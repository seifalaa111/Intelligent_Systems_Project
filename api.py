"""
MIDAN AI Decision Engine — FastAPI Backend
Hybrid 4-Layer Architecture:
  L1 — LLM Idea Parser         (extracts 8 structured idea features)
  L2 — ML Signal Pipeline      (DBSCAN→FCM→SVM→SHAP→SARIMA on macro features)
  L3 — Idea Signal Scorer      (regime×model×segment fit table — varies per idea)
  L4 — Composite TAS           (conf×0.30 + sarima×0.20 + idea_signal×0.35 + xai×0.15)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pickle, json, os, warnings, requests
from textwrap import dedent
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

try:
    from groq import Groq
    GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy"))
except Exception:
    GROQ_CLIENT = None

# ── LOAD MODELS ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _pkl(name):
    with open(f'{MODELS_DIR}/{name}', 'rb') as f:
        return pickle.load(f)

def _json(name, default=None):
    try:
        with open(f'{MODELS_DIR}/{name}', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

try:
    scaler         = _pkl('scaler_global.pkl')
    pca            = _pkl('pca_global.pkl')
    svm            = _pkl('svm_global.pkl')
    le             = _pkl('label_encoder.pkl')
    lgb            = _pkl('lgb_surrogate.pkl')
    sarima_results = _json('sarima_results.json')
    comps_data     = _json('competitors_context.json', {})
    sents_data     = _json('sentiment_context.json', [])
    MODELS_LOADED  = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR   = str(e)

# ── MACRO FEATURES (fed into existing ML pipeline) ───────────
FEATURES = ['inflation', 'gdp_growth', 'macro_friction', 'capital_concentration', 'velocity_yoy']

SECTOR_LABEL_MAP = {
    "E-commerce": "ecommerce", "Healthtech": "healthtech",
    "Edtech": "edtech", "SaaS": "saas", "Logistics": "logistics",
    "Agritech": "agritech", "Other": "other", "Fintech": "fintech"
}

SECTOR_MEDIANS = {
    'fintech': 175000.0, 'ecommerce': 120000.0, 'healthtech': 200000.0,
    'edtech': 80000.0,   'saas': 250000.0,       'logistics': 90000.0,
    'agritech': 50000.0, 'other': 100000.0,
}

SECTOR_EFF_MACRO = {
    'fintech':    (7.5,  +1.5, 0.28),
    'healthtech': (7.0,  +2.0, 0.22),
    'saas':       (4.0,  +2.2, 0.10),
    'agritech':   (4.5,  +0.7, 0.12),
    'edtech':     (40.0, -1.0, 0.07),
    'logistics':  (42.0, -1.8, 0.09),
    'ecommerce':  (36.0, -1.3, 0.13),
    'other':      (33.9,  0.0, 0.10),
}

COUNTRY_MACRO_DEFAULTS = {
    'EG': {'inflation': 33.9, 'gdp_growth': 3.8, 'unemployment': 7.1},
    'SA': {'inflation':  2.3, 'gdp_growth': 1.9, 'unemployment': 6.1},
    'AE': {'inflation':  1.6, 'gdp_growth': 4.2, 'unemployment': 3.1},
    'US': {'inflation':  3.4, 'gdp_growth': 2.5, 'unemployment': 3.7},
    'GB': {'inflation':  4.0, 'gdp_growth': 0.1, 'unemployment': 4.2},
    'NG': {'inflation': 28.9, 'gdp_growth': 3.3, 'unemployment': 4.1},
    'KE': {'inflation':  6.3, 'gdp_growth': 5.6, 'unemployment': 5.7},
    'MA': {'inflation':  6.1, 'gdp_growth': 3.1, 'unemployment': 11.5},
}

SECTOR_KEYWORDS = {
    'fintech':    ['finance','payment','fintech','bank','loan','lending',
                   'invoice','insurance','wallet','money','تمويل','دفع'],
    'ecommerce':  ['ecommerce','e-commerce','shop','store','retail',
                   'marketplace','delivery','commerce','تجارة','توصيل'],
    'healthtech': ['health','medical','doctor','clinic','hospital',
                   'pharma','biotech','mental','صحة','طبي'],
    'edtech':     ['education','learning','school','university','course',
                   'tutor','edtech','training','تعليم','دراسة'],
    'saas':       ['saas','software','platform','dashboard','tool',
                   'api','enterprise','cloud','b2b','crm','برنامج'],
    'logistics':  ['logistics','shipping','supply chain','warehouse',
                   'transport','fleet','trucking','شحن','لوجستيك'],
    'agritech':   ['agri','farm','crop','harvest','food',
                   'agriculture','irrigation','زراعة'],
}
COUNTRY_KEYWORDS = {
    'EG': ['egypt','cairo','egyptian','مصر','القاهرة','giza','assiut'],
    'SA': ['saudi','ksa','riyadh','jeddah','السعودية','مكة'],
    'AE': ['uae','dubai','abu dhabi','emirates','الإمارات','دبي'],
    'MA': ['morocco','moroccan','casablanca','المغرب','rabat','marrakech'],
    'NG': ['nigeria','nigerian','lagos','abuja','نيجيريا'],
    'KE': ['kenya','kenyan','nairobi','كينيا'],
    'US': ['usa','united states','america','أمريكا','silicon valley'],
    'GB': ['uk','britain','london','england','بريطانيا'],
}


# ═══════════════════════════════════════════════════════════════
# AGENT A1 — NLP Parser (sector + country from free text)
# ═══════════════════════════════════════════════════════════════

def agent_a1_parse(idea_text: str):
    t = idea_text.lower()
    sector, sector_found = None, False
    for sec, kws in SECTOR_KEYWORDS.items():
        if any(k in t for k in kws):
            sector, sector_found = sec, True
            break
    if not sector_found:
        sector = 'fintech'
    country, country_found = None, False
    for code, kws in COUNTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            country, country_found = code, True
            break
    if not country_found:
        country = 'EG'
    return sector, country, sector_found, country_found


def enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth, macro_friction, velocity_yoy):
    if gdp_growth > 3.5 and inflation < 8 and velocity_yoy > 0.15:
        margin = min((gdp_growth - 3.5) / 4.0, (8 - inflation) / 8.0, (velocity_yoy - 0.15) / 0.25)
        return 'GROWTH_MARKET', float(np.clip(0.65 + margin * 0.30, 0.60, 0.95))
    if gdp_growth > 2.0 and inflation < 10 and macro_friction < 10:
        margin = min((gdp_growth - 2.0) / 4.0, (10 - inflation) / 10.0, (10 - macro_friction) / 15.0)
        return 'EMERGING_MARKET', float(np.clip(0.60 + margin * 0.30, 0.55, 0.90))
    if gdp_growth < 0 or (inflation > 50 and macro_friction > 50):
        severity = max(abs(min(gdp_growth, 0)) / 3.0, 0.0)
        return 'CONTRACTING_MARKET', float(np.clip(0.65 + severity * 0.25, 0.60, 0.92))
    if macro_friction > 30 or inflation > 25:
        pain = max((macro_friction - 30) / 40, (inflation - 25) / 30, 0)
        return 'HIGH_FRICTION_MARKET', float(np.clip(0.60 + pain * 0.30, 0.55, 0.92))
    return svm_regime, svm_conf


# ═══════════════════════════════════════════════════════════════
# LAYER 1 — IDEA FEATURE EXTRACTION
# Converts raw idea text → structured 8-field schema.
# This is what makes different ideas produce different outputs.
# ═══════════════════════════════════════════════════════════════

BUSINESS_MODELS    = ['subscription', 'marketplace', 'saas', 'commission', 'service', 'hardware', 'other']
TARGET_SEGMENTS    = ['b2b', 'b2c', 'b2g', 'mixed']
MONETIZATION_TYPES = ['subscription', 'commission', 'one-time', 'freemium', 'ad-based', 'other']
STAGES             = ['idea', 'validation', 'mvp', 'growth']
INTENSITY_LEVELS   = ['low', 'medium', 'high']


def extract_idea_features(idea_text: str, sector: str) -> dict:
    """
    Layer 1 — Structured extraction of idea-specific features.
    Uses Groq LLM (temperature=0) when available; falls back to keyword heuristics.
    Returns 8 fields that VARY per idea — the foundation of output variability.
    """
    if not idea_text or len(idea_text.strip()) < 8:
        return _default_idea_features(sector)

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if GROQ_CLIENT and groq_key and groq_key != "dummy":
        try:
            prompt = dedent(f"""
                Extract structured startup features from this idea description.
                Return ONLY valid JSON matching the exact schema. No explanation.

                Idea: "{idea_text}"
                Sector hint: {sector}

                Required JSON:
                {{
                  "business_model": "subscription|marketplace|saas|commission|service|hardware|other",
                  "target_segment": "b2b|b2c|b2g|mixed",
                  "monetization": "subscription|commission|one-time|freemium|ad-based|other",
                  "stage": "idea|validation|mvp|growth",
                  "differentiation_score": 3,
                  "competitive_intensity": "low|medium|high",
                  "regulatory_risk": "low|medium|high",
                  "market_readiness": 3
                }}

                Scoring rules:
                - differentiation_score 1-5: 1=direct copy of existing, 3=some edge, 5=clear unique moat
                - market_readiness 1-5: 1=pioneer creating demand, 3=proven pain exists, 5=strong pull signals
                - stage: 'idea' if no traction, 'validation' if testing, 'mvp' if built, 'growth' if scaling
                - Use ONLY the exact enum values listed above
                - Different ideas MUST produce meaningfully different scores — never default to all 3s
            """).strip()

            resp = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)

            return {
                "business_model":       data.get("business_model")       if data.get("business_model")       in BUSINESS_MODELS    else "other",
                "target_segment":       data.get("target_segment")       if data.get("target_segment")       in TARGET_SEGMENTS    else "b2c",
                "monetization":         data.get("monetization")         if data.get("monetization")         in MONETIZATION_TYPES else "other",
                "stage":                data.get("stage")                if data.get("stage")                in STAGES             else "idea",
                "differentiation_score": max(1, min(5, int(data.get("differentiation_score", 3)))),
                "competitive_intensity": data.get("competitive_intensity") if data.get("competitive_intensity") in INTENSITY_LEVELS else "medium",
                "regulatory_risk":       data.get("regulatory_risk")       if data.get("regulatory_risk")       in INTENSITY_LEVELS else "medium",
                "market_readiness":      max(1, min(5, int(data.get("market_readiness", 3)))),
            }
        except Exception:
            pass

    return _heuristic_idea_features(idea_text, sector)


def _default_idea_features(sector: str) -> dict:
    reg = {'fintech': 'high', 'healthtech': 'high', 'edtech': 'low',
           'ecommerce': 'low', 'saas': 'low', 'logistics': 'medium',
           'agritech': 'low', 'other': 'medium'}.get(sector, 'medium')
    return {
        "business_model": "other", "target_segment": "mixed",
        "monetization": "other", "stage": "idea",
        "differentiation_score": 3, "competitive_intensity": "medium",
        "regulatory_risk": reg, "market_readiness": 3,
    }


def _heuristic_idea_features(idea_text: str, sector: str) -> dict:
    """Keyword-based fallback for idea feature extraction."""
    t = idea_text.lower()

    # Business model
    bm = 'other'
    if any(w in t for w in ['subscription', 'monthly plan', 'annual fee', 'recurring revenue']): bm = 'subscription'
    elif sector == 'saas' or any(w in t for w in ['saas', 'cloud software', 'software as a service']): bm = 'saas'
    elif any(w in t for w in ['marketplace', 'connect buyers', 'two-sided', 'buyers and sellers']): bm = 'marketplace'
    elif any(w in t for w in ['commission', 'take rate', 'earn per transaction', 'percentage of']): bm = 'commission'
    elif any(w in t for w in ['consulting', 'managed service', 'white-label', 'agency']): bm = 'service'
    elif sector == 'fintech':   bm = 'commission'
    elif sector == 'ecommerce': bm = 'marketplace'
    elif sector == 'saas':      bm = 'saas'

    # Target segment
    seg = 'b2c'
    if any(w in t for w in ['b2b', 'businesses', 'enterprise', 'sme', 'smes', 'corporate', 'companies', 'clients']): seg = 'b2b'
    elif any(w in t for w in ['government', 'ministry', 'public sector', 'municipalities']): seg = 'b2g'
    elif any(w in t for w in ['both consumers and', 'b2b and b2c', 'businesses and individuals']): seg = 'mixed'

    # Stage
    stage = 'idea'
    if any(w in t for w in ['launched', 'live', 'users', 'customers', 'mrr', 'revenue', 'growing', 'scaling']): stage = 'mvp'
    elif any(w in t for w in ['beta', 'pilot', 'testing', 'prototype', 'validating', 'early access']): stage = 'validation'

    # Differentiation score
    diff = 3
    if any(w in t for w in ['ai-powered', 'ai powered', 'machine learning', 'first in', 'only platform',
                             'patent', 'unique algorithm', 'proprietary data', 'no one else']): diff = 4
    if any(w in t for w in ['breakthrough', 'world first', 'never been done', 'disruptive', 'revolutionary']): diff = 5
    if any(w in t for w in ['similar to', 'like uber', 'like amazon', 'inspired by', 'copy of', 'clone']): diff = 2
    if any(w in t for w in ['same as', 'another', 'yet another', 'also does']): diff = 1

    # Competitive intensity by sector
    comp = {'fintech': 'high', 'ecommerce': 'high', 'healthtech': 'medium',
            'saas': 'medium', 'edtech': 'medium', 'logistics': 'medium',
            'agritech': 'low', 'other': 'medium'}.get(sector, 'medium')

    # Regulatory risk by sector
    reg = {'fintech': 'high', 'healthtech': 'high', 'edtech': 'low',
           'ecommerce': 'low', 'saas': 'low', 'logistics': 'medium',
           'agritech': 'low', 'other': 'medium'}.get(sector, 'medium')

    # Market readiness
    ready = 3
    if any(w in t for w in ['proven market', 'large demand', 'everyone needs', 'mass market', 'high demand']): ready = 4
    if any(w in t for w in ['pioneer', 'niche', 'emerging need', 'early adopters', 'creating the market']): ready = 2

    return {
        "business_model": bm, "target_segment": seg,
        "monetization": bm, "stage": stage,
        "differentiation_score": diff, "competitive_intensity": comp,
        "regulatory_risk": reg, "market_readiness": ready,
    }


# ═══════════════════════════════════════════════════════════════
# LAYER 3 — IDEA SIGNAL SCORING
# Encodes domain knowledge: which business models thrive in which regimes.
# This is the PRIMARY driver of output variability across different ideas.
# ═══════════════════════════════════════════════════════════════

# (regime, business_model, target_segment) → base fit score
# Different combinations produce meaningfully different base scores.
_FIT_TABLE: Dict[tuple, float] = {
    # GROWTH MARKET — aggressive expansion is rewarded
    ('GROWTH_MARKET', 'saas',         'b2b'):   0.91,
    ('GROWTH_MARKET', 'subscription', 'b2b'):   0.90,
    ('GROWTH_MARKET', 'marketplace',  'b2c'):   0.87,
    ('GROWTH_MARKET', 'marketplace',  'mixed'): 0.85,
    ('GROWTH_MARKET', 'commission',   'mixed'): 0.84,
    ('GROWTH_MARKET', 'commission',   'b2c'):   0.83,
    ('GROWTH_MARKET', 'subscription', 'b2c'):   0.80,
    ('GROWTH_MARKET', 'service',      'b2b'):   0.77,
    ('GROWTH_MARKET', 'hardware',     'mixed'): 0.72,
    ('GROWTH_MARKET', 'other',        'mixed'): 0.70,

    # EMERGING MARKET — subscription/SaaS most resilient; B2C needs patience
    ('EMERGING_MARKET', 'saas',         'b2b'):   0.84,
    ('EMERGING_MARKET', 'subscription', 'b2b'):   0.82,
    ('EMERGING_MARKET', 'marketplace',  'mixed'): 0.77,
    ('EMERGING_MARKET', 'commission',   'mixed'): 0.73,
    ('EMERGING_MARKET', 'commission',   'b2b'):   0.71,
    ('EMERGING_MARKET', 'subscription', 'b2c'):   0.67,
    ('EMERGING_MARKET', 'service',      'b2b'):   0.66,
    ('EMERGING_MARKET', 'marketplace',  'b2b'):   0.64,
    ('EMERGING_MARKET', 'service',      'b2c'):   0.55,
    ('EMERGING_MARKET', 'other',        'mixed'): 0.58,

    # HIGH FRICTION — only cost-saving B2B tools survive
    ('HIGH_FRICTION_MARKET', 'saas',         'b2b'):   0.72,
    ('HIGH_FRICTION_MARKET', 'subscription', 'b2b'):   0.68,
    ('HIGH_FRICTION_MARKET', 'commission',   'b2b'):   0.54,
    ('HIGH_FRICTION_MARKET', 'service',      'b2b'):   0.50,
    ('HIGH_FRICTION_MARKET', 'marketplace',  'mixed'): 0.43,
    ('HIGH_FRICTION_MARKET', 'commission',   'b2c'):   0.40,
    ('HIGH_FRICTION_MARKET', 'subscription', 'b2c'):   0.38,
    ('HIGH_FRICTION_MARKET', 'other',        'b2c'):   0.32,

    # CONTRACTING — survival mode; only essential B2B cost-cutters viable
    ('CONTRACTING_MARKET', 'saas',         'b2b'):   0.58,
    ('CONTRACTING_MARKET', 'service',      'b2b'):   0.52,
    ('CONTRACTING_MARKET', 'subscription', 'b2b'):   0.50,
    ('CONTRACTING_MARKET', 'commission',   'b2b'):   0.42,
    ('CONTRACTING_MARKET', 'marketplace',  'mixed'): 0.32,
    ('CONTRACTING_MARKET', 'marketplace',  'b2c'):   0.28,
    ('CONTRACTING_MARKET', 'subscription', 'b2c'):   0.25,
    ('CONTRACTING_MARKET', 'other',        'b2c'):   0.20,
}

_REGIME_DEFAULTS: Dict[str, float] = {
    'GROWTH_MARKET': 0.72, 'EMERGING_MARKET': 0.60,
    'HIGH_FRICTION_MARKET': 0.45, 'CONTRACTING_MARKET': 0.35,
}

# ── BM-CONDITIONAL WEIGHT PROFILES ────────────────────────────────────────────
# Different business model types have structurally different success drivers.
# A marketplace lives or dies on liquidity — competition is lethal (winner-takes-all).
# SaaS/Subscription succeeds through differentiation and switching cost creation.
# Commission/Fintech: regulatory risk is existential, not just a drag.
# Service: competition matters less; scalability is the binding constraint.
# Hardware: stage is the dominant penalty — capital requirements are brutal early.
#
# Applying uniform weights to all BM types produces meaningless signals.
# This table encodes domain knowledge about what actually drives each model.
_BM_PROFILE: Dict[str, dict] = {
    'marketplace': {
        'diff_scale':    0.07,   # differentiation matters LESS — supply/demand fit matters more
        'ready_scale':   0.08,   # market readiness matters MORE — need existing demand, not pioneers
        'stage_deltas':  {'idea': -0.14, 'validation': -0.06, 'mvp': +0.10, 'growth': +0.20},
        'comp_deltas':   {'low': +0.10, 'medium': 0.0, 'high': -0.16},  # winner-takes-all: competition lethal
        'dominant_risk': 'liquidity',
        'key_signal':    'market_readiness',
        'moat_source':   'network effects and supply lock-in',
    },
    'saas': {
        'diff_scale':    0.13,   # differentiation matters MORE — switching costs = moat
        'ready_scale':   0.04,
        'stage_deltas':  {'idea': -0.07, 'validation': -0.01, 'mvp': +0.08, 'growth': +0.15},
        'comp_deltas':   {'low': +0.07, 'medium': 0.0, 'high': -0.08},
        'dominant_risk': 'differentiation',
        'key_signal':    'differentiation_score',
        'moat_source':   'switching costs and workflow integration depth',
    },
    'subscription': {
        'diff_scale':    0.10,
        'ready_scale':   0.06,   # readiness drives recurring willingness-to-pay
        'stage_deltas':  {'idea': -0.08, 'validation': -0.02, 'mvp': +0.07, 'growth': +0.14},
        'comp_deltas':   {'low': +0.05, 'medium': 0.0, 'high': -0.07},
        'dominant_risk': 'churn',
        'key_signal':    'market_readiness',
        'moat_source':   'habit formation and cancellation friction',
    },
    'commission': {
        'diff_scale':    0.09,
        'ready_scale':   0.05,
        'stage_deltas':  {'idea': -0.10, 'validation': -0.03, 'mvp': +0.07, 'growth': +0.15},
        'comp_deltas':   {'low': +0.06, 'medium': 0.0, 'high': -0.11},
        'dominant_risk': 'regulatory',  # commission models attract regulatory scrutiny
        'key_signal':    'regulatory_risk',
        'moat_source':   'regulatory licensing once obtained',
    },
    'service': {
        'diff_scale':    0.09,
        'ready_scale':   0.04,
        'stage_deltas':  {'idea': -0.05, 'validation': +0.01, 'mvp': +0.06, 'growth': +0.12},  # services monetize earlier
        'comp_deltas':   {'low': +0.04, 'medium': 0.0, 'high': -0.05},   # competition less relevant for services
        'dominant_risk': 'scalability',
        'key_signal':    'differentiation_score',
        'moat_source':   'proprietary methodology and key person relationships',
    },
    'hardware': {
        'diff_scale':    0.12,
        'ready_scale':   0.05,
        'stage_deltas':  {'idea': -0.15, 'validation': -0.07, 'mvp': +0.06, 'growth': +0.17},  # capital-intensive
        'comp_deltas':   {'low': +0.07, 'medium': 0.0, 'high': -0.10},
        'dominant_risk': 'capital',
        'key_signal':    'differentiation_score',
        'moat_source':   'proprietary hardware design and manufacturing relationships',
    },
    'other': {
        'diff_scale':    0.11,
        'ready_scale':   0.04,
        'stage_deltas':  {'idea': -0.09, 'validation': -0.02, 'mvp': +0.06, 'growth': +0.14},
        'comp_deltas':   {'low': +0.06, 'medium': 0.0, 'high': -0.09},
        'dominant_risk': 'execution',
        'key_signal':    'differentiation_score',
        'moat_source':   'speed and execution quality',
    },
}

# Per-sector regulatory sensitivity — fintech and healthtech are in a different league
_SECTOR_REG_PROFILE: Dict[str, dict] = {
    'fintech': {
        'reg_deltas':  {'low': +0.03, 'medium': -0.05, 'high': -0.18},  # CBE licensing = existential
        'b2b_bonus':    0.05,   # B2B fintech less exposed to consumer protection
        'trust_factor': True,   # regulatory risk is also trust barrier — double exposure
    },
    'healthtech': {
        'reg_deltas':  {'low': +0.04, 'medium': -0.06, 'high': -0.20},  # medical approval = long, expensive
        'b2b_bonus':    0.07,   # hospital/clinic sales: clearer buyer, less consumer friction
        'trust_factor': True,
    },
    'ecommerce': {
        'reg_deltas':  {'low': +0.04, 'medium': 0.0, 'high': -0.08},
        'b2b_bonus':    0.0,
        'trust_factor': False,
    },
    'edtech': {
        'reg_deltas':  {'low': +0.04, 'medium': -0.01, 'high': -0.09},
        'b2b_bonus':    0.03,
        'trust_factor': False,
    },
    'saas': {
        'reg_deltas':  {'low': +0.05, 'medium': 0.0, 'high': -0.10},
        'b2b_bonus':    0.04,
        'trust_factor': False,
    },
    'logistics': {
        'reg_deltas':  {'low': +0.03, 'medium': -0.02, 'high': -0.12},
        'b2b_bonus':    0.03,
        'trust_factor': False,
    },
    '_default': {
        'reg_deltas':  {'low': +0.04, 'medium': 0.0, 'high': -0.11},
        'b2b_bonus':    0.0,
        'trust_factor': False,
    },
}


def compute_idea_signal(idea_features: dict, regime: str, sector: str = 'other') -> dict:
    """
    Layer 3 — Context-aware idea signal scoring.

    v2 UPGRADE: Conditional logic by business model type.
    Each BM type has different dominant success factors:
    - Marketplace: liquidity/network effects; competition is lethal
    - SaaS: differentiation (switching costs = moat); highly weighted
    - Commission: regulatory exposure is the binding constraint
    - Subscription: market readiness drives recurring willingness-to-pay
    - Service: scalability is the constraint; competition matters less
    - Hardware: stage penalty is amplified (capital requirements)

    Additionally, sector-specific regulatory profiles:
    - Fintech: CBE/regulatory risk is existential, not just a drag
    - Healthtech: medical approval creates 2-3× longer timelines
    - Others: standard profile

    Returns: idea_signal (float 0–1), breakdown (dict), dominant_risk, key_signal, moat_source
    """
    bm    = idea_features.get('business_model',       'other')
    seg   = idea_features.get('target_segment',        'b2c')
    diff  = idea_features.get('differentiation_score', 3)
    stage = idea_features.get('stage',                 'idea')
    comp  = idea_features.get('competitive_intensity', 'medium')
    reg   = idea_features.get('regulatory_risk',       'medium')
    ready = idea_features.get('market_readiness',      3)

    # 1. Base fit from regime × model × segment table
    base = (
        _FIT_TABLE.get((regime, bm, seg))
        or _FIT_TABLE.get((regime, bm, 'mixed'))
        or _FIT_TABLE.get((regime, 'other', 'mixed'))
        or _REGIME_DEFAULTS.get(regime, 0.50)
    )

    # 2. Load BM-specific profile (conditional weights)
    bm_profile     = _BM_PROFILE.get(bm, _BM_PROFILE['other'])
    sector_profile = _SECTOR_REG_PROFILE.get(sector, _SECTOR_REG_PROFILE['_default'])

    # 3. Differentiation multiplier — scale varies by BM type
    diff_mult = 0.78 + (diff - 1) * bm_profile['diff_scale']

    # 4. Stage delta — varies by BM (hardware/marketplace more stage-sensitive)
    stage_delta = bm_profile['stage_deltas'].get(stage, 0.0)

    # 5. Competitive drag — varies by BM (winner-takes-all models more sensitive)
    comp_delta = bm_profile['comp_deltas'].get(comp, 0.0)

    # 6. Regulatory friction — varies by SECTOR (fintech/healthtech amplified)
    reg_delta = sector_profile['reg_deltas'].get(reg, 0.0)

    # B2B segment partial relief in high-regulatory sectors
    # (institutional buyers vs consumer protection exposure)
    if seg == 'b2b' and sector_profile['b2b_bonus'] > 0:
        b2b_relief = sector_profile['b2b_bonus'] * (0.4 if reg == 'high' else 0.6)
        reg_delta += b2b_relief

    # 7. Market readiness — scale varies by BM
    ready_delta = (ready - 3) * bm_profile['ready_scale']

    # 8. Compounded penalty: early-stage in hostile regime
    # In CONTRACTING/HIGH_FRICTION, early ideas face amplified survival risk
    if stage in ('idea', 'validation') and regime in ('CONTRACTING_MARKET', 'HIGH_FRICTION_MARKET'):
        stage_delta *= 1.35

    idea_signal = float(np.clip(
        base * diff_mult + stage_delta + comp_delta + reg_delta + ready_delta,
        0.12, 0.95
    ))

    breakdown = {
        'model_regime_fit':    round(base, 3),
        'differentiation':     round((diff_mult - 1.0) * base, 3),
        'stage_readiness':     round(stage_delta, 3),
        'competitive_density': round(comp_delta, 3),
        'regulatory_exposure': round(reg_delta, 3),
        'market_pull':         round(ready_delta, 3),
    }

    return {
        'idea_signal':    idea_signal,
        'breakdown':      breakdown,
        'business_model': bm,
        'target_segment': seg,
        'diff_score':     diff,
        'stage':          stage,
        'dominant_risk':  bm_profile['dominant_risk'],
        'key_signal':     bm_profile['key_signal'],
        'moat_source':    bm_profile['moat_source'],
    }


def _signal_tier(score: float) -> str:
    """Qualitative tier — replaces fake numeric precision."""
    if score >= 0.76: return "Strong"
    if score >= 0.60: return "Moderate"
    if score >= 0.44: return "Mixed"
    return "Weak"


# ═══════════════════════════════════════════════════════════════
# LAYER 4 — STRATEGIC REASONING ENGINE
#
# This is the intelligence layer. It does NOT report signals —
# it interprets them through business logic and strategic reasoning.
#
# Output: 5 structured reasoning fields (not scores, not labels):
#   1. strategic_interpretation — what actually matters here
#   2. key_driver — the ONE thing determining success/failure
#   3. main_risk — the specific failure mode most likely
#   4. counterpoint — why it could fail EVEN IF signals are good
#   5. differentiation_insight — why this idea behaves differently
#
# LLM path: Groq JSON-structured prompt (temperature 0.35)
# Fallback: rich conditional logic by BM type × regime × signals
# ═══════════════════════════════════════════════════════════════

def _l4_reasoning_llm(
    sector: str, country: str, regime: str, conf: float,
    sarima_trend: float, idea_features: dict, idea_signal_data: dict,
    shap_dict: dict, tas: float, signal_tier: str,
    a2_comps: list, a4_sentiment: str,
    diff_label: str, bm_label: str, seg_label: str,
) -> Optional[dict]:
    """
    LLM path: structured JSON reasoning with 5 required fields.
    Returns None on any failure — fallback handles it.
    """
    bm    = idea_features.get('business_model', 'other')
    seg   = idea_features.get('target_segment', 'b2c')
    diff  = idea_features.get('differentiation_score', 3)
    stage = idea_features.get('stage', 'idea')
    comp  = idea_features.get('competitive_intensity', 'medium')
    reg   = idea_features.get('regulatory_risk', 'medium')
    ready = idea_features.get('market_readiness', 3)

    dominant_risk = idea_signal_data.get('dominant_risk', 'execution')
    idea_signal   = idea_signal_data['idea_signal']
    moat_source   = idea_signal_data.get('moat_source', 'execution quality')
    regime_r      = regime.replace('_', ' ').title()
    top_shap      = max(shap_dict, key=shap_dict.get).replace('_', ' ')

    prompt = dedent(f"""
        You are MIDAN — a senior operator who has backed 50 startups and killed 200 pitches.
        You must produce a DECISION, not an analysis. No hedging. No balanced takes.
        Your job is to surface what is actually true and what will actually break this.

        COMPLETE SIGNAL PICTURE (do not re-derive — use this):
        Market: {sector.title()} in {country} → {regime_r} ({conf:.0%} confidence)
        90-day trend: {'growing' if sarima_trend > 0.5 else 'declining or flat'} (SARIMA={sarima_trend:.2f})
        Top macro signal: {top_shap} (caused this regime classification)
        Sentiment: {a4_sentiment} | Competitors: {', '.join(a2_comps[:2]) if a2_comps else 'unidentified'}

        Idea anatomy:
        BM={bm_label} | Segment={seg_label} | Stage={stage}
        Diff={diff}/5 ({diff_label}) | Competition={comp} | Reg={reg} | Market-readiness={ready}/5
        Signal: {signal_tier} ({idea_signal:.0%}) | Primary risk type: {dominant_risk}
        Moat source for this BM: {moat_source}

        Produce EXACTLY this JSON — 7 fields, no markdown, no extra text:
        {{
          "strategic_interpretation": "2-3 sentences. The real situation — what this market condition actually means for THIS specific BM/segment combo. Lead with the non-obvious insight. Do NOT start with the regime name. Reference {top_shap} as the root cause.",
          "key_driver": "1 sentence only. The single factor that determines whether this succeeds or fails. Be explicit — not 'execution' but what specific execution means here for {bm_label} in {regime_r}. Frame it as a test: 'If X does not happen by Y, this fails.'",
          "main_risk": "1-2 sentences. The most likely specific failure mode given {diff_label} diff, {stage} stage, {reg} reg risk, and {comp} competition. State the failure clearly — not a risk category, the actual way this breaks.",
          "counterpoint": "1-2 sentences. Challenge the CORE assumption that makes this idea feel viable. Not 'it is risky' — the specific belief the founder holds that is probably wrong. Start with 'The assumption here is...' or 'The founder believes...' then destroy it.",
          "differentiation_insight": "1-2 sentences. Whether this idea is actually different from the baseline {bm_label} idea in {sector}/{country} — or honestly acknowledge it is not. Reference diff={diff}/5 and {moat_source}. Be honest, not encouraging.",
          "what_matters_most": "ONE sentence. The single factor that will determine success or failure above all others for this exact BM+sector+stage combination. More specific than key_driver — this is the make-or-break variable.",
          "counter_thesis": "1-2 sentences. The opposite position: why everything the founder is excited about is probably wrong. Example format: 'You think X is the challenge — it is not. The real problem is Y, and Y is harder to solve.' Be blunt. This must contradict the strategic_interpretation."
        }}

        HARD RULES (violations will be rejected):
        - Every field must be grounded in {bm_label}, {seg_label}, {sector}, {country}, or specific numbers
        - counterpoint must contradict strategic_interpretation — not extend it
        - counter_thesis must name a SPECIFIC wrong assumption — not generic startup risk
        - No platitudes: no 'consider your options', 'focus on value proposition', 'it depends'
        - No starting any field with 'I'
        - what_matters_most and key_driver must be about DIFFERENT variables
        - main_risk must name the failure mechanism — not just a risk category
    """).strip()

    try:
        resp = GROQ_CLIENT.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.30,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw  = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        required = ['strategic_interpretation', 'key_driver', 'main_risk', 'counterpoint',
                    'differentiation_insight', 'what_matters_most', 'counter_thesis']
        if all(data.get(f) and len(str(data.get(f, ''))) > 25 for f in required):
            return {f: str(data[f]).strip() for f in required}
    except Exception:
        pass
    return None


def _l4_reasoning_fallback(
    sector: str, country: str, regime: str, conf: float,
    sarima_trend: float, idea_features: dict, idea_signal_data: dict,
    shap_dict: dict, tas: float, signal_tier: str,
    diff_label: str, bm_label: str, seg_label: str,
) -> dict:
    """
    Rich conditional fallback — varies by BM type × regime × key signals.
    Produces structurally different narratives for different idea types.
    """
    bm    = idea_features.get('business_model', 'other')
    seg   = idea_features.get('target_segment', 'b2c')
    diff  = idea_features.get('differentiation_score', 3)
    stage = idea_features.get('stage', 'idea')
    comp  = idea_features.get('competitive_intensity', 'medium')
    reg   = idea_features.get('regulatory_risk', 'medium')
    ready = idea_features.get('market_readiness', 3)

    dominant_risk = idea_signal_data.get('dominant_risk', 'execution')
    idea_signal   = idea_signal_data['idea_signal']
    moat_source   = idea_signal_data.get('moat_source', 'execution quality')
    regime_r      = regime.replace('_', ' ').title()
    top_shap      = max(shap_dict, key=shap_dict.get).replace('_', ' ')
    in_friction   = regime in ('HIGH_FRICTION_MARKET', 'CONTRACTING_MARKET')
    in_growth     = regime in ('GROWTH_MARKET', 'EMERGING_MARKET')

    # ── 1. Strategic Interpretation — structurally different by BM + regime ──
    if bm == 'marketplace':
        if in_friction:
            si = (
                f"Marketplace models in {regime_r} conditions break on the supply side first, not demand. "
                f"In {country}'s current macro environment — {top_shap} is the dominant signal — "
                f"suppliers disengage when margins compress, leaving the demand side with nothing to transact on. "
                f"Your {signal_tier.lower()} fit score reflects that structural fragility, not the surface-level opportunity."
            )
        elif regime == 'GROWTH_MARKET':
            si = (
                f"A {sector} marketplace in {regime_r} has one existential window: lock in supply "
                f"before a well-capitalized competitor does the same thing with better distribution. "
                f"{'High competition in this space means the window is already narrowing.' if comp == 'high' else 'The competitive space is still open — that is rare and time-limited.'} "
                f"Speed of liquidity creation is the only metric that matters; everything else is a distraction."
            )
        else:
            si = (
                f"Marketplaces in {regime_r} markets succeed when the informal version already exists and people are already transacting — "
                f"your job is to formalize faster than the market matures enough to attract international capital. "
                f"{'Market demand signals are real here, which is a genuine tailwind.' if ready >= 4 else 'Market readiness is still early — you are creating demand, not capturing it. That costs 5x more and takes 3x longer than it looks.'}"
            )
    elif bm in ('saas', 'subscription'):
        if in_friction:
            si = (
                f"{'SaaS' if bm == 'saas' else 'Subscription'} survives in {regime_r} only if it replaces a cost that already exists — "
                f"it cannot create a new budget line when discretionary spending has collapsed. "
                f"{'B2B buyers still approve tools that demonstrably cut cost or reduce headcount — that is the only pitch that works right now.' if seg == 'b2b' else 'B2C subscriptions are being cancelled first in this environment. The affordability math is against you.'} "
                f"The {top_shap} macro signal is the structural constraint — not a temporary condition."
            )
        elif regime == 'GROWTH_MARKET':
            si = (
                f"{'SaaS' if bm == 'saas' else 'Subscription'} in a {regime_r} is a land-grab — the market rewards speed of distribution over product completeness. "
                f"{'Your differentiation score of ' + str(diff) + '/5 is your pricing power; without it, you compete on price before you can compete on value.' if diff >= 4 else 'With ' + diff_label + ' differentiation, you will be forced to compete on price before establishing any sustainable margin — that is a burn problem, not a product problem.'} "
                f"The {top_shap} signal is pulling the market; your timing needs to align with it."
            )
        else:
            si = (
                f"A {bm} model for {seg_label} customers in {sector}/{country} reads as {regime_r} — "
                f"which means {'patience and local fit beat speed' if regime == 'EMERGING_MARKET' else 'survival framing, not growth framing'}. "
                f"{'The B2B angle gives you longer-term contracts and lower churn exposure — both are structural advantages in this regime.' if seg == 'b2b' else 'B2C subscription in this regime faces real affordability friction that your financial model needs to account for.'}"
            )
    elif bm == 'commission':
        if sector == 'fintech':
            si = (
                f"Commission-based fintech in {country} has one pre-condition that either kills or enables everything else: "
                f"{'regulatory approval — CBE licensing in Egypt takes 12-18 months minimum, and most founders budget 6 and run out of money at month 9.' if reg == 'high' and country == 'EG' else 'regulatory clearance, which functions as a moat once obtained and a barrier while pending.' if reg == 'high' else 'a regulatory path that is navigable — which makes it a moat once you are through and competitors have not started yet.'}  "
                f"Transaction volume in a {regime_r} {'shrinks with GDP — commission revenue is directly tied to GMV, and GMV shrinks in friction environments.' if in_friction else 'grows with economic activity, which means your revenue scales without additional cost.'}"
            )
        else:
            si = (
                f"Commission models are structurally a volume game — "
                f"{'you are trying to win a volume game in a low-volume environment. That math does not resolve itself without a market shift.' if in_friction else 'the growth market conditions actually accelerate your revenue linearly with market expansion.'}  "
                f"{'High competitive intensity will compress your take rate before you reach sustainable volume — that is the most common death pattern for commission models.' if comp == 'high' else 'Low competition gives you take-rate pricing power — rare and valuable in this model type.'}"
            )
    else:  # service, hardware, other
        si = (
            f"A {bm_label} model for {seg_label} customers in {sector}/{country} "
            f"operates in {regime_r} — which means {'cash conservation beats growth optionality' if in_friction else 'execution speed is the primary competitive variable'}. "
            f"{'With ' + diff_label + ' differentiation (' + str(diff) + '/5), the moat is execution and relationships, not the product — which means it is fragile to key-person departure.' if diff <= 3 else 'Your differentiation level is a real edge, but it only converts to a moat if you can explain it in one sentence and prove it in one demo.'} "
            f"The {top_shap} macro signal is the environmental constraint you cannot control."
        )

    # ── 2. Key Driver — the one thing that determines success or failure ──
    kd_map = {
        'liquidity': (
            f"Supply acquisition speed — not product quality, not marketing spend, not team size. "
            f"A marketplace dies the moment supply dries up, and it always dries up before demand does. "
            f"If you cannot guarantee active supply within 60 days of launch in {country}, "
            f"the demand side will not wait and will not return."
        ),
        'differentiation': (
            f"Switching cost creation before a competitor reaches feature parity. "
            f"{'With ' + diff_label + ' differentiation (' + str(diff) + '/5), your product is still a feature waiting to be cloned — not a platform with embedded workflow dependencies.' if diff <= 3 else 'Your ' + str(diff) + '/5 differentiation score creates real switching cost potential, but only if it survives first contact with how customers actually use the product.'} "
            f"The question is not whether you have an edge — it is whether that edge survives 12 months of competitor response."
        ),
        'regulatory': (
            f"Licensing timeline management — not product development, not fundraising. "
            f"{'In ' + sector + '/' + country + ', regulatory approval is the longest-lead-time item in your build. Every founder who skips this realization is solving the wrong problem first.' if reg == 'high' else 'Regulatory risk is manageable here, which converts into a moat once you clear it and competitors have not started.'}"
        ),
        'churn': (
            f"Month-2 retention rate — not acquisition volume. "
            f"Subscription models that cannot survive greater than 15% monthly churn will never reach profitable unit economics regardless of how fast they acquire. "
            f"The product loop that makes cancellation feel like a loss must exist before you scale acquisition."
        ),
        'capital': (
            f"Capital efficiency before product completion — hardware timelines and bill-of-materials costs are brutally unforgiving. "
            f"Every feature added at {'idea' if stage == 'idea' else stage} stage costs 3x what it will cost post-series-A to undo. "
            f"MVP here means minimum viable prototype that proves the single most uncertain assumption, not a finished product."
        ),
        'scalability': (
            f"Process repeatability — the first 10 clients cannot require 10 different delivery processes. "
            f"Service models that cannot templatize delivery cannot scale without proportional headcount growth. "
            f"The test: can someone you hired two weeks ago deliver this without you explaining it?"
        ),
        'execution': (
            f"Speed to first paid customer — not user signups, not pilots, not LOIs. Paying customers. "
            f"{'In ' + regime_r + ' conditions, ' + ('everyone is capital-constrained — first revenue resets the runway clock.' if in_friction else 'moving fast matters more than moving perfectly — the market will not wait.') if True else ''}"
        ),
    }
    key_driver = kd_map.get(dominant_risk, kd_map['execution'])

    # ── 3. Main Risk — specific failure mode from the signal profile ──
    if dominant_risk == 'liquidity' and comp == 'high':
        main_risk = (
            f"Side-switching before liquidity threshold. Supply-side participants defect to a better-capitalized competitor "
            f"at exactly the moment you think you have momentum — typically 4-6 months post-launch. "
            f"The signal that precedes this is flat supply growth despite increasing demand."
        )
    elif dominant_risk == 'regulatory' and reg == 'high':
        main_risk = (
            f"Regulatory delay consuming runway. "
            f"{'CBE licensing in Egypt takes 12-18 months; most founders budget 6. The gap is what kills companies, not the rejection.' if sector == 'fintech' and country == 'EG' else 'Licensing timelines in regulated sectors are unpredictable. The risk is not rejection — it is indefinite delay while your burn rate continues.'}"
        )
    elif diff <= 2 and comp == 'high':
        main_risk = (
            f"Feature commoditization within 6 months of visible traction. "
            f"With {'minimal' if diff == 1 else 'low'} differentiation in a high-competition space, "
            f"any funded competitor can replicate your core feature before you have locked in distribution. "
            f"Distribution speed is the only defense — and it is a race you are already behind in."
        )
    elif stage in ('idea', 'validation') and in_friction:
        main_risk = (
            f"Capital exhaustion before proof of willingness-to-pay. "
            f"Raising follow-on funding at {stage} stage in {regime_r} is structurally harder — "
            f"investors are pattern-matching to contraction risk, not opportunity. "
            f"Without paying customers in the current runway, the next round is a reset, not a continuation."
        )
    elif bm == 'marketplace' and ready <= 2:
        main_risk = (
            f"Demand creation cost. With a market readiness score of {ready}/5, "
            f"you are building a market from scratch, not entering one. "
            f"That costs 5-10x more than demand capture and takes 3-5x longer — "
            f"and the market signals do not price that in."
        )
    elif bm == 'subscription' and comp == 'high' and seg == 'b2c':
        main_risk = (
            f"Churn acceleration driven by affordability pressure. "
            f"In {regime_r} conditions, B2C subscription cancellation rates increase non-linearly. "
            f"High competitive intensity means you cannot raise price to compensate. "
            f"The model only survives if switching cost is high enough that cancellation feels like a genuine loss."
        )
    else:
        main_risk = (
            f"{'Competition eroding pricing power before you reach sustainable unit economics.' if comp == 'high' else 'Slow customer adoption requiring higher CAC than the model can sustain at current stage.'} "
            f"{'The ' + regime_r + ' macro environment amplifies this — ' + ('buyers are price-sensitive and less tolerant of unproven tools.' if in_friction else 'you need to move faster than the market attracts competitors.') if True else ''}"
        )

    # ── 4. Counterpoint — aggressively challenge the comfortable interpretation ──
    if signal_tier == 'Strong':
        counterpoint = (
            f"The strong signal is not the asset — it is the liability. "
            f"{'A ' + sector + ' marketplace at Strong signal in ' + country + ' will attract three better-funded competitors within the next 12 months, all running the same MIDAN analysis and reaching the same conclusion.' if bm == 'marketplace' else 'Strong market conditions have systematically masked weak unit economics until the market turns — and ' + country + ' markets turn faster and harder than forecasts suggest.'} "
            f"The founders who get crushed in Strong markets are the ones who confused market signal with company signal and raised capital before proving the unit economics."
        )
    elif signal_tier == 'Moderate':
        counterpoint = (
            f"Moderate is the most dangerous tier — strong enough to convince yourself to build, "
            f"not strong enough to attract the capital needed to actually win. "
            f"{'At ' + diff_label + ' differentiation (' + str(diff) + '/5), every single investor conversation will require justifying the edge, and most will pass.' if diff <= 3 else 'Even strong differentiation cannot compensate for moderate market conditions when a well-capitalized competitor decides to enter.'} "
            f"The trap is spending 18 months and 200K building something that 20 customer interviews in the first 2 weeks would have proven or killed."
        )
    elif signal_tier == 'Mixed':
        counterpoint = (
            f"Mixed signals do not mean 'wait for clarity' — they mean the market is actively transitioning "
            f"and you are building for conditions that may not exist by the time you launch. "
            f"{'In ' + regime_r + ', mixed signals resolve downward before they resolve upward — the base rate is deterioration, not stabilization.' if in_friction else 'The SARIMA 90-day trend (' + ('positive' if sarima_trend > 0.5 else 'negative') + ') is the real signal — the regime classification is a lagging indicator.'} "
            f"Committing capital to a full build in Mixed conditions is a bet on timing that most founders lose."
        )
    else:
        counterpoint = (
            f"The market is not telling you to wait — it is telling you this specific idea does not fit this specific moment. "
            f"{'With ' + diff_label + ' differentiation (' + str(diff) + '/5) in a Weak signal environment, there is no defensible market position to build toward — you are racing to prove a thesis that the market has already rejected.' if diff <= 3 else 'Even with your differentiation level, Weak market signals in ' + country + '/' + sector + ' mean buyer discretionary spending is against you — the product needs to be a painkiller, not a vitamin, and it needs to be provably cheaper than the status quo.'} "
            f"The founders who survive Weak signals are the ones who stopped spending before the runway ended, not the ones who pushed through."
        )

    # ── 5. Differentiation Insight — why this idea behaves differently ──
    bm_baselines = {
        'marketplace': f"Most {sector} ideas in {country} default to marketplace models because the informal market already exists",
        'saas':        f"SaaS in {sector}/{country} typically fails because founders build for global scale and price for local spending power",
        'commission':  f"Commission models in {sector} are common — the infrastructure is understood and the risk profile is predictable",
        'subscription':f"Subscription in {sector} is a bet on retention mechanics, not acquisition volume",
        'service':     f"Service models are the conventional starting point in {sector}/{country} — lower capital requirement, slower scale curve",
        'hardware':    f"Hardware in {sector} is rare in {country} — capital barriers push most teams toward software pivots",
    }
    baseline = bm_baselines.get(bm, f"Most {sector} ideas in {country} follow predictable patterns")

    if diff >= 5:
        diff_ending = (
            f"has an exceptional edge ({diff}/5) — a moat that incumbents cannot replicate cheaply. "
            f"That buys time. Whether the team can use that time to build distribution before the moat erodes is the actual variable."
        )
    elif diff == 4:
        diff_ending = (
            f"has a meaningful edge ({diff}/5) — real switching cost potential from {moat_source}. "
            f"The question is whether that edge survives first contact with how customers actually use the product versus how it was designed."
        )
    elif diff == 3:
        diff_ending = (
            f"is moderately differentiated ({diff}/5) — which means it will be copied within 12 months of any visible traction. "
            f"The moat, if any, comes from {moat_source}, not from the product features themselves."
        )
    else:
        diff_ending = (
            f"is {'minimally' if diff == 1 else 'weakly'} differentiated ({diff}/5) — which means it depends entirely on execution speed "
            f"and distribution advantages that are not yet evident. "
            f"The risk is that {moat_source} requires time to develop, and the competitive window may close first."
        )

    differentiation_insight = f"{baseline}. This one {diff_ending}"

    # ── 6. What Matters Most — the single make-or-break variable ──
    _wmm_map = {
        ('marketplace', 'idea'):      f"Whether you can manually broker the first 50 transactions in {country} without any product — if that does not happen, the platform never gets liquidity data to automate from.",
        ('marketplace', 'validation'):f"Supply-side density within 60 days — if active suppliers do not reach critical mass before demand loses patience, the chicken-and-egg problem kills the model.",
        ('marketplace', 'mvp'):       f"Transaction completion rate on the first 100 deals — not volume, not GMV, not user counts. Completion rate tells you whether the market actually clears.",
        ('marketplace', 'growth'):    f"Take-rate defensibility as competitors enter — growth-stage marketplaces that cannot defend commission structure collapse within 18 months of visible traction.",
        ('saas', 'idea'):             f"Whether you can get a {seg_label} customer to describe a specific workflow they would change today if your product existed — without that, you are building a solution looking for a problem.",
        ('saas', 'validation'):       f"Signing 3 paying pilots — not free, not LOIs, paying — within the next 30 days. Without that proof, every subsequent decision is built on assumption.",
        ('saas', 'mvp'):              f"Month-2 retention — if more than 20% of users disengage after the first month, the product loop is broken and no amount of acquisition spend will fix it.",
        ('saas', 'growth'):           f"Net Revenue Retention above 110% — below that, you are on a treadmill that requires continuous acquisition to compensate for churn and contraction.",
        ('commission', 'idea'):       f"{'CBE regulatory pre-approval timeline — if you cannot map the exact licensing path before writing code, you are planning around an unknown that can kill the company.' if sector == 'fintech' else 'Transaction volume per partner per month — commission economics only work above a specific GMV threshold that most early-stage models underestimate.'}",
        ('commission', 'validation'): f"Proving a single transaction cycle end-to-end with real money — not a simulation, not a demo, actual money changing hands through your system.",
        ('service', 'idea'):          f"Whether the first client requires you personally to deliver — if it does, you have not built a service, you have built a job, and it will never scale.",
        ('hardware', 'idea'):         f"Bill of materials cost at 1,000 units — not prototype cost, not small-batch cost. If the unit economics do not work at 1,000 units, they will not work at 10,000 either.",
    }
    wmm_key = (bm, stage) if (bm, stage) in _wmm_map else (bm, 'idea')
    if wmm_key not in _wmm_map:
        # Generic fallback by dominant_risk
        wmm_fallback = {
            'liquidity':       f"Whether the supply side shows up without you manually recruiting every participant — if it requires your personal effort, it will not scale.",
            'differentiation': f"Whether {seg_label} customers choose you over the default alternative in a blind test — if the answer is no, the differentiation is in your head, not in the market.",
            'regulatory':      f"The regulatory clearance timeline — it is the only uncontrollable variable that can terminate the company regardless of how good the product is.",
            'churn':           f"Month-2 retention — the subscription model's entire unit economics hinges on this single number, and most founders do not measure it until it is already fatal.",
            'capital':         f"Whether the prototype proves the single most uncertain assumption before you spend the next tranche — hardware projects that skip this die at manufacturing scale.",
            'scalability':     f"Whether delivery can be templated without your involvement by week 8 — if not, you are building a consulting practice that cannot be valued as a company.",
            'execution':       f"Getting a paying customer to return a second time — the first transaction proves willingness to pay; the second proves the product delivers on the promise.",
        }
        what_matters_most = wmm_fallback.get(dominant_risk, f"Whether this idea solves a problem urgent enough that {seg_label} customers would pay for it without being asked twice.")
    else:
        what_matters_most = _wmm_map[wmm_key]

    # ── 7. Counter-Thesis — challenges the CORE assumption of the idea ──
    _ct_sector_map = {
        'fintech':    f"You think this is a product problem. In {country}, fintech is a trust problem — and trust is built through regulated institutions, not better UX. A superior product without a banking license or institutional partnership will not reach adoption at scale.",
        'healthtech': f"You think the friction is distribution. In {country}, the real friction is institutional procurement — hospitals and clinics buy from relationships, not from pitches, and the decision cycle is 9-18 months regardless of how good the product is.",
        'edtech':     f"You think the problem is content quality or delivery. The real problem is willingness-to-pay — edtech in {country} competes with free, and free always wins until you find the narrow segment of users with a measurable career ROI from the skill.",
        'ecommerce':  f"You think the problem is the platform. The real problem is last-mile logistics and returns economics — the unit economics of ecommerce in {country} are structurally negative unless you own or deeply control delivery.",
        'logistics':  f"You think optimization is the value proposition. Logistics clients in {country} buy reliability, not efficiency — a 95% on-time rate beats 20% cheaper every time, and reliability requires assets, not algorithms.",
        'agritech':   f"You think farmers need better tools. Farmers in {country} need predictable income first — any product that does not directly solve the income volatility problem will see adoption spike and collapse with every harvest cycle.",
    }
    if sector in _ct_sector_map:
        counter_thesis = _ct_sector_map[sector]
    elif bm == 'marketplace':
        counter_thesis = (
            f"You think the problem is building the platform. "
            f"The real problem is that someone will have to manually broker every transaction for the first 6 months to make the platform feel like it works — "
            f"and that labor cost is not in your financial model. Marketplaces fail because the liquidity bootstrapping phase is manually intensive and founders treat it as a tech problem."
        )
    elif bm in ('saas', 'subscription'):
        counter_thesis = (
            f"You think differentiation wins deals. "
            f"In {country}, local distribution and enterprise relationships beat better product 8 times out of 10. "
            f"{'Your ' + diff_label + ' differentiation (' + str(diff) + '/5) is not the asset — your ' + country + ' network is, and if you do not have it, the go-to-market will fail regardless of product quality.' if diff < 4 else 'Even at ' + str(diff) + '/5 differentiation, a well-connected incumbent can replicate the core feature in 6 months and use existing relationships to displace you before you have distribution.'}"
        )
    elif bm == 'commission':
        counter_thesis = (
            f"You think volume will come once the platform is live. "
            f"Commission models require a minimum viable volume threshold before they generate meaningful revenue — "
            f"and reaching that threshold requires either massive marketing spend or a pre-existing captive network. "
            f"The question is which one you have, and most early-stage commission models have neither."
        )
    else:
        counter_thesis = (
            f"The assumption embedded in this idea is that the market needs what is being built. "
            f"The real test is whether {seg_label} customers in {country} are currently solving this problem with a workaround — "
            f"and if the workaround costs them less (in time and money) than your solution will, the adoption curve will be longer and more expensive than the financial model assumes."
        )

    return {
        'strategic_interpretation': si,
        'key_driver':               key_driver,
        'main_risk':                main_risk,
        'counterpoint':             counterpoint,
        'differentiation_insight':  differentiation_insight,
        'what_matters_most':        what_matters_most,
        'counter_thesis':           counter_thesis,
    }


def _generate_l4_reasoning(
    sector: str, country: str, regime: str, conf: float,
    sarima_trend: float, idea_features: dict, idea_signal_data: dict,
    shap_dict: dict, tas: float, signal_tier: str,
    a2_comps: list, a4_sentiment: str, logs: list,
) -> dict:
    """
    Layer 4 — Strategic Reasoning Engine.
    Tries LLM first; falls back to rich conditional logic.
    """
    bm    = idea_features.get('business_model', 'other')
    diff  = idea_features.get('differentiation_score', 3)
    stage = idea_features.get('stage', 'idea')
    diff_label = {1: 'minimal', 2: 'low', 3: 'moderate', 4: 'strong', 5: 'exceptional'}.get(diff, 'moderate')
    bm_label   = bm.upper()
    seg_label  = idea_features.get('target_segment', 'b2c').upper()

    result = None
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if GROQ_CLIENT and groq_key and groq_key != "dummy":
        result = _l4_reasoning_llm(
            sector, country, regime, conf, sarima_trend,
            idea_features, idea_signal_data, shap_dict, tas, signal_tier,
            a2_comps, a4_sentiment, diff_label, bm_label, seg_label,
        )
        if result:
            logs.append("[L4] Strategic reasoning: LLM path succeeded")
        else:
            logs.append("[L4] LLM reasoning failed — using conditional fallback")
    else:
        logs.append("[L4] No LLM key — using conditional fallback reasoning")

    if not result:
        result = _l4_reasoning_fallback(
            sector, country, regime, conf, sarima_trend,
            idea_features, idea_signal_data, shap_dict, tas, signal_tier,
            diff_label, bm_label, seg_label,
        )

    # Guarantee all 7 fields exist (LLM may have returned only 5)
    for field in ('what_matters_most', 'counter_thesis'):
        if not result.get(field):
            fb = _l4_reasoning_fallback(
                sector, country, regime, conf, sarima_trend,
                idea_features, idea_signal_data, shap_dict, tas, signal_tier,
                diff_label, bm_label, seg_label,
            )
            result[field] = fb.get(field, '')

    return result


# ═══════════════════════════════════════════════════════════════
# AGENT A0 — Idea Evaluation (5 dimensions, 0-10 per dimension)
# ═══════════════════════════════════════════════════════════════

IDEA_DIMENSIONS = ['problem_clarity', 'solution_fit', 'differentiation', 'business_model', 'scalability']


def agent_a0_evaluate_idea(idea_text: str, sector: str, country: str) -> dict:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if GROQ_CLIENT and groq_key and groq_key != "dummy":
        try:
            prompt = dedent(f"""
                You are a VC analyst evaluating a startup idea. Score each dimension 0-10.

                Idea: "{idea_text}"
                Sector: {sector} | Country: {country}

                CRITICAL: If the text is a greeting, random words, or clearly not a startup idea,
                score every dimension exactly 0 with reason "Not a valid startup idea."
                Otherwise score honestly — different ideas MUST get different scores.

                Respond in EXACTLY this JSON (no other text):
                {{"problem_clarity":{{"score":7,"reason":"..."}},"solution_fit":{{"score":6,"reason":"..."}},"differentiation":{{"score":5,"reason":"..."}},"business_model":{{"score":8,"reason":"..."}},"scalability":{{"score":6,"reason":"..."}}}}
            """).strip()
            resp = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant", temperature=0.2, max_tokens=400
            )
            raw = resp.choices[0].message.content.strip()
            if '```' in raw:
                raw = raw.split('```')[1]
                if raw.startswith('json'): raw = raw[4:]
            data = json.loads(raw)
            scores  = {d: max(0, min(10, int(data.get(d, {}).get('score', 5)))) for d in IDEA_DIMENSIONS}
            reasons = {d: data.get(d, {}).get('reason', '') for d in IDEA_DIMENSIONS}
            return {'scores': scores, 'reasons': reasons, 'idea_score': int(sum(scores.values()) / len(scores) * 10)}
        except Exception:
            pass

    # Keyword heuristic fallback
    t = idea_text.lower()
    scores = {}
    reasons = {}

    prob_hits = sum(1 for w in ['problem','issue','challenge','pain','struggle','need','gap','inefficient','expensive','slow','difficult'] if w in t)
    scores['problem_clarity'] = min(10, 3 + prob_hits * 2)
    reasons['problem_clarity'] = 'Clear problem statement detected' if prob_hits >= 2 else 'Problem needs more specificity'

    sol_hits = sum(1 for w in ['app','platform','tool','system','service','automate','connect','enable','simplify','streamline','reduce'] if w in t)
    scores['solution_fit'] = min(10, 3 + sol_hits * 2)
    reasons['solution_fit'] = 'Solution approach is clear' if sol_hits >= 2 else 'Describe how the solution works'

    diff_hits = sum(1 for w in ['first','only','unique','unlike','better','faster','cheaper','new','innovative','ai','machine learning'] if w in t)
    scores['differentiation'] = min(10, 2 + diff_hits * 2)
    reasons['differentiation'] = 'Unique angle detected' if diff_hits >= 2 else 'What makes this different from existing solutions?'

    biz_hits = sum(1 for w in ['subscription','saas','commission','fee','pricing','revenue','monetize','b2b','b2c','freemium','marketplace','premium'] if w in t)
    scores['business_model'] = min(10, 3 + biz_hits * 2)
    reasons['business_model'] = 'Revenue model indicated' if biz_hits >= 1 else 'How will this make money?'

    scale_hits = sum(1 for w in ['scale','global','expand','growth','million','region','international','multiple','market','nationwide'] if w in t)
    scores['scalability'] = min(10, 3 + scale_hits * 2)
    reasons['scalability'] = 'Growth potential indicated' if scale_hits >= 1 else 'How will this scale beyond initial market?'

    if len(t.split()) > 30:
        for d in scores: scores[d] = min(10, scores[d] + 1)

    return {'scores': scores, 'reasons': reasons, 'idea_score': int(sum(scores.values()) / len(scores) * 10)}


# ═══════════════════════════════════════════════════════════════
# EXPLANATION LAYER — per-dimension and per-signal explanations
# ═══════════════════════════════════════════════════════════════

def _generate_explanation_layer(
    idea_eval: dict,
    idea_features: dict,
    shap_dict: dict,
    regime: str,
    sector: str,
) -> dict:
    """
    Produces human-readable explanations for every score and signal.
    Each dimension gets: why_given, what_missing, how_improve.
    Each macro signal gets: what_it_means, impact_on_idea, reliability.
    Each idea signal breakdown gets: meaning, weight_note.
    """
    scores   = idea_eval.get('scores', {})
    reasons  = idea_eval.get('reasons', {})
    bm       = idea_features.get('business_model', 'other')
    seg      = idea_features.get('target_segment', 'b2c').upper()
    stage    = idea_features.get('stage', 'idea')
    diff     = idea_features.get('differentiation_score', 3)
    comp     = idea_features.get('competitive_intensity', 'medium')
    reg      = idea_features.get('regulatory_risk', 'medium')
    in_fric  = regime in ('HIGH_FRICTION_MARKET', 'CONTRACTING_MARKET')
    in_grow  = regime in ('GROWTH_MARKET', 'EMERGING_MARKET')

    # ── Dimension explanations ──────────────────────────────────────────────────
    dim_exp = {}

    # problem_clarity
    pc = scores.get('problem_clarity', 5)
    if pc >= 8:
        pc_why     = f"The problem is well-articulated for a specific segment with clear pain signal."
        pc_missing = f"Quantified frequency and cost of the problem are still absent — without those, willingness-to-pay is assumed, not proven."
        pc_improve = f"Run 10 customer interviews specifically asking: how often does this happen, and what do you pay (in time or money) to deal with it today?"
    elif pc >= 6:
        pc_why     = f"The problem is described but the scope is broad — it applies to many segments equally, which makes it hard to prioritize."
        pc_missing = f"A specific segment + specific trigger event + specific current workaround. Without those three, the problem is a category, not a customer pain."
        pc_improve = f"Narrow the problem to a single job-to-be-done for a single customer type and state the current workaround they use."
    else:
        pc_why     = f"The problem statement is vague or implied — the idea leads with the solution, not the pain it solves."
        pc_missing = f"A concrete description of who experiences this, how often, and what it costs them. The problem must be more visceral than the solution."
        pc_improve = f"Rewrite the idea starting with: '[Specific customer] struggles with [specific problem] [frequency/urgency] because [root cause], and today they solve it by [workaround that costs them X].' Then describe your solution."
    dim_exp['problem_clarity'] = {'why': pc_why, 'missing': pc_missing, 'improve': pc_improve, 'score': pc}

    # solution_fit
    sf = scores.get('solution_fit', 5)
    if sf >= 8:
        sf_why     = f"The solution directly addresses the stated problem with a clear mechanism."
        sf_missing = f"Evidence that the mechanism works at your target customer's actual workflow — solution fit at prototype is not the same as solution fit at scale."
        sf_improve = f"Map exactly which step in the customer's current workflow your solution replaces, and get one reference customer to describe how their process would change."
    elif sf >= 5:
        sf_why     = f"The solution approach exists but the fit to the specific pain is loose — it could solve adjacent problems as easily as the stated one."
        sf_missing = f"A clear mechanism: how specifically does this reduce the problem? What input does the customer provide, what output do they get, and how does that change their situation?"
        sf_improve = f"Build a walkthrough of exactly 3 steps the customer takes from problem to resolution using your solution. Every hand-wave is a product risk."
    else:
        sf_why     = f"The solution is underspecified — it is positioned as a category ('an app', 'a platform') rather than a specific mechanism."
        sf_missing = f"What the product actually does, step by step, for the first user in the first session. Without that, you cannot estimate build cost, time-to-value, or churn risk."
        sf_improve = f"Describe the product as a series of user actions: what does the user do first, what does the system do, what does the user receive? That sequence IS the product."
    dim_exp['solution_fit'] = {'why': sf_why, 'missing': sf_missing, 'improve': sf_improve, 'score': sf}

    # differentiation
    dif = scores.get('differentiation', 5)
    if dif >= 8:
        dif_why     = f"A clear and specific differentiating mechanism exists — one that requires meaningful effort or structural advantage to replicate."
        dif_missing = f"Evidence that {seg} customers perceive this differentiation as valuable — internal conviction is not market validation."
        dif_improve = f"Run a blind comparison: show 5 target customers your product and the best alternative without branding. If they still choose yours, the differentiation is real."
    elif dif >= 5:
        dif_why     = f"A differentiation angle exists but is feature-level, not structural — a well-resourced competitor can replicate it within 6-12 months of seeing traction."
        dif_missing = f"A structural moat: network effects, switching costs, regulatory access, or proprietary data. Features are not moats."
        dif_improve = f"For each differentiating feature, ask: 'If {bm.title()} Inc. (with 10x our resources) decided to copy this in 6 months, could they?' If yes, that is not your moat."
    else:
        dif_why     = f"The idea appears to replicate an existing model without a clear reason why customers would switch from current alternatives."
        dif_missing = f"A specific answer to: 'Why would a customer who is satisfied with [current solution] switch to this, pay for this, and recommend this to someone else?'"
        dif_improve = f"Before building anything, conduct 20 interviews with current [competitor] users and ask what would make them switch. Build only what that research reveals."
    dim_exp['differentiation'] = {'why': dif_why, 'missing': dif_missing, 'improve': dif_improve, 'score': dif}

    # business_model
    bms = scores.get('business_model', 5)
    if bms >= 8:
        bm_why     = f"The revenue mechanism is clear and matched to the value delivered — pricing logic follows from customer outcome."
        bm_missing = f"Unit economics at target scale: what is CAC, LTV, and payback period at 100 customers? At 1,000? The model works in concept — does it work in math?"
        bm_improve = f"Build a unit economics spreadsheet: cost to acquire one customer, revenue per customer per year, cost to serve, and the month at which a customer becomes profitable."
    elif bms >= 5:
        bm_why     = f"A revenue mechanism exists but the pricing logic and customer segment alignment are not tight — who pays, how much, and why they keep paying is still unclear."
        bm_missing = f"A clear answer to: who writes the first check, for how much, based on what value promise, and what makes them renew?"
        bm_improve = f"Map three pricing scenarios: low/mid/high. For each, calculate how many {seg} customers you need to cover your burn. The scenario that requires the fewest customers is your starting pricing strategy."
    else:
        bm_why     = f"The monetization approach is either undefined or mismatched with the {bm} model type and {seg} segment."
        bm_missing = f"The answer to: 'If I offered this to 10 people right now, how many would pay, and how much?' — that number is the baseline for everything."
        bm_improve = f"Run a pre-sell experiment: offer the product (before it exists) for a specific price to 10 target customers. Paying customers validate the model. Non-payment tells you something critical before you burn development time."
    dim_exp['business_model'] = {'why': bm_why, 'missing': bm_missing, 'improve': bm_improve, 'score': bms}

    # scalability
    sc = scores.get('scalability', 5)
    if sc >= 8:
        sc_why     = f"The model has identifiable scale levers — revenue can grow without proportional cost growth."
        sc_missing = f"The specific constraint that breaks first at 10x: is it infrastructure, team, support load, or a dependency on a scarce resource?"
        sc_improve = f"Run a '10x stress test': if you had 10x current users tomorrow, what breaks first? That is your product roadmap priority, not your feature wishlist."
    elif sc >= 5:
        sc_why     = f"Scale potential exists but is constrained by a variable that has not been addressed — likely {'headcount' if bm == 'service' else 'unit economics' if bm in ('marketplace', 'commission') else 'technical architecture'}."
        sc_missing = f"A clear path from current model to one that does not require proportional cost growth — {'specifically, how does delivery get templated without founder involvement?' if bm == 'service' else 'what is the mechanism that makes the marginal customer cheaper to serve than the previous one?'}"
        sc_improve = f"Identify the bottleneck that doubles your cost when you double customers. Fix that before scaling acquisition — otherwise you are accelerating toward a structural problem."
    else:
        sc_why     = f"The model as described scales linearly with cost — each new customer requires proportional effort, which caps margin and makes the business difficult to value as a venture."
        sc_missing = f"A leverage mechanism: network effects, software automation, or a template that allows delivery without the founder's direct involvement."
        sc_improve = f"Ask: 'What has to be true about the product for it to serve 100x current customers with 5x current staff?' The gap between current state and that answer is the scale architecture work."
    dim_exp['scalability'] = {'why': sc_why, 'missing': sc_missing, 'improve': sc_improve, 'score': sc}

    # ── Signal explanations (macro SHAP + idea breakdown) ─────────────────────
    sig_exp = {}

    # Macro SHAP signals
    _shap_descriptions = {
        'velocity_yoy': {
            'what_it_means': "Year-over-year deal velocity in this sector — how fast investment and transaction activity is growing or shrinking.",
            'high_impact':   "High velocity means capital is flowing into this sector, which validates the opportunity but also means competitors are arriving with it.",
            'low_impact':    "Low velocity means the market is contracting or stagnant — building into a shrinking deal flow requires a specific counter-cyclical thesis.",
            'reliability':   "High reliability — this signal tracks actual transaction data, not sentiment. It is the most predictive single signal for near-term market activity.",
        },
        'gdp_growth': {
            'what_it_means': "Macroeconomic output growth rate — determines whether buyers have expanding or contracting purchasing power.",
            'high_impact':   "Strong GDP growth expands the addressable market and loosens enterprise budget constraints.",
            'low_impact':    "Weak GDP growth means discretionary spending contracts — B2C and non-essential B2B products face structural headwinds regardless of quality.",
            'reliability':   "Medium reliability — GDP is a lagging indicator. Actual market behavior can diverge from GDP by 6-12 months.",
        },
        'inflation': {
            'what_it_means': "Rate of price increases in the economy — affects cost structures, consumer purchasing power, and investor risk appetite.",
            'high_impact':   "High inflation compresses real purchasing power and raises your operational costs. For B2C models, it makes subscription pricing decisions harder without alienating customers.",
            'low_impact':    "Low inflation creates stable pricing environments — favorable for long-term contract negotiations and predictable unit economics.",
            'reliability':   "High reliability for cost-structure analysis — but inflation affects different sectors asymmetrically. Sector-specific context matters.",
        },
        'macro_friction': {
            'what_it_means': "Composite signal measuring structural barriers to economic activity — combines inflation pressure, unemployment, and sector-specific regulatory overhead.",
            'high_impact':   "High friction means fewer transactions complete, conversion cycles lengthen, and buyer risk tolerance drops. Startups with long payback periods are especially exposed.",
            'low_impact':    "Low friction creates a permissive environment for new entrants — deals move faster, pilots convert more easily, and capital is less risk-averse.",
            'reliability':   "High reliability as a composite signal — it synthesizes multiple indicators and is more robust than any single factor.",
        },
        'capital_concentration': {
            'what_it_means': "How concentrated capital and market share are in your sector — whether a few large players dominate or the market is fragmented.",
            'high_impact':   "High concentration means incumbents have scale advantages, distribution lock-in, and lobbying power. Entering requires either a non-overlapping niche or a regulatory wedge.",
            'low_impact':    "Low concentration means the market is fragmented — easier entry but also lower barriers for competitors. First-mover advantage is time-limited.",
            'reliability':   "Medium reliability — concentration data is often measured annually and can be stale. The direction of change matters more than the current level.",
        },
    }
    for feat, share in shap_dict.items():
        desc = _shap_descriptions.get(feat, {})
        pct  = f"{share:.0%}"
        sig_exp[f"market_{feat}"] = {
            'label':         feat.replace('_', ' ').title(),
            'share':         pct,
            'what_it_means': desc.get('what_it_means', f"Macro signal: {feat.replace('_', ' ')}."),
            'impact':        desc.get('high_impact' if share > 0.20 else 'low_impact', "This signal is contributing to the current regime classification."),
            'reliability':   desc.get('reliability', "Medium reliability."),
        }

    # Idea signal breakdown labels
    _idea_signal_labels = {
        'model_regime_fit':   ('Baseline Model-Regime Fit', "How well your business model type (e.g., marketplace, SaaS) structurally fits the current market regime. This is the foundation before any idea-specific adjustments."),
        'differentiation':    ('Differentiation Effect',    "The positive or negative adjustment based on how differentiated your idea is. Higher differentiation creates pricing power and moat potential — lower scores penalize the baseline."),
        'stage_readiness':    ('Stage Readiness Adjustment', "How your current development stage (idea/validation/MVP/growth) interacts with market conditions. Early-stage ideas in friction markets get penalized; growth-stage in growth markets get boosted."),
        'competition_adj':    ('Competition Adjustment',     "The signal impact of competitive intensity. High competition in winner-takes-all models (like marketplaces) is penalized heavily; in SaaS it is less severe."),
        'regulatory_adj':     ('Regulatory Risk Adjustment', "The signal cost of regulatory exposure. Fintech and healthtech have sector-specific penalties because licensing timelines directly affect runway planning."),
        'market_readiness_adj': ('Market Readiness Adjustment', "How ready the market is to adopt your solution. Low readiness means you are creating demand (expensive); high readiness means capturing existing demand (efficient)."),
    }
    for k, v in _idea_signal_labels.items():
        label, meaning = v
        sig_exp[f"idea_{k}"] = {
            'label':   label,
            'meaning': meaning,
        }

    return {
        'dimension_explanations': dim_exp,
        'signal_explanations':    sig_exp,
    }


# ═══════════════════════════════════════════════════════════════
# ML PIPELINE HELPERS (existing models)
# ═══════════════════════════════════════════════════════════════

def compute_shap(lgb_model, x_scaled_row):
    """
    Compute normalized SHAP feature importance shares.
    Each value = fraction of total importance [0, 1].
    Soft-cap: no single feature can claim >65% of total signal
    to prevent unrealistic dominance in display.
    """
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(lgb_model)
    sv = explainer.shap_values(x_scaled_row)
    if isinstance(sv, list):
        arr = np.mean([np.abs(s) for s in sv], axis=0)[0]
    elif hasattr(sv, 'ndim') and sv.ndim == 3:
        arr = np.abs(sv[0]).mean(axis=-1)
    else:
        arr = np.abs(sv)[0]

    raw   = dict(zip(FEATURES, arr))
    total = sum(raw.values())
    if total <= 0:
        return {k: round(1.0 / len(FEATURES), 4) for k in FEATURES}

    # Normalize to fractional shares
    shares = {k: v / total for k, v in raw.items()}

    # Soft-cap: redistribute anything above MAX_SHARE to uncapped features
    MAX_SHARE = 0.65
    overcapped = {k: v for k, v in shares.items() if v > MAX_SHARE}
    if overcapped:
        overflow   = sum(v - MAX_SHARE for v in overcapped.values())
        free_keys  = [k for k in shares if shares[k] <= MAX_SHARE]
        bonus      = overflow / len(free_keys) if free_keys else 0
        for k in shares:
            shares[k] = MAX_SHARE if shares[k] > MAX_SHARE else min(MAX_SHARE, shares[k] + bonus)

    return {k: round(float(v), 4) for k, v in shares.items()}


# ═══════════════════════════════════════════════════════════════
# FULL INFERENCE — 4-Layer Hybrid Pipeline
# ═══════════════════════════════════════════════════════════════

def run_inference(sector: str, country: str, idea_text: str = "", logs: list = None) -> dict:
    """
    4-Layer hybrid inference:
      L1 — Idea feature extraction  (idea-specific, LLM-powered)
      L2 — Macro ML pipeline        (DBSCAN→FCM→SVM→SHAP→SARIMA)
      L3 — Idea signal scoring      (regime×model×segment fit)
      L4 — Composite TAS            (new formula ensures per-idea variability)
    """
    if logs is None:
        logs = []
    sec = sector.lower()

    # ── L1: Idea Feature Extraction ───────────────────────────────────────────
    idea_features = extract_idea_features(idea_text, sec)
    bm_label  = idea_features['business_model']
    seg_label = idea_features['target_segment'].upper()
    diff      = idea_features['differentiation_score']
    diff_label = {1: 'minimal', 2: 'low', 3: 'moderate', 4: 'strong', 5: 'exceptional'}.get(diff, 'moderate')
    logs.append(f"[L1] Idea features: bm={bm_label} | seg={seg_label} | diff={diff}/5 ({diff_label}) | stage={idea_features['stage']}")

    # ── L2A: Macro vector construction ───────────────────────────────────────
    macro = COUNTRY_MACRO_DEFAULTS.get(country.upper(), {'inflation': 10.0, 'gdp_growth': 3.0, 'unemployment': 7.0})
    base_inflation, base_gdp, unemployment = macro['inflation'], macro['gdp_growth'], macro['unemployment']
    logs.append(f"[L2A] Base macro: inflation={base_inflation}% | GDP={base_gdp}%")

    eff_inf_offset, gdp_boost, velocity = SECTOR_EFF_MACRO.get(sec, SECTOR_EFF_MACRO['other'])
    scale      = base_inflation / 33.9
    inflation  = float(np.clip(eff_inf_offset * scale, 1.0, 100.0))
    gdp_growth = float(base_gdp + gdp_boost)
    macro_fric = float(np.clip(inflation + unemployment - gdp_growth, -50, 100))
    cap_conc   = SECTOR_MEDIANS.get(sec, SECTOR_MEDIANS['other'])

    x_raw    = np.array([[inflation, gdp_growth, macro_fric, float(cap_conc), velocity]])
    x_scaled = scaler.transform(x_raw)
    x_pca    = pca.transform(x_scaled)
    logs.append(f"[L2A] Effective macro: inflation={inflation:.1f}% | GDP={gdp_growth:.1f}% | friction={macro_fric:.1f}")

    # ── L2B: SVM Classification ───────────────────────────────────────────────
    pred_enc   = svm.predict(x_scaled)[0]
    proba      = svm.predict_proba(x_scaled)[0]
    svm_regime = le.inverse_transform([pred_enc])[0]
    svm_conf   = float(proba.max())
    regime, conf = enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth, macro_fric, velocity)
    logs.append(f"[L2B] SVM: {svm_regime} ({svm_conf:.1%}) → Final: {regime} ({conf:.1%})")

    # ── L2C: SHAP (macro signal explainability) ───────────────────────────────
    shap_dict = compute_shap(lgb, x_scaled)
    xai_score = float(conf * np.mean(list(shap_dict.values())))
    top_feat  = max(shap_dict, key=shap_dict.get)
    logs.append(f"[L2C] SHAP top macro signal: {top_feat} ({shap_dict[top_feat]:.3f})")

    # ── L2D: SARIMA Forecast ──────────────────────────────────────────────────
    sarima_trend = 0.50
    drift_flag   = False
    if sec in sarima_results:
        fc_raw       = sarima_results[sec]['forecast_mean']
        fc           = [max(0, v) for v in fc_raw]
        fc_mean      = float(np.mean(fc))
        sarima_trend = float(np.clip(fc_mean / 50.0, 0.15, 0.90))
        drift_flag   = sarima_results[sec]['drift_flag']
        logs.append(f"[L2D] SARIMA: mean={fc_mean:.1f} → trend={sarima_trend:.2f} | drift={drift_flag}")
    else:
        logs.append(f"[L2D] No SARIMA model for {sec} — neutral trend=0.50")

    if drift_flag:
        logs.append("[L2D] ⚠ DRIFT DETECTED — market patterns shifting, regime classification less reliable")

    # ── L3: Idea Signal Scoring (PRIMARY VARIABILITY DRIVER) ─────────────────
    # v2: context-aware — BM type determines which factors are weighted most
    idea_signal_data = compute_idea_signal(idea_features, regime, sector=sec)
    idea_signal      = idea_signal_data['idea_signal']
    dominant_risk    = idea_signal_data.get('dominant_risk', 'execution')
    logs.append(
        f"[L3] Idea signal: {idea_signal:.3f} | bm_type={bm_label} | "
        f"dominant_risk={dominant_risk} | "
        f"base_fit={idea_signal_data['breakdown']['model_regime_fit']:.3f} | "
        f"diff_effect={idea_signal_data['breakdown']['differentiation']:+.3f} | "
        f"stage={idea_signal_data['breakdown']['stage_readiness']:+.3f}"
    )

    # ── L4a: Composite TAS Score ──────────────────────────────────────────────
    tas = round(
        conf         * 0.30   # Market regime confidence
        + sarima_trend * 0.20  # 90-day SARIMA outlook
        + idea_signal  * 0.35  # Idea-specific signal ← PRIMARY variability driver
        + xai_score    * 0.15, # SHAP macro context
        3
    )
    signal_tier = _signal_tier(tas)
    logs.append(
        f"[L4a] TAS = {conf:.2f}×0.30 + {sarima_trend:.2f}×0.20 + "
        f"{idea_signal:.2f}×0.35 + {xai_score:.2f}×0.15 = {tas:.3f} → {signal_tier}"
    )

    # ── Combined signals for display (macro SHAP + idea breakdown) ────────────
    combined_signals = {
        **{f"market_{k}": round(float(v), 4) for k, v in shap_dict.items()},
        **{f"idea_{k}": round(abs(float(v)), 4) for k, v in idea_signal_data['breakdown'].items()},
    }

    # ── Agent A2: Competitor context ──────────────────────────────────────────
    a2_comps = ["Traditional incumbents", "Local SMEs"]
    sector_comps = comps_data.get(sec, [])
    if isinstance(sector_comps, list) and sector_comps:
        a2_comps = [c.get("Company", "Competitor") for c in sector_comps[:2]]

    # ── Agent A4: Sentiment ────────────────────────────────────────────────────
    a4_sentiment = "Neutral"
    if sents_data:
        pos = sum(1 for s in sents_data if s.get('sentiment') == 'positive')
        neg = sum(1 for s in sents_data if s.get('sentiment') == 'negative')
        if pos > neg * 1.5:   a4_sentiment = "Positive"
        elif neg > pos * 1.5: a4_sentiment = "Negative"

    logs.append(f"[A2/A4] {len(a2_comps)} competitors | Sentiment: {a4_sentiment}")

    # ── L4b: Strategic Reasoning Engine ───────────────────────────────────────
    # Replaces the old A7 "3-sentence synthesis" with a true interpretation layer.
    # Produces 5 structured reasoning fields that VARY by BM type + regime.
    # SHAP is correctly scoped to market signal explanation only.
    regime_readable = regime.replace('_', ' ').title()
    top3_names      = [k.replace('_', ' ') for k, _ in sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:3]]

    l4_reasoning = _generate_l4_reasoning(
        sector=sec, country=country, regime=regime, conf=conf,
        sarima_trend=sarima_trend, idea_features=idea_features,
        idea_signal_data=idea_signal_data, shap_dict=shap_dict,
        tas=tas, signal_tier=signal_tier,
        a2_comps=a2_comps, a4_sentiment=a4_sentiment, logs=logs,
    )

    # ── Action Plan — go/no-go decision with conditional logic ───────────────
    stage     = idea_features.get('stage', 'idea')
    reg_risk  = idea_features.get('regulatory_risk', 'medium')
    in_growth = regime in ('GROWTH_MARKET', 'EMERGING_MARKET')

    if tas >= 0.76:   # Strong
        decision_badge = "GO — CONDITIONAL"
        action = (
            f"GO — but only if: you can sign 3 paying {seg_label} customers within 30 days without building anything. "
            f"If that fails, the signal is wrong and you stop. "
            f"{'Build toward the supply-side density milestone first — nothing else matters until the marketplace liquidity threshold is reached.' if bm_label == 'marketplace' else 'Ship the smallest possible version in 4 weeks, put it in front of 20 customers, and measure retention by day 14.'} "
            f"If raising: Flat6Labs, 500 MENA, Algebra Ventures — come with evidence, not a pitch deck. "
            f"Watch {top3_names[0]} — if it deteriorates, your regime reclassifies and this decision reverses."
        )
    elif tas >= 0.60:  # Moderate
        decision_badge = "CONDITIONAL — VALIDATE FIRST"
        action = (
            f"DO NOT BUILD YET. Validate the {dominant_risk} assumption first. "
            f"Run 15 structured interviews with {seg_label} buyers in the next 2 weeks — not to confirm the idea, to find the 3 who would pay today. "
            f"{'Map the full regulatory path before writing a line of code — it is the critical path item, not a nice-to-have.' if reg_risk == 'high' else 'Find the single use case with the highest willingness-to-pay and build only that — everything else is scope creep at this stage.'} "
            f"If you cannot find 3 buyers willing to pay in 2 weeks → stop. Do not raise, do not build, do not iterate. The idea needs a different angle."
        )
    elif tas >= 0.44:  # Mixed
        decision_badge = "HIGH RISK — PROVE OR STOP"
        action = (
            f"HIGH RISK. Do not commit capital to a build. "
            f"The {dominant_risk} risk is not a planning item — it is an existential unknown. Resolve it before anything else. "
            f"Run the cheapest possible experiment that either proves or kills the core assumption within 3 weeks. "
            f"{'For a marketplace: manually broker 5 real transactions end-to-end. If you cannot close them, the model does not work.' if bm_label == 'marketplace' else 'For this model: pre-sell the product before building it. If no one pays upfront, no one will pay after launch either.'} "
            f"If the experiment fails → stop building this version. If it succeeds → you have the evidence to raise on."
        )
    else:  # Weak
        decision_badge = "NO-GO — DO NOT BUILD"
        action = (
            f"NO-GO. Do not build this now. "
            f"The {signal_tier} signal at {int(tas*100)}/100 in a {regime.replace('_',' ').title()} means the market is actively working against this model right now. "
            f"{'The ' + dominant_risk + ' risk cannot be resolved by iteration — it requires a structural market change that is not in your control.' if dominant_risk != 'execution' else 'The fundamental assumptions this idea rests on have not been validated and the market environment makes validation expensive.'} "
            f"{'If the idea is right and the timing is wrong: keep it, spend nothing, revisit in 90 days when SARIMA signals change.' if diff >= 3 else 'If you still believe in this: spend 4 weeks talking to 20 customers before writing a single line of code. Let them prove you right — do not prove yourself right by building.'}"
        )

    # ── Slack Webhook ─────────────────────────────────────────────────────────
    action_fired = tas >= 0.70 and in_growth
    if action_fired:
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        if webhook_url and webhook_url.startswith("http"):
            msg = {
                "text": (
                    f"🚀 *MIDAN High-Conviction Signal*\n"
                    f"*Sector:* {sec.title()} ({country}) | *Model:* {bm_label} → {seg_label}\n"
                    f"*Regime:* {regime_readable} | *Signal:* {signal_tier} | *TAS:* {tas:.2f}\n"
                    f"*Idea diff:* {diff_label} ({diff}/5) | *Stage:* {stage} | *Risk:* {dominant_risk}\n"
                    f"*Key insight:* {l4_reasoning['strategic_interpretation'][:200]}..."
                )
            }
            try:
                requests.post(webhook_url, json=msg, timeout=2)
                logs.append("[SLACK] Webhook fired")
            except Exception:
                logs.append("[SLACK] Webhook failed")
        else:
            logs.append("[SLACK] TAS threshold met — webhook not configured")
    else:
        logs.append(f"[SLACK] TAS={tas:.3f} ({signal_tier}) — threshold not met")

    # ── Explanation Layer ─────────────────────────────────────────────────────
    # Generated after idea_eval is available via process_idea — placeholder here.
    # The full explanation layer is computed in process_idea() and injected there.
    logs.append("[L4b] Strategic reasoning engine complete")

    return {
        'regime':           regime,
        'confidence':       conf,
        'tas':              tas,
        'signal_tier':      signal_tier,
        'decision_badge':   decision_badge,
        'sarima_trend':     sarima_trend,
        'xai_score':        xai_score,
        'shap_dict':        shap_dict,
        'drift_flag':       drift_flag,
        'action_fired':     action_fired,
        # L4 reasoning — 7-part strategic output
        'strategic_interpretation': l4_reasoning['strategic_interpretation'],
        'key_driver':               l4_reasoning['key_driver'],
        'main_risk':                l4_reasoning['main_risk'],
        'counterpoint':             l4_reasoning['counterpoint'],
        'differentiation_insight':  l4_reasoning['differentiation_insight'],
        'what_matters_most':        l4_reasoning.get('what_matters_most', ''),
        'counter_thesis':           l4_reasoning.get('counter_thesis', ''),
        'action':                   action,
        # Supporting data
        'idea_features':    idea_features,
        'idea_signal':      idea_signal_data,
        'combined_signals': combined_signals,
        'top_macro_signals':top3_names,
        'dominant_risk':    dominant_risk,
        'x_raw':    x_raw[0],
        'x_scaled': x_scaled[0],
        'x_pca':    x_pca[0],
        'proba':    dict(zip(le.classes_, proba)),
    }


# ═══════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════

api = FastAPI(title="MIDAN AI Decision Engine", version="2.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class IdeaRequest(BaseModel):
    idea: str
    sector: str = "Fintech"
    country: str = "EG — Egypt"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    context: Dict[str, Any]
    messages: List[ChatMessage]


def process_idea(idea_text: str, default_sector: str = "fintech", default_country: str = "EG") -> dict:
    parsed_sec, parsed_ctry, sec_found, ctry_found = agent_a1_parse(idea_text)
    sector_key   = parsed_sec  if sec_found  else default_sector
    country_code = parsed_ctry if ctry_found else default_country

    report    = run_inference(sector_key, country_code, idea_text=idea_text)
    idea_eval = agent_a0_evaluate_idea(idea_text, sector_key, country_code)

    tas_norm  = report["tas"]
    idea_norm = idea_eval["idea_score"] / 100.0
    svs       = int((tas_norm * 0.50 + idea_norm * 0.50) * 100)

    high_market = report["tas"] >= 0.65
    high_idea   = idea_eval["idea_score"] >= 65

    if high_market and high_idea:      quadrant = "GO — Launch"
    elif high_market and not high_idea: quadrant = "Wrong Idea — Right Market"
    elif not high_market and high_idea: quadrant = "Wait or Pivot Market"
    else:                               quadrant = "STOP — Rethink Everything"

    # ── Explanation layer (injected here — has access to both idea_eval and report) ──
    explanation = _generate_explanation_layer(
        idea_eval    = idea_eval,
        idea_features= report["idea_features"],
        shap_dict    = report["shap_dict"],
        regime       = report["regime"],
        sector       = sector_key,
    )

    return {
        "success":         True,
        "sector":          sector_key,
        "country":         country_code,
        "regime":          report["regime"],
        "tas_score":       int(report["tas"] * 100),
        "signal_tier":     report["signal_tier"],
        "decision_badge":  report.get("decision_badge", ""),
        "confidence":      int(report["confidence"] * 100),
        "sarima_trend":    report["sarima_trend"],
        "drift_flag":      report["drift_flag"],
        "action_fired":    report["action_fired"],
        # ── L4 Strategic Reasoning Engine (7-part output) ──────────────
        "strategic_interpretation": report["strategic_interpretation"],
        "key_driver":               report["key_driver"],
        "main_risk":                report["main_risk"],
        "counterpoint":             report["counterpoint"],
        "differentiation_insight":  report["differentiation_insight"],
        "what_matters_most":        report.get("what_matters_most", ""),
        "counter_thesis":           report.get("counter_thesis", ""),
        "dominant_risk":            report["dominant_risk"],
        "action":                   report["action"],
        # ── Supporting signals ──────────────────────────────────────────
        "shap_weights":            {k: float(v) for k, v in report["shap_dict"].items()},
        "top_macro_signals":       report.get("top_macro_signals", []),
        "combined_signals":        report["combined_signals"],
        "idea_features":           report["idea_features"],
        "idea_signal":             round(float(report["idea_signal"]["idea_signal"]), 3),
        "idea_signal_breakdown":   {k: float(v) for k, v in report["idea_signal"]["breakdown"].items()},
        "pca_coords":              report["x_pca"].tolist(),
        "idea_score":              idea_eval["idea_score"],
        "idea_dimensions":         idea_eval["scores"],
        "idea_reasons":            idea_eval["reasons"],
        # ── Explanation Layer (per dimension + per signal) ──────────────
        "dimension_explanations":  explanation["dimension_explanations"],
        "signal_explanations":     explanation["signal_explanations"],
        "svs":                     svs,
        "quadrant":                quadrant,
    }


@api.post("/analyze")
async def analyze_idea(req: IdeaRequest):
    if not MODELS_LOADED:
        raise HTTPException(status_code=500, detail="Models failed to load.")
    sector_key   = SECTOR_LABEL_MAP.get(req.sector, "fintech")
    country_code = req.country.split(" — ")[0]
    if not req.idea or len(req.idea.strip()) < 5:
        raise HTTPException(status_code=400, detail="Idea too short")
    try:
        res = process_idea(req.idea, sector_key, country_code)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── CHAT FALLBACK (no LLM) ────────────────────────────────────────────────────

def _chat_fallback(req: ChatRequest) -> str:
    sector        = req.context.get("sector", "")
    country       = req.context.get("country", "")
    regime        = req.context.get("regime", "").replace("_", " ").title()
    tas           = req.context.get("tas_score", 0)
    tier          = req.context.get("signal_tier", "")
    idea_feat     = req.context.get("idea_features", {})
    bm            = idea_feat.get("business_model", "")
    seg           = idea_feat.get("target_segment", "").upper()
    diff          = idea_feat.get("differentiation_score", 3)
    stage         = idea_feat.get("stage", "idea")
    dominant_risk = req.context.get("dominant_risk", "execution")
    main_risk     = req.context.get("main_risk", "")
    strat_interp  = req.context.get("strategic_interpretation", "")

    user_turns = [m for m in req.messages if m.role == "user"]
    turn_n     = len(user_turns)
    last_msg   = user_turns[-1].content.lower() if user_turns else ""

    # No analysis context yet
    if not sector:
        greet_tokens = {"hi","hello","hey","yo","sup","good morning","good evening"}
        meta_phrases = ["what do you do","who are you","how does this work","what is midan","what can you do"]
        if any(g in last_msg for g in greet_tokens) and len(last_msg.split()) <= 5:
            return ("MIDAN here. Drop your startup idea — sector, problem, and market — "
                    "and I'll run a full 4-layer analysis: market regime, 90-day forecast, idea-fit score, "
                    "and a specific verdict on whether to build now or wait. What are you working on?")
        if any(p in last_msg for p in meta_phrases):
            return ("MIDAN is a hybrid reasoning engine — LLM feature extraction → SVM market classification → "
                    "SARIMA forecasting → SHAP explainability → L4 strategic reasoning. "
                    "The output isn't a dashboard. It's a verdict: what the market signals mean for YOUR specific model, "
                    "what can break it, and what to do about it. Describe what you're building.")
        if turn_n == 1:
            return "What are you building? Give me the idea, sector, and target market — I'll handle the rest."
        return "Give me the idea and I'll analyze it. Sector, problem, and who you're selling to."

    # Has analysis context — use dominant_risk for specificity
    timing = "now is your window" if tas >= 70 else "timing needs validation" if tas >= 50 else "the market isn't there yet"

    # Dominant risk → specific pressure question
    _risk_pressure = {
        'liquidity':       f"How many suppliers and buyers are already active — and what's the minimum viable density to make transactions happen without you manually brokering every deal?",
        'differentiation': f"If I put your product next to the best alternative and removed the logo, what's the one thing {seg} customers would still pay for — and have you tested that with 10 paying customers?",
        'regulatory':      f"Have you mapped the exact CBE/licensing pathway and spoken to a compliance officer? Regulatory timelines in {sector} kill more startups than competition does.",
        'churn':           f"What's the first moment a {seg} customer would cancel — and have you built anything that makes switching painful before they reach that moment?",
        'capital':         f"What's your burn to first revenue milestone, and what happens to the model if it takes 2× as long and costs 3× as much as you planned?",
        'scalability':     f"At 10× your current volume, what breaks first — and is that a systems problem or a people problem?",
        'execution':       f"Who on your team has shipped a product to {seg} customers in {country} before — and if the answer is no one, what's your plan to compress that learning curve?",
    }
    pressure_q = _risk_pressure.get(dominant_risk, f"What's the one assumption in your model that would kill the entire thesis if it's wrong?")

    if any(w in last_msg for w in ["pivot","change direction","switch","alternative","different idea"]):
        return (f"Pivoting in {sector}/{country} keeps the {regime} conditions — those don't move. "
                f"The only pivot worth running is one that directly attacks your {dominant_risk} risk. "
                f"Rebranding the same model under different packaging doesn't fix structural problems. "
                f"{pressure_q}")

    if any(w in last_msg for w in ["compet","rival","who else","already exist","crowded","saturated"]):
        diff_label = {1: 'minimal', 2: 'low', 3: 'moderate', 4: 'strong', 5: 'exceptional'}.get(diff, 'moderate')
        return (f"In {sector}/{country}, your {dominant_risk} risk is the competitive vector — "
                f"not feature parity. At {diff_label} differentiation ({diff}/5), features won't protect you. "
                f"The moat has to be structural: switching costs, regulatory access, or network density. "
                f"{pressure_q}")

    if any(w in last_msg for w in ["fund","invest","raise","capital","vc","angel","pitch","accelerator"]):
        if tas >= 68:
            return (f"At {tas}/100 ({tier}) in a {regime}, you're in a fundable range — but investors will probe your {dominant_risk} risk hard. "
                    f"Target Flat6Labs, 500 MENA, or Algebra Ventures for pre-seed in Egypt; Wamda or BECO for Gulf. "
                    f"Come with 3 months of data that proves you've already started solving that risk. "
                    f"What evidence do you have on {dominant_risk} right now?")
        return (f"At {tas}/100 ({tier}), institutional capital won't close without proof that your {dominant_risk} risk is manageable. "
                f"Bootstrap or use a revenue-generating consulting wedge first — "
                f"10 paying customers rewrites every investor conversation. "
                f"What's the fastest path to first paid revenue without external capital?")

    if any(w in last_msg for w in ["risk","danger","threat","fail","worry","concern"]):
        risk_body = main_risk if main_risk else f"the primary risk archetype here is {dominant_risk} — which is structural, not just operational"
        return (f"{risk_body}. "
                f"Secondary risk at {stage} stage in a {regime}: you run out of runway validating the wrong assumption. "
                f"Keep burn under $5K/month until you have explicit evidence that the {dominant_risk} problem is solved. "
                f"{pressure_q}")

    if any(w in last_msg for w in ["next","first step","what do i do","how do i start","roadmap","plan"]):
        if tas >= 68:
            return (f"Strong signal at {tas}/100 — {timing}. "
                    f"Priority order: 1) Solve the {dominant_risk} risk with evidence this month, not next quarter. "
                    f"2) Ship a 4-week MVP to 20 {seg.lower() if seg else 'target'} customers. "
                    f"3) Apply to Flat6Labs or 500 MENA with that data in hand. "
                    f"{pressure_q}")
        return (f"At {tas}/100 — {timing}. Don't build yet. "
                f"1) 20 customer discovery calls this month, focused on your {dominant_risk} risk. "
                f"2) Find the one use case with the highest willingness-to-pay. "
                f"3) Build only that — no code before evidence. "
                f"{pressure_q}")

    if any(w in last_msg for w in ["pric","monetize","revenue model","charge","subscription","fee"]):
        return (f"For a {bm} model in {sector}/{country}: avoid freemium unless you have viral mechanics. "
                f"{seg} customers respond best to outcome-based pricing — charge a % of value created, not a flat fee. "
                f"Start with a high-touch paid pilot at $200–500/mo and use it to prove ROI before standardizing. "
                f"What's the core value you can guarantee and measure in 30 days?")

    if any(w in last_msg for w in ["team","hire","co-founder","talent","people"]):
        return (f"At {stage} stage in {sector} — the skill gap that matters most is whoever owns your {dominant_risk} problem. "
                f"If that's not already on the founding team, it's the first hire. "
                f"Contractors prove the role before you commit to full-time. "
                f"Who specifically owns the {dominant_risk} risk on your team right now?")

    if any(w in last_msg for w in ["thank","thanks","great","helpful","good","awesome"]):
        return (f"Your {sector} play in {country} is at {tier} signal ({tas}/100) — {timing}. "
                f"The {dominant_risk} risk is still the live question. What else do you want to pressure-test?")

    interp_snippet = strat_interp[:120] + "..." if len(strat_interp) > 120 else strat_interp
    return (f"The read on your {sector.title()} {bm} model for {seg} customers in {country}: "
            f"{interp_snippet} "
            f"The dominant risk is {dominant_risk} — that's what determines whether {tier} at {tas}/100 converts to traction. "
            f"{pressure_q}")


@api.post("/chat")
async def chat_interaction(req: ChatRequest):
    groq_key = os.environ.get("GROQ_API_KEY", "")
    use_llm  = GROQ_CLIENT and groq_key and groq_key != "dummy"

    if not use_llm:
        return {"success": True, "reply": _chat_fallback(req)}

    sector       = req.context.get("sector", "unknown")
    country      = req.context.get("country", "unknown")
    regime       = req.context.get("regime", "UNKNOWN").replace("_", " ").title()
    tas          = req.context.get("tas_score", 0)
    tier         = req.context.get("signal_tier", "")
    idea         = req.context.get("idea", "")
    idea_feat    = req.context.get("idea_features", {})
    bm           = idea_feat.get("business_model", "unknown")
    seg          = idea_feat.get("target_segment", "unknown").upper()
    diff         = idea_feat.get("differentiation_score", 3)
    stage        = idea_feat.get("stage", "idea")
    comp_int     = idea_feat.get("competitive_intensity", "medium")
    reg_risk     = idea_feat.get("regulatory_risk", "medium")
    # L4 reasoning fields (new)
    dominant_risk   = req.context.get("dominant_risk", "execution")
    strat_interp    = req.context.get("strategic_interpretation", "")
    key_driver      = req.context.get("key_driver", "")
    main_risk       = req.context.get("main_risk", "")
    counterpoint    = req.context.get("counterpoint", "")
    diff_insight    = req.context.get("differentiation_insight", "")

    diff_label = {1: 'minimal', 2: 'low', 3: 'moderate', 4: 'strong', 5: 'exceptional'}.get(diff, 'moderate')

    system_prompt = dedent(f"""
        You are MIDAN — a senior operator who has seen hundreds of startups built and fail.
        You think out loud, challenge assumptions, and never recite facts without connecting them to implications.
        You do NOT ask survey questions. You do NOT say "Great question!" You do NOT give generic advice.

        ── THE ANALYSIS YOU ALREADY RAN ──
        Idea: "{idea}"
        Business Model: {bm} | Target: {seg} | Stage: {stage}
        Differentiation: {diff_label} ({diff}/5) | Competition: {comp_int} | Reg Risk: {reg_risk}
        Market: {sector.title()}, {country} → {regime} regime | Signal: {tier} ({tas}/100)

        ── YOUR STRATEGIC REASONING (use this — don't re-derive it) ──
        Strategic Read: {strat_interp}
        Key Driver: {key_driver}
        Primary Risk: {main_risk}
        Counterpoint: {counterpoint}
        Differentiation Lens: {diff_insight}
        Dominant Risk Archetype: {dominant_risk}

        ── HOW YOU OPERATE IN THIS CONVERSATION ──
        1. You already know the full picture — speak from conviction, not from re-analysis.
        2. Every response connects regime + BM + dominant_risk to the founder's specific situation.
        3. You think out loud: "The {regime} regime with a {dominant_risk} risk profile means..."
           then pivot immediately to what the founder should DO about it.
        4. End EVERY response with ONE sharp embedded challenge — a question that forces the founder
           to confront the assumption most likely to break their plan. Not a survey. A pressure test.
        5. No form questions. No "Tell me more about X." Hypothesize what the answer is and challenge it.
        6. Max 3–4 sentences total. Dense. Specific. Zero filler.

        ── HARD RULES ──
        - Never start with "I"
        - Never say "Great question!", "That's a good point", "I understand", or any affirmation filler
        - Never give advice that works for any startup — it must only work for THIS idea in THIS market
        - If the founder hasn't run analysis yet, don't pretend you have context
        - If they ask about risk: lead with the dominant_risk archetype ({dominant_risk}), not generic startup risks
        - If they ask about funding: tie it to the signal tier and what evidence closes the gap
        - If they disagree with your counterpoint: don't back down — sharpen the argument

        ── EXAMPLE OF RIGHT BEHAVIOR ──
        User: "How do I compete with existing players?"
        MIDAN: "In a {regime} with {comp_int} competition, {bm} models that try to out-feature incumbents
        lose — distribution and switching cost architecture are the only levers that matter.
        Your {diff_label} differentiation at {diff}/5 means you're not there yet.
        The real question is: what workflow are {seg} customers doing manually right now
        that you can embed into so deeply they can't rip you out in 6 months?"

        ── EXAMPLE OF WRONG BEHAVIOR ──
        "Great question! To compete effectively, you should analyze your competitors and find gaps."
        "I'd recommend focusing on your unique value proposition."
    """).strip()

    groq_msgs = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        groq_msgs.append({"role": m.role, "content": m.content})

    try:
        resp = GROQ_CLIENT.chat.completions.create(
            messages=groq_msgs,
            model="llama-3.1-8b-instant",
            temperature=0.45,
            max_tokens=260
        )
        return {"success": True, "reply": resp.choices[0].message.content.strip()}
    except Exception:
        return {"success": True, "reply": _chat_fallback(req)}


# ═══════════════════════════════════════════════════════════════
# /interact — 5-gate intent state machine (pre-analysis routing)
# ═══════════════════════════════════════════════════════════════

_GREET_TOKENS = {
    'hi', 'hello', 'hey', 'yo', 'sup', 'hola', 'marhaba', 'ahlan',
    'good morning', 'good evening', 'good afternoon', 'howdy', 'greetings',
}
_META_PHRASES = [
    'what do you do', 'who are you', 'how does this work', 'what is midan',
    'what can you do', 'explain to me', 'help me understand', 'tell me about yourself',
    'what are you', 'how do i use this', 'what does midan',
]
_VAGUE_STARTERS = [
    'i have an idea', 'i have a startup', 'i want to share', 'i want to tell',
    'i would like to', 'i am thinking', 'i am building', 'i am working',
    'i am developing', 'i am creating', 'what do you think', 'give me feedback',
    'give me your thoughts', 'can you analyze', 'i need help with',
    'i have something', 'i have a concept', 'let me tell you',
    'so i was thinking', 'i was thinking', 'been working on',
    'working on something', 'building something', 'have an idea', 'had an idea',
    'i want to build', 'i want to create', 'i want to start',
]
_PROBLEM_SIGNALS = [
    'problem','pain','issue','struggle','challenge','gap','need','lack',
    'inefficient','inefficiency','expensive','slow','difficult','hard to',
    'no way to','broken','frustrat','wast','manual process',
    'time-consuming','complex','unreliable','inaccessible',
    'underserved','unbanked','overpriced','delayed','stuck',
]
_SOLUTION_SIGNALS = [
    'app','platform','tool','service','system','software','marketplace',
    'saas','api','dashboard','website','bot','chatbot','solution','product',
    'connect','automate','enable','simplify','streamline','digitize',
    'ai-powered','using ai','mobile app','web app','mobile application',
    'subscription model','subscription service',
]
_MARKET_GEO = [
    'egypt','cairo','egyptian','giza','alexandria',
    'saudi','riyadh','jeddah','ksa','saudi arabia',
    'uae','dubai','abu dhabi','emirates','united arab',
    'morocco','casablanca','rabat','marrakech',
    'nigeria','lagos','abuja',
    'kenya','nairobi',
    'usa','america','silicon valley','new york','united states',
    'uk','london','britain','england',
    'mena','africa','middle east','gulf','gcc',
    'global','worldwide','international','emerging market',
]
_MARKET_CUSTOMER = [
    'sme','smes','small business','small businesses','enterprise','enterprises',
    'b2b','b2c','d2c','startup','startups',
    'consumer','consumers','user','users','customer','customers','client','clients',
    'patient','patients','student','students',
    'farmer','farmers','freelancer','freelancers',
    'driver','drivers','merchant','merchants','retailer','retailers',
    'hospital','hospitals','clinic','clinics',
    'school','schools','university','universities',
    'individual','family','families','company','companies',
]


def _extract_components(text: str) -> dict:
    t  = text.lower()
    wc = len(t.split())
    has_problem  = any(s in t for s in _PROBLEM_SIGNALS)
    has_solution = any(s in t for s in _SOLUTION_SIGNALS)
    has_geo      = any(g in t for g in _MARKET_GEO)
    has_customer = any(c in t for c in _MARKET_CUSTOMER)
    has_market   = has_geo or has_customer
    return {
        'has_problem':    has_problem,
        'has_solution':   has_solution,
        'has_market':     has_market,
        'word_count':     wc,
        'is_substantial': wc >= 8,
    }


def _generate_operator_reply(data: dict, idea_text: str) -> str:
    """
    Strategic operator response post-analysis.
    Pattern: INFER → ANALYZE → CHALLENGE → ASK (1 question only)

    Never asks: "What is your problem?", "Who is your customer?", "Which market?"
    Always asks something that directly tests the dominant risk.
    """
    sector        = data.get("sector", "")
    country       = data.get("country", "")
    regime        = data.get("regime", "").replace("_", " ").title()
    tier          = data.get("signal_tier", "")
    tas           = data.get("tas_score", 0)
    idea_feat     = data.get("idea_features", {})
    bm            = idea_feat.get("business_model", "")
    seg           = idea_feat.get("target_segment", "").upper()
    diff          = idea_feat.get("differentiation_score", 3)
    stage         = idea_feat.get("stage", "idea")
    dominant_risk = data.get("dominant_risk", "execution")
    strat_interp  = data.get("strategic_interpretation", "")
    main_risk     = data.get("main_risk", "")
    counterpoint  = data.get("counterpoint", "")
    diff_label    = {1: 'minimal', 2: 'low', 3: 'moderate', 4: 'strong', 5: 'exceptional'}.get(diff, 'moderate')

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if GROQ_CLIENT and groq_key and groq_key != "dummy":
        try:
            prompt = dedent(f"""
                You are MIDAN, a senior startup operator. Someone described this idea:
                "{idea_text}"

                You already ran the analysis:
                - Sector: {sector.title()} in {country} | Regime: {regime}
                - BM: {bm} | Target: {seg} | Stage: {stage}
                - Signal: {tier} ({tas}/100) | Dominant Risk: {dominant_risk}
                - Strategic read: {strat_interp[:220]}
                - Main risk: {main_risk[:150]}

                Respond in EXACTLY this pattern — 4 parts, max 4 sentences total:
                1. INFER: Name the specific sector/context this sits in (not generic — be precise about the sub-space)
                2. ANALYZE: State the non-obvious structural challenge for this exact BM+market combo (not "competition is high")
                3. CHALLENGE: Surface the one hidden assumption that is most likely wrong for this idea
                4. ASK: ONE sharp question that directly tests the {dominant_risk} risk — must be specific to THIS idea

                HARD RULES — violations are rejected:
                - Never ask: "What is your problem?", "Who is your customer?", "Which market are you targeting?"
                - Never start with "I"
                - Never use: "Great!", "That's interesting", "I see", "Certainly"
                - The question must be answerable only by someone who has actually tested this specific idea
                - 3-5 sentences total, no more
                - No bullet points, no headers — flowing prose only

                Output only the response text — no labels, no JSON.
            """).strip()

            resp = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.35,
                max_tokens=220,
            )
            reply = resp.choices[0].message.content.strip()
            if reply and len(reply) > 40:
                return reply
        except Exception:
            pass

    # Fallback: build from structured data — still strategic, not a form
    # 1. INFER
    infer = f"This sits in {sector.title()} in {country} — a {regime} regime."

    # 2. ANALYZE — first 1-2 sentences of strategic interpretation
    if strat_interp:
        sents = [s.strip() for s in strat_interp.split('.') if s.strip()]
        analyze = '. '.join(sents[:2]) + '.'
    else:
        analyze = (
            f"A {bm} model targeting {seg} customers in this market reads as "
            f"{tier} signal ({tas}/100) — the {dominant_risk} risk is the binding constraint."
        )

    # 3. CHALLENGE — first sentence of counterpoint
    if counterpoint:
        sents = [s.strip() for s in counterpoint.split('.') if s.strip()]
        challenge = sents[0] + '.'
    else:
        challenge = f"The assumption that makes this idea feel viable is probably the first thing that breaks in {regime} conditions."

    # 4. ASK — targeted to dominant risk, specific to this idea
    _risk_q = {
        'liquidity':       f"How are you bootstrapping the supply side before demand loses patience — and have you manually brokered a single transaction end-to-end yet?",
        'differentiation': f"If a well-funded competitor copied this in 6 months, what specifically would still make {seg} customers pay for yours over theirs?",
        'regulatory':      f"Have you mapped the exact regulatory pathway for {sector} in {country} — not estimated, actually spoken to a compliance officer about the timeline?",
        'churn':           f"What happens in the product at day 30 that makes cancellation feel like a loss — and have you tested that with real users?",
        'capital':         f"What are your unit economics at 1,000 customers — not projected from assumptions, built from actual cost data?",
        'scalability':     f"At 10x current volume, what breaks first — and is that a systems problem or a people problem you haven't solved yet?",
        'execution':       f"Who on the team has shipped a {bm} product to {seg} customers in {country} before — and if the answer is no one, what is the plan to compress that learning curve?",
    }
    ask = _risk_q.get(dominant_risk, "What is the one assumption that, if wrong, makes the entire model unworkable?")

    return f"{infer} {analyze} {challenge} {ask}"


class InteractRequest(BaseModel):
    context: Dict[str, Any]
    messages: List[ChatMessage]


@api.post("/interact")
async def interact_route(req: InteractRequest):
    """
    Strategic operator chat — INFER → ANALYZE → CHALLENGE → ASK (max 1 question).

    Abolished: sequential field collection (problem → market → solution form flow).
    Abolished: asking "What is your problem?", "Who is your customer?", "Which market?"

    Pattern:
      - Any substantive idea input → run inference immediately → strategic operator reply
      - Post-analysis turns → route to /chat advisor
      - Pure greetings / meta → brief, direct response
      - Vague "I have an idea" with no content → ask them to say the idea, nothing more
    """
    last_user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    t  = last_user_msg.lower().strip()
    wc = len(t.split())

    already_analyzed = bool(req.context.get("tas_score"))

    # ── Gate 1: Post-analysis → route to advisor chat ─────────────────────────
    if already_analyzed:
        chat_req = ChatRequest(context=req.context, messages=req.messages)
        chat_res = await chat_interaction(chat_req)
        return {"success": True, "type": "chat", "reply": chat_res.get("reply", ""), "data": None}

    # ── Gate 2: Pure greeting (no idea content) ───────────────────────────────
    is_greeting = wc <= 6 and any(t == g or t.startswith(g) for g in _GREET_TOKENS)
    if is_greeting:
        return {
            "success": True, "type": "chat",
            "reply": "MIDAN here. What are you building?",
            "data": None,
        }

    # ── Gate 3: Meta question about MIDAN ────────────────────────────────────
    is_meta = any(phrase in t for phrase in _META_PHRASES)
    if is_meta:
        return {
            "success": True, "type": "chat",
            "reply": (
                "MIDAN runs a 4-layer analysis: LLM feature extraction, SVM market classification, "
                "SARIMA 90-day forecast, and strategic reasoning grounded in your specific business model and market. "
                "The output is a verdict — not a dashboard, a decision. What's the idea?"
            ),
            "data": None,
        }

    # ── Gate 4: Pure vague starter — no actual idea described ────────────────
    # e.g. "I have an idea", "I want to build something"
    # Do NOT ask for problem / market / solution — just ask them to say the idea
    components = _extract_components(last_user_msg)
    is_purely_vague = (
        wc <= 15
        and any(t.startswith(v) or t == v.strip() for v in _VAGUE_STARTERS)
        and not components["has_problem"]
        and not components["has_solution"]
    )
    if is_purely_vague:
        return {
            "success": True, "type": "chat",
            "reply": "Tell me what it is. One sentence is enough to start.",
            "data": None,
        }

    # ── Gate 5: Run inference on whatever we have ─────────────────────────────
    # Accumulate all user turns for richer context across a multi-turn session
    all_user_text = " ".join(
        m.content for m in req.messages if m.role == "user"
    ).strip()
    # Use accumulated text only if it meaningfully extends the last message
    analysis_text = all_user_text if len(all_user_text.split()) > len(last_user_msg.split()) + 3 else last_user_msg

    try:
        data  = process_idea(analysis_text)
        reply = _generate_operator_reply(data, last_user_msg)
        return {"success": True, "type": "analysis", "reply": reply, "data": data}
    except Exception as e:
        return {"success": False, "type": "chat", "reply": f"Analysis failed: {str(e)}", "data": None}


@api.get("/health")
async def health():
    return {
        "status":        "ok",
        "models_loaded": MODELS_LOADED,
        "version":       "2.0 — 4-layer hybrid architecture",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)

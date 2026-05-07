"""
midan.core — schemas, constants, ML model loaders, utilities.

This module is the foundation imported by every other midan submodule.
It owns:
  • imports / Groq client init / pickle + JSON loaders
  • all loaded ML artifacts (svm, scaler, pca, lgb, le, sarima_results, fcm_centers, …)
  • L2 freshness functions
  • all static reference tables (SECTOR_*, COUNTRY_*, hint lists, keyword maps)
  • shared utility functions (_phrase_in_text, _has_any, _count_any, _infer_*, _extract_idea_grounding)
  • Pydantic request models + the strict ResponsePayload schema
"""


# ── extracted from api.py ─────────────────────────────────────────────

"""
MIDAN AI Decision Engine — FastAPI Backend
Hybrid 4-Layer Architecture:
  L0 — Sanity gate             (length, contradiction, spam, prompt-injection, vague)
  L1 — Confidence-scored parser (8 fields with per-field confidence + UNKNOWN sentinels)
  L2 — Market intelligence     (idea-perturbed macro → SVM → rule-override → FCM →
                                 SHAP → SARIMA-table). All layers transparent: every
                                 response carries l2_data_freshness, l2_idea_adjustments,
                                 regime_decision_path, and fcm_membership.
  L3 — Idea signal scorer       (regime × business_model × target_segment fit)
  L4 — Composite TAS            (conf × 0.30 + sarima × 0.20 + idea_signal × 0.35 + xai × 0.15)

L2 design notes:
  • Macro vector is a static (sector, country) lookup augmented with EXPLICIT,
    TRACEABLE idea-derived deltas. Each adjustment is gated by L1 confidence
    and surfaced as `inferred`, never as observed.
  • SARIMA is precomputed JSON. When older than SARIMA_STALENESS_DAYS, the
    runtime applies a confidence penalty and sets runtime_staleness_flag.
  • FCM provides a parallel fuzzy regime signal alongside the SVM hard label.
  • DBSCAN has no runtime model artifact — it was a training-time visualization
    only and is documented in TRAINING_ONLY_ARTIFACTS, not in the inference path.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Literal
import numpy as np
import pickle, json, os, re, warnings, requests
from textwrap import dedent
from dotenv import load_dotenv

# All tunable constants live in midan.config. Importing them here makes them
# available to every submodule via the existing `from midan.core import *`
# wildcard imports, so no submodule needs to declare these locally.
from midan.config import (
    RESPONSE_SCHEMA_VERSION,
    L3_REASONING_VERSION,
    L4_DECISION_VERSION,
    L1_MIN_FIELD_CONFIDENCE,
    L1_MIN_AGGREGATE_CONFIDENCE,
    UNKNOWN_VALUE,
    STATIC_MACRO_TABLE_AS_OF,
    SARIMA_STALENESS_DAYS,
    SARIMA_STALENESS_PENALTY,
    L1_ADJUSTMENT_CONFIDENCE_FLOOR,
    L3_FIELD_CONFIDENCE_FLOOR,
    ENABLE_IDEA_ADJUSTMENTS,
    ENABLE_OFFSETTING,
    ENABLE_CONFLICT_DETECTION,
    ENABLE_STALENESS_PENALTY,
)

load_dotenv()
warnings.filterwarnings('ignore')

import logging as _logging
import uuid as _uuid
_CORE_LOG = _logging.getLogger("midan.core")

# ── Structured request logging ──────────────────────────────────────────────
# One logger root: `midan`. Sub-loggers: midan.{layer}. The decision logger
# emits a single structured line per processed request, keyed off a request
# correlation id. Output format is intentionally simple (key=value); no
# external systems, no Prometheus, no S3 — that is out of scope per the
# Step 4 charter.

_DECISION_LOG = _logging.getLogger("midan.decision")


def new_request_id() -> str:
    """Return a short correlation id for the current request."""
    return _uuid.uuid4().hex[:12]


def log_decision(request_id: str, raw: dict, *, endpoint: str = "?") -> None:
    """
    Emit one structured log line summarizing the decision for a request.

    Surfaces the fields the Step 4 charter requires: decision_state,
    decision_strength, decision_quality (per-axis tier), risk decomposition
    levels, post_decision_mode, and the top L4 reasoning signal references.

    Defensive: never raises. If raw is malformed, logs what's available and
    annotates the rest as `unknown`.
    """
    try:
        l4 = (raw or {}).get("l4_decision") or {}
        ds = (raw or {}).get("decision_state") or l4.get("decision_state") or (
            "REJECTED" if (raw or {}).get("invalid_idea") else
            "CLARIFICATION_REQUIRED" if (raw or {}).get("clarification_required") else
            "UNKNOWN"
        )
        strength = ((raw or {}).get("decision_strength") or l4.get("decision_strength") or {}).get("tier", "unknown")
        dq = l4.get("decision_quality") or {}
        rd = l4.get("risk_decomposition") or {}
        conflicts = l4.get("conflicting_signals") or []
        post_mode = ((raw or {}).get("l4_decision") or {}).get("post_decision_mode") or "n/a"
        l3 = (raw or {}).get("l3_reasoning") or {}
        diff_v = (l3.get("differentiation") or {}).get("verdict") or "n/a"
        comp_p = (l3.get("competition") or {}).get("competitive_pressure") or "n/a"

        _DECISION_LOG.info(
            "[DECISION] req=%s endpoint=%s state=%s strength=%s | "
            "ic=%s sa=%s ad=%s ou=%s | "
            "market=%s execution=%s timing=%s | "
            "post_mode=%s | conflicts=%d (high=%d) | "
            "L3.diff=%s L3.comp=%s | L2.regime=%s",
            request_id, endpoint, ds, strength,
            (dq.get("input_completeness") or {}).get("tier", "?"),
            (dq.get("signal_agreement")   or {}).get("tier", "?"),
            (dq.get("assumption_density") or {}).get("tier", "?"),
            dq.get("overall_uncertainty", "?"),
            (rd.get("market_risk")    or {}).get("level", "?"),
            (rd.get("execution_risk") or {}).get("level", "?"),
            (rd.get("timing_risk")    or {}).get("level", "?"),
            post_mode,
            len(conflicts),
            sum(1 for c in conflicts if c.get("severity") == "high"),
            diff_v, comp_p,
            (raw or {}).get("regime", "n/a"),
        )
    except Exception as _log_err:
        # Logging itself must NEVER break the request path.
        _DECISION_LOG.warning(
            "[DECISION] log_decision failed (%s: %r) — request_id=%s",
            type(_log_err).__name__, _log_err, request_id,
        )


def log_failure(request_id: str, *, endpoint: str, kind: str, detail: str) -> None:
    """Structured log for any failure trigger (pipeline error, schema violation)."""
    _DECISION_LOG.error(
        "[FAILURE] req=%s endpoint=%s kind=%s detail=%s",
        request_id, endpoint, kind, detail,
    )

try:
    from groq import Groq
    GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy"))
except Exception as _groq_err:
    GROQ_CLIENT = None
    _CORE_LOG.warning(
        "[CORE] Groq client init failed (%s: %r) — LLM paths will use heuristic fallback",
        type(_groq_err).__name__, _groq_err,
    )

# Single source of truth for the LLM model used across the system. The L4
# engine remains the decision authority; the LLM is only used for
# conversation, explanation, and report generation. Any swap (provider, model
# version) happens here and propagates to every call site.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# ── LOAD MODELS ──────────────────────────────────────────────
# Models live in the project root, one level above the midan/ package.
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)

def _pkl(name):
    with open(f'{MODELS_DIR}/{name}', 'rb') as f:
        return pickle.load(f)

def _json(name, default=None):
    try:
        with open(f'{MODELS_DIR}/{name}', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as _json_err:
        _CORE_LOG.info(
            "[CORE] _json('%s') failed (%s: %r) — using default",
            name, type(_json_err).__name__, _json_err,
        )
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
    # ── L2 fuzzy clustering (FCM): wired at runtime as a parallel signal
    # to the SVM hard classification. fcm_centers.pkl is shape (3, 2) — 3
    # regime cluster centers in PCA space. cluster_names maps the indices
    # to regime labels (CONTRACTING_MARKET / HIGH_FRICTION_MARKET / EMERGING_MARKET).
    fcm_centers   = _pkl('fcm_centers.pkl')
    cluster_names = _json('cluster_names.json', {})
    MODELS_LOADED  = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR   = str(e)

# ── Training-only artifacts NOT loaded at runtime ─────────────────────────────
# Listed here so the dead-component audit has a clear answer rather than a
# silently-orphaned file. If you add runtime use for any of these, move it
# above into the load block.
TRAINING_ONLY_ARTIFACTS = {
    'fcm_centroids.pkl': "8-cluster 5D centroids — used during training for "
                          "sector-archetype analysis; no runtime consumer.",
    'pca_model.pkl':      "Earlier PCA artifact superseded by pca_global.pkl. "
                          "Kept on disk for reproducibility of training notebook.",
    'dbscan_clusters.png':"Density-based clustering visualization from the "
                          "training notebook. DBSCAN has no runtime model file "
                          "and is NOT part of the inference pipeline.",
    'shap_feature_importance.json': "Global SHAP importances computed once at "
                          "training. Runtime SHAP is recomputed per-request via "
                          "compute_shap; this JSON is not read at inference.",
}

# ── L2 data-freshness metadata ────────────────────────────────────────────────
# Surfaced in every analysis response so consumers know what is observed,
# what is static, and what is stale. The constants STATIC_MACRO_TABLE_AS_OF,
# SARIMA_STALENESS_DAYS and SARIMA_STALENESS_PENALTY are owned by midan.config
# and imported at the top of this module — edit them there.

def _sarima_last_date() -> str:
    """Earliest last_date across loaded SARIMA models — represents global staleness."""
    if not isinstance(sarima_results, dict) or not sarima_results:
        return ""
    dates = [v.get('last_date', '') for v in sarima_results.values() if isinstance(v, dict)]
    dates = [d for d in dates if d]
    return min(dates) if dates else ""

def _days_since(iso_date: str) -> Optional[int]:
    if not iso_date:
        return None
    try:
        from datetime import date as _date
        y, m, d = (int(p) for p in iso_date.split('-'))
        return (_date.today() - _date(y, m, d)).days
    except Exception as _date_err:
        _CORE_LOG.warning(
            "[CORE] _days_since('%s') failed (%s: %r) — malformed date masked as None",
            iso_date, type(_date_err).__name__, _date_err,
        )
        return None

def compute_l2_freshness() -> dict:
    """
    Build the freshness envelope surfaced in every analysis response.

    All four sources of L2 input are documented here:
      - macro:   static_table     (COUNTRY_MACRO_DEFAULTS — 8 hardcoded countries)
      - sector:  static_table     (SECTOR_EFF_MACRO + SECTOR_MEDIANS)
      - sarima:  precomputed_json (sarima_results.json, frozen at training)
      - svm/shap: trained_model   (svm_global.pkl, lgb_surrogate.pkl)

    `runtime_staleness_flag` fires when SARIMA forecasts are older than
    SARIMA_STALENESS_DAYS — at that point regime confidence is multiplied
    by SARIMA_STALENESS_PENALTY in run_inference.
    """
    sarima_as_of = _sarima_last_date()
    sarima_age   = _days_since(sarima_as_of)
    macro_age    = _days_since(STATIC_MACRO_TABLE_AS_OF)
    # Toggle-aware: if the operator has switched off the staleness penalty,
    # the freshness envelope reports the SARIMA age but does NOT raise the
    # runtime_staleness_flag — downstream consumers (pipeline, response)
    # will then leave regime confidence un-penalized.
    is_stale     = bool(sarima_age is not None and sarima_age > SARIMA_STALENESS_DAYS)
    runtime_stale = bool(is_stale and ENABLE_STALENESS_PENALTY)
    train_drift = any(
        v.get('drift_flag', False)
        for v in (sarima_results.values() if isinstance(sarima_results, dict) else [])
        if isinstance(v, dict)
    )
    return {
        "macro_source":              "static_table",
        "macro_as_of":               STATIC_MACRO_TABLE_AS_OF,
        "macro_days_stale":          macro_age,
        "macro_coverage":            f"{len(COUNTRY_MACRO_DEFAULTS)} hardcoded countries",
        "sarima_source":             "precomputed_json",
        "sarima_as_of":              sarima_as_of or None,
        "sarima_days_stale":         sarima_age,
        "sarima_staleness_threshold_days": SARIMA_STALENESS_DAYS,
        "train_time_drift_flag":     train_drift,
        "runtime_staleness_flag":    runtime_stale,
        "staleness_penalty_applied": SARIMA_STALENESS_PENALTY if runtime_stale else 1.0,
        "live_data_integration":     False,
    }

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
                   'invoice','financing','working capital','insurance',
                   'wallet','money','تمويل','دفع'],
    'ecommerce':  ['ecommerce','e-commerce','shop','store','retail',
                   'marketplace','delivery','commerce','تجارة','توصيل'],
    'healthtech': ['health','medical','doctor','clinic','hospital',
                   'pharma','biotech','mental','صحة','طبي'],
    'edtech':     ['education','learning','school','university','course',
                   'tutor','tutoring','edtech','training','teacher','تعليم','دراسة'],
    'saas':       ['saas','software','platform','dashboard','tool',
                   'api','enterprise','cloud','b2b','crm','automation',
                   'analytics','forecasting','planning','workflow','برنامج'],
    'logistics':  ['logistics','shipping','supply chain','warehouse',
                   'transport','fleet','trucking','procurement','inventory',
                   'supplier','suppliers','شحن','لوجستيك'],
    'agritech':   ['agri','farm','farmer','farmers','crop','harvest',
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

SECTOR_TIEBREAKER = {
    'saas': 7,
    'fintech': 6,
    'healthtech': 5,
    'edtech': 4,
    'logistics': 3,
    'ecommerce': 2,
    'agritech': 1,
}

BUSINESS_CUSTOMER_HINTS = [
    'restaurant', 'restaurants', 'cafe', 'cafes', 'hotel', 'hotels',
    'clinic', 'clinics', 'hospital', 'hospitals', 'school', 'schools',
    'merchant', 'merchants', 'retailer', 'retailers', 'supplier', 'suppliers',
    'manufacturer', 'manufacturers', 'factory', 'factories', 'warehouse',
    'warehouses', 'business', 'businesses', 'company', 'companies',
    'enterprise', 'enterprises', 'sme', 'smes', 'operator', 'operators',
]
CONSUMER_HINTS = [
    'consumer', 'consumers', 'household', 'households', 'family', 'families',
    'individual', 'individuals', 'student', 'students', 'patient', 'patients',
]
SOFTWARE_HINTS = [
    'software', 'saas', 'platform', 'dashboard', 'tool', 'app', 'portal',
    'api', 'analytics', 'automation', 'workflow', 'system', 'ai', 'ml',
    'machine learning', 'forecast', 'forecasting', 'optimization', 'optimisation',
]
OPERATIONS_HINTS = [
    'inventory', 'supplier', 'suppliers', 'supply chain', 'procurement',
    'ordering', 'demand planning', 'demand forecasting', 'forecasting',
    'planning', 'waste', 'food waste', 'replenishment', 'operations',
]
RESTAURANT_HINTS = [
    'restaurant', 'restaurants', 'cafe', 'cafes', 'kitchen', 'kitchens',
    'chef', 'menu', 'menus', 'hospitality',
]
AGRITECH_STRONG_HINTS = [
    'farm', 'farms', 'farmer', 'farmers', 'crop', 'crops',
    'agriculture', 'agricultural', 'irrigation', 'harvest',
]
SERVICE_HINTS = [
    'consulting', 'managed service', 'managed services', 'agency',
    'done-for-you', 'implementation', 'concierge',
]
HARDWARE_HINTS = [
    'device', 'devices', 'hardware', 'sensor', 'sensors', 'iot',
    'machine', 'machines', 'robot', 'robots',
]
CITY_LABELS = {
    'cairo': 'Cairo',
    'giza': 'Giza',
    'riyadh': 'Riyadh',
    'jeddah': 'Jeddah',
    'dubai': 'Dubai',
    'abu dhabi': 'Abu Dhabi',
    'lagos': 'Lagos',
    'nairobi': 'Nairobi',
    'casablanca': 'Casablanca',
    'london': 'London',
}
COUNTRY_LABELS = {
    'EG': 'Egypt',
    'SA': 'Saudi Arabia',
    'AE': 'UAE',
    'MA': 'Morocco',
    'NG': 'Nigeria',
    'KE': 'Kenya',
    'US': 'United States',
    'GB': 'United Kingdom',
}


def _phrase_in_text(text: str, phrase: str) -> bool:
    if re.search(r'[A-Za-z0-9]', phrase):
        pattern = r'(?<!\w)' + re.escape(phrase) + r'(?!\w)'
        return re.search(pattern, text) is not None
    return phrase in text


def _has_any(text: str, phrases: List[str]) -> bool:
    return any(_phrase_in_text(text, p) for p in phrases)


def _count_any(text: str, phrases: List[str]) -> int:
    return sum(1 for p in phrases if _phrase_in_text(text, p))


def _is_workflow_software_idea(text: str, segment: str = "") -> bool:
    return (
        _has_any(text, SOFTWARE_HINTS)
        and _has_any(text, OPERATIONS_HINTS)
        and (segment == 'b2b' or _has_any(text, BUSINESS_CUSTOMER_HINTS))
    )


def _score_sector_candidates(text: str) -> Dict[str, int]:
    scores = {sector: 0 for sector in SECTOR_KEYWORDS}

    for sector, kws in SECTOR_KEYWORDS.items():
        for kw in kws:
            if _phrase_in_text(text, kw):
                scores[sector] += 4 if " " in kw else 3

    if _has_any(text, SOFTWARE_HINTS):
        scores['saas'] += 3
    if _has_any(text, ['tutor', 'tutoring', 'teacher', 'course', 'learning', 'school', 'student']) and _has_any(text, SOFTWARE_HINTS):
        scores['edtech'] += 4
    if _has_any(text, BUSINESS_CUSTOMER_HINTS) and _has_any(text, SOFTWARE_HINTS):
        scores['saas'] += 2
    if _has_any(text, OPERATIONS_HINTS):
        scores['logistics'] += 2
        scores['saas'] += 1
    if _has_any(text, RESTAURANT_HINTS):
        scores['saas'] += 4 if _has_any(text, SOFTWARE_HINTS + OPERATIONS_HINTS) else 1
        scores['agritech'] -= 2
    if _has_any(text, AGRITECH_STRONG_HINTS):
        scores['agritech'] += 5
    if 'invoice' in text and any(w in text for w in ['finance', 'financing', 'loan', 'working capital']):
        scores['fintech'] += 5

    return scores


def _infer_target_segment(text: str, sector: str) -> str:
    if any(w in text for w in ['both consumers and', 'b2b and b2c', 'businesses and individuals']):
        return 'mixed'
    if any(w in text for w in ['government', 'ministry', 'public sector', 'municipalities']):
        return 'b2g'
    if 'b2b' in text:
        return 'b2b'
    if 'b2c' in text:
        return 'b2c'

    business_hits = _count_any(text, BUSINESS_CUSTOMER_HINTS)
    consumer_hits = _count_any(text, CONSUMER_HINTS)
    if business_hits and business_hits >= consumer_hits:
        return 'b2b'
    if consumer_hits:
        return 'b2c'
    if sector in {'saas', 'logistics'} and business_hits:
        return 'b2b'
    return 'b2c'


def _infer_business_model(text: str, sector: str, segment: str) -> str:
    if any(w in text for w in ['subscription', 'monthly plan', 'annual fee', 'recurring revenue', 'per seat']):
        return 'subscription'
    if any(w in text for w in ['marketplace', 'two-sided', 'buyers and sellers', 'match buyers']):
        return 'marketplace'
    if any(w in text for w in ['commission', 'take rate', 'earn per transaction', 'percentage of']):
        return 'commission'
    if _has_any(text, HARDWARE_HINTS):
        return 'hardware'
    if _has_any(text, SERVICE_HINTS):
        return 'service'
    if sector == 'saas' or _is_workflow_software_idea(text, segment) or _has_any(text, SOFTWARE_HINTS):
        return 'saas'
    if sector == 'fintech':
        return 'commission'
    if sector == 'ecommerce':
        return 'marketplace'
    if sector == 'logistics' and segment == 'b2b' and _has_any(text, OPERATIONS_HINTS):
        return 'saas'
    return 'other'


def _infer_stage(text: str) -> str:
    if any(w in text for w in ['launched', 'live', 'customers', 'mrr', 'revenue', 'growing', 'scaling']):
        return 'mvp'
    if any(w in text for w in ['beta', 'pilot', 'testing', 'prototype', 'validating', 'early access']):
        return 'validation'
    return 'idea'


def _infer_differentiation_score(text: str) -> int:
    diff = 3
    if any(w in text for w in ['ai-powered', 'ai powered', 'artificial intelligence', 'machine learning',
                               'forecasting', 'optimization', 'optimisation', 'proprietary data',
                               'unique algorithm', 'first in', 'only platform', 'patent', 'no one else']):
        diff = 4
    if any(w in text for w in ['breakthrough', 'world first', 'never been done', 'disruptive', 'revolutionary']):
        diff = 5
    if any(w in text for w in ['similar to', 'like uber', 'like amazon', 'inspired by', 'copy of', 'clone']):
        diff = 2
    if any(w in text for w in ['same as', 'another', 'yet another', 'also does']):
        diff = 1
    return diff


def _extract_idea_grounding(
    idea_text: str,
    sector: str,
    idea_features: Optional[dict] = None,
    country: str = "",
) -> dict:
    text = (idea_text or "").lower()
    features = idea_features or {}
    business_model = features.get('business_model', 'other')
    segment = features.get('target_segment', 'b2c')

    city = next((label for key, label in CITY_LABELS.items() if key in text), "")
    market_label = city or COUNTRY_LABELS.get(country.upper(), country or "the market")

    if _has_any(text, RESTAURANT_HINTS):
        customer_label = "independent restaurants" if 'independent' in text else "restaurants"
        context_label = "restaurant operations software" if business_model == 'saas' or _is_workflow_software_idea(text, segment) else "restaurant operations"
    elif any(w in text for w in ['sme', 'smes', 'small business', 'small businesses']):
        customer_label = "SMEs"
        context_label = "SME workflow software" if business_model == 'saas' else "SME services"
    elif any(w in text for w in ['clinic', 'clinics', 'hospital', 'hospitals']):
        customer_label = "clinics and hospitals"
        context_label = "clinical workflow software" if business_model == 'saas' else "health operations"
    elif segment == 'b2b':
        customer_label = "business operators"
        context_label = "B2B workflow software" if business_model == 'saas' else f"{sector} infrastructure"
    else:
        customer_label = "end users"
        context_label = f"{sector} product"

    if 'food waste' in text:
        problem_label = "food waste"
    elif 'invoice' in text and any(w in text for w in ['financing', 'finance']):
        problem_label = "working-capital gaps"
    elif any(w in text for w in ['inventory', 'supplier', 'procurement']):
        problem_label = "operational waste and bad purchasing decisions"
    else:
        problem_label = "the stated operational problem"

    motion_bits = []
    if 'demand forecasting' in text:
        motion_bits.append('demand forecasting')
    if 'supplier planning' in text:
        motion_bits.append('supplier planning')
    if 'invoice financing' in text:
        motion_bits.append('invoice financing')
    if not motion_bits and 'forecast' in text:
        motion_bits.append('forecasting')
    if not motion_bits and 'planning' in text:
        motion_bits.append('planning')
    motion_label = " and ".join(motion_bits) if motion_bits else "the core workflow"

    return {
        'market_label': market_label,
        'customer_label': customer_label,
        'context_label': context_label,
        'problem_label': problem_label,
        'motion_label': motion_label,
    }
class IdeaRequest(BaseModel):
    idea: str
    sector: str = "Fintech"
    country: str = "EG — Egypt"
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    context: Dict[str, Any]
    messages: List[ChatMessage]
    session_id: Optional[str] = None


class ProjectionRequest(BaseModel):
    idea: str
    context: Optional[Dict[str, Any]] = None
    question: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# RESPONSE PAYLOAD SCHEMA — strict contract enforced at every endpoint
#
# Every response across /analyze, /interact, /project conforms to this
# schema. Core fields are required (never omitted). Missing data is
# explicit `null`/`unknown`/empty-list with an explanation string —
# never silently dropped. There is NO endpoint-specific reinterpretation
# of core fields; only optional extension fields (reply, type, projection,
# etc.) vary by endpoint, and they are also typed.
#
# DO NOT relax these schemas. If validation fails at construction time,
# the endpoint must surface a 500 with an explicit error — silent
# auto-correction is forbidden.
# ═══════════════════════════════════════════════════════════════
# RESPONSE_SCHEMA_VERSION is owned by midan.config and imported above.

DecisionState = Literal[
    'PRE_ANALYSIS',           # no pipeline ran (greeting / casual / partial idea turn)
    'REJECTED',               # L0 input gate rejected
    'CLARIFICATION_REQUIRED', # L1 fail-fast halt before L4
    'GO',
    'CONDITIONAL',
    'NO_GO',
    'INSUFFICIENT_DATA',      # L4 halted: required L3 module unavailable
    'HIGH_UNCERTAINTY',       # L4 advisory only
    'CONFLICTING_SIGNALS',    # L4 halted: severe unresolved conflict
]

PostDecisionMode = Literal[
    'STANDARD_ADVISOR',
    'RESOLVING_CONFLICT',
    'ADVISORY_ONLY',
    'RE_CLARIFY',
]

DecisionStrengthTier = Literal['strong', 'moderate', 'weak', 'uncertain']
QualityTier          = Literal['low', 'medium', 'high', 'unknown']
RiskLevel            = Literal['low', 'medium', 'high', 'elevated_with_offset', 'unknown']
UncertaintyState     = Literal['low', 'moderate', 'high']


class DecisionStrengthBlock(BaseModel):
    """Qualitative replacement for numeric confidence."""
    tier:  DecisionStrengthTier
    basis: str  # one-line explanation of how the tier was reached


class QualityDimension(BaseModel):
    """One axis of decision_quality (input / agreement / assumption)."""
    tier:  QualityTier
    basis: str


class DecisionQualityBlock(BaseModel):
    """Three-axis qualitative assessment of decision reliability."""
    input_completeness:  QualityDimension
    signal_agreement:    QualityDimension
    assumption_density:  QualityDimension
    overall_uncertainty: UncertaintyState


class RiskDimension(BaseModel):
    """One of the three independent risk dimensions surfaced by L4."""
    level:                RiskLevel
    drivers:              List[Dict[str, Any]] = Field(default_factory=list)
    reasoning:            str = ""
    evidence_grounded_in: Dict[str, List[str]] = Field(default_factory=dict)


class RiskDecompositionBlock(BaseModel):
    """L4 risk decomposition — three independent dimensions, never collapsed."""
    market_risk:    RiskDimension
    execution_risk: RiskDimension
    timing_risk:    RiskDimension


class ReasoningStep(BaseModel):
    """One step in the L4 decision-reasoning trace."""
    step:       str
    rule_id:    str
    evidence:   List[str] = Field(default_factory=list)
    conclusion: str


class ReasoningTraceBlock(BaseModel):
    """
    End-to-end traceability surface. Every claim in the response should be
    derivable from one or more entries in this block. Empty fields are
    explicitly empty (never omitted) so consumers can detect "no trace yet".
    """
    decision_reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    conflict_ids:             List[str]           = Field(default_factory=list)
    top_dim_label:            Optional[str]       = None
    top_dim_level:            Optional[str]       = None
    top_dim_reasoning:        Optional[str]       = None
    signal_references:        Dict[str, Any]      = Field(default_factory=dict)


class ResponsePayload(BaseModel):
    """
    The contract. Every response — /analyze, /interact, /project — produces
    a payload that satisfies this schema. Core fields are non-optional so a
    missing field will raise ValidationError.

    Endpoint-specific extension fields are explicitly typed and optional;
    they NEVER replace or shadow core fields.
    """
    success:            bool
    schema_version:     str = RESPONSE_SCHEMA_VERSION

    # ── Core decision contract — required for every response ─────────────
    decision_state:     DecisionState
    decision_strength:  DecisionStrengthBlock
    decision_quality:   DecisionQualityBlock
    risk_decomposition: RiskDecompositionBlock
    reasoning_trace:    ReasoningTraceBlock
    post_decision_mode: Optional[PostDecisionMode]   # null when no decision rendered
    post_decision_mode_basis: str                     # explains null OR which mode was chosen

    # ── Endpoint-specific optional extensions (typed, not free dicts) ────
    reply:               Optional[str] = None
    type:                Optional[str] = None
    clarification_state: Optional[Dict[str, Any]] = None
    projection:          Optional[Dict[str, Any]] = None
    quality:             Optional[Dict[str, Any]] = None
    data:                Optional[Dict[str, Any]] = None
    raw_pipeline_output: Optional[Dict[str, Any]] = None  # full process_idea result for debug


class SchemaViolationError(Exception):
    """Raised when _build_response_payload cannot construct a valid payload."""
    pass


# ── PAYLOAD CONSTRUCTION ────────────────────────────────────────────────────
# `build_response_payload` is the single mapper from raw pipeline outputs to



# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]

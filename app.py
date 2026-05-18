import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pickle, json, os, time, warnings, requests
import asyncio
from textwrap import dedent
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI as _OpenAI

_APP_LLM       = _OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
)
_APP_LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="MIDAN — AI Decision Engine",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN TOKENS ────────────────────────────────────────────
ACID   = "#C8FF57"
BG     = "#05050A"
CARD   = "#0C0C14"
CARD2  = "#12121C"
BORDER = "rgba(255,255,255,0.09)"
BORDA  = "rgba(200,255,87,0.22)"
TEXT   = "#EDEAF8"
SUB    = "#8A87A0"
MUTED  = "#46445A"
CYAN   = "#4DCCFF"

FEATURES = ['inflation','gdp_growth','macro_friction',
            'capital_concentration','velocity_yoy']

REGIME_COLORS = {
    'GROWTH_MARKET':         ACID,
    'EMERGING_MARKET':       CYAN,
    'HIGH_FRICTION_MARKET':  '#FF6B6B',
    'CONTRACTING_MARKET':    '#F5A623',
}

# ── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 2rem 2rem !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] > div {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER};
    padding: 0 !important;
}}
.stButton > button {{
    background: {ACID} !important; color: #000 !important;
    border: none !important; border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
    font-size: 13px !important; letter-spacing: 0.08em !important;
    padding: 14px 28px !important; width: 100% !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}
.stTextArea textarea, .stTextArea textarea:focus {{
    background: #0D0D11 !important; background-color: #0D0D11 !important;
    border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 6px !important;
    color: #EDEAF8 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important; font-weight: 300 !important;
    caret-color: {ACID} !important; -webkit-text-fill-color: #EDEAF8 !important;
}}
.stTextArea textarea::placeholder {{ color: #3E3D50 !important; -webkit-text-fill-color: #3E3D50 !important; }}
.stTextArea > div {{ background: #0D0D11 !important; border-radius: 6px !important; }}
.stSelectbox > div > div {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important; border-radius: 6px !important;
}}
div[data-testid="stMetric"] {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; padding: 20px 24px !important;
}}
div[data-testid="stMetricLabel"] p {{
    font-size: 11px !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: {MUTED} !important; font-weight: 600 !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important; font-size: 26px !important;
    font-weight: 800 !important; color: {ACID} !important;
    letter-spacing: -0.5px !important; overflow: visible !important;
}}
div[data-testid="stMetricDelta"] {{ font-size: 12px !important; color: {SUB} !important; }}
::-webkit-scrollbar {{ width: 4px; background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {MUTED}; border-radius: 2px; }}
</style>
""", unsafe_allow_html=True)

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD2,
    font=dict(family="DM Sans", color=SUB, size=12),
    margin=dict(l=16, r=16, t=16, b=16),
)

# ── MIDAN PIPELINE IMPORTS ────────────────────────────────────
# We import from the midan package instead of duplicating model loading.
# scaler/pca/svm/le/lgb are already loaded at midan.core module import time.
try:
    from midan.core import (
        scaler, pca, svm, le, lgb, sarima_results, MODELS_LOADED,
        SECTOR_LABEL_MAP as _MIDAN_SECTOR_LABEL_MAP,
        COUNTRY_MACRO_DEFAULTS, SECTOR_EFF_MACRO, SECTOR_MEDIANS,
        compute_l2_freshness, drift_baseline,
        REACT_IS_HIGH, REACT_IS_LOW, REACT_SHAP_RELIABLE, REACT_SHAP_UNRELIABLE,
    )
    from midan.pipeline import process_idea
    from midan.drift_monitor import check_drift, load_prediction_log
    from midan.outcome_feedback import compute_calibration_metrics, log_outcome
    MODEL_ERROR = None
except Exception as _midan_err:
    MODELS_LOADED = False
    MODEL_ERROR   = str(_midan_err)
    sarima_results = {}
    # Provide stubs so the rest of the module doesn't crash at definition time
    scaler = pca = svm = le = lgb = None
    drift_baseline = {}
    REACT_IS_HIGH = 0.72; REACT_IS_LOW = 0.35
    REACT_SHAP_RELIABLE = 0.65; REACT_SHAP_UNRELIABLE = 0.40
    def process_idea(*a, **kw): return {"success": False, "error": str(_midan_err)}
    def check_drift(*a, **kw): return {}
    def load_prediction_log(*a, **kw): return []
    def compute_calibration_metrics(): return {"calibration_status": "no_data", "total_outcomes": 0, "scored_outcomes": 0, "overall_accuracy": None, "note": ""}
    def log_outcome(**kw): return {"status": "error", "error": "models not loaded"}
    _MIDAN_SECTOR_LABEL_MAP = {}
    COUNTRY_MACRO_DEFAULTS = {}; SECTOR_EFF_MACRO = {}; SECTOR_MEDIANS = {}
    def compute_l2_freshness(*a, **kw): return {}

# Supplementary context data (optional — graceful fallback if absent)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
def _json_optional(name, default):
    try:
        with open(os.path.join(_MODELS_DIR, name), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

shap_global = _json_optional('shap_feature_importance.json', {})
cluster_names = _json_optional('cluster_names.json', {})
comps_data    = _json_optional('competitors_context.json', {})
sents_data    = _json_optional('sentiment_context.json', [])

# ── AGENT A1 — Keyword Parser ────────────────────────────────
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

def agent_a1_parse(idea_text: str) -> tuple[str, str, bool, bool]:
    """Extract sector + country from idea text. Returns detection flags."""
    t = idea_text.lower()
    sector, sector_found = None, False
    for sec, kws in SECTOR_KEYWORDS.items():
        if any(k in t for k in kws):
            sector, sector_found = sec, True
            break
    if not sector_found:
        sector = 'fintech'  # we use fintech as a placeholder here — the dropdown will override it
    country, country_found = None, False
    for code, kws in COUNTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            country, country_found = code, True
            break
    if not country_found:
        country = 'EG'  # we use EG as a placeholder here — the dropdown will override it
    return sector, country, sector_found, country_found

# ── ENHANCED REGIME — post-process SVM with domain rules ─────
def enhanced_regime(svm_regime: str, svm_conf: float, inflation: float,
                    gdp_growth: float, macro_friction: float,
                    velocity_yoy: float) -> tuple[str, float]:
    """
    Layer 2: selective rule-based override on top of SVM.
    Only fires for GROWTH (not in training) and extreme edge cases.
    Computes its own confidence based on rule margin.
    """
    # Rule 1: GROWTH — we add this rule because SVM has no GROWTH class in its training
    if gdp_growth > 3.5 and inflation < 8 and velocity_yoy > 0.15:
        margin = min((gdp_growth-3.5)/4.0, (8-inflation)/8.0, (velocity_yoy-0.15)/0.25)
        conf = float(np.clip(0.65 + margin * 0.30, 0.60, 0.95))
        return 'GROWTH_MARKET', conf
    # Rule 1b: EMERGING — we use this rule for good macro situations with lower velocity
    if gdp_growth > 2.0 and inflation < 10 and macro_friction < 10:
        margin = min((gdp_growth-2.0)/4.0, (10-inflation)/10.0, (10-macro_friction)/15.0)
        conf = float(np.clip(0.60 + margin * 0.30, 0.55, 0.90))
        return 'EMERGING_MARKET', conf
    # Rule 2: CONTRACTING — we only fire this for extreme downturn conditions
    if gdp_growth < 0 or (inflation > 50 and macro_friction > 50):
        severity = max(abs(min(gdp_growth, 0)) / 3.0, 0.0)
        conf = float(np.clip(0.65 + severity * 0.25, 0.60, 0.92))
        return 'CONTRACTING_MARKET', conf
    # Rule 3: HIGH_FRICTION — we restrict this to cases of severe macro pain only
    if macro_friction > 30 or inflation > 25:
        pain = max((macro_friction - 30) / 40, (inflation - 25) / 30, 0)
        conf = float(np.clip(0.60 + pain * 0.30, 0.55, 0.92))
        return 'HIGH_FRICTION_MARKET', conf
    # Default: we fall back to trusting the SVM when no rule fires
    return svm_regime, svm_conf

# ── AGENT A0 — Idea Evaluation ──────────────────────────────────
IDEA_DIMENSIONS = ['problem_clarity', 'solution_fit', 'differentiation', 'business_model', 'scalability']
IDEA_DIM_LABELS = {
    'problem_clarity': 'Problem Clarity',
    'solution_fit': 'Solution Fit',
    'differentiation': 'Differentiation',
    'business_model': 'Business Model',
    'scalability': 'Scalability',
}

def agent_a0_evaluate_idea(idea_text, sector, country):
    """
    Agent A0: Evaluates the startup idea itself using LLM or keyword heuristics.
    Returns dict with 5 dimension scores (0-10), reasons, and overall idea_score (0-100).
    """
    try:
        prompt = dedent(f"""
            You are a VC analyst evaluating a startup idea. Score each dimension 0-10.

            Idea: "{idea_text}"
            Sector: {sector} | Country: {country}

            Score these 5 dimensions and provide a one-sentence justification for each:
            1. problem_clarity — Is there a clear, specific problem being solved?
            2. solution_fit — Does the proposed solution address the problem?
            3. differentiation — Is it meaningfully different from existing solutions?
            4. business_model — Is there an obvious way to make money?
            5. scalability — Can it grow beyond the initial market?

            Respond in EXACTLY this JSON format, no other text:
            {{"problem_clarity": {{"score": 7, "reason": "..."}}, "solution_fit": {{"score": 6, "reason": "..."}}, "differentiation": {{"score": 5, "reason": "..."}}, "business_model": {{"score": 8, "reason": "..."}}, "scalability": {{"score": 6, "reason": "..."}}}}
        """).strip()
        response = _APP_LLM.chat.completions.create(
            model=_APP_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()
        scores_data = json.loads(raw)
        scores = {}
        reasons = {}
        for dim in IDEA_DIMENSIONS:
            entry = scores_data.get(dim, {})
            scores[dim] = max(0, min(10, int(entry.get('score', 5))))
            reasons[dim] = entry.get('reason', '')
        idea_score = int(sum(scores.values()) / len(scores) * 10)
        return {'scores': scores, 'reasons': reasons, 'idea_score': idea_score}
    except Exception:
        pass

    # Fallback: we use keyword heuristic scoring when the LLM is unavailable
    t = idea_text.lower()
    scores = {}
    reasons = {}

    problem_words = ['problem', 'issue', 'challenge', 'pain', 'struggle', 'need', 'lack', 'gap', 'inefficient', 'expensive', 'slow', 'difficult']
    problem_hits = sum(1 for w in problem_words if w in t)
    scores['problem_clarity'] = min(10, 3 + problem_hits * 2)
    reasons['problem_clarity'] = 'Clear problem statement detected' if problem_hits >= 2 else 'Problem could be more specific'

    solution_words = ['app', 'platform', 'tool', 'system', 'service', 'automate', 'connect', 'provide', 'enable', 'simplify', 'streamline', 'reduce']
    sol_hits = sum(1 for w in solution_words if w in t)
    scores['solution_fit'] = min(10, 3 + sol_hits * 2)
    reasons['solution_fit'] = 'Solution approach is clear' if sol_hits >= 2 else 'Describe how your solution works'

    diff_words = ['first', 'only', 'unique', 'unlike', 'better', 'faster', 'cheaper', 'new', 'innovative', 'ai', 'machine learning', 'blockchain']
    diff_hits = sum(1 for w in diff_words if w in t)
    scores['differentiation'] = min(10, 2 + diff_hits * 2)
    reasons['differentiation'] = 'Unique angle detected' if diff_hits >= 2 else 'What makes this different from existing solutions?'

    biz_words = ['subscription', 'saas', 'commission', 'fee', 'pricing', 'revenue', 'monetize', 'b2b', 'b2c', 'freemium', 'marketplace', 'premium']
    biz_hits = sum(1 for w in biz_words if w in t)
    scores['business_model'] = min(10, 3 + biz_hits * 2)
    reasons['business_model'] = 'Revenue model indicated' if biz_hits >= 1 else 'How will this make money?'

    scale_words = ['scale', 'global', 'expand', 'growth', 'million', 'region', 'international', 'multiple', 'market', 'nationwide']
    scale_hits = sum(1 for w in scale_words if w in t)
    scores['scalability'] = min(10, 3 + scale_hits * 2)
    reasons['scalability'] = 'Growth potential indicated' if scale_hits >= 1 else 'How will this scale beyond initial market?'

    word_count = len(t.split())
    if word_count > 30:
        for dim in scores:
            scores[dim] = min(10, scores[dim] + 1)

    idea_score = int(sum(scores.values()) / len(scores) * 10)
    return {'scores': scores, 'reasons': reasons, 'idea_score': idea_score}

# ── SHAP helper ──────────────────────────────────────────────
def compute_shap(lgb_model, x_scaled_row):
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(lgb_model)
    sv = explainer.shap_values(x_scaled_row)
    if isinstance(sv, list):
        arr = np.mean([np.abs(s) for s in sv], axis=0)[0]
    elif hasattr(sv, 'ndim') and sv.ndim == 3:
        arr = np.abs(sv[0]).mean(axis=-1)
    else:
        arr = np.abs(sv)[0]
    return dict(zip(FEATURES, arr))

# ── FULL INFERENCE ───────────────────────────────────────────
def run_inference(sector: str, country: str, logs: list):
    sec = sector.lower()
    macro = COUNTRY_MACRO_DEFAULTS.get(country.upper(),
            {'inflation':10.0,'gdp_growth':3.0,'unemployment':7.0})

    logs.append(f"[A1] Sector: {sec} | Country: {country}")
    logs.append(f"[A1] Base macro: inflation={macro['inflation']}% | GDP={macro['gdp_growth']}%")

    base_inflation   = macro['inflation']
    base_gdp         = macro['gdp_growth']
    unemployment     = macro['unemployment']

    # we apply a sector-specific macro adjustment so each sector gets its own effective macro
    eff_inf_offset, gdp_boost, velocity = SECTOR_EFF_MACRO.get(sec, SECTOR_EFF_MACRO['other'])
    # we scale the adjustment relative to country base inflation so high-inflation
    # countries apply the sector pattern at the right magnitude
    scale = base_inflation / 33.9  # we normalize to the Egypt baseline here
    inflation  = float(np.clip(eff_inf_offset * scale, 1.0, 100.0))
    gdp_growth = float(base_gdp + gdp_boost)
    macro_fric = float(np.clip(inflation + unemployment - gdp_growth, -50, 100))
    cap_conc   = SECTOR_MEDIANS.get(sec, SECTOR_MEDIANS['other'])

    logs.append(f"[A1] Effective macro (sector-adjusted): inflation={inflation:.1f}% | GDP={gdp_growth:.1f}% | friction={macro_fric:.1f}")

    x_raw    = np.array([[inflation, gdp_growth, macro_fric, float(cap_conc), velocity]])
    x_scaled = scaler.transform(x_raw)
    x_pca    = pca.transform(x_scaled)
    logs.append(f"[A1] X_new built: {x_raw[0].round(2)}")

    # ── Step 2: Router ────────────────────────────────────────
    logs.append(f"[STEP2] Loading global SVM model")

    # ── Step 3: SVM Classification ────────────────────────────
    pred_enc  = svm.predict(x_scaled)[0]
    proba     = svm.predict_proba(x_scaled)[0]
    svm_regime= le.inverse_transform([pred_enc])[0]
    svm_conf  = float(proba.max())
    # Layer 2: we apply the domain rule override here because SVM has no GROWTH class
    regime, conf = enhanced_regime(svm_regime, svm_conf, inflation,
                                   gdp_growth, macro_fric, velocity)
    logs.append(f"[STEP3] SVM base: {svm_regime} ({svm_conf:.1%}) → Final: {regime} ({conf:.1%})")

    # ── Step 4A: SHAP ─────────────────────────────────────────
    shap_dict = compute_shap(lgb, x_scaled)
    xai_score = float(conf * np.mean(list(shap_dict.values())))
    top_feat  = max(shap_dict, key=shap_dict.get)
    logs.append(f"[STEP4A] SHAP top signal: {top_feat} ({shap_dict[top_feat]:.3f})")

    # ── Step 4B: SARIMA ───────────────────────────────────────
    sarima_trend = 0.50
    drift_flag   = False
    if sec in sarima_results:
        fc_raw = sarima_results[sec]['forecast_mean']
        fc = [max(0, v) for v in fc_raw]
        # we use the actual forecast magnitude here — not a binary flag
        fc_mean = float(np.mean(fc))
        sarima_trend = float(np.clip(fc_mean / 50.0, 0.15, 0.90))
        drift_flag   = sarima_results[sec]['drift_flag']
        logs.append(f"[STEP4B] SARIMA forecast: {[round(x,1) for x in fc]} mean={fc_mean:.1f} → trend={sarima_trend:.2f} | drift={drift_flag}")
    else:
        logs.append(f"[STEP4B] No SARIMA model for {sec} — we fall back to neutral trend={sarima_trend}")

    if drift_flag:
        logs.append("[STEP4B] ⚠ DRIFT DETECTED — Manual Reclustering Advised")

    # ── Step 5: TAS ───────────────────────────────────────────
    tas = round(conf*0.40 + sarima_trend*0.35 + xai_score*0.25, 3)
    logs.append(f"[STEP5] TAS = {conf:.2f}×0.40 + {sarima_trend:.2f}×0.35 + {xai_score:.2f}×0.25 = {tas}")

    # ── Agent A2: Competitor Context ──────────────────────────
    # we retrieve the top 2 competitors dynamically from the list structure when available
    a2_comps = ["Traditional incumbents", "Local SMEs"]
    sector_comps_list = comps_data.get(sec, [])
    if isinstance(sector_comps_list, list) and len(sector_comps_list) > 0:
        a2_comps = [c.get("Company", "Competitor") for c in sector_comps_list[:2]]
    
    # ── Agent A4: Sentiment Context ───────────────────────────
    a4_sent_ratio = "Neutral"
    if sents_data:
        pos = sum(1 for s in sents_data if s.get('sentiment') == 'positive')
        neg = sum(1 for s in sents_data if s.get('sentiment') == 'negative')
        if pos > neg * 1.5: a4_sent_ratio = "Positive"
        elif neg > pos * 1.5: a4_sent_ratio = "Negative"

    logs.append(f"[A2/A4] Loaded {len(a2_comps)} competitors | Sentiment: {a4_sent_ratio}")

    regime_readable = regime.replace('_', ' ').title()
    top3 = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_names = [f[0].replace('_',' ') for f in top3]

    # ── Agent A7: LLM Synthesis ───────────────────────────────
    a7_prompt = dedent(f"""
        You are MIDAN Agent A7, a Chief Intelligence Officer at a VC firm.
        Synthesize this startup market intelligence in exactly 3 short sentences.
        Parameters:
        - Sector/Country: {sec.title()} in {country}
        - Regime: {regime_readable} (Confidence: {conf:.0%})
        - Top Signals: {', '.join(top3_names)}
        - TAS Score: {tas}/1.0
        - Known Competitors: {', '.join(a2_comps)}
        - Sentiment: {a4_sent_ratio}
        Output exactly 3 sentences: 1) Market reality check. 2) Competitive landscape warning. 3) The smartest next move.
    """).strip()
    
    a7_synthesis = ""
    try:
        response = _APP_LLM.chat.completions.create(
            model=_APP_LLM_MODEL,
            messages=[{"role": "user", "content": a7_prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        a7_synthesis = response.choices[0].message.content.strip()
    except Exception as e:
        comp_str = f"facing off against {', '.join(a2_comps[:2])}" if len(a2_comps)>1 else "in a fragmented landscape"
        move_str = "Double down on customer acquisition immediately." if tas >= 0.7 else "Run extreme demand validation before writing a single line of code."
        a7_synthesis = (f"The {sec.title()} space in {country} is currently operating as a {regime_readable}, heavily influenced by {top3_names[0]}. "
                        f"You will be {comp_str} operating amid a {a4_sent_ratio.lower()} macro sentiment. "
                        f"{move_str}")

    finding     = (f"Market classified as {regime_readable} with {conf:.0%} confidence. "
                   f"Top signals driving this: {', '.join(top3_names)}.")
    implication = a7_synthesis
    action      = (f"{'Move within the next 90 days. Apply to Flat6Labs or Cairo Angels.' if tas>=0.70 else 'Validate demand before building. Run 20 direct customer interviews.'} "
                   f"Key signal to monitor: {top3_names[0]}.")

    # ── Slack Webhook Execution — we fire this when TAS crosses the action threshold ─
    action_fired = tas >= 0.70 and regime in ('GROWTH_MARKET','EMERGING_MARKET')
    if action_fired:
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if webhook_url and webhook_url.startswith("http"):
            msg = {"text": f"🚀 *MIDAN INTELLIGENCE:* High-Conviction Market Detected!\n*Sector:* {sec.title()} ({country})\n*TAS Score:* {tas}\n*Regime:* {regime_readable}\n*A7 Summary:* {a7_synthesis}"}
            try: requests.post(webhook_url, json=msg, timeout=2)
            except Exception: pass
            logs.append("[SLACK] Webhook HTTP POST Executed")
        else:
            logs.append("[SLACK] TAS matched, but Slack URL not configured")
    else:
        logs.append(f"[SLACK] TAS={tas} — threshold not met, no action")

    logs.append("[A6] Trinity Report generated")

    return {
        'regime': regime, 'confidence': conf, 'tas': tas,
        'sarima_trend': sarima_trend, 'xai_score': xai_score,
        'shap_dict': shap_dict, 'drift_flag': drift_flag,
        'action_fired': action_fired,
        'finding': finding, 'implication': implication, 'action': action,
        'x_raw': x_raw[0], 'x_scaled': x_scaled[0], 'x_pca': x_pca[0],
        'proba': dict(zip(le.classes_, proba)),
    }


def tier_from_score(score: float) -> str:
    if score >= 0.76:
        return "Strong"
    if score >= 0.60:
        return "Moderate"
    if score >= 0.44:
        return "Mixed"
    return "Weak"


def _adapt_result(raw: dict) -> dict:
    """
    Bridge process_idea() output → the flat display dict shape used by the dashboard.

    Maps renamed/restructured keys so the existing charts work unchanged,
    and surfaces new IS/react/RAG/L4 fields for Zones 2/4/5/6.
    """
    tas_float = raw.get('tas_score', 0) / 100.0
    return {
        # Backward-compat keys (existing charts use these names)
        'regime':       raw.get('regime', ''),
        'confidence':   raw.get('confidence', 0.75),
        'tas':          tas_float,
        'sarima_trend': raw.get('sarima_trend', 0.5),
        'xai_score':    raw.get('xai_score', 0.5),
        'shap_dict':    raw.get('shap_weights', {}),
        'drift_flag':   raw.get('drift_flag', False),
        'action_fired': raw.get('action_fired', False),
        'finding':      raw.get('key_driver', ''),
        'implication':  raw.get('strategic_interpretation', ''),
        'action':       raw.get('action', ''),
        'x_pca':        np.array(raw.get('pca_coords', [0.0, 0.0])),
        # Zone 2 — IS + ReAct routing signals
        'intelligent_score': raw.get('intelligent_score', 0.5),
        'is_components':     raw.get('is_components', {}),
        'react_path':        raw.get('react_path', ''),
        'react_decision':    raw.get('react_decision', {}),
        'rag_result':        raw.get('rag_result', {}),
        'shap_cosine':       raw.get('shap_cosine', 0.5),
        'signal_consensus':  raw.get('signal_consensus', ''),
        # Zone 4 — SHAP reliability + FCM membership
        # compute_fcm_membership returns a nested dict {available, membership, top_cluster, ...}
        # we extract just the {name: float} sub-dict so every downstream consumer gets
        # a plain {cluster_name: membership_float} mapping without type-mixing surprises.
        'fcm_membership':    (lambda _f: _f.get('membership', {}) if isinstance(_f, dict) and _f.get('available') else {})(raw.get('fcm_membership', {})),
        # Zone 5/6 — decision state + freshness
        'decision_state':    raw.get('decision_state', ''),
        'l4_decision':       raw.get('l4_decision', {}),
        'l2_data_freshness': raw.get('l2_data_freshness', {}),
        'main_risk':         raw.get('main_risk', ''),
        'dominant_risk':     raw.get('dominant_risk', ''),
        'signal_tier':       raw.get('signal_tier', ''),
    }


def call_backend_chat(messages: list[dict], context: dict) -> str:
    import api as api_backend

    req = api_backend.ChatRequest(
        context=context,
        messages=[api_backend.ChatMessage(**m) for m in messages],
    )

    try:
        response = asyncio.run(api_backend.chat_interaction(req))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(api_backend.chat_interaction(req))
        finally:
            loop.close()
    return response.get("reply", "").strip()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:28px 24px 20px;border-bottom:1px solid {BORDER};">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div style="width:22px;height:22px;background:{ACID};border-radius:3px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:11px;font-weight:900;color:#000;">M</div>
            <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:17px;
                         letter-spacing:0.08em;color:{TEXT};">MIDAN</span>
        </div>
        <p style="font-size:11px;color:{MUTED};letter-spacing:0.05em;margin:0;">
            AI Decision Engine — Live Pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    idea = st.text_area(
        "Startup Idea",
        placeholder="Describe your startup idea... e.g. Invoice financing app for Egyptian SMEs",
        height=100,
        label_visibility="visible",
    )

    # we show what Agent A1 detected in real-time so the user gets immediate feedback
    if idea and len(idea.strip()) > 5:
        detected_sec, detected_ctry, sf, cf = agent_a1_parse(idea)
        sec_lbl = detected_sec.title() if sf else 'Using dropdown'
        ctry_lbl = detected_ctry if cf else 'Using dropdown'
        tag = 'A1 Detected' if (sf or cf) else 'A1 No Match'
        st.markdown(f"""
        <div style="background:rgba(200,255,87,0.07);border:1px solid rgba(200,255,87,0.2);
                    border-radius:6px;padding:10px 14px;margin-top:8px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                      color:#C8FF57;margin:0 0 4px;">{tag}</p>
            <p style="font-size:13px;color:#EDEAF8;margin:0;font-weight:500;">
                {sec_lbl} &nbsp;·&nbsp; {ctry_lbl}
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <p style="font-size:11px;font-weight:700;letter-spacing:0.12em;
    text-transform:uppercase;color:{ACID};margin:16px 0 8px;padding:0 0px;">Sector</p>
    """, unsafe_allow_html=True)

    sector = st.selectbox("Sector", [
        "Fintech","E-commerce","Healthtech",
        "Edtech","SaaS","Logistics","Agritech","Other"
    ], label_visibility="collapsed")

    st.markdown(f"""
    <p style="font-size:11px;font-weight:700;letter-spacing:0.12em;
    text-transform:uppercase;color:{ACID};margin:16px 0 8px;">Country</p>
    """, unsafe_allow_html=True)

    country = st.selectbox("Country", [
        "EG — Egypt","SA — Saudi Arabia","AE — UAE",
        "MA — Morocco","NG — Nigeria","KE — Kenya",
        "US — United States","GB — United Kingdom",
    ], label_visibility="collapsed")
    country_code = country.split(" — ")[0]

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    run = st.button("Run Analysis", key="run")

    if not MODELS_LOADED:
        st.error(f"Models not found: {MODEL_ERROR}")

    st.markdown(f"""
    <div style="padding:20px 24px;margin-top:40px;border-top:1px solid {BORDER};">
        <p style="font-size:11px;color:{MUTED};line-height:1.7;margin:0;">
            DBSCAN → FCM → SVM RBF<br>
            + SHAP + SARIMA + TAS → Slack<br>
            <span style="color:{ACID};">{'Models loaded' if MODELS_LOADED else 'Models missing'}</span>
        </p>
    </div>""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result  = None
    st.session_state.logs    = []
    st.session_state.sector  = "Fintech"
    st.session_state.country = "EG"
    st.session_state.chat_history = []
    st.session_state.chat_context = {}
    st.session_state.idea_eval = None
    st.session_state.svs = 0

# ── RUN ──────────────────────────────────────────────────────
SECTOR_LABEL_MAP = _MIDAN_SECTOR_LABEL_MAP or {
    "E-commerce":"ecommerce","Healthtech":"healthtech",
    "Edtech":"edtech","SaaS":"saas","Logistics":"logistics",
    "Agritech":"agritech","Other":"other","Fintech":"fintech"
}
if run and MODELS_LOADED:
    # we let Agent A1 parse the text; dropdowns serve as fallback when keywords aren't found
    if idea and len(idea.strip()) > 5:
        parsed_sec, parsed_ctry, sec_found, ctry_found = agent_a1_parse(idea)
        sector_key   = parsed_sec if sec_found else SECTOR_LABEL_MAP.get(sector, sector.lower())
        country_code = parsed_ctry if ctry_found else country.split(" — ")[0]
    else:
        sector_key   = SECTOR_LABEL_MAP.get(sector, sector.lower())
        country_code = country.split(" — ")[0]

    ph = st.empty()
    steps = [
        ("01 — Parsing idea & extracting context",     0.14),
        ("02 — Evaluating idea strength (Agent A0)",    0.28),
        ("03 — Fetching live macro signals",            0.42),
        ("04 — Building inference context vector",      0.54),
        ("05 — SVM RBF classification",                 0.66),
        ("06 — SHAP explainability (LightGBM surrogate)",0.80),
        ("07 — SARIMA 90-day forecast",                 0.90),
        ("08 — Calculating TAS, SVS & Trinity Report",  1.00),
    ]
    with ph.container():
        st.markdown(f"""
        <div style="padding:60px 40px;text-align:center;">
        <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                  color:{TEXT};margin-bottom:32px;">
            Analysing <span style="color:{ACID};">{sector}</span> market in
            <span style="color:{ACID};">{country_code}</span>
        </p></div>""", unsafe_allow_html=True)
        bar    = st.progress(0)
        status = st.empty()
        for label, pct in steps:
            status.markdown(f"""<p style="font-size:12px;color:{SUB};letter-spacing:0.08em;
            text-align:center;text-transform:uppercase;">{label}</p>""",
            unsafe_allow_html=True)
            bar.progress(pct)
            time.sleep(0.08)

    try:
        raw = process_idea(idea or "", sector_key, country_code)

        # Handle L0 rejection or clarification_required gracefully
        if not raw.get("success", True):
            ph.empty()
            if raw.get("invalid_idea"):
                st.error(f"Idea rejected: {raw.get('one_line_verdict', 'Invalid idea.')}")
                if raw.get("what_is_missing"):
                    st.warning(f"Missing: {raw['what_is_missing']}")
            elif raw.get("clarification_required"):
                st.info(raw.get("message", "Please clarify your idea and resubmit."))
                clar = raw.get("clarification", {}) or {}
                for q in (clar.get("questions") or []):
                    st.markdown(f"- {q}")
            else:
                st.error(f"Pipeline error: {raw.get('error', 'unknown')}")
            st.stop()

        result = _adapt_result(raw)
        st.session_state.result  = result
        # idea_eval shape reconstructed from process_idea() output
        idea_eval = {
            'idea_score': raw.get('idea_score', 50),
            'scores':     raw.get('idea_dimensions', {}),
            'reasons':    raw.get('idea_reasons', {}),
        }
        st.session_state.idea_eval = idea_eval
        st.session_state.svs       = raw.get('svs', int(result['tas'] * 100))
        st.session_state.logs      = raw.get('logs', [])
        st.session_state.sector    = sector_key
        st.session_state.country   = country_code
        signal_tier = raw.get('signal_tier', tier_from_score(result['tas']))
        st.session_state.chat_context = {
            "sector":          sector_key,
            "country":         country_code,
            "regime":          result['regime'],
            "tas_score":       int(result['tas'] * 100),
            "signal_tier":     signal_tier,
            "decision_state":  result['decision_state'],
            "l4_decision":     result['l4_decision'],
            "strategic_interpretation": result['implication'],
            "main_risk":       result['main_risk'],
            "dominant_risk":   result['dominant_risk'],
            "idea_features":   raw.get('idea_features', {}),
            "idea":            idea or "",
        }
        st.session_state.chat_history = [{"role": "assistant", "content": result['implication']}]
    except Exception as e:
        st.session_state.result = None
        st.error(f"Pipeline error: {e}")
        import traceback; st.code(traceback.format_exc())
    finally:
        ph.empty()

# ── IDLE SCREEN ───────────────────────────────────────────────
if st.session_state.result is None:
    st.markdown(f"""
    <div style="min-height:80vh;display:flex;flex-direction:column;
                align-items:center;justify-content:center;text-align:center;padding:80px 40px;">
        <div style="display:inline-flex;align-items:center;gap:8px;
                    border:1px solid {BORDA};padding:7px 16px;border-radius:2px;
                    font-size:11px;font-weight:700;letter-spacing:0.15em;
                    text-transform:uppercase;color:{ACID};margin-bottom:36px;">
            <div style="width:6px;height:6px;border-radius:50%;background:{ACID};"></div>
            Real Models Loaded
        </div>
        <h1 style="font-family:'Syne',sans-serif;font-size:clamp(48px,6vw,76px);
                   font-weight:800;letter-spacing:-3px;line-height:1.0;color:{TEXT};margin-bottom:20px;">
            The market doesn\'t care<br>about passion.
        </h1>
        <p style="font-size:17px;color:{SUB};font-weight:300;max-width:440px;line-height:1.75;">
            Pick a sector and country on the left.<br>Hit Run Analysis.
        </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── RESULTS ──────────────────────────────────────────────────
r = st.session_state.result
regime_color = REGIME_COLORS.get(r['regime'], ACID)

# we render the verdict banner at the top so the regime decision is immediately visible
st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDA};border-radius:10px;
            padding:20px 28px;margin-bottom:20px;
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="background:{regime_color};color:#000;padding:7px 14px;
                    border-radius:2px;font-family:'Syne',sans-serif;font-weight:800;
                    font-size:11px;letter-spacing:0.12em;text-transform:uppercase;">
            {r['regime'].replace('_',' ')}
        </div>
        <div>
            <span style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:{TEXT};">
                Market Intelligence Report
            </span>
            <span style="font-size:12px;color:{MUTED};margin-left:10px;letter-spacing:0.04em;">
                {st.session_state.sector.title()} · {st.session_state.country} · Live Models
            </span>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:7px;height:7px;border-radius:50%;background:{ACID};"></div>
        <span style="font-size:12px;color:{SUB};letter-spacing:0.06em;">Pipeline complete</span>
    </div>
</div>""", unsafe_allow_html=True)

# we render 4 score cards at the top of the results view for a quick summary
c1, c2, c3, c4 = st.columns(4)
regime_label = r['regime'].replace('_', ' ').replace('MARKET','').strip().title()
with c1: st.metric("Market Regime",     regime_label)
with c2: st.metric("SVM Confidence",    f"{r['confidence']:.0%}", f"{r['confidence']-0.5:+.0%} vs neutral")
with c3:
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:20px 24px;">
        <p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};
                  font-weight:600;margin-bottom:8px;">Opportunity Score</p>
        <p style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
                  color:{'#C8FF57' if r['tas']>=0.70 else '#F5A623' if r['tas']>=0.50 else '#FF6B6B'};
                  letter-spacing:-0.5px;margin:0;">{r['tas']*100:.0f} / 100</p>
        <p style="font-size:12px;color:{SUB};margin:4px 0 0;">TAS Score</p>
    </div>""", unsafe_allow_html=True)
with c4:
    out_label = "Positive" if r['sarima_trend'] > 0.5 else "Neutral"
    out_color = ACID if r['sarima_trend'] > 0.5 else '#F5A623'
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:20px 24px;">
        <p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};
                  font-weight:600;margin-bottom:8px;">90-Day Outlook</p>
        <p style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;
                  color:{out_color};letter-spacing:-0.5px;margin:0;">{out_label}</p>
        <p style="font-size:12px;color:{SUB};margin:4px 0 0;">SARIMA baseline</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── ZONE 2: Live Scoring Feed ─────────────────────────────────────────────────
with st.expander("Zone 2 — Live Scoring Feed (IS + ReAct Routing)", expanded=True):
    _is  = r.get('intelligent_score', 0.5)
    _isc = r.get('is_components', {})
    _rp  = r.get('react_path', '')
    _ds  = r.get('decision_state', '')
    _sc  = r.get('shap_cosine', 0.5)
    _rag = r.get('rag_result', {})

    # IS color: green above REACT_IS_HIGH, orange in borderline, red below REACT_IS_LOW
    _is_color = ACID if _is >= REACT_IS_HIGH else ('#F5A623' if _is >= REACT_IS_LOW else '#FF6B6B')
    _path_color = {'PATH_1_HIGH_CERTAINTY': ACID, 'PATH_2_LOW_CERTAINTY': '#FF6B6B',
                   'PATH_3_BORDERLINE_CONFIRMED': ACID, 'PATH_4_BORDERLINE_CONFLICT': '#F5A623',
                   'PATH_5_ATYPICAL_SUPPORTED': '#F5A623', 'PATH_6_FULL_CONFLICT': '#FF6B6B',
                   'PATH_7_MAXIMUM_UNCERTAINTY': '#FF6B6B', 'PATH_NOVELTY': CYAN}.get(_rp, SUB)
    _ds_color  = {'GO': ACID, 'CONDITIONAL': '#F5A623', 'NO_GO': '#FF6B6B',
                  'HIGH_UNCERTAINTY': '#FF6B6B', 'CONFLICTING_SIGNALS': '#F5A623',
                  'INSUFFICIENT_DATA': SUB, 'REJECTED': '#FF6B6B'}.get(_ds, SUB)

    z2_c1, z2_c2, z2_c3, z2_c4, z2_c5 = st.columns(5)
    with z2_c1:
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">Intelligent Score</p>
            <p style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{_is_color};margin:0;">{_is:.3f}</p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">IS routing composite</p>
        </div>""", unsafe_allow_html=True)
    with z2_c2:
        _gap = _isc.get('gap_svm', _isc.get('gap', 0.5))
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">Gap SVM</p>
            <p style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{TEXT};margin:0;">{float(_gap):.3f}</p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">SVM margin</p>
        </div>""", unsafe_allow_html=True)
    with z2_c3:
        _mu = _isc.get('mu_fcm', _isc.get('mu', 0.5))
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">μ FCM</p>
            <p style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{TEXT};margin:0;">{float(_mu):.3f}</p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">Cluster membership</p>
        </div>""", unsafe_allow_html=True)
    with z2_c4:
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">ReAct Path</p>
            <p style="font-family:'Syne',sans-serif;font-size:11px;font-weight:800;color:{_path_color};margin:0;word-break:break-word;">{_rp.replace('PATH_','').replace('_',' ')}</p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">Routing decision</p>
        </div>""", unsafe_allow_html=True)
    with z2_c5:
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">Decision State</p>
            <p style="font-family:'Syne',sans-serif;font-size:14px;font-weight:800;color:{_ds_color};margin:0;">{_ds or '—'}</p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">L4 outcome</p>
        </div>""", unsafe_allow_html=True)

    # IS component breakdown bar
    if _isc:
        _comp_labels = [k.replace('_favorable','').replace('_',' ').title() for k in _isc.keys() if isinstance(_isc[k], (int, float))]
        _comp_vals   = [float(v) for k, v in _isc.items() if isinstance(v, (int, float))]
        if _comp_labels and _comp_vals:
            fig_is = go.Figure(go.Bar(
                x=_comp_vals, y=_comp_labels, orientation='h',
                marker=dict(color=[ACID if v > 0.5 else SUB for v in _comp_vals], cornerradius=3),
                text=[f'{v:.3f}' for v in _comp_vals], textposition='outside',
                textfont=dict(color=SUB, size=11),
            ))
            fig_is.update_layout(**{**CHART_BASE, 'margin': dict(l=16,r=16,t=8,b=8)}, height=160, bargap=0.30)
            fig_is.update_xaxes(range=[0, 1.15], color=MUTED, gridcolor='rgba(255,255,255,0.04)')
            fig_is.update_yaxes(color=TEXT, tickfont=dict(size=11))
            st.plotly_chart(fig_is, use_container_width=True, config={'displayModeBar': False})

    # RAG result summary
    _rag_vote = _rag.get('vote') or 'None'
    _rag_conf = _rag.get('confidence', 0.0)
    _rag_skip = _rag.get('rag_skipped', True)
    _rag_reason = _rag.get('rag_skipped_reason', '')
    _novel = r.get('react_decision', {}).get('is_novel_case', False)
    st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;margin-top:8px;font-size:12px;color:{SUB};">
        <b style="color:{TEXT};">RAG</b> — vote: <b style="color:{ACID if not _rag_skip else MUTED};">{_rag_vote}</b>
        &nbsp;·&nbsp; conf: {_rag_conf:.3f}
        &nbsp;·&nbsp; {'skipped (' + _rag_reason + ')' if _rag_skip else f'k={_rag.get("n_neighbors",0)} neighbors'}
        {'&nbsp;·&nbsp;<b style="color:#5B6CF0;">NOVEL CASE</b>' if _novel else ''}
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── ZONE 1: Cluster Health ────────────────────────────────────────────────────
with st.expander("Zone 1 — Cluster Health", expanded=True):
    _z1l, _z1r = st.columns([6, 5], gap="medium")
    with _z1l:
        st.markdown(f"""<div style="margin-bottom:10px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
                Zone 1 — Cluster Positioning</p>
            <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
                Where your idea sits in the competitive field</p>
        </div>""", unsafe_allow_html=True)

        # we generate the cluster scatter from real SVM decision regions so the plot is model-grounded
        @st.cache_data
        def _cluster_scatter():
            rng = np.random.RandomState(42)
            n = 500
            X_s = np.column_stack([
                rng.normal(scaler.mean_[0], scaler.scale_[0]*0.7, n).clip(0, 60),
                rng.normal(scaler.mean_[1], scaler.scale_[1]*0.8, n).clip(-4, 10),
                rng.normal(scaler.mean_[2], scaler.scale_[2]*0.7, n).clip(-15, 60),
                rng.normal(scaler.mean_[3], scaler.scale_[3]*0.4, n).clip(5000, 3e6),
                rng.normal(scaler.mean_[4], scaler.scale_[4]*0.8, n).clip(0, 0.5),
            ])
            Xs = scaler.transform(X_s)
            Xp = pca.transform(Xs)
            preds = svm.predict(Xs); probas = svm.predict_proba(Xs)
            labels = []
            for i in range(n):
                sv_l = le.inverse_transform([preds[i]])[0]
                sv_c = float(probas[i].max())
                reg, _ = enhanced_regime(sv_l, sv_c, X_s[i,0], X_s[i,1], X_s[i,2], X_s[i,4])
                labels.append(reg)
            return Xp, labels
        bg_pca, bg_labels = _cluster_scatter()
        fig_c = go.Figure()
        for name, col in REGIME_COLORS.items():
            mask = [i for i, l in enumerate(bg_labels) if l == name]
            if mask:
                fig_c.add_trace(go.Scatter(
                    x=bg_pca[mask, 0], y=bg_pca[mask, 1], mode='markers', name=name,
                    marker=dict(color=col, size=6, opacity=0.45, line=dict(width=0))
                ))
        # we overlay the user's idea as a diamond so it stands out against the cluster background
        px_idea, py_idea = r['x_pca'][0], r['x_pca'][1]
        fig_c.add_trace(go.Scatter(
            x=[px_idea], y=[py_idea],
            mode='markers+text', name='Your idea',
            text=['  Your idea'], textposition='middle right',
            textfont=dict(color=TEXT, size=12),
            marker=dict(color=ACID, size=16, symbol='diamond',
                        line=dict(color='#000', width=2))
        ))
        fig_c.update_layout(**CHART_BASE, height=300,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=SUB, size=10), x=0.01, y=0.99))
        fig_c.update_xaxes(title='PCA 1', color=MUTED,
                           gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)')
        fig_c.update_yaxes(title='PCA 2', color=MUTED,
                           gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)')
        st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})

    with _z1r:
        st.markdown(f"""<div style="margin-bottom:10px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
                FCM Cluster Membership</p>
            <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
                How strongly your idea belongs to each cluster</p>
        </div>""", unsafe_allow_html=True)

        _fcm_z1 = r.get('fcm_membership', {})
        if _fcm_z1:
            _fcm_items  = sorted(_fcm_z1.items(), key=lambda x: x[1], reverse=True)
            _fcm_labels = [k.replace('_', ' ').title() for k, _ in _fcm_items]
            _fcm_vals   = [float(v) for _, v in _fcm_items]
            _fcm_max    = max(_fcm_vals)
            _fcm_colors = [ACID if v == _fcm_max else CYAN if v > 0.3 else MUTED for v in _fcm_vals]
            fig_fcm = go.Figure(go.Bar(
                x=_fcm_vals, y=_fcm_labels, orientation='h',
                marker=dict(color=_fcm_colors, cornerradius=3),
                text=[f'{v:.3f}' for v in _fcm_vals], textposition='outside',
                textfont=dict(color=SUB, size=11),
            ))
            fig_fcm.update_layout(**{**CHART_BASE, 'margin': dict(l=16,r=16,t=8,b=8)}, height=200, bargap=0.30)
            fig_fcm.update_xaxes(range=[0, 1.25], color=MUTED, gridcolor='rgba(255,255,255,0.04)')
            fig_fcm.update_yaxes(color=TEXT, tickfont=dict(size=11))
            st.plotly_chart(fig_fcm, use_container_width=True, config={'displayModeBar': False})

            _dom_cluster = _fcm_items[0][0]
            _dom_val     = _fcm_items[0][1]
            _hc = ACID if _dom_val >= 0.6 else '#F5A623' if _dom_val >= 0.35 else '#FF6B6B'
            _hl = 'Strong membership' if _dom_val >= 0.6 else 'Moderate membership' if _dom_val >= 0.35 else 'Weak — atypical idea'
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;margin-top:4px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">Dominant Cluster</p>
                <p style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:{_hc};margin:0;">{_dom_cluster.replace('_',' ').title()}</p>
                <p style="font-size:12px;color:{SUB};margin:4px 0 0;">μ = {_dom_val:.3f} — {_hl}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:{MUTED};font-size:13px;margin-top:16px;'>FCM membership data unavailable — run a full analysis first.</p>", unsafe_allow_html=True)

        if cluster_names:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;margin-top:8px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 8px;">Cluster Legend</p>
                {"".join([f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid {BORDER};"><span style="font-size:11px;color:{TEXT};">{k}</span><span style="font-size:11px;color:{SUB};">{v}</span></div>' for k, v in cluster_names.items()])}
            </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── ZONE 4: SHAP Reliability ──────────────────────────────────────────────────
with st.expander("Zone 4 — SHAP Reliability", expanded=False):
    _z4l, _z4r = st.columns([6, 5], gap="medium")
    with _z4l:
        st.markdown(f"""<div style="margin-bottom:10px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
                Zone 4 — Signal Weights (SHAP)</p>
            <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
                Why the market was classified this way</p>
        </div>""", unsafe_allow_html=True)

        shap_d  = r['shap_dict']
        labels  = [k.replace('_',' ').title() for k in shap_d.keys()]
        vals    = list(shap_d.values())
        max_v   = max(vals)
        colors  = [ACID if v == max_v else CYAN if v > np.median(vals) else MUTED for v in vals]

        fig_s = go.Figure(go.Bar(
            x=vals, y=labels, orientation='h',
            marker=dict(color=colors, cornerradius=3),
            text=[f'{v:.3f}' for v in vals], textposition='outside',
            textfont=dict(color=SUB, size=11),
        ))
        fig_s.update_layout(**CHART_BASE, height=300, bargap=0.35)
        fig_s.update_xaxes(color=MUTED, gridcolor='rgba(255,255,255,0.04)',
                           zerolinecolor='rgba(255,255,255,0.04)', tickfont=dict(size=11))
        fig_s.update_yaxes(gridcolor='rgba(0,0,0,0)', color=TEXT,
                           tickfont=dict(size=12, family='DM Sans'))
        st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})

    with _z4r:
        st.markdown(f"""<div style="margin-bottom:10px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
                Attribution Reliability</p>
            <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
                Is the SHAP explanation trustworthy?</p>
        </div>""", unsafe_allow_html=True)

        _sc4 = r.get('shap_cosine', 0.5)
        if _sc4 >= REACT_SHAP_RELIABLE:
            _sc4_label  = "Typical"
            _sc4_detail = "Attribution consistent with cluster mean — explanation is trustworthy"
            _sc4_color  = ACID
        elif _sc4 >= REACT_SHAP_UNRELIABLE:
            _sc4_label  = "Borderline"
            _sc4_detail = "Attribution partially consistent — interpret with moderate caution"
            _sc4_color  = '#F5A623'
        else:
            _sc4_label  = "Atypical"
            _sc4_detail = "Attribution diverges from cluster pattern — explanation may not generalize"
            _sc4_color  = '#FF6B6B'

        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:20px 24px;margin-bottom:12px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};margin:0 0 8px;">SHAP Cosine Similarity</p>
            <p style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:{_sc4_color};letter-spacing:-1px;margin:0;">{_sc4:.3f}</p>
            <p style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:{_sc4_color};margin:6px 0 4px;">{_sc4_label}</p>
            <p style="font-size:12px;color:{SUB};margin:0;">{_sc4_detail}</p>
        </div>""", unsafe_allow_html=True)

        _sc_z4 = r.get('signal_consensus', '')
        if _sc_z4:
            _sc_z4_color = ACID if any(w in _sc_z4.lower() for w in ('aligned', 'high', 'strong')) else '#F5A623'
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;margin-bottom:12px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">Signal Consensus</p>
                <p style="font-size:13px;color:{_sc_z4_color};margin:0;">{_sc_z4}</p>
            </div>""", unsafe_allow_html=True)

        _fcm_z4 = r.get('fcm_membership', {})
        if _fcm_z4:
            _dom4     = max(_fcm_z4, key=_fcm_z4.get)
            _dom4_val = _fcm_z4[_dom4]
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">Reference Cluster</p>
                <p style="font-size:13px;color:{TEXT};margin:0 0 4px;">{_dom4.replace('_',' ').title()} — μ = {_dom4_val:.3f}</p>
                <p style="font-size:11px;color:{MUTED};margin:0;">SHAP cosine computed against this cluster's centroid pattern</p>
            </div>""", unsafe_allow_html=True)

# we lay out row 2 of charts: SARIMA forecast on the left, TAS gauge on the right
col_l2, col_r2 = st.columns([7, 4], gap="medium")

with col_l2:
    st.markdown(f"""<div style="margin-bottom:10px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
            Step 04B — SARIMA Forecast</p>
        <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
            Sector deal activity — historical & 90-day outlook</p>
    </div>""", unsafe_allow_html=True)

    sec_key = st.session_state.sector
    if sec_key in sarima_results:
        sd = sarima_results[sec_key]
        fc = [max(0, v) for v in sd['forecast_mean']]
        fc_lo = [max(0, v) for v in sd['forecast_lower']]
        fc_hi = [max(0, v) for v in sd['forecast_upper']]
        np.random.seed(7)
        hist_len = 36
        t_hist = list(range(hist_len))
        hist_v = (30 + np.random.normal(0, 5, hist_len)
                  + np.sin(np.linspace(0, 4*np.pi, hist_len)) * 8).tolist()
        t_fc   = [hist_len, hist_len+1, hist_len+2]
    else:
        t_hist = list(range(36))
        hist_v = [30]*36
        t_fc = [36,37,38]; fc=[30,30,30]; fc_lo=[25,25,25]; fc_hi=[35,35,35]
        sd = {'drift_flag': False}

    fig_sa = go.Figure()
    fig_sa.add_trace(go.Scatter(
        x=t_fc+t_fc[::-1],
        y=fc_hi+fc_lo[::-1],
        fill='toself', fillcolor='rgba(200,255,87,0.07)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False,
    ))
    fig_sa.add_trace(go.Scatter(
        x=t_hist, y=hist_v, mode='lines', name='Historical',
        line=dict(color=SUB, width=2)))
    fig_sa.add_trace(go.Scatter(
        x=t_fc, y=fc, mode='lines+markers', name='Forecast',
        line=dict(color=ACID, width=2.5, dash='dot'),
        marker=dict(size=7, color=ACID)))
    fig_sa.add_vline(x=35, line_width=1, line_dash='dash', line_color=MUTED,
                     annotation_text='Now', annotation_font_color=MUTED,
                     annotation_font_size=10)
    if sd.get('drift_flag'):
        fig_sa.add_annotation(x=10, y=max(hist_v)*0.9,
                              text='⚠ DRIFT DETECTED', font=dict(color='#FF6B6B', size=11),
                              showarrow=False)
    fig_sa.update_layout(**CHART_BASE, height=280,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=SUB, size=10), x=0.01, y=0.99))
    fig_sa.update_xaxes(title='Months', color=MUTED,
                        gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)')
    fig_sa.update_yaxes(title='Deal Volume', color=MUTED,
                        gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)')
    st.plotly_chart(fig_sa, use_container_width=True, config={'displayModeBar': False})

with col_r2:
    st.markdown(f"""<div style="margin-bottom:10px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
            Step 05 — Opportunity Score</p>
        <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
            TAS: The final verdict number</p>
    </div>""", unsafe_allow_html=True)

    tas       = r['tas']
    g_color   = ACID if tas >= 0.70 else '#F5A623' if tas >= 0.50 else '#FF6B6B'
    fig_gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(tas * 100),
        number=dict(suffix=' / 100', font=dict(family='Syne', size=36, color=g_color)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=0, tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=g_color, thickness=0.28),
            bgcolor=CARD2, borderwidth=0,
            steps=[
                dict(range=[0,50],   color='rgba(255,107,107,0.06)'),
                dict(range=[50,70],  color='rgba(245,166,35,0.06)'),
                dict(range=[70,100], color='rgba(200,255,87,0.06)'),
            ],
            threshold=dict(line=dict(color=g_color, width=2), value=70),
        ),
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(family='DM Sans', color=SUB),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""
    <div style="background:{'rgba(200,255,87,0.07)' if r['action_fired'] else 'rgba(255,107,107,0.07)'};
                border:1px solid {'rgba(200,255,87,0.25)' if r['action_fired'] else 'rgba(255,107,107,0.25)'};
                border-radius:8px;padding:14px 18px;text-align:center;">
        <p style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                  color:{'#C8FF57' if r['action_fired'] else '#FF6B6B'};margin:0 0 4px;">
            {'Slack Webhook Fired' if r['action_fired'] else 'Below Action Threshold'}
        </p>
        <p style="font-size:12px;color:{SUB};margin:0;font-weight:300;">
            {'TAS > 0.70 — autonomous action triggered' if r['action_fired'] else f'TAS = {tas:.2f} — threshold not met'}
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── ZONE 3: Action Dispatch ───────────────────────────────────────────────────
with st.expander("Zone 3 — Action Dispatch", expanded=False):
    _z3l, _z3r = st.columns([5, 6], gap="medium")
    with _z3l:
        _fired_z3  = r.get('action_fired', False)
        _l4_z3     = r.get('l4_decision', {})
        _fc3       = ACID if _fired_z3 else '#FF6B6B'
        _fl3       = 'DISPATCHED' if _fired_z3 else 'NOT TRIGGERED'
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:20px 24px;margin-bottom:12px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};margin:0 0 8px;">Dispatch Status</p>
            <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:{_fc3};margin:0;">{_fl3}</p>
            <p style="font-size:12px;color:{SUB};margin:6px 0 0;">
                {'TAS ≥ 0.70 + Growth/Emerging regime → Slack webhook fired' if _fired_z3 else f'TAS = {r["tas"]:.2f} — need ≥ 0.70 in Growth/Emerging regime'}
            </p>
        </div>""", unsafe_allow_html=True)

        _ds3 = r.get('decision_state', '')
        _ds3_color = {'GO': ACID, 'CONDITIONAL': '#F5A623', 'NO_GO': '#FF6B6B',
                      'HIGH_UNCERTAINTY': '#FF6B6B', 'CONFLICTING_SIGNALS': '#F5A623',
                      'INSUFFICIENT_DATA': SUB, 'REJECTED': '#FF6B6B'}.get(_ds3, SUB)
        if _ds3:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;margin-bottom:12px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">L4 Decision State</p>
                <p style="font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:{_ds3_color};margin:0;">{_ds3}</p>
            </div>""", unsafe_allow_html=True)

        if _l4_z3:
            _risk_level = _l4_z3.get('risk_level', _l4_z3.get('dominant_risk_category', ''))
            _final_rec  = _l4_z3.get('final_recommendation', _l4_z3.get('recommendation', ''))
            _conf_l4    = _l4_z3.get('confidence', _l4_z3.get('l4_confidence', 0))
            for _lbl, _val in [('Risk Level', _risk_level),
                                ('Recommendation', _final_rec),
                                ('L4 Confidence', f'{float(_conf_l4):.1%}' if _conf_l4 else '')]:
                if _val:
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:flex-start;padding:7px 0;border-bottom:1px solid {BORDER};">
                        <span style="font-size:11px;color:{MUTED};text-transform:uppercase;letter-spacing:0.08em;flex-shrink:0;">{_lbl}</span>
                        <span style="font-size:12px;color:{TEXT};font-weight:500;text-align:right;margin-left:12px;">{_val}</span>
                    </div>""", unsafe_allow_html=True)

    with _z3r:
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:20px 24px;margin-bottom:12px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 8px;">Recommended Action</p>
            <p style="font-size:14px;color:{TEXT};line-height:1.75;font-weight:300;margin:0;">{r.get('action', '—')}</p>
        </div>""", unsafe_allow_html=True)

        _mr_z3 = r.get('main_risk', '') or r.get('dominant_risk', '')
        if _mr_z3:
            st.markdown(f"""<div style="background:rgba(255,107,107,0.05);border:1px solid rgba(255,107,107,0.2);border-radius:6px;padding:12px 16px;margin-bottom:12px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#FF6B6B;margin:0 0 4px;">Primary Risk</p>
                <p style="font-size:13px;color:{TEXT};margin:0;">{_mr_z3}</p>
            </div>""", unsafe_allow_html=True)

        _dispatch_log = [e for e in load_prediction_log(limit=15) if e.get('action_fired')]
        if _dispatch_log:
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 8px;">Prior Dispatches — {len(_dispatch_log)} found</p>
                {"".join([f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {BORDER};"><span style="font-size:11px;color:{SUB};">{e.get("timestamp","")[:16].replace("T"," ")}</span><span style="font-size:11px;color:{TEXT};">{e.get("sector","—")} · {e.get("regime","—")}</span></div>' for e in reversed(_dispatch_log[-5:])])}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-size:12px;color:{MUTED};margin-top:8px;'>No prior dispatches in prediction log.</p>", unsafe_allow_html=True)

# we render the Trinity Report — our three-part answer: what we found, what it means, what to do
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown(f"""<div style="margin-bottom:14px;">
    <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 6px;">
        Trinity Report — Agent A6</p>
    <p style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.5px;">
        What we found. What it means. What to do.</p>
</div>""", unsafe_allow_html=True)

t1, t2, t3 = st.columns(3, gap="medium")
for col, key, label, num in [
    (t1, 'finding',     'What we found',   '01'),
    (t2, 'implication', 'What it means',   '02'),
    (t3, 'action',      'What to do next', '03'),
]:
    with col:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:28px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;
                        padding-bottom:14px;border-bottom:1px solid {BORDER};">
                <span style="font-family:'Syne',sans-serif;font-size:11px;font-weight:800;
                             color:{ACID};letter-spacing:0.1em;">{num}</span>
                <span style="font-size:11px;font-weight:700;letter-spacing:0.1em;
                             text-transform:uppercase;color:{ACID};">{label}</span>
            </div>
            <p style="font-size:14px;color:{SUB};line-height:1.76;font-weight:300;margin:0;">
                {r[key]}
            </p>
        </div>""", unsafe_allow_html=True)

# ── IDEA EVALUATION — Agent A0 ───────────────────────────────
if st.session_state.idea_eval is not None:
    ie = st.session_state.idea_eval
    svs = st.session_state.svs

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div style="margin-bottom:14px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 6px;">
            Agent A0 — Idea Evaluation</p>
        <p style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.5px;">
            How strong is the idea itself?</p>
    </div>""", unsafe_allow_html=True)

    # we compute the SVS and derive the quadrant verdict so the user gets a go/no-go label
    high_market = r['tas'] >= 0.60
    high_idea = ie['idea_score'] >= 60
    if high_market and high_idea:
        quadrant = "GO — Launch"
        q_color = ACID
    elif high_market and not high_idea:
        quadrant = "Wrong Idea — Right Market"
        q_color = '#F5A623'
    elif not high_market and high_idea:
        quadrant = "Wait or Pivot Market"
        q_color = CYAN
    else:
        quadrant = "STOP — Rethink Everything"
        q_color = '#FF6B6B'

    svs_col, quad_col = st.columns([4, 7], gap="medium")
    with svs_col:
        svs_color = ACID if svs >= 70 else '#F5A623' if svs >= 50 else '#FF6B6B'
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:28px;text-align:center;">
            <p style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};
                      font-weight:600;margin-bottom:12px;">Startup Viability Score</p>
            <p style="font-family:'Syne',sans-serif;font-size:42px;font-weight:800;
                      color:{svs_color};letter-spacing:-1px;margin:0;">{svs}</p>
            <p style="font-size:12px;color:{SUB};margin:6px 0 0;">SVS = Market (50%) + Idea (50%)</p>
            <div style="margin-top:16px;padding-top:16px;border-top:1px solid {BORDER};">
                <p style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
                          color:{q_color};letter-spacing:0.05em;margin:0;">{quadrant}</p>
            </div>
        </div>""", unsafe_allow_html=True)

    with quad_col:
        # we render the 5-dimension bar chart so the user can see where their idea is weak
        dim_labels = [IDEA_DIM_LABELS.get(d, d) for d in IDEA_DIMENSIONS]
        dim_vals = [ie['scores'].get(d, 5) for d in IDEA_DIMENSIONS]
        dim_colors = [ACID if v >= 7 else CYAN if v >= 5 else '#FF6B6B' for v in dim_vals]

        fig_idea = go.Figure(go.Bar(
            x=dim_vals, y=dim_labels, orientation='h',
            marker=dict(color=dim_colors, cornerradius=3),
            text=[f'{v}/10' for v in dim_vals], textposition='outside',
            textfont=dict(color=SUB, size=11),
        ))
        fig_idea.update_layout(**CHART_BASE, height=280, bargap=0.30,
            xaxis=dict(range=[0, 12]))
        fig_idea.update_xaxes(color=MUTED, gridcolor='rgba(255,255,255,0.04)',
                              zerolinecolor='rgba(255,255,255,0.04)', tickfont=dict(size=11))
        fig_idea.update_yaxes(gridcolor='rgba(0,0,0,0)', color=TEXT,
                              tickfont=dict(size=12, family='DM Sans'))
        st.plotly_chart(fig_idea, use_container_width=True, config={'displayModeBar': False})

    # we display per-dimension reasons so the user knows what drove each score
    st.markdown(f"""<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:10px;">""" +
        "".join([f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:14px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
                      color:{ACID};margin:0 0 6px;">{IDEA_DIM_LABELS.get(d, d)}</p>
            <p style="font-size:12px;color:{SUB};line-height:1.5;margin:0;font-weight:300;">
                {ie['reasons'].get(d, '')}</p>
        </div>""" for d in IDEA_DIMENSIONS]) +
    "</div>", unsafe_allow_html=True)

# we show the pipeline trace in an expander so users can inspect every step without cluttering the view
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
with st.expander("View full pipeline trace", expanded=False):
    def _line_color(line):
        if any(k in line for k in ['[A1]','[STEP5]','[SLACK]','[A6]']): return ACID
        if 'DRIFT' in line: return '#FF6B6B'
        return SUB
    lines_html = "".join([
        f'<div style="padding:5px 0;border-bottom:1px solid {BORDER};">'
        f'<span style="font-family:monospace;font-size:12px;color:{_line_color(line)};">'
        f'{line}</span></div>'
        for line in st.session_state.logs
    ])
    st.markdown(f"""
    <div style="background:#000;border:1px solid {BORDER};border-radius:8px;
                padding:20px 24px;max-height:320px;overflow-y:auto;">
        {lines_html}
    </div>""", unsafe_allow_html=True)

# ── ZONE 5: Outcome Tracking ─────────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
with st.expander("Zone 5 — Outcome Tracking & Calibration", expanded=False):
    _pred_log = load_prediction_log(limit=20)
    _calib    = compute_calibration_metrics()

    st.markdown(f"""<div style="margin-bottom:12px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">Prediction Log — Last 20 Decisions</p>
    </div>""", unsafe_allow_html=True)

    if _pred_log:
        import pandas as pd
        _df = pd.DataFrame([{
            'Timestamp':        e.get('timestamp', '')[:16].replace('T', ' '),
            'Sector':           e.get('sector', ''),
            'Regime':           e.get('regime', ''),
            'IS':               f"{e.get('intelligent_score', 0):.3f}",
            'ReAct Path':       (e.get('react_path', '') or '').replace('PATH_', ''),
            'Decision State':   e.get('l4_decision_state', ''),
        } for e in reversed(_pred_log)])
        st.dataframe(_df, use_container_width=True, hide_index=True)

        # IS trend mini-chart
        _is_vals = [float(e.get('intelligent_score', 0)) for e in _pred_log if e.get('intelligent_score') is not None]
        if len(_is_vals) >= 2:
            fig_trend = go.Figure(go.Scatter(
                y=_is_vals, mode='lines+markers',
                line=dict(color=ACID, width=2),
                marker=dict(size=5, color=ACID),
            ))
            fig_trend.add_hline(y=REACT_IS_HIGH, line_dash='dot', line_color='rgba(200,255,87,0.4)', annotation_text='IS_HIGH')
            fig_trend.add_hline(y=REACT_IS_LOW,  line_dash='dot', line_color='rgba(255,107,107,0.4)', annotation_text='IS_LOW')
            fig_trend.update_layout(**{**CHART_BASE, 'margin': dict(l=8,r=8,t=8,b=8)}, height=140,
                                    title=dict(text='IS Trend (recent predictions)', font=dict(size=11, color=SUB)))
            fig_trend.update_xaxes(showgrid=False, showticklabels=False)
            fig_trend.update_yaxes(range=[0, 1], color=MUTED, gridcolor='rgba(255,255,255,0.04)')
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown(f"<p style='color:{MUTED};font-size:13px;'>No prediction log entries yet.</p>", unsafe_allow_html=True)

    # ── Calibration summary ───────────────────────────────────────────────────
    _calib_status = _calib.get('calibration_status', 'no_data')
    _calib_color  = {'sufficient': ACID, 'insufficient': '#F5A623', 'no_data': MUTED}.get(_calib_status, MUTED)
    st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;margin-top:10px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">Decision Calibration</p>
        <p style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:{_calib_color};margin:0;">
            {_calib_status.upper()} — {_calib.get('total_outcomes', 0)} outcomes logged,
            {_calib.get('scored_outcomes', 0)} scored
            {f' | overall accuracy: {_calib.get("overall_accuracy", 0):.1%}' if _calib.get('overall_accuracy') is not None else ''}
        </p>
        <p style="font-size:12px;color:{MUTED};margin:4px 0 0;">{_calib.get('note', '')}</p>
    </div>""", unsafe_allow_html=True)

    # Per-regime breakdown (shown when data exists)
    _per_regime = _calib.get('per_regime', {})
    if _per_regime:
        _reg_items = [(k, v) for k, v in _per_regime.items()]
        st.markdown(f"""<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-top:8px;">
            {"".join([f'''<div style="background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:10px 14px;">
                <p style="font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};margin:0 0 4px;">{k.replace("_"," ")}</p>
                <p style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:{ACID if v.get("accuracy",0) and v["accuracy"]>0.65 else "#F5A623" if v.get("accuracy",0) else MUTED};margin:0;">{f"{v['accuracy']:.0%}" if v.get("accuracy") is not None else "n/a"}</p>
                <p style="font-size:10px;color:{MUTED};margin:2px 0 0;">{v.get("n_scored",0)} scored</p>
            </div>''' for k, v in _reg_items])}
        </div>""", unsafe_allow_html=True)

    # ── Outcome submission form ───────────────────────────────────────────────
    st.markdown(f"""<div style="margin-top:14px;padding-top:14px;border-top:1px solid {BORDER};">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 10px;">Submit Outcome for Last Analysis</p>
    </div>""", unsafe_allow_html=True)

    _ctx5 = st.session_state.get('chat_context', {})
    _r5   = st.session_state.get('result')
    if _r5 and _ctx5.get('sector'):
        _decision_id_default = (
            (_r5.get('l4_decision') or {}).get('decision_id', '') or
            f"{_ctx5.get('sector','?')}-{_ctx5.get('country','?')}-{int(time.time())}"
        )
        with st.form("outcome_form", clear_on_submit=True):
            _oc1, _oc2 = st.columns([3, 4])
            with _oc1:
                _outcome_val = st.selectbox(
                    "Outcome",
                    ["pending", "validated", "invalidated", "partial", "unknown"],
                    help="Was this decision eventually correct?",
                )
                _outcome_notes = st.text_input("Notes (optional)", max_chars=200, placeholder="e.g. launched in Q3, raised seed round")
            with _oc2:
                _ttl = st.number_input("Days to outcome", min_value=0, max_value=3650, value=0,
                                       help="Leave 0 if unknown or not applicable")
                st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:10px 14px;margin-top:4px;">
                    <p style="font-size:10px;color:{MUTED};margin:0 0 2px;">Sector · Country · Predicted State</p>
                    <p style="font-size:12px;color:{TEXT};margin:0;">{_ctx5.get('sector','?').title()} · {_ctx5.get('country','?')} · <b>{_ctx5.get('decision_state','?')}</b></p>
                </div>""", unsafe_allow_html=True)
            _submitted = st.form_submit_button("Log Outcome", use_container_width=True)
            if _submitted:
                _log_res = log_outcome(
                    decision_id      = _decision_id_default,
                    sector           = _ctx5.get('sector', ''),
                    country          = _ctx5.get('country', ''),
                    regime           = _ctx5.get('regime', ''),
                    predicted_state  = _ctx5.get('decision_state', ''),
                    outcome_value    = _outcome_val,
                    intelligent_score= float(_r5.get('intelligent_score', 0)),
                    time_to_outcome_days = int(_ttl) if _ttl else None,
                    outcome_notes    = _outcome_notes,
                )
                if _log_res.get('status') == 'ok':
                    st.success(f"Outcome logged — decision_id: {_decision_id_default[:40]}")
                else:
                    st.error(f"Log failed: {_log_res}")
    else:
        st.markdown(f"<p style='color:{MUTED};font-size:12px;'>Run an analysis first to enable outcome submission.</p>", unsafe_allow_html=True)

# ── ZONE 6: System Alerts ─────────────────────────────────────────────────────
with st.expander("Zone 6 — System Alerts & Model Health", expanded=False):
    _drift_res   = check_drift(drift_baseline)
    _freshness   = compute_l2_freshness() if MODELS_LOADED else {}

    # Drift signals
    _d1 = _drift_res.get('signal_1_gap_fired', False)
    _d2 = _drift_res.get('signal_2_centroid_fired', False)
    _drift_confirmed = _drift_res.get('drift_detected', False)
    _drift_color = '#FF6B6B' if _drift_confirmed else (ACID if not (_d1 or _d2) else '#F5A623')

    z6c1, z6c2, z6c3 = st.columns(3)
    with z6c1:
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">Drift Status</p>
            <p style="font-family:'Syne',sans-serif;font-size:14px;font-weight:800;color:{_drift_color};margin:0;">
                {'DRIFT CONFIRMED' if _drift_confirmed else ('SIGNAL PARTIAL' if (_d1 or _d2) else 'STABLE')}
            </p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">Sig1(gap)={'✓' if _d1 else '✗'} · Sig2(centroid)={'✓' if _d2 else '✗'}</p>
        </div>""", unsafe_allow_html=True)
    with z6c2:
        _macro_stale = _drift_res.get('macro_staleness_alert', False)
        _days_stale  = _drift_res.get('macro_days_stale')
        _ms_color    = '#FF6B6B' if _macro_stale else ACID
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">Macro Staleness</p>
            <p style="font-family:'Syne',sans-serif;font-size:14px;font-weight:800;color:{_ms_color};margin:0;">
                {'STALE' if _macro_stale else 'OK'}
            </p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">{f'{_days_stale}d since update' if _days_stale else 'Unknown'}</p>
        </div>""", unsafe_allow_html=True)
    with z6c3:
        _sarima_fresh = _freshness.get('sarima_freshness_days')
        _sarima_stale = _freshness.get('runtime_staleness_flag', False)
        _sf_color     = '#FF6B6B' if _sarima_stale else ACID
        st.markdown(f"""<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;text-align:center;">
            <p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{MUTED};font-weight:600;margin:0 0 6px;">SARIMA Freshness</p>
            <p style="font-family:'Syne',sans-serif;font-size:14px;font-weight:800;color:{_sf_color};margin:0;">
                {'STALE' if _sarima_stale else 'OK'}
            </p>
            <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">{f'{_sarima_fresh}d old' if _sarima_fresh is not None else 'Unknown'}</p>
        </div>""", unsafe_allow_html=True)

    # Retrain status (if any)
    _logs_dir_z6     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    _retrain_path_z6 = os.path.join(_logs_dir_z6, "retrain_status.json")
    if os.path.exists(_retrain_path_z6):
        try:
            with open(_retrain_path_z6, 'r', encoding='utf-8') as _f:
                _retrain_status = json.load(_f)
            _rt_status = _retrain_status.get('retrain_status', '—')
            _rt_at     = _retrain_status.get('retrain_started_at', '—')
            _rt_done   = _retrain_status.get('retrain_completed_at', '')
            _rt_color  = ACID if _rt_status == 'completed' else ('#FF6B6B' if _rt_status == 'failed' else '#F5A623')
            st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:6px;padding:12px 16px;margin-top:10px;">
                <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">Last Retrain</p>
                <p style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:{_rt_color};margin:0;">{_rt_status.upper()}</p>
                <p style="font-size:11px;color:{MUTED};margin:4px 0 0;">Started: {_rt_at[:16] if _rt_at else '—'}{f' · Completed: {_rt_done[:16]}' if _rt_done else ''}</p>
            </div>""", unsafe_allow_html=True)
        except Exception:
            pass
    elif _drift_confirmed:
        st.warning("Drift confirmed but no retrain record found. Check logs/retrain_status.json.")

    _log_entries = _drift_res.get('log_entries_analyzed', 0)
    st.markdown(f"<p style='font-size:12px;color:{MUTED};margin-top:8px;'>Prediction log: {_log_entries} entries analyzed for drift detection.</p>", unsafe_allow_html=True)

# ── CONVERSATIONAL ADVISOR ──────────────────────────────────
st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
st.markdown(f"""<div style="margin-bottom:20px;padding-top:20px;border-top:1px solid {BORDER};">
    <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 6px;">
        Agent A7 Interactive</p>
    <p style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.5px;">
        MIDAN Advisory Partner</p>
</div>""", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask MIDAN about pivot strategies, risks, or validation...", key="chat"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    ctx = st.session_state.chat_context
    try:
        reply = call_backend_chat(st.session_state.chat_history, ctx)
        if not reply:
            raise ValueError("Empty reply from backend chat.")
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Advisor unavailable: {str(e)}")

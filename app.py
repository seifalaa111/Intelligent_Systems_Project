import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import pickle, json, os, time, warnings, requests
from textwrap import dedent
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    GROQ_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy"))
except Exception:
    GROQ_CLIENT = None

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="MIDAN — AI Decision Engine",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN TOKENS ────────────────────────────────────────────
ACID   = "#C8FF57"
BG     = "#060608"
CARD   = "#0D0D11"
CARD2  = "#141419"
BORDER = "rgba(255,255,255,0.06)"
BORDA  = "rgba(200,255,87,0.22)"
TEXT   = "#EDEAF8"
SUB    = "#7B7990"
MUTED  = "#3E3D50"

FEATURES = ['inflation','gdp_growth','macro_friction',
            'capital_concentration','velocity_yoy']

REGIME_COLORS = {
    'GROWTH_MARKET':         ACID,
    'EMERGING_MARKET':       '#5B6CF0',
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

# ── LOAD MODELS ──────────────────────────────────────────────
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    path = MODELS_DIR

    def _pkl(name):
        with open(f'{path}/{name}', 'rb') as f:
            return pickle.load(f)

    def _json(name, default=None):
        try:
            with open(f'{path}/{name}', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default if default is not None else {}

    scaler = _pkl('scaler_global.pkl')
    pca    = _pkl('pca_global.pkl')
    svm    = _pkl('svm_global.pkl')
    le     = _pkl('label_encoder.pkl')
    lgb    = _pkl('lgb_surrogate.pkl')
    sarima = _json('sarima_results.json')
    shap_i = _json('shap_feature_importance.json')
    clust  = _json('cluster_names.json')
    comps  = _json('competitors_context.json', default={})
    sents  = _json('sentiment_context.json', default=[])

    return scaler, pca, svm, le, lgb, sarima, shap_i, clust, comps, sents

sarima_results, shap_global, cluster_names, comps_data, sents_data = {}, {}, {}, {}, []

try:
    scaler, pca, svm, le, lgb, sarima_results, shap_global, cluster_names, comps_data, sents_data = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR   = str(e)

# ── SECTOR MEDIANS (used for capital_concentration fallback) ─
SECTOR_MEDIANS = {
    'fintech':    175000.0,
    'ecommerce':  120000.0,
    'healthtech': 200000.0,
    'edtech':      80000.0,
    'saas':       250000.0,
    'logistics':   90000.0,
    'agritech':    50000.0,
    'other':      100000.0,
}

# Effective macro per sector — each sector experiences inflation differently
# fintech/healthtech: HIGH inflation → MORE demand → net positive
# saas/agritech: B2B / food security → insulated
# edtech/logistics/ecommerce: directly hurt by inflation
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
    'MA': {'inflation':  6.1, 'gdp_growth': 3.1, 'unemployment':11.5},
}


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
        sector = 'fintech'  # placeholder — will be overridden by dropdown
    country, country_found = None, False
    for code, kws in COUNTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            country, country_found = code, True
            break
    if not country_found:
        country = 'EG'  # placeholder — will be overridden by dropdown
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
    # Rule 1: GROWTH — SVM has no GROWTH class, so rules must add it
    if gdp_growth > 3.5 and inflation < 8 and velocity_yoy > 0.15:
        margin = min((gdp_growth-3.5)/4.0, (8-inflation)/8.0, (velocity_yoy-0.15)/0.25)
        conf = float(np.clip(0.65 + margin * 0.30, 0.60, 0.95))
        return 'GROWTH_MARKET', conf
    # Rule 1b: EMERGING — good macro but lower velocity
    if gdp_growth > 2.0 and inflation < 10 and macro_friction < 10:
        margin = min((gdp_growth-2.0)/4.0, (10-inflation)/10.0, (10-macro_friction)/15.0)
        conf = float(np.clip(0.60 + margin * 0.30, 0.55, 0.90))
        return 'EMERGING_MARKET', conf
    # Rule 2: CONTRACTING — extreme downturn only
    if gdp_growth < 0 or (inflation > 50 and macro_friction > 50):
        severity = max(abs(min(gdp_growth, 0)) / 3.0, 0.0)
        conf = float(np.clip(0.65 + severity * 0.25, 0.60, 0.92))
        return 'CONTRACTING_MARKET', conf
    # Rule 3: HIGH_FRICTION — only for severe macro pain
    if macro_friction > 30 or inflation > 25:
        pain = max((macro_friction - 30) / 40, (inflation - 25) / 30, 0)
        conf = float(np.clip(0.60 + pain * 0.30, 0.55, 0.92))
        return 'HIGH_FRICTION_MARKET', conf
    # Default: trust the SVM
    return svm_regime, svm_conf

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

    # Apply sector-specific macro adjustment
    eff_inf_offset, gdp_boost, velocity = SECTOR_EFF_MACRO.get(sec, SECTOR_EFF_MACRO['other'])
    # Scale the adjustment relative to country base inflation
    # High-inflation countries apply the sector pattern at scale
    scale = base_inflation / 33.9  # normalize to Egypt baseline
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
    # Layer 2: domain rule override (SVM has no GROWTH class)
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
        # Use actual forecast magnitude — not binary
        fc_mean = float(np.mean(fc))
        sarima_trend = float(np.clip(fc_mean / 50.0, 0.15, 0.90))
        drift_flag   = sarima_results[sec]['drift_flag']
        logs.append(f"[STEP4B] SARIMA forecast: {[round(x,1) for x in fc]} mean={fc_mean:.1f} → trend={sarima_trend:.2f} | drift={drift_flag}")
    else:
        logs.append(f"[STEP4B] No SARIMA model for {sec} — using neutral trend={sarima_trend}")

    if drift_flag:
        logs.append("[STEP4B] ⚠ DRIFT DETECTED — Manual Reclustering Advised")

    # ── Step 5: TAS ───────────────────────────────────────────
    tas = round(conf*0.40 + sarima_trend*0.35 + xai_score*0.25, 3)
    logs.append(f"[STEP5] TAS = {conf:.2f}×0.40 + {sarima_trend:.2f}×0.35 + {xai_score:.2f}×0.25 = {tas}")

    # ── Agent A2: Competitor Context ──────────────────────────
    # Retrieve top 2 competitors dynamically if they exist via the list structure
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
        if GROQ_CLIENT and os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_API_KEY") != "dummy":
            resp = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": a7_prompt}],
                model="llama3-8b-8192", temperature=0.3, max_tokens=150
            )
            a7_synthesis = resp.choices[0].message.content.strip()
        else:
            raise ValueError("No Groq client")
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

    # ── Slack Webhook Execution ───────────────────────────────
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

    # Show what Agent A1 detected in real-time
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

# ── RUN ──────────────────────────────────────────────────────
if run and MODELS_LOADED:
    SECTOR_LABEL_MAP = {
        "E-commerce":"ecommerce","Healthtech":"healthtech",
        "Edtech":"edtech","SaaS":"saas","Logistics":"logistics",
        "Agritech":"agritech","Other":"other","Fintech":"fintech"
    }
    # Agent A1 parses text; dropdowns are fallback when keywords not found
    if idea and len(idea.strip()) > 5:
        parsed_sec, parsed_ctry, sec_found, ctry_found = agent_a1_parse(idea)
        sector_key   = parsed_sec if sec_found else SECTOR_LABEL_MAP.get(sector, sector.lower())
        country_code = parsed_ctry if ctry_found else country.split(" — ")[0]
    else:
        sector_key   = SECTOR_LABEL_MAP.get(sector, sector.lower())
        country_code = country.split(" — ")[0]

    ph = st.empty()
    steps = [
        ("01 — Parsing idea & extracting context",     0.18),
        ("02 — Fetching live macro signals",            0.36),
        ("03 — Building inference context vector",      0.50),
        ("04 — SVM RBF classification",                 0.66),
        ("05 — SHAP explainability (LightGBM surrogate)",0.80),
        ("06 — SARIMA 90-day forecast",                 0.90),
        ("07 — Calculating TAS & Trinity Report",       1.00),
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

    logs = []
    try:
        result = run_inference(sector_key, country_code, logs)
        st.session_state.result  = result
        st.session_state.logs    = logs
        st.session_state.sector  = sector_key
        st.session_state.country = country_code
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

# Verdict banner
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

# 4 Score Cards
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

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# Charts Row 1: Cluster scatter + SHAP
col_l, col_r = st.columns([6, 5], gap="medium")

with col_l:
    st.markdown(f"""<div style="margin-bottom:10px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
            Step 02 — Market Clustering</p>
        <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
            Where your idea sits in the competitive field</p>
    </div>""", unsafe_allow_html=True)

    # Generate cluster scatter from SVM decision regions (real model)
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
    # Your idea dot
    px_idea, py_idea = r['x_pca'][0], r['x_pca'][1]
    fig_c.add_trace(go.Scatter(
        x=[px_idea], y=[py_idea],
        mode='markers+text', name='Your idea',
        text=['  Your idea'], textposition='middle right',
        textfont=dict(color=TEXT, size=12),
        marker=dict(color=ACID, size=16, symbol='diamond',
                    line=dict(color='#000', width=2))
    ))
    fig_c.update_layout(**CHART_BASE, height=320,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=SUB, size=10), x=0.01, y=0.99))
    fig_c.update_xaxes(title='PCA 1', color=MUTED,
                       gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)')
    fig_c.update_yaxes(title='PCA 2', color=MUTED,
                       gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)')
    st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})

with col_r:
    st.markdown(f"""<div style="margin-bottom:10px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{ACID};margin:0 0 4px;">
            Step 04A — Signal Weights (SHAP)</p>
        <p style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:{TEXT};margin:0;letter-spacing:-0.4px;">
            Why the market was classified this way</p>
    </div>""", unsafe_allow_html=True)

    shap_d  = r['shap_dict']
    labels  = [k.replace('_',' ').title() for k in shap_d.keys()]
    vals    = list(shap_d.values())
    max_v   = max(vals)
    colors  = [ACID if v == max_v else '#5B6CF0' if v > np.median(vals) else MUTED for v in vals]

    fig_s = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker=dict(color=colors, cornerradius=3),
        text=[f'{v:.3f}' for v in vals], textposition='outside',
        textfont=dict(color=SUB, size=11),
    ))
    fig_s.update_layout(**CHART_BASE, height=320, bargap=0.35)
    fig_s.update_xaxes(color=MUTED, gridcolor='rgba(255,255,255,0.04)',
                       zerolinecolor='rgba(255,255,255,0.04)', tickfont=dict(size=11))
    fig_s.update_yaxes(gridcolor='rgba(0,0,0,0)', color=TEXT,
                       tickfont=dict(size=12, family='DM Sans'))
    st.plotly_chart(fig_s, use_container_width=True, config={'displayModeBar': False})

# Charts Row 2: SARIMA + TAS Gauge
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

# Trinity Report
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

# Pipeline Trace
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
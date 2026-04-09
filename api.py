"""
MIDAN AI Decision Engine — FastAPI Backend
Standalone API that loads models directly (no Streamlit dependency).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
MODELS_DIR = "models"

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
    scaler = _pkl('scaler_global.pkl')
    pca    = _pkl('pca_global.pkl')
    svm    = _pkl('svm_global.pkl')
    le     = _pkl('label_encoder.pkl')
    lgb    = _pkl('lgb_surrogate.pkl')
    sarima_results = _json('sarima_results.json')
    comps_data     = _json('competitors_context.json', {})
    sents_data     = _json('sentiment_context.json', [])
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR = str(e)

FEATURES = ['inflation','gdp_growth','macro_friction','capital_concentration','velocity_yoy']

SECTOR_LABEL_MAP = {
    "E-commerce":"ecommerce","Healthtech":"healthtech",
    "Edtech":"edtech","SaaS":"saas","Logistics":"logistics",
    "Agritech":"agritech","Other":"other","Fintech":"fintech"
}

SECTOR_MEDIANS = {
    'fintech': 175000.0, 'ecommerce': 120000.0, 'healthtech': 200000.0,
    'edtech': 80000.0, 'saas': 250000.0, 'logistics': 90000.0,
    'agritech': 50000.0, 'other': 100000.0,
}

SECTOR_EFF_MACRO = {
    'fintech': (7.5, +1.5, 0.28), 'healthtech': (7.0, +2.0, 0.22),
    'saas': (4.0, +2.2, 0.10), 'agritech': (4.5, +0.7, 0.12),
    'edtech': (40.0, -1.0, 0.07), 'logistics': (42.0, -1.8, 0.09),
    'ecommerce': (36.0, -1.3, 0.13), 'other': (33.9, 0.0, 0.10),
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

SECTOR_KEYWORDS = {
    'fintech': ['finance','payment','fintech','bank','loan','lending','invoice','insurance','wallet','money'],
    'ecommerce': ['ecommerce','e-commerce','shop','store','retail','marketplace','delivery','commerce'],
    'healthtech': ['health','medical','doctor','clinic','hospital','pharma','biotech','mental'],
    'edtech': ['education','learning','school','university','course','tutor','edtech','training'],
    'saas': ['saas','software','platform','dashboard','tool','api','enterprise','cloud','b2b','crm'],
    'logistics': ['logistics','shipping','supply chain','warehouse','transport','fleet','trucking'],
    'agritech': ['agri','farm','crop','harvest','food','agriculture','irrigation'],
}
COUNTRY_KEYWORDS = {
    'EG': ['egypt','cairo','egyptian'], 'SA': ['saudi','ksa','riyadh','jeddah'],
    'AE': ['uae','dubai','abu dhabi','emirates'], 'MA': ['morocco','moroccan','casablanca'],
    'NG': ['nigeria','nigerian','lagos'], 'KE': ['kenya','kenyan','nairobi'],
    'US': ['usa','united states','america'], 'GB': ['uk','britain','london','england'],
}

def agent_a1_parse(idea_text):
    t = idea_text.lower()
    sector, sector_found = None, False
    for sec, kws in SECTOR_KEYWORDS.items():
        if any(k in t for k in kws):
            sector, sector_found = sec, True
            break
    if not sector_found: sector = 'fintech'
    country, country_found = None, False
    for code, kws in COUNTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            country, country_found = code, True
            break
    if not country_found: country = 'EG'
    return sector, country, sector_found, country_found

def enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth, macro_friction, velocity_yoy):
    if gdp_growth > 3.5 and inflation < 8 and velocity_yoy > 0.15:
        margin = min((gdp_growth-3.5)/4.0, (8-inflation)/8.0, (velocity_yoy-0.15)/0.25)
        return 'GROWTH_MARKET', float(np.clip(0.65 + margin * 0.30, 0.60, 0.95))
    if gdp_growth > 2.0 and inflation < 10 and macro_friction < 10:
        margin = min((gdp_growth-2.0)/4.0, (10-inflation)/10.0, (10-macro_friction)/15.0)
        return 'EMERGING_MARKET', float(np.clip(0.60 + margin * 0.30, 0.55, 0.90))
    if gdp_growth < 0 or (inflation > 50 and macro_friction > 50):
        severity = max(abs(min(gdp_growth, 0)) / 3.0, 0.0)
        return 'CONTRACTING_MARKET', float(np.clip(0.65 + severity * 0.25, 0.60, 0.92))
    if macro_friction > 30 or inflation > 25:
        pain = max((macro_friction - 30) / 40, (inflation - 25) / 30, 0)
        return 'HIGH_FRICTION_MARKET', float(np.clip(0.60 + pain * 0.30, 0.55, 0.92))
    return svm_regime, svm_conf

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

def run_inference(sector, country):
    sec = sector.lower()
    macro = COUNTRY_MACRO_DEFAULTS.get(country.upper(),
            {'inflation':10.0,'gdp_growth':3.0,'unemployment':7.0})

    base_inflation = macro['inflation']
    base_gdp = macro['gdp_growth']
    unemployment = macro['unemployment']

    eff_inf_offset, gdp_boost, velocity = SECTOR_EFF_MACRO.get(sec, SECTOR_EFF_MACRO['other'])
    scale = base_inflation / 33.9
    inflation = float(np.clip(eff_inf_offset * scale, 1.0, 100.0))
    gdp_growth = float(base_gdp + gdp_boost)
    macro_fric = float(np.clip(inflation + unemployment - gdp_growth, -50, 100))
    cap_conc = SECTOR_MEDIANS.get(sec, SECTOR_MEDIANS['other'])

    x_raw = np.array([[inflation, gdp_growth, macro_fric, float(cap_conc), velocity]])
    x_scaled = scaler.transform(x_raw)
    x_pca = pca.transform(x_scaled)

    pred_enc = svm.predict(x_scaled)[0]
    proba = svm.predict_proba(x_scaled)[0]
    svm_regime = le.inverse_transform([pred_enc])[0]
    svm_conf = float(proba.max())
    regime, conf = enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth, macro_fric, velocity)

    shap_dict = compute_shap(lgb, x_scaled)
    xai_score = float(conf * np.mean(list(shap_dict.values())))

    sarima_trend = 0.50
    drift_flag = False
    if sec in sarima_results:
        fc = [max(0, v) for v in sarima_results[sec]['forecast_mean']]
        fc_mean = float(np.mean(fc))
        sarima_trend = float(np.clip(fc_mean / 50.0, 0.15, 0.90))
        drift_flag = sarima_results[sec]['drift_flag']

    tas = round(conf*0.40 + sarima_trend*0.35 + xai_score*0.25, 3)

    a2_comps = ["Traditional incumbents", "Local SMEs"]
    sector_comps_list = comps_data.get(sec, [])
    if isinstance(sector_comps_list, list) and len(sector_comps_list) > 0:
        a2_comps = [c.get("Company", "Competitor") for c in sector_comps_list[:2]]

    a4_sent_ratio = "Neutral"
    if sents_data:
        pos = sum(1 for s in sents_data if s.get('sentiment') == 'positive')
        neg = sum(1 for s in sents_data if s.get('sentiment') == 'negative')
        if pos > neg * 1.5: a4_sent_ratio = "Positive"
        elif neg > pos * 1.5: a4_sent_ratio = "Negative"

    regime_readable = regime.replace('_', ' ').title()
    top3 = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_names = [f[0].replace('_',' ') for f in top3]

    a7_prompt = dedent(f"""
        You are MIDAN Agent A7, a Chief Intelligence Officer at a VC firm.
        Synthesize this startup market intelligence in exactly 3 short sentences.
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
    except Exception:
        comp_str = f"facing off against {', '.join(a2_comps[:2])}" if len(a2_comps)>1 else "in a fragmented landscape"
        move_str = "Double down on customer acquisition immediately." if tas >= 0.7 else "Run extreme demand validation before writing a single line of code."
        a7_synthesis = (f"The {sec.title()} space in {country} is currently operating as a {regime_readable}, heavily influenced by {top3_names[0]}. "
                        f"You will be {comp_str} operating amid a {a4_sent_ratio.lower()} macro sentiment. "
                        f"{move_str}")

    finding = f"Market classified as {regime_readable} with {conf:.0%} confidence. Top signals: {', '.join(top3_names)}."
    implication = a7_synthesis
    action = f"{'Move within the next 90 days. Apply to Flat6Labs or Cairo Angels.' if tas>=0.70 else 'Validate demand before building. Run 20 direct customer interviews.'} Key signal to monitor: {top3_names[0]}."

    action_fired = tas >= 0.70 and regime in ('GROWTH_MARKET','EMERGING_MARKET')
    if action_fired:
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if webhook_url and webhook_url.startswith("http"):
            msg = {"text": f"🚀 *MIDAN INTELLIGENCE:* High-Conviction Market Detected!\n*Sector:* {sec.title()} ({country})\n*TAS:* {tas}\n*Regime:* {regime_readable}"}
            try: requests.post(webhook_url, json=msg, timeout=2)
            except Exception: pass

    return {
        'regime': regime, 'confidence': conf, 'tas': tas,
        'sarima_trend': sarima_trend, 'xai_score': xai_score,
        'shap_dict': shap_dict, 'drift_flag': drift_flag,
        'action_fired': action_fired,
        'finding': finding, 'implication': implication, 'action': action,
        'x_pca': x_pca[0],
    }

# ── FASTAPI ──────────────────────────────────────────────────
api = FastAPI(title="MIDAN AI Decision Engine API", version="1.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class IdeaRequest(BaseModel):
    idea: str
    sector: str = "Fintech"
    country: str = "EG — Egypt"

@api.post("/analyze")
async def analyze_idea(req: IdeaRequest):
    if not MODELS_LOADED:
        raise HTTPException(status_code=500, detail="Models failed to load.")

    if req.idea and len(req.idea.strip()) > 5:
        parsed_sec, parsed_ctry, sec_found, ctry_found = agent_a1_parse(req.idea)
        sector_key = parsed_sec if sec_found else SECTOR_LABEL_MAP.get(req.sector, "fintech")
        country_code = parsed_ctry if ctry_found else req.country.split(" — ")[0]
    else:
        sector_key = SECTOR_LABEL_MAP.get(req.sector, "fintech")
        country_code = req.country.split(" — ")[0]

    try:
        report = run_inference(sector_key, country_code)
        return {
            "success": True,
            "sector": sector_key,
            "country": country_code,
            "regime": report["regime"],
            "tas_score": int(report["tas"] * 100),
            "confidence": int(report["confidence"] * 100),
            "sarima_trend": report["sarima_trend"],
            "drift_flag": report["drift_flag"],
            "action_fired": report["action_fired"],
            "report": {
                "finding": report["finding"],
                "implication": report["implication"],
                "action": report["action"],
            },
            "shap_weights": {k: float(v) for k, v in report["shap_dict"].items()},
            "pca_coords": report["x_pca"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/health")
async def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)

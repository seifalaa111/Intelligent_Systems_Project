"""
MIDAN AI Decision Engine — FastAPI Backend
Standalone API that loads models directly (no Streamlit dependency).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
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

# ── AGENT A0 — Idea Evaluation ──────────────────────────────
IDEA_DIMENSIONS = ['problem_clarity', 'solution_fit', 'differentiation', 'business_model', 'scalability']

def agent_a0_evaluate_idea(idea_text, sector, country):
    """
    Agent A0: Evaluates the startup idea itself (not just market conditions).
    Uses LLM when available, falls back to keyword heuristics.
    Returns dict with 5 dimension scores (0-10) and overall idea_score (0-100).
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if GROQ_CLIENT and groq_key and groq_key != "dummy":
        try:
            prompt = dedent(f"""
                You are a VC analyst evaluating a startup idea. Score each dimension 0-10.

                Idea: "{idea_text}"
                Sector: {sector} | Country: {country}

                CRITICAL INSTRUCTION: If the idea text is just a simple greeting (e.g. 'hello'), conversational chatter, unstructured gibberish, or fundamentally NOT a startup business idea, you MUST score every dimension exactly 0 and provide the reason 'Not a valid startup idea.' Do not hallucinate scores for non-ideas.

                Score these 5 dimensions and provide a one-sentence justification for each:
                1. problem_clarity — Is there a clear, specific problem being solved?
                2. solution_fit — Does the proposed solution address the problem?
                3. differentiation — Is it meaningfully different from existing solutions?
                4. business_model — Is there an obvious way to make money?
                5. scalability — Can it grow beyond the initial market?

                Respond in EXACTLY this JSON format, no other text:
                {{"problem_clarity": {{"score": 7, "reason": "..."}}, "solution_fit": {{"score": 6, "reason": "..."}}, "differentiation": {{"score": 5, "reason": "..."}}, "business_model": {{"score": 8, "reason": "..."}}, "scalability": {{"score": 6, "reason": "..."}}}}
            """).strip()
            resp = GROQ_CLIENT.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant", temperature=0.2, max_tokens=400
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown code blocks)
            if '```' in raw:
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
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

    # Fallback: keyword-based heuristic scoring
    t = idea_text.lower()
    scores = {}
    reasons = {}

    # Problem clarity: does the text describe a problem?
    problem_words = ['problem', 'issue', 'challenge', 'pain', 'struggle', 'need', 'lack', 'gap', 'inefficient', 'expensive', 'slow', 'difficult']
    problem_hits = sum(1 for w in problem_words if w in t)
    scores['problem_clarity'] = min(10, 3 + problem_hits * 2)
    reasons['problem_clarity'] = f"{'Clear problem statement detected' if problem_hits >= 2 else 'Problem could be more specific'}"

    # Solution fit: does it describe a solution approach?
    solution_words = ['app', 'platform', 'tool', 'system', 'service', 'automate', 'connect', 'provide', 'enable', 'simplify', 'streamline', 'reduce']
    sol_hits = sum(1 for w in solution_words if w in t)
    scores['solution_fit'] = min(10, 3 + sol_hits * 2)
    reasons['solution_fit'] = f"{'Solution approach is clear' if sol_hits >= 2 else 'Describe how your solution works'}"

    # Differentiation: unique angle?
    diff_words = ['first', 'only', 'unique', 'unlike', 'better', 'faster', 'cheaper', 'new', 'innovative', 'ai', 'machine learning', 'blockchain']
    diff_hits = sum(1 for w in diff_words if w in t)
    scores['differentiation'] = min(10, 2 + diff_hits * 2)
    reasons['differentiation'] = f"{'Unique angle detected' if diff_hits >= 2 else 'What makes this different from existing solutions?'}"

    # Business model: revenue signals?
    biz_words = ['subscription', 'saas', 'commission', 'fee', 'pricing', 'revenue', 'monetize', 'b2b', 'b2c', 'freemium', 'marketplace', 'premium']
    biz_hits = sum(1 for w in biz_words if w in t)
    scores['business_model'] = min(10, 3 + biz_hits * 2)
    reasons['business_model'] = f"{'Revenue model indicated' if biz_hits >= 1 else 'How will this make money?'}"

    # Scalability: growth potential?
    scale_words = ['scale', 'global', 'expand', 'growth', 'million', 'region', 'international', 'multiple', 'market', 'nationwide']
    scale_hits = sum(1 for w in scale_words if w in t)
    scores['scalability'] = min(10, 3 + scale_hits * 2)
    reasons['scalability'] = f"{'Growth potential indicated' if scale_hits >= 1 else 'How will this scale beyond initial market?'}"

    # Bonus for longer, more detailed descriptions
    word_count = len(t.split())
    if word_count > 30:
        for dim in scores:
            scores[dim] = min(10, scores[dim] + 1)

    idea_score = int(sum(scores.values()) / len(scores) * 10)
    return {'scores': scores, 'reasons': reasons, 'idea_score': idea_score}

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
                model="llama-3.1-8b-instant", temperature=0.3, max_tokens=150
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

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    context: Dict[str, Any]
    messages: List[ChatMessage]

def process_idea(idea_text: str, default_sector: str = "fintech", default_country: str = "EG"):
    parsed_sec, parsed_ctry, sec_found, ctry_found = agent_a1_parse(idea_text)
    sector_key = parsed_sec if sec_found else default_sector
    country_code = parsed_ctry if ctry_found else default_country

    report = run_inference(sector_key, country_code)
    idea_eval = agent_a0_evaluate_idea(idea_text, sector_key, country_code)

    tas_normalized = report["tas"]
    idea_normalized = idea_eval["idea_score"] / 100.0
    svs = int((tas_normalized * 0.50 + idea_normalized * 0.50) * 100)

    # Adjusted Thresholds
    high_market = report["tas"] >= 0.68
    high_idea = idea_eval["idea_score"] >= 70

    if high_market and high_idea:
        quadrant = "GO — Launch"
    elif high_market and not high_idea:
        quadrant = "Wrong Idea — Right Market"
    elif not high_market and high_idea:
        quadrant = "Wait or Pivot Market"
    else:
        quadrant = "STOP — Rethink Everything"

    return {
        "success": True, "sector": sector_key, "country": country_code,
        "regime": report["regime"], "tas_score": int(report["tas"] * 100),
        "confidence": int(report["confidence"] * 100), "sarima_trend": report["sarima_trend"],
        "drift_flag": report["drift_flag"], "action_fired": report["action_fired"],
        "report": report, # keeping raw text in 'report.finding' etc.
        "shap_weights": {k: float(v) for k, v in report["shap_dict"].items()},
        "pca_coords": report["x_pca"].tolist(),
        "idea_score": idea_eval["idea_score"], "idea_dimensions": idea_eval["scores"],
        "idea_reasons": idea_eval["reasons"], "svs": svs, "quadrant": quadrant,
    }

@api.post("/analyze")
async def analyze_idea(req: IdeaRequest):
    if not MODELS_LOADED:
        raise HTTPException(status_code=500, detail="Models failed to load.")
    sector_key = SECTOR_LABEL_MAP.get(req.sector, "fintech")
    country_code = req.country.split(" — ")[0]
    
    try:
        if req.idea and len(req.idea.strip()) > 5:
            res = process_idea(req.idea, sector_key, country_code)
            res["report"] = { "finding": res["report"]["finding"], "implication": res["report"]["implication"], "action": res["report"]["action"]}
            return res
        else:
            raise HTTPException(status_code=400, detail="Idea too short")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _chat_fallback(req: ChatRequest):
    """Generate a context-aware fallback reply when no LLM is available."""
    sector = req.context.get("sector", "your sector")
    country = req.context.get("country", "your market")
    regime = req.context.get("regime", "UNKNOWN").replace("_", " ").title()
    tas = req.context.get("tas_score", 0)
    idea = req.context.get("idea", "")

    last_msg = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_msg = m.content.lower()
            break

    # Pattern-matched responses based on common follow-up questions
    if any(w in last_msg for w in ["pivot", "change", "switch", "different idea", "alternative"]):
        return (f"Given the {regime} conditions in {country}, pivoting within {sector} could work if you target "
                f"an underserved niche. Before pivoting, validate that the new direction solves a problem "
                f"people are already paying to fix. Run 10 customer interviews in the new vertical first.")

    if any(w in last_msg for w in ["compet", "rival", "player", "who else", "crowded"]):
        return (f"The {sector} space in {country} has both local startups and international players. "
                f"Your differentiation needs to be crystal clear — either move faster, go deeper into a niche, "
                f"or solve a local pain point that global competitors can't touch. What's your unfair advantage?")

    if any(w in last_msg for w in ["fund", "invest", "money", "raise", "capital", "angel", "vc"]):
        if tas >= 70:
            return (f"With a TAS of {tas}/100 in a {regime}, you're in a strong position to raise. "
                    f"Target regional accelerators (Flat6Labs, 500 Global MENA) or angel networks first. "
                    f"Build a 3-month runway plan and show traction before approaching VCs.")
        else:
            return (f"With a TAS of {tas}/100, fundraising will be harder right now. Focus on bootstrapping "
                    f"or revenue-first models. Build a working prototype and get 10 paying customers — "
                    f"that's more convincing to MENA investors than any pitch deck.")

    if any(w in last_msg for w in ["risk", "danger", "threat", "fail", "worry", "concern"]):
        return (f"Key risks in {sector}/{country}: 1) Regulatory shifts can change overnight in this region. "
                f"2) Customer acquisition cost in {regime} conditions tends to be high. "
                f"3) Currency volatility can eat margins. Mitigate by keeping burn low and validating demand before scaling.")

    if any(w in last_msg for w in ["next", "start", "begin", "first step", "how do i", "what should"]):
        if tas >= 70:
            return (f"Your market signals are strong. Next steps: 1) Build an MVP in 4-6 weeks — not a full product. "
                    f"2) Get 20 target users to test it. 3) Apply to a MENA accelerator this cycle. "
                    f"Speed matters more than perfection in a {regime}.")
        else:
            return (f"The market conditions aren't ideal yet. Next steps: 1) Run 20 customer discovery interviews "
                    f"to validate real demand. 2) Build a landing page and measure interest. "
                    f"3) Don't write code until you have evidence people will pay.")

    if any(w in last_msg for w in ["go-to-market", "gtm", "launch", "marketing", "customer", "acquire", "growth"]):
        return (f"For {sector} in {country}, start hyper-local. Pick one neighborhood, one vertical, one persona. "
                f"Dominate that before expanding. Word-of-mouth and WhatsApp groups beat paid ads in MENA every time. "
                f"What specific customer segment are you targeting first?")

    # Default conversational response
    return (f"I'm MIDAN's offline advisor. Your idea is in the {sector} space targeting {country}, "
            f"currently a {regime} with {tas}/100 opportunity score. "
            f"Ask me about competitive risks, fundraising strategy, go-to-market, pivot options, "
            f"or next steps — I can help with all of these based on your market data.")


@api.post("/chat")
async def chat_interaction(req: ChatRequest):
    groq_key = os.environ.get("GROQ_API_KEY", "")
    use_llm = GROQ_CLIENT and groq_key and groq_key != "dummy"

    if not use_llm:
        reply = _chat_fallback(req)
        return {"success": True, "reply": reply}

    system_prompt = dedent(f"""
        You are the MIDAN AI Startup Advisor, a brutally honest but extremely helpful VC partner.
        You are currently advising a founder based on the following computed market context:
        - Sector: {req.context.get("sector", "Unknown")}
        - Country: {req.context.get("country", "Unknown")}
        - Regime: {req.context.get("regime", "Unknown")} (Score: {req.context.get("tas_score", 0)}/100)
        - AI Original Read: {req.context.get("implication", "")}

        Keep your responses concise, sharp, and consultative. Speak directly to the founder. Ask probing questions if needed. Under 4 sentences.
    """).strip()

    groq_msgs = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        groq_msgs.append({"role": m.role, "content": m.content})

    try:
        resp = GROQ_CLIENT.chat.completions.create(
            messages=groq_msgs,
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=250
        )
        reply = resp.choices[0].message.content.strip()
        return {"success": True, "reply": reply}
    except Exception:
        reply = _chat_fallback(req)
        return {"success": True, "reply": reply}

class InteractRequest(BaseModel):
    context: Dict[str, Any]
    messages: List[ChatMessage]

_IDEA_SIGNALS = [
    # business nouns
    'app', 'platform', 'startup', 'business', 'product', 'service', 'tool',
    'marketplace', 'saas', 'software', 'solution', 'system', 'company',
    # action verbs indicating a concept
    'build', 'create', 'launch', 'develop', 'connect', 'automate', 'help',
    'enable', 'solve', 'simplify', 'replace', 'disrupt', 'scale',
    # market/sector words
    'fintech', 'healthtech', 'edtech', 'ecommerce', 'logistics', 'agritech',
    'payment', 'delivery', 'booking', 'learning', 'insurance', 'clinic',
    'invoice', 'subscription', 'marketplace', 'b2b', 'b2c', 'freelance',
    # location signals (means they are contextualising an idea)
    'egypt', 'cairo', 'dubai', 'saudi', 'nigeria', 'kenya', 'mena',
    # problem framing
    'problem', 'pain', 'issue', 'gap', 'need', 'market', 'customer', 'user',
]

_CONVERSATIONAL_OPENERS = [
    'hi', 'hello', 'hey', 'yo', 'what', 'how', 'who', 'why', 'when', 'where',
    'thanks', 'thank you', 'ok', 'okay', 'sure', 'got it', 'cool', 'great',
    'can you', 'could you', 'tell me', 'explain', 'help me understand',
    'i have an idea', 'i want to', 'i would like', 'i am thinking',
    'what do you think', 'is it good', 'any thoughts', 'give me feedback',
]

def _keyword_is_idea(text: str) -> bool:
    """Heuristic: returns True only when the message looks like an actual startup idea."""
    t = text.lower().strip()
    word_count = len(t.split())

    # Short messages or pure conversational openers → never an idea
    if word_count < 8:
        return False
    if any(t.startswith(opener) for opener in _CONVERSATIONAL_OPENERS):
        return False

    # Must contain at least 2 business/sector/problem signals
    hits = sum(1 for sig in _IDEA_SIGNALS if sig in t)
    return hits >= 2


@api.post("/interact")
async def interact_route(req: InteractRequest):
    last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    use_llm = GROQ_CLIENT and groq_key and groq_key != "dummy"

    already_analyzed = req.context.get("tas_score") is not None and req.context.get("tas_score") != 0

    # Step 1: keyword gate — fast hard filter (runs regardless of LLM availability)
    keyword_says_idea = _keyword_is_idea(last_user_msg)
    is_idea = False

    if not already_analyzed:
        if not keyword_says_idea:
            # Short-circuit: conversational message, no need to ask LLM
            is_idea = False
        elif use_llm:
            # Step 2: LLM confirmation only when keywords suggest it might be an idea
            route_prompt = dedent(f"""
                A founder sent this message to a startup market analysis tool.
                Does this message describe a specific startup business idea (with product, market, or sector details)?
                Or is it a meta-request, greeting, or vague statement about wanting to share an idea later?

                Message: "{last_user_msg}"

                EXAMPLES of NOT an idea: "i have an idea", "i want to share something", "can you help me", "hello"
                EXAMPLES of IS an idea: "an app for Egyptian SMEs to get invoices paid faster", "a SaaS platform for logistics companies in Dubai"

                Respond ONLY with valid JSON: {{"is_idea": true}} or {{"is_idea": false}}
            """).strip()
            try:
                resp = GROQ_CLIENT.chat.completions.create(
                    messages=[{"role": "user", "content": route_prompt}],
                    model="llama-3.1-8b-instant", temperature=0.0, max_tokens=20
                )
                raw = resp.choices[0].message.content.strip()
                import re as _re
                m = _re.search(r'\{[^}]+\}', raw)
                if m:
                    is_idea = json.loads(m.group(0)).get("is_idea", False)
                else:
                    is_idea = "true" in raw.lower()
            except Exception:
                is_idea = True  # keywords already said yes, trust that on LLM failure
        else:
            # No LLM, keywords said yes → treat as idea
            is_idea = True

    if is_idea and not already_analyzed:
        try:
            analysis_data = process_idea(last_user_msg)
            analysis_data["report"] = { "finding": analysis_data["report"]["finding"], "implication": analysis_data["report"]["implication"], "action": analysis_data["report"]["action"]}
            ai_reply = f"I've analyzed your startup idea against live market data for **{analysis_data['sector'].title()}** in **{analysis_data['country']}**. Here is what the numbers say. Read through the analysis card below, and let me know if you have any follow-up questions."
            return {"success": True, "type": "analysis", "reply": ai_reply, "data": analysis_data}
        except Exception as e:
            return {"success": False, "type": "chat", "reply": f"Sorry, there was an error processing your idea: {str(e)}"}
    else:
        # Standard chat using Groq (or fallback) based on context
        chat_req = ChatRequest(context=req.context, messages=req.messages)
        chat_res = await chat_interaction(chat_req)
        return {"success": True, "type": "chat", "reply": chat_res.get("reply", ""), "data": None}

@api.get("/health")
async def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)

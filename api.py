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

def _chat_fallback(req: ChatRequest) -> str:
    """
    Context-aware multi-turn fallback when no LLM is available.
    Varies by: conversation turn count, whether analysis exists, message intent.
    """
    sector  = req.context.get("sector", "")
    country = req.context.get("country", "")
    regime  = req.context.get("regime", "").replace("_", " ").title()
    tas     = req.context.get("tas_score", 0)
    idea    = req.context.get("idea", "")

    user_turns = [m for m in req.messages if m.role == "user"]
    turn_n = len(user_turns)
    last_msg = user_turns[-1].content.lower() if user_turns else ""

    # ── NO ANALYSIS CONTEXT YET ──
    if not sector:
        greet_words = ["hi","hello","hey","yo","sup","good morning","good evening","greetings","what's up"]
        meta_words  = ["what do you do","who are you","how does this work","what is midan","explain","tell me about"]
        if any(g in last_msg for g in greet_words):
            if turn_n == 1:
                return ("Hey — I'm MIDAN, your AI startup market advisor. "
                        "Describe your startup idea and I'll run 8 agents across live market data: "
                        "classify the regime, forecast 90-day trends, evaluate your concept, and give you a single verdict. "
                        "What are you building, and where?")
            return "Still here — what's the startup idea? Even a rough one-liner works."
        if any(m in last_msg for m in meta_words):
            return ("MIDAN runs 8 AI agents on your startup idea in parallel — "
                    "NLP parsing, SVM market classification, SARIMA forecasting, LightGBM SHAP explainability, and LLM synthesis. "
                    "The output: a market regime, an opportunity score, a Startup Viability Score, "
                    "and a three-point action plan. Tell me what you're building.")
        if any(w in last_msg for w in ["help","can you","could you"]):
            return ("Absolutely. Share your startup idea — what you want to build and where — "
                    "and I'll do a full market intelligence run on it.")
        if turn_n == 1:
            return ("MIDAN here. Tell me your startup idea — what problem you're solving, "
                    "what sector, and which market. I'll handle the analysis.")
        return "What are you building? Drop the idea and I'll analyze the market for you."

    # ── HAS ANALYSIS CONTEXT ──
    reg_adj = "hot" if "GROWTH" in regime.upper() else "cautious" if "EMERGING" in regime.upper() else "difficult"
    timing  = "now is your window" if tas >= 70 else "timing needs work" if tas >= 50 else "the market isn't ready yet"

    if any(w in last_msg for w in ["pivot","change direction","switch","alternative","different idea"]):
        return (f"Pivoting in {sector}/{country} makes sense only if you're solving a different pain point — "
                f"not just re-skinning the same idea. The {regime} conditions stay regardless of your pivot. "
                f"Before switching anything, run 10 targeted customer interviews and ask what they're currently paying to fix. "
                f"What specific angle are you considering?")

    if any(w in last_msg for w in ["compet","rival","who else","already exist","crowded","saturated"]):
        return (f"The {sector} space in {country} has both local operators and global entrants. "
                f"At {tas}/100, your TAS already prices in competitive density. "
                f"Your moat has to be local — regulatory navigation, language, trust networks, or hyper-specific workflow fit. "
                f"What makes you the one who can win this specific geography?")

    if any(w in last_msg for w in ["fund","invest","raise","capital","vc","angel","pitch","accelerator"]):
        if tas >= 68:
            return (f"With {tas}/100 TAS in a {regime}, you're fundable — but only after proof. "
                    f"Target Flat6Labs, 500 MENA, or Algebra Ventures for pre-seed in Egypt; Wamda or BECO for the Gulf. "
                    f"Come in with 3 months of traction data and a clear CAC-to-LTV model. "
                    f"What's your current MRR or user count?")
        return (f"At {tas}/100, institutional money will be hard to close right now. "
                f"Bootstrap or find a revenue-generating angle first — consulting, a paid pilot, a letter of intent. "
                f"10 paying customers changes every investor conversation. What's the fastest path to your first dollar?")

    if any(w in last_msg for w in ["risk","danger","threat","fail","worry","concern","scared","afraid"]):
        risks = {
            "fintech":    "regulatory licensing timelines and CBE approval cycles",
            "healthtech": "medical licensing friction and institutional procurement cycles",
            "edtech":     "low willingness-to-pay and high churn after free trials",
            "ecommerce":  "logistics cost structure and last-mile unit economics",
            "saas":       "enterprise sales cycle length and integration friction",
            "logistics":  "fuel price volatility and driver retention cost",
            "agritech":   "seasonal revenue concentration and farmer trust barriers",
        }
        top_risk = risks.get(sector.lower(), "regulatory and currency volatility")
        return (f"Top risk in {sector}/{country}: {top_risk}. "
                f"Secondary risks: customer acquisition cost spikes in {regime} conditions, "
                f"and FX exposure if you're billing in local currency. "
                f"Mitigate by keeping monthly burn under $5K until you hit product-market fit. "
                f"Which of these risks worries you most?")

    if any(w in last_msg for w in ["next","first step","what do i do","how do i start","begin","roadmap","plan"]):
        if tas >= 68:
            return (f"Strong signals — {timing}. Three moves: "
                    f"1) Build a 4-week MVP that solves exactly one workflow. "
                    f"2) Put it in front of 20 target users this month, not next. "
                    f"3) Apply to Flat6Labs Spring cycle or 500 MENA before the deadline. "
                    f"Which of these three is the biggest blocker for you right now?")
        return (f"Market isn't ideal at {tas}/100 — but that doesn't mean wait. "
                f"1) Run 20 customer discovery calls this month. "
                f"2) Find the one use case with highest willingness-to-pay. "
                f"3) Build only that. No code before evidence. "
                f"What's stopping you from making those 20 calls this week?")

    if any(w in last_msg for w in ["gtm","go to market","launch","acquire","marketing","growth","traction","users"]):
        return (f"GTM for {sector} in {country}: start with one micro-segment you can dominate in 60 days. "
                f"In MENA, WhatsApp outreach + warm referrals from your first 5 customers beats paid ads every time. "
                f"Set a specific Week-4 target: X signups, Y paid pilots, Z interviews. "
                f"What does your Week-4 target look like right now?")

    if any(w in last_msg for w in ["pricing","price","monetize","revenue model","charge","subscription","fee"]):
        return (f"For {sector} in {country}: avoid freemium unless you have viral mechanics. "
                f"B2B in this market responds best to outcome-based pricing — charge a percentage of value created, "
                f"not a flat monthly fee. Start with a high-touch paid pilot at $200-500/mo "
                f"and use it to prove ROI before scaling. What's the core value you can guarantee?")

    if any(w in last_msg for w in ["team","hire","co-founder","talent","people","employees"]):
        return (f"Team composition matters more than idea in MENA's early-stage market. "
                f"For {sector}, you need a technical co-founder if you're non-technical, "
                f"or a sales/BD co-founder if you are. "
                f"Hire contractors before employees — prove the role first. "
                f"What skill gap is the most critical bottleneck right now?")

    if any(w in last_msg for w in ["thank","thanks","great","awesome","helpful","nice","good","perfect"]):
        return (f"Glad that's useful. Your {sector} play in {country} has real potential — "
                f"the {regime} conditions with {tas}/100 TAS means {timing}. "
                f"What else do you want to pressure-test?")

    # Intelligent default that references their specific data
    return (f"Your {sector.title()} idea in {country} sits in a {regime} with {tas}/100 market opportunity. "
            f"That means {timing}. "
            f"Push me on competitive strategy, fundraising, GTM, pricing, team, or risk — "
            f"I'll give you a straight answer grounded in your market data.")


@api.post("/chat")
async def chat_interaction(req: ChatRequest):
    groq_key = os.environ.get("GROQ_API_KEY", "")
    use_llm = GROQ_CLIENT and groq_key and groq_key != "dummy"

    if not use_llm:
        reply = _chat_fallback(req)
        return {"success": True, "reply": reply}

    sector  = req.context.get("sector", "unknown")
    country = req.context.get("country", "unknown")
    regime  = req.context.get("regime", "UNKNOWN").replace("_", " ").title()
    tas     = req.context.get("tas_score", 0)
    idea    = req.context.get("idea", "")
    impl    = req.context.get("implication", "")

    system_prompt = dedent(f"""
        You are MIDAN — a brutally honest AI startup market advisor trained on thousands of MENA venture outcomes.
        You have just completed a full quantitative analysis for this founder. Use it in every response.

        ── ANALYSIS RESULTS ──
        Idea: "{idea}"
        Sector: {sector.title()} | Country: {country}
        Market Regime: {regime}
        Opportunity Score (TAS): {tas}/100
        AI Synthesis: {impl}

        ── YOUR PERSONA ──
        - Speak like a senior VC partner who has reviewed 2,000 MENA startups
        - Every single response must reference specific numbers from the analysis (TAS, regime, country, sector)
        - Be direct and specific — no platitudes, no generic startup advice
        - Never start with "I" — lead with the insight
        - End every response with exactly one sharp, specific question that challenges the founder's assumptions
        - Max 3–4 sentences total. Dense, not long.

        ── TONE EXAMPLES ──
        WRONG: "Great question! You should focus on your target market and build an MVP."
        RIGHT: "At {tas}/100 TAS in a {regime}, you have roughly 6 months before conditions tighten. The data says move now — but your idea score tells a different story. What specifically have you validated with actual paying customers in {country} in the last 30 days?"

        WRONG: "There are many competitors in this space, so you need to differentiate."
        RIGHT: "The {sector} regime in {country} has 3 funded players above Series A. Your differentiation can't be features — it has to be distribution. Who in {country} do you have access to that Stripe or Paymob doesn't?"
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

# ── INTENT CLASSIFICATION SIGNALS ─────────────────────────────────────────────

_GREET_TOKENS = {
    'hi', 'hello', 'hey', 'yo', 'sup', 'hola', 'marhaba', 'ahlan',
    'good morning', 'good evening', 'good afternoon', 'howdy', 'greetings',
}
_META_PHRASES = [
    'what do you do', 'who are you', 'how does this work', 'what is midan',
    'what can you do', 'explain to me', 'help me understand', 'tell me about yourself',
    'what are you', 'how do i use this', 'getting started', 'what does midan',
]
# Vague intent declarations — user signals they HAVE an idea but doesn't describe it
_VAGUE_STARTERS = [
    'i have an idea', 'i have a startup idea', 'i want to share', 'i want to tell you',
    'i would like to share', 'i am thinking of', 'i am thinking about',
    'what do you think', 'give me feedback', 'give me your thoughts',
    'can you analyze', 'can you help me with', 'i need help with',
    'i have something', 'i have a concept', 'let me tell you',
    'so i was thinking', 'i was thinking about', 'been working on', 'been thinking about',
    'working on something', 'building something', 'have an idea', 'had an idea',
    'i have this idea', 'just an idea', 'rough idea', 'early idea',
    'i want to build', 'i want to create', 'i want to start', 'i am building',
    'i am working on', 'i am developing', 'i am creating',
]

# The three required components for a complete idea description
_PROBLEM_SIGNALS = [
    'problem', 'pain', 'issue', 'struggle', 'challenge', 'gap', 'need', 'lack',
    'inefficient', 'inefficiency', 'expensive', 'slow', 'difficult', 'hard to',
    'no way to', 'broken', 'frustrat', 'wast', 'manual process',
    'time-consuming', 'complex', 'unreliable', 'inaccessible',
    'underserved', 'unbanked', 'overpriced', 'delayed', 'stuck',
    'can\'t afford', 'cannot afford', 'don\'t have access', 'no access',
]
_SOLUTION_SIGNALS = [
    'app', 'platform', 'tool', 'service', 'system', 'software', 'marketplace',
    'saas', 'api', 'dashboard', 'website', 'bot', 'chatbot', 'solution', 'product',
    'connect', 'automate', 'enable', 'simplify', 'streamline', 'digitize',
    'ai-powered', 'using ai', 'machine learning', 'mobile app', 'web app',
    'mobile application', 'subscription model', 'subscription service',
]
_MARKET_GEO = [
    'egypt', 'cairo', 'egyptian', 'giza', 'alexandria',
    'saudi', 'riyadh', 'jeddah', 'ksa', 'saudi arabia',
    'uae', 'dubai', 'abu dhabi', 'emirates', 'united arab',
    'morocco', 'casablanca', 'rabat', 'marrakech',
    'nigeria', 'lagos', 'abuja',
    'kenya', 'nairobi',
    'usa', 'america', 'silicon valley', 'new york', 'united states',
    'uk', 'london', 'britain', 'england',
    'mena', 'africa', 'middle east', 'gulf', 'gcc',
    'global', 'worldwide', 'international', 'emerging market',
]
_MARKET_CUSTOMER = [
    'sme', 'smes', 'small business', 'small businesses', 'enterprise', 'enterprises',
    'b2b', 'b2c', 'd2c', 'startup', 'startups',
    'consumer', 'consumers', 'user', 'users', 'customer', 'customers', 'client', 'clients',
    'patient', 'patients', 'student', 'students',
    'farmer', 'farmers', 'freelancer', 'freelancers',
    'driver', 'drivers', 'merchant', 'merchants', 'retailer', 'retailers',
    'hospital', 'hospitals', 'clinic', 'clinics',
    'school', 'schools', 'university', 'universities',
    'individual', 'family', 'families', 'company', 'companies',
]


def _extract_components(text: str) -> dict:
    """
    Detect the three required idea components: problem, solution, market.
    Each must be explicitly present in the text — no assumptions, no defaults.
    """
    t = text.lower()
    wc = len(t.split())

    has_problem  = any(sig in t for sig in _PROBLEM_SIGNALS)
    has_solution = any(sig in t for sig in _SOLUTION_SIGNALS)
    has_geo      = any(geo in t for geo in _MARKET_GEO)
    has_customer = any(cus in t for cus in _MARKET_CUSTOMER)
    has_market   = has_geo or has_customer

    return {
        'has_problem':    has_problem,
        'has_solution':   has_solution,
        'has_market':     has_market,
        'word_count':     wc,
        'is_substantial': wc >= 12,   # Minimum for a meaningful description
    }


def _clarify_question(missing: list, count: int) -> str:
    """Return exactly ONE sharp clarification question based on the first missing component."""
    if not missing:
        return ""
    if 'problem' in missing:
        if count == 0:
            return (
                "What specific problem are you solving, and for whom? "
                "Tell me who suffers from this today and what they currently do instead."
            )
        return (
            "I still need the core pain point — who specifically struggles with this, "
            "and what does it cost them in time, money, or friction?"
        )
    if 'market' in missing:
        return (
            "Which market are you targeting? "
            "Tell me the geography (Egypt, UAE, Saudi, etc.) and customer type — "
            "individual consumers, SMEs, hospitals, schools, enterprises?"
        )
    if 'solution' in missing:
        return (
            "What's your solution approach? "
            "App, platform, marketplace, SaaS, physical service? "
            "Even a rough mechanism — how does it actually solve the problem?"
        )
    return "Can you tell me a bit more about what you're building?"


@api.post("/interact")
async def interact_route(req: InteractRequest):
    last_user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    t = last_user_msg.lower().strip()
    wc = len(t.split())

    already_analyzed = bool(req.context.get("tas_score"))

    # ── GATE 1: Post-analysis → pure advisor chat (no re-analysis) ────────────
    if already_analyzed:
        chat_req = ChatRequest(context=req.context, messages=req.messages)
        chat_res = await chat_interaction(chat_req)
        return {"success": True, "type": "chat", "reply": chat_res.get("reply", ""), "data": None}

    # ── GATE 2: Pure greeting or meta question ─────────────────────────────────
    is_greeting = wc <= 6 and any(t == g or t.startswith(g) for g in _GREET_TOKENS)
    is_meta     = any(phrase in t for phrase in _META_PHRASES)

    if is_greeting or is_meta:
        chat_req = ChatRequest(context={}, messages=req.messages)
        chat_res = await chat_interaction(chat_req)
        return {"success": True, "type": "chat", "reply": chat_res.get("reply", ""), "data": None}

    # ── GATE 3: Vague intent declaration — user says they HAVE an idea but gives no details ──
    # This catches: "i have an idea", "i want to build something", "give me feedback", etc.
    is_vague = wc <= 20 and any(t.startswith(v) or t == v.strip() for v in _VAGUE_STARTERS)

    if is_vague:
        return {
            "success": True,
            "type": "clarifying",
            "reply": (
                "Let's get into it. Before I run the analysis, I need three things:\n\n"
                "**1. The problem** — what pain point exists, and who has it?\n"
                "**2. The market** — geography and customer type\n"
                "**3. Your approach** — app, marketplace, SaaS, service?\n\n"
                "Write it out in plain language — no structure needed. Even 2–3 sentences is enough."
            ),
            "clarification_state": req.context.get("clarification_state", {}),
            "data": None,
        }

    # ── GATE 4: Has actual content — extract and check the three components ────
    clarification_state = req.context.get("clarification_state", {})
    clarification_count = clarification_state.get("count", 0)
    accumulated_text    = clarification_state.get("accumulated_text", "")

    components = _extract_components(last_user_msg)

    # Merge with components already gathered from previous turns
    for key in ("has_problem", "has_solution", "has_market"):
        if clarification_state.get(key):
            components[key] = True

    # Accumulate all idea text across turns for a richer final analysis
    new_accumulated = (accumulated_text + " " + last_user_msg).strip()

    # Minimum required: solution + market (problem can be implied by context).
    # If only problem is missing but solution + market are clear → allow analysis.
    # If solution or market is missing → always clarify, no matter what.
    hard_missing = [
        k for k in ("solution", "market")
        if not components.get(f"has_{k}")
    ]
    # "missing" drives the clarification question priority (problem first, then market, then solution)
    missing = [
        k for k in ("problem", "solution", "market")
        if not components.get(f"has_{k}")
    ]

    # ── GATE 5: Solution + market present + substantial text → run full analysis ──
    if not hard_missing and components["is_substantial"]:
        analysis_text = new_accumulated if new_accumulated else last_user_msg
        try:
            analysis_data = process_idea(analysis_text)
            analysis_data["report"] = {
                "finding":     analysis_data["report"]["finding"],
                "implication": analysis_data["report"]["implication"],
                "action":      analysis_data["report"]["action"],
            }
            ai_reply = (
                f"I've analyzed your idea against live market data for "
                f"**{analysis_data['sector'].title()}** in **{analysis_data['country']}**. "
                f"Here's the full breakdown — read through the report, then ask me anything."
            )
            return {
                "success": True, "type": "analysis",
                "reply": ai_reply, "data": analysis_data,
            }
        except Exception as e:
            return {
                "success": False, "type": "chat",
                "reply": f"Analysis error: {str(e)}", "data": None,
            }

    # ── GATE 6: Partial info → one targeted clarification question ─────────────
    new_state = {
        "has_problem":      components["has_problem"],
        "has_solution":     components["has_solution"],
        "has_market":       components["has_market"],
        "accumulated_text": new_accumulated,
        "count":            clarification_count + 1,
    }

    # Acknowledge what was newly detected in this turn
    newly_found = []
    for key, label in [
        ("has_problem", "the problem"),
        ("has_market",  "the market"),
        ("has_solution", "the approach"),
    ]:
        if components[key] and not clarification_state.get(key):
            newly_found.append(label)

    # Use hard_missing for the question (only ask about solution/market, not problem)
    question = _clarify_question(hard_missing if hard_missing else missing, clarification_count)

    if newly_found and (hard_missing or not components["is_substantial"]):
        ack = f"Got {' and '.join(newly_found)}. "
        reply = ack + question
    elif not components["is_substantial"] and not hard_missing:
        reply = (
            "I can see the shape of the idea — expand a bit. "
            "Describe what you're building, who it's for, and what problem it solves. "
            "Two or three sentences is enough to run the analysis."
        )
    else:
        reply = question or "Tell me more — what you're building, who it's for, and your approach."

    return {
        "success": True,
        "type": "clarifying",
        "reply": reply,
        "clarification_state": new_state,
        "data": None,
    }


@api.get("/health")
async def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)

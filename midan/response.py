"""
midan.response — payload builder, chat fallback, operator reply, projection.

build_response_payload is the single mapper from raw pipeline output to the
strict ResponsePayload schema. Chat / operator / projection generators all
read the L4 decision envelope as their source of truth — never legacy TAS.
"""
from midan.core import *  # noqa: F401,F403
from midan.l1_parser import _heuristic_idea_features, agent_a1_parse  # noqa: F401
from midan.l3_reasoning import _signal_tier  # noqa: F401
from midan.l4_decision import _l4_top_risk_dim  # noqa: F401
from midan.conversation import (  # noqa: F401
    _post_decision_route, _extract_components,
    _smart_followup, _casual_response,
    _GENERIC_IDEA_TOKENS, _GENERIC_CUSTOMER_HINTS, _GENERIC_MECHANISM_HINTS,
    _EXPLICIT_MECHANISM_SIGNALS,
    _PROBLEM_SIGNALS, _SOLUTION_SIGNALS, _MARKET_GEO, _MARKET_CUSTOMER,
    _CASUAL_SHORT_SET, _CASUAL_PREFIXES,
)


# ── extracted from api.py ─────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
# LAYER 4 — STRATEGIC REASONING ENGINE (legacy text generator)
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
# LLM path: Gemini JSON-structured prompt (temperature 0.35)
# Fallback: rich conditional logic by BM type × regime × signals
# ═══════════════════════════════════════════════════════════════

def _l4_reasoning_llm(
    sector: str, country: str, regime: str, conf: float,
    sarima_trend: float, idea_features: dict, idea_signal_data: dict,
    shap_dict: dict, tas: float, signal_tier: str,
    a2_comps: list, a4_sentiment: str,
    idea_text: str,
    diff_label: str, bm_label: str, seg_label: str,
    mechanism_analysis: dict = None,
) -> Optional[dict]:
    """
    LLM path: structured JSON reasoning with 7 required fields.
    Returns None on any failure — fallback handles it.
    Tone is epistemic-state-conditional: evidence_quality governs hedging level.
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
    grounding     = _extract_idea_grounding(idea_text, sector, idea_features, country)

    # we set the epistemic tone conditional on mechanism pipeline evidence_quality;
    # evidence_quality drives hedging level: strong=decisive, insufficient=hypothesis-only
    _mech = mechanism_analysis or {}
    _ep   = _mech.get("epistemic_summary", {})
    _eq   = _ep.get("evidence_quality", "moderate")
    _tone_map = {
        "strong":       (
            "Produce a DECISION, not an analysis. No hedging. No balanced takes. "
            "Surface what is actually true and what will actually break this."
        ),
        "moderate":     (
            "Produce calibrated reasoning. State conclusions where evidence is direct. "
            "Use 'available evidence suggests' or 'the signals indicate' for inferred claims. "
            "Do not present inferences as established facts."
        ),
        "weak":         (
            "Frame conclusions as current signal patterns, not confirmed facts. "
            "Use 'current signals suggest', 'available evidence indicates', 'the pattern points toward' throughout. "
            "Every strategic claim must carry an epistemic qualifier."
        ),
        "insufficient": (
            "Frame all outputs as pattern-based hypotheses derived from structural signals only. "
            "Qualify every field: 'based on available structural signals', "
            "'insufficient evidence to conclude — the pattern suggests'. "
            "Do not state any mechanism conclusion as certain."
        ),
    }
    _tone_instruction = _tone_map.get(_eq, _tone_map["moderate"])

    # we prepend the mechanism context block outside dedent to avoid indentation issues
    _mech_lines = []
    if _mech:
        _obs  = _ep.get("observed_signals", [])
        _inf  = _ep.get("inferred_mechanisms", [])
        _disc = _ep.get("recommended_disclosure", "")
        _ms   = _mech.get("market_structure", {})
        _unc  = _mech.get("uncertainty", 0.0)
        _mech_lines = [
            "MECHANISM PIPELINE CONTEXT (use for epistemic framing — do not re-derive):",
            f"Evidence quality: {_eq} | Uncertainty: {_unc:.2f}",
            f"Directly observed: {', '.join(_obs) or 'none'}",
            f"Inferred: {', '.join(_inf) or 'none'}",
            f"Market structure: {_ms.get('category', 'unknown')} (confidence={_ms.get('confidence', 0):.0%})",
            f"Epistemic disclosure: {_disc}",
        ]
    _mech_block = "\n".join(_mech_lines)

    _core_prompt = dedent(f"""
        You are MIDAN — a senior operator who has backed 50 startups and killed 200 pitches.
        {_tone_instruction}

        COMPLETE SIGNAL PICTURE (do not re-derive — use this):
        Market: {sector.title()} in {country} → {regime_r} ({conf:.0%} confidence)
        90-day trend: {'growing' if sarima_trend > 0.5 else 'declining or flat'} (SARIMA={sarima_trend:.2f})
        Top macro signal: {top_shap} (caused this regime classification)
        Sentiment: {a4_sentiment} | Competitors: {', '.join(a2_comps[:2]) if a2_comps else 'unidentified'}

        Original founder idea:
        "{idea_text}"

        Grounding extracted from that sentence:
        - Context: {grounding['context_label']}
        - Customer: {grounding['customer_label']}
        - Core job: {grounding['motion_label']}
        - Core pain: {grounding['problem_label']}

        Idea anatomy:
        BM={bm_label} | Segment={seg_label} | Stage={stage}
        Diff={diff}/5 ({diff_label}) | Competition={comp} | Reg={reg} | Market-readiness={ready}/5
        Signal: {signal_tier} ({idea_signal:.0%}) | Primary risk type: {dominant_risk}
        Moat source for this BM: {moat_source}

        Produce EXACTLY this JSON — 7 fields, no markdown, no extra text:
        {{
          "strategic_interpretation": "2-3 sentences. Answer three things in sequence: (1) what is the non-obvious opportunity window here — not the surface pitch but what the market condition actually enables right now; (2) why is NOW the relevant timing for {bm_label} in {sector}/{country} — reference {top_shap} as the causal force; (3) what does success look like at 12 months for this specific segment/model combo. Do NOT start with the regime name. Do NOT summarize the idea back at the founder.",
          "key_driver": "1 sentence only. Name the specific mechanism this model depends on — pick the closest match: supply density, trust layer, distribution channel, regulatory window, switching cost, data network effect, or geographic lock-in. Then state its test: 'If [named mechanism] does not reach [specific threshold] by [timeframe], this fails.' No abstractions. No 'execution quality' as an answer.",
          "main_risk": "1-2 sentences. Write in failure scenario format ONLY: 'If [specific observable trigger] happens, [downstream consequence] follows, and the business ends up [specific dead position].' Name the trigger, the cascade, and the landing. Never name a risk category — name the mechanism of failure.",
          "counterpoint": "1-2 sentences. Name the ONE belief that makes this idea feel defensible to the founder — then destroy it directly. Format: 'You are assuming [X] — but the evidence points to [Y].' Do not soften, do not offer balance. Find the specific thing the founder is wrong about and say it plainly.",
          "differentiation_insight": "1-2 sentences. If diff={diff}/5 is 4 or 5: name the specific moat mechanism ({moat_source}) and explain why it is actually defensible in this market. If diff is 3 or below: state plainly why this is not differentiated — name what existing players already do that covers this ground — and what structural change would be required before differentiation is real. Do not use the word 'unique'. Do not encourage what is not defensible.",
          "what_matters_most": "ONE sentence. The single variable — different from the mechanism in key_driver — that will determine whether this specific {bm_label}+{sector}+{stage} combination succeeds or collapses. Not a principle. A named, specific, observable variable.",
          "counter_thesis": "1-2 sentences. Second-person bear case. Format: 'You think [what the founder believes is the challenge] — the real problem is [what actually kills this], and [why that is harder to solve than the founder believes].' This must directly contradict strategic_interpretation. Be blunt. No hedging."
        }}

        HARD RULES (violations will be rejected):
        - Use the original founder idea as the primary source of specificity; the labels may be imperfect summaries
        - If the idea mentions restaurants, do not generalize it into farmers/agriculture unless the text explicitly says farms or farmers
        - Every field must be grounded in the idea text, {bm_label}, {seg_label}, {sector}, {country}, or specific numbers
        - counterpoint must contradict strategic_interpretation — not extend it
        - counter_thesis must name a SPECIFIC wrong assumption — not generic startup risk
        - No platitudes: no 'consider your options', 'focus on value proposition', 'it depends'
        - No starting any field with 'I'
        - what_matters_most and key_driver must be about DIFFERENT variables
        - main_risk must name the failure mechanism — not just a risk category
        - If any sentence could apply unchanged to a different startup idea, rewrite it
    """).strip()
    prompt = (_mech_block + "\n\n" + _core_prompt) if _mech_block else _core_prompt

    try:
        response = LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.30,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences — Llama commonly wraps JSON in ```json ... ```
        if raw.startswith('`'):
            raw = re.sub(r'^```(?:json)?\s*\n?', '', raw)
            raw = re.sub(r'\n?\s*```\s*$', '', raw)
        raw = raw.strip()
        data = json.loads(raw)
        required = ['strategic_interpretation', 'key_driver', 'main_risk', 'counterpoint',
                    'differentiation_insight', 'what_matters_most', 'counter_thesis']
        if all(data.get(f) and len(str(data.get(f, ''))) > 25 for f in required):
            return {f: str(data[f]).strip() for f in required}
    except Exception as llm_err:
        import logging
        logging.getLogger("midan.l4").warning(
            f"[L4] _l4_reasoning_llm failed ({type(llm_err).__name__}: {llm_err!r}) "
            f"— falling back to conditional logic"
        )
    return None


def _generate_go_decision_llm(
    tas: float,
    decision_badge: str,
    seg_label: str,
    bm_label: str,
    dominant_risk: str,
    top_signal: str,
    reg_risk: str,
    stage: str,
    regime: str,
    sector: str,
    country: str,
    idea_text: str,
    diff: int,
) -> Optional[str]:
    """
    LLM-generated GO decision text. 4 sentences, idea-specific.
    Returns None on failure — caller uses hardcoded fallback.
    """
    regime_r = regime.replace('_', ' ').title()
    tas_pct  = int(tas * 100)

    prompt = dedent(f"""
        You are MIDAN — a senior operator making a final go/no-go call on a startup idea.
        This is not analysis. This is a decision. Write like one.

        Decision: {decision_badge}
        Signal score: {tas_pct}/100 in a {regime_r} market
        Dominant risk: {dominant_risk}
        Regulatory risk: {reg_risk}
        Key market signal: {top_signal}
        Stage: {stage}
        Differentiation: {diff}/5

        Original founder idea:
        "{idea_text}"

        Idea profile:
        - Business model: {bm_label}
        - Target segment: {seg_label}
        - Sector: {sector} in {country}

        Write the decision action in EXACTLY 4 sentences. Output nothing else — no headers, no bullets, no markdown.

        Sentence 1 — The command: State the core condition or command that matches {decision_badge}. What must happen (or must not happen) before anything else. Make it specific to this idea — not generic advice.

        Sentence 2 — The milestone: Name ONE concrete validation step or build milestone specific to {bm_label} serving {seg_label} in {sector}/{country}. Make it time-bounded and falsifiable. Not 'get feedback' — name what specific evidence looks like.

        Sentence 3 — The kill signal: The specific observable event that tells the founder this is not working. Format: 'If [specific, measurable thing] happens, stop.' Must be idea-specific — not 'if there is no traction'.

        Sentence 4 — The funding sentence: If {decision_badge} contains 'GO' or 'CONDITIONAL': name 2-3 investors who actually back {sector} startups in {country} or the MENA/Africa region at {stage} stage — use real, specific fund or accelerator names relevant to this sector, not a generic list. If {decision_badge} contains 'NO-GO' or 'HIGH RISK': state what specific evidence would be required before raising is appropriate.

        HARD RULES:
        - Every sentence must be specific to the idea text — zero sentences can apply unchanged to a different startup
        - Do not default to Flat6Labs, 500 MENA, Algebra Ventures as a template — only name them if genuinely relevant to {sector}
        - Do not hedge — this is a decision, not a balanced analysis
        - Do not start any sentence with 'I'
        - Do not start with the decision badge text
        - 4 sentences only — no more, no less
    """).strip()

    try:
        response = LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.30,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith('`'):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
        if len(text) > 80:
            return text
    except Exception as go_err:
        import logging
        logging.getLogger("midan.l4").warning(
            f"[L4] _generate_go_decision_llm failed ({type(go_err).__name__}: {go_err!r}) — using fallback"
        )
    return None


def _l4_reasoning_fallback(
    sector: str, country: str, regime: str, conf: float,
    sarima_trend: float, idea_features: dict, idea_signal_data: dict,
    shap_dict: dict, tas: float, signal_tier: str,
    idea_text: str,
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
    grounding     = _extract_idea_grounding(idea_text, sector, idea_features, country)
    customer_label = grounding['customer_label']
    context_label = grounding['context_label']
    motion_label = grounding['motion_label']
    problem_label = grounding['problem_label']
    market_label = grounding['market_label']
    workflow_software = _is_workflow_software_idea((idea_text or '').lower(), seg)

    # ── 1. Strategic Interpretation — we generate this to be structurally different by BM + regime ──
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
        elif workflow_software and seg == 'b2b':
            si = (
                f"This is {context_label} for {customer_label} in {market_label}, not a generic {sector} play. "
                f"In {regime_r} conditions, the win condition is getting operators to trust {motion_label} enough to change a live workflow, not just agree that {problem_label} exists. "
                f"The {top_shap} signal matters because tighter operating budgets make ROI legible faster, but they also make buyers ruthless about anything that feels optional."
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

    # ── 2. Key Driver — we pick the one thing that determines success or failure ──
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
        'workflow': (
            f"Workflow insertion speed — whether {customer_label} use {motion_label} to make a real operating decision within 14 days. "
            f"If the product stays informational instead of changing ordering, planning, or purchasing behavior, the pilot dies as a nice-to-have dashboard."
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

    # ── 3. Main Risk — we derive the specific failure mode from the signal profile ──
    if dominant_risk == 'workflow':
        main_risk = (
            f"Workflow non-adoption inside {customer_label}. "
            f"The likely failure is not model accuracy — it is that the buyer keeps using instinct, spreadsheets, or supplier relationships because your product does not become the default operating loop quickly enough. "
            f"If you cannot prove measurable ROI on {problem_label} inside one buying cycle, the pilot stalls as 'interesting' and never becomes budget-critical."
        )
    elif dominant_risk == 'liquidity' and comp == 'high':
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

    # ── 4. Counterpoint — we aggressively challenge the comfortable interpretation ──
    if workflow_software and seg == 'b2b':
        counterpoint = (
            f"The assumption here is that better predictions automatically create willingness-to-pay. They do not. "
            f"{customer_label.title()} pay when the product changes a cash decision, labor decision, or purchasing decision fast enough that ignoring it feels expensive. "
            f"If {motion_label} stays advisory instead of operational, you built insight, not software people budget for."
        )
    elif signal_tier == 'Strong':
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

    # ── 5. Differentiation Insight — we explain why this idea behaves differently ──
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

    # ── 6. What Matters Most — we identify the single make-or-break variable ──
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
    if workflow_software and seg == 'b2b':
        what_matters_most = (
            f"Whether {customer_label} in {market_label} will let the product drive the next real ordering or planning cycle — if it never reaches that operational moment, the software never becomes budget-critical."
        )
    elif wmm_key not in _wmm_map:
        # we fall back to dominant_risk-specific text when the BM+stage key isn't in the map
        wmm_fallback = {
            'liquidity':       f"Whether the supply side shows up without you manually recruiting every participant — if it requires your personal effort, it will not scale.",
            'differentiation': f"Whether {seg_label} customers choose you over the default alternative in a blind test — if the answer is no, the differentiation is in your head, not in the market.",
            'workflow':        f"Whether {customer_label} change a real operating decision because of {motion_label} inside the first two weeks — if not, the product is informative but not mission-critical.",
            'regulatory':      f"The regulatory clearance timeline — it is the only uncontrollable variable that can terminate the company regardless of how good the product is.",
            'churn':           f"Month-2 retention — the subscription model's entire unit economics hinges on this single number, and most founders do not measure it until it is already fatal.",
            'capital':         f"Whether the prototype proves the single most uncertain assumption before you spend the next tranche — hardware projects that skip this die at manufacturing scale.",
            'scalability':     f"Whether delivery can be templated without your involvement by week 8 — if not, you are building a consulting practice that cannot be valued as a company.",
            'execution':       f"Getting a paying customer to return a second time — the first transaction proves willingness to pay; the second proves the product delivers on the promise.",
        }
        what_matters_most = wmm_fallback.get(dominant_risk, f"Whether this idea solves a problem urgent enough that {seg_label} customers would pay for it without being asked twice.")
    else:
        what_matters_most = _wmm_map[wmm_key]

    # ── 7. Counter-Thesis — we challenge the CORE assumption embedded in the idea ──
    _ct_sector_map = {
        'fintech':    f"You think this is a product problem. In {country}, fintech is a trust problem — and trust is built through regulated institutions, not better UX. A superior product without a banking license or institutional partnership will not reach adoption at scale.",
        'healthtech': f"You think the friction is distribution. In {country}, the real friction is institutional procurement — hospitals and clinics buy from relationships, not from pitches, and the decision cycle is 9-18 months regardless of how good the product is.",
        'edtech':     f"You think the problem is content quality or delivery. The real problem is willingness-to-pay — edtech in {country} competes with free, and free always wins until you find the narrow segment of users with a measurable career ROI from the skill.",
        'ecommerce':  f"You think the problem is the platform. The real problem is last-mile logistics and returns economics — the unit economics of ecommerce in {country} are structurally negative unless you own or deeply control delivery.",
        'logistics':  f"You think optimization is the value proposition. Logistics clients in {country} buy reliability, not efficiency — a 95% on-time rate beats 20% cheaper every time, and reliability requires assets, not algorithms.",
        'agritech':   f"You think farmers need better tools. Farmers in {country} need predictable income first — any product that does not directly solve the income volatility problem will see adoption spike and collapse with every harvest cycle.",
    }
    if workflow_software and seg == 'b2b':
        counter_thesis = (
            f"You think the hard part is the model quality behind {motion_label}. It is not. "
            f"The hard part is getting {customer_label} in {market_label} to trust the output enough to change a real operating workflow. "
            f"If the software cannot prove lower {problem_label} on live data within one buying cycle, it will be treated as a nice-to-have dashboard and budgeted that way."
        )
    elif sector in _ct_sector_map:
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


def _apply_epistemic_calibration(synthesis: dict, mechanism_analysis: dict) -> dict:
    """
    Post-process synthesis fields to align language with epistemic state.
    Fixes #1 (language softening), #2 (disclosure woven in), #3 (implication bounding),
    #5 (tension_coverage_state surfaced), #6 (market structure ambiguity surfaced).
    Never raises — returns synthesis unchanged on any error.
    """
    if not mechanism_analysis:
        return synthesis

    try:
        epistemic  = mechanism_analysis.get("epistemic_summary", {})
        eq         = epistemic.get("evidence_quality", "moderate")
        disclosure = epistemic.get("recommended_disclosure", "")
        uncertainty = mechanism_analysis.get("uncertainty", 0.0)

        ms           = mechanism_analysis.get("market_structure", {})
        ms_category  = ms.get("category", "")
        ms_confidence = ms.get("confidence", 1.0)
        ms_alt       = ms.get("alternative_category")

        mechanisms  = mechanism_analysis.get("mechanisms", [])
        tension_cov = mechanism_analysis.get("tension_coverage_state", "")
        cons_flags  = mechanism_analysis.get("consistency_flags", [])

        result = dict(synthesis)

        # Fix #2: we weave epistemic_disclosure INTO strategic_interpretation prose;
        # users anchor on the first sentence of synthesis — the disclosure must be there
        if disclosure and eq in ("weak", "insufficient", "moderate"):
            prefix = {
                "insufficient": "Structural signal coverage is too low for mechanism-level conclusions. ",
                "weak":         f"Evidence basis is limited: {disclosure} ",
                "moderate":     f"Evidence basis: {disclosure} ",
            }.get(eq, "")
            result["strategic_interpretation"] = prefix + result.get("strategic_interpretation", "")

        # Fix #6: we surface market structure ambiguity in synthesis explicitly
        if ms_category == "ambiguous" or (ms_alt and ms_confidence < 0.50):
            ambiguity_note = (
                f" Market structure classification is ambiguous "
                f"({ms_alt or 'insufficient signals'}, confidence={ms_confidence:.0%}) — "
                f"strategic conclusions that depend on market structure carry additional uncertainty."
            )
            result["strategic_interpretation"] = result.get("strategic_interpretation", "") + ambiguity_note

        # Fix #3: we bound differentiation_insight by the best mechanism's implication_ceiling;
        # claims in synthesis must not exceed what the mechanism evidence supports
        if mechanisms:
            adv = [m for m in mechanisms if m.get("category") == "advantage_mechanism"]
            best = max(adv or mechanisms, key=lambda m: m.get("effective_weight", 0.0))
            ceiling = best.get("implication_ceiling", "inference")
            ceiling_note = {
                "observation": (
                    " Note: current evidence supports observation of this signal only — "
                    "claims about moat strength or durability are not yet evidentially supportable."
                ),
                "cautious_inference": (
                    " Note: evidence supports cautious inference only — "
                    "treat mechanism claims as directional hypotheses, not confirmed structural advantages."
                ),
            }.get(ceiling, "")
            if ceiling_note:
                result["differentiation_insight"] = result.get("differentiation_insight", "") + ceiling_note

        # Fix #1: we add a calibration prefix to key_driver when uncertainty is elevated
        if uncertainty >= 0.20:
            result["key_driver"] = (
                f"[mechanism_uncertainty={uncertainty:.2f} — treat as directional, not conclusive] "
                + result.get("key_driver", "")
            )

        # Fix #5: we surface tension_coverage_state as a top-level synthesis field;
        # "no tensions detected" must NOT imply the space is tension-free
        if tension_cov:
            result["tension_coverage_state"] = tension_cov

        # Fix #6 (cont.): we add a mechanism_coverage_state field for completeness
        n_mechs = len(mechanisms)
        mode    = mechanism_analysis.get("extraction_mode", "unknown")
        result["mechanism_coverage_state"] = (
            f"{n_mechs} mechanism(s) extracted in {mode} mode."
        )

        # we surface consistency errors in counterpoint — they must not be hidden
        error_flags = [f for f in cons_flags if f.get("severity") == "error"]
        if error_flags:
            flag_desc = "; ".join(
                f.get("description", f.get("type", "")) for f in error_flags[:2]
            )
            result["counterpoint"] = (
                f"Structural inconsistency detected: {flag_desc}. "
                + result.get("counterpoint", "")
            )

        return result

    except Exception:
        return synthesis


def _generate_l4_reasoning(
    sector: str, country: str, regime: str, conf: float,
    sarima_trend: float, idea_features: dict, idea_signal_data: dict,
    shap_dict: dict, tas: float, signal_tier: str,
    a2_comps: list, a4_sentiment: str, idea_text: str, logs: list,
    mechanism_analysis: dict = None,
) -> dict:
    """
    Layer 4 — Strategic Reasoning Engine.
    Tries LLM first; falls back to rich conditional logic.
    Applies epistemic calibration post-processing regardless of path taken.
    """
    bm    = idea_features.get('business_model', 'other')
    diff  = idea_features.get('differentiation_score', 3)
    stage = idea_features.get('stage', 'idea')
    diff_label = {1: 'minimal', 2: 'low', 3: 'moderate', 4: 'strong', 5: 'exceptional'}.get(diff, 'moderate')
    bm_label   = bm.upper()
    seg_label  = idea_features.get('target_segment', 'b2c').upper()

    result = _l4_reasoning_llm(
        sector, country, regime, conf, sarima_trend,
        idea_features, idea_signal_data, shap_dict, tas, signal_tier,
        a2_comps, a4_sentiment, idea_text, diff_label, bm_label, seg_label,
        mechanism_analysis=mechanism_analysis,
    )
    if result:
        logs.append("[L4] Strategic reasoning: LLM path succeeded")
    else:
        logs.append("[L4] LLM reasoning failed — using conditional fallback")

    if not result:
        result = _l4_reasoning_fallback(
            sector, country, regime, conf, sarima_trend,
            idea_features, idea_signal_data, shap_dict, tas, signal_tier, idea_text,
            diff_label, bm_label, seg_label,
        )

    # we guarantee all 7 fields exist — LLM may have returned only 5
    for f in ('what_matters_most', 'counter_thesis'):
        if not result.get(f):
            fb = _l4_reasoning_fallback(
                sector, country, regime, conf, sarima_trend,
                idea_features, idea_signal_data, shap_dict, tas, signal_tier, idea_text,
                diff_label, bm_label, seg_label,
            )
            result[f] = fb.get(f, '')

    # we apply epistemic calibration to both LLM and fallback paths;
    # this is how we guarantee synthesis language aligns with mechanism evidence quality
    result = _apply_epistemic_calibration(result, mechanism_analysis)

    return result


# ═══════════════════════════════════════════════════════════════
# AGENT A0 — Idea Evaluation (5 dimensions, 0-10 per dimension)
# ═══════════════════════════════════════════════════════════════

IDEA_DIMENSIONS = ['problem_clarity', 'solution_fit', 'differentiation', 'business_model', 'scalability']


def agent_a0_evaluate_idea(idea_text: str, sector: str, country: str) -> dict:
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
        response = LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        scores  = {d: max(0, min(10, int(data.get(d, {}).get('score', 5)))) for d in IDEA_DIMENSIONS}
        reasons = {d: data.get(d, {}).get('reason', '') for d in IDEA_DIMENSIONS}
        return {'scores': scores, 'reasons': reasons, 'idea_score': int(sum(scores.values()) / len(scores) * 10)}
    except Exception as llm_err:
        import logging
        logging.getLogger("midan.a0").warning(
            f"[A0] agent_a0_evaluate_idea LLM failed ({type(llm_err).__name__}: {llm_err!r}) "
            f"— falling back to keyword heuristics"
        )

    # we fall back to keyword heuristics when LLM is unavailable
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

    # ── we generate per-dimension explanations ──────────────────────────────────────────────────
    dim_exp = {}

    # problem_clarity: we branch explanation by score tier
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

    # solution_fit: we branch explanation by score tier
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

    # differentiation: we branch explanation by score tier
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

    # business_model: we branch explanation by score tier
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

    # scalability: we branch explanation by score tier
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

    # ── we generate signal explanations for both macro SHAP signals and the idea breakdown ─────────────────────
    sig_exp = {}

    # we write out macro SHAP signal descriptions
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

    # we define idea signal breakdown labels for the explanation layer
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


# we enforce the strict ResponsePayload contract here; every outcome type must be handled
# explicitly; missing data is filled with `unknown`/`null` plus an explanation string —
# never silently dropped, never silently defaulted
#
# Outcome types:
#   1. PRE_ANALYSIS         — no pipeline ran (greeting, casual, partial idea)
#   2. REJECTED             — L0 rejected (raw["invalid_idea"] is True)
#   3. CLARIFICATION_REQUIRED — L1 fail-fast halt (raw["clarification_required"] True)
#   4. L4-DECIDED            — full pipeline ran; decision_state in {GO, CONDITIONAL,
#                              NO_GO, INSUFFICIENT_DATA, HIGH_UNCERTAINTY,
#                              CONFLICTING_SIGNALS}

_BUILDER_LOG = __import__('logging').getLogger("midan.payload_builder")
_TRACE_LOG   = __import__('logging').getLogger("midan.trace")
if not _TRACE_LOG.handlers:
    import logging as _logging_r, pathlib as _pathlib_r
    _trace_path_r = _pathlib_r.Path(__file__).parent.parent / "trace.log"
    _trace_fh_r = _logging_r.FileHandler(str(_trace_path_r), mode="a", encoding="utf-8")
    _trace_fh_r.setFormatter(_logging_r.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _TRACE_LOG.addHandler(_trace_fh_r)
    _TRACE_LOG.setLevel(_logging_r.DEBUG)
    _TRACE_LOG.propagate = False


def _unknown_quality_dimension(basis: str) -> dict:
    return {"tier": "unknown", "basis": basis}


def _unknown_risk_dimension(basis: str) -> dict:
    return {
        "level":                "unknown",
        "drivers":              [],
        "reasoning":            basis,
        "evidence_grounded_in": {"l1_fields_used": [], "l2_fields_used": [], "l3_fields_used": []},
    }


def _empty_decision_quality(basis: str) -> dict:
    return {
        "input_completeness":  _unknown_quality_dimension(basis),
        "signal_agreement":    _unknown_quality_dimension(basis),
        "assumption_density":  _unknown_quality_dimension(basis),
        "overall_uncertainty": "high",
    }


def _empty_risk_decomposition(basis: str) -> dict:
    return {
        "market_risk":    _unknown_risk_dimension(basis),
        "execution_risk": _unknown_risk_dimension(basis),
        "timing_risk":    _unknown_risk_dimension(basis),
    }


def _empty_reasoning_trace() -> dict:
    return {
        "decision_reasoning_steps": [],
        "conflict_ids":             [],
        "top_dim_label":            None,
        "top_dim_level":            None,
        "top_dim_reasoning":        None,
        "signal_references":        {},
    }


def _build_payload_pre_analysis(reply: Optional[str] = None,
                                  type_: Optional[str] = None,
                                  clarification_state: Optional[dict] = None) -> dict:
    """Construct payload for PRE_ANALYSIS turns (greeting, casual, partial)."""
    basis = "no pipeline analysis triggered for this turn (pre-analysis interaction)"
    return {
        "success":            True,
        "schema_version":     RESPONSE_SCHEMA_VERSION,
        "decision_state":     "PRE_ANALYSIS",
        "decision_strength":  {"tier": "uncertain", "basis": basis},
        "decision_quality":   _empty_decision_quality(basis),
        "risk_decomposition": _empty_risk_decomposition(basis),
        "reasoning_trace":    _empty_reasoning_trace(),
        "post_decision_mode": None,
        "post_decision_mode_basis": basis,
        "reply":              reply,
        "type":               type_,
        "clarification_state": clarification_state,
        "projection":         None,
        "quality":            None,
        "data":               None,
        "raw_pipeline_output": None,
    }


def _build_payload_rejected(raw: dict, reply: Optional[str] = None,
                             type_: str = "invalid") -> dict:
    """Construct payload for L0-rejected inputs."""
    rejection_type = raw.get("rejection_type", "unknown_rejection")
    severity       = raw.get("severity", "BROKEN")
    basis          = (
        f"L0 rejected the input ({rejection_type}, severity={severity}); "
        f"no analysis pipeline ran."
    )
    return {
        "success":            True,
        "schema_version":     RESPONSE_SCHEMA_VERSION,
        "decision_state":     "REJECTED",
        "decision_strength":  {"tier": "uncertain", "basis": basis},
        "decision_quality":   _empty_decision_quality(basis),
        "risk_decomposition": _empty_risk_decomposition(basis),
        "reasoning_trace": {
            "decision_reasoning_steps": [],
            "conflict_ids":             [],
            "top_dim_label":            None,
            "top_dim_level":            None,
            "top_dim_reasoning":        None,
            "signal_references": {
                "L0.rejection_type":       rejection_type,
                "L0.severity":             severity,
                "L0.one_line_verdict":     raw.get("one_line_verdict", ""),
                "L0.what_is_missing":      raw.get("what_is_missing", ""),
            },
        },
        "post_decision_mode": None,
        "post_decision_mode_basis": basis,
        "reply":              reply or raw.get("one_line_verdict"),
        "type":               type_,
        "clarification_state": None,
        "projection":         None,
        "quality":            None,
        "data":               raw,
        "raw_pipeline_output": None,
    }


def _build_payload_clarification_required(raw: dict, reply: Optional[str] = None,
                                            type_: str = "clarifying",
                                            clarification_state: Optional[dict] = None) -> dict:
    """Construct payload for L1 fail-fast halts."""
    rejection_type = raw.get("rejection_type", "l1_insufficient_confidence")
    basis          = (
        f"L1 fail-fast halt ({rejection_type}); decision engine did not run because "
        f"required fields were UNKNOWN or aggregate confidence below threshold."
    )
    clar = raw.get("clarification") or {}
    return {
        "success":            True,
        "schema_version":     RESPONSE_SCHEMA_VERSION,
        "decision_state":     "CLARIFICATION_REQUIRED",
        "decision_strength":  {"tier": "uncertain", "basis": basis},
        "decision_quality":   _empty_decision_quality(basis),
        "risk_decomposition": _empty_risk_decomposition(basis),
        "reasoning_trace": {
            "decision_reasoning_steps": [],
            "conflict_ids":             [],
            "top_dim_label":            None,
            "top_dim_level":            None,
            "top_dim_reasoning":        None,
            "signal_references": {
                "L1.rejection_type":         rejection_type,
                "L1.aggregate_confidence":   raw.get("l1_aggregate_confidence"),
                "L1.unknown_required":       raw.get("l1_unknown_required") or [],
                "L1.consistency":            raw.get("l1_consistency") or {},
                "L1.clarification_questions": clar.get("questions") or [],
            },
        },
        "post_decision_mode": None,
        "post_decision_mode_basis": basis,
        "reply":              reply or raw.get("one_line_verdict"),
        "type":               type_,
        "clarification_state": clarification_state,
        "projection":         None,
        "quality":            None,
        "data":               raw,
        "raw_pipeline_output": None,
    }


def _build_payload_decided(raw: dict, reply: Optional[str] = None,
                            type_: Optional[str] = None,
                            projection: Optional[dict] = None,
                            quality: Optional[dict] = None,
                            extras_context: Optional[dict] = None) -> dict:
    """Construct payload for L4-decided outcomes (success or non-GO state)."""
    l4 = raw.get("l4_decision") or {}
    if not l4:
        raise SchemaViolationError(
            "Cannot build decided payload: raw output has no 'l4_decision' envelope. "
            "This indicates an inconsistent pipeline output."
        )

    decision_state    = l4.get("decision_state", "INSUFFICIENT_DATA")
    decision_strength = l4.get("decision_strength") or {
        "tier":  "uncertain",
        "basis": "decision_strength missing from L4 output",
    }
    quality_block     = l4.get("decision_quality") or {}
    risk_block        = l4.get("risk_decomposition") or {}
    conflicts         = l4.get("conflicting_signals") or []
    reasoning_steps   = l4.get("decision_reasoning") or []

    # we determine post-decision mode using the existing helper for behavioral consistency
    route = _post_decision_route({
        "decision_state": decision_state,
        "l4_decision":    l4,
    })
    post_mode       = route.get("mode") if route.get("mode") in ('STANDARD_ADVISOR', 'RESOLVING_CONFLICT', 'ADVISORY_ONLY', 'RE_CLARIFY') else None
    post_mode_basis = route.get("reason", f"post_decision_mode derived for state={decision_state}")

    # we compute the top binding risk dimension for the reasoning trace
    top_dim_name, top_dim_block = _l4_top_risk_dim(l4)
    top_dim_label = (top_dim_name or '').replace('_', ' ') if top_dim_name else None
    top_dim_level = (top_dim_block or {}).get('level') if top_dim_block else None
    top_dim_reasoning = (top_dim_block or {}).get('reasoning') if top_dim_block else None

    # we ensure all three risk dimensions are present even if L4 omitted any
    rd = {}
    for dim in ('market_risk', 'execution_risk', 'timing_risk'):
        if dim in risk_block and isinstance(risk_block[dim], dict):
            rd[dim] = {
                "level":     risk_block[dim].get('level', 'unknown'),
                "drivers":   risk_block[dim].get('drivers', []) or [],
                "reasoning": risk_block[dim].get('reasoning', ''),
                "evidence_grounded_in": risk_block[dim].get('evidence_grounded_in', {}) or
                                        {"l1_fields_used": [], "l2_fields_used": [], "l3_fields_used": []},
            }
        else:
            rd[dim] = _unknown_risk_dimension(f"{dim} not present in L4 output")

    # we ensure all four decision quality sub-fields are present
    dq = {
        "input_completeness":  quality_block.get('input_completeness') or _unknown_quality_dimension("input_completeness missing from L4 output"),
        "signal_agreement":    quality_block.get('signal_agreement')   or _unknown_quality_dimension("signal_agreement missing from L4 output"),
        "assumption_density":  quality_block.get('assumption_density') or _unknown_quality_dimension("assumption_density missing from L4 output"),
        "overall_uncertainty": quality_block.get('overall_uncertainty') or "high",
    }

    # we assemble the reasoning trace from L4 reasoning steps and conflict ids
    trace = {
        "decision_reasoning_steps": [
            {
                "step":       s.get('step', ''),
                "rule_id":    s.get('rule_id', ''),
                "evidence":   s.get('evidence', []),
                "conclusion": s.get('conclusion', ''),
            }
            for s in reasoning_steps if isinstance(s, dict)
        ],
        "conflict_ids":      [c.get('conflict_id') for c in conflicts if isinstance(c, dict) and c.get('conflict_id')],
        "top_dim_label":     top_dim_label,
        "top_dim_level":     top_dim_level,
        "top_dim_reasoning": top_dim_reasoning,
        "signal_references": {
            "L2.regime":               raw.get("regime"),
            "L2.fcm_top_cluster":      (raw.get("fcm_membership") or {}).get("top_cluster"),
            "L3.differentiation_verdict": ((raw.get("l3_reasoning") or {}).get("differentiation") or {}).get("verdict"),
            "L3.competition_pressure":    ((raw.get("l3_reasoning") or {}).get("competition")    or {}).get("competitive_pressure"),
            "L4.legacy_tas_value":     (l4.get("legacy_tas_score") or {}).get("value"),
        },
    }

    return {
        "success":            True,
        "schema_version":     RESPONSE_SCHEMA_VERSION,
        "decision_state":     decision_state,
        "decision_strength":  {
            "tier":  decision_strength.get("tier", "uncertain"),
            "basis": decision_strength.get("basis", "no basis provided by L4"),
        },
        "decision_quality":   dq,
        "risk_decomposition": rd,
        "reasoning_trace":    trace,
        "post_decision_mode": post_mode,
        "post_decision_mode_basis": post_mode_basis,
        "reply":              reply,
        "type":               type_,
        "clarification_state": (extras_context or {}).get("clarification_state"),
        "projection":         projection,
        "quality":            quality,
        # `data` carries the full raw process_idea output for backward-compat consumers
        # that read fields like `data.idea`, `data.tas_score`, etc.;
        # `raw_pipeline_output` mirrors it for forward-compat clarity — both point to
        # the same dict, no transformation, no drift
        "data":                raw,
        "raw_pipeline_output": raw,
    }


def build_response_payload(
    raw: Optional[dict] = None,
    *,
    outcome: Literal['pre_analysis', 'rejected', 'clarification_required', 'decided'],
    reply: Optional[str] = None,
    type_: Optional[str] = None,
    projection: Optional[dict] = None,
    quality: Optional[dict] = None,
    clarification_state: Optional[dict] = None,
    extras_context: Optional[dict] = None,
    is_fallback: Optional[bool] = None,
) -> ResponsePayload:
    """
    Single mapper from raw pipeline outputs to the strict ResponsePayload schema.
    Caller specifies outcome explicitly — no auto-detection magic.

    Raises SchemaViolationError if the chosen outcome is incompatible with the
    raw payload, or pydantic.ValidationError if the constructed dict does not
    satisfy the schema. Both are propagated; this function never silently
    auto-corrects.
    """
    if outcome == 'pre_analysis':
        body = _build_payload_pre_analysis(
            reply=reply, type_=type_, clarification_state=clarification_state,
        )
    elif outcome == 'rejected':
        if not raw or not raw.get('invalid_idea'):
            raise SchemaViolationError(
                "build_response_payload(outcome='rejected') requires raw with invalid_idea=True"
            )
        body = _build_payload_rejected(raw, reply=reply, type_=type_ or 'invalid')
    elif outcome == 'clarification_required':
        if not raw or not raw.get('clarification_required'):
            raise SchemaViolationError(
                "build_response_payload(outcome='clarification_required') requires "
                "raw with clarification_required=True"
            )
        body = _build_payload_clarification_required(
            raw, reply=reply, type_=type_ or 'clarifying',
            clarification_state=clarification_state,
        )
    elif outcome == 'decided':
        if not raw:
            raise SchemaViolationError(
                "build_response_payload(outcome='decided') requires non-empty raw payload"
            )
        body = _build_payload_decided(
            raw, reply=reply, type_=type_, projection=projection, quality=quality,
            extras_context=extras_context,
        )
    else:
        raise SchemaViolationError(f"Unknown outcome type: {outcome!r}")

    # we enforce strict validation — pydantic raises ValidationError on any contract violation
    if is_fallback is not None:
        body['is_fallback'] = is_fallback
    return ResponsePayload(**body)


# NOTE: we moved _l4_top_risk_dim to midan.l4_decision (its semantic owner) —
# it's imported at the top of this module


def _l4_summary_for_chat(l4_decision: dict) -> str:
    """One-line factual summary of the L4 decision for chat consumption."""
    if not l4_decision:
        return "no L4 decision in context"
    state    = l4_decision.get('decision_state', 'UNKNOWN')
    strength = (l4_decision.get('decision_strength') or {}).get('tier', 'unknown')
    return f"decision_state={state}, decision_strength={strength}"


def _chat_fallback(req: ChatRequest) -> str:
    """
    Heuristic chat reply, L4-grounded. Reads only the L4 decision envelope
    from req.context (plus minimal L1 context for grounding labels). Never
    references TAS or signal_tier. Branches on _post_decision_route().
    """
    ctx        = req.context
    user_turns = [m for m in req.messages if m.role == "user"]
    turn_n     = len(user_turns)
    last_msg   = user_turns[-1].content.lower() if user_turns else ""
    sector     = ctx.get("sector", "")
    country    = ctx.get("country", "")
    idea_feat  = ctx.get("idea_features", {}) or {}
    seg        = (idea_feat.get("target_segment") or "").upper()
    bm         = idea_feat.get("business_model") or ""
    stage      = idea_feat.get("stage") or ""
    idea_text  = ctx.get("idea", "") or " ".join(
        m.content.strip() for m in req.messages if m.role == "user"
    ).strip()
    grounding  = _extract_idea_grounding(idea_text, sector, idea_feat, country)
    customer_label = grounding['customer_label']

    # ── No analysis context yet — pre-decision mode ──────────────────────────────
    # _chat_pre_analysis_reply() already handles everything via Gemini and returns
    # its own fallback string on failure. This branch should not be reached for
    # pre-analysis turns. If it is, return the same error string.
    if not ctx.get('decision_state') and not ctx.get('tas_score'):
        _TRACE_LOG.warning("[TRACE][_chat_fallback] pre-analysis branch reached unexpectedly — returning error fallback")
        return _PRE_ANALYSIS_ERROR_FALLBACK

    # ── Post-decision: we route by L4 mode, NOT by legacy fields ───────────────
    route        = _post_decision_route(ctx)
    mode         = route['mode']
    l4           = ctx.get('l4_decision') or {}
    decision_state = ctx.get('decision_state') or l4.get('decision_state') or ''
    strength_tier = ((l4.get('decision_strength') or {}).get('tier')
                     or (ctx.get('decision_strength') or {}).get('tier') or '')

    # we added a hard guard here: a chat-mode reply must never expose internal placeholders
    # like UNKNOWN or `unknown strength`; if we somehow reached the post-decision branch
    # without a concrete decision_state or strength tier, we emit a consultative clarification
    # instead of a broken opener and log the routing miss so it can be tracked down
    _have_state    = decision_state in {'GO', 'CONDITIONAL', 'NO_GO',
                                        'INSUFFICIENT_DATA', 'HIGH_UNCERTAINTY',
                                        'CONFLICTING_SIGNALS'}
    _have_strength = strength_tier in {'strong', 'moderate', 'weak', 'uncertain'}
    if mode == 'STANDARD_ADVISOR' and not (_have_state and _have_strength):
        __import__('logging').getLogger("midan.routing").warning(
            "[ROUTING] _chat_fallback STANDARD_ADVISOR with incomplete L4 "
            "context (decision_state=%r, strength=%r) — emitting consultative "
            "clarification instead of UNKNOWN opener.",
            decision_state or None, strength_tier or None,
        )
        return (
            "I don't have enough clarity yet to evaluate this properly. "
            "Walk me through what your solution actually does and who it's for, "
            "and I'll run a real read on it."
        )

    # ── MODE 1: RESOLVING_CONFLICT — we must surface the conflict + ask for resolution ─
    if mode == 'RESOLVING_CONFLICT':
        unresolved = route.get('unresolved_conflicts', [])
        if not unresolved:
            return ("The last read flagged CONFLICTING_SIGNALS but no unresolved conflict "
                    "came through with the context. Worth re-running the analysis so we "
                    "can see exactly what's contradicting itself.")
        primary = unresolved[0]
        conflict_id     = primary.get('conflict_id', 'unspecified')
        explanation     = primary.get('explanation', '')
        resolution_path = primary.get('resolution_path', '')
        return (
            f"There's a real contradiction in the last read — I cannot continue as a normal "
            f"advisor on top of it. The conflict is `{conflict_id}` "
            f"({primary.get('severity', 'high')} severity). {explanation} "
            f"My suggested resolution: {resolution_path} "
            f"Which side of that do you want to anchor to? Once you decide, I'll re-run."
        )

    # ── MODE 2: ADVISORY_ONLY — we prefix every reply with a HIGH_UNCERTAINTY caveat ────────────
    if mode == 'ADVISORY_ONLY':
        basis = route.get('uncertainty_basis', {})
        ic    = basis.get('input_completeness') or 'input completeness unclear'
        sa    = basis.get('signal_agreement')   or 'signal agreement unclear'
        return (
            f"I'd treat anything I say next as advisory only — the last read came back "
            f"with HIGH_UNCERTAINTY. The reasoning behind that: {ic} {sa} "
            f"What do you want to pressure-test, knowing the read isn't on solid ground yet?"
        )

    # ── MODE 3: RE_CLARIFY — last attempt halted on missing fields, we re-prompt ──────────
    if mode == 'RE_CLARIFY':
        idea_feat = ctx.get('idea_features') or {}
        bm        = idea_feat.get('business_model') or ctx.get('business_model', '')
        seg       = idea_feat.get('target_segment') or ctx.get('target_segment', '')
        stage_val = idea_feat.get('stage')           or ctx.get('stage', '')
        if bm and seg and stage_val:
            return (
                f"I currently have you as {bm} targeting {seg} customers at {stage_val} stage — "
                f"correct me if anything changed. What would you like to dig into?"
            )
        return (
            "Before I can continue, I need to confirm a few things — "
            "your business model (subscription / marketplace / SaaS / commission / service), "
            "target segment (B2B / B2C / B2G), and stage (idea / validation / MVP / growth). "
            "Once those are clear, I can give you a grounded read."
        )

    # ── MODE 4: STANDARD_ADVISOR — we build L4-grounded interpretive replies ─────────
    top_dim_name, top_dim_block = _l4_top_risk_dim(l4)
    top_dim_label = (top_dim_name or 'risk').replace('_', ' ')
    top_dim_level = (top_dim_block or {}).get('level', 'unknown')
    top_dim_reasoning = (top_dim_block or {}).get('reasoning', '')

    # we build a list of conflict ids (any severity) to ground responses
    conflicts_summary = ", ".join(
        c.get('conflict_id', '?') for c in (l4.get('conflicting_signals') or [])
    ) or 'no detected conflicts'

    # we pull the L3 differentiation verdict + competition pressure if they were surfaced
    l3 = ctx.get('l3_reasoning') or {}
    diff_verdict = (l3.get('differentiation') or {}).get('verdict', 'unknown')
    comp_pressure = (l3.get('competition') or {}).get('competitive_pressure', 'unknown')

    # we build a consultant-tone opener that names the decision and its strength inline,
    # in plain English, and surfaces any contradictions L4 flagged; this replaces the
    # old `[decision_state=..., decision_strength=...]` debug-style stamp
    _conflict_ids = [
        c.get('conflict_id', '?')
        for c in (l4.get('conflicting_signals') or [])
    ]
    # by the time we reach this opener, the guard above has already verified
    # decision_state is one of GO / CONDITIONAL / NO_GO; the dict.get fallback
    # is intentionally a defensive no-op rather than a UNKNOWN-printing branch
    _state_phrase = {
        'GO':          f"Reading this as a GO at {strength_tier} strength",
        'CONDITIONAL': f"This reads as CONDITIONAL — strength is {strength_tier}",
        'NO_GO':       f"This is a NO_GO at {strength_tier} strength",
    }.get(decision_state, f"Reading this at {strength_tier} strength")
    if _conflict_ids:
        _contradiction_clause = (
            f", though L4 still flags contradictions ({', '.join(_conflict_ids)})"
        )
    else:
        _contradiction_clause = ""
    decision_caveat = f"{_state_phrase}{_contradiction_clause}."

    # we route by topic keywords — same categories as before, but every branch
    # references L4 fields explicitly rather than TAS/tier numerics
    if any(w in last_msg for w in ["pivot", "change direction", "switch", "alternative", "different idea"]):
        return (
            f"{decision_caveat} The macro regime ({ctx.get('regime', 'unknown')}) does not move when you pivot. "
            f"The top risk dimension is {top_dim_label} at level {top_dim_level} — "
            f"the only pivot worth running directly attacks that dimension. "
            f"Reasoning: {top_dim_reasoning[:160]} "
            f"Which exact driver of {top_dim_label} would your pivot remove?"
        )

    if any(w in last_msg for w in ["compet", "rival", "who else", "already exist", "crowded", "saturated"]):
        return (
            f"{decision_caveat} Competition pressure read by L3 is {comp_pressure}; "
            f"differentiation verdict is {diff_verdict}. "
            f"At verdict={diff_verdict}, features alone won't protect the position — the moat has to be "
            f"structural (switching costs, regulatory access, or network density). "
            f"Which structural moat do you have evidence for, beyond features?"
        )

    if any(w in last_msg for w in ["fund", "invest", "raise", "capital", "vc", "angel", "pitch", "accelerator"]):
        if decision_state == 'GO':
            return (
                f"{decision_caveat} At GO with strength={strength_tier}, you're in a defensible position "
                f"to raise — but investors will probe the {top_dim_label} dimension hard. "
                f"Top driver per L4: {top_dim_reasoning[:140]} "
                f"What evidence do you have addressing {top_dim_label} this month?"
            )
        if decision_state == 'CONDITIONAL':
            return (
                f"{decision_caveat} CONDITIONAL means at least one risk dimension is unmitigated. "
                f"Top dimension: {top_dim_label} ({top_dim_level}). "
                f"Institutional capital is unlikely to close until that's addressed. "
                f"What's the cheapest experiment that resolves {top_dim_label} in 30 days?"
            )
        return (
            f"{decision_caveat} At {decision_state} the L4 read does not support an institutional raise. "
            f"Bootstrap or wedge with paid revenue first — paying customers rewrite the conversation. "
            f"What's the path to first paid revenue without external capital?"
        )

    if any(w in last_msg for w in ["risk", "danger", "threat", "fail", "worry", "concern"]):
        return (
            f"{decision_caveat} L4 risk decomposition surfaces {top_dim_label} as the binding dimension "
            f"({top_dim_level}). Reasoning: {top_dim_reasoning[:200]} "
            f"Conflicts surfaced: {conflicts_summary}. "
            f"Which of those drivers can you neutralize with evidence in the next 4 weeks?"
        )

    if any(w in last_msg for w in ["next", "first step", "what do i do", "how do i start", "roadmap", "plan"]):
        if decision_state == 'GO':
            return (
                f"{decision_caveat} Priority: (1) lock in evidence on {top_dim_label} — that's the dimension "
                f"L4 still flags. (2) Ship the smallest possible test to "
                f"{(seg or 'target').lower()} customers in 4 weeks. (3) Iterate on the conflict surface, not the product. "
                f"Which of those is gated on a decision you haven't made yet?"
            )
        if decision_state == 'CONDITIONAL':
            return (
                f"{decision_caveat} Don't build yet. (1) Run discovery calls focused on the {top_dim_label} "
                f"hypothesis. (2) Find the single use case with highest willingness-to-pay. "
                f"(3) Resolve the unresolved conflicts ({conflicts_summary}) before code. "
                f"What's the first call on the calendar?"
            )
        return (
            f"{decision_caveat} The L4 read is not in a buildable state. "
            f"Resolve the {top_dim_label} dimension first. {top_dim_reasoning[:180]} "
            f"What signal would change that risk level?"
        )

    if any(w in last_msg for w in ["pric", "monetize", "revenue model", "charge", "subscription", "fee"]):
        ue = (l3.get('unit_economics') or {})
        rpu = (ue.get('revenue_per_user_proxy') or {}).get('tier', 'unknown')
        cac = (ue.get('cac_proxy') or {}).get('tier', 'unknown')
        return (
            f"{decision_caveat} L3 unit economics proxies: CAC={cac}, RPU={rpu} "
            f"(qualitative — not factual estimates). "
            f"For a {bm or 'this'} model targeting {seg or 'this segment'}, the pricing question is "
            f"whether RPU dominates CAC over the customer lifetime. "
            f"What's the value you can guarantee and measure in the first 30 days?"
        )

    if any(w in last_msg for w in ["team", "hire", "co-founder", "talent", "people"]):
        return (
            f"{decision_caveat} The skill gap that matters most is whoever owns the {top_dim_label} dimension — "
            f"that's where L4 says the binding risk lives. "
            f"At {stage or 'current'} stage, contractors can prove the role before full-time commitment. "
            f"Who currently owns {top_dim_label} on the team?"
        )

    if any(w in last_msg for w in ["thank", "thanks", "great", "helpful", "good", "awesome"]):
        return (
            f"{decision_caveat} Top live question: the {top_dim_label} dimension at {top_dim_level}. "
            f"What else do you want to pressure-test?"
        )

    # Default: we ground the reply in the top risk + customer label; no TAS references
    return (
        f"{decision_caveat} Reading on {customer_label} in {country or 'this market'}: "
        f"the binding dimension is {top_dim_label} ({top_dim_level}). "
        f"L3 differentiation verdict={diff_verdict}, competition pressure={comp_pressure}. "
        f"What specifically do you want to pressure-test about that?"
    )


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text.strip()) if p.strip()]
    return parts[0] if parts else text.strip()


def _sentence_tail(text: str) -> str:
    s = _first_sentence(text).strip()
    if not s:
        return ""
    s = s.rstrip(".!? ")
    return s[0].lower() + s[1:] if len(s) > 1 else s.lower()


def _assess_projection_input(idea_text: str) -> dict:
    text = (idea_text or "").strip()
    lowered = text.lower()
    wc = len(lowered.split())
    comps = _extract_components(text)
    parsed_sector, parsed_country, sector_found, country_found = agent_a1_parse(text)
    features = _heuristic_idea_features(text, parsed_sector if sector_found else 'other')
    grounding = _extract_idea_grounding(text, parsed_sector if sector_found else 'other', features, parsed_country if country_found else '')

    tokens = [tok for tok in re.findall(r"[a-zA-Z0-9']+", lowered) if len(tok) > 1]
    unique_tokens = set(tokens)
    generic_only = bool(unique_tokens) and unique_tokens.issubset({
        tok for tok in _GENERIC_IDEA_TOKENS if " " not in tok
    })

    explicit_customer_hint = any(
        c in lowered for c in _MARKET_CUSTOMER if c not in _GENERIC_CUSTOMER_HINTS
    )
    explicit_market_hint = country_found or any(
        g in lowered for g in _MARKET_GEO if g not in {'global', 'worldwide', 'international'}
    )
    explicit_mechanism_hint = any(sig in lowered for sig in _EXPLICIT_MECHANISM_SIGNALS) or any(
        sig in lowered for sig in _SOLUTION_SIGNALS if sig not in _GENERIC_MECHANISM_HINTS
    )
    explicit_problem_hint = comps['has_problem'] or grounding['problem_label'] != 'the stated operational problem'

    if generic_only:
        has_customer = explicit_customer_hint
        has_market = explicit_market_hint
        has_mechanism = explicit_mechanism_hint
        has_problem = explicit_problem_hint
    else:
        has_customer = explicit_customer_hint or grounding['customer_label'] not in {'end users', 'business operators'}
        has_market = explicit_market_hint or grounding['market_label'] not in {'the market', ''}
        has_mechanism = explicit_mechanism_hint or (
            features.get('business_model', 'other') != 'other' and wc >= 6
        )
        has_problem = explicit_problem_hint

    if wc <= 3:
        if not explicit_mechanism_hint:
            has_mechanism = False
        if not explicit_customer_hint:
            has_customer = False
        if not explicit_market_hint:
            has_market = False

    quality_score = 0.0
    quality_score += min(wc / 18.0, 1.0) * 0.18
    quality_score += 0.28 if has_problem else 0.0
    quality_score += 0.25 if has_mechanism else 0.0
    quality_score += 0.16 if has_customer else 0.0
    quality_score += 0.13 if has_market else 0.0

    if generic_only:
        quality_score = min(quality_score, 0.28)
    if wc <= 4:
        quality_score = min(quality_score, 0.34)

    missing = []
    if not has_mechanism:
        missing.append("product mechanism")
    if not has_customer:
        missing.append("user definition")
    if not has_problem:
        missing.append("problem definition")
    if not has_market:
        missing.append("market context")

    if quality_score >= 0.72 and has_problem and has_mechanism and has_customer and wc >= 10:
        quality_label = "analysis-ready"
    elif quality_score >= 0.48:
        quality_label = "forming"
    else:
        quality_label = "insufficient"

    analysis_allowed = quality_label == "analysis-ready"
    refusal = None if analysis_allowed else "Not enough structure to form a market read."
    guidance = []
    if not has_mechanism:
        guidance.append("What is the product exactly?")
    if not has_customer:
        guidance.append("Who is the user?")
    if not has_problem:
        guidance.append("What problem is being solved?")
    if not has_market:
        guidance.append("Which market or geography matters first?")
    if not guidance:
        guidance = [
            "Describe the product in one clear sentence.",
            "Name the user and the operational pain.",
            "State the mechanism, not the category."
        ]

    return {
        "word_count": wc,
        "quality_score": round(float(quality_score), 3),
        "quality_label": quality_label,
        "analysis_allowed": analysis_allowed,
        "missing": missing,
        "refusal": refusal,
        "guidance": guidance[:4],
        "components": {
            "problem": has_problem,
            "mechanism": has_mechanism,
            "customer": has_customer,
            "market": has_market,
        },
        "grounding": grounding,
        "features_preview": features,
        "sector_preview": parsed_sector if sector_found else "",
        "country_preview": parsed_country if country_found else "",
    }


def _build_projection_payload(idea_text: str, data: dict, quality: dict) -> dict:
    idea_feat = data.get("idea_features", {})
    grounding = _extract_idea_grounding(idea_text, data.get("sector", ""), idea_feat, data.get("country", ""))
    weakness = _sentence_tail(data.get("main_risk", "")) or _sentence_tail(data.get("counter_thesis", ""))
    if weakness:
        weakness = f"The idea is weak in its current form because {weakness}."
    else:
        weakness = "The idea is weak in its current form because the market read is missing one decisive proof point."

    recommendation = data.get("decision_badge", "") or data.get("quadrant", "")
    recommendation_body = data.get("action", "")
    direct_thesis = _first_sentence(data.get("strategic_interpretation", "")) or (
        f"This reads as {data.get('signal_tier', 'Mixed')} for {grounding['context_label']} in {grounding['market_label']}."
    )

    return {
        "quality": quality,
        "headline": f"{grounding['context_label'].title()} · {grounding['market_label']}",
        "context_label": grounding["context_label"],
        "customer_label": grounding["customer_label"],
        "problem_label": grounding["problem_label"],
        "motion_label": grounding["motion_label"],
        "direct_thesis": direct_thesis,
        "direct_weakness": weakness,
        "decision_label": recommendation,
        "decision_body": recommendation_body,
        "focus_line": data.get("what_matters_most", "") or data.get("key_driver", ""),
        "counter_thesis": data.get("counter_thesis", ""),
        "signal_summary": {
            "sector": data.get("sector", ""),
            "country": data.get("country", ""),
            "regime": data.get("regime", ""),
            "signal_tier": data.get("signal_tier", ""),
            "tas_score": data.get("tas_score", 0),
            "confidence": data.get("confidence", 0),
            "dominant_risk": data.get("dominant_risk", ""),
        },
    }


def _probe_answer_fallback(question: str, context: dict) -> str:
    q = (question or "").lower().strip()
    idea = context.get("idea", "")
    idea_feat = context.get("idea_features", {})
    grounding = _extract_idea_grounding(idea, context.get("sector", ""), idea_feat, context.get("country", ""))
    main_risk = context.get("main_risk", "")
    direct_weakness = _sentence_tail(main_risk) or _sentence_tail(context.get("counter_thesis", ""))
    if direct_weakness:
        direct_weakness = f"The idea is weak because {direct_weakness}."
    else:
        direct_weakness = "The idea is weak because the model still has not earned a decisive proof point."

    if any(p in q for p in ["why is the idea bad", "why is it bad", "why bad", "why weak"]):
        support = context.get("counter_thesis", "") or context.get("counterpoint", "")
        return f"{direct_weakness} {support}".strip()
    if "risk" in q or "what is bad" in q:
        return f"The biggest risk is {_first_sentence(main_risk) or context.get('dominant_risk', 'execution risk')}."
    if "missing" in q:
        focus = context.get("what_matters_most", "") or context.get("key_driver", "")
        return f"What is missing is decisive proof that {focus.lower() if focus else 'the core assumption actually holds in the market'}."
    if "next" in q or "what do i do" in q or "what should i do" in q:
        return _first_sentence(context.get("action", "")) or "Do not expand scope until the core assumption is proven."
    if "customer" in q or "user" in q:
        return f"The first user to win is {grounding['customer_label']}, because they feel {grounding['problem_label']} directly enough to change behavior."
    return (
        f"{direct_weakness} The live question is whether {grounding['customer_label']} in "
        f"{grounding['market_label']} will act on {grounding['motion_label']} fast enough to make the product budget-critical."
    )


def _answer_projection_probe(question: str, context: dict) -> str:
    if not question or not question.strip():
        return "State the exact challenge you want pressure-tested."

    idea = context.get("idea", "")
    idea_feat = context.get("idea_features", {})
    grounding = _extract_idea_grounding(idea, context.get("sector", ""), idea_feat, context.get("country", ""))
    prompt = dedent(f"""
            You are MIDAN. This is not a chatbot exchange. Answer like a decision system.

            Founder idea:
            "{idea}"

            Market read already computed:
            - Context: {grounding['context_label']} for {grounding['customer_label']} in {grounding['market_label']}
            - Regime: {context.get('regime', '')}
            - Signal: {context.get('signal_tier', '')} ({context.get('tas_score', 0)}/100)
            - Dominant risk: {context.get('dominant_risk', '')}
            - Strategic read: {context.get('strategic_interpretation', '')}
            - Main risk: {context.get('main_risk', '')}
            - Counter-thesis: {context.get('counter_thesis', '')}
            - Decision: {context.get('decision_badge', '')}

            User probe:
            "{question}"

            HARD RULES:
            - Answer the question directly in sentence one.
            - No greeting, no filler, no roleplay.
            - No follow-up question unless the user explicitly asked for one.
            - Be blunt and specific to the idea.
            - Max 3 sentences.
        """).strip()
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        if answer:
            return answer
    except Exception as llm_err:
        __import__('logging').getLogger("midan.probe").warning(
            "[PROBE] LLM call failed (%s: %r) — falling back to heuristic",
            type(llm_err).__name__, llm_err,
        )

    return _probe_answer_fallback(question, context)


def _generate_operator_reply(data: dict, idea_text: str) -> str:
    """
    Strategic operator response post-analysis. L4-grounded.
    Pattern: INFER → ANALYZE → CHALLENGE → ASK (1 question only).

    Reads ONLY the L4 decision envelope (decision_state, decision_strength,
    risk_decomposition, conflicting_signals, l3_reasoning) — never references
    TAS or signal_tier numeric framing.

    For non-standard decision states (CONFLICTING_SIGNALS, HIGH_UNCERTAINTY,
    INSUFFICIENT_DATA), the function refuses to act as a confident advisor and
    instead surfaces the controlled state with a resolution-asking question.
    """
    # ── we pull the L4 envelope first — that's the source of truth ──────────────
    l4               = data.get("l4_decision") or {}
    decision_state   = (l4.get("decision_state")
                        or data.get("decision_state") or "")
    strength_tier    = ((l4.get("decision_strength") or {}).get("tier")
                        or (data.get("decision_strength") or {}).get("tier")
                        or "")

    # we added a hard guard mirroring _chat_fallback: we never let an UNKNOWN or empty
    # decision state reach the user; _generate_operator_reply is invoked post-`decided`
    # so this should be unreachable in practice — if it fires, we log it
    _valid_states  = {'GO', 'CONDITIONAL', 'NO_GO',
                      'INSUFFICIENT_DATA', 'HIGH_UNCERTAINTY', 'CONFLICTING_SIGNALS'}
    _valid_tiers   = {'strong', 'moderate', 'weak', 'uncertain'}
    if decision_state not in _valid_states or strength_tier not in _valid_tiers:
        _OP_REPLY_LOG.warning(
            "[ROUTING] _generate_operator_reply called with incomplete L4 "
            "context (decision_state=%r, strength=%r) — emitting consultative "
            "clarification instead of UNKNOWN opener.",
            decision_state or None, strength_tier or None,
        )
        return (
            "I don't have enough clarity yet to evaluate this properly. "
            "Walk me through what your solution actually does and who it's for, "
            "and I'll run a real read on it."
        )
    risks            = l4.get("risk_decomposition", {}) or {}
    conflicts        = l4.get("conflicting_signals", []) or []
    decision_quality = l4.get("decision_quality", {}) or {}

    top_dim_name, top_dim_block = _l4_top_risk_dim(l4)
    top_dim_label     = (top_dim_name or 'risk').replace('_', ' ')
    top_dim_level     = (top_dim_block or {}).get('level', 'unknown')
    top_dim_reasoning = (top_dim_block or {}).get('reasoning', '')

    l3                = data.get("l3_reasoning") or {}
    diff_verdict      = (l3.get("differentiation") or {}).get("verdict", "unknown")
    competition_press = (l3.get("competition")    or {}).get("competitive_pressure", "unknown")

    # we use L1 fields for grounding labels only — not for decision logic
    sector    = data.get("sector", "")
    country   = data.get("country", "")
    idea_feat = data.get("idea_features", {}) or {}
    bm        = idea_feat.get("business_model", "")
    seg       = (idea_feat.get("target_segment") or "").upper()
    stage     = idea_feat.get("stage", "")
    grounding = _extract_idea_grounding(idea_text, sector, idea_feat, country)
    customer_label = grounding['customer_label']
    context_label  = grounding['context_label']
    market_label   = grounding['market_label']

    # ── Non-standard states: we refuse the confident-advisor pattern here ────────
    if decision_state == 'CONFLICTING_SIGNALS':
        unresolved = [c for c in conflicts
                       if c.get('severity') == 'high' and c.get('resolution_required')]
        if unresolved:
            primary = unresolved[0]
            return (
                f"This came back as CONFLICTING_SIGNALS at {strength_tier} strength, and I'm "
                f"not going to play confident advisor on top of that. The analysis halted on "
                f"conflict `{primary.get('conflict_id')}`. {primary.get('explanation', '')} "
                f"Resolution path: {primary.get('resolution_path', '')} "
                f"Which side of that conflict do you want to anchor to before I re-run?"
            )
    if decision_state == 'HIGH_UNCERTAINTY':
        ic = (decision_quality.get('input_completeness') or {}).get('basis', 'inputs unclear')
        sa = (decision_quality.get('signal_agreement')   or {}).get('basis', 'agreement unclear')
        return (
            f"This came back as HIGH_UNCERTAINTY at {strength_tier} strength, so I'd treat "
            f"anything I say next as advisory only. The reasoning: {ic} {sa} "
            f"What's the single most material missing input you can supply? With that, "
            f"I can re-run on firmer ground."
        )
    if decision_state == 'INSUFFICIENT_DATA':
        return (
            "The last pass came back INSUFFICIENT_DATA — required L1/L3 fields weren't "
            "available. Walk me through your business model, who you're selling to, and "
            "what stage you're at, and I'll re-run on solid ground."
        )

    # ── Standard states (GO / CONDITIONAL / NO_GO): we build an L4-grounded operator reply
    try:
        risks_summary = "; ".join(
            f"{dim}={(risks.get(dim) or {}).get('level','?')}"
            for dim in ('market_risk', 'execution_risk', 'timing_risk')
        )
        conflict_lines = "\n".join(
            f"  - {c.get('conflict_id')} (severity={c.get('severity')}): {c.get('explanation','')[:120]}"
            for c in conflicts[:3]
        ) or "  (none)"
        prompt = dedent(f"""
            You are MIDAN, a senior startup operator. The user described this idea:
            "{idea_text}"

            The L4 decision engine produced these structured outputs (DO NOT invent any others):
            - decision_state: {decision_state}
            - decision_strength.tier: {strength_tier}
            - risk_decomposition: {risks_summary}
            - top binding risk dimension: {top_dim_label} (level={top_dim_level})
            - top dim reasoning: {top_dim_reasoning[:240]}
            - L3.differentiation.verdict: {diff_verdict}
            - L3.competition.competitive_pressure: {competition_press}
            - conflicts:
{conflict_lines}
            - L1 grounding: business_model={bm}, segment={seg}, stage={stage}
            - text grounding: {context_label} for {customer_label} in {market_label}

            Respond in EXACTLY this pattern — 4 parts, max 5 sentences total:
            1. INFER: Name the specific sub-space this sits in.
            2. ANALYZE: State the non-obvious structural challenge ANCHORED in the top binding
               risk dimension above (not "competition is high"). Reference the specific dimension.
            3. CHALLENGE: Surface the hidden assumption most likely to be wrong, ANCHORED in the
               L3 differentiation verdict OR a listed conflict.
            4. ASK: ONE sharp question that tests the top binding risk — specific to THIS idea.

            HARD RULES — violations are rejected:
            - NEVER reference TAS, signal_tier, or any X/100 percentage. The decision system
              is qualitative; numeric framing is forbidden.
            - NEVER ask: "What is your problem?", "Who is your customer?", "Which market?"
            - Never start with "I". Never use: "Great!", "That's interesting", "Certainly".
            - Every claim must trace to one of the structured outputs listed above.
            - 4-5 sentences total, flowing prose, no bullet points, no labels.

            Output only the response text — no labels, no JSON.
        """).strip()

        response = LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=2048,
        )
        reply = response.choices[0].message.content.strip()
        if reply and len(reply) > 40:
            data["analysis_summary"] = reply
            return reply
        _OP_REPLY_LOG.warning(
            "[OP] LLM returned empty/too-short reply (len=%d) — falling back",
            len(reply or ''),
        )
    except Exception as llm_err:
        _OP_REPLY_LOG.warning(
            "[OP] LLM call failed (%s: %r) — falling back to L4-grounded heuristic",
            type(llm_err).__name__, llm_err,
        )

    # ── Heuristic fallback — we use this when LLM is unavailable; fully L4-grounded, no TAS, no fake numbers ──
    infer = (
        f"This is {context_label} for {customer_label} in {market_label}, "
        f"not a generic {sector.title() or 'sector'} play."
    )
    analyze = (
        f"L4 places the binding constraint on {top_dim_label} at level {top_dim_level}: "
        f"{top_dim_reasoning[:200]}"
    )
    if conflicts:
        c0 = conflicts[0]
        challenge = (
            f"The assumption likely to break first is the one that triggers conflict "
            f"`{c0.get('conflict_id')}` — {c0.get('explanation','')[:160]}"
        )
    else:
        challenge = (
            f"With L3 differentiation verdict={diff_verdict} and competition pressure={competition_press}, "
            f"the assumption likely to break first is that the moat will form before the market notices."
        )

    # we pick a risk-dimension-specific question grounded in the L4 dimension, not legacy dominant_risk
    _risk_q_by_dim = {
        'market_risk':    f"What in your model survives if the macro regime ({data.get('regime','unknown')}) deteriorates further over the next 90 days?",
        'execution_risk': f"What's the single operational metric that, if it doesn't move in 6 weeks, kills the build?",
        'timing_risk':    f"What evidence do you have that {customer_label} are ready to change behavior NOW, not in 12 months?",
    }
    ask = _risk_q_by_dim.get(
        top_dim_name,
        "What is the one assumption that, if wrong, makes the entire model unworkable?"
    )

    return (
        f"Reading this as a {decision_state} at {strength_tier} strength. "
        f"{infer} {analyze} {challenge} {ask}"
    )


_MIDAN_PRE_ANALYSIS_SYSTEM = """\
You are MIDAN — a startup idea evaluator. You have heard hundreds of pitches. Most were vague. Most people who come here have been sitting on something and finally decided to think it through with someone. That someone is you.

Your job right now is to collect three pieces of information so you can run a real analysis. You do not have them yet. You do not move forward until you do.

THE THREE PIECES YOU NEED:
1. PROBLEM — a specific pain, for a specific person, at a specific severity
2. SOLUTION — the concrete mechanism (app, marketplace, SaaS, service, hardware) — not "an app", a real thing
3. MARKET — a specific country AND a specific customer type

══════════════════════════════════════
THE 3-SENTENCE RULE — NO EXCEPTIONS
══════════════════════════════════════

Every response you write in this phase follows this exact structure:

Sentence 1 — React to what they just said. Acknowledge it like a real person would. If it was vague, name it. If it was interesting, show it. If it was funny, match it. Do not perform warmth — be it.
Sentence 2 — Add a thought, observation, or small insight that shows you were actually listening and thinking, not just waiting for the next data point. This is what separates a conversation from a form.
Sentence 3 — Ask the one question that moves things forward. One question. Never two.

The farewell situation is the only exception: 1-2 natural sentences, then stop. Do not ask anything after a goodbye.
The trigger situation is the only other exception: once all 3 inputs are confirmed, extract+confirm in one sentence, then fire the trigger. Do not add filler.

══════════════════════════════════════
TONE STANDARD
══════════════════════════════════════

MIDAN is a smart friend who happens to know a lot about startups. Not a professor. Not a consultant. Not a form.
- Warm but not soft. Curious but not eager. Direct but not cold.
- Reads like a real person typed it, not like it was generated.
- Occasional dry humor is fine if it fits. Never forced.
- Never uses filler: no "Great!", "Interesting!", "Of course!", "Sure!", "Absolutely!", "Certainly!", "That's helpful!"
- Never sounds like it's evaluating the user — it's evaluating the idea together with them.
- No bullet points in replies. No corporate phrasing.

══════════════════════════════════════
COLLECTION RULES
══════════════════════════════════════

- Ask ONE question per reply. Never two.
- If an answer is present but too vague ("small businesses", "young people", "an app", "the Middle East", "everyone"), treat it as incomplete. Push back on the vague part. Do not accept it and move on.
- Never repeat a question in the same wording. Come at the same gap from a different angle.
- If the user has given 2 of 3 inputs and drifts off-topic, weave back to the missing one naturally within the 3-sentence structure. Do not restart from zero.
- The trigger fires ONLY when all three are specific and usable: problem names a real pain (not a category), solution names a real mechanism (not just "an app"), market names a specific country AND a specific customer type. If any of the three is still vague, push back — do not trigger.
- Once all three are confirmed and specific, say exactly this and nothing else: "Alright, I have what I need. Give me a second."

══════════════════════════════════════
EDGE CASE LIBRARY — FOLLOW EXACTLY
══════════════════════════════════════

CASE 1 — GREETING
Detected: hi, hello, hey, sup, yo, salam, marhaba, ezzayak, what's up, howdy, good morning
Instruction: Open warmly. Make them feel like they landed somewhere worth being.
Example:
"Hey — glad you're here. Most people who show up have been sitting on something for a while and just needed somewhere to actually think it through. What are you working on?"

CASE 2 — NO IDEA YET / DON'T KNOW WHERE TO START
Detected: I don't have an idea, I don't know where to start, I'm not sure what to build, I don't have anything yet
Instruction: Validate the honesty, reframe the starting point, redirect to the source of ideas.
Example:
"Honestly, that's a better starting point than most — people who say they have everything figured out usually have less figured out than they think. The clearest ideas tend to start with a frustration, not a business plan. What's something that's been bugging you lately that you keep coming back to?"

CASE 3 — WORKING ON SOMETHING FOR A WHILE BUT HASN'T EXPLAINED
Detected: I've been working on something, I have an idea I've been thinking about, I've had this idea for a while
Instruction: Show that longevity signals something real. Invite the rough version.
Example:
"A while usually means it's real — the ideas that don't survive past the first week usually weren't the right ones. I'd rather hear the rough version than the polished pitch. What is it?"

CASE 4 — WHO IS MIDAN / HOW DOES IT WORK
Detected: what are you, who are you, what is MIDAN, how does this work, what do you do, explain yourself
Instruction: One clear sentence on identity. One sentence on what you need. Ask for the idea.
Example:
"I'm MIDAN — I look at startup ideas before founders spend months building the wrong thing, and I give you a real read, not a pep talk. I need three things: the problem you've spotted, how you're thinking of fixing it, and which market you're going after. What's the idea?"

CASE 5 — OFF-TOPIC QUESTION
Detected: what's the weather, what's my name, tell me a joke, what time is it, who's the president, what's [anything unrelated]
Instruction: Honest about the limitation, no fake apology, redirect with warmth.
Example:
"That one's outside my range — I only know what you bring to me here, and that's not it. I'm built for startup ideas specifically, not general questions. What are you working on?"

CASE 6 — CAN'T EXPLAIN THE IDEA
Detected: I don't know how to explain it, it's hard to explain, I can't put it into words, I'm not sure how to describe it
Instruction: Reframe difficulty as a sign of something real, invite the messy version.
Example:
"Most good ideas are hard to explain at first — if it were easy to describe, someone would've built it already. Skip the polished version and just tell me what's broken, from your own experience. What's the thing that made you think there was a gap here?"

CASE 7 — ONE-WORD OR ONE-PHRASE ANSWER
Detected: single word or phrase with no context — "fintech", "app", "Egypt", "delivery", "SaaS", "B2B", "healthcare", "edtech"
Instruction: Name the vagueness directly. Show why it matters. Ask the specific follow-up.
Example (for "fintech"):
"Fintech covers a lot of territory — payments, lending, insurance, remittances, compliance, it branches everywhere. I can't tell yet if you're onto something until I understand what specifically is broken. What's the problem within that space, and who has it?"
Example (for "Egypt"):
"Egypt as the market — useful starting point, that's one piece. I still need to understand what you're solving there and for who. What's the problem that exists in Egypt that your idea fixes?"

CASE 8 — VAGUE CUSTOMER OR MARKET
Detected: everyone, businesses, all companies, young people, users, people, SMEs, anyone, the general public
Instruction: Name the error and why it matters. Make the ask specific.
Example:
"'Everyone' is the answer that kills more startups than bad products — when you build for everyone, you usually end up building for no one. The most defensible startups start with one specific type of person who has this problem badly enough to pay to fix it. Who specifically is that person?"

CASE 9 — FAREWELL
Detected: bye, byeee, goodbye, later, ciao, yalla bye, good night, ttyl, see you, take care, peace
Instruction: 1-2 warm sentences. Stop. Do not ask anything. Do not redirect to the startup.
Example:
"Good luck — come back whenever the idea's ready to think through. I'll be here."

CASE 10 — FRUSTRATED OR PUSHING BACK
Detected: this is pointless, you don't get it, forget it, why do you keep asking, this is stupid, you're useless, whatever, ugh
Instruction: Take the frustration seriously. Explain the why honestly. Open the door back in.
Example:
"That's fair — I'm asking a lot before giving you anything back, and I get why that's annoying. The reason is that a wrong read from me is worse than no read, and I'd rather ask one more question than give you something useless. What's the part that feels like it's not landing?"

CASE 11 — ASKS IF THEIR IDEA IS GOOD BEFORE FULL INPUTS
Detected: is my idea good, do you think this will work, is this a good idea, what do you think so far
Instruction: Honest about not being able to answer yet. Explain what's coming. Ask for the missing piece.
Example:
"I genuinely don't know yet — I'd be making it up, and a guess from me isn't worth more than a guess from anyone else at this point. Once I have the full picture, I run it through a real model and give you a specific read. What's the problem you're solving?"

CASE 12 — META QUESTION ABOUT THE ANALYSIS
Detected: will this be accurate, what happens after, how do you evaluate, what do you do with my idea, what does the analysis look like
Instruction: Answer the question directly. Tease what's coming. Redirect.
Example:
"Once you give me the problem, solution, and market, I run it through a scoring model that looks at market conditions, competition, business model viability, and more — not just vibes. The output is a real read with specific callouts, not a generic answer. What are you working on?"

CASE 13 — ALL THREE INPUTS IN ONE MESSAGE
Detected: user gives problem, solution, and specific market in a single message
Instruction: Extract all three. Confirm back in one natural sentence. Fire the trigger immediately.
Example:
"Got it — [problem] solved with [solution] for [customer type] in [country]. Alright, I have what I need. Give me a second."

CASE 14 — CONTRADICTION MID-CONVERSATION
Detected: earlier said Egypt, now says UAE; earlier said B2B, now says B2C; conflicting market, customer, or solution signals
Instruction: Flag the specific contradiction. Explain why it matters. Ask which is right.
Example:
"You mentioned Egypt earlier — now you're saying UAE, and those are different markets with different dynamics and different reads. I want to make sure I'm looking at the right one before I go further. Which market are we actually building for?"

CASE 15 — TESTING THE SYSTEM
Detected: "test", "123", "asdf", random characters, gibberish, keyboard mashing
Instruction: Light acknowledgment that you noticed. One sentence on what you actually need. Ask.
Example:
"Looks like you're checking if I'm alive — I am. I work best with a real idea, not test strings. What are you building?"

CASE 16 — EMOTIONAL OR PERSONAL MESSAGE
Detected: I'm stressed, I've been having a rough time, I'm overwhelmed, I'm exhausted, I've been going through a lot
Instruction: Acknowledge it genuinely. Be honest about your scope. Offer a gentle bridge.
Example:
"Heard — I'm not going to rush past that. I'm not built for this part, but if working through an idea gives you a bit of focus right now, that's not a bad trade. What's on your mind?"

CASE 17 — ASKS TO COMPARE TO A COMPETITOR PRE-ANALYSIS
Detected: how does this compare to [X], is this better than [competitor], what's different from [existing product]
Instruction: Honest about why you can't do it yet. Signal that it's coming. Redirect.
Example:
"I'd need to understand your idea properly before I can stack it against anything — comparing something I don't fully know yet is just noise. Competitive positioning is actually one of the things I look at once the full picture is in. What's your idea?"

CASE 18 — TRYING TO SKIP THE PROCESS / DEMANDS ANALYSIS NOW
Detected: just analyze it, skip the questions, give me the analysis, stop asking me things, just run it, go already
Instruction: Acknowledge the friction honestly. Explain the tradeoff in one sentence. Come back to the specific missing piece.
Example:
"I get it — the questions feel like friction before the thing you actually came for. The analysis is only as good as what I put in, and a specific read is worth more than a fast generic one. What's the one thing I'm still missing from you?"

CASE 19 — USER SPEAKS IN ARABIC
Detected: message is in Arabic (Egyptian dialect, Gulf dialect, or MSA)
Instruction: Respond in the same language and dialect naturally. Follow the 3-sentence rule. Do not switch back to English.
Example response in Egyptian Arabic:
"أهلاً — معظم اللي بييجوا هنا بيكونوا شايلين حاجة في دماغهم من فترة وبس محتاجين حد يسمعهم ويفكر معاهم فيها. أنا MIDAN — بشتغل على أفكار الستارتاب وبدي رأي حقيقي، مش مجرد تشجيع. إيه اللي شغال عليه؟"
Example response in MSA:
"أهلاً — معظم من يأتون إلى هنا لديهم فكرة ينتظرون فرصة للتفكير فيها بشكل جدي. أنا MIDAN — أحلل أفكار الشركات الناشئة وأقدم قراءة حقيقية قبل أن يبدأ المؤسس في البناء. ما الذي تعمل عليه؟"

CASE 20 — DETAILED, COMPLETE, WELL-ARTICULATED IDEA
Detected: user gives a clear, specific problem + specific solution + specific market in enough detail to be usable
Instruction: Confirm that all three are there clearly. Fire the trigger. Do not ask unnecessary follow-up questions.
Example:
"Clear and specific — problem is there, the mechanism makes sense, and the market is narrow enough to work with. Alright, I have what I need. Give me a second."

══════════════════════════════════════
HARD RULES — NEVER BREAK
══════════════════════════════════════

- Every response is 3 sentences (React → Observe → Ask). The only exceptions are: farewell (1-2 sentences, no question), trigger (confirm + trigger line, stop).
- Never ask for more than one piece of information.
- Never accept a vague answer as complete and move on.
- Never repeat a question in the same wording.
- Never fire the trigger unless all three inputs are specific and usable.
- Never ignore a farewell and push for startup info.
- Never mention L1, L2, L3, L4, pipeline, SHAP, or any internal system names.
- Never use bullet points in your replies.
- Never start a response with a filler word or validation phrase.\
"""


_PRE_ANALYSIS_ERROR_FALLBACK = "I'm having trouble connecting right now — give me a second and try again."


def _chat_pre_analysis_reply(req) -> str:
    """
    Pre-analysis LLM chat. ALWAYS returns a string — never None.
    LLM handles every case via the system prompt.
    On any API failure the single fallback string is returned; no other
    hardcoded strings exist in this path.
    """
    all_messages = list(req.messages)
    if not all_messages:
        return _PRE_ANALYSIS_ERROR_FALLBACK

    last_msg = all_messages[-1]

    # Build messages list: system prompt first, then history, then latest user message.
    # Skip any leading assistant turns — the conversation must start with a user message.
    messages: list = [{"role": "system", "content": _MIDAN_PRE_ANALYSIS_SYSTEM}]
    first_user_seen = False
    for msg in all_messages[:-1]:
        role = "assistant" if msg.role == "assistant" else "user"
        if not first_user_seen:
            if role != "user":
                continue
            first_user_seen = True
        messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": last_msg.content})

    _TRACE_LOG.info("[TRACE][_chat_pre_analysis_reply] LLM CALLED | history_turns=%d | msg=%r",
                    len(messages) - 2, last_msg.content[:100])
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )
        result = (response.choices[0].message.content or "").strip()
        if result:
            _TRACE_LOG.info("[TRACE][_chat_pre_analysis_reply] LLM RESPONDED: %r", result[:120])
            return result
        _TRACE_LOG.warning("[TRACE][_chat_pre_analysis_reply] LLM RETURNED EMPTY → fallback")
        return _PRE_ANALYSIS_ERROR_FALLBACK
    except Exception as e:
        __import__('logging').getLogger("midan.chat").warning("[pre-analysis] LLM error: %r", e)
        _TRACE_LOG.warning("[TRACE][_chat_pre_analysis_reply] LLM FAILED: %r → fallback", e)
        return _PRE_ANALYSIS_ERROR_FALLBACK


def _sanitize_chat_output(text: str) -> str:
    """Strip internal labels, debug stamps, UNKNOWN leakage."""
    if not text:
        return ""
    # we strip any internal tags if present
    text = re.sub(r'\[decision_state=.*?\]', '', text)
    # we sanitize UNKNOWN leakage before returning
    text = text.replace("UNKNOWN", "unknown")
    return text.strip()


def _chat_llm_reply(req) -> Optional[str]:
    """
    Gemini multi-turn chat with full conversation memory.
    Injects analysis context, maintains complete history, responds naturally.
    Returns None on failure — caller falls back to heuristic.
    """
    ctx = req.context
    decision_state = ctx.get('decision_state') or (ctx.get('l4_decision') or {}).get('decision_state')

    if not decision_state:
        _TRACE_LOG.info("[TRACE][_chat_llm_reply] no decision_state → routing to _chat_pre_analysis_reply (Gemini)")
        return _chat_pre_analysis_reply(req)

    route = _post_decision_route(ctx)
    mode = route['mode']

    l4 = ctx.get('l4_decision') or {}
    top_dim_name, top_dim_block = _l4_top_risk_dim(l4)
    top_dim_label = (top_dim_name or 'risk').replace('_', ' ')
    top_dim_level = (top_dim_block or {}).get('level', 'unknown')
    top_dim_reasoning = (top_dim_block or {}).get('reasoning', '')

    strength_tier = ((l4.get('decision_strength') or {}).get('tier') or
                     (ctx.get('decision_strength') or {}).get('tier') or 'unknown')

    idea_text = ctx.get("idea", "") or " ".join(m.content for m in req.messages if m.role == "user").strip()

    idea_feat = ctx.get("idea_features") or {}
    bm     = idea_feat.get("business_model") or ctx.get("business_model") or "unknown"
    seg    = idea_feat.get("target_segment") or ctx.get("target_segment") or "unknown"
    stage  = idea_feat.get("stage")          or ctx.get("stage")          or "unknown"
    sector = ctx.get("sector") or ctx.get("industry") or ""
    region = ctx.get("country") or ctx.get("region") or ""

    system_instruction = f"""You are MIDAN. The analysis on this idea is done. The data is in. You are now the founder's strategic advisor — not a consultant hedging, not a chatbot validating. You speak from the data. You state what you found. You tell them what it means for them specifically.

You are in {mode} mode.

STARTUP PROFILE ON FILE — do not ask for any of these again:
- Idea: "{idea_text}"
- Business model: {bm}
- Target segment: {seg}
- Stage: {stage}
- Sector: {sector}
- Region: {region}
- Decision: {decision_state} at {strength_tier} strength
- Top risk: {top_dim_label} at {top_dim_level} — {top_dim_reasoning}

DECISION STATE BEHAVIOR — match exactly:

If decision is GO:
  Confident and direct. Focus on what to do next. The finding is clear — say it without hedging.
  Acknowledge the top risk but do not let it dominate. This is a green light with a specific watch item.

If decision is CONDITIONAL:
  Clear about what the condition is and exactly why it matters for this specific idea.
  Do not say "needs more validation" without naming what to validate, how, and why it is the blocking factor.
  The condition comes from {top_dim_label} at {top_dim_level} — name it directly.

If decision is NO_GO:
  Honest, not cruel. State the specific reason from {top_dim_label} at {top_dim_level}.
  Do not soften the finding with hope language. The founder deserves a clear read, not false comfort.
  Do not say "it might not work" — say what specifically makes it not work based on the data.

MEMORY RULES — enforce on every reply:
- Every follow-up must be answered using the analysis data already on file. Never ask the user to re-explain something already in the profile above.
- If the user asks "why did you give me that score" or "why that decision": answer directly from {top_dim_label}, {top_dim_level}, and {top_dim_reasoning}. Do not generalize.
- If the user asks about something not in the profile: answer from general startup knowledge, stay in MIDAN's voice, do not break character.
- If the user asks the same question twice: go deeper the second time, not wider. Add a layer they have not heard yet.
- If the user implies something changed: say "I currently have you as {bm} targeting {seg} at {stage} stage — is that still accurate?" then proceed.

MODE-SPECIFIC BEHAVIOR:

RESOLVING_CONFLICT mode:
  Do not just surface the contradiction. State which side the data supports more strongly, then ask the founder which reality is accurate.
  Take a position. Do not be neutral about what the signals say.
  Frame it as: "The data leans toward [X]. [Y] would require [specific condition] to hold — which is actually your situation?"

ADVISORY_ONLY mode:
  Prefix every reply with a specific uncertainty caveat — not generic. Name the actual gap.
  Example: "The market data here is thin — I'm working with limited signals on {region} + {sector}."
  Then give your best read from what you do have. Do not refuse to advise — contextualize the uncertainty, then advise.

PUSHBACK (implicit — fires when the founder disputes the finding):
  Do not cave. Restate the finding from the data. Acknowledge their point only if it adds genuinely new information.
  If they are pushing back emotionally: "The data says [X] because {top_dim_reasoning}. If your situation differs from that assumption, tell me how."
  If they surface new information that materially changes the analysis: "That changes the read — specifically on {top_dim_label}."

FORBIDDEN — never do any of these:
- Ask for information already in the startup profile above
- Say "Great question", "Absolutely", "Of course", "Certainly", or any filler validation
- Give generic startup advice that does not reference the specific profile on file
- Say "it depends" without immediately stating what it depends on and why
- Hedge with "it could be" or "it might be" without immediately following with what the data actually shows
- Break {mode} behavior mid-conversation — if in RESOLVING_CONFLICT, do not drift into casual advisor mode
- Mention L1, L2, L3, L4, or any internal system labels
- Start any sentence with "I"

TONE:
- Peer-to-peer. Not corporate, not therapeutic, not cheerful.
- Short paragraphs. No bullet points unless listing genuinely distinct items.
- Direct. If the finding is uncomfortable, say it directly. The founder came here for a real read, not comfort."""

    all_messages = list(req.messages)

    try:
        # Build messages list: system prompt first, then synthetic analysis turn, then history
        messages: list = [{"role": "system", "content": system_instruction.strip()}]

        # Inject the analysis result as a synthetic opening exchange so the model
        # can answer follow-up questions without the user re-explaining the analysis.
        analysis_summary = ctx.get("analysis_summary")
        if analysis_summary:
            messages.append({"role": "user", "content": "What is the initial analysis result for my startup idea?"})
            messages.append({"role": "assistant", "content": analysis_summary})

        # Append all prior turns (every message except the final user message)
        for msg in all_messages[:-1]:
            role = "assistant" if msg.role == "assistant" else "user"
            messages.append({"role": role, "content": msg.content})

        # Append the latest user message
        last_msg = all_messages[-1]
        messages.append({"role": "user", "content": last_msg.content})

        response = LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=8192,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        __import__('logging').getLogger("midan.chat").warning(f"LLM chat error: {e!r}")
        return None

class InteractRequest(BaseModel):
    context: Dict[str, Any]
    messages: List[ChatMessage]

# we export everything defined in this module — including underscore-prefixed helpers —
# so other midan submodules can wildcard-import the full surface
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]

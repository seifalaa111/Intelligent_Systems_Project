"""
midan.pipeline — process_idea + run_inference orchestrators.

process_idea: L0 gate → A1 sector/country parse → L1 confidence-scored
extraction (with consistency + sufficiency gate) → L2 inference → L3
reasoning → L4 decision → response shaping.

run_inference: L2/L3/L4 + legacy strategic generators. Called by process_idea
once L1 has produced a sufficient envelope.
"""
from midan.core import *  # noqa: F401,F403
from midan.l0_gate import _layer0_sanity_check, _l0_how_to_fix, _log_rejection  # noqa: F401
from midan.l1_parser import (  # noqa: F401
    extract_idea_features, agent_a1_parse, _l1_clarification_message,
    _validate_l1_consistency,
)
from midan.l2_intelligence import (  # noqa: F401
    enhanced_regime_with_path, compute_fcm_membership, compute_shap,
    _idea_macro_adjustments, apply_idea_adjustments,
)
from midan.l3_reasoning import (  # noqa: F401
    compute_idea_signal, compute_l3_reasoning, _signal_tier,
)
from midan.l4_decision import compute_l4_decision  # noqa: F401
from midan.mechanism_extractor import run_mechanism_pipeline, serialize_envelope  # noqa: F401
from midan.response import (  # noqa: F401
    _generate_l4_reasoning, agent_a0_evaluate_idea, _generate_explanation_layer,
)


# ── extracted from api.py ─────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# FULL INFERENCE — 4-Layer Hybrid Pipeline
# ═══════════════════════════════════════════════════════════════

def run_inference(
    sector: str,
    country: str,
    idea_text: str = "",
    logs: list = None,
    idea_features_result: Optional[dict] = None,
) -> dict:
    """
    4-Layer hybrid inference:
      L1 — Idea feature extraction  (idea-specific, LLM-powered)
      L2 — Macro ML pipeline        (DBSCAN→FCM→SVM→SHAP→SARIMA)
      L3 — Idea signal scoring      (regime×model×segment fit)
      L4 — Composite TAS            (new formula ensures per-idea variability)

    `idea_features_result`: pre-computed L1 envelope from `extract_idea_features`.
    When provided, the caller has already enforced the sufficiency gate and
    L2–L4 may safely consume the values. When omitted (legacy path), L1 is
    extracted here AND the caller is responsible for not feeding UNKNOWN values
    downstream — but this branch should not be used by `process_idea`.
    """
    if logs is None:
        logs = []
    sec = sector.lower()

    # ── L1: Idea Feature Extraction ───────────────────────────────────────────
    if idea_features_result is None:
        idea_features_result = extract_idea_features(idea_text, sec)
        if not idea_features_result["is_sufficient"]:
            # Defensive: process_idea is the canonical caller and gates upstream.
            # If anyone bypasses that gate, halt here rather than feed UNKNOWN
            # values into L2–L4.
            raise ValueError(
                f"L1 extraction insufficient for inference: "
                f"unknown_required={idea_features_result['unknown_required']} "
                f"aggregate_confidence={idea_features_result['aggregate_confidence']}"
            )
    # Use runtime_values for downstream arithmetic — non-required UNKNOWNs
    # get documented neutral defaults here. Required UNKNOWNs are impossible
    # at this point (the gate would have halted upstream).
    idea_features = idea_features_result["runtime_values"]
    bm_label  = idea_features['business_model']
    seg_label = idea_features['target_segment'].upper()
    diff      = idea_features['differentiation_score']
    diff_label = {1: 'minimal', 2: 'low', 3: 'moderate', 4: 'strong', 5: 'exceptional'}.get(diff, 'moderate')
    logs.append(f"[L1] Idea features: bm={bm_label} | seg={seg_label} | diff={diff}/5 ({diff_label}) | stage={idea_features['stage']} | conf={idea_features_result['aggregate_confidence']:.2f}")

    # ════════════════════════════════════════════════════════════════════════
    # L2 — Market intelligence layer
    #
    # Transparency contract:
    #   l2_macro_base       — pure (sector, country) static lookup, source=static_table
    #   l2_idea_adjustments — explicit, traceable deltas, source=inferred
    #   l2_macro_adjusted   — base + capped deltas; what the SVM actually sees
    #   regime_decision_path— SVM step → optional rule_override → final
    #   fcm_membership      — parallel fuzzy regime signal, source=fcm_static_centers
    #   l2_data_freshness   — staleness metadata; runtime_staleness_flag triggers
    #                          a confidence penalty downstream
    # ════════════════════════════════════════════════════════════════════════

    # ── L2A: Static macro vector — pure (sector, country) lookup ──────────────
    macro_country = COUNTRY_MACRO_DEFAULTS.get(country.upper(), {'inflation': 10.0, 'gdp_growth': 3.0, 'unemployment': 7.0})
    base_inflation, base_gdp, unemployment = macro_country['inflation'], macro_country['gdp_growth'], macro_country['unemployment']
    logs.append(f"[L2A] Base macro (static_table): inflation={base_inflation}% | GDP={base_gdp}%")

    eff_inf_offset, gdp_boost, sector_velocity = SECTOR_EFF_MACRO.get(sec, SECTOR_EFF_MACRO['other'])
    scale     = base_inflation / 33.9
    base_inflation_eff = float(np.clip(eff_inf_offset * scale, 1.0, 100.0))
    base_gdp_eff       = float(base_gdp + gdp_boost)
    base_friction      = float(np.clip(base_inflation_eff + unemployment - base_gdp_eff, -50, 100))
    base_cap_conc      = SECTOR_MEDIANS.get(sec, SECTOR_MEDIANS['other'])

    macro_base = {
        "inflation":             base_inflation_eff,
        "gdp_growth":            base_gdp_eff,
        "macro_friction":        base_friction,
        "capital_concentration": float(base_cap_conc),
        "velocity_yoy":          float(sector_velocity),
    }

    # ── L2A.5: Idea-perturbed macro vector ────────────────────────────────────
    # Adjustments only fire on high-confidence L1 fields (>= 0.70). Each delta
    # is logged with its source field, source value, and reason code. The
    # response carries both base and adjusted vectors so consumers can see
    # exactly what was inferred vs. what was looked up.
    idea_adjustments = _idea_macro_adjustments(macro_base, idea_features_result)
    macro_adjusted, applied_deltas = apply_idea_adjustments(macro_base, idea_adjustments)
    macro_adjusted = {
        f: float(np.clip(macro_adjusted[f],
                         {'inflation': 1.0, 'gdp_growth': -10.0, 'macro_friction': -50.0,
                          'capital_concentration': 0.0, 'velocity_yoy': 0.0}.get(f, -1e9),
                         {'inflation': 100.0, 'gdp_growth': 15.0, 'macro_friction': 100.0,
                          'capital_concentration': 1e9, 'velocity_yoy': 1.0}.get(f, 1e9)))
        for f in macro_adjusted
    }
    if idea_adjustments:
        logs.append(
            f"[L2A.5] Idea-perturbed macro: "
            f"{', '.join(f'{a['feature']}{a['delta']:+.2f}({a['reason_code']})' for a in idea_adjustments)}"
        )
    else:
        logs.append("[L2A.5] No idea adjustments applied (L1 confidence below threshold for all rules).")

    # Use the adjusted vector for SVM input. SHAP runs on the same vector so
    # explainability matches what the model actually saw.
    inflation  = macro_adjusted["inflation"]
    gdp_growth = macro_adjusted["gdp_growth"]
    macro_fric = macro_adjusted["macro_friction"]
    cap_conc   = macro_adjusted["capital_concentration"]
    velocity   = macro_adjusted["velocity_yoy"]

    x_raw    = np.array([[inflation, gdp_growth, macro_fric, float(cap_conc), velocity]])
    x_scaled = scaler.transform(x_raw)
    x_pca    = pca.transform(x_scaled)
    logs.append(f"[L2A] Effective macro (post-adjustment): inflation={inflation:.1f}% | GDP={gdp_growth:.1f}% | friction={macro_fric:.1f}")

    # ── L2B: SVM Classification + visible decision path ──────────────────────
    pred_enc   = svm.predict(x_scaled)[0]
    proba      = svm.predict_proba(x_scaled)[0]
    svm_regime = le.inverse_transform([pred_enc])[0]
    svm_conf   = float(proba.max())
    pred_class_idx = int(np.argmax(proba))

    regime_trace = enhanced_regime_with_path(
        svm_regime, svm_conf, inflation, gdp_growth, macro_fric, velocity,
    )
    regime         = regime_trace["regime"]
    conf           = regime_trace["confidence"]
    decision_path  = regime_trace["decision_path"]
    overridden     = any(step["step"] == "rule_override" for step in decision_path)
    logs.append(
        f"[L2B] SVM: {svm_regime} ({svm_conf:.1%}) → "
        f"{'rule_override:' + next(s['rule_id'] for s in decision_path if s['step']=='rule_override') + ' → ' if overridden else ''}"
        f"final: {regime} ({conf:.1%})"
    )

    # ── L2B.5: FCM parallel fuzzy regime signal ──────────────────────────────
    fcm_signal = compute_fcm_membership(x_pca[0], fcm_centers)
    if fcm_signal.get("available"):
        logs.append(
            f"[L2B.5] FCM membership: top={fcm_signal['top_cluster']} "
            f"({fcm_signal['top_membership']:.2f}) | entropy={fcm_signal['entropy']:.2f} "
            f"| ambiguous={fcm_signal['is_ambiguous']}"
        )

    # ── L2C: SHAP (macro signal explainability over the ADJUSTED vector) ─────
    shap_dict = compute_shap(lgb, x_scaled, predicted_class_idx=pred_class_idx)
    shap_max  = float(max(shap_dict.values()))
    xai_score = float(conf * shap_max)
    top_feat  = max(shap_dict, key=shap_dict.get)
    logs.append(
        f"[L2C] SHAP top macro signal: {top_feat} ({shap_dict[top_feat]:.3f}) | "
        f"xai_score={xai_score:.3f} (conf×shap_max)"
    )

    # ── L2D: SARIMA Forecast (precomputed JSON) ──────────────────────────────
    sarima_trend = 0.50
    train_drift  = False
    if sec in sarima_results:
        fc_raw       = sarima_results[sec]['forecast_mean']
        fc           = [max(0, v) for v in fc_raw]
        fc_mean      = float(np.mean(fc))
        sarima_trend = float(np.clip(fc_mean / 50.0, 0.15, 0.90))
        train_drift  = sarima_results[sec].get('drift_flag', False)
        logs.append(f"[L2D] SARIMA: mean={fc_mean:.1f} → trend={sarima_trend:.2f} | train_time_drift={train_drift}")
    else:
        logs.append(f"[L2D] No SARIMA model for {sec} — neutral trend=0.50")
    if train_drift:
        logs.append("[L2D] Training-time drift flag set — patterns shifted during training window")

    # ── L2E: Freshness gate — penalize confidence when SARIMA is stale ───────
    freshness = compute_l2_freshness()
    if freshness["runtime_staleness_flag"]:
        old_conf = conf
        conf = float(np.clip(conf * SARIMA_STALENESS_PENALTY, 0.0, 1.0))
        decision_path.append({
            "step":        "staleness_penalty",
            "regime":      regime,
            "confidence":  conf,
            "source":      "freshness_rule",
            "explanation": (
                f"SARIMA last_date={freshness['sarima_as_of']} "
                f"({freshness['sarima_days_stale']} days stale, threshold "
                f"{SARIMA_STALENESS_DAYS}). Confidence multiplied by "
                f"{SARIMA_STALENESS_PENALTY}."
            ),
        })
        logs.append(
            f"[L2E] STALENESS PENALTY: conf {old_conf:.2f} → {conf:.2f} "
            f"(SARIMA {freshness['sarima_days_stale']} days old)"
        )

    # legacy alias for downstream consumers; the freshness envelope carries
    # the canonical training-time vs runtime breakdown.
    drift_flag = train_drift

    # ── L3: Idea Signal Scoring (legacy scalar — kept for L4 TAS contract) ──
    # The scalar idea_signal feeds L4 TAS computation (kept stable so existing
    # consumers don't break). The PRIMARY explanation surface for L3 is the
    # structured `l3_reasoning` envelope built right after this call.
    idea_signal_data = compute_idea_signal(idea_features, regime, sector=sec, idea_text=idea_text)
    idea_signal      = idea_signal_data['idea_signal']
    dominant_risk    = idea_signal_data.get('dominant_risk', 'execution')
    logs.append(
        f"[L3] Idea signal (legacy scalar): {idea_signal:.3f} | bm_type={bm_label} | "
        f"dominant_risk={dominant_risk} | "
        f"base_fit={idea_signal_data['breakdown']['model_regime_fit']:.3f} | "
        f"diff_effect={idea_signal_data['breakdown']['differentiation']:+.3f} | "
        f"stage={idea_signal_data['breakdown']['stage_readiness']:+.3f}"
    )

    # ── L3 reasoning — primary structured explanation ───────────────────────
    l3_reasoning = compute_l3_reasoning(
        idea_text=idea_text,
        l1_result=idea_features_result,
        regime=regime,
        sector=sec,
        idea_signal_data=idea_signal_data,
    )
    n_interactions = len(l3_reasoning.get('signal_interactions', []))
    n_insufficient = len(l3_reasoning.get('insufficient_information', []))
    logs.append(
        f"[L3] Reasoning: diff_verdict={l3_reasoning['differentiation'].get('verdict','n/a')} | "
        f"comp_pressure={l3_reasoning['competition'].get('competitive_pressure','n/a')} | "
        f"bm_available={l3_reasoning['business_model'].get('available')} | "
        f"interactions_fired={n_interactions} | insufficient_modules={n_insufficient}"
    )

    # ── Mechanism extraction pipeline (between L3 and L4) ─────────────────────
    # run_mechanism_pipeline never raises — returns insufficient envelope on error.
    # mechanism_uncertainty feeds L4 as a continuous probabilistic modifier.
    mechanism_envelope = run_mechanism_pipeline(
        l3_reasoning  = l3_reasoning,
        l1_values     = idea_features,
        l1_confidence = idea_features_result.get('confidence', {}),
        idea_text     = idea_text,
        sector        = sec,
        country       = country,
    )
    logs.append(
        f"[MECH] mode={mechanism_envelope.extraction_mode} | "
        f"mechanisms={len(mechanism_envelope.mechanisms)} | "
        f"uncertainty={mechanism_envelope.uncertainty:.3f} | "
        f"evidence_quality={mechanism_envelope.epistemic_summary.evidence_quality} | "
        f"consistency_passed={mechanism_envelope.consistency_report.passed}"
    )

    # ── L4a: Legacy TAS — preserved for backward compat, ZERO decision influence ──
    # The decision is now made by compute_l4_decision (state machine over risks,
    # conflicts, uncertainty). TAS is kept only because some response fields and
    # external consumers still read it. It does NOT factor into decision_state.
    tas = round(
        conf         * 0.30
        + sarima_trend * 0.20
        + idea_signal  * 0.35
        + xai_score    * 0.15,
        3
    )
    signal_tier = _signal_tier(tas)
    logs.append(
        f"[L4a-legacy] tas={tas:.3f} ({signal_tier}) — informational only, no decision impact."
    )

    # ── L4b: Decision engine — the actual decision is made here ──────────────
    l4_decision = compute_l4_decision(
        l1_result           = idea_features_result,
        regime              = regime,
        regime_conf         = conf,
        fcm_membership      = fcm_signal,
        l2_freshness        = freshness,
        l3_reasoning        = l3_reasoning,
        legacy_tas          = tas,
        mechanism_uncertainty = mechanism_envelope.uncertainty,
    )
    logs.append(
        f"[L4b] Decision state={l4_decision['decision_state']} | "
        f"strength={l4_decision['decision_strength']['tier']} | "
        f"market={l4_decision['risk_decomposition']['market_risk']['level']} | "
        f"execution={l4_decision['risk_decomposition']['execution_risk']['level']} | "
        f"timing={l4_decision['risk_decomposition']['timing_risk']['level']} | "
        f"offsets={len(l4_decision['offsetting_applied'])} | "
        f"conflicts(high)={sum(1 for c in l4_decision['conflicting_signals'] if c.get('severity')=='high')}"
    )

    # ── Combined signals for display (macro SHAP + idea breakdown) ────────────
    # IMPORTANT: market_* values are SHAP fractional shares (sum≈1.0, display as v×100%).
    # idea_* values are signed adjustment deltas + one absolute base score.
    # model_regime_fit is the absolute base (0.58–0.91) and is sent separately so
    # the UI can render it on its own scale rather than conflating it with small deltas.
    # Removing abs(): signs are preserved so the UI can color positive vs negative.
    idea_breakdown = idea_signal_data['breakdown']
    combined_signals = {
        **{f"market_{k}": round(float(v), 4) for k, v in shap_dict.items()},
        # Base fit exposed alone so UI knows it's the baseline, not an adjustment
        "idea_model_regime_fit": round(float(idea_breakdown['model_regime_fit']), 4),
        # Signed adjustments — preserve sign so UI can distinguish boost vs penalty
        **{
            f"idea_{k}": round(float(v), 4)
            for k, v in idea_breakdown.items()
            if k != 'model_regime_fit'
        },
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
        a2_comps=a2_comps, a4_sentiment=a4_sentiment, idea_text=idea_text, logs=logs,
        mechanism_analysis=serialize_envelope(mechanism_envelope),
    )

    # ── Tier 3: signal anchors for every L4 text field (full traceability) ──
    # Each text field is paired with the canonical L1/L2/L3 signal paths that
    # justify it. Anchors are computed deterministically from the L4 decision
    # envelope — no LLM-generated trace fabrication.
    _diff_block = l3_reasoning.get('differentiation', {})
    _comp_block = l3_reasoning.get('competition', {})
    strategic_anchors = {
        'strategic_interpretation': [
            f"L2.regime={regime}",
            f"L4.decision_state={l4_decision['decision_state']}",
            f"L1.business_model={idea_features.get('business_model')}",
            f"L1.target_segment={idea_features.get('target_segment')}",
        ],
        'key_driver': [
            f"L4.risk_decomposition.market_risk={l4_decision['risk_decomposition']['market_risk']['level']}",
            f"L4.risk_decomposition.execution_risk={l4_decision['risk_decomposition']['execution_risk']['level']}",
            f"L4.risk_decomposition.timing_risk={l4_decision['risk_decomposition']['timing_risk']['level']}",
        ],
        'main_risk': [
            f"L3.dominant_risk={dominant_risk}",
            f"L3.differentiation.verdict={_diff_block.get('verdict')}",
            f"L1.regulatory_risk={idea_features.get('regulatory_risk')}",
        ] + [f"L4.conflict={c['conflict_id']}" for c in l4_decision['conflicting_signals']],
        'counterpoint': [
            f"L4.offsetting_applied=[{', '.join(o['offsetting_factor'] for o in l4_decision['offsetting_applied']) or 'none'}]",
            f"L3.competition.competitive_pressure={_comp_block.get('competitive_pressure', 'unknown')}",
        ],
        'differentiation_insight': [
            f"L1.differentiation_score={idea_features.get('differentiation_score')}",
            f"L3.differentiation.verdict={_diff_block.get('verdict')}",
            f"L3.differentiation.what_is_new=[{len((_diff_block.get('what_is_new') or []))} items]",
        ],
        'action': [
            f"L4.decision_state={l4_decision['decision_state']}",
            f"L4.decision_strength.tier={l4_decision['decision_strength']['tier']}",
        ],
    }
    logs.append(
        f"[L4-anchors] each text field annotated with signal_paths "
        f"({sum(len(v) for v in strategic_anchors.values())} anchors total)"
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
            except Exception as _slack_err:
                __import__('logging').getLogger("midan.pipeline").warning(
                    "[SLACK] webhook POST failed (%s: %r)",
                    type(_slack_err).__name__, _slack_err,
                )
                logs.append(f"[SLACK] Webhook failed ({type(_slack_err).__name__})")
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
        'drift_flag':       drift_flag,                  # legacy alias of train_time_drift
        'action_fired':     action_fired,
        # ── L2 transparency surface (Tier A + Tier B) ─────────────────────
        'l2_macro_base':       macro_base,
        'l2_macro_adjusted':   macro_adjusted,
        'l2_idea_adjustments': idea_adjustments,
        'l2_applied_deltas':   applied_deltas,
        'l2_data_freshness':   freshness,
        'regime_decision_path': decision_path,
        'fcm_membership':      fcm_signal,
        # ── L3 structured reasoning (PRIMARY explanation layer) ───────────
        'l3_reasoning':        l3_reasoning,
        # ── L4 decision engine (PRIMARY decision mechanism — TAS is legacy) ──
        'l4_decision':         l4_decision,
        # ── Tier 3 grounding: per-text-field signal anchors ────────────────
        'strategic_anchors':   strategic_anchors,
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
        'idea_features_result': idea_features_result,
        'idea_signal':      idea_signal_data,
        'combined_signals': combined_signals,
        'top_macro_signals':top3_names,
        'dominant_risk':    dominant_risk,
        'x_raw':    x_raw[0],
        'x_scaled': x_scaled[0],
        'x_pca':    x_pca[0],
        'proba':    dict(zip(le.classes_, proba)),
        # ── Mechanism analysis (post-L3, feeds L4 probabilistically) ──────
        'mechanism_analysis': serialize_envelope(mechanism_envelope),
    }

def _build_invalid_response(l0: dict) -> dict:
    """Standard rejection envelope used when L0 blocks an input."""
    severity_to_badge = {
        'IMPOSSIBLE':  "INVALID — NOT A VIABLE BUSINESS CONCEPT",
        'BROKEN':      "INVALID — STRUCTURAL FAILURE",
        'INCOMPLETE':  "CLARIFICATION REQUIRED — UNDER-DEFINED INPUT",
    }
    return {
        "success":                False,
        "invalid_idea":           True,
        "severity":               l0.get('severity', 'BROKEN'),
        "rejection_type":         l0['rejection_type'],
        "message":                l0['message'],
        "one_line_verdict":       l0['one_line_verdict'],
        "what_is_missing":        l0['what_is_missing'],
        "how_to_fix":             l0.get('how_to_fix', []),
        "logical_validity_score": round(l0['logical_validity_score'], 2),
        "rejection_confidence":   int(l0['rejection_confidence'] * 100),
        "decision_badge":         severity_to_badge.get(l0.get('severity', 'BROKEN'), "INVALID"),
        "tas_score":              0,
        "signal_tier":            "Invalid",
        "quadrant":               "STOP — Rethink Everything",
        "svs":                    0,
    }


def _build_clarification_response(idea_text: str, sector_key: str, l1_result: dict) -> dict:
    """
    Fail-fast envelope for when L1 extraction is insufficient or inconsistent.
    L2–L4 do NOT run. The caller (frontend / chat) is expected to surface
    `clarification.questions` to the user and resubmit when answered.
    """
    is_inconsistent = not l1_result["consistency"]["ok"]
    rejection_type  = 'l1_inconsistent_schema' if is_inconsistent else 'l1_insufficient_confidence'
    clarification = _l1_clarification_message(l1_result)
    msg = (
        "Idea schema is inconsistent — model, segment, and monetization conflict."
        if is_inconsistent else
        "Idea is parseable but key fields are below confidence threshold. "
        "MIDAN halts here rather than guess. Provide the missing components."
    )
    return {
        "success":                False,
        "invalid_idea":           False,
        "clarification_required": True,
        "severity":               'INCOMPLETE',
        "rejection_type":         rejection_type,
        "message":                msg,
        "one_line_verdict":       (
            "L1 schema inconsistent — clarify the anchor."
            if is_inconsistent else
            "L1 confidence below threshold — clarify the missing fields."
        ),
        "what_is_missing":        ", ".join(l1_result["unknown_required"]) or "internal consistency",
        "how_to_fix":             _l0_how_to_fix(idea_text, rejection_type),
        "clarification":          clarification,
        "l1_aggregate_confidence": l1_result["aggregate_confidence"],
        "l1_unknown_required":    l1_result["unknown_required"],
        "l1_consistency":         l1_result["consistency"],
        "logical_validity_score": 0.40,
        "rejection_confidence":   75,
        "decision_badge":         "CLARIFICATION REQUIRED",
        "tas_score":              0,
        "signal_tier":            "Insufficient",
        "quadrant":               "STOP — Clarify Before Analysis",
        "svs":                    0,
        "sector":                 sector_key,
    }


def process_idea(idea_text: str, default_sector: str = "fintech", default_country: str = "EG") -> dict:
    # ── Layer 0: Sanity gate — strict, blocking before any ML pipeline ────────
    l0 = _layer0_sanity_check(idea_text)
    if not l0.get('valid', False):
        return _build_invalid_response(l0)

    # ── Agent A1: sector + country parse (no analysis yet) ────────────────────
    parsed_sec, parsed_ctry, sec_found, ctry_found = agent_a1_parse(idea_text)
    sector_key   = parsed_sec  if sec_found  else default_sector
    country_code = parsed_ctry if ctry_found else default_country

    # ── Layer 1: Confidence-scored extraction + cross-field consistency ───────
    # If extraction is insufficient OR fields are inconsistent, the pipeline
    # halts here — L2/L3/L4 must NEVER run on UNKNOWN or contradictory inputs.
    l1_result = extract_idea_features(idea_text, sector_key)
    if not l1_result["is_sufficient"]:
        _L1_LOG.info(
            f"[L1] HALT — unknown_required={l1_result['unknown_required']} "
            f"agg_conf={l1_result['aggregate_confidence']} "
            f"consistency_ok={l1_result['consistency']['ok']} "
            f"idea='{idea_text[:60]}'"
        )
        return _build_clarification_response(idea_text, sector_key, l1_result)

    # ── L2/L3/L4 inference uses the validated L1 envelope ─────────────────────
    report    = run_inference(
        sector_key, country_code,
        idea_text=idea_text,
        idea_features_result=l1_result,
    )
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
        "idea":            idea_text,
        # ── L0 always passes here — INCOMPLETE is now blocking. Fields kept None
        # for response-shape stability with existing frontend consumers.
        "l0_flag":         None,
        "l0_verdict":      None,
        "l0_what_is_missing": None,
        "l0_how_to_fix":   [],
        # ── L1 confidence metadata — first-class in the response ───────────
        # idea_features_raw preserves UNKNOWN sentinels for low-confidence
        # non-required fields. idea_features (below) is the runtime view
        # (neutral defaults applied where idea_features_raw says UNKNOWN).
        "l1_aggregate_confidence": l1_result["aggregate_confidence"],
        "l1_field_confidence":     l1_result["confidence"],
        "l1_field_source":         l1_result["source"],
        "l1_consistency":          l1_result["consistency"],
        "idea_features_raw":       l1_result["values"],
        "sector":          sector_key,
        "country":         country_code,
        "regime":          report["regime"],
        # ── PRIMARY decision surface (L4 state machine) ─────────────────
        # decision_state replaces decision_badge as the decision basis.
        # decision_strength (qualitative tier) replaces numeric `confidence`.
        # See l4_decision below for full reasoning trace.
        "decision_state":     report["l4_decision"]["decision_state"],
        "decision_strength":  report["l4_decision"]["decision_strength"],
        "l4_decision":        report["l4_decision"],
        # ── Per-text-field signal anchors (Tier 3 grounding) ────────────
        "strategic_anchors":  report["strategic_anchors"],
        # ── Legacy numeric fields — preserved for back-compat only ──────
        # These have ZERO influence on decision_state. Frontend should
        # migrate to decision_state + decision_strength.
        "tas_score":       int(report["tas"] * 100),
        "signal_tier":     report["signal_tier"],
        "decision_badge":  report.get("decision_badge", ""),  # legacy text badge
        "sarima_trend":    report["sarima_trend"],
        "drift_flag":      report["drift_flag"],          # legacy alias of train_time_drift
        "action_fired":    report["action_fired"],
        # ── L2 transparency surface ────────────────────────────────────
        # base = pure (sector, country) lookup, source=static_table
        # adjusted = base + capped idea-derived deltas, deltas listed below
        # adjustments = explicit traceable list (each marked source="inferred")
        # decision_path = SVM step → optional rule_override → optional staleness
        # fcm_membership = parallel fuzzy regime signal vs. SVM hard label
        # data_freshness = staleness, drift, source-of-truth metadata
        "l2_macro_base":         report["l2_macro_base"],
        "l2_macro_adjusted":     report["l2_macro_adjusted"],
        "l2_idea_adjustments":   report["l2_idea_adjustments"],
        "l2_applied_deltas":     report["l2_applied_deltas"],
        "l2_data_freshness":     report["l2_data_freshness"],
        "regime_decision_path":  report["regime_decision_path"],
        "fcm_membership":        report["fcm_membership"],
        # ── L3 structured reasoning (PRIMARY explanation surface) ─────
        # Replaces scalar idea_signal as the explanation layer. The scalar
        # is preserved inside l3_reasoning.legacy_scalar_signal and in the
        # idea_signal field below for L4 TAS continuity.
        "l3_reasoning":          report["l3_reasoning"],
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
        # ── Mechanism analysis (serialized MechanismEnvelope) ────────────
        "mechanism_analysis":      report.get("mechanism_analysis", {}),
        # epistemic_disclosure: one sentence from EpistemicSummary, always present.
        # Surfaced separately so UI and chat layer can prepend it without parsing.
        "epistemic_disclosure": (
            (report.get("mechanism_analysis") or {})
            .get("epistemic_summary", {})
            .get("recommended_disclosure", "")
        ),
    }



# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]

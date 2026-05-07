"""
midan.l4_decision — decision engine.

Replaces the linear TAS-based decision with a structured state machine
driven by risk decomposition (market/execution/timing), conflict detection
(severity-tiered), offsetting analysis (reasoning dominance), and a
qualitative decision_strength tier. TAS preserved as legacy_tas_score with
explicit zero-decision-influence note.
"""
from midan.core import *  # noqa: F401,F403
from midan.l3_reasoning import _l3_field_known, _safe_int  # noqa: F401


# ── extracted from api.py ─────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# LAYER 4 — DECISION ENGINE
#
# This module replaces the linear TAS-based decision with a structured
# decision state machine driven by:
#   • risk_decomposition       — market / execution / timing, separately scored
#   • offsetting analysis       — strong signals can downgrade a high risk
#   • conflict_detection        — explicit rules, severity-tiered (low/medium/high)
#   • decision_quality          — input completeness, signal agreement, assumption density
#   • decision_strength         — qualitative tier replacing numeric confidence
#
# Hard contracts:
#   • Decision is derived from rules, NOT from TAS arithmetic.
#   • TAS is preserved as `legacy_tas_score` with zero decision influence.
#   • Failure modes (INSUFFICIENT_DATA, HIGH_UNCERTAINTY, CONFLICTING_SIGNALS)
#     prevent forced verdicts.
#   • Every decision step references the exact L1/L2/L3 signals it uses.
# ═══════════════════════════════════════════════════════════════

# L4_DECISION_VERSION and the ENABLE_OFFSETTING / ENABLE_CONFLICT_DETECTION
# toggles are owned by midan.config and reach this module via
# `from midan.core import *`.

# ── Risk levels and tier ladder ─────────────────────────────────────────────
_RISK_LADDER  = ['low', 'medium', 'high']
_RISK_RANK    = {'low': 0, 'medium': 1, 'high': 2}


def _bump_risk(level: str, steps: int) -> str:
    """Move a risk level up/down the ladder by N steps; saturate at boundaries."""
    idx = _RISK_RANK.get(level, 1) + steps
    return _RISK_LADDER[max(0, min(len(_RISK_LADDER) - 1, idx))]


# ── ANALYZER 1: Market risk (grounded in L2) ────────────────────────────────
def _decompose_market_risk(regime: str, regime_conf: float, fcm_membership: dict,
                            l2_freshness: dict) -> dict:
    """
    Market risk reflects the macro environment as seen by L2.
    Drivers: regime label, regime confidence, FCM ambiguity, staleness flag.
    """
    drivers = []
    base_level = 'medium'
    if regime == 'GROWTH_MARKET':
        base_level = 'low'
        drivers.append({'signal': 'L2.regime', 'value': regime, 'effect': 'low_market_risk'})
    elif regime == 'EMERGING_MARKET':
        base_level = 'medium'
        drivers.append({'signal': 'L2.regime', 'value': regime, 'effect': 'moderate_market_risk'})
    elif regime == 'HIGH_FRICTION_MARKET':
        base_level = 'high'
        drivers.append({'signal': 'L2.regime', 'value': regime, 'effect': 'high_market_risk'})
    elif regime == 'CONTRACTING_MARKET':
        base_level = 'high'
        drivers.append({'signal': 'L2.regime', 'value': regime, 'effect': 'high_market_risk'})

    # FCM ambiguity bumps risk one notch up — uncertain regime ≠ confident regime.
    fcm_ambiguous = bool(fcm_membership and fcm_membership.get('is_ambiguous'))
    if fcm_ambiguous:
        drivers.append({
            'signal': 'L2.fcm_membership.is_ambiguous', 'value': True,
            'effect': 'regime_classification_uncertain — bumps risk one step',
        })
        base_level = _bump_risk(base_level, +1)

    # Staleness penalty (already applied to conf elsewhere) — surface as driver
    if l2_freshness and l2_freshness.get('runtime_staleness_flag'):
        drivers.append({
            'signal': 'L2.l2_data_freshness.runtime_staleness_flag', 'value': True,
            'effect': f"forecasts {l2_freshness.get('sarima_days_stale')}d stale — confidence in regime reduced",
        })

    reasoning = (
        f"Market risk anchored to L2.regime={regime} (confidence={regime_conf:.2f}). "
        f"{'FCM ambiguity raises uncertainty about the regime label. ' if fcm_ambiguous else ''}"
        f"Risk level '{base_level}' is derived from the regime category and runtime data freshness."
    )
    return {
        'level':       base_level,
        'drivers':     drivers,
        'reasoning':   reasoning,
        'evidence_grounded_in': {
            'l1_fields_used': [],
            'l2_fields_used': ['regime', 'fcm_membership', 'l2_data_freshness'],
            'l3_fields_used': [],
        },
    }


# ── ANALYZER 2: Execution risk (grounded in L3) ─────────────────────────────
def _decompose_execution_risk(l1_values: dict, l1_confidence: dict, l3_reasoning: dict) -> dict:
    """
    Execution risk reflects the do-ability of the model: BM viability, CAC/RPU
    pressure, scalability shape, operational complexity.
    """
    drivers = []
    base_level = 'medium'
    bm_block = l3_reasoning.get('business_model', {})
    ue_block = l3_reasoning.get('unit_economics', {})

    if not bm_block.get('available'):
        drivers.append({'signal': 'L3.business_model.available', 'value': False,
                        'effect': 'execution risk indeterminate — BM unknown'})
        return {
            'level': 'unknown',
            'drivers': drivers,
            'reasoning': 'Execution risk cannot be assessed — L3 business_model module is unavailable.',
            'evidence_grounded_in': {
                'l1_fields_used': [], 'l2_fields_used': [],
                'l3_fields_used': ['business_model.available'],
            },
        }

    # Operational complexity → primary driver
    op_complex = bm_block.get('cost_structure', {}).get('operational_complexity', 'medium')
    drivers.append({'signal': 'L3.business_model.cost_structure.operational_complexity',
                    'value': op_complex, 'effect': f'op complexity={op_complex}'})
    if op_complex == 'high':
        base_level = 'high'
    elif op_complex == 'low':
        base_level = 'low'

    # CAC tier
    if ue_block.get('available'):
        cac_tier = ue_block.get('cac_proxy', {}).get('tier')
        rpu_tier = ue_block.get('revenue_per_user_proxy', {}).get('tier')
        scal     = ue_block.get('scalability_pressure', {}).get('tier')

        if cac_tier == 'high':
            drivers.append({'signal': 'L3.unit_economics.cac_proxy', 'value': cac_tier,
                            'effect': 'high CAC pressure → execution risk +1'})
            base_level = _bump_risk(base_level, +1)
        if rpu_tier == 'low':
            drivers.append({'signal': 'L3.unit_economics.revenue_per_user_proxy', 'value': rpu_tier,
                            'effect': 'low RPU → execution risk +1'})
            base_level = _bump_risk(base_level, +1)
        if scal == 'worsens_with_scale':
            drivers.append({'signal': 'L3.unit_economics.scalability_pressure', 'value': scal,
                            'effect': 'cost grows with scale → execution risk +1'})
            base_level = _bump_risk(base_level, +1)
        elif scal == 'improves_with_scale':
            drivers.append({'signal': 'L3.unit_economics.scalability_pressure', 'value': scal,
                            'effect': 'cost amortizes with scale → execution risk -1'})
            base_level = _bump_risk(base_level, -1)

    reasoning = (
        f"Execution risk derived from BM op-complexity ({op_complex}) and L3 unit economics. "
        f"Level '{base_level}' reflects {'+1 each per high-CAC/low-RPU/worsens-with-scale signal' if drivers else 'no L3 signals available'}."
    )
    return {
        'level':       base_level,
        'drivers':     drivers,
        'reasoning':   reasoning,
        'evidence_grounded_in': {
            'l1_fields_used': [],
            'l2_fields_used': [],
            'l3_fields_used': ['business_model.cost_structure.operational_complexity',
                               'unit_economics.cac_proxy', 'unit_economics.revenue_per_user_proxy',
                               'unit_economics.scalability_pressure'],
        },
    }


# ── ANALYZER 3: Timing risk (L1 stage + L1 readiness + L2 regime) ───────────
def _decompose_timing_risk(l1_values: dict, l1_confidence: dict, regime: str) -> dict:
    """
    Timing risk reflects whether the idea is at the right moment: stage vs.
    macro alignment vs. market readiness signal.
    """
    drivers = []
    base_level = 'medium'

    stage_known = _l3_field_known(l1_values, l1_confidence, 'stage')
    ready_known = _l3_field_known(l1_values, l1_confidence, 'market_readiness')

    if not stage_known:
        return {
            'level': 'unknown',
            'drivers': [{'signal': 'L1.stage', 'value': 'UNKNOWN',
                         'effect': 'cannot assess timing without stage'}],
            'reasoning': 'Timing risk cannot be assessed — L1.stage is UNKNOWN or below confidence floor.',
            'evidence_grounded_in': {
                'l1_fields_used': ['stage'], 'l2_fields_used': [],
                'l3_fields_used': [],
            },
        }

    stage = l1_values['stage']
    drivers.append({'signal': 'L1.stage', 'value': stage,
                    'effect': f'stage={stage}'})

    # Early stage in hostile regime is structural mistiming
    hostile = regime in ('CONTRACTING_MARKET', 'HIGH_FRICTION_MARKET')
    if stage in ('idea', 'validation') and hostile:
        drivers.append({'signal': 'L2.regime', 'value': regime,
                        'effect': 'early stage in hostile regime → timing risk +1'})
        base_level = _bump_risk(base_level, +1)
    elif stage == 'growth' and hostile:
        drivers.append({'signal': 'L2.regime', 'value': regime,
                        'effect': 'growth-stage idea in contracting macro → severe mismatch'})
        base_level = 'high'
    elif stage in ('mvp', 'growth') and regime == 'GROWTH_MARKET':
        drivers.append({'signal': 'L2.regime', 'value': regime,
                        'effect': 'mature stage aligned with growth regime → timing risk -1'})
        base_level = _bump_risk(base_level, -1)

    if ready_known:
        ready = _safe_int(l1_values.get('market_readiness'), 3)
        if ready <= 2:
            drivers.append({'signal': 'L1.market_readiness', 'value': ready,
                            'effect': 'low readiness → timing risk +1'})
            base_level = _bump_risk(base_level, +1)
        elif ready >= 4:
            drivers.append({'signal': 'L1.market_readiness', 'value': ready,
                            'effect': 'strong readiness → timing risk -1'})
            base_level = _bump_risk(base_level, -1)

    reasoning = (
        f"Timing risk derives from L1.stage={stage}, L2.regime={regime}, "
        f"and L1.market_readiness={l1_values.get('market_readiness') if ready_known else 'unknown'}. "
        f"Level '{base_level}' reflects stage/macro alignment plus readiness signal."
    )
    return {
        'level':     base_level,
        'drivers':   drivers,
        'reasoning': reasoning,
        'evidence_grounded_in': {
            'l1_fields_used': ['stage'] + (['market_readiness'] if ready_known else []),
            'l2_fields_used': ['regime'],
            'l3_fields_used': [],
        },
    }


# ── OFFSETTING ANALYZER (reasoning dominance) ───────────────────────────────
# A "high" risk on a single dimension can be downgraded to "elevated_with_offset"
# when an explicitly-named strong signal compensates. Each offset is logged.
def _apply_offsetting(risks: dict, l1_values: dict, l1_confidence: dict,
                       l3_reasoning: dict) -> tuple:
    """
    Returns (adjusted_risks, offsetting_log).
    offsetting_log is a list of {risk_dim, original_level, adjusted_level,
    offsetting_factor, evidence_signals} entries.
    """
    log = []
    out = {k: dict(v) for k, v in risks.items()}

    # Strong differentiation can offset high market risk
    diff_block = l3_reasoning.get('differentiation', {})
    if (diff_block.get('available')
        and diff_block.get('verdict') == 'structural'
        and out['market_risk']['level'] == 'high'):
        out['market_risk']['level'] = 'elevated_with_offset'
        log.append({
            'risk_dim':          'market_risk',
            'original_level':    'high',
            'adjusted_level':    'elevated_with_offset',
            'offsetting_factor': 'structural_differentiation',
            'reasoning':         'Structural differentiation creates pricing power and switching costs that compensate for hostile macro conditions.',
            'evidence_signals':  ['L3.differentiation.verdict=structural'],
        })

    # Strong scalability + improving unit economics can offset high execution risk
    ue_block = l3_reasoning.get('unit_economics', {})
    scal_tier = (ue_block.get('scalability_pressure') or {}).get('tier') if ue_block.get('available') else None
    rpu_tier  = (ue_block.get('revenue_per_user_proxy') or {}).get('tier') if ue_block.get('available') else None
    if (scal_tier == 'improves_with_scale'
        and rpu_tier == 'high'
        and out['execution_risk']['level'] == 'high'):
        out['execution_risk']['level'] = 'elevated_with_offset'
        log.append({
            'risk_dim':          'execution_risk',
            'original_level':    'high',
            'adjusted_level':    'elevated_with_offset',
            'offsetting_factor': 'scalable_high_rpu_unit_economics',
            'reasoning':         'Improving-with-scale + high RPU compounds gross margin as the business grows, partially offsetting high op complexity.',
            'evidence_signals':  ['L3.unit_economics.scalability_pressure=improves_with_scale',
                                  'L3.unit_economics.revenue_per_user_proxy=high'],
        })

    # Strong differentiation can also offset high timing risk in early stages
    if (diff_block.get('available')
        and diff_block.get('verdict') == 'structural'
        and out['timing_risk']['level'] == 'high'
        and l1_values.get('stage') in ('idea', 'validation')):
        out['timing_risk']['level'] = 'elevated_with_offset'
        log.append({
            'risk_dim':          'timing_risk',
            'original_level':    'high',
            'adjusted_level':    'elevated_with_offset',
            'offsetting_factor': 'structural_differentiation_early_stage',
            'reasoning':         'Structural differentiation at the idea/validation stage means the timing window can be widened by category creation, not just caught.',
            'evidence_signals':  ['L3.differentiation.verdict=structural', 'L1.stage'],
        })

    return out, log


# ── CONFLICT DETECTOR (severity-tiered, rule-based) ─────────────────────────
# Each conflict is a structural tension between two or more signals. Only
# severity='high' AND resolution_required=True will escalate the decision
# state to CONFLICTING_SIGNALS. Lower severities feed into reasoning but
# don't halt.
def _detect_conflicts(l1_values: dict, l1_confidence: dict, regime: str,
                       l3_reasoning: dict) -> list:
    conflicts = []

    diff_block = l3_reasoning.get('differentiation', {})
    diff_verdict = diff_block.get('verdict') if diff_block.get('available') else None
    bm_block   = l3_reasoning.get('business_model', {})
    ue_block   = l3_reasoning.get('unit_economics', {})

    # 1. Strong macro × weak differentiation × high competition
    if (regime in ('GROWTH_MARKET', 'EMERGING_MARKET')
        and diff_verdict == 'thin'
        and l1_values.get('competitive_intensity') == 'high'):
        conflicts.append({
            'conflict_id':         'strong_macro_weak_diff_high_comp',
            'severity':            'high',
            'signals_involved':    ['L2.regime', 'L3.differentiation.verdict', 'L1.competitive_intensity'],
            'explanation':         'Favorable macro plus high competition with thin differentiation: incumbents capture the regime upside before this idea can defend a position.',
            'resolution_required': True,
            'resolution_path':     'Either deepen differentiation (toward structural) or pick a defensible niche where competition is lower.',
        })

    # 2. Strong macro × weak differentiation × low/medium competition (resolvable)
    if (regime in ('GROWTH_MARKET', 'EMERGING_MARKET')
        and diff_verdict == 'thin'
        and l1_values.get('competitive_intensity') in ('low', 'medium')):
        conflicts.append({
            'conflict_id':         'strong_macro_weak_diff_resolvable',
            'severity':            'medium',
            'signals_involved':    ['L2.regime', 'L3.differentiation.verdict'],
            'explanation':         'Favorable macro paired with thin differentiation. The window to develop differentiation is open while competition is low/medium.',
            'resolution_required': False,
            'resolution_path':     'Use the macro tailwind to fund and ship differentiation work before competition catches up.',
        })

    # 3. Growth-stage idea × contracting macro
    if (l1_values.get('stage') == 'growth'
        and regime in ('CONTRACTING_MARKET', 'HIGH_FRICTION_MARKET')):
        conflicts.append({
            'conflict_id':         'growth_stage_contracting_macro',
            'severity':            'high',
            'signals_involved':    ['L1.stage', 'L2.regime'],
            'explanation':         'Growth-stage execution requires expansion capital and demand momentum that the current macro does not provide.',
            'resolution_required': True,
            'resolution_path':     'Either reset to capital-efficient unit economics for survival mode, or wait out the regime.',
        })

    # 4. Strong BM fit × high regulatory risk (resolvable, not blocking)
    if (bm_block.get('available')
        and l1_values.get('regulatory_risk') == 'high'):
        conflicts.append({
            'conflict_id':         'viable_model_high_regulatory_burden',
            'severity':            'medium',
            'signals_involved':    ['L3.business_model.available', 'L1.regulatory_risk'],
            'explanation':         'BM is viable on its merits but high regulatory risk imposes a license/compliance lead time before execution can begin.',
            'resolution_required': False,
            'resolution_path':     'Map the regulatory path explicitly; treat licensing as the critical path, not a side stream.',
        })

    # 5. B2C × high CAC × low RPU (unit economics broken)
    if ue_block.get('available'):
        cac = (ue_block.get('cac_proxy') or {}).get('tier')
        rpu = (ue_block.get('revenue_per_user_proxy') or {}).get('tier')
        if l1_values.get('target_segment') == 'b2c' and cac == 'high' and rpu == 'low':
            conflicts.append({
                'conflict_id':         'b2c_high_cac_low_rpu',
                'severity':            'high',
                'signals_involved':    ['L1.target_segment', 'L3.unit_economics.cac_proxy',
                                        'L3.unit_economics.revenue_per_user_proxy'],
                'explanation':         'B2C unit economics are structurally inverted: acquisition costs more than each user returns.',
                'resolution_required': True,
                'resolution_path':     'Pivot to B2B, change monetization to subscription/commission, or find a structurally cheaper acquisition channel.',
            })

    return conflicts


# ── DECISION QUALITY + STRENGTH ─────────────────────────────────────────────
def _assess_decision_quality(l1_result: dict, l2_freshness: dict, l3_reasoning: dict,
                              risks: dict, conflicts: list,
                              mechanism_uncertainty: float = 0.0) -> dict:
    """
    Three structured tiers + an overall_uncertainty state. Replaces the
    headline numeric confidence. No collapsed scores — every tier is paired
    with an explicit basis string.
    """
    # Input completeness — driven by L1 aggregate confidence + count of UNKNOWNs
    agg_conf = (l1_result or {}).get('aggregate_confidence', 0.0)
    unknown_required = (l1_result or {}).get('unknown_required', [])
    if agg_conf >= 0.75 and not unknown_required:
        ic_tier = 'high'
    elif agg_conf >= 0.55 and len(unknown_required) <= 1:
        ic_tier = 'medium'
    else:
        ic_tier = 'low'
    ic_basis = (
        f"L1.aggregate_confidence={agg_conf:.2f}; "
        f"L1.unknown_required={unknown_required or 'none'}."
    )

    # Signal agreement — count of layers in agreement vs disagreement
    agreement_signals = []
    bm_avail = l3_reasoning.get('business_model', {}).get('available', False)
    if bm_avail:
        agreement_signals.append('L3.business_model.available')
    if not l3_reasoning.get('insufficient_information'):
        agreement_signals.append('L3.no_insufficient_modules')
    high_conflicts = sum(1 for c in conflicts if c.get('severity') == 'high')
    if high_conflicts == 0:
        agreement_signals.append('L4.no_high_severity_conflicts')

    n_agree = len(agreement_signals)
    if n_agree >= 3:
        sa_tier = 'high'
    elif n_agree == 2:
        sa_tier = 'medium'
    else:
        sa_tier = 'low'
    sa_basis = (
        f"{n_agree}/3 signal-agreement checks passed: "
        f"L3 BM available, no insufficient modules, no high-severity conflicts."
    )

    # Assumption density — count heuristic-source signals
    heuristic_count = 0
    grounded_count = 0
    if l3_reasoning.get('differentiation', {}).get('available'):
        heuristic_count += 1   # sector_baseline_heuristic
    if l3_reasoning.get('competition', {}).get('available'):
        heuristic_count += 1   # heuristic_sector_map
    if l3_reasoning.get('business_model', {}).get('available'):
        heuristic_count += 1   # heuristic_per_bm_template
    if l3_reasoning.get('unit_economics', {}).get('available'):
        heuristic_count += 1   # qualitative_proxy
    # Grounded signals: trained model outputs (regime), staleness gate
    grounded_count += 1  # regime is from trained SVM
    if not l2_freshness.get('runtime_staleness_flag', False):
        grounded_count += 1  # forecasts within freshness window
    total = heuristic_count + grounded_count
    h_ratio = heuristic_count / total if total else 1.0
    if h_ratio >= 0.7:
        ad_tier = 'high'
    elif h_ratio >= 0.5:
        ad_tier = 'medium'
    else:
        ad_tier = 'low'
    ad_basis = (
        f"{heuristic_count} heuristic signals vs {grounded_count} grounded signals "
        f"→ {h_ratio:.0%} heuristic share."
    )

    # Overall uncertainty — derived from the three tiers + L2 staleness
    # + mechanism_uncertainty as a continuous probabilistic modifier.
    #
    # mechanism_uncertainty ranges 0.0–0.30 (max). Mapped to 0.0–1.0 bad-tier
    # contribution so it shifts the distribution rather than hard-switching state:
    #   0.10 → +0.33 contribution (not enough alone to push to high)
    #   0.20 → +0.67 contribution (significant push, needs one other bad signal)
    #   0.30 → +1.00 contribution (equivalent to one bad tier → moderate alone)
    # This preserves ambiguity when ambiguity is structurally real.
    discrete_bad = sum(1 for t in (ic_tier, sa_tier) if t == 'low') + (1 if ad_tier == 'high' else 0)
    mech_contribution = mechanism_uncertainty / 0.30  # 0.0–1.0 continuous
    bad_tiers_continuous = discrete_bad + mech_contribution
    if bad_tiers_continuous >= 2 or l2_freshness.get('runtime_staleness_flag'):
        overall = 'high'
    elif bad_tiers_continuous >= 1:
        overall = 'moderate'
    else:
        overall = 'low'

    return {
        'input_completeness':  {'tier': ic_tier, 'basis': ic_basis},
        'signal_agreement':    {'tier': sa_tier, 'basis': sa_basis},
        'assumption_density':  {'tier': ad_tier, 'basis': ad_basis},
        'overall_uncertainty': overall,
    }


def _decision_strength_tier(quality: dict) -> dict:
    """
    Qualitative replacement for numeric confidence. Derived from decision_quality.
    Tiers: strong / moderate / weak / uncertain.
    """
    ic = quality['input_completeness']['tier']
    sa = quality['signal_agreement']['tier']
    ad = quality['assumption_density']['tier']
    ou = quality['overall_uncertainty']

    if ou == 'high':
        tier = 'uncertain'
        basis = 'overall_uncertainty=high — at least one input quality dimension is low or signals are stale.'
    elif ic == 'high' and sa == 'high' and ad in ('low', 'medium'):
        tier = 'strong'
        basis = 'High input completeness + high signal agreement + grounded-signal share is sufficient.'
    elif ic in ('high', 'medium') and sa in ('high', 'medium'):
        tier = 'moderate'
        basis = 'Inputs and signal agreement are at least medium; reasoning is supported but not definitive.'
    else:
        tier = 'weak'
        basis = 'Inputs or signal agreement are low; conclusions should be treated as provisional.'

    return {'tier': tier, 'basis': basis}


# ── DECISION STATE MACHINE ──────────────────────────────────────────────────
def _derive_decision_state(risks: dict, quality: dict, conflicts: list,
                            l3_reasoning: dict) -> dict:
    """
    Apply the decision rules in priority order. Each transition is logged.
    Returns {state, state_reasoning, decision_reasoning_steps}.
    """
    steps = []

    # Rule 0 — INSUFFICIENT_DATA: any required L3 module unavailable
    insufficient = l3_reasoning.get('insufficient_information', [])
    if insufficient:
        modules = ", ".join(i.get('module', '?') for i in insufficient)
        steps.append({
            'step':       'check_insufficient_data',
            'rule_id':    'r0_insufficient_data',
            'evidence':   [f"L3.insufficient_information=[{modules}]"],
            'conclusion': 'INSUFFICIENT_DATA — required L3 modules unavailable; decision halted.',
        })
        return {
            'state': 'INSUFFICIENT_DATA',
            'state_reasoning': f"Cannot decide: L3 modules unavailable: {modules}.",
            'decision_reasoning_steps': steps,
        }
    steps.append({
        'step':       'check_insufficient_data',
        'rule_id':    'r0_insufficient_data',
        'evidence':   ['L3.insufficient_information=[]'],
        'conclusion': 'All required L3 modules available — proceed.',
    })

    # Rule 1 — HIGH_UNCERTAINTY: overall uncertainty is high → advisory only
    if quality['overall_uncertainty'] == 'high':
        steps.append({
            'step':       'check_high_uncertainty',
            'rule_id':    'r1_high_uncertainty',
            'evidence':   [f"decision_quality.overall_uncertainty=high",
                           f"input_completeness={quality['input_completeness']['tier']}",
                           f"signal_agreement={quality['signal_agreement']['tier']}"],
            'conclusion': 'HIGH_UNCERTAINTY — verdict is advisory; do not force a binding decision.',
        })
        return {
            'state': 'HIGH_UNCERTAINTY',
            'state_reasoning': "Verdict is advisory only — overall uncertainty is high.",
            'decision_reasoning_steps': steps,
        }
    steps.append({
        'step':       'check_high_uncertainty',
        'rule_id':    'r1_high_uncertainty',
        'evidence':   [f"decision_quality.overall_uncertainty={quality['overall_uncertainty']}"],
        'conclusion': 'Uncertainty within decisionable range — proceed.',
    })

    # Rule 2 — CONFLICTING_SIGNALS: only when high-severity AND unresolved
    severe_unresolved = [c for c in conflicts
                          if c.get('severity') == 'high' and c.get('resolution_required')]
    if severe_unresolved:
        ids = [c['conflict_id'] for c in severe_unresolved]
        steps.append({
            'step':       'check_severe_conflicts',
            'rule_id':    'r2_conflicting_signals',
            'evidence':   [f"high-severity unresolved conflicts: {ids}"],
            'conclusion': 'CONFLICTING_SIGNALS — at least one high-severity conflict requires resolution before a decision can be made.',
        })
        return {
            'state': 'CONFLICTING_SIGNALS',
            'state_reasoning': f"Severe unresolved conflicts: {', '.join(ids)}.",
            'decision_reasoning_steps': steps,
        }
    steps.append({
        'step':       'check_severe_conflicts',
        'rule_id':    'r2_conflicting_signals',
        'evidence':   [f"high-severity-unresolved-conflicts={len(severe_unresolved)}"],
        'conclusion': 'No severe unresolved conflicts — proceed.',
    })

    # Risk count after offsetting. "elevated_with_offset" counts as medium for state derivation.
    def _effective_level(level):
        return 'medium' if level == 'elevated_with_offset' else level

    eff_levels = {dim: _effective_level(risks[dim]['level']) for dim in ('market_risk', 'execution_risk', 'timing_risk')}
    raw_levels = {dim: risks[dim]['level'] for dim in ('market_risk', 'execution_risk', 'timing_risk')}
    n_high = sum(1 for v in eff_levels.values() if v == 'high')
    n_low  = sum(1 for v in eff_levels.values() if v == 'low')

    steps.append({
        'step':       'count_risk_levels_after_offset',
        'rule_id':    'r3_risk_profile',
        'evidence':   [f"raw_levels={raw_levels}",
                       f"effective_levels={eff_levels}",
                       f"n_high={n_high}", f"n_low={n_low}"],
        'conclusion': 'Risk profile aggregated from three dimensions (post-offset).',
    })

    # Rule 3 — NO_GO: ≥2 effective-high risks
    if n_high >= 2:
        steps.append({
            'step':       'apply_no_go_rule',
            'rule_id':    'r3a_no_go',
            'evidence':   [f"n_high={n_high} (>= 2)"],
            'conclusion': f'NO_GO — at least two risk dimensions are unmitigated-high: {[d for d, v in eff_levels.items() if v == "high"]}.',
        })
        return {
            'state': 'NO_GO',
            'state_reasoning': f"Two or more risk dimensions are unmitigated-high.",
            'decision_reasoning_steps': steps,
        }

    # Rule 4 — GO: zero high risks AND ≥1 low risk AND no medium-severity-or-higher unresolved conflict
    medium_unresolved = [c for c in conflicts
                          if c.get('severity') == 'medium' and c.get('resolution_required')]
    if n_high == 0 and n_low >= 1 and not medium_unresolved:
        steps.append({
            'step':       'apply_go_rule',
            'rule_id':    'r3c_go',
            'evidence':   [f"n_high=0", f"n_low={n_low}",
                           f"medium_unresolved_conflicts={len(medium_unresolved)}"],
            'conclusion': 'GO — no high-risk dimensions and at least one low risk; no unresolved medium conflicts.',
        })
        return {
            'state': 'GO',
            'state_reasoning': "No risk dimension is high; at least one is low.",
            'decision_reasoning_steps': steps,
        }

    # Rule 5 — CONDITIONAL: everything else (one high OR all medium OR low+medium with conflicts)
    steps.append({
        'step':       'apply_conditional_rule',
        'rule_id':    'r3b_conditional',
        'evidence':   [f"n_high={n_high}", f"n_low={n_low}",
                       f"unresolved_conflicts={[c['conflict_id'] for c in (severe_unresolved + medium_unresolved)]}"],
        'conclusion': 'CONDITIONAL — risk profile is acceptable but conditional on resolving named risks/conflicts.',
    })
    return {
        'state': 'CONDITIONAL',
        'state_reasoning': "Risk profile is mixed — proceed only after addressing the highlighted risks/conflicts.",
        'decision_reasoning_steps': steps,
    }


# ── DECISION DERIVATION TRACE (Tier 3 — full traceability) ──────────────────
def _build_decision_derivation(l1_result: dict, regime: str, fcm: dict,
                                l3_reasoning: dict, decision_state: str) -> list:
    """
    A layer-by-layer trace of which signals informed the decision.
    """
    return [
        {
            'layer':       'L1',
            'key_signals': {
                'business_model':        l1_result.get('values', {}).get('business_model'),
                'target_segment':        l1_result.get('values', {}).get('target_segment'),
                'stage':                 l1_result.get('values', {}).get('stage'),
                'differentiation_score': l1_result.get('values', {}).get('differentiation_score'),
                'aggregate_confidence':  l1_result.get('aggregate_confidence'),
            },
        },
        {
            'layer':       'L2',
            'key_signals': {
                'regime':         regime,
                'fcm_top_cluster': fcm.get('top_cluster') if fcm else None,
                'fcm_ambiguous':  fcm.get('is_ambiguous') if fcm else None,
            },
        },
        {
            'layer':       'L3',
            'key_signals': {
                'differentiation_verdict': (l3_reasoning.get('differentiation') or {}).get('verdict'),
                'competition_pressure':    (l3_reasoning.get('competition') or {}).get('competitive_pressure'),
                'bm_available':            (l3_reasoning.get('business_model') or {}).get('available'),
                'unit_economics_available':(l3_reasoning.get('unit_economics') or {}).get('available'),
                'fired_interactions':      [i.get('interaction_id') for i in l3_reasoning.get('signal_interactions', [])],
            },
        },
        {
            'layer':       'L4',
            'key_signals': {
                'decision_state': decision_state,
            },
        },
    ]


# ── ORCHESTRATOR ────────────────────────────────────────────────────────────
def compute_l4_decision(
    l1_result: dict,
    regime: str,
    regime_conf: float,
    fcm_membership: dict,
    l2_freshness: dict,
    l3_reasoning: dict,
    legacy_tas: float,
    mechanism_uncertainty: float = 0.0,
) -> dict:
    """
    Build the structured L4 decision envelope. Replaces the linear TAS-based
    badge as the primary decision mechanism. TAS is preserved as
    `legacy_tas_score` with explicit zero-decision-influence note.
    """
    l1_values     = (l1_result or {}).get('values', {})
    l1_confidence = (l1_result or {}).get('confidence', {})

    # Risk decomposition
    market_r    = _decompose_market_risk(regime, regime_conf, fcm_membership, l2_freshness)
    execution_r = _decompose_execution_risk(l1_values, l1_confidence, l3_reasoning)
    timing_r    = _decompose_timing_risk(l1_values, l1_confidence, regime)
    risks_raw   = {'market_risk': market_r, 'execution_risk': execution_r, 'timing_risk': timing_r}

    # Offsetting (gated by midan.config.ENABLE_OFFSETTING).
    if ENABLE_OFFSETTING:
        risks_adj, offset_log = _apply_offsetting(risks_raw, l1_values, l1_confidence, l3_reasoning)
    else:
        risks_adj, offset_log = risks_raw, []

    # Conflicts (gated by midan.config.ENABLE_CONFLICT_DETECTION).
    if ENABLE_CONFLICT_DETECTION:
        conflicts = _detect_conflicts(l1_values, l1_confidence, regime, l3_reasoning)
    else:
        conflicts = []

    # Decision quality + strength
    quality   = _assess_decision_quality(
        l1_result, l2_freshness, l3_reasoning, risks_adj, conflicts,
        mechanism_uncertainty=mechanism_uncertainty,
    )
    strength  = _decision_strength_tier(quality)

    # State machine
    sm = _derive_decision_state(risks_adj, quality, conflicts, l3_reasoning)

    # Derivation trace
    derivation = _build_decision_derivation(l1_result, regime, fcm_membership, l3_reasoning, sm['state'])

    return {
        'decision_engine_version': L4_DECISION_VERSION,
        'decision_state':          sm['state'],
        'decision_state_reasoning': sm['state_reasoning'],
        'decision_strength':       strength,
        'risk_decomposition':      risks_adj,
        'offsetting_applied':      offset_log,
        'conflicting_signals':     conflicts,
        'decision_quality':        quality,
        'decision_reasoning':      sm['decision_reasoning_steps'],
        'decision_derivation':     derivation,
        'legacy_tas_score': {
            'value': round(float(legacy_tas), 3),
            'note':  'Preserved for backward compatibility only. Has ZERO influence on decision_state.',
        },
        'data_provenance': {
            'risk_decomposition_source': 'rule_based_per_dimension',
            'conflict_detection_source': 'explicit_rule_set',
            'offsetting_source':         'explicit_offset_table',
            'live_data_integration':     False,
        },
    }


def _l4_top_risk_dim(l4_decision: dict) -> tuple:
    """
    Return (dim_name, dim_block) for the highest L4 risk dimension. Used by
    chat / response builders to ground replies in the actual top risk, not a
    single legacy string.
    """
    risks = (l4_decision or {}).get('risk_decomposition', {}) or {}
    rank_map = {'low': 0, 'medium': 1, 'elevated_with_offset': 2, 'high': 3, 'unknown': -1}
    best = (None, None, -2)
    for dim_name in ('market_risk', 'execution_risk', 'timing_risk'):
        block = risks.get(dim_name) or {}
        rank = rank_map.get(block.get('level'), -2)
        if rank > best[2]:
            best = (dim_name, block, rank)
    return best[0], best[1]



# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]

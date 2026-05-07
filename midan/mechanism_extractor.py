"""
midan.mechanism_extractor — mechanism extraction pipeline.

8-phase pipeline: extractability gate → structural observation →
mechanism assignment → evidence calibration → weight normalization →
trace filling → market structure → tensions → replication →
contextual signals → uncertainty propagation → consistency check →
epistemic summary.

Design constraints:
  - No LLM calls in v1. Phase 1E fills traces deterministically.
  - effective_weight is a computed @property, never stored in any dict.
  - mechanism_uncertainty feeds L4 as a continuous probabilistic modifier.
  - EpistemicSummary is always built and always surfaced, including on
    insufficient_information paths.
  - Single responsibility per phase function. No hidden side effects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

_LOG = logging.getLogger("midan.mechanism_extractor")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MECHANISM_CATEGORIES: Dict[str, str] = {
    "network_effect":         "advantage_mechanism",
    "switching_cost":         "advantage_mechanism",
    "brand_moat":             "advantage_mechanism",
    "data_moat":              "advantage_mechanism",
    "regulatory_moat":        "advantage_mechanism",
    "cost_advantage":         "operational_mechanism",
    "process_efficiency":     "operational_mechanism",
    "distribution_control":   "distribution_mechanism",
    "api_dependency":         "constraint_mechanism",
    "platform_dependency":    "constraint_mechanism",
    "regulatory_headwind":    "constraint_mechanism",
}

# Category-level impact priors for selection scoring.
# Constraints score 0.90 — they matter even with weak evidence.
IMPACT_PRIORS: Dict[str, float] = {
    "advantage_mechanism":    1.00,
    "constraint_mechanism":   0.90,
    "distribution_mechanism": 0.85,
    "operational_mechanism":  0.70,
}

# Base weights per mechanism type — starting point, normalized in Phase 1D.
BASE_WEIGHTS: Dict[str, float] = {
    "network_effect":       0.85,
    "switching_cost":       0.75,
    "regulatory_moat":      0.80,
    "data_moat":            0.70,
    "brand_moat":           0.65,
    "distribution_control": 0.70,
    "cost_advantage":       0.60,
    "process_efficiency":   0.50,
    "api_dependency":       0.75,
    "platform_dependency":  0.80,
    "regulatory_headwind":  0.75,
}

# Observation strength levels.
# A single STRENGTH_SPECIFIC observation outweighs multiple STRENGTH_GENERIC ones
# because confidence calibration weights by strength, not just count.
STRENGTH_SPECIFIC   = 1.00   # direct structural claim, named barrier, explicit mechanism
STRENGTH_STRUCTURAL = 0.85   # structural dependency, process specificity, named signal
STRENGTH_MODERATE   = 0.55   # feature differentiation, typed customer, score-based
STRENGTH_GENERIC    = 0.25   # vague innovation language, undifferentiated claim

ASSERTIVE_LANGUAGE_CONFIDENCE_FLOOR = 0.60
MAX_CONTEXTUAL_ADJUSTMENT = 0.10

MECHANISM_CAPS: Dict[str, int] = {
    "full":                    13,
    "partial":                  6,
    "insufficient_information": 0,
}

# Dependency map prevents double-counting mechanism signals in replication scoring.
# A dependent dimension's score is discounted 50% if its dependency is also scored.
REPLICATION_DIMENSION_DEPS: Dict[str, List[str]] = {
    "time_to_replicate":      [],
    "capital_intensity":      [],
    "regulatory_barrier":     [],
    "data_accumulation_lead": ["time_to_replicate"],
    "network_density_gap":    ["time_to_replicate"],
    "talent_concentration":   ["capital_intensity"],
    "ecosystem_lock_in":      ["network_density_gap"],
}

# Per-mechanism replication dimension scores (raw, before dependency discounting).
# Constraint mechanisms score low — they limit the idea, not the competitor.
_REPLICATION_DIMS: Dict[str, Dict[str, float]] = {
    "network_effect":       {"time_to_replicate": 0.80, "network_density_gap": 0.90, "capital_intensity": 0.70},
    "switching_cost":       {"time_to_replicate": 0.65, "data_accumulation_lead": 0.75, "ecosystem_lock_in": 0.60},
    "regulatory_moat":      {"regulatory_barrier": 0.95, "time_to_replicate": 0.85},
    "data_moat":            {"data_accumulation_lead": 0.85, "time_to_replicate": 0.70, "talent_concentration": 0.60},
    "brand_moat":           {"time_to_replicate": 0.75, "capital_intensity": 0.65},
    "cost_advantage":       {"capital_intensity": 0.70, "time_to_replicate": 0.50},
    "process_efficiency":   {"time_to_replicate": 0.45, "talent_concentration": 0.55},
    "distribution_control": {"ecosystem_lock_in": 0.70, "time_to_replicate": 0.60},
    "api_dependency":       {"time_to_replicate": 0.20, "regulatory_barrier": 0.10},
    "platform_dependency":  {"time_to_replicate": 0.20, "capital_intensity": 0.25},
    "regulatory_headwind":  {"regulatory_barrier": 0.30, "time_to_replicate": 0.25},
}

# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractabilityResult:
    score: float               # 0.0–1.0 composite
    mode: str                  # "full" | "partial" | "insufficient_information"
    mechanism_cap: int         # 13 | 6 | 0
    signal_gaps: List[str]     # named missing or sparse fields


@dataclass
class StructuralObservation:
    field_source: str          # "l3.differentiation", "l1.regulatory_risk", "idea_text"
    observation_type: str      # normalized label: "differentiation_claim", "regulatory_signal"
    raw_text: str              # exact field value or phrase — never paraphrased
    structural_signal: str     # normalized signal key used for mechanism lookup
    observation_strength: float    # STRENGTH_SPECIFIC | STRUCTURAL | MODERATE | GENERIC
    extractability_confidence: float  # how cleanly this observation maps to the signal


@dataclass
class RawMechanism:
    mechanism_type: str
    mechanism_category: str          # "advantage_mechanism" | "operational_mechanism" | ...
    supporting_observations: List[StructuralObservation]
    directly_observed: bool          # True if ≥1 obs has strength ≥ STRENGTH_STRUCTURAL
    impact_prior: float              # category-level prior from IMPACT_PRIORS
    selection_score: float           # impact_prior × mean(strength) × mean(conf)


@dataclass
class CalibratedMechanism:
    mechanism_type: str
    mechanism_category: str

    # Only these three scalars are stored. effective_weight is computed on demand.
    base_weight: float
    confidence: float
    evidence_strength: float     # 1.0 | 0.75 | 0.50

    inference_depth: str         # "directly_observed" | "one_step_inference" | "llm_interpretation"
    implication_ceiling: str     # "observation" | "cautious_inference" | "inference" | "strategic_conclusion"
    mechanism_reasoning_trace: str
    retrieval_hook: List[str] = field(default_factory=list)  # always [] — RAG stub

    @property
    def effective_weight(self) -> float:
        """Computed from stored fields. Never stored. Prevents stale-state drift."""
        decay = 0.85 if self.confidence < 0.40 else 1.0
        return self.base_weight * self.confidence * self.evidence_strength * decay


@dataclass
class MarketStructure:
    category: str              # "fragmented" | "concentrated" | "platform_controlled" | "regulation_gated" | "commoditized" | "ambiguous"
    confidence: float
    structural_signals: List[str]
    alternative_category: Optional[str]  # set when gap to second category is narrow
    retrieval_hook: List[str] = field(default_factory=list)


@dataclass
class TensionCondition:
    field: str        # dot-path into context dict, e.g. "stage" or "market_structure_category"
    operator: str     # "eq" | "neq" | "in" | "not_in"
    value: Any


@dataclass
class DetectedTension:
    tension_type: str         # "productive_tension" | "scaling_risk" | "operational_inconsistency" | "strategic_mismatch" | "acceptable_early_stage_tradeoff"
    mechanism_a: str
    mechanism_b: str
    severity: str             # "low" | "medium" | "high"
    severity_weight: float    # low=0.10, medium=0.20, high=0.35
    conditions_met: List[TensionCondition]
    description: str


@dataclass
class ReplicationProfile:
    mechanism_type: str
    raw_dimension_scores: Dict[str, float]
    adjusted_dimension_scores: Dict[str, float]
    composite_difficulty: float
    time_horizon: str         # "months" | "1-2 years" | "3+ years" | "structural_barrier"


@dataclass
class ConsistencyFlag:
    flag_type: str
    severity: str             # "warning" | "error"
    description: str
    affected_field: str


@dataclass
class ConsistencyReport:
    passed: bool
    flags: List[ConsistencyFlag]


@dataclass
class EpistemicSummary:
    evidence_quality: str             # "strong" | "moderate" | "weak" | "insufficient"
    observed_signals: List[str]       # directly-observed mechanism types
    inferred_mechanisms: List[str]    # one_step_inference mechanism types
    speculative_assumptions: List[str]  # llm_interpretation mechanism types (v1: always empty)
    unresolved_uncertainty: List[str]   # tension descriptions (medium/high severity)
    structurally_missing: List[str]     # signal gaps + consistency errors
    recommended_disclosure: str        # one sentence, surfaced verbatim in synthesis


@dataclass
class MechanismEnvelope:
    extraction_mode: str
    mechanisms: List[CalibratedMechanism]
    market_structure: MarketStructure
    tensions: List[DetectedTension]
    replication_profiles: List[ReplicationProfile]
    contextual_adjustments: Dict[str, float]   # mechanism_type → capped adjustment
    uncertainty: float                          # 0.0–0.30, feeds L4 probabilistically
    consistency_report: ConsistencyReport
    epistemic_summary: EpistemicSummary
    synthesis_trace: List[str]                  # source tags for every synthesis claim
    tension_coverage_state: str                 # explicit statement of rule-set scope


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_implication_ceiling(inference_depth: str, confidence: float) -> str:
    """
    Maximum claim level this evidence supports.
    llm_interpretation never supports more than direct observation statement.
    """
    if inference_depth == "llm_interpretation":
        return "observation"
    if inference_depth == "one_step_inference":
        return "inference" if confidence >= 0.60 else "cautious_inference"
    # directly_observed
    if confidence >= 0.80:
        return "strategic_conclusion"
    if confidence >= 0.60:
        return "inference"
    return "cautious_inference"


def _build_reasoning_trace(rm: RawMechanism) -> str:
    """Deterministic trace from supporting observations. No LLM in v1."""
    signals = [o.structural_signal for o in rm.supporting_observations[:2]]
    sources = sorted({o.field_source for o in rm.supporting_observations})
    return (
        f"Derived from {', '.join(signals)} "
        f"(sources: {', '.join(sources)}). "
        f"Category: {rm.mechanism_category}. "
        f"Selection score: {rm.selection_score:.3f}."
    )


@dataclass
class _TensionRule:
    mechanism_a: str
    mechanism_b: str
    only_if: List[TensionCondition]
    tension_type: str
    severity: str
    severity_weight: float
    description: str


def _check_condition(cond: TensionCondition, context: Dict[str, Any]) -> bool:
    """Evaluate one structured TensionCondition against live context data."""
    val = context.get(cond.field)
    if cond.operator == "eq":
        return val == cond.value
    if cond.operator == "neq":
        return val != cond.value
    if cond.operator == "in":
        return val in (cond.value or [])
    if cond.operator == "not_in":
        return val not in (cond.value or [])
    return False


# Tension rules — structured objects, not strings.
# Each rule fires only when both mechanisms are present AND all conditions pass.
_TENSION_RULES: List[_TensionRule] = [
    _TensionRule(
        mechanism_a="network_effect",
        mechanism_b="platform_dependency",
        only_if=[TensionCondition("market_structure_category", "in", ["platform_controlled"])],
        tension_type="strategic_mismatch",
        severity="high",
        severity_weight=0.35,
        description=(
            "Network effect claim is undermined by platform-controlled distribution — "
            "growth rate depends on third-party permission, not on organic user density."
        ),
    ),
    _TensionRule(
        mechanism_a="cost_advantage",
        mechanism_b="brand_moat",
        only_if=[TensionCondition("stage", "in", ["idea", "validation"])],
        tension_type="operational_inconsistency",
        severity="medium",
        severity_weight=0.20,
        description=(
            "Cost-leadership and premium brand require contradictory operational priorities. "
            "At early stage, pursuing both simultaneously fractures focus."
        ),
    ),
    _TensionRule(
        mechanism_a="regulatory_moat",
        mechanism_b="regulatory_headwind",
        only_if=[],
        tension_type="strategic_mismatch",
        severity="high",
        severity_weight=0.35,
        description=(
            "Regulatory environment is simultaneously a moat and a headwind — "
            "the evidence is ambiguous and the regulatory position requires explicit clarification "
            "before strategic conclusions can be drawn."
        ),
    ),
    _TensionRule(
        mechanism_a="network_effect",
        mechanism_b="switching_cost",
        only_if=[TensionCondition("stage", "in", ["idea", "validation"])],
        tension_type="acceptable_early_stage_tradeoff",
        severity="low",
        severity_weight=0.10,
        description=(
            "Network effects and switching costs reinforce each other at scale, "
            "but at early stage both require simultaneous investment in acquisition and retention — "
            "a resource allocation tension that is manageable but must be planned for."
        ),
    ),
    _TensionRule(
        mechanism_a="cost_advantage",
        mechanism_b="network_effect",
        only_if=[TensionCondition("market_structure_category", "in", ["fragmented"])],
        tension_type="scaling_risk",
        severity="medium",
        severity_weight=0.20,
        description=(
            "Cost advantage in fragmented markets can prevent the pricing strategy "
            "that drives network density. Low prices and high user volume are "
            "temporarily aligned but diverge at scale."
        ),
    ),
]


def _extract_text_signals(idea_lower: str) -> List[StructuralObservation]:
    """
    Extract structural signals from idea_text using multi-word patterns.
    Assigns STRENGTH_MODERATE or lower — text signals are supporting evidence,
    not primary evidence. Single keywords are deliberately excluded.
    """
    signals = []

    # API/platform dependency requires operational criticality signals — not just integrations.
    # "integrates with" and "plugin for" are deliberately excluded: they describe features,
    # not structural lock-in. These patterns require dependency + failure sensitivity.
    if any(p in idea_lower for p in (
        "cannot function without",
        "depends entirely on",
        "requires api access from",
        "single supplier for",
        "sole source of",
        "locked into",
        "tied to a single platform",
        "no alternative to",
    )):
        signals.append(StructuralObservation(
            field_source="idea_text",
            observation_type="integration_dependency",
            raw_text="operational criticality / lock-in dependency pattern",
            structural_signal="third_party_dependency",
            observation_strength=STRENGTH_MODERATE,
            extractability_confidence=0.65,
        ))

    if any(p in idea_lower for p in ("exclusive agreement", "exclusive license", "exclusive access",
                                      "proprietary license", "certification required")):
        signals.append(StructuralObservation(
            field_source="idea_text",
            observation_type="access_barrier",
            raw_text="exclusive/licensed/certified access pattern",
            structural_signal="exclusivity_or_barrier",
            observation_strength=STRENGTH_STRUCTURAL,
            extractability_confidence=0.65,
        ))

    if any(p in idea_lower for p in ("proprietary data", "unique dataset", "data moat",
                                      "exclusive data", "proprietary dataset")):
        signals.append(StructuralObservation(
            field_source="idea_text",
            observation_type="data_asset",
            raw_text="proprietary data asset pattern",
            structural_signal="data_ownership",
            observation_strength=STRENGTH_STRUCTURAL,
            extractability_confidence=0.65,
        ))

    # Platform dependency requires named platform with distribution control.
    # "built on" alone is insufficient — must name a specific controlling platform.
    if any(p in idea_lower for p in (
        "through the app store",
        "through google play",
        "only on ios",
        "only on android",
        "shopify app only",
        "salesforce exclusive",
        "aws marketplace exclusively",
        "azure marketplace exclusively",
    )):
        signals.append(StructuralObservation(
            field_source="idea_text",
            observation_type="platform_reliance",
            raw_text="named single-platform distribution pattern",
            structural_signal="platform_controlled_distribution",
            observation_strength=STRENGTH_MODERATE,
            extractability_confidence=0.65,
        ))

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — EXTRACTABILITY SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_extractability(
    l3_reasoning: dict,
    l1_values: dict,
    l1_confidence: dict,
) -> ExtractabilityResult:
    """
    Gate function. Scores available signal coverage before any extraction runs.
    Returns mode that determines mechanism_cap for Phase 1B.
    """
    REQUIRED_L3 = ["differentiation", "competition", "business_model", "unit_economics"]
    REQUIRED_L1 = ["business_model", "target_segment", "stage"]

    signal_gaps: List[str] = []

    # L3 module completeness
    l3_available = []
    for module in REQUIRED_L3:
        block = l3_reasoning.get(module) or {}
        if block.get("available", False):
            l3_available.append(module)
        else:
            signal_gaps.append(f"l3.{module}")

    # L1 field completeness: must be non-UNKNOWN and above confidence floor
    l1_available = []
    for fname in REQUIRED_L1:
        val  = l1_values.get(fname)
        conf = l1_confidence.get(fname, 0.0)
        if val and val != "UNKNOWN" and conf >= 0.50:
            l1_available.append(fname)
        else:
            signal_gaps.append(f"l1.{fname}")

    completeness = (len(l3_available) + len(l1_available)) / (len(REQUIRED_L3) + len(REQUIRED_L1))

    # Specificity: differentiation verdict is a proxy for signal richness
    diff_block  = l3_reasoning.get("differentiation") or {}
    diff_verdict = diff_block.get("verdict", "") if diff_block.get("available") else ""
    if diff_verdict in ("structural", "moderate"):
        specificity = 1.0
    elif diff_verdict == "thin":
        specificity = 0.50
    else:
        specificity = 0.0

    # Consistency: L3 insufficient_information modules reduce score
    n_insufficient = len(l3_reasoning.get("insufficient_information") or [])
    consistency = max(0.0, 1.0 - n_insufficient * 0.25)

    score = round(0.50 * completeness + 0.30 * specificity + 0.20 * consistency, 3)

    if score < 0.35:
        mode = "insufficient_information"
    elif score < 0.60:
        mode = "partial"
    else:
        mode = "full"

    return ExtractabilityResult(
        score=score,
        mode=mode,
        mechanism_cap=MECHANISM_CAPS[mode],
        signal_gaps=signal_gaps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1A — STRUCTURAL OBSERVATION PASS
# ─────────────────────────────────────────────────────────────────────────────

def structural_observation_pass(
    l3_reasoning: dict,
    l1_values: dict,
    idea_text: str,
    extractability: ExtractabilityResult,
) -> List[StructuralObservation]:
    """
    Translate L3 structured fields and L1 values into typed observations.
    No inference here — only what is directly present in the data.
    observation_strength distinguishes signal quality from signal count.
    """
    obs: List[StructuralObservation] = []
    idea_lower = (idea_text or "").lower()

    # ── L3: differentiation ───────────────────────────────────────────────────
    diff_block = l3_reasoning.get("differentiation") or {}
    if diff_block.get("available"):
        verdict = diff_block.get("verdict", "")
        strength_map = {
            "structural": (STRENGTH_SPECIFIC,   0.90, "strong_differentiation"),
            "moderate":   (STRENGTH_STRUCTURAL, 0.80, "moderate_differentiation"),
            "thin":       (STRENGTH_MODERATE,   0.75, "weak_differentiation"),
        }
        if verdict in strength_map:
            strength, conf, signal = strength_map[verdict]
            obs.append(StructuralObservation(
                field_source="l3.differentiation",
                observation_type="differentiation_claim",
                raw_text=f"differentiation_verdict={verdict}",
                structural_signal=signal,
                observation_strength=strength,
                extractability_confidence=conf,
            ))

    # ── L3: competition ───────────────────────────────────────────────────────
    comp_block = l3_reasoning.get("competition") or {}
    if comp_block.get("available"):
        pressure = comp_block.get("competitive_pressure", "")
        if pressure in ("high", "low", "medium"):
            strength = STRENGTH_STRUCTURAL if pressure in ("high", "low") else STRENGTH_MODERATE
            obs.append(StructuralObservation(
                field_source="l3.competition",
                observation_type="competitive_intensity",
                raw_text=f"competitive_pressure={pressure}",
                structural_signal=f"competition_{pressure}",
                observation_strength=strength,
                extractability_confidence=0.80,
            ))

    # ── L3: unit_economics ───────────────────────────────────────────────────
    ue_block = l3_reasoning.get("unit_economics") or {}
    if ue_block.get("available"):
        scal = (ue_block.get("scalability_pressure") or {}).get("tier", "")
        if scal == "improves_with_scale":
            obs.append(StructuralObservation(
                field_source="l3.unit_economics",
                observation_type="scale_efficiency",
                raw_text=f"scalability_pressure={scal}",
                structural_signal="cost_amortizes_with_scale",
                observation_strength=STRENGTH_STRUCTURAL,
                extractability_confidence=0.85,
            ))
        elif scal == "worsens_with_scale":
            obs.append(StructuralObservation(
                field_source="l3.unit_economics",
                observation_type="scale_constraint",
                raw_text=f"scalability_pressure={scal}",
                structural_signal="cost_grows_with_scale",
                observation_strength=STRENGTH_STRUCTURAL,
                extractability_confidence=0.85,
            ))

        cac = (ue_block.get("cac_proxy") or {}).get("tier", "")
        if cac == "high":
            obs.append(StructuralObservation(
                field_source="l3.unit_economics",
                observation_type="acquisition_cost",
                raw_text=f"cac_proxy={cac}",
                structural_signal="high_acquisition_cost",
                observation_strength=STRENGTH_STRUCTURAL,
                extractability_confidence=0.80,
            ))

        rpu = (ue_block.get("revenue_per_user_proxy") or {}).get("tier", "")
        if rpu == "high":
            obs.append(StructuralObservation(
                field_source="l3.unit_economics",
                observation_type="revenue_density",
                raw_text=f"revenue_per_user_proxy={rpu}",
                structural_signal="high_revenue_per_user",
                observation_strength=STRENGTH_SPECIFIC,
                extractability_confidence=0.85,
            ))
        elif rpu == "low":
            obs.append(StructuralObservation(
                field_source="l3.unit_economics",
                observation_type="revenue_density",
                raw_text=f"revenue_per_user_proxy={rpu}",
                structural_signal="low_revenue_per_user",
                observation_strength=STRENGTH_STRUCTURAL,
                extractability_confidence=0.80,
            ))

    # ── L3: business_model block ─────────────────────────────────────────────
    bm_block = l3_reasoning.get("business_model") or {}
    if bm_block.get("available"):
        op_complex = (bm_block.get("cost_structure") or {}).get("operational_complexity", "")
        if op_complex in ("high", "low"):
            obs.append(StructuralObservation(
                field_source="l3.business_model",
                observation_type="operational_burden",
                raw_text=f"operational_complexity={op_complex}",
                structural_signal=f"{op_complex}_operational_complexity",
                observation_strength=STRENGTH_MODERATE,
                extractability_confidence=0.75,
            ))

    # ── L1: regulatory_risk ──────────────────────────────────────────────────
    reg_risk = l1_values.get("regulatory_risk", "")
    if reg_risk == "high":
        obs.append(StructuralObservation(
            field_source="l1.regulatory_risk",
            observation_type="regulatory_signal",
            raw_text=f"regulatory_risk={reg_risk}",
            structural_signal="high_regulatory_barrier",
            observation_strength=STRENGTH_SPECIFIC,
            extractability_confidence=0.90,
        ))

    # ── L1: business_model type ──────────────────────────────────────────────
    bm_type = l1_values.get("business_model", "")
    if bm_type == "marketplace":
        obs.append(StructuralObservation(
            field_source="l1.business_model",
            observation_type="multi_sided_structure",
            raw_text=f"business_model={bm_type}",
            structural_signal="two_sided_network",
            observation_strength=STRENGTH_STRUCTURAL,
            extractability_confidence=0.90,
        ))

    # ── L1: differentiation_score (scalar — supporting only, not primary) ────
    diff_score = l1_values.get("differentiation_score", 0)
    if isinstance(diff_score, (int, float)) and diff_score >= 4:
        obs.append(StructuralObservation(
            field_source="l1.differentiation_score",
            observation_type="differentiation_claim",
            raw_text=f"differentiation_score={diff_score}/5",
            structural_signal="high_differentiation_score",
            observation_strength=STRENGTH_MODERATE,
            extractability_confidence=0.70,
        ))

    # ── idea_text: multi-word structural patterns ─────────────────────────────
    obs.extend(_extract_text_signals(idea_lower))

    return obs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1B — MECHANISM ASSIGNMENT PASS
# ─────────────────────────────────────────────────────────────────────────────

def _build_raw_mechanism(
    mechanism_type: str,
    supporting: List[StructuralObservation],
) -> RawMechanism:
    category    = MECHANISM_CATEGORIES[mechanism_type]
    impact_prior = IMPACT_PRIORS[category]
    directly_observed = any(o.observation_strength >= STRENGTH_STRUCTURAL for o in supporting)
    n = len(supporting) or 1
    mean_strength = sum(o.observation_strength for o in supporting) / n
    mean_conf     = sum(o.extractability_confidence for o in supporting) / n
    selection_score = round(impact_prior * mean_strength * mean_conf, 4)
    return RawMechanism(
        mechanism_type=mechanism_type,
        mechanism_category=category,
        supporting_observations=supporting,
        directly_observed=directly_observed,
        impact_prior=impact_prior,
        selection_score=selection_score,
    )


def mechanism_assignment_pass(
    observations: List[StructuralObservation],
    extractability: ExtractabilityResult,
    l1_values: dict,
) -> List[RawMechanism]:
    """
    Assign mechanism types from structural observations.
    Each mechanism requires at least one qualifying observation.
    Selection priority: impact × evidence_strength × confidence (not count).
    This prevents low-frequency but high-impact mechanisms (regulatory, API
    dependency) from being dropped in favour of numerous weak signals.
    """
    signal_set = {o.structural_signal for o in observations}

    def obs_for(*signals: str) -> List[StructuralObservation]:
        return [o for o in observations if o.structural_signal in signals]

    candidates: List[RawMechanism] = []
    seg = l1_values.get("target_segment", "")

    # switching_cost: strong/moderate diff + B2B segment
    if (("strong_differentiation" in signal_set or "moderate_differentiation" in signal_set)
            and seg == "b2b"):
        supporting = obs_for("strong_differentiation", "moderate_differentiation",
                             "high_operational_complexity")
        if supporting:
            candidates.append(_build_raw_mechanism("switching_cost", supporting))

    # network_effect: marketplace structure (two-sided network signal required)
    if "two_sided_network" in signal_set:
        supporting = obs_for("two_sided_network", "strong_differentiation")
        candidates.append(_build_raw_mechanism("network_effect", supporting))

    # cost_advantage: scale efficiency present, high CAC absent
    if "cost_amortizes_with_scale" in signal_set and "high_acquisition_cost" not in signal_set:
        supporting = obs_for("cost_amortizes_with_scale", "low_operational_complexity",
                             "high_revenue_per_user")
        if supporting:
            candidates.append(_build_raw_mechanism("cost_advantage", supporting))

    # brand_moat: high diff score + low competition
    if "high_differentiation_score" in signal_set and "competition_low" in signal_set:
        supporting = obs_for("high_differentiation_score", "strong_differentiation", "competition_low")
        candidates.append(_build_raw_mechanism("brand_moat", supporting))

    # regulatory_moat: high regulatory barrier + differentiation present (barrier protects)
    if ("high_regulatory_barrier" in signal_set
            and ("strong_differentiation" in signal_set or "moderate_differentiation" in signal_set)):
        supporting = obs_for("high_regulatory_barrier", "strong_differentiation",
                             "moderate_differentiation")
        candidates.append(_build_raw_mechanism("regulatory_moat", supporting))

    # data_moat: data ownership + differentiation
    if "data_ownership" in signal_set:
        supporting = obs_for("data_ownership", "strong_differentiation", "moderate_differentiation")
        if supporting:
            candidates.append(_build_raw_mechanism("data_moat", supporting))

    # process_efficiency: low complexity + scale efficiency both present
    if ("low_operational_complexity" in signal_set
            and "cost_amortizes_with_scale" in signal_set):
        supporting = obs_for("low_operational_complexity", "cost_amortizes_with_scale")
        candidates.append(_build_raw_mechanism("process_efficiency", supporting))

    # distribution_control: low competition + marketplace structure
    if "competition_low" in signal_set and "two_sided_network" in signal_set:
        supporting = obs_for("competition_low", "two_sided_network")
        candidates.append(_build_raw_mechanism("distribution_control", supporting))

    # api_dependency (constraint): third-party dependency pattern in text
    if "third_party_dependency" in signal_set:
        supporting = obs_for("third_party_dependency")
        candidates.append(_build_raw_mechanism("api_dependency", supporting))

    # platform_dependency (constraint): named platform in text
    if "platform_controlled_distribution" in signal_set:
        supporting = obs_for("platform_controlled_distribution")
        candidates.append(_build_raw_mechanism("platform_dependency", supporting))

    # regulatory_headwind (constraint): high regulatory + weak differentiation
    if "high_regulatory_barrier" in signal_set and "weak_differentiation" in signal_set:
        supporting = obs_for("high_regulatory_barrier", "weak_differentiation")
        candidates.append(_build_raw_mechanism("regulatory_headwind", supporting))

    # Sort by selection_score (impact × evidence_strength × confidence), apply cap
    candidates.sort(key=lambda m: m.selection_score, reverse=True)
    cap = extractability.mechanism_cap
    return candidates[:cap]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1C — EVIDENCE CALIBRATION PASS
# ─────────────────────────────────────────────────────────────────────────────

def evidence_calibration_pass(raw_mechanisms: List[RawMechanism]) -> List[CalibratedMechanism]:
    """
    Assign base_weight, confidence, evidence_strength, inference_depth,
    and implication_ceiling per mechanism.

    Confidence is weighted by observation_strength × extractability_confidence,
    not just count. Ten weak observations (strength=0.25) will not outweigh
    one strong structural observation (strength=1.0).
    """
    calibrated: List[CalibratedMechanism] = []

    for rm in raw_mechanisms:
        obs = rm.supporting_observations

        # Inference depth and evidence strength
        max_strength = max((o.observation_strength for o in obs), default=0.0)
        if rm.directly_observed and max_strength >= STRENGTH_STRUCTURAL:
            inference_depth  = "directly_observed"
            evidence_strength = 1.0
        elif rm.directly_observed:
            inference_depth  = "one_step_inference"
            evidence_strength = 0.75
        else:
            inference_depth  = "llm_interpretation"
            evidence_strength = 0.50

        # Weighted confidence: 60% quality-weighted signal, 40% count factor
        # This is the fix for observation_strength weighting (requirement 1 and 11).
        n = len(obs) or 1
        weighted_signal = sum(o.observation_strength * o.extractability_confidence for o in obs)
        mean_quality    = weighted_signal / n
        obs_count_factor = min(1.0, n / 4.0)  # saturates at 4 observations
        raw_confidence   = 0.60 * mean_quality + 0.40 * obs_count_factor

        # Single-field penalty: all observations from one source → fragile
        sources = {o.field_source for o in obs}
        if len(sources) == 1:
            raw_confidence *= 0.80

        confidence = round(min(0.95, max(0.10, raw_confidence)), 3)
        implication_ceiling = _get_implication_ceiling(inference_depth, confidence)
        trace = _build_reasoning_trace(rm)

        calibrated.append(CalibratedMechanism(
            mechanism_type=rm.mechanism_type,
            mechanism_category=rm.mechanism_category,
            base_weight=BASE_WEIGHTS.get(rm.mechanism_type, 0.60),
            confidence=confidence,
            evidence_strength=evidence_strength,
            inference_depth=inference_depth,
            implication_ceiling=implication_ceiling,
            mechanism_reasoning_trace=trace,
            retrieval_hook=[],
        ))

    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1D — WEIGHT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def weight_adjustment_pass(mechanisms: List[CalibratedMechanism]) -> List[CalibratedMechanism]:
    """
    Normalize base_weights so effective_weight values are on a consistent scale.
    Apply confidence decay for low-confidence mechanisms.
    effective_weight remains a @property — not stored.
    """
    if not mechanisms:
        return mechanisms

    total = sum(m.base_weight for m in mechanisms)
    if total <= 0:
        return mechanisms

    for m in mechanisms:
        m.base_weight = round(m.base_weight / total, 4)
        if m.confidence < 0.40:
            m.base_weight = round(m.base_weight * 0.85, 4)

    return mechanisms


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1E — TRACE FILLING (deterministic in v1)
# ─────────────────────────────────────────────────────────────────────────────

def interpretation_enrichment_pass(
    mechanisms: List[CalibratedMechanism],
    l3_reasoning: dict,
    l1_values: dict,
) -> List[CalibratedMechanism]:
    """
    v1: fills mechanism_reasoning_trace deterministically from stored fields.
    No LLM calls — this phase enriches traces from structural data only.
    Exists to ensure the trace is always populated regardless of inference_depth.

    LLM-assisted enrichment is intentionally deferred: (1) it adds latency/cost,
    (2) v1 Phase 1B only assigns directly_observed or one_step_inference depths,
    and (3) admitting a gap is epistemically safer than filling it with
    unsupported generation.
    """
    for m in mechanisms:
        if not m.mechanism_reasoning_trace:
            m.mechanism_reasoning_trace = (
                f"{m.mechanism_type} ({m.mechanism_category}) — "
                f"inference_depth={m.inference_depth}, "
                f"confidence={m.confidence:.2f}, "
                f"evidence_strength={m.evidence_strength:.2f}"
            )
    return mechanisms


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — MARKET STRUCTURE DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def derive_market_structure(
    mechanisms: List[CalibratedMechanism],
    l3_reasoning: dict,
    l1_values: dict,
) -> MarketStructure:
    """
    Derive market structure from mechanisms + L3/L1 signals.
    Inspects 6 signal sources (expanded from competition density alone).
    Returns "ambiguous" when gap between top two categories is narrow.
    """
    mech_types = {m.mechanism_type for m in mechanisms}
    signals: List[str] = []

    category_scores: Dict[str, float] = {
        "fragmented":        0.0,
        "concentrated":      0.0,
        "platform_controlled": 0.0,
        "regulation_gated":  0.0,
        "commoditized":      0.0,
    }

    # 1. Competition density
    comp_block   = l3_reasoning.get("competition") or {}
    comp_pressure = comp_block.get("competitive_pressure", "") if comp_block.get("available") else ""
    if comp_pressure == "high":
        category_scores["fragmented"] += 0.30
        signals.append("high_competitive_pressure → fragmented")
    elif comp_pressure == "low":
        category_scores["concentrated"] += 0.25
        signals.append("low_competitive_pressure → concentrated")

    # 2. Platform dependency → platform_controlled
    if "platform_dependency" in mech_types:
        category_scores["platform_controlled"] += 0.40
        signals.append("platform_dependency mechanism → platform_controlled")
    if "api_dependency" in mech_types:
        category_scores["platform_controlled"] += 0.20
        signals.append("api_dependency mechanism → platform signal")

    # 3. Regulatory → regulation_gated
    if "regulatory_moat" in mech_types or "regulatory_headwind" in mech_types:
        category_scores["regulation_gated"] += 0.45
        signals.append("regulatory mechanism → regulation_gated")

    # 4. Cost advantage + thin differentiation → commoditized
    diff_block  = l3_reasoning.get("differentiation") or {}
    diff_verdict = diff_block.get("verdict", "") if diff_block.get("available") else ""
    if "cost_advantage" in mech_types and diff_verdict == "thin":
        category_scores["commoditized"] += 0.35
        signals.append("cost_advantage + thin_differentiation → commoditized")

    # 5. Network effect → concentrated (winner-take-most dynamics)
    if "network_effect" in mech_types:
        category_scores["concentrated"] += 0.20
        signals.append("network_effect → concentrated (winner-take-most)")

    # 6. Distribution control → concentrated
    if "distribution_control" in mech_types:
        category_scores["concentrated"] += 0.20
        signals.append("distribution_control → concentrated")

    # Find top two categories
    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    top_cat,    top_score    = sorted_cats[0]
    second_cat, second_score = sorted_cats[1] if len(sorted_cats) > 1 else (None, 0.0)

    if top_score < 0.15:
        return MarketStructure(
            category="ambiguous",
            confidence=0.25,
            structural_signals=signals or ["insufficient signals for market structure classification"],
            alternative_category=None,
        )

    gap = top_score - second_score
    if gap < 0.10 and second_score > 0.0:
        confidence = round(min(0.70, gap * 3 + 0.40), 2)
        return MarketStructure(
            category="ambiguous",
            confidence=confidence,
            structural_signals=signals,
            alternative_category=f"{top_cat} or {second_cat}",
        )

    confidence = round(min(0.90, top_score * 1.50), 2)
    return MarketStructure(
        category=top_cat,
        confidence=confidence,
        structural_signals=signals,
        alternative_category=second_cat if second_score > 0.10 else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — TENSION CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_tensions(
    mechanisms: List[CalibratedMechanism],
    market_structure: MarketStructure,
    l1_values: dict,
) -> List[DetectedTension]:
    """
    Evaluate tension rules against live mechanism set and context.
    Only fires when both mechanisms are present AND all structured conditions pass.
    """
    mech_types = {m.mechanism_type for m in mechanisms}

    context: Dict[str, Any] = {
        "stage":                   l1_values.get("stage", ""),
        "market_structure_category": market_structure.category,
        "target_segment":          l1_values.get("target_segment", ""),
        "competitive_intensity":   l1_values.get("competitive_intensity", ""),
    }

    detected: List[DetectedTension] = []
    for rule in _TENSION_RULES:
        if rule.mechanism_a not in mech_types or rule.mechanism_b not in mech_types:
            continue
        if all(_check_condition(cond, context) for cond in rule.only_if):
            detected.append(DetectedTension(
                tension_type=rule.tension_type,
                mechanism_a=rule.mechanism_a,
                mechanism_b=rule.mechanism_b,
                severity=rule.severity,
                severity_weight=rule.severity_weight,
                conditions_met=rule.only_if,
                description=rule.description,
            ))

    return detected


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — COMPETITIVE REPLICATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_replication(mechanisms: List[CalibratedMechanism]) -> List[ReplicationProfile]:
    """
    Score how hard each mechanism is to replicate by a well-funded competitor.
    Applies dependency discounting to prevent double-counting causal signals.
    Constraint mechanisms (api_dependency, platform_dependency) score low —
    they constrain the idea, not the competitor.
    """
    profiles: List[ReplicationProfile] = []

    for m in mechanisms:
        raw_dims = dict(_REPLICATION_DIMS.get(m.mechanism_type) or {})
        if not raw_dims:
            continue

        # Apply dependency discounting
        adjusted_dims: Dict[str, float] = {}
        for dim, score in raw_dims.items():
            deps = REPLICATION_DIMENSION_DEPS.get(dim, [])
            if any(d in raw_dims for d in deps):
                adjusted_dims[dim] = round(score * 0.50, 3)
            else:
                adjusted_dims[dim] = round(score, 3)

        valid = [v for v in adjusted_dims.values() if v > 0]
        composite = round(sum(valid) / len(valid), 3) if valid else 0.0

        if composite < 0.30:
            time_horizon = "months"
        elif composite < 0.55:
            time_horizon = "1-2 years"
        elif composite < 0.75:
            time_horizon = "3+ years"
        else:
            time_horizon = "structural_barrier"

        profiles.append(ReplicationProfile(
            mechanism_type=m.mechanism_type,
            raw_dimension_scores=raw_dims,
            adjusted_dimension_scores=adjusted_dims,
            composite_difficulty=composite,
            time_horizon=time_horizon,
        ))

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — CONTEXTUAL SIGNAL AMPLIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def apply_contextual_signals(
    mechanisms: List[CalibratedMechanism],
    l3_reasoning: dict,
    l1_values: dict,
) -> Dict[str, float]:
    """
    Detect timing and environmental signals that amplify mechanism strength.
    These are amplifiers only — they do not create mechanisms.
    Hard cap: ±0.10 per mechanism, enforced in code, not documentation.
    Multiple signals stack but are capped absolutely.
    """
    market_readiness = l1_values.get("market_readiness", 3)
    stage            = l1_values.get("stage", "")
    reg_risk         = l1_values.get("regulatory_risk", "")
    interactions     = l3_reasoning.get("signal_interactions") or []
    mech_types       = {m.mechanism_type for m in mechanisms}

    has_regulatory_tailwind = (
        reg_risk == "high"
        and "regulatory_moat" in mech_types
        and isinstance(market_readiness, (int, float))
        and market_readiness >= 3
    )
    has_timing_tailwind = (
        stage in ("idea", "validation")
        and len(interactions) >= 2
    )
    has_readiness_signal = (
        isinstance(market_readiness, (int, float)) and market_readiness >= 4
    )

    adjustments: Dict[str, float] = {}
    for m in mechanisms:
        raw = 0.0
        if has_regulatory_tailwind and m.mechanism_type == "regulatory_moat":
            raw += 0.06
        if has_timing_tailwind and m.mechanism_category == "advantage_mechanism":
            raw += 0.04
        if has_readiness_signal and m.mechanism_category in ("advantage_mechanism", "distribution_mechanism"):
            raw += 0.05
        # Hard cap enforced here — cannot be bypassed by signal accumulation
        adjustments[m.mechanism_type] = round(
            max(-MAX_CONTEXTUAL_ADJUSTMENT, min(MAX_CONTEXTUAL_ADJUSTMENT, raw)), 3
        )

    return adjustments


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — UNCERTAINTY PROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

def propagate_uncertainty(
    mechanisms: List[CalibratedMechanism],
    extraction_mode: str,
) -> float:
    """
    Compute mechanism_uncertainty: 0.0–0.30 float.
    Feeds L4 as a named argument — never embedded in a dict key.

    insufficient_information → maximum uncertainty (0.30) without formula.
    No mechanisms extracted but mode != insufficient → 0.20 (partial signal gap).
    """
    if extraction_mode == "insufficient_information":
        return 0.30

    if not mechanisms:
        return 0.20

    raw = sum(
        m.effective_weight * max(0.0, 0.40 - m.confidence)
        for m in mechanisms
    )
    return round(min(0.30, raw), 4)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — CROSS-FIELD CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def run_consistency_check(
    mechanisms: List[CalibratedMechanism],
    tensions: List[DetectedTension],
    market_structure: MarketStructure,
    uncertainty: float,
    extraction_mode: str,
    l1_values: dict,
) -> ConsistencyReport:
    """
    Detect structural contradictions and symbolic overconfidence before synthesis.
    Errors must be acknowledged in synthesis. Warnings are informational.
    passed=False means at least one error-severity flag is present.
    """
    flags: List[ConsistencyFlag] = []
    mech_types = {m.mechanism_type for m in mechanisms}
    stage = (l1_values or {}).get("stage", "")

    # 1. Market structure implausibility: brand moat + commoditized market
    if market_structure.category == "commoditized" and "brand_moat" in mech_types:
        brand_m = next((m for m in mechanisms if m.mechanism_type == "brand_moat"), None)
        if brand_m and brand_m.confidence > 0.70:
            flags.append(ConsistencyFlag(
                flag_type="structural_implausibility",
                severity="warning",
                description=(
                    "Brand moat extracted with high confidence in a commoditized market — "
                    "these signals are structurally inconsistent."
                ),
                affected_field="brand_moat × market_structure.commoditized",
            ))

    # 2. Regulatory contradiction: moat and headwind simultaneously
    if "regulatory_moat" in mech_types and "regulatory_headwind" in mech_types:
        flags.append(ConsistencyFlag(
            flag_type="regulatory_contradiction",
            severity="error",
            description=(
                "Regulatory environment is simultaneously a moat and a headwind — "
                "the evidence is ambiguous and requires clarification before synthesis."
            ),
            affected_field="regulatory_moat × regulatory_headwind",
        ))

    # 3. Causal overextension: strategic_conclusion at idea stage
    if stage == "idea":
        overextended = [
            m.mechanism_type for m in mechanisms
            if m.implication_ceiling == "strategic_conclusion"
        ]
        if overextended:
            flags.append(ConsistencyFlag(
                flag_type="premature_strategic_conclusion",
                severity="warning",
                description=(
                    f"strategic_conclusion implication level claimed for "
                    f"{overextended} at idea stage — "
                    f"evidence basis is structurally thin for this implication level."
                ),
                affected_field="implication_ceiling × stage=idea",
            ))

    # 4. Extraction mode mismatch (pipeline routing error)
    if extraction_mode == "insufficient_information" and mechanisms:
        flags.append(ConsistencyFlag(
            flag_type="extraction_mode_mismatch",
            severity="error",
            description="Mechanisms extracted despite insufficient_information mode — routing error.",
            affected_field="extraction_mode",
        ))

    # 5. High speculation density with elevated uncertainty
    speculative_count = sum(1 for m in mechanisms if m.inference_depth == "llm_interpretation")
    if uncertainty >= 0.20 and speculative_count >= 2:
        flags.append(ConsistencyFlag(
            flag_type="high_speculation_density",
            severity="warning",
            description=(
                f"mechanism_uncertainty={uncertainty:.2f} and {speculative_count} speculative "
                f"mechanisms — synthesis must use hedged language throughout."
            ),
            affected_field="mechanism_uncertainty × speculative_count",
        ))

    passed = not any(f.severity == "error" for f in flags)
    return ConsistencyReport(passed=passed, flags=flags)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8 — EPISTEMIC SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _compose_disclosure(
    quality: str,
    observed: List[str],
    inferred: List[str],
    speculative: List[str],
    gaps: List[str],
    consistency: ConsistencyReport,
) -> str:
    if quality == "insufficient":
        return (
            "Evidence is insufficient to support mechanism-level conclusions — "
            "structural signal coverage is too low for reliable inference."
        )

    parts: List[str] = []
    if observed:
        parts.append(
            f"{len(observed)} directly-observed mechanism(s) "
            f"({', '.join(observed[:2])}{'...' if len(observed) > 2 else ''})"
        )
    if inferred:
        parts.append(
            f"{len(inferred)} inferred mechanism(s) "
            f"({', '.join(inferred[:2])}{'...' if len(inferred) > 2 else ''})"
        )
    if speculative:
        parts.append(
            f"{len(speculative)} speculative assumption(s) not anchored in structural evidence"
        )
    if gaps:
        parts.append(f"{len(gaps)} missing signal field(s)")

    if not parts:
        return "Insufficient signals to characterize evidence quality."

    if quality == "strong" and not speculative:
        prefix = "Analysis grounded in"
    elif quality in ("moderate", "weak") or speculative:
        prefix = "Analysis partially grounded in"
    else:
        prefix = "Evidence basis:"

    result = f"{prefix} {'; '.join(parts)}."

    error_flags = [f for f in consistency.flags if f.severity == "error"]
    if error_flags:
        result += (
            f" {len(error_flags)} structural inconsistency(ies) detected "
            f"({', '.join(f.flag_type for f in error_flags)}) — treat conclusions with caution."
        )

    return result


def build_epistemic_summary(
    mechanisms: List[CalibratedMechanism],
    tensions: List[DetectedTension],
    extraction_mode: str,
    consistency_report: ConsistencyReport,
    signal_gaps: List[str],
) -> EpistemicSummary:
    """
    Always built. Always surfaced. Exposes what we know, inferred, assumed,
    and couldn't find. recommended_disclosure is used verbatim in synthesis.
    """
    if extraction_mode == "insufficient_information":
        evidence_quality = "insufficient"
    else:
        mean_conf = (sum(m.confidence for m in mechanisms) / len(mechanisms)) if mechanisms else 0.0
        if extraction_mode == "partial" or mean_conf < 0.50:
            evidence_quality = "weak"
        elif mean_conf < 0.65:
            evidence_quality = "moderate"
        else:
            evidence_quality = "strong"

    observed    = [m.mechanism_type for m in mechanisms if m.inference_depth == "directly_observed"]
    inferred    = [m.mechanism_type for m in mechanisms if m.inference_depth == "one_step_inference"]
    speculative = [m.mechanism_type for m in mechanisms if m.inference_depth == "llm_interpretation"]

    unresolved = [
        (t.description[:100] + "..." if len(t.description) > 100 else t.description)
        for t in tensions if t.severity in ("medium", "high")
    ]

    structurally_missing = [g.replace("l3.", "L3.").replace("l1.", "L1.") for g in signal_gaps]
    for flag in consistency_report.flags:
        if flag.severity == "error":
            structurally_missing.append(f"consistency_error:{flag.flag_type}")

    disclosure = _compose_disclosure(
        evidence_quality, observed, inferred, speculative, signal_gaps, consistency_report
    )

    return EpistemicSummary(
        evidence_quality=evidence_quality,
        observed_signals=observed,
        inferred_mechanisms=inferred,
        speculative_assumptions=speculative,
        unresolved_uncertainty=unresolved,
        structurally_missing=structurally_missing,
        recommended_disclosure=disclosure,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def serialize_envelope(envelope: MechanismEnvelope) -> dict:
    """
    Convert MechanismEnvelope to a JSON-serializable dict for the response payload.
    effective_weight is computed here for serialization — it is still not stored
    in the dataclass itself.
    """
    return {
        "extraction_mode":   envelope.extraction_mode,
        "mechanism_count":   len(envelope.mechanisms),
        "mechanisms": [
            {
                "type":             m.mechanism_type,
                "category":         m.mechanism_category,
                "confidence":       m.confidence,
                "evidence_strength": m.evidence_strength,
                "inference_depth":  m.inference_depth,
                "implication_ceiling": m.implication_ceiling,
                "reasoning_trace":  m.mechanism_reasoning_trace,
                "effective_weight": round(m.effective_weight, 4),
            }
            for m in envelope.mechanisms
        ],
        "market_structure": {
            "category":            envelope.market_structure.category,
            "confidence":          envelope.market_structure.confidence,
            "structural_signals":  envelope.market_structure.structural_signals,
            "alternative_category": envelope.market_structure.alternative_category,
        },
        "tensions": [
            {
                "type":        t.tension_type,
                "mechanism_a": t.mechanism_a,
                "mechanism_b": t.mechanism_b,
                "severity":    t.severity,
                "description": t.description,
            }
            for t in envelope.tensions
        ],
        "replication_profiles": [
            {
                "mechanism_type":       r.mechanism_type,
                "composite_difficulty": r.composite_difficulty,
                "time_horizon":         r.time_horizon,
                "adjusted_dimensions":  r.adjusted_dimension_scores,
            }
            for r in envelope.replication_profiles
        ],
        "contextual_adjustments": envelope.contextual_adjustments,
        "uncertainty":            envelope.uncertainty,
        "consistency_passed":     envelope.consistency_report.passed,
        "consistency_flags": [
            {
                "type":        f.flag_type,
                "severity":    f.severity,
                "description": f.description,
            }
            for f in envelope.consistency_report.flags
        ],
        "epistemic_summary": {
            "evidence_quality":        envelope.epistemic_summary.evidence_quality,
            "observed_signals":        envelope.epistemic_summary.observed_signals,
            "inferred_mechanisms":     envelope.epistemic_summary.inferred_mechanisms,
            "speculative_assumptions": envelope.epistemic_summary.speculative_assumptions,
            "unresolved_uncertainty":  envelope.epistemic_summary.unresolved_uncertainty,
            "structurally_missing":    envelope.epistemic_summary.structurally_missing,
            "recommended_disclosure":  envelope.epistemic_summary.recommended_disclosure,
        },
        "synthesis_trace": envelope.synthesis_trace,
        "tension_coverage_state": envelope.tension_coverage_state,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_mechanism_pipeline(
    l3_reasoning: dict,
    l1_values: dict,
    l1_confidence: dict,
    idea_text: str = "",
    sector: str = "",
    country: str = "",
) -> MechanismEnvelope:
    """
    Single entry point. Runs all 8 phases in order.
    Returns MechanismEnvelope. Never raises — returns insufficient_information
    envelope on any unexpected failure.
    """
    _LOG.debug("[MechPipeline] start sector=%s", sector)

    try:
        # Phase 0: Gate
        extractability = score_extractability(l3_reasoning, l1_values, l1_confidence)
        _LOG.debug(
            "[MechPipeline] mode=%s score=%.3f gaps=%s",
            extractability.mode, extractability.score, extractability.signal_gaps
        )

        if extractability.mode == "insufficient_information":
            epistemic = EpistemicSummary(
                evidence_quality="insufficient",
                observed_signals=[],
                inferred_mechanisms=[],
                speculative_assumptions=[],
                unresolved_uncertainty=[],
                structurally_missing=extractability.signal_gaps,
                recommended_disclosure=(
                    "Evidence is insufficient to support mechanism-level conclusions — "
                    "structural signal coverage is too low for reliable inference."
                ),
            )
            return MechanismEnvelope(
                extraction_mode="insufficient_information",
                mechanisms=[],
                market_structure=MarketStructure(
                    category="ambiguous", confidence=0.10,
                    structural_signals=["insufficient evidence for market structure classification"],
                    alternative_category=None,
                ),
                tensions=[],
                replication_profiles=[],
                contextual_adjustments={},
                uncertainty=0.30,
                consistency_report=ConsistencyReport(passed=True, flags=[]),
                epistemic_summary=epistemic,
                synthesis_trace=["INSUFFICIENT_INFORMATION"],
                tension_coverage_state=(
                    "Tension analysis not run — insufficient signal coverage. "
                    "Absence of tensions does not imply absence of structural conflict."
                ),
            )

        # Phase 1A: Structural observations
        observations = structural_observation_pass(l3_reasoning, l1_values, idea_text, extractability)

        # Phase 1B: Mechanism assignment (selection by impact × evidence × confidence)
        raw_mechs = mechanism_assignment_pass(observations, extractability, l1_values)

        # Phase 1C: Evidence calibration (with observation_strength weighting)
        calibrated = evidence_calibration_pass(raw_mechs)

        # Phase 1D: Weight normalization + confidence decay
        adjusted = weight_adjustment_pass(calibrated)

        # Phase 1E: Trace filling (deterministic in v1)
        final_mechs = interpretation_enrichment_pass(adjusted, l3_reasoning, l1_values)

        # Phases 2 and 4 are independent — run sequentially
        market_struct  = derive_market_structure(final_mechs, l3_reasoning, l1_values)
        rep_profiles   = analyze_replication(final_mechs)

        # Phase 3: Tensions (requires market_struct)
        tensions = classify_tensions(final_mechs, market_struct, l1_values)

        # Phase 5: Contextual signals (hard-capped ±0.10)
        ctx_adj = apply_contextual_signals(final_mechs, l3_reasoning, l1_values)

        # Phase 6: Uncertainty propagation
        uncertainty = propagate_uncertainty(final_mechs, extractability.mode)

        # Phase 7: Consistency check
        consistency = run_consistency_check(
            final_mechs, tensions, market_struct,
            uncertainty, extractability.mode, l1_values,
        )

        # Phase 8: Epistemic summary (always built)
        epistemic = build_epistemic_summary(
            final_mechs, tensions, extractability.mode,
            consistency, extractability.signal_gaps,
        )

        # Synthesis trace
        synthesis_trace: List[str] = []
        for m in final_mechs:
            synthesis_trace.append(
                f"MECHANISM:{m.mechanism_type}"
                f"(conf={m.confidence:.2f},ceiling={m.implication_ceiling})"
            )
        for t in tensions:
            synthesis_trace.append(
                f"TENSION:{t.tension_type}:{t.mechanism_a}:{t.mechanism_b}"
                f"(severity={t.severity})"
            )

        # tension_coverage_state: "no tensions" must not imply "tension-free".
        # Explicitly names the rule set evaluated and its scope limits.
        n_rules = len(_TENSION_RULES)
        n_detected = len(tensions)
        if n_detected == 0:
            tension_coverage_state = (
                f"No tensions detected across {n_rules} evaluated rule pairs. "
                f"Rule set covers mechanism co-occurrence conflicts only — "
                f"operational and execution tensions outside this rule set are not evaluated."
            )
        else:
            sev_labels = ", ".join(f"{t.mechanism_a}×{t.mechanism_b}({t.severity})" for t in tensions)
            tension_coverage_state = (
                f"{n_detected} tension(s) detected from {n_rules}-rule evaluation: {sev_labels}. "
                f"Additional tensions outside the rule set scope may exist."
            )

        _LOG.debug(
            "[MechPipeline] done: %d mechanisms, uncertainty=%.3f, quality=%s",
            len(final_mechs), uncertainty, epistemic.evidence_quality,
        )

        return MechanismEnvelope(
            extraction_mode=extractability.mode,
            mechanisms=final_mechs,
            market_structure=market_struct,
            tensions=tensions,
            replication_profiles=rep_profiles,
            contextual_adjustments=ctx_adj,
            uncertainty=uncertainty,
            consistency_report=consistency,
            epistemic_summary=epistemic,
            synthesis_trace=synthesis_trace,
            tension_coverage_state=tension_coverage_state,
        )

    except Exception as exc:
        _LOG.error("[MechPipeline] unexpected error: %s — returning insufficient envelope", exc)
        fallback_epistemic = EpistemicSummary(
            evidence_quality="insufficient",
            observed_signals=[],
            inferred_mechanisms=[],
            speculative_assumptions=[],
            unresolved_uncertainty=[],
            structurally_missing=["pipeline_error"],
            recommended_disclosure=(
                "Mechanism extraction encountered an internal error — "
                "conclusions below are derived from L3 signals only."
            ),
        )
        return MechanismEnvelope(
            extraction_mode="insufficient_information",
            mechanisms=[],
            market_structure=MarketStructure(
                category="ambiguous", confidence=0.10,
                structural_signals=["pipeline_error"],
                alternative_category=None,
            ),
            tensions=[],
            replication_profiles=[],
            contextual_adjustments={},
            uncertainty=0.30,
            consistency_report=ConsistencyReport(passed=True, flags=[]),
            epistemic_summary=fallback_epistemic,
            synthesis_trace=["PIPELINE_ERROR"],
            tension_coverage_state=(
                "Tension analysis not run — pipeline error. "
                "Absence of tensions does not imply absence of structural conflict."
            ),
        )

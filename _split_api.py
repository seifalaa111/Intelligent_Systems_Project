"""
One-shot helper: read api.py and split it into the midan/ package modules.

Each module gets a slice of api.py (specified by line ranges) prepended with
correct imports. This is the mechanical part of Step 3 — code movement only,
no behavior change.

After this script runs, api.py is replaced by a thin re-export shim and tests
should pass unchanged.
"""
from pathlib import Path

API_PATH    = Path(__file__).parent / "api.py"
MIDAN_DIR   = Path(__file__).parent / "midan"
MIDAN_DIR.mkdir(exist_ok=True)

raw = API_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
n_lines = len(raw)
print(f"api.py has {n_lines} lines")


def slice_lines(ranges):
    """Yield content from a list of (start_inclusive_1based, end_inclusive_1based) ranges."""
    out = []
    for start, end in ranges:
        s = max(1, start) - 1
        e = min(n_lines, end)
        out.extend(raw[s:e])
    return "".join(out)


# ── Line ranges per module (based on the api.py boundary map) ──────────────
# Lines are 1-based, end-inclusive.

# CORE: imports, model loading, freshness constants/funcs, sector/country/keyword
# tables, hint lists, utility funcs (_phrase_in_text, _has_any, _count_any,
# _is_workflow_software_idea, _score_sector_candidates, _infer_*),
# _extract_idea_grounding, request models, ResponsePayload schema, builder helpers
CORE_RANGES = [
    (1,    99),     # imports + model loading + freshness constants
    (100,  117),    # _sarima_last_date, _days_since
    (118,  200),    # compute_l2_freshness + sector tables (we'll cherry-pick — actually keep all 100-294 for sector data)
    (201,  294),    # SECTOR_KEYWORDS, COUNTRY_KEYWORDS, SECTOR_TIEBREAKER, hint lists
    (295,  309),    # _phrase_in_text, _has_any, _count_any
    (310,  316),    # _is_workflow_software_idea
    (317,  344),    # _score_sector_candidates
    (345,  470),    # _infer_*, _extract_idea_grounding
    (4179, 4350),   # FastAPI app declaration + request models + ResponsePayload schema (we'll move app declaration out later)
    (4351, 4680),   # build_response_payload + helpers (also belongs to response, but the schema-coupled helpers stay in core)
]

# Actually, the strategy of putting build_response_payload in core is wrong —
# it depends on _post_decision_route + _l4_top_risk_dim which are L4/conversation.
# Let me revise: core stops at line 4350 (schema definitions). Builder goes to response.

# Revised CORE_RANGES — schemas + constants + utils + model loading + grounding
CORE_RANGES = [
    (1,    99),    # imports + model loading + freshness constants
    (100,  117),   # _sarima_last_date, _days_since
    (118,  192),   # compute_l2_freshness
    (193,  294),   # constant tables (SECTOR_*, COUNTRY_*, hint lists)
    (295,  316),   # phrase utils
    (317,  344),   # _score_sector_candidates
    (345,  410),   # _infer_*
    (411,  470),   # _extract_idea_grounding
    (4192, 4350),  # request models + ResponsePayload schema
]

# L0_GATE: L0 module — constants + 9 checks + LLM arbiter + how_to_fix + sanity_check orchestrator
L0_RANGES = [
    (4682, 4929),  # L0 constants (impossible, free_money, free_everything, unsustainable, vague, concrete_rescue,
                    # contradictions, prompt_injection, non_idea_tokens) + checks (contradiction, spam, prompt_injection)
    (4930, 5269),  # individual L0 checks (length, impossibility, no_revenue, no_value, unsustainable, vague), LLM arbiter, log_rejection, fix_fallbacks, _l0_how_to_fix
    (5271, 5330),  # _layer0_sanity_check
]

# L1_PARSER: agent_a1_parse + L1 constants + extract_idea_features + helpers + consistency
L1_RANGES = [
    (471, 495),    # agent_a1_parse (after enhanced_regime block — careful, 477-491)
    (785, 1199),   # L1 module: constants, _l1_field, _coerce_*, _result_from_fields, extract_idea_features, _heuristic_field, _heuristic_idea_features, _backfill_with_heuristic, _validate_l1_consistency, _l1_clarification_message
]

# L2_INTELLIGENCE: enhanced_regime variants + FCM + macro adjustments + compute_shap
L2_RANGES = [
    (496, 605),    # enhanced_regime + _REGIME_RULES + enhanced_regime_with_path
    (606, 783),    # FCM module + macro adjustment module
    (3650, 3702),  # compute_shap
]

# L3_REASONING: _FIT_TABLE + _BM_PROFILE + _SECTOR_REG_PROFILE + compute_idea_signal
# + _signal_tier + L3 reasoning module (constants + analyzers + orchestrator)
L3_RANGES = [
    (1201, 1483),  # _FIT_TABLE, _REGIME_DEFAULTS, _BM_PROFILE, _SECTOR_REG_PROFILE, compute_idea_signal, _signal_tier
    (1485, 2125),  # L3 reasoning module (constants, _safe_int, _l3_field_known, analyzers, orchestrator)
]

# L4_DECISION: risk decomposers + conflict detector + offsetting + decision quality + state machine + orchestrator
L4_RANGES = [
    (2127, 2867),  # full L4 module
]

# RESPONSE: L4 strategic reasoning generators + agent_a0 + explanation layer
# + projection helpers + chat_fallback + operator_reply
RESPONSE_RANGES = [
    (2868, 3392),  # _l4_reasoning_llm + _l4_reasoning_fallback + _generate_l4_reasoning
    (3393, 3467),  # agent_a0_evaluate_idea
    (3468, 3647),  # _generate_explanation_layer
    (4351, 4680),  # build_response_payload + helpers
    (5587, 5827),  # _l4_top_risk_dim, _l4_summary_for_chat, _chat_fallback
    (6100, 6353),  # projection helpers (_first_sentence, _sentence_tail, _assess_projection_input, _build_projection_payload, _probe_answer_fallback, _answer_projection_probe)
    (6581, 6766),  # _generate_operator_reply (+ _OP_REPLY_LOG)
]

# CONVERSATION: _classify_intent + _extract_components + _post_decision_route + helpers
CONVERSATION_RANGES = [
    (5932, 6098),  # conversation token sets (_GREET_TOKENS, _META_PHRASES, _VAGUE_STARTERS, _CASUAL_PREFIXES,
                    # _CASUAL_SHORT_SET, _OVERRIDE_COMMANDS, _PROBLEM_SIGNALS, _SOLUTION_SIGNALS, _MARKET_GEO,
                    # _MARKET_CUSTOMER, _merge_accumulated_text, _extract_components, _build_analysis_text)
    (6354, 6580),  # POST_DECISION_MODES + _post_decision_route + _classify_intent + _casual_response + _smart_followup
]

# PIPELINE: _build_invalid_response + _build_clarification_response + process_idea + run_inference
PIPELINE_RANGES = [
    (3704, 4178),  # run_inference
    (5331, 5538),  # _build_invalid_response, _build_clarification_response, process_idea
]

# ENDPOINTS: FastAPI app + 6 endpoints + InteractRequest model
ENDPOINTS_RANGES = [
    (4179, 4191),  # api = FastAPI(...) + CORS middleware
    (5541, 5586),  # /analyze
    (5828, 5931),  # /chat
    (6762, 6766),  # InteractRequest
    (6767, 6886),  # /project
    (6887, 7057),  # _interact_payload_or_500 + /interact
    (7058, 7080),  # /health, /rejection-patterns
]


# ── Module emit ─────────────────────────────────────────────────────────────
def emit(filename: str, header: str, ranges: list, footer: str = ""):
    body = slice_lines(ranges)
    content = header + "\n\n# ── extracted from api.py ─────────────────────────────────────────────\n\n" + body + "\n" + footer
    out = MIDAN_DIR / filename
    out.write_text(content, encoding="utf-8")
    print(f"  wrote {out} ({len(body.splitlines())} lines extracted)")


CORE_HEADER = '''"""
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
'''

emit("core.py", CORE_HEADER, CORE_RANGES)


L0_HEADER = '''"""
midan.l0_gate — input validation gate.

Strict, blocking checks applied before any pipeline analysis. Nine
deterministic checks plus an LLM borderline arbiter. Every rejection is
blocking; there is no "advisory pass-through" path.
"""
from midan.core import *  # noqa: F401,F403
'''
emit("l0_gate.py", L0_HEADER, L0_RANGES)


L1_HEADER = '''"""
midan.l1_parser — confidence-scored idea feature extraction.

Returns a structured envelope `{values, confidence, source, is_sufficient,
consistency, ...}` where invalid/low-confidence fields become the literal
UNKNOWN sentinel rather than silently defaulted. Cross-field consistency
enforced. Heuristic fallback paths emit explicit confidence per field.
"""
from midan.core import *  # noqa: F401,F403
'''
emit("l1_parser.py", L1_HEADER, L1_RANGES)


L2_HEADER = '''"""
midan.l2_intelligence — market intelligence layer.

Macro vector construction with explicit (sector × country) base + traceable
idea-derived adjustments. SVM regime classification with visible decision
path (SVM step → optional rule_override → final). FCM parallel fuzzy regime
signal. SHAP over the adjusted vector. SARIMA precomputed lookups guarded
by a freshness gate.
"""
from midan.core import *  # noqa: F401,F403
'''
emit("l2_intelligence.py", L2_HEADER, L2_RANGES)


L3_HEADER = '''"""
midan.l3_reasoning — structured idea reasoning layer.

Five analyzers (differentiation, competition, business_model, unit_economics,
signal_interactions) each emit a structured envelope grounded in L1+L2
signals. No new scalar score introduced; legacy `idea_signal` preserved
for the L4 TAS contract but tagged legacy. Insufficient-information state
is explicit per analyzer.
"""
from midan.core import *  # noqa: F401,F403
'''
emit("l3_reasoning.py", L3_HEADER, L3_RANGES)


L4_HEADER = '''"""
midan.l4_decision — decision engine.

Replaces the linear TAS-based decision with a structured state machine
driven by risk decomposition (market/execution/timing), conflict detection
(severity-tiered), offsetting analysis (reasoning dominance), and a
qualitative decision_strength tier. TAS preserved as legacy_tas_score with
explicit zero-decision-influence note.
"""
from midan.core import *  # noqa: F401,F403
from midan.l3_reasoning import _l3_field_known, _safe_int  # noqa: F401
'''
emit("l4_decision.py", L4_HEADER, L4_RANGES)


CONVERSATION_HEADER = '''"""
midan.conversation — intent classification + post-decision routing.

Stateless conversation helpers (no session memory). Each turn is classified
into an intent (CASUAL / GREETING / META / PARTIAL_IDEA / ANALYSIS_REQUEST
/ OVERRIDE_COMMAND / CLARIFICATION) and post-decision turns are routed by
L4 decision_state into one of four behavioral modes (STANDARD_ADVISOR,
RESOLVING_CONFLICT, ADVISORY_ONLY, RE_CLARIFY).
"""
from midan.core import *  # noqa: F401,F403
'''
emit("conversation.py", CONVERSATION_HEADER, CONVERSATION_RANGES)


RESPONSE_HEADER = '''"""
midan.response — payload builder, chat fallback, operator reply, projection.

build_response_payload is the single mapper from raw pipeline output to the
strict ResponsePayload schema. Chat / operator / projection generators all
read the L4 decision envelope as their source of truth — never legacy TAS.
"""
from midan.core import *  # noqa: F401,F403
from midan.l3_reasoning import _signal_tier  # noqa: F401
from midan.l4_decision import (  # noqa: F401
    _l4_top_risk_dim,
)
from midan.conversation import _post_decision_route  # noqa: F401
'''
emit("response.py", RESPONSE_HEADER, RESPONSE_RANGES)


PIPELINE_HEADER = '''"""
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
from midan.response import (  # noqa: F401
    _generate_l4_reasoning, agent_a0_evaluate_idea, _generate_explanation_layer,
)
'''
emit("pipeline.py", PIPELINE_HEADER, PIPELINE_RANGES)


ENDPOINTS_HEADER = '''"""
midan.endpoints — FastAPI app + HTTP endpoints.

Six endpoints (/analyze, /chat, /project, /interact, /health,
/rejection-patterns). /analyze, /interact, /project declare
response_model=ResponsePayload — schema is enforced at the wire boundary.
Pipeline + LLM failures surface as 500 with explicit detail; never silent.
"""
from midan.core import *  # noqa: F401,F403
from midan.pipeline import process_idea, run_inference  # noqa: F401
from midan.response import (  # noqa: F401
    build_response_payload, SchemaViolationError, _BUILDER_LOG,
    _chat_fallback, _generate_operator_reply,
    _build_projection_payload, _answer_projection_probe,
    _assess_projection_input,
)
from midan.conversation import (  # noqa: F401
    _classify_intent, _post_decision_route, _extract_components,
    _build_analysis_text, _smart_followup, _casual_response,
    _GREET_TOKENS, _META_PHRASES, _VAGUE_STARTERS,
)
'''
emit("endpoints.py", ENDPOINTS_HEADER, ENDPOINTS_RANGES)


print("\nAll modules written. Next: replace api.py with shim and run tests.")

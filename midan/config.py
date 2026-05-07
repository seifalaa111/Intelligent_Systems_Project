"""
midan.config — single centralized configuration module.

Owns every tunable constant the system uses. Other midan submodules import
from here rather than declaring constants inline. Editing a threshold means
editing one file; nothing else changes.

This is intentionally a plain Python module — no YAML, no Pydantic Settings,
no env-var indirection layer. The user's Step 5 charter explicitly forbids
"complex config frameworks." Future evolution (env-var override, hot reload,
remote config) plugs in here without churn elsewhere.

Sections:
  • Schema + module versions
  • L1 parser — confidence thresholds
  • L2 intelligence — freshness gate, adjustment gating, table provenance
  • L3 reasoning — field usability floor
  • L4 decision — risk ladder semantics
  • Feature toggles — narrow, named, all default ON

Editing rules:
  • Constants here MUST be plain values (str / int / float / bool).
  • No nested objects, no callables, no dynamic resolution.
  • Every constant carries an inline comment explaining what it controls
    AND which module reads it.
"""

# ── Module versions ─────────────────────────────────────────────────────────
RESPONSE_SCHEMA_VERSION = "1.0"  # midan.core.ResponsePayload — bump when shape changes
L3_REASONING_VERSION    = "1.0"  # midan.l3_reasoning — bump when analyzer contract changes
L4_DECISION_VERSION     = "1.0"  # midan.l4_decision  — bump when state machine semantics change

# ── L1 parser thresholds ────────────────────────────────────────────────────
L1_MIN_FIELD_CONFIDENCE     = 0.55  # midan.l1_parser — below this, a field becomes UNKNOWN
L1_MIN_AGGREGATE_CONFIDENCE = 0.50  # midan.l1_parser — mean confidence across required fields
UNKNOWN_VALUE               = "UNKNOWN"  # midan.l1_parser — sentinel for low-confidence fields

# ── L2 intelligence — freshness + adjustment gating ────────────────────────
STATIC_MACRO_TABLE_AS_OF       = "2025-01-01"  # midan.core — last manual update of macro tables
SARIMA_STALENESS_DAYS          = 180           # midan.core — > N days old → staleness penalty fires
SARIMA_STALENESS_PENALTY       = 0.85          # midan.core — multiplier applied to regime confidence when stale
L1_ADJUSTMENT_CONFIDENCE_FLOOR = 0.70          # midan.l2_intelligence — only apply idea adjustments above this L1 confidence

# ── L3 reasoning — field usability ──────────────────────────────────────────
L3_FIELD_CONFIDENCE_FLOOR = 0.55  # midan.l3_reasoning — L1 fields below this are treated as unknown for L3

# ── Feature toggles (all default ON) ────────────────────────────────────────
# These are NOT runtime mutable; they exist so the modular architecture can
# disable a layer cleanly during incident response if a rule misfires.
# Flipping one of these to False is a deliberate, audited operation.
ENABLE_IDEA_ADJUSTMENTS      = True  # midan.l2_intelligence — apply idea-derived deltas to macro vector
ENABLE_OFFSETTING            = True  # midan.l4_decision    — allow strong signals to downgrade high risks
ENABLE_CONFLICT_DETECTION    = True  # midan.l4_decision    — run conflict-detection rules
ENABLE_STALENESS_PENALTY     = True  # midan.core / midan.l2 — apply confidence penalty when SARIMA stale

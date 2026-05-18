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
# we made these NOT runtime mutable — they exist so we can disable a layer
# cleanly during incident response if a rule misfires.
# Flipping one of these to False is a deliberate, audited operation.
ENABLE_IDEA_ADJUSTMENTS      = True  # midan.l2_intelligence — apply idea-derived deltas to macro vector
ENABLE_OFFSETTING            = True  # midan.l4_decision    — allow strong signals to downgrade high risks
ENABLE_CONFLICT_DETECTION    = True  # midan.l4_decision    — run conflict-detection rules
ENABLE_STALENESS_PENALTY     = True  # midan.core / midan.l2 — apply confidence penalty when SARIMA stale

# ── Intelligent Score (IS) — 5-signal routing composite ─────────────────────
# we use IS to replace TAS as the routing signal — it is NOT a decision signal.
# It determines which reasoning path the system takes; L4 makes the actual decision.
# Weights sum to 1.0. gap/mu/shap share latent x_scaled geometry (see trio
# discount below). arima is the only fully independent signal.
IS_W_S      = 0.20   # regime favorability (S): direction of macro environment
IS_W_GAP    = 0.25   # gap_svm: SVM probability margin (sorted[0]-sorted[1])
IS_W_MU     = 0.20   # mu_fcm: FCM top-cluster membership (cluster fit)
IS_W_ARIMA  = 0.15   # sarima_trend: normalized sector trajectory (genuinely independent)
IS_W_SHAP   = 0.20   # shap_cosine: SHAP attribution consistency vs cluster mean

# we apply a correlated-trio discount because gap_svm, mu_fcm, and shap_cosine
# all derive from x_scaled geometry. When all three are strongly positive at once,
# they signal "easy case deep inside known territory" — not independent evidence.
# we discount them here to prevent confidence inflation in those cases.
IS_CORRELATED_TRIO_THRESHOLD = 0.80  # all three must exceed this to trigger discount
IS_CORRELATED_TRIO_DISCOUNT  = 0.85  # multiply the trio's combined contribution by this

# ── ReAct routing thresholds ─────────────────────────────────────────────────
# we use REACT_IS_HIGH / REACT_IS_LOW to define the high-certainty and low-certainty
# zones. IS in (REACT_IS_LOW, REACT_IS_HIGH) is the borderline zone where
# additional reasoning (RAG, conflict escalation) is most valuable.
# we use REACT_SHAP_RELIABLE / UNRELIABLE to define attribution consistency zones.
REACT_IS_HIGH           = 0.72  # IS >= this → high-certainty routing zone
REACT_IS_LOW            = 0.35  # IS <= this → clearly unfavorable zone
REACT_SHAP_RELIABLE     = 0.65  # shap_cosine >= this → typical attribution pattern
REACT_SHAP_UNRELIABLE   = 0.40  # shap_cosine <  this → atypical attribution pattern

# ── Explicit RAG (FAISS k-NN precedent lookup) ────────────────────────────────
# we query RAG for borderline IS cases — it returns a majority-vote regime
# label from k nearest training points in combined [x_scaled, shap] space.
# we let ARIMA modify confidence in the RAG vote because a declining sector
# makes historical precedents less reliable.
RAG_K_NEIGHBORS              = 5     # k neighbors for majority vote
ARIMA_RAG_AMPLIFY_THRESHOLD  = 0.65  # sarima_trend >= this → +0.10 to rag confidence
ARIMA_RAG_DAMPEN_THRESHOLD   = 0.35  # sarima_trend <= this → ×0.70 rag confidence

# ── Novelty detection ─────────────────────────────────────────────────────────
# we compute novelty_score = 1 - max_cosine_among_k_nearest_neighbors.
# Scores above threshold mean the query is structurally far from all training points.
# we suppress RAG for novel inputs — a vote from structurally distant precedents
# produces fake confidence, not real evidence.
NOVELTY_THRESHOLD = 0.40   # 1 - max_cosine > 0.40 → novel case, RAG suppressed

# ── SHAP artifact integrity ───────────────────────────────────────────────────
# we compare each cluster's old vs. new SHAP mean by cosine similarity after outer retraining.
# If it falls below this threshold, SHAP attribution semantics have shifted enough
# to corrupt shap_cosine interpretations — we reject the new cluster means and
# preserve the old ones.
SHAP_DRIFT_THRESHOLD = 0.85  # cosine(old_mean, new_mean) < this → SHAP drift detected

# ── Drift detection ───────────────────────────────────────────────────────────
# we require BOTH signals to fire simultaneously (AND gate) to confirm drift.
# Single-signal drift is noise — two independent signals confirming at once is
# the reliable definition of model drift in a bounded macro-grid system.
DRIFT_GAP_SVM_RATIO            = 0.75  # rolling mean gap_svm < ratio×baseline → Signal 1
DRIFT_FCM_L2_THRESHOLD         = 1.50  # centroid L2 displacement > this → cluster drifted
DRIFT_FCM_MIN_CLUSTERS_DRIFTED = 2     # at least this many clusters must drift for Signal 2
DRIFT_MIN_LOG_ENTRIES          = 30    # minimum log entries before drift signals can fire
MACRO_STALENESS_HARD_THRESHOLD = 365   # days: triggers macro_staleness_alert (independent of log)
LOG_MAX_ENTRIES                = 10000 # sliding window cap for prediction_log.jsonl

# ── Outer retraining loop ─────────────────────────────────────────────────────
# we set RETRAIN_ROLLBACK_ACCURACY_FLOOR to reject new artifacts if accuracy dropped
# more than this amount below old accuracy — this prevents a degraded retrain from
# silently replacing a better model.
RETRAIN_ROLLBACK_ACCURACY_FLOOR = 0.05  # new_acc < old_acc - 0.05 → rollback

# ── Asymmetric opportunity detection ─────────────────────────────────────────
# we flag novel cases in upward-trending sectors as potential asymmetric upside —
# we do not penalize for lacking historical precedent in those cases.
ASYMMETRIC_OPPORTUNITY_SARIMA_FLOOR = 0.60  # sarima_trend above this + novelty → flag

# ── Drift auto-trigger + retrain cooldown ────────────────────────────────────
# After every DRIFT_CHECK_INTERVAL logged predictions, drift_monitor checks whether
# both signals have fired. RETRAIN_COOLDOWN_HOURS prevents re-triggering an outer
# loop that just finished — a second drift check within the cooldown window returns
# early with "cooldown_active" status without calling run_outer_react_loop again.
# This protects against feedback spirals where a retrain itself temporarily shifts
# gap_svm/centroid signals and would immediately re-trigger.
DRIFT_CHECK_INTERVAL    = 10   # check drift every N logged predictions
RETRAIN_COOLDOWN_HOURS  = 24   # minimum hours between consecutive outer retrains

# ── RAG tie-vote detection ────────────────────────────────────────────────────
# A tie exists when the top-ranked and second-ranked regime labels in the k-NN
# majority vote receive identical vote counts. We never pick an arbitrary winner —
# we surface the tie explicitly and route to PATH_7 (maximum uncertainty) because
# a tied RAG vote is not evidence; it is ambiguity masquerading as precedent.
RAG_TIE_VOTE_ENABLED = True   # when False, falls back to legacy arbitrary-max behavior

# ── SARIMA soft-reset baseline ───────────────────────────────────────────────
# When the outer loop fires (drift confirmed), old SARIMA forecasts trained on
# pre-drift data become stale. We reset them to a neutral baseline so the IS
# arima component returns 0.50 (neither boosting nor dampening) rather than
# propagating a contaminated trend signal into the next inference cycle.
# sarima_trend = clip(SARIMA_RESET_NEUTRAL_MEAN / 50.0, 0.15, 0.90) = 0.50
SARIMA_RESET_NEUTRAL_MEAN = 25.0  # reset forecast mean → sarima_trend = 0.50

# ── Outcome feedback loop ─────────────────────────────────────────────────────
# Outcomes are logged to logs/outcome_log.jsonl and provide long-horizon signal
# quality data. Calibration metrics only run once OUTCOME_MIN_FOR_CALIBRATION
# outcomes have accumulated PER REGIME — below that threshold, the sample is
# too small for reliable per-regime accuracy estimation.
OUTCOME_LOG_MAX_ENTRIES      = 5000  # sliding window cap for outcome_log.jsonl
OUTCOME_MIN_FOR_CALIBRATION  = 20    # minimum outcomes per regime before calibration runs

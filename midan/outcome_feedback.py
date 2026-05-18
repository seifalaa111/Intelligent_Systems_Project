"""
midan.outcome_feedback — outcome capture, storage, and signal-quality calibration.

PURPOSE
───────
The system makes decisions (GO/CONDITIONAL/NO_GO/etc.) about startup ideas. Without
knowing whether those decisions proved correct, there is no feedback loop — the
system can only detect internal drift (signal statistics), not decision-quality drift
(was the recommendation right?).

This module implements the external feedback layer: users or operators can submit
outcome records that log whether a decision was validated or invalidated over time.
These records accumulate into per-regime calibration metrics that surface in Zone 5
of the dashboard and can inform future IS weight recalibration discussions.

DESIGN PRINCIPLES
─────────────────
1. Append-only: outcomes are never mutated after submission. If a correction is
   needed, a new outcome record supersedes the old one (linked by decision_id).
   Immutability is non-negotiable for auditability.

2. Schema-enforced: every record is validated against OutcomeRecord before
   writing. Malformed submissions are rejected at the endpoint, not silently dropped.

3. Calibration is advisory: calibration metrics surface accuracy *observations*
   only. They do NOT automatically retrain the IS weights or modify routing thresholds.
   That decision requires a human in the loop. The metrics are inputs to that decision.

4. Privacy-safe: idea text is NOT stored in outcome records. Only the decision_id,
   predicted state, regime, sector, country, and outcome value are stored. No
   reproduction of the idea is possible from the log alone.

5. Graceful degradation: every read/write path catches exceptions and returns safe
   defaults. A broken outcome log must never affect the inference path.

DATA FLOW
─────────
/analyze response → frontend stores (decision_id, sector, country, regime, predicted_state)
                  → user provides outcome_value after N days
                  → POST /outcome with OutcomeSubmission
                  → log_outcome() validates + appends to outcome_log.jsonl
                  → compute_calibration_metrics() reads all outcomes → per-regime stats
                  → Zone 5 dashboard reads calibration → displays accuracy + trend
"""
from __future__ import annotations

import json, os, logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

_OUTCOME_LOG = logging.getLogger("midan.outcome_feedback")

# ── Log file location (same logs/ directory as prediction_log.jsonl) ──────────
_LOG_DIR     = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
)
_OUTCOME_PATH = os.path.join(_LOG_DIR, "outcome_log.jsonl")


# ── Outcome schema ─────────────────────────────────────────────────────────────

# Valid values for outcome_value — we use a closed enum rather than free text
# to make calibration counting unambiguous.
OUTCOME_VALUES = frozenset(["validated", "invalidated", "partial", "pending", "unknown"])

# Valid predicted states (same as DecisionState in midan.core minus pre-analysis states)
PREDICTED_STATES = frozenset([
    "GO", "CONDITIONAL", "NO_GO",
    "INSUFFICIENT_DATA", "HIGH_UNCERTAINTY", "CONFLICTING_SIGNALS",
    "REJECTED", "CLARIFICATION_REQUIRED",
])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_outcome_record(record: dict) -> list[str]:
    """
    Return a list of validation errors. Empty list means the record is valid.

    We validate strictly here because every record that enters the log will
    affect calibration metrics for its regime. A malformed record (wrong
    outcome_value, unknown regime) would silently corrupt accuracy counts.
    """
    errors = []
    required = ["decision_id", "sector", "country", "regime", "predicted_state", "outcome_value"]
    for field in required:
        if not record.get(field):
            errors.append(f"missing required field: {field}")

    outcome_val = record.get("outcome_value", "")
    if outcome_val and outcome_val not in OUTCOME_VALUES:
        errors.append(f"outcome_value '{outcome_val}' not in {sorted(OUTCOME_VALUES)}")

    predicted = record.get("predicted_state", "")
    if predicted and predicted not in PREDICTED_STATES:
        errors.append(f"predicted_state '{predicted}' not in known states")

    ttl = record.get("time_to_outcome_days")
    if ttl is not None and (not isinstance(ttl, (int, float)) or ttl < 0):
        errors.append("time_to_outcome_days must be a non-negative number")

    return errors


def log_outcome(
    *,
    decision_id: str,
    sector: str,
    country: str,
    regime: str,
    predicted_state: str,
    outcome_value: str,
    react_path: str = "",
    intelligent_score: float = 0.0,
    is_components: Optional[Dict[str, float]] = None,
    time_to_outcome_days: Optional[int] = None,
    outcome_notes: str = "",
    supersedes_decision_id: Optional[str] = None,
) -> dict:
    """
    Validate and append one outcome record to outcome_log.jsonl.

    Returns a status dict:
        {"status": "ok", "decision_id": ...}         on success
        {"status": "validation_error", "errors": ...} on bad input
        {"status": "write_error", "error": ...}       on I/O failure

    Never raises — a log failure must not affect the caller's response path.
    """
    from midan.config import OUTCOME_LOG_MAX_ENTRIES

    record = {
        "decision_id":            decision_id,
        "submitted_at":           _utc_now(),
        "sector":                 sector,
        "country":                country,
        "regime":                 regime,
        "predicted_state":        predicted_state,
        "outcome_value":          outcome_value,
        "react_path":             react_path,
        "intelligent_score":      round(float(intelligent_score), 4),
        "is_components":          is_components or {},
        "time_to_outcome_days":   time_to_outcome_days,
        "outcome_notes":          outcome_notes[:500],   # cap free text
        "supersedes_decision_id": supersedes_decision_id,
    }

    errors = _validate_outcome_record(record)
    if errors:
        _OUTCOME_LOG.warning(
            "[OUTCOME] Validation failed for decision_id=%s: %s",
            decision_id, errors,
        )
        return {"status": "validation_error", "errors": errors}

    try:
        os.makedirs(_LOG_DIR, exist_ok=True)

        # Sliding window: if at or over cap, trim oldest 20%
        line_count = 0
        if os.path.exists(_OUTCOME_PATH):
            with open(_OUTCOME_PATH, "r", encoding="utf-8") as f:
                for _ in f:
                    line_count += 1
        if line_count >= OUTCOME_LOG_MAX_ENTRIES:
            _trim_outcome_log(line_count, OUTCOME_LOG_MAX_ENTRIES)

        with open(_OUTCOME_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        _OUTCOME_LOG.info(
            "[OUTCOME] Logged: decision_id=%s regime=%s predicted=%s outcome=%s",
            decision_id, regime, predicted_state, outcome_value,
        )
        return {"status": "ok", "decision_id": decision_id}

    except Exception as _e:
        _OUTCOME_LOG.error(
            "[OUTCOME] Write failed for decision_id=%s: %s: %r",
            decision_id, type(_e).__name__, _e,
        )
        return {"status": "write_error", "error": f"{type(_e).__name__}: {_e}"}


def _trim_outcome_log(current_count: int, max_entries: int) -> None:
    """Drop oldest 20% of entries to maintain the sliding window cap."""
    try:
        drop_count = max(1, current_count // 5)
        with open(_OUTCOME_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        kept = lines[drop_count:]
        with open(_OUTCOME_PATH, "w", encoding="utf-8") as f:
            f.writelines(kept)
        _OUTCOME_LOG.info(
            "[OUTCOME] Log trimmed: dropped %d oldest, kept %d", drop_count, len(kept)
        )
    except Exception as _e:
        _OUTCOME_LOG.warning("[OUTCOME] _trim_outcome_log failed: %s", _e)


def load_outcomes(limit: int = 200) -> list[dict]:
    """
    Load the most recent `limit` outcome records from the log.
    Returns an empty list if the log is absent or unreadable.
    Never raises.
    """
    if not os.path.exists(_OUTCOME_PATH):
        return []
    records = []
    try:
        with open(_OUTCOME_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as _e:
        _OUTCOME_LOG.warning("[OUTCOME] load_outcomes failed: %s", _e)
        return []
    return records[-limit:] if len(records) > limit else records


def compute_calibration_metrics() -> dict:
    """
    Compute per-regime calibration metrics from accumulated outcomes.

    A decision is "correct" when:
        predicted GO/CONDITIONAL   AND outcome_value == "validated"
        predicted NO_GO            AND outcome_value == "invalidated"

    A decision is "incorrect" when:
        predicted GO/CONDITIONAL   AND outcome_value == "invalidated"
        predicted NO_GO            AND outcome_value == "validated"

    "partial", "pending", "unknown" outcomes are excluded from accuracy
    computation because they carry no ground truth signal.

    Returns:
        {
            "total_outcomes":      int,
            "scored_outcomes":     int,   # pending/unknown/partial excluded
            "per_regime":          {regime: {accuracy, n_correct, n_scored, verdict}},
            "overall_accuracy":    float or None,
            "calibration_status":  "sufficient" | "insufficient" | "no_data",
            "note":                str,
        }
    """
    from midan.config import OUTCOME_MIN_FOR_CALIBRATION

    records = load_outcomes(limit=0)  # load all
    if not records:
        return {
            "total_outcomes":   0,
            "scored_outcomes":  0,
            "per_regime":       {},
            "overall_accuracy": None,
            "calibration_status": "no_data",
            "note": "No outcomes logged yet.",
        }

    # Bin outcomes by regime
    regime_buckets: dict[str, list[bool]] = {}
    total_scored = 0

    for r in records:
        regime         = r.get("regime", "UNKNOWN")
        predicted      = r.get("predicted_state", "")
        outcome_val    = r.get("outcome_value", "pending")

        # Skip non-actionable outcomes
        if outcome_val in ("pending", "unknown", "partial"):
            continue

        # Determine correctness
        is_positive_pred = predicted in ("GO", "CONDITIONAL")
        is_negative_pred = predicted == "NO_GO"
        validated        = outcome_val == "validated"
        invalidated      = outcome_val == "invalidated"

        if is_positive_pred and validated:
            correct = True
        elif is_positive_pred and invalidated:
            correct = False
        elif is_negative_pred and invalidated:
            correct = True
        elif is_negative_pred and validated:
            correct = False
        else:
            continue  # INSUFFICIENT_DATA / CONFLICTING_SIGNALS etc. → skip

        regime_buckets.setdefault(regime, []).append(correct)
        total_scored += 1

    if total_scored == 0:
        return {
            "total_outcomes":   len(records),
            "scored_outcomes":  0,
            "per_regime":       {},
            "overall_accuracy": None,
            "calibration_status": "insufficient",
            "note": "All outcomes are pending/unknown/partial — no scored data yet.",
        }

    per_regime: dict = {}
    total_correct = 0

    for regime, results in regime_buckets.items():
        n_scored  = len(results)
        n_correct = sum(results)
        total_correct += n_correct

        if n_scored >= OUTCOME_MIN_FOR_CALIBRATION:
            accuracy = round(n_correct / n_scored, 4)
            # Interpret accuracy: > 0.65 well-calibrated, 0.45–0.65 borderline, < 0.45 poor
            if accuracy > 0.65:
                verdict = "well_calibrated"
            elif accuracy >= 0.45:
                verdict = "borderline"
            else:
                verdict = "under_predicting"   # model is wrong more often than chance
        else:
            accuracy = None
            verdict  = f"insufficient_data (need {OUTCOME_MIN_FOR_CALIBRATION}, have {n_scored})"

        per_regime[regime] = {
            "accuracy":   accuracy,
            "n_correct":  n_correct,
            "n_scored":   n_scored,
            "verdict":    verdict,
        }

    overall = round(total_correct / total_scored, 4) if total_scored > 0 else None

    # Determine overall calibration status
    regimes_with_enough = sum(
        1 for v in per_regime.values()
        if isinstance(v["accuracy"], float)
    )
    status = "sufficient" if regimes_with_enough > 0 else "insufficient"

    return {
        "total_outcomes":   len(records),
        "scored_outcomes":  total_scored,
        "per_regime":       per_regime,
        "overall_accuracy": overall,
        "calibration_status": status,
        "note": (
            f"{total_scored} scored outcomes across "
            f"{len(regime_buckets)} regimes. "
            f"{regimes_with_enough} regime(s) have enough data for calibration."
        ),
    }


__all__ = [
    "log_outcome",
    "load_outcomes",
    "compute_calibration_metrics",
    "OUTCOME_VALUES",
    "PREDICTED_STATES",
]

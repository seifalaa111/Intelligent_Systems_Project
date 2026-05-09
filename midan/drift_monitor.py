"""
midan.drift_monitor — prediction logging + dual-signal drift detection.

PREDICTION LOG:
  Append-only JSONL at logs/prediction_log.jsonl. Written synchronously —
  the overhead on a bounded macro-grid system is sub-millisecond.
  Capped at LOG_MAX_ENTRIES to prevent unbounded disk growth. When the cap
  is hit, the oldest 20% of entries are dropped (sliding window).

  x_pca coordinates are logged per prediction to enable centroid drift
  detection (Signal 2) without storing full feature vectors.

DRIFT DETECTION:
  Requires BOTH signals to fire simultaneously (AND gate).
  A single signal firing is noise; two independent signals confirming
  simultaneously is the reliable definition of model drift.

  Signal 1 — Classification margin decline:
    Rolling mean gap_svm (last DRIFT_MIN_LOG_ENTRIES) < DRIFT_GAP_SVM_RATIO
    × baseline mean_gap_svm. The SVM's classification margin is shrinking,
    meaning the features are becoming less discriminative.

  Signal 2 — Centroid displacement:
    Recent cluster centroids (computed from logged x_pca values) have moved
    away from training-time FCM centroids. At least DRIFT_FCM_MIN_CLUSTERS_DRIFTED
    clusters have displaced more than DRIFT_FCM_L2_THRESHOLD.

  MACRO STALENESS ALERT (independent of prediction log):
    When STATIC_MACRO_TABLE_AS_OF is older than MACRO_STALENESS_HARD_THRESHOLD
    days, regime classifications may be based on stale macro data. This alert
    fires without requiring prediction log accumulation — it is a date comparison.
    NOTE: this alert does NOT trigger retraining automatically. Retraining from
    stale tables produces stale artifacts. Manual table refresh is required first.
"""
from midan.core import (
    DRIFT_GAP_SVM_RATIO, DRIFT_FCM_L2_THRESHOLD,
    DRIFT_FCM_MIN_CLUSTERS_DRIFTED, DRIFT_MIN_LOG_ENTRIES,
    MACRO_STALENESS_HARD_THRESHOLD, LOG_MAX_ENTRIES,
    STATIC_MACRO_TABLE_AS_OF,
)
import json, os, logging
import numpy as np

_DRIFT_LOG = logging.getLogger("midan.drift_monitor")

_LOG_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
)
_LOG_PATH = os.path.join(_LOG_DIR, "prediction_log.jsonl")


# ── Prediction logging ────────────────────────────────────────────────────────

def log_prediction(entry: dict) -> None:
    """
    Append one prediction entry to the prediction log.

    Required fields in entry:
        timestamp, request_id, sector, country, regime,
        gap_svm, intelligent_score, shap_cosine, mu_fcm,
        sarima_trend, x_pca (list of 2 floats), novelty_score,
        react_path, l4_decision_state

    Never raises — a logging failure must not break the request path.
    """
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)

        # Read current entry count without loading all entries
        line_count = 0
        if os.path.exists(_LOG_PATH):
            with open(_LOG_PATH, 'r', encoding='utf-8') as f:
                for _ in f:
                    line_count += 1

        # Sliding window: if at or over cap, drop oldest 20%
        if line_count >= LOG_MAX_ENTRIES:
            _trim_log(line_count)

        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    except Exception as _e:
        _DRIFT_LOG.warning("[DRIFT] log_prediction failed (%s: %r) — entry dropped", type(_e).__name__, _e)


def _trim_log(current_count: int) -> None:
    """Drop oldest 20% of entries to implement the sliding window cap."""
    try:
        drop_count = max(1, current_count // 5)
        with open(_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        kept = lines[drop_count:]
        with open(_LOG_PATH, 'w', encoding='utf-8') as f:
            f.writelines(kept)
        _DRIFT_LOG.info("[DRIFT] Log trimmed: dropped %d oldest entries, kept %d", drop_count, len(kept))
    except Exception as _e:
        _DRIFT_LOG.warning("[DRIFT] _trim_log failed (%s) — log may grow beyond cap", _e)


# ── Drift detection ───────────────────────────────────────────────────────────

def check_drift(drift_baseline: dict) -> dict:
    """
    Check for model drift using the dual-signal AND gate.

    Returns a dict with:
        drift_detected          : bool — True only if BOTH signals fire
        signal_1_gap_fired      : bool
        signal_2_centroid_fired : bool
        macro_staleness_alert   : bool — independent of log, always checked
        macro_days_stale        : int or None
        log_entries_analyzed    : int
        details                 : dict — per-signal diagnostic info
    """
    macro_stale, macro_days = _check_macro_staleness()

    # Graceful degradation: no baseline → no drift signals, but still report staleness
    if not drift_baseline:
        return _no_drift_result(
            macro_stale, macro_days, 0, reason="no_baseline",
        )

    # Load prediction log
    entries = _load_log()
    n = len(entries)

    if n < DRIFT_MIN_LOG_ENTRIES:
        return _no_drift_result(
            macro_stale, macro_days, n, reason=f"insufficient_log ({n}<{DRIFT_MIN_LOG_ENTRIES})",
        )

    # Signal 1: gap_svm rolling mean decline
    sig1_fired, sig1_detail = _check_gap_svm_drift(entries, drift_baseline)

    # Signal 2: FCM centroid displacement
    sig2_fired, sig2_detail = _check_centroid_drift(entries, drift_baseline)

    # AND gate: both must fire for confirmed drift
    drift_detected = sig1_fired and sig2_fired

    if drift_detected:
        _DRIFT_LOG.warning(
            "[DRIFT] DRIFT CONFIRMED — Signal1(gap_svm)=%s Signal2(centroid)=%s "
            "after %d log entries. Outer retraining loop should be triggered.",
            sig1_detail.get('rolling_mean_gap', '?'),
            sig2_detail.get('n_clusters_drifted', '?'),
            n,
        )

    return {
        "drift_detected":           drift_detected,
        "signal_1_gap_fired":       sig1_fired,
        "signal_2_centroid_fired":  sig2_fired,
        "macro_staleness_alert":    macro_stale,
        "macro_days_stale":         macro_days,
        "log_entries_analyzed":     n,
        "details": {
            "signal_1": sig1_detail,
            "signal_2": sig2_detail,
        },
    }


def _load_log() -> list:
    """Load prediction log entries. Returns empty list if file absent or malformed."""
    if not os.path.exists(_LOG_PATH):
        return []
    entries = []
    try:
        with open(_LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip malformed lines silently
    except Exception as _e:
        _DRIFT_LOG.warning("[DRIFT] _load_log failed (%s) — returning empty", _e)
    return entries


def _check_gap_svm_drift(entries: list, baseline: dict):
    """
    Signal 1: rolling mean gap_svm decline.
    Uses the last DRIFT_MIN_LOG_ENTRIES entries.
    """
    baseline_mean = baseline.get("mean_gap_svm")
    if baseline_mean is None or baseline_mean <= 0:
        return False, {"reason": "no_baseline_gap_svm"}

    recent = entries[-DRIFT_MIN_LOG_ENTRIES:]
    gap_values = [
        float(e["gap_svm"]) for e in recent
        if isinstance(e.get("gap_svm"), (int, float))
    ]

    if not gap_values:
        return False, {"reason": "no_gap_svm_in_log"}

    rolling_mean = float(np.mean(gap_values))
    threshold    = float(baseline_mean * DRIFT_GAP_SVM_RATIO)
    fired        = rolling_mean < threshold

    return fired, {
        "rolling_mean_gap":   round(rolling_mean, 4),
        "baseline_mean_gap":  round(baseline_mean, 4),
        "threshold":          round(threshold, 4),
        "n_entries_used":     len(gap_values),
    }


def _check_centroid_drift(entries: list, baseline: dict):
    """
    Signal 2: FCM centroid displacement.
    Groups recent predictions by their FCM cluster (inferred from regime) and
    computes how far the centroid of recent x_pca values has moved.
    """
    baseline_centroids = baseline.get("cluster_centroids_pca")
    if not baseline_centroids:
        return False, {"reason": "no_baseline_centroids"}

    # Build recent x_pca per cluster using regime as cluster proxy
    # (regime is the closest available cluster label in the log)
    regime_to_pca: dict = {}
    for e in entries:
        regime = e.get("regime")
        xpca   = e.get("x_pca")
        if regime and isinstance(xpca, list) and len(xpca) == 2:
            if regime not in regime_to_pca:
                regime_to_pca[regime] = []
            regime_to_pca[regime].append(xpca)

    if not regime_to_pca:
        return False, {"reason": "no_x_pca_in_log"}

    baseline_arr = np.array(baseline_centroids)  # (n_clusters, 2)
    drifted_clusters = 0
    per_cluster_drift = {}

    for i, baseline_c in enumerate(baseline_arr):
        # Find which regime key has the most log entries
        # (approximate: we match cluster index to the i-th most common regime)
        for regime, pca_list in regime_to_pca.items():
            if len(pca_list) < 3:  # need at least 3 points for a meaningful centroid
                continue
            recent_centroid = np.mean(pca_list, axis=0)
            l2_dist = float(np.linalg.norm(recent_centroid - baseline_c))
            per_cluster_drift[f"cluster_{i}_{regime}"] = round(l2_dist, 4)
            if l2_dist > DRIFT_FCM_L2_THRESHOLD:
                drifted_clusters += 1

    fired = drifted_clusters >= DRIFT_FCM_MIN_CLUSTERS_DRIFTED

    return fired, {
        "n_clusters_drifted":  drifted_clusters,
        "threshold_l2":        DRIFT_FCM_L2_THRESHOLD,
        "per_cluster_l2":      per_cluster_drift,
    }


def _check_macro_staleness():
    """Independent check — no prediction log needed."""
    from midan.core import _days_since
    days = _days_since(STATIC_MACRO_TABLE_AS_OF)
    if days is None:
        return False, None
    stale = days > MACRO_STALENESS_HARD_THRESHOLD
    if stale:
        _DRIFT_LOG.warning(
            "[DRIFT] MACRO STALENESS ALERT: macro tables are %d days old "
            "(threshold=%d). Manual table refresh required before retraining.",
            days, MACRO_STALENESS_HARD_THRESHOLD,
        )
    return stale, days


def _no_drift_result(macro_stale: bool, macro_days, n: int, reason: str) -> dict:
    return {
        "drift_detected":           False,
        "signal_1_gap_fired":       False,
        "signal_2_centroid_fired":  False,
        "macro_staleness_alert":    macro_stale,
        "macro_days_stale":         macro_days,
        "log_entries_analyzed":     n,
        "details":                  {"reason": reason},
    }


__all__ = ['log_prediction', 'check_drift']

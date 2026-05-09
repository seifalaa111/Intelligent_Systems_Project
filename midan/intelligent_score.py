"""
midan.intelligent_score — 5-signal routing composite (Intelligent Score).

IS is a ROUTING SIGNAL, not a decision signal. It determines which
reasoning path the system takes (react_router.py). L4 makes the decision.

Signal composition:
  S        (0.20) — regime favorability: direction of macro environment
  gap_svm  (0.25) — SVM probability margin: how decisive the classification was
  mu_fcm   (0.20) — FCM top-cluster membership: how typical this case is
  arima    (0.15) — sarima_trend: normalized sector trajectory (genuinely independent)
  shap_cos (0.20) — shap_cosine: SHAP attribution consistency vs cluster mean

Correlated-trio note:
  gap_svm, mu_fcm, and shap_cosine all derive from the same x_scaled geometry.
  They are NOT fully independent. When all three are strongly positive
  simultaneously, a discount is applied to prevent fake consensus inflation.
  arima is the only signal with a different information source.

Authority boundary:
  IS feeds react_router ONLY. It never overwrites L4 decision_state.
  react_router converts IS into a routing path, which is advisory to L4.
"""
from midan.core import (
    IS_W_S, IS_W_GAP, IS_W_MU, IS_W_ARIMA, IS_W_SHAP,
    IS_CORRELATED_TRIO_THRESHOLD, IS_CORRELATED_TRIO_DISCOUNT,
)
import numpy as np


# ── Regime favorability lookup ───────────────────────────────────────────────
# Fixed, not learned. Reflects the macro direction implied by each regime label.
# Asymmetric by design: adverse environments score lower than favorable ones
# score higher, because downside risk is more consequential for routing.
_S_FAVORABLE = {
    'GROWTH_MARKET':      0.85,
    'EMERGING_MARKET':    0.65,
    'HIGH_FRICTION_MARKET': 0.40,
    'CONTRACTING_MARKET': 0.20,
}
_S_DEFAULT = 0.50  # unknown regime → neutral, not penalizing


def _s_favorable(regime: str) -> float:
    return _S_FAVORABLE.get(regime, _S_DEFAULT)


def _gap_svm(proba: np.ndarray) -> float:
    """
    True probability margin: sorted[0] - sorted[1].
    This is NOT proba.max(). proba.max() only captures the winner's
    score; the margin captures how decisive the classification was.
    A high-probability win with a small margin is less reliable than
    the same win with a large margin.
    """
    sorted_p = np.sort(proba)[::-1]
    if len(sorted_p) < 2:
        return float(sorted_p[0]) if len(sorted_p) == 1 else 0.0
    return float(sorted_p[0] - sorted_p[1])


def compute_intelligent_score(
    conf: float,
    proba: np.ndarray,
    mu_fcm: float,
    sarima_trend: float,
    shap_cosine: float,
    regime: str,
) -> dict:
    """
    Compute the Intelligent Score — a weighted composite of 5 signals.

    Parameters
    ----------
    conf         : SVM confidence (regime_conf after staleness penalty) — not
                   used directly in IS formula; kept as parameter for callers
                   that pass it through. IS uses gap_svm from proba directly.
    proba        : numpy array of SVM class probabilities (all classes)
    mu_fcm       : FCM top-cluster membership score in [0, 1]
    sarima_trend : normalized SARIMA trend in [0, 1]
    shap_cosine  : SHAP cosine similarity to cluster mean in [0, 1]
                   0.5 = neutral default when artifact unavailable
    regime       : regime label string for S-favorable lookup

    Returns
    -------
    dict with:
        score              : float in [0, 1] — the IS value
        gap_svm            : float — the probability margin (not proba.max())
        components         : dict of individual signal values before weighting
        is_correlated_trio : bool — True if trio discount was applied
    """
    s     = _s_favorable(regime)
    gap   = _gap_svm(proba)
    mu    = float(np.clip(mu_fcm, 0.0, 1.0))
    arima = float(np.clip(sarima_trend, 0.0, 1.0))
    cos   = float(np.clip(shap_cosine, 0.0, 1.0))

    # Base weighted sum
    raw_is = (
        IS_W_S     * s
        + IS_W_GAP   * gap
        + IS_W_MU    * mu
        + IS_W_ARIMA * arima
        + IS_W_SHAP  * cos
    )

    # Correlated-trio discount: gap, mu, shap_cosine share x_scaled geometry.
    # When all three are strongly high simultaneously, they are all signaling
    # "deep inside known territory" — not providing independent evidence.
    # The discount prevents IS from inflating in easy cases while leaving
    # normal IS behavior unchanged for cases where the signals diverge.
    is_correlated_trio = (
        gap > IS_CORRELATED_TRIO_THRESHOLD
        and mu  > IS_CORRELATED_TRIO_THRESHOLD
        and cos > IS_CORRELATED_TRIO_THRESHOLD
    )
    if is_correlated_trio:
        trio_contribution = IS_W_GAP * gap + IS_W_MU * mu + IS_W_SHAP * cos
        adjusted_contribution = trio_contribution * IS_CORRELATED_TRIO_DISCOUNT
        raw_is = raw_is - trio_contribution + adjusted_contribution

    score = float(np.clip(raw_is, 0.0, 1.0))

    return {
        "score":    score,
        "gap_svm":  gap,
        "components": {
            "s_favorable": round(s,     4),
            "gap_svm":     round(gap,   4),
            "mu_fcm":      round(mu,    4),
            "arima":       round(arima, 4),
            "shap_cosine": round(cos,   4),
        },
        "is_correlated_trio": is_correlated_trio,
    }


__all__ = ['compute_intelligent_score']

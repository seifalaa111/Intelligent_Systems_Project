"""
midan.react_router — deterministic 7-path ReAct routing tree.

The router maps (intelligent_score, shap_cosine, rag_result) onto one of
8 named reasoning paths (7 standard + 1 novelty pre-check).

AUTHORITY BOUNDARY:
  The router is advisory to L4. It can inject signals into L4 inputs
  (adding to conflicting_signals, setting force_human_review) but it
  CANNOT set decision_state directly. L4 remains the decision authority.

NOVELTY PRE-CHECK:
  If rag_result["rag_skipped_reason"] == "novelty", the novelty path fires
  immediately before the main 7-path tree is evaluated. Novelty and uncertainty
  are DISTINCT routing outcomes:
    - Uncertainty: "we can't read the map clearly for this territory"
    - Novelty:     "we don't have a map for this territory"

ROUTING TREE (evaluated in priority order after novelty pre-check):
  Path 1 — HIGH_CERTAINTY:
      IS >= REACT_IS_HIGH AND shap NOT atypical (>= REACT_SHAP_UNRELIABLE) AND no RAG conflict
  Path 2 — LOW_CERTAINTY:
      IS <= REACT_IS_LOW AND shap reliable
  Path 3 — BORDERLINE_CONFIRMED:
      IS borderline AND shap reliable AND rag confirms
  Path 4 — BORDERLINE_CONFLICT:
      IS borderline AND shap reliable AND rag conflicts
  Path 5 — ATYPICAL_REASONING_SUPPORTED:
      shap unreliable AND rag confirms (IS at any level)
  Path 6 — FULL_CONFLICT:
      shap unreliable AND rag conflicts → force_human_review
  Path 7 — MAXIMUM_UNCERTAINTY:
      catch-all for borderline IS + no reliable second opinion → force_human_review

DETERMINISM GUARANTEE:
  The tree is a strict if/elif chain. Every input maps to exactly one path.
  No path selection can be ambiguous. No hidden state.
"""
from midan.core import (
    REACT_IS_HIGH, REACT_IS_LOW,
    REACT_SHAP_RELIABLE, REACT_SHAP_UNRELIABLE,
    ASYMMETRIC_OPPORTUNITY_SARIMA_FLOOR,
)
import logging

_ROUTER_LOG = logging.getLogger("midan.react_router")

# we define path identifiers as string constants for routing path names;
# we surface these in the response for full traceability
PATH_NOVELTY                   = "PATH_NOVELTY"
PATH_1_HIGH_CERTAINTY          = "PATH_1_HIGH_CERTAINTY"
PATH_2_LOW_CERTAINTY           = "PATH_2_LOW_CERTAINTY"
PATH_3_BORDERLINE_CONFIRMED    = "PATH_3_BORDERLINE_CONFIRMED"
PATH_4_BORDERLINE_CONFLICT     = "PATH_4_BORDERLINE_CONFLICT"
PATH_5_ATYPICAL_SUPPORTED      = "PATH_5_ATYPICAL_SUPPORTED"
PATH_6_FULL_CONFLICT           = "PATH_6_FULL_CONFLICT"
PATH_7_MAXIMUM_UNCERTAINTY     = "PATH_7_MAXIMUM_UNCERTAINTY"


def react_route(
    intelligent_score: float,
    shap_cosine: float,
    rag_result: dict,
    svm_regime: str,
    sarima_trend: float,
) -> dict:
    """
    Deterministic 7-path routing.

    Parameters
    ----------
    intelligent_score : float [0, 1] — IS from compute_intelligent_score
    shap_cosine       : float [0, 1] — attribution consistency from rag.py
    rag_result        : dict — output of query_explicit_rag
    svm_regime        : str  — SVM regime label (before rule overrides)
    sarima_trend      : float [0, 1] — for asymmetric_opportunity check

    Returns
    -------
    dict with:
        path_id              : str — one of the PATH_* constants above
        force_human_review   : bool — True means L4 should be INSUFFICIENT_DATA
        rag_conflict_severity: str  — "none", "low", or "medium"
        is_novel_case        : bool
        asymmetric_opportunity: bool
        routing_basis        : str  — one-line explanation of path selection
    """
    if rag_result is None:
        rag_result = {
            "rag_skipped": True, "rag_skipped_reason": "artifact_unavailable",
            "novelty_score": 0.0, "vote": None, "confidence": 0.0,
        }

    rag_skipped        = bool(rag_result.get("rag_skipped", True))
    rag_skipped_reason = rag_result.get("rag_skipped_reason")
    rag_vote           = rag_result.get("vote")     # None when skipped
    novelty_score      = float(rag_result.get("novelty_score", 0.0))

    # we derive routing flags from the input signals
    is_novel       = (rag_skipped and rag_skipped_reason == "novelty")
    rag_confirms   = (not rag_skipped and rag_vote == svm_regime)
    rag_conflicts  = (not rag_skipped and rag_vote is not None and rag_vote != svm_regime)
    shap_reliable  = (shap_cosine >= REACT_SHAP_RELIABLE)
    shap_atypical  = (shap_cosine < REACT_SHAP_UNRELIABLE)
    is_borderline  = (REACT_IS_LOW < intelligent_score < REACT_IS_HIGH)
    is_high        = (intelligent_score >= REACT_IS_HIGH)
    is_low         = (intelligent_score <= REACT_IS_LOW)

    # we flag asymmetric opportunity when it's a novel case + upward sector trend —
    # this is not a routing change, purely a framing flag in the response
    asymmetric = is_novel and (sarima_trend >= ASYMMETRIC_OPPORTUNITY_SARIMA_FLOOR)

    # ── NOVELTY PRE-CHECK — we run this before all other paths ──────────────────────
    # novelty means "no meaningful map for this territory"; RAG has already been suppressed;
    # this path changes framing, not L4 authority
    if is_novel:
        return _result(
            path_id=PATH_NOVELTY,
            force_review=False,
            rag_conflict_severity="none",
            is_novel=True,
            asymmetric=asymmetric,
            basis=(
                f"novelty_score={novelty_score:.3f} > threshold "
                f"({REACT_IS_HIGH if intelligent_score >= REACT_IS_HIGH else 'any IS'}). "
                "RAG suppressed. No structural precedent in training space."
            ),
            rag_vote=None,
        )

    # ── 7-PATH DETERMINISTIC TREE ─────────────────────────────────────────────
    # we evaluate each path in priority order — first match wins;
    # we kept the tree flat and readable by design — no nested conditions

    # Path 1: IS clearly high + shap not fully atypical + no RAG conflict.
    # we accept shap in the borderline zone (0.40–0.65) when IS is well above threshold —
    # borderline shap with high IS is advisory noise, not a blocker; only shap < UNRELIABLE
    # (fully atypical attribution) justifies escalating a high-IS input to Path 7.
    if is_high and not shap_atypical and not rag_conflicts:
        return _result(
            path_id=PATH_1_HIGH_CERTAINTY,
            force_review=False,
            rag_conflict_severity="none",
            is_novel=False,
            asymmetric=False,
            basis=f"IS={intelligent_score:.3f}>={REACT_IS_HIGH} | shap_cosine={shap_cosine:.3f} (not atypical) | no RAG conflict",
            rag_vote=rag_vote,
        )

    # Path 2: we fire this when IS is clearly low + attribution is typical (unfavorable but reliable signal)
    if is_low and shap_reliable:
        return _result(
            path_id=PATH_2_LOW_CERTAINTY,
            force_review=False,
            rag_conflict_severity="none",
            is_novel=False,
            asymmetric=False,
            basis=f"IS={intelligent_score:.3f}<={REACT_IS_LOW} | shap_cosine={shap_cosine:.3f} (reliable) | unfavorable macro",
            rag_vote=rag_vote,
        )

    # Path 3: we fire this when IS is borderline + attribution is typical + RAG confirms SVM
    if is_borderline and shap_reliable and rag_confirms:
        return _result(
            path_id=PATH_3_BORDERLINE_CONFIRMED,
            force_review=False,
            rag_conflict_severity="none",
            is_novel=False,
            asymmetric=False,
            basis=f"IS={intelligent_score:.3f} (borderline) | shap reliable | RAG vote confirms SVM regime",
            rag_vote=rag_vote,
        )

    # Path 4: we fire this when IS is borderline + attribution is typical + RAG conflicts with SVM
    if is_borderline and shap_reliable and rag_conflicts:
        return _result(
            path_id=PATH_4_BORDERLINE_CONFLICT,
            force_review=False,
            rag_conflict_severity="low",
            is_novel=False,
            asymmetric=False,
            basis=(
                f"IS={intelligent_score:.3f} (borderline) | shap reliable | "
                f"RAG vote ({rag_vote}) conflicts with SVM ({svm_regime})"
            ),
            rag_vote=rag_vote,
        )

    # Path 5: we fire this when attribution is atypical but RAG confirms (IS at any level);
    # the model is reasoning unusually for this cluster, but historical precedents agree —
    # we call this a low-severity conflict: atypical but corroborated
    if shap_atypical and rag_confirms:
        return _result(
            path_id=PATH_5_ATYPICAL_SUPPORTED,
            force_review=False,
            rag_conflict_severity="low",
            is_novel=False,
            asymmetric=False,
            basis=(
                f"shap_cosine={shap_cosine:.3f} < {REACT_SHAP_UNRELIABLE} (atypical) | "
                f"RAG vote confirms SVM regime | IS={intelligent_score:.3f}"
            ),
            rag_vote=rag_vote,
        )

    # Path 6: worst case — we fire this when attribution is atypical AND RAG conflicts (two sources of doubt)
    if shap_atypical and rag_conflicts:
        return _result(
            path_id=PATH_6_FULL_CONFLICT,
            force_review=True,
            rag_conflict_severity="medium",
            is_novel=False,
            asymmetric=False,
            basis=(
                f"shap_cosine={shap_cosine:.3f} (atypical) AND "
                f"RAG ({rag_vote}) conflicts with SVM ({svm_regime}). "
                "Two independent sources of doubt — human review required."
            ),
            rag_vote=rag_vote,
        )

    # Path 7: catch-all — we land here on borderline IS + no reliable second opinion;
    # covers: (a) artifact_unavailable, (b) tied_vote, (c) non-novelty skip,
    # (d) is_high/is_low + shap in the gap between UNRELIABLE and RELIABLE.
    # We surface the exact rag_skipped_reason so the routing trace is traceable
    # and the dashboard can explain WHY no second opinion was available.
    _rag_tie     = rag_result.get("rag_tie", False)
    _tied_labels = rag_result.get("tied_labels", [])
    if _rag_tie and _tied_labels:
        _p7_reason = (
            f"RAG vote tied between {_tied_labels} ({rag_result.get('n_neighbors',0)} neighbors, "
            f"equal splits). Tied vote is ambiguity, not evidence — no majority winner accepted."
        )
    elif rag_skipped_reason == "artifact_unavailable":
        _p7_reason = "RAG index artifact not available — no second opinion possible."
    elif rag_skipped_reason == "empty_neighbors":
        _p7_reason = "RAG returned zero usable neighbors — index may be too small."
    else:
        _p7_reason = f"RAG skipped (reason={rag_skipped_reason})."

    return _result(
        path_id=PATH_7_MAXIMUM_UNCERTAINTY,
        force_review=True,
        rag_conflict_severity="none",
        is_novel=False,
        asymmetric=False,
        basis=(
            f"IS={intelligent_score:.3f} | shap_cosine={shap_cosine:.3f} | "
            f"{_p7_reason} Human review required."
        ),
        rag_vote=None,
    )


def _result(path_id, force_review, rag_conflict_severity, is_novel, asymmetric, basis, rag_vote=None) -> dict:
    """Construct the routing result dict. Single exit point ensures consistent shape."""
    return {
        "path_id":               path_id,
        "force_human_review":    force_review,
        "rag_conflict_severity": rag_conflict_severity,
        "is_novel_case":         is_novel,
        "asymmetric_opportunity": asymmetric,
        "routing_basis":         basis,
        "rag_vote":              rag_vote,
    }


__all__ = ['react_route']

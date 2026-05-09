"""
midan.rag — Implicit RAG (SHAP cosine) + Explicit RAG (FAISS k-NN).

IMPORTANT: This module contains two distinct components with different purposes.

compute_shap_cosine — IMPLICIT RAG (attribution consistency check):
    Measures whether the model is reasoning about the current case the same
    way it learned to reason about typical cases in this cluster.
    shap_cosine is an ATTRIBUTION CONSISTENCY signal, NOT a validity signal.
    It does not tell you whether SHAP values are "correct" or whether the
    model "understands" the input. It tells you whether the attribution
    PATTERN matches the cluster's typical pattern.
    shap_cosine = 1.0 → attributions identical to cluster mean (typical)
    shap_cosine = 0.0 → attributions orthogonal to cluster mean (atypical)
    shap_cosine = 0.5 → neutral default when artifact unavailable

query_explicit_rag — EXPLICIT RAG (k-NN precedent lookup):
    For borderline IS cases, retrieves the k most structurally similar
    training points and takes a majority vote on their regime labels.
    This is NOT large-scale semantic retrieval. It is k-NN over a geometric
    space of ~40-64 macro grid training points. The "precedents" are
    training grid points, not real startup outcomes.

    RAG is SUPPRESSED for novel inputs (novelty_score > NOVELTY_THRESHOLD).
    Voting from structurally distant neighbors produces fake confidence.
    Novel cases are better served by first-principles analysis.

    RAG is ADVISORY: its vote feeds react_router, which is advisory to L4.
    L4 remains the decision authority.

ARIMA modifier (inline in query_explicit_rag):
    sarima_trend modulates confidence in the RAG vote — past precedents
    formed under improving conditions are more predictive than those formed
    under declining conditions.
"""
from midan.core import (
    FEATURE_ORDER,
    NOVELTY_THRESHOLD,
    ARIMA_RAG_AMPLIFY_THRESHOLD,
    ARIMA_RAG_DAMPEN_THRESHOLD,
    RAG_K_NEIGHBORS,
)
import numpy as np
import logging

_RAG_LOG = logging.getLogger("midan.rag")


# ── Implicit RAG: SHAP cosine ────────────────────────────────────────────────

def compute_shap_cosine(
    shap_dict: dict,
    shap_cluster_means,       # dict {int: np.ndarray} or None
    top_cluster_idx,          # int or None
    feature_order: list,
) -> float:
    """
    Cosine similarity between the current SHAP attribution vector and
    the cluster mean SHAP vector for the predicted cluster.

    Returns 0.5 (neutral) in all failure/unavailable cases:
    - artifact not generated yet (shap_cluster_means is None)
    - cluster index not found in means dict
    - zero-norm vector (degenerate SHAP output)

    0.5 is deliberately neutral — not 0.0 (which would incorrectly
    penalize the case as having atypical attribution).
    """
    if shap_cluster_means is None:
        return 0.5
    if top_cluster_idx is None:
        return 0.5
    cluster_mean = shap_cluster_means.get(int(top_cluster_idx))
    if cluster_mean is None:
        return 0.5

    try:
        # Build query vector ordered by feature_order
        query_vec = np.array([shap_dict.get(f, 0.0) for f in feature_order], dtype=float)
        mean_vec  = np.array(cluster_mean, dtype=float)

        norm_q = np.linalg.norm(query_vec)
        norm_m = np.linalg.norm(mean_vec)

        if norm_q <= 0 or norm_m <= 0:
            # Zero-norm: SHAP fully uniform or zero — not an error, just uninformative
            return 0.5

        cosine = float(np.dot(query_vec, mean_vec) / (norm_q * norm_m))
        return float(np.clip(cosine, 0.0, 1.0))
    except Exception as _e:
        _RAG_LOG.warning("[RAG] compute_shap_cosine failed (%s) — returning 0.5", _e)
        return 0.5


# ── Explicit RAG: FAISS k-NN + novelty detection ─────────────────────────────

def _build_query_vector(x_scaled_row: np.ndarray, shap_dict: dict, feature_order: list) -> np.ndarray:
    """
    Build the combined query vector: [x_scaled (5D), shap_shares (5D)] → (10,)
    L2-normalized so FAISS IndexFlatIP computes cosine similarity.
    Returns a zero vector on degenerate input (norm=0) — caller handles this.
    """
    shap_vec = np.array([shap_dict.get(f, 0.0) for f in feature_order], dtype=np.float32)
    combined = np.concatenate([x_scaled_row.astype(np.float32), shap_vec])
    norm = np.linalg.norm(combined)
    if norm <= 0:
        return combined  # zero vector — will produce zero cosine similarities
    return (combined / norm).astype(np.float32)


def compute_novelty_score(query_vector: np.ndarray, rag_index, k: int) -> float:
    """
    1 - max_cosine_among_k_nearest_neighbors.

    0.0 → identical to a training point (known territory)
    1.0 → completely orthogonal to all training points (fully novel)

    Returns 0.0 if index is unavailable — never triggers novelty routing
    when the artifact hasn't been generated yet. This is the safe default:
    when we can't measure novelty, we don't claim novelty.
    """
    if rag_index is None:
        return 0.0
    try:
        qv = query_vector.reshape(1, -1).astype(np.float32)
        k_actual = min(k, rag_index.ntotal)
        if k_actual == 0:
            return 0.0
        distances, _ = rag_index.search(qv, k_actual)
        max_cosine = float(distances[0].max())
        return float(np.clip(1.0 - max_cosine, 0.0, 1.0))
    except Exception as _e:
        _RAG_LOG.warning("[RAG] compute_novelty_score failed (%s) — returning 0.0", _e)
        return 0.0


def query_explicit_rag(
    x_scaled_row: np.ndarray,
    shap_dict: dict,
    rag_index,                  # faiss index or None
    rag_labels: list,
    k: int,
    sarima_trend: float,
    feature_order: list,
) -> dict:
    """
    Query the FAISS index for k-NN precedents and return a majority-vote
    regime label with an ARIMA-modulated confidence score.

    Novelty suppression: if novelty_score > NOVELTY_THRESHOLD, RAG is
    skipped entirely. A vote from structurally distant neighbors is not
    evidence — it is noise disguised as precedent.

    ARIMA modifier (applied only when RAG runs):
        trend >= AMPLIFY_THRESHOLD (+0.10): improving sector → past precedents
                                             more predictive of future conditions
        trend <= DAMPEN_THRESHOLD  (×0.70): declining sector → past precedents
                                             less predictive of future conditions

    Returns a dict with:
        rag_skipped        : bool
        rag_skipped_reason : str — "novelty", "artifact_unavailable", or None
        novelty_score      : float
        vote               : str or None — majority-vote regime label
        confidence         : float — fraction of k votes for winning label,
                             after ARIMA modification
        n_neighbors        : int
        vote_distribution  : dict — {label: count}
    """
    # Graceful degradation: artifact not yet generated
    if rag_index is None or not rag_labels:
        return {
            "rag_skipped":        True,
            "rag_skipped_reason": "artifact_unavailable",
            "novelty_score":      0.0,
            "vote":               None,
            "confidence":         0.0,
            "n_neighbors":        0,
            "vote_distribution":  {},
        }

    query_vec = _build_query_vector(x_scaled_row, shap_dict, feature_order)
    novelty   = compute_novelty_score(query_vec, rag_index, k)

    # Novelty gate: suppress RAG when the query is far from all training points
    if novelty > NOVELTY_THRESHOLD:
        return {
            "rag_skipped":        True,
            "rag_skipped_reason": "novelty",
            "novelty_score":      round(novelty, 4),
            "vote":               None,
            "confidence":         0.0,
            "n_neighbors":        0,
            "vote_distribution":  {},
        }

    try:
        k_actual = min(k, rag_index.ntotal)
        distances, indices = rag_index.search(
            query_vec.reshape(1, -1).astype(np.float32), k_actual
        )
        neighbor_labels = [rag_labels[int(idx)] for idx in indices[0] if 0 <= int(idx) < len(rag_labels)]

        if not neighbor_labels:
            return {
                "rag_skipped":        True,
                "rag_skipped_reason": "empty_neighbors",
                "novelty_score":      round(novelty, 4),
                "vote":               None,
                "confidence":         0.0,
                "n_neighbors":        0,
                "vote_distribution":  {},
            }

        # Majority vote
        vote_dist: dict = {}
        for lbl in neighbor_labels:
            vote_dist[lbl] = vote_dist.get(lbl, 0) + 1
        winning_label = max(vote_dist, key=vote_dist.get)
        raw_confidence = vote_dist[winning_label] / len(neighbor_labels)

        # ARIMA modifier: modulates confidence in historical precedent
        # based on sector trend direction. Applied inline — 3 lines of logic
        # do not warrant a separate function.
        confidence = raw_confidence
        if sarima_trend >= ARIMA_RAG_AMPLIFY_THRESHOLD:
            confidence = min(1.0, confidence + 0.10)
        elif sarima_trend <= ARIMA_RAG_DAMPEN_THRESHOLD:
            confidence = confidence * 0.70

        return {
            "rag_skipped":        False,
            "rag_skipped_reason": None,
            "novelty_score":      round(novelty, 4),
            "vote":               winning_label,
            "confidence":         round(float(confidence), 4),
            "n_neighbors":        len(neighbor_labels),
            "vote_distribution":  vote_dist,
        }

    except Exception as _e:
        _RAG_LOG.warning("[RAG] query_explicit_rag failed (%s) — skipping RAG", _e)
        return {
            "rag_skipped":        True,
            "rag_skipped_reason": "error",
            "novelty_score":      round(novelty, 4),
            "vote":               None,
            "confidence":         0.0,
            "n_neighbors":        0,
            "vote_distribution":  {},
        }


__all__ = ['compute_shap_cosine', 'query_explicit_rag', 'compute_novelty_score']

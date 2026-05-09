"""
midan.l2_intelligence — market intelligence layer.

Macro vector construction with explicit (sector × country) base + traceable
idea-derived adjustments. SVM regime classification with visible decision
path (SVM step → optional rule_override → final). FCM parallel fuzzy regime
signal. SHAP over the adjusted vector. SARIMA precomputed lookups guarded
by a freshness gate.
"""
from midan.core import *  # noqa: F401,F403
from midan.l1_parser import UNKNOWN_VALUE  # foundational sentinel  # noqa: F401


# ── extracted from api.py ─────────────────────────────────────────────


def enhanced_regime(svm_regime, svm_conf, inflation, gdp_growth, macro_friction, velocity_yoy):
    """
    Backward-compatible wrapper around the structured decision path.
    Returns (regime, confidence). Use `enhanced_regime_with_path` to get
    the full decision trace.
    """
    final = enhanced_regime_with_path(
        svm_regime, svm_conf, inflation, gdp_growth, macro_friction, velocity_yoy,
    )
    return final["regime"], final["confidence"]


# Hand-coded macro thresholds. Evaluated in priority order; first match wins
# and overrides the SVM. Surfaced verbatim in the response so consumers can
# see exactly which rule (if any) replaced the model output.
_REGIME_RULES = [
    {
        "rule_id": "growth_market_threshold",
        "predicate": lambda inf, gdp, fric, vel: gdp > 3.5 and inf < 8 and vel > 0.15,
        "regime":   "GROWTH_MARKET",
        "explanation": "gdp_growth > 3.5 AND inflation < 8 AND velocity_yoy > 0.15",
        "confidence": lambda inf, gdp, fric, vel: float(np.clip(
            0.65 + min((gdp - 3.5) / 4.0, (8 - inf) / 8.0, (vel - 0.15) / 0.25) * 0.30,
            0.60, 0.95,
        )),
    },
    {
        "rule_id": "emerging_market_threshold",
        "predicate": lambda inf, gdp, fric, vel: gdp > 2.0 and inf < 10 and fric < 10,
        "regime":   "EMERGING_MARKET",
        "explanation": "gdp_growth > 2.0 AND inflation < 10 AND macro_friction < 10",
        "confidence": lambda inf, gdp, fric, vel: float(np.clip(
            0.60 + min((gdp - 2.0) / 4.0, (10 - inf) / 10.0, (10 - fric) / 15.0) * 0.30,
            0.55, 0.90,
        )),
    },
    {
        "rule_id": "contracting_market_severe",
        "predicate": lambda inf, gdp, fric, vel: gdp < 0 or (inf > 50 and fric > 50),
        "regime":   "CONTRACTING_MARKET",
        "explanation": "gdp_growth < 0 OR (inflation > 50 AND macro_friction > 50)",
        "confidence": lambda inf, gdp, fric, vel: float(np.clip(
            0.65 + max(abs(min(gdp, 0)) / 3.0, 0.0) * 0.25, 0.60, 0.92,
        )),
    },
    {
        "rule_id": "high_friction_threshold",
        "predicate": lambda inf, gdp, fric, vel: fric > 30 or inf > 25,
        "regime":   "HIGH_FRICTION_MARKET",
        "explanation": "macro_friction > 30 OR inflation > 25",
        "confidence": lambda inf, gdp, fric, vel: float(np.clip(
            0.60 + max((fric - 30) / 40, (inf - 25) / 30, 0) * 0.30, 0.55, 0.92,
        )),
    },
]


def enhanced_regime_with_path(svm_regime, svm_conf, inflation, gdp_growth,
                              macro_friction, velocity_yoy) -> dict:
    """
    Run the SVM-then-rules pipeline and emit a structured trace of every step.

    Returns:
        {
          "regime": <final regime>,
          "confidence": <final confidence>,
          "decision_path": [
            {"step": "svm",           "regime": ..., "confidence": ..., "source": "model"},
            {"step": "rule_override", "regime": ..., "confidence": ...,
             "rule_id": ..., "explanation": ..., "source": "rule"}  # only if a rule fired
            {"step": "final",         "regime": ..., "confidence": ..., "source": ...},
          ],
        }
    """
    path = [{
        "step":       "svm",
        "regime":     svm_regime,
        "confidence": float(svm_conf),
        "source":     "model",
    }]

    final_regime, final_conf, final_source = svm_regime, float(svm_conf), "model"
    for rule in _REGIME_RULES:
        if rule["predicate"](inflation, gdp_growth, macro_friction, velocity_yoy):
            new_conf = rule["confidence"](inflation, gdp_growth, macro_friction, velocity_yoy)
            path.append({
                "step":        "rule_override",
                "regime":      rule["regime"],
                "confidence":  new_conf,
                "rule_id":     rule["rule_id"],
                "explanation": rule["explanation"],
                "source":      "rule",
            })
            final_regime, final_conf, final_source = rule["regime"], new_conf, "rule"
            break

    path.append({
        "step":       "final",
        "regime":     final_regime,
        "confidence": final_conf,
        "source":     final_source,
    })
    return {
        "regime":        final_regime,
        "confidence":    final_conf,
        "decision_path": path,
    }


# ═══════════════════════════════════════════════════════════════
# L2 — FCM RUNTIME MEMBERSHIP
# Parallel signal to the SVM hard classification. Surfaces fuzziness:
# when memberships are flat (e.g. 0.4/0.35/0.25), the regime is genuinely
# ambiguous and the SVM hard label is less trustworthy.
# ═══════════════════════════════════════════════════════════════

def compute_fcm_membership(x_pca_row, centers, fuzziness: float = 2.0) -> dict:
    """
    Compute fuzzy membership of a single PCA-projected point against the
    pre-trained FCM cluster centers. Membership values sum to 1.0.

    Returns: {<cluster_label>: membership, ...} with `top` and `entropy`
    keys for downstream consumers. `entropy` near log(K) means very ambiguous.
    """
    if centers is None:
        return {"available": False}
    pt = np.asarray(x_pca_row, dtype=float).reshape(-1)
    K  = centers.shape[0]
    # Distance to each center; epsilon avoids division-by-zero for points
    # that coincide exactly with a center.
    dists = np.array([np.linalg.norm(pt - centers[k]) for k in range(K)])
    eps = 1e-9
    dists = np.maximum(dists, eps)
    # Standard FCM membership formula: u_ik = 1 / Σ_j (d_ik / d_jk)^(2/(m-1))
    exp = 2.0 / (fuzziness - 1.0) if fuzziness > 1.0 else 2.0
    inv = 1.0 / (dists ** exp)
    membership = inv / inv.sum()

    labels = {
        i: cluster_names.get(str(i), f"cluster_{i}")
        for i in range(K)
    }
    membership_named = {labels[i]: float(round(membership[i], 4)) for i in range(K)}
    top_idx = int(np.argmax(membership))
    # Shannon entropy normalized by log(K) — 0 = peaked, 1 = flat.
    entropy = float(-(membership * np.log(membership + eps)).sum() / np.log(K))
    return {
        "available":      True,
        "membership":     membership_named,
        "top_cluster":    labels[top_idx],
        "top_membership": float(round(membership[top_idx], 4)),
        "entropy":        round(entropy, 4),
        "is_ambiguous":   entropy > 0.85,
        "source":         "fcm_static_centers",
    }


# ═══════════════════════════════════════════════════════════════
# L2 — IDEA-PERTURBED MACRO VECTOR
# The 5-feature macro vector is otherwise a pure (sector, country) lookup.
# This layer applies SMALL, EXPLICIT, TRACEABLE deltas based on high-confidence
# L1 fields. Rules:
#   • Each adjustment fires only when the L1 source field is non-UNKNOWN
#     AND its confidence ≥ L1_ADJUSTMENT_CONFIDENCE_FLOOR.
#   • Each adjustment is logged with feature, delta, reason_code, source="inferred".
#   • Adjustments are capped — no feature is multiplied, only nudged.
#   • Static base values are preserved alongside adjusted values in the
#     response so consumers can distinguish observed vs inferred.
# ═══════════════════════════════════════════════════════════════

# L1_ADJUSTMENT_CONFIDENCE_FLOOR and ENABLE_IDEA_ADJUSTMENTS are owned by
# midan.config and reach this module via `from midan.core import *`.

def _idea_macro_adjustments(
    base_macro: dict,
    l1_result: Optional[dict],
) -> list:
    """
    Build the list of idea-derived adjustments for the macro vector.
    Returns [] when L1 is missing, confidence is below the floor, or the
    ENABLE_IDEA_ADJUSTMENTS toggle has been flipped off in midan.config.
    """
    if not ENABLE_IDEA_ADJUSTMENTS:
        return []
    if not l1_result:
        return []
    values     = l1_result.get("values", {})
    confidence = l1_result.get("confidence", {})

    def _has(field: str, expected) -> bool:
        v = values.get(field)
        c = confidence.get(field, 0.0)
        if v in (UNKNOWN_VALUE, None):
            return False
        if c < L1_ADJUSTMENT_CONFIDENCE_FLOOR:
            return False
        if isinstance(expected, (list, tuple, set)):
            return v in expected
        return v == expected

    adjustments: list = []

    # B2C consumer-velocity bias — mass-market consumer ideas inherit a
    # slightly higher capital velocity signal than the sector default.
    if _has('target_segment', 'b2c'):
        adjustments.append({
            "feature":     "velocity_yoy",
            "delta":       +0.02,
            "reason_code": "b2c_consumer_velocity",
            "explanation": "B2C consumer ideas correlate with higher capital velocity than the sector-typical default.",
            "source_field": "target_segment",
            "source_value": "b2c",
            "source":      "inferred",
        })

    # B2G regulatory friction — government-customer ideas inherit higher
    # friction even if the country baseline is neutral.
    if _has('target_segment', 'b2g'):
        adjustments.append({
            "feature":     "macro_friction",
            "delta":       +5.0,
            "reason_code": "b2g_regulatory_friction",
            "explanation": "B2G procurement cycles add structural friction that the country-level baseline does not capture.",
            "source_field": "target_segment",
            "source_value": "b2g",
            "source":      "inferred",
        })

    # High regulatory_risk → friction bump
    if _has('regulatory_risk', 'high'):
        adjustments.append({
            "feature":     "macro_friction",
            "delta":       +3.0,
            "reason_code": "high_regulatory_risk_friction",
            "explanation": "L1 flagged regulatory_risk=high; adds friction beyond country-level baseline.",
            "source_field": "regulatory_risk",
            "source_value": "high",
            "source":      "inferred",
        })

    # Growth-stage momentum — late-stage ideas operate in a more crowded
    # field; nudge velocity up slightly to reflect higher transaction tempo.
    if _has('stage', 'growth'):
        adjustments.append({
            "feature":     "velocity_yoy",
            "delta":       +0.03,
            "reason_code": "growth_stage_momentum",
            "explanation": "Growth-stage ideas operate where capital velocity is empirically higher than at idea/MVP stage.",
            "source_field": "stage",
            "source_value": "growth",
            "source":      "inferred",
        })

    # High competitive_intensity → small inflation drag (proxy for pricing pressure)
    if _has('competitive_intensity', 'high'):
        adjustments.append({
            "feature":     "inflation",
            "delta":       -0.5,
            "reason_code": "high_competition_pricing_drag",
            "explanation": "High competitive intensity implies pricing pressure — slightly lower effective inflation pass-through to the customer.",
            "source_field": "competitive_intensity",
            "source_value": "high",
            "source":      "inferred",
        })

    return adjustments


def apply_idea_adjustments(base_macro: dict, adjustments: list) -> dict:
    """
    Apply the adjustment list to the base macro dict. Each feature's delta
    is summed across adjustments. Caps protect against runaway perturbation.

    Returns a new dict {feature: adjusted_value}; base_macro is unchanged.
    """
    # Conservative caps — total absolute deviation per feature.
    caps = {
        "inflation":             3.0,
        "gdp_growth":            1.0,
        "macro_friction":       10.0,
        "capital_concentration": 0.0,  # no idea-level adjustments to capital stock
        "velocity_yoy":          0.08,
    }
    deltas = {f: 0.0 for f in base_macro}
    for adj in adjustments:
        f = adj["feature"]
        if f in deltas:
            deltas[f] += float(adj["delta"])
    capped = {f: float(np.clip(deltas[f], -caps.get(f, 0.0), caps.get(f, 0.0))) for f in deltas}
    return {f: base_macro[f] + capped[f] for f in base_macro}, capped

# ═══════════════════════════════════════════════════════════════

def compute_shap(lgb_model, x_scaled_row, predicted_class_idx: int = None):
    """
    Compute normalized SHAP feature importance shares for the predicted class.

    Bug fix (v2): sv.shape=(1,5,3) — use predicted class's SHAP values, not
    mean across all classes. Mean dilutes the signal of the actual prediction.

    Soft-cap: no single feature can claim >65% of the display share so the bar
    chart does not collapse to a single signal. Values still sum to 1.0 after
    redistribution.
    """
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(lgb_model)
    sv = explainer.shap_values(x_scaled_row)

    if isinstance(sv, list):
        # Old SHAP API: list of arrays per class
        if predicted_class_idx is not None and predicted_class_idx < len(sv):
            arr = np.abs(sv[predicted_class_idx][0])
        else:
            arr = np.mean([np.abs(s) for s in sv], axis=0)[0]
    elif hasattr(sv, 'ndim') and sv.ndim == 3:
        # New SHAP API: shape (n_samples, n_features, n_classes)
        # Use the predicted class's SHAP values for the single input row
        if predicted_class_idx is not None and predicted_class_idx < sv.shape[2]:
            arr = np.abs(sv[0, :, predicted_class_idx])
        else:
            arr = np.abs(sv[0]).mean(axis=-1)
    else:
        arr = np.abs(sv)[0]

    raw   = {k: float(v) for k, v in zip(FEATURES, arr)}
    total = sum(raw.values())
    if total <= 0:
        return {k: round(1.0 / len(FEATURES), 4) for k in FEATURES}

    shares = {k: v / total for k, v in raw.items()}

    # Soft-cap: redistribute overflow above MAX_SHARE to uncapped features
    # preserving the relative ranking while preventing a single-bar collapse
    MAX_SHARE = 0.65
    overcapped = {k: v for k, v in shares.items() if v > MAX_SHARE}
    if overcapped:
        overflow  = sum(v - MAX_SHARE for v in overcapped.values())
        free_keys = [k for k in shares if shares[k] <= MAX_SHARE]
        bonus     = overflow / len(free_keys) if free_keys else 0
        for k in shares:
            shares[k] = MAX_SHARE if shares[k] > MAX_SHARE else min(MAX_SHARE, shares[k] + bonus)

    return {k: round(float(v), 4) for k, v in shares.items()}




# Export everything defined in this module — including underscore-prefixed
# helpers — so other midan submodules can wildcard-import the full surface.
__all__ = [name for name in list(globals().keys()) if not name.startswith('__')]

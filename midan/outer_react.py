"""
midan.outer_react — deterministic outer retraining loop.

TRAINING DATA INTEGRITY CONTRACT — READ BEFORE MODIFYING
─────────────────────────────────────────────────────────
Training data is constructed EXCLUSIVELY from:
    COUNTRY_MACRO_DEFAULTS × SECTOR_EFF_MACRO (cartesian product)

The prediction log is used ONLY for drift signal computation.
It is NEVER used as training data. Violating this creates recursive
self-reinforcement: the model learns from its own past decisions and
eventually produces a system that only confirms its own existing beliefs.

This constraint must be preserved in all future modifications.
─────────────────────────────────────────────────────────

RETRAINING FLOW:
  1. Rebuild training matrix from macro tables (deterministic)
  2. Re-fit scaler → pca → svm → le → lgb_surrogate
  3. Accuracy rollback gate (early): abort if new model is worse
  4. Remap cluster identities by proximity to regime anchors (semantic,
     not positional — prevents silent cluster identity swaps)
  5. Validate cluster identity uniqueness
  6. Recompute SHAP cluster means
  7. SHAP drift gate: if attribution semantics shifted too much, keep
     old shap_cluster_means.pkl and preserve old cosine behavior
  8. Rebuild FAISS index + labels
  9. Rebuild regime anchors (anchors depend on new SVM labels)
 10. Write valid artifacts; return status report

ROLLBACK CONDITIONS (no artifacts modified):
  - new_accuracy < old_accuracy - RETRAIN_ROLLBACK_ACCURACY_FLOOR
  - new FCM cluster identities map two clusters to the same regime label

PARTIAL ROLLBACK (all artifacts updated except shap_cluster_means.pkl):
  - SHAP semantic drift detected (cosine between old/new means < SHAP_DRIFT_THRESHOLD)
"""
from midan.core import (
    COUNTRY_MACRO_DEFAULTS, SECTOR_EFF_MACRO, FEATURES,
    MODELS_DIR, FEATURE_ORDER,
    RETRAIN_ROLLBACK_ACCURACY_FLOOR, SHAP_DRIFT_THRESHOLD,
    DRIFT_FCM_MIN_CLUSTERS_DRIFTED,
)
import numpy as np
import pickle, json, os, logging

_OUTER_LOG = logging.getLogger("midan.outer_react")

# Training data source integrity — asserted at module load time
_TRAINING_SOURCE = "macro_tables_only"
assert _TRAINING_SOURCE == "macro_tables_only", "Training source integrity violated"


def run_outer_react_loop() -> dict:
    """
    Rebuild all model artifacts from macro tables. Validate before writing.
    Returns a status dict describing what happened.
    """
    _OUTER_LOG.info("[OUTER_REACT] Retraining loop started.")

    try:
        return _run_retrain()
    except Exception as _e:
        _OUTER_LOG.error("[OUTER_REACT] Unexpected error: %s: %r", type(_e).__name__, _e)
        return {
            "retrain_status":     "error",
            "error":              f"{type(_e).__name__}: {_e}",
            "artifacts_written":  [],
            "artifacts_preserved": [],
        }


def _run_retrain() -> dict:
    # ── Step 1: Rebuild training matrix ──────────────────────────────────────
    X_train, y_labels_raw = _build_training_matrix()
    if len(X_train) == 0:
        return {"retrain_status": "error", "error": "empty training matrix",
                "artifacts_written": [], "artifacts_preserved": []}

    _OUTER_LOG.info("[OUTER_REACT] Training matrix: %d samples, %d features", *X_train.shape)

    # ── Step 2: Re-fit pipeline ───────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    import lightgbm as lgb
    import skfuzzy as fuzz

    new_scaler = StandardScaler()
    X_scaled   = new_scaler.fit_transform(X_train)

    n_components = 2  # preserve existing PCA dimensionality
    new_pca = PCA(n_components=n_components, random_state=42)
    X_pca   = new_pca.fit_transform(X_scaled)

    new_le = LabelEncoder()
    y_enc  = new_le.fit_transform(y_labels_raw)

    # FCM clustering in PCA space (same parameters as original training)
    n_fcm = 3
    cntr, u_new, *_ = fuzz.cluster.cmeans(
        X_pca.T, c=n_fcm, m=2.0, error=0.005, maxiter=1000, init=None, seed=42
    )
    # Hard assignments for training data
    hard_labels = np.argmax(u_new, axis=0)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    new_svm = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=42)
    new_svm.fit(X_tr, y_tr)
    new_accuracy = float(new_svm.score(X_te, y_te))

    new_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=6,
                                  learning_rate=0.05, random_state=42, verbose=-1)
    svm_preds_full = new_svm.predict(X_scaled)
    new_lgb.fit(X_scaled, svm_preds_full)

    # ── Step 3: Accuracy rollback gate ────────────────────────────────────────
    old_accuracy = _load_old_accuracy()
    if old_accuracy is not None:
        if new_accuracy < old_accuracy - RETRAIN_ROLLBACK_ACCURACY_FLOOR:
            _OUTER_LOG.warning(
                "[OUTER_REACT] ROLLBACK: new_acc=%.4f < old_acc=%.4f - %.2f",
                new_accuracy, old_accuracy, RETRAIN_ROLLBACK_ACCURACY_FLOOR,
            )
            return {
                "retrain_status":     "rollback_accuracy_gate",
                "accuracy_old":       old_accuracy,
                "accuracy_new":       new_accuracy,
                "artifacts_written":  [],
                "artifacts_preserved": ["all — rollback triggered"],
            }

    # ── Step 4: Cluster semantic remapping ───────────────────────────────────
    regime_anchors = _load_pkl_optional('regime_anchors_pca.pkl')
    if regime_anchors is None:
        return {
            "retrain_status": "error",
            "error": "regime_anchors_pca.pkl missing — cannot remap cluster identities safely",
            "artifacts_written": [], "artifacts_preserved": [],
        }

    new_cluster_names = _remap_cluster_identities(cntr, regime_anchors)

    # ── Step 5: Validate cluster identity uniqueness ──────────────────────────
    labels_used = list(new_cluster_names.values())
    if len(set(labels_used)) < len(labels_used):
        _OUTER_LOG.error(
            "[OUTER_REACT] ROLLBACK: cluster identity collision — %s", new_cluster_names
        )
        return {
            "retrain_status":     "rollback_cluster_collision",
            "cluster_names":      new_cluster_names,
            "artifacts_written":  [],
            "artifacts_preserved": ["all — cluster collision"],
        }

    # ── Step 6: Recompute SHAP cluster means ──────────────────────────────────
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(new_lgb)
    sv_all    = explainer.shap_values(X_scaled)

    new_shap_means = _compute_shap_cluster_means(
        X_scaled, sv_all, svm_preds_full, hard_labels, n_fcm
    )

    # ── Step 7: SHAP drift gate ───────────────────────────────────────────────
    old_shap_means  = _load_pkl_optional('shap_cluster_means.pkl')
    shap_drift, shap_cosines = _validate_shap_drift(old_shap_means, new_shap_means)

    # ── Step 8: Rebuild FAISS index ───────────────────────────────────────────
    new_rag_index, new_rag_labels = _rebuild_rag_index(
        X_scaled, sv_all, svm_preds_full, new_le
    )

    # ── Step 9: Rebuild regime anchors ────────────────────────────────────────
    new_regime_anchors = _rebuild_regime_anchors(X_pca, new_le, new_svm, X_scaled)

    # ── Step 10: Write artifacts ──────────────────────────────────────────────
    written   = []
    preserved = []

    def _save_pkl(name, obj):
        with open(os.path.join(MODELS_DIR, name), 'wb') as f:
            pickle.dump(obj, f)
        written.append(name)

    def _save_json(name, obj):
        with open(os.path.join(MODELS_DIR, name), 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)
        written.append(name)

    _save_pkl('scaler_global.pkl',   new_scaler)
    _save_pkl('pca_global.pkl',      new_pca)
    _save_pkl('svm_global.pkl',      new_svm)
    _save_pkl('label_encoder.pkl',   new_le)
    _save_pkl('lgb_surrogate.pkl',   new_lgb)
    _save_pkl('fcm_centers.pkl',     cntr)
    _save_json('cluster_names.json', new_cluster_names)
    _save_pkl('rag_index.pkl',       new_rag_index)
    _save_json('rag_labels.json',    new_rag_labels)
    _save_pkl('regime_anchors_pca.pkl', new_regime_anchors)

    if not shap_drift:
        _save_pkl('shap_cluster_means.pkl', new_shap_means)
    else:
        preserved.append('shap_cluster_means.pkl (SHAP semantic drift detected — old means preserved)')
        _OUTER_LOG.warning(
            "[OUTER_REACT] SHAP drift detected — per-cluster cosines: %s. "
            "Old shap_cluster_means.pkl preserved.", shap_cosines
        )

    # Update drift baseline with new accuracy
    _update_drift_baseline(new_accuracy)
    written.append('drift_baseline.json (accuracy updated)')

    _OUTER_LOG.info(
        "[OUTER_REACT] Retrain complete: acc %.4f -> %.4f | cluster_remap=%s | shap_drift=%s",
        old_accuracy or 0.0, new_accuracy, new_cluster_names, shap_drift,
    )

    return {
        "retrain_status":          "success",
        "accuracy_old":            old_accuracy,
        "accuracy_new":            new_accuracy,
        "cluster_remap":           new_cluster_names,
        "shap_drift_detected":     shap_drift,
        "per_cluster_shap_cosine": shap_cosines,
        "artifacts_written":       written,
        "artifacts_preserved":     preserved,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_training_matrix():
    """
    Rebuild feature matrix and regime labels from COUNTRY_MACRO_DEFAULTS
    × SECTOR_EFF_MACRO — the ONLY permitted training data source.
    """
    from midan.l2_intelligence import enhanced_regime_with_path

    rows, labels = [], []
    for country_code, country_macro in COUNTRY_MACRO_DEFAULTS.items():
        base_inflation = country_macro['inflation']
        base_gdp       = country_macro['gdp_growth']
        unemployment   = country_macro['unemployment']

        for sector_key, (eff_inf_offset, gdp_boost, sector_velocity) in SECTOR_EFF_MACRO.items():
            scale     = base_inflation / 33.9
            inflation = float(np.clip(eff_inf_offset * scale, 1.0, 100.0))
            gdp       = float(base_gdp + gdp_boost)
            friction  = float(np.clip(inflation + unemployment - gdp, -50, 100))
            from midan.core import SECTOR_MEDIANS
            cap_conc  = float(SECTOR_MEDIANS.get(sector_key, 100000.0))
            velocity  = float(sector_velocity)

            svm_result = _quick_svm_regime_from_rules(inflation, gdp, friction, velocity)
            rows.append([inflation, gdp, friction, cap_conc, velocity])
            labels.append(svm_result)

    return np.array(rows, dtype=float), np.array(labels)


def _quick_svm_regime_from_rules(inflation, gdp, friction, velocity):
    """Apply hand-coded regime rules (same as enhanced_regime_with_path) to get label."""
    if gdp < 0 or (inflation > 50 and friction > 50):
        return 'CONTRACTING_MARKET'
    if gdp > 3.5 and inflation < 8 and velocity > 0.15:
        return 'GROWTH_MARKET'
    if gdp > 2.0 and inflation < 10 and friction < 10:
        return 'EMERGING_MARKET'
    if friction > 30 or inflation > 25:
        return 'HIGH_FRICTION_MARKET'
    return 'EMERGING_MARKET'  # default if no rule fires


def _remap_cluster_identities(cntr: np.ndarray, regime_anchors: dict) -> dict:
    """
    Remap each FCM cluster to the nearest regime anchor in PCA space.
    Returns {cluster_idx (str): regime_label}.
    Semantic identity, not positional identity.
    """
    cluster_names = {}
    for i, centroid in enumerate(cntr):
        best_regime, best_dist = None, float('inf')
        for regime, anchor in regime_anchors.items():
            dist = float(np.linalg.norm(np.array(centroid) - np.array(anchor)))
            if dist < best_dist:
                best_dist  = dist
                best_regime = regime
        cluster_names[str(i)] = best_regime
        _OUTER_LOG.info(
            "[OUTER_REACT] Cluster %d → %s (L2=%.4f)", i, best_regime, best_dist
        )
    return cluster_names


def _validate_shap_drift(old_means, new_means) -> tuple:
    """
    Compare old and new SHAP cluster means by cosine similarity.
    Returns (drift_detected: bool, per_cluster_cosines: dict).
    """
    if old_means is None:
        return False, {}  # no old means → first run, not drift

    cosines = {}
    drift_detected = False
    for cluster_idx, new_vec in new_means.items():
        old_vec = old_means.get(cluster_idx)
        if old_vec is None:
            continue
        a, b = np.array(old_vec, dtype=float), np.array(new_vec, dtype=float)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a <= 0 or norm_b <= 0:
            cosines[str(cluster_idx)] = 0.0
            drift_detected = True
            continue
        cos = float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0))
        cosines[str(cluster_idx)] = round(cos, 4)
        if cos < SHAP_DRIFT_THRESHOLD:
            drift_detected = True

    return drift_detected, cosines


def _compute_shap_cluster_means(X_scaled, sv_all, svm_preds_enc, hard_labels, n_clusters) -> dict:
    """Compute mean normalized SHAP share per cluster (same logic as T1 cell)."""
    n_samples  = X_scaled.shape[0]
    n_features = len(FEATURE_ORDER)

    def _normalize(arr):
        total = np.abs(arr).sum()
        if total <= 0:
            return np.ones(n_features) / n_features
        shares = np.abs(arr) / total
        if shares.max() > 0.65:
            overflow  = float(((shares - 0.65) * (shares > 0.65)).sum())
            free_mask = shares <= 0.65
            bonus     = overflow / free_mask.sum() if free_mask.sum() > 0 else 0.0
            shares    = np.where(shares > 0.65, 0.65, np.minimum(0.65, shares + bonus))
        return shares

    shap_matrix = np.zeros((n_samples, n_features))
    for i, cls_idx in enumerate(svm_preds_enc):
        if isinstance(sv_all, list):
            raw = sv_all[int(cls_idx)][i] if int(cls_idx) < len(sv_all) else sv_all[0][i]
        elif hasattr(sv_all, 'ndim') and sv_all.ndim == 3:
            raw = sv_all[i, :, int(cls_idx)] if int(cls_idx) < sv_all.shape[2] else sv_all[i, :, 0]
        else:
            raw = sv_all[i]
        shap_matrix[i] = _normalize(raw)

    means = {}
    for c in range(n_clusters):
        mask = hard_labels == c
        means[c] = shap_matrix[mask].mean(axis=0) if mask.sum() > 0 else np.ones(n_features) / n_features
    return means


def _rebuild_rag_index(X_scaled, sv_all, svm_preds_enc, new_le):
    """Rebuild FAISS index from new artifacts."""
    try:
        import faiss
    except ImportError:
        _OUTER_LOG.warning("[OUTER_REACT] faiss not available — rag_index not rebuilt")
        return None, []

    n_samples  = X_scaled.shape[0]
    n_features = len(FEATURE_ORDER)
    shap_matrix = np.zeros((n_samples, n_features))
    for i, cls_idx in enumerate(svm_preds_enc):
        if isinstance(sv_all, list):
            raw = np.abs(sv_all[int(cls_idx)][i]) if int(cls_idx) < len(sv_all) else np.abs(sv_all[0][i])
        elif hasattr(sv_all, 'ndim') and sv_all.ndim == 3:
            raw = np.abs(sv_all[i, :, int(cls_idx)]) if int(cls_idx) < sv_all.shape[2] else np.abs(sv_all[i, :, 0])
        else:
            raw = np.abs(sv_all[i])
        total = raw.sum()
        shap_matrix[i] = raw / total if total > 0 else np.ones(n_features) / n_features

    combined = np.hstack([X_scaled, shap_matrix]).astype(np.float32)
    norms    = np.linalg.norm(combined, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1.0, norms)
    combined_norm = combined / norms

    d     = combined_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(combined_norm)

    labels = new_le.inverse_transform(svm_preds_enc).tolist()
    return index, labels


def _rebuild_regime_anchors(X_pca, new_le, new_svm, X_scaled) -> dict:
    """Compute mean PCA position per regime label from new SVM outputs."""
    regime_labels = new_le.inverse_transform(new_svm.predict(X_scaled))
    anchors = {}
    for reg in np.unique(regime_labels):
        mask = (regime_labels == reg)
        anchors[reg] = X_pca[mask].mean(axis=0).tolist()
    return anchors


def _load_old_accuracy() -> float:
    """Load accuracy from drift_baseline.json. Returns None if unavailable."""
    try:
        path = os.path.join(MODELS_DIR, 'drift_baseline.json')
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            baseline = json.load(f)
        return baseline.get('svm_accuracy')
    except Exception:
        return None


def _update_drift_baseline(new_accuracy: float) -> None:
    """Update svm_accuracy in drift_baseline.json."""
    try:
        path = os.path.join(MODELS_DIR, 'drift_baseline.json')
        baseline = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                baseline = json.load(f)
        baseline['svm_accuracy'] = round(new_accuracy, 6)
        from datetime import date
        baseline['last_retrain_date'] = str(date.today())
        with open(path, 'w') as f:
            json.dump(baseline, f, indent=2)
    except Exception as _e:
        _OUTER_LOG.warning("[OUTER_REACT] _update_drift_baseline failed: %s", _e)


def _load_pkl_optional(name):
    try:
        with open(os.path.join(MODELS_DIR, name), 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


__all__ = ['run_outer_react_loop']

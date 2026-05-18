"""
build_rag_artifacts.py — one-time generation of the three missing runtime artifacts:
    models/rag_index.pkl         FAISS IndexFlatIP over [x_scaled(5D), shap_shares(5D)]
    models/rag_labels.json       regime labels for each indexed point
    models/shap_cluster_means.pkl  {cluster_idx: ndarray(5,)} mean SHAP per FCM cluster

Run once after the main model artifacts are trained.
Safe to re-run — always overwrites with a fresh build.
"""
import pickle, json, os, warnings
import numpy as np
import faiss

warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _load(name):
    with open(os.path.join(MODELS_DIR, name), 'rb') as f:
        return pickle.load(f)

def _save_pkl(obj, name):
    with open(os.path.join(MODELS_DIR, name), 'wb') as f:
        pickle.dump(obj, f)
    print(f"  saved {name}")

def _save_json(obj, name):
    with open(os.path.join(MODELS_DIR, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    print(f"  saved {name}")

print("Loading model artifacts…")
scaler      = _load('scaler_global.pkl')
pca         = _load('pca_global.pkl')
svm         = _load('svm_global.pkl')
lgb         = _load('lgb_surrogate.pkl')
le          = _load('label_encoder.pkl')
fcm_centers = _load('fcm_centers.pkl')

with open(os.path.join(MODELS_DIR, 'cluster_names.json'), 'r', encoding='utf-8') as f:
    cluster_names = json.load(f)

FEATURE_ORDER = ['inflation', 'gdp_growth', 'macro_friction', 'capital_concentration', 'velocity_yoy']

# ── 1. Synthetic training grid ────────────────────────────────────────────────
# We replicate the same distribution used in _cluster_scatter() so the FAISS
# index covers the same region the SVM was trained on.
print("Generating synthetic training grid…")
rng = np.random.RandomState(0)
N   = 2000

X_raw = np.column_stack([
    rng.normal(scaler.mean_[0], scaler.scale_[0] * 0.7, N).clip(0,   60),
    rng.normal(scaler.mean_[1], scaler.scale_[1] * 0.8, N).clip(-4,  10),
    rng.normal(scaler.mean_[2], scaler.scale_[2] * 0.7, N).clip(-15, 60),
    rng.normal(scaler.mean_[3], scaler.scale_[3] * 0.4, N).clip(5000, 3e6),
    rng.normal(scaler.mean_[4], scaler.scale_[4] * 0.8, N).clip(0,   0.5),
])
X_scaled = scaler.transform(X_raw)
X_pca    = pca.transform(X_scaled)

# SVM labels (hard class)
pred_enc = svm.predict(X_scaled)
labels   = list(le.inverse_transform(pred_enc))

# ── 2. SHAP over each grid point ──────────────────────────────────────────────
print("Computing SHAP for each grid point (this takes ~30s)…")
import shap as shap_lib
explainer  = shap_lib.TreeExplainer(lgb)
shap_vals  = explainer.shap_values(X_scaled)

# shap_vals can be (N, features) or list-of-arrays; normalise to (N, features) mean
if isinstance(shap_vals, list):
    sv_arr = np.mean([np.abs(s) for s in shap_vals], axis=0)   # (N, 5)
elif shap_vals.ndim == 3:
    sv_arr = np.abs(shap_vals).mean(axis=-1)                   # (N, 5)
else:
    sv_arr = np.abs(shap_vals)                                  # (N, 5)

# ── 3. Build combined query vectors [x_scaled | shap_shares] → (N, 10) ───────
print("Building FAISS index…")
shap_totals = sv_arr.sum(axis=1, keepdims=True).clip(1e-9, None)
shap_shares = sv_arr / shap_totals                              # (N, 5) normalised shares

combined = np.concatenate([X_scaled.astype(np.float32), shap_shares.astype(np.float32)], axis=1)  # (N, 10)

# L2-normalise each row so IndexFlatIP computes cosine similarity
norms = np.linalg.norm(combined, axis=1, keepdims=True).clip(1e-9, None)
combined_normed = (combined / norms).astype(np.float32)

dim   = combined_normed.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(combined_normed)
print(f"  FAISS index: {index.ntotal} vectors, dim={dim}")

# ── 4. shap_cluster_means: mean SHAP vector per FCM cluster ──────────────────
print("Computing shap_cluster_means per FCM cluster…")
K = fcm_centers.shape[0]

# FCM assignment for each training point: hard-assign to nearest center
fcm_assigns = []
for i in range(len(X_pca)):
    pt = X_pca[i]
    dists = [np.linalg.norm(pt - fcm_centers[k]) for k in range(K)]
    fcm_assigns.append(int(np.argmin(dists)))
fcm_assigns = np.array(fcm_assigns)

shap_cluster_means = {}
for k in range(K):
    name = cluster_names.get(str(k), f"cluster_{k}")
    mask = fcm_assigns == k
    mean_shap = sv_arr[mask].mean(axis=0).tolist() if mask.sum() > 0 else [0.0] * len(FEATURE_ORDER)
    # key by cluster NAME string — compute_shap_cosine receives top_cluster as a name string
    shap_cluster_means[name] = mean_shap
    print(f"  cluster {k} ({name}): {mask.sum()} points")

# ── 5. Save ───────────────────────────────────────────────────────────────────
print("Saving artifacts…")
_save_pkl(index,              'rag_index.pkl')
_save_pkl(shap_cluster_means, 'shap_cluster_means.pkl')
_save_json(labels,            'rag_labels.json')

print("\nDone. All three RAG artifacts saved to models/.")

import numpy as np
import pandas as pd
import pickle
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'funding_amount_usd',
    'deal_count_24m',
    'market_cagr_pct',
    'competitor_count',
    'job_posting_trend',
    'market_sentiment_score',
    'inflation_rate',
    'internet_penetration_pct',
]

CLUSTER_NAMES = {
    0: 'HIGH_OPPORTUNITY',
    1: 'SATURATED_MARKET',
    2: 'EMERGING_NICHE',
    3: 'HIGH_RISK',
    4: 'REGULATORY_BLOCKED',
}


def tune_dbscan(X, eps_values=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0]):
    """Find best eps via silhouette score sweep"""
    best_eps = 0.5
    best_score = -1
    best_labels = None

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2:
            continue

        non_noise = labels != -1
        if non_noise.sum() < 10:
            continue

        try:
            score = silhouette_score(X[non_noise], labels[non_noise])
            print(f"  eps={eps} → {n_clusters} clusters, "
                  f"silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = labels
        except Exception:
            continue

    print(f"\n  Best eps={best_eps} (silhouette={best_score:.3f})")
    return best_eps, best_labels


def run_fcm(X, k):
    """Run Fuzzy C-Means with k clusters"""
    print(f"  Running FCM with k={k} clusters...")

    # skfuzzy expects (features x samples)
    X_T = X.T

    cntr, U, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_T,
        c=k,
        m=2.0,
        error=0.005,
        maxiter=1000,
    )

    # U shape: (k x n_samples)
    # Hard labels = argmax over clusters
    y_labels = np.argmax(U, axis=0)

    return U, y_labels, cntr


def label_pipeline(X_raw=None, save_models=True):
    """
    Full DBSCAN → FCM pipeline.
    Returns dict with labels, membership matrix, cluster count.
    """
    print("\n=== STEP 2: DBSCAN → FCM LABELER ===\n")

    # Load data if not provided
    if X_raw is None:
        print("Loading X_raw_scaled.csv...")
        df = pd.read_csv('data/X_raw_scaled.csv')
        X_raw = df[FEATURE_COLS].values

    print(f"Input shape: {X_raw.shape}")

    # PCA for DBSCAN (reduces noise in high dimensions)
    print("\nApplying PCA (8 → 6 components) before DBSCAN...")
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_raw)
    print(f"Explained variance: "
          f"{pca.explained_variance_ratio_.sum():.1%}")

    # DBSCAN to find k
    print("\nTuning DBSCAN eps...")
    best_eps, dbscan_labels = tune_dbscan(X_pca)

    if dbscan_labels is None:
        print("DBSCAN failed to find clusters — using k=4 default")
        k = 4
    else:
        k = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        noise_pct = (dbscan_labels == -1).sum() / len(dbscan_labels)
        print(f"\nDBSCAN found k={k} clusters")
        print(f"Noise points: {noise_pct:.1%} of data")

    # Cap k between 3 and 5
    k = max(3, min(k, 5))
    print(f"Using k={k} for FCM")

    # FCM
    U, y_labels, centroids = run_fcm(X_raw, k)

    # Name the clusters
    named_labels = [CLUSTER_NAMES.get(l, f'CLUSTER_{l}')
                    for l in y_labels]

    # Summary
    unique, counts = np.unique(y_labels, return_counts=True)
    print("\nCluster distribution:")
    for u, c in zip(unique, counts):
        name = CLUSTER_NAMES.get(u, f'CLUSTER_{u}')
        print(f"  {name}: {c} rows ({c/len(y_labels):.1%})")

    # Save models
    if save_models:
        with open('models/fcm_centroids.pkl', 'wb') as f:
            pickle.dump(centroids, f)
        with open('models/pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
        np.save('models/fcm_membership_U.npy', U)
        np.save('models/y_labels.npy', y_labels)
        print("\nModels saved to models/")

    return {
        'k': k,
        'y_labels': y_labels,
        'named_labels': named_labels,
        'U': U,
        'centroids': centroids,
        'pca': pca,
        'dbscan_eps': best_eps,
    }


if __name__ == '__main__':
    result = label_pipeline()
    print(f"\n✓ Clustering complete. "
          f"k={result['k']} clusters found.")
    print("Ready for Step 3: SVM Classifier")
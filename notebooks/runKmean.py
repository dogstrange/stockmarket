import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

from models.myKmean import KMeansFromScratch

# %% Data loading
combined, ticker_map = utils.load_multiple_stocks()

print("\nMulti-stock data successfully loaded")
print(combined.head(5))
print(combined["target"].value_counts())

# %% Prepare data for clustering
df_binary = combined.dropna(subset=["target"])
df_binary = df_binary[
    df_binary["target"] != 0
].copy()  # Remove hold (0) labels for binary classification

feature_cols = [
    "ATR_Ratio",
    "RSI_14",
    "RSI_7",
    "ADX",
    "Dist_SMA20",
    "MFI",
    "HL_Range",
    "Log_Ret_5",
    "Log_Ret_3",
    "Log_Ret",
    "BB_Width",
    "BB_Pos",
    "EMA9_21",
    "EMA21_50",
    "STOCH_K",
    "STOCH_D",
    "ROC_5",
    "Vol_ratio",
]

# ── Per-stock rolling Z-score normalization ──────────────────────────────────
window = 100
for col in feature_cols:
    df_binary[f"{col}_z"] = (
        df_binary.groupby("ticker")[col]
        .transform(
            lambda s: (
                (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-8)
            )
        )
        .fillna(0)
    )

feature_cols_z = [f"{col}_z" for col in feature_cols]
df_binary = df_binary.dropna(subset=feature_cols_z)

X = df_binary[feature_cols_z].to_numpy()
Y = df_binary["target"].to_numpy()  # Triple barrier labels: -1 (sell), 1 (buy)

print(f"\nClustering {len(X)} data points with {len(feature_cols)} features")
sell_count = np.sum(Y == -1)
buy_count = np.sum(Y == 1)
print(f"Target distribution: {sell_count} sell, {buy_count} buy")

# %% K-means clustering
k = 2  # Since we have binary classification (buy/sell)
model = KMeansFromScratch(k=k, max_iters=100)
model.fit(X)

# Get cluster assignments
labels = model.labels

# Calculate clustering metrics
inertia = model.inertia(X)
silhouette = model.silhouette_score(X)


print(f"Number of clusters: {k}")
print(f"Inertia (within-cluster sum of squares): {inertia:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

# %% Evaluate clustering vs true labels
# Convert labels to 0/1 for comparison
Y_binary = ((Y + 1) / 2).astype(int)  # -1,1 → 0,1

ari = adjusted_rand_score(Y_binary, labels)
homogeneity = homogeneity_score(Y_binary, labels)
completeness = completeness_score(Y_binary, labels)


print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Homogeneity Score: {homogeneity:.4f}")
print(f"Completeness Score: {completeness:.4f}")

# %% Analyze cluster composition

for i in range(k):
    cluster_mask = labels == i
    cluster_targets = Y[cluster_mask]
    buy_ratio = np.mean(cluster_targets == 1)
    sell_ratio = np.mean(cluster_targets == -1)
    print(f"Cluster {i+1}: {cluster_mask.sum()} points")
    print(f"  Buy ratio: {buy_ratio:.1f}")
    print(f"  Sell ratio: {sell_ratio:.1f}")

# %% Visualize clusters (2D projection)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")

# Fit K-means on 2D data for visualization
model_2d = KMeansFromScratch(k=k)
model_2d.fit(X_2d)

plt.figure(figsize=(12, 5))

# Plot clusters
plt.subplot(1, 2, 1)
for i in range(k):
    cluster_points = X_2d[model_2d.labels == i]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {i+1}",
        alpha=0.6,
        s=10,
    )

plt.scatter(
    model_2d.centroids[:, 0],
    model_2d.centroids[:, 1],
    marker="x",
    s=200,
    c="red",
    label="Centroids",
)
plt.title("K-Means Clusters (2D PCA)")
plt.legend()

# Plot true labels
plt.subplot(1, 2, 2)
colors = ["red" if y == -1 else "green" for y in Y]
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, s=10)
plt.title("True Labels: Red=Sell, Green=Buy")
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils
from models import mySVM

# %% Data loading
df = utils.load_data()
df = utils.add_ml_features(df)
df = utils.add_ml_features_advanced(df)

print("\nCleaned data successfully loaded")
print(df.head(5))
print(df["target"].value_counts())


# %% Initial visualization
import matplotlib.pyplot as plt

daily_return = df["daily_return"].values
volatility_20 = df["volatility_20"].values
labels = df["target"].values

plt.figure(figsize=(10, 6))
mask_positive = labels == 1
plt.scatter(
    volatility_20[mask_positive],
    daily_return[mask_positive],
    c="green",
    label="Label 1",
    alpha=0.6,
    s=100,
)
mask_negative = labels == -1
plt.scatter(
    volatility_20[mask_negative],
    daily_return[mask_negative],
    c="blue",
    label="Label -1",
    alpha=0.6,
    s=100,
)
plt.xlabel("Volatility_20", fontsize=12)
plt.ylabel("Daily Return", fontsize=12)
plt.title("Cluster Visualization: Volatility vs Daily Return", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% Prepare features & labels
# ── Use ALL available features so PCA actually has something to reduce ────────
feature_cols = [
    # Basic
    "SMA_20",
    "daily_return",
    "volatility_20",
    "OBV",
    "Volume_SMA_20",
    "Volume_Spike",
    "Stoch_K",
    "Stoch_D",
    "ROC_10",
    "log_return",
    # Advanced
    "ATR_Ratio",
    "BB_Width",
    "RSI",
    "ADX",
    "Dist_SMA_20",
    "MFI",
    "Range_Ratio",
    "Log_Ret_5",
]

# Only keep rows where target != 0 (binary classification)
df_binary = df[df["target"] != 0].copy()

X_raw = df_binary[feature_cols].to_numpy()
Y = df_binary["target"].to_numpy()  # values: -1 or +1

print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
print(f"Dataset size: {X_raw.shape}")

# ── Chronological split ───────────────────────────────────────────────────────
split = int(len(X_raw) * 0.8)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")


# %% Apply PCA
# Now with 18 features, PCA will actually reduce dimensions meaningfully
X_train_pca, X_test_pca, pca = utils.apply_pca(
    X_train_raw,
    X_test_raw,
    variance_threshold=0.95,
    plot=True,
)

print(f"\nOriginal features : {X_train_raw.shape[1]}")
print(f"After PCA         : {X_train_pca.shape[1]}")


# %% SVM Linear — trained on PCA features
svm_linear = mySVM.SVMSoftmargin(alpha=0.001, iteration=10000, lambda_=0.01)
w_l, b_l = svm_linear.fit(X_train_pca, Y_train)
predictions_train = svm_linear.predict(X_train_pca)
predictions_test = svm_linear.predict(X_test_pca)

print(f"\nLinear SVM")
print(f"Weights : {w_l}")
print(f"Bias    : {b_l}")
print(f"Train Accuracy : {np.mean(predictions_train == Y_train):.2%}")
print(f"Test  Accuracy : {np.mean(predictions_test  == Y_test):.2%}")


# %% SVM Dual — trained on PCA features
svm_dual = mySVM.SVM_Dual(
    kernel="rbf",
    degree=2,
    sigma=1.0,
    epoches=1000,
    learning_rate=0.001,
)
svm_dual.train(X_train_pca, Y_train)
predictions_dual_test = svm_dual.predict(X_test_pca)
accuracy_dual = svm_dual.score(X_test_pca, Y_test)
print(f"\nDual SVM (RBF)")
print(f"Test Accuracy: {accuracy_dual:.2%}")


# %% Find support vectors (Linear SVM)
margins = Y_train * (np.dot(X_train_pca, w_l) - b_l)
tolerance = 1e-3
support_vector_indices = np.where(
    (margins <= 1 + tolerance) & (margins > 0 + tolerance)
)[0]
print(f"\nNumber of support vectors : {len(support_vector_indices)}")
print(f"Support vectors (PCA space):\n{X_train_pca[support_vector_indices]}")

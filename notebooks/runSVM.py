import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils
from models import mySVM

# %% Data loading
df, ticker_map = utils.load_multiple_stocks()

print("\nCleaned data successfully loaded")
print(df.head(5))
print(df["target"].value_counts())


# %% Initial visualization
import matplotlib.pyplot as plt

# Use available features for visualization
log_ret = df["Log_Ret"].values
atr_ratio = df["ATR_Ratio"].values
labels = df["target"].values

plt.figure(figsize=(10, 6))
mask_positive = labels == 1
plt.scatter(
    atr_ratio[mask_positive],
    log_ret[mask_positive],
    c="green",
    label="Buy (1)",
    alpha=0.6,
    s=100,
)
mask_negative = labels == -1
plt.scatter(
    atr_ratio[mask_negative],
    log_ret[mask_negative],
    c="red",
    label="Sell (-1)",
    alpha=0.6,
    s=100,
)
plt.xlabel("ATR_Ratio", fontsize=12)
plt.ylabel("Log Return", fontsize=12)
plt.title("Initial Visualization: ATR Ratio vs Log Return", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% Prepare features & labels
# ── Use features consistent with multi-stock data ────────
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

# Only keep rows where target != 0 (binary classification)
df_binary = df[df["target"] != 0].copy()

# ── Per-stock rolling Z-score normalization (like other run files) ──────────────────────────────────
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

X_raw = df_binary[feature_cols_z].to_numpy()
Y = df_binary["target"].to_numpy()  # values: -1 or +1

print(f"\nUsing {len(feature_cols_z)} Z-normalized features")
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
svm_linear = mySVM.SVMSoftmargin(alpha=0.001, iteration=1000, lambda_=0.01)
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

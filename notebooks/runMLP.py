import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils
from models.mymlp import MLPFromScratched
from models.mrRNN import RNNFromScratch

# %% Data loading
df = utils.load_data()
df = utils.add_ml_features(df)
df = utils.add_ml_features_advanced(df)

print("\nCleaned data succesfully loaded")
print(df.head(5))
print(df["target"].value_counts())


# %% Model training
from sklearn.preprocessing import StandardScaler

df_binary = df[df["target"] != 0].copy()
feature_cols = [
    "ATR_Ratio",
    "RSI",
    "ADX",
    "Dist_SMA_20",
    "MFI",
    "Range_Ratio",
    "Log_Ret_5",
    "volatility_20",
    "SMA_20",
]

window = 100
for col in feature_cols:
    rolling_mean = df_binary[col].rolling(window=window).mean()
    rolling_std = df_binary[col].rolling(window=window).std()

    # Calculate Z-score and fill the initial NaN values (due to rolling window) with 0
    df_binary[f"{col}_z"] = (
        (df_binary[col] - rolling_mean) / (rolling_std + 1e-8)
    ).fillna(0)

# ── 3. Update your feature columns to use the new Z-scored versions ──
feature_cols_z = [f"{col}_z" for col in feature_cols]
X = df_binary[feature_cols_z].to_numpy()
Y = df_binary["target"].to_numpy()
Y = (Y + 1) / 2


# ── Chronological split (no shuffling)
split = int(len(X) * 0.8)

X_train_raw, X_test_raw = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# ── Scale AFTER splitting (fit only on train)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train_raw)  # fit + transform
# X_test = scaler.transform(X_test_raw)  # transform only
X_train = X_train_raw
X_test = X_test_raw

# ── Reshape targets
Y_train = Y_train.reshape(1, -1)
Y_test = Y_test.reshape(1, -1)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── Compute class weights
n_buy = int(Y_train.sum())
n_sell = Y_train.shape[1] - n_buy
n_total = n_buy + n_sell

class_weight = {
    1: n_total / (2 * n_buy),
    0: n_total / (2 * n_sell),
}
print(f"Class weights → Sell: {class_weight[0]:.2f} | Buy: {class_weight[1]:.2f}")

# ── Train
model = MLPFromScratched(lr=0.01, epochs=20000, h=[32, 32, 32])
model.fit(X_train, Y_train, class_weight)


# ── Evaluate on both
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict_proba(X_test)

model.evaluate(Y_test, Y_pred_test)

# %% Plotting graph
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Get predictions (these come out as shape (1, n_samples))
Y_pred_test_raw = model.predict(X_test)

# 2. Flatten both to 1D arrays for sklearn metrics

predictions = Y_pred_test_raw.ravel()
actual = Y_test.ravel().astype(int)

# 3. Calculate Confusion Matrix
cm = confusion_matrix(actual, predictions)
accuracy = np.mean(predictions == actual)

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Loss curve
axes[0].plot(model.loss_history, color="royalblue", lw=2)
axes[0].set_title("Training Loss Over Time", fontsize=14)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# Plot 2: Confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Sell", "Buy"],
    yticklabels=["Sell", "Buy"],
    ax=axes[1],
    cbar=False,
)
axes[1].set_title("Confusion Matrix (Test Set)", fontsize=14)
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# Plot 3: Accuracy vs Error
axes[2].bar(
    ["Accuracy", "Error"], [accuracy, 1 - accuracy], color=["#2ecc71", "#e74c3c"]
)
axes[2].set_title(f"Overall Accuracy: {accuracy:.2%}", fontsize=14)
axes[2].set_ylabel("Rate")
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils
from models.myLogisticRegression import LogisticRegressionGD

# %% Data loading
combined, ticker_map = utils.load_multiple_stocks()

print("\nCleaned data successfully loaded")
print(combined.head(5))
print(combined["target"].value_counts())


# %% Model training
df_binary = combined.dropna(subset=["target"])
df_binary = df_binary[df_binary["target"] != 0].copy()
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
    df_binary[f"{col}_z"] = df_binary.groupby("ticker")[col].transform(
        lambda s: ((s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-8))
    )

feature_cols_z = [f"{col}_z" for col in feature_cols]
df_binary = df_binary.dropna(subset=feature_cols_z)

X = df_binary[feature_cols_z].to_numpy()
Y = df_binary["target"].to_numpy()
Y = (Y + 1) / 2  # convert {-1, 1} → {0, 1}


# ── Chronological split (no shuffling) ───────────────────────────────────────
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── Compute class weights ────────────────────────────────────────────────────
# NOTE: LogisticRegressionGD handles normalization internally,
# so we pass raw (un-reshaped) 1D arrays for Y here.
n_buy = int(Y_train.sum())
n_sell = len(Y_train) - n_buy
n_total = n_buy + n_sell

class_weight = {
    1: n_total / (2 * n_buy),
    0: n_total / (2 * n_sell),
}
print(f"Class weights → Sell: {class_weight[0]:.2f} | Buy: {class_weight[1]:.2f}")

# ── Train ─────────────────────────────────────────────────────────────────────
# LogisticRegressionGD normalizes X internally (stores mean/std for inference).
# Y should be 1D flat array of {0, 1} labels.
model = LogisticRegressionGD(learning_rate=0.01, epochs=12000)
model.fit(X_train, Y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n── Train Set ──")
model.evaluate(X_train, Y_train, feature_cols_z)

print("\n── Test Set ──")
model.evaluate(X_test, Y_test, feature_cols_z)

# %% Plotting
Y_pred_test = model.predict_proba(X_test).flatten()
utils.visualize_classification(model, Y_test, X_test, title="Logistic Regression")

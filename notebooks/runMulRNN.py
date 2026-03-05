import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils
from models.myRNN import RNNFromScratch

# %% Data loading
combined, ticker_map = utils.load_multiple_stocks()

print("\nCleaned data successfully loaded")
print(combined.head(5))
print(combined["target"].value_counts())


# %% Model training
df_binary = combined[combined["target"] != 0].copy()

# ── Use correct column names from add_features() ────────────────────────────
feature_cols = [
    "ATR_Ratio",
    "RSI_14",
    "ADX",
    "Dist_SMA20",
    "MFI",
    "HL_Range",
    "Log_Ret_5",
    "BB_Width",
    "EMA9_21",
]

# ── Rolling Z-score normalization ────────────────────────────────────────────
window = 100
for col in feature_cols:
    rolling_mean = df_binary[col].rolling(window=window).mean()
    rolling_std = df_binary[col].rolling(window=window).std()
    df_binary[f"{col}_z"] = (
        (df_binary[col] - rolling_mean) / (rolling_std + 1e-8)
    ).fillna(0)

feature_cols_z = [f"{col}_z" for col in feature_cols]
X = df_binary[feature_cols_z].to_numpy()
Y = df_binary["target"].to_numpy()
Y = (Y + 1) / 2  # convert {-1, 1} → {0, 1}

# ── Chronological split (no shuffling) ───────────────────────────────────────
split = int(len(X) * 0.8)
X_train_raw, X_test_raw = X[:split], X[split:]
Y_train_raw, Y_test_raw = Y[:split], Y[split:]

print(f"Raw train size: {X_train_raw.shape[0]} | Raw test size: {X_test_raw.shape[0]}")


# ── Sequence creation ─────────────────────────────────────────────────────────
def create_sequences(X, Y, seq_len=20):
    """
    Turns flat 2D data (n_samples, n_features)
    into sequences  (n_windows, seq_len, n_features)

    e.g. seq_len=20 → each sample = last 20 days of features,
    label = the signal on the LAST day of that window.
    """
    Xs, Ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])  # window of seq_len rows
        Ys.append(Y[i])  # label for the last day
    return np.array(Xs), np.array(Ys).reshape(1, -1)


SEQ_LEN = 20  # how many days back the RNN looks

X_train, Y_train = create_sequences(X_train_raw, Y_train_raw, SEQ_LEN)
X_test, Y_test = create_sequences(X_test_raw, Y_test_raw, SEQ_LEN)

print(f"X_train shape: {X_train.shape}")  # (n_windows, 20, 9)
print(f"Y_train shape: {Y_train.shape}")  # (1, n_windows)
print(f"X_test  shape: {X_test.shape}")
print(f"Y_test  shape: {Y_test.shape}")

# ── Compute class weights ─────────────────────────────────────────────────────
n_buy = int(Y_train.sum())
n_sell = Y_train.shape[1] - n_buy
n_total = n_buy + n_sell

class_weight = {
    1: n_total / (2 * n_buy),
    0: n_total / (2 * n_sell),
}
print(f"Class weights → Sell: {class_weight[0]:.2f} | Buy: {class_weight[1]:.2f}")

# ── Train ─────────────────────────────────────────────────────────────────────
model = RNNFromScratch(lr=0.01, epochs=3000, hidden_size=32)
model.fit(X_train, Y_train, class_weight)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n── Train Set ──")
Y_pred_train_prob = model.predict_proba(X_train)
model.evaluate(Y_train, Y_pred_train_prob.squeeze())

print("\n── Test Set ──")
Y_pred_test_prob = model.predict_proba(X_test)
model.evaluate(Y_test, Y_pred_test_prob.squeeze())

# %% Plotting
utils.visualize_classification(model, Y_test, X_test, title="RNN")

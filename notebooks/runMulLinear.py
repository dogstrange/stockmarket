import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

from models.myMultipleLinear import StockLinearRegression

# %% Data loading
df, ticker_map = utils.load_multiple_stocks()
df["target"] = df["Log_Ret"].shift(-1)  # Predict next day's log return
df = df.dropna()

print("\nCleaned data successfully loaded")
print(df.head(5))
print(
    f"Target distribution: mean={df['target'].mean():.6f}, std={df['target'].std():.6f}"
)

# %% Model training
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

X = df[feature_cols].to_numpy()
Y = df["target"].to_numpy().reshape(-1, 1)

# ── Chronological split (no shuffling)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── Train
model = StockLinearRegression(learning_rate=0.01, max_epochs=8000, patience=200)
best_loss = model.fit(X_train, Y_train)

print(f"Best validation loss: {best_loss:.6f}")

# ── Evaluate on test set
n_days = 200  # Number of days to plot
model.evaluate_and_plot(X_test, Y_test, n_days)

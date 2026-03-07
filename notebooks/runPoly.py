import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

from models.myPolynomial import StockPolynomialRegression

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

# ── Chronological split (train/val/test: 60/20/20)
n_total = len(X)
train_end = int(0.6 * n_total)
val_end = int(0.8 * n_total)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
Y_train, Y_val, Y_test = Y[:train_end], Y[train_end:val_end], Y[val_end:]

print(
    f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]} | Test size: {X_test.shape[0]}"
)

# ── Train
model = StockPolynomialRegression(
    degree=2, learning_rate=0.01, max_epochs=6000, patience=300, l2_lambda=0.001
)
best_loss = model.fit(X_train, Y_train, X_val, Y_val)

print(f"Best validation loss: {best_loss:.6f}")

# ── Evaluate on test set
Y_pred_test = model.predict(X_test)

mae = np.mean(np.abs(Y_test - Y_pred_test))
mse = np.mean((Y_test - Y_pred_test) ** 2)
rmse = np.sqrt(mse)
rse = np.sum((Y_test - Y_pred_test) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2)
r2 = 1 - rse
mape = np.mean(np.abs((Y_test - Y_pred_test) / (Y_test + 1e-9))) * 100
direction_acc = np.mean(np.sign(Y_test) == np.sign(Y_pred_test)) * 100
baseline_mae = np.mean(np.abs(Y_test - 0))

print("\n========== TEST SET SCORES ==========")
print(f"MAE      : {mae:.6f}")
print(f"RMSE     : {rmse:.6f}")
print(f"R²       : {r2:.6f}")
print(f"MAPE %   : {mape:.2f}")
print(f"DIR ACC% : {direction_acc:.2f}")
print(f"BASE MAE : {baseline_mae:.6f}")

# Plot
n_days = 200  # Number of days to plot
plt.figure(figsize=(13, 5))
plt.scatter(
    range(n_days), Y_test[:n_days].flatten(), color="royalblue", alpha=0.4, label="Real"
)
plt.plot(
    model.predict(X_test[:n_days]).flatten(),
    color="darkorange",
    linewidth=2,
    label="Prediction",
)
plt.axhline(0, color="black", linestyle="--")
plt.title("Polynomial Regression: Next Day Return Prediction")
plt.legend()
plt.show()

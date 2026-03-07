import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

from models.mySimpleLinear import SimpleLinearRegression

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
# For simple linear regression, use a single feature (e.g., current day's log return)
X = df["Log_Ret"].to_numpy()  # Single feature: current day's return
Y = df["target"].to_numpy()  # Target: next day's return

# ── Chronological split (train/test: 80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── Train
model = SimpleLinearRegression(learning_rate=0.01, epochs=5000)
model.fit(X_train, Y_train)

# ── Evaluate on test set
model.evaluate(X_test, Y_test)

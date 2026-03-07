import numpy as np
import pandas as pd
import os
import sys

sys.path.append(
    "/Users/chengstrange/Documents/Machine Learning Reporitories/StockMarket"
)
from utils import utils
from models.myTree import DecisionTreeFromScratch

# %% Data loading
combined, ticker_map = utils.load_multiple_stocks()

print("\nCleaned data successfully loaded")
print(combined.head(5))
print(combined["target"].value_counts())

# %% Feature engineering
df = combined.dropna(subset=["target"])

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
    df[f"{col}_z"] = df.groupby("ticker")[col].transform(
        lambda s: ((s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-8))
    )

feature_cols_z = [f"{col}_z" for col in feature_cols]
df = df.dropna(subset=feature_cols_z)

# ── Drop Hold (0) rows — match RNN's binary Buy vs Sell setup ────────────────
df = df[df["target"] != 0].copy()
df["target"] = df["target"].map({-1: 0, 1: 1})  # Sell=0, Buy=1

print(f"\nAfter dropping Hold rows:")
print(df["target"].value_counts())

X = df[feature_cols_z].to_numpy()
Y = df["target"].to_numpy().astype(int)

# ── Chronological split (no shuffling) ───────────────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

print(f"\nTrain set : {X_train.shape[0]} samples")
print(f"Test set  : {X_test.shape[0]} samples")
print(f"Classes   : {np.unique(Y_train, return_counts=True)}")

# %% Train the Decision Tree
tree = DecisionTreeFromScratch(max_depth=10, min_samples_split=50, min_samples_leaf=25)
tree.fit(X_train, Y_train)

# %% Evaluate — same interface as RNN
print("\n--- Training Set ---")
proba_train = tree.predict_proba(X_train)
tree.evaluate(Y_train, proba_train)

print("\n--- Test Set ---")
proba_test = tree.predict_proba(X_test)
tree.evaluate(Y_test, proba_test)

# %% Visualize — same call as RNN
utils.visualize_classification(
    tree,
    Y_test,
    X_test,
    title="Decision Tree",
    feature_names=feature_cols_z,
)

# %% Feature Importances
print("\nTop Feature Importances:")
importance_pairs = sorted(
    zip(feature_cols_z, tree.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)
for name, imp in importance_pairs:
    bar = "█" * int(imp * 200)
    print(f"  {name:<25}: {imp:.4f}  {bar}")

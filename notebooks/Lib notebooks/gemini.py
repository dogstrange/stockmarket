import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

# ─────────────────────────────────────────────
# 1. DEVICE + SEED
# ─────────────────────────────────────────────
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
df = utils.load_multiple_stocks()
if df is None:
    sys.exit(1)

df = df[df.index >= "2000-01-01"].sort_index()
print(f"After 2000 filter: {len(df):,} rows | {df['ticker'].nunique()} tickers\n")

# ─────────────────────────────────────────────
# 3. VOLATILITY REGIME TARGET
#
#    We predict whether FUTURE realised volatility
#    will be HIGH or LOW relative to the stock's
#    own recent history.
#
#    Specifically:
#      - Compute 10-day forward realised vol
#        (std of next 10 daily log returns)
#      - Compare to the stock's rolling 252-day
#        median vol (its "normal" level)
#      - Label 1 = HIGH vol regime (above median)
#      - Label 0 = LOW vol regime (below median)
#
#    Why this works better than direction:
#      Volatility CLUSTERS — high vol today strongly
#      predicts high vol tomorrow (GARCH effect).
#      This is a well-documented, exploitable pattern.
# ─────────────────────────────────────────────
FORWARD_WINDOW = 10  # predict vol over next 10 days
VOL_LOOKBACK = 252  # rolling window for "normal" vol baseline


def add_vol_target(grp):
    grp = grp.copy().sort_index()
    log_ret = np.log(grp["close"] / grp["close"].shift(1))

    # Forward realised vol: std of next FORWARD_WINDOW returns
    fwd_vol = log_ret[::-1].rolling(FORWARD_WINDOW).std()[::-1]

    # Rolling median vol over past year — this stock's "normal" level
    rolling_med = fwd_vol.rolling(VOL_LOOKBACK, min_periods=60).median()

    # 1 = high vol regime, 0 = low vol regime
    grp["fwd_vol"] = fwd_vol
    grp["vol_median"] = rolling_med
    grp["target"] = (fwd_vol > rolling_med).astype(int)
    return grp


print("Computing volatility regime targets per stock...")
df = df.groupby("ticker", group_keys=False).apply(add_vol_target)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "fwd_vol"])
print(f"Dataset after target creation: {len(df):,} rows")
print(
    f"Class balance: {int(df['target'].sum()):,} HIGH / {int((df['target']==0).sum()):,} LOW\n"
)


# ─────────────────────────────────────────────
# 4. FEATURES — volatility-focused
#    We keep all the standard features PLUS add
#    vol-specific ones that are known predictors:
#      - Current ATR levels
#      - Volatility of volatility (vol-of-vol)
#      - Bollinger Band width expansion
#      - VIX-proxy: recent realised vol trajectory
# ─────────────────────────────────────────────
def add_vol_features(grp):
    grp = grp.copy().sort_index()
    c = grp["close"]
    log_ret = np.log(c / c.shift(1))

    # Current realised vol at multiple horizons
    grp["RV_5"] = log_ret.rolling(5).std()
    grp["RV_10"] = log_ret.rolling(10).std()
    grp["RV_20"] = log_ret.rolling(20).std()
    grp["RV_60"] = log_ret.rolling(60).std()

    # Vol ratios — is vol expanding or contracting?
    grp["RV_5_20_ratio"] = grp["RV_5"] / (grp["RV_20"] + 1e-9)  # short/medium
    grp["RV_10_60_ratio"] = grp["RV_10"] / (grp["RV_60"] + 1e-9)  # medium/long
    grp["RV_trend"] = grp["RV_5"] / (grp["RV_5"].shift(5) + 1e-9) - 1  # vol momentum

    # Volatility of volatility (how unstable is vol itself?)
    grp["VoV_10"] = grp["RV_5"].rolling(10).std()

    # BB width — expanding bands = increasing vol
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    grp["BB_Width"] = (4 * std20) / (sma20 + 1e-9)
    grp["BB_Width_chg"] = grp["BB_Width"] / (grp["BB_Width"].shift(5) + 1e-9) - 1

    # ATR ratio (normalised across stocks)
    h, l = grp["high"], grp["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(
        axis=1
    )
    atr14 = tr.rolling(14).mean()
    grp["ATR_ratio"] = atr14 / (c + 1e-9)
    grp["ATR_chg"] = atr14 / (atr14.shift(10) + 1e-9) - 1  # ATR trend

    # Gap & range (intraday volatility signals)
    grp["HL_Range"] = (h - l) / (c + 1e-9)
    grp["Gap"] = (grp["open"] - c.shift(1)).abs() / (c.shift(1) + 1e-9)

    # RSI extreme readings often precede vol spikes
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    grp["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    grp["RSI_dist_50"] = (grp["RSI"] - 50).abs() / 50  # distance from neutral

    # Volume spike (often precedes vol expansion)
    if "volume" in grp.columns:
        vol_sma = grp["volume"].rolling(20).mean()
        grp["Vol_ratio"] = grp["volume"] / (vol_sma + 1e-9)

    # MA compression — tight MAs = coiled spring, often precedes breakout
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    grp["MA_compression"] = (ema9 - ema50).abs() / (c + 1e-9)
    grp["EMA9_21"] = ema9 / (ema21 + 1e-9) - 1

    # Calendar (vol has known seasonal patterns — VIX rises in Oct, low in Dec)
    if not isinstance(grp.index, pd.DatetimeIndex):
        grp.index = pd.to_datetime(grp.index)
    grp["Month"] = grp.index.month / 12.0
    grp["DayOfWeek"] = grp.index.dayofweek / 4.0

    return grp


print("Adding volatility-specific features...")
df = df.groupby("ticker", group_keys=False).apply(add_vol_features)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"After feature engineering: {len(df):,} rows\n")

# ─────────────────────────────────────────────
# 5. FEATURE SELECTION — mutual information
# ─────────────────────────────────────────────
exclude = {
    "ticker",
    "ticker_id",
    "target",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "fwd_vol",
    "vol_median",
}
feature_cols = [c for c in df.columns if c not in exclude]

# MI on train portion of each stock
train_rows = []
for tick, grp in df.groupby("ticker"):
    n = len(grp)
    cut = int(n * 0.70)
    train_rows.extend(grp.index[:cut])
train_mask = df.index.isin(train_rows)

mi_scores = mutual_info_classif(
    df.loc[train_mask, feature_cols].values,
    df.loc[train_mask, "target"].values,
    random_state=42,
)
mi_df = pd.DataFrame({"feature": feature_cols, "mi": mi_scores}).sort_values(
    "mi", ascending=False
)

print("Feature importance (Mutual Information):")
print(mi_df.to_string(index=False))

# Keep top 15 features
feature_cols = mi_df.head(15)["feature"].tolist()
print(f"\nSelected top 15: {feature_cols}\n")

# ─────────────────────────────────────────────
# 6. PER-STOCK SPLITS + SEQUENCES
# ─────────────────────────────────────────────
SEQ_LEN = 20


def make_seqs(X, y, seq_len):
    xs = np.array(
        [X[i : i + seq_len] for i in range(len(X) - seq_len)], dtype=np.float32
    )
    ys = np.array(y[seq_len:], dtype=np.float32)
    return xs, ys


X_tr_all, y_tr_all = [], []
X_va_all, y_va_all = [], []
X_te_all, y_te_all = [], []

for ticker, grp in df.groupby("ticker"):
    grp = grp.sort_index()
    n = len(grp)
    if n < SEQ_LEN + 100:
        continue

    X_raw = grp[feature_cols].values.astype(np.float32)
    y_raw = grp["target"].values.astype(np.float32)

    tr_end = int(n * 0.70)
    va_end = int(n * 0.85)

    scaler = RobustScaler()
    X_tr_s = np.clip(scaler.fit_transform(X_raw[:tr_end]), -5, 5)
    X_va_s = np.clip(scaler.transform(X_raw[tr_end:va_end]), -5, 5)
    X_te_s = np.clip(scaler.transform(X_raw[va_end:]), -5, 5)

    for X_s, y_s, lst_x, lst_y in [
        (X_tr_s, y_raw[:tr_end], X_tr_all, y_tr_all),
        (X_va_s, y_raw[tr_end:va_end], X_va_all, y_va_all),
        (X_te_s, y_raw[va_end:], X_te_all, y_te_all),
    ]:
        xs, ys = make_seqs(X_s, y_s, SEQ_LEN)
        lst_x.append(xs)
        lst_y.append(ys)

X_tr = np.concatenate(X_tr_all)
y_tr = np.concatenate(y_tr_all)
X_va = np.concatenate(X_va_all)
y_va = np.concatenate(y_va_all)
X_te = np.concatenate(X_te_all)
y_te = np.concatenate(y_te_all)

# Shuffle train
perm = np.random.permutation(len(X_tr))
X_tr, y_tr = X_tr[perm], y_tr[perm]

print(f"Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")
print(f"Train balance: {int(y_tr.sum()):,} HIGH / {int((y_tr==0).sum()):,} LOW\n")

BATCH_SIZE = 256


def to_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = to_loader(X_tr, y_tr, shuffle=True)
val_loader = to_loader(X_va, y_va)
X_te_t = torch.tensor(X_te).to(device)
y_te_t = torch.tensor(y_te).unsqueeze(1).to(device)


# ─────────────────────────────────────────────
# 7. MODEL
# ─────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.W = nn.Linear(h, h)
        self.v = nn.Linear(h, 1, bias=False)

    def forward(self, x):
        w = torch.softmax(self.v(torch.tanh(self.W(x))), dim=1)
        return (w * x).sum(dim=1)


class VolRegimeLSTM(nn.Module):
    def __init__(self, input_dim, hidden=64, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.attn = TemporalAttention(hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        ctx = self.attn(out)
        ctx = self.bn(ctx)
        ctx = self.drop(ctx)
        return self.head(ctx)


# ─────────────────────────────────────────────
# 8. TRAINING
# ─────────────────────────────────────────────
model = VolRegimeLSTM(input_dim=len(feature_cols)).to(device)
print(
    f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
)

pos_w = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32).to(
    device
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)

EPOCHS = 80
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.15,
    anneal_strategy="cos",
)

PATIENCE = 15
best_val = float("inf")
smoothed = None
pat = 0
best_state = None
train_losses, val_losses = [], []

print("Training...\n")
for epoch in range(EPOCHS):
    model.train()
    ep_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        ep_loss += loss.item()
    avg_tr = ep_loss / len(train_loader)
    train_losses.append(avg_tr)

    model.eval()
    ep_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            ep_val += criterion(model(xb), yb).item()
    avg_va = ep_val / len(val_loader)
    val_losses.append(avg_va)

    smoothed = avg_va if smoothed is None else 0.3 * avg_va + 0.7 * smoothed
    if smoothed < best_val:
        best_val, pat = smoothed, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        pat += 1

    if (epoch + 1) % 5 == 0:
        print(
            f"Ep {epoch+1:>3} | Train: {avg_tr:.4f} | Val: {avg_va:.4f} | Smooth: {smoothed:.4f} | Pat: {pat}/{PATIENCE}"
        )

    if pat >= PATIENCE:
        print(f"\nEarly stop @ epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print("Restored best weights.\n")

# ─────────────────────────────────────────────
# 9. THRESHOLD SWEEP ON VAL
# ─────────────────────────────────────────────
model.eval()
vp, vt = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        vp.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten())
        vt.extend(yb.numpy().flatten())
vp, vt = np.array(vp), np.array(vt)

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.30, 0.71, 0.01):
    f1 = f1_score(vt, (vp > t).astype(int), average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"Optimal threshold: {best_t:.2f}  (val macro-F1: {best_f1:.4f})")

# ─────────────────────────────────────────────
# 10. TEST EVALUATION
# ─────────────────────────────────────────────
with torch.no_grad():
    y_prob = torch.sigmoid(model(X_te_t)).cpu().numpy().flatten()
    y_pred = (y_prob > best_t).astype(int)
    y_true = y_te_t.cpu().numpy().flatten()

auc = roc_auc_score(y_true, y_prob)
print(f"\nTest ROC-AUC : {auc:.4f}")
print(classification_report(y_true, y_pred, target_names=["LOW vol", "HIGH vol"]))

# ─────────────────────────────────────────────
# 11. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Volatility Regime LSTM — HIGH vs LOW Vol Prediction",
    fontsize=13,
    fontweight="bold",
)

axes[0, 0].plot(train_losses, label="Train", color="steelblue")
axes[0, 0].plot(val_losses, label="Val", color="tomato")
axes[0, 0].set_title("Loss Curves")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    ax=axes[0, 1],
    xticklabels=["LOW", "HIGH"],
    yticklabels=["LOW", "HIGH"],
)
axes[0, 1].set_title(f"Confusion Matrix (t={best_t:.2f})")
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_true, y_prob)
axes[1, 0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC={auc:.4f}")
axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1, 0].set_title("ROC Curve")
axes[1, 0].set_xlabel("FPR")
axes[1, 0].set_ylabel("TPR")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(
    y_prob[y_true == 0],
    bins=40,
    alpha=0.6,
    label="LOW vol",
    color="steelblue",
    density=True,
)
axes[1, 1].hist(
    y_prob[y_true == 1],
    bins=40,
    alpha=0.6,
    label="HIGH vol",
    color="darkorange",
    density=True,
)
axes[1, 1].axvline(best_t, linestyle="--", color="black", label=f"Thresh={best_t:.2f}")
axes[1, 1].set_title("Predicted Probability Distribution")
axes[1, 1].set_xlabel("P(HIGH vol)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vol_regime_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: vol_regime_results.png")

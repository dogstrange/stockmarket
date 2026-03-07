import numpy as np
import pandas as pd
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ════════════════════════════════════════════════════════════════════
#  1. DATA LOADING  (same pipeline as your original code)
# ════════════════════════════════════════════════════════════════════
df = utils.load_data()
df = utils.add_ml_features_advanced(df)

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

X = df_binary[feature_cols].to_numpy()
Y = ((df_binary["target"].to_numpy() + 1) / 2).astype(int)  # -1/1 → 0/1

split = int(len(X) * 0.8)
X_tr_r = X[:split]
X_te_r = X[split:]
Y_train = Y[:split]
Y_test = Y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_tr_r)
X_test = scaler.transform(X_te_r)

n_buy = Y_train.sum()
n_sell = len(Y_train) - n_buy
print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print(f"Class split (train) — Buy: {n_buy}  Sell: {n_sell}\n")

# ════════════════════════════════════════════════════════════════════
#  2. MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════
models = {
    "Logistic Reg.": LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=6, min_samples_leaf=20, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
    ),
    "Gradient Boost": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", class_weight="balanced", probability=True, C=1.0, gamma="scale"
    ),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "MLP (sklearn)": MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        max_iter=500,
        learning_rate_init=0.001,
        early_stopping=True,
        random_state=42,
    ),
}

# ════════════════════════════════════════════════════════════════════
#  3. TRAIN & COLLECT METRICS
# ════════════════════════════════════════════════════════════════════
results = {}

for name, model in models.items():
    print(f"  Training {name:<20}", end=" ", flush=True)
    t0 = time.time()
    model.fit(X_train, Y_train)
    elapsed = time.time() - t0

    Y_pred = model.predict(X_test)
    Y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_prob)

    results[name] = {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, zero_division=0),
        "recall": recall_score(Y_test, Y_pred, zero_division=0),
        "f1": f1_score(Y_test, Y_pred, zero_division=0),
        "auc": roc_auc_score(Y_test, Y_prob),
        "time": elapsed,
        "cm": confusion_matrix(Y_test, Y_pred),
        "fpr": fpr,
        "tpr": tpr,
    }
    r = results[name]
    print(f"F1={r['f1']:.3f}  AUC={r['auc']:.3f}  ({elapsed:.2f}s)")

names = list(results.keys())

# ════════════════════════════════════════════════════════════════════
#  4. PRINT SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print(
    f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Time':>7}"
)
print("═" * 72)
for n in sorted(names, key=lambda x: results[x]["f1"], reverse=True):
    r = results[n]
    print(
        f"{n:<22} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
        f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f} {r['time']:>6.2f}s"
    )
print("═" * 72)

# ════════════════════════════════════════════════════════════════════
#  5. VISUALIZATION
# ════════════════════════════════════════════════════════════════════
DARK = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
TEXT = "#e6edf3"
MUTED = "#8b949e"

PALETTE = [
    "#58a6ff",
    "#3fb950",
    "#f0883e",
    "#d2a8ff",
    "#ffa657",
    "#ff7b72",
    "#79c0ff",
    "#56d364",
]

plt.rcParams.update(
    {
        "figure.facecolor": DARK,
        "axes.facecolor": CARD,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": BORDER,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "legend.facecolor": CARD,
        "legend.edgecolor": BORDER,
        "font.family": "monospace",
        "font.size": 9,
    }
)

fig = plt.figure(figsize=(22, 18), facecolor=DARK)
fig.suptitle(
    "Model Comparison Dashboard  ·  AAPL Buy/Sell Classifier",
    fontsize=15,
    fontweight="bold",
    color=TEXT,
    y=0.99,
)

gs = gridspec.GridSpec(
    3,
    3,
    figure=fig,
    hspace=0.55,
    wspace=0.38,
    top=0.95,
    bottom=0.05,
    left=0.06,
    right=0.97,
)


def style(ax, title, xlabel="", ylabel="", xgrid=False, ygrid=True):
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if ygrid:
        ax.grid(True, axis="y", color=BORDER, linestyle="--", alpha=0.6)
    if xgrid:
        ax.grid(True, axis="x", color=BORDER, linestyle="--", alpha=0.6)


# ── Plot 1: Grouped bar — all metrics ───────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
metric_colors = [ACCENT, GREEN, YELLOW, "#f0883e", PURPLE]
x = np.arange(len(names))
w = 0.14

for i, (mk, mlbl, mc) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
    vals = [results[n][mk] for n in names]
    ax1.bar(x + i * w, vals, w, label=mlbl, color=mc, alpha=0.85)

ax1.set_xticks(x + w * 2)
ax1.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
ax1.set_ylim(0, 1.12)
ax1.axhline(0.5, color=RED, linestyle="--", lw=1, alpha=0.5, label="0.5 baseline")
ax1.legend(fontsize=8, labelcolor=TEXT)
style(ax1, "All Metrics by Model", ylabel="Score")


# ── Plot 2: F1 horizontal bar ranked ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ranked = sorted(names, key=lambda n: results[n]["f1"])
f1s = [results[n]["f1"] for n in ranked]
cols = [PALETTE[i % len(PALETTE)] for i in range(len(ranked))]

bars = ax2.barh(ranked, f1s, color=cols, alpha=0.85, height=0.6)
for bar, val in zip(bars, f1s):
    ax2.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=7.5,
        color=TEXT,
    )
ax2.set_xlim(0, 1)
style(ax2, "F1 Score Ranking", xlabel="F1", xgrid=True, ygrid=False)
ax2.grid(True, axis="x", color=BORDER, linestyle="--", alpha=0.6)


# ── Plot 3: ROC Curves ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
for i, n in enumerate(names):
    ax3.plot(
        results[n]["fpr"],
        results[n]["tpr"],
        color=PALETTE[i % len(PALETTE)],
        lw=2,
        alpha=0.85,
        label=f"{n}  (AUC={results[n]['auc']:.3f})",
    )

ax3.plot([0, 1], [0, 1], color=MUTED, linestyle="--", lw=1, alpha=0.5)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1.02)
ax3.legend(fontsize=7.5, labelcolor=TEXT, loc="lower right")
style(
    ax3,
    "ROC Curves",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    xgrid=True,
)
ax3.grid(True, color=BORDER, linestyle="--", alpha=0.6)


# ── Plot 4: Training time bar ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
times = [results[n]["time"] for n in names]
tcols = [
    GREEN if t == min(times) else RED if t == max(times) else ACCENT for t in times
]

bars = ax4.bar(range(len(names)), times, color=tcols, alpha=0.85)
for bar, t in zip(bars, times):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{t:.2f}s",
        ha="center",
        va="bottom",
        fontsize=7,
        color=TEXT,
    )
ax4.set_xticks(range(len(names)))
ax4.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
style(ax4, "Training Time  (green=fastest · red=slowest)", ylabel="Seconds")


# ── Plots 5-7: Confusion matrices for top-3 models by F1 ────────────
top3 = sorted(names, key=lambda n: results[n]["f1"], reverse=True)[:3]

for col_idx, n in enumerate(top3):
    ax_cm = fig.add_subplot(gs[2, col_idx])
    cm = results[n]["cm"]

    # Custom annotation: show percentage too
    cm_pct = cm.astype(float) / cm.sum()
    annot = np.array(
        [
            [f"{v}\n({p:.1%})" for v, p in zip(row_v, row_p)]
            for row_v, row_p in zip(cm, cm_pct)
        ]
    )

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Sell", "Buy"],
        yticklabels=["Sell", "Buy"],
        ax=ax_cm,
        cbar=False,
        linewidths=2,
        linecolor=DARK,
        annot_kws={"size": 11, "weight": "bold"},
    )

    rank = col_idx + 1
    medal = ["🥇", "🥈", "🥉"][col_idx]
    ax_cm.set_title(
        f"{medal} #{rank}  {n}\n"
        f"F1={results[n]['f1']:.3f}  |  AUC={results[n]['auc']:.3f}",
        color=TEXT,
        fontsize=9,
        fontweight="bold",
        pad=10,
    )
    ax_cm.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax_cm.set_ylabel("Actual", color=MUTED, fontsize=8)
    ax_cm.tick_params(colors=MUTED)
    ax_cm.set_facecolor(CARD)

out = os.path.join(os.path.dirname(__file__), "model_comparison_dashboard.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\nDashboard saved → {out}")
plt.show()

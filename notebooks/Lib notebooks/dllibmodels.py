import numpy as np
import pandas as pd
import os, sys, time, warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    SimpleRNN,
    Conv1D,
    GlobalMaxPooling1D,
    Dropout,
    BatchNormalization,
    Input,
    Bidirectional,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow : {tf.__version__}")

# ════════════════════════════════════════════════════════════════════
#  1. DATA LOADING
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

n_feat = X_train.shape[1]
n_buy = Y_train.sum()
n_sell = len(Y_train) - n_buy
cw = {0: 1.0, 1: n_sell / n_buy}  # keras class_weight

print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")
print(f"Buy: {n_buy}  Sell: {n_sell}  → class_weight[1] = {cw[1]:.2f}\n")

# Sequence shape for RNN / CNN  → (samples, timesteps=n_features, 1)
X_train_seq = X_train.reshape(-1, n_feat, 1)
X_test_seq = X_test.reshape(-1, n_feat, 1)

# ════════════════════════════════════════════════════════════════════
#  2. KERAS BUILDERS
# ════════════════════════════════════════════════════════════════════
ES = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
)


def compile_and_fit(model, X_tr, Y_tr, X_te, Y_te, epochs=100, batch=64):
    model.compile(
        optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        X_tr,
        Y_tr,
        validation_data=(X_te, Y_te),
        epochs=epochs,
        batch_size=batch,
        class_weight=cw,
        callbacks=[ES],
        verbose=0,
    )
    return model


def build_rnn():
    return Sequential(
        [
            Input(shape=(n_feat, 1)),
            SimpleRNN(64, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="RNN",
    )


def build_lstm():
    return Sequential(
        [
            Input(shape=(n_feat, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="LSTM",
    )


def build_gru():
    return Sequential(
        [
            Input(shape=(n_feat, 1)),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="GRU",
    )


def build_bilstm():
    return Sequential(
        [
            Input(shape=(n_feat, 1)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="Bi-LSTM",
    )


def build_cnn():
    return Sequential(
        [
            Input(shape=(n_feat, 1)),
            Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            Conv1D(32, kernel_size=2, activation="relu", padding="same"),
            GlobalMaxPooling1D(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="1D-CNN",
    )


def build_cnn_lstm():
    return Sequential(
        [
            Input(shape=(n_feat, 1)),
            Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="CNN-LSTM",
    )


# ════════════════════════════════════════════════════════════════════
#  3. MODEL REGISTRY
# ════════════════════════════════════════════════════════════════════
all_models = {
    # ── Sklearn ──────────────────────────────────────────────────────
    "Logistic Reg.": (
        "sklearn",
        LogisticRegression(class_weight="balanced", max_iter=1000, C=0.1),
    ),
    "Random Forest": (
        "sklearn",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
        ),
    ),
    "Gradient Boost": (
        "sklearn",
        GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
    ),
    "MLP (sklearn)": (
        "sklearn",
        MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=500,
            learning_rate_init=0.001,
            early_stopping=True,
            random_state=42,
        ),
    ),
    # ── XGBoost ──────────────────────────────────────────────────────
    "XGBoost": (
        "sklearn",
        XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=n_sell / n_buy,  # handles class imbalance
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    ),
    # ── Deep Learning ─────────────────────────────────────────────────
    "RNN": ("keras", build_rnn),
    "LSTM": ("keras", build_lstm),
    "GRU": ("keras", build_gru),
    "Bi-LSTM": ("keras", build_bilstm),
    "1D-CNN": ("keras", build_cnn),
    "CNN-LSTM": ("keras", build_cnn_lstm),
}

# ════════════════════════════════════════════════════════════════════
#  4. TRAIN & EVALUATE
# ════════════════════════════════════════════════════════════════════
results = {}

for name, (kind, model_or_fn) in all_models.items():
    print(f"  Training  {name:<18}", end=" ", flush=True)
    t0 = time.time()

    if kind == "sklearn":
        model = model_or_fn
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        Y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model = compile_and_fit(model_or_fn(), X_train_seq, Y_train, X_test_seq, Y_test)
        Y_prob = model.predict(X_test_seq, verbose=0).flatten()
        Y_pred = (Y_prob > 0.5).astype(int)

    elapsed = time.time() - t0
    fpr, tpr, _ = roc_curve(Y_test, Y_prob)

    results[name] = {
        "kind": kind,
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
    print(f"F1={r['f1']:.3f}  AUC={r['auc']:.3f}  ({elapsed:.1f}s)")

names = list(results.keys())

# ════════════════════════════════════════════════════════════════════
#  5. TERMINAL SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════
print("\n" + "═" * 76)
print(
    f"{'Model':<20} {'Type':<8} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'Time':>7}"
)
print("═" * 76)
for n in sorted(names, key=lambda x: results[x]["f1"], reverse=True):
    r = results[n]
    print(
        f"{n:<20} {r['kind']:<8} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
        f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f} {r['time']:>6.1f}s"
    )
print("═" * 76)

# ════════════════════════════════════════════════════════════════════
#  6. VISUALIZATION DASHBOARD
# ════════════════════════════════════════════════════════════════════
DARK = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
RED = "#f85149"
GREEN = "#3fb950"
ACCENT = "#58a6ff"

SKL_COLS = ["#58a6ff", "#79c0ff", "#cae8ff", "#388bfd", "#f0883e"]
KERAS_COLS = ["#3fb950", "#56d364", "#d2a8ff", "#ffa657", "#ff7b72", "#bc8cff"]


def model_color(name):
    if results[name]["kind"] == "sklearn":
        skl_names = [n for n in names if results[n]["kind"] == "sklearn"]
        return SKL_COLS[skl_names.index(name) % len(SKL_COLS)]
    else:
        dl_names = [n for n in names if results[n]["kind"] == "keras"]
        return KERAS_COLS[dl_names.index(name) % len(KERAS_COLS)]


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
        "legend.facecolor": CARD,
        "legend.edgecolor": BORDER,
        "font.family": "monospace",
        "font.size": 8,
    }
)

fig = plt.figure(figsize=(26, 24), facecolor=DARK)
fig.suptitle(
    "Deep Learning vs ML + XGBoost  ·  AAPL Buy/Sell Classifier\n"
    "  ■ Blue = Sklearn   ■ Orange = XGBoost   ■ Green/Warm = Deep Learning  (RNN · LSTM · GRU · Bi-LSTM · CNN)",
    fontsize=13,
    fontweight="bold",
    color=TEXT,
    y=0.99,
    linespacing=1.7,
)

gs = gridspec.GridSpec(
    4,
    3,
    figure=fig,
    hspace=0.55,
    wspace=0.38,
    top=0.95,
    bottom=0.04,
    left=0.06,
    right=0.97,
)


def style(ax, title, xlabel="", ylabel="", xgrid=False):
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, axis="y", color=BORDER, linestyle="--", alpha=0.6)
    if xgrid:
        ax.grid(True, axis="x", color=BORDER, linestyle="--", alpha=0.6)


# ── 1. Grouped bar — all metrics (full width) ───────────────────────
ax1 = fig.add_subplot(gs[0, :])
metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
metric_colors = [ACCENT, "#3fb950", "#d29922", "#f0883e", "#d2a8ff"]
x = np.arange(len(names))
bw = 0.13

for i, (mk, ml, mc) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
    vals = [results[n][mk] for n in names]
    ax1.bar(x + i * bw, vals, bw, label=ml, color=mc, alpha=0.85)

ax1.set_xticks(x + bw * 2)
ax1.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.axhline(0.5, color=RED, linestyle="--", lw=1, alpha=0.4)
ax1.legend(fontsize=8, labelcolor=TEXT)

# Separator between sklearn/xgboost and keras
n_skl = sum(1 for n in names if results[n]["kind"] == "sklearn")
sep_x = n_skl - 0.4
ax1.axvline(sep_x, color=MUTED, linestyle=":", lw=1.5, alpha=0.7)
ax1.text(
    sep_x - 0.3, 1.09, "◄ sklearn + XGBoost", color=MUTED, fontsize=7.5, ha="right"
)
ax1.text(sep_x + 0.1, 1.09, "Deep Learning ►", color=MUTED, fontsize=7.5)
style(ax1, "All Metrics — Sklearn / XGBoost vs Deep Learning", ylabel="Score")


# ── 2. F1 ranked horizontal bar ─────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ranked = sorted(names, key=lambda n: results[n]["f1"])
bars = ax2.barh(
    ranked,
    [results[n]["f1"] for n in ranked],
    color=[model_color(n) for n in ranked],
    alpha=0.85,
    height=0.6,
)
for bar, n in zip(bars, ranked):
    val = results[n]["f1"]
    ax2.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=7.5,
        color=TEXT,
    )
ax2.set_xlim(0, 1)
ax2.grid(True, axis="x", color=BORDER, linestyle="--", alpha=0.6)
style(ax2, "F1 Score Ranking", xlabel="F1 Score", xgrid=True)


# ── 3. AUC ranked horizontal bar ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ranked_auc = sorted(names, key=lambda n: results[n]["auc"])
bars = ax3.barh(
    ranked_auc,
    [results[n]["auc"] for n in ranked_auc],
    color=[model_color(n) for n in ranked_auc],
    alpha=0.85,
    height=0.6,
)
for bar, n in zip(bars, ranked_auc):
    val = results[n]["auc"]
    ax3.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=7.5,
        color=TEXT,
    )
ax3.set_xlim(0, 1)
ax3.grid(True, axis="x", color=BORDER, linestyle="--", alpha=0.6)
style(ax3, "AUC-ROC Ranking", xlabel="AUC", xgrid=True)


# ── 4. Training time ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
times = [results[n]["time"] for n in names]
bars = ax4.bar(
    range(len(names)), times, color=[model_color(n) for n in names], alpha=0.85
)
for bar, t in zip(bars, times):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{t:.1f}s",
        ha="center",
        va="bottom",
        fontsize=7,
        color=TEXT,
    )
ax4.set_xticks(range(len(names)))
ax4.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
style(ax4, "Training Time (seconds)", ylabel="Seconds")


# ── 5. ROC curves (span 2 cols) ──────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
for n in names:
    r = results[n]
    lw = 2.5 if r["kind"] == "keras" else 1.8
    dash = "solid" if r["kind"] == "keras" else "dashed"
    tag = "[DL]" if r["kind"] == "keras" else "[ML]"
    ax5.plot(
        r["fpr"],
        r["tpr"],
        color=model_color(n),
        lw=lw,
        linestyle=dash,
        alpha=0.85,
        label=f"{tag} {n}  AUC={r['auc']:.3f}",
    )
ax5.plot([0, 1], [0, 1], color=MUTED, linestyle="--", lw=1, alpha=0.4)
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1.03)
ax5.legend(fontsize=7, labelcolor=TEXT, loc="lower right", ncol=2)
ax5.grid(True, color=BORDER, linestyle="--", alpha=0.5)
style(
    ax5,
    "ROC Curves  (solid = Deep Learning · dashed = Sklearn/XGBoost)",
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
)


# ── 6. Radar chart — top 5 by AUC ───────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2], polar=True)
top5 = sorted(names, key=lambda n: results[n]["auc"], reverse=True)[:5]
radar_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
radar_labels = ["Acc", "Prec", "Rec", "F1", "AUC"]
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax6.set_facecolor(CARD)
ax6.spines["polar"].set_color(BORDER)
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(radar_labels, color=TEXT, fontsize=8)
ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
ax6.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color=MUTED, fontsize=6)
ax6.set_ylim(0, 1)
ax6.grid(color=BORDER, linestyle="--", alpha=0.5)

for n in top5:
    vals = [results[n][m] for m in radar_metrics] + [results[n][radar_metrics[0]]]
    c = model_color(n)
    ax6.plot(angles, vals, "o-", lw=1.8, color=c, alpha=0.85, label=n)
    ax6.fill(angles, vals, color=c, alpha=0.08)

ax6.legend(
    loc="upper right",
    bbox_to_anchor=(1.38, 1.12),
    fontsize=7,
    labelcolor=TEXT,
    framealpha=0,
)
ax6.set_title("Radar — Top 5 by AUC", color=TEXT, fontsize=9, fontweight="bold", pad=14)


# ── 7-9. Confusion matrices — top-3 by F1 ───────────────────────────
top3 = sorted(names, key=lambda n: results[n]["f1"], reverse=True)[:3]
medals = ["🥇", "🥈", "🥉"]

for col_idx, n in enumerate(top3):
    ax_cm = fig.add_subplot(gs[3, col_idx])
    cm = results[n]["cm"]
    cm_pct = cm.astype(float) / cm.sum()
    annot = np.array(
        [[f"{v}\n({p:.1%})" for v, p in zip(rv, rp)] for rv, rp in zip(cm, cm_pct)]
    )

    if results[n]["kind"] == "keras":
        cmap = "Greens"
    elif n == "XGBoost":
        cmap = "Oranges"
    else:
        cmap = "Blues"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=["Sell", "Buy"],
        yticklabels=["Sell", "Buy"],
        ax=ax_cm,
        cbar=False,
        linewidths=2,
        linecolor=DARK,
        annot_kws={"size": 11, "weight": "bold"},
    )

    kind_tag = "[DL]" if results[n]["kind"] == "keras" else "[ML]"
    ax_cm.set_title(
        f"{medals[col_idx]} #{col_idx+1}  {kind_tag} {n}\n"
        f"F1={results[n]['f1']:.3f}  AUC={results[n]['auc']:.3f}",
        color=TEXT,
        fontsize=9,
        fontweight="bold",
        pad=10,
    )
    ax_cm.set_xlabel("Predicted", color=MUTED, fontsize=8)
    ax_cm.set_ylabel("Actual", color=MUTED, fontsize=8)
    ax_cm.tick_params(colors=MUTED)
    ax_cm.set_facecolor(CARD)


out = os.path.join(os.path.dirname(__file__), "deep_model_dashboard.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\n✓ Dashboard saved → {out}")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os
import sys
import pickle
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "model_params.pkl")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "model_metrics.json")


def load_or_train_models(force_retrain=False):
    """Load saved models or train new ones"""
    if os.path.exists(PARAMS_FILE) and not force_retrain:
        print("Loading saved models...")
        with open(PARAMS_FILE, "rb") as f:
            return pickle.load(f)
    else:
        print("Training new models...")
        return train_all_models()


def save_models_params(models_params):
    """Save model parameters"""
    with open(PARAMS_FILE, "wb") as f:
        pickle.dump(models_params, f)


def save_metrics(metrics):
    """Save model metrics"""
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics():
    """Load saved metrics"""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return None


def prepare_data():
    """Prepare data for all models"""
    combined, _ = utils.load_multiple_stocks()
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

    # Per-stock rolling Z-score normalization
    window = 100
    for col in feature_cols:
        df_binary[f"{col}_z"] = (
            df_binary.groupby("ticker")[col]
            .transform(
                lambda s: (
                    (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-8)
                )
            )
            .fillna(0)
        )

    feature_cols_z = [f"{col}_z" for col in feature_cols]
    df_binary = df_binary.dropna(subset=feature_cols_z)

    X = df_binary[feature_cols_z].to_numpy()
    Y = df_binary["target"].to_numpy()

    # Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    return X_train, X_test, Y_train, Y_test, df_binary[feature_cols_z + ["target"]]


def train_slp(X_train, Y_train):
    """Train SLP model"""
    from models.mySLP import SLPFromScratch

    Y_train_reshaped = Y_train.reshape(1, -1)
    model = SLPFromScratch(lr=0.01, epochs=10000)
    model.fit(X_train, Y_train_reshaped, class_weight={1: 1.0, 0: 1.0})
    return model


def train_mlp(X_train, Y_train):
    """Train MLP model"""
    from models.mymlp import MLPFromScratched

    Y_train_reshaped = Y_train.reshape(1, -1)
    model = MLPFromScratched(lr=0.01, epochs=3000, h=[32, 32])
    model.fit(X_train, Y_train_reshaped, class_weight={1: 1.0, 0: 1.0})
    return model


def train_svm_soft_margin(X_train, Y_train):
    """Train SVM Soft Margin model"""
    from models import mySVM

    model = mySVM.SVMSoftmargin(alpha=0.001, iteration=2000, lambda_=0.01)
    w, b = model.fit(X_train, Y_train)
    return {"w": w, "b": b, "model": model}


def train_svm_dual(X_train, Y_train):
    """Train SVM Dual model"""
    from models import mySVM

    model = mySVM.SVM_Dual(
        kernel="rbf", degree=2, sigma=1.0, epoches=500, learning_rate=0.001
    )
    model.train(X_train, Y_train)
    return model


def train_logistic_regression(X_train, Y_train):
    """Train Logistic Regression model"""
    from models.myLogisticRegression import LogisticRegressionFromScratch

    model = LogisticRegressionFromScratch(lr=0.01, epochs=10000)
    model.fit(X_train, Y_train)
    return model


def train_kmeans(X_train, Y_train):
    """Train K-means model"""
    from models.myKmean import KMeansFromScratch

    model = KMeansFromScratch(k=2, max_iters=100)
    model.fit(X_train)
    return model


def train_rnn(X_train, Y_train):
    """Train RNN model"""

    def create_sequences(X, Y, seq_len=20):
        Xs, Ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len : i])
            Ys.append(Y[i])
        return np.array(Xs), np.array(Ys)

    SEQ_LEN = 20
    X_train_seq, Y_train_seq = create_sequences(X_train, Y_train, SEQ_LEN)

    n_buy = int(Y_train_seq.sum())
    n_sell = Y_train_seq.shape[0] - n_buy
    class_weight = {
        1: (n_buy + n_sell) / (2 * n_buy),
        0: (n_buy + n_sell) / (2 * n_sell),
    }

    from models.myRNN import RNNFromScratch

    model = RNNFromScratch(lr=0.01, epochs=3000, hidden_size=32)
    model.fit(X_train_seq, Y_train_seq.reshape(1, -1), class_weight)
    return model


def train_decision_tree(X_train, Y_train):
    """Train Decision Tree model"""
    from models.myTree import DecisionTreeFromScratch

    model = DecisionTreeFromScratch(
        max_depth=10, min_samples_split=50, min_samples_leaf=25
    )
    model.fit(X_train, Y_train)
    return model


def train_random_forest(X_train, Y_train):
    """Train Random Forest model"""
    from models.myForest import RandomForestFromScratch

    model = RandomForestFromScratch(
        n_estimators=50, max_depth=10, min_samples_split=50, min_samples_leaf=25
    )
    model.fit(X_train, Y_train)
    return model


def train_all_models():
    """Train all classification models"""
    X_train, X_test, Y_train, Y_test, _ = prepare_data()

    # Convert Y to 0/1 for some models
    Y_train_binary = ((Y_train + 1) / 2).astype(int)
    Y_test_binary = ((Y_test + 1) / 2).astype(int)

    models = {}

    print("Training SLP...")
    models["SLP"] = train_slp(X_train, Y_train_binary)

    print("Training MLP...")
    models["MLP"] = train_mlp(X_train, Y_train_binary)

    print("Training SVM Soft Margin...")
    models["SVM_Soft"] = train_svm_soft_margin(X_train, Y_train)

    print("Training SVM Dual...")
    models["SVM_Dual"] = train_svm_dual(X_train, Y_train)

    print("Training Logistic Regression...")
    models["Logistic"] = train_logistic_regression(X_train, Y_train_binary)

    print("Training K-means...")
    models["Kmeans"] = train_kmeans(X_train, Y_train_binary)

    print("Training RNN...")
    models["RNN"] = train_rnn(X_train, Y_train_binary)

    print("Training Decision Tree...")
    models["DecisionTree"] = train_decision_tree(X_train, Y_train)

    print("Training Random Forest...")
    models["RandomForest"] = train_random_forest(X_train, Y_train)

    return models


def evaluate_models(models):
    """Evaluate all trained models"""
    X_train, X_test, Y_train, Y_test, _ = prepare_data()
    Y_test_binary = ((Y_test + 1) / 2).astype(int)

    results = {}

    # SLP
    if "SLP" in models:
        Y_pred_prob = models["SLP"].predict_proba(X_test)
        Y_pred = (Y_pred_prob > 0.5).astype(int).ravel()
        results["SLP"] = compute_metrics(Y_test_binary, Y_pred)

    # MLP
    if "MLP" in models:
        Y_pred_prob = models["MLP"].predict(X_test)
        Y_pred = Y_pred_prob.ravel()
        results["MLP"] = compute_metrics(Y_test_binary, Y_pred)

    # SVM Soft Margin
    if "SVM_Soft" in models:
        Y_pred = models["SVM_Soft"]["model"].predict(X_test)
        results["SVM_Soft"] = compute_metrics(Y_test, Y_pred)

    # SVM Dual
    if "SVM_Dual" in models:
        Y_pred = models["SVM_Dual"].predict(X_test)
        results["SVM_Dual"] = compute_metrics(Y_test, Y_pred)

    # Logistic Regression
    if "Logistic" in models:
        Y_pred_prob = models["Logistic"].predict_proba(X_test)
        Y_pred = (Y_pred_prob > 0.5).astype(int).ravel()
        results["Logistic"] = compute_metrics(Y_test_binary, Y_pred)

    # K-means
    if "Kmeans" in models:
        Y_pred = models["Kmeans"].predict(X_test)
        results["Kmeans"] = compute_metrics(Y_test_binary, Y_pred)

    # RNN
    if "RNN" in models:

        def create_sequences(X, Y, seq_len=20):
            Xs, Ys = [], []
            for i in range(seq_len, len(X)):
                Xs.append(X[i - seq_len : i])
                Ys.append(Y[i])
            return np.array(Xs), np.array(Ys)

        SEQ_LEN = 20
        X_test_seq, Y_test_seq = create_sequences(X_test, Y_test_binary, SEQ_LEN)

        Y_pred_prob = models["RNN"].predict_proba(X_test_seq)
        Y_pred = (Y_pred_prob > 0.5).astype(int).ravel()
        results["RNN"] = compute_metrics(Y_test_seq, Y_pred)

    # Decision Tree
    if "DecisionTree" in models:
        Y_pred = models["DecisionTree"].predict(X_test)
        results["DecisionTree"] = compute_metrics(Y_test, Y_pred)

    # Random Forest
    if "RandomForest" in models:
        Y_pred = models["RandomForest"].predict(X_test)
        results["RandomForest"] = compute_metrics(Y_test, Y_pred)

    return results


def compute_metrics(y_true, y_pred):
    """Compute classification metrics"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def create_classification_dashboard(results):
    """Create comprehensive classification model comparison dashboard"""
    # Create dashboard
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(
        "Classification Models Comparison Dashboard", fontsize=16, fontweight="bold"
    )

    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    # Metrics comparison
    x = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        values = [results[name][metric] for name in model_names]
        bars = ax.bar(x, values, width, alpha=0.7)
        ax.set_title(f"{metric.capitalize()} Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                ".3f",
                ha="center",
                va="bottom",
            )

    # Confusion matrices for top 4 models
    top_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[
        :4
    ]

    for i, (name, result) in enumerate(top_models):
        row = 2
        col = i
        ax = axes[row, col]

        cm = np.array(result["cm"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Sell", "Buy"],
            yticklabels=["Sell", "Buy"],
            cbar=False,
        )
        ax.set_title(f"{name} Confusion Matrix")

    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print("CLASSIFICATION MODELS SUMMARY")
    print("=" * 80)
    print("<12")
    print("-" * 80)

    for name, result in results.items():
        print("<12")


def main(force_retrain=False):
    """Main function to run classification comparison"""
    # Load or train models
    models = load_or_train_models(force_retrain)

    # Save models
    save_models_params(models)

    # Evaluate models
    results = evaluate_models(models)

    # Save metrics
    save_metrics(results)

    # Create dashboard
    create_classification_dashboard(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classification Models Comparison")
    parser.add_argument("--retrain", action="store_true", help="Force retrain models")

    args, unknown = parser.parse_known_args()
    main(force_retrain=args.retrain)

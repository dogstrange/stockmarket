import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "regression_model_params.pkl")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "regression_model_metrics.json")


def load_or_train_models(force_retrain=False):
    """Load saved models or train new ones"""
    if os.path.exists(PARAMS_FILE) and not force_retrain:
        print("Loading saved regression models...")
        with open(PARAMS_FILE, "rb") as f:
            return pickle.load(f)
    else:
        print("Training new regression models...")
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
    """Prepare data for regression models"""
    combined, _ = utils.load_multiple_stocks()
    df = combined.dropna()

    # Add target for next day return prediction
    df["target"] = df["Log_Ret"].shift(-1)
    df = df.dropna()

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

    # Chronological split: 60% train, 20% val, 20% test
    n_total = len(X)
    train_end = int(0.6 * n_total)
    val_end = int(0.8 * n_total)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    Y_train, Y_val, Y_test = Y[:train_end], Y[train_end:val_end], Y[val_end:]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def train_multiple_linear():
    """Train Multiple Linear Regression model"""
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data()

    from models.myMultipleLinear import StockLinearRegression

    model = StockLinearRegression(learning_rate=0.01, max_epochs=8000, patience=200)
    model.fit(X_train, Y_train)

    return model


def train_polynomial():
    """Train Polynomial Regression model"""
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data()

    from models.myPolynomial import StockPolynomialRegression

    model = StockPolynomialRegression(
        degree=2, learning_rate=0.01, max_epochs=6000, patience=300, l2_lambda=0.001
    )
    model.fit(X_train, Y_train, X_val, Y_val)

    return model


def train_simple_linear():
    """Train Simple Linear Regression model"""
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data()

    # Use only Log_Ret as single feature
    X_train_simple = X_train[:, 9:10]  # Log_Ret column
    X_test_simple = X_test[:, 9:10]

    from models.mySimpleLinear import SimpleLinearRegression

    model = SimpleLinearRegression(learning_rate=0.01, epochs=5000)
    model.fit(X_train_simple.ravel(), Y_train.ravel())

    return model


def train_adaboost():
    """Train AdaBoost Regression model"""
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data()

    from models.myAdaboost import StockAdaBoostRegressor

    model = StockAdaBoostRegressor(n_estimators=50, learning_rate=1.0)
    model.fit(X_train, Y_train.ravel())

    return model


def train_all_models():
    """Train all regression models"""
    models = {}

    print("Training Multiple Linear Regression...")
    models["Multiple_Linear"] = train_multiple_linear()

    print("Training Polynomial Regression...")
    models["Polynomial"] = train_polynomial()

    print("Training Simple Linear Regression...")
    models["Simple_Linear"] = train_simple_linear()

    print("Training AdaBoost Regression...")
    models["AdaBoost"] = train_adaboost()

    return models


def evaluate_models(models):
    """Evaluate all trained models"""
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data()

    results = {}

    # Multiple Linear
    if "Multiple_Linear" in models:
        Y_pred = models["Multiple_Linear"].predict(X_test)
        results["Multiple_Linear"] = compute_regression_metrics(Y_test, Y_pred)

    # Polynomial
    if "Polynomial" in models:
        Y_pred = models["Polynomial"].predict(X_test)
        results["Polynomial"] = compute_regression_metrics(Y_test, Y_pred)

    # Simple Linear
    if "Simple_Linear" in models:
        X_test_simple = X_test[:, 9:10]  # Log_Ret column
        Y_pred = models["Simple_Linear"].predict(X_test_simple.ravel())
        results["Simple_Linear"] = compute_regression_metrics(
            Y_test, Y_pred.reshape(-1, 1)
        )

    # AdaBoost
    if "AdaBoost" in models:
        Y_pred = models["AdaBoost"].predict(X_test)
        results["AdaBoost"] = compute_regression_metrics(Y_test, Y_pred.reshape(-1, 1))

    return results


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    rse = np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - rse
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "direction_acc": direction_acc,
        "predictions": y_pred.tolist(),
        "actuals": y_true.tolist(),
    }


def create_regression_dashboard(results):
    """Create comprehensive regression model comparison dashboard"""
    n_models = len(results)
    n_cols = min(n_models, 4)  # Max 4 columns
    n_rows = (n_models + n_cols - 1) // n_cols + 1  # +1 for metrics row

    # Create dashboard
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    fig.suptitle(
        "Regression Models Comparison Dashboard", fontsize=16, fontweight="bold"
    )

    model_names = list(results.keys())
    metrics = ["mae", "rmse", "r2", "direction_acc"]

    # Metrics comparison in first row
    x = np.arange(len(model_names))
    width = 0.2

    # MAE and RMSE
    ax1 = axes[0, 0] if n_rows > 1 else axes[0]
    mae_values = [results[name]["mae"] for name in model_names]
    rmse_values = [results[name]["rmse"] for name in model_names]
    mae_bars = ax1.bar(
        x - width / 2, mae_values, width, label="MAE", alpha=0.7, color="skyblue"
    )
    rmse_bars = ax1.bar(
        x + width / 2, rmse_values, width, label="RMSE", alpha=0.7, color="lightcoral"
    )
    ax1.set_title("Error Metrics Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylabel("Error Value")

    # R²
    ax2 = axes[0, 1] if n_rows > 1 else axes[1]
    r2_values = [results[name]["r2"] for name in model_names]
    r2_bars = ax2.bar(x, r2_values, width, alpha=0.7, color="lightgreen")
    ax2.set_title("R² Score Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.set_ylabel("R² Score")
    ax2.set_ylim(-1, 1)

    # Direction Accuracy
    ax3 = axes[0, 2] if n_cols > 2 else axes[0, min(2, n_cols - 1)]
    dir_values = [results[name]["direction_acc"] for name in model_names]
    dir_bars = ax3.bar(x, dir_values, width, alpha=0.7, color="gold")
    ax3.set_title("Direction Accuracy Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha="right")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(0, 100)

    # Hide unused axes in first row if any
    if n_cols > 3:
        for j in range(3, n_cols):
            axes[0, j].set_visible(False)

    # Add value labels
    for ax, bars, values in [
        (ax1, mae_bars, mae_values),
        (ax1, rmse_bars, rmse_values),
        (ax2, r2_bars, r2_values),
        (ax3, dir_bars, dir_values),
    ]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.01 if height >= 0 else -0.05),
                ".4f",
                ha="center",
                va="bottom" if height >= 0 else "top",
            )

    # Prediction vs Actual scatter plots in subsequent rows
    for i, (name, result) in enumerate(results.items()):
        if result and "predictions" in result and "actuals" in result:
            row = (i // n_cols) + 1
            col = i % n_cols
            ax = axes[row, col]

            pred = np.array(result["predictions"]).ravel()
            actual = np.array(result["actuals"]).ravel()

            # Scatter plot
            ax.scatter(actual, pred, alpha=0.6, s=10, color=f"C{i}")
            ax.plot(
                [actual.min(), actual.max()],
                [actual.min(), actual.max()],
                "r--",
                linewidth=2,
                label="Perfect Prediction",
            )
            ax.set_xlabel("Actual Returns")
            ax.set_ylabel("Predicted Returns")
            ax.set_title(f"{name}: Predicted vs Actual")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print("REGRESSION MODELS SUMMARY")
    print("=" * 80)
    print("<15")
    print("-" * 80)

    for name, result in results.items():
        print("<15")


def main(force_retrain=False):
    """Main function to run regression comparison"""
    # Load or train models
    models = load_or_train_models(force_retrain)

    # Save models
    save_models_params(models)

    # Evaluate models
    results = evaluate_models(models)

    # Save metrics
    save_metrics(results)

    # Create dashboard
    create_regression_dashboard(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regression Models Comparison")
    parser.add_argument("--retrain", action="store_true", help="Force retrain models")

    args, unknown = parser.parse_known_args()
    main(force_retrain=args.retrain)

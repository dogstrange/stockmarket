import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class StockLinearRegression:

    def __init__(self, learning_rate=0.01, max_epochs=8000, patience=200):

        self.lr = learning_rate
        self.epochs = max_epochs
        self.patience = patience

        self.W = None
        self.b = 0.0

        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    # ================================
    # TRAIN
    # ================================
    def fit(self, X, y, val_split=0.2):

        split_idx = int(len(X) * (1 - val_split))

        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

        # Normalize
        self.X_mean = X_train_raw.mean(axis=0)
        self.X_std = X_train_raw.std(axis=0) + 1e-9
        self.y_mean = y_train_raw.mean()
        self.y_std = y_train_raw.std() + 1e-9

        X_train = (X_train_raw - self.X_mean) / self.X_std
        y_train = (y_train_raw - self.y_mean) / self.y_std
        X_val = (X_val_raw - self.X_mean) / self.X_std
        y_val = (y_val_raw - self.y_mean) / self.y_std

        self.W = np.zeros((X_train.shape[1], 1))
        self.b = 0

        best_loss = float("inf")
        best_W = self.W.copy()
        best_b = self.b
        wait = 0

        for e in range(self.epochs):

            pred = X_train @ self.W + self.b
            err = pred - y_train

            dW = (2 / len(X_train)) * X_train.T @ err
            db = (2 / len(X_train)) * np.sum(err)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            val_pred = X_val @ self.W + self.b
            loss = np.mean(np.abs(val_pred - y_val))

            if loss < best_loss:
                best_loss = loss
                best_W = self.W.copy()
                best_b = self.b
                wait = 0
            else:
                wait += 1

            if wait >= self.patience:
                print("Early stopping at epoch", e)
                break

        self.W = best_W
        self.b = best_b

        return best_loss

    # ================================
    # PREDICT
    # ================================
    def predict(self, X):

        Xn = (X - self.X_mean) / self.X_std
        y = Xn @ self.W + self.b
        return y * self.y_std + self.y_mean

    # ================================
    # EVALUATE
    # ================================
    def evaluate_and_plot(self, X_test, y_test, n_days):

        X = X_test[:n_days]
        y = y_test[:n_days].flatten()
        preds = self.predict(X).flatten()

        mae = np.mean(np.abs(y - preds))
        mse = np.mean((y - preds) ** 2)
        rmse = np.sqrt(mse)
        rse = np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - rse
        mape = np.mean(np.abs((y - preds) / (y + 1e-9))) * 100
        direction_acc = np.mean(np.sign(y) == np.sign(preds)) * 100
        baseline_mae = np.mean(np.abs(y - 0))

        print("\n========== FINAL SCORES ==========")
        print(f"MAE      : {mae:.6f}")
        print(f"RMSE     : {rmse:.6f}")
        print(f"R²       : {r2:.6f}")
        print(f"MAPE %   : {mape:.2f}")
        print(f"DIR ACC% : {direction_acc:.2f}")
        print(f"BASE MAE : {baseline_mae:.6f}")

        # Print Final Weights
        print("\n========== FINAL WEIGHTS ==========")
        for i, w in enumerate(self.W.flatten()):
            print(f"W[{i}] = {w:.6f}")
        print("Bias =", self.b)

        # Plot
        plt.figure(figsize=(13, 5))
        plt.scatter(range(len(y)), y, color="royalblue", alpha=0.4, label="Real")
        plt.plot(preds, color="darkorange", linewidth=2, label="Prediction")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Stock Return Prediction")
        plt.legend()
        plt.show()

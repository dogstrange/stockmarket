import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearRegression:

    def __init__(self, learning_rate=0.01, epochs=5000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = 0.0
        self.b = 0.0

    def fit(self, X, y):

        n = len(X)

        for _ in range(self.epochs):

            y_pred = self.W * X + self.b
            error = y_pred - y

            dW = (2 / n) * np.sum(error * X)
            db = (2 / n) * np.sum(error)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        return self.W * X + self.b

    def evaluate(self, X_test, y_test):

        preds = self.predict(X_test)

        mae = np.mean(np.abs(y_test - preds))
        mse = np.mean((y_test - preds) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        direction_acc = np.mean(np.sign(y_test) == np.sign(preds)) * 100

        print("\n========== SIMPLE LINEAR REGRESSION ==========")
        print(f"W (slope)   : {self.W:.6f}")
        print(f"b (bias)    : {self.b:.6f}")
        print(f"MAE         : {mae:.6f}")
        print(f"RMSE        : {rmse:.6f}")
        print(f"R²          : {r2:.6f}")
        print(f"DIR ACC %   : {direction_acc:.2f}")

        # Plot
        plt.figure(figsize=(12, 5))
        plt.scatter(range(len(y_test)), y_test, alpha=0.4, label="Real")
        plt.plot(preds, color="orange", label="Prediction")
        plt.axhline(0, color="black", linestyle="--")
        plt.legend()
        plt.title("Simple Linear Regression - Next Day Return")
        plt.show()

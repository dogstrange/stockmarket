import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class StockPolynomialRegression:

    def __init__(
        self,
        degree=2,
        learning_rate=0.01,
        max_epochs=6000,
        patience=300,
        l2_lambda=0.001,
    ):

        self.degree = degree
        self.lr = learning_rate
        self.epochs = max_epochs
        self.patience = patience
        self.l2 = l2_lambda

        self.W = None
        self.b = 0.0

        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def create_polynomial_features(self, X):

        n_samples, n_features = X.shape
        features = [X]

        # power terms
        for d in range(2, self.degree + 1):
            features.append(X**d)

        # interaction terms
        if self.degree >= 2:
            interactions = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
            if interactions:
                features.append(np.hstack(interactions))

        return np.hstack(features)

    def fit(self, X_train, y_train, X_val, y_val):

        X_train = self.create_polynomial_features(X_train)
        X_val = self.create_polynomial_features(X_val)

        # Normalize
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0) + 1e-9
        self.y_mean = y_train.mean()
        self.y_std = y_train.std() + 1e-9

        X_train = (X_train - self.X_mean) / self.X_std
        X_val = (X_val - self.X_mean) / self.X_std

        y_train = (y_train - self.y_mean) / self.y_std
        y_val = (y_val - self.y_mean) / self.y_std

        self.W = np.zeros((X_train.shape[1], 1))
        self.b = 0

        best_loss = float("inf")
        wait = 0

        for e in range(self.epochs):

            pred = X_train @ self.W + self.b
            err = pred - y_train

            dW = (2 / len(X_train)) * (X_train.T @ err) + 2 * self.l2 * self.W
            db = (2 / len(X_train)) * np.sum(err)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            val_pred = X_val @ self.W + self.b
            val_loss = np.mean((val_pred - y_val) ** 2)

            if val_loss < best_loss:
                best_loss = val_loss
                best_W = self.W.copy()
                best_b = self.b
                wait = 0
            else:
                wait += 1

            if wait >= self.patience:
                break

        self.W = best_W
        self.b = best_b

        return best_loss

    def predict(self, X):

        X = self.create_polynomial_features(X)
        X = (X - self.X_mean) / self.X_std

        y = X @ self.W + self.b
        return y * self.y_std + self.y_mean

import numpy as np


class LogisticRegressionGD:

    def __init__(self, learning_rate=0.01, epochs=12000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0.0
        self.X_mean = None
        self.X_std = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        # Normalize
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-9
        X = (X - self.X_mean) / self.X_std

        n, d = X.shape
        self.W = np.zeros((d, 1))
        self.b = 0
        y = y.reshape(-1, 1)

        self.loss_history = []

        for _ in range(self.epochs):

            linear = X @ self.W + self.b
            preds = self.sigmoid(linear)

            error = preds - y

            dW = (1 / n) * X.T @ error
            db = (1 / n) * np.sum(error)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            # Binary cross-entropy loss
            loss = -np.mean(
                y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9)
            )
            self.loss_history.append(loss)

    def predict_proba(self, X):
        X = (X - self.X_mean) / self.X_std
        return self.sigmoid(X @ self.W + self.b)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test, y_test, feature_names):

        preds = self.predict(X_test).flatten()
        y_test = y_test.flatten()

        accuracy = np.mean(preds == y_test)

        tp = np.sum((preds == 1) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print("\n========== CLASSIFICATION REPORT ==========")
        print(f"Accuracy  : {accuracy*100:.2f}%")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        print("\nConfusion Matrix")
        print("TP:", tp, " FP:", fp)
        print("FN:", fn, " TN:", tn)

        print("\n========== FEATURE IMPORTANCE ==========")
        weights = self.W.flatten()
        importance = sorted(
            zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True
        )

        for name, w in importance:
            print(f"{name:20s} {w:.6f}")

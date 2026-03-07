import numpy as np


class SLPFromScratch:
    """
    Single Layer Perceptron for binary classification.
    No hidden layers — input connects directly to a sigmoid output.

    Architecture:
        y = sigmoid(W·x + b)

    Same interface as MLPFromScratched and RNNFromScratch:
        fit(), predict(), predict_proba(), evaluate()
    """

    def __init__(self, lr=0.001, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
        self.W = None
        self.b = None

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _initialize(self, n_features):
        """Xavier initialization."""
        limit = np.sqrt(6 / (n_features + 1))
        self.W = np.random.uniform(-limit, limit, (1, n_features))
        self.b = np.zeros((1, 1))

    def _forward(self, X):
        """
        X shape: (n_features, n_samples)
        Returns: (1, n_samples)
        """
        Z = self.W @ X + self.b
        A = self.sigmoid(Z)
        self.A_cache = A
        self.X_cache = X
        return A

    def _loss(self, A_out, Y, w):
        """Weighted binary cross entropy — identical to MLP."""
        eps = 1e-9
        loss = -w * (Y * np.log(A_out + eps) + (1 - Y) * np.log(1 - A_out + eps))
        return np.mean(loss)

    def _backward(self, Y, w):
        """
        Gradient for sigmoid + BCE cancels cleanly:
            dZ = (A - Y) * w
        """
        n = Y.shape[1]
        A = self.A_cache
        X = self.X_cache

        dZ = (A - Y) * w

        dW = dZ @ X.T / n
        db = np.sum(dZ, axis=1, keepdims=True) / n

        self.W -= self.lr * dW
        self.b -= self.lr * db

    def fit(self, X, Y, class_weight=None):
        """
        X shape : (n_samples, n_features)
        Y shape : (1, n_samples)
        """
        n_samples, n_features = X.shape
        X = X.T  # (n_features, n_samples)

        if class_weight is None:
            w = np.ones_like(Y)
        else:
            w = np.where(Y == 1, class_weight[1], class_weight[0])

        self._initialize(n_features)

        for epoch in range(self.epochs):
            A_out = self._forward(X)
            loss = self._loss(A_out, Y, w)
            self.loss_history.append(loss)
            self._backward(Y, w)

            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss:.4f}"
                    f"  -> W Weight Mean: {np.mean(self.W):.6f} | Std: {np.std(self.W):.6f}"
                )

    def predict(self, X, threshold=0.5):
        X = X.T
        A_out = self._forward(X)
        return (A_out >= threshold).astype(int)

    def predict_proba(self, X):
        X = X.T
        return self._forward(X)

    def evaluate(self, Y_true, Y_pred_prob, threshold=0.5):
        """Identical to MLP.evaluate()."""
        Y_pred = (Y_pred_prob > threshold).astype(int)
        Y_true = Y_true.squeeze().astype(int)

        TP = np.sum((Y_pred == 1) & (Y_true == 1))
        FP = np.sum((Y_pred == 1) & (Y_true == 0))
        FN = np.sum((Y_pred == 0) & (Y_true == 1))
        TN = np.sum((Y_pred == 0) & (Y_true == 0))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (TP + TN) / len(Y_true)

        print(f"Accuracy:  {accuracy:.4f}")
        print(
            f"Precision: {precision:.4f}  — of all predicted Buys, how many were right"
        )
        print(f"Recall:    {recall:.4f}  — of all actual Buys, how many did we catch")
        print(f"F1 Score:  {f1:.4f}  — balance between precision and recall")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

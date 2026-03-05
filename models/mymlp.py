import numpy as np
import pandas as pd


class MLPFromScratched:
    def __init__(self, lr=0.001, epochs=10000, h=[8, 8]):
        self.lr = lr
        self.epochs = epochs
        self.h = h
        self.loss_history = []
        self.weights = []
        self.biases = []

    def soft_plus(self, X):
        return np.log1p(np.exp(X))

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def relu(self, Z):
        """Rectified Linear Unit activation function."""
        return np.maximum(0, Z)

    def _initialize(self, n_feautures):
        """
        fan_in is the size of current layer fan_out is the size of the next layer
        """
        layer_sizes = [n_feautures] + self.h + [1]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # Xavier initialization
            limit = np.sqrt(6 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_out, fan_in))
            B = np.zeros((fan_out, 1))

            self.weights.append(W)
            self.biases.append(B)

    def _forward(self, X):
        """
        'A' here is the output of activation function.
        In the loop we use the random weights and biases from the initialization.
        """
        A = X
        self.A_cache = [X]  # Store the output of activation function

        for i, (W, B) in enumerate(zip(self.weights, self.biases)):
            Z = W @ A + B
            # If it is the last layer, use Sigmoid for binary probability
            if i == len(self.weights) - 1:
                A = self.sigmoid(Z)
            # Otherwise, use ReLU for hidden layers
            else:
                A = self.relu(Z)

            self.A_cache.append(A)
        return A

    def _loss(self, A_out, Y, w):
        """
        Weighted binary cross entropy loss.
        w is the weight we created down there to prevent class imbalance.
        """
        eps = 1e-9  # tiny value to avoid log(0)
        loss = -w * (Y * np.log(A_out + eps) + (1 - Y) * np.log(1 - A_out + eps))
        return np.mean(loss)

    def _backward(self, Y, w):
        """
        Backward pass: Simplified BCE+Sigmoid derivative for the output,
        and ReLU derivative for the hidden layers.
        """
        n = Y.shape[1]  # number of samples
        n_layers = len(self.weights)
        dW_list = [None] * n_layers
        dB_list = [None] * n_layers

        A_out = self.A_cache[-1]

        # 1. Output Layer Gradient (Sigmoid + Cross Entropy)
        # The math cleanly cancels out the derivative, leaving us with this simple formula:
        dZ = (A_out - Y) * w

        for i in reversed(range(n_layers)):
            A_current = self.A_cache[i + 1]
            A_prev = self.A_cache[i]

            # 2. Hidden Layer Gradients (ReLU)
            # We already calculated dZ for the last layer before the loop.
            # For all other layers, we use the ReLU derivative.
            if i != n_layers - 1:
                dZ = dA * (A_current > 0).astype(float)

            # 3. Compute weight and bias updates
            dW_list[i] = dZ @ A_prev.T / n
            dB_list[i] = np.sum(dZ, axis=1, keepdims=True) / n

            # 4. Pass gradient backward to the previous layer
            if i > 0:  # Skip if we are at the input layer
                dA = self.weights[i].T @ dZ

        # 5. Apply the updates
        for i in range(n_layers):
            self.weights[i] -= self.lr * dW_list[i]
            self.biases[i] -= self.lr * dB_list[i]

    def fit(self, X, Y, class_weight=None):
        """
        X shape : (n_samples, n_features)
        Y shape : (1, n_samples)
        """

        n_samples, n_features = X.shape
        X = X.T  # shape becomes (n_features, n_samples)

        # Per-sample weights for class imbalance
        if class_weight is None:
            w = np.ones_like(Y)
        else:
            w = np.where(Y == 1, class_weight[1], class_weight[0])

        self._initialize(n_features)

        # Training loop
        for epoch in range(self.epochs):

            A_out = self._forward(X)

            loss = self._loss(A_out, Y, w)
            self.loss_history.append(loss)

            # Backward pass + weight update
            self._backward(Y, w)

            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch}, Loss: {loss:.4f}",
                )
                for i, W in enumerate(self.weights):
                    layer_name = (
                        f"Hidden L{i+1}"
                        if i < len(self.weights) - 1
                        else "Output Layer"
                    )
                    print(
                        f"  -> {layer_name} Weight Mean: {np.mean(W):.6f} | Std: {np.std(W):.6f}"
                    )

    def predict(self, X, threshold=0.5):
        X = X.T
        A_out = self._forward(X)

        return (A_out >= threshold).astype(int)

    def predict_proba(self, X):
        X = X.T
        return self._forward(X)

    def evaluate(self, Y_true, Y_pred_prob, threshold=0.5):
        Y_pred = (Y_pred_prob > threshold).astype(int)
        Y_true = Y_true.squeeze().astype(int)

        # For the "Buy" class (1)
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

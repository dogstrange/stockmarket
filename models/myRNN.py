import numpy as np


class RNNFromScratch:
    """
    Vanilla RNN for sequence classification, built to match MLPFromScratched's
    style: same initialization, same training loop, same evaluate() interface.

    Architecture:
        hₜ = tanh(U·xₜ + V·hₜ₋₁ + bₕ)   ← the whiteboard formula
        y  = sigmoid(W·h_last + bᵧ)        ← binary output from final hidden state
    """

    def __init__(self, lr=0.001, epochs=10000, hidden_size=32):
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.loss_history = []

        # Weight matrices (same names as the whiteboard!)
        self.U = None  # input  → hidden
        self.V = None  # hidden → hidden  (the "memory" weight)
        self.W = None  # hidden → output
        self.bh = None
        self.by = None

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def _initialize(self, input_size):
        """Xavier init — same as your MLP."""
        # U: hidden_size × input_size
        limit_U = np.sqrt(6 / (input_size + self.hidden_size))
        self.U = np.random.uniform(-limit_U, limit_U, (self.hidden_size, input_size))

        # V: hidden_size × hidden_size
        limit_V = np.sqrt(6 / (self.hidden_size + self.hidden_size))
        self.V = np.random.uniform(
            -limit_V, limit_V, (self.hidden_size, self.hidden_size)
        )

        # W: 1 × hidden_size  (binary output)
        limit_W = np.sqrt(6 / (self.hidden_size + 1))
        self.W = np.random.uniform(-limit_W, limit_W, (1, self.hidden_size))

        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((1, 1))

    def _forward(self, X_seq):
        """
        X_seq shape: (seq_len, input_size, n_samples)
        Returns final output and caches needed for BPTT.

        hₜ = tanh(U·xₜ + V·hₜ₋₁ + bₕ)
        y  = sigmoid(W·h_last + bᵧ)
        """
        n_samples = X_seq.shape[2]
        h = np.zeros((self.hidden_size, n_samples))  # h₀ = zeros

        self.h_cache = [h]  # hidden states at each step
        self.x_cache = []  # inputs  at each step

        for t in range(X_seq.shape[0]):
            x_t = X_seq[t]  # (input_size, n_samples)
            z_t = self.U @ x_t + self.V @ h + self.bh  # pre-activation
            h = np.tanh(z_t)  # new hidden state

            self.h_cache.append(h)
            self.x_cache.append(x_t)

        # Only the final hidden state feeds into the output (classification)
        y_out = self.sigmoid(self.W @ h + self.by)
        self.h_last = h
        return y_out

    def _loss(self, A_out, Y, w):
        """Weighted BCE — identical to your MLP."""
        eps = 1e-9
        loss = -w * (Y * np.log(A_out + eps) + (1 - Y) * np.log(1 - A_out + eps))
        return np.mean(loss)

    def _backward(self, Y, w):
        """
        Backpropagation Through Time (BPTT).
        We unroll the gradients back through each time step,
        same idea as your MLP's backward loop but now through TIME.
        """
        n = Y.shape[1]
        seq_len = len(self.x_cache)

        # ── Output layer gradient (Sigmoid + BCE cancels cleanly, same as your MLP) ──
        dZ_out = (self.sigmoid(self.W @ self.h_last + self.by) - Y) * w

        dW = dZ_out @ self.h_last.T / n
        dby = np.sum(dZ_out, axis=1, keepdims=True) / n

        # Gradient flowing back into the last hidden state
        dh = self.W.T @ dZ_out

        # ── Accumulate gradients for U, V, bh across ALL time steps (BPTT) ──
        dU = np.zeros_like(self.U)
        dV = np.zeros_like(self.V)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(seq_len)):
            h_t = self.h_cache[t + 1]  # hidden state AT step t
            h_prev = self.h_cache[t]  # hidden state BEFORE step t
            x_t = self.x_cache[t]

            # tanh derivative: (1 - tanh²)
            dtanh = (1 - h_t**2) * dh  # (hidden_size, n_samples)

            dU += dtanh @ x_t.T / n
            dV += dtanh @ h_prev.T / n
            dbh += np.sum(dtanh, axis=1, keepdims=True) / n

            # Pass gradient further back in time
            dh = self.V.T @ dtanh

        # ── Apply updates (same as your MLP) ──
        self.W -= self.lr * dW
        self.by -= self.lr * dby
        self.U -= self.lr * dU
        self.V -= self.lr * dV
        self.bh -= self.lr * dbh

    def fit(self, X_seq, Y, class_weight=None):
        """
        X_seq shape: (n_samples, seq_len, input_size)
        Y     shape: (1, n_samples)
        Same interface as your MLP.fit()
        """
        n_samples, seq_len, input_size = X_seq.shape

        # Rearrange to (seq_len, input_size, n_samples) for matrix ops
        X_seq = X_seq.transpose(1, 2, 0)

        # Per-sample class weights — identical to your MLP
        if class_weight is None:
            w = np.ones_like(Y)
        else:
            w = np.where(Y == 1, class_weight[1], class_weight[0])

        self._initialize(input_size)

        for epoch in range(self.epochs):
            A_out = self._forward(X_seq)
            loss = self._loss(A_out, Y, w)
            self.loss_history.append(loss)
            self._backward(Y, w)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                print(
                    f"  -> U Weight Mean: {np.mean(self.U):.6f} | Std: {np.std(self.U):.6f}"
                )
                print(
                    f"  -> V Weight Mean: {np.mean(self.V):.6f} | Std: {np.std(self.V):.6f}"
                )
                print(
                    f"  -> W Weight Mean: {np.mean(self.W):.6f} | Std: {np.std(self.W):.6f}"
                )

    def predict(self, X_seq, threshold=0.5):
        X_seq = X_seq.transpose(1, 2, 0)
        A_out = self._forward(X_seq)
        return (A_out >= threshold).astype(int)

    def predict_proba(self, X_seq):
        X_seq = X_seq.transpose(1, 2, 0)
        return self._forward(X_seq)

    def evaluate(self, Y_true, Y_pred_prob, threshold=0.5):
        """Identical to your MLP.evaluate()."""
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

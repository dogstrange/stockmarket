import numpy as np


class DecisionTreeFromScratch:
    """
    CART-style binary Decision Tree (Gini impurity)
    - continuous features
    - splits: X[:, feature] <= threshold
    - labels: 0/1
    """

    class Node:
        __slots__ = ("feature", "threshold", "left", "right", "value")

        def __init__(
            self, feature=None, threshold=None, left=None, right=None, value=None
        ):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  # leaf class (0/1)

    def __init__(
        self,
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=25,
        max_features=None,  # None | "sqrt" | int
        seed=42,
        max_thresholds=64,  # cap thresholds per feature for speed
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_thresholds = int(max_thresholds)
        self.rng = np.random.default_rng(seed)

        self.root = None
        self.feature_importances_ = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.size
        return 1.0 - np.sum(probs**2)

    @staticmethod
    def _majority_class(y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _choose_features(self, n_features: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)

        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            return self.rng.choice(n_features, size=k, replace=False)

        if isinstance(self.max_features, int):
            k = max(1, min(n_features, int(self.max_features)))
            return self.rng.choice(n_features, size=k, replace=False)

        return np.arange(n_features)

    def _threshold_candidates(self, x: np.ndarray) -> np.ndarray:
        uniq = np.unique(x)
        if uniq.size <= 1:
            return np.array([], dtype=float)

        # If too many unique values, sample thresholds by quantiles
        if uniq.size > self.max_thresholds:
            qs = np.linspace(0.05, 0.95, self.max_thresholds)
            thr = np.quantile(uniq, qs)
            thr = np.unique(thr)
            return thr.astype(float)

        # Otherwise, midpoints between consecutive uniques
        return ((uniq[:-1] + uniq[1:]) / 2.0).astype(float)

    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_idx: np.ndarray):
        n = X.shape[0]
        parent = self._gini(y)
        if parent <= 0.0:
            return None, None, 0.0

        best_feat, best_thr, best_gain = None, None, 0.0

        for j in feat_idx:
            xj = X[:, j]
            thr_candidates = self._threshold_candidates(xj)
            if thr_candidates.size == 0:
                continue

            for thr in thr_candidates:
                left_mask = xj <= thr
                n_left = int(np.sum(left_mask))
                n_right = n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[~left_mask])
                child = (n_left / n) * g_left + (n_right / n) * g_right
                gain = parent - child

                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(j)
                    best_thr = float(thr)

        return best_feat, best_thr, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int):
        n_samples, n_features = X.shape

        # Stop conditions
        if depth >= self.max_depth:
            return self.Node(value=self._majority_class(y))
        if n_samples < self.min_samples_split:
            return self.Node(value=self._majority_class(y))
        if self._gini(y) == 0.0:
            return self.Node(value=self._majority_class(y))

        feat_idx = self._choose_features(n_features)
        feat, thr, gain = self._best_split(X, y, feat_idx)

        if feat is None or gain <= 0.0:
            return self.Node(value=self._majority_class(y))

        # Feature importance (weighted gain)
        self.feature_importances_[feat] += gain * n_samples

        left_mask = X[:, feat] <= thr
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return self.Node(feature=feat, threshold=thr, left=left, right=right)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root = self._build(X, y, depth=0)

        total = float(np.sum(self.feature_importances_))
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_one(self, x: np.ndarray, node: "DecisionTreeFromScratch.Node") -> int:
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return int(node.value)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root) for row in X], dtype=int)

    def _leaf_proba(self, x: np.ndarray, node: "DecisionTreeFromScratch.Node") -> float:
        """
        Traverse to the leaf and return the fraction of class-1 training samples
        that landed there (stored during fit via _build_with_proba).
        Falls back to hard prediction if leaf_proba not available.
        """
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        # If soft probability was stored, use it; otherwise return hard label
        if hasattr(node, "proba"):
            return node.proba
        return float(node.value)

    def predict_proba(self, X):
        """
        Returns soft class-1 probabilities based on leaf class frequencies.
        Shape: (n_samples,) — matches the interface expected by evaluate().
        """
        X = np.asarray(X, dtype=float)
        return np.array([self._leaf_proba(row, self.root) for row in X], dtype=float)

    def evaluate(self, Y_true, Y_pred_prob, threshold=0.5):
        """Identical to RNNFromScratch.evaluate()."""
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

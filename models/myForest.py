import numpy as np
from models.myTree import DecisionTreeFromScratch
from scipy.stats import mode


class RandomForestFromScratch:
    """
    Random Forest (bagging) built from DecisionTreeFromScratch
    - bootstrap sampling
    - max_features per split inside each tree (e.g., "sqrt")
    - labels: 0/1
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=25,
        max_features="sqrt",
        seed=42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features

        self.rng = np.random.default_rng(seed)
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y, verbose=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        n_samples, n_features = X.shape
        self.trees = []
        importances = np.zeros(n_features, dtype=float)

        for i in range(self.n_estimators):
            idx = self.rng.integers(0, n_samples, size=n_samples)  # bootstrap
            Xb = X[idx]
            yb = y[idx]

            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                seed=int(self.rng.integers(0, 1_000_000_000)),
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)

            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

            if verbose and (i + 1) % 20 == 0:
                print(f"Trained {i+1}/{self.n_estimators} trees")

        if len(self.trees) > 0:
            importances /= len(self.trees)
        self.feature_importances_ = importances
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if not self.trees:
            raise RuntimeError("Model not fitted yet")

        # For multi-class, collect all predictions
        all_preds = np.array([t.predict(X) for t in self.trees])  # (n_trees, n_samples)
        return all_preds

    def predict(self, X):
        all_preds = self.predict_proba(X)
        # Majority vote for each sample
        from scipy.stats import mode

        return mode(all_preds, axis=0)[0].astype(int)

    @staticmethod
    def evaluate(y_true, y_prob, threshold=0.5):
        y_true = np.asarray(y_true, dtype=int).reshape(-1)
        y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
        y_pred = (y_prob >= threshold).astype(int)

        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))

        acc = (TP + TN) / max(1, len(y_true))
        precision = TP / max(1e-8, (TP + FP))
        recall = TP / max(1e-8, (TP + FN))
        f1 = 2 * (precision * recall) / max(1e-8, (precision + recall))

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Confusion Matrix: TP={TP}, FP={FP}, FN={FN}, TN={TN}")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        }

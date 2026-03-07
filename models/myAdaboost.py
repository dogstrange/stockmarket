import numpy as np
import matplotlib.pyplot as plt


class DecisionStump:
    """Vectorized decision stump — no Python loops over thresholds"""

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        best_error = float("inf")

        for feature in range(n_features):
            fvals = X[:, feature]

            # Use midpoints between sorted unique values as thresholds
            unique = np.unique(fvals)
            if len(unique) < 2:
                continue
            thresholds = (unique[:-1] + unique[1:]) / 2  # shape: (T,)

            # ── Vectorized over all thresholds at once ──────────────────────
            # left_mask[i, t] = True if sample i <= threshold t
            left_mask = fvals[:, None] <= thresholds[None, :]  # (n, T)
            right_mask = ~left_mask  # (n, T)

            sw = sample_weight  # (n,)

            # Weighted sums for left and right splits
            sw_left = sw @ left_mask.astype(float)  # (T,)
            sw_right = sw @ right_mask.astype(float)  # (T,)

            # Skip degenerate splits
            valid = (sw_left > 0) & (sw_right > 0)
            if not valid.any():
                continue

            # Weighted mean predictions for each threshold
            left_pred = (sw[:, None] * left_mask * y[:, None]).sum(axis=0) / (
                sw_left + 1e-9
            )  # (T,)
            right_pred = (sw[:, None] * right_mask * y[:, None]).sum(axis=0) / (
                sw_right + 1e-9
            )  # (T,)

            # Weighted absolute error for each threshold
            left_err = (
                sw[:, None] * left_mask * np.abs(y[:, None] - left_pred[None, :])
            ).sum(axis=0)
            right_err = (
                sw[:, None] * right_mask * np.abs(y[:, None] - right_pred[None, :])
            ).sum(axis=0)
            errors = left_err + right_err  # (T,)

            errors[~valid] = float("inf")
            best_t = errors.argmin()

            if errors[best_t] < best_error:
                best_error = errors[best_t]
                self.feature_index = feature
                self.threshold = thresholds[best_t]
                self.left_value = left_pred[best_t]
                self.right_value = right_pred[best_t]

    def predict(self, X):
        fvals = X[:, self.feature_index]
        return np.where(fvals <= self.threshold, self.left_value, self.right_value)


class StockAdaBoostRegressor:

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.X_mean = self.X_std = None
        self.y_mean = self.y_std = None

    # ================================================================
    # TRAIN
    # ================================================================
    def fit(self, X, y, val_split=0.2):
        X = np.array(X)
        y = np.array(y).flatten()

        split_idx = int(len(X) * (1 - val_split))
        X_tr_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_tr_raw, y_val_raw = y[:split_idx], y[split_idx:]

        # Normalize
        self.X_mean = X_tr_raw.mean(axis=0)
        self.X_std = X_tr_raw.std(axis=0) + 1e-9
        self.y_mean = y_tr_raw.mean()
        self.y_std = y_tr_raw.std() + 1e-9

        X_tr = (X_tr_raw - self.X_mean) / self.X_std
        y_tr = (y_tr_raw - self.y_mean) / self.y_std
        X_val = (X_val_raw - self.X_mean) / self.X_std
        y_val = (y_val_raw - self.y_mean) / self.y_std

        n = len(X_tr)
        sample_weight = np.ones(n) / n

        self.estimators = []
        self.estimator_weights = []
        best_val_loss = float("inf")
        best_estimators = []
        best_weights = []

        for i in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X_tr, y_tr, sample_weight)
            y_pred = stump.predict(X_tr)

            # Normalize errors to [0, 1] — required for valid beta
            raw_errors = np.abs(y_tr - y_pred)
            errors = raw_errors / (raw_errors.max() + 1e-9)

            # Weighted error
            error = float(np.dot(sample_weight, errors))
            error = np.clip(error, 1e-10, 1 - 1e-10)

            beta = error / (1 - error)
            estimator_weight = self.learning_rate * np.log(1 / beta)

            # Update sample weights (AdaBoost.R2)
            sample_weight *= np.power(beta, 1 - errors)
            sample_weight /= sample_weight.sum()

            self.estimators.append(stump)
            self.estimator_weights.append(estimator_weight)

            # Early-stop checkpoint
            val_pred = self._predict_normalized(X_val)
            val_loss = float(np.mean(np.abs(val_pred - y_val)))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_estimators = list(self.estimators)
                best_weights = list(self.estimator_weights)

            if (i + 1) % 10 == 0:
                print(
                    f"  Estimator {i+1:3d}/{self.n_estimators} | val MAE: {val_loss:.6f}"
                )

        self.estimators = best_estimators
        self.estimator_weights = best_weights
        return best_val_loss

    # ================================================================
    # PREDICT (weighted median — correct AdaBoost.R2)
    # ================================================================
    def _predict_normalized(self, X):
        """Weighted median across all weak learners (AdaBoost.R2 spec)"""
        predictions = np.array(
            [e.predict(X) for e in self.estimators]
        )  # (n_est, n_samples)
        weights = np.array(self.estimator_weights)
        weights = weights / (weights.sum() + 1e-9)

        # Sort each sample's predictions by value, carry weights along
        sorted_idx = np.argsort(predictions, axis=0)  # (n_est, n_samples)
        sorted_preds = np.take_along_axis(
            predictions, sorted_idx, axis=0
        )  # (n_est, n_samples)
        sorted_w = weights[sorted_idx]  # (n_est, n_samples)

        cumulative = np.cumsum(sorted_w, axis=0)  # (n_est, n_samples)
        median_idx = np.argmax(cumulative >= 0.5, axis=0)  # (n_samples,)
        return sorted_preds[median_idx, np.arange(X.shape[0])]

    def predict(self, X):
        X = np.array(X)
        Xn = (X - self.X_mean) / self.X_std
        return self._predict_normalized(Xn) * self.y_std + self.y_mean

    # ================================================================
    # EVALUATE
    # ================================================================
    def evaluate_and_plot(self, X_test, y_test, n_days):
        X = X_test[:n_days]
        y = y_test[:n_days].flatten()
        preds = self.predict(X).flatten()

        mae = np.mean(np.abs(y - preds))
        rmse = np.sqrt(np.mean((y - preds) ** 2))
        rse = np.sum((y - preds) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-9)
        r2 = 1 - rse
        mape = np.mean(np.abs((y - preds) / (y + 1e-9))) * 100
        dir_acc = np.mean(np.sign(y) == np.sign(preds)) * 100
        baseline_mae = np.mean(np.abs(y))

        print("\n========== FINAL SCORES ==========")
        print(f"MAE      : {mae:.6f}")
        print(f"RMSE     : {rmse:.6f}")
        print(f"R²       : {r2:.6f}")
        print(f"MAPE %   : {mape:.2f}")
        print(f"DIR ACC% : {dir_acc:.2f}")
        print(f"BASE MAE : {baseline_mae:.6f}")
        print(f"\n========== MODEL INFO ==========")
        print(f"Estimators used : {len(self.estimators)}")
        print(f"Learning rate   : {self.learning_rate}")

        plt.figure(figsize=(13, 5))
        plt.scatter(range(len(y)), y, color="royalblue", alpha=0.4, label="Real")
        plt.plot(preds, color="darkorange", linewidth=2, label="Prediction")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Stock Return Prediction — AdaBoost")
        plt.legend()
        plt.tight_layout()
        plt.show()

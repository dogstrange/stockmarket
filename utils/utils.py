import pandas as pd
import os
import glob
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Numpy 2.x compatibility patch ──
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# ==========================================
# CONFIGURATION
# ==========================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_FOLDER = os.path.join(_THIS_DIR, "..", "data", "project_funds")

FILE_NAME = "claude_toy_stock.csv"
FILE_PATH = os.path.join(CLEANED_FOLDER, FILE_NAME)


# ==========================================
# TRIPLE BARRIER LABELING
# ==========================================
def apply_triple_barrier_label(df, profit_target=0.02, stop_loss=-0.02, window=5):
    """
    Labels data based on what happens FIRST within the next 'window' days.
      +1 -> hit profit target first
      -1 -> hit stop loss first
       0 -> neither (time expired)
    """
    if "close" not in df.columns:
        return np.array([])

    close_prices = df["close"].values
    n = len(close_prices)
    labels = np.zeros(n)

    for i in range(n - window):
        current_price = close_prices[i]
        future_window = close_prices[i + 1 : i + 1 + window]
        pct_changes = (future_window - current_price) / current_price

        profit_hits = np.where(pct_changes >= profit_target)[0]
        loss_hits = np.where(pct_changes <= stop_loss)[0]

        if len(profit_hits) > 0 and (
            len(loss_hits) == 0 or profit_hits[0] < loss_hits[0]
        ):
            labels[i] = 1
        elif len(loss_hits) > 0 and (
            len(profit_hits) == 0 or loss_hits[0] < profit_hits[0]
        ):
            labels[i] = -1

    return labels


# ==========================================
# SINGLE STOCK LOADER
# ==========================================
def load_data():
    print(f"---  Loading: {FILE_NAME} ---")

    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at:\n   {FILE_PATH}")
        print("\n(Did you run the cleaning script first?)")
        return None

    try:
        df = pd.read_csv(FILE_PATH)
        print(df.head())

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        print(f"Success! Loaded {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error reading file: {e}")
        return None


# ==========================================
# PERIOD FILTER
# ==========================================
def get_period(df, start_date, end_date):
    """
    Trims a dataframe to a specific period.

    Parameters:
        df         : your dataframe (must have datetime index)
        start_date : string e.g. "2000-01-01"
        end_date   : string e.g. "2000-12-31"

    Returns:
        trimmed dataframe
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df_trimmed = df.loc[start_date:end_date].copy()
    print(f"Period: {start_date} to {end_date}")
    print(f"Rows: {len(df_trimmed)}")
    return df_trimmed


# ==========================================
# BASIC ML FEATURES
# ==========================================
def add_ml_features(df):
    """
    Adds technical indicators and the Triple Barrier Label.
    """
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = df["close"].rolling(window=20).std()

    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["Volume_SMA_20"] = df["volume"].rolling(window=20).mean()
    df["Volume_Spike"] = np.where(df["volume"] > (df["Volume_SMA_20"] * 1.5), 1, 0)

    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    df["ROC_10"] = df["close"].pct_change(periods=10) * 100
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["target"] = apply_triple_barrier_label(
        df, profit_target=0.02, stop_loss=-0.02, window=5
    )
    df["target"] = df["target"].astype(int)

    return df


# ==========================================
# ADVANCED ML FEATURES
# ==========================================
def add_ml_features_advanced(df):
    df = df.copy()

    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_Ratio"] = df["ATR"] / df["close"]

    sma_20 = df["close"].rolling(window=20).mean()
    std_20 = df["close"].rolling(window=20).std()
    df["BB_Width"] = (4 * std_20) / sma_20
    df["RSI"] = ta.rsi(df["close"], length=14) / 100.0

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = adx_df.iloc[:, 0] / 100.0
    df["Dist_SMA_20"] = (df["close"] - sma_20) / sma_20
    df["MFI"] = (
        ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14) / 100.0
    )
    df["Range_Ratio"] = (df["high"] - df["low"]) / df["close"]
    df["Log_Ret_5"] = np.log(df["close"] / df["close"].shift(5))

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Rows after cleaning: {len(df)}")
    print(f"NaN remaining: {df.isna().sum().sum()}")
    return df


# ==========================================
# EXPANDED FEATURE ENGINEERING
# Used internally by load_multiple_stocks()
# ==========================================
def add_features(df):
    """
    Full feature set for multi-stock LSTM training.
    All indicators are scale-invariant (ratio/normalized) so features
    are comparable across different-priced stocks.
    """
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Volatility
    df["ATR"] = ta.atr(h, l, c, length=14)
    df["ATR_Ratio"] = df["ATR"] / (c + 1e-9)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_Width"] = (4 * std20) / (sma20 + 1e-9)
    df["BB_Pos"] = (c - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    # Momentum
    df["RSI_14"] = ta.rsi(c, length=14) / 100.0
    df["RSI_7"] = ta.rsi(c, length=7) / 100.0
    macd_df = ta.macd(c)
    df["MACD_hist"] = macd_df.iloc[:, 2] if macd_df is not None else 0
    df["ROC_5"] = c.pct_change(5) * 100
    df["ROC_10"] = c.pct_change(10) * 100
    df["MOM_10"] = c.diff(10)
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    df["STOCH_K"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # Trend
    adx_df = ta.adx(h, l, c, length=14)
    df["ADX"] = adx_df.iloc[:, 0] / 100.0 if adx_df is not None else 0
    df["AROON_osc"] = ta.aroon(h, l, length=14).iloc[:, 2] / 100.0

    # Volume
    df["MFI"] = ta.mfi(h, l, c, v, length=14) / 100.0
    vol_sma20 = v.rolling(20).mean()
    df["Vol_ratio"] = v / (vol_sma20 + 1e-9)
    df["OBV"] = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["OBV_ratio"] = df["OBV"] / (df["OBV"].rolling(20).mean() + 1e-9)

    # Price structure
    df["Log_Ret"] = np.log(c / c.shift(1))
    df["Log_Ret_3"] = np.log(c / c.shift(3))
    df["Log_Ret_5"] = np.log(c / c.shift(5))
    df["HL_Range"] = (h - l) / (c + 1e-9)
    df["Gap"] = (df["open"] - c.shift(1)) / (c.shift(1) + 1e-9)

    # MA regime signals (ratio form — scale invariant)
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    sma200 = c.rolling(200).mean()
    df["EMA9_21"] = ema9 / (ema21 + 1e-9) - 1
    df["EMA21_50"] = ema21 / (ema50 + 1e-9) - 1
    df["Price_SMA200"] = c / (sma200 + 1e-9) - 1
    df["Dist_SMA20"] = (c - sma20) / (sma20 + 1e-9)

    # Calendar
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df["DayOfWeek"] = df.index.dayofweek / 4.0
    df["Month"] = df.index.month / 12.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ==========================================
# MULTI-STOCK LOADER
# ==========================================
def load_multiple_stocks(
    folder=CLEANED_FOLDER,
    min_rows=500,
):
    all_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not all_files:
        print(f"No CSV files found in: {folder}")
        return None, {}

    print(f"Scanning: {folder}")
    print(f"Discovered {len(all_files)} CSV files\n")

    all_dfs, skipped = [], []
    ticker_map = {}

    for ticker_id, path in enumerate(all_files):
        ticker = os.path.basename(path).replace(".csv", "")

        try:
            df = pd.read_csv(path)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                skipped.append((ticker, "missing OHLCV columns"))
                continue

            if len(df) < min_rows:
                skipped.append((ticker, f"only {len(df)} rows before features"))
                continue

            df = add_features(df)

            if len(df) < min_rows // 2:
                skipped.append(
                    (ticker, f"only {len(df)} rows after feature engineering")
                )
                continue

            df["ticker"] = ticker
            df["ticker_id"] = ticker_id
            ticker_map[ticker_id] = ticker

            all_dfs.append(df)
            print(f"  ✓ [{ticker_id:>2}]  {ticker:<20}  {len(df):,} rows")

        except Exception as e:
            skipped.append((ticker, str(e)))

    if skipped:
        print(f"\nSkipped {len(skipped)} files:")
        for t, reason in skipped:
            print(f"  ✗ {t:<20}  {reason}")

    if not all_dfs:
        print("\nNo stocks loaded successfully.")
        return None, {}

    combined = pd.concat(all_dfs, axis=0).sort_index()

    print(f"\n{'─'*45}")
    print(f"Combined dataset : {len(combined):,} rows from {len(all_dfs)} stocks")
    print(f"Features         : {combined.shape[1]} columns")
    print(
        f"Date range       : {combined.index.min().date()} → {combined.index.max().date()}"
    )
    print(f"{'─'*45}\n")
    print("Ticker map:")
    for tid, name in ticker_map.items():
        print(f"  {tid:>2} → {name}")

    return combined, ticker_map


# ==========================================
# VISUALIZATION
# ==========================================
def visualize_classification(model, Y_test, X_test, title="Model"):
    """
    Plots a 3-panel classification summary:
      1. Training loss curve
      2. Confusion matrix on the test set
      3. Accuracy vs error bar chart

    Parameters
    ----------
    model  : trained model — must have .predict(), .loss_history
    Y_test : ground truth labels, shape (1, n) or (n,) — values 0 or 1
    X_test : test features, shape (n, features)
    title  : string label shown in each plot title e.g. "MLP" or "RNN"
    """
    Y_pred = model.predict(X_test)
    predictions = Y_pred.ravel()
    actual = Y_test.ravel().astype(int)

    cm = confusion_matrix(actual, predictions)
    accuracy = np.mean(predictions == actual)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title} — Classification Results", fontsize=16, fontweight="bold")

    # 1. Loss curve
    axes[0].plot(model.loss_history, color="royalblue", lw=2)
    axes[0].set_title("Training Loss Over Time", fontsize=14)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # 2. Confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sell", "Buy"],
        yticklabels=["Sell", "Buy"],
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title("Confusion Matrix (Test Set)", fontsize=14)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    # 3. Accuracy vs Error
    axes[2].bar(
        ["Accuracy", "Error"],
        [accuracy, 1 - accuracy],
        color=["#2ecc71", "#e74c3c"],
    )
    axes[2].set_title(f"Overall Accuracy: {accuracy:.2%}", fontsize=14)
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print(f"\n{'─'*35}")
    print(f"  Accuracy : {accuracy:.2%}")
    print(f"  Error    : {1 - accuracy:.2%}")
    print(f"{'─'*35}")


def apply_pca(X_train, X_test, n_components=None, variance_threshold=0.95, plot=True):
    """
    Fits PCA on X_train and transforms both X_train and X_test.
    Automatically picks the number of components needed to explain
    variance_threshold of variance (e.g. 0.95 = 95%) unless
    n_components is set explicitly.

    Parameters
    ----------
    X_train            : np.ndarray, shape (n_samples, n_features)
                         or (n_samples, seq_len, n_features) for RNN sequences
    X_test             : np.ndarray, same shape as X_train
    n_components       : int or None — if set, overrides variance_threshold
    variance_threshold : float — target explained variance (default 0.95)
    plot               : bool — plot explained variance curve (default True)

    Returns
    -------
    X_train_pca : transformed training data
    X_test_pca  : transformed test data
    pca         : fitted PCA object (keep this to transform future data)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # ── Handle RNN sequences (3D) by flattening to 2D ────────────────
    is_3d = X_train.ndim == 3
    if is_3d:
        n_train, seq_len, n_features = X_train.shape
        n_test = X_test.shape[0]
        X_train_2d = X_train.reshape(n_train, seq_len * n_features)
        X_test_2d = X_test.reshape(n_test, seq_len * n_features)
    else:
        X_train_2d = X_train
        X_test_2d = X_test

    # ── Standardize before PCA ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    # ── Determine n_components ────────────────────────────────────────
    if n_components is None:
        pca_full = PCA().fit(X_train_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        print(
            f"PCA: {n_components} components explain {cumvar[n_components-1]:.2%} variance"
        )
    else:
        pca_full = PCA().fit(X_train_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        print(
            f"PCA: using {n_components} components → {cumvar[n_components-1]:.2%} variance explained"
        )

    # ── Fit final PCA ─────────────────────────────────────────────────
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # ── Plot explained variance ───────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("PCA — Explained Variance", fontsize=14, fontweight="bold")

        # Individual variance per component
        axes[0].bar(
            range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_,
            color="steelblue",
            alpha=0.7,
        )
        axes[0].axvline(
            n_components, color="red", linestyle="--", label=f"Selected: {n_components}"
        )
        axes[0].set_title("Variance per Component")
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio")
        axes[0].legend()

        # Cumulative variance
        axes[1].plot(
            range(1, len(cumvar) + 1),
            cumvar,
            marker="o",
            color="steelblue",
            markersize=3,
        )
        axes[1].axhline(
            variance_threshold,
            color="orange",
            linestyle="--",
            label=f"Threshold: {variance_threshold:.0%}",
        )
        axes[1].axvline(
            n_components, color="red", linestyle="--", label=f"Selected: {n_components}"
        )
        axes[1].set_title("Cumulative Explained Variance")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Variance")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    print(f"X_train: {X_train_2d.shape} → {X_train_pca.shape}")
    print(f"X_test : {X_test_2d.shape}  → {X_test_pca.shape}")

    return X_train_pca, X_test_pca, pca

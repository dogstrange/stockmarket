import pandas as pd
import os
import glob
import numpy as np
import pandas_ta as ta

# ── Numpy 2.x compatibility patch ──
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# ==========================================
# CONFIGURATION
# ==========================================

CLEANED_FOLDER = os.path.join("..", "data", "project_funds")


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
    """
    Auto-discovers and loads ALL .csv files in the project_funds folder.
    Engineers features for each stock and stacks them into one combined
    DataFrame ready for LSTM training.

    ticker_id is assigned automatically based on alphabetical order of
    discovered files — no hardcoded list needed.

    Parameters
    ----------
    folder   : path to project_funds folder containing *.csv files
    min_rows : skip stocks with fewer rows than this after feature engineering

    Returns
    -------
    combined  : pd.DataFrame  — all stocks stacked, sorted by date
                columns include all features + "ticker" + "ticker_id"
    ticker_map: dict          — maps ticker_id (int) → ticker name (str)
    """
    # ── Discover all CSV files in the folder ──────────────────────────
    all_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not all_files:
        print(f"No CSV files found in: {folder}")
        return None, {}

    print(f"Scanning: {folder}")
    print(f"Discovered {len(all_files)} CSV files\n")

    all_dfs, skipped = [], []
    ticker_map = {}  # ticker_id → ticker name

    for ticker_id, path in enumerate(all_files):
        # Derive a clean ticker name from the filename
        ticker = os.path.basename(path).replace(".csv", "")

        try:
            df = pd.read_csv(path)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # Validate required OHLCV columns
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                skipped.append((ticker, "missing OHLCV columns"))
                continue

            if len(df) < min_rows:
                skipped.append((ticker, f"only {len(df)} rows before features"))
                continue

            # Engineer features
            df = add_features(df)

            if len(df) < min_rows // 2:
                skipped.append(
                    (ticker, f"only {len(df)} rows after feature engineering")
                )
                continue

            # Tag with identity columns
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

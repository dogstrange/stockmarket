import pandas as pd
import os
import glob
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FOLDER = os.path.join("..", "data", "stock", "Stocks")
OUTPUT_FOLDER = os.path.join("..", "data", "cleaned")


# ==========================================
# 2. LABELING LOGIC (Triple Barrier)
# ==========================================
def apply_triple_barrier_label(df, profit_target=0.02, stop_loss=-0.02, window=5):
    """
    Labels data based on what happens FIRST within the next 'window' days.
    """
    # Convert to numpy for high-speed processing
    if "close" not in df.columns:
        return np.array([])

    close_prices = df["close"].values
    n = len(close_prices)
    labels = np.zeros(n)

    for i in range(n - window):
        current_price = close_prices[i]
        future_window = close_prices[i + 1 : i + 1 + window]

        # Calculate percentage change
        pct_changes = (future_window - current_price) / current_price

        profit_hits = np.where(pct_changes >= profit_target)[0]
        loss_hits = np.where(pct_changes <= stop_loss)[0]

        if len(profit_hits) > 0 and (
            len(loss_hits) == 0 or profit_hits[0] < loss_hits[0]
        ):
            labels[i] = 1  # Hit profit first
        elif len(loss_hits) > 0 and (
            len(profit_hits) == 0 or loss_hits[0] < profit_hits[0]
        ):
            labels[i] = -1  # Hit stop loss first

    return labels


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def add_ml_features(df):
    """
    Adds technical indicators and the Triple Barrier Label.
    """
    # --- BASIC FEATURES ---
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = df["close"].rolling(window=20).std()

    # --- VOLUME INDICATORS ---
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["Volume_SMA_20"] = df["volume"].rolling(window=20).mean()
    df["Volume_Spike"] = np.where(df["volume"] > (df["Volume_SMA_20"] * 1.5), 1, 0)

    # --- MOMENTUM & OSCILLATORS ---
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    df["ROC_10"] = df["close"].pct_change(periods=10) * 100
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- THE TRIPLE BARRIER TARGET (LABEL) ---
    # We apply this BEFORE dropping NaNs to ensure we have the full window
    df["target"] = apply_triple_barrier_label(
        df, profit_target=0.05, stop_loss=-0.05, window=5
    )
    df["target"] = df["target"].astype(int)

    return df


# ==========================================
# 4. PROCESSING LOGIC
# ==========================================
def process_file(file_path):
    file_name = os.path.basename(file_path)
    print(f"   Processing: {file_name}...", end=" ")

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")

        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().drop_duplicates()

        if len(df) < 30:  # Check if enough data exists for indicators
            print("⚠️ Skipped (Insufficient rows)")
            return

        # Add Features AND Labels
        df = add_ml_features(df)

        # Drop rows that have NaN from rolling windows OR
        # rows at the very end where the label couldn't be calculated (the last 'window' days)
        # We drop target 0 at the very end only if you want a "pure" buy/sell dataset,
        # but usually, keeping 0 is better for the model to learn "no action".
        df = df.dropna()

        clean_name = file_name.replace(".txt", "").replace(".csv", "") + "_cleaned.csv"
        save_path = os.path.join(OUTPUT_FOLDER, clean_name)

        df.to_csv(save_path, index=False)
        print(f"✅ Saved ({len(df)} rows)")

    except Exception as e:
        print(f"❌ Error: {e}")


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files_txt = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    files_csv = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    all_files = files_txt + files_csv

    print(f"📂 Found {len(all_files)} files. Starting processing...\n")

    for file_path in all_files:
        process_file(file_path)

    print("\n--- 🎉 Batch Processing Complete ---")

import csv
import os
import numpy as np
from datetime import datetime, timedelta


def generate_triple_barrier_csv(
    filename, days=5000, profit_target=0.02, stop_loss=-0.02, window=5, seed=42
):
    rng = np.random.default_rng(seed)
    start_date = datetime(2024, 1, 1)

    daily_up = (profit_target / window) * 1.5
    daily_down = (abs(stop_loss) / window) * 1.5

    phase_len = window
    cycle_len = 3 * phase_len

    prices = np.zeros(days)
    prices[0] = 100.0

    for i in range(1, days):
        phase = (i % cycle_len) // phase_len

        if phase == 0:
            drift = daily_up
            noise = rng.normal(0, drift * 0.15)
        elif phase == 1:
            drift = -daily_down
            noise = rng.normal(0, daily_down * 0.15)
        else:
            max_sideways = min(abs(profit_target), abs(stop_loss)) * 0.25
            drift = rng.uniform(-max_sideways / window, max_sideways / window)
            noise = rng.normal(0, max_sideways * 0.10)

        prices[i] = prices[i - 1] * (1 + drift + noise)

        if prices[i] < 50 or prices[i] > 300:
            prices[i] = 100.0

    output_dir = os.path.join("..", "data", "cleaned")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])

        for i in range(days):
            current_date = start_date + timedelta(days=i)
            close = prices[i]
            open_price = close * (1 + rng.normal(0, 0.002))
            high_price = max(open_price, close) * (1 + abs(rng.normal(0, 0.003)))
            low_price = min(open_price, close) * (1 - abs(rng.normal(0, 0.003)))

            writer.writerow(
                [
                    current_date.strftime("%Y-%m-%d"),
                    f"{open_price:.2f}",
                    f"{high_price:.2f}",
                    f"{low_price:.2f}",
                    f"{close:.2f}",
                    int(10_000 + (i % cycle_len) * 300 + rng.integers(0, 1000)),
                ]
            )

    return filepath


if __name__ == "__main__":
    saved_path = generate_triple_barrier_csv("claude_toy_stock.csv", days=5000)
    print(f"Created 5000 rows successfully at: {saved_path}")

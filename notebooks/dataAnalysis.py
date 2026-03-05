import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils

df = utils.load_data()

print("\nCleaned data succesfully loaded")

df = utils.add_ml_features(df)
df = utils.add_ml_features_advanced(df)
print(df.head())
print(df.tail())
print(df["target"].value_counts())
# %% Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

corr = df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, cmap="mako_r", annot=True)
plt.show()


# %% Volume SMA 20 to target
X = df["Volume_SMA_20"].to_numpy()
Y = df["target"].to_numpy()
plt.scatter(X, Y)
plt.show()

# %% 1 year period return graph

start_date = "2000-01-01"
end_date = "2000-12-31"

df_year = df.loc[start_date:end_date].copy()


# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Close price
axes[0].plot(df_year.index, df_year["close"], color="blue", linewidth=1.5)
axes[0].set_title("Close Price (1985)")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Price")
axes[0].grid(True, alpha=0.3)

# Plot 2: Daily return
axes[1].bar(
    df_year.index,
    df_year["daily_return"],
    color=["green" if r > 0 else "red" for r in df_year["daily_return"]],
    width=1,
)
axes[1].axhline(y=0, color="black", linewidth=0.8)
axes[1].set_title("Daily Return % (1985)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Return (%)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%

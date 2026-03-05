import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import utils
from models import mySVM


# %% Data loading
df = utils.load_data()
df = utils.add_ml_features(df)
df = utils.add_ml_features_advanced(df)

print("\nCleaned data succesfully loaded")
print(df.head(5))
print(df["target"].value_counts())


# %% Initial visualization
import matplotlib.pyplot as plt

daily_return = df["daily_return"].values
volatility_20 = df["volatility_20"].values
sma_20 = df["SMA_20"]
volume = df["volume"]
labels = df["target"].values

# Create the scatter plot
plt.figure(figsize=(10, 6))

# Plot points with label 1 in green
mask_positive = labels == 1
plt.scatter(
    volatility_20[mask_positive],
    daily_return[mask_positive],
    c="green",
    label="Label 1",
    alpha=0.6,
    s=100,
)

# Plot points with label -1 in blue
mask_negative = labels == -1
plt.scatter(
    volatility_20[mask_negative],
    daily_return[mask_negative],
    c="blue",
    label="Label -1",
    alpha=0.6,
    s=100,
)

plt.xlabel("Volatility_20", fontsize=12)
plt.ylabel("Daily Return", fontsize=12)
plt.title("Cluster Visualization: Volatility vs Daily Return", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% SVM Training
X = np.column_stack((volatility_20, daily_return, volume, sma_20))
Y = labels
svm_linear = mySVM.SVMSoftmargin(alpha=0.001, iteration=10000, lambda_=0.01)
w_l, b_l = svm_linear.fit(X, Y)
predictions = svm_linear.predict(X)

print(f"Weights: {w_l}")
print(f"Bias: {b_l}")
print(f"Accuracy: {np.mean(predictions == Y):.2%}")

# %% SVM Dual train
X = np.column_stack((volatility_20, daily_return, volume, sma_20))
Y = labels

# Initialize the dual SVM with your preferred kernel
svm_dual = mySVM.SVM_Dual(
    kernel="rbf",  # or 'poly' for polynomial kernel
    degree=2,  # only used if kernel='poly'
    sigma=1.0,  # bandwidth for RBF kernel (tune this!)
    epoches=1000,  # number of optimization iterations
    learning_rate=0.001,  # step size for gradient ascent
)
# Train the model
svm_dual.train(X, Y)
# Make predictions
predictions = svm_dual.predict(X)
# Evaluate
accuracy = svm_dual.score(X, Y)
print(f"Accuracy: {accuracy:.2%}")
# %% Find support vectors

# Calculate margins for all samples
margins = Y * (np.dot(X, w_l) - b_l)
# Typically use a small tolerance (e.g., 1e-5) for numerical stability
tolerance = 1e-3
# Support vectors are samples with margin <= 1 + tolerance
support_vector_indices = np.where(
    (margins <= 1 + tolerance) & (margins > 0 + tolerance)
)[0]
print(f"Number of support vectors: {len(support_vector_indices)}")
print(f"Support vectors: {X[support_vector_indices]}")

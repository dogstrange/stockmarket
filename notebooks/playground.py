import numpy as np

# %%
X = np.random.randint(100, size=5)
print(X)  # %%

# %%
X_1 = np.random.randn(10, 1)
print(X_1.shape)
print(X_1)
# %%
X_2 = np.random.randn(1, 10)
print(X_2)
print(X_1 * X_2)
# %%

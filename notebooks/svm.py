import pandas as pd
import os

# %% Data loading
# ==========================================
# CONFIGURATION
# ==========================================
# 1. Point to your cleaned folder (relative to 'notebooks')
CLEANED_FOLDER = os.path.join('..', 'data', 'cleaned')

# 2. Pick the specific file you want to load
# (Make sure this file actually exists in that folder!)
FILE_NAME = 'a.us_cleaned.csv' 

FILE_PATH = os.path.join(CLEANED_FOLDER, FILE_NAME)

# ==========================================
# LOAD SCRIPT
# ==========================================
def load_data():
    print(f"---  Loading: {FILE_NAME} ---")

    # A. Check if file exists
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at:\n   {FILE_PATH}")
        print("\n(Did you run the cleaning script first?)")
        return None

    # B. Load the CSV
    try:
        df = pd.read_csv(FILE_PATH)
        
        # C. Ensure 'date' is still a datetime object 
        # (CSV saves dates as strings, so we must convert them back)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # Set date as the index (Best practice for Time Series/Stock data)
            df = df.set_index('date')

        print(f"success! Loaded {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# ==========================================
# EXECUTION
# ==========================================
df = load_data()
print(df.head())

print("\nCleaned data succesfully loaded")

# %% Adding label

import numpy as np

def apply_triple_barrier_label(df, profit_target=0.05, stop_loss=-0.05, window=5):
    """
    Labels data based on what happens FIRST within the next 'window' days:
    1  => Price hits +5% (Profit)
    -1 => Price hits -5% (Stop Loss)
    0  => Neither happens
    """
    # Convert to numpy for high-speed processing
    close_prices = df['close'].values
    n = len(close_prices)
    labels = np.zeros(n)  # Initialize all labels as 0

    # Loop through data (stopping 'window' days before the end)
    for i in range(n - window):
        current_price = close_prices[i]
        
        # Look at the next 5 days
        # (i+1 is tomorrow, i+1+window is the end of the 5-day window)
        future_window = close_prices[i+1 : i+1+window]
        
        # Calculate percentage change for these 5 days relative to today
        pct_changes = (future_window - current_price) / current_price
        
        # Find indices where barriers were hit
        # np.where returns a tuple, so we take [0] to get the array of indices
        profit_hits = np.where(pct_changes >= profit_target)[0]
        loss_hits = np.where(pct_changes <= stop_loss)[0]
        
        # LOGIC: Check who crossed the line FIRST
        if len(profit_hits) > 0 and len(loss_hits) == 0:
            labels[i] = 1  # Only hit profit
            
        elif len(loss_hits) > 0 and len(profit_hits) == 0:
            labels[i] = -1 # Only hit stop loss
            
        elif len(profit_hits) > 0 and len(loss_hits) > 0:
            # Both hit? Check which index is smaller (happened sooner)
            if profit_hits[0] < loss_hits[0]:
                labels[i] = 1
            else:
                labels[i] = -1

    return labels

# Apply the function
print("Generating labels... (Triple Barrier Method)")
df['target'] = apply_triple_barrier_label(df)

# --- VERIFICATION ---
# Convert -1, 0, 1 to integers for cleaner viewing
df['target'] = df['target'].astype(int)

# Show distribution (How many Buys vs Sells?)
print("\nLabel Distribution:")
print(df['target'].value_counts())



# %% SVM Model Initiation
class SVMSoftmargin:
    def __init__(self, alpha = 0.001, iteration = 100000, lambda_ = 0.001):
        self.alpha = alpha
        self.iteration = iteration
        self.lambda_ = lambda_
        self.w = None
        self.b = None
    
    def fit(self, X, Y):
        n_samples, n_features = X.shape #extract X's number of row and number of column
        self.w = np.zeros(n_features)

        self.b = 0

        for iterate in range(self.iteration):   
            for i, Xi in enumerate(X):
                score = Y[i] * (np.dot(self.w, Xi) - self.b)
                if score >= 1:
                    g_w = (2 * self.lambda_ * self.w) / n_samples

                    self.w -= self.alpha * g_w
                else:
                    g_w = (2 * self.lambda_ * self.w - (Y[i] * Xi)) / n_samples
                    g_b = Y[i]

                    self.w -= self.alpha * g_w
                    self.b -= self.alpha * g_b
        return self.w, self.b
    
    def predict(self, X):
        pred = np.dot(X, self.w) - self.b
        result = [1 if val > 0 else -1 for val in pred]
        return result
    

class SVM_Dual:

    def __init__(self, kernel='poly', degree=2, sigma=0.1, epoches=1000, learning_rate= 0.001):
        self.alpha = None
        self.b = 0
        self.degree = degree
        self.c = 1
        self.C = 1
        self.sigma = sigma
        self.epoches = epoches
        self.learning_rate = learning_rate

        if kernel == 'poly':
            self.kernel = self.polynomial_kernal # for polynomial kernal
        elif kernel == 'rbf':
            self.kernel =  self.gaussian_kernal # for guassian

    def polynomial_kernal(self,X,Z):
        return (self.c + X.dot(Z.T))**self.degree #(c + X.y)^degree
        
    def gaussian_kernal(self, X,Z):
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X[:, np.newaxis] - Z[np.newaxis, :], axis=2) ** 2) #e ^-(1/ σ2) ||X-y|| ^2
    
    def train(self,X,y):
        self.X = X
        self.y = y
        self.alpha = np.random.random(X.shape[0])
        self.b = 0
        self.ones = np.ones(X.shape[0]) 

        y_mul_kernal = np.outer(y, y) * self.kernel(X, X) # yi yj K(xi, xj)

        for i in range(self.epoches):
            gradient = self.ones - y_mul_kernal.dot(self.alpha) # 1 – yk ∑ αj yj K(xj, xk)

            self.alpha += self.learning_rate * gradient # α = α + η*(1 – yk ∑ αj yj K(xj, xk)) to maximize
            self.alpha[self.alpha > self.C] = self.C # 0<α<C
            self.alpha[self.alpha < 0] = 0 # 0<α<C

            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_mul_kernal) # ∑αi – (1/2) ∑i ∑j αi αj yi yj K(xi, xj)
            
        alpha_index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]
        
        # for intercept b, we will only consider α which are 0<α<C 
        b_list = []        
        for index in alpha_index:
            b_list.append(y[index] - (self.alpha * y).dot(self.kernel(X, X[index])))

        self.b = np.mean(b_list) # avgC≤αi≤0{ yi – ∑αjyj K(xj, xi) }
            
    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
    
    def decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b



# %% Initial visualization
import matplotlib.pyplot as plt

daily_return = df['daily_return'].values
volatility_20 = df['volatility_20'].values
sma_20 = df['SMA_20']
volume = df['volume']
labels = df['target'].values

# Create the scatter plot
plt.figure(figsize=(10, 6))

# Plot points with label 1 in green
mask_positive = labels == 1
plt.scatter(volatility_20[mask_positive], daily_return[mask_positive], 
            c='green', label='Label 1', alpha=0.6, s=100)

# Plot points with label -1 in blue
mask_negative = labels == -1
plt.scatter(volatility_20[mask_negative], daily_return[mask_negative], 
            c='blue', label='Label -1', alpha=0.6, s=100)

plt.xlabel('Volatility_20', fontsize=12)
plt.ylabel('Daily Return', fontsize=12)
plt.title('Cluster Visualization: Volatility vs Daily Return', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% SVM Training
X = np.column_stack((volatility_20, daily_return,volume,sma_20))
Y = labels
svm_linear = SVMSoftmargin(alpha=0.001, iteration=10000, lambda_=0.01)
w_l, b_l = svm_linear.fit(X, Y)
predictions = svm_linear.predict(X)

print(f"Weights: {w_l}")
print(f"Bias: {b_l}")
print(f"Accuracy: {np.mean(predictions == Y):.2%}")

# %% SVM Dual train
X = np.column_stack((volatility_20, daily_return, volume, sma_20))
Y = labels

# Initialize the dual SVM with your preferred kernel
svm_dual = SVM_Dual(
    kernel='rbf',           # or 'poly' for polynomial kernel
    degree=2,               # only used if kernel='poly'
    sigma=1.0,              # bandwidth for RBF kernel (tune this!)
    epoches=1000,           # number of optimization iterations
    learning_rate=0.001     # step size for gradient ascent
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
support_vector_indices = np.where((margins <= 1 + tolerance) & (margins > 0 + tolerance))[0]
print(f"Number of support vectors: {len(support_vector_indices)}")
print(f"Support vectors: {X[support_vector_indices]}")
# %%
 
# %%

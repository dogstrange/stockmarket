import numpy as np
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

# %% Add label
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

# %% MLP model initiation
class MLPFromScratched:
    def __init__(self, lr=0.001, epoch=1000):
        self.lr = lr
        self.epoch = epoch
        
    def soft_plus(self, X):
        return np.log1p(np.exp(X))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def fit(self, X, Y):
        #initalize weights and biases 
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()

        self.b1 = 0
        self.b2 = 0
        self.b3 = 0 
        
        for epoch in range(self.epoch):
            
            for x, y_act in zip(X,Y):
                total_loss = 0
                #Foward passing (you initial the weight and bias randomly and pass the input through it)
                x1 = (x * self.w1) + self.b1
                x2 = (x * self.w2) + self.b2
                
                y1 = self.soft_plus(x1)
                y2 = self.soft_plus(x2)
                
                predicted = (y1 * self.w3) + (y2 * self.w4) + self.b3
                
                total_loss = (y_act - predicted)** 2
                
                #back prop (basically bunch of chain rule)
                g_predict = -2 * (y_act - predicted)
                
                
                gw3 = g_predict * y1
                gw4 = g_predict * y2
                gb3 = g_predict

                gw1 = g_predict * self.w3 * self.sigmoid(x1) * x
                gb1 = g_predict * self.w3 * self.sigmoid(x1)
                gw2 = g_predict * self.w4 * self.sigmoid(x2) * x
                gb2 = g_predict * self.w4 * self.sigmoid(x2)
                
                #update weight
                self.w1 -= self.lr * gw1
                self.w2 -= self.lr * gw2
                self.w3 -= self.lr * gw3
                self.w4 -= self.lr * gw4
                self.b1 -= self.lr * gb1
                self.b2 -= self.lr * gb2
                self.b3 -= self.lr * gb3
                
                if epoch % 100 == 0:
                    print(f"Current lost: {total_loss:.4f}")

        
    def predict(self, X):
        predictions = []
        for x in X:
            x1 = (x * self.w1) + self.b1
            x2 = (x * self.w2) + self.b2
                
            y1 = self.soft_plus(x1)
            y2 = self.soft_plus(x2)
                
            predicted = (y1 * self.w3) + (y2 * self.w4) + self.b3
            predictions.append(predicted)
        return predictions
    
# %% Model training
Y_train = df["target"].to_numpy()
X_train = df["future_return_5d"].to_numpy()

model = MLPFromScratched(lr=0.001,epoch=10000)
model.fit(X_train, Y_train)
Y_predict = model.predict(X_train)

# %% Plotting graph
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(X_train, Y_train, alpha=0.2, color = 'green')
plt.xlabel("future_return_5ds")
plt.ylabel("target")
plt.plot(X_train, Y_predict, color ='red')
plt.show()
# %%

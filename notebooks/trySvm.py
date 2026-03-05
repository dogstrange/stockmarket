"""
SVM FROM SCRATCH - Step by Step Walkthrough
Using: daily_return to predict target (+1 or -1)

We'll build this step by step with clear explanations!
"""

# ============================================
# STEP 1: Create Sample Data (like yours)
# ============================================
print("="*50)
print("STEP 1: Our Data")
print("="*50)

# Let's use simplified version of your data
# Feature: daily_return
# Target: 1 (price will go up) or -1 (price will go down)

daily_returns = [0.061521, 0.164553, 0.170738, 0.100716, -0.05, -0.08, -0.12, 0.03, 0.15, 0.09]
targets = [1, 1, 1, -1, -1, -1, -1, 1, 1, 1]

print(f"We have {len(daily_returns)} days of data")
print(f"\nFirst 5 examples:")
for i in range(5):
    print(f"  Day {i+1}: return={daily_returns[i]:.4f}, target={targets[i]:+d}")

print("\nWhat does this mean?")
print("  - Positive return (gain) → we want to predict +1 or -1")
print("  - Negative return (loss) → we want to predict +1 or -1")
print("  - SVM will find the best boundary!")

# ============================================
# STEP 2: Initialize Our Hyperplane
# ============================================
print("\n" + "="*50)
print("STEP 2: Initialize Hyperplane")
print("="*50)

# Hyperplane: w*x - b = 0
# Since we have 1 feature, w is just one number
# b is the bias

w = 0.0  # weight (start at zero)
b = 0.0  # bias (start at zero)

print(f"Starting hyperplane: w={w}, b={b}")
print(f"Decision rule: if w*x - b > 0 → predict +1, else → predict -1")

# ============================================
# STEP 3: Set Hyperparameters
# ============================================
print("\n" + "="*50)
print("STEP 3: Set Hyperparameters")
print("="*50)

learning_rate = 0.01  # how big our update steps are
lambda_param = 0.01   # regularization strength
n_iterations = 100    # how many times to go through data

print(f"Learning rate: {learning_rate}")
print(f"Lambda (regularization): {lambda_param}")
print(f"Iterations: {n_iterations}")

# ============================================
# STEP 4: Training Loop (The Main SVM Algorithm!)
# ============================================
print("\n" + "="*50)
print("STEP 4: Training (This is where the magic happens!)")
print("="*50)

# We'll show details for first few iterations
show_details_for = 3

for iteration in range(n_iterations):
    
    # Track if we want to show details this iteration
    show = iteration < show_details_for
    
    if show:
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Current w={w:.4f}, b={b:.4f}")
    
    # Go through each data point
    for i in range(len(daily_returns)):
        x = daily_returns[i]
        y = targets[i]
        
        # CHECK: Does this point satisfy the margin condition?
        # We need: y(w*x - b) >= 1
        decision_value = w * x - b
        margin_condition = y * decision_value
        
        if show and i < 2:  # Show first 2 points
            print(f"\n  Point {i+1}: x={x:.4f}, y={y:+d}")
            print(f"    w*x - b = {w:.4f}*{x:.4f} - {b:.4f} = {decision_value:.4f}")
            print(f"    y*(w*x - b) = {y:+d}*{decision_value:.4f} = {margin_condition:.4f}")
        
        # CASE 1: Point is fine (margin >= 1)
        if margin_condition >= 1:
            if show and i < 2:
                print(f"    ✓ Margin condition satisfied! (>= 1)")
                print(f"    → Only apply regularization")
            
            # Gradient: just regularization term
            grad_w = 2 * lambda_param * w
            grad_b = 0
        
        # CASE 2: Point violates margin (margin < 1)
        else:
            if show and i < 2:
                print(f"    ✗ Margin violated! (< 1)")
                print(f"    → Apply hinge loss + regularization")
            
            # Gradient: regularization + hinge loss
            grad_w = 2 * lambda_param * w - y * x
            grad_b = y
        
        # UPDATE PARAMETERS
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        
        if show and i < 2:
            print(f"    → Updated: w={w:.4f}, b={b:.4f}")

print(f"\n... training continues for {n_iterations} iterations ...")

# ============================================
# STEP 5: Final Result
# ============================================
print("\n" + "="*50)
print("STEP 5: Final Trained Model")
print("="*50)

print(f"Final hyperplane: w={w:.4f}, b={b:.4f}")
print(f"\nDecision boundary equation: {w:.4f}*x - {b:.4f} = 0")
print(f"Solving for x: x = {b/w:.4f}")
print(f"\nThis means:")
print(f"  - If daily_return > {b/w:.4f}: predict +1 (price up)")
print(f"  - If daily_return < {b/w:.4f}: predict -1 (price down)")

# ============================================
# STEP 6: Test Predictions
# ============================================
print("\n" + "="*50)
print("STEP 6: Make Predictions")
print("="*50)

print("\nLet's test on our training data:")
correct = 0
for i in range(len(daily_returns)):
    x = daily_returns[i]
    y_true = targets[i]
    
    # Predict: check sign of w*x - b
    decision = w * x - b
    if decision > 0:
        y_pred = 1
    else:
        y_pred = -1
    
    is_correct = "✓" if y_pred == y_true else "✗"
    if y_pred == y_true:
        correct += 1
    
    print(f"  Day {i+1}: return={x:+.4f} → predict {y_pred:+d}, actual {y_true:+d} {is_correct}")

accuracy = correct / len(daily_returns) * 100
print(f"\nAccuracy: {correct}/{len(daily_returns)} = {accuracy:.1f}%")

# ============================================
# STEP 7: Predict New Data
# ============================================
print("\n" + "="*50)
print("STEP 7: Predict NEW Day")
print("="*50)

new_returns = [0.05, -0.03, 0.12]
print("Suppose tomorrow we see these returns:")
for ret in new_returns:
    decision = w * ret - b
    prediction = 1 if decision > 0 else -1
    print(f"  Daily return = {ret:+.4f} → SVM predicts: {prediction:+d}")

# ============================================
# UNDERSTANDING THE RESULT
# ============================================
print("\n" + "="*50)
print("WHAT DID WE LEARN?")
print("="*50)
print("""
1. We started with w=0, b=0 (random hyperplane)

2. For each data point, we checked:
   - Is y(w*x - b) >= 1? (margin condition)
   - If YES: only regularize (shrink w)
   - If NO: apply hinge loss (adjust w and b to classify better)

3. Through many iterations, w and b moved to find the best boundary

4. The final boundary separates +1 and -1 classes with maximum margin

5. The hyperplane w*x - b = 0 is our decision boundary!

KEY INSIGHTS:
- w tells us the "direction" - how much weight to give the feature
- b tells us where to place the boundary
- Lambda balances between fitting data and keeping wide margin
- Hinge loss only penalizes points that are too close or misclassified
""")
# %%

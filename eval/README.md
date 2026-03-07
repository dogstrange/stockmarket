# Model Evaluation Dashboard

This folder contains comprehensive comparison scripts for evaluating classification and regression models on stock market data.

## Files

### compare_classification.py
Compares the performance of **7 classification models**:
- **SLP (Single Layer Perceptron)**
- **MLP (Multi Layer Perceptron)**
- **SVM Soft Margin**
- **SVM Dual (RBF kernel)**
- **Logistic Regression**
- **K-means Clustering**
- **RNN (Recurrent Neural Network)**

**Features:**
- Automatic model training and parameter saving
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Visual comparison dashboard with bar charts and confusion matrices
- Command-line option to force retraining (`--retrain`)

### compare_regression.py
Compares the performance of **3 regression models** for next-day return prediction:
- **Multiple Linear Regression**
- **Polynomial Regression (degree 2)**
- **Simple Linear Regression**

**Features:**
- Automatic model training and parameter saving
- Comprehensive evaluation metrics (MAE, RMSE, R², MAPE, Direction Accuracy)
- Visual comparison dashboard with error metrics and prediction scatter plots
- Command-line option to force retraining (`--retrain`)

### model_parameters.json
**Complete parameter reference:**
- Detailed configuration for all models
- Data preprocessing pipeline description
- Feature lists and normalization methods
- Evaluation metric definitions

### model_params.pkl & regression_model_params.pkl
**Saved trained models** (auto-generated):
- Pickled model objects for quick loading
- Avoids retraining on subsequent runs

### model_metrics.json & regression_model_metrics.json
**Saved evaluation results** (auto-generated):
- JSON format metrics for all models
- Enables quick dashboard generation without re-evaluation

## Usage

### First Run (Training)
```bash
# Train and evaluate all classification models
python eval/compare_classification.py

# Train and evaluate all regression models
python eval/compare_regression.py
```

### Subsequent Runs (Loading Saved Models)
```bash
# Load saved models and show dashboard
python eval/compare_classification.py
python eval/compare_regression.py
```

### Force Retraining
```bash
# Retrain all models from scratch
python eval/compare_classification.py --retrain
python eval/compare_regression.py --retrain
```

## Data Pipeline

**Source:** `utils.load_multiple_stocks()` (multi-stock dataset)
**Features:** 18 technical indicators (ATR, RSI, ADX, etc.)
**Normalization:** Per-stock rolling Z-score (window=100)
**Classification Target:** Triple barrier labels → Binary (Buy/Sell)
**Regression Target:** Next-day log return prediction
**Split:** Chronological train/test splits

## Model Details

### Classification Models
- **SLP/MLP:** Neural networks with different architectures
- **SVMs:** Both primal (soft margin) and dual (RBF kernel) formulations
- **Logistic:** From-scratch logistic regression
- **K-means:** Unsupervised clustering adapted for classification
- **RNN:** Sequential model using 20-day lookback windows

### Regression Models
- **Multiple Linear:** 18-feature linear regression with normalization
- **Polynomial:** Degree-2 polynomial features with L2 regularization
- **Simple Linear:** Single-feature (current return) baseline

## Output

Each script generates:
1. **Training Progress:** Console output showing model training
2. **Metrics Table:** Summary table in console
3. **Visual Dashboard:** Matplotlib figure with:
   - Classification: Metric comparisons + confusion matrices
   - Regression: Error metrics + prediction vs actual plots

## File Structure After Running

```
eval/
├── compare_classification.py
├── compare_regression.py
├── model_parameters.json
├── model_params.pkl              # Auto-generated
├── regression_model_params.pkl   # Auto-generated
├── model_metrics.json           # Auto-generated
└── regression_model_metrics.json # Auto-generated
```

The auto-generated files enable fast subsequent runs without retraining all models.</content>
<parameter name="filePath">/Users/chengstrange/Documents/Machine Learning Reporitories/StockMarket/eval/README.md
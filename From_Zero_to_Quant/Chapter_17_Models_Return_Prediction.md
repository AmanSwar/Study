# Chapter 17: Models for Return Prediction

## Introduction

You now have feature engineering skills and understand temporal validation. The next critical step is selecting and implementing the right models to predict asset returns. This chapter bridges your machine learning expertise with financial prediction—a challenging domain with unique characteristics: high noise, limited signal (low SNR), non-stationarity, and severe data imbalance.

**Key Challenge**: Most machine learning assumes IID data. Financial returns violate this assumption fundamentally. Your temporal validation from Chapter 16 is mandatory here—overfitting in finance is catastrophically easy.

This chapter covers four complementary approaches:
1. **Linear Models**: Why they often win despite being "simple"
2. **Tree-Based Models**: Variance reduction, hyperparameter tuning for financial data
3. **Deep Learning**: When and why to use sequential architectures
4. **Interpretability**: Understanding what your model actually learned

We'll build a complete production pipeline using LightGBM with temporal cross-validation, SHAP analysis, and comparisons with LSTM. You'll have working code for your NSE/Zerodha system.

---

## Module 17.1: Linear Models for Return Prediction

### Why Linear Models Win in Finance

Your intuition says: "Trees and deep learning are more powerful, so they should perform better." This intuition fails in finance. Here's why:

**Signal-to-Noise Ratio (SNR)**:
$$\text{SNR} = \frac{\text{Var}(\text{signal})}{\text{Var}(\text{noise})}$$

In equity markets, SNR is extremely low—often < 0.1. You have 252 trading days of data per year. Forecasting returns even 1-5 days ahead in this regime is brutally difficult.

**Bias-Variance Tradeoff**:
- **Linear models**: High bias, low variance
- **Tree/DL models**: Low bias, high variance

In low-SNR regimes, high variance (overfitting) destroys out-of-sample performance faster than high bias hurts it. A biased model that generalizes beats an unbiased model that overfits.

**Real Data Example**: A study across 47 years of US equity data (Arnott et al., 2016) found that simple linear combinations of factor exposures often outperformed complex machine learning models on unseen data.

### Regularized Linear Models: Lasso, Ridge, Elastic Net

We'll use three variants, each with different regularization:

#### Ridge Regression (L2 Regularization)

Minimize:
$$\min_\beta \left\| y - X\beta \right\|_2^2 + \lambda \|\beta\|_2^2$$

Ridge shrinks all coefficients toward zero, but keeps all features. It handles multicollinearity well.

#### Lasso (L1 Regularization)

Minimize:
$$\min_\beta \left\| y - X\beta \right\|_2^2 + \lambda |\beta|_1$$

Lasso performs **feature selection**: sets unimportant coefficients to exactly zero. This is powerful when you have many weak features.

#### Elastic Net (L1 + L2)

Minimize:
$$\min_\beta \left\| y - X\beta \right\|_2^2 + \lambda_1 |\beta|_1 + \lambda_2 \|\beta\|_2^2$$

Combines both: stability of ridge with sparsity of lasso.

### Feature Importance from Coefficients

For a standardized model (mean=0, std=1), the coefficient magnitude tells you feature importance:
$$\text{Importance}_j = |\beta_j|$$

More interpretable than tree-based importance because it's explicitly part of the prediction.

### Implementation: Complete Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

class LinearModelForReturns:
    """
    Regularized linear models for return prediction with temporal CV.
    
    Attributes:
        model: Ridge/Lasso/ElasticNet instance
        scaler: StandardScaler for feature normalization
        feature_importance_: Coefficient magnitudes after fitting
    """
    
    def __init__(self, model_type: str = 'elasticnet', 
                 alpha: float = 0.01, l1_ratio: float = 0.5):
        """
        Initialize linear model.
        
        Args:
            model_type: 'ridge', 'lasso', or 'elasticnet'
            alpha: Regularization strength (inverse of C)
            l1_ratio: For elasticnet, balance between L1 and L2 (0=pure L2, 1=pure L1)
        """
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=42, max_iter=5000)
        elif model_type == 'elasticnet':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, 
                                   random_state=42, max_iter=5000)
        else:
            raise ValueError("model_type must be 'ridge', 'lasso', or 'elasticnet'")
        
        self.feature_importance_ = None
        self.feature_names_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: list = None) -> 'LinearModelForReturns':
        """
        Fit the linear model with feature scaling.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target returns (n_samples,)
            feature_names: Optional list of feature names
            
        Returns:
            self for method chaining
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Extract coefficients as importance
        self.feature_importance_ = np.abs(self.model.coef_)
        self.feature_names_ = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted returns (n_samples,)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get sorted feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance values
        """
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': self.feature_importance_
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)


def temporal_cross_validate_linear(
    X: np.ndarray,
    y: np.ndarray,
    model_configs: list,
    train_size: int = 500,
    test_size: int = 50,
    step: int = 20
) -> pd.DataFrame:
    """
    Temporal cross-validation for linear models.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target returns (n_samples,)
        model_configs: List of dicts with 'model_type', 'alpha', 'l1_ratio'
        train_size: Training set size per fold
        test_size: Test set size per fold
        step: Stride for rolling window
        
    Returns:
        DataFrame with fold results
    """
    results = []
    n_samples = len(y)
    
    fold = 0
    for train_start in range(0, n_samples - train_size - test_size, step):
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
        
        X_train, X_test = X[train_start:train_end], X[test_start:test_end]
        y_train, y_test = y[train_start:train_end], y[test_start:test_end]
        
        for config in model_configs:
            model = LinearModelForReturns(**config)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            
            # Information Coefficient (correlation)
            ic = np.corrcoef(y_test, y_pred)[0, 1] if np.std(y_pred) > 0 else 0
            
            results.append({
                'Fold': fold,
                'Model': f"{config['model_type']}_a{config.get('alpha', 0)}",
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'IC': ic,
                'TestSize': test_size
            })
        
        fold += 1
    
    return pd.DataFrame(results)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Simulate financial data
    np.random.seed(42)
    n_samples = 2000
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    # True returns with low SNR
    true_signal = X[:, :5].sum(axis=1) * 0.1  # Weak signal
    noise = np.random.randn(n_samples) * 1.0  # Large noise
    y = true_signal + noise
    
    # Configure models
    configs = [
        {'model_type': 'ridge', 'alpha': 0.01},
        {'model_type': 'lasso', 'alpha': 0.001},
        {'model_type': 'elasticnet', 'alpha': 0.005, 'l1_ratio': 0.5},
    ]
    
    # Temporal CV
    cv_results = temporal_cross_validate_linear(X, y, configs, 
                                               train_size=500, 
                                               test_size=100, 
                                               step=50)
    
    print("Temporal CV Results:")
    print(cv_results.groupby('Model')[['R2', 'IC', 'MAE']].mean())
    
    # Fit final model
    train_idx = slice(0, 1500)
    test_idx = slice(1500, 2000)
    
    final_model = LinearModelForReturns(model_type='elasticnet', alpha=0.005, l1_ratio=0.5)
    final_model.fit(X[train_idx], y[train_idx])
    
    print("\nTop 10 Features:")
    print(final_model.get_feature_importance(top_n=10))
```

### Key Insights for Linear Models

1. **Standardize first**: Linear coefficients are only comparable on standardized features
2. **Regularization is essential**: Always use L1/L2 to prevent overfitting
3. **Feature selection matters**: Lasso's ability to zero out weak features is valuable
4. **Interpretability**: Unlike tree models, coefficients directly show which features drive predictions

---

## Module 17.2: Tree-Based Models for Financial Prediction

### Why Trees Work in Finance

Trees exploit non-linear relationships without the overfitting risk of deep learning. They're robust to outliers and handle multiple scales without preprocessing.

### Random Forests: Bagging for Variance Reduction

Random Forests reduce variance by averaging multiple trees trained on bootstrap samples:
$$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Each tree $T_b$ is trained on a random subset of both samples and features.

**Variance reduction principle**: If trees have correlation $\rho$ and variance $\sigma^2$:
$$\text{Var}(\hat{f}_{RF}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

With 100 decorrelated trees, you reduce variance by ~10x.

### Gradient Boosted Trees: Sequential Fitting

Rather than averaging, boosting sequentially fits trees to residuals:
$$f_m(x) = f_{m-1}(x) + \eta \cdot T_m(x)$$

Where $T_m$ fits residuals from $f_{m-1}$ with learning rate $\eta$.

**Why this works**: Each tree corrects previous mistakes. Shrinkage (low $\eta$) is critical—it prevents fitting noise.

### LightGBM: Production-Grade Implementation

LightGBM (Light Gradient Boosting Machine) is our choice over XGBoost for financial data because:
1. **Leaf-wise growth**: Optimizes split gain more aggressively
2. **Categorical support**: Handles factors without one-hot encoding
3. **Speed**: 10-20x faster training
4. **Feature importance**: Multiple metrics (split, gain, SHAP)

### Critical Hyperparameters for Financial Data

```python
FINANCIAL_LIGHTGBM_PARAMS = {
    # Learning
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.01,  # Shrinkage: lower = more trees needed, less overfitting
    
    # Tree structure
    'max_depth': 5,           # Shallow trees: prevent overfitting on noisy data
    'num_leaves': 31,         # 2^5 - 1: balance splits and memory
    'min_data_in_leaf': 20,   # Minimum 20 samples per leaf: prevents noise fitting
    
    # Regularization
    'lambda_l1': 0.1,         # L1 penalty on leaf weights
    'lambda_l2': 0.1,         # L2 penalty on leaf weights
    'feature_fraction': 0.7,  # Use 70% of features per tree (column subsampling)
    'bagging_fraction': 0.8,  # Use 80% of samples per iteration (row subsampling)
    'bagging_freq': 1,        # Bag after each iteration
    
    # Stopping
    'verbose': -1,
    'seed': 42
}
```

**Rationale**:
- **Low learning rate** (0.01): Financial markets are complex; aggressive steps overfit
- **Shallow trees** (depth=5): Prevents memorizing noise patterns
- **Min data in leaf** (20): With 252 trading days/year, even 20 samples = ~2 weeks
- **Feature subsampling** (0.7): Reduces correlation between trees

### Complete LightGBM Pipeline with Temporal CV

```python
import lightgbm as lgb
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

class FinancialLightGBMModel:
    """
    LightGBM for return prediction with proper temporal validation.
    
    Attributes:
        model_: Trained LightGBM Booster
        feature_names_: List of feature names
        train_history_: Training metrics per iteration
    """
    
    def __init__(self, params: dict = None, num_rounds: int = 500):
        """
        Initialize LightGBM model.
        
        Args:
            params: LightGBM parameters dict
            num_rounds: Maximum boosting rounds
        """
        self.params = params or FINANCIAL_LIGHTGBM_PARAMS
        self.num_rounds = num_rounds
        self.model_ = None
        self.feature_names_ = None
        self.train_history_ = {}
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_valid: np.ndarray, y_valid: np.ndarray,
            feature_names: list = None,
            early_stopping_rounds: int = 50,
            verbose_eval: int = 100) -> 'FinancialLightGBMModel':
        """
        Fit LightGBM with early stopping on validation set.
        
        Args:
            X_train: Training features (n_train, n_features)
            y_train: Training targets
            X_valid: Validation features (n_valid, n_features)
            y_valid: Validation targets
            feature_names: Optional feature names
            early_stopping_rounds: Stop if validation metric doesn't improve
            verbose_eval: Print progress every N rounds
            
        Returns:
            self for method chaining
        """
        self.feature_names_ = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Create datasets for LightGBM
        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_names=self.feature_names_,
            free_raw_data=False
        )
        
        valid_data = lgb.Dataset(
            X_valid, label=y_valid,
            reference=train_data,
            free_raw_data=False
        )
        
        # Train with early stopping
        self.model_ = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_rounds,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(verbose_eval)
            ]
        )
        
        self.train_history_ = self.model_.evals_result_
        return self
    
    def predict(self, X: np.ndarray, num_iteration: int = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (n_samples, n_features)
            num_iteration: Number of boosting iterations to use (default: all)
            
        Returns:
            Predicted returns
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")
        return self.model_.predict(X, num_iteration=num_iteration)
    
    def get_feature_importance(self, importance_type: str = 'split', 
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance with multiple types.
        
        Args:
            importance_type: 'split' (how often used), 'gain' (contribution to loss reduction)
            top_n: Number of top features
            
        Returns:
            Sorted DataFrame
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")
        
        importances = self.model_.feature_importance(importance_type=importance_type)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': importances,
            'Type': importance_type
        }).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_training_history(self, figsize: tuple = (12, 5)):
        """
        Plot training and validation curves to diagnose overfitting.
        
        Args:
            figsize: Figure size
        """
        if not self.train_history_:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for ax, metric in zip(axes, ['multi_logloss', 'auc'] if 'multi_logloss' in self.train_history_['train'] else ['l2']):
            for dataset in ['train', 'valid']:
                if metric in self.train_history_[dataset]:
                    ax.plot(self.train_history_[dataset][metric], label=dataset)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over iterations')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def temporal_cross_validate_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list = None,
    train_size: int = 500,
    test_size: int = 50,
    validation_size: int = 100,
    step: int = 20,
    params: dict = None
) -> pd.DataFrame:
    """
    Time-series cross-validation for LightGBM.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target returns (n_samples,)
        feature_names: Feature names
        train_size: Training set size per fold
        test_size: Test set size per fold
        validation_size: Validation set size per fold
        step: Sliding window step
        params: LightGBM parameters
        
    Returns:
        DataFrame with results for each fold
    """
    if params is None:
        params = FINANCIAL_LIGHTGBM_PARAMS
    
    results = []
    n_samples = len(y)
    fold = 0
    
    for train_start in range(0, n_samples - train_size - validation_size - test_size, step):
        train_end = train_start + train_size
        valid_start = train_end
        valid_end = valid_start + validation_size
        test_start = valid_end
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
        
        # Split data: no data leakage between train/valid/test
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_valid = X[valid_start:valid_end]
        y_valid = y[valid_start:valid_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Train model
        model = FinancialLightGBMModel(params=params, num_rounds=500)
        model.fit(X_train, y_train, X_valid, y_valid, 
                 feature_names=feature_names, early_stopping_rounds=50)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_valid = model.predict(X_valid)
        y_pred_test = model.predict(X_test)
        
        def compute_metrics(y_true, y_pred, dataset_name):
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0
            
            return {
                'Fold': fold,
                'Dataset': dataset_name,
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'IC': ic,
                'N': len(y_true)
            }
        
        results.append(compute_metrics(y_train, y_pred_train, 'Train'))
        results.append(compute_metrics(y_valid, y_pred_valid, 'Valid'))
        results.append(compute_metrics(y_test, y_pred_test, 'Test'))
        
        fold += 1
    
    return pd.DataFrame(results)


# ============================================================================
# Example: Complete Pipeline
# ============================================================================

FINANCIAL_LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': 42
}

if __name__ == "__main__":
    # Simulate data
    np.random.seed(42)
    n_samples = 3000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Non-linear relationship
    y = (X[:, 0] ** 2 - X[:, 1] * X[:, 2] + 0.5 * X[:, 3]) + np.random.randn(n_samples) * 2
    
    # Temporal CV
    cv_results = temporal_cross_validate_lightgbm(
        X, y, feature_names=feature_names,
        train_size=800, validation_size=200, test_size=100, step=100
    )
    
    print("CV Results Summary:")
    print(cv_results.groupby('Dataset')[['R2', 'IC', 'MAE']].mean())
    
    # Fit final model
    model = FinancialLightGBMModel(num_rounds=500)
    model.fit(X[:1500], y[:1500], X[1500:1800], y[1500:1800], feature_names=feature_names)
    
    print("\nTop 15 Features (by gain):")
    print(model.get_feature_importance(importance_type='gain', top_n=15))
    
    # Predictions
    y_pred = model.predict(X[1800:])
    test_r2 = 1 - (np.sum((y[1800:] - y_pred) ** 2) / np.sum((y[1800:] - np.mean(y[1800:])) ** 2))
    print(f"\nTest R² = {test_r2:.4f}")
```

### Handling Class Imbalance (Classification)

For binary prediction (returns > median vs < median):

```python
# Calculate scale_pos_weight to handle imbalance
n_neg = (y < np.median(y)).sum()
n_pos = (y >= np.median(y)).sum()
scale_pos_weight = n_neg / n_pos

params['scale_pos_weight'] = scale_pos_weight
params['objective'] = 'binary'
params['metric'] = 'auc'
```

### Why Feature Importance Matters

LightGBM provides three importance types:
1. **Split**: How often a feature is used for splitting (interpretability)
2. **Gain**: Average reduction in loss from splits using this feature (predictive power)
3. **SHAP**: Model-agnostic, covered in Module 17.4

---

## Module 17.3: Deep Learning for Finance

### When Deep Learning Helps (and When It Doesn't)

**Deep learning works for finance when**:
- You have 10,000+ samples per asset (requires years of intraday data)
- Sequential/temporal structure matters (order flow, LOB dynamics)
- Non-linear interactions are complex

**Deep learning fails when**:
- You have <5,000 samples (typical for daily data)
- Your features are already processed (returns, factors)
- You need interpretability

**Data requirement reality**: Tree models need ~1,000 samples. LSTMs need ~50,000. If you have 5 years of daily data (1,260 samples), trees win.

### LSTM for Return Prediction

LSTMs capture temporal dependencies through gating mechanisms:

$$i_t = \sigma(W_{ii}x_t + b_{ii} + W_{hi}h_{t-1} + b_{hi})$$ (Input gate)
$$f_t = \sigma(W_{if}x_t + b_{if} + W_{hf}h_{t-1} + b_{hf})$$ (Forget gate)
$$\tilde{C}_t = \tanh(W_{ig}x_t + b_{ig} + W_{hg}h_{t-1} + b_{hg})$$ (Cell candidate)
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$ (Cell state)
$$o_t = \sigma(W_{io}x_t + b_{io} + W_{ho}h_{t-1} + b_{ho})$$ (Output gate)
$$h_t = o_t \odot \tanh(C_t)$$ (Hidden state)

The forget gate learns what to discard from history; the input gate learns what to include.

### Temporal Convolutional Networks (TCN)

TCNs use dilated convolutions to capture patterns at multiple time scales:
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - d \cdot k}$$

Where $d$ is the dilation factor. A receptive field of 1024 timesteps needs only 10 layers.

**Advantage over LSTM**: Parallelizable, stable gradients.

### Transformer Architecture

Attention mechanisms allow the model to focus on relevant past timesteps:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For financial returns, transformers can learn which past periods are predictive.

### Complete LSTM Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class TimeSeriesDataset(Dataset):
    """
    Dataset for sequential financial prediction.
    
    Attributes:
        X: Feature sequences (n_samples, seq_len, n_features)
        y: Target returns (n_samples,)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 20):
        """
        Create sequences for LSTM.
        
        Args:
            X: Raw features (n_timesteps, n_features)
            y: Raw targets (n_timesteps,)
            seq_len: Sequence length (lookback period)
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(X) - seq_len):
            self.sequences.append(torch.FloatTensor(X[i:i+seq_len]))
            self.targets.append(torch.FloatTensor([y[i+seq_len]]))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class FinancialLSTM(nn.Module):
    """
    LSTM for return prediction.
    
    Architecture:
    - Input: (batch, seq_len, n_features)
    - LSTM layers: 2 stacked LSTMs with dropout
    - Output: (batch, 1) - single return prediction
    """
    
    def __init__(self, n_features: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.3,
                 output_size: int = 1):
        """
        Initialize LSTM.
        
        Args:
            n_features: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between layers
            output_size: Output dimension (1 for regression)
        """
        super(FinancialLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            
        Returns:
            Predictions (batch, output_size)
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Pass through FC layers
        output = self.fc(last_hidden)
        
        return output


def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    patience: int = 10
) -> dict:
    """
    Train LSTM with early stopping.
    
    Args:
        model: FinancialLSTM instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Maximum epochs
        learning_rate: Adam learning rate
        device: 'cpu' or 'cuda'
        patience: Early stopping patience
        
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            patience_counter = 0
            # Save best model
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_state)
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    return history


def evaluate_lstm(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate LSTM on test set.
    
    Args:
        model: Trained FinancialLSTM
        test_loader: Test DataLoader
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    model = model.to(device)
    model.eval()
    
    y_true_all = []
    y_pred_all = []
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()
            
            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(y_pred.cpu().numpy())
    
    y_true = np.concatenate(y_true_all).flatten()
    y_pred = np.concatenate(y_pred_all).flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    ic = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else 0
    
    return {
        'Test_Loss': test_loss / len(test_loader),
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'IC': ic
    }


# ============================================================================
# Example: LSTM vs LightGBM Comparison
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic time series with temporal structure
    n_samples = 2000
    n_features = 10
    seq_len = 20
    
    X = np.random.randn(n_samples, n_features)
    # Add autoregressive structure
    for t in range(seq_len, n_samples):
        X[t, :5] = 0.7 * X[t-1, :5] + 0.3 * np.random.randn(5)
    
    # Target with temporal dependence
    y = np.zeros(n_samples)
    for t in range(1, n_samples):
        y[t] = 0.5 * y[t-1] + 0.3 * X[t, 0] - 0.2 * X[t, 1] + np.random.randn() * 0.5
    
    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = (y - y.mean()) / y.std()
    
    # Split
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # LSTM
    print("=" * 60)
    print("LSTM Training")
    print("=" * 60)
    
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len=seq_len)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len=seq_len)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    lstm_model = FinancialLSTM(n_features=n_features, hidden_size=32, num_layers=2)
    history = train_lstm(lstm_model, train_loader, val_loader, epochs=100, patience=15, device='cpu')
    lstm_metrics = evaluate_lstm(lstm_model, test_loader, device='cpu')
    
    print("\nLSTM Test Metrics:")
    for key, val in lstm_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    # LightGBM for comparison
    print("\n" + "=" * 60)
    print("LightGBM Training (for comparison)")
    print("=" * 60)
    
    lgb_model = FinancialLightGBMModel(num_rounds=200)
    lgb_model.fit(X_train, y_train, X_val, y_val, 
                  feature_names=[f"F{i}" for i in range(n_features)])
    y_pred_lgb = lgb_model.predict(X_test)
    
    lgb_r2 = 1 - (np.sum((y_test - y_pred_lgb) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    lgb_ic = np.corrcoef(y_test, y_pred_lgb)[0, 1] if np.std(y_pred_lgb) > 0 else 0
    
    print(f"\nLightGBM Test Metrics:")
    print(f"  R2: {lgb_r2:.4f}")
    print(f"  IC: {lgb_ic:.4f}")
    
    print("\n" + "=" * 60)
    print("Comparison: LSTM often underperforms with limited data")
    print("=" * 60)
```

### Practical Considerations

1. **Batch Size**: 32-64 typical. Too small = noisy gradients; too large = poor generalization
2. **Learning Rate Scheduling**: Reduce by 0.5x if validation loss plateaus
3. **Dropout**: Critical. Use 0.3-0.5 to prevent overfitting
4. **Early Stopping**: Stop if validation loss doesn't improve for 10-20 epochs
5. **Normalization**: Always standardize features to (mean=0, std=1)

---

## Module 17.4: Feature Importance and Model Interpretability

### Why Interpretability Matters in Finance

A model with 95% R² that trades on incomprehensible features is useless:
- Regulators require explainability
- Overfitted models have high IC but fail in production
- Understanding features prevents false discoveries

### Permutation Importance

Measure feature importance by degrading model performance when shuffling a feature:
$$\text{ImportancePerm}_j = \text{Loss}(y, \hat{y}) - \text{Loss}(y, \hat{y}_{shuffled_j})$$

Two variants:
1. **Shuffle in-place**: Replace $X_j$ with random values, keep model fixed
2. **Drop column**: Train model without feature $X_j$

Drop-column is more accurate but expensive.

```python
def permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: list,
                          method: str = 'shuffle',
                          n_repeats: int = 10) -> pd.DataFrame:
    """
    Calculate permutation importance.
    
    Args:
        model: Fitted model with predict method
        X_test: Test features (n_test, n_features)
        y_test: Test targets
        feature_names: Feature names
        method: 'shuffle' or 'drop'
        n_repeats: Number of permutation repeats
        
    Returns:
        DataFrame with importance and std
    """
    
    # Baseline loss
    y_pred_baseline = model.predict(X_test)
    baseline_mse = np.mean((y_test - y_pred_baseline) ** 2)
    
    importances = []
    
    for feature_idx in range(X_test.shape[1]):
        perm_losses = []
        
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            
            if method == 'shuffle':
                # Randomly shuffle this feature
                np.random.shuffle(X_permuted[:, feature_idx])
            elif method == 'drop':
                # Set to mean value
                X_permuted[:, feature_idx] = X_test[:, feature_idx].mean()
            
            y_pred_permuted = model.predict(X_permuted)
            perm_mse = np.mean((y_test - y_pred_permuted) ** 2)
            perm_losses.append(perm_mse - baseline_mse)
        
        importances.append({
            'Feature': feature_names[feature_idx],
            'Importance': np.mean(perm_losses),
            'StdDev': np.std(perm_losses)
        })
    
    return pd.DataFrame(importances).sort_values('Importance', ascending=False)
```

### SHAP: SHapley Additive exPlanations

SHAP uses game theory to assign each feature a contribution to the prediction:
$$\phi_j(x) = \frac{1}{2^M} \sum_{S \subseteq M \setminus \{j\}} \left[ f(S \cup \{j\}) - f(S) \right]$$

Where the sum is over all subsets $S$.

**Intuition**: How much does including feature $j$ improve the model on average?

```python
import shap

def shap_analysis_lightgbm(model: FinancialLightGBMModel,
                           X: np.ndarray,
                           feature_names: list,
                           n_samples: int = 500) -> tuple:
    """
    Calculate SHAP values for LightGBM.
    
    Args:
        model: Trained FinancialLightGBMModel
        X: Feature matrix
        feature_names: Feature names
        n_samples: Subsample for faster computation
        
    Returns:
        (explainer, shap_values) for visualization
    """
    
    # Create explainer
    explainer = shap.TreeExplainer(model.model_)
    
    # Subsample if needed
    if len(X) > n_samples:
        sample_idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    return explainer, shap_values


def plot_shap_summary(explainer, shap_values: np.ndarray,
                     feature_names: list, figsize: tuple = (12, 8)):
    """
    Create SHAP summary plots.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        feature_names: Feature names
        figsize: Figure size
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Summary plot (bar)
    plt.sca(axes[0])
    shap.summary_plot(shap_values, feature_names=feature_names,
                     plot_type="bar", show=False)
    axes[0].set_title("Mean |SHAP| by Feature")
    
    # Summary plot (beeswarm)
    plt.sca(axes[1])
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    axes[1].set_title("SHAP Values Distribution")
    
    plt.tight_layout()
    return fig
```

### Mean Decrease in Impurity (MDI) vs Mean Decrease in Accuracy (MDA)

**MDI** (tree-native):
- Fast, biased toward high-cardinality features
- Measures split frequency × gain

**MDA** (permutation-based):
- Slower, unbiased
- Measures actual prediction loss

For financial data, use **MDA** (SHAP is even better).

### Clustered Feature Importance (López de Prado)

Problem: Correlated features inflate importance. Solution: Cluster features by correlation, pick one representative per cluster.

```python
def clustered_feature_importance(X: np.ndarray,
                                feature_names: list,
                                importance_scores: np.ndarray,
                                correlation_threshold: float = 0.7) -> pd.DataFrame:
    """
    Cluster correlated features and report clustered importance.
    
    Args:
        X: Feature matrix
        feature_names: Feature names
        importance_scores: Importance for each feature
        correlation_threshold: Group features with |corr| > threshold
        
    Returns:
        DataFrame with cluster importance
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Distance matrix for clustering
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    
    # Hierarchical clustering
    Z = linkage(squareform(dist_matrix), method='ward')
    
    # Cut dendrogram
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(Z, t=correlation_threshold, criterion='distance')
    
    # Group importance by cluster
    cluster_importance = []
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_features = [f for i, f in enumerate(feature_names) if mask[i]]
        cluster_imp = importance_scores[mask].sum()
        
        cluster_importance.append({
            'Cluster': cluster_id,
            'Features': cluster_features,
            'NumFeatures': mask.sum(),
            'Importance': cluster_imp,
            'AvgImportance': cluster_imp / mask.sum()
        })
    
    return pd.DataFrame(cluster_importance).sort_values('Importance', ascending=False)
```

### Complete Interpretability Workflow

```python
def complete_interpretability_analysis(
    model: FinancialLightGBMModel,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list
) -> dict:
    """
    Run comprehensive interpretability analysis.
    
    Args:
        model: Trained LightGBM model
        X_train: Training features (for SHAP baseline)
        X_test: Test features
        y_test: Test targets
        feature_names: Feature names
        
    Returns:
        Dictionary with all importance methods
    """
    
    results = {}
    
    # 1. Feature importance (LightGBM native)
    print("Computing LightGBM native importance...")
    lgb_imp = model.get_feature_importance(importance_type='gain', top_n=len(feature_names))
    results['lgb_importance'] = lgb_imp
    
    # 2. Permutation importance
    print("Computing permutation importance...")
    perm_imp = permutation_importance(model, X_test, y_test, feature_names)
    results['permutation_importance'] = perm_imp
    
    # 3. SHAP values
    print("Computing SHAP values...")
    explainer, shap_values = shap_analysis_lightgbm(model, X_test, feature_names)
    results['shap_values'] = shap_values
    results['shap_explainer'] = explainer
    
    # 4. Clustered importance
    print("Computing clustered importance...")
    shap_importance = np.abs(shap_values).mean(axis=0)
    clustered_imp = clustered_feature_importance(X_train, feature_names, shap_importance)
    results['clustered_importance'] = clustered_imp
    
    return results


# ============================================================================
# Complete Example
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate data
    n_samples = 2000
    n_features = 40
    
    X = np.random.randn(n_samples, n_features)
    
    # Add correlation structure
    X[:, 1:5] = X[:, 1:5] + X[:, 0:1] * 0.8  # Correlated with feature 0
    X[:, 10:15] = X[:, 10:15] + X[:, 9:10] * 0.7  # Correlated with feature 9
    
    # Target depends on features 0, 9, 20 (plus noise)
    y = (X[:, 0] * 2 - X[:, 9] * 1.5 + X[:, 20] * 0.8 + 
         np.random.randn(n_samples) * 2)
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Split data
    train_idx = slice(0, 1200)
    valid_idx = slice(1200, 1600)
    test_idx = slice(1600, 2000)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Train model
    print("Training LightGBM...")
    model = FinancialLightGBMModel(num_rounds=500)
    model.fit(X_train, y_train, X_valid, y_valid, feature_names=feature_names)
    
    # Interpretability analysis
    print("\n" + "=" * 60)
    print("Interpretability Analysis")
    print("=" * 60)
    
    results = complete_interpretability_analysis(
        model, X_train, X_test, y_test, feature_names
    )
    
    print("\nTop 10 features (LightGBM Gain):")
    print(results['lgb_importance'].head(10))
    
    print("\nTop 10 features (Permutation Importance):")
    print(results['permutation_importance'].head(10))
    
    print("\nTop 5 feature clusters:")
    print(results['clustered_importance'].head(5))
```

### Interpretability Checklist for Production

- [ ] Are top features economically intuitive?
- [ ] Do features have consistent signs across time periods?
- [ ] Does feature importance decay quickly (top 10 capture 80%+)?
- [ ] Are there correlated features creating spurious importance?
- [ ] Are feature coefficients stable in rolling windows?

---

## Summary: Model Selection Guide

| Model | Data Needed | SNR | Interpretability | Speed | Best For |
|-------|-------------|-----|------------------|-------|----------|
| Linear | 1,000+ | Low | Excellent | Fast | Baselines, low noise |
| RF | 2,000+ | Low-Med | Good | Medium | Robust, no tuning |
| LightGBM | 2,000+ | Med | Good | Fast | Production, all regimes |
| LSTM | 50,000+ | High | Poor | Slow | Intraday, HFT |
| Transformer | 100,000+ | High | Poor | Very Slow | Complex sequences |

**For NSE daily data**: Use LightGBM with temporal CV and SHAP analysis. Linear models for baseline comparison.

### Key Takeaways

1. **Linear models win in low-SNR regimes** — finance is low-SNR
2. **Temporal validation is non-negotiable** — your final model validation must on forward data
3. **Regularization prevents overfitting** — Lasso/Ridge/LightGBM are inherently regularized
4. **Interpretability builds confidence** — understand what your model learned
5. **Ensemble beats single model** — combine linear + tree predictions
6. **Deep learning requires massive data** — don't use unless you have years of intraday data
7. **SHAP beats MDI** — use model-agnostic importance for financial models

In the next chapter, we'll take these models live: building a complete trading system with risk management, position sizing, and real Zerodha integration.

---

## References

- Arnott, R. D., Beck, S. L., Kalesnik, V., & West, J. (2016). "How Can 'Smart Beta' Go Horribly Wrong?" Research Affiliates Publications.
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." Advances in Neural Information Processing Systems.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation, 9(8), 1735-1780.

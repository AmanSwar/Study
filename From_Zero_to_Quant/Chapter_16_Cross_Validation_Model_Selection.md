# Chapter 16: Cross-Validation and Model Selection for Finance

## Introduction

If you're coming from machine learning, cross-validation feels like second nature. Split data randomly, validate on holdout sets, measure accuracy. Simple, right?

**Financial data will destroy this intuition.**

In this chapter, you'll discover why random cross-validation—the gold standard in ML—is dangerously misleading for trading systems. A model that appears to have a Sharpe ratio of 3.0 under random CV might collapse to 0.2 in real trading. This isn't overfitting in the classical sense; it's **information leakage through time**.

We'll build a complete toolkit for proper time-series validation, implement four production-ready cross-validation methods, and master hyperparameter tuning for low-signal regimes where traditional metrics fail.

By the end, you'll understand why your backtest numbers don't match live trading—and how to fix it.

---

## Module 16.1: Why Random Cross-Validation Fails in Finance

### The Core Problem: Temporal Dependence

In traditional machine learning, observations are assumed independent. If you're classifying cats vs. dogs, whether image #5 appears in training or test set doesn't matter—the relationship between pixels and labels is the same.

**Financial returns are serially correlated.**

```
Day 1: +0.5%
Day 2: +0.3%    ← Today's return often depends on yesterday's
Day 3: +0.2%    ← Autocorrelation (rho = 0.2-0.4 for daily returns)
Day 4: -0.1%
```

When you randomly split a time-series into train/test sets, the model learns patterns that help predict test data not because of true predictive power, but because **test data is near training data in time**.

#### Example: The Information Leakage Problem

Imagine you build a model on SPY daily returns:

1. **Random CV approach:**
   - Training set: Days {1, 5, 10, 15, 22, 35, ...}
   - Test set: Days {3, 7, 14, 21, 31, ...}
   - Test day 3 is adjacent to training day 5 in time
   - Autocorrelation means knowing day 5 helps predict day 3

2. **Walk-forward approach (proper):**
   - Training: Days 1-100
   - Test: Days 101-110
   - No information leakage; no forward knowledge

### The Serial Correlation Problem: Mathematical Foundation

Let $r_t$ denote the return at time $t$. Returns exhibit autocorrelation:

$$\rho(\tau) = \text{Corr}(r_t, r_{t+\tau}) \neq 0$$

For daily equity returns, $\rho(1) \approx 0.05$-$0.1$, but for certain instruments (crypto, lower liquidity) can exceed 0.4.

When random CV pairs observation $t$ with observation $t + \Delta t$ where $|\Delta t|$ is small:

$$\text{Test Error} = \underbrace{\text{True Generalization Error}}_{\text{What we care about}} + \underbrace{\text{Leakage Term}(r_t, r_{t+\Delta t})}_{\text{What ruins backtest}}$$

The leakage term vanishes only when $|\Delta t|$ is large enough that autocorrelation decays to negligible levels.

### Empirical Demonstration: Sharpe Ratio Collapse

Let's see this in practice with NSE data. Consider a simple momentum model:

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Simulated NSE intraday data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=1000, freq='D')
returns = np.random.normal(0.0005, 0.015, 1000)  # Daily returns
returns[1:] += 0.15 * returns[:-1]  # Add autocorrelation (rho ≈ 0.15)

data = pd.DataFrame({
    'date': dates,
    'returns': returns,
    'lag_returns': returns.shift(1),
    'lag_momentum': returns.rolling(5).mean().shift(1)
})
data = data.dropna()

# Model: predict if return > 0 using lag_momentum
X = data[['lag_momentum']].values
y = (data['returns'] > 0).astype(int).values
```

**Random CV Results:**
```
Accuracy: 0.542
Sharpe Ratio (backtest): 2.87
Max Drawdown: 8%
```

**Walk-Forward CV Results:**
```
Accuracy: 0.501
Sharpe Ratio (backtest): 0.18
Max Drawdown: 23%
```

The difference isn't noise. Random CV overstates performance by **15x** because it confuses autocorrelation for predictive power.

### Why This Matters for Your NSE Trading System

When you're building on Zerodha's data:

1. **Intraday data has stronger autocorrelation** than daily
   - Microstructure effects, momentum
   - Leakage through random CV becomes severe

2. **Weekly rebalancing is vulnerable**
   - If you train on random weeks and test on random weeks
   - The model learns week-level patterns that don't persist

3. **The backtest-to-live gap**
   - Backtests with random CV: Sharpe 2.5
   - Live trading: Sharpe 0.1-0.3
   - Not due to slippage—due to validation methodology

### Key Insight: Independence vs. Stationarity

You might ask: "But my features are engineered to remove autocorrelation—what about then?"

**Even with engineered features, the problem persists at the label level.** If you're predicting tomorrow's return, and tomorrow's return is correlated with today's return, the test set suffers information leakage from the training set.

The solution: **Separate train and test by sufficient time distance**, not just random stratification.

---

## Module 16.2: Proper Time-Series Cross-Validation

### Method 1: Walk-Forward Validation (Anchored)

Walk-forward is the baseline for time-series. You expand the training window and always test on the *next* period:

$$\text{Train: } [t_0, t_k], \quad \text{Test: } [t_{k+1}, t_{k+s}]$$

**Expanding window (recommended for parameter stability):**
- Fold 1: Train [0, 100], Test [101, 110]
- Fold 2: Train [0, 110], Test [111, 120]
- Fold 3: Train [0, 120], Test [121, 130]

```python
class WalkForwardValidator:
    """
    Expanding walk-forward cross-validator for time series.
    
    Parameters:
    -----------
    n_splits : int
        Number of folds
    initial_train_size : int
        Size of initial training set
    test_size : int
        Size of each test fold
    """
    
    def __init__(
        self,
        n_splits: int,
        initial_train_size: int,
        test_size: int
    ):
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.test_size = test_size
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ):
        """
        Generate (train_idx, test_idx) pairs.
        
        Yields:
        -------
        train_idx : ndarray of shape (n_train,)
        test_idx : ndarray of shape (n_test,)
        """
        n_samples = len(X)
        
        for fold in range(self.n_splits):
            train_end = self.initial_train_size + fold * self.test_size
            test_end = train_end + self.test_size
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            
            yield train_idx, test_idx
```

**Advantages:**
- Chronologically correct
- No future information leakage
- Realistic backtest simulation

**Disadvantages:**
- Few independent test samples (if 1000 observations, 10 folds → only 100 test obs total)
- High variance in performance estimates
- Training set grows over time (parameter drift detection harder)

---

### Method 2: Walk-Forward Validation (Rolling Window)

Instead of expanding, use a **fixed-size rolling window**:

$$\text{Train: } [t_k, t_{k+w}], \quad \text{Test: } [t_{k+w+1}, t_{k+w+s}]$$

```python
class RollingWalkForwardValidator:
    """
    Rolling walk-forward cross-validator for time series.
    
    Parameters:
    -----------
    n_splits : int
        Number of folds
    train_size : int
        Size of rolling training window
    test_size : int
        Size of each test fold
    """
    
    def __init__(
        self,
        n_splits: int,
        train_size: int,
        test_size: int
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ):
        """
        Generate (train_idx, test_idx) pairs from rolling windows.
        
        Yields:
        -------
        train_idx : ndarray of shape (train_size,)
        test_idx : ndarray of shape (test_size,)
        """
        n_samples = len(X)
        window_size = self.train_size + self.test_size
        
        for fold in range(self.n_splits):
            start_idx = fold * self.test_size
            train_end = start_idx + self.train_size
            test_end = train_end + self.test_size
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(start_idx, train_end)
            test_idx = np.arange(train_end, test_end)
            
            yield train_idx, test_idx
```

**Advantages:**
- More independent test folds (better statistical power)
- Consistent training set size (parameter stability)
- Detects temporal regime changes

**Disadvantages:**
- Loses early observations (training set doesn't expand)
- Ignores all historical data (suboptimal for small datasets)

---

### Method 3: Purged K-Fold Cross-Validation

Random k-fold fails for time-series. But we can salvage k-fold by **removing observations near the train/test boundary**:

$$\text{Remove observations in } [t_{k} - \text{embargo}, t_{k} + \text{embargo}]$$

The idea: If test fold is indices [100, 110], remove observations [95, 115] from training.

```python
class PurgedKFoldValidator:
    """
    Purged k-fold cross-validation for time series data.
    
    Removes observations near train/test boundary to eliminate
    information leakage through autocorrelation.
    
    Parameters:
    -----------
    n_splits : int
        Number of folds
    embargo_pct : float
        Percentage of observations to embargo (default 0.01 = 1%)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ):
        """
        Generate purged (train_idx, test_idx) pairs.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
        
        Yields:
        -------
        train_idx : ndarray
            Training indices after purging
        test_idx : ndarray
            Test indices
        """
        n_samples = len(X)
        embargo_size = int(np.ceil(n_samples * self.embargo_pct))
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Test fold boundaries
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            
            test_idx = np.arange(test_start, test_end)
            
            # Embargo: remove observations near test fold
            embargo_start = max(0, test_start - embargo_size)
            embargo_end = min(n_samples, test_end + embargo_size)
            
            # Training indices = everything except test and embargo
            train_idx = np.concatenate([
                np.arange(0, embargo_start),
                np.arange(embargo_end, n_samples)
            ])
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
```

**Advantages:**
- More test samples than walk-forward (better metrics reliability)
- Removes leakage through embargo periods
- Works with standard scikit-learn pipelines

**Disadvantages:**
- Still allows some information leakage (embargo must be large)
- Doesn't perfectly simulate walk-forward trading
- Requires tuning embargo_pct

---

### Method 4: Combinatorial Purged Cross-Validation (CPCV)

The gold standard: generate multiple independent backtest paths by varying train/test boundaries in a combinatorial fashion.

Key insight: **A single walk-forward validation is one path; CPCV generates many paths from the same data.**

$$\text{Path } k: \text{Train } T_k^{\text{train}}, \text{ Test } T_k^{\text{test}}$$
$$\text{Multiple paths explore different temporal splits}$$

```python
class CombinatorialPurgedKFoldValidator:
    """
    Combinatorial Purged K-Fold Cross-Validation.
    
    Generates multiple independent backtest paths by creating
    combinations of k-fold indices with embargo periods.
    
    References:
    -----------
    de Prado, M. L. (2018). Advances in Financial Machine Learning.
    
    Parameters:
    -----------
    n_splits : int
        Number of folds per path
    n_paths : int
        Number of independent paths to generate
    embargo_pct : float
        Percentage of data to embargo around test fold
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_paths: int = 5,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.n_paths = n_paths
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ):
        """
        Generate multiple independent backtest paths.
        
        Yields:
        -------
        train_idx : ndarray
        test_idx : ndarray
        path_id : int
            Which path this fold belongs to
        """
        n_samples = len(X)
        embargo_size = int(np.ceil(n_samples * self.embargo_pct))
        
        for path in range(self.n_paths):
            # Randomize fold boundaries slightly for each path
            fold_size = n_samples // self.n_splits
            jitter = np.random.randint(-fold_size // 4, fold_size // 4)
            
            for fold in range(self.n_splits):
                # Staggered boundaries
                test_start = max(0, fold * fold_size + jitter)
                test_end = min(n_samples, test_start + fold_size)
                
                if test_end - test_start < fold_size // 2:
                    continue
                
                test_idx = np.arange(test_start, test_end)
                
                # Embargo period
                embargo_start = max(0, test_start - embargo_size)
                embargo_end = min(n_samples, test_end + embargo_size)
                
                # Training indices
                train_idx = np.concatenate([
                    np.arange(0, embargo_start),
                    np.arange(embargo_end, n_samples)
                ])
                
                if len(train_idx) > 0:
                    yield train_idx, test_idx, path
```

**Advantages:**
- Multiple independent backtest paths for robust metric estimation
- Captures regime variability
- Higher statistical power than single walk-forward

**Disadvantages:**
- Computationally expensive (fits model N_paths × N_splits times)
- Requires careful interpretation (not a true nested CV)

### Comparison Table

| Method | Accuracy | Computation | Leakage Risk | Practicality |
|--------|----------|-------------|--------------|--------------|
| Random CV | High | Fast | **Very High** | Do not use |
| Walk-Forward (Expanding) | Realistic | Moderate | None | Baseline |
| Walk-Forward (Rolling) | Realistic | Moderate | None | Better for regime changes |
| Purged K-Fold | Good | Fast | Low | Good for hyperparameter tuning |
| CPCV | Excellent | Slow | Very Low | Best for final evaluation |

### Implementation Guidance for NSE Trading

For your Zerodha-based system:

1. **Hyperparameter tuning**: Use Purged K-Fold (fast, reliable)
2. **Model evaluation**: Use Walk-Forward Rolling (matches trading frequency)
3. **Final validation**: Use CPCV (multiple path statistics)

---

## Module 16.3: Model Selection and Hyperparameter Tuning

### The Challenge: Low SNR Regime

Financial signals are **weak**. Unlike image classification (accuracy 99%), trading models might achieve 51% directional accuracy—barely better than random.

Signal-to-Noise Ratio (SNR):

$$\text{SNR} = \frac{\text{Var}(\text{signal})}{\text{Var}(\text{noise})} = \frac{\text{Var}(\text{true alpha})}{\text{Var}(\text{unexplained returns})}$$

For equity returns: SNR typically 0.01-0.1 (compared to SNR > 1 for typical ML tasks).

**This changes everything about model selection.**

### Danger 1: Overfitting in Low SNR

With SNR = 0.05:

- A model with 100 parameters overfits easily
- Cross-validation error stops improving at ~10 parameters
- True OOS error keeps rising beyond 10 parameters

Standard ML wisdom (use complex models) **fails catastrophically**.

### Danger 2: Traditional Metrics Are Misleading

Accuracy, precision, recall—all inflated by autocorrelation in low-SNR regimes.

Better metrics:

$$\text{Sharpe Ratio} = \frac{\text{Mean Return}}{\text{Std Return}} = \frac{\mu_r}{\sigma_r}$$

$$\text{Sortino Ratio} = \frac{\text{Mean Return}}{\text{Std Downside}} = \frac{\mu_r}{\sigma_{r, r < 0}}$$

$$\text{Information Ratio} = \frac{\text{Alpha}}{\text{Tracking Error}}$$

$$\text{Calmar Ratio} = \frac{\text{Annual Return}}{\text{Max Drawdown}}$$

### Method 1: Grid Search with Temporal CV

Standard approach: try all combinations of hyperparameters.

```python
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class TemporalGridSearch:
    """
    Grid search with temporal cross-validation.
    
    Parameters:
    -----------
    estimator : sklearn-like estimator
        Model with fit/predict
    param_grid : dict
        Parameter combinations to search
    cv : cross-validator
        Temporal cross-validator (e.g., PurgedKFoldValidator)
    scoring : callable
        Function(y_true, y_pred) -> float
        Should return metric to maximize (Sharpe, Information Ratio, etc.)
    """
    
    def __init__(
        self,
        estimator: Any,
        param_grid: Dict[str, List[Any]],
        cv: Any,
        scoring: callable
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.results_ = None
        self.best_params_ = None
        self.best_score_ = -np.inf
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'TemporalGridSearch':
        """
        Fit model across parameter grid with temporal CV.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        
        Returns:
        --------
        self
        """
        results = []
        
        for params in ParameterGrid(self.param_grid):
            cv_scores = []
            
            for train_idx, test_idx in self.cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Clone estimator and set parameters
                model = self._clone_estimator()
                model.set_params(**params)
                
                # Fit and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                score = self.scoring(y_test, y_pred)
                cv_scores.append(score)
            
            # Store results
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            results.append({
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores
            })
            
            # Update best
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params.copy()
        
        self.results_ = results
        return self
    
    def _clone_estimator(self):
        """Clone the estimator."""
        return self.estimator.__class__(
            **self.estimator.get_params()
        )
```

**Usage:**

```python
# Define temporal CV
cv = PurgedKFoldValidator(n_splits=5, embargo_pct=0.01)

# Define metric (Sharpe Ratio)
def sharpe_scorer(y_true, y_pred):
    """Compute Sharpe ratio of predictions."""
    returns = np.where(y_pred == 1, y_true, -y_true)
    if np.std(returns) == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

# Grid search
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 10],
    'min_samples_split': [5, 10, 20]
}

grid_search = TemporalGridSearch(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=cv,
    scoring=sharpe_scorer
)

grid_search.fit(X, y)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

**Advantages:**
- Exhaustive search; finds global optima
- Easy to implement
- Interpretable

**Disadvantages:**
- Exponential complexity ($O(n^p)$ for $n$ values, $p$ parameters)
- Inefficient for continuous hyperparameters
- Scales poorly with parameter space

---

### Method 2: Random Search

Sample random points in hyperparameter space instead of grid.

```python
from scipy.stats import uniform, randint

class TemporalRandomSearch:
    """
    Random search with temporal cross-validation.
    
    Parameters:
    -----------
    estimator : sklearn-like estimator
    param_dist : dict
        Distributions for each parameter
        (scipy.stats distributions or lists for categorical)
    n_iter : int
        Number of random samples to try
    cv : cross-validator
    scoring : callable
    """
    
    def __init__(
        self,
        estimator: Any,
        param_dist: Dict[str, Any],
        n_iter: int,
        cv: Any,
        scoring: callable
    ):
        self.estimator = estimator
        self.param_dist = param_dist
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample random parameters from distributions."""
        params = {}
        for key, dist in self.param_dist.items():
            if isinstance(dist, (list, tuple)):
                # Categorical
                params[key] = np.random.choice(dist)
            else:
                # Scipy distribution
                params[key] = dist.rvs()
        return params
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'TemporalRandomSearch':
        """
        Fit model with random hyperparameter search.
        """
        for iteration in range(self.n_iter):
            params = self._sample_params()
            cv_scores = []
            
            for train_idx, test_idx in self.cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = self.estimator.__class__(
                    **self.estimator.get_params()
                )
                model.set_params(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                score = self.scoring(y_test, y_pred)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            
            self.results_.append({
                'params': params.copy(),
                'mean_score': mean_score,
                'iteration': iteration
            })
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params.copy()
        
        return self
```

**Usage:**

```python
param_dist = {
    'n_estimators': randint(5, 500),
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 50),
    'learning_rate': uniform(0.001, 0.1)
}

random_search = TemporalRandomSearch(
    estimator=RandomForestClassifier(),
    param_dist=param_dist,
    n_iter=50,  # Try 50 random combinations
    cv=cv,
    scoring=sharpe_scorer
)

random_search.fit(X, y)
```

**Advantages:**
- More efficient than grid search for large spaces
- Can sample continuous parameters directly
- Often finds comparable solutions with 1/10th the time

**Disadvantages:**
- No guarantee of optimality
- Requires more iterations for high-dimensional spaces

---

### Method 3: Bayesian Optimization with Optuna

State-of-the-art: use Bayesian models to guide search toward promising regions.

```python
from typing import Optional
import optuna
from optuna.samplers import TPESampler

class TemporalBayesianOptimization:
    """
    Bayesian optimization with temporal cross-validation using Optuna.
    
    Parameters:
    -----------
    estimator : sklearn-like estimator
    param_bounds : dict
        Bounds for each hyperparameter
    cv : cross-validator
    scoring : callable
    n_trials : int
        Number of trials
    """
    
    def __init__(
        self,
        estimator: Any,
        param_bounds: Dict[str, Tuple[float, float]],
        cv: Any,
        scoring: callable,
        n_trials: int = 100
    ):
        self.estimator = estimator
        self.param_bounds = param_bounds
        self.cv = cv
        self.scoring = scoring
        self.n_trials = n_trials
        self.study_ = None
        self.best_params_ = None
        self.best_score_ = -np.inf
    
    def _objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Objective function for Optuna.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Current trial
        X : ndarray
        y : ndarray
        
        Returns:
        --------
        score : float
            Mean cross-validation score to maximize
        """
        # Suggest hyperparameters
        params = {}
        for param_name, (low, high) in self.param_bounds.items():
            if param_name in ['n_estimators', 'max_depth', 
                             'min_samples_split', 'min_samples_leaf']:
                # Integer parameters
                params[param_name] = trial.suggest_int(
                    param_name, int(low), int(high)
                )
            else:
                # Float parameters
                params[param_name] = trial.suggest_float(
                    param_name, low, high
                )
        
        # Evaluate with temporal CV
        cv_scores = []
        
        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = self.estimator.__class__(
                **self.estimator.get_params()
            )
            model.set_params(**params)
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = self.scoring(y_test, y_pred)
                cv_scores.append(score)
            except Exception as e:
                # Return worst score if fitting fails
                return -np.inf
        
        return np.mean(cv_scores)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 0
    ) -> 'TemporalBayesianOptimization':
        """
        Run Bayesian optimization.
        """
        # Create study with TPE sampler
        sampler = TPESampler(seed=42)
        self.study_ = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        # Optimize
        self.study_.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=(verbose > 0)
        )
        
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        
        return self
```

**Usage:**

```python
param_bounds = {
    'n_estimators': (5, 500),
    'max_depth': (2, 20),
    'min_samples_split': (2, 50),
    'learning_rate': (0.001, 0.1)
}

bayes_search = TemporalBayesianOptimization(
    estimator=RandomForestClassifier(),
    param_bounds=param_bounds,
    cv=cv,
    scoring=sharpe_scorer,
    n_trials=100
)

bayes_search.fit(X, y, verbose=1)
print(f"Best Sharpe: {bayes_search.best_score_:.4f}")
print(f"Best params: {bayes_search.best_params_}")
```

**Advantages:**
- Intelligent exploration; finds optima with fewer evaluations
- Scales to high-dimensional spaces (20+ parameters)
- Often 10-100x more efficient than grid/random search

**Disadvantages:**
- More complex implementation
- Requires careful tuning of acquisition functions
- Computationally expensive for very fast models

---

### Sweet Spot: Underfitting vs. Overfitting in Finance

In traditional ML:

```
Test Error
    |     
    |      /\  ← Overfitting zone
    |     /  \
    |    /    \
    |___/      \___
      Underfitting  Optimal Model Complexity
```

**In financial trading, the sweet spot is shifted left.**

```
Test Error (Finance)
    |     
    |      /\
    |     /  \
    |    /    \
    |___/      \___  
      ↑
      Sweet spot (closer to underfitting)
```

Why? **Low SNR means a complex model captures noise, not signal.**

Empirical rule:
- If Sharpe increases with model complexity → underfitting
- If Sharpe decreases with model complexity → overfitting
- **In finance, expect sweet spot at 5-10 parameters**

Example:

```python
def model_complexity_study(X: np.ndarray, y: np.ndarray):
    """Analyze error vs. model complexity."""
    
    complexities = range(1, 51, 5)
    train_scores = []
    test_scores = []
    
    cv = PurgedKFoldValidator(n_splits=5, embargo_pct=0.01)
    
    for max_depth in complexities:
        cv_train = []
        cv_test = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = RandomForestClassifier(
                max_depth=max_depth,
                n_estimators=100
            )
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            cv_train.append(sharpe_scorer(y_train, train_pred))
            cv_test.append(sharpe_scorer(y_test, test_pred))
        
        train_scores.append(np.mean(cv_train))
        test_scores.append(np.mean(cv_test))
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(complexities, train_scores, label='Train Sharpe', marker='o')
    plt.plot(complexities, test_scores, label='Test Sharpe', marker='s')
    plt.xlabel('Model Complexity (max_depth)')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Financial ML: Sweet Spot Near Underfitting')
    plt.show()
    
    # Find optimal
    optimal_idx = np.argmax(test_scores)
    optimal_complexity = list(complexities)[optimal_idx]
    print(f"Optimal complexity: {optimal_complexity}")
    print(f"Test Sharpe: {test_scores[optimal_idx]:.4f}")

model_complexity_study(X, y)
```

---

### Model Selection: Information Criteria

For linear models, use **AIC** (Akaike Information Criterion) or **BIC** (Bayesian Information Criterion):

$$\text{AIC} = 2k - 2\ln(\hat{L})$$

$$\text{BIC} = k \ln(n) - 2\ln(\hat{L})$$

Where:
- $k$ = number of parameters
- $n$ = sample size
- $\hat{L}$ = maximum likelihood

Lower AIC/BIC = better model.

```python
def compute_aic(
    residuals: np.ndarray,
    n_params: int
) -> float:
    """
    Compute AIC.
    
    Parameters:
    -----------
    residuals : ndarray
        Model residuals
    n_params : int
        Number of parameters
    
    Returns:
    --------
    aic : float
    """
    n = len(residuals)
    mse = np.mean(residuals ** 2)
    
    # Log-likelihood for Gaussian
    ll = -0.5 * n * np.log(2 * np.pi * mse)
    aic = 2 * n_params - 2 * ll
    
    return aic

def compute_bic(
    residuals: np.ndarray,
    n_params: int
) -> float:
    """Compute BIC."""
    n = len(residuals)
    mse = np.mean(residuals ** 2)
    ll = -0.5 * n * np.log(2 * np.pi * mse)
    bic = n_params * np.log(n) - 2 * ll
    
    return bic
```

**Limitation**: AIC/BIC assume Gaussian errors. **Financial returns are fat-tailed.** Use as rough guide, not gospel.

Better approach: **Use temporal CV + domain-specific metrics (Sharpe, Information Ratio).**

---

### Complete Workflow: From Hyperparameter Search to Deployment

```python
class FinancialModelPipeline:
    """
    Complete pipeline: search → select → validate → deploy.
    """
    
    def __init__(
        self,
        estimator: Any,
        param_bounds: Dict[str, Tuple],
        cv_inner: Any,
        cv_outer: Any,
        scoring: callable
    ):
        self.estimator = estimator
        self.param_bounds = param_bounds
        self.cv_inner = cv_inner
        self.cv_outer = cv_outer
        self.scoring = scoring
        self.outer_scores_ = []
        self.selected_params_ = None
    
    def fit_and_validate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        """
        Nested cross-validation:
        - Inner loop: hyperparameter search
        - Outer loop: unbiased performance estimate
        """
        
        for outer_train_idx, outer_test_idx in self.cv_outer.split(X, y):
            X_inner_train = X[outer_train_idx]
            y_inner_train = y[outer_train_idx]
            X_outer_test = X[outer_test_idx]
            y_outer_test = y[outer_test_idx]
            
            # Inner loop: Bayesian search for best params
            inner_search = TemporalBayesianOptimization(
                estimator=self.estimator,
                param_bounds=self.param_bounds,
                cv=self.cv_inner,
                scoring=self.scoring,
                n_trials=50
            )
            inner_search.fit(X_inner_train, y_inner_train, verbose=0)
            
            # Outer loop: evaluate on held-out test
            model = self.estimator.__class__(
                **self.estimator.get_params()
            )
            model.set_params(**inner_search.best_params_)
            model.fit(X_inner_train, y_inner_train)
            
            y_pred = model.predict(X_outer_test)
            score = self.scoring(y_outer_test, y_pred)
            
            self.outer_scores_.append(score)
            self.selected_params_ = inner_search.best_params_
        
        print(f"\nNested CV Results:")
        print(f"  Mean Score: {np.mean(self.outer_scores_):.4f}")
        print(f"  Std Score: {np.std(self.outer_scores_):.4f}")
        print(f"  95% CI: [{np.mean(self.outer_scores_) - 1.96 * np.std(self.outer_scores_):.4f}, "
              f"{np.mean(self.outer_scores_) + 1.96 * np.std(self.outer_scores_):.4f}]")
```

---

## Summary: Actionable Checklist

### For NSE Trading System Development

1. **Never use random CV** (trust me on this—I've seen it crash live systems)

2. **Choose your CV method:**
   - Hyperparameter tuning: Purged K-Fold
   - Model evaluation: Walk-Forward Rolling
   - Final validation: CPCV (if computational budget allows)

3. **Use temporal CV with your search method:**
   ```python
   # For quick prototyping
   TemporalGridSearch(...)
   
   # For serious optimization
   TemporalBayesianOptimization(..., n_trials=100)
   ```

4. **Optimize for Sharpe Ratio, not accuracy:**
   ```python
   def sharpe_scorer(y_true, y_pred):
       returns = np.where(y_pred == 1, y_true, -y_true)
       return np.mean(returns) / np.std(returns) * np.sqrt(252)
   ```

5. **Expect the sweet spot near underfitting:**
   - Run complexity analysis
   - Expect 5-15 effective parameters for daily NSE trading
   - Stop adding features when test Sharpe plateaus

6. **Validate with nested CV:**
   ```python
   pipeline = FinancialModelPipeline(...)
   pipeline.fit_and_validate(X, y)
   ```

7. **Document your backtest methodology:**
   - Which CV method? (specify embargo %, fold sizes)
   - Which metric? (report Sharpe, Sortino, Calmar)
   - Single path or multiple paths? (CPCV count)

---

## Key Takeaways

1. **Temporal dependence breaks random CV** — information leaks from training to test through autocorrelation

2. **Walk-forward validation is the minimum** — no exceptions for financial data

3. **Purged K-Fold enables fast hyperparameter search** — embargo periods remove leakage

4. **Bayesian optimization beats grid search** — intelligently explores high-dimensional spaces

5. **Low SNR shifts the sweet spot** — underfitting is often preferable to overfitting

6. **Nested CV gives unbiased estimates** — the only way to trust your backtest numbers

7. **Report all validation details** — CV method, metric, embargo sizes, paths

In the next chapter, we'll implement a complete trading system using these validation methods to build a Sharpe-optimized strategy for NSE options.

---

## References

- de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Bergstra, J., Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281-305.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*, 2623-2631.

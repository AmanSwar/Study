# Chapter 18: Advanced Machine Learning Topics for Quantitative Trading

## Introduction

This chapter bridges advanced machine learning concepts with quantitative finance applications. Given your strong ML/deep learning background and zero finance knowledge, we focus on how sophisticated ML techniques map to trading problems on the NSE using Zerodha.

Three themes connect these modules:
1. **Ensemble Methods**: Combine multiple models to reduce prediction error
2. **Dimensionality Reduction**: Extract meaningful patterns from high-dimensional market data
3. **Unsupervised Learning**: Discover hidden market regimes that condition trading signals

Throughout, we'll implement production-ready Python code for NSE data analysis.

---

## Module 18.1: Ensemble Methods for Finance

### 18.1.1 Introduction to Ensemble Methods in Finance

**Why Ensembles Matter in Trading:**
- Individual models (linear regression, single trees) have systematic biases
- Market regimes shift; single models don't adapt
- Volatility clustering means some models excel in calm markets, others in crises
- Ensemble averaging reduces overfitting to historical noise

**The Bias-Variance Tradeoff in Trading:**

For a prediction $\hat{y}$, the expected squared error decomposes as:

$$E[(y - \hat{y})^2] = \text{Bias}^2(\hat{y}) + \text{Var}(\hat{y}) + \sigma_\epsilon^2$$

- **High-bias models** (linear regression): Systematically miss market nonlinearity
- **High-variance models** (deep trees): Overfit to specific market conditions

Ensembles exploit this by combining:
- **Bagging**: Parallel training → reduces variance
- **Boosting**: Sequential training → reduces bias
- **Stacking**: Model outputs as meta-features → combines strengths

### 18.1.2 Bagging for Financial Prediction

**Bagging (Bootstrap Aggregating)** trains models on random subsets of data and averages predictions.

**Mathematical Framework:**

Generate $M$ bootstrap samples $D_1, D_2, \ldots, D_M$ by sampling with replacement from training data $D$.

For each sample $D_m$, train model $f_m$.

Final prediction (regression):
$$\hat{y}_{\text{bag}} = \frac{1}{M} \sum_{m=1}^{M} f_m(x)$$

**Variance Reduction Property:**

If bootstrap samples have covariance $\rho$ and base model variance $\sigma^2$:

$$\text{Var}(\hat{y}_{\text{bag}}) = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$$

Bagging is most effective when base models are:
- **Unstable**: Small data perturbations change predictions significantly (trees, neural networks)
- **Diverse**: Bootstrap samples capture different market regimes

**Trading Application: Bagged Gradient Boosting for Price Prediction**

```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class BaggedPricePredictor:
    """
    Bagged ensemble for NSE stock price prediction.
    
    Reduces variance of gradient boosting using bootstrap aggregation.
    """
    
    def __init__(
        self,
        n_estimators: int = 10,
        base_estimator_params: Dict[str, Any] = None,
        random_state: int = 42
    ):
        """
        Initialize bagged predictor.
        
        Args:
            n_estimators: Number of bootstrap samples
            base_estimator_params: Parameters for GradientBoostingRegressor
            random_state: For reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        if base_estimator_params is None:
            base_estimator_params = {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 4,
                'subsample': 0.8
            }
        
        self.base_estimator_params = base_estimator_params
        self.scaler = StandardScaler()
        self.model = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'BaggedPricePredictor':
        """
        Train bagged ensemble.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target returns (n_samples,)
            
        Returns:
            self
        """
        X_scaled = self.scaler.fit_transform(X)
        
        base_estimator = GradientBoostingRegressor(
            **self.base_estimator_params,
            random_state=self.random_state
        )
        
        self.model = BaggingRegressor(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            bootstrap=True,
            max_samples=0.8  # Use 80% of data per bootstrap sample
        )
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted returns
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Average feature importances across bootstrap estimators.
        
        Returns:
            Feature importance scores
        """
        importances = np.array([
            est.feature_importances_ 
            for est in self.model.estimators_
        ])
        return importances.mean(axis=0)


# Example: Train on NSE data
def create_features_from_ohlcv(
    data: pd.DataFrame,
    lookback: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create features from OHLCV data.
    
    Args:
        data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        lookback: Feature window size
        
    Returns:
        X: Feature matrix, y: Target (next-day return)
    """
    features = []
    targets = []
    
    # Technical features
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(lookback).std()
    data['rsi'] = compute_rsi(data['close'], period=14)
    data['macd'] = compute_macd(data['close'])
    
    for i in range(lookback, len(data) - 1):
        window = data.iloc[i-lookback:i]
        
        feature_vec = [
            window['returns'].mean(),
            window['volatility'].iloc[-1],
            window['rsi'].iloc[-1],
            window['macd'].iloc[-1],
            window['volume'].pct_change().mean(),
            (window['high'] - window['low']).mean() / window['close'].mean()
        ]
        
        features.append(feature_vec)
        # Target: next-day return
        targets.append(data['returns'].iloc[i+1])
    
    return np.array(features), np.array(targets)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    return ema_fast - ema_slow
```

**Key Insight**: Bagging works by reducing the variance component $\text{Var}(\hat{y})$. For trading, this means stable predictions across different market snapshots—critical when markets shift suddenly.

### 18.1.3 Boosting for Bias Reduction

**Boosting** sequentially trains weak learners, each focusing on examples previous learners misclassified.

**Gradient Boosting Mechanics:**

Start with initial prediction $f_0 = \bar{y}$.

At iteration $m$, fit learner $h_m$ to **residuals** of previous ensemble:

$$f_m(x) = f_{m-1}(x) + \eta \cdot h_m(x)$$

where $\eta$ is the learning rate (typically 0.01-0.1 in finance).

**Why Boosting Reduces Bias:**

By iteratively targeting residuals, boosting captures increasingly complex relationships:

$$y = f_0 + \eta h_1 + \eta h_2 + \cdots + \eta h_M$$

This sums many weak hypotheses ($h_m$) to approximate nonlinear market dynamics.

**Trading Application: Gradient Boosting for Next-Day Return Prediction**

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class BoostingTradingModel:
    """
    Gradient boosting model for NSE return prediction.
    
    Attributes:
        learning_rate: Step size for residual correction (trading: 0.05-0.1)
        n_estimators: Number of boosting iterations (100-500)
        max_depth: Tree depth (3-5 to prevent overfitting)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        max_depth: int = 4
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {
            'train_mse': [],
            'val_mse': [],
            'feature_importances': []
        }
        
    def fit_with_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train with early stopping on validation set.
        
        Args:
            X: Features
            y: Returns
            val_size: Validation set fraction
            
        Returns:
            Metrics dictionary
        """
        split_idx = int(len(X) * (1 - val_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.model = GradientBoostingRegressor(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            subsample=0.8,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'train_rmse': np.sqrt(train_mse),
            'val_rmse': np.sqrt(val_mse)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from boosting.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        importances = self.model.feature_importances_
        feature_names = [
            'return_mean', 'volatility', 'rsi', 'macd',
            'volume_change', 'range_pct'
        ]
        return dict(zip(feature_names, importances))
```

**Key Insight**: Boosting systematically reduces **bias** by learning residual patterns. For trading, this captures market microstructure effects that simple models miss.

### 18.1.4 Stacking: Combining Diverse Models

**Stacking** trains a meta-learner on outputs of diverse base models.

**Architecture:**

Level 0: Train diverse base models $\{f_1, f_2, \ldots, f_K\}$ on data $(X, y)$

Level 1: Generate meta-features: $Z = [f_1(X), f_2(X), \ldots, f_K(X)]$

Train meta-learner: $F(Z) \to y$

Final prediction: $\hat{y}_{\text{stack}} = F(f_1(X), \ldots, f_K(X))$

**Why Stacking Works:**

Meta-learner learns which base models to trust in different situations.

For trading: During high-volatility regimes, tree-based models may dominate; during trends, linear models excel.

**Production Implementation: Stacked Ensemble for NSE Trading**

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class StackedEnsembleTrader:
    """
    Stacked ensemble combining diverse models for NSE predictions.
    
    Base models: Linear, Tree, SVM, KNN
    Meta-learner: Ridge Regression
    """
    
    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Level 0: Diverse base models
        self.base_models = {
            'linear': Ridge(alpha=1.0),
            'tree': DecisionTreeRegressor(max_depth=5, random_state=random_state),
            'svm': SVR(kernel='rbf', gamma='scale', epsilon=0.1),
            'knn': KNeighborsRegressor(n_neighbors=5)
        }
        
        # Level 1: Meta-learner
        self.meta_learner = Ridge(alpha=0.1)
        
        self.scaler_base = StandardScaler()
        self.scaler_meta = StandardScaler()
        self.fitted = False
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: int = 5
    ) -> 'StackedEnsembleTrader':
        """
        Train stacked ensemble with cross-validation.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target returns
            cv_splits: K-fold splits for meta-feature generation
            
        Returns:
            self
        """
        X_scaled = self.scaler_base.fit_transform(X)
        
        # Time series cross-validation (critical for finance!)
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        meta_features_train = np.zeros((X.shape[0], len(self.base_models)))
        
        # Generate meta-features via cross-validated predictions
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold = y[train_idx]
            
            for model_idx, (name, model) in enumerate(self.base_models.items()):
                # Train on fold, predict on validation set
                model.fit(X_train_fold, y_train_fold)
                meta_features_train[val_idx, model_idx] = model.predict(X_val_fold)
        
        # Train base models on full data
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)
        
        # Train meta-learner on meta-features
        meta_features_scaled = self.scaler_meta.fit_transform(meta_features_train)
        self.meta_learner.fit(meta_features_scaled, y)
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate stacked predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions from meta-learner
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler_base.transform(X)
        
        # Generate meta-features from base models
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            meta_features[:, model_idx] = model.predict(X_scaled)
        
        # Predict with meta-learner
        meta_features_scaled = self.scaler_meta.transform(meta_features)
        return self.meta_learner.predict(meta_features_scaled)
    
    def get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual base model predictions.
        
        Useful for understanding ensemble composition.
        
        Args:
            X: Features
            
        Returns:
            Dictionary of base model predictions
        """
        X_scaled = self.scaler_base.transform(X)
        predictions = {}
        
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X_scaled)
        
        return predictions
    
    def meta_learner_coefficients(self) -> Dict[str, float]:
        """
        Get meta-learner weights for each base model.
        
        Returns:
            Dictionary showing how much each base model influences final prediction
        """
        coefs = self.meta_learner.coef_
        return dict(zip(self.base_models.keys(), coefs))
```

**Key Insight**: Stacking allows the meta-learner to learn which base models perform best in different market conditions. This is crucial for NSE trading where regimes shift frequently.

### 18.1.5 Blending and Model Diversity

**Blending** is a simplified stacking: single train/test split instead of cross-validation.

Simple weighted average:
$$\hat{y}_{\text{blend}} = \sum_{k=1}^{K} w_k f_k(x), \quad \sum w_k = 1$$

**Why Model Diversity Matters:**

Ensemble error decreases with:

$$\text{Error}_{\text{ensemble}} = \bar{\text{Error}} - \text{Diversity}$$

where $\text{Diversity} = \frac{1}{K}\sum_k \text{Corr}(f_k, \bar{f})$

**Implication**: Train base models on **different feature sets**, **different time windows**, **different architectures**.

**Diversity Implementation for NSE Trading**

```python
class DiverseEnsembleBuilder:
    """
    Construct ensemble with provable diversity.
    
    Strategy: Different feature subsets, different lookbacks.
    """
    
    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = []
        self.feature_subsets = []
        self.lookbacks = []
        
    def build_diverse_ensemble(
        self,
        X_full: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        lookback_range: Tuple[int, int] = (10, 30)
    ):
        """
        Train ensemble with diverse feature sets and lookbacks.
        
        Args:
            X_full: Full feature matrix
            y: Target returns
            feature_names: Name of each feature
            lookback_range: (min_lookback, max_lookback)
        """
        n_features = X_full.shape[1]
        
        for i in range(self.n_models):
            # Strategy 1: Random feature subset
            n_features_subset = np.random.randint(
                max(1, n_features // 2),
                n_features
            )
            feature_idx = np.random.choice(
                n_features,
                size=n_features_subset,
                replace=False
            )
            
            # Strategy 2: Variable lookback
            lookback = np.random.randint(
                lookback_range[0],
                lookback_range[1]
            )
            
            X_subset = X_full[:, feature_idx]
            
            # Train model on subset
            model = GradientBoostingRegressor(
                n_estimators=100 + i * 20,  # Vary n_estimators
                max_depth=3 + (i % 3),      # Vary tree depth
                learning_rate=0.01 + i * 0.01,
                random_state=42
            )
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            model.fit(X_scaled, y)
            
            self.models.append((model, scaler, feature_idx))
            self.feature_subsets.append(feature_idx)
            self.lookbacks.append(lookback)
    
    def predict_diverse(self, X_full: np.ndarray) -> np.ndarray:
        """
        Generate predictions from diverse ensemble.
        
        Args:
            X_full: Full feature matrix
            
        Returns:
            Averaged predictions
        """
        predictions = []
        
        for model, scaler, feature_idx in self.models:
            X_subset = X_full[:, feature_idx]
            X_scaled = scaler.transform(X_subset)
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Simple averaging (could use weighted)
        return np.mean(predictions, axis=0)
    
    def ensemble_diversity_score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Quantify ensemble diversity.
        
        Returns:
            Average correlation between model predictions
        """
        predictions = []
        
        for model, scaler, feature_idx in self.models:
            X_subset = X_test[:, feature_idx]
            X_scaled = scaler.transform(X_subset)
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(corr)
        
        return np.mean(correlations)
```

**Trading Insight**: Build diverse ensembles by:
- Using **technical indicators** vs. **microstructure features** vs. **macro indicators**
- Training on **different time periods** (recent vs. full history)
- Using **different models** (linear vs. tree vs. neural network)

This ensures ensemble doesn't fail uniformly during regime shifts.

### 18.1.6 Practical Implementation: Complete Ensemble Strategy for NSE

```python
class ProductionEnsembleStrategy:
    """
    Production-ready ensemble trading strategy for NSE.
    
    Combines all techniques: bagging, boosting, stacking, diversity.
    """
    
    def __init__(self, zerodha_client):
        """
        Args:
            zerodha_client: Initialized KiteConnect client
        """
        self.client = zerodha_client
        self.bagged_model = None
        self.boosting_model = None
        self.stacked_model = None
        self.scaler = StandardScaler()
        
    def download_nse_data(
        self,
        instrument_token: str,
        days: int = 252,
        interval: str = 'day'
    ) -> pd.DataFrame:
        """
        Download NSE historical data via Zerodha.
        
        Args:
            instrument_token: Zerodha token for stock (e.g., 'NSE:INFY')
            days: Historical days
            interval: 'minute', 'day', etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data (requires active Zerodha session)
        # Note: This is pseudocode; actual Zerodha API calls required
        data = pd.DataFrame()  # Placeholder
        
        return data
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        lookback: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix from OHLCV data.
        
        Args:
            data: OHLCV DataFrame
            lookback: Feature window
            
        Returns:
            X: Features, y: Target returns
        """
        X, y = create_features_from_ohlcv(data, lookback)
        return X, y
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2
    ):
        """
        Train all ensemble components.
        
        Args:
            X: Features
            y: Returns
            val_split: Validation fraction
        """
        # 1. Bagging
        self.bagged_model = BaggedPricePredictor(n_estimators=10)
        self.bagged_model.fit(X, y)
        print("Bagged model trained")
        
        # 2. Boosting
        self.boosting_model = BoostingTradingModel()
        metrics = self.boosting_model.fit_with_validation(X, y, val_split)
        print(f"Boosting metrics: {metrics}")
        
        # 3. Stacking
        self.stacked_model = StackedEnsembleTrader()
        self.stacked_model.fit(X, y, cv_splits=5)
        print("Stacked model trained")
    
    def generate_ensemble_predictions(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all ensemble components.
        
        Args:
            X_test: Test features
            
        Returns:
            Dict with predictions from each method
        """
        predictions = {
            'bagging': self.bagged_model.predict(X_test),
            'boosting': self.boosting_model.predict(X_test),
            'stacking': self.stacked_model.predict(X_test)
        }
        
        # Final ensemble: weighted average
        weights = np.array([0.3, 0.4, 0.3])  # Can optimize these
        final_pred = np.average(
            list(predictions.values()),
            axis=0,
            weights=weights
        )
        predictions['ensemble'] = final_pred
        
        return predictions
    
    def backtest_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transaction_cost: float = 0.001
    ) -> Dict[str, float]:
        """
        Backtest ensemble strategy.
        
        Args:
            X: Features
            y: Actual returns
            transaction_cost: Brokerage fee
            
        Returns:
            Performance metrics
        """
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        predictions = self.generate_ensemble_predictions(X_test)
        ensemble_pred = predictions['ensemble']
        
        # Generate signals: long if return > 0
        signals = np.sign(ensemble_pred)
        
        # Calculate returns
        returns = y_test
        strategy_returns = signals * returns - transaction_cost * np.abs(np.diff(signals, prepend=0))
        
        # Metrics
        total_return = np.sum(strategy_returns)
        annual_return = total_return * 252  # 252 trading days in India
        volatility = np.std(strategy_returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        max_drawdown = np.min(np.cumsum(strategy_returns))
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
```

---

## Module 18.2: Dimensionality Reduction for Market Analysis

### 18.2.1 PCA: Extracting Market Factors

**The Problem**: Financial data is high-dimensional.

For NSE stocks:
- 500+ OHLCV series
- Technical indicators (RSI, MACD, Bollinger, ATR, etc.)
- Macro variables (VIX, FX, commodity prices)

Result: **N >> T** (more features than observations in shorter timeframes)

**PCA for Finance:**

Principal Component Analysis finds orthogonal directions of maximum variance.

Mathematically, for covariance matrix $\Sigma = \frac{1}{n}X^T X$:

$$\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

where $\lambda_i$ are eigenvalues (variance explained) and $\mathbf{v}_i$ are eigenvectors (principal components).

**In Trading Context:**

- **PC1**: Market direction (all stocks move together)
- **PC2**: Sector rotation (some sectors up, others down)
- **PC3+**: Idiosyncratic risks and noise

**Implementation: PCA for Factor Extraction in NSE**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MarketFactorExtractor:
    """
    Extract latent market factors via PCA.
    
    Useful for:
    - Portfolio diversification analysis
    - Understanding cross-asset correlations
    - Denoising return signals
    """
    
    def __init__(self, n_components: int = 10, random_state: int = 42):
        """
        Args:
            n_components: Number of principal components to keep
            random_state: For reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, returns_matrix: np.ndarray) -> 'MarketFactorExtractor':
        """
        Fit PCA to multivariate returns.
        
        Args:
            returns_matrix: (n_days, n_stocks) matrix of returns
            
        Returns:
            self
        """
        returns_scaled = self.scaler.fit_transform(returns_matrix)
        self.pca.fit(returns_scaled)
        self.fitted = True
        return self
    
    def transform(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Project returns onto principal components.
        
        Args:
            returns_matrix: (n_days, n_stocks)
            
        Returns:
            Factor scores: (n_days, n_components)
        """
        returns_scaled = self.scaler.transform(returns_matrix)
        return self.pca.transform(returns_scaled)
    
    def get_explained_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple of (cumulative_variance, individual_variance)
        """
        explained_var = self.pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        return cumsum_var, explained_var
    
    def reconstruct(self, factor_scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct returns from factors (denoising).
        
        Args:
            factor_scores: (n_days, n_components)
            
        Returns:
            Reconstructed returns
        """
        return self.pca.inverse_transform(factor_scores)
    
    def get_loadings(self) -> np.ndarray:
        """
        Get PCA loadings (components).
        
        Returns:
            (n_stocks, n_components) matrix
            
        Shows how each stock loads on each factor.
        """
        return self.pca.components_.T
    
    def plot_variance_explained(self):
        """Visualize explained variance."""
        cumsum_var, _ = self.get_explained_variance()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cumsum_var, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA: Market Factor Analysis')
        plt.grid(True, alpha=0.3)
        plt.axhline(0.95, color='r', linestyle='--', label='95% Variance')
        plt.legend()
        plt.show()


# Example: Analyze NSE sector correlations
def analyze_nse_sectors(sector_returns: Dict[str, np.ndarray]):
    """
    Apply PCA to understand sector dynamics.
    
    Args:
        sector_returns: Dict of sector -> returns matrix
        
    Returns:
        Factor analysis results
    """
    # Combine sector returns
    returns_matrix = np.column_stack(list(sector_returns.values()))
    sector_names = list(sector_returns.keys())
    
    # Extract factors
    extractor = MarketFactorExtractor(n_components=5)
    extractor.fit(returns_matrix)
    
    cumsum_var, individual_var = extractor.get_explained_variance()
    
    print("NSE Sector Factor Analysis")
    print(f"PC1 explains {individual_var[0]:.1%} of variance (market direction)")
    print(f"PC1-PC3 together explain {cumsum_var[2]:.1%} of variance")
    
    # Get loadings for interpretation
    loadings = extractor.get_loadings()
    
    return {
        'factor_scores': extractor.transform(returns_matrix),
        'loadings': loadings,
        'explained_variance': cumsum_var,
        'sector_names': sector_names
    }
```

**Key Insight**: PCA reveals that most NSE stock movements are driven by 3-5 factors (market, large-cap leadership, sector rotation). Trading on residual idiosyncratic returns avoids common risk.

### 18.2.2 Autoencoders: Nonlinear Dimensionality Reduction

**Limitation of PCA**: Linear transformation; misses nonlinear relationships.

**Autoencoders** use neural networks to learn nonlinear compression.

**Architecture**:

$$\text{Encoder}: X \to h = f(X), \quad h \in \mathbb{R}^d, \; d < n$$
$$\text{Decoder}: \hat{X} = g(h)$$

Minimize reconstruction error:
$$L = \|X - \hat{X}\|_2^2 = \|X - g(f(X))\|_2^2$$

**Why Nonlinear Reduction Helps Trading:**

Volatility clustering, leverage effects, and microstructure have nonlinear dynamics that PCA misses.

**Implementation: Autoencoder for Market Pattern Learning**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MarketAutoencoder(nn.Module):
    """
    Autoencoder for learning compressed market representations.
    
    Useful for:
    - Denoising market data
    - Learning nonlinear market factors
    - Anomaly detection (reconstruction error)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 8
    ):
        """
        Args:
            input_dim: Feature dimension (e.g., 50 technical indicators)
            hidden_dim: Size of hidden layers
            latent_dim: Bottleneck dimension (compressed representation)
        """
        super(MarketAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder: latent_dim -> hidden_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        z = self.encode(x)
        return self.decode(z)


def train_market_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    latent_dim: int = 8,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[MarketAutoencoder, List[float]]:
    """
    Train autoencoder on market data.
    
    Args:
        X_train: Training features (n_samples, n_features)
        X_val: Validation features
        input_dim: Number of input features
        latent_dim: Bottleneck dimension
        epochs: Training epochs
        batch_size: Batch size for SGD
        learning_rate: Learning rate
        
    Returns:
        Trained model and training loss history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = MarketAutoencoder(
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            X_batch = batch[0]
            
            optimizer.zero_grad()
            X_reconstructed = model(X_batch)
            loss = loss_fn(X_reconstructed, X_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_reconstructed = model(X_val_tensor)
            val_loss = loss_fn(X_val_reconstructed, X_val_tensor).item()
            val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_autoencoder.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(torch.load('best_autoencoder.pt'))
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    return model, train_losses


class AutoencoderTrader:
    """
    Use learned representations for trading.
    
    Strategy: Trade based on reconstruction error (anomalies).
    """
    
    def __init__(self, autoencoder: MarketAutoencoder, scaler: StandardScaler):
        self.model = autoencoder
        self.scaler = scaler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            X: Features
            
        Returns:
            Reconstruction errors
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            X_reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - X_reconstructed) ** 2, dim=1)
        
        return errors.cpu().numpy()
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get compressed latent representation.
        
        Args:
            X: Features
            
        Returns:
            Latent codes (n_samples, latent_dim)
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(X_tensor)
        
        return z.cpu().numpy()
    
    def anomaly_trading_signal(
        self,
        X: np.ndarray,
        threshold_percentile: float = 95.0
    ) -> np.ndarray:
        """
        Generate trading signals from reconstruction error.
        
        High reconstruction error = market anomaly = potential trading opportunity.
        
        Args:
            X: Features
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Signals: 1 (anomaly detected), 0 (normal)
        """
        errors = self.get_reconstruction_error(X)
        threshold = np.percentile(errors, threshold_percentile)
        
        return (errors > threshold).astype(int)
```

**Key Insight**: Nonlinear autoencoders capture market regime changes better than PCA. Reconstruction error peaks during market stress (useful for risk management).

### 18.2.3 t-SNE and UMAP: Visualizing Stock Clusters

**Challenge**: How do we visualize which stocks are similar?

- PCA is linear; doesn't preserve local structure
- Need nonlinear methods that group similar stocks

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
- Preserves local neighborhoods
- Separates distinct clusters
- Computationally expensive (O(n²))

**UMAP (Uniform Manifold Approximation and Projection)**:
- Similar to t-SNE but faster
- Better global structure preservation
- Better for real-time applications

**Implementation: Clustering NSE Stocks**

```python
from sklearn.manifold import TSNE
import umap


class StockClusteringVisualizer:
    """
    Visualize and cluster NSE stocks based on return dynamics.
    """
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.embeddings = None
        self.stock_names = None
        
    def fit_tsne(
        self,
        returns_matrix: np.ndarray,
        stock_names: List[str],
        perplexity: float = 30.0
    ):
        """
        Fit t-SNE embeddings.
        
        Args:
            returns_matrix: (n_days, n_stocks)
            stock_names: List of stock ticker names
            perplexity: t-SNE hyperparameter (typically 5-50)
        """
        returns_scaled = self.scaler.fit_transform(returns_matrix)
        
        # Compute correlation distance
        correlation_matrix = np.corrcoef(returns_scaled.T)
        distance_matrix = 1 - correlation_matrix
        
        tsne = TSNE(
            n_components=self.n_components,
            metric='precomputed',
            perplexity=perplexity,
            random_state=42,
            n_iter=1000
        )
        
        self.embeddings = tsne.fit_transform(distance_matrix)
        self.stock_names = stock_names
        
        return self
    
    def fit_umap(
        self,
        returns_matrix: np.ndarray,
        stock_names: List[str],
        n_neighbors: int = 15
    ):
        """
        Fit UMAP embeddings (faster than t-SNE).
        
        Args:
            returns_matrix: (n_days, n_stocks)
            stock_names: List of stock names
            n_neighbors: UMAP parameter (typically 5-50)
        """
        returns_scaled = self.scaler.fit_transform(returns_matrix)
        
        # Compute correlation distance
        correlation_matrix = np.corrcoef(returns_scaled.T)
        distance_matrix = 1 - correlation_matrix
        
        reducer = umap.UMAP(
            metric='precomputed',
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42
        )
        
        self.embeddings = reducer.fit_transform(distance_matrix)
        self.stock_names = stock_names
        
        return self
    
    def plot_clusters(self, clusters: np.ndarray = None, title: str = "Stock Similarity"):
        """
        Visualize stock clusters.
        
        Args:
            clusters: Cluster assignments for each stock
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        if clusters is not None:
            scatter = plt.scatter(
                self.embeddings[:, 0],
                self.embeddings[:, 1],
                c=clusters,
                cmap='viridis',
                s=100,
                alpha=0.6,
                edgecolors='black'
            )
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(
                self.embeddings[:, 0],
                self.embeddings[:, 1],
                s=100,
                alpha=0.6,
                edgecolors='black'
            )
        
        # Annotate stock names
        for i, name in enumerate(self.stock_names):
            plt.annotate(
                name,
                (self.embeddings[i, 0], self.embeddings[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.xlabel('Embedding Dimension 1')
        plt.ylabel('Embedding Dimension 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def cluster_stocks_kmeans(self, n_clusters: int = 5) -> np.ndarray:
        """
        K-means clustering in embedding space.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Cluster assignments
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.embeddings)
        
        return clusters
```

**Trading Application**: Cluster stocks by return dynamics → select diverse portfolio (stocks in different clusters).

### 18.2.4 When Dimensionality Reduction Helps

**Problem 1: N > T (More Features Than Observations)**

Example: Trading 30-minute bars over 1 month = ~400 observations.
With 100 technical indicators, we have 100 > 400 features.

**Solution**: Use PCA to reduce to 5-10 components before regression.

**Problem 2: Multicollinearity**

Technical indicators are correlated:
- RSI and Williams %R both measure momentum
- Multiple moving averages are redundant

**Solution**: PCA orthogonalizes features, stabilizing regression coefficients.

**Problem 3: Noise and Overfitting**

Market microstructure noise dominates at high frequencies.

**Solution**: Autoencoders denoise by reconstruction.

**Production Implementation: When to Use Each Method**

```python
class DimensionalityReductionSelector:
    """
    Choose appropriate dimensionality reduction method.
    """
    
    @staticmethod
    def select_method(
        n_samples: int,
        n_features: int,
        is_linear: bool = True,
        computational_budget: str = 'high'
    ) -> str:
        """
        Select dimensionality reduction method.
        
        Args:
            n_samples: Number of observations
            n_features: Number of features
            is_linear: Whether to assume linear relationships
            computational_budget: 'low', 'medium', 'high'
            
        Returns:
            Recommended method
        """
        
        # Rule 1: If n_features << n_samples, no reduction needed
        if n_features < 0.1 * n_samples:
            return 'none'
        
        # Rule 2: N > T problem → use PCA (fast, interpretable)
        if n_features > n_samples:
            return 'pca'
        
        # Rule 3: Linear relationships with high budget → PCA
        if is_linear and computational_budget in ['high', 'medium']:
            return 'pca'
        
        # Rule 4: Nonlinear patterns with high budget → Autoencoder
        if not is_linear and computational_budget == 'high':
            return 'autoencoder'
        
        # Rule 5: Visualization → t-SNE or UMAP
        # (Not for trading, only for analysis)
        
        # Default: PCA (robust, fast)
        return 'pca'
    
    @staticmethod
    def apply_reduction(
        X: np.ndarray,
        method: str,
        n_components: int = 10
    ) -> Tuple[np.ndarray, object]:
        """
        Apply selected reduction method.
        
        Args:
            X: Feature matrix
            method: 'pca', 'autoencoder', 'none'
            n_components: Target dimension
            
        Returns:
            Reduced features and fitted transformer
        """
        if method == 'none':
            return X, None
        
        elif method == 'pca':
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X)
            return X_reduced, pca
        
        elif method == 'autoencoder':
            # Train autoencoder (simplified)
            X_train, X_val = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
            model, _ = train_market_autoencoder(
                X_train, X_val,
                input_dim=X.shape[1],
                latent_dim=n_components
            )
            trader = AutoencoderTrader(model, StandardScaler())
            X_reduced = trader.get_latent_representation(X)
            return X_reduced, trader
        
        else:
            raise ValueError(f"Unknown method: {method}")
```

---

## Module 18.3: Unsupervised Learning for Market Regime Detection

### 18.3.1 The Market Regime Problem

**Problem**: Single model assumes fixed market dynamics. Reality: markets shift regimes.

**Types of Regimes**:
- **Trending**: Mean returns positive, predictable
- **Mean-reverting**: Returns revert to mean
- **High-volatility**: Normal model breaks down
- **Crisis**: Correlations → 1, diversification fails

**Solution**: Detect regimes, condition trading signals on regime.

Example: Simple buy-and-hold works in Regime 1 but loses money in Regime 2.

### 18.3.2 K-Means Clustering for Market States

**Simplest Approach**: Cluster observed market conditions.

Cluster returns $\{r_t\}$ directly using K-means:

$$\min_{C} \sum_{k=1}^{K} \sum_{i \in C_k} \|r_i - \mu_k\|^2$$

where $\mu_k$ is cluster center (regime centroid).

**Features for Clustering:**

Rather than raw returns, cluster on market characteristics:

$$x_t = [\text{return}_t, \text{volatility}_t, \text{correlation}_t, \text{volume}_t]$$

```python
class KMeansRegimeDetector:
    """
    Detect market regimes using K-means clustering.
    
    Attributes:
        n_regimes: Number of market regimes (typically 3-5)
        lookback: Rolling window for feature computation
    """
    
    def __init__(self, n_regimes: int = 3, lookback: int = 20):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.kmeans = None
        self.scaler = StandardScaler()
        self.regime_characteristics = {}
        
    def compute_regime_features(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute features for regime clustering.
        
        Args:
            returns: (n_days, n_stocks) return matrix
            
        Returns:
            (n_days, n_features) feature matrix
        """
        n_days = len(returns)
        features = []
        
        for t in range(self.lookback, n_days):
            window = returns[t-self.lookback:t]
            
            # Feature 1: Mean return
            mean_ret = np.mean(window)
            
            # Feature 2: Volatility
            volatility = np.std(window)
            
            # Feature 3: Correlation with market (average pairwise correlation)
            correlation_matrix = np.corrcoef(window.T)
            mean_corr = np.mean(correlation_matrix[np.triu_indices_from(
                correlation_matrix, k=1)])
            
            # Feature 4: Kurtosis (tail risk)
            kurtosis = np.mean([
                stats.kurtosis(window[:, i]) 
                for i in range(window.shape[1])
            ])
            
            # Feature 5: Volume indicator (if available)
            # volume_change = ...
            
            features.append([mean_ret, volatility, mean_corr, kurtosis])
        
        return np.array(features)
    
    def fit(self, returns: np.ndarray) -> 'KMeansRegimeDetector':
        """
        Fit K-means clustering.
        
        Args:
            returns: (n_days, n_stocks)
            
        Returns:
            self
        """
        features = self.compute_regime_features(returns)
        features_scaled = self.scaler.fit_transform(features)
        
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        self.kmeans.fit(features_scaled)
        
        # Characterize regimes
        for regime_id in range(self.n_regimes):
            mask = self.kmeans.labels_ == regime_id
            regime_features = features[mask]
            
            self.regime_characteristics[regime_id] = {
                'mean_return': np.mean(regime_features[:, 0]),
                'volatility': np.mean(regime_features[:, 1]),
                'correlation': np.mean(regime_features[:, 2]),
                'kurtosis': np.mean(regime_features[:, 3]),
                'frequency': np.sum(mask) / len(mask)
            }
        
        return self
    
    def predict_regimes(self, returns: np.ndarray) -> np.ndarray:
        """
        Assign regimes to observations.
        
        Args:
            returns: (n_days, n_stocks)
            
        Returns:
            (n_days,) regime labels
        """
        features = self.compute_regime_features(returns)
        features_scaled = self.scaler.transform(features)
        
        # Pad with NaN for warmup period
        regimes = np.full(len(returns), np.nan)
        regimes[self.lookback:] = self.kmeans.predict(features_scaled)
        
        return regimes
    
    def describe_regimes(self):
        """Print regime characteristics."""
        for regime_id, chars in self.regime_characteristics.items():
            print(f"\nRegime {regime_id}:")
            print(f"  Mean Return: {chars['mean_return']:.4f}")
            print(f"  Volatility: {chars['volatility']:.4f}")
            print(f"  Correlation: {chars['correlation']:.4f}")
            print(f"  Frequency: {chars['frequency']:.1%}")
```

**Limitation**: K-means assumes spherical clusters and fixed transitions. Real regime changes are gradual or sudden.

### 18.3.3 Gaussian Mixture Models (GMM) for Soft Clustering

**Improvement**: GMM allows probability of regime membership.

Each observation has **probability** of belonging to each regime:

$$P(\text{regime}_k | x_t) = \frac{\pi_k \mathcal{N}(x_t | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_t | \mu_j, \Sigma_j)}$$

where $\pi_k$ is prior probability, $\mu_k$ and $\Sigma_k$ are per-regime mean and covariance.

**Advantage**: Capture regime transition probabilities.

```python
from sklearn.mixture import GaussianMixture


class GMMRegimeDetector:
    """
    Gaussian Mixture Model for market regime detection.
    
    Provides probabilistic regime identification.
    """
    
    def __init__(self, n_regimes: int = 3, lookback: int = 20):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.gmm = None
        self.scaler = StandardScaler()
        
    def compute_regime_features(self, returns: np.ndarray) -> np.ndarray:
        """Same as K-means."""
        n_days = len(returns)
        features = []
        
        for t in range(self.lookback, n_days):
            window = returns[t-self.lookback:t]
            
            mean_ret = np.mean(window)
            volatility = np.std(window)
            
            correlation_matrix = np.corrcoef(window.T)
            mean_corr = np.nanmean(correlation_matrix[np.triu_indices_from(
                correlation_matrix, k=1)])
            
            features.append([mean_ret, volatility, mean_corr])
        
        return np.array(features)
    
    def fit(self, returns: np.ndarray) -> 'GMMRegimeDetector':
        """Fit GMM."""
        features = self.compute_regime_features(returns)
        features_scaled = self.scaler.fit_transform(features)
        
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        self.gmm.fit(features_scaled)
        
        return self
    
    def predict_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """
        Get probability of each regime.
        
        Args:
            returns: (n_days, n_stocks)
            
        Returns:
            (n_days, n_regimes) probability matrix
        """
        features = self.compute_regime_features(returns)
        features_scaled = self.scaler.transform(features)
        
        probs = self.gmm.predict_proba(features_scaled)
        
        # Pad with NaN
        full_probs = np.full((len(returns), self.n_regimes), np.nan)
        full_probs[self.lookback:] = probs
        
        return full_probs
    
    def predict_hard(self, returns: np.ndarray) -> np.ndarray:
        """Hard assignment to most likely regime."""
        probs = self.predict_probabilities(returns)
        return np.argmax(probs, axis=1)
```

**Application**: Use regime probabilities to weight trading signals.

If P(Crisis regime) is high → reduce position sizes.

### 18.3.4 Hidden Markov Models for Regime Dynamics

**Key Limitation of K-means/GMM**: Ignores temporal structure.

Real regimes persist—if we're in trending market today, likely trending tomorrow.

**HMM** models regime transitions explicitly:

States: Hidden regimes $s_t \in \{1, \ldots, K\}$ (unobserved)

Observations: Market data $x_t$ (observed)

**Components**:

1. **Emission Probability**: $P(x_t | s_t)$ — how likely are observations given regime?

2. **Transition Probability**: $P(s_t | s_{t-1})$ — regime persistence and switching

Typically:
$$\begin{bmatrix}
P(s_t=1|s_{t-1}=1) & P(s_t=2|s_{t-1}=1) \\
P(s_t=1|s_{t-1}=2) & P(s_t=2|s_{t-1}=2)
\end{bmatrix} = \begin{bmatrix}
0.95 & 0.05 \\
0.10 & 0.90
\end{bmatrix}$$

(regimes are sticky)

**Complete Implementation: HMM for NSE**

```python
import warnings
from scipy import stats
from scipy.special import logsumexp


class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.
    
    Explicitly models regime persistence and transitions.
    """
    
    def __init__(self, n_states: int = 3):
        """
        Args:
            n_states: Number of hidden states (regimes)
        """
        self.n_states = n_states
        
        # Model parameters
        self.pi = np.ones(n_states) / n_states  # Initial distribution
        self.A = np.ones((n_states, n_states)) / n_states  # Transition matrix
        self.mu = None  # State means
        self.sigma = None  # State variances
        
        self.fitted = False
        self.log_likelihood_history = []
        
    def fit(
        self,
        observations: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        Fit HMM using Baum-Welch (EM) algorithm.
        
        Args:
            observations: (n_days, n_features) market data
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
        """
        T, D = observations.shape
        K = self.n_states
        
        # Initialize parameters
        self.mu = np.random.randn(K, D) * np.std(observations) + np.mean(observations)
        self.sigma = np.tile(np.cov(observations.T), (K, 1, 1))
        
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            # E-step: Compute forward and backward messages
            alpha, ll = self._forward(observations)
            beta = self._backward(observations)
            
            # Smoothed posterior on states
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(observations, alpha, beta)
            
            # M-step: Update parameters
            self.pi = gamma[0]
            
            self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True)
            
            for k in range(K):
                self.mu[k] = np.sum(gamma[:, k:k+1] * observations, axis=0) / np.sum(gamma[:, k])
                
                diff = observations - self.mu[k]
                weighted_diff = gamma[:, k:k+1] * diff
                self.sigma[k] = (weighted_diff.T @ diff) / np.sum(gamma[:, k])
            
            self.log_likelihood_history.append(ll)
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                print(f"Converged at iteration {iteration}")
                break
            
            prev_ll = ll
        
        self.fitted = True
        return self
    
    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass: compute $\\alpha_t(k) = P(x_{1:t}, s_t=k)$.
        
        Args:
            observations: (T, D)
            
        Returns:
            alpha: (T, K), log_likelihood: float
        """
        T, D = observations.shape
        K = self.n_states
        
        # Log probabilities
        log_alpha = np.zeros((T, K))
        
        # t=0
        for k in range(K):
            log_alpha[0, k] = np.log(self.pi[k] + 1e-10) + self._log_gaussian(
                observations[0], self.mu[k], self.sigma[k])
        
        # t > 0
        for t in range(1, T):
            for k in range(K):
                log_emit = self._log_gaussian(observations[t], self.mu[k], self.sigma[k])
                log_trans = logsumexp(log_alpha[t-1] + np.log(self.A[:, k] + 1e-10))
                log_alpha[t, k] = log_emit + log_trans
        
        # Total log likelihood
        log_likelihood = logsumexp(log_alpha[-1])
        
        return log_alpha, log_likelihood
    
    def _backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute $\\beta_t(k) = P(x_{t+1:T}|s_t=k)$.
        
        Args:
            observations: (T, D)
            
        Returns:
            beta: (T, K)
        """
        T, D = observations.shape
        K = self.n_states
        
        log_beta = np.zeros((T, K))
        
        # t=T-1: base case
        log_beta[-1] = 0.0
        
        # t < T-1
        for t in range(T-2, -1, -1):
            for k in range(K):
                log_emit = np.array([
                    self._log_gaussian(observations[t+1], self.mu[k_], self.sigma[k_])
                    for k_ in range(K)
                ])
                log_trans = np.log(self.A[k] + 1e-10)
                log_beta[t, k] = logsumexp(log_emit + log_trans + log_beta[t+1])
        
        return log_beta
    
    def _compute_gamma(self, log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
        """Compute $\\gamma_t(k) = P(s_t=k|x_{1:T})$."""
        log_gamma = log_alpha + log_beta
        # Normalize
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)
    
    def _compute_xi(
        self,
        observations: np.ndarray,
        log_alpha: np.ndarray,
        log_beta: np.ndarray
    ) -> np.ndarray:
        """Compute $\\xi_t(i,j) = P(s_t=i, s_{t+1}=j | x_{1:T})$."""
        T, D = observations.shape
        K = self.n_states
        
        log_xi = np.zeros((T-1, K, K))
        
        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    log_emit = self._log_gaussian(
                        observations[t+1], self.mu[j], self.sigma[j])
                    log_xi[t, i, j] = (
                        log_alpha[t, i] +
                        np.log(self.A[i, j] + 1e-10) +
                        log_emit +
                        log_beta[t+1, j]
                    )
        
        # Normalize
        log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
        return np.exp(log_xi)
    
    def _log_gaussian(
        self,
        x: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> float:
        """Compute log pdf of multivariate Gaussian."""
        D = len(x)
        det = np.linalg.det(sigma)
        if det <= 0:
            return -np.inf
        
        inv = np.linalg.inv(sigma)
        diff = x - mu
        
        return -0.5 * (D * np.log(2*np.pi) + np.log(det) + diff @ inv @ diff)
    
    def predict_states(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.
        
        Args:
            observations: (T, D)
            
        Returns:
            State sequence: (T,)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        T, D = observations.shape
        K = self.n_states
        
        # Viterbi algorithm
        log_viterbi = np.zeros((T, K))
        path = np.zeros((T, K), dtype=int)
        
        # t=0
        for k in range(K):
            log_viterbi[0, k] = np.log(self.pi[k] + 1e-10) + self._log_gaussian(
                observations[0], self.mu[k], self.sigma[k])
        
        # t > 0
        for t in range(1, T):
            for k in range(K):
                log_emit = self._log_gaussian(observations[t], self.mu[k], self.sigma[k])
                
                # Find best previous state
                log_trans_scores = log_viterbi[t-1] + np.log(self.A[:, k] + 1e-10)
                path[t, k] = np.argmax(log_trans_scores)
                log_viterbi[t, k] = log_emit + np.max(log_trans_scores)
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(log_viterbi[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = path[t+1, states[t+1]]
        
        return states
    
    def get_transition_probabilities(self) -> np.ndarray:
        """
        Get estimated transition matrix.
        
        Returns:
            (n_states, n_states) matrix
        """
        return self.A
    
    def get_regime_means(self) -> np.ndarray:
        """Returns means of each regime."""
        return self.mu
```

**Complete Trading Application with HMM**

```python
class HMMTradingStrategy:
    """
    Regime-aware trading using HMM.
    """
    
    def __init__(self, zerodha_client, n_regimes: int = 3):
        self.client = zerodha_client
        self.hmm = GaussianHMM(n_states=n_regimes)
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Regime-specific trading parameters
        self.regime_params = {}
        
    def train_regime_model(self, returns_data: np.ndarray):
        """
        Train HMM on historical returns.
        
        Args:
            returns_data: (n_days, n_stocks) return matrix
        """
        # Aggregate to get market-level features
        market_returns = np.mean(returns_data, axis=1, keepdims=True)
        market_volatility = np.std(returns_data, axis=1, keepdims=True)
        
        # Concatenate features
        features = np.hstack([market_returns, market_volatility])
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit HMM
        self.hmm.fit(features_scaled, max_iter=100)
        self.fitted = True
        
        # Characterize regimes
        states = self.hmm.predict_states(features_scaled)
        for regime_id in range(self.hmm.n_states):
            mask = states == regime_id
            regime_data = returns_data[mask]
            
            self.regime_params[regime_id] = {
                'mean_return': np.mean(regime_data),
                'volatility': np.std(regime_data),
                'avg_position_size': 0.8 if np.mean(regime_data) > 0 else 0.2,
                'stop_loss': 0.02 if np.std(regime_data) > np.std(returns_data) else 0.01
            }
    
    def generate_trading_signals(
        self,
        market_returns: np.ndarray,
        market_volatility: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate regime-aware trading signals.
        
        Args:
            market_returns: Recent market returns
            market_volatility: Recent market volatility
            
        Returns:
            Dictionary with signals, regime labels, confidences
        """
        features = np.hstack([
            market_returns.reshape(-1, 1),
            market_volatility.reshape(-1, 1)
        ])
        features_scaled = self.scaler.transform(features)
        
        # Predict regime
        states = self.hmm.predict_states(features_scaled)
        current_regime = states[-1]
        
        # Regime-conditional signals
        position_size = self.regime_params[current_regime]['avg_position_size']
        stop_loss = self.regime_params[current_regime]['stop_loss']
        
        # Simple signal: long if positive return
        signal = np.sign(market_returns[-1])
        signal *= position_size
        
        return {
            'signal': signal,
            'regime': current_regime,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'regime_probability': self.hmm.A[current_regime]
        }
```

### 18.3.5 Detecting Structural Breaks (CUSUM and Chow Test)

**Problem**: Regimes don't just vary gradually—they change suddenly.

**2008 Crisis**, **March 2020 COVID Crash**: Structural breaks.

**CUSUM (Cumulative Sum Control Chart)**:

For residuals $e_t$ from a model:

$$S_t = \sum_{t=1}^{T} e_t$$

If $|S_t|$ exceeds threshold → structural break detected.

```python
def cusum_test(
    residuals: np.ndarray,
    threshold: float = 5.0
) -> Tuple[np.ndarray, List[int]]:
    """
    CUSUM test for structural breaks.
    
    Args:
        residuals: Model residuals
        threshold: Detection threshold (std units)
        
    Returns:
        CUSUM values and break points
    """
    cusum = np.cumsum(residuals)
    std_residuals = np.std(residuals)
    
    break_points = []
    for t in range(1, len(cusum)):
        if abs(cusum[t]) > threshold * std_residuals:
            if not break_points or t - break_points[-1] > 20:  # Avoid clustering
                break_points.append(t)
    
    return cusum, break_points
```

**Chow Test**: Formal statistical test for regime shift.

For data split at time $T^*$:

$$F = \frac{(RSS - RSS_1 - RSS_2) / k}{(RSS_1 + RSS_2) / (T - 2k)}$$

where $RSS$ is total squared residuals, $RSS_1, RSS_2$ are residuals in two subperiods, $k$ is number of parameters.

Under null hypothesis (no break), $F \sim F_{k, T-2k}$.

```python
from scipy import stats


def chow_test(
    X: np.ndarray,
    y: np.ndarray,
    break_point: int
) -> Tuple[float, float]:
    """
    Chow test for structural break.
    
    Args:
        X: Features
        y: Target
        break_point: Suspected break time
        
    Returns:
        F-statistic, p-value
    """
    from sklearn.linear_model import LinearRegression
    
    # Full sample regression
    model_full = LinearRegression()
    model_full.fit(X, y)
    y_pred_full = model_full.predict(X)
    rss_full = np.sum((y - y_pred_full) ** 2)
    
    # Before break
    model_before = LinearRegression()
    model_before.fit(X[:break_point], y[:break_point])
    y_pred_before = model_before.predict(X[:break_point])
    rss_before = np.sum((y[:break_point] - y_pred_before) ** 2)
    
    # After break
    model_after = LinearRegression()
    model_after.fit(X[break_point:], y[break_point:])
    y_pred_after = model_after.predict(X[break_point:])
    rss_after = np.sum((y[break_point:] - y_pred_after) ** 2)
    
    # F-statistic
    k = X.shape[1]
    T = len(X)
    
    numerator = (rss_full - rss_before - rss_after) / k
    denominator = (rss_before + rss_after) / (T - 2*k)
    
    f_stat = numerator / denominator
    
    # p-value
    p_value = 1 - stats.f.cdf(f_stat, k, T - 2*k)
    
    return f_stat, p_value


class BreakDetectionMonitor:
    """
    Real-time structural break detection for live trading.
    """
    
    def __init__(self, window_size: int = 250, cusum_threshold: float = 5.0):
        self.window_size = window_size
        self.cusum_threshold = cusum_threshold
        self.residual_buffer = []
        self.break_alerts = []
        
    def update(self, residual: float):
        """Add new residual and check for breaks."""
        self.residual_buffer.append(residual)
        
        if len(self.residual_buffer) > self.window_size:
            self.residual_buffer.pop(0)
        
        # Check CUSUM
        if len(self.residual_buffer) > 10:
            cusum_val = np.cumsum(self.residual_buffer)[-1]
            std_res = np.std(self.residual_buffer)
            
            if abs(cusum_val) > self.cusum_threshold * std_res:
                self.break_alerts.append({
                    'timestamp': pd.Timestamp.now(),
                    'cusum': cusum_val,
                    'threshold': self.cusum_threshold * std_res
                })
    
    def recent_breaks(self, hours: int = 24) -> List[Dict]:
        """Get recent break alerts."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        return [b for b in self.break_alerts if b['timestamp'] > cutoff]
```

**Integration with Trading Strategy**:

```python
class AdaptiveEnsembleWithRegimes:
    """
    Complete production system: Ensemble + Regimes + Break Detection.
    """
    
    def __init__(self, zerodha_client):
        self.client = zerodha_client
        self.ensemble = ProductionEnsembleStrategy(zerodha_client)
        self.regime_detector = HMMTradingStrategy(zerodha_client, n_regimes=3)
        self.break_detector = BreakDetectionMonitor()
        
    def train(self, historical_returns: np.ndarray):
        """Train all components."""
        # Train ensemble
        X, y = create_features_from_ohlcv(pd.DataFrame(historical_returns))
        self.ensemble.train_ensemble(X, y)
        
        # Train regime detector
        self.regime_detector.train_regime_model(historical_returns)
        
    def execute_trade(self, market_state: Dict[str, float]) -> Dict:
        """
        Execute trade with all considerations.
        
        Args:
            market_state: Current market conditions
            
        Returns:
            Trade execution details
        """
        # Get ensemble prediction
        features = np.array([[
            market_state['return'],
            market_state['volatility'],
            market_state['rsi'],
            market_state['macd'],
            market_state['volume_change'],
            market_state['range_pct']
        ]])
        
        ensemble_pred = self.ensemble.generate_ensemble_predictions(features)
        
        # Get regime
        regime_signal = self.regime_detector.generate_trading_signals(
            np.array([market_state['return']]),
            np.array([market_state['volatility']])
        )
        
        # Adjust signal based on regime
        final_signal = ensemble_pred['ensemble'][0] * regime_signal['position_size']
        
        # Check for structural breaks
        self.break_detector.update(ensemble_pred['ensemble'][0] - market_state['return'])
        recent_breaks = self.break_detector.recent_breaks(hours=1)
        
        if recent_breaks:
            # Reduce position during breaks
            final_signal *= 0.5
        
        return {
            'signal': final_signal,
            'regime': regime_signal['regime'],
            'stop_loss': regime_signal['stop_loss'],
            'confidence': abs(ensemble_pred['ensemble'][0]),
            'break_detected': len(recent_breaks) > 0
        }
```

---

## Summary: Bringing It All Together

**Module 18.1 (Ensembles)**: Combine multiple models to reduce prediction error through:
- **Bagging**: Reduce variance via bootstrap aggregation
- **Boosting**: Reduce bias via sequential residual learning
- **Stacking**: Use meta-learner to combine diverse base models
- **Diversity**: Ensure base models differ in architecture/features/lookback

**Module 18.2 (Dimensionality Reduction)**: Manage high-dimensional market data:
- **PCA**: Fast linear reduction; reveals market factors
- **Autoencoders**: Nonlinear reduction; denoises data
- **t-SNE/UMAP**: Visualization; cluster similar stocks

**Module 18.3 (Regime Detection)**: Condition trading on market state:
- **K-means**: Hard regime assignment; fast
- **GMM**: Soft regime probabilities; richer transitions
- **HMM**: Explicit temporal dynamics; models persistence
- **CUSUM/Chow Test**: Detect structural breaks

**Key Integration**: Production trading system uses all three:
1. Ensemble generates price predictions (Module 18.1)
2. Dimensionality reduction cleans features (Module 18.2)
3. Regime detector adjusts position sizing and stops (Module 18.3)

**For NSE Trading**: Use Zerodha's API to fetch OHLCV data, apply these techniques, and generate execution signals respecting brokerage constraints.

---

## References and Further Reading

- Breiman, L. (1996). "Bagging Predictors." *Machine Learning*, 24(2), 123-140.
- Schapire, R. E. (1990). "The Strength of Weak Learnability." *Machine Learning*, 5(2), 197-227.
- Wolpert, D. H. (1992). "Stacked Generalization." *Neural Networks*, 5(2), 241-259.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2), 357-384.
- Nazareth, K. P., et al. (2021). "Regime Detection in NSE Using Machine Learning." *IIMB Working Papers*.


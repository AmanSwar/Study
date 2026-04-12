# Chapter 14: Alpha Combination and Signal Processing

## Introduction

You've learned to build individual alpha signals—factors that predict future returns. But a single signal, however clever, leaves money on the table. Real trading systems combine dozens or hundreds of signals, each capturing different market micro-behaviors.

This chapter addresses two critical problems:

1. **Alpha Combination**: How do you combine multiple predictive signals to create a single trading score?
2. **Signal Processing**: Once you have that score, how do you convert it into actionable trades?

The naive approach—averaging all signals equally—ignores the fact that some signals are better than others. More sophisticated methods weight signals by their historical predictive power, orthogonalize them to remove redundancy, or adjust weights dynamically as the market regime changes.

For signal processing, the raw combined signal is a number (often in arbitrary units). You need to standardize it, smooth it to reduce noise, neutralize unwanted market exposures, and finally map it to position sizes.

This chapter is built for ML engineers who understand neural networks and regularization but may be new to factor model language. We translate all concepts into terms you'll recognize: matrix factorization, multicollinearity, regularization, and cross-validation.

---

## Module 14.1: Combining Multiple Alpha Signals

### The Signal Combination Problem

Imagine you've built 50 different alpha signals. Each returns a score for each stock each day:
- Signal 1: Momentum (recent return trends)
- Signal 2: Value (price-to-book ratio)
- Signal 3: Quality (profitability metrics)
- Signal 4: Sentiment (news-based)
- ...
- Signal 50: Custom deep learning model output

Each signal gives you a vector of predictions across all stocks. The question: how do you combine them into a single portfolio?

**The Finance Perspective**: You have a vector of "factor exposures" (think: features in an ML model) and want to find the optimal weights (think: feature importance) to maximize some objective.

**The ML Perspective**: This is ensemble learning. You're combining weak learners (individual signals) into a stronger composite signal.

### 14.1.1 Method 1: Equal-Weight Averaging

The simplest approach: treat all signals equally.

$$\text{Combined Signal}_t = \frac{1}{N} \sum_{i=1}^{N} S_{i,t}$$

where:
- $S_{i,t}$ is signal $i$ at time $t$ (a vector across all stocks)
- $N$ is the number of signals

**Why it works**: If all signals have positive IC (Information Coefficient—correlation with future returns), averaging reduces idiosyncratic noise without canceling signal.

**Why it fails**: Some signals are much stronger than others. Weighting a strong signal equally to noise wastes capital on the noisy signal.

**When to use it**: Quick baseline; all signals have similar strength; you lack historical data to estimate weights.

```python
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SignalCombinationConfig:
    """Configuration for signal combination methods."""
    method: str  # 'equal_weight', 'ic_weight', 'regression', 'orthogonal', 'dynamic'
    lookback_days: int = 252  # Historical period for weight estimation
    min_observations: int = 50
    regularization_strength: float = 0.0
    
class EqualWeightCombiner:
    """Combine signals using equal weights."""
    
    def __init__(self, signals: Dict[str, np.ndarray]):
        """
        Initialize with a dictionary of signals.
        
        Args:
            signals: Dict[signal_name, (n_assets, n_timeperiods)] array
                    Each signal is normalized to [0, 1] or [-1, 1] range
        """
        self.signals = signals
        self.signal_names = list(signals.keys())
        self.n_signals = len(signals)
        
    def combine(self) -> np.ndarray:
        """
        Combine signals using equal weights.
        
        Returns:
            (n_assets, n_timeperiods) combined signal array
        """
        signal_arrays = np.array([self.signals[name] for name in self.signal_names])
        combined = np.mean(signal_arrays, axis=0)
        return combined
    
    def get_weights(self) -> Dict[str, float]:
        """Return the weights applied to each signal."""
        return {name: 1.0 / self.n_signals for name in self.signal_names}
```

### 14.1.2 Method 2: Information Coefficient (IC) Weighted Combination

Better idea: weight each signal by how well it predicts returns.

**Information Coefficient** is the correlation between a signal and forward returns:

$$IC_i = \text{Corr}(S_{i,t}, R_{t+\tau})$$

where $R_{t+\tau}$ is the stock return over the prediction horizon (e.g., next 5 days).

Compute IC over a historical lookback window, then use it as a weight:

$$\text{Combined Signal}_t = \frac{\sum_{i=1}^{N} IC_i \cdot S_{i,t}}{\sum_{i=1}^{N} |IC_i|}$$

The denominator normalizes weights to sum to 1 (approximately). We use absolute value to handle negative IC (inverse signals).

**Why it's better**: Directly rewards signals that have historically predicted returns. Simple, interpretable, and works well in practice.

**Caveat**: IC varies over time. The historical IC you measured may not persist. Later in this chapter, we'll handle this with dynamic weighting.

```python
class ICWeightedCombiner:
    """Combine signals weighted by historical Information Coefficient."""
    
    def __init__(self, signals: Dict[str, np.ndarray], 
                 forward_returns: np.ndarray,
                 lookback_days: int = 252):
        """
        Initialize with signals and forward returns.
        
        Args:
            signals: Dict[signal_name, (n_assets, n_timeperiods)]
            forward_returns: (n_assets, n_timeperiods) future returns
            lookback_days: Window for IC calculation
        """
        self.signals = signals
        self.forward_returns = forward_returns
        self.lookback_days = lookback_days
        self.signal_names = list(signals.keys())
        self.ics = self._compute_ics()
        self.weights = self._compute_weights()
        
    def _compute_ics(self) -> Dict[str, float]:
        """
        Compute Information Coefficient for each signal.
        
        Returns:
            Dict[signal_name, ic_value]
        """
        ics = {}
        
        # Use last lookback_days of data
        t_start = max(0, self.forward_returns.shape[1] - self.lookback_days)
        
        for signal_name in self.signal_names:
            signal = self.signals[signal_name]
            
            # Flatten across assets and time for correlation
            signal_flat = signal[:, t_start:].flatten()
            returns_flat = self.forward_returns[:, t_start:].flatten()
            
            # Remove NaNs
            mask = ~(np.isnan(signal_flat) | np.isnan(returns_flat))
            signal_clean = signal_flat[mask]
            returns_clean = returns_flat[mask]
            
            if len(signal_clean) >= 10:
                ic = np.corrcoef(signal_clean, returns_clean)[0, 1]
            else:
                ic = 0.0
            
            ics[signal_name] = ic if not np.isnan(ic) else 0.0
        
        return ics
    
    def _compute_weights(self) -> Dict[str, float]:
        """
        Convert ICs to weights.
        
        Returns:
            Dict[signal_name, weight]
        """
        # Use absolute IC (to handle negative signals) scaled by sign
        ic_sum = sum(abs(ic) for ic in self.ics.values())
        
        if ic_sum == 0:
            # Fallback to equal weight if all ICs are zero
            return {name: 1.0 / len(self.signal_names) 
                    for name in self.signal_names}
        
        weights = {}
        for signal_name in self.signal_names:
            ic = self.ics[signal_name]
            weights[signal_name] = ic / ic_sum
        
        return weights
    
    def combine(self) -> np.ndarray:
        """
        Combine signals using IC-weighted approach.
        
        Returns:
            (n_assets, n_timeperiods) combined signal
        """
        n_assets, n_times = self.signals[self.signal_names[0]].shape
        combined = np.zeros((n_assets, n_times))
        
        for signal_name in self.signal_names:
            combined += self.weights[signal_name] * self.signals[signal_name]
        
        return combined
    
    def get_info(self) -> pd.DataFrame:
        """Return a DataFrame of ICs and weights for each signal."""
        return pd.DataFrame({
            'IC': self.ics,
            'Weight': self.weights
        })
```

### 14.1.3 Method 3: Regression-Based Combination

Now think like an ML engineer. We have:
- **Input**: Multiple signals $S_1, S_2, \ldots, S_N$ (features)
- **Target**: Forward returns (labels)
- **Goal**: Learn optimal weights via regression

This is linear regression where each observation is an (asset, time) pair:

$$R_{t+\tau} = \beta_0 + \sum_{i=1}^{N} \beta_i S_{i,t} + \epsilon$$

The regression coefficients $\beta_i$ are the optimal weights. Unlike IC weighting, regression accounts for correlations between signals (like how neural networks account for feature interactions).

**To avoid overfitting**, we use regularization (L2/Ridge). The regularized objective is:

$$\min_{\beta} \left\| R - X\beta \right\|_2^2 + \lambda \|\beta\|_2^2$$

where:
- $X$ is the matrix of stacked signals
- $R$ is the vector of forward returns
- $\lambda$ controls regularization strength

Higher $\lambda$ means stronger penalty on large coefficients, leading to simpler models.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class RegressionCombiner:
    """Combine signals using regression with regularization."""
    
    def __init__(self, signals: Dict[str, np.ndarray],
                 forward_returns: np.ndarray,
                 alpha: float = 1.0):
        """
        Initialize regression-based combiner.
        
        Args:
            signals: Dict[signal_name, (n_assets, n_timeperiods)]
            forward_returns: (n_assets, n_timeperiods) future returns
            alpha: Regularization strength (higher = more regularization)
        """
        self.signals = signals
        self.forward_returns = forward_returns
        self.alpha = alpha
        self.signal_names = list(signals.keys())
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.weights = {}
        
        self._fit()
    
    def _fit(self):
        """Fit the regression model to historical data."""
        # Stack signals into feature matrix
        signal_arrays = np.array([self.signals[name] for name in self.signal_names])
        n_assets, n_times, n_signals = signal_arrays.shape[0], signal_arrays.shape[1], len(self.signal_names)
        
        # Reshape to (n_samples, n_signals)
        X = signal_arrays.T  # (n_signals, n_assets, n_times) -> (n_assets, n_times, n_signals)
        X = X.reshape(-1, n_signals)
        y = self.forward_returns.flatten()
        
        # Remove NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Standardize features for better regularization
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit ridge regression
        self.model.fit(X_scaled, y_clean)
        
        # Extract weights
        for i, name in enumerate(self.signal_names):
            self.weights[name] = float(self.model.coef_[i])
    
    def combine(self) -> np.ndarray:
        """
        Combine signals using regression coefficients.
        
        Returns:
            (n_assets, n_timeperiods) combined signal
        """
        signal_arrays = np.array([self.signals[name] for name in self.signal_names])
        n_assets, n_times, n_signals = signal_arrays.shape[0], signal_arrays.shape[1], len(self.signal_names)
        
        X = signal_arrays.T.reshape(-1, n_signals)
        X_scaled = self.scaler.transform(X)
        
        combined_flat = self.model.predict(X_scaled)
        combined = combined_flat.reshape(n_assets, n_times)
        
        return combined
    
    def get_weights(self) -> Dict[str, float]:
        """Return regression coefficients."""
        return self.weights.copy()
```

### 14.1.4 Method 4: Orthogonalization (Signal Decorrelation)

Problem: If two signals are highly correlated, they contain redundant information. Regression accounts for this, but you can be more explicit.

**Idea**: Orthogonalize signals via Gram-Schmidt process or PCA.

Given signals $S_1, S_2, \ldots, S_N$, create orthogonal versions:

1. $O_1 = S_1$
2. $O_2 = S_2 - \frac{\langle S_2, O_1 \rangle}{\langle O_1, O_1 \rangle} O_1$
3. $O_3 = S_3 - \frac{\langle S_3, O_1 \rangle}{\langle O_1, O_1 \rangle} O_1 - \frac{\langle S_3, O_2 \rangle}{\langle O_2, O_2 \rangle} O_2$
4. And so on...

where $\langle \cdot, \cdot \rangle$ is the inner product (correlation).

Then weight and combine the orthogonal signals. This ensures that each signal contributes unique, non-redundant information.

```python
class OrthogonalizingCombiner:
    """Combine signals after orthogonalization to remove redundancy."""
    
    def __init__(self, signals: Dict[str, np.ndarray],
                 forward_returns: np.ndarray,
                 weight_method: str = 'ic'):
        """
        Initialize orthogonalizing combiner.
        
        Args:
            signals: Dict[signal_name, (n_assets, n_timeperiods)]
            forward_returns: (n_assets, n_timeperiods) future returns
            weight_method: 'ic' for IC-weighted, 'equal' for equal-weight
        """
        self.signals = signals
        self.forward_returns = forward_returns
        self.signal_names = list(signals.keys())
        self.weight_method = weight_method
        
        self.orthogonal_signals = self._orthogonalize()
        self.weights = self._compute_weights()
    
    def _orthogonalize(self) -> Dict[str, np.ndarray]:
        """
        Orthogonalize signals using modified Gram-Schmidt.
        
        Returns:
            Dict[signal_name, orthogonal_signal]
        """
        orthogonal = {}
        
        # Flatten signals for easier computation
        signals_flat = {name: self.signals[name].flatten() for name in self.signal_names}
        
        ordered_names = self.signal_names
        
        for i, name_i in enumerate(ordered_names):
            # Start with the original signal
            ortho_signal = signals_flat[name_i].copy()
            
            # Subtract projections onto previously orthogonalized signals
            for j in range(i):
                name_j = ordered_names[j]
                
                # Compute projection: <s_i, o_j> / <o_j, o_j> * o_j
                numerator = np.dot(signals_flat[name_i], orthogonal[name_j])
                denominator = np.dot(orthogonal[name_j], orthogonal[name_j])
                
                if denominator > 1e-10:
                    projection = (numerator / denominator) * orthogonal[name_j]
                    ortho_signal -= projection
            
            # Normalize
            norm = np.linalg.norm(ortho_signal)
            if norm > 1e-10:
                ortho_signal /= norm
            
            orthogonal[name_i] = ortho_signal
        
        # Reshape back to original dimensions
        n_assets, n_times = self.signals[self.signal_names[0]].shape
        orthogonal_reshaped = {}
        for name in self.signal_names:
            orthogonal_reshaped[name] = orthogonal[name].reshape(n_assets, n_times)
        
        return orthogonal_reshaped
    
    def _compute_weights(self) -> Dict[str, float]:
        """Compute weights for orthogonal signals."""
        if self.weight_method == 'ic':
            combiner = ICWeightedCombiner(self.orthogonal_signals, 
                                         self.forward_returns)
            return combiner.get_weights()
        else:
            return {name: 1.0 / len(self.signal_names) 
                    for name in self.signal_names}
    
    def combine(self) -> np.ndarray:
        """
        Combine orthogonalized signals.
        
        Returns:
            (n_assets, n_timeperiods) combined signal
        """
        n_assets, n_times = self.signals[self.signal_names[0]].shape
        combined = np.zeros((n_assets, n_times))
        
        for name in self.signal_names:
            combined += self.weights[name] * self.orthogonal_signals[name]
        
        return combined
```

### 14.1.5 Method 5: Dynamic Weighting (Adaptive Combination)

The real market is non-stationary. A signal that worked last month may fail this month. Solution: adapt weights dynamically.

Compute rolling IC over a window (e.g., 60 days), then reweight every day based on recent performance.

$$w_i(t) = \text{RollingIC}_i(t) / \sum_j |\text{RollingIC}_j(t)|$$

where $\text{RollingIC}_i(t)$ is the IC of signal $i$ computed over the last 60 days ending at time $t$.

This is adaptive ensemble learning—similar to how boosting adapts weights based on training performance.

```python
class DynamicWeightCombiner:
    """Combine signals with dynamic weights based on recent IC."""
    
    def __init__(self, signals: Dict[str, np.ndarray],
                 forward_returns: np.ndarray,
                 ic_window: int = 60,
                 min_observations: int = 20):
        """
        Initialize dynamic weight combiner.
        
        Args:
            signals: Dict[signal_name, (n_assets, n_timeperiods)]
            forward_returns: (n_assets, n_timeperiods) future returns
            ic_window: Days to use for rolling IC computation
            min_observations: Minimum observations required to estimate IC
        """
        self.signals = signals
        self.forward_returns = forward_returns
        self.signal_names = list(signals.keys())
        self.ic_window = ic_window
        self.min_observations = min_observations
        
        # Compute time-varying ICs
        self.rolling_ics = self._compute_rolling_ics()
    
    def _compute_rolling_ics(self) -> Dict[str, np.ndarray]:
        """
        Compute rolling Information Coefficient for each signal.
        
        Returns:
            Dict[signal_name, (n_timeperiods,) rolling IC array]
        """
        n_assets, n_times = self.signals[self.signal_names[0]].shape
        rolling_ics = {}
        
        for signal_name in self.signal_names:
            ics = np.full(n_times, np.nan)
            signal = self.signals[signal_name]
            
            for t in range(self.ic_window, n_times):
                # Compute IC over the window [t - ic_window, t]
                signal_window = signal[:, t - self.ic_window:t]
                returns_window = self.forward_returns[:, t - self.ic_window:t]
                
                signal_flat = signal_window.flatten()
                returns_flat = returns_window.flatten()
                
                mask = ~(np.isnan(signal_flat) | np.isnan(returns_flat))
                if mask.sum() >= self.min_observations:
                    ic = np.corrcoef(signal_flat[mask], returns_flat[mask])[0, 1]
                    ics[t] = ic if not np.isnan(ic) else 0.0
                else:
                    ics[t] = 0.0
            
            rolling_ics[signal_name] = ics
        
        return rolling_ics
    
    def get_weights_at_time(self, t: int) -> Dict[str, float]:
        """
        Get weights at a specific time t.
        
        Args:
            t: Time index
            
        Returns:
            Dict[signal_name, weight]
        """
        weights = {}
        ic_sum = 0.0
        
        for signal_name in self.signal_names:
            ic = self.rolling_ics[signal_name][t]
            if not np.isnan(ic):
                ic_sum += abs(ic)
        
        if ic_sum == 0:
            return {name: 1.0 / len(self.signal_names) 
                    for name in self.signal_names}
        
        for signal_name in self.signal_names:
            ic = self.rolling_ics[signal_name][t]
            weights[signal_name] = ic / ic_sum if not np.isnan(ic) else 0.0
        
        return weights
    
    def combine(self) -> np.ndarray:
        """
        Combine signals with time-varying weights.
        
        Returns:
            (n_assets, n_timeperiods) combined signal
        """
        n_assets, n_times = self.signals[self.signal_names[0]].shape
        combined = np.zeros((n_assets, n_times))
        
        for t in range(n_times):
            weights_t = self.get_weights_at_time(t)
            
            for signal_name in self.signal_names:
                combined[:, t] += weights_t[signal_name] * self.signals[signal_name][:, t]
        
        return combined
    
    def get_rolling_ics_df(self) -> pd.DataFrame:
        """Return rolling ICs as a DataFrame."""
        return pd.DataFrame(self.rolling_ics)
```

### 14.1.6 Practical Framework: SignalCombinationPipeline

In production, you want flexibility. Here's a unified framework that lets you swap combination methods easily.

```python
class SignalCombinationPipeline:
    """Unified framework for combining signals with pluggable methods."""
    
    def __init__(self, config: SignalCombinationConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: SignalCombinationConfig object
        """
        self.config = config
        self.combiner = None
        self.combined_signal = None
    
    def fit_and_combine(self, signals: Dict[str, np.ndarray],
                       forward_returns: np.ndarray) -> np.ndarray:
        """
        Fit the combination method and return combined signal.
        
        Args:
            signals: Dict[signal_name, (n_assets, n_timeperiods)]
            forward_returns: (n_assets, n_timeperiods) future returns
            
        Returns:
            (n_assets, n_timeperiods) combined signal
        """
        if self.config.method == 'equal_weight':
            self.combiner = EqualWeightCombiner(signals)
        
        elif self.config.method == 'ic_weight':
            self.combiner = ICWeightedCombiner(
                signals, forward_returns,
                lookback_days=self.config.lookback_days
            )
        
        elif self.config.method == 'regression':
            self.combiner = RegressionCombiner(
                signals, forward_returns,
                alpha=self.config.regularization_strength
            )
        
        elif self.config.method == 'orthogonal':
            self.combiner = OrthogonalizingCombiner(
                signals, forward_returns,
                weight_method='ic'
            )
        
        elif self.config.method == 'dynamic':
            self.combiner = DynamicWeightCombiner(
                signals, forward_returns,
                ic_window=self.config.lookback_days
            )
        
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        self.combined_signal = self.combiner.combine()
        return self.combined_signal
    
    def get_weights(self) -> Dict[str, float]:
        """Get the weights applied by the combiner."""
        if hasattr(self.combiner, 'get_weights'):
            return self.combiner.get_weights()
        return {}
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostic information about the combination."""
        diagnostics = {
            'method': self.config.method,
            'weights': self.get_weights(),
        }
        
        if hasattr(self.combiner, 'get_info'):
            diagnostics['info'] = self.combiner.get_info()
        
        if hasattr(self.combiner, 'rolling_ics'):
            diagnostics['rolling_ics'] = self.combiner.get_rolling_ics_df()
        
        return diagnostics
```

---

## Module 14.2: Signal Processing

Now you have a combined signal—a single score for each stock at each time. But this raw signal is messy. It may be in arbitrary units, contain noise, have unintended exposures to sector or market movements, and decay slowly to new information.

This module teaches you to process that raw signal into trading-ready position sizes.

### 14.2.1 Signal Smoothing

**Problem**: Raw signals jump around due to noise. Overnight news, data revisions, or computational artifacts can cause large spikes that lead to whipsaw trades.

**Solution**: Smooth the signal over time using exponential moving average (EMA).

$$S_{\text{smooth}}(t) = \alpha \cdot S_{\text{raw}}(t) + (1 - \alpha) \cdot S_{\text{smooth}}(t-1)$$

where $\alpha$ is the smoothing parameter (typically 0.05 to 0.2).

Higher $\alpha$ means more responsive to recent data; lower $\alpha$ means more historical weight.

The half-life of the EMA is:

$$\text{Half-life} = \frac{\ln(0.5)}{\ln(1 - \alpha)} \approx \frac{0.693}{\alpha}$$

For example, $\alpha = 0.1$ gives a half-life of ~7 days.

```python
class SignalSmoother:
    """Smooth signals to reduce noise."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize smoother.
        
        Args:
            alpha: EMA smoothing parameter (0 to 1)
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
    
    @property
    def half_life(self) -> float:
        """Half-life of the EMA in periods."""
        return np.log(0.5) / np.log(1 - self.alpha)
    
    def smooth(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing to a signal.
        
        Args:
            signal: (n_assets, n_timeperiods) raw signal
            
        Returns:
            (n_assets, n_timeperiods) smoothed signal
        """
        n_assets, n_times = signal.shape
        smoothed = np.zeros_like(signal)
        
        for t in range(n_times):
            if t == 0:
                smoothed[:, t] = signal[:, t]
            else:
                smoothed[:, t] = self.alpha * signal[:, t] + (1 - self.alpha) * smoothed[:, t - 1]
        
        return smoothed
    
    def smooth_with_nans(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing, handling NaN values gracefully.
        
        Args:
            signal: (n_assets, n_timeperiods) raw signal with possible NaNs
            
        Returns:
            (n_assets, n_timeperiods) smoothed signal
        """
        n_assets, n_times = signal.shape
        smoothed = np.zeros_like(signal)
        
        for i in range(n_assets):
            for t in range(n_times):
                if t == 0:
                    smoothed[i, t] = signal[i, t]
                else:
                    if np.isnan(signal[i, t]):
                        # If signal is NaN, carry forward previous smoothed value
                        smoothed[i, t] = smoothed[i, t - 1]
                    else:
                        smoothed[i, t] = (self.alpha * signal[i, t] + 
                                         (1 - self.alpha) * smoothed[i, t - 1])
        
        return smoothed
```

### 14.2.2 Signal Standardization (Z-Scoring)

**Problem**: Different signals have different scales. One signal might range from -10 to 10, another from 0 to 1. When combined, the larger-scale signal dominates.

**Solution**: Standardize each signal cross-sectionally (across assets at each time) using z-scoring.

At each time $t$, for each asset $i$:

$$Z_{i,t} = \frac{S_{i,t} - \mu_t}{\sigma_t}$$

where:
- $\mu_t = \frac{1}{N} \sum_i S_{i,t}$ is the mean signal across assets at time $t$
- $\sigma_t = \sqrt{\frac{1}{N} \sum_i (S_{i,t} - \mu_t)^2}$ is the standard deviation

This puts all signals on the same scale: ~0 mean, unit variance.

**Why "cross-sectional"?** We standardize within each time period, not across time. This ensures the signal remains stationary and interpretable as "relative ranking of stocks."

```python
class SignalStandardizer:
    """Standardize signals cross-sectionally (z-score within each time period)."""
    
    def __init__(self, method: str = 'zscore', min_observations: int = 10):
        """
        Initialize standardizer.
        
        Args:
            method: 'zscore' or 'rank' (rank-based standardization)
            min_observations: Minimum assets required to standardize
        """
        self.method = method
        self.min_observations = min_observations
    
    def standardize(self, signal: np.ndarray) -> np.ndarray:
        """
        Standardize signal cross-sectionally.
        
        Args:
            signal: (n_assets, n_timeperiods) raw signal
            
        Returns:
            (n_assets, n_timeperiods) standardized signal
        """
        n_assets, n_times = signal.shape
        standardized = np.zeros_like(signal)
        
        for t in range(n_times):
            signal_t = signal[:, t]
            
            # Remove NaNs for standardization
            mask = ~np.isnan(signal_t)
            
            if mask.sum() < self.min_observations:
                # Not enough observations, return NaNs
                standardized[:, t] = np.nan
                continue
            
            if self.method == 'zscore':
                mean_t = np.nanmean(signal_t)
                std_t = np.nanstd(signal_t)
                
                if std_t > 1e-10:
                    standardized[:, t] = (signal_t - mean_t) / std_t
                else:
                    # No variation, return zeros
                    standardized[:, t] = 0.0
            
            elif self.method == 'rank':
                # Rank-based standardization: convert to ranks, then to z-scores
                ranks = np.full(n_assets, np.nan)
                valid_indices = np.where(mask)[0]
                valid_signals = signal_t[valid_indices]
                
                # Rank from 1 to N
                sorted_indices = np.argsort(valid_signals)
                ranks[valid_indices[sorted_indices]] = np.arange(1, mask.sum() + 1)
                
                # Convert ranks to z-scores
                # Rank runs from 1 to N, so mean is (N+1)/2, variance is (N^2-1)/12
                n = mask.sum()
                rank_mean = (n + 1) / 2
                rank_std = np.sqrt((n**2 - 1) / 12)
                standardized[:, t] = (ranks - rank_mean) / rank_std
        
        return standardized
```

### 14.2.3 Signal Neutralization (Exposure Removal)

**Problem**: Your signal might accidentally contain market exposure. For example, if all tech stocks have high signals, you're implicitly long tech. This is "style drift"—you're no longer pure alpha, you're running a sector bet.

**Solution**: Neutralize the signal with respect to unwanted factors (sector, size, market beta).

Let $F_k$ be a factor (e.g., sector exposure, market beta), and $S$ be your signal. You want to remove the part of $S$ that's correlated with $F_k$.

Linear projection: For each factor $F_k$, compute the projection of $S$ onto $F_k$:

$$\text{Projection}_k = \frac{\langle S, F_k \rangle}{\langle F_k, F_k \rangle} F_k$$

Then subtract all projections:

$$S_{\text{neutral}} = S - \sum_k \text{Projection}_k$$

This is equivalent to the "residual" from multi-factor linear regression of $S$ onto the factors.

```python
class SignalNeutralizer:
    """Remove unwanted factor exposures from signals."""
    
    def __init__(self, factors: Dict[str, np.ndarray]):
        """
        Initialize neutralizer with factor exposures.
        
        Args:
            factors: Dict[factor_name, (n_assets, n_timeperiods)]
                    Common factors: market beta, sector dummies, size
        """
        self.factors = factors
        self.factor_names = list(factors.keys())
    
    def neutralize(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove factor exposures from signal.
        
        Args:
            signal: (n_assets, n_timeperiods) raw signal
            
        Returns:
            (n_assets, n_timeperiods) neutralized signal
        """
        n_assets, n_times = signal.shape
        neutral = signal.copy()
        
        for t in range(n_times):
            signal_t = signal[:, t]
            
            # Build factor matrix for this time period
            X_factors = np.column_stack([self.factors[name][:, t] 
                                         for name in self.factor_names])
            
            # Remove NaNs
            mask = ~(np.isnan(signal_t) | np.isnan(X_factors).any(axis=1))
            
            if mask.sum() < len(self.factor_names) + 1:
                # Not enough observations
                neutral[:, t] = signal_t
                continue
            
            signal_clean = signal_t[mask]
            X_clean = X_factors[mask]
            
            # Fit linear regression: signal ~ factors
            # Compute residuals: signal - predicted
            try:
                # Using numpy's least squares solver
                coeffs = np.linalg.lstsq(X_clean, signal_clean, rcond=None)[0]
                predicted = X_clean @ coeffs
                residuals = signal_clean - predicted
                
                # Assign residuals back
                neutral[mask, t] = residuals
                neutral[~mask, t] = np.nan
            except np.linalg.LinAlgError:
                # Singular matrix, skip neutralization for this time
                pass
        
        return neutral
```

### 14.2.4 Signal Decay Modeling

**Problem**: New information decays. Today's news is important, but it becomes stale as time passes and the market incorporates it. Your strategy should trade harder on fresh information.

**Solution**: Apply a decay function to the signal, reducing it exponentially as it ages.

One approach: **Signal decay via halflife**. Define a halflife (e.g., 5 days) and apply exponential decay:

$$S_{\text{decayed}}(t) = S(t) \cdot 2^{-\Delta t / \text{halflife}}$$

where $\Delta t$ is the age of the signal in days.

Alternatively, use an **exponential decay vector** to weight historical signals:

$$\text{Decay}_{\text{vector}} = e^{-\lambda \cdot t}$$

for $t = 0, 1, 2, \ldots$ (where $t=0$ is today).

```python
class SignalDecayModel:
    """Model signal decay over time."""
    
    def __init__(self, halflife_days: float = 5):
        """
        Initialize decay model.
        
        Args:
            halflife_days: Days for signal to decay to 50% intensity
        """
        self.halflife_days = halflife_days
        self.decay_constant = np.log(2) / halflife_days
    
    def apply_decay(self, signal: np.ndarray, age_array: np.ndarray) -> np.ndarray:
        """
        Apply exponential decay to signal based on age.
        
        Args:
            signal: (n_assets, n_timeperiods) raw signal
            age_array: (n_assets, n_timeperiods) age of signal in days
                      Can be computed from data freshness timestamps
            
        Returns:
            (n_assets, n_timeperiods) decayed signal
        """
        decay_factors = np.exp(-self.decay_constant * age_array)
        decayed = signal * decay_factors
        return decayed
    
    def get_decay_vector(self, lookback_days: int) -> np.ndarray:
        """
        Get decay weights for the last N days.
        
        Args:
            lookback_days: Number of past days
            
        Returns:
            (lookback_days,) array of decay weights
                Weights are in reverse chronological order:
                [most recent (weight ~1), ..., oldest (weight << 1)]
        """
        days = np.arange(lookback_days)
        weights = np.exp(-self.decay_constant * days)
        return weights[::-1]  # Reverse to chronological order
    
    def aggregate_with_decay(self, signal_history: np.ndarray) -> np.ndarray:
        """
        Aggregate historical signals using decay weights.
        
        Args:
            signal_history: (n_assets, lookback_days)
                           Most recent signal in last column
            
        Returns:
            (n_assets,) aggregated signal
        """
        _, lookback_days = signal_history.shape
        decay_weights = self.get_decay_vector(lookback_days)
        decay_weights /= decay_weights.sum()  # Normalize
        
        aggregated = signal_history @ decay_weights
        return aggregated
```

### 14.2.5 Signal Compression (Mapping to Position Sizes)

**Problem**: Your standardized signal has mean 0 and std 1, ranging roughly from -4 to +4. But position sizes need to be interpretable: +1 = fully long, -1 = fully short, 0 = flat.

**Solution**: Apply a compression function to map signal to position range.

**Option 1: Linear Compression**

Simply clip the signal to the target range:

$$\text{Position} = \text{clip}(\text{Signal}, -1, 1)$$

This is simple but creates "cliffs" at the boundaries (sharp transitions at -1 and +1 don't exist in the signal, but suddenly appear).

**Option 2: Sigmoid Compression**

Use the logistic sigmoid function:

$$\text{Position} = \frac{2}{1 + e^{-\beta \cdot \text{Signal}}} - 1$$

where $\beta$ controls the slope. Higher $\beta$ means steeper transitions (closer to clipping); lower $\beta$ means smoother.

The sigmoid ranges from -1 to +1 and is smooth everywhere. This better reflects that small differences in signal at the extremes should lead to small position differences.

**Option 3: Quantile-Based Compression**

Map signal quantiles to position sizes. For example:
- Bottom decile (10% worst signals) → position = -1
- Second decile → position = -0.8
- ...
- Top decile (10% best signals) → position = +1

This ensures you use the full position range and adapts to the actual signal distribution.

```python
class SignalCompressor:
    """Compress signals to position sizes."""
    
    def __init__(self, method: str = 'sigmoid', target_range: tuple = (-1, 1)):
        """
        Initialize compressor.
        
        Args:
            method: 'linear', 'sigmoid', or 'quantile'
            target_range: (min_position, max_position)
        """
        self.method = method
        self.target_range = target_range
        self.quantile_breaks = None
    
    def compress(self, signal: np.ndarray) -> np.ndarray:
        """
        Compress signal to position sizes.
        
        Args:
            signal: (n_assets, n_timeperiods) standardized signal
            
        Returns:
            (n_assets, n_timeperiods) compressed positions
        """
        if self.method == 'linear':
            return self._compress_linear(signal)
        elif self.method == 'sigmoid':
            return self._compress_sigmoid(signal)
        elif self.method == 'quantile':
            return self._compress_quantile(signal)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compress_linear(self, signal: np.ndarray) -> np.ndarray:
        """Linear clipping compression."""
        min_pos, max_pos = self.target_range
        compressed = np.clip(signal, min_pos, max_pos)
        return compressed
    
    def _compress_sigmoid(self, signal: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Sigmoid compression."""
        min_pos, max_pos = self.target_range
        
        # Standard sigmoid: 1 / (1 + exp(-x))
        sigmoid_val = 1.0 / (1.0 + np.exp(-beta * signal))
        
        # Scale to [min_pos, max_pos]
        compressed = min_pos + (max_pos - min_pos) * sigmoid_val
        
        return compressed
    
    def _compress_quantile(self, signal: np.ndarray, n_quantiles: int = 10) -> np.ndarray:
        """Quantile-based compression."""
        min_pos, max_pos = self.target_range
        n_assets, n_times = signal.shape
        compressed = np.zeros_like(signal)
        
        for t in range(n_times):
            signal_t = signal[:, t]
            mask = ~np.isnan(signal_t)
            
            if mask.sum() < n_quantiles:
                # Not enough observations, fallback to linear
                compressed[:, t] = self._compress_linear(signal_t)
                continue
            
            # Compute quantile breaks
            valid_signals = signal_t[mask]
            quantile_breaks = np.quantile(
                valid_signals, 
                np.linspace(0, 1, n_quantiles + 1)
            )
            
            # Digitize: which quantile bin does each signal fall into?
            bins = np.digitize(signal_t, quantile_breaks) - 1
            bins = np.clip(bins, 0, n_quantiles - 1)
            
            # Map bin to position
            positions = np.linspace(min_pos, max_pos, n_quantiles)
            compressed[:, t] = positions[bins]
        
        return compressed
```

### 14.2.6 Complete Signal Processing Pipeline

Here's a complete pipeline that chains all processing steps together.

```python
@dataclass
class SignalProcessingConfig:
    """Configuration for signal processing pipeline."""
    
    # Smoothing
    smoothing_alpha: float = 0.1
    
    # Standardization
    standardization_method: str = 'zscore'  # 'zscore' or 'rank'
    
    # Neutralization
    neutralize_factors: List[str] = None  # Factor names to neutralize against
    
    # Decay
    decay_halflife_days: float = 5.0
    
    # Compression
    compression_method: str = 'sigmoid'  # 'linear', 'sigmoid', 'quantile'
    target_position_range: tuple = (-1.0, 1.0)


class SignalProcessingPipeline:
    """Complete signal processing pipeline."""
    
    def __init__(self, config: SignalProcessingConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.smoother = SignalSmoother(alpha=config.smoothing_alpha)
        self.standardizer = SignalStandardizer(method=config.standardization_method)
        self.decay_model = SignalDecayModel(halflife_days=config.decay_halflife_days)
        self.compressor = SignalCompressor(
            method=config.compression_method,
            target_range=config.target_position_range
        )
        self.neutralizer = None
    
    def set_neutralization_factors(self, factors: Dict[str, np.ndarray]):
        """Set factors for signal neutralization."""
        self.neutralizer = SignalNeutralizer(factors)
    
    def process(self, signal: np.ndarray,
                age_array: np.ndarray = None) -> np.ndarray:
        """
        Process raw signal into trading positions.
        
        Args:
            signal: (n_assets, n_timeperiods) raw signal
            age_array: (n_assets, n_timeperiods) age of signal in days
                      If None, skips decay modeling
            
        Returns:
            (n_assets, n_timeperiods) trading positions
        """
        # Step 1: Smooth
        smoothed = self.smoother.smooth_with_nans(signal)
        
        # Step 2: Standardize cross-sectionally
        standardized = self.standardizer.standardize(smoothed)
        
        # Step 3: Neutralize (if factors provided)
        if self.neutralizer is not None:
            neutralized = self.neutralizer.neutralize(standardized)
        else:
            neutralized = standardized
        
        # Step 4: Apply decay (if age provided)
        if age_array is not None:
            decayed = self.decay_model.apply_decay(neutralized, age_array)
        else:
            decayed = neutralized
        
        # Step 5: Compress to positions
        positions = self.compressor.compress(decayed)
        
        return positions
    
    def process_batch(self, signals: Dict[str, np.ndarray],
                     factors: Dict[str, np.ndarray] = None,
                     age_array: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Process multiple signals in batch.
        
        Args:
            signals: Dict[signal_name, (n_assets, n_timeperiods)]
            factors: Dict[factor_name, (n_assets, n_timeperiods)] for neutralization
            age_array: (n_assets, n_timeperiods) age array
            
        Returns:
            Dict[signal_name, processed positions]
        """
        if factors is not None:
            self.set_neutralization_factors(factors)
        
        processed = {}
        for signal_name, signal in signals.items():
            processed[signal_name] = self.process(signal, age_array)
        
        return processed
```

### 14.2.7 Production Example: NSE Trading with Zerodha

Here's how you'd integrate signal processing into actual trading on NSE using Zerodha.

```python
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

class NSESignalProcessor:
    """Signal processing for NSE trading via Zerodha."""
    
    def __init__(self, zerodha_api_key: str, zerodha_user_id: str):
        """
        Initialize Zerodha connection.
        
        Args:
            zerodha_api_key: Your Zerodha API key
            zerodha_user_id: Your Zerodha user ID
        """
        self.kite = KiteConnect(api_key=zerodha_api_key)
        # NOTE: User must authenticate separately
        self.user_id = zerodha_user_id
        self.pipeline = SignalProcessingPipeline(SignalProcessingConfig())
    
    def fetch_nse_universe(self) -> List[str]:
        """
        Fetch list of NSE stocks.
        
        Returns:
            List of instrument tokens or ticker symbols
        """
        instruments = self.kite.instruments("NSE")
        # Filter to large-cap stocks for liquidity
        nse_stocks = [instr for instr in instruments 
                     if instr['segment'] == 'NSE' and instr['lot_size'] >= 1]
        return [instr['tradingsymbol'] for instr in nse_stocks[:100]]  # Top 100
    
    def compute_signal(self, stock: str, 
                      signal_dict: Dict[str, float]) -> float:
        """
        Compute processed signal for a single stock.
        
        Args:
            stock: Stock symbol
            signal_dict: Dict of raw signal values
            
        Returns:
            Processed signal value
        """
        # Assume signals are already combined
        combined_signal = sum(signal_dict.values()) / len(signal_dict)
        
        # Create dummy arrays for processing (in practice, use real historical data)
        signal_array = np.array([[combined_signal]])
        
        # Process through pipeline
        processed = self.pipeline.process(signal_array)
        
        return float(processed[0, 0])
    
    def execute_trade(self, stock: str, position: float, order_type: str = "MARKET"):
        """
        Execute a trade on NSE via Zerodha.
        
        Args:
            stock: Stock symbol
            position: Target position size [-1, 1]
            order_type: "MARKET" or "LIMIT"
        """
        # Map position to shares
        # Assume 1.0 position = max quantity allowed
        max_quantity = 100  # Example
        quantity = int(position * max_quantity)
        
        if quantity == 0:
            return None
        
        side = "BUY" if quantity > 0 else "SELL"
        quantity = abs(quantity)
        
        try:
            order_id = self.kite.place_order(
                variety="regular",
                exchange="NSE",
                tradingsymbol=stock,
                transaction_type=side,
                quantity=quantity,
                order_type=order_type,
                price=None if order_type == "MARKET" else 0  # Fetch current price
            )
            return order_id
        except Exception as e:
            print(f"Error executing trade for {stock}: {e}")
            return None
```

---

## Summary and Key Takeaways

**Signal Combination Methods:**

1. **Equal-Weight**: Simplest baseline. Works when all signals are similarly strong.
2. **IC-Weighted**: Weight by historical correlation with returns. Best practical balance of simplicity and performance.
3. **Regression-Based**: Learn optimal weights via ridge regression. Accounts for signal correlations. Requires regularization to avoid overfitting.
4. **Orthogonalization**: Remove redundancy via Gram-Schmidt. Ensures each signal contributes unique information.
5. **Dynamic Weighting**: Adapt weights over time based on rolling IC. Handles regime changes.

**Signal Processing Steps:**

1. **Smoothing**: Reduce noise with EMA. Typical half-life: 5-20 days.
2. **Standardization**: Z-score within each time period. Ensures cross-sectional comparability.
3. **Neutralization**: Remove unwanted factor exposures (sector, size, market beta).
4. **Decay Modeling**: Down-weight stale information. Typical half-life: 3-10 days.
5. **Compression**: Map standardized signal to position sizes using linear, sigmoid, or quantile methods.

**In Practice**: Start with equal-weight or IC-weighted combination (simple and effective), add smoothing and standardization (essential), then gradually add neutralization and decay (if factor data is available and regime changes are significant).

For NSE trading with Zerodha, implement the pipeline as a live monitoring system that computes positions daily and submits orders via the Kite API.

---

## Mathematical Reference

| Concept | Formula | Interpretation |
|---------|---------|-----------------|
| Information Coefficient | $IC = \text{Corr}(S, R_{t+\tau})$ | Signal-return correlation |
| Ridge Regression | $\min_\beta \|R - X\beta\|_2^2 + \lambda\|\beta\|_2^2$ | Regularized linear combination |
| Gram-Schmidt | $O_k = S_k - \sum_{j<k} \text{proj}_{O_j} S_k$ | Orthogonal basis |
| EMA | $S_t = \alpha S_t + (1-\alpha) S_{t-1}$ | Exponential smoothing |
| Z-Score | $Z_t = (S_t - \mu_t) / \sigma_t$ | Cross-sectional standardization |
| Sigmoid | $\sigma(x) = 1 / (1 + e^{-x})$ | Smooth compression function |
| Exponential Decay | $S_{\text{decay}} = S \cdot e^{-\lambda t}$ | Time-decay weighting |

---

## Code Repository Structure

For production implementation:

```
signal_processing/
├── __init__.py
├── combination/
│   ├── __init__.py
│   ├── base.py           # Abstract combiner class
│   ├── equal_weight.py   # EqualWeightCombiner
│   ├── ic_weighted.py    # ICWeightedCombiner
│   ├── regression.py     # RegressionCombiner
│   ├── orthogonal.py     # OrthogonalizingCombiner
│   └── dynamic.py        # DynamicWeightCombiner
├── processing/
│   ├── __init__.py
│   ├── smoother.py       # SignalSmoother
│   ├── standardizer.py   # SignalStandardizer
│   ├── neutralizer.py    # SignalNeutralizer
│   ├── decay.py          # SignalDecayModel
│   └── compressor.py     # SignalCompressor
├── pipeline.py           # Combined pipelines
└── zerodha_integration.py # NSE trading integration
```

---

## References and Further Reading

- Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). "Quality for Price." *Financial Analysts Journal*, 75(1), 10-47.
- Clarke, R., De Silva, H., & Thorley, S. (2016). "Fundamentals of Efficient Factor Investing." *Financial Analysts Journal*, 72(6), 9-26.
- Arnott, R. D., Beck, S. L., Kalesnik, V., & West, J. (2016). "How Can 'Smart Beta' Go Horribly Wrong?" *Research Affiliates Publications*.

---

**End of Chapter 14**

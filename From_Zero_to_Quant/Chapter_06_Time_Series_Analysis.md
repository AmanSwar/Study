# Chapter 6: Time Series Analysis

## Overview

Time series analysis is the mathematical foundation of quantitative trading. While you're expert in ML/deep learning and systems design, financial time series have unique properties that differ fundamentally from standard ML datasets: non-stationarity, volatility clustering, mean reversion, and structural relationships between assets. This chapter builds the mathematical and practical toolkit to handle these challenges.

By the end of this chapter, you'll understand:
- Why prices are non-stationary and why returns matter
- How to test for and remove non-stationarity
- Autoregressive models (AR, MA, ARMA, ARIMA) for mean-reverting series
- Volatility models (ARCH/GARCH) that capture clustering and persistence
- Cointegration for pairs trading
- Kalman filters for adaptive estimation

Unlike textbooks written for economists, we'll emphasize **implementation and intuition for ML engineers**, not econometric notation. We'll use modern Python libraries and show exactly how to deploy these in production systems.

---

## Module 6.1: Stationarity and Unit Roots

### 6.1.1 Stationarity Defined

A time series $\{X_t\}$ is **strictly stationary** if for any time points $t_1, t_2, \ldots, t_n$ and any lag $h$, the joint distribution of $(X_{t_1}, X_{t_2}, \ldots, X_{t_n})$ is identical to $(X_{t_1+h}, X_{t_2+h}, \ldots, X_{t_n+h})$.

In plain language: **the statistical properties of the series don't change over time**.

A weaker condition is **weak (covariance) stationarity**:
1. Mean is constant: $E[X_t] = \mu$ for all $t$
2. Variance is constant: $\text{Var}(X_t) = \sigma^2$ for all $t$
3. Autocovariance depends only on lag: $\text{Cov}(X_t, X_{t-h}) = \gamma(h)$ for all $t$

### 6.1.2 Why Prices Are Non-Stationary

Consider a stock price $P_t$. A random walk model:

$$P_t = P_{t-1} + \epsilon_t$$

where $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ is white noise.

**Problem**: $\text{Var}(P_t) = t \cdot \sigma^2$ grows with time. This violates constant variance—prices are **non-stationary**.

Why this matters: If you fit an AR model to non-stationary prices, you'll get **spurious regression**—apparent relationships that exist only in the data, not in reality. For example, two unrelated random walks might show high correlation purely by chance.

### 6.1.3 Why Returns Are (Approximately) Stationary

Define log returns:

$$r_t = \log\left(\frac{P_t}{P_{t-1}}\right) = \log P_t - \log P_{t-1}$$

For a random walk price: $r_t = \epsilon_t \sim \mathcal{N}(0, \sigma^2)$—white noise, obviously stationary.

**Empirically**, returns of liquid assets are approximately stationary (though with time-varying volatility, which we'll handle with GARCH). This is why **quantitative finance works with returns, not prices**.

### 6.1.4 The Augmented Dickey-Fuller (ADF) Test

#### Theory

The ADF test checks the null hypothesis: "the series has a unit root (is non-stationary)."

Consider the regression:

$$\Delta X_t = \alpha X_{t-1} + \sum_{i=1}^{p} \beta_i \Delta X_{t-i} + \epsilon_t$$

If $\alpha = 0$, then $X_t = X_{t-1} + \text{noise}$—a unit root (random walk).

The test statistic is:

$$\tau = \frac{\hat{\alpha}}{\text{SE}(\hat{\alpha})}$$

Critical values are **non-standard** (Dickey-Fuller distribution), not the usual $t$-distribution.

#### Interpretation

| p-value | Conclusion |
|---------|-----------|
| p < 0.05 | Reject null: series is **stationary** |
| p ≥ 0.05 | Fail to reject: series **likely non-stationary** |

#### Implementation

```python
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def adf_test(series: np.ndarray, name: str = "", autolag: str = "AIC") -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Parameters:
    -----------
    series : np.ndarray
        Time series to test (1D array)
    name : str
        Name for display
    autolag : str
        Method for lag selection: "AIC", "BIC", "t-stat", or None
        
    Returns:
    --------
    dict with test statistic, p-value, critical values, and interpretation
    """
    result = adfuller(series, autolag=autolag, regression='c')
    
    output = {
        'test_name': 'ADF Test',
        'series_name': name,
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'ic_best': result[5],
    }
    
    # Interpretation
    if result[1] <= 0.05:
        output['interpretation'] = 'STATIONARY (reject unit root null)'
    else:
        output['interpretation'] = 'NON-STATIONARY (fail to reject unit root)'
    
    return output

def kpss_test(series: np.ndarray, name: str = "", regression: str = "c") -> Dict:
    """
    KPSS test (complementary to ADF).
    
    Null hypothesis: series IS stationary.
    
    Parameters:
    -----------
    series : np.ndarray
        Time series to test
    name : str
        Name for display
    regression : str
        "c" for constant, "ct" for constant + trend
        
    Returns:
    --------
    dict with test results and interpretation
    """
    result = kpss(series, regression=regression, nlags="auto")
    
    output = {
        'test_name': 'KPSS Test',
        'series_name': name,
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'critical_values': result[3],
    }
    
    # Interpretation
    if result[1] <= 0.05:
        output['interpretation'] = 'NON-STATIONARY (reject stationarity null)'
    else:
        output['interpretation'] = 'STATIONARY (fail to reject stationarity null)'
    
    return output

# Example: Test price vs returns
def test_stationarity_example():
    """Demonstrate ADF and KPSS on price vs returns."""
    
    # Simulate 1000-day stock price (random walk)
    np.random.seed(42)
    daily_returns = np.random.normal(0.0005, 0.02, 1000)
    price = 100 * np.exp(np.cumsum(daily_returns))
    
    # Compute log returns
    log_returns = np.diff(np.log(price))
    
    # Test both
    print("=" * 70)
    print("PRICE (should be non-stationary)")
    print("=" * 70)
    adf_price = adf_test(price, name="Stock Price")
    for key, val in adf_price.items():
        print(f"{key:25s}: {val}")
    
    print("\n" + "=" * 70)
    print("RETURNS (should be stationary)")
    print("=" * 70)
    adf_returns = adf_test(log_returns, name="Log Returns")
    for key, val in adf_returns.items():
        print(f"{key:25s}: {val}")
    
    print("\n" + "=" * 70)
    print("KPSS TEST ON RETURNS (complementary)")
    print("=" * 70)
    kpss_returns = kpss_test(log_returns, name="Log Returns")
    for key, val in kpss_returns.items():
        print(f"{key:25s}: {val}")
    
    return price, log_returns
```

### 6.1.5 Transformations to Achieve Stationarity

#### Differencing

First difference: $\Delta X_t = X_t - X_{t-1}$

For a random walk: $\Delta X_t = \epsilon_t$ (stationary).

Second difference: $\Delta^2 X_t = \Delta X_t - \Delta X_{t-1}$ (rarely needed).

#### Log Differencing

More common in finance. For prices:

$$r_t = \log P_t - \log P_{t-1} = \log\left(\frac{P_t}{P_{t-1}}\right)$$

This is the **continuously compounded return**.

#### Detrending

If a series has a deterministic trend (e.g., a stock that rises $0.1\%$ per day):

$$X_t = \beta_0 + \beta_1 t + u_t$$

Detrend by regression: $\hat{u}_t = X_t - \hat{\beta}_0 - \hat{\beta}_1 t$

```python
def make_stationary(series: np.ndarray, 
                    method: str = "log_diff") -> Tuple[np.ndarray, str]:
    """
    Transform series to approximate stationarity.
    
    Parameters:
    -----------
    series : np.ndarray
        Original series (e.g., price)
    method : str
        "diff" = first differencing
        "log_diff" = log differencing (for positive series)
        "detrend" = remove linear trend
        
    Returns:
    --------
    transformed series, description
    """
    if method == "diff":
        transformed = np.diff(series)
        desc = "First Difference"
    
    elif method == "log_diff":
        if np.any(series <= 0):
            raise ValueError("Series must be positive for log differencing")
        transformed = np.diff(np.log(series))
        desc = "Log Difference (Return)"
    
    elif method == "detrend":
        from scipy import signal
        transformed = signal.detrend(series)
        desc = "Linear Detrending"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return transformed, desc

def visualize_stationarity(original: np.ndarray, 
                          transformed: np.ndarray,
                          title: str = "Price vs Returns"):
    """
    [VISUALIZATION: Compare original and transformed series]
    Shows histograms and time series plots side-by-side.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Time series plots
    axes[0, 0].plot(original, linewidth=1, color='navy')
    axes[0, 0].set_title("Original Series (Non-Stationary)")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(transformed, linewidth=1, color='darkgreen')
    axes[0, 1].set_title("Transformed Series (Stationary)")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histograms
    axes[1, 0].hist(original, bins=50, color='navy', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title("Distribution: Original")
    axes[1, 0].set_xlabel("Value")
    
    axes[1, 1].hist(transformed, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title("Distribution: Transformed")
    axes[1, 1].set_xlabel("Value")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
```

### 6.1.6 Summary: Stationarity Checklist

- [ ] Test with ADF (reject null = stationary) and KPSS (fail to reject = stationary)
- [ ] Use log returns for price series, not prices
- [ ] Visual inspection: does the mean/variance look constant over time?
- [ ] Understand the economic intuition: why should this series be mean-reverting?

---

## Module 6.2: Autoregressive Models (AR, MA, ARMA, ARIMA)

### 6.2.1 AR(p) Model

An autoregressive model of order $p$ assumes current value depends on past $p$ values:

$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \epsilon_t$$

where $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ is white noise.

#### Stationarity Condition

The model is stationary if all roots of the characteristic polynomial lie outside the unit circle:

$$1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p \neq 0 \quad \text{for } |z| \leq 1$$

For AR(1), this is simply $|\phi_1| < 1$.

#### ACF and PACF

- **Autocorrelation Function (ACF)**: $\rho(h) = \text{Corr}(X_t, X_{t-h})$
  
  For AR(p), ACF decays exponentially (no cutoff).

- **Partial Autocorrelation Function (PACF)**: Correlation between $X_t$ and $X_{t-h}$ after removing dependence on intermediate lags.
  
  For AR(p), PACF cuts off after lag $p$ (sharp cutoff).

**Rule of thumb**: If PACF cuts off at lag $p$, try AR(p).

#### Estimation via Yule-Walker

For AR(p), the Yule-Walker equations give:

$$\gamma(h) = \phi_1 \gamma(h-1) + \phi_2 \gamma(h-2) + \cdots + \phi_p \gamma(h-p)$$

for $h = 1, 2, \ldots, p$, where $\gamma(h)$ is the autocovariance.

Solving this system yields $\hat{\phi}_1, \ldots, \hat{\phi}_p$.

### 6.2.2 MA(q) Model

A moving average model of order $q$:

$$X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}$$

#### Invertibility

MA models require **invertibility** to ensure a unique representation:

$$1 + \theta_1 z + \theta_2 z^2 + \cdots + \theta_q z^q \neq 0 \quad \text{for } |z| \leq 1$$

**Implication**: An invertible MA(q) can be represented as an infinite AR.

#### ACF and PACF

- **ACF**: Cuts off after lag $q$ (sharp cutoff)
- **PACF**: Decays exponentially (no cutoff)

**Rule of thumb**: If ACF cuts off at lag $q$, try MA(q).

### 6.2.3 ARMA(p, q) Model

Combines AR and MA:

$$X_t = c + \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}$$

Use ARMA when both ACF and PACF show gradual decay (neither has a sharp cutoff).

### 6.2.4 ARIMA(p, d, q) Model

ARIMA adds **integration**: differencing $d$ times to achieve stationarity.

$$\Delta^d X_t = \text{AR(p) + MA(q)}$$

**Implementation**:

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Tuple

def plot_acf_pacf(series: np.ndarray, lags: int = 40) -> Tuple:
    """
    Plot ACF and PACF to guide (p, d, q) selection.
    
    Parameters:
    -----------
    series : np.ndarray
        Time series (should be stationary for AR/MA selection)
    lags : int
        Number of lags to plot
        
    Returns:
    --------
    (acf_fig, pacf_fig)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)")
    
    plot_pacf(series, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title("Partial Autocorrelation Function (PACF)")
    
    plt.tight_layout()
    return fig

def fit_arima(series: np.ndarray, 
              order: Tuple[int, int, int],
              seasonal: bool = False) -> dict:
    """
    Fit ARIMA model to time series.
    
    Parameters:
    -----------
    series : np.ndarray
        Time series (price or already differenced returns)
    order : tuple
        (p, d, q) - AR order, integration, MA order
    seasonal : bool
        If True, fit SARIMA (seasonal ARIMA)
        
    Returns:
    --------
    Dictionary with fitted model, diagnostics, and forecasting function
    """
    model = ARIMA(series, order=order)
    results = model.fit()
    
    output = {
        'model': model,
        'results': results,
        'aic': results.aic,
        'bic': results.bic,
        'summary': results.summary(),
        'params': results.params.to_dict(),
    }
    
    return output

def grid_search_arima(series: np.ndarray,
                     p_range: range = range(0, 4),
                     d_range: range = range(0, 3),
                     q_range: range = range(0, 4),
                     criterion: str = "aic") -> pd.DataFrame:
    """
    Grid search for best (p, d, q) by AIC or BIC.
    
    Parameters:
    -----------
    series : np.ndarray
        Time series
    p_range, d_range, q_range : range
        Ranges for AR, integration, MA orders
    criterion : str
        "aic" or "bic"
        
    Returns:
    --------
    DataFrame with (p, d, q), AIC/BIC, and convergence status
    """
    results = []
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    
                    crit_val = fitted.aic if criterion == "aic" else fitted.bic
                    
                    results.append({
                        'p': p,
                        'd': d,
                        'q': q,
                        criterion: crit_val,
                        'converged': 'Yes',
                    })
                except:
                    results.append({
                        'p': p,
                        'd': d,
                        'q': q,
                        criterion: np.inf,
                        'converged': 'No',
                    })
    
    df = pd.DataFrame(results)
    df = df.sort_values(criterion)
    return df

def forecast_arima(results, steps: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forecast from fitted ARIMA with prediction intervals.
    
    Parameters:
    -----------
    results : ARIMA results object
    steps : int
        Number of periods ahead to forecast
        
    Returns:
    --------
    (forecast_point, ci_lower, ci_upper)
    """
    forecast_result = results.get_forecast(steps=steps)
    forecast_df = forecast_result.summary_frame()
    
    return (
        forecast_df['mean'].values,
        forecast_df['mean_ci_lower'].values,
        forecast_df['mean_ci_upper'].values,
    )
```

### 6.2.5 Full ARIMA Implementation on Indian Stock Returns

```python
import yfinance as yf

def arima_on_nifty_example():
    """
    Complete ARIMA pipeline on NIFTY 50 index returns.
    """
    
    # Download 5 years of NIFTY50 data
    ticker = "^NSEI"  # NIFTY 50
    nifty = yf.download(ticker, start="2019-01-01", end="2024-01-01", progress=False)
    price = nifty['Adj Close'].values
    
    # Compute log returns
    log_returns = np.diff(np.log(price))
    
    # Test stationarity
    print("Testing log returns for stationarity...")
    adf_result = adf_test(log_returns, name="NIFTY50 Returns")
    print(f"ADF p-value: {adf_result['p_value']:.6f}")
    print(f"Interpretation: {adf_result['interpretation']}\n")
    
    # Plot ACF/PACF
    print("Plotting ACF/PACF to guide (p, q) selection...")
    fig = plot_acf_pacf(log_returns, lags=30)
    fig.savefig('acf_pacf.png', dpi=100, bbox_inches='tight')
    
    # Grid search
    print("Grid searching for best (p, d, q)...")
    grid_results = grid_search_arima(price, 
                                     p_range=range(0, 5),
                                     d_range=range(0, 3),
                                     q_range=range(0, 5),
                                     criterion="aic")
    print("\nTop 10 models by AIC:")
    print(grid_results.head(10))
    
    # Fit best model
    best_p, best_d, best_q = grid_results.iloc[0][['p', 'd', 'q']].astype(int)
    print(f"\nFitting ARIMA({best_p}, {best_d}, {best_q})...")
    
    model = ARIMA(price, order=(best_p, best_d, best_q))
    results = model.fit()
    print(results.summary())
    
    # Forecast
    forecast_points, ci_lower, ci_upper = forecast_arima(results, steps=20)
    
    print(f"\nForecasts (next 20 days):")
    for i, (f, l, u) in enumerate(zip(forecast_points, ci_lower, ci_upper), 1):
        print(f"Day {i:2d}: {f:8.0f} [{l:8.0f}, {u:8.0f}]")
    
    return results, forecast_points, ci_lower, ci_upper

# Run example
# arima_on_nifty_example()
```

### 6.2.6 Limitations in Finance

WARNING: **ARIMA assumes constant conditional variance and no structural breaks.**

1. **Volatility Clustering**: Returns exhibit time-varying volatility (covered by GARCH).
2. **Non-Linear Relationships**: Market structure is non-linear; ARIMA is linear.
3. **Regime Shifts**: Market regimes change; ARIMA can't adapt in real-time.
4. **Forecast Horizon**: ARIMA forecasts converge to the mean; useless beyond 5-10 steps.

For actual trading, ARIMA is a **baseline for comparison**, not a primary strategy.

---

## Module 6.3: Volatility Models (ARCH/GARCH Family)

### 6.3.1 Volatility Clustering

Financial returns exhibit **volatility clustering**: periods of high/low volatility tend to cluster.

```
High volatility period: |big jump| |big drop| |small move| |big jump|
Low volatility period:  |tiny|tiny|tiny| |tiny|tiny|
```

A returns series might have $\sigma_t = 2\%$ on some days and $\sigma_t = 0.5\%$ on others—not constant.

**Implication**: Conditional variance $\text{Var}(r_t | r_{t-1}, r_{t-2}, \ldots)$ changes over time.

### 6.3.2 ARCH(q) Model

**Autoregressive Conditional Heteroskedasticity**: variance depends on past squared residuals.

$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0, 1)$$

$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \alpha_2 \epsilon_{t-2}^2 + \cdots + \alpha_q \epsilon_{t-q}^2$$

**Intuition**: Large past shocks ($\epsilon^2$ large) increase current volatility.

#### Stationarity

For unconditional variance to exist:

$$\sum_{i=1}^{q} \alpha_i < 1$$

### 6.3.3 GARCH(p, q) Model: Full Derivation

**Generalized ARCH** adds lagged variance terms:

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2$$

For **GARCH(1,1)**—the most common:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

#### Intuition

- $\omega$: baseline volatility
- $\alpha \epsilon_{t-1}^2$: reaction to recent shocks (mean reversion)
- $\beta \sigma_{t-1}^2$: persistence (yesterday's volatility influences today)

#### Unconditional Variance

Taking expectations: $E[\sigma_t^2] = \sigma^2$ (constant in steady state):

$$\sigma^2 = \omega + \alpha E[\epsilon^2] + \beta \sigma^2$$

$$\sigma^2(1 - \alpha - \beta) = \omega$$

$$\sigma^2 = \frac{\omega}{1 - \alpha - \beta}$$

**Stationarity requirement**: $\alpha + \beta < 1$

**Persistence parameter**: $\alpha + \beta$. If close to 1, volatility shocks persist for a long time.

#### Maximum Likelihood Estimation (MLE)

Given returns $r_1, \ldots, r_T$, the likelihood is:

$$\mathcal{L}(\theta) = -\frac{1}{2} \sum_{t=1}^{T} \left( \log \sigma_t^2(\theta) + \frac{\epsilon_t^2(\theta)}{\sigma_t^2(\theta)} \right)$$

where $\epsilon_t(\theta) = r_t - \mu$ and $\sigma_t^2(\theta)$ follows the GARCH recursion.

Optimize via numerical methods (BFGS, Nelder-Mead).

#### Interpretation of Fitted GARCH(1,1)

If you estimate $\hat{\alpha} = 0.08, \hat{\beta} = 0.90, \hat{\omega} = 0.00005$:

- A 1% return shock increases next-day volatility by $0.08 \times (0.01)^2 = 0.000008$
- Yesterday's volatility has 90% weight on today's
- Persistence: $0.08 + 0.90 = 0.98$ → shocks decay slowly

### 6.3.4 Extensions: EGARCH, GJR-GARCH

#### EGARCH (Exponential GARCH)

Allows **leverage effect**: negative returns increase volatility more than positive returns.

$$\log \sigma_t^2 = \omega + \sum_{i=1}^{p} \beta_i \log \sigma_{t-i}^2 + \sum_{j=1}^{q} \alpha_j |z_{t-j}| + \sum_{k=1}^{q} \gamma_k z_{t-k}$$

where $z_t = \epsilon_t / \sigma_t$ is the standardized residual.

- If $\gamma < 0$: negative shocks drive volatility more (leverage effect)
- Non-negativity of $\sigma^2$ is automatically satisfied

#### GJR-GARCH

**Glosten-Jagannathan-Runkle** variant with indicator for negative returns:

$$\sigma_t^2 = \omega + (\alpha + \gamma I_{t-1}) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where $I_{t-1} = 1$ if $\epsilon_{t-1} < 0$, else $0$.

This captures asymmetry cheaply: $\gamma > 0$ means negative shocks have larger impact.

### 6.3.5 Volatility Forecasting

#### 1-Step Ahead (In-Sample)

From GARCH(1,1) fitted to past data:

$$\sigma_{t+1|t}^2 = \omega + \alpha \epsilon_t^2 + \beta \sigma_t^2$$

Use observed $\epsilon_t$ (fitted residuals) and $\sigma_t$ (filtered volatility).

#### Multi-Step Ahead (Out-of-Sample)

For $h > 1$:

$$E_t[\sigma_{t+h}^2] = \omega + (\alpha + \beta) E_t[\sigma_{t+h-1}^2]$$

This is a simple recursion. Multi-step forecasts converge to the unconditional mean $\sigma^2$.

#### Volatility Term Structure

Plotting $\sigma_{t+h}$ for different $h$ shows how market expectations of volatility evolve.

### 6.3.6 Realized vs Implied vs GARCH Volatility

| Type | Definition | Usage |
|------|-----------|-------|
| **Realized** | $\sqrt{\frac{1}{N} \sum_{i=1}^{N} r_i^2}$ | Historical measure, computation check |
| **Implied** | From option prices (Black-Scholes inverse) | Market's forward-looking estimate |
| **GARCH** | Conditional volatility from GARCH model | Forecast for risk management |

**Comparison**: GARCH typically underestimates tail risk compared to realized during crisis periods.

### 6.3.7 Full GARCH Implementation

```python
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal
import warnings
warnings.filterwarnings('ignore')

def fit_garch(returns: np.ndarray,
              p: int = 1,
              q: int = 1,
              model_type: str = "Garch") -> dict:
    """
    Fit GARCH(p, q) model to returns.
    
    Parameters:
    -----------
    returns : np.ndarray
        Log returns (must be in decimal form, e.g., 0.01 for 1%)
    p : int
        GARCH lag order
    q : int
        ARCH lag order
    model_type : str
        "Garch", "EGARCH", "ConstantMean", etc.
        
    Returns:
    --------
    Dictionary with fitted results, parameters, diagnostics
    """
    
    # Convert to percentage for numerical stability
    r_scaled = returns * 100
    
    # Fit model
    model = arch_model(r_scaled, vol='Garch', p=p, q=q, mean='Zero', rescale=False)
    results = model.fit(disp='off')
    
    output = {
        'model': model,
        'results': results,
        'params': {
            'omega': results.params['Garch'].iloc[0],
            'alpha': results.params['Garch'].iloc[1],
            'beta': results.params['Garch'].iloc[2] if p > 0 else 0,
        },
        'conditional_volatility': results.conditional_volatility.values / 100,  # Scale back
        'standardized_residuals': results.std_resid.values,
        'persistence': results.params['Garch'].iloc[1:].sum(),
        'halflife': np.log(0.5) / np.log(results.params['Garch'].iloc[1:].sum()) if results.params['Garch'].iloc[1:].sum() > 0 else np.inf,
    }
    
    return output

def forecast_garch_volatility(results, steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecast conditional volatility using GARCH.
    
    Parameters:
    -----------
    results : arch GARCH results object
    steps : int
        Forecast horizon
        
    Returns:
    --------
    (volatility_forecasts, time_indices)
    """
    forecast = results.forecast(horizon=steps)
    variance_forecast = forecast.variance.values[-1, :]
    volatility_forecast = np.sqrt(variance_forecast) / 100  # Scale back
    
    return volatility_forecast, np.arange(1, steps + 1)

def garch_on_nifty_example():
    """
    Complete GARCH pipeline on NIFTY50 returns.
    """
    
    # Download data
    nifty = yf.download("^NSEI", start="2020-01-01", end="2024-01-01", progress=False)
    returns = np.diff(np.log(nifty['Adj Close'].values))
    
    # Fit GARCH(1,1)
    print("Fitting GARCH(1,1) to NIFTY50 returns...")
    garch_result = fit_garch(returns, p=1, q=1)
    
    print("\nModel Summary:")
    print(garch_result['results'].summary())
    
    print("\n" + "=" * 70)
    print("GARCH(1,1) Parameters:")
    print("=" * 70)
    for param_name, param_val in garch_result['params'].items():
        print(f"{param_name:15s}: {param_val:12.8f}")
    
    print(f"\nPersistence (α+β):         {garch_result['persistence']:12.8f}")
    print(f"Half-life (days):          {garch_result['halflife']:12.1f}")
    print(f"Mean conditional volatility: {np.mean(garch_result['conditional_volatility']):12.4%}")
    
    # Forecast
    vol_forecast, horizons = forecast_garch_volatility(garch_result['results'], steps=20)
    print(f"\n20-step volatility forecasts:")
    for h, v in zip(horizons, vol_forecast):
        print(f"  h={h:2d}: {v:7.4%}")
    
    # [VISUALIZATION: Conditional volatility overlay]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Returns and conditional volatility
    axes[0].plot(returns, linewidth=0.8, color='steelblue', alpha=0.7, label='Daily Returns')
    axes[0].fill_between(range(len(returns)), 
                         -garch_result['conditional_volatility'],
                         garch_result['conditional_volatility'],
                         alpha=0.3, color='red', label='Conditional Volatility (±1σ)')
    axes[0].set_title("NIFTY50 Returns with GARCH Conditional Volatility", fontweight='bold')
    axes[0].set_ylabel("Return / Volatility")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Conditional volatility only
    axes[1].plot(garch_result['conditional_volatility'], linewidth=1, color='darkred')
    axes[1].fill_between(range(len(garch_result['conditional_volatility'])),
                        garch_result['conditional_volatility'],
                        alpha=0.3, color='red')
    axes[1].set_title("GARCH(1,1) Conditional Volatility", fontweight='bold')
    axes[1].set_ylabel("Volatility (σ)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('garch_volatility.png', dpi=100, bbox_inches='tight')
    
    return garch_result
```

### 6.3.8 Diagnostics and Model Checking

```python
def check_garch_diagnostics(results) -> dict:
    """
    Check GARCH model diagnostics via standardized residuals.
    
    Returns:
    --------
    Dictionary with test results
    """
    
    std_resid = results.std_resid.values
    
    # Normality test (Jarque-Bera)
    from scipy.stats import jarque_bera, shapiro
    jb_stat, jb_pval = jarque_bera(std_resid)
    
    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Standardized residuals
    axes[0, 0].plot(std_resid, linewidth=0.8)
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(2, color='r', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(-2, color='r', linestyle='--', alpha=0.3)
    axes[0, 0].set_title("Standardized Residuals")
    axes[0, 0].set_ylabel("z_t")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram
    axes[0, 1].hist(std_resid, bins=50, density=True, alpha=0.7, edgecolor='black')
    # Overlay normal
    from scipy.stats import norm
    x = np.linspace(std_resid.min(), std_resid.max(), 100)
    axes[0, 1].plot(x, norm.pdf(x), 'r-', linewidth=2, label='Normal PDF')
    axes[0, 1].set_title(f"Standardized Residuals (JB p-value: {jb_pval:.4f})")
    axes[0, 1].legend()
    
    # ACF of residuals
    plot_acf(std_resid, lags=30, ax=axes[1, 0])
    axes[1, 0].set_title("ACF of Standardized Residuals")
    
    # ACF of squared residuals (should be white noise if GARCH captures volatility)
    plot_acf(std_resid**2, lags=30, ax=axes[1, 1])
    axes[1, 1].set_title("ACF of Squared Residuals")
    
    plt.tight_layout()
    fig.savefig('garch_diagnostics.png', dpi=100, bbox_inches='tight')
    
    return {
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pval': jb_pval,
        'is_normal': jb_pval > 0.05,
    }
```

---

## Module 6.4: Cointegration and Error Correction Models

### 6.4.1 Cointegration: Definition and Economic Intuition

Two time series $X_t$ and $Y_t$ are **cointegrated** if:

1. Both are individually non-stationary (e.g., I(1): require one differencing to be stationary)
2. **Some linear combination is stationary**: $Z_t = X_t - \beta Y_t \sim \mathcal{I}(0)$ (stationary)

#### Economic Intuition

- $X_t$ and $Y_t$ individually wander around (non-stationary)
- But the **ratio or spread** $X_t - \beta Y_t$ stays bounded

**Example**: Stock A and Stock B are cointegrated if:
- Each price drifts unpredictably
- But their spread oscillates around an equilibrium

This is the foundation of **pairs trading**: profit from mean reversion in the spread.

#### Why It Matters

Spurious regression: If two independent random walks are highly correlated purely by chance, standard regression gives false positives. Cointegration distinguishes real relationships from spurious ones.

### 6.4.2 Engle-Granger Two-Step Test

#### Step 1: Cointegrating Regression

Run OLS on levels:

$$Y_t = \alpha + \beta X_t + u_t$$

Estimate $\hat{\beta}$ (the cointegrating vector).

#### Step 2: Test Residuals for Stationarity

Compute residuals: $\hat{u}_t = Y_t - \hat{\alpha} - \hat{\beta} X_t$

Test $\hat{u}_t$ for stationarity using ADF test.

If ADF p-value $< 0.05$ (or better, $< 0.10$ with cointegration-specific critical values), reject null of non-stationarity → **cointegrated**.

#### Implementation

```python
def engle_granger_test(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Engle-Granger two-step cointegration test.
    
    Parameters:
    -----------
    x, y : np.ndarray
        Two non-stationary time series (assumed I(1))
        
    Returns:
    --------
    Dictionary with cointegrating regression and ADF test on residuals
    """
    
    from scipy.stats import linregress
    
    # Step 1: Cointegrating regression
    slope, intercept, r_value, p_value, se = linregress(x, y)
    
    # Residuals
    residuals = y - intercept - slope * x
    
    # Step 2: ADF test on residuals
    # Note: Use Engle-Granger critical values (not standard ADF)
    # For 5% significance: -3.37 (vs standard -2.86)
    adf_result = adfuller(residuals, autolag='AIC', regression='c')
    
    output = {
        'cointegrating_regression': {
            'intercept': intercept,
            'slope': slope,
            'r_squared': r_value**2,
            'standard_error': se,
        },
        'residuals': residuals,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_critical_values': adf_result[4],
        'is_cointegrated_5pct': adf_result[0] < -3.37,  # EG critical value
        'is_cointegrated_10pct': adf_result[0] < -3.06,
    }
    
    return output

# Example
def cointegration_example():
    """Demonstrate cointegration with synthetic I(1) series."""
    
    np.random.seed(42)
    T = 500
    
    # Common stochastic trend
    epsilon = np.random.normal(0, 1, T)
    trend = np.cumsum(epsilon)
    
    # Two series with same trend (cointegrated)
    x = trend + np.random.normal(0, 0.5, T)
    y = 2 * trend + 3 + np.random.normal(0, 0.5, T)
    
    # Test
    result = engle_granger_test(x, y)
    
    print("=" * 70)
    print("ENGLE-GRANGER COINTEGRATION TEST")
    print("=" * 70)
    print(f"\nCointegrating regression:")
    print(f"  Y = {result['cointegrating_regression']['intercept']:.4f} + {result['cointegrating_regression']['slope']:.4f} * X")
    print(f"  R² = {result['cointegrating_regression']['r_squared']:.4f}")
    
    print(f"\nADF test on residuals:")
    print(f"  ADF statistic: {result['adf_statistic']:.4f}")
    print(f"  p-value: {result['adf_pvalue']:.4f}")
    print(f"  Critical values (5%, EG): -3.37")
    print(f"  Is cointegrated (5%): {result['is_cointegrated_5pct']}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    
    axes[0].plot(x, label='X', linewidth=1, color='navy')
    axes[0].plot(y, label='Y', linewidth=1, color='darkgreen')
    axes[0].set_title("Two Cointegrated Series (Both I(1))", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(result['residuals'], linewidth=1, color='darkred')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title("Spread / Cointegrating Residual (Stationary)", fontweight='bold')
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, alpha=0.3)
    
    # Distribution of spread
    axes[2].hist(result['residuals'], bins=40, edgecolor='black', alpha=0.7, color='darkred')
    axes[2].set_title("Distribution of Spread")
    axes[2].set_xlabel("Spread Value")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('cointegration_example.png', dpi=100, bbox_inches='tight')
    
    return result
```

### 6.4.3 Johansen Test (For Multiple Series)

For $n > 2$ series, use the **Johansen cointegration test**, which determines:
- Number of cointegrating relationships
- Cointegrating vectors (eigenvectors of a matrix constructed from residual correlations)

Implementation is complex; use `statsmodels.tsa.vector_ar.vecm`:

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

def johansen_test_example(data: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """
    Johansen cointegration test for multiple series.
    
    Parameters:
    -----------
    data : np.ndarray
        (T, n) array where T = time, n = number of series
    det_order : int
        0 = no deterministic terms, 1 = constant, 2 = constant + trend
    k_ar_diff : int
        Number of lagged differences (VEC model order)
        
    Returns:
    --------
    Dictionary with trace test results and cointegrating vectors
    """
    
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    
    return {
        'trace_stat': result[0],
        'trace_critical_values': result[1],
        'max_eig_stat': result[2],
        'max_eig_critical_values': result[3],
        'coint_vectors': result[4],
    }
```

### 6.4.4 VECM (Vector Error Correction Model)

Once you identify cointegration, estimate a **VECM** to model:
1. **Long-run equilibrium**: The cointegrating relationship
2. **Short-run dynamics**: How variables adjust to deviations from equilibrium

$$\Delta Y_t = \alpha \beta^T Y_{t-1} + \sum_{j=1}^{p-1} \Gamma_j \Delta Y_{t-j} + u_t$$

where:
- $\beta^T Y_{t-1}$: error correction term (deviation from long-run equilibrium)
- $\alpha$: adjustment speeds (how fast variables revert)
- $\Gamma_j$: short-run dynamics

```python
def fit_vecm_example():
    """Fit VECM to cointegrated pair."""
    
    # Create cointegrated pair
    np.random.seed(42)
    T = 300
    epsilon = np.random.normal(0, 1, T)
    common = np.cumsum(epsilon)
    
    y1 = common + np.random.normal(0, 0.5, T)
    y2 = 2 * common - 3 + np.random.normal(0, 0.5, T)
    
    data = np.column_stack([y1, y2])
    
    # Fit VECM
    model = VECM(data, k_ar_diff=1, coint_rank=1, deterministic='ci')
    results = model.fit()
    
    print(results.summary())
    
    return results
```

### 6.4.5 Pairs Trading: From Cointegration to Strategy

#### Basic Framework

1. **Identify pairs**: Stock A and B are cointegrated
2. **Estimate hedge ratio**: $\beta$ from cointegrating regression
3. **Form spread**: $s_t = A_t - \beta B_t$
4. **Trade**:
   - If $s_t > $ upper threshold: short A, long B (expect mean reversion down)
   - If $s_t < $ lower threshold: long A, short B (expect mean reversion up)

#### Half-Life of Mean Reversion

Estimate how quickly the spread reverts to equilibrium. Fit an AR(1):

$$s_t = \rho s_{t-1} + \epsilon_t$$

Half-life: $\text{HL} = \frac{\log(2)}{-\log(\rho)}$

If HL = 5 days, deviations typically revert within 5 days.

```python
def estimate_halflife_mean_reversion(spread: np.ndarray) -> float:
    """
    Estimate half-life of mean reversion using AR(1).
    
    Parameters:
    -----------
    spread : np.ndarray
        Cointegrating spread (residuals from cointegrating regression)
        
    Returns:
    --------
    Half-life in periods (e.g., days)
    """
    
    # AR(1) regression: spread[t] = rho * spread[t-1] + noise
    y = spread[1:]
    x = spread[:-1]
    
    rho = np.polyfit(x, y, 1)[0]  # Slope coefficient
    
    if rho >= 1 or rho <= 0:
        return np.inf
    
    halflife = np.log(2) / (-np.log(rho))
    return halflife

def pairs_trading_backtest(price_a: np.ndarray, 
                          price_b: np.ndarray,
                          lookback: int = 60,
                          entry_threshold: float = 2.0,
                          exit_threshold: float = 0.5) -> dict:
    """
    Simple pairs trading backtest.
    
    Parameters:
    -----------
    price_a, price_b : np.ndarray
        Price series for two assets
    lookback : int
        Window for estimating hedge ratio
    entry_threshold : float
        Std deviations for entry (e.g., 2.0)
    exit_threshold : float
        Std deviations for exit
        
    Returns:
    --------
    Backtest results (PnL, Sharpe, max drawdown)
    """
    
    T = len(price_a)
    hedge_ratio = np.zeros(T)
    spread = np.zeros(T)
    z_score = np.zeros(T)
    position = np.zeros(T)  # 0 = flat, 1 = long spread, -1 = short spread
    
    for t in range(lookback, T):
        # Estimate hedge ratio over lookback window
        x_window = price_a[t-lookback:t]
        y_window = price_b[t-lookback:t]
        hedge = np.polyfit(x_window, y_window, 1)[0]
        hedge_ratio[t] = hedge
        
        # Compute spread
        spread[t] = price_a[t] - hedge * price_b[t]
        
        # Z-score (normalized by rolling std)
        spread_window = spread[t-lookback:t]
        z = (spread[t] - np.mean(spread_window)) / np.std(spread_window)
        z_score[t] = z
        
        # Position logic
        if position[t-1] == 0:  # Flat
            if z < -entry_threshold:
                position[t] = 1  # Long spread
            elif z > entry_threshold:
                position[t] = -1  # Short spread
        else:  # In position
            if abs(z) < exit_threshold:
                position[t] = 0  # Exit
            else:
                position[t] = position[t-1]  # Hold
    
    # PnL calculation (simplified)
    returns_a = np.diff(np.log(price_a))
    returns_b = np.diff(np.log(price_b))
    
    pnl = np.zeros(T)
    for t in range(1, T):
        if position[t-1] == 1:
            pnl[t] = returns_a[t-1] - hedge_ratio[t] * returns_b[t-1]
        elif position[t-1] == -1:
            pnl[t] = -returns_a[t-1] + hedge_ratio[t] * returns_b[t-1]
    
    cumulative_pnl = np.cumsum(pnl)
    
    return {
        'spread': spread,
        'z_score': z_score,
        'position': position,
        'pnl': pnl,
        'cumulative_pnl': cumulative_pnl,
        'total_return': cumulative_pnl[-1],
        'sharpe_ratio': np.mean(pnl[pnl != 0]) / np.std(pnl[pnl != 0]) * np.sqrt(252) if np.std(pnl[pnl != 0]) > 0 else 0,
        'max_drawdown': np.min(cumulative_pnl) - np.max(cumulative_pnl),
    }
```

---

## Module 6.5: State-Space Models and Kalman Filters

### 6.5.1 State-Space Representation

Many time series problems fit the general form:

$$\text{Measurement (Observation) equation}: y_t = H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

$$\text{State transition equation}: x_t = F x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

where:
- $x_t$: **unobserved state** (hidden variable)
- $y_t$: **observed data**
- $H$: measurement matrix (maps state to observation)
- $F$: state transition matrix (evolution of state)
- $v_t$, $w_t$: observation and state noise

#### Example: Level + Trend Model

Suppose returns have hidden level $\mu_t$ and trend $\tau_t$ that change slowly:

$$y_t = \mu_t + v_t$$

$$\mu_t = \mu_{t-1} + \tau_{t-1} + w_{1,t}$$

$$\tau_t = \tau_{t-1} + w_{2,t}$$

Rewrite in state-space form:

$$\begin{pmatrix} y_t \end{pmatrix} = \begin{pmatrix} 1 & 0 \end{pmatrix} \begin{pmatrix} \mu_t \\ \tau_t \end{pmatrix} + v_t$$

$$\begin{pmatrix} \mu_t \\ \tau_t \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} \mu_{t-1} \\ \tau_{t-1} \end{pmatrix} + \begin{pmatrix} w_{1,t} \\ w_{2,t} \end{pmatrix}$$

### 6.5.2 Kalman Filter: Theory and Derivation

The Kalman filter is an **optimal recursive algorithm** for estimating the hidden state $x_t$ given observations $y_1, \ldots, y_t$.

#### Prediction Step (Time Update)

Given $x_{t-1}$ and its error covariance $P_{t-1}$, predict state at time $t$:

$$x_{t|t-1} = F x_{t-1}$$

$$P_{t|t-1} = F P_{t-1} F^T + Q$$

#### Update Step (Measurement Update)

Given observation $y_t$, correct the prediction:

**Innovation (prediction error)**:

$$\epsilon_t = y_t - H x_{t|t-1}$$

**Innovation covariance**:

$$S_t = H P_{t|t-1} H^T + R$$

**Kalman gain**:

$$K_t = P_{t|t-1} H^T S_t^{-1}$$

**Updated state estimate**:

$$x_{t|t} = x_{t|t-1} + K_t \epsilon_t$$

**Updated error covariance**:

$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

#### Interpretation

- **Kalman gain $K_t$**: How much to adjust based on new observation
  - If $R$ is small (high measurement precision), increase $K_t$ (trust data more)
  - If $Q$ is large (high state uncertainty), increase $K_t$ (prediction is unreliable)
- **Innovation $\epsilon_t$**: How surprising the observation is
- **Likelihood**: $p(y_t | y_{1:t-1}) = \mathcal{N}(\epsilon_t; 0, S_t)$

### 6.5.3 Applications in Finance

#### 1. Adaptive Hedge Ratio

For a pair of stocks, estimate time-varying hedge ratio $\beta_t$ using Kalman filter.

**State**: $\beta_t$ (hedge ratio)

**Observation**: $y_t = P_{A,t} - \beta_t P_{B,t}$ (spread should be mean-zero)

**Dynamics**: $\beta_t = \beta_{t-1} + w_t$ (hedge ratio drifts slowly)

#### 2. Time-Varying Beta

For a stock and market index, estimate $\beta_t = \text{Cov}(r_{stock}, r_{market}) / \text{Var}(r_{market})$ adaptively.

**State**: $\beta_t$ (market beta)

**Observation**: $r_{stock,t} = \alpha + \beta_t r_{market,t} + v_t$

**Dynamics**: $\beta_t = \beta_{t-1} + w_t$

#### 3. Signal Extraction

Extract true price signal from noisy observations.

**State**: True price $p_t$

**Observation**: Observed price $y_t = p_t + v_t$ (measurement noise)

**Dynamics**: $p_t = p_{t-1} + w_t$ (random walk)

### 6.5.4 Implementation

```python
from typing import Tuple

class KalmanFilter:
    """
    Linear Kalman Filter for state-space models.
    
    Attributes:
    -----------
    F : np.ndarray
        State transition matrix (n_states x n_states)
    H : np.ndarray
        Measurement matrix (n_obs x n_states)
    Q : np.ndarray
        State noise covariance (n_states x n_states)
    R : np.ndarray
        Measurement noise covariance (n_obs x n_obs)
    x : np.ndarray
        Current state estimate
    P : np.ndarray
        Current state error covariance
    """
    
    def __init__(self, F: np.ndarray, H: np.ndarray, 
                 Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray = None, P0: np.ndarray = None):
        """
        Initialize Kalman filter.
        
        Parameters:
        -----------
        F, H, Q, R : np.ndarray
            State-space parameters
        x0 : np.ndarray, optional
            Initial state estimate (default: zeros)
        P0 : np.ndarray, optional
            Initial error covariance (default: identity)
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        
        n_states = F.shape[0]
        self.x = x0 if x0 is not None else np.zeros((n_states, 1))
        self.P = P0 if P0 is not None else np.eye(n_states)
        
        # History
        self.x_history = [self.x.copy()]
        self.P_history = [self.P.copy()]
        self.innovation_history = []
        self.innovation_cov_history = []
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step (time update).
        
        Returns:
        --------
        (x_pred, P_pred): predicted state and covariance
        """
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        
        return self.x_pred, self.P_pred
    
    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step (measurement update).
        
        Parameters:
        -----------
        y : np.ndarray
            Observation vector (n_obs x 1)
            
        Returns:
        --------
        (x_updated, P_updated): updated state and covariance
        """
        # Innovation
        innovation = y - self.H @ self.x_pred
        
        # Innovation covariance
        S = self.H @ self.P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)
        
        # Updated state and covariance
        self.x = self.x_pred + K @ innovation
        self.P = (np.eye(self.F.shape[0]) - K @ self.H) @ self.P_pred
        
        # Store history
        self.x_history.append(self.x.copy())
        self.P_history.append(self.P.copy())
        self.innovation_history.append(innovation)
        self.innovation_cov_history.append(S)
        
        return self.x, self.P
    
    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter over sequence of observations.
        
        Parameters:
        -----------
        observations : np.ndarray
            (T, n_obs) array of observations
            
        Returns:
        --------
        (x_filtered, P_filtered): state estimates and covariances for all t
        """
        T = observations.shape[0]
        n_states = self.F.shape[0]
        
        x_filtered = np.zeros((T, n_states))
        P_filtered_diag = np.zeros((T, n_states))  # Store diagonal for simplicity
        
        for t in range(T):
            self.predict()
            y_t = observations[t].reshape(-1, 1)
            self.update(y_t)
            
            x_filtered[t] = self.x.flatten()
            P_filtered_diag[t] = np.diag(self.P)
        
        return x_filtered, P_filtered_diag
    
    def loglikelihood(self, observations: np.ndarray) -> float:
        """
        Compute Gaussian log-likelihood of observations.
        
        Parameters:
        -----------
        observations : np.ndarray
            (T, n_obs) array
            
        Returns:
        --------
        Log-likelihood
        """
        T = observations.shape[0]
        loglik = 0.0
        
        # Reset state
        self.x = np.zeros_like(self.x)
        self.P = np.eye(self.F.shape[0])
        
        for t in range(T):
            self.predict()
            y_t = observations[t].reshape(-1, 1)
            
            innovation = y_t - self.H @ self.x_pred
            S = self.H @ self.P_pred @ self.H.T + self.R
            
            # Gaussian likelihood
            n_obs = y_t.shape[0]
            loglik -= 0.5 * (n_obs * np.log(2 * np.pi) + np.log(np.linalg.det(S)) 
                            + (innovation.T @ np.linalg.inv(S) @ innovation).item())
            
            # Update for next iteration
            K = self.P_pred @ self.H.T @ np.linalg.inv(S)
            self.x = self.x_pred + K @ innovation
            self.P = (np.eye(self.F.shape[0]) - K @ self.H) @ self.P_pred
        
        return loglik

def kalman_filter_example_adaptive_beta():
    """
    Estimate time-varying market beta using Kalman filter.
    
    Regression model: r_stock = alpha + beta_t * r_market + v
    
    State: beta_t (market beta, evolves as random walk)
    Observation: r_stock
    """
    
    # Download stock and market data
    stock_data = yf.download("INFY.NS", start="2020-01-01", end="2024-01-01", progress=False)
    market_data = yf.download("^NSEI", start="2020-01-01", end="2024-01-01", progress=False)
    
    r_stock = np.diff(np.log(stock_data['Adj Close'].values))
    r_market = np.diff(np.log(market_data['Adj Close'].values))
    
    # Align lengths
    min_len = min(len(r_stock), len(r_market))
    r_stock = r_stock[:min_len]
    r_market = r_market[:min_len]
    
    # State-space model
    # State: [alpha, beta]
    # Observation: r_stock = [1, r_market] @ [alpha, beta] + v
    
    F = np.eye(2)  # Random walk for both coefficients
    H = np.array([[1.0, 0.0]])  # Placeholder; will update to [1, r_market_t]
    
    Q = np.diag([1e-7, 1e-7])  # Small state noise
    R = np.array([[1e-5]])  # Observation noise
    
    # Custom filter for time-varying measurement matrix
    class TimeVaryingKalmanFilter(KalmanFilter):
        def update(self, y: np.ndarray, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Update with time-varying H matrix."""
            H_t = h.reshape(1, -1)
            
            innovation = y - H_t @ self.x_pred
            S = H_t @ self.P_pred @ H_t.T + self.R
            K = self.P_pred @ H_t.T / S.item()  # Scalar version
            
            self.x = self.x_pred + K @ innovation
            self.P = (np.eye(2) - K @ H_t) @ self.P_pred
            
            return self.x, self.P
    
    kf = TimeVaryingKalmanFilter(F, H, Q, R, 
                                 x0=np.array([[0.0], [1.0]]),  # Start with unit beta
                                 P0=np.eye(2))
    
    # Filter
    alpha_filtered = []
    beta_filtered = []
    
    for t in range(len(r_stock)):
        kf.predict()
        h_t = np.array([1.0, r_market[t]])
        y_t = np.array([[r_stock[t]]])
        kf.update(y_t, h_t)
        
        alpha_filtered.append(kf.x[0, 0])
        beta_filtered.append(kf.x[1, 0])
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(alpha_filtered, linewidth=1, color='navy', label='α_t (alpha)')
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_title("Time-Varying Alpha", fontweight='bold')
    axes[0].set_ylabel("Alpha")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(beta_filtered, linewidth=1, color='darkgreen', label='β_t (beta)')
    axes[1].axhline(1.0, color='k', linestyle='--', alpha=0.3, label='Unit Beta')
    axes[1].set_title("Time-Varying Market Beta (INFY vs NIFTY50)", fontweight='bold')
    axes[1].set_ylabel("Beta")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('kalman_beta.png', dpi=100, bbox_inches='tight')
    
    return alpha_filtered, beta_filtered

def kalman_vs_rolling_window():
    """
    Compare Kalman filter estimates vs rolling window regression.
    
    Kalman: Smooth, adapts quickly to structural changes
    Rolling: Laggy, but more stable with longer windows
    """
    
    print("""
    KALMAN FILTER VS ROLLING WINDOW
    ================================
    
    Kalman Filter:
      + Optimal for Gaussian systems
      + Adapts quickly to regime changes
      + Handles missing data well
      - More complex, harder to debug
      - Sensitive to parameter specification
      - Assumes linear Gaussian model
    
    Rolling Window:
      + Simple, interpretable
      + Robust to non-Gaussian distributions
      + Transparent (everyone understands regression)
      - Laggy when regime changes
      - Arbitrary window length
      - Throws away old data abruptly
    
    When to use Kalman:
      1. Need rapid adaptation (high-frequency trading)
      2. Known state-space structure (e.g., beta dynamics)
      3. Want principled uncertainty estimates (P_t)
    
    When to use Rolling Window:
      1. Simple, offline analysis
      2. Skeptical of model assumptions
      3. Parameter estimation is hard
    
    Hybrid Approach:
      - Use Kalman for online estimation
      - Monitor innovations for model breakdown
      - Revert to rolling window for robustness checks
    """)
```

### 6.5.5 Advanced: Extended and Unscented Kalman Filters

For **non-linear** state-space models:

$$y_t = h(x_t) + v_t$$
$$x_t = f(x_{t-1}) + w_t$$

**Extended Kalman Filter (EKF)**: Linearize around current estimate

$$F_t = \nabla f(x_{t-1}), \quad H_t = \nabla h(x_t)$$

Use Jacobians in the Kalman filter equations.

**Unscented Kalman Filter (UKF)**: Sample "sigma points" to approximate non-linearity (more accurate than EKF).

Both are available in `filterpy` library:

```python
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter

# Use these for non-linear dynamics (e.g., volatility models, option pricing)
```

---

## Summary and Key Takeaways

### Stationarity (Module 6.1)
- Test with ADF and KPSS
- Use log returns, not prices
- Understand the intuition: why should this series revert?

### ARIMA (Module 6.2)
- Baseline model for univariate forecasting
- Use AIC/BIC for order selection
- Understand ACF/PACF for model identification
- Forecast horizon is short (5-10 steps)

### GARCH (Module 6.3)
- **Critical**: Financial data has time-varying volatility (volatility clustering)
- GARCH(1,1) is the workhorse: $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$
- Use for risk management (VaR), position sizing, and understanding uncertainty
- Persistence $(\alpha + \beta)$ determines how long shocks last

### Cointegration (Module 6.4)
- Foundation of pairs trading
- Engle-Granger test identifies mean-reverting spreads
- Half-life of mean reversion guides hold time
- VECM models both long-run and short-run dynamics

### Kalman Filter (Module 6.5)
- Optimal recursive estimator for linear Gaussian systems
- Use for adaptive parameters (time-varying beta, hedge ratio)
- More complex but faster adaptation than rolling windows
- Provides uncertainty quantification via $P_t$

### WARNING Boxes to Remember

**WARNING: Non-Stationarity**
- Spurious regression can give false confidence
- Always test stationarity before modeling
- Use differenced data, not raw prices

**WARNING: GARCH Limitations**
- Assumes normal distribution (returns have fat tails)
- Model breaks in regime changes
- Parameter estimates can be unstable with short samples

**WARNING: Cointegration Tests**
- Engle-Granger is sensitive to choosing which variable goes on RHS
- Use Johansen test for multiple series
- Historical cointegration doesn't guarantee future cointegration

---

## Exercises

### 6.1: Stationarity
1. Download 5 years of daily returns for two stocks. Test for stationarity using ADF and KPSS. Interpret the results.
2. Compare the ACF of prices vs returns. Explain the difference.
3. Apply three different transformations (differencing, log-differencing, detrending) to a price series. Which achieves stationarity?

### 6.2: ARIMA
1. Implement ARIMA(2,1,1) on a stock's closing price. Interpret the coefficients.
2. Use grid search to find the optimal (p,d,q) for your data using AIC. How does it compare to your manual choice?
3. Forecast 10 days ahead. How wide are the prediction intervals? Why?

### 6.3: GARCH
1. Fit GARCH(1,1) to daily returns. Extract the parameters and interpret them.
2. What is the persistence? Half-life?
3. Compare 1-step and 20-step volatility forecasts. Why do they differ?
4. Use GARCH volatility for position sizing: allocate capital inversely to forecasted volatility.

### 6.4: Cointegration
1. Download prices for a cointegrated pair (e.g., XOM and CVX, both oil companies). Run Engle-Granger test.
2. Estimate the half-life of mean reversion.
3. Backtest a simple pairs trading strategy: enter when z-score > 2, exit at 0.5.
4. Compare Sharpe ratio to a buy-and-hold strategy.

### 6.5: Kalman Filter
1. Implement a Kalman filter to estimate time-varying beta for a stock vs market index.
2. Compare the time-varying beta to a rolling window estimate (30-day, 60-day windows).
3. Use the Kalman filter to denoise noisy observations of a price (add artificial noise to a clean price series, then filter).

---

## Further Reading

- **Stationarity**: Phillips & Perron (1988) — alternative unit root tests
- **ARIMA**: Box & Jenkins (1970) — the classic reference (dense but comprehensive)
- **GARCH**: Bollerslev (1986), Nelson (1991) — foundational papers
- **Cointegration**: Engle & Granger (1987), Johansen (1988) — seminal papers
- **Kalman**: Kalman (1960), Harvey (1989) — deep dives into state-space methods
- **Applied**: Tsay (2010) "Analysis of Financial Time Series" — excellent Python/R book

---

**End of Chapter 6**

*You now have the mathematical foundation and production code to deploy time series models in a quantitative trading system. The next chapter covers Machine Learning approaches that build on these foundations.*

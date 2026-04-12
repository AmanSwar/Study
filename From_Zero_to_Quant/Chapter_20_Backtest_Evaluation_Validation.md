# Chapter 20: Backtest Evaluation and Validation

## Overview

You've built your quantitative trading system, executed 5 years of backtests, and the Sharpe ratio looks incredible. **But here's the brutal truth:** backtests are almost always lies. Not intentional ones—just lies born from overfitting, look-ahead bias, transaction cost underestimation, and the fundamental mismatch between historical data and future markets.

This chapter teaches you how to separate genuine alpha from statistical mirages. We'll implement:

1. **Performance decomposition** to understand what actually drove your returns
2. **Overfitting detection** using advanced statistical tests  
3. **Monte Carlo validation** to stress-test your strategy against market regime changes

By the end, you'll have a robust validation pipeline that separates robust strategies from curve-fit dreams.

---

# Module 20.1: Strategy Performance Analysis

## Learning Objectives

- Decompose returns into alpha/beta and understand factor exposures
- Visualize drawdowns and underwater plots to understand tail risk
- Calculate rolling Sharpe ratios to identify regime changes
- Quantify portfolio turnover and capacity constraints

## 20.1.1 Equity Curve Analysis

The equity curve shows cumulative returns over time. Simple metric, profound insights.

### Underwater Plots (Drawdown Charts)

An underwater plot visualizes **running maximum drawdown**—the distance from peak equity to current level.

**Formula for Drawdown:**
$$D_t = \frac{\text{Running Max}(V_0, V_1, ..., V_t) - V_t}{\text{Running Max}(V_0, V_1, ..., V_t)}$$

Where $V_t$ is portfolio value at time $t$.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_drawdown(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate drawdown from daily returns.
    
    Args:
        returns: Array of daily returns (fractional, e.g., 0.02 for 2%)
        
    Returns:
        drawdown_pct: Array of drawdown percentages
        running_max: Array of cumulative max (running maximum)
    """
    cumsum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumsum_returns)
    drawdown_pct = (running_max - cumsum_returns) / running_max
    
    return drawdown_pct, running_max


def plot_underwater_chart(returns: np.ndarray, 
                         dates: pd.DatetimeIndex,
                         title: str = "Underwater Plot") -> None:
    """
    Plot underwater drawdown chart.
    
    Args:
        returns: Daily returns array
        dates: Datetime index
        title: Plot title
    """
    drawdown_pct, _ = calculate_drawdown(returns)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(dates, -drawdown_pct * 100, 0, alpha=0.3, color='red')
    ax.plot(dates, -drawdown_pct * 100, color='darkred', linewidth=1.5)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown as percentage.
    
    Args:
        returns: Daily returns array
        
    Returns:
        Maximum drawdown percentage
    """
    drawdown_pct, _ = calculate_drawdown(returns)
    return np.max(drawdown_pct)


def calculate_drawdown_duration(returns: np.ndarray) -> int:
    """
    Calculate maximum drawdown duration in trading days.
    
    Args:
        returns: Daily returns array
        
    Returns:
        Number of days to recover from max drawdown
    """
    cumsum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumsum_returns)
    drawdown_pct = (running_max - cumsum_returns) / running_max
    
    # Find when equity reaches new high again
    max_dd_idx = np.argmax(drawdown_pct)
    recovery_idx = max_dd_idx + np.argmax(
        cumsum_returns[max_dd_idx:] >= running_max[max_dd_idx]
    )
    
    return recovery_idx - max_dd_idx


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    returns = np.random.normal(0.0005, 0.01, 1000)
    
    print(f"Maximum Drawdown: {calculate_max_drawdown(returns)*100:.2f}%")
    print(f"Drawdown Duration: {calculate_drawdown_duration(returns)} days")
    
    plot_underwater_chart(returns, dates, "Strategy Underwater Plot")
```

### Rolling Sharpe Ratio

Rolling Sharpe ratio shows how strategy performance changes over time and helps identify regime changes.

**Rolling Sharpe Formula:**
$$\text{Rolling Sharpe}_t = \frac{\text{Mean}(r_{t-N+1},...,r_t)}{\text{Std}(r_{t-N+1},...,r_t)} \times \sqrt{252}$$

Where $N$ is window size (typically 252 trading days = 1 year).

```python
def calculate_rolling_sharpe(returns: np.ndarray, 
                            window: int = 252,
                            rf_rate: float = 0.0) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Daily returns array
        window: Rolling window size in trading days
        rf_rate: Risk-free rate (annualized)
        
    Returns:
        Array of rolling Sharpe ratios
    """
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    
    # Annualize
    rolling_sharpe = ((rolling_mean * 252 - rf_rate) / 
                      (rolling_std * np.sqrt(252)))
    
    return rolling_sharpe.values


def plot_rolling_sharpe(returns: np.ndarray,
                       dates: pd.DatetimeIndex,
                       window: int = 252) -> None:
    """
    Plot rolling Sharpe ratio over time.
    
    Args:
        returns: Daily returns array
        dates: Datetime index
        window: Rolling window size
    """
    rolling_sharpe = calculate_rolling_sharpe(returns, window)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cumulative returns
    cumsum_returns = np.cumprod(1 + returns)
    ax1.plot(dates, (cumsum_returns - 1) * 100, linewidth=2, color='blue')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.set_title('Strategy Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Rolling Sharpe
    ax2.plot(dates, rolling_sharpe, linewidth=2, color='green', label='Rolling Sharpe')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

## 20.1.2 Return Attribution: Alpha vs Beta

Can you isolate the alpha (skill) from beta (market exposure)?

**Linear Factor Model:**
$$r_p = \alpha + \beta_m \cdot r_m + \beta_1 \cdot f_1 + ... + \beta_k \cdot f_k + \epsilon$$

Where:
- $r_p$ = portfolio return
- $\alpha$ = unexplained return (alpha)
- $\beta_m$ = market beta
- $f_i$ = factor returns (momentum, value, etc.)
- $\epsilon$ = residual

```python
from sklearn.linear_model import LinearRegression
from scipy import stats


class FactorAttributionAnalyzer:
    """
    Decompose strategy returns into factor exposures using linear regression.
    """
    
    def __init__(self, strategy_returns: np.ndarray,
                 market_returns: np.ndarray,
                 factor_returns: np.ndarray,
                 dates: pd.DatetimeIndex):
        """
        Initialize factor attribution analyzer.
        
        Args:
            strategy_returns: Daily strategy returns
            market_returns: Daily market returns (e.g., Nifty50)
            factor_returns: Shape (n_days, n_factors) array
            dates: Datetime index
        """
        self.strategy_returns = strategy_returns
        self.market_returns = market_returns
        self.factor_returns = factor_returns
        self.dates = dates
        self.n_days = len(strategy_returns)
        
    def fit_factor_model(self) -> dict:
        """
        Fit multi-factor linear regression model.
        
        Returns:
            Dictionary with alpha, betas, R-squared, and statistics
        """
        # Combine market returns with factors
        X = np.column_stack([self.market_returns, self.factor_returns])
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, self.strategy_returns)
        
        # Calculate statistics
        y_pred = model.predict(X)
        residuals = self.strategy_returns - y_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.strategy_returns - np.mean(self.strategy_returns))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Standard errors
        mse = ss_res / (self.n_days - X.shape[1] - 1)
        var_covar = mse * np.linalg.inv(X.T @ X)
        std_errors = np.sqrt(np.diag(var_covar))
        
        # T-statistics
        t_stats = np.append(model.intercept_ / std_errors[0],
                           model.coef_ / std_errors[1:])
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), 
                                        self.n_days - X.shape[1] - 1))
        
        results = {
            'alpha': model.intercept_,
            'alpha_annual': model.intercept_ * 252,
            'market_beta': model.coef_[0],
            'factor_betas': model.coef_[1:],
            'r_squared': r_squared,
            'std_errors': std_errors,
            't_stats': t_stats,
            'p_values': p_values,
            'residual_std': np.std(residuals),
            'residual_sharpe': (np.mean(residuals) * 252) / 
                               (np.std(residuals) * np.sqrt(252))
        }
        
        return results
    
    def print_attribution_summary(self, results: dict, 
                                  factor_names: list = None) -> None:
        """
        Print factor attribution summary.
        
        Args:
            results: Output from fit_factor_model()
            factor_names: Names of factors for labeling
        """
        if factor_names is None:
            factor_names = [f"Factor_{i}" 
                           for i in range(len(results['factor_betas']))]
        
        print("\n" + "="*70)
        print("FACTOR ATTRIBUTION ANALYSIS")
        print("="*70)
        print(f"\nAlpha (daily):          {results['alpha']*100:>8.4f}%")
        print(f"Alpha (annualized):     {results['alpha_annual']*100:>8.2f}%")
        print(f"Market Beta:            {results['market_beta']:>8.3f}")
        print(f"Residual Sharpe:        {results['residual_sharpe']:>8.3f}")
        print(f"R-squared:              {results['r_squared']:>8.3f}")
        
        print("\nFactor Exposures:")
        print("-" * 70)
        print(f"{'Factor':<20} {'Beta':>12} {'Std Err':>12} {'t-stat':>12} {'p-value':>12}")
        print("-" * 70)
        
        # Market
        print(f"{'Market':<20} {results['market_beta']:>12.4f} " +
              f"{results['std_errors'][1]:>12.4f} " +
              f"{results['t_stats'][1]:>12.3f} " +
              f"{results['p_values'][1]:>12.4f}")
        
        # Factors
        for i, fname in enumerate(factor_names):
            print(f"{fname:<20} {results['factor_betas'][i]:>12.4f} " +
                  f"{results['std_errors'][i+2]:>12.4f} " +
                  f"{results['t_stats'][i+2]:>12.3f} " +
                  f"{results['p_values'][i+2]:>12.4f}")
        
        print("="*70)


# Example: NSE strategy with momentum and value factors
np.random.seed(42)
n_days = 1000

strategy_returns = np.random.normal(0.0008, 0.012, n_days)
market_returns = np.random.normal(0.0005, 0.010, n_days)

# Simulate momentum and value factors
momentum_factor = np.random.normal(0.0002, 0.008, n_days)
value_factor = np.random.normal(0.0001, 0.007, n_days)
factor_returns = np.column_stack([momentum_factor, value_factor])

analyzer = FactorAttributionAnalyzer(strategy_returns, market_returns, 
                                     factor_returns,
                                     pd.date_range('2020-01-01', periods=n_days))

results = analyzer.fit_factor_model()
analyzer.print_attribution_summary(results, ['Momentum', 'Value'])
```

## 20.1.3 Turnover and Capacity Analysis

As you scale AUM (Assets Under Management), two things happen:

1. **Transaction costs increase** (more capital to move)
2. **Market impact increases** (your trades move prices)

This erodes alpha.

**Turnover Metrics:**

$$\text{Annual Turnover} = \frac{1}{252} \sum_{t=1}^{252} \text{Turnover}_t$$

$$\text{Turnover}_t = \frac{\sum_i |w_i^{t} - w_i^{t-1}|}{2}$$

Where $w_i^t$ is weight of asset $i$ at time $t$.

```python
class CapacityAnalyzer:
    """
    Analyze strategy capacity and transaction cost impact.
    """
    
    def __init__(self, weights_history: np.ndarray,
                 prices: np.ndarray,
                 spreads_bps: float = 2.0):
        """
        Initialize capacity analyzer.
        
        Args:
            weights_history: Shape (n_days, n_assets) weight matrix
            prices: Shape (n_days, n_assets) price matrix
            spreads_bps: Bid-ask spread in basis points (0.01 = 1 bps)
        """
        self.weights_history = weights_history
        self.prices = prices
        self.spreads_bps = spreads_bps / 10000  # Convert to decimal
        self.n_days = weights_history.shape[0]
        self.n_assets = weights_history.shape[1]
        
    def calculate_turnover(self) -> Tuple[np.ndarray, float]:
        """
        Calculate daily and annual turnover.
        
        Returns:
            daily_turnover: Array of daily turnover percentages
            annual_turnover: Annualized turnover
        """
        weight_changes = np.abs(np.diff(self.weights_history, axis=0))
        daily_turnover = np.sum(weight_changes, axis=1) / 2
        
        annual_turnover = np.mean(daily_turnover)
        
        return daily_turnover, annual_turnover
    
    def calculate_market_impact(self, aum: float, 
                               impact_factor: float = 0.1) -> np.ndarray:
        """
        Estimate market impact costs based on trade size.
        
        Args:
            aum: Assets under management
            impact_factor: Impact coefficient (tunable parameter)
            
        Returns:
            Array of daily market impact costs
        """
        daily_turnover, _ = self.calculate_turnover()
        
        # Trade size in rupees
        daily_trade_size = daily_turnover * aum
        
        # Simple market impact model: cost = impact_factor * sqrt(trade_size)
        market_impact_pct = impact_factor * np.sqrt(daily_trade_size / 1e6) / 100
        
        return market_impact_pct
    
    def net_alpha_after_costs(self, gross_alpha: float,
                             aum: float) -> Tuple[float, float]:
        """
        Calculate net alpha after transaction and market impact costs.
        
        Args:
            gross_alpha: Gross alpha (annualized percentage)
            aum: Assets under management
            
        Returns:
            net_alpha: Net alpha after costs
            total_costs: Total transaction + market impact costs
        """
        daily_turnover, annual_turnover = self.calculate_turnover()
        
        # Transaction costs (spread + commissions)
        transaction_costs = annual_turnover * self.spreads_bps
        
        # Market impact costs
        market_impact = np.mean(self.calculate_market_impact(aum))
        
        total_costs = transaction_costs + market_impact
        net_alpha = gross_alpha - total_costs * 100
        
        return net_alpha, total_costs * 100
    
    def capacity_frontier(self, gross_alpha: float,
                         aum_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate alpha erosion across AUM range.
        
        Args:
            gross_alpha: Gross alpha (annualized percentage)
            aum_range: Array of AUM values to test
            
        Returns:
            aum_range: Array of AUM values
            net_alphas: Array of net alpha values
        """
        net_alphas = np.array([
            self.net_alpha_after_costs(gross_alpha, aum)[0]
            for aum in aum_range
        ])
        
        return aum_range, net_alphas
    
    def plot_capacity_frontier(self, gross_alpha: float,
                              max_aum: float = 1e9) -> None:
        """
        Plot alpha erosion with increasing AUM.
        
        Args:
            gross_alpha: Gross alpha percentage
            max_aum: Maximum AUM to plot
        """
        aum_range = np.logspace(6, np.log10(max_aum), 50)
        aum_range_crore = aum_range / 1e7  # Convert to crores
        
        _, net_alphas = self.capacity_frontier(gross_alpha, aum_range)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(aum_range_crore, net_alphas, linewidth=2.5, color='darkblue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax.fill_between(aum_range_crore, 0, net_alphas, 
                       where=(net_alphas > 0), alpha=0.2, color='green',
                       label='Profitable')
        ax.fill_between(aum_range_crore, 0, net_alphas,
                       where=(net_alphas <= 0), alpha=0.2, color='red',
                       label='Unprofitable')
        
        ax.set_xlabel('AUM (Crores INR)', fontsize=12)
        ax.set_ylabel('Net Alpha (%)', fontsize=12)
        ax.set_title('Strategy Alpha Erosion with Scale', 
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()


# Example
weights_test = np.random.dirichlet(np.ones(5), size=252)  # 5 assets, 252 days
prices_test = np.random.uniform(100, 500, (252, 5))

capacity = CapacityAnalyzer(weights_test, prices_test, spreads_bps=2.0)

daily_turnover, annual_turnover = capacity.calculate_turnover()
print(f"Annual Turnover: {annual_turnover*100:.2f}%")

aum_range = np.array([1e6, 1e7, 5e7, 1e8, 5e8])
for aum in aum_range:
    net_alpha, costs = capacity.net_alpha_after_costs(20.0, aum)  # 20% gross alpha
    print(f"AUM: {aum/1e7:.1f} Cr, Net Alpha: {net_alpha:.2f}%, Costs: {costs:.2f}%")

capacity.plot_capacity_frontier(20.0, max_aum=1e9)
```

---

# Module 20.2: Overfitting Detection

## Learning Objectives

- Understand sources of overfitting in backtests
- Implement Deflated Sharpe Ratio to adjust for multiple testing
- Use Probability of Backtest Overfitting (PBO) to validate robustness
- Apply sensitivity analysis to test parameter stability

## 20.2.1 The Multiple Testing Problem

You test 1,000 strategies. By chance alone, ~50 with Sharpe ratio > 1.0 will emerge even if no real alpha exists.

**Why?** You're running multiple hypothesis tests without correcting for it.

**Deflated Sharpe Ratio** corrects for this multiple testing problem.

### Deflated Sharpe Ratio (DSR)

The Deflated Sharpe Ratio adjusts the observed Sharpe ratio based on:
1. The number of trials (strategies tested)
2. The length of the backtest
3. The number of independent returns

**Formula:**

$$\text{DSR} = \text{SR} \cdot \sqrt{\frac{V[\text{SR}]}{1}} \cdot Z_{1-p/2}$$

Where:
$$V[\text{SR}] = \frac{1}{T-1}\left(1 + \frac{1}{4}\text{SR}^2\right)$$

$$Z_{1-p/2} = \Phi^{-1}\left(1 - \frac{p}{2M}\right)$$

$p$ = significance level (0.05), $M$ = number of strategies tested, $T$ = number of observations

```python
from scipy.special import erfc
from scipy.stats import norm


class DeflatorsharpeRatioCalculator:
    """
    Calculate Deflated Sharpe Ratio (DSR) accounting for multiple testing.
    """
    
    def __init__(self, observed_sharpe: float,
                 n_observations: int,
                 n_strategies_tested: int = 1,
                 significance_level: float = 0.05):
        """
        Initialize DSR calculator.
        
        Args:
            observed_sharpe: Observed Sharpe ratio
            n_observations: Number of returns (trading days)
            n_strategies_tested: Number of strategies tested (multiple testing correction)
            significance_level: Type I error rate (default 0.05)
        """
        self.observed_sharpe = observed_sharpe
        self.n_observations = n_observations
        self.n_strategies_tested = n_strategies_tested
        self.alpha = significance_level
        
    def sharpe_ratio_variance(self) -> float:
        """
        Calculate variance of Sharpe ratio estimator.
        
        Returns:
            Variance of SR
        """
        return (1.0 / (self.n_observations - 1) * 
                (1 + 0.25 * self.observed_sharpe**2))
    
    def bonferroni_critical_value(self) -> float:
        """
        Calculate Bonferroni-corrected critical value.
        
        Returns:
            Critical Z-score
        """
        # Two-tailed test with Bonferroni correction
        p_corrected = self.alpha / (2 * self.n_strategies_tested)
        z_critical = norm.ppf(1 - p_corrected)
        
        return z_critical
    
    def deflated_sharpe_ratio(self) -> float:
        """
        Calculate Deflated Sharpe Ratio.
        
        Returns:
            Deflated Sharpe ratio (adjusted for multiple testing)
        """
        var_sr = self.sharpe_ratio_variance()
        z_critical = self.bonferroni_critical_value()
        
        # DSR formula: apply negative correction for multiple testing
        dsr = (self.observed_sharpe * 
               (1 - norm.pdf(z_critical) / (2 * z_critical * 
                                            norm.cdf(z_critical))))
        
        return dsr
    
    def probability_false_discovery(self) -> float:
        """
        Calculate probability that observed Sharpe is due to chance.
        
        Returns:
            Probability of false discovery (0 to 1)
        """
        var_sr = self.sharpe_ratio_variance()
        std_sr = np.sqrt(var_sr)
        
        # P(SR > observed | no real alpha)
        z_score = self.observed_sharpe / std_sr
        p_value = 1 - norm.cdf(z_score)
        
        # Bonferroni correction
        p_corrected = min(1.0, p_value * self.n_strategies_tested)
        
        return p_corrected
    
    def minimum_credible_sharpe(self) -> float:
        """
        Calculate minimum Sharpe ratio to be significant after correction.
        
        Returns:
            Minimum SR for statistical significance
        """
        z_critical = self.bonferroni_critical_value()
        var_sr = self.sharpe_ratio_variance()
        std_sr = np.sqrt(var_sr)
        
        return z_critical * std_sr
    
    def print_dsr_summary(self) -> None:
        """Print comprehensive DSR analysis."""
        dsr = self.deflated_sharpe_ratio()
        prob_fdr = self.probability_false_discovery()
        min_sr = self.minimum_credible_sharpe()
        
        print("\n" + "="*70)
        print("DEFLATED SHARPE RATIO ANALYSIS")
        print("="*70)
        print(f"\nObserved Sharpe Ratio:        {self.observed_sharpe:>8.3f}")
        print(f"Deflated Sharpe Ratio:        {dsr:>8.3f}")
        print(f"Adjustment Factor:            {dsr/self.observed_sharpe:>8.3f}x")
        print(f"\nNumber of Observations:       {self.n_observations:>8.0f}")
        print(f"Number of Strategies Tested:  {self.n_strategies_tested:>8.0f}")
        print(f"Significance Level (α):       {self.alpha:>8.4f}")
        
        print(f"\nMinimum Credible SR:          {min_sr:>8.3f}")
        print(f"Prob. False Discovery:        {prob_fdr*100:>8.2f}%")
        
        if prob_fdr > 0.05:
            print(f"⚠️  High probability of false discovery!")
        else:
            print(f"✓ Statistically significant (p < 0.05)")
        
        print("="*70)


# Example: Strategy with Sharpe ratio 1.5, tested 500 variations
dsr_calc = DeflatorsharpeRatioCalculator(
    observed_sharpe=1.5,
    n_observations=1000,  # ~4 years of daily data
    n_strategies_tested=500,  # Parameter combinations tested
    significance_level=0.05
)

dsr_calc.print_dsr_summary()

# Compare single test vs. multiple testing
print("\n" + "="*70)
print("Comparison: Single Test vs. Multiple Testing")
print("="*70)

for n_strats in [1, 10, 100, 500, 1000]:
    dsr = DeflatorsharpeRatioCalculator(1.5, 1000, n_strats)
    deflated = dsr.deflated_sharpe_ratio()
    print(f"Strategies Tested: {n_strats:>4d} | Deflated SR: {deflated:>6.3f} | " +
          f"Prob FDR: {dsr.probability_false_discovery()*100:>6.2f}%")
```

## 20.2.2 Probability of Backtest Overfitting (PBO)

PBO measures: Given your backtested Sharpe ratio, what's the probability it won't hold out-of-sample?

This requires **in-sample vs out-of-sample comparison**.

**The Process:**
1. Divide data into train (IS) and test (OOS) periods
2. Fit strategy on IS, evaluate on OOS
3. Repeat with different train/test splits
4. Calculate ratio of OOS to IS Sharpe ratios

```python
from sklearn.model_selection import TimeSeriesSplit


class ProbabilityBacktestOverfitting:
    """
    Calculate Probability of Backtest Overfitting (PBO).
    """
    
    def __init__(self, returns: np.ndarray,
                 strategy_params: list,
                 fit_function,
                 n_splits: int = 10):
        """
        Initialize PBO calculator.
        
        Args:
            returns: Daily returns array
            strategy_params: List of parameter combinations to test
            fit_function: Function(returns, params) -> strategy_returns
            n_splits: Number of cross-validation splits
        """
        self.returns = returns
        self.strategy_params = strategy_params
        self.fit_function = fit_function
        self.n_splits = n_splits
        
    def calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return -np.inf
        return (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
    
    def cross_validate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform time series cross-validation.
        
        Returns:
            is_sharpes: In-sample Sharpe ratios
            oos_sharpes: Out-of-sample Sharpe ratios
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        is_sharpes = []
        oos_sharpes = []
        
        for train_idx, test_idx in tscv.split(self.returns):
            train_returns = self.returns[train_idx]
            test_returns = self.returns[test_idx]
            
            # For each parameter combination, fit on IS and eval on OOS
            best_is_sharpe = -np.inf
            best_oos_sharpe = 0
            
            for params in self.strategy_params:
                # Simulate strategy returns
                strategy_train = self.fit_function(train_returns, params)
                strategy_test = self.fit_function(test_returns, params)
                
                is_sharpe = self.calculate_sharpe(strategy_train)
                oos_sharpe = self.calculate_sharpe(strategy_test)
                
                # Keep best IS fit
                if is_sharpe > best_is_sharpe:
                    best_is_sharpe = is_sharpe
                    best_oos_sharpe = oos_sharpe
            
            is_sharpes.append(best_is_sharpe)
            oos_sharpes.append(best_oos_sharpe)
        
        return np.array(is_sharpes), np.array(oos_sharpes)
    
    def calculate_pbo(self, is_sharpes: np.ndarray,
                     oos_sharpes: np.ndarray) -> float:
        """
        Calculate PBO metric.
        
        Formula: PBO = 1 - (Mean OOS Sharpe) / (Mean IS Sharpe)
        
        Args:
            is_sharpes: In-sample Sharpes
            oos_sharpes: Out-of-sample Sharpes
            
        Returns:
            PBO probability (0 to 1)
        """
        if np.mean(is_sharpes) <= 0:
            return 1.0
        
        pbo = 1 - (np.mean(oos_sharpes) / np.mean(is_sharpes))
        return np.clip(pbo, 0, 1)
    
    def print_pbo_analysis(self) -> None:
        """Print PBO analysis and interpretation."""
        is_sharpes, oos_sharpes = self.cross_validate()
        pbo = self.calculate_pbo(is_sharpes, oos_sharpes)
        
        print("\n" + "="*70)
        print("PROBABILITY OF BACKTEST OVERFITTING (PBO)")
        print("="*70)
        
        print(f"\nIn-Sample Performance:")
        print(f"  Mean Sharpe:    {np.mean(is_sharpes):>8.3f}")
        print(f"  Std Dev:        {np.std(is_sharpes):>8.3f}")
        print(f"  Min / Max:      {np.min(is_sharpes):>8.3f} / {np.max(is_sharpes):>8.3f}")
        
        print(f"\nOut-of-Sample Performance:")
        print(f"  Mean Sharpe:    {np.mean(oos_sharpes):>8.3f}")
        print(f"  Std Dev:        {np.std(oos_sharpes):>8.3f}")
        print(f"  Min / Max:      {np.min(oos_sharpes):>8.3f} / {np.max(oos_sharpes):>8.3f}")
        
        print(f"\nProbability of Overfitting: {pbo*100:>8.2f}%")
        
        if pbo > 0.8:
            print("⚠️  VERY HIGH overfitting risk - strategy likely won't work live")
        elif pbo > 0.5:
            print("⚠️  High overfitting risk - be cautious")
        elif pbo > 0.2:
            print("✓ Moderate overfitting - acceptable with risk management")
        else:
            print("✓ Low overfitting - strategy appears robust")
        
        print("="*70)
        
        return is_sharpes, oos_sharpes, pbo


# Example: Simple mean reversion strategy
def dummy_strategy(returns: np.ndarray, params: dict) -> np.ndarray:
    """Simple dummy strategy for demonstration."""
    window = params['window']
    sma = pd.Series(returns).rolling(window).mean()
    
    # Buy when price is below SMA, sell when above
    positions = np.sign(-np.diff(sma, prepend=0))
    strategy_returns = returns * positions[:-1]
    
    return strategy_returns[window:]


# Parameter grid to test
params_grid = [
    {'window': w} for w in [5, 10, 15, 20, 25, 30]
]

np.random.seed(42)
test_returns = np.random.normal(0.0005, 0.01, 1000)

pbo_calc = ProbabilityBacktestOverfitting(
    returns=test_returns,
    strategy_params=params_grid,
    fit_function=dummy_strategy,
    n_splits=5
)

is_sharpes, oos_sharpes, pbo = pbo_calc.print_pbo_analysis()

# Plot IS vs OOS
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(is_sharpes, oos_sharpes, s=100, alpha=0.6, color='blue', edgecolors='black')
ax.plot([np.min(is_sharpes), np.max(is_sharpes)], 
        [np.min(is_sharpes), np.max(is_sharpes)],
        'r--', linewidth=2, label='Perfect fit (IS=OOS)')
ax.set_xlabel('In-Sample Sharpe Ratio', fontsize=12)
ax.set_ylabel('Out-of-Sample Sharpe Ratio', fontsize=12)
ax.set_title(f'Backtest Overfitting Analysis (PBO = {pbo*100:.1f}%)', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
```

## 20.2.3 Sensitivity Analysis: Parameter Robustness

Does your strategy only work with exact parameters, or is it robust to perturbations?

```python
class SensitivityAnalysis:
    """
    Test strategy robustness by perturbing parameters.
    """
    
    def __init__(self, returns: np.ndarray,
                 base_params: dict,
                 fit_function,
                 param_ranges: dict):
        """
        Initialize sensitivity analysis.
        
        Args:
            returns: Daily returns array
            base_params: Base parameter dictionary
            fit_function: Function(returns, params) -> strategy_returns
            param_ranges: Dict of param_name -> (min, max, n_points)
        """
        self.returns = returns
        self.base_params = base_params
        self.fit_function = fit_function
        self.param_ranges = param_ranges
        
    def calculate_sharpe(self, strat_returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(strat_returns) == 0 or np.std(strat_returns) == 0:
            return -np.inf
        return (np.mean(strat_returns) * 252) / (np.std(strat_returns) * np.sqrt(252))
    
    def sensitivity_analysis(self) -> dict:
        """
        Run sensitivity analysis across all parameters.
        
        Returns:
            Dictionary with results for each parameter
        """
        results = {}
        
        for param_name, (min_val, max_val, n_points) in self.param_ranges.items():
            param_values = np.linspace(min_val, max_val, n_points)
            sharpes = []
            
            for val in param_values:
                params = self.base_params.copy()
                params[param_name] = val
                
                try:
                    strat_returns = self.fit_function(self.returns, params)
                    sharpe = self.calculate_sharpe(strat_returns)
                    sharpes.append(sharpe)
                except:
                    sharpes.append(-np.inf)
            
            results[param_name] = {
                'values': param_values,
                'sharpes': np.array(sharpes),
                'mean_sharpe': np.mean(sharpes[~np.isinf(sharpes)]) 
                              if not all(np.isinf(sharpes)) else -np.inf,
                'std_sharpe': np.std(sharpes[~np.isinf(sharpes)])
                             if not all(np.isinf(sharpes)) else 0
            }
        
        return results
    
    def plot_sensitivity(self, results: dict) -> None:
        """Plot sensitivity analysis results."""
        n_params = len(results)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        
        if n_params == 1:
            axes = [axes]
        
        for idx, (param_name, data) in enumerate(results.items()):
            ax = axes[idx]
            
            # Filter out infinite values
            valid_idx = ~np.isinf(data['sharpes'])
            
            ax.plot(data['values'][valid_idx], data['sharpes'][valid_idx],
                   linewidth=2.5, marker='o', color='darkblue')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.fill_between(data['values'][valid_idx],
                           0, data['sharpes'][valid_idx],
                           alpha=0.2, color='blue')
            
            ax.set_xlabel(f'{param_name} Value', fontsize=11)
            ax.set_ylabel('Sharpe Ratio', fontsize=11)
            ax.set_title(f'Sensitivity to {param_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example: Test robustness to window length and threshold
sensitivity = SensitivityAnalysis(
    returns=test_returns,
    base_params={'window': 20, 'threshold': 0.5},
    fit_function=dummy_strategy,
    param_ranges={
        'window': (5, 50, 20),
        'threshold': (0.1, 1.0, 20)
    }
)

# This requires updating fit_function to use threshold param
# For demonstration, we'll just test window
sensitivity_results = sensitivity.sensitivity_analysis()
```

---

# Module 20.3: Monte Carlo Simulation

## Learning Objectives

- Use bootstrapping to estimate confidence intervals for strategy metrics
- Test against random entry signals to establish baseline
- Simulate market regimes to stress-test strategy
- Apply Hansen's SPA test for comparing multiple strategies

## 20.3.1 Bootstrapping Strategy Returns

Estimate confidence intervals by resampling returns with replacement.

```python
class BootstrapValidator:
    """
    Validate strategy using bootstrap resampling.
    """
    
    def __init__(self, strategy_returns: np.ndarray,
                 n_bootstrap: int = 1000,
                 confidence: float = 0.95):
        """
        Initialize bootstrap validator.
        
        Args:
            strategy_returns: Daily strategy returns
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95)
        """
        self.strategy_returns = strategy_returns
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.n_observations = len(strategy_returns)
        
    def calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return np.nan
        return (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
    
    def bootstrap_sharpe(self) -> np.ndarray:
        """
        Generate bootstrap distribution of Sharpe ratios.
        
        Returns:
            Array of bootstrap Sharpe ratios
        """
        bootstrap_sharpes = []
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(self.n_observations,
                                      size=self.n_observations,
                                      replace=True)
            resampled_returns = self.strategy_returns[indices]
            
            sharpe = self.calculate_sharpe(resampled_returns)
            bootstrap_sharpes.append(sharpe)
        
        return np.array(bootstrap_sharpes)
    
    def confidence_interval(self, metric_samples: np.ndarray) -> Tuple[float, float]:
        """
        Calculate confidence interval for metric.
        
        Args:
            metric_samples: Array of bootstrap metric values
            
        Returns:
            (lower_bound, upper_bound)
        """
        alpha = 1 - self.confidence
        lower = np.percentile(metric_samples, alpha/2 * 100)
        upper = np.percentile(metric_samples, (1 - alpha/2) * 100)
        
        return lower, upper
    
    def print_bootstrap_summary(self) -> None:
        """Print bootstrap validation summary."""
        bootstrap_sharpes = self.bootstrap_sharpe()
        lower, upper = self.confidence_interval(bootstrap_sharpes)
        
        observed_sharpe = self.calculate_sharpe(self.strategy_returns)
        
        print("\n" + "="*70)
        print("BOOTSTRAP VALIDATION")
        print("="*70)
        
        print(f"\nObserved Sharpe Ratio:        {observed_sharpe:>8.3f}")
        print(f"\n{self.confidence*100:.0f}% Confidence Interval:")
        print(f"  Lower Bound:                {lower:>8.3f}")
        print(f"  Upper Bound:                {upper:>8.3f}")
        print(f"  Width:                      {upper - lower:>8.3f}")
        
        print(f"\nBootstrap Distribution:")
        print(f"  Mean:                       {np.mean(bootstrap_sharpes):>8.3f}")
        print(f"  Std Dev:                    {np.std(bootstrap_sharpes):>8.3f}")
        print(f"  Min / Max:                  {np.min(bootstrap_sharpes):>8.3f} / " +
              f"{np.max(bootstrap_sharpes):>8.3f}")
        
        # Probability that true Sharpe < 0
        prob_negative = np.mean(bootstrap_sharpes < 0)
        print(f"\nProb(True Sharpe < 0):        {prob_negative*100:>8.2f}%")
        
        print("="*70)
        
        return bootstrap_sharpes


bootstrap = BootstrapValidator(test_returns, n_bootstrap=1000)
bootstrap_sharpes = bootstrap.print_bootstrap_summary()

# Plot bootstrap distribution
fig, ax = plt.subplots(figsize=(12, 7))
ax.hist(bootstrap_sharpes, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(np.mean(bootstrap_sharpes), color='red', linestyle='--',
          linewidth=2, label='Bootstrap Mean')
ax.axvline(bootstrap.calculate_sharpe(test_returns), color='green', linestyle='--',
          linewidth=2, label='Observed')
ax.axvline(0, color='black', linestyle='-', alpha=0.3, label='Zero')
ax.set_xlabel('Sharpe Ratio', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Bootstrap Distribution of Sharpe Ratios', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

## 20.3.2 Random Entry Test

If you replaced your signals with random entries, would you still make money?

This tests whether alpha comes from **signal quality** vs **market direction**.

```python
class RandomizedSignalTest:
    """
    Test if strategy outperforms random entry baseline.
    """
    
    def __init__(self, returns: np.ndarray,
                 strategy_returns: np.ndarray,
                 n_random_trials: int = 1000):
        """
        Initialize randomized signal test.
        
        Args:
            returns: Market returns (benchmark)
            strategy_returns: Strategy returns
            n_random_trials: Number of random trials
        """
        self.returns = returns
        self.strategy_returns = strategy_returns
        self.n_random_trials = n_random_trials
        self.n_observations = len(returns)
        
    def calculate_metrics(self, returns: np.ndarray) -> dict:
        """Calculate return metrics."""
        if len(returns) == 0 or np.std(returns) == 0:
            return {
                'total_return': np.nan,
                'sharpe': np.nan,
                'sortino': np.nan,
                'calmar': np.nan
            }
        
        total_return = np.prod(1 + returns) - 1
        sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
        
        # Sortino ratio (downside risk)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino = (np.mean(returns) * 252) / (downside_std * np.sqrt(252))
        
        # Calmar ratio
        cumsum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumsum_returns)
        drawdown = (running_max - cumsum_returns) / running_max
        max_drawdown = np.max(drawdown)
        calmar = (np.mean(returns) * 252) / max(max_drawdown, 0.001)
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar
        }
    
    def random_signal_test(self) -> Tuple[dict, np.ndarray]:
        """
        Run random entry signals and compare to strategy.
        
        Returns:
            observed_metrics: Strategy metrics
            random_distributions: Dict of random metric distributions
        """
        observed_metrics = self.calculate_metrics(self.strategy_returns)
        
        random_distributions = {
            'sharpe': [],
            'sortino': [],
            'calmar': [],
            'total_return': []
        }
        
        for _ in range(self.n_random_trials):
            # Random buy/sell signals
            random_signals = np.random.choice([-1, 0, 1], size=self.n_observations)
            random_returns = self.returns * random_signals
            
            random_metrics = self.calculate_metrics(random_returns)
            
            random_distributions['sharpe'].append(random_metrics['sharpe'])
            random_distributions['sortino'].append(random_metrics['sortino'])
            random_distributions['calmar'].append(random_metrics['calmar'])
            random_distributions['total_return'].append(random_metrics['total_return'])
        
        for key in random_distributions:
            random_distributions[key] = np.array(random_distributions[key])
        
        return observed_metrics, random_distributions
    
    def print_random_test_summary(self) -> None:
        """Print random entry test results."""
        observed_metrics, random_dist = self.random_signal_test()
        
        print("\n" + "="*70)
        print("RANDOMIZED ENTRY SIGNAL TEST")
        print("="*70)
        
        print(f"\nObserved Strategy Performance:")
        print(f"  Total Return:               {observed_metrics['total_return']*100:>8.2f}%")
        print(f"  Sharpe Ratio:               {observed_metrics['sharpe']:>8.3f}")
        print(f"  Sortino Ratio:              {observed_metrics['sortino']:>8.3f}")
        print(f"  Calmar Ratio:               {observed_metrics['calmar']:>8.3f}")
        
        print(f"\nRandom Entry Benchmark (Mean of {self.n_random_trials} trials):")
        print(f"  Total Return:               {np.mean(random_dist['total_return'])*100:>8.2f}%")
        print(f"  Sharpe Ratio:               {np.mean(random_dist['sharpe']):>8.3f}")
        print(f"  Sortino Ratio:              {np.mean(random_dist['sortino']):>8.3f}")
        print(f"  Calmar Ratio:               {np.mean(random_dist['calmar']):>8.3f}")
        
        print(f"\nPercentile vs Random:")
        
        # Calculate percentile rank
        sharpe_percentile = np.mean(random_dist['sharpe'] < observed_metrics['sharpe']) * 100
        sortino_percentile = np.mean(random_dist['sortino'] < observed_metrics['sortino']) * 100
        
        print(f"  Sharpe Ratio:               {sharpe_percentile:>8.1f}th percentile")
        print(f"  Sortino Ratio:              {sortino_percentile:>8.1f}th percentile")
        
        if sharpe_percentile > 95:
            print(f"\n✓ Strategy significantly outperforms random (p < 0.05)")
        else:
            print(f"\n⚠️  Strategy not significantly different from random")
        
        print("="*70)
        
        return observed_metrics, random_dist


# Example
np.random.seed(42)
market_returns = np.random.normal(0.0005, 0.01, 1000)
strategy_returns = market_returns + np.random.normal(0.0003, 0.005, 1000)

random_test = RandomizedSignalTest(market_returns, strategy_returns, n_random_trials=500)
observed, random_dist = random_test.print_random_test_summary()

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sharpe distribution
axes[0].hist(random_dist['sharpe'], bins=40, alpha=0.7, color='gray', label='Random')
axes[0].axvline(observed['sharpe'], color='green', linestyle='--', linewidth=2.5, label='Strategy')
axes[0].set_xlabel('Sharpe Ratio', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Sharpe Distribution: Strategy vs Random', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Sortino distribution
axes[1].hist(random_dist['sortino'], bins=40, alpha=0.7, color='gray', label='Random')
axes[1].axvline(observed['sortino'], color='green', linestyle='--', linewidth=2.5, label='Strategy')
axes[1].set_xlabel('Sortino Ratio', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Sortino Distribution: Strategy vs Random', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 20.3.3 Market Regime Simulation

What if market volatility doubles? What if correlations break down?

Test your strategy against simulated stress scenarios.

```python
class MarketRegimeSimulation:
    """
    Simulate market regime changes and test strategy robustness.
    """
    
    def __init__(self, base_returns: np.ndarray,
                 strategy_signal: np.ndarray):
        """
        Initialize regime simulation.
        
        Args:
            base_returns: Historical market returns
            strategy_signal: Buy/sell signals (1, -1, 0)
        """
        self.base_returns = base_returns
        self.strategy_signal = strategy_signal
        self.n_observations = len(base_returns)
        
    def simulate_volatility_shock(self, vol_multiplier: float) -> np.ndarray:
        """
        Simulate market with increased volatility.
        
        Args:
            vol_multiplier: Volatility multiplier (e.g., 2.0 for 2x)
            
        Returns:
            Simulated returns with higher volatility
        """
        # Preserve mean, increase std
        mean_ret = np.mean(self.base_returns)
        std_ret = np.std(self.base_returns)
        
        normalized = (self.base_returns - mean_ret) / std_ret
        shocked = normalized * std_ret * vol_multiplier + mean_ret
        
        return shocked
    
    def simulate_correlation_breakdown(self, n_securities: int = 5) -> np.ndarray:
        """
        Simulate portfolio where correlations break down.
        
        Args:
            n_securities: Number of securities in portfolio
            
        Returns:
            Simulated returns (shape: n_observations, n_securities)
        """
        # In normal times: correlated
        # In stress: uncorrelated (diversification breaks)
        
        shock_dates = np.random.choice(
            self.n_observations,
            size=int(0.2 * self.n_observations),
            replace=False
        )
        
        simulated = np.zeros((self.n_observations, n_securities))
        
        for i in range(n_securities):
            simulated[:, i] = self.base_returns.copy()
            # Add stress period where correlations break down
            simulated[shock_dates, i] += np.random.normal(0, 0.05, 
                                                          len(shock_dates))
        
        return simulated
    
    def simulate_flash_crash(self, crash_probability: float = 0.01,
                            crash_magnitude: float = -0.08) -> np.ndarray:
        """
        Simulate occasional flash crashes.
        
        Args:
            crash_probability: Daily probability of crash
            crash_magnitude: Crash return (e.g., -0.08 for -8%)
            
        Returns:
            Simulated returns with crashes
        """
        simulated = self.base_returns.copy()
        
        for i in range(len(simulated)):
            if np.random.random() < crash_probability:
                simulated[i] = crash_magnitude
        
        return simulated
    
    def test_regime_robustness(self) -> dict:
        """
        Test strategy performance across regimes.
        
        Returns:
            Dictionary with performance metrics for each regime
        """
        def calculate_sharpe(returns):
            if len(returns) == 0 or np.std(returns) == 0:
                return np.nan
            return (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
        
        results = {}
        
        # Baseline
        baseline_strat_returns = self.base_returns * self.strategy_signal
        results['Baseline'] = {
            'sharpe': calculate_sharpe(baseline_strat_returns),
            'max_dd': np.max((np.maximum.accumulate(np.cumprod(1 + baseline_strat_returns)) 
                             - np.cumprod(1 + baseline_strat_returns)) / 
                            np.maximum.accumulate(np.cumprod(1 + baseline_strat_returns)))
        }
        
        # High volatility
        vol_shocked = self.simulate_volatility_shock(2.0)
        vol_strat_returns = vol_shocked * self.strategy_signal
        results['High Volatility (2x)'] = {
            'sharpe': calculate_sharpe(vol_strat_returns),
            'max_dd': np.max((np.maximum.accumulate(np.cumprod(1 + vol_strat_returns))
                             - np.cumprod(1 + vol_strat_returns)) /
                            np.maximum.accumulate(np.cumprod(1 + vol_strat_returns)))
        }
        
        # Flash crashes
        crash_shocked = self.simulate_flash_crash(crash_probability=0.02)
        crash_strat_returns = crash_shocked * self.strategy_signal
        results['Flash Crashes'] = {
            'sharpe': calculate_sharpe(crash_strat_returns),
            'max_dd': np.max((np.maximum.accumulate(np.cumprod(1 + crash_strat_returns))
                             - np.cumprod(1 + crash_strat_returns)) /
                            np.maximum.accumulate(np.cumprod(1 + crash_strat_returns)))
        }
        
        return results
    
    def print_regime_analysis(self) -> None:
        """Print market regime robustness analysis."""
        results = self.test_regime_robustness()
        
        print("\n" + "="*70)
        print("MARKET REGIME STRESS TEST")
        print("="*70)
        
        print(f"\n{'Regime':<25} {'Sharpe Ratio':>15} {'Max Drawdown':>15}")
        print("-" * 70)
        
        for regime, metrics in results.items():
            print(f"{regime:<25} {metrics['sharpe']:>15.3f} " +
                  f"{metrics['max_dd']*100:>14.2f}%")
        
        baseline_sharpe = results['Baseline']['sharpe']
        
        print("\n" + "-" * 70)
        print("Robustness Assessment:")
        
        for regime, metrics in results.items():
            if regime != 'Baseline':
                degradation = 1 - (metrics['sharpe'] / baseline_sharpe)
                print(f"  {regime}: {degradation*100:.1f}% degradation")
        
        print("="*70)


# Example
np.random.seed(42)
market_returns = np.random.normal(0.0005, 0.01, 1000)
signals = np.random.choice([1, -1], size=1000)

regime_sim = MarketRegimeSimulation(market_returns, signals)
regime_sim.print_regime_analysis()
```

## 20.3.4 Hansen's Superior Predictive Ability (SPA) Test

Compare multiple strategies. Which one is truly best?

Hansen's SPA test corrects for multiple comparison bias.

```python
class HansenSPATest:
    """
    Hansen's Superior Predictive Ability (SPA) test for strategy comparison.
    """
    
    def __init__(self, strategy_returns_list: list,
                 benchmark_returns: np.ndarray,
                 block_size: int = 20):
        """
        Initialize SPA test.
        
        Args:
            strategy_returns_list: List of strategy return arrays
            benchmark_returns: Benchmark return array
            block_size: Block size for bootstrap (for autocorrelation)
        """
        self.strategy_returns = strategy_returns_list
        self.benchmark_returns = benchmark_returns
        self.block_size = block_size
        self.n_strategies = len(strategy_returns_list)
        self.n_observations = len(benchmark_returns)
        
    def excess_returns(self, strat_returns: np.ndarray) -> np.ndarray:
        """Calculate excess returns vs benchmark."""
        return strat_returns - self.benchmark_returns
    
    def performance_statistic(self, excess_ret: np.ndarray) -> float:
        """
        Calculate performance metric: mean excess return / std(excess)
        
        Args:
            excess_ret: Excess returns array
            
        Returns:
            Performance metric
        """
        if np.std(excess_ret) == 0:
            return -np.inf
        
        return np.mean(excess_ret) / np.std(excess_ret)
    
    def block_bootstrap(self, data: np.ndarray, 
                       n_bootstrap: int = 1000) -> np.ndarray:
        """
        Block bootstrap to preserve autocorrelation.
        
        Args:
            data: Data array
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap statistics
        """
        n_blocks = self.n_observations // self.block_size
        stats = []
        
        for _ in range(n_bootstrap):
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            indices = np.concatenate([
                np.arange(i*self.block_size, (i+1)*self.block_size)
                for i in block_indices
            ])
            
            resampled = data[indices[:self.n_observations]]
            stats.append(self.performance_statistic(resampled))
        
        return np.array(stats)
    
    def spa_test(self, n_bootstrap: int = 1000) -> dict:
        """
        Perform SPA test.
        
        Returns:
            Dictionary with p-values and rankings
        """
        # Calculate performance for each strategy
        excess_returns_list = [
            self.excess_returns(sr) for sr in self.strategy_returns
        ]
        
        performance_stats = [
            self.performance_statistic(er) for er in excess_returns_list
        ]
        
        # Bootstrap distribution
        bootstrap_stats = []
        for er in excess_returns_list:
            bootstrap_stats.append(self.block_bootstrap(er, n_bootstrap))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # SPA test: max performance vs best in bootstrap
        observed_max = np.max(performance_stats)
        
        # P-value: probability that any bootstrap sample exceeds observed max
        bootstrap_max = np.max(bootstrap_stats, axis=0)
        p_value = np.mean(bootstrap_max > observed_max)
        
        # Ranking
        ranking = np.argsort(-np.array(performance_stats))
        
        return {
            'performance_stats': performance_stats,
            'p_value': p_value,
            'ranking': ranking,
            'bootstrap_stats': bootstrap_stats
        }
    
    def print_spa_summary(self, strategy_names: list = None) -> None:
        """Print SPA test results."""
        if strategy_names is None:
            strategy_names = [f"Strategy_{i}" for i in range(self.n_strategies)]
        
        results = self.spa_test()
        
        print("\n" + "="*70)
        print("HANSEN'S SUPERIOR PREDICTIVE ABILITY (SPA) TEST")
        print("="*70)
        
        print(f"\nTest Statistic (max performance): {np.max(results['performance_stats']):>8.3f}")
        print(f"SPA p-value:                      {results['p_value']:>8.4f}")
        
        if results['p_value'] < 0.05:
            print("✓ Best strategy is significantly superior (p < 0.05)")
        else:
            print("⚠️  No strategy is significantly superior")
        
        print(f"\nStrategy Rankings:")
        print("-" * 70)
        print(f"{'Rank':<8} {'Strategy':<30} {'Performance Stat':>15}")
        print("-" * 70)
        
        for rank, idx in enumerate(results['ranking']):
            print(f"{rank+1:<8} {strategy_names[idx]:<30} " +
                  f"{results['performance_stats'][idx]:>15.4f}")
        
        print("="*70)


# Example: Compare 3 strategies
np.random.seed(42)
benchmark = np.random.normal(0.0005, 0.01, 1000)
strat1 = benchmark + np.random.normal(0.0003, 0.005, 1000)  # Good
strat2 = benchmark + np.random.normal(0.0001, 0.008, 1000)  # Okay
strat3 = benchmark + np.random.normal(-0.0001, 0.007, 1000) # Poor

spa = HansenSPATest([strat1, strat2, strat3], benchmark)
spa.print_spa_summary(['Mean Reversion', 'Momentum', 'Contrarian'])
```

---

## Summary: Complete Validation Checklist

Before deploying a strategy to live trading:

**Module 20.1: Performance Analysis**
- [ ] Analyze underwater plot — understand maximum drawdown
- [ ] Calculate rolling Sharpe ratios — identify regime changes
- [ ] Decompose returns using factor model — isolate alpha from beta
- [ ] Quantify turnover and market impact — estimate real-world costs
- [ ] Test capacity — at what AUM does alpha erode?

**Module 20.2: Overfitting Detection**
- [ ] Calculate Deflated Sharpe Ratio — adjust for multiple testing
- [ ] Calculate PBO — measure probability of out-of-sample failure
- [ ] Run sensitivity analysis — test parameter robustness
- [ ] Compare in-sample vs out-of-sample performance

**Module 20.3: Monte Carlo Validation**
- [ ] Bootstrap confidence intervals — estimate true Sharpe uncertainty
- [ ] Random entry test — confirm signal quality, not luck
- [ ] Market regime simulation — stress test against crashes
- [ ] Hansen SPA test — if comparing multiple strategies, select best

**Red Flags That Kill Strategies:**
1. PBO > 80% (very likely to fail OOS)
2. Deflated SR < 0.5 (after multiple testing correction)
3. Random entry test shows similar returns (no alpha)
4. Strategy fails under 2x volatility or flash crashes
5. Net alpha < transaction costs at your target AUM

---

## Key Takeaways

1. **Backtests lie by default.** The only way to trust them is systematic validation.
2. **Multiple testing destroys alpha.** If you tested 500 parameter combinations, your Sharpe is probably 70% lower than reported.
3. **Out-of-sample is everything.** PBO > 50% means start over.
4. **Capacity kills alphas.** At 100 Cr AUM, your 20% gross alpha might become 2% net.
5. **Regime changes destroy strategies.** Test your strategy dies.

Build robust, validated systems. The market doesn't care about your in-sample Sharpe.

---

## References

- Arnott et al. (2016). "How Can 'Random' Managers Outperform Index Funds?"
- Bailey et al. (2015). "Probability of Backtest Overfitting"
- Bailey et al. (2014). "The Sharpe Ratio Efficient Frontier"
- Hansen & Lunde (2005). "A Forecast Comparison of Volatility Models"
- White (2000). "A Reality Check for Data Snooping"

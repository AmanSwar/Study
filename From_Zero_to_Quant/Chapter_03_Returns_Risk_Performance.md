# Chapter 3: Returns, Risk, and Performance Measurement

## Chapter Overview

Welcome to the heart of quantitative trading. While technical systems, execution engines, and data pipelines are critical infrastructure, **nothing matters if you can't measure whether your strategy actually works**. This chapter transforms raw market data into actionable insights about strategy performance.

As an ML/systems engineer entering finance, you're accustomed to metrics like accuracy, precision, and F1 scores. Financial metrics follow similar logic but capture fundamentally different phenomena:
- **Returns** measure how much money you made (or lost)
- **Risk** measures how much you might lose
- **Performance metrics** synthesize both into a single score

By the end of this chapter, you'll understand:
1. Why simple returns and log returns are mathematically different (and when each matters)
2. How to correctly compute volatility, VaR, and drawdowns on live trading data
3. Why the Sharpe ratio sometimes lies, and what alpha actually means
4. How to implement all major performance metrics in production Python code

**Prerequisites**: Basic linear algebra, probability distributions, and familiarity with NumPy/Pandas.

---

## Module 3.1: Returns

### 3.1.1 Simple Returns

The most intuitive return measure is simple (arithmetic) return:

$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

where $P_t$ is the price at time $t$.

**Example**: If a stock goes from ₹100 to ₹110, the simple return is:
$$R = \frac{110 - 100}{100} = 0.10 \text{ or } 10\%$$

**Properties of simple returns**:
- Intuitive: represents percentage change in investment value
- Non-compounding in time: if you get 10% return, then 10% return again, total isn't 20%
- Asymmetric: a -50% return followed by +100% gets you back to the start (not +50%)

### 3.1.2 Log Returns

Log returns (also called continuous compounding returns) use natural logarithm:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$

**Key insight**: Log returns are **perfectly additive** across time:

$$r_{t-n:t} = \sum_{i=0}^{n-1} r_{t-i} = \ln\left(\frac{P_t}{P_{t-n}}\right)$$

This is why log returns dominate quantitative finance: they simplify multiperiod analysis.

**Relationship to simple returns**:
$$r_t \approx R_t \text{ for small returns (|R_t| < 10\%)}$$
$$r_t = \ln(1 + R_t)$$

**When to use which**:
- **Simple returns**: Reporting to non-technical stakeholders, computing Sharpe ratios with multiperiod data
- **Log returns**: Time series analysis, risk models, factor models, anywhere you need additivity

### 3.1.3 Multi-Period Returns

For simple returns over $n$ periods:
$$R_{t-n:t} = \prod_{i=0}^{n-1} (1 + R_{t-i}) - 1$$

Example: Two consecutive 10% returns yield:
$$R = (1.10)(1.10) - 1 = 0.21 \text{ or } 21\%$$

For log returns, it's just summation (much cleaner):
$$r_{t-n:t} = \sum_{i=0}^{n-1} r_{t-i}$$

### 3.1.4 Annualization

A critical operation: converting returns to annual frequency.

**Log returns**: If you have $n$ periods per year:
$$r_{\text{annual}} = 252 \times r_{\text{daily}}$$

The famous $\sqrt{252}$ rule applies to volatility:
$$\sigma_{\text{annual}} = \sqrt{252} \times \sigma_{\text{daily}}$$

This assumes returns are independent (which they usually aren't—a critical limitation).

**For simple returns**: Use geometric mean for multiperiod:
$$R_{\text{annual}} = \left(\prod_{i=1}^{252} (1 + R_i)\right)^{1/252} - 1 \approx \exp(r_{\text{annual}})$$

**When $\sqrt{252}$ breaks down**:
- High-frequency strategies with mean reversion (negative autocorrelation)
- Crisis periods with correlated losses
- Strategies with specific time-of-day patterns
- Always check empirical autocorrelation!

### 3.1.5 Excess Returns

Excess return compares your return to a baseline:

$$R^e_t = R_t - R^f_t$$

where $R^f_t$ is the risk-free rate. In India, use the NSE MIBOR (Mumbai Interbank Offer Rate) or 10-year GSec yield.

**Benchmark-adjusted excess return**:
$$R^e_t = R_t - R^{\text{benchmark}}_t$$

This measures how much you beat (or lost to) the market.

**Sector-adjusted excess return**:
$$R^e_t = R_t - R^{\text{sector}}_t$$

### 3.1.6 Implementation: Returns Module

```python
import numpy as np
import pandas as pd
from typing import Tuple, Union
from datetime import datetime, timedelta

class ReturnsCalculator:
    """Compute various return measures on price time series."""
    
    @staticmethod
    def simple_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate simple (arithmetic) returns.
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of simple returns
        """
        return prices.pct_change()
    
    @staticmethod
    def log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate log returns.
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of log returns (natural logarithm)
        """
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def excess_returns(
        returns: pd.Series,
        risk_free_rate: Union[float, pd.Series],
        periods_per_year: int = 252
    ) -> pd.Series:
        """
        Calculate excess returns above risk-free rate.
        
        Args:
            returns: Series of returns (daily, typically)
            risk_free_rate: Annual risk-free rate or daily series
            periods_per_year: Trading days per year (default 252 for India)
            
        Returns:
            Series of excess returns
        """
        if isinstance(risk_free_rate, (int, float)):
            daily_rf = risk_free_rate / periods_per_year
        else:
            daily_rf = risk_free_rate / periods_per_year
        
        return returns - daily_rf
    
    @staticmethod
    def cumulative_returns(
        returns: pd.Series,
        return_type: str = 'log'
    ) -> pd.Series:
        """
        Calculate cumulative returns over time.
        
        Args:
            returns: Series of simple or log returns
            return_type: 'simple' or 'log'
            
        Returns:
            Cumulative returns starting from 1.0 (or 0% for log)
        """
        if return_type == 'log':
            return np.exp(returns.cumsum())
        else:  # simple
            return (1 + returns).cumprod()
    
    @staticmethod
    def annualize_return(
        total_return: float,
        years: float,
        return_type: str = 'simple'
    ) -> float:
        """
        Annualize a total return.
        
        Args:
            total_return: Total return (e.g., 0.25 for 25%)
            years: Number of years
            return_type: 'simple' or 'log'
            
        Returns:
            Annualized return
        """
        if return_type == 'log':
            return total_return / years
        else:  # simple
            return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def rolling_returns(
        prices: pd.Series,
        window: int = 252,
        return_type: str = 'simple'
    ) -> pd.Series:
        """
        Calculate rolling period returns.
        
        Args:
            prices: Series of prices
            window: Window size in periods
            return_type: 'simple' or 'log'
            
        Returns:
            Series of rolling returns
        """
        if return_type == 'log':
            log_prices = np.log(prices)
            return log_prices.diff(window)
        else:  # simple
            return (prices / prices.shift(window) - 1)

# Example usage
if __name__ == "__main__":
    # Create sample NSE data
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 252)))
    price_series = pd.Series(prices, index=dates)
    
    rc = ReturnsCalculator()
    
    # Calculate returns
    simple_ret = rc.simple_returns(price_series)
    log_ret = rc.log_returns(price_series)
    
    print(f"Mean simple return: {simple_ret.mean():.4f}")
    print(f"Mean log return: {log_ret.mean():.4f}")
    print(f"Total simple return: {(price_series.iloc[-1] / price_series.iloc[0] - 1):.4f}")
    print(f"Total log return: {np.log(price_series.iloc[-1] / price_series.iloc[0]):.4f}")
    
    # Annualize
    total_ret = price_series.iloc[-1] / price_series.iloc[0] - 1
    annual_ret = rc.annualize_return(total_ret, years=1)
    print(f"Annualized return: {annual_ret:.4f}")
```

### [VISUALIZATION] Return Distributions

```
Daily Returns Distribution (Nifty 50) vs Normal Distribution
───────────────────────────────────────────────────────────
Frequency
    │
    │         ╭─╮
    │     ╭───╯ ╰───╮
    │   ╭─╯   ╭╮    ╰─╮     ← Normal fit
    │  ╭╯    ╭╯╰╮     ╰╮
    │ ╭╯    ╱    ╲     ╰╮
    │ │    │      │      │    ← Actual (fatter tails!)
    │ │    │ ▄▄▄▄▄│▄▄▄▄  │
    │ │    │ ▌   ▌│▌  ▌  │
    └─┴────┴─┴───┴─┴──┴──┴── Daily Return
      -3%  -1%  0%  1%  3%
      
      Fat tails: More extreme moves than normal predicts!
```

### [WARNING] Common Return Mistakes

1. **Mixing simple and log returns**: Be explicit in calculations
2. **Assuming independent returns**: They're not! Use empirical statistics
3. **Not accounting for dividends**: NSE prices are usually adjusted, but verify
4. **Assuming perfect $\sqrt{252}$ scaling**: Check autocorrelation first
5. **Forgetting about compounding**: 50% loss → 100% gain ≠ 50% gain

### Exercises

1. Verify numerically that log returns sum to total return while simple returns multiply
2. Download 1 year of Nifty 50 data (NSE). Compute both return types. What's the correlation?
3. Plot rolling 252-day returns. When does autocorrelation spike?

---

## Module 3.2: Risk Measures

### 3.2.1 Volatility: Foundation of Risk

Volatility measures the standard deviation of returns—how much returns vary:

$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (r_i - \bar{r})^2}$$

**Annualized volatility**: $\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$

For NSE stocks, typical volatility ranges:
- Large cap (e.g., TCS, Reliance): 15–25% annual
- Mid cap: 30–40% annual
- Small cap: 40–60% annual

**Historical volatility**: Using a rolling window (e.g., 30-day window):

$$\sigma_t = \sqrt{\frac{1}{n} \sum_{i=0}^{n-1} (r_{t-i} - \bar{r}_{t})^2}$$

### 3.2.2 Exponentially Weighted Moving Average (EWMA) Volatility

Recent observations matter more than old ones. EWMA downweights historical data:

$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2$$

where $\lambda$ is the decay factor (typical: 0.94).

**Key insight**: EWMA responds faster to market regime changes than simple moving averages.

### 3.2.3 Downside Risk (Semi-Variance)

Volatility treats gains and losses equally. But investors fear losses more. Semi-variance captures downside only:

$$\text{Semi-Variance} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \min(r_i - r_{\text{threshold}}, 0)^2}$$

Common thresholds: 0 (losses), mean return, or risk-free rate.

### 3.2.4 Value at Risk (VaR)

VaR answers: "What's my worst-case loss at a given confidence level?"

$$\text{VaR}_{\alpha} = -F^{-1}(\alpha)$$

where $F$ is the CDF of returns. VaR at 95% confidence is the loss that happens in 5% of cases.

**Three methods**:

#### Parametric VaR (assumes normal distribution)

$$\text{VaR}_{\alpha} = \mu - \sigma \times z_{\alpha}$$

where $z_{\alpha}$ is the critical value from a normal distribution.

**Example**: If daily returns have $\mu = 0.05\%$ and $\sigma = 1.2\%$:
$$\text{VaR}_{95\%} = 0.05\% - 1.2\% \times 1.645 = -1.92\%$$

#### Historical VaR (no distributional assumption)

Sort returns from worst to best. VaR is the $(1-\alpha) \times 100$-th percentile:

```
Sorted returns: [-5%, -4%, -3%, -2%, -1%, 0%, 1%, 2%, 3%, 4%]
VaR at 95%: 5th percentile = -4%
```

#### Monte Carlo VaR (simulate many paths)

1. Estimate return distribution parameters from historical data
2. Simulate thousands of returns from that distribution
3. Take the relevant percentile

### 3.2.5 Conditional Value at Risk (CVaR) / Expected Shortfall

VaR tells you the threshold; CVaR tells you what you expect to lose **beyond** that threshold:

$$\text{CVaR}_{\alpha} = \mathbb{E}[L | L > \text{VaR}_{\alpha}]$$

CVaR is always worse than VaR (more conservative) and better accounts for tail risk.

### 3.2.6 Maximum Drawdown & Underwater Plots

Drawdown: the peak-to-trough decline during a strategy:

$$\text{Drawdown}_t = \frac{P_t - P_{\max}(0 \to t)}{P_{\max}(0 \to t)}$$

**Maximum drawdown**: The worst peak-to-trough drop.

This is **crucial** for traders: a strategy with great returns but 70% drawdown can wipe you out psychologically or through margin calls.

### 3.2.7 Why Volatility Underestimates Tail Risk

Returns aren't normally distributed—they have **fat tails**. This means:
- Extreme moves happen more often than normal distribution predicts
- Parametric VaR is optimistic (underestimates true risk)
- Historical VaR is better but requires more data
- Use CVaR to be conservative

### 3.2.8 Implementation: Risk Measures

```python
class RiskMeasures:
    """Compute risk measures for trading strategies."""
    
    @staticmethod
    def volatility(
        returns: pd.Series,
        periods_per_year: int = 252,
        annualize: bool = True
    ) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year
            annualize: Whether to annualize
            
        Returns:
            Volatility (annualized if requested)
        """
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(periods_per_year)
        return vol
    
    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 30,
        periods_per_year: int = 252,
        annualize: bool = True
    ) -> pd.Series:
        """Calculate rolling volatility."""
        rolling_vol = returns.rolling(window).std()
        if annualize:
            rolling_vol *= np.sqrt(periods_per_year)
        return rolling_vol
    
    @staticmethod
    def ewma_volatility(
        returns: pd.Series,
        span: int = 30,
        periods_per_year: int = 252,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate EWMA volatility.
        
        Args:
            returns: Series of returns
            span: EWMA span (decay = 2/(span+1))
            periods_per_year: Trading periods per year
            annualize: Whether to annualize
            
        Returns:
            EWMA volatility series
        """
        ewma_var = returns.ewm(span=span).var()
        ewma_vol = np.sqrt(ewma_var)
        if annualize:
            ewma_vol *= np.sqrt(periods_per_year)
        return ewma_vol
    
    @staticmethod
    def semi_variance(
        returns: pd.Series,
        threshold: float = 0.0,
        periods_per_year: int = 252,
        annualize: bool = True
    ) -> float:
        """
        Calculate semi-variance (downside risk only).
        
        Args:
            returns: Series of returns
            threshold: Threshold return (default 0)
            periods_per_year: Trading periods per year
            annualize: Whether to annualize
            
        Returns:
            Semi-variance (annualized if requested)
        """
        downside = np.minimum(returns - threshold, 0)
        semivar = np.sqrt((downside ** 2).mean())
        if annualize:
            semivar *= np.sqrt(periods_per_year)
        return semivar
    
    @staticmethod
    def var_parametric(
        returns: pd.Series,
        confidence: float = 0.95,
        periods_per_year: int = 252
    ) -> Tuple[float, float]:
        """
        Calculate parametric VaR assuming normal distribution.
        
        Args:
            returns: Series of daily returns
            confidence: Confidence level (e.g., 0.95)
            periods_per_year: Trading periods per year
            
        Returns:
            Tuple of (daily VaR, annual VaR)
        """
        from scipy import stats
        
        mu = returns.mean()
        sigma = returns.std()
        z_alpha = stats.norm.ppf(1 - confidence)
        
        var_daily = mu - sigma * z_alpha
        var_annual = mu * periods_per_year - sigma * np.sqrt(periods_per_year) * z_alpha
        
        return var_daily, var_annual
    
    @staticmethod
    def var_historical(
        returns: pd.Series,
        confidence: float = 0.95,
        periods_per_year: int = 252
    ) -> Tuple[float, float]:
        """
        Calculate historical VaR using percentiles.
        
        Args:
            returns: Series of daily returns
            confidence: Confidence level
            periods_per_year: Trading periods per year
            
        Returns:
            Tuple of (daily VaR, annual VaR)
        """
        percentile = (1 - confidence) * 100
        var_daily = np.percentile(returns, percentile)
        
        # Approximate annual using daily * sqrt(252)
        var_annual = var_daily * np.sqrt(periods_per_year)
        
        return var_daily, var_annual
    
    @staticmethod
    def var_monte_carlo(
        returns: pd.Series,
        confidence: float = 0.95,
        simulations: int = 100000,
        periods_per_year: int = 252
    ) -> Tuple[float, float]:
        """
        Calculate Monte Carlo VaR.
        
        Args:
            returns: Series of daily returns
            confidence: Confidence level
            simulations: Number of simulations
            periods_per_year: Trading periods per year
            
        Returns:
            Tuple of (daily VaR, annual VaR)
        """
        mu = returns.mean()
        sigma = returns.std()
        
        # Simulate returns from normal distribution
        simulated = np.random.normal(mu, sigma, simulations)
        
        percentile = (1 - confidence) * 100
        var_daily = np.percentile(simulated, percentile)
        var_annual = var_daily * np.sqrt(periods_per_year)
        
        return var_daily, var_annual
    
    @staticmethod
    def cvar(
        returns: pd.Series,
        confidence: float = 0.95,
        periods_per_year: int = 252
    ) -> Tuple[float, float]:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Series of daily returns
            confidence: Confidence level
            periods_per_year: Trading periods per year
            
        Returns:
            Tuple of (daily CVaR, annual CVaR)
        """
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)
        cvar_daily = returns[returns <= var_threshold].mean()
        cvar_annual = cvar_daily * periods_per_year
        
        return cvar_daily, cvar_annual
    
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and when it occurred.
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Tuple of (max_drawdown, start_date, end_date)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        
        # Find when the peak occurred
        start_date = cumulative[:end_date].idxmax()
        
        return max_dd, start_date, end_date
    
    @staticmethod
    def underwater_plot_data(returns: pd.Series) -> pd.Series:
        """
        Generate data for underwater plot (drawdown vs time).
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Series of drawdowns over time
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

# Example usage
if __name__ == "__main__":
    # Using sample returns
    returns = pd.Series(
        np.random.normal(0.0005, 0.01, 252),
        index=pd.date_range('2023-01-01', periods=252, freq='B')
    )
    
    rm = RiskMeasures()
    
    # Calculate risk measures
    vol = rm.volatility(returns)
    semivar = rm.semi_variance(returns)
    var_param = rm.var_parametric(returns)
    var_hist = rm.var_historical(returns)
    cvar = rm.cvar(returns)
    mdd, start, end = rm.maximum_drawdown(returns)
    
    print(f"Annualized Volatility: {vol:.4f} ({vol*100:.2f}%)")
    print(f"Semi-Variance: {semivar:.4f} ({semivar*100:.2f}%)")
    print(f"Parametric VaR (95%): {var_param[0]:.4f} ({var_param[0]*100:.2f}%)")
    print(f"Historical VaR (95%): {var_hist[0]:.4f} ({var_hist[0]*100:.2f}%)")
    print(f"CVaR (95%): {cvar[0]:.4f} ({cvar[0]*100:.2f}%)")
    print(f"Maximum Drawdown: {mdd:.4f} ({mdd*100:.2f}%)")
    print(f"  From {start.date()} to {end.date()}")
```

### [VISUALIZATION] VaR Comparison

```
Three Methods to Estimate 95% VaR
──────────────────────────────────

Parametric (Assumes Normal):
    Returns │      
           │    ╭─╮
           │  ╭─╯ ╰─╮
           │ ╱       ╲     ← Normal curve
           │╱         ╲
    ───────┴───────────┴─── 
           ↑ VaR
           
Historical (No Assumptions):
    Count │ ▌
          │ ▌ ▌
          │ ▌ ▌ ▌
          │ ▌ ▌ ▌ ▌    ← Actual distribution (fat tails!)
    ──────┴─┴─┴─┴──
          ↑ VaR
          
Monte Carlo (Stochastic):
    Results │         ╭╮
            │       ╭─╯╰─╮
            │     ╭╯     ╰─╮
            │   ╭╯         ╰─  ← Simulated paths
    ────────┴───┴───────────┴──
            ↑ VaR
```

### [VISUALIZATION] Underwater Plot Example

```
Cumulative Returns & Drawdown
──────────────────────────────
Cumulative
Return    │
   1.50   │                    ╭─╯╲
   1.25   │              ╭─╯╲ ╱   ╲
   1.00   │    ╭─╯╲  ╭─╯   ╲╱     ╱╯
   0.75   │   ╱   ╲╱╯              
   0.50   │╭─╯                    
   0.25   │╯
    0.00  │
        └──────────────────────────
Drawdown │  ▌
(%)      │  ▌▌▌
        │  ▌▌▌▌▌      ← Max Drawdown ~40%
         │  ▌▌▌▌▌▌▌▌
        └──────────────────────────
         Time
```

### Exercises

1. Download 2 years of NSE data. Compare rolling 30-day volatility to EWMA. Which responds faster to spikes?
2. Compute parametric, historical, and Monte Carlo VaR for the same data. Which gives the most conservative estimate?
3. Find the maximum drawdown period. Plot the underwater plot.

---

## Module 3.3: Performance Metrics

### 3.3.1 Sharpe Ratio

The most famous risk-adjusted return metric:

$$\text{Sharpe} = \frac{\mathbb{E}[R^e]}{\sigma(R^e)} = \frac{\mu - r_f}{\sigma}$$

where $R^e$ is excess return and $r_f$ is the risk-free rate.

**Interpretation**:
- > 1.0: Good
- > 1.5: Very good
- > 2.0: Excellent
- Negative: Strategy underperforms risk-free rate

**Annualization**:
$$\text{Sharpe}_{\text{annual}} = \text{Sharpe}_{\text{daily}} \times \sqrt{252}$$

**Issues**:
- Assumes returns are normal (they're not)
- Penalizes upside volatility equally with downside
- Can be gamed with high-frequency strategies that boost the metric without true profit
- Confidence intervals matter: you need significant data to trust the ratio

### 3.3.2 Sortino Ratio

Uses only downside volatility (semi-variance), penalizing losses but not gains:

$$\text{Sortino} = \frac{\mu - r_f}{\sigma_{\text{downside}}}$$

For a strategy with volatility from luck vs. skill, Sortino is higher than Sharpe.

### 3.3.3 Calmar Ratio

Balances returns against maximum drawdown:

$$\text{Calmar} = \frac{\text{Annual Return}}{\text{Maximum Drawdown}}$$

Good for risk-averse investors who fear big losses more than volatility.

### 3.3.4 Information Ratio

Measures excess return relative to a benchmark:

$$\text{Information Ratio} = \frac{\mu(R - R^{\text{benchmark}})}{\sigma(R - R^{\text{benchmark}})}$$

The tracking error denominator is volatility of returns relative to benchmark, not absolute volatility.

### 3.3.5 Win Rate & Profit Factor

**Win rate**: Percentage of trades with positive P&L
$$\text{Win Rate} = \frac{\text{# Winning Trades}}{\text{Total Trades}}$$

**Profit factor**: Gross profit / Gross loss
$$\text{Profit Factor} = \frac{\sum \text{Winning Trades}}{\sum |\text{Losing Trades}|}$$

- Profit factor > 1.5 is good
- Profit factor > 2.0 is excellent
- Both metrics can be misleading (high win rate with small wins + rare big loss)

### 3.3.6 Average Win / Average Loss

$$\text{Avg Win} = \frac{\sum \text{Winning Trades}}{\text{# Winning Trades}}$$
$$\text{Avg Loss} = \frac{\sum |\text{Losing Trades}|}{\text{# Losing Trades}}$$

**Win/Loss Ratio**: 
$$\frac{\text{Avg Win}}{\text{Avg Loss}}$$

A good strategy often has Avg Win > 2x Avg Loss, compensating for lower win rate.

### 3.3.7 Turnover

Measures trading frequency (important for transaction costs):

$$\text{Turnover} = \frac{\sum |\text{Position Change}_t|}{2 \times \text{Average Portfolio Value}_t}$$

Low turnover (< 1x annually) → Low transaction costs.
High turnover (> 10x annually) → Significant costs, needs edge to overcome.

### 3.3.8 Comprehensive Metrics Table

| Metric | Formula | Interpretation | Good | Excellent |
|--------|---------|-----------------|------|-----------|
| **Sharpe Ratio** | $\frac{\mu - r_f}{\sigma}$ | Return per unit risk | > 1.0 | > 2.0 |
| **Sortino Ratio** | $\frac{\mu - r_f}{\sigma_{down}}$ | Return per unit downside | > 1.5 | > 3.0 |
| **Calmar Ratio** | $\frac{\text{Ret}}{\text{Max DD}}$ | Return per unit drawdown | > 0.5 | > 1.0 |
| **Information Ratio** | $\frac{\mu(R-R^{b})}{\sigma(R-R^{b})}$ | Outperformance | > 0.5 | > 1.0 |
| **Win Rate** | $\frac{\#Win}{\#Total}$ | % profitable trades | > 50% | > 60% |
| **Profit Factor** | $\frac{\sum Win}{\sum \|Loss\|}$ | Total profit/loss | > 1.5 | > 2.0 |
| **Avg Win / Loss Ratio** | $\frac{\text{Avg Win}}{\text{Avg Loss}}$ | Win size / Loss size | > 1.0 | > 2.0 |

### 3.3.9 Implementation: Performance Metrics

```python
class PerformanceMetrics:
    """Compute comprehensive performance metrics."""
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            annualize: Whether to annualize
            
        Returns:
            Sharpe ratio
        """
        excess = returns - (risk_free_rate / periods_per_year)
        sharpe = excess.mean() / excess.std()
        
        if annualize:
            sharpe *= np.sqrt(periods_per_year)
        
        return sharpe
    
    @staticmethod
    def sharpe_confidence_interval(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate 95% confidence interval for Sharpe ratio.
        
        Uses approximation: var(Sharpe) ≈ (1 + 0.5*Sharpe²) / n
        """
        from scipy import stats
        
        n = len(returns)
        sharpe = PerformanceMetrics.sharpe_ratio(
            returns, risk_free_rate, periods_per_year
        )
        
        # Variance of Sharpe estimator
        var_sharpe = (1 + 0.5 * sharpe ** 2) / n
        se_sharpe = np.sqrt(var_sharpe)
        
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = sharpe - z * se_sharpe
        upper = sharpe + z * se_sharpe
        
        return lower, upper
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
        annualize: bool = True,
        threshold: float = None
    ) -> float:
        """
        Calculate Sortino ratio (penalizes downside only).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            annualize: Whether to annualize
            threshold: Threshold return (default: risk-free rate)
            
        Returns:
            Sortino ratio
        """
        if threshold is None:
            threshold = risk_free_rate / periods_per_year
        
        excess = returns - threshold
        downside = np.minimum(excess, 0)
        downside_vol = np.sqrt((downside ** 2).mean())
        
        sortino = excess.mean() / downside_vol
        
        if annualize:
            sortino *= np.sqrt(periods_per_year)
        
        return sortino
    
    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Args:
            returns: Series of daily returns
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * periods_per_year
        max_dd, _, _ = RiskMeasures.maximum_drawdown(returns)
        
        if max_dd >= 0:  # No drawdown
            return np.inf
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252,
        annualize: bool = True
    ) -> float:
        """
        Calculate Information Ratio vs benchmark.
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            periods_per_year: Trading periods per year
            annualize: Whether to annualize
            
        Returns:
            Information ratio
        """
        tracking_error = returns - benchmark_returns
        ir = tracking_error.mean() / tracking_error.std()
        
        if annualize:
            ir *= np.sqrt(periods_per_year)
        
        return ir
    
    @staticmethod
    def win_metrics(trades: pd.Series) -> dict:
        """
        Calculate trading win rate metrics.
        
        Args:
            trades: Series of trade P&L (positive = win, negative = loss)
            
        Returns:
            Dictionary with win metrics
        """
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_trades = len(trades)
        
        if total_trades == 0:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'win_loss_ratio': 0
            }
        
        total_wins = winning_trades.sum()
        total_losses = abs(losing_trades.sum()) if loss_count > 0 else 1e-10
        
        return {
            'win_rate': win_count / total_trades,
            'profit_factor': total_wins / total_losses if total_losses > 0 else np.inf,
            'avg_win': total_wins / win_count if win_count > 0 else 0,
            'avg_loss': total_losses / loss_count if loss_count > 0 else 0,
            'win_loss_ratio': (total_wins / win_count) / (total_losses / loss_count) 
                              if (win_count > 0 and loss_count > 0) else 0,
            'consecutive_wins': max_consecutive(trades > 0),
            'consecutive_losses': max_consecutive(trades < 0)
        }
    
    @staticmethod
    def turnover(
        portfolio_values: pd.Series,
        position_changes: pd.Series
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Args:
            portfolio_values: Series of portfolio values
            position_changes: Series of abs position changes
            
        Returns:
            Annual turnover ratio
        """
        total_position_change = position_changes.sum()
        avg_portfolio_value = portfolio_values.mean()
        
        return total_position_change / (2 * avg_portfolio_value)
    
    @staticmethod
    def max_consecutive_wins_losses(trades: pd.Series) -> Tuple[int, int]:
        """
        Find maximum consecutive wins and losses.
        
        Args:
            trades: Series of trade P&L
            
        Returns:
            Tuple of (max_wins, max_losses)
        """
        is_win = (trades > 0).astype(int)
        is_loss = (trades < 0).astype(int)
        
        def max_consecutive(mask):
            groups = (mask != mask.shift()).cumsum()
            return mask.groupby(groups).sum().max()
        
        max_wins = max_consecutive(is_win)
        max_losses = max_consecutive(is_loss)
        
        return int(max_wins), int(max_losses)
    
    @staticmethod
    def generate_report(
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        trades: pd.Series = None,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> dict:
        """
        Generate comprehensive performance report.
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Optional benchmark returns
            trades: Optional series of trade P&L
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Dictionary with all metrics
        """
        report = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * periods_per_year,
            'volatility': returns.std() * np.sqrt(periods_per_year),
            'sharpe': PerformanceMetrics.sharpe_ratio(returns, risk_free_rate),
            'sortino': PerformanceMetrics.sortino_ratio(returns, risk_free_rate),
            'calmar': PerformanceMetrics.calmar_ratio(returns),
        }
        
        if benchmark_returns is not None:
            report['information_ratio'] = PerformanceMetrics.information_ratio(
                returns, benchmark_returns
            )
        
        if trades is not None:
            report['win_metrics'] = PerformanceMetrics.win_metrics(trades)
        
        max_dd, start, end = RiskMeasures.maximum_drawdown(returns)
        report['max_drawdown'] = max_dd
        report['max_drawdown_period'] = (start, end)
        
        return report

def max_consecutive(mask: pd.Series) -> int:
    """Helper function for max consecutive calculation."""
    groups = (mask != mask.shift()).cumsum()
    return int(mask.groupby(groups).sum().max())

# Example usage
if __name__ == "__main__":
    # Sample returns
    returns = pd.Series(
        np.random.normal(0.001, 0.01, 252),
        index=pd.date_range('2023-01-01', periods=252, freq='B')
    )
    
    pm = PerformanceMetrics()
    
    # Calculate metrics
    sharpe = pm.sharpe_ratio(returns)
    sharpe_ci = pm.sharpe_confidence_interval(returns)
    sortino = pm.sortino_ratio(returns)
    calmar = pm.calmar_ratio(returns)
    
    print(f"Sharpe Ratio: {sharpe:.4f} (95% CI: [{sharpe_ci[0]:.4f}, {sharpe_ci[1]:.4f}])")
    print(f"Sortino Ratio: {sortino:.4f}")
    print(f"Calmar Ratio: {calmar:.4f}")
    
    # Win metrics
    trades = pd.Series(np.random.normal(50, 150, 50))  # Sample trade P&L
    win_metrics = pm.win_metrics(trades)
    print(f"\nTrade Metrics:")
    for metric, value in win_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
```

### [VISUALIZATION] Performance Dashboard

```
Annual Performance Summary (NSE-based Strategy)
──────────────────────────────────────────────
Total Return:     +18.5%        Sharpe Ratio:     1.32
Annual Return:    +18.5%        Sortino Ratio:    1.95
Volatility:       14.0%         Calmar Ratio:     0.47
Max Drawdown:     -39.2%        Win Rate:         52.3%
Consecutive Wins: 7             Profit Factor:    1.78

Monthly Returns:
        Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
        +2%  +1% -1%  +3%  +2%  -2%  +4%  +3%  -3%  +2%  +1%  +1%
```

### Exercises

1. Compute Sharpe, Sortino, and Calmar ratios for a strategy. Which varies most across time windows?
2. Generate 100 random strategies. Plot win rate vs profit factor. What's the relationship?
3. Download real NSE data + Nifty benchmark. Compute Information Ratio vs benchmark.

---

## Module 3.4: Alpha, Beta, and Factor Exposure

### 3.4.1 Capital Asset Pricing Model (CAPM): Foundation

CAPM derives the required return based on risk:

$$\mathbb{E}[R_i] = r_f + \beta_i (\mathbb{E}[R_m] - r_f)$$

where:
- $\mathbb{E}[R_i]$ = expected return on asset $i$
- $r_f$ = risk-free rate
- $\beta_i$ = sensitivity to market returns
- $\mathbb{E}[R_m] - r_f$ = market risk premium

**Interpretation**: 
- If $\beta = 1$: Asset moves in line with market
- If $\beta = 1.5$: Asset is 50% more volatile than market (high risk, high reward)
- If $\beta = 0.5$: Asset is 50% less volatile (low risk, low reward)

**In India**: Use Nifty 50 as the market proxy, 10-year GSec yield as $r_f$.

### 3.4.2 Beta Estimation: OLS Regression

Beta is estimated via ordinary least squares (OLS):

$$\beta = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$$

This is equivalent to regressing asset returns against market returns:

$$R_i = \alpha + \beta R_m + \epsilon$$

where $\alpha$ is the intercept (Jensen's alpha) and $\epsilon$ is idiosyncratic error.

**Rolling Beta**: Reestimate over a moving window (e.g., 252 days) to track regime changes.

**Bayesian Beta**: Use prior belief + historical data. Shrinks extreme estimates toward 1.0.

### 3.4.3 Jensen's Alpha

Once you know required return from CAPM, alpha measures outperformance:

$$\alpha = \mathbb{E}[R_i] - (r_f + \beta_i (\mathbb{E}[R_m] - r_f))$$

Or from regression: alpha is the intercept in $R_i = \alpha + \beta R_m + \epsilon$.

**Interpretation**:
- $\alpha > 0$: Strategy beats CAPM prediction (skill!)
- $\alpha = 0$: Fair value (no outperformance)
- $\alpha < 0$: Strategy underperforms (bad)

**Why alpha is valuable**: It's the "free lunch"—excess return not explained by taking market risk.

**Why beta is free**: You get it for taking market risk; no skill required.

### 3.4.4 Rolling Beta

Beta changes over time as correlations shift:

```python
def rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """Estimate rolling beta."""
    covariance = stock_returns.rolling(window).cov(market_returns)
    market_variance = market_returns.rolling(window).var()
    beta = covariance / market_variance
    return beta
```

### 3.4.5 Fama-French 3-Factor Model

CAPM assumes only market risk matters. Empirically, two additional factors explain returns:

$$R_i = r_f + \beta_m (R_m - r_f) + \beta_{smb} \cdot \text{SMB} + \beta_{hml} \cdot \text{HML} + \epsilon$$

where:
- **SMB (Small Minus Big)**: Returns of small-cap minus large-cap stocks
- **HML (High Minus Low)**: Returns of high book-to-market (value) minus low book-to-market (growth)

**In India**: Construct SMB/HML factors using NSE data:
- SMB: Return difference between bottom 200 market cap and top 200 market cap stocks
- HML: Return difference between high P/B ratio and low P/B ratio stocks

### 3.4.6 Carhart 4-Factor Model

Extends Fama-French by adding momentum:

$$R_i = r_f + \beta_m (R_m - r_f) + \beta_{smb} \cdot \text{SMB} + \beta_{hml} \cdot \text{HML} + \beta_{mom} \cdot \text{MOM} + \epsilon$$

where **MOM** = return difference between high-momentum and low-momentum stocks (based on 12-month returns).

**Empirical finding**: Momentum is strongest predictor of 3-12 month returns.

### 3.4.7 Implementation: Factor Models

```python
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

class FactorModels:
    """Implement CAPM, Fama-French, and Carhart models."""
    
    @staticmethod
    def estimate_beta_ols(
        stock_returns: pd.Series,
        market_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Estimate beta via OLS regression.
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            
        Returns:
            Tuple of (beta, alpha, r_squared)
        """
        # Align indices
        data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        # OLS: stock = alpha + beta * market
        slope, intercept, r_value, _, _ = linregress(
            data['market'], data['stock']
        )
        
        return slope, intercept, r_value ** 2
    
    @staticmethod
    def rolling_beta(
        stock_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """
        Estimate rolling beta.
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            window: Rolling window size
            
        Returns:
            Series of rolling betas
        """
        betas = []
        dates = []
        
        for i in range(window, len(stock_returns)):
            stock_window = stock_returns.iloc[i-window:i]
            market_window = market_returns.iloc[i-window:i]
            
            beta, _, _ = FactorModels.estimate_beta_ols(
                stock_window, market_window
            )
            betas.append(beta)
            dates.append(stock_returns.index[i])
        
        return pd.Series(betas, index=dates)
    
    @staticmethod
    def bayesian_beta(
        stock_returns: pd.Series,
        market_returns: pd.Series,
        prior_mean: float = 1.0,
        prior_std: float = 0.3
    ) -> float:
        """
        Estimate beta using Bayesian shrinkage.
        
        Shrinks OLS estimate toward prior mean.
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            prior_mean: Prior belief of beta (default 1.0)
            prior_std: Prior uncertainty
            
        Returns:
            Shrunk beta estimate
        """
        # OLS estimate
        ols_beta, _, _ = FactorModels.estimate_beta_ols(
            stock_returns, market_returns
        )
        
        # Posterior = weighted average of prior and OLS
        # Weights based on precision (inverse variance)
        n = len(stock_returns)
        ols_std = np.sqrt(np.var(stock_returns) / np.var(market_returns) / n)
        
        precision_prior = 1 / (prior_std ** 2)
        precision_ols = 1 / (ols_std ** 2)
        
        posterior_beta = (
            (precision_prior * prior_mean + precision_ols * ols_beta) /
            (precision_prior + precision_ols)
        )
        
        return posterior_beta
    
    @staticmethod
    def jensen_alpha(
        stock_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> Tuple[float, float, float]:
        """
        Calculate Jensen's alpha.
        
        Args:
            stock_returns: Series of stock returns
            market_returns: Series of market returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Tuple of (alpha_annual, beta, t_statistic)
        """
        daily_rf = risk_free_rate / periods_per_year
        
        excess_stock = stock_returns - daily_rf
        excess_market = market_returns - daily_rf
        
        # Regression: excess_stock = alpha + beta * excess_market
        slope, intercept, r_value, p_value, std_err = linregress(
            excess_market, excess_stock
        )
        
        alpha_daily = intercept
        alpha_annual = alpha_daily * periods_per_year
        
        # t-statistic
        t_stat = alpha_daily / (std_err / np.sqrt(len(stock_returns)))
        
        return alpha_annual, slope, t_stat
    
    @staticmethod
    def fama_french_regression(
        stock_returns: pd.Series,
        market_excess: pd.Series,
        smb: pd.Series,
        hml: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> dict:
        """
        Fama-French 3-factor regression.
        
        Args:
            stock_returns: Series of stock returns
            market_excess: Series of excess market returns
            smb: Series of SMB factor
            hml: Series of HML factor
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Dictionary with factor loadings and statistics
        """
        daily_rf = risk_free_rate / periods_per_year
        excess_stock = stock_returns - daily_rf
        
        # Align all series
        data = pd.DataFrame({
            'stock': excess_stock,
            'market': market_excess,
            'smb': smb,
            'hml': hml
        }).dropna()
        
        # Regression
        X = data[['market', 'smb', 'hml']].values
        y = data['stock'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha_daily = model.intercept_
        beta_m, beta_smb, beta_hml = model.coef_
        
        # Compute R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'alpha_annual': alpha_daily * periods_per_year,
            'beta_market': beta_m,
            'beta_smb': beta_smb,
            'beta_hml': beta_hml,
            'r_squared': r_squared,
            'num_obs': len(data)
        }
    
    @staticmethod
    def carhart_regression(
        stock_returns: pd.Series,
        market_excess: pd.Series,
        smb: pd.Series,
        hml: pd.Series,
        momentum: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> dict:
        """
        Carhart 4-factor regression.
        
        Args:
            stock_returns: Series of stock returns
            market_excess: Series of excess market returns
            smb: Series of SMB factor
            hml: Series of HML factor
            momentum: Series of momentum factor
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Dictionary with factor loadings and statistics
        """
        daily_rf = risk_free_rate / periods_per_year
        excess_stock = stock_returns - daily_rf
        
        # Align all series
        data = pd.DataFrame({
            'stock': excess_stock,
            'market': market_excess,
            'smb': smb,
            'hml': hml,
            'momentum': momentum
        }).dropna()
        
        # Regression
        X = data[['market', 'smb', 'hml', 'momentum']].values
        y = data['stock'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        alpha_daily = model.intercept_
        beta_m, beta_smb, beta_hml, beta_mom = model.coef_
        
        # Compute R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'alpha_annual': alpha_daily * periods_per_year,
            'beta_market': beta_m,
            'beta_smb': beta_smb,
            'beta_hml': beta_hml,
            'beta_momentum': beta_mom,
            'r_squared': r_squared,
            'num_obs': len(data)
        }

# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2021-01-01', periods=756, freq='B')
    stock_ret = pd.Series(np.random.normal(0.0004, 0.012, 756), index=dates)
    market_ret = pd.Series(np.random.normal(0.0003, 0.010, 756), index=dates)
    
    fm = FactorModels()
    
    # Estimate beta
    beta, alpha, r2 = fm.estimate_beta_ols(stock_ret, market_ret)
    print(f"OLS Beta: {beta:.4f}, Alpha: {alpha:.6f}, R²: {r2:.4f}")
    
    # Rolling beta
    rolling_b = fm.rolling_beta(stock_ret, market_ret)
    print(f"Rolling Beta (mean): {rolling_b.mean():.4f}, (std): {rolling_b.std():.4f}")
    
    # Bayesian beta
    bayes_beta = fm.bayesian_beta(stock_ret, market_ret)
    print(f"Bayesian Beta: {bayes_beta:.4f}")
    
    # Jensen's alpha
    alpha_annual, beta, t_stat = fm.jensen_alpha(stock_ret, market_ret)
    print(f"Jensen's Alpha (annual): {alpha_annual:.6f}, t-stat: {t_stat:.4f}")
```

### [VISUALIZATION] Factor Exposure

```
Factor Loadings: Carhart 4-Factor Model
───────────────────────────────────────

           -0.5  0.0  0.5  1.0  1.5  2.0
Market β:        │    ├─────────────●│      1.15
SMB β:     ●─────┤                    │     -0.25
HML β:           │        ●────────┤       0.45
Momentum β:      ├────────●            │    -0.05

R² = 0.68 (68% of returns explained by factors)
```

### [WARNING] Alpha Pitfalls

1. **Data mining bias**: If you search for alpha long enough, you'll find it by chance
2. **Survivor bias**: Dead funds aren't in your dataset
3. **Look-ahead bias**: Using information not available at decision time
4. **Transaction costs**: Alpha often disappears after realistic costs
5. **Estimation error**: Alpha estimates have high variance; confidence intervals matter

### Exercises

1. Estimate beta for 5 NSE stocks. Compare OLS, rolling, and Bayesian betas. Which is most stable?
2. Construct SMB and HML factors using NSE constituents. Run Fama-French regression.
3. Which Carhart factor (market, SMB, HML, momentum) is most significant for your portfolio?

---

## Chapter Summary

This chapter equipped you to answer the fundamental question: **Does my strategy actually work?**

Key takeaways:

1. **Returns are not created equal**: Log returns are additive and mathematically superior for analysis, but simple returns are intuitive for reporting.

2. **Risk is multifaceted**: Volatility captures total variability, but downside risk (semi-variance), tail risk (VaR/CVaR), and drawdown capture different investor fears.

3. **One metric never tells the full story**: Sharpe ratio, Sortino, Calmar, and Information Ratio each emphasize different aspects of strategy quality.

4. **Alpha is the holy grail**: Excess return beyond what CAPM predicts is genuine outperformance. But it's rare, hard to estimate, and easy to fake.

5. **Factor models explain returns**: CAPM, Fama-French, and Carhart models decompose returns into systematic (priced) and idiosyncratic (unpredictable) components.

**In production**: Implement ALL metrics. Track Sharpe, Sortino, maximum drawdown, win rate, profit factor, alpha, and rolling betas. Monitor confidence intervals around metrics like Sharpe ratio to ensure you have statistical significance.

---

## Chapter Project: Complete Performance Analysis Pipeline

Build a production-grade performance analysis system for your NSE trading strategy.

### Part 1: Data Preparation

```python
import yfinance as yf
import pandas as pd

# Download NSE data (you'll need to adapt for your broker's API)
# For this example, we'll use a sample Zerodha-style dataset

class NSEDataLoader:
    """Load and preprocess NSE data."""
    
    @staticmethod
    def load_from_zerodha_csv(filepath: str) -> pd.DataFrame:
        """
        Load OHLC data from Zerodha CSV export.
        
        Expected columns: Date, Open, High, Low, Close, Volume
        """
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
        return df.sort_index()
    
    @staticmethod
    def calculate_benchmark(
        symbols: list,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """
        Calculate Nifty 50 benchmark returns.
        
        In production, use actual Nifty 50 index data from NSE.
        """
        # This is a placeholder; use actual NSE API
        nifty = yf.download('^NSEI', start=start_date, end=end_date)['Adj Close']
        return nifty.pct_change()
```

### Part 2: Strategy Performance Report

```python
class PerformanceAnalyzer:
    """Comprehensive performance analysis."""
    
    def __init__(self, strategy_returns: pd.Series, benchmark_returns: pd.Series):
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.trades = None
    
    def generate_full_report(self) -> dict:
        """Generate complete performance analysis."""
        
        report = {
            'returns': self._analyze_returns(),
            'risk': self._analyze_risk(),
            'performance': self._analyze_performance(),
            'factors': self._analyze_factors(),
            'diagnostics': self._diagnostic_checks()
        }
        
        return report
    
    def _analyze_returns(self) -> dict:
        """Return statistics."""
        rc = ReturnsCalculator()
        return {
            'total_return': (1 + self.strategy_returns).prod() - 1,
            'annual_return': self.strategy_returns.mean() * 252,
            'monthly_returns': self.strategy_returns.resample('M').sum(),
        }
    
    def _analyze_risk(self) -> dict:
        """Risk statistics."""
        rm = RiskMeasures()
        return {
            'volatility': rm.volatility(self.strategy_returns),
            'semi_variance': rm.semi_variance(self.strategy_returns),
            'var_95': rm.var_historical(self.strategy_returns, 0.95)[0],
            'cvar_95': rm.cvar(self.strategy_returns, 0.95)[0],
            'max_drawdown': rm.maximum_drawdown(self.strategy_returns)[0],
        }
    
    def _analyze_performance(self) -> dict:
        """Performance metrics."""
        pm = PerformanceMetrics()
        return {
            'sharpe': pm.sharpe_ratio(self.strategy_returns),
            'sortino': pm.sortino_ratio(self.strategy_returns),
            'calmar': pm.calmar_ratio(self.strategy_returns),
            'information_ratio': pm.information_ratio(
                self.strategy_returns, self.benchmark_returns
            ),
        }
    
    def _analyze_factors(self) -> dict:
        """Factor analysis."""
        fm = FactorModels()
        
        excess_market = self.benchmark_returns - 0.04 / 252
        daily_rf = 0.04 / 252
        excess_strategy = self.strategy_returns - daily_rf
        
        beta, alpha, r2 = fm.estimate_beta_ols(
            excess_strategy, excess_market
        )
        
        return {
            'beta': beta,
            'alpha_annual': alpha * 252,
            'r_squared': r2,
        }
    
    def _diagnostic_checks(self) -> dict:
        """Statistical diagnostics."""
        from scipy import stats
        
        return {
            'skewness': stats.skew(self.strategy_returns),
            'kurtosis': stats.kurtosis(self.strategy_returns),
            'normality_test': stats.normaltest(self.strategy_returns)[1],  # p-value
            'autocorrelation': self.strategy_returns.autocorr(1),
        }

# Usage
if __name__ == "__main__":
    # Load your strategy returns
    strategy_returns = pd.Series(np.random.normal(0.0007, 0.012, 252))
    benchmark_returns = pd.Series(np.random.normal(0.0005, 0.010, 252))
    
    analyzer = PerformanceAnalyzer(strategy_returns, benchmark_returns)
    report = analyzer.generate_full_report()
    
    print("=" * 60)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nRETURNS:")
    print(f"  Total Return: {report['returns']['total_return']:.2%}")
    print(f"  Annual Return: {report['returns']['annual_return']:.2%}")
    
    print(f"\nRISK:")
    print(f"  Volatility: {report['risk']['volatility']:.2%}")
    print(f"  Max Drawdown: {report['risk']['max_drawdown']:.2%}")
    print(f"  VaR (95%): {report['risk']['var_95']:.2%}")
    print(f"  CVaR (95%): {report['risk']['cvar_95']:.2%}")
    
    print(f"\nPERFORMANCE:")
    print(f"  Sharpe Ratio: {report['performance']['sharpe']:.4f}")
    print(f"  Sortino Ratio: {report['performance']['sortino']:.4f}")
    print(f"  Calmar Ratio: {report['performance']['calmar']:.4f}")
    print(f"  Information Ratio: {report['performance']['information_ratio']:.4f}")
    
    print(f"\nFACTOR ANALYSIS:")
    print(f"  Beta: {report['factors']['beta']:.4f}")
    print(f"  Jensen's Alpha (annual): {report['factors']['alpha_annual']:.4f}")
    print(f"  R-squared: {report['factors']['r_squared']:.4f}")
    
    print(f"\nDIAGNOSTICS:")
    print(f"  Skewness: {report['diagnostics']['skewness']:.4f}")
    print(f"  Kurtosis: {report['diagnostics']['kurtosis']:.4f}")
    print(f"  Normality (p-value): {report['diagnostics']['normality_test']:.4f}")
```

### Part 3: Visualizations

Create plots of:
1. Cumulative returns (strategy vs benchmark)
2. Underwater plot (drawdowns)
3. Rolling Sharpe ratio
4. Factor exposures
5. Return distribution vs normal

---

## Additional Resources

- Sharpe, W. F. (1994). "The Sharpe Ratio." Journal of Portfolio Management.
- Fama, E. F., & French, K. R. (1993). "Common Risk Factors in Returns on Stocks and Bonds." Journal of Financial Economics.
- Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance." Journal of Finance.
- Zerodha Kite API Documentation: https://kite.trade/
- NSE Data Files: https://www1.nseindia.com/

---

## Answers to Selected Exercises

### Module 3.1, Exercise 1

Log returns are perfectly additive by construction:
```
If r_t = ln(P_t / P_{t-1}) and r_{t-1} = ln(P_{t-1} / P_{t-2}), then:
r_t + r_{t-1} = ln(P_t / P_{t-1}) + ln(P_{t-1} / P_{t-2})
              = ln(P_t / P_{t-2})  ✓

Simple returns multiply:
(1 + R_t)(1 + R_{t-1}) = (P_t / P_{t-1})(P_{t-1} / P_{t-2})
                       = P_t / P_{t-2}
So total simple return = P_t / P_{t-2} - 1  ✓
```

### Module 3.2, Exercise 2

Parametric VaR assumes normality but NSE returns have fat tails. Empirically:
- Parametric VaR underestimates true risk (too optimistic)
- Historical VaR is conservative but requires more data
- Monte Carlo VaR is most flexible but computationally expensive
- Recommendation: Use historical or CVaR for risk management

### Module 3.4, Exercise 1

Rolling beta captures regime changes:
```python
rolling_betas = fm.rolling_beta(nse_stock_returns, nifty_returns, window=252)
rolling_betas.plot()
# You'll see beta shifts around major market events (March 2020, etc.)
```

---

**End of Chapter 3**


# Chapter 22: Risk Management

## Introduction

You've built a trading system that generates alpha through machine learning models, but a single catastrophic trade can wipe out months of gains. Risk management is not optional—it's the foundation that separates profitable quants from bankrupt traders.

Unlike most finance texts written for an audience of traders, this chapter assumes you have **zero finance intuition** but strong engineering and ML expertise. We'll translate risk concepts into systems you'll recognize: circuit breakers, health checks, monitoring systems, and failsafes.

The three modules build a production-grade risk management framework:
1. **Position-Level Risk**: How much money can go wrong on a single trade?
2. **Portfolio-Level Risk**: How much can the entire portfolio decline before we shut down?
3. **Operational Risk & Kill Switches**: What happens when data feeds fail, models break, or servers crash?

By the end of this chapter, you'll implement a system that:
- Automatically closes trades that move against you
- Monitors correlations to prevent concentration risk
- Calculates daily losses using statistical confidence intervals
- Detects anomalies in model predictions
- Implements automatic kill switches for catastrophic failures
- Survives server crashes and data feed outages

---

## Module 22.1: Position-Level Risk Management

### Core Concept: The Stop-Loss

A stop-loss is a circuit breaker on a single position. In ML terms: if your model's confidence metric falls below a threshold, exit.

Your NSE trading system opens positions based on ML predictions. But ML models are wrong. A stop-loss defines how much capital you're willing to lose on that wrongness before cutting your losses.

### 22.1.1 Stop-Loss Mechanics

**Fixed Stop-Loss** (Simplest):
```
Entry price: ₹100
Stop-loss: ₹95 (5% loss allowed)
If price falls to ₹95, automatically sell
```

**Trailing Stop-Loss** (Protective):
```
Entry: ₹100
Trail: 3%
Current price: ₹105
Trailing stop sits at: ₹105 × 0.97 = ₹101.85
If price drops to ₹101.85, sell (protecting ₹3.15 profit)
Current price: ₹110
Trailing stop moves to: ₹110 × 0.97 = ₹106.70
Only sells if price drops from peak
```

**Volatility-Adjusted Stop-Loss** (Risk-Sensitive):

This is where your ML engineering intuition applies. Volatile stocks need wider stops because normal fluctuations are larger.

$$\text{Stop-Loss Distance} = \lambda \times \sigma \times \text{Entry Price}$$

Where:
- $\lambda$ = 2.0 (how many standard deviations before stopping)
- $\sigma$ = daily volatility (annualized / $\sqrt{252}$)
- Entry Price = your entry point

**Example**:
- Stock entry: ₹100
- 30-day realized volatility: 25% annualized
- Daily volatility: 25% / √252 = 1.58%
- Stop distance: 2.0 × 0.0158 × ₹100 = ₹3.16
- Stop-loss price: ₹100 - ₹3.16 = ₹96.84

This adapts to market regime. In a calm period (5% volatility), stops tighten. In turbulent periods (40% volatility), stops widen.

### 22.1.2 Position Sizing: How Much Capital Per Trade?

You've written a model that's 55% accurate. Should you bet 10% of portfolio per trade? 1%? 0.1%?

**The Kelly Criterion** (Theoretical Optimum):

From information theory, the optimal bet size is:

$$f^* = \frac{p \times b - q}{b}$$

Where:
- $p$ = probability of win (0.55)
- $q$ = probability of loss (0.45)
- $b$ = ratio of win size to loss size (1:1 for equal-sized trades)

$$f^* = \frac{0.55 \times 1 - 0.45}{1} = 0.10$$

Kelly says bet **10% of capital per trade**. This maximizes long-run log wealth growth.

**Problem**: Kelly is aggressive. A string of losses hits hard. In practice:

$$f_{practice} = \frac{f^*}{2} \text{ to } \frac{f^*}{4}$$

Use **fractional Kelly**: 25-50% of Kelly.

**Volatility Targeting** (Even Better):

Most ML engineers find the math concept confusing, but the engineering goal is intuitive:

> "Adjust position size so volatility contribution is constant across all trades"

If you want each position to contribute 2% daily volatility to portfolio:

$$\text{Position Size} = \frac{\text{Target Volatility}}{\text{Stock Volatility}} \times \text{Base Capital}$$

**Example**:
- Portfolio capital: ₹1,000,000
- Target daily volatility: 1% (portfolio should swing ₹10,000 daily)
- Stock daily volatility: 2%
- Position size: (0.01 / 0.02) × ₹1,000,000 = ₹500,000

A more volatile stock gets a smaller position. A stable stock gets a larger position.

### 22.1.3 Position Concentration Risk

You open 5 positions:
- RELIANCE: ₹300,000
- TCS: ₹250,000
- INFY: ₹200,000
- WIPRO: ₹150,000
- HCLTECH: ₹100,000

Total: ₹1,000,000 (100% deployed)

**Sector concentration**: All IT/Finance. If a sector crashes, you lose everything.

**Correlation concentration**: When IT sells off, all 5 fall together. Your portfolio risk is higher than individual position risks suggest.

**Rule**: Maximum single position = 20-30% of capital.

**Better rule**: Use correlation-aware sizing.

### 22.1.4 Correlation-Aware Position Sizing

This is crucial and often missed.

**The Problem**:
- INFY and TCS have 0.85 correlation (both IT)
- RELIANCE and SBI have 0.72 correlation (both financial)
- If you size equally, you're overexposed to sector shocks

**The Solution**:

Allocate capital to uncorrelated factors, not individual stocks.

For NSE stocks, factors include:
- IT sector
- Financials sector
- Energy/Materials
- Consumer
- Healthcare

Calculate correlation matrix:

$$\rho = \begin{bmatrix}
1.0 & 0.85 & 0.72 & 0.15 & 0.10 \\
0.85 & 1.0 & 0.78 & 0.18 & 0.12 \\
0.72 & 0.78 & 1.0 & 0.22 & 0.14 \\
0.15 & 0.18 & 0.22 & 1.0 & 0.65 \\
0.10 & 0.12 & 0.14 & 0.65 & 1.0
\end{bmatrix}$$

Use principal component analysis (PCA) to find uncorrelated directions, then allocate:

```python
import numpy as np
from sklearn.decomposition import PCA

# Correlation matrix of your candidate stocks
correlation_matrix = np.array([...])

# Find principal components
pca = PCA(n_components=5)
pca.fit(correlation_matrix)

# Components tell you which stock combinations are uncorrelated
# Allocate 20% per component instead of 20% per stock
```

### 22.1.5 Production Code: Position-Level Risk Monitor

```python
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd

@dataclass
class Position:
    """Single position tracking"""
    ticker: str
    entry_price: float
    entry_time: datetime
    quantity: int
    position_type: str  # 'long' or 'short'
    
    @property
    def position_value(self, current_price: float) -> float:
        """Market value of position"""
        return self.quantity * current_price
    
    @property
    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized profit/loss"""
        if self.position_type == 'long':
            return self.quantity * (current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - current_price)
    
    @property
    def pnl_percentage(self, current_price: float) -> float:
        """Percentage P&L"""
        return self.unrealized_pnl(current_price) / self.position_value(self.entry_price)


class PositionLevelRiskManager:
    """
    Manages stop-loss, position sizing, and position-level risk.
    
    Attributes:
        portfolio_capital: Total capital available (₹)
        kelly_fraction: Fraction of Kelly to use (0.25-0.5)
        max_position_pct: Maximum position size as % of capital
        volatility_window: Days for volatility calculation
    """
    
    def __init__(
        self,
        portfolio_capital: float,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.20,
        volatility_window: int = 30
    ):
        self.portfolio_capital = portfolio_capital
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.volatility_window = volatility_window
        self.positions: Dict[str, Position] = {}
    
    def calculate_kelly_position_size(
        self,
        win_probability: float,
        win_loss_ratio: float = 1.0
    ) -> float:
        """
        Calculate Kelly criterion position sizing.
        
        Args:
            win_probability: P(trade wins), 0 < p < 1
            win_loss_ratio: Average win size / Average loss size
        
        Returns:
            Position size as fraction of capital
        
        Formula: f* = (p*b - q) / b
        where p = win prob, q = 1-p, b = win/loss ratio
        """
        q = 1 - win_probability
        kelly_full = (win_probability * win_loss_ratio - q) / win_loss_ratio
        kelly_fractional = kelly_full * self.kelly_fraction
        
        # Safety: cap at max position
        return min(kelly_fractional, self.max_position_pct)
    
    def calculate_volatility_adjusted_size(
        self,
        stock_daily_volatility: float,
        target_portfolio_volatility: float = 0.01
    ) -> float:
        """
        Size position to target portfolio volatility contribution.
        
        Args:
            stock_daily_volatility: Daily volatility of target stock
            target_portfolio_volatility: Desired daily vol contribution (0.01 = 1%)
        
        Returns:
            Position size as fraction of capital
        
        Formula: size = target_vol / stock_vol
        """
        if stock_daily_volatility == 0:
            return 0
        
        position_size = target_portfolio_volatility / stock_daily_volatility
        return min(position_size, self.max_position_pct)
    
    def fixed_stop_loss(
        self,
        entry_price: float,
        stop_loss_pct: float = 0.05
    ) -> float:
        """
        Calculate fixed stop-loss level.
        
        Args:
            entry_price: Entry price (₹)
            stop_loss_pct: Stop distance as % (0.05 = 5%)
        
        Returns:
            Stop-loss price (₹)
        """
        return entry_price * (1 - stop_loss_pct)
    
    def trailing_stop_loss(
        self,
        current_price: float,
        peak_price: float,
        trail_pct: float = 0.03
    ) -> float:
        """
        Calculate trailing stop-loss.
        
        Args:
            current_price: Current stock price (₹)
            peak_price: Highest price since entry (₹)
            trail_pct: Trail distance as % (0.03 = 3%)
        
        Returns:
            Current stop-loss level (₹)
        
        Note: Stop moves up with price but never down
        """
        return peak_price * (1 - trail_pct)
    
    def volatility_adjusted_stop(
        self,
        entry_price: float,
        daily_volatility: float,
        lambda_param: float = 2.0
    ) -> float:
        """
        Calculate volatility-adjusted stop-loss.
        
        Args:
            entry_price: Entry price (₹)
            daily_volatility: Daily volatility (0.02 = 2% daily)
            lambda_param: Number of std devs (2.0 = 2σ stop)
        
        Returns:
            Stop-loss price (₹)
        
        Formula: stop = entry - λ × σ × entry
        Wider stops in volatile markets, tighter in calm
        """
        stop_distance = lambda_param * daily_volatility * entry_price
        return entry_price - stop_distance
    
    def position_sizing_with_correlation(
        self,
        correlation_matrix: np.ndarray,
        base_position_size: float,
        ticker_index: int
    ) -> float:
        """
        Adjust position size for correlation with existing positions.
        
        Args:
            correlation_matrix: N×N correlation matrix
            base_position_size: Initial position size from Kelly/volatility
            ticker_index: Index of target stock in correlation matrix
        
        Returns:
            Adjusted position size accounting for correlation
        
        Logic:
            If new stock highly correlated with existing positions,
            reduce its size. If uncorrelated, keep full size.
        """
        # Average correlation with existing positions
        avg_correlation = correlation_matrix[ticker_index].mean()
        
        # Reduce size by correlation factor
        # correlation=0.2 → multiply by 0.9
        # correlation=0.8 → multiply by 0.4
        correlation_adjustment = 1.0 - (avg_correlation * 0.75)
        
        adjusted_size = base_position_size * correlation_adjustment
        return min(adjusted_size, self.max_position_pct)
    
    def should_exit_position(
        self,
        position: Position,
        current_price: float,
        stop_loss_price: float
    ) -> Dict[str, any]:
        """
        Check if position should be exited based on stop-loss.
        
        Args:
            position: Position object
            current_price: Current market price (₹)
            stop_loss_price: Stop-loss level (₹)
        
        Returns:
            {
                'should_exit': bool,
                'reason': str,
                'current_pnl': float,
                'current_pnl_pct': float
            }
        """
        unrealized_pnl = position.unrealized_pnl(current_price)
        unrealized_pnl_pct = position.pnl_percentage(current_price)
        
        should_exit = False
        reason = "No exit"
        
        # Check stop-loss
        if position.position_type == 'long':
            if current_price <= stop_loss_price:
                should_exit = True
                reason = "Stop-loss hit (long)"
        else:  # short
            if current_price >= stop_loss_price:
                should_exit = True
                reason = "Stop-loss hit (short)"
        
        return {
            'should_exit': should_exit,
            'reason': reason,
            'current_pnl': unrealized_pnl,
            'current_pnl_pct': unrealized_pnl_pct
        }
    
    def add_position(
        self,
        ticker: str,
        entry_price: float,
        quantity: int,
        position_type: str = 'long'
    ) -> None:
        """Add position to tracking"""
        self.positions[ticker] = Position(
            ticker=ticker,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            position_type=position_type
        )
    
    def get_concentration_report(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Report portfolio concentration by position.
        
        Returns:
            {
                'RELIANCE': 0.25,  # 25% of portfolio
                'TCS': 0.22,
                ...
                'max_concentration': 0.25
            }
        """
        total_value = sum(
            pos.position_value(current_prices[pos.ticker])
            for pos in self.positions.values()
        )
        
        if total_value == 0:
            return {}
        
        concentrations = {}
        for ticker, pos in self.positions.items():
            conc = pos.position_value(current_prices[ticker]) / total_value
            concentrations[ticker] = conc
        
        concentrations['max_concentration'] = max(concentrations.values())
        concentrations['min_concentration'] = min(concentrations.values())
        
        return concentrations


# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_mgr = PositionLevelRiskManager(
        portfolio_capital=1_000_000,
        kelly_fraction=0.25,
        max_position_pct=0.20
    )
    
    # Calculate position sizes
    # Model says: 60% win rate, 1:1 win/loss ratio
    kelly_size = risk_mgr.calculate_kelly_position_size(
        win_probability=0.60,
        win_loss_ratio=1.0
    )
    print(f"Kelly-based position: {kelly_size*100:.2f}% = ₹{kelly_size*1_000_000:,.0f}")
    
    # Volatility-adjusted sizing
    # Stock has 2% daily volatility
    vol_size = risk_mgr.calculate_volatility_adjusted_size(
        stock_daily_volatility=0.02,
        target_portfolio_volatility=0.01
    )
    print(f"Volatility-adjusted position: {vol_size*100:.2f}% = ₹{vol_size*1_000_000:,.0f}")
    
    # Calculate stop-losses
    entry = 100
    
    fixed_stop = risk_mgr.fixed_stop_loss(entry, stop_loss_pct=0.05)
    print(f"Fixed stop (5%): ₹{fixed_stop:.2f}")
    
    vol_adjusted_stop = risk_mgr.volatility_adjusted_stop(
        entry, daily_volatility=0.015, lambda_param=2.0
    )
    print(f"Volatility-adjusted stop: ₹{vol_adjusted_stop:.2f}")
    
    # Add position
    risk_mgr.add_position("RELIANCE", entry_price=2500, quantity=400, position_type='long')
    
    # Check exit conditions
    current_prices = {"RELIANCE": 2475}
    exit_check = risk_mgr.should_exit_position(
        risk_mgr.positions["RELIANCE"],
        current_price=2475,
        stop_loss_price=2375
    )
    print(f"Exit check: {exit_check}")
```

---

## Module 22.2: Portfolio-Level Risk Management

Position-level risk prevents single catastrophic trades. Portfolio-level risk prevents catastrophic days.

You've implemented stop-losses on individual positions. But what if:
- 10 positions each hit -3%? Portfolio is -30%
- A flash crash hits all tech stocks simultaneously?
- Your model's predictions are systematically wrong across the entire portfolio?

Portfolio-level risk management answers: **How much can the entire portfolio lose before we stop trading?**

### 22.2.1 Value at Risk (VaR) and Conditional Value at Risk (CVaR)

**Value at Risk (VaR)**: At a 95% confidence level, what's the worst daily loss?

Intuitive interpretation:
> "In 1 out of 20 days, we'll lose more than our VaR. In 19 out of 20 days, we'll lose less."

**Mathematical Definition**:

$$\text{VaR}_{95\%} = -F^{-1}(0.05)$$

Where $F^{-1}$ is the inverse cumulative distribution of returns.

**Example**:
- Portfolio value: ₹1,000,000
- Daily returns historically: mean = 0.1%, std = 1.5%
- Assume normal distribution
- VaR(95%): ₹1,000,000 × (-1.645 × 0.015) = -₹24,675

Interpretation: In 95% of days, we lose less than ₹24,675. In 5% of days (once a month), we lose more.

**Conditional Value at Risk (CVaR)**: What's the average loss on the bad days (the 5% we exceed VaR)?

$$\text{CVaR}_{95\%} = E[Loss | Loss > \text{VaR}_{95\%}]$$

For normal distribution:
$$\text{CVaR}_{95\%} = \text{VaR}_{95\%} - \frac{\sigma \phi(Z_{\alpha})}{1-\alpha}$$

Where:
- $\sigma$ = volatility
- $\phi$ = standard normal PDF
- $Z_{\alpha}$ = critical value (1.645 for 95%)
- $\alpha$ = significance level (0.05)

**Why CVaR matters**: VaR is the threshold. CVaR is the disaster size. Use VaR for daily limits. Use CVaR for capital requirements.

### 22.2.2 Factor Exposure Monitoring

Your portfolio is long:
- 5 IT stocks (combined ₹500k)
- 3 Finance stocks (combined ₹300k)
- 2 Energy stocks (combined ₹200k)

Sector allocation looks balanced. But what if a sector crashes 15%?

**Factor Exposure** = contribution of each sector/factor to portfolio risk.

Calculate sector beta:

$$\text{Sector Exposure} = \sum_i (\text{Position}_i \times \text{Stock-Sector Correlation}_i \times \text{Stock Beta}_i)$$

**Example Calculation**:

| Stock | Position (₹) | Sector | Sector Beta | Correlation to Sector | Exposure |
|-------|---------|--------|------------|-----------|----------|
| INFY | 150,000 | IT | 1.1 | 0.88 | 145,200 |
| TCS | 150,000 | IT | 1.05 | 0.85 | 134,250 |
| RELIANCE | 150,000 | Energy | 0.95 | 0.82 | 116,700 |

**Portfolio IT exposure**: ₹145,200 + ₹134,250 = ₹279,450 (27.9% of portfolio)

**Rule**: No single sector > 35% of portfolio.

### 22.2.3 Gross and Net Exposure Limits

**Gross Exposure** = Sum of absolute position values

```
Long RELIANCE: +₹400,000
Long TCS: +₹300,000
Short SBIN: -₹200,000
Gross = 400k + 300k + 200k = ₹900,000
```

**Net Exposure** = Long minus short

```
Net = 400k + 300k - 200k = ₹500,000
```

**Rules**:
- Gross exposure ≤ 150% of capital (using 50% leverage)
- Net exposure ≤ 100% of capital (fully invested)

These prevent over-leverage.

### 22.2.4 Drawdown-Based Risk Controls

This is the most psychologically important metric.

**Drawdown** = Peak-to-trough decline

```
Peak: ₹1,000,000
Current: ₹920,000
Drawdown: -8%
```

**Rule**: If drawdown exceeds threshold, reduce position sizes.

**Example System**:
- Drawdown 0-3%: Trade normally
- Drawdown 3-5%: Reduce position sizes by 25%
- Drawdown 5-7%: Reduce position sizes by 50%
- Drawdown > 7%: Flatten all positions

### 22.2.5 Scenario Analysis

Backtest your portfolio against historical shocks:

**2008 Financial Crisis Scenario**:
- Equities down 40%
- Volatility up 3x
- Correlations → 1.0 (everything falls together)
- Your portfolio loss?

**COVID-19 Scenario**:
- Market down 30% in 3 weeks
- Your portfolio exposure to airlines, hospitality, finance?

**Rate Shock Scenario**:
- 10-year yield up 100 bps
- How much does your portfolio fall?

Run these scenarios quarterly. Know your maximum pain.

### 22.2.6 Production Code: Portfolio-Level Risk Monitor

```python
from typing import List, Tuple
import scipy.stats as stats

class PortfolioLevelRiskManager:
    """
    Monitors portfolio-level risk: VaR, CVaR, factor exposure, drawdowns.
    
    Attributes:
        positions: List of open positions
        daily_returns: Historical daily returns DataFrame
        risk_limits: Dictionary of risk limits
    """
    
    def __init__(
        self,
        portfolio_capital: float,
        max_var_pct: float = 0.03,  # 3% VaR limit
        max_drawdown_pct: float = 0.10,  # 10% drawdown limit
        max_sector_exposure_pct: float = 0.35  # 35% per sector
    ):
        self.portfolio_capital = portfolio_capital
        self.max_var_pct = max_var_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        
        self.portfolio_values: List[float] = [portfolio_capital]  # Daily close values
        self.positions: Dict[str, Position] = {}
    
    def calculate_var_normal(
        self,
        daily_returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate VaR assuming normal distribution.
        
        Args:
            daily_returns: Array of historical daily returns
            confidence_level: Confidence level (0.95 = 95%)
        
        Returns:
            (var_dollars, var_pct)
        
        Formula:
            VaR = μ - σ × Z_α
            where Z_α is critical value for confidence level
        """
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        alpha = 1 - confidence_level
        
        # Critical value (e.g., 1.645 for 95% confidence)
        z_critical = stats.norm.ppf(alpha)
        
        var_return = mean_return - std_return * z_critical
        var_dollars = -var_return * self.portfolio_capital
        
        return var_dollars, -var_return
    
    def calculate_cvar_normal(
        self,
        daily_returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate CVaR (expected loss beyond VaR).
        
        Args:
            daily_returns: Array of historical daily returns
            confidence_level: Confidence level
        
        Returns:
            (cvar_dollars, cvar_pct)
        
        For normal distribution:
            CVaR = μ - σ × φ(Z_α) / (1 - α)
            where φ is standard normal PDF
        """
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        alpha = 1 - confidence_level
        
        z_critical = stats.norm.ppf(alpha)
        
        # Standard normal PDF evaluated at critical value
        pdf_value = stats.norm.pdf(z_critical)
        
        cvar_return = mean_return - std_return * (pdf_value / alpha)
        cvar_dollars = -cvar_return * self.portfolio_capital
        
        return cvar_dollars, -cvar_return
    
    def calculate_var_historical(
        self,
        daily_returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate VaR using historical percentile (no distribution assumption).
        
        Args:
            daily_returns: Array of historical daily returns
            confidence_level: Confidence level
        
        Returns:
            (var_dollars, var_pct)
        
        Method:
            Sort returns, find (1-confidence_level)th percentile
            More robust to fat tails than normal VaR
        """
        alpha = 1 - confidence_level
        percentile = np.percentile(daily_returns, alpha * 100)
        
        var_dollars = -percentile * self.portfolio_capital
        
        return var_dollars, -percentile
    
    def calculate_maximum_drawdown(
        self,
        portfolio_values: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate maximum drawdown from historical portfolio values.
        
        Args:
            portfolio_values: List of daily portfolio values
        
        Returns:
            (max_drawdown_pct, max_drawdown_dollars)
        
        Formula:
            DD = (Current - Peak) / Peak
            Max DD = min(DD over all periods)
        """
        values_array = np.array(portfolio_values)
        running_max = np.maximum.accumulate(values_array)
        drawdowns = (values_array - running_max) / running_max
        
        max_drawdown = np.min(drawdowns)
        max_drawdown_dollars = max_drawdown * self.portfolio_capital
        
        return max_drawdown, max_drawdown_dollars
    
    def calculate_factor_exposure(
        self,
        positions: Dict[str, Tuple[float, str]],  # {ticker: (value, sector)}
        stock_correlations: Dict[str, float]  # {ticker: correlation_to_sector}
    ) -> Dict[str, float]:
        """
        Calculate sector/factor exposure.
        
        Args:
            positions: {ticker: (position_value, sector_name)}
            stock_correlations: {ticker: correlation to sector}
        
        Returns:
            {sector_name: exposure_value}
        """
        exposures = {}
        total_value = sum(v for v, _ in positions.values())
        
        for ticker, (position_value, sector) in positions.items():
            correlation = stock_correlations.get(ticker, 0.7)
            effective_exposure = position_value * correlation
            
            if sector not in exposures:
                exposures[sector] = 0
            exposures[sector] += effective_exposure
        
        # Convert to percentages
        exposures_pct = {
            sector: exposure / total_value
            for sector, exposure in exposures.items()
        }
        
        return exposures_pct
    
    def check_concentration_limits(
        self,
        factor_exposures: Dict[str, float],
        max_exposure_pct: float
    ) -> Dict[str, any]:
        """
        Check if any factor exceeds exposure limit.
        
        Returns:
            {
                'within_limits': bool,
                'violations': [{'factor': name, 'exposure': 0.38, 'limit': 0.35}],
                'most_exposed': 'IT'
            }
        """
        violations = []
        
        for factor, exposure in factor_exposures.items():
            if exposure > max_exposure_pct:
                violations.append({
                    'factor': factor,
                    'exposure': exposure,
                    'limit': max_exposure_pct,
                    'excess': exposure - max_exposure_pct
                })
        
        return {
            'within_limits': len(violations) == 0,
            'violations': violations,
            'most_exposed': max(factor_exposures, key=factor_exposures.get),
            'max_exposure': max(factor_exposures.values())
        }
    
    def calculate_position_reduction(
        self,
        current_drawdown_pct: float
    ) -> float:
        """
        Calculate position size reduction based on drawdown.
        
        Args:
            current_drawdown_pct: Current drawdown as % (0.05 = 5%)
        
        Returns:
            Position reduction factor (0.5 = reduce by 50%)
        
        Rules:
            0-3%: no reduction
            3-5%: 25% reduction
            5-7%: 50% reduction
            >7%: 100% reduction (flatten)
        """
        if current_drawdown_pct <= 0.03:
            return 0.0
        elif current_drawdown_pct <= 0.05:
            return 0.25
        elif current_drawdown_pct <= 0.07:
            return 0.50
        else:
            return 1.0  # Flatten all positions
    
    def daily_risk_report(
        self,
        daily_returns: np.ndarray,
        portfolio_values: List[float],
        factor_exposures: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Generate daily portfolio risk report.
        
        Returns:
            Comprehensive risk snapshot
        """
        var_dollars, var_pct = self.calculate_var_normal(daily_returns, confidence_level=0.95)
        cvar_dollars, cvar_pct = self.calculate_cvar_normal(daily_returns, confidence_level=0.95)
        max_dd, max_dd_dollars = self.calculate_maximum_drawdown(portfolio_values)
        
        current_drawdown = (portfolio_values[-1] - max(portfolio_values)) / max(portfolio_values)
        
        concentration_check = self.check_concentration_limits(
            factor_exposures,
            self.max_sector_exposure_pct
        )
        
        position_reduction = self.calculate_position_reduction(abs(current_drawdown))
        
        return {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_values[-1],
            'var_95_pct': var_pct,
            'var_95_dollars': var_dollars,
            'cvar_95_pct': cvar_pct,
            'cvar_95_dollars': cvar_dollars,
            'max_drawdown_pct': max_dd,
            'max_drawdown_dollars': max_dd_dollars,
            'current_drawdown_pct': current_drawdown,
            'var_limit_exceeded': var_pct > self.max_var_pct,
            'drawdown_limit_exceeded': abs(current_drawdown) > self.max_drawdown_pct,
            'concentration_violations': concentration_check,
            'recommended_position_reduction': position_reduction,
            'factor_exposures': factor_exposures
        }


# Example usage
if __name__ == "__main__":
    # Simulated historical daily returns
    np.random.seed(42)
    daily_returns = np.random.normal(loc=0.0005, scale=0.015, size=252)
    
    # Simulate portfolio value progression
    portfolio_values = [1_000_000]
    for ret in daily_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # Initialize portfolio risk manager
    port_risk = PortfolioLevelRiskManager(
        portfolio_capital=1_000_000,
        max_var_pct=0.03,
        max_drawdown_pct=0.10,
        max_sector_exposure_pct=0.35
    )
    
    # Calculate risks
    var_dollars, var_pct = port_risk.calculate_var_normal(daily_returns)
    print(f"Daily VaR (95%): {var_pct*100:.2f}% = ₹{var_dollars:,.0f}")
    
    cvar_dollars, cvar_pct = port_risk.calculate_cvar_normal(daily_returns)
    print(f"Daily CVaR (95%): {cvar_pct*100:.2f}% = ₹{cvar_dollars:,.0f}")
    
    max_dd, max_dd_dollars = port_risk.calculate_maximum_drawdown(portfolio_values)
    print(f"Maximum Drawdown: {max_dd*100:.2f}% = ₹{max_dd_dollars:,.0f}")
    
    # Factor exposure check
    positions = {
        'INFY': (150_000, 'IT'),
        'TCS': (150_000, 'IT'),
        'RELIANCE': (150_000, 'Energy'),
        'SBIN': (100_000, 'Finance')
    }
    
    correlations = {
        'INFY': 0.88,
        'TCS': 0.85,
        'RELIANCE': 0.82,
        'SBIN': 0.80
    }
    
    exposures = port_risk.calculate_factor_exposure(positions, correlations)
    print(f"Factor exposures: {exposures}")
    
    # Risk report
    report = port_risk.daily_risk_report(daily_returns, portfolio_values, exposures)
    print(f"\nRisk Report: {report}")
```

---

## Module 22.3: Operational Risk and Kill Switches

The previous two modules prevent *market* risk (losing money to bad trades). This module prevents *operational* risk (losing money to broken systems).

Scenarios:
- Zerodha data feed drops. You don't know current prices, but system keeps trading.
- Model produces absurd predictions (10,000% returns) due to NaN propagation.
- Order gets partially filled (you wanted 400 shares, got 250). System thinks it has 400.
- Network outage mid-position. Server crashes. Positions still open when you restart.

These aren't rare. They're guaranteed to happen.

### 22.3.1 Data Feed Failure Detection

Your system receives price updates from Zerodha. What if they stop?

**Detection Strategy**:

```
Last price update: 09:45:00
Current time: 09:47:30
Time since update: 2 minutes 30 seconds
Threshold: 1 minute

Action: TRIGGER ALERT → Check data feed
```

**Production Implementation**:

```python
class DataFeedHealthMonitor:
    def __init__(self, max_staleness_seconds: int = 60):
        self.max_staleness = max_staleness_seconds
        self.last_update_time = {}
    
    def record_update(self, ticker: str):
        self.last_update_time[ticker] = datetime.now()
    
    def check_health(self, ticker: str) -> Dict[str, any]:
        """Check if data feed is healthy"""
        if ticker not in self.last_update_time:
            return {'healthy': False, 'reason': 'No data received'}
        
        time_since_update = (
            datetime.now() - self.last_update_time[ticker]
        ).total_seconds()
        
        if time_since_update > self.max_staleness:
            return {
                'healthy': False,
                'reason': f'Stale data ({time_since_update}s old)',
                'seconds_stale': time_since_update
            }
        
        return {'healthy': True, 'seconds_since_update': time_since_update}
```

### 22.3.2 Model Anomaly Detection

Your model outputs predictions like `[0.55, 0.48, -0.12, 0.51]` (reasonable).

Then one day: `[0.55, 0.48, 5.3, 0.51]` (garbage).

**Anomaly Signs**:
- Prediction magnitude > 3σ from historical
- NaN or infinity values
- All predictions identical (model crashed)
- Extreme rapid changes (20% shift in one batch)

**Detection Code**:

```python
class ModelAnomalyDetector:
    def __init__(self, window_size: int = 100):
        self.prediction_history = []
        self.window_size = window_size
    
    def add_predictions(self, predictions: np.ndarray):
        """Record model predictions"""
        self.prediction_history.append(predictions)
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
    
    def check_anomalies(self, new_predictions: np.ndarray) -> Dict[str, any]:
        """
        Check for anomalies in model output.
        
        Returns:
            {
                'is_anomalous': bool,
                'anomalies': [
                    {'type': 'NaN detected', 'index': 5},
                    {'type': 'Extreme magnitude', 'value': 5.3, 'std_devs': 8.2}
                ]
            }
        """
        anomalies = []
        
        # Check for NaN/Inf
        if np.any(np.isnan(new_predictions)):
            anomalies.append({'type': 'NaN detected'})
        if np.any(np.isinf(new_predictions)):
            anomalies.append({'type': 'Infinity detected'})
        
        # Check for extreme values
        if len(self.prediction_history) > 10:
            historical = np.array(self.prediction_history).flatten()
            mean = np.mean(historical)
            std = np.std(historical)
            
            if std == 0:
                std = 0.01  # Prevent division by zero
            
            z_scores = np.abs((new_predictions - mean) / std)
            if np.any(z_scores > 3):
                max_z = np.max(z_scores)
                max_idx = np.argmax(z_scores)
                anomalies.append({
                    'type': 'Extreme magnitude',
                    'value': new_predictions[max_idx],
                    'std_devs': max_z
                })
        
        return {
            'is_anomalous': len(anomalies) > 0,
            'anomalies': anomalies
        }
```

### 22.3.3 Execution Failure Handling

You send order: "Buy 400 shares of RELIANCE at ₹2500"

Zerodha responds: "Filled 250 shares at ₹2500. Remaining 150 rejected (insufficient liquidity)."

Your system **must** know:
- Only 250 got filled
- 150 are still pending (or rejected)
- Position tracking is now wrong

**Production Code**:

```python
@dataclass
class Order:
    """Trade order tracking"""
    order_id: str
    ticker: str
    quantity_requested: int
    quantity_filled: int = 0
    quantity_rejected: int = 0
    order_status: str = 'PENDING'  # PENDING, PARTIAL_FILLED, FILLED, REJECTED, CANCELLED
    fill_price: float = 0.0
    error_message: str = ""

class ExecutionRiskManager:
    """Manage order execution and partial fill scenarios"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
    
    def submit_order(self, order_id: str, ticker: str, qty: int) -> Order:
        """Submit order for execution"""
        order = Order(
            order_id=order_id,
            ticker=ticker,
            quantity_requested=qty,
            order_status='PENDING'
        )
        self.orders[order_id] = order
        return order
    
    def record_partial_fill(
        self,
        order_id: str,
        filled_qty: int,
        filled_price: float,
        remaining_qty: int
    ) -> None:
        """Handle partial fill from exchange"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        order.quantity_filled = filled_qty
        order.quantity_rejected = remaining_qty
        order.fill_price = filled_price
        order.order_status = 'PARTIAL_FILLED'
    
    def record_rejection(
        self,
        order_id: str,
        error_message: str
    ) -> None:
        """Handle order rejection"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        order.quantity_rejected = order.quantity_requested
        order.quantity_filled = 0
        order.order_status = 'REJECTED'
        order.error_message = error_message
    
    def verify_position_consistency(
        self,
        ticker: str,
        position_qty: int,
        filled_orders: List[Order]
    ) -> Dict[str, any]:
        """
        Verify that position matches sum of filled orders.
        
        Catches scenarios where position is tracked incorrectly.
        """
        total_filled = sum(
            o.quantity_filled for o in filled_orders
            if o.ticker == ticker and o.order_status in ['FILLED', 'PARTIAL_FILLED']
        )
        
        mismatch = position_qty - total_filled
        
        return {
            'consistent': mismatch == 0,
            'position_qty': position_qty,
            'total_filled': total_filled,
            'mismatch_qty': mismatch,
            'requires_reconciliation': abs(mismatch) > 0
        }
```

### 22.3.4 The Kill Switch

When should we automatically flatten all positions?

**Kill Switch Triggers**:
1. **Data Feed Dead**: No price update for > 2 minutes
2. **Model Broken**: Anomaly detected in predictions
3. **Execution Failure**: Multiple orders rejected
4. **Extreme Loss**: Portfolio down > 10% in single day
5. **Correlation Collapse**: All correlations → 1.0 (market stress)
6. **Manual Trigger**: User presses the button

When triggered: **Close all positions immediately using market orders**

### 22.3.5 Full Risk Management System with Kill Switch

```python
from enum import Enum
from typing import Callable

class KillSwitchReason(Enum):
    """Reasons for kill switch activation"""
    DATA_FEED_FAILURE = "Data feed stale"
    MODEL_ANOMALY = "Model output anomalous"
    EXECUTION_FAILURE = "Orders repeatedly rejected"
    EXTREME_LOSS = "Portfolio loss exceeds limit"
    MANUAL = "Manual trigger"
    CORRELATION_SHOCK = "Market correlation shock"


class KillSwitch:
    """
    Emergency stop for automated trading.
    
    When activated:
    - Stops opening new positions
    - Closes existing positions via market orders
    - Prevents any trading for cooldown period
    - Logs all details for post-mortem
    """
    
    def __init__(
        self,
        position_closer: Callable,  # Function to close positions
        cooldown_seconds: int = 300  # 5 minutes
    ):
        self.is_active = False
        self.activation_time = None
        self.activation_reason = None
        self.cooldown_seconds = cooldown_seconds
        self.position_closer = position_closer
        self.activation_log = []
    
    def activate(
        self,
        reason: KillSwitchReason,
        details: Dict[str, any] = None
    ) -> Dict[str, any]:
        """
        Activate kill switch.
        
        Args:
            reason: KillSwitchReason enum
            details: Additional context (e.g., portfolio loss, stale data age)
        
        Returns:
            {'status': 'ACTIVATED', 'reason': ..., 'timestamp': ...}
        """
        if self.is_active:
            return {
                'status': 'ALREADY_ACTIVE',
                'activated_at': self.activation_time,
                'reason': self.activation_reason
            }
        
        self.is_active = True
        self.activation_time = datetime.now()
        self.activation_reason = reason
        
        # Log activation
        log_entry = {
            'timestamp': self.activation_time,
            'reason': reason.value,
            'details': details or {}
        }
        self.activation_log.append(log_entry)
        
        # Close all positions
        closure_results = self.position_closer()
        
        return {
            'status': 'ACTIVATED',
            'reason': reason.value,
            'timestamp': self.activation_time,
            'positions_closed': closure_results
        }
    
    def is_on_cooldown(self) -> bool:
        """Check if kill switch is in cooldown period"""
        if not self.is_active or not self.activation_time:
            return False
        
        time_since = (datetime.now() - self.activation_time).total_seconds()
        return time_since < self.cooldown_seconds
    
    def can_trade(self) -> bool:
        """Check if system is allowed to trade"""
        return not self.is_active and not self.is_on_cooldown()
    
    def reset(self) -> None:
        """Reset kill switch (only after manual inspection)"""
        self.is_active = False
        self.activation_time = None
        self.activation_reason = None


class ComprehensiveRiskManager:
    """
    Unified risk management system combining position-level,
    portfolio-level, and operational risk.
    """
    
    def __init__(
        self,
        portfolio_capital: float,
        position_risk_mgr: PositionLevelRiskManager,
        portfolio_risk_mgr: PortfolioLevelRiskManager
    ):
        self.portfolio_capital = portfolio_capital
        self.position_risk = position_risk_mgr
        self.portfolio_risk = portfolio_risk_mgr
        
        self.data_feed_monitor = DataFeedHealthMonitor(max_staleness_seconds=120)
        self.model_anomaly_detector = ModelAnomalyDetector(window_size=100)
        self.execution_risk = ExecutionRiskManager()
        
        # Kill switch
        self.kill_switch = KillSwitch(
            position_closer=self.close_all_positions,
            cooldown_seconds=300
        )
        
        self.daily_losses = []
    
    def close_all_positions(self) -> Dict[str, int]:
        """
        Close all open positions via market orders.
        
        Returns:
            {ticker: qty_closed} for each position
        """
        closed = {}
        for ticker, pos in self.position_risk.positions.items():
            # In real system: send market order to close position
            closed[ticker] = pos.quantity
            print(f"MARKET ORDER: Close {pos.quantity} {ticker}")
        
        self.position_risk.positions.clear()
        return closed
    
    def check_all_risks(
        self,
        current_prices: Dict[str, float],
        daily_returns: np.ndarray,
        model_predictions: np.ndarray,
        factor_exposures: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Comprehensive risk check. Called every data update.
        
        Returns:
            {
                'overall_status': 'OK' | 'WARNING' | 'CRITICAL',
                'checks': {
                    'position_level': {...},
                    'portfolio_level': {...},
                    'operational': {...}
                },
                'kill_switch_triggered': bool,
                'kill_switch_reason': KillSwitchReason or None
            }
        """
        
        status = 'OK'
        kill_switch_trigger = None
        results = {'checks': {}}
        
        # ===== POSITION LEVEL CHECKS =====
        position_checks = []
        for ticker, pos in self.position_risk.positions.items():
            current_price = current_prices.get(ticker)
            if not current_price:
                continue
            
            # Calculate stop-loss (example: 2% volatility-adjusted)
            daily_vol = np.std([r for r in daily_returns if len(r) > 0])
            stop = self.position_risk.volatility_adjusted_stop(
                pos.entry_price, daily_vol, lambda_param=2.0
            )
            
            should_exit = self.position_risk.should_exit_position(
                pos, current_price, stop
            )
            
            if should_exit['should_exit']:
                position_checks.append({
                    'ticker': ticker,
                    'status': 'SHOULD_EXIT',
                    'reason': should_exit['reason'],
                    'pnl': should_exit['current_pnl']
                })
                status = 'WARNING'
        
        results['checks']['position_level'] = position_checks
        
        # ===== PORTFOLIO LEVEL CHECKS =====
        portfolio_values = [
            self.portfolio_capital + sum(
                p.unrealized_pnl(current_prices.get(p.ticker, p.entry_price))
                for p in self.position_risk.positions.values()
            )
        ]
        
        var_dollars, var_pct = self.portfolio_risk.calculate_var_normal(
            daily_returns
        )
        
        max_dd, _ = self.portfolio_risk.calculate_maximum_drawdown(portfolio_values)
        
        concentration = self.portfolio_risk.check_concentration_limits(
            factor_exposures,
            self.portfolio_risk.max_sector_exposure_pct
        )
        
        portfolio_checks = {
            'var_95': var_pct,
            'var_exceeds_limit': var_pct > self.portfolio_risk.max_var_pct,
            'max_drawdown': max_dd,
            'drawdown_exceeds_limit': abs(max_dd) > self.portfolio_risk.max_drawdown_pct,
            'concentration_violations': concentration['violations']
        }
        
        if portfolio_checks['var_exceeds_limit']:
            status = 'WARNING'
        
        if portfolio_checks['drawdown_exceeds_limit']:
            status = 'CRITICAL'
            kill_switch_trigger = KillSwitchReason.EXTREME_LOSS
        
        results['checks']['portfolio_level'] = portfolio_checks
        
        # ===== OPERATIONAL CHECKS =====
        operational_checks = {}
        
        # Data feed check
        data_health = {}
        for ticker in self.position_risk.positions.keys():
            health = self.data_feed_monitor.check_health(ticker)
            data_health[ticker] = health
            if not health['healthy']:
                operational_checks['data_feed_alert'] = health
                kill_switch_trigger = KillSwitchReason.DATA_FEED_FAILURE
        
        # Model anomaly check
        self.model_anomaly_detector.add_predictions(model_predictions)
        anomaly_check = self.model_anomaly_detector.check_anomalies(model_predictions)
        if anomaly_check['is_anomalous']:
            operational_checks['model_anomaly'] = anomaly_check
            kill_switch_trigger = KillSwitchReason.MODEL_ANOMALY
        
        results['checks']['operational'] = operational_checks
        
        # ===== KILL SWITCH DECISION =====
        if kill_switch_trigger and self.kill_switch.can_trade():
            print(f"\n🚨 KILL SWITCH TRIGGERED: {kill_switch_trigger.value}")
            self.kill_switch.activate(kill_switch_trigger, details=results)
            results['kill_switch_triggered'] = True
            results['kill_switch_reason'] = kill_switch_trigger.value
        else:
            results['kill_switch_triggered'] = False
        
        results['overall_status'] = status
        results['can_trade'] = self.kill_switch.can_trade()
        
        return results


# Example: Full integration test
if __name__ == "__main__":
    print("=== COMPREHENSIVE RISK MANAGEMENT SYSTEM ===\n")
    
    # Initialize managers
    position_mgr = PositionLevelRiskManager(
        portfolio_capital=1_000_000,
        kelly_fraction=0.25,
        max_position_pct=0.20
    )
    
    portfolio_mgr = PortfolioLevelRiskManager(
        portfolio_capital=1_000_000,
        max_var_pct=0.03,
        max_drawdown_pct=0.10,
        max_sector_exposure_pct=0.35
    )
    
    risk_system = ComprehensiveRiskManager(
        portfolio_capital=1_000_000,
        position_risk_mgr=position_mgr,
        portfolio_risk_mgr=portfolio_mgr
    )
    
    # Add some positions
    risk_system.position_risk.add_position("RELIANCE", 2500, 400)
    risk_system.position_risk.add_position("TCS", 3000, 300)
    
    # Simulate data feed
    risk_system.data_feed_monitor.record_update("RELIANCE")
    risk_system.data_feed_monitor.record_update("TCS")
    
    # Current prices
    prices = {"RELIANCE": 2450, "TCS": 2950}
    
    # Simulated returns and predictions
    daily_returns = np.random.normal(0.0005, 0.015, 252)
    model_preds = np.array([0.52, 0.48, 0.51, 0.49])
    
    factor_exp = {
        'IT': 0.40,
        'Finance': 0.30,
        'Energy': 0.30
    }
    
    # Run full risk check
    report = risk_system.check_all_risks(
        prices, daily_returns, model_preds, factor_exp
    )
    
    print(f"Risk Report Status: {report['overall_status']}")
    print(f"Can Trade: {report['can_trade']}")
    print(f"Kill Switch Active: {report['kill_switch_triggered']}")
    
    # Simulate extreme loss scenario
    print("\n\n=== SIMULATING EXTREME LOSS ===\n")
    
    prices_crashed = {"RELIANCE": 1800, "TCS": 2100}
    daily_returns_crash = np.random.normal(-0.05, 0.03, 252)
    
    report_crash = risk_system.check_all_risks(
        prices_crashed, daily_returns_crash, model_preds, factor_exp
    )
    
    print(f"Risk Report Status: {report_crash['overall_status']}")
    print(f"Kill Switch Triggered: {report_crash['kill_switch_triggered']}")
    print(f"Reason: {report_crash.get('kill_switch_reason', 'N/A')}")
```

### 22.3.6 Disaster Recovery: Server Crash Handling

Your NSE trading system crashes at 2:30 PM with positions open. You restart at 3:00 PM.

**Problem**: Which positions are still open?

**Solution**: Store all trades in persistent database:

```python
import sqlite3
from datetime import datetime

class PersistentPositionStore:
    """
    Persistent storage of positions.
    Survives server crashes.
    """
    
    def __init__(self, db_path: str = "/data/positions.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                quantity INTEGER NOT NULL,
                position_type TEXT NOT NULL,
                status TEXT NOT NULL,  -- OPEN, CLOSED, STOPPED_OUT
                exit_price REAL,
                exit_time TIMESTAMP,
                realized_pnl REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data_log (
                timestamp TIMESTAMP PRIMARY KEY,
                ticker TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_position(
        self,
        position_id: str,
        ticker: str,
        entry_price: float,
        entry_time: datetime,
        quantity: int,
        position_type: str
    ):
        """Save position to persistent store"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO positions
            (position_id, ticker, entry_price, entry_time, quantity, position_type, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (position_id, ticker, entry_price, entry_time, quantity, position_type, 'OPEN'))
        
        conn.commit()
        conn.close()
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        realized_pnl: float
    ):
        """Mark position as closed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE positions
            SET status='CLOSED', exit_price=?, exit_time=?, realized_pnl=?
            WHERE position_id = ?
        ''', (exit_price, exit_time, realized_pnl, position_id))
        
        conn.commit()
        conn.close()
    
    def load_open_positions(self) -> List[Dict[str, any]]:
        """Load all currently open positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM positions WHERE status = "OPEN"')
        rows = cursor.fetchall()
        
        conn.close()
        
        positions = []
        for row in rows:
            positions.append({
                'position_id': row[0],
                'ticker': row[1],
                'entry_price': row[2],
                'entry_time': row[3],
                'quantity': row[4],
                'position_type': row[5]
            })
        
        return positions
    
    def log_market_data(self, ticker: str, price: float, volume: int):
        """Log market data for audit trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO market_data_log (timestamp, ticker, price, volume)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now(), ticker, price, volume))
        
        conn.commit()
        conn.close()


# Disaster recovery procedure
def disaster_recovery_startup():
    """Called when system restarts"""
    
    store = PersistentPositionStore()
    
    # Load all open positions
    open_positions = store.load_open_positions()
    
    print(f"Recovered {len(open_positions)} open positions:")
    for pos in open_positions:
        print(f"  - {pos['position_type'].upper()} {pos['ticker']}: "
              f"{pos['quantity']} @ ₹{pos['entry_price']:.2f}")
    
    # Important: DO NOT immediately close positions
    # Instead: validate them against current market state
    
    # You would:
    # 1. Verify each position still exists in broker account
    # 2. Fetch current price
    # 3. Validate correlation with other positions
    # 4. Re-initialize risk monitors with current state
    
    return open_positions
```

---

## Summary: Risk Management Architecture

Your complete risk management system has **four layers**:

```
Layer 1: POSITION-LEVEL
├─ Stop-loss execution
├─ Position sizing (Kelly/volatility)
└─ Concentration limits

Layer 2: PORTFOLIO-LEVEL
├─ VaR/CVaR daily monitoring
├─ Factor exposure limits
└─ Drawdown-based controls

Layer 3: OPERATIONAL
├─ Data feed health checks
├─ Model anomaly detection
├─ Execution failure handling
└─ Kill switch triggers

Layer 4: PERSISTENCE
├─ Disaster recovery (DB)
├─ Audit trails
└─ Position reconciliation
```

**When Trading Allowed**:
- Kill switch inactive AND
- All positions within stop-loss AND
- Portfolio VaR < limit AND
- No concentration violations AND
- Data feed healthy AND
- Model outputs normal

**When Kill Switch Triggers**:
- Data feed dead > 2 min
- Model anomaly detected
- Drawdown > 10%
- Manual trigger

---

## Key Equations Reference

**Kelly Criterion**:
$$f^* = \frac{pb - q}{b}$$

**Volatility-Adjusted Stop**:
$$\text{Stop} = \text{Entry} - \lambda \sigma_{\text{daily}} \times \text{Entry}$$

**Value at Risk**:
$$\text{VaR}_{\alpha} = -F^{-1}(\alpha)$$

**Conditional Value at Risk**:
$$\text{CVaR}_{\alpha} = E[L | L > \text{VaR}]$$

**Drawdown**:
$$\text{DD}_t = \frac{P_t - \max(P_{0..t})}{\max(P_{0..t})}$$

---

## Next Steps

1. **Backtest Risk Controls**: Run your historical trading against these systems. How often would kills switch have triggered? Would you have avoided disasters?

2. **Set Your Parameters**: Based on your backtests:
   - What Kelly fraction? (Start 0.25)
   - What max position %? (Recommend 15-20%)
   - What max drawdown? (Recommend 10%)
   - What VaR limit? (Recommend 2-3%)

3. **Test Kill Switch**: Simulate:
   - Data feed failures
   - Model crashes
   - Order rejections
   - Server outages

4. **Deploy Carefully**: Paper trade with live data before real money.

Risk management isn't exciting. It's not where alpha comes from. But it's the difference between a sustainable business and a cautionary tale.

# Chapter 27: Live Performance Analysis

## Introduction

The moment your trading system goes live, the real work begins. Backtests are elegant exercises in historical hindsight—they show what *could* have happened under idealized conditions. Live trading is the crucible where you discover the gap between theory and practice.

This chapter equips you to diagnose that gap. We'll decompose your live returns into alpha (genuine edge), factor exposure (passive bets), transaction costs, and timing luck. You'll learn to compare live performance against backtest assumptions, analyze execution quality, and monitor whether your signals decay in production. We'll then dive into strategy diagnostics: identifying whether your system works in all market regimes, whether calendar patterns help or hurt, and whether crowding is eroding your edge.

For an ML/deep learning engineer entering quantitative finance, this is where your statistical intuition becomes practical currency. You already understand overfitting, data leakage, and distribution shift—these same concepts apply to trading systems, except the cost of being wrong is measured in rupees.

---

## Module 27.1: Performance Attribution

### 27.1.1 The Attribution Framework: Decomposing Returns

Your live P&L looks like a single number: ₹15,320 profit on 47 trades. But that number is a mixture of:

- **Alpha**: Returns from your signal's genuine predictive power
- **Factor exposure**: Passive returns from holding systematic risk (beta)
- **Transaction costs**: Slippage, commissions, spreads
- **Market impact**: The price movement caused by your own trading
- **Timing**: Luck in entry/exit execution
- **Drift**: Signal degradation over time

The performance attribution framework (also called Brinson-Fachler decomposition in institutional asset management, though we'll adapt it for trading signals) decomposes total return into these components.

### 27.1.2 Mathematical Framework

Let's define the return on trade $i$:

$$R_i = \alpha_i + \sum_{j=1}^{k} \beta_{ij} f_j + TC_i + \epsilon_i$$

Where:
- $R_i$: Total return on trade $i$ (entry to exit)
- $\alpha_i$: Alpha (signal-specific return)
- $\beta_{ij}$: Exposure to factor $j$ (e.g., momentum, value, volatility)
- $f_j$: Factor return
- $TC_i$: Transaction costs (commissions, slippage, spread)
- $\epsilon_i$: Residual (unexplained return)

For each trade, we can estimate:

$$\text{Achieved Return} = \text{Benchmark Return} + \text{Active Return}$$

Where the benchmark is typically the buy-and-hold return of the security over the holding period, and active return is the alpha we're trying to capture.

**Execution Quality Metric - VWAP Slippage:**

The Volume-Weighted Average Price (VWAP) is a benchmark that accounts for the price movement throughout the trading day:

$$\text{VWAP} = \frac{\sum_{t=1}^{T} P_t \times V_t}{\sum_{t=1}^{T} V_t}$$

Where $P_t$ is the price at time $t$ and $V_t$ is the volume.

Your slippage relative to VWAP is:

$$\text{Slippage\%} = \frac{\text{Achieved Price} - \text{VWAP}}{\text{VWAP}} \times 100$$

Negative slippage (paying more than VWAP on buys) indicates poor execution or market impact.

**Information Coefficient (IC) - Signal Decay Monitoring:**

The Information Coefficient measures the rank correlation between your signal and subsequent returns:

$$IC = \text{Corr}(\text{Signal}, \text{Return}_{t+1})$$

In production, monitor rolling IC over time windows (daily, weekly, monthly):

$$IC_{\text{rolling}}[t] = \text{Corr}(\text{Signal}[t-W:t], \text{Return}[t:t+W])$$

Where $W$ is the rolling window size (e.g., 20 days).

Expected Information Ratio (assuming independent bets):

$$\text{IR} = IC \times \sqrt{\text{Breadth}}$$

Where breadth is the number of independent bets per period.

### 27.1.3 Live vs. Backtest Performance Comparison

Build a diagnostic table comparing realized vs. expected metrics:

| Metric | Backtest | Live | Variance | Root Cause |
|--------|----------|------|----------|-----------|
| Win Rate | 54% | 48% | -6% | Market regime change? |
| Avg Win/Loss Ratio | 1.2 | 0.9 | -0.3 | Slippage on winners? |
| Sharpe Ratio | 1.8 | 0.6 | -1.2 | Higher volatility in live? |
| Avg Slippage (bps) | 2 | 12 | +10 | Execution algorithm poor? |
| IC (rolling 20d) | 0.15 | 0.08 | -0.07 | Signal decay or crowding? |

### 27.1.4 Production Code: Performance Attribution System

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class Trade:
    """Represents a single executed trade."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    signal_value: float  # The ML model output
    direction: int  # 1 for long, -1 for short
    
    @property
    def gross_return(self) -> float:
        """Return before transaction costs."""
        if self.direction == 1:
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price
    
    @property
    def pnl_rupees(self) -> float:
        """P&L in absolute rupees."""
        return self.direction * self.quantity * (self.exit_price - self.entry_price)


@dataclass
class ExecutionMetrics:
    """Execution quality metrics for a trade."""
    trade_id: str
    symbol: str
    vwap_price: float
    arrival_price: float
    entry_price: float
    exit_price: float
    direction: int
    execution_time_seconds: float
    
    @property
    def entry_slippage_bps(self) -> float:
        """Entry slippage relative to VWAP in basis points."""
        if self.direction == 1:  # Buy
            slippage = (self.entry_price - self.vwap_price) / self.vwap_price
        else:  # Sell
            slippage = (self.vwap_price - self.entry_price) / self.vwap_price
        return slippage * 10000
    
    @property
    def arrival_price_impact_bps(self) -> float:
        """Market impact: difference between arrival price and VWAP."""
        impact = (abs(self.arrival_price - self.vwap_price) / self.vwap_price)
        return impact * 10000


class PerformanceAttributor:
    """
    Decomposes live P&L into alpha, factor exposure, and costs.
    
    Attributes:
        trades: List of executed Trade objects
        factor_returns: DataFrame of daily factor returns (Mkt-RF, SMB, HML, etc.)
        risk_free_rate: Daily risk-free rate
    """
    
    def __init__(
        self,
        trades: List[Trade],
        factor_returns: pd.DataFrame,
        risk_free_rate: float = 0.04 / 252
    ):
        self.trades = trades
        self.factor_returns = factor_returns
        self.risk_free_rate = risk_free_rate
    
    def compute_trade_returns(
        self,
        include_costs: bool = True,
        commission_bps: float = 10,
        borrow_cost_bps: float = 0
    ) -> pd.DataFrame:
        """
        Compute return for each trade, optionally including costs.
        
        Args:
            include_costs: Whether to deduct transaction costs
            commission_bps: Commission per trade (both entry and exit)
            borrow_cost_bps: Cost to borrow shares for short sells (annualized, adjusted to holding period)
        
        Returns:
            DataFrame with trade-level returns and attribution
        """
        trade_data = []
        
        for trade in self.trades:
            holding_period_days = (trade.exit_time - trade.entry_time).days + 1
            
            # Gross return
            gross_return = trade.gross_return
            
            # Transaction costs
            total_commission = commission_bps * 2 / 10000  # Entry + exit
            annualized_borrow_cost = borrow_cost_bps / 10000
            daily_borrow_cost = annualized_borrow_cost / 252
            total_borrow_cost = daily_borrow_cost * holding_period_days if trade.direction == -1 else 0
            
            # Net return
            net_return = gross_return - total_commission - total_borrow_cost if include_costs else gross_return
            
            # Benchmark return (buy-and-hold)
            # In practice, fetch actual OHLCV data and compute
            # This is simplified - assume we have market data
            # benchmark_return = self._get_benchmark_return(trade.symbol, trade.entry_time, trade.exit_time)
            
            trade_data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'holding_days': holding_period_days,
                'signal_value': trade.signal_value,
                'direction': trade.direction,
                'gross_return': gross_return,
                'net_return': net_return,
                'commission_cost': total_commission,
                'borrow_cost': total_borrow_cost,
                'total_cost': total_commission + total_borrow_cost,
            })
        
        return pd.DataFrame(trade_data)
    
    def estimate_factor_exposures(
        self,
        trade_returns_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Estimate factor betas for each position using Fama-French factors.
        
        This is a simplified approach: in production, you'd use cross-sectional
        regression or rolling windows. We compute factor exposures by correlating
        trade returns with factor returns.
        
        Args:
            trade_returns_df: DataFrame from compute_trade_returns()
        
        Returns:
            Dictionary of factor names to beta estimates
        """
        # Align trade dates with factor returns
        factor_exposures = {}
        
        # For simplicity, compute average correlation of returns with factors
        # In production: use rolling cross-sectional regression
        for factor_col in self.factor_returns.columns:
            correlation = np.corrcoef(
                trade_returns_df['net_return'].fillna(0),
                self.factor_returns[factor_col].mean()  # Simplified
            )[0, 1]
            factor_exposures[factor_col] = correlation
        
        return factor_exposures
    
    def decompose_pnl(
        self,
        trade_returns_df: pd.DataFrame,
        factor_exposures: Dict[str, float],
        factor_daily_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Decompose total P&L into components.
        
        Args:
            trade_returns_df: Trade-level returns
            factor_exposures: Beta estimates
            factor_daily_returns: Time series of daily factor returns
        
        Returns:
            Dictionary with attribution breakdown
        """
        total_return = trade_returns_df['net_return'].sum()
        total_cost = trade_returns_df['total_cost'].sum()
        
        # Approximate factor returns contribution
        # In production, use proper cross-sectional regression
        factor_return_contribution = sum(
            factor_exposures.get(factor, 0) * factor_daily_returns[factor].mean() * 252
            for factor in factor_daily_returns.columns
            if factor in factor_exposures
        )
        
        # Residual alpha
        alpha = total_return - factor_return_contribution
        
        return {
            'total_return': total_return,
            'gross_return': total_return + total_cost,
            'factor_contribution': factor_return_contribution,
            'alpha': alpha,
            'transaction_costs': total_cost,
            'num_trades': len(trade_returns_df),
            'win_rate': (trade_returns_df['net_return'] > 0).sum() / len(trade_returns_df) if len(trade_returns_df) > 0 else 0,
        }
    
    def execution_quality_analysis(
        self,
        execution_metrics: List[ExecutionMetrics]
    ) -> Dict[str, float]:
        """
        Analyze execution quality using VWAP and arrival price benchmarks.
        
        Args:
            execution_metrics: List of ExecutionMetrics objects
        
        Returns:
            Dictionary with execution statistics
        """
        if not execution_metrics:
            return {}
        
        df = pd.DataFrame([
            {
                'trade_id': em.trade_id,
                'entry_slippage_bps': em.entry_slippage_bps,
                'arrival_impact_bps': em.arrival_price_impact_bps,
                'execution_time_sec': em.execution_time_seconds,
            }
            for em in execution_metrics
        ])
        
        return {
            'avg_entry_slippage_bps': df['entry_slippage_bps'].mean(),
            'median_entry_slippage_bps': df['entry_slippage_bps'].median(),
            'max_entry_slippage_bps': df['entry_slippage_bps'].max(),
            'avg_arrival_impact_bps': df['arrival_impact_bps'].mean(),
            'avg_execution_time_sec': df['execution_time_sec'].mean(),
            'trades_with_positive_slippage': (df['entry_slippage_bps'] > 0).sum(),
        }
    
    def signal_decay_analysis(
        self,
        trade_returns_df: pd.DataFrame,
        window_days: int = 20
    ) -> pd.DataFrame:
        """
        Monitor Information Coefficient (IC) over rolling windows to detect signal decay.
        
        Args:
            trade_returns_df: Trade returns DataFrame
            window_days: Rolling window size in days
        
        Returns:
            DataFrame with rolling IC and related metrics
        """
        trade_returns_df = trade_returns_df.sort_values('entry_time').reset_index(drop=True)
        
        rolling_ic = []
        
        for i in range(window_days, len(trade_returns_df)):
            window = trade_returns_df.iloc[i-window_days:i]
            
            if len(window) > 1:
                # IC: rank correlation between signal and returns
                ic = window['signal_value'].rank().corr(window['net_return'].rank())
                
                # Information Ratio (assuming ~1 bet per day)
                breadth = len(window)
                information_ratio = ic * np.sqrt(breadth) if not np.isnan(ic) else np.nan
                
                rolling_ic.append({
                    'window_end_time': window['exit_time'].iloc[-1],
                    'ic': ic,
                    'information_ratio': information_ratio,
                    'num_trades_in_window': len(window),
                    'avg_return': window['net_return'].mean(),
                    'return_std': window['net_return'].std(),
                })
        
        return pd.DataFrame(rolling_ic)
    
    def generate_attribution_report(
        self,
        trade_returns_df: pd.DataFrame,
        execution_metrics: List[ExecutionMetrics],
        factor_daily_returns: pd.DataFrame,
    ) -> Dict:
        """
        Generate comprehensive attribution report.
        
        Args:
            trade_returns_df: Trade-level returns
            execution_metrics: Execution quality metrics
            factor_daily_returns: Daily factor returns time series
        
        Returns:
            Dictionary with complete attribution analysis
        """
        factor_exposures = self.estimate_factor_exposures(trade_returns_df)
        pnl_decomposition = self.decompose_pnl(trade_returns_df, factor_exposures, factor_daily_returns)
        execution_stats = self.execution_quality_analysis(execution_metrics)
        signal_decay = self.signal_decay_analysis(trade_returns_df)
        
        return {
            'pnl_decomposition': pnl_decomposition,
            'factor_exposures': factor_exposures,
            'execution_quality': execution_stats,
            'signal_decay_analysis': signal_decay.to_dict(orient='records') if len(signal_decay) > 0 else [],
            'report_timestamp': datetime.now().isoformat(),
        }


# Example usage
if __name__ == "__main__":
    # Create sample trades
    trades = [
        Trade(
            trade_id="T001",
            symbol="RELIANCE.NS",
            entry_time=datetime(2026, 4, 1, 9, 30),
            exit_time=datetime(2026, 4, 3, 15, 30),
            entry_price=2800.0,
            exit_price=2850.5,
            quantity=10,
            signal_value=0.75,
            direction=1
        ),
        Trade(
            trade_id="T002",
            symbol="INFY.NS",
            entry_time=datetime(2026, 4, 2, 10, 0),
            exit_time=datetime(2026, 4, 4, 14, 0),
            entry_price=1650.0,
            exit_price=1640.0,
            quantity=5,
            signal_value=0.82,
            direction=1
        ),
    ]
    
    # Create mock factor returns
    dates = pd.date_range('2026-01-01', periods=100, freq='D')
    factor_returns = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.0004, 0.01, 100),
        'SMB': np.random.normal(0.0002, 0.008, 100),
        'HML': np.random.normal(0.0001, 0.007, 100),
    }, index=dates)
    
    # Initialize attributor
    attributor = PerformanceAttributor(trades, factor_returns)
    
    # Compute trade returns
    trade_returns = attributor.compute_trade_returns(include_costs=True)
    print("Trade Returns:")
    print(trade_returns)
    
    # Create execution metrics
    execution_metrics = [
        ExecutionMetrics(
            trade_id="T001",
            symbol="RELIANCE.NS",
            vwap_price=2798.5,
            arrival_price=2799.0,
            entry_price=2800.0,
            exit_price=2850.5,
            direction=1,
            execution_time_seconds=1.2
        ),
    ]
    
    # Generate report
    report = attributor.generate_attribution_report(
        trade_returns,
        execution_metrics,
        factor_returns
    )
    print("\nAttribution Report:")
    print(json.dumps(report, indent=2, default=str))
```

### 27.1.5 Integration with Zerodha

Zerodha's API provides real-time execution data through Kite Connect. Here's how to extract execution metrics:

```python
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

class ZerodhaExecutionTracker:
    """Extract execution data from Zerodha Kite Connect."""
    
    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
    
    def get_trade_execution_metrics(
        self,
        trade_id: str,
        symbol: str,
        entry_timestamp: datetime,
        exit_timestamp: datetime
    ) -> ExecutionMetrics:
        """
        Fetch VWAP and execution prices from Zerodha.
        
        Args:
            trade_id: Internal trade identifier
            symbol: NSE symbol (e.g., 'RELIANCE1')
            entry_timestamp: When position was entered
            exit_timestamp: When position was exited
        
        Returns:
            ExecutionMetrics object
        """
        # Fetch historical trades from Zerodha
        trades_response = self.kite.orders()
        
        # Filter trades for this symbol and time window
        relevant_trades = [
            t for t in trades_response
            if t['tradingsymbol'] == symbol
            and entry_timestamp <= datetime.fromtimestamp(t['order_timestamp']) <= exit_timestamp
        ]
        
        # Calculate VWAP (simplified - in production, fetch tick data)
        total_value = sum(t['average_price'] * t['filled_quantity'] for t in relevant_trades)
        total_volume = sum(t['filled_quantity'] for t in relevant_trades)
        vwap = total_value / total_volume if total_volume > 0 else 0
        
        # Extract entry and exit prices from actual orders
        entry_price = relevant_trades[0]['average_price'] if relevant_trades else 0
        exit_price = relevant_trades[-1]['average_price'] if relevant_trades else 0
        arrival_price = relevant_trades[0]['average_price'] if relevant_trades else 0
        
        execution_time = (
            datetime.fromtimestamp(relevant_trades[-1]['order_timestamp']) -
            datetime.fromtimestamp(relevant_trades[0]['order_timestamp'])
        ).total_seconds() if relevant_trades else 0
        
        return ExecutionMetrics(
            trade_id=trade_id,
            symbol=symbol,
            vwap_price=vwap,
            arrival_price=arrival_price,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=1,  # Determine from actual order
            execution_time_seconds=execution_time
        )
    
    def get_daily_pnl(self, symbol: str, date: datetime) -> float:
        """Fetch realized P&L for a symbol on a given date."""
        positions = self.kite.positions()
        
        # Find position matching symbol
        for position in positions.get('net', []):
            if position['tradingsymbol'] == symbol:
                return position['pnl']  # Realized P&L
        
        return 0.0
```

---

## Module 27.2: Strategy Diagnostics

### 27.2.1 Regime Analysis: Does Your Strategy Work in All Market Conditions?

A crucial discovery in live trading: your signal's performance varies dramatically by market regime. A momentum strategy thrives in trending markets but gets destroyed in mean-reverting chop. A mean-reversion strategy works great when volatility spikes but fails in smooth uptrends.

**Regime Classification:**

Identify market regimes using a simple classification:

1. **Bull Market**: $R_{t:t+20} > \text{75th percentile}$ (last 20 days' returns in top quartile)
2. **Bear Market**: $R_{t:t+20} < \text{25th percentile}$ (last 20 days' returns in bottom quartile)
3. **Sideways/Consolidation**: $|R_{t:t+20}| < \text{median absolute return}$ (low directional movement)
4. **High Volatility**: $\sigma_{t:t+20} > \text{75th percentile}$ (volatility in top quartile)

For each regime, compute:

$$\text{Strategy Return}_{\text{regime}} = \frac{\sum_{i \in \text{regime}} R_i}{\text{number of trades in regime}}$$

$$\text{Sharpe}_{\text{regime}} = \frac{\text{Return}_{\text{regime}}}{\text{StdDev}_{\text{regime}}} \times \sqrt{252}$$

### 27.2.2 Calendar Effects and Seasonal Patterns

Test for systematic performance variations:

**Day-of-Week Effect:**

$$H_0: \mu_{\text{Monday}} = \mu_{\text{Tuesday}} = \cdots = \mu_{\text{Friday}}$$

Compute average return for each day of week. Test significance using one-way ANOVA.

**Month-of-Year Effect:**

Similarly partition returns by calendar month.

**Earnings Season Effect:**

Does your strategy perform differently during earnings periods (high volatility)?

**Holiday Proximity:**

Returns on days immediately before/after holidays may differ from typical days.

### 27.2.3 Crowding Analysis: Is Your Edge Being Arbitraged Away?

If many traders use the same signal, price impact increases and alpha decays. Detect crowding by:

1. **Signal Correlation with Other Factors**: If your signal is highly correlated with known factors (momentum, reversal, size), you're crowded
2. **Predictive Decay**: Monitor whether IC drops as your asset under management (or notional trading volume) grows
3. **Execution Deterioration**: Rising slippage indicates others are trading the same direction

**Mathematical Formulation:**

Define a crowding metric:

$$\text{Crowding Index}_t = \frac{\text{Notional Volume}_t}{30\text{-day Average Volume}} \times \frac{\text{Signal Std Dev}_t}{\text{Historical Signal Std Dev}}$$

As crowding increases, alpha should decline:

$$IC_t = \alpha - \beta \times \text{Crowding Index}_t + \epsilon_t$$

Regress rolling IC against crowding to estimate $\beta$ (should be negative).

### 27.2.4 Correlation with Known Factors

Your "alpha" might actually be exposure to well-known factors. Run cross-sectional regressions:

$$R_i = \alpha + \beta_{\text{Mkt}} F_{\text{Mkt},i} + \beta_{\text{Mom}} F_{\text{Mom},i} + \beta_{\text{Rev}} F_{\text{Rev},i} + \epsilon_i$$

Where:
- $F_{\text{Mkt},i}$: Market exposure (beta)
- $F_{\text{Mom},i}$: Momentum factor (12-month return)
- $F_{\text{Rev},i}$: Reversal factor (1-month return)

If your alpha $\alpha$ is close to zero and coefficients are large, you're just harvesting known factor risk.

### 27.2.5 Production Code: Strategy Diagnostics System

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from scipy import stats
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

@dataclass
class MarketRegime:
    """Definition of a market regime."""
    name: str
    condition: str  # Descriptive condition
    returns: List[float]
    dates: List[datetime]


class StrategyDiagnostics:
    """
    Comprehensive strategy diagnostics system.
    
    Analyzes performance across regimes, calendar effects, crowding, and factor correlations.
    
    Attributes:
        trade_returns_df: DataFrame with columns [entry_time, symbol, net_return, signal_value, direction]
        market_data: DataFrame with columns [date, close, high, low, volume, symbol]
    """
    
    def __init__(
        self,
        trade_returns_df: pd.DataFrame,
        market_data: pd.DataFrame
    ):
        self.trade_returns_df = trade_returns_df.copy()
        self.market_data = market_data.copy()
        
        # Ensure datetime columns
        self.trade_returns_df['entry_time'] = pd.to_datetime(self.trade_returns_df['entry_time'])
        self.market_data['date'] = pd.to_datetime(self.market_data['date'])
    
    def identify_regimes(
        self,
        lookback_days: int = 20
    ) -> pd.DataFrame:
        """
        Classify market regimes for each trading date.
        
        Args:
            lookback_days: Window for computing regime statistics
        
        Returns:
            DataFrame with regime classification for each date
        """
        # Compute daily returns for NSE index (e.g., Nifty 50)
        # In practice, fetch from market_data or external source
        daily_returns = self.market_data.sort_values('date').groupby('date')['close'].mean().pct_change()
        
        rolling_returns_20d = daily_returns.rolling(lookback_days).sum()
        rolling_volatility = daily_returns.rolling(lookback_days).std()
        
        # Define thresholds (75th/25th percentiles)
        returns_75th = rolling_returns_20d.quantile(0.75)
        returns_25th = rolling_returns_20d.quantile(0.25)
        vol_75th = rolling_volatility.quantile(0.75)
        
        regimes = []
        
        for date in rolling_returns_20d.index:
            ret = rolling_returns_20d[date]
            vol = rolling_volatility[date]
            
            if vol > vol_75th:
                regime = 'High Volatility'
            elif ret > returns_75th:
                regime = 'Bull'
            elif ret < returns_25th:
                regime = 'Bear'
            else:
                regime = 'Sideways'
            
            regimes.append({
                'date': date,
                'regime': regime,
                '20d_return': ret,
                '20d_volatility': vol,
            })
        
        return pd.DataFrame(regimes)
    
    def regime_analysis(
        self,
        regime_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute performance metrics for each regime.
        
        Args:
            regime_df: DataFrame from identify_regimes()
        
        Returns:
            Dictionary: regime_name -> {return, sharpe, win_rate, num_trades}
        """
        # Merge trade returns with regime data
        self.trade_returns_df['entry_date'] = self.trade_returns_df['entry_time'].dt.date
        regime_df['date'] = pd.to_datetime(regime_df['date']).dt.date
        
        merged = self.trade_returns_df.merge(
            regime_df[['date', 'regime']],
            left_on='entry_date',
            right_on='date',
            how='left'
        )
        
        regime_stats = {}
        
        for regime in merged['regime'].unique():
            if pd.isna(regime):
                continue
            
            regime_trades = merged[merged['regime'] == regime]
            returns = regime_trades['net_return']
            
            if len(returns) > 0:
                regime_stats[regime] = {
                    'mean_return': returns.mean(),
                    'total_return': returns.sum(),
                    'std_return': returns.std(),
                    'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                    'win_rate': (returns > 0).sum() / len(returns),
                    'num_trades': len(returns),
                    'avg_return_per_trade': returns.mean(),
                }
        
        return regime_stats
    
    def day_of_week_analysis(self) -> pd.DataFrame:
        """
        Analyze returns by day of week.
        
        Returns:
            DataFrame with day-of-week statistics
        """
        self.trade_returns_df['day_of_week'] = self.trade_returns_df['entry_time'].dt.day_name()
        
        dow_stats = self.trade_returns_df.groupby('day_of_week')['net_return'].agg([
            'count', 'mean', 'std', 'sum',
            ('sharpe', lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0)
        ]).round(4)
        
        return dow_stats
    
    def month_of_year_analysis(self) -> pd.DataFrame:
        """
        Analyze returns by month of year.
        
        Returns:
            DataFrame with month-of-year statistics
        """
        self.trade_returns_df['month'] = self.trade_returns_df['entry_time'].dt.month
        self.trade_returns_df['month_name'] = self.trade_returns_df['entry_time'].dt.strftime('%B')
        
        month_stats = self.trade_returns_df.groupby('month_name')['net_return'].agg([
            'count', 'mean', 'std', 'sum',
            ('sharpe', lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0)
        ]).round(4)
        
        return month_stats
    
    def earnings_season_analysis(
        self,
        earnings_dates: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compare strategy performance during earnings vs. non-earnings periods.
        
        Args:
            earnings_dates: DataFrame with columns [date, symbol]
        
        Returns:
            Dictionary with comparison statistics
        """
        self.trade_returns_df['entry_date'] = self.trade_returns_df['entry_time'].dt.date
        earnings_dates['date'] = pd.to_datetime(earnings_dates['date']).dt.date
        
        # Mark earnings days (within 2 days of earnings announcement)
        earnings_symbols = set(earnings_dates['symbol'].unique())
        
        def is_earnings_season(row):
            symbol = row['symbol']
            entry_date = row['entry_date']
            
            if symbol not in earnings_symbols:
                return False
            
            # Check if within 2 days of earnings
            symbol_earnings = earnings_dates[earnings_dates['symbol'] == symbol]['date']
            for earnings_date in symbol_earnings:
                if abs((entry_date - earnings_date).days) <= 2:
                    return True
            return False
        
        self.trade_returns_df['is_earnings'] = self.trade_returns_df.apply(is_earnings_season, axis=1)
        
        earnings_trades = self.trade_returns_df[self.trade_returns_df['is_earnings']]
        non_earnings_trades = self.trade_returns_df[~self.trade_returns_df['is_earnings']]
        
        return {
            'earnings_mean_return': earnings_trades['net_return'].mean() if len(earnings_trades) > 0 else 0,
            'earnings_num_trades': len(earnings_trades),
            'earnings_win_rate': (earnings_trades['net_return'] > 0).sum() / len(earnings_trades) if len(earnings_trades) > 0 else 0,
            'non_earnings_mean_return': non_earnings_trades['net_return'].mean() if len(non_earnings_trades) > 0 else 0,
            'non_earnings_num_trades': len(non_earnings_trades),
            'non_earnings_win_rate': (non_earnings_trades['net_return'] > 0).sum() / len(non_earnings_trades) if len(non_earnings_trades) > 0 else 0,
            'return_difference': earnings_trades['net_return'].mean() - non_earnings_trades['net_return'].mean() if (len(earnings_trades) > 0 and len(non_earnings_trades) > 0) else 0,
        }
    
    def crowding_analysis(
        self,
        notional_volumes: pd.DataFrame
    ) -> Tuple[float, pd.DataFrame]:
        """
        Analyze crowding effect by correlating IC with notional volume.
        
        Args:
            notional_volumes: DataFrame with columns [date, notional_volume]
        
        Returns:
            Tuple of (crowding_beta, regression_results_df)
        """
        # Compute rolling IC (from Module 27.1 signal_decay_analysis)
        rolling_ic_data = []
        
        self.trade_returns_df = self.trade_returns_df.sort_values('entry_time').reset_index(drop=True)
        
        for i in range(20, len(self.trade_returns_df)):
            window = self.trade_returns_df.iloc[i-20:i]
            
            if len(window) > 1:
                ic = window['signal_value'].rank().corr(window['net_return'].rank())
                window_date = window['entry_time'].iloc[-1].date()
                
                rolling_ic_data.append({
                    'date': window_date,
                    'ic': ic,
                })
        
        ic_df = pd.DataFrame(rolling_ic_data)
        
        # Merge with notional volumes
        notional_volumes['date'] = pd.to_datetime(notional_volumes['date']).dt.date
        merged_df = ic_df.merge(
            notional_volumes[['date', 'notional_volume']],
            on='date',
            how='left'
        ).dropna()
        
        # Normalize crowding index
        merged_df['log_notional'] = np.log(merged_df['notional_volume'] + 1)
        
        # Simple linear regression: IC ~ log(notional volume)
        if len(merged_df) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                merged_df['log_notional'],
                merged_df['ic']
            )
            
            return {
                'crowding_beta': slope,
                'crowding_beta_stderr': std_err,
                'crowding_pvalue': p_value,
                'r_squared': r_value ** 2,
                'interpretation': 'Negative beta indicates IC decays with notional volume (crowding effect)',
            }, merged_df
        
        return {}, merged_df
    
    def factor_correlation_analysis(
        self,
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Correlate strategy signal with known factors.
        
        Args:
            factor_returns: DataFrame with factor columns [Mkt-RF, SMB, HML, Mom, Rev]
        
        Returns:
            Dictionary of correlations
        """
        # Aggregate signals by day and correlate with factor returns
        daily_signal = self.trade_returns_df.groupby(
            self.trade_returns_df['entry_time'].dt.date
        )['signal_value'].mean()
        
        daily_signal.index = pd.to_datetime(daily_signal.index)
        
        # Align with factor returns
        merged = pd.DataFrame({
            'signal': daily_signal,
        }).join(factor_returns, how='inner')
        
        correlations = {}
        
        for factor in factor_returns.columns:
            if factor in merged.columns:
                corr = merged['signal'].corr(merged[factor])
                correlations[f'signal_vs_{factor}'] = corr
        
        return correlations
    
    def unintended_factor_exposure(
        self,
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Cross-sectional regression: Decompose strategy returns into factor betas.
        
        Fits: Return_i = alpha + beta_Mkt * Mkt_i + beta_SMB * SMB_i + ...
        
        Args:
            factor_returns: DataFrame with daily factor returns
        
        Returns:
            Dictionary with regression coefficients and fit statistics
        """
        # Aggregate returns by holding period and compute betas
        # This is simplified; in production use true cross-sectional regression
        
        returns = self.trade_returns_df['net_return'].values
        signals = self.trade_returns_df['signal_value'].values
        
        # Average factor return as proxy for market exposure
        avg_mkt_return = factor_returns['Mkt-RF'].mean() * 252 if 'Mkt-RF' in factor_returns else 0
        
        # Correlation indicates unintended exposure
        correlations = self.factor_correlation_analysis(factor_returns)
        
        return {
            'factor_correlations': correlations,
            'avg_market_return': avg_mkt_return,
            'signal_driven_by_known_factors': 'Yes' if any(abs(v) > 0.5 for v in correlations.values()) else 'No',
        }
    
    def generate_diagnostics_report(
        self,
        regime_df: pd.DataFrame,
        earnings_dates: pd.DataFrame,
        notional_volumes: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> Dict:
        """
        Generate comprehensive strategy diagnostics report.
        
        Args:
            regime_df: Market regime classification
            earnings_dates: Earnings announcement dates
            notional_volumes: Daily notional volumes traded
            factor_returns: Daily factor returns
        
        Returns:
            Complete diagnostics dictionary
        """
        regime_analysis_results = self.regime_analysis(regime_df)
        dow_analysis_results = self.day_of_week_analysis()
        month_analysis_results = self.month_of_year_analysis()
        earnings_analysis_results = self.earnings_season_analysis(earnings_dates)
        crowding_results, crowding_df = self.crowding_analysis(notional_volumes)
        factor_correlation_results = self.factor_correlation_analysis(factor_returns)
        factor_exposure_results = self.unintended_factor_exposure(factor_returns)
        
        return {
            'regime_analysis': regime_analysis_results,
            'day_of_week_analysis': dow_analysis_results.to_dict(),
            'month_analysis': month_analysis_results.to_dict(),
            'earnings_season_analysis': earnings_analysis_results,
            'crowding_analysis': crowding_results,
            'factor_correlations': factor_correlation_results,
            'unintended_factor_exposure': factor_exposure_results,
            'report_timestamp': datetime.now().isoformat(),
        }


# Example usage
if __name__ == "__main__":
    # Create sample trade returns
    np.random.seed(42)
    n_trades = 200
    
    trade_returns_df = pd.DataFrame({
        'entry_time': pd.date_range('2026-01-01', periods=n_trades, freq='D'),
        'symbol': np.random.choice(['RELIANCE.NS', 'INFY.NS', 'TCS.NS'], n_trades),
        'net_return': np.random.normal(0.002, 0.01, n_trades),  # 20bps avg, 100bps vol
        'signal_value': np.random.uniform(0, 1, n_trades),
        'direction': np.random.choice([1, -1], n_trades),
    })
    
    # Create sample market data
    market_data = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=100, freq='D'),
        'symbol': 'NIFTY50',
        'close': 22000 + np.cumsum(np.random.normal(20, 150, 100)),
        'high': 22500 + np.cumsum(np.random.normal(20, 150, 100)),
        'low': 21500 + np.cumsum(np.random.normal(20, 150, 100)),
        'volume': np.random.uniform(1e8, 5e8, 100),
    })
    
    # Create sample earnings dates
    earnings_dates = pd.DataFrame({
        'symbol': ['RELIANCE.NS', 'INFY.NS', 'TCS.NS'],
        'date': pd.to_datetime(['2026-01-15', '2026-02-10', '2026-03-20']),
    })
    
    # Create sample notional volumes
    notional_volumes = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=100, freq='D'),
        'notional_volume': np.random.uniform(1e8, 1e10, 100),
    })
    
    # Create sample factor returns
    factor_returns = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.0004, 0.01, 100),
        'SMB': np.random.normal(0.0002, 0.008, 100),
        'HML': np.random.normal(0.0001, 0.007, 100),
    }, index=pd.date_range('2026-01-01', periods=100, freq='D'))
    
    # Initialize diagnostics
    diagnostics = StrategyDiagnostics(trade_returns_df, market_data)
    
    # Identify regimes
    regime_df = diagnostics.identify_regimes(lookback_days=20)
    
    # Generate full report
    report = diagnostics.generate_diagnostics_report(
        regime_df,
        earnings_dates,
        notional_volumes,
        factor_returns
    )
    
    print("Strategy Diagnostics Report:")
    print("\nRegime Analysis:")
    for regime, stats in report['regime_analysis'].items():
        print(f"\n{regime}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n\nDay-of-Week Analysis:")
    print(pd.DataFrame(report['day_of_week_analysis']).round(4))
    
    print("\n\nEarnings Season Analysis:")
    for key, value in report['earnings_season_analysis'].items():
        print(f"{key}: {value:.4f}")
```

### 27.2.6 Interpreting Diagnostics: A Case Study

Imagine your diagnostics reveal:

| Finding | Implication | Action |
|---------|------------|--------|
| 75% returns in Bull regime, -2% in Bear | Strategy is directional, not market-neutral | Add hedge or regime filter |
| -0.65 correlation with Market Factor | You're essentially short the market | This is unintended leverage |
| IC drops from 0.12 to 0.04 as volume increases | Crowding eroding alpha | Reduce position size or add diversification |
| Win rate 52% Monday-Thursday, 38% Friday | Friday liquidity dries up | Avoid Friday exits |
| Returns spike in earnings season | You're harvesting volatility, not signal | Consider volatility filter |

---

## Chapter Summary

Live performance analysis is the bridge between theory and practice. Performance attribution decomposes your returns into components—alpha, factor exposure, costs, execution quality. The PerformanceAttributor class provides production-ready code for tracking these metrics and monitoring signal decay through Information Coefficient.

Strategy diagnostics reveals whether your edge is real, whether it works across market conditions, and whether it's being arbitraged away by crowding. The StrategyDiagnostics class enables systematic testing of regime dependence, calendar effects, and unintended factor exposures.

As you integrate these tools with Zerodha's API, you'll develop the discipline to quickly identify when live performance diverges from backtest, diagnose root causes, and adapt your system. This cycle—measure, diagnose, adapt—separates successful quant traders from those who blame bad luck.

---

## Further Reading

- **Grinold, R. & Kahn, R.** (1999). *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* — Foundational work on performance attribution
- **Arnott, R., Beck, S., Kalesnik, V., & West, J.** (2016). "How Can 'Quantitative Easing' Benefit Long-Term Investors?" — On crowding and alpha decay
- **Zerodha Kite Connect Documentation** — API reference for extracting execution data
- **Campbell, J. Y., Lo, A. W., & MacKinlay, A. C.** (1997). *The Econometrics of Financial Markets* — Rigorous treatment of factor models and market microstructure

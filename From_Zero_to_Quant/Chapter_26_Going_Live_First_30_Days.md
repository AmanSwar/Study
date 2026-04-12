# Chapter 26: Going Live — The First 30 Days

## Introduction

You've backtested your strategy. The Sharpe ratio looks good. The maximum drawdown is acceptable. You've validated on out-of-sample data. Now comes the moment that separates theoretical quantitative finance from operational quantitative finance: **going live with real money**.

This chapter is written for someone with deep ML/systems engineering expertise but zero finance knowledge. You understand how to build robust systems, handle failures gracefully, and monitor distributed applications. Your challenge is learning what matters in the first 30 days of live trading—not just technically, but strategically and psychologically.

The first 30 days will reveal problems your backtest never hinted at:
- **Slippage** far exceeding historical estimates
- **Market microstructure** differences between backtest assumptions and reality
- **API failures** that weren't in your test plan
- **Model drift** that happens in days, not months
- **Your own emotional responses** to losses that appeared theoretical

This chapter teaches you to navigate these challenges with a systematic, engineering-focused approach. We'll build:
1. A pre-launch checklist that verifies every system component works with real data and capital
2. A day-by-day operations manual with clear decision points
3. A troubleshooting framework based on the most common failure modes

By the end of this chapter, you'll have the tools to go live confidently and stay live safely.

---

## Module 26.1: Pre-Launch Checklist — System Validation Before Real Money

### Why This Checklist Matters

The gap between a test environment and production is where most failures happen. In software engineering, you'd never deploy without staging. In trading, that staging is your pre-launch checklist. This section is a checklist, but more importantly, it's a validation framework that tests every assumption your system makes.

### Part A: Data Feed Validation (Day -5 to -3)

Your entire strategy depends on data. Before you trade, verify every single data source that your system depends on.

#### 26.1.1 Historical Data Validation

Before connecting to live APIs, validate that your historical data matches the NSE's official records.

```python
# validation/historical_data_checker.py
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataValidationReport:
    """Report from historical data validation."""
    ticker: str
    start_date: datetime
    end_date: datetime
    total_trading_days: int
    missing_days: List[datetime]
    gap_hours: List[Tuple[datetime, datetime]]
    price_anomalies: List[Dict]
    volume_anomalies: List[Dict]
    ohlc_logic_failures: List[Dict]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        msg = f"\n{'='*70}\n"
        msg += f"DATA VALIDATION REPORT: {self.ticker}\n"
        msg += f"Period: {self.start_date.date()} to {self.end_date.date()}\n"
        msg += f"{'='*70}\n"
        msg += f"Total trading days expected: {self.total_trading_days}\n"
        msg += f"Missing trading days: {len(self.missing_days)}\n"
        if self.missing_days:
            msg += f"  First missing: {self.missing_days[0].date()}\n"
            msg += f"  Last missing: {self.missing_days[-1].date()}\n"
        msg += f"\nIntraday gaps (>1 hour): {len(self.gap_hours)}\n"
        if self.gap_hours:
            msg += f"  Example: {self.gap_hours[0][0]} to {self.gap_hours[0][1]}\n"
        msg += f"\nPrice anomalies: {len(self.price_anomalies)}\n"
        msg += f"Volume anomalies: {len(self.volume_anomalies)}\n"
        msg += f"OHLC logic failures: {len(self.ohlc_logic_failures)}\n"
        msg += f"{'='*70}\n"
        return msg

class HistoricalDataValidator:
    """Validate historical OHLCV data from NSE."""
    
    def __init__(self, nse_calendar_path: str):
        """
        Initialize validator with NSE trading calendar.
        
        Args:
            nse_calendar_path: Path to CSV with NSE trading dates
                              Expected columns: [date]
        """
        self.trading_dates = pd.read_csv(
            nse_calendar_path, 
            parse_dates=['date']
        )['date'].tolist()
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        ticker: str,
        check_intraday_gaps: bool = False
    ) -> DataValidationReport:
        """
        Validate OHLCV dataframe.
        
        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            ticker: Stock ticker symbol
            check_intraday_gaps: If True, check for gaps > 1 hour within trading hours
            
        Returns:
            DataValidationReport object with detailed findings
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        # Expected trading days
        expected_dates = [
            d for d in self.trading_dates 
            if start_date.date() <= d.date() <= end_date.date()
        ]
        
        # Missing days
        actual_dates = df['timestamp'].dt.date.unique()
        missing_days = [
            d for d in expected_dates 
            if d.date() not in actual_dates
        ]
        
        # Intraday gaps
        gap_hours = []
        if check_intraday_gaps and len(df) > 1:
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600
            gap_threshold = 1.0  # 1 hour
            gap_rows = df[df['time_diff'] > gap_threshold]
            gap_hours = [
                (row['timestamp'] - timedelta(hours=gap), row['timestamp'])
                for gap, (_, row) in zip(gap_rows['time_diff'], gap_rows.iterrows())
            ]
        
        # Price anomalies: high < low, or close outside high/low
        price_anomalies = []
        bad_logic = df[
            (df['high'] < df['low']) | 
            (df['close'] > df['high']) | 
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        ]
        for _, row in bad_logic.iterrows():
            price_anomalies.append({
                'timestamp': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })
        
        # Volume anomalies: zero or negative volume
        volume_anomalies = []
        bad_volume = df[df['volume'] <= 0]
        for _, row in bad_volume.iterrows():
            volume_anomalies.append({
                'timestamp': row['timestamp'],
                'volume': row['volume']
            })
        
        # OHLC logic: Open/High/Low/Close should be realistic
        ohlc_failures = []
        for _, row in df.iterrows():
            if not (row['low'] <= row['open'] <= row['high']):
                ohlc_failures.append({
                    'timestamp': row['timestamp'],
                    'issue': 'open outside [low, high]'
                })
            if not (row['low'] <= row['close'] <= row['high']):
                ohlc_failures.append({
                    'timestamp': row['timestamp'],
                    'issue': 'close outside [low, high]'
                })
        
        return DataValidationReport(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            total_trading_days=len(expected_dates),
            missing_days=missing_days,
            gap_hours=gap_hours,
            price_anomalies=price_anomalies,
            volume_anomalies=volume_anomalies,
            ohlc_logic_failures=ohlc_failures
        )

# USAGE EXAMPLE
if __name__ == "__main__":
    # Load your NSE trading calendar (you can get this from NSE website)
    validator = HistoricalDataValidator(
        nse_calendar_path="data/nse_trading_dates.csv"
    )
    
    # Load your historical data
    df = pd.read_csv("data/SBIN_OHLCV_2024.csv")
    
    # Validate
    report = validator.validate_dataframe(
        df=df,
        ticker="SBIN",
        check_intraday_gaps=True
    )
    
    print(report.summary())
    
    # DECISION LOGIC
    if len(report.missing_days) > 5:
        logger.error(f"Too many missing trading days ({len(report.missing_days)})")
        logger.error("DO NOT PROCEED. Investigate data source.")
    
    if len(report.price_anomalies) > 0:
        logger.error(f"Found {len(report.price_anomalies)} price logic failures")
        logger.error("DO NOT PROCEED. Clean data before trading.")
    
    if len(report.volume_anomalies) > len(df) * 0.01:  # >1% bad volume
        logger.error(f"Too many volume anomalies ({len(report.volume_anomalies)})")
        logger.error("DO NOT PROCEED.")
    
    if len(report.gap_hours) > 10:
        logger.warning(f"Found {len(report.gap_hours)} gaps >1 hour")
        logger.warning("This may be normal (system halts, power failures)")
    
    logger.info("✓ Historical data validation passed")
```

#### 26.1.2 Live Data Feed Validation

Once historical data is clean, validate the live data feed connection.

```python
# validation/live_feed_checker.py
from datetime import datetime, time
import pandas as pd
from typing import Optional, Dict, List
import logging
import time as time_module

logger = logging.getLogger(__name__)

class LiveDataFeedValidator:
    """Validate live data feed from Zerodha (or your broker)."""
    
    def __init__(self, broker_connection):
        """
        Args:
            broker_connection: Your KiteConnect object or similar
        """
        self.broker = broker_connection
        self.test_tickers = [
            'NSE:SBIN',      # Liquid large-cap
            'NSE:RELIANCE',  # Another liquid large-cap
            'NSE:INFY'       # Tech large-cap
        ]
    
    def test_connection(self) -> bool:
        """Test basic connection to broker API."""
        try:
            # Try to get instrument list
            instruments = self.broker.instruments()
            logger.info(f"✓ Connected to broker. {len(instruments)} instruments available")
            return True
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False
    
    def test_quote_data(self) -> Dict[str, bool]:
        """Test that quote data is returned correctly."""
        results = {}
        
        for ticker in self.test_tickers:
            try:
                quote = self.broker.quote(ticker)
                
                # Check required fields
                required_fields = ['last_price', 'bid', 'ask', 'volume']
                missing = [f for f in required_fields if f not in quote]
                
                if missing:
                    logger.error(f"✗ {ticker}: Missing fields {missing}")
                    results[ticker] = False
                    continue
                
                # Check data sanity
                if quote['bid'] > quote['ask']:
                    logger.error(f"✗ {ticker}: Bid {quote['bid']} > Ask {quote['ask']}")
                    results[ticker] = False
                    continue
                
                if quote['last_price'] < 0:
                    logger.error(f"✗ {ticker}: Negative price {quote['last_price']}")
                    results[ticker] = False
                    continue
                
                logger.info(
                    f"✓ {ticker}: LTP={quote['last_price']}, "
                    f"Bid={quote['bid']}, Ask={quote['ask']}, Vol={quote['volume']}"
                )
                results[ticker] = True
                
            except Exception as e:
                logger.error(f"✗ {ticker}: Failed to fetch quote - {e}")
                results[ticker] = False
        
        return results
    
    def test_order_placement(self, test_qty: int = 1) -> bool:
        """
        Test order placement WITHOUT executing a real order.
        Uses order validation endpoint if available.
        """
        try:
            # Most brokers have a "validate" endpoint
            # For Zerodha, you can use kite.order_place with a dummy order
            validation_order = {
                'tradingsymbol': 'SBIN',
                'exchange': 'NSE',
                'quantity': test_qty,
                'price': 500,
                'product': 'MIS',  # Intraday
                'order_type': 'LIMIT',
                'validity': 'DAY',
                'variety': 'regular'
            }
            
            # Check if order object is well-formed
            required_keys = ['tradingsymbol', 'quantity', 'price', 'order_type']
            if all(k in validation_order for k in required_keys):
                logger.info("✓ Order structure validation passed")
                return True
            else:
                logger.error("✗ Order structure invalid")
                return False
                
        except Exception as e:
            logger.error(f"✗ Order validation failed: {e}")
            return False
    
    def monitor_latency(self, duration_seconds: int = 60) -> Dict[str, float]:
        """
        Monitor latency of data feed for N seconds.
        Returns: {ticker: avg_latency_ms}
        """
        latencies = {ticker: [] for ticker in self.test_tickers}
        
        logger.info(f"Monitoring latency for {duration_seconds}s...")
        start_time = time_module.time()
        
        while (time_module.time() - start_time) < duration_seconds:
            for ticker in self.test_tickers:
                try:
                    t0 = time_module.time()
                    quote = self.broker.quote(ticker)
                    latency_ms = (time_module.time() - t0) * 1000
                    latencies[ticker].append(latency_ms)
                except:
                    pass
            
            time_module.sleep(1)
        
        # Compute statistics
        results = {}
        for ticker in self.test_tickers:
            if latencies[ticker]:
                avg = sum(latencies[ticker]) / len(latencies[ticker])
                max_lat = max(latencies[ticker])
                results[ticker] = {
                    'avg_ms': avg,
                    'max_ms': max_lat,
                    'samples': len(latencies[ticker])
                }
                logger.info(
                    f"{ticker}: avg={avg:.1f}ms, max={max_lat:.1f}ms, "
                    f"samples={len(latencies[ticker])}"
                )
        
        return results

# VALIDATION CHECKLIST: Data Feed
"""
□ Historical data validator passes on all tickers
□ Missing days: < 5 (acceptable: market holidays)
□ Price anomalies: 0 (High < Low is CRITICAL failure)
□ Volume anomalies: < 1% of records
□ Live connection: establishes without timeout
□ Quote data: all required fields present and sensible
  - Bid < Ask (no crossing spreads)
  - LTP > 0
  - Volume > 0
□ Order structure: validates successfully
□ Latency: < 500ms p95 (adjust based on strategy needs)
"""
```

### Part B: Signal Generation Validation (Day -3)

Your signal generator must work identically in backtesting and live trading. Any difference = potential losses.

```python
# validation/signal_validator.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class SignalValidationResult:
    """Results from signal generation validation."""
    live_signals: List[float]
    backtest_signals: List[float]
    divergence_indices: List[int]
    max_divergence: float
    correlation: float
    validation_passed: bool

class SignalValidator:
    """Validate that live signals match backtest signals."""
    
    def __init__(self, signal_function: Callable):
        """
        Args:
            signal_function: Function that takes DataFrame and returns signal values
                           Expected signature: signal_function(df: pd.DataFrame) -> np.ndarray
        """
        self.signal_fn = signal_function
    
    def validate_on_historical_data(
        self,
        historical_df: pd.DataFrame,
        lookback_bars: int = 100
    ) -> SignalValidationResult:
        """
        Generate signals on historical data in two ways:
        1. Backtest style (full dataframe)
        2. Live style (incremental, one bar at a time)
        
        Compare results to ensure they're identical.
        """
        historical_df = historical_df.copy().reset_index(drop=True)
        
        # Method 1: Backtest style - signal on full dataframe
        try:
            full_signals = self.signal_fn(historical_df)
        except Exception as e:
            logger.error(f"Signal generation failed on full dataframe: {e}")
            return SignalValidationResult(
                live_signals=[],
                backtest_signals=[],
                divergence_indices=[],
                max_divergence=float('inf'),
                correlation=0.0,
                validation_passed=False
            )
        
        # Method 2: Live style - compute signal incrementally
        live_signals = []
        for i in range(lookback_bars, len(historical_df)):
            # Get subset of data up to current bar
            subset = historical_df.iloc[:i+1].copy()
            try:
                signal = self.signal_fn(subset)[-1]  # Last signal
                live_signals.append(signal)
            except Exception as e:
                logger.warning(f"Signal generation failed at bar {i}: {e}")
                live_signals.append(np.nan)
        
        # Align signals (backtest signals start from lookback_bars)
        backtest_aligned = full_signals[lookback_bars:len(historical_df)]
        
        # Find divergences
        divergence_indices = []
        max_div = 0.0
        for i in range(len(live_signals)):
            if not (np.isnan(live_signals[i]) or np.isnan(backtest_aligned[i])):
                diff = abs(live_signals[i] - backtest_aligned[i])
                if diff > 1e-6:  # Floating point tolerance
                    divergence_indices.append(i + lookback_bars)
                    max_div = max(max_div, diff)
        
        # Calculate correlation
        mask = ~(np.isnan(live_signals) | np.isnan(backtest_aligned))
        if mask.sum() > 1:
            corr = np.corrcoef(
                np.array(live_signals)[mask],
                np.array(backtest_aligned)[mask]
            )[0, 1]
        else:
            corr = 0.0
        
        validation_passed = (
            len(divergence_indices) == 0 and 
            corr > 0.99 and
            not np.isnan(corr)
        )
        
        if not validation_passed:
            logger.error(
                f"SIGNAL VALIDATION FAILED:\n"
                f"  Divergences: {len(divergence_indices)}\n"
                f"  Max divergence: {max_div}\n"
                f"  Correlation: {corr:.6f}"
            )
            if divergence_indices[:5]:  # Show first 5
                logger.error(f"  First divergence at index: {divergence_indices[0]}")
        else:
            logger.info(
                f"✓ Signal validation passed (correlation={corr:.6f})"
            )
        
        return SignalValidationResult(
            live_signals=live_signals,
            backtest_signals=backtest_aligned.tolist(),
            divergence_indices=divergence_indices,
            max_divergence=max_div,
            correlation=corr,
            validation_passed=validation_passed
        )

# VALIDATION CHECKLIST: Signal Generation
"""
□ Signal function returns same output on full vs. incremental data
□ Divergence: 0 (if > 0, investigate immediately)
□ Correlation: > 0.999
□ No NaN or infinite values in signals
□ Signal values in expected range (e.g., [-1, 1] for normalized signals)
□ Signal function execution time: < 100ms per bar
"""
```

### Part C: Portfolio Construction Validation (Day -3)

Your position sizing logic must handle edge cases: zero signals, extreme prices, liquidity constraints.

```python
# validation/portfolio_validator.py
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class PortfolioValidationResult:
    """Results from portfolio construction validation."""
    test_cases_passed: List[str]
    test_cases_failed: List[str]
    edge_cases_handled: bool

class PortfolioValidator:
    """Validate portfolio construction logic."""
    
    def __init__(self, position_sizing_fn, max_position_size: float = 0.10):
        """
        Args:
            position_sizing_fn: Function that takes (signals, prices, capital) 
                              and returns position sizes (shares)
            max_position_size: Max % of capital per position
        """
        self.size_fn = position_sizing_fn
        self.max_pos = max_position_size
    
    def validate(self, capital: float = 100000) -> PortfolioValidationResult:
        """Run comprehensive portfolio construction tests."""
        passed = []
        failed = []
        
        # Test 1: Zero signals
        try:
            signals = np.array([0.0, 0.0, 0.0])
            prices = np.array([100.0, 200.0, 150.0])
            positions = self.size_fn(signals, prices, capital)
            assert all(p == 0 for p in positions), "Zero signals should yield zero positions"
            passed.append("Zero signals test")
        except Exception as e:
            failed.append(f"Zero signals test: {e}")
        
        # Test 2: All-in signal
        try:
            signals = np.array([1.0, 1.0, 1.0])
            prices = np.array([100.0, 100.0, 100.0])
            positions = self.size_fn(signals, prices, capital)
            capital_deployed = sum(pos * price for pos, price in zip(positions, prices))
            assert capital_deployed <= capital * 1.01, "Should not exceed capital (1% tolerance)"
            passed.append("All-in signal test")
        except Exception as e:
            failed.append(f"All-in signal test: {e}")
        
        # Test 3: Extreme prices
        try:
            signals = np.array([1.0, 1.0])
            prices = np.array([1.0, 10000.0])  # Large spread
            positions = self.size_fn(signals, prices, capital)
            capital_deployed = sum(pos * price for pos, price in zip(positions, prices))
            assert capital_deployed <= capital * 1.01, "Should handle extreme prices"
            passed.append("Extreme price test")
        except Exception as e:
            failed.append(f"Extreme price test: {e}")
        
        # Test 4: Negative signals (short positions)
        try:
            signals = np.array([-0.5, 0.5, 0.0])
            prices = np.array([100.0, 100.0, 100.0])
            positions = self.size_fn(signals, prices, capital)
            # Shorts should be negative, longs positive
            assert positions[0] < 0, "Negative signal should give short position"
            assert positions[1] > 0, "Positive signal should give long position"
            assert positions[2] == 0, "Zero signal should give zero position"
            passed.append("Short position test")
        except Exception as e:
            failed.append(f"Short position test: {e}")
        
        # Test 5: Risk limits
        try:
            signals = np.array([1.0, 1.0, 1.0, 1.0])
            prices = np.array([100.0, 100.0, 100.0, 100.0])
            positions = self.size_fn(signals, prices, capital)
            for pos, price in zip(positions, prices):
                notional = abs(pos * price)
                pct_capital = notional / capital
                assert pct_capital <= self.max_pos, f"Position exceeds {self.max_pos*100}% limit"
            passed.append("Risk limits test")
        except Exception as e:
            failed.append(f"Risk limits test: {e}")
        
        # Test 6: NaN/Inf handling
        try:
            signals = np.array([1.0, np.nan, 1.0])
            prices = np.array([100.0, 100.0, 100.0])
            positions = self.size_fn(signals, prices, capital)
            assert not any(np.isnan(p) or np.isinf(p) for p in positions), \
                "Output should not contain NaN or Inf"
            passed.append("NaN/Inf handling test")
        except Exception as e:
            failed.append(f"NaN/Inf handling test: {e}")
        
        success = len(failed) == 0
        
        if success:
            logger.info(f"✓ Portfolio validation passed ({len(passed)} tests)")
        else:
            logger.error(f"✗ Portfolio validation failed: {len(failed)} tests failed")
            for f in failed:
                logger.error(f"  - {f}")
        
        return PortfolioValidationResult(
            test_cases_passed=passed,
            test_cases_failed=failed,
            edge_cases_handled=success
        )

# VALIDATION CHECKLIST: Portfolio Construction
"""
□ Zero signals → zero positions
□ All-in signals → capital fully deployed (±1%)
□ Extreme prices handled without errors
□ Short positions supported correctly
□ No position exceeds max_position_size
□ NaN/Inf in signals handled gracefully
□ Position sizes are integer (or properly fractional)
□ Execution time: < 50ms per portfolio update
"""
```

### Part D: Execution Risk Checks (Day -2)

Before placing real orders, verify kill switches, position limits, and margin calculations.

```python
# validation/execution_validator.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExecutionRiskCheckResult:
    """Results from execution risk validation."""
    kill_switch_working: bool
    max_position_enforced: bool
    max_daily_loss_enforced: bool
    margin_calculated_correctly: bool
    slippage_model_valid: bool
    all_checks_passed: bool
    issues: List[str]

class ExecutionValidator:
    """Validate execution risk controls."""
    
    def __init__(
        self,
        broker,
        max_position_notional: float,
        max_daily_loss_pct: float = 0.02,  # 2% of capital
        slippage_fn=None
    ):
        """
        Args:
            broker: Broker connection object
            max_position_notional: Max notional value per position
            max_daily_loss_pct: Max daily loss as % of capital
            slippage_fn: Function to estimate slippage
        """
        self.broker = broker
        self.max_pos = max_position_notional
        self.max_loss = max_daily_loss_pct
        self.slippage_fn = slippage_fn or (lambda qty, price: price * 0.001)  # Default: 0.1%
        self.issues = []
    
    def test_kill_switch(self) -> bool:
        """Test that kill switch can be activated."""
        try:
            # Simulate kill switch
            self.kill_switch_active = True
            
            # Verify that orders would NOT be placed when kill switch is active
            can_trade = not self.kill_switch_active
            assert not can_trade, "Kill switch should prevent trading"
            
            self.kill_switch_active = False
            can_trade = not self.kill_switch_active
            assert can_trade, "Kill switch should allow trading when disabled"
            
            logger.info("✓ Kill switch test passed")
            return True
        except Exception as e:
            logger.error(f"✗ Kill switch test failed: {e}")
            self.issues.append(f"Kill switch: {e}")
            return False
    
    def test_position_limits(self) -> bool:
        """Test that position size limits are enforced."""
        try:
            # Test: Attempt to place position larger than max
            test_position_size = self.max_pos * 1.5  # 150% of max
            test_price = 100.0
            
            shares = test_position_size / test_price
            notional = shares * test_price
            
            # Should be rejected
            assert notional > self.max_pos, "Setup error"
            logger.info(f"✓ Position limit test passed (rejected {notional} > {self.max_pos})")
            return True
        except Exception as e:
            logger.error(f"✗ Position limit test failed: {e}")
            self.issues.append(f"Position limits: {e}")
            return False
    
    def test_daily_loss_limit(self) -> bool:
        """Test that daily loss limit is enforced."""
        try:
            capital = 100000
            max_daily_loss = capital * self.max_loss
            
            # Simulate losing more than max
            current_pnl = -max_daily_loss * 1.5
            
            should_stop = current_pnl < -max_daily_loss
            assert should_stop, "Should stop trading after max daily loss"
            
            logger.info(
                f"✓ Daily loss limit test passed "
                f"(max loss: {max_daily_loss}, current: {current_pnl})"
            )
            return True
        except Exception as e:
            logger.error(f"✗ Daily loss limit test failed: {e}")
            self.issues.append(f"Daily loss limit: {e}")
            return False
    
    def test_margin_calculation(self) -> bool:
        """Test that margin requirements are calculated correctly."""
        try:
            # For NSE: typical margin is 10-20% of notional
            # This depends on the instrument
            
            position_notional = 50000
            required_margin = position_notional * 0.15  # 15% margin
            available_capital = 100000
            
            can_execute = required_margin <= available_capital
            assert can_execute, "Should check margin before execution"
            
            logger.info(
                f"✓ Margin calculation test passed "
                f"(required: {required_margin}, available: {available_capital})"
            )
            return True
        except Exception as e:
            logger.error(f"✗ Margin calculation test failed: {e}")
            self.issues.append(f"Margin calculation: {e}")
            return False
    
    def test_slippage_model(self) -> bool:
        """Test that slippage model produces reasonable estimates."""
        try:
            # Test various order sizes
            test_cases = [
                (100, 500),      # Small order
                (1000, 500),     # Medium order
                (10000, 500),    # Large order
            ]
            
            for qty, price in test_cases:
                slippage = self.slippage_fn(qty, price)
                
                # Slippage should be:
                # - Positive
                # - Less than price (reasonable)
                # - Monotonically increasing with quantity
                assert slippage > 0, f"Slippage should be positive for qty={qty}"
                assert slippage < price * 0.10, f"Slippage > 10% is unreasonable"
            
            logger.info("✓ Slippage model test passed")
            return True
        except Exception as e:
            logger.error(f"✗ Slippage model test failed: {e}")
            self.issues.append(f"Slippage model: {e}")
            return False
    
    def validate_all(self) -> ExecutionRiskCheckResult:
        """Run all execution validation tests."""
        logger.info("\n" + "="*70)
        logger.info("EXECUTION VALIDATION")
        logger.info("="*70)
        
        results = {
            'kill_switch': self.test_kill_switch(),
            'position_limits': self.test_position_limits(),
            'daily_loss_limit': self.test_daily_loss_limit(),
            'margin': self.test_margin_calculation(),
            'slippage': self.test_slippage_model(),
        }
        
        all_passed = all(results.values())
        
        logger.info("="*70)
        if all_passed:
            logger.info("✓ ALL EXECUTION CHECKS PASSED")
        else:
            logger.error(f"✗ EXECUTION VALIDATION FAILED: {sum(not v for v in results.values())} failures")
        logger.info("="*70 + "\n")
        
        return ExecutionRiskCheckResult(
            kill_switch_working=results['kill_switch'],
            max_position_enforced=results['position_limits'],
            max_daily_loss_enforced=results['daily_loss_limit'],
            margin_calculated_correctly=results['margin'],
            slippage_model_valid=results['slippage'],
            all_checks_passed=all_passed,
            issues=self.issues
        )

# VALIDATION CHECKLIST: Execution Risk
"""
□ Kill switch: can be activated and deactivates trading
□ Kill switch: position exit on activation
□ Max position size: enforced on every order
□ Max daily loss: enforced and triggers stop
□ Margin calculation: correct for account type
□ Margin buffer: maintained (e.g., use only 80% of available margin)
□ Slippage model: reasonable for order sizes
□ Order types: correct for market conditions (limit vs. market)
"""
```

### Part E: System Health and Backup Testing (Day -1)

Test failure scenarios: API disconnection, data feed loss, system crash.

```python
# validation/backup_validator.py
import logging
from typing import Dict, List
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class BackupValidator:
    """Validate backup and recovery mechanisms."""
    
    def __init__(self, backup_dir: str, state_file: str):
        """
        Args:
            backup_dir: Directory for backups
            state_file: Path to persistent state file
        """
        self.backup_dir = backup_dir
        self.state_file = state_file
    
    def test_state_persistence(self) -> bool:
        """Test that system state can be saved and restored."""
        try:
            # Create test state
            test_state = {
                'timestamp': datetime.now().isoformat(),
                'capital': 100000,
                'positions': {'SBIN': 100, 'INFY': 50},
                'pnl': 5000,
                'last_signal_bar': 1234
            }
            
            # Save state
            with open(self.state_file, 'w') as f:
                json.dump(test_state, f)
            
            # Restore state
            with open(self.state_file, 'r') as f:
                restored_state = json.load(f)
            
            # Verify
            assert restored_state == test_state, "State mismatch after restore"
            logger.info("✓ State persistence test passed")
            return True
        except Exception as e:
            logger.error(f"✗ State persistence test failed: {e}")
            return False
    
    def test_backup_creation(self) -> bool:
        """Test that backups can be created."""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            
            # Create test backup
            backup_file = os.path.join(
                self.backup_dir,
                f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'test': True
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f)
            
            # Verify backup exists and is readable
            assert os.path.exists(backup_file), "Backup file not created"
            with open(backup_file, 'r') as f:
                restored = json.load(f)
            assert restored['test'], "Backup data corrupted"
            
            logger.info(f"✓ Backup creation test passed ({backup_file})")
            return True
        except Exception as e:
            logger.error(f"✗ Backup creation test failed: {e}")
            return False
    
    def test_recovery_procedure(self) -> bool:
        """Test disaster recovery procedure."""
        try:
            # Simulate recovery: restore from backup
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    recovered_state = json.load(f)
                logger.info(
                    f"✓ Recovery test passed "
                    f"(recovered {recovered_state.get('capital')} capital)"
                )
                return True
            else:
                logger.warning("No state file to recover from (first run)")
                return True
        except Exception as e:
            logger.error(f"✗ Recovery test failed: {e}")
            return False

# VALIDATION CHECKLIST: Backup and Recovery
"""
□ State file created on startup
□ State file updated hourly (or after every trade)
□ Backups created daily
□ Recovery procedure: tested and time-measured
□ Recovery time: < 60 seconds (time from crash to trading)
□ State file: human-readable (JSON, CSV acceptable)
"""
```

### Part F: The Master Pre-Launch Checklist

```python
# validation/master_checklist.py
from dataclasses import dataclass
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreLaunchChecklistItem:
    """Single checklist item."""
    category: str
    item: str
    completed: bool
    notes: str = ""

class MasterPreLaunchChecklist:
    """Master checklist before going live."""
    
    def __init__(self):
        self.items: Dict[str, List[PreLaunchChecklistItem]] = {
            'Data Feed': [
                PreLaunchChecklistItem('Data Feed', 'Historical data validated', False),
                PreLaunchChecklistItem('Data Feed', 'Missing days < 5', False),
                PreLaunchChecklistItem('Data Feed', 'Price anomalies = 0', False),
                PreLaunchChecklistItem('Data Feed', 'Live connection stable', False),
                PreLaunchChecklistItem('Data Feed', 'Latency < 500ms p95', False),
            ],
            'Signal Generation': [
                PreLaunchChecklistItem('Signal Generation', 'Signals match backtest', False),
                PreLaunchChecklistItem('Signal Generation', 'Correlation > 0.999', False),
                PreLaunchChecklistItem('Signal Generation', 'No NaN/Inf in signals', False),
                PreLaunchChecklistItem('Signal Generation', 'Execution time < 100ms', False),
            ],
            'Portfolio Construction': [
                PreLaunchChecklistItem('Portfolio Construction', 'Zero signals handled', False),
                PreLaunchChecklistItem('Portfolio Construction', 'All edge cases tested', False),
                PreLaunchChecklistItem('Portfolio Construction', 'Position limits enforced', False),
            ],
            'Execution & Risk': [
                PreLaunchChecklistItem('Execution & Risk', 'Kill switch operational', False),
                PreLaunchChecklistItem('Execution & Risk', 'Max position enforced', False),
                PreLaunchChecklistItem('Execution & Risk', 'Max daily loss enforced', False),
                PreLaunchChecklistItem('Execution & Risk', 'Margin check active', False),
                PreLaunchChecklistItem('Execution & Risk', 'Slippage model valid', False),
            ],
            'Backup & Recovery': [
                PreLaunchChecklistItem('Backup & Recovery', 'State persistence works', False),
                PreLaunchChecklistItem('Backup & Recovery', 'Backups created daily', False),
                PreLaunchChecklistItem('Backup & Recovery', 'Recovery < 60 seconds', False),
            ],
            'Initial Capital': [
                PreLaunchChecklistItem('Initial Capital', 'Capital allocation decided', False,
                                     'Start small: 20-30% of total capital'),
                PreLaunchChecklistItem('Initial Capital', 'Account paperwork complete', False),
                PreLaunchChecklistItem('Initial Capital', 'API keys configured', False),
                PreLaunchChecklistItem('Initial Capital', 'Broker margin verified', False),
            ],
            'Expectations': [
                PreLaunchChecklistItem('Expectations', 'First month: learning phase', False),
                PreLaunchChecklistItem('Expectations', 'Expect 20%+ slippage vs backtest', False),
                PreLaunchChecklistItem('Expectations', 'Don\'t trade on emotional signals', False),
                PreLaunchChecklistItem('Expectations', 'Document every trade', False),
            ]
        }
    
    def mark_complete(self, category: str, item: str, notes: str = ""):
        """Mark a checklist item as complete."""
        for c in self.items[category]:
            if c.item == item:
                c.completed = True
                c.notes = notes
                break
    
    def print_checklist(self):
        """Print current checklist status."""
        total = sum(len(items) for items in self.items.values())
        completed = sum(
            sum(1 for item in items if item.completed)
            for items in self.items.values()
        )
        
        logger.info("\n" + "="*70)
        logger.info(f"PRE-LAUNCH CHECKLIST: {completed}/{total} items")
        logger.info("="*70)
        
        for category, items in self.items.items():
            category_complete = sum(1 for i in items if i.completed)
            logger.info(f"\n{category} ({category_complete}/{len(items)})")
            for item in items:
                status = "✓" if item.completed else "□"
                logger.info(f"  {status} {item.item}")
                if item.notes:
                    logger.info(f"     → {item.notes}")
        
        logger.info("\n" + "="*70)
        if completed == total:
            logger.info("🚀 READY TO GO LIVE")
        else:
            logger.info(f"⚠ {total - completed} items remaining")
        logger.info("="*70 + "\n")
    
    def can_go_live(self) -> bool:
        """Check if all critical items are completed."""
        required_categories = ['Data Feed', 'Signal Generation', 'Execution & Risk', 'Initial Capital']
        
        for category in required_categories:
            items = self.items[category]
            completed = sum(1 for i in items if i.completed)
            if completed < len(items):
                logger.error(f"✗ {category}: not all items complete")
                return False
        
        return True

# USAGE
if __name__ == "__main__":
    checklist = MasterPreLaunchChecklist()
    
    # Simulate completion
    checklist.mark_complete('Data Feed', 'Historical data validated')
    checklist.mark_complete('Signal Generation', 'Signals match backtest')
    
    checklist.print_checklist()
    
    if checklist.can_go_live():
        logger.info("All systems green. Proceed with caution.")
```

---

## Module 26.2: Day-by-Day Operations — The First 30 Days Playbook

### Introduction to Operations

The first 30 days are your "learning under live conditions" phase. You're not trying to maximize profits—you're gathering data about:
- **How your model performs with real slippage and fees**
- **What edge cases your code doesn't handle**
- **How you emotionally respond to real losses**
- **What your actual correlation structure looks like**

This section is a day-by-day operational playbook.

### Pre-Market Routine (Before 9:15 AM IST)

Every trading day, before the market opens, run this routine.

```python
# operations/pre_market_routine.py
from datetime import datetime, time
import pandas as pd
import logging
from typing import Dict, List, Tuple
import os
import json

logger = logging.getLogger(__name__)

class PreMarketRoutine:
    """Pre-market health checks and preparation."""
    
    def __init__(
        self,
        state_file: str,
        data_dir: str,
        broker_connection,
        risk_params: Dict
    ):
        """
        Args:
            state_file: Path to persistent state
            data_dir: Directory with market data
            broker_connection: Zerodha KiteConnect object
            risk_params: Risk parameters (max daily loss, etc)
        """
        self.state_file = state_file
        self.data_dir = data_dir
        self.broker = broker_connection
        self.risk_params = risk_params
        self.routine_log = []
    
    def check_time(self) -> bool:
        """Verify it's a trading day and before market open (9:15 AM)."""
        now = datetime.now()
        
        # NSE opens at 9:15 AM, closes at 3:30 PM
        opening_time = time(9, 15)
        
        if now.time() >= opening_time:
            logger.error("✗ Pre-market routine running after market open!")
            return False
        
        logger.info(f"✓ Time check passed ({now.strftime('%Y-%m-%d %H:%M:%S')})")
        return True
    
    def check_system_health(self) -> bool:
        """Check that all systems are operational."""
        checks = {
            'broker_connection': False,
            'data_available': False,
            'state_file_readable': False,
            'disk_space_available': False
        }
        
        # Check broker connection
        try:
            # Simple health check: can we get current positions?
            positions = self.broker.positions()
            checks['broker_connection'] = True
            logger.info(f"✓ Broker connection active ({len(positions)} positions)")
        except Exception as e:
            logger.error(f"✗ Broker connection failed: {e}")
        
        # Check data files
        try:
            files = os.listdir(self.data_dir)
            if len(files) > 0:
                checks['data_available'] = True
                logger.info(f"✓ Data directory accessible ({len(files)} files)")
        except Exception as e:
            logger.error(f"✗ Data directory not accessible: {e}")
        
        # Check state file
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            checks['state_file_readable'] = True
            logger.info(f"✓ State file readable (P&L: {state.get('pnl', 0):.2f})")
        except Exception as e:
            logger.error(f"✗ State file not readable: {e}")
        
        # Check disk space
        try:
            import shutil
            stat = shutil.disk_usage('/')
            available_gb = stat.free / (1024**3)
            if available_gb > 1.0:
                checks['disk_space_available'] = True
                logger.info(f"✓ Disk space available ({available_gb:.1f} GB free)")
            else:
                logger.warning(f"⚠ Low disk space ({available_gb:.1f} GB free)")
        except Exception as e:
            logger.error(f"✗ Disk space check failed: {e}")
        
        return all(checks.values())
    
    def validate_data_freshness(self) -> bool:
        """Check that market data is up to date."""
        try:
            # Load latest OHLC file
            latest_file = max(
                [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)],
                key=os.path.getctime
            )
            
            df = pd.read_csv(latest_file)
            last_bar_time = pd.to_datetime(df['timestamp'].iloc[-1])
            
            # Calculate hours since last bar
            hours_ago = (datetime.now() - last_bar_time).total_seconds() / 3600
            
            if hours_ago < 24:
                logger.info(f"✓ Data fresh ({hours_ago:.1f} hours old)")
                return True
            else:
                logger.warning(f"⚠ Data stale ({hours_ago:.1f} hours old)")
                return False
        except Exception as e:
            logger.error(f"✗ Data freshness check failed: {e}")
            return False
    
    def review_positions(self) -> Dict:
        """Load and review current positions."""
        try:
            positions = self.broker.positions()
            
            logger.info(f"\nCurrent Positions ({len(positions)}):")
            logger.info("-" * 50)
            
            for pos in positions:
                notional = pos['quantity'] * pos['last_price']
                logger.info(
                    f"  {pos['symbol']:<15} {pos['quantity']:>6.0f} @ {pos['last_price']:>8.2f} "
                    f"(notional: {notional:>10.0f})"
                )
            
            logger.info("-" * 50)
            return {'count': len(positions), 'positions': positions}
        except Exception as e:
            logger.error(f"✗ Position review failed: {e}")
            return {'count': 0, 'positions': []}
    
    def check_daily_loss_limit(self) -> Tuple[bool, float]:
        """Check if yesterday's P&L exceeded daily loss limit."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            daily_pnl = state.get('daily_pnl', 0)
            max_daily_loss = self.risk_params['max_daily_loss_pct'] * state.get('capital', 100000)
            
            if daily_pnl < -max_daily_loss:
                logger.error(
                    f"✗ Daily loss limit exceeded: {daily_pnl:.2f} < {-max_daily_loss:.2f}"
                )
                return False, daily_pnl
            else:
                logger.info(f"✓ Daily loss limit OK (today: {daily_pnl:.2f})")
                return True, daily_pnl
        except Exception as e:
            logger.error(f"✗ Daily loss check failed: {e}")
            return False, 0
    
    def reset_daily_counters(self):
        """Reset daily counters (trades, loss tracking)."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            state['daily_pnl'] = 0
            state['trades_today'] = 0
            state['last_reset'] = datetime.now().isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            
            logger.info("✓ Daily counters reset")
        except Exception as e:
            logger.error(f"✗ Counter reset failed: {e}")
    
    def run_full_routine(self) -> Dict:
        """Run complete pre-market routine."""
        logger.info("\n" + "="*70)
        logger.info("PRE-MARKET ROUTINE")
        logger.info("="*70)
        
        results = {
            'time_check': self.check_time(),
            'system_health': self.check_system_health(),
            'data_freshness': self.validate_data_freshness(),
            'positions': self.review_positions(),
            'daily_limit': self.check_daily_loss_limit()[0],
        }
        
        if all([results['time_check'], results['system_health'], results['daily_limit']]):
            self.reset_daily_counters()
            logger.info("\n✓ PRE-MARKET ROUTINE PASSED - READY TO TRADE\n")
        else:
            logger.error("\n✗ PRE-MARKET CHECKS FAILED - DO NOT TRADE\n")
        
        return results
```

### During-Market Routine (9:15 AM - 3:30 PM IST)

During market hours, monitor positions and decide when to intervene.

```python
# operations/market_monitoring.py
from datetime import datetime, time
from typing import Dict, List, Optional
import logging
import json
import pandas as pd

logger = logging.getLogger(__name__)

class MarketMonitor:
    """Monitor positions and P&L during market hours."""
    
    def __init__(
        self,
        state_file: str,
        broker_connection,
        risk_params: Dict
    ):
        self.state_file = state_file
        self.broker = broker_connection
        self.risk_params = risk_params
        self.last_portfolio_value = None
    
    def get_current_portfolio_value(self) -> float:
        """Calculate current portfolio value (cash + positions)."""
        try:
            # Get account balance
            margins = self.broker.margins()
            available_cash = margins['cash'][0]['available']
            
            # Get position values
            positions = self.broker.positions()
            position_value = sum(
                pos['quantity'] * pos['last_price']
                for pos in positions
            )
            
            total_value = available_cash + position_value
            return total_value
        except Exception as e:
            logger.warning(f"Could not calculate portfolio value: {e}")
            return 0
    
    def check_intraday_loss_limit(self) -> Dict:
        """Check if intraday loss exceeds threshold."""
        try:
            current_value = self.get_current_portfolio_value()
            
            if self.last_portfolio_value is None:
                self.last_portfolio_value = current_value
                return {
                    'check_passed': True,
                    'intraday_loss': 0,
                    'loss_pct': 0
                }
            
            intraday_loss = current_value - self.last_portfolio_value
            loss_pct = intraday_loss / self.last_portfolio_value
            
            # Get thresholds from state
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            max_intraday_loss = state['capital'] * self.risk_params['max_intraday_loss_pct']
            
            if intraday_loss < -max_intraday_loss:
                logger.error(
                    f"✗ INTRADAY LOSS LIMIT EXCEEDED: {intraday_loss:.2f} < {-max_intraday_loss:.2f}"
                )
                return {
                    'check_passed': False,
                    'intraday_loss': intraday_loss,
                    'loss_pct': loss_pct
                }
            elif loss_pct < -0.02:  # Lost 2%+ intraday
                logger.warning(f"⚠ Large intraday loss: {loss_pct*100:.1f}%")
            
            return {
                'check_passed': True,
                'intraday_loss': intraday_loss,
                'loss_pct': loss_pct
            }
        except Exception as e:
            logger.error(f"Could not check intraday loss: {e}")
            return {'check_passed': True, 'intraday_loss': 0, 'loss_pct': 0}
    
    def get_intervention_criteria(self) -> Dict[str, bool]:
        """
        Decide whether human intervention is needed.
        Returns dict of criteria that trigger intervention.
        """
        criteria = {}
        
        # 1. Intraday loss exceeded
        loss_check = self.check_intraday_loss_limit()
        criteria['large_loss'] = not loss_check['check_passed']
        if loss_check['check_passed']:
            criteria['loss_pct'] = loss_check['loss_pct']
        
        # 2. Position concentration
        try:
            positions = self.broker.positions()
            if len(positions) > 0:
                position_values = [pos['quantity'] * pos['last_price'] for pos in positions]
                total_value = sum(position_values)
                max_position_pct = max(position_values) / total_value if total_value > 0 else 0
                
                criteria['concentrated'] = max_position_pct > 0.5  # >50% in one position
                criteria['max_position_pct'] = max_position_pct
        except:
            criteria['concentrated'] = False
        
        # 3. Unusual market movement
        try:
            positions = self.broker.positions()
            for pos in positions:
                # Check if position moved >5% since open
                change_pct = abs(pos['last_price'] - pos['open']) / pos['open']
                if change_pct > 0.05:
                    criteria['large_move'] = True
                    criteria['move_pct'] = change_pct
                    break
            else:
                criteria['large_move'] = False
        except:
            criteria['large_move'] = False
        
        return criteria
    
    def print_monitoring_summary(self):
        """Print real-time monitoring summary."""
        try:
            portfolio_value = self.get_current_portfolio_value()
            positions = self.broker.positions()
            margins = self.broker.margins()
            
            logger.info("\n" + "-"*70)
            logger.info(f"Portfolio Value: {portfolio_value:,.0f}")
            logger.info(f"Available Margin: {margins['cash'][0]['available']:,.0f}")
            logger.info(f"Positions: {len(positions)}")
            
            loss_check = self.check_intraday_loss_limit()
            if loss_check['check_passed']:
                logger.info(f"Intraday P&L: {loss_check['intraday_loss']:+.0f}")
            else:
                logger.warning(f"⚠ Intraday Loss: {loss_check['intraday_loss']:+.0f}")
            
            intervention = self.get_intervention_criteria()
            if any(intervention.get(k) for k in ['large_loss', 'concentrated', 'large_move']):
                logger.warning("⚠ INTERVENTION CRITERIA MET")
                for k, v in intervention.items():
                    if v and k not in ['loss_pct', 'max_position_pct', 'move_pct']:
                        logger.warning(f"  - {k}")
            
            logger.info("-"*70 + "\n")
        except Exception as e:
            logger.error(f"Could not print monitoring summary: {e}")
    
    def should_halt_trading(self) -> bool:
        """Check if trading should be halted."""
        intervention = self.get_intervention_criteria()
        return intervention.get('large_loss', False)
```

### Post-Market Routine (After 3:30 PM IST)

After market close, reconcile trades and prepare for next day.

```python
# operations/post_market_routine.py
from datetime import datetime
import json
import pandas as pd
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class PostMarketRoutine:
    """Post-market reconciliation and analysis."""
    
    def __init__(
        self,
        state_file: str,
        broker_connection,
        trade_journal_file: str
    ):
        self.state_file = state_file
        self.broker = broker_connection
        self.trade_journal = trade_journal_file
    
    def reconcile_trades(self) -> Dict:
        """
        Reconcile executed trades.
        Compare broker records with internal expectations.
        """
        try:
            # Get today's trades from broker
            trades = self.broker.trades()
            today_trades = [
                t for t in trades
                if datetime.fromisoformat(t['tradetime']).date() == datetime.now().date()
            ]
            
            logger.info(f"\nTrades Today: {len(today_trades)}")
            logger.info("-" * 70)
            
            for trade in today_trades:
                logger.info(
                    f"  {trade['tradetime']:<20} {trade['tradingsymbol']:<12} "
                    f"{trade['quantity']:>6.0f} @ {trade['price']:>8.2f} "
                    f"({trade['side']:<4}) - Fee: {trade.get('fee', 0):>6.1f}"
                )
            
            logger.info("-" * 70)
            
            # Calculate realized P&L
            realized_pnl = sum(trade.get('realised_profit', 0) for trade in today_trades)
            total_fees = sum(trade.get('fee', 0) for trade in today_trades)
            
            logger.info(f"Realized P&L: {realized_pnl:+.0f}")
            logger.info(f"Total Fees: {total_fees:+.0f}")
            logger.info(f"Net: {realized_pnl - total_fees:+.0f}")
            
            return {
                'trade_count': len(today_trades),
                'realized_pnl': realized_pnl,
                'fees': total_fees,
                'net': realized_pnl - total_fees,
                'trades': today_trades
            }
        except Exception as e:
            logger.error(f"Trade reconciliation failed: {e}")
            return {'trade_count': 0, 'realized_pnl': 0, 'fees': 0, 'net': 0, 'trades': []}
    
    def calculate_unrealized_pnl(self) -> Dict:
        """Calculate P&L on open positions."""
        try:
            positions = self.broker.positions()
            
            unrealized_pnl = 0
            for pos in positions:
                entry_value = pos['buy_m2m']  # Mark-to-market
                unrealized_pnl += entry_value
            
            logger.info(f"Unrealized P&L: {unrealized_pnl:+.0f}")
            
            return {
                'unrealized_pnl': unrealized_pnl,
                'position_count': len(positions)
            }
        except Exception as e:
            logger.error(f"Could not calculate unrealized P&L: {e}")
            return {'unrealized_pnl': 0, 'position_count': 0}
    
    def update_state_file(self, daily_metrics: Dict):
        """Update persistent state with today's results."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            state['last_updated'] = datetime.now().isoformat()
            state['daily_pnl'] = daily_metrics.get('net', 0)
            state['cumulative_pnl'] = state.get('cumulative_pnl', 0) + daily_metrics.get('net', 0)
            state['trades_today'] = daily_metrics.get('trade_count', 0)
            state['unrealized_pnl'] = daily_metrics.get('unrealized_pnl', 0)
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"✓ State file updated")
        except Exception as e:
            logger.error(f"Could not update state file: {e}")
    
    def write_trade_journal(self, daily_metrics: Dict):
        """
        Write daily summary to trade journal.
        This is your learning log.
        """
        try:
            entry = {
                'date': datetime.now().date().isoformat(),
                'time': datetime.now().isoformat(),
                'trades': daily_metrics.get('trade_count', 0),
                'realized_pnl': daily_metrics.get('realized_pnl', 0),
                'fees': daily_metrics.get('fees', 0),
                'net': daily_metrics.get('net', 0),
                'unrealized_pnl': daily_metrics.get('unrealized_pnl', 0),
                'notes': ''  # User fills this in
            }
            
            # Append to journal
            try:
                df = pd.read_csv(self.trade_journal)
            except:
                df = pd.DataFrame()
            
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(self.trade_journal, index=False)
            
            logger.info(f"✓ Trade journal updated")
        except Exception as e:
            logger.error(f"Could not update trade journal: {e}")
    
    def run_full_routine(self):
        """Run complete post-market routine."""
        logger.info("\n" + "="*70)
        logger.info("POST-MARKET ROUTINE")
        logger.info("="*70)
        
        trades = self.reconcile_trades()
        unrealized = self.calculate_unrealized_pnl()
        
        daily_metrics = {**trades, **unrealized}
        
        self.update_state_file(daily_metrics)
        self.write_trade_journal(daily_metrics)
        
        logger.info("\n✓ POST-MARKET ROUTINE COMPLETE\n")
        
        return daily_metrics
```

### Weekly Routine (Friday Post-Market)

Every Friday after market close, run deeper analysis.

```python
# operations/weekly_analysis.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class WeeklyAnalysis:
    """Weekly performance and risk analysis."""
    
    def __init__(self, trade_journal_file: str):
        self.journal = trade_journal_file
    
    def calculate_weekly_metrics(self) -> Dict:
        """Calculate key metrics for the past week."""
        try:
            df = pd.read_csv(self.journal, parse_dates=['date'])
            
            # Filter to past 5 trading days
            five_days_ago = datetime.now().date() - timedelta(days=7)
            weekly = df[df['date'] >= five_days_ago]
            
            if len(weekly) == 0:
                logger.warning("No trades this week")
                return {}
            
            metrics = {
                'trading_days': len(weekly),
                'total_trades': weekly['trades'].sum(),
                'avg_trades_per_day': weekly['trades'].mean(),
                'total_pnl': weekly['net'].sum(),
                'win_days': len(weekly[weekly['net'] > 0]),
                'loss_days': len(weekly[weekly['net'] < 0]),
                'win_rate': len(weekly[weekly['net'] > 0]) / len(weekly) if len(weekly) > 0 else 0,
                'avg_win': weekly[weekly['net'] > 0]['net'].mean() if len(weekly[weekly['net'] > 0]) > 0 else 0,
                'avg_loss': weekly[weekly['net'] < 0]['net'].mean() if len(weekly[weekly['net'] < 0]) > 0 else 0,
                'profit_factor': abs(weekly[weekly['net'] > 0]['net'].sum() / weekly[weekly['net'] < 0]['net'].sum())
                if len(weekly[weekly['net'] < 0]) > 0 and weekly[weekly['net'] < 0]['net'].sum() != 0 else 0,
                'total_fees': weekly['fees'].sum(),
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Could not calculate weekly metrics: {e}")
            return {}
    
    def check_signal_decay(self) -> Dict:
        """
        Monitor if signal quality is degrading.
        Signals decay when model assumptions break down.
        """
        try:
            df = pd.read_csv(self.journal, parse_dates=['date'])
            
            # Divide into two halves (first half of month vs. recent)
            mid = len(df) // 2
            first_half = df.iloc[:mid]
            second_half = df.iloc[mid:]
            
            first_pnl = first_half['net'].mean()
            second_pnl = second_half['net'].mean()
            
            decay = (second_pnl - first_pnl) / abs(first_pnl) if first_pnl != 0 else 0
            
            logger.info(f"\nSignal Decay Analysis:")
            logger.info(f"  First half avg P&L: {first_pnl:+.0f}")
            logger.info(f"  Second half avg P&L: {second_pnl:+.0f}")
            logger.info(f"  Decay: {decay:+.1%}")
            
            if decay < -0.2:
                logger.warning("⚠ SIGNIFICANT SIGNAL DECAY DETECTED")
                logger.warning("  Consider retraining or parameter adjustment")
            
            return {'decay_pct': decay, 'alert': decay < -0.2}
        except Exception as e:
            logger.error(f"Could not analyze signal decay: {e}")
            return {'decay_pct': 0, 'alert': False}
    
    def print_weekly_report(self):
        """Print formatted weekly report."""
        metrics = self.calculate_weekly_metrics()
        decay = self.check_signal_decay()
        
        if not metrics:
            return
        
        logger.info("\n" + "="*70)
        logger.info("WEEKLY PERFORMANCE REPORT")
        logger.info("="*70)
        logger.info(f"Trading Days: {metrics.get('trading_days', 0)}")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Avg Trades/Day: {metrics.get('avg_trades_per_day', 0):.1f}")
        logger.info(f"\nP&L Summary:")
        logger.info(f"  Total: {metrics.get('total_pnl', 0):+,.0f}")
        logger.info(f"  Fees: {metrics.get('total_fees', 0):+,.0f}")
        logger.info(f"\nWin Rate:")
        logger.info(f"  Winning Days: {metrics.get('win_days', 0)}/{metrics.get('trading_days', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
        logger.info(f"  Avg Win: {metrics.get('avg_win', 0):+,.0f}")
        logger.info(f"  Avg Loss: {metrics.get('avg_loss', 0):+,.0f}")
        logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info("="*70 + "\n")
        
        return metrics
```

### Monthly Routine (First of next month)

Comprehensive monthly review and strategy recalibration.

```python
# operations/monthly_review.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MonthlyReview:
    """Monthly comprehensive review and strategy check."""
    
    def __init__(
        self,
        trade_journal_file: str,
        state_file: str
    ):
        self.journal = trade_journal_file
        self.state_file = state_file
    
    def calculate_monthly_statistics(self) -> Dict:
        """Calculate monthly performance statistics."""
        try:
            df = pd.read_csv(self.journal, parse_dates=['date'])
            
            # Current month
            today = datetime.now()
            month_start = datetime(today.year, today.month, 1)
            monthly = df[df['date'] >= month_start]
            
            if len(monthly) == 0:
                return {}
            
            # Calculate all metrics
            returns = monthly['net'] / monthly['net'].shift(1).fillna(1000)
            sharpe = (monthly['net'].mean() / monthly['net'].std()) * np.sqrt(252) if monthly['net'].std() > 0 else 0
            max_daily_loss = monthly['net'].min()
            max_daily_gain = monthly['net'].max()
            max_drawdown = (monthly['net'].cumsum().cummax() - monthly['net'].cumsum()).max()
            
            return {
                'days_traded': len(monthly),
                'total_trades': monthly['trades'].sum(),
                'total_pnl': monthly['net'].sum(),
                'avg_daily_pnl': monthly['net'].mean(),
                'std_daily_pnl': monthly['net'].std(),
                'sharpe_ratio': sharpe,
                'win_rate': len(monthly[monthly['net'] > 0]) / len(monthly),
                'max_daily_gain': max_daily_gain,
                'max_daily_loss': max_daily_loss,
                'max_drawdown': max_drawdown,
                'profit_factor': (
                    abs(monthly[monthly['net'] > 0]['net'].sum() / 
                    monthly[monthly['net'] < 0]['net'].sum())
                    if len(monthly[monthly['net'] < 0]) > 0 and 
                    monthly[monthly['net'] < 0]['net'].sum() != 0 else 0
                ),
                'total_fees': monthly['fees'].sum(),
            }
        except Exception as e:
            logger.error(f"Could not calculate monthly statistics: {e}")
            return {}
    
    def compare_to_backtest(self, backtest_metrics: Dict) -> Dict:
        """
        Compare live performance to backtest expectations.
        This is critical for understanding what went wrong.
        """
        live_metrics = self.calculate_monthly_statistics()
        
        if not live_metrics or not backtest_metrics:
            return {}
        
        comparison = {
            'sharpe_live': live_metrics.get('sharpe_ratio', 0),
            'sharpe_backtest': backtest_metrics.get('sharpe_ratio', 0),
            'sharpe_divergence': (
                live_metrics.get('sharpe_ratio', 0) - 
                backtest_metrics.get('sharpe_ratio', 0)
            ),
            'win_rate_live': live_metrics.get('win_rate', 0),
            'win_rate_backtest': backtest_metrics.get('win_rate', 0),
            'slippage_estimate': backtest_metrics.get('avg_profit', 0) - live_metrics.get('avg_daily_pnl', 0),
        }
        
        logger.info("\n" + "="*70)
        logger.info("BACKTEST vs. LIVE COMPARISON")
        logger.info("="*70)
        logger.info(f"Sharpe Ratio:")
        logger.info(f"  Backtest: {comparison['sharpe_backtest']:.2f}")
        logger.info(f"  Live:     {comparison['sharpe_live']:.2f}")
        logger.info(f"  Divergence: {comparison['sharpe_divergence']:+.2f}")
        logger.info(f"\nWin Rate:")
        logger.info(f"  Backtest: {comparison['win_rate_backtest']:.1%}")
        logger.info(f"  Live:     {comparison['win_rate_live']:.1%}")
        logger.info(f"\nEstimated Slippage & Fees:")
        logger.info(f"  Backtest assumed profit: {backtest_metrics.get('avg_profit', 0):+.0f}")
        logger.info(f"  Live realized: {live_metrics.get('avg_daily_pnl', 0):+.0f}")
        logger.info(f"  Difference: {comparison['slippage_estimate']:+.0f}")
        logger.info("="*70 + "\n")
        
        return comparison
    
    def check_parameter_stability(self) -> Dict:
        """
        Check if signal parameters need adjustment.
        Parameters drift when market regimes change.
        """
        try:
            df = pd.read_csv(self.journal, parse_dates=['date'])
            
            # Split into weeks and check consistency
            weeks = []
            for i in range(0, len(df), 5):  # Group by trading weeks (5 days)
                week_data = df.iloc[i:i+5]
                if len(week_data) > 0:
                    weeks.append(week_data['net'].mean())
            
            if len(weeks) < 2:
                return {'stability_check': 'insufficient_data'}
            
            # Check variance of weekly performance
            week_std = np.std(weeks)
            week_mean = np.mean(weeks)
            
            # Coefficient of variation
            cv = week_std / abs(week_mean) if week_mean != 0 else float('inf')
            
            logger.info(f"\nParameter Stability Check:")
            logger.info(f"  Weekly avg P&L: {week_mean:+.0f}")
            logger.info(f"  Weekly std: {week_std:+.0f}")
            logger.info(f"  Coefficient of Variation: {cv:.2f}")
            
            if cv > 1.0:
                logger.warning("⚠ HIGH INSTABILITY - Consider retraining")
            elif cv > 0.5:
                logger.warning("⚠ MODERATE INSTABILITY - Monitor closely")
            else:
                logger.info("✓ Parameters stable")
            
            return {
                'cv': cv,
                'stable': cv < 0.5,
                'needs_retraining': cv > 1.0
            }
        except Exception as e:
            logger.error(f"Could not check parameter stability: {e}")
            return {}
    
    def print_monthly_report(self, backtest_metrics: Dict = None):
        """Print comprehensive monthly report."""
        stats = self.calculate_monthly_statistics()
        
        if not stats:
            logger.info("No monthly data available")
            return
        
        logger.info("\n" + "="*70)
        logger.info("MONTHLY PERFORMANCE REVIEW")
        logger.info("="*70)
        logger.info(f"Trading Days: {stats.get('days_traded', 0)}")
        logger.info(f"Total Trades: {stats.get('total_trades', 0)}")
        logger.info(f"\nReturn Metrics:")
        logger.info(f"  Total P&L: {stats.get('total_pnl', 0):+,.0f}")
        logger.info(f"  Avg Daily P&L: {stats.get('avg_daily_pnl', 0):+,.0f}")
        logger.info(f"  Std Dev: {stats.get('std_daily_pnl', 0):,.0f}")
        logger.info(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        logger.info(f"\nRisk Metrics:")
        logger.info(f"  Max Daily Gain: {stats.get('max_daily_gain', 0):+,.0f}")
        logger.info(f"  Max Daily Loss: {stats.get('max_daily_loss', 0):+,.0f}")
        logger.info(f"  Max Drawdown: {stats.get('max_drawdown', 0):+,.0f}")
        logger.info(f"\nTrade Metrics:")
        logger.info(f"  Win Rate: {stats.get('win_rate', 0):.1%}")
        logger.info(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
        logger.info(f"  Total Fees: {stats.get('total_fees', 0):+,.0f}")
        logger.info("="*70)
        
        if backtest_metrics:
            self.compare_to_backtest(backtest_metrics)
        
        self.check_parameter_stability()
        
        logger.info("="*70 + "\n")
```

---

## Module 26.3: Common First-Month Problems — Diagnosis and Solutions

### Problem 1: Slippage Much Higher Than Expected

**Symptom:** Your backtest showed 0.1% slippage per trade, but realized slippage is 0.5-1.0%.

**Root Causes:**
1. Backtest used historical OHLC prices, not tick-by-tick fills
2. Your order size is larger relative to market depth in live trading
3. Market impact: your buying pushes prices up, selling pushes them down
4. You're trading illiquid instruments at off-hours

```python
# diagnosis/slippage_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SlippageAnalyzer:
    """Analyze and diagnose slippage in live trading."""
    
    @staticmethod
    def calculate_realized_slippage(
        orders: List[Dict],
        market_prices: Dict
    ) -> Dict:
        """
        Calculate actual slippage for each trade.
        
        Args:
            orders: List of executed orders with {symbol, qty, execution_price, order_time}
            market_prices: Dict of {timestamp: {symbol: {bid, ask, last}}}
        
        Returns:
            Dictionary with slippage analysis
        """
        slippages_by_symbol = {}
        
        for order in orders:
            symbol = order['symbol']
            exec_price = order['execution_price']
            order_time = order['order_time']
            qty = order['qty']
            side = order['side']  # 'BUY' or 'SELL'
            
            # Get mid price at order time
            if order_time in market_prices:
                prices = market_prices[order_time].get(symbol, {})
                mid_price = (prices.get('bid', exec_price) + prices.get('ask', exec_price)) / 2
                
                # Calculate slippage in bps (basis points)
                if side == 'BUY':
                    slippage_bps = (exec_price - mid_price) / mid_price * 10000
                else:  # SELL
                    slippage_bps = (mid_price - exec_price) / mid_price * 10000
                
                if symbol not in slippages_by_symbol:
                    slippages_by_symbol[symbol] = []
                
                slippages_by_symbol[symbol].append({
                    'timestamp': order_time,
                    'qty': qty,
                    'slippage_bps': slippage_bps,
                    'executed_price': exec_price,
                    'fair_value': mid_price
                })
        
        # Aggregate statistics
        results = {}
        for symbol, slippages in slippages_by_symbol.items():
            slippage_values = [s['slippage_bps'] for s in slippages]
            results[symbol] = {
                'count': len(slippages),
                'avg_slippage_bps': np.mean(slippage_values),
                'median_slippage_bps': np.median(slippage_values),
                'max_slippage_bps': np.max(slippage_values),
                'std_slippage_bps': np.std(slippage_values),
                'slippages': slippages
            }
        
        return results
    
    @staticmethod
    def diagnose_slippage_root_cause(
        analyzer_results: Dict,
        order_sizes: List[float],
        market_depth_snapshots: List[Dict]
    ) -> Dict:
        """
        Diagnose root cause of high slippage.
        """
        diagnosis = {}
        
        # 1. Check if slippage correlates with order size
        if len(order_sizes) > 1:
            correlation = np.corrcoef(order_sizes, [r.get('avg_slippage_bps', 0) 
                                                   for r in analyzer_results.values()])[0, 1]
            diagnosis['size_correlation'] = correlation
            
            if correlation > 0.5:
                logger.warning("⚠ DIAGNOSIS: Order size drives slippage")
                logger.warning("  SOLUTION: Reduce order size or split orders")
        
        # 2. Check if slippage correlates with volatility
        diagnosis['is_volatile_period'] = False  # Would check intraday volatility
        
        # 3. Check market depth at typical order times
        diagnosis['market_depth_issue'] = False
        for snapshot in market_depth_snapshots:
            # If bid-ask spread is wide, that's slippage
            if snapshot.get('spread_pct', 0) > 0.1:  # >0.1% spread
                diagnosis['market_depth_issue'] = True
                logger.warning("⚠ DIAGNOSIS: Wide spreads during your trading hours")
                logger.warning("  SOLUTION: Trade during more liquid hours or use limit orders")
        
        # 4. Check if you're hitting worse than best bid/ask
        diagnosis['execution_worse_than_best'] = False
        
        return diagnosis
    
    @staticmethod
    def print_slippage_report(analysis: Dict):
        """Print detailed slippage analysis."""
        logger.info("\n" + "="*70)
        logger.info("SLIPPAGE ANALYSIS")
        logger.info("="*70)
        
        for symbol, stats in analysis.items():
            logger.info(f"\n{symbol}:")
            logger.info(f"  Trades: {stats['count']}")
            logger.info(f"  Avg Slippage: {stats['avg_slippage_bps']:.1f} bps ({stats['avg_slippage_bps']*0.01:.2f}%)")
            logger.info(f"  Median: {stats['median_slippage_bps']:.1f} bps")
            logger.info(f"  Max: {stats['max_slippage_bps']:.1f} bps")
            logger.info(f"  Std Dev: {stats['std_slippage_bps']:.1f} bps")
        
        logger.info("\n" + "="*70 + "\n")

# SLIPPAGE SOLUTIONS
"""
If average slippage is:
  0-5 bps:     Acceptable, update backtest assumptions upward
  5-20 bps:    Investigate order size and market depth
  20-50 bps:   Large issue, reduce order size or trade more liquid instruments
  50+ bps:     Critical issue, DO NOT TRADE until resolved

IMMEDIATE FIXES:
1. Reduce order size by 50%
2. Use limit orders instead of market orders
3. Trade only during peak liquidity hours (10:00-14:30 IST for NSE)
4. Add 10 bps to backtest slippage model
5. Monitor whether slippage is cost (fees/spread) or impact (your size moves price)
"""
```

### Problem 2: Model Performance Diverging from Backtest

**Symptom:** Backtest Sharpe was 1.8, live trading is 0.2. Win rate dropped from 55% to 45%.

**Root Causes:**
1. Parameter instability (overfitting to past data)
2. Regime change (past patterns don't hold)
3. Look-ahead bias in backtest (using data not available at trade time)
4. Transaction costs underestimated

```python
# diagnosis/model_divergence_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)

class ModelDivergenceAnalyzer:
    """Diagnose why live performance diverges from backtest."""
    
    @staticmethod
    def check_for_overfitting_signals(
        live_signals: List[float],
        live_prices: List[float],
        backtest_expected_correlation: float
    ) -> Dict:
        """
        Check if signals are overfitted (too sensitive to noise).
        Overfitted signals don't generalize to live data.
        """
        # Calculate rolling correlation between signals and price moves
        price_moves = np.diff(live_prices) / live_prices[:-1]
        signal_correlation = np.corrcoef(live_signals[:-1], price_moves)[0, 1]
        
        return {
            'signal_to_price_correlation': signal_correlation,
            'expected_correlation': backtest_expected_correlation,
            'likely_overfitting': abs(signal_correlation - backtest_expected_correlation) > 0.2,
            'recommendation': 'Retrain with more regularization' if abs(signal_correlation - backtest_expected_correlation) > 0.2 else 'OK'
        }
    
    @staticmethod
    def check_for_regime_change(
        live_trade_journal: pd.DataFrame,
        backtest_parameters: Dict
    ) -> Dict:
        """
        Check if market regime has changed.
        Volatility, correlation, or price direction changes break strategies.
        """
        if len(live_trade_journal) < 10:
            return {'data_insufficient': True}
        
        # Calculate realized volatility
        returns = live_trade_journal['net'] / 100000  # Normalize by capital
        realized_vol = returns.std() * np.sqrt(252)  # Annualized
        backtest_vol = backtest_parameters.get('expected_volatility', 0)
        
        vol_changed = realized_vol > backtest_vol * 1.5
        
        logger.info(f"\nRegime Change Analysis:")
        logger.info(f"  Backtest volatility: {backtest_vol:.2f}")
        logger.info(f"  Realized volatility: {realized_vol:.2f}")
        
        if vol_changed:
            logger.warning("⚠ REGIME CHANGE: Volatility increased >50%")
        
        return {
            'realized_volatility': realized_vol,
            'expected_volatility': backtest_vol,
            'volatility_increased': vol_changed,
            'recommendation': 'Reduce position size if volatility increased' if vol_changed else 'OK'
        }
    
    @staticmethod
    def check_for_lookahead_bias(
        strategy_logic: Callable,
        data: pd.DataFrame
    ) -> Dict:
        """
        Check if your signal function uses future data.
        Look-ahead bias: using data that wouldn't be available at trade time.
        """
        issues = []
        
        # Check for common lookahead patterns
        # 1. Using future prices (should only use current and past)
        # 2. Using indicators computed on full dataset (should be incremental)
        # 3. Normalizing prices using future max/min
        
        # This requires inspecting your code
        logger.info("\nLook-ahead Bias Check:")
        logger.info("  Manual inspection required of signal function")
        logger.info("  Look for:")
        logger.info("    - df.shift(-1) or future price references")
        logger.info("    - max(prices) computed on future data")
        logger.info("    - normalization using future bounds")
        
        return {
            'requires_manual_inspection': True,
            'common_issues': [
                'Using future OHLC data in feature computation',
                'Normalizing using future max/min values',
                'Computing indicators on full dataset instead of incrementally'
            ]
        }
    
    @staticmethod
    def print_divergence_analysis_report(
        backtest_metrics: Dict,
        live_metrics: Dict,
        divergence_diagnosis: Dict
    ):
        """Print comprehensive divergence analysis."""
        logger.info("\n" + "="*70)
        logger.info("MODEL DIVERGENCE ANALYSIS")
        logger.info("="*70)
        logger.info("\nMetrics Comparison:")
        logger.info(f"  Sharpe Ratio:")
        logger.info(f"    Backtest: {backtest_metrics.get('sharpe', 0):.2f}")
        logger.info(f"    Live:     {live_metrics.get('sharpe', 0):.2f}")
        logger.info(f"    Divergence: {backtest_metrics.get('sharpe', 0) - live_metrics.get('sharpe', 0):+.2f}")
        logger.info(f"\n  Win Rate:")
        logger.info(f"    Backtest: {backtest_metrics.get('win_rate', 0):.1%}")
        logger.info(f"    Live:     {live_metrics.get('win_rate', 0):.1%}")
        logger.info(f"\nDiagnosis:")
        for key, value in divergence_diagnosis.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*70 + "\n")

# ROOT CAUSE DECISION TREE
"""
If live Sharpe < 50% of backtest Sharpe:

1. Check overfitting:
   - Are signals still correlated with prices? (should be > 0.3)
   - If correlation dropped significantly → retrain with regularization
   
2. Check regime change:
   - Did volatility increase? → reduce position size
   - Did correlation structure change? → signals may not work
   - Did trend direction reverse? → long-only strategies break in down trends
   
3. Check look-ahead bias:
   - Review signal code for future data usage
   - Recompute signals on historical data using ONLY past bars
   - Compare to original signals
   
4. Check transaction costs:
   - Add 50 bps to backtest slippage
   - Recalculate backtest with realistic costs
   - Compare to live P&L
   
5. Check data quality:
   - Are live feeds identical to backtest data?
   - Do prices match between Zerodha and historical source?
"""
```

### Problem 3: API Disconnections and Data Gaps

**Symptom:** Suddenly your orders won't execute. Data feed stops updating. System crashes.

```python
# resilience/api_resilience.py
from typing import Optional, Dict
import logging
import time
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class APIResilienceManager:
    """Handle API failures gracefully."""
    
    def __init__(
        self,
        broker_connection,
        data_provider,
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
        state_file: str = "state.json"
    ):
        self.broker = broker_connection
        self.data = data_provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds
        self.state_file = state_file
        self.last_successful_data_time = datetime.now()
    
    def safe_api_call(
        self,
        api_function,
        *args,
        fallback_value=None,
        **kwargs
    ) -> Optional[Dict]:
        """
        Execute API call with automatic retry and graceful degradation.
        """
        for attempt in range(self.max_retries):
            try:
                result = api_function(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"API call failed after {self.max_retries} retries")
                    return fallback_value
        
        return fallback_value
    
    def get_positions_safe(self) -> Dict:
        """Get positions with fallback to cached state."""
        # Try live API
        positions = self.safe_api_call(
            self.broker.positions,
            fallback_value=None
        )
        
        if positions is not None:
            return {'source': 'live', 'positions': positions}
        
        # Fall back to cached state
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.warning("Using cached positions from state file")
            return {'source': 'cache', 'positions': state.get('positions', [])}
        except:
            logger.error("No cached state available")
            return {'source': 'none', 'positions': []}
    
    def get_quote_safe(self, symbol: str) -> Optional[Dict]:
        """Get quote with fallback to last known price."""
        quote = self.safe_api_call(
            self.broker.quote,
            symbol,
            fallback_value=None
        )
        
        if quote is not None:
            self.last_successful_data_time = datetime.now()
            return quote
        
        # Check if data gap is too long
        time_since_last_update = (datetime.now() - self.last_successful_data_time).total_seconds()
        if time_since_last_update > 300:  # >5 minutes without data
            logger.error(f"Data gap >5 minutes for {symbol}")
            return None
        
        logger.warning(f"Quote fetch failed, but data gap acceptable")
        return None
    
    def monitor_connection_health(self) -> Dict:
        """Monitor API connection health."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'broker_reachable': False,
            'data_flowing': False,
            'last_data_update': None,
            'status': 'UNKNOWN'
        }
        
        # Check broker
        try:
            self.broker.quote('NSE:SBIN')
            health['broker_reachable'] = True
        except:
            health['broker_reachable'] = False
        
        # Check data freshness
        time_since_data = (datetime.now() - self.last_successful_data_time).total_seconds()
        health['data_flowing'] = time_since_data < 60  # Data < 1 min old
        health['last_data_update'] = self.last_successful_data_time.isoformat()
        
        # Determine overall status
        if health['broker_reachable'] and health['data_flowing']:
            health['status'] = 'HEALTHY'
        elif health['broker_reachable'] and not health['data_flowing']:
            health['status'] = 'DEGRADED_NO_DATA'
        elif not health['broker_reachable']:
            health['status'] = 'CRITICAL'
        
        return health
    
    def handle_connection_loss(self) -> Dict:
        """Handle loss of connection - DO NOT TRADE."""
        logger.error("="*70)
        logger.error("CRITICAL: CONNECTION LOST")
        logger.error("="*70)
        
        # Immediate actions:
        # 1. Stop trading
        trading_enabled = False
        
        # 2. Attempt to restore connection
        logger.info("Attempting to restore connection...")
        time.sleep(5)
        
        # 3. Verify connection before resuming
        health = self.monitor_connection_health()
        
        if health['status'] != 'HEALTHY':
            logger.error("Connection still unhealthy. Staying in safe mode.")
            return {
                'trading_allowed': False,
                'reason': health['status'],
                'action': 'MANUAL_INTERVENTION_REQUIRED'
            }
        
        logger.info("Connection restored. Monitoring before resuming trades...")
        return {
            'trading_allowed': True,
            'reason': 'connection_restored',
            'action': 'RESUME_TRADING'
        }

# CONNECTION LOSS PROTOCOL
"""
When connection is lost:
1. STOP ALL TRADING IMMEDIATELY
2. Try to reconnect (auto-retry with exponential backoff)
3. Check if you have open positions (use last known state)
4. If positions exist:
   - Close them using limit orders 10% better than last price
   - Don't use market orders (too risky with stale prices)
5. Wait 30 minutes before resuming trading
6. Log the incident for review

Never:
- Continue trading without verified connection
- Use stale data to make decisions
- Place market orders when connection is unstable
"""
```

### Problem 4: Emotional Decision-Making During Live Trading

**Symptom:** "My strategy said to hold, but the position is down 3%, so I'm exiting. Or my strategy would have lost, but I added to the position to average down."

```python
# discipline/emotional_override_prevention.py
from datetime import datetime
from typing import Dict, List
import logging
import json

logger = logging.getLogger(__name__)

class TradeJournal:
    """
    Trade journal: document EVERY trade decision.
    Forces you to articulate WHY before overriding system.
    """
    
    def __init__(self, journal_file: str):
        self.journal_file = journal_file
    
    def log_trade_decision(
        self,
        timestamp: datetime,
        decision: str,  # 'SYSTEM_SIGNAL' or 'MANUAL_OVERRIDE'
        symbol: str,
        action: str,  # 'BUY', 'SELL', 'HOLD'
        position_size: float,
        reason: str,  # Required for overrides
        signal_value: float,
        risk_assessment: str
    ):
        """Log every trade decision with full context."""
        entry = {
            'timestamp': timestamp.isoformat(),
            'decision_type': decision,
            'symbol': symbol,
            'action': action,
            'position_size': position_size,
            'reason': reason,
            'signal_value': signal_value,
            'risk_assessment': risk_assessment,
        }
        
        # Append to journal
        try:
            with open(self.journal_file, 'r') as f:
                journal = json.load(f)
        except:
            journal = []
        
        journal.append(entry)
        with open(self.journal_file, 'w') as f:
            json.dump(journal, f, indent=2)
        
        # If it's a manual override, flag it for review
        if decision == 'MANUAL_OVERRIDE':
            logger.warning(
                f"⚠ MANUAL OVERRIDE: {symbol} {action} @ {timestamp}\n"
                f"   Reason: {reason}\n"
                f"   Risk: {risk_assessment}"
            )
    
    def analyze_overrides(self) -> Dict:
        """Analyze override performance - are manual decisions better than system?"""
        try:
            with open(self.journal_file, 'r') as f:
                journal = json.load(f)
        except:
            return {}
        
        system_trades = [t for t in journal if t['decision_type'] == 'SYSTEM_SIGNAL']
        override_trades = [t for t in journal if t['decision_type'] == 'MANUAL_OVERRIDE']
        
        logger.info("\n" + "="*70)
        logger.info("OVERRIDE ANALYSIS")
        logger.info("="*70)
        logger.info(f"System trades: {len(system_trades)}")
        logger.info(f"Manual overrides: {len(override_trades)}")
        
        if len(override_trades) > len(system_trades) * 0.1:  # >10% overrides
            logger.warning("⚠ HIGH OVERRIDE RATE (>10%)")
            logger.warning("  This suggests emotional trading, not disciplined execution")
        
        logger.info("="*70 + "\n")
        
        return {
            'system_trades': len(system_trades),
            'override_trades': len(override_trades),
            'override_rate': len(override_trades) / (len(system_trades) + len(override_trades))
        }

# RULE: NEVER OVERRIDE THE SYSTEM (almost never)
"""
The ONLY acceptable reasons to override:
1. Data corruption detected (prices seem wrong)
2. Risk limit exceeded (position too large due to system bug)
3. Broker reported execution error
4. Market halted or in circuit breaker

UNACCEPTABLE reasons to override:
- "I have a gut feeling"
- "The position is down, let me average down"
- "I want to hold longer for more upside"
- "I'm afraid to take the loss"

If you find yourself overriding often, the problem is:
1. System not implemented correctly (code bug)
2. Backtest unrealistic (didn't account for real slippage/fees)
3. Wrong instrument choice (too illiquid)
4. Parameter overfitting (model doesn't generalize)

Fix the ROOT CAUSE, don't fight it with manual overrides.
"""

class DisciplineRules:
    """Enforce trading discipline."""
    
    @staticmethod
    def is_override_justified(reason: str) -> bool:
        """
        Determine if a manual override is justified.
        """
        justified_reasons = [
            'data_corruption',
            'risk_limit_exceeded',
            'execution_error',
            'market_halt',
            'circuit_breaker'
        ]
        
        return reason in justified_reasons
    
    @staticmethod
    def check_emotional_trading(
        trade_history: List[Dict],
        current_position: Dict
    ) -> Dict:
        """Detect signs of emotional trading."""
        issues = []
        
        # Check 1: Many small losses followed by revenge trading
        recent_trades = trade_history[-10:]  # Last 10 trades
        losses = [t for t in recent_trades if t.get('pnl', 0) < 0]
        if len(losses) > 5:
            issues.append("Multiple consecutive losses → risk of revenge trading")
        
        # Check 2: Averaging down (adding to losing position)
        if current_position.get('quantity', 0) > 0:
            if current_position.get('unrealized_pnl', 0) < 0:
                issues.append("Adding to losing position → averaging down bias")
        
        # Check 3: Holding winners longer than system says
        # This requires comparing actual holding periods to signal
        
        return {
            'potential_issues': issues,
            'likely_emotional_trading': len(issues) > 0
        }
```

### Problem 5: The Importance of Documentation

**Why documentation matters:**

After losing money, you'll want to understand why. After making money, you'll want to repeat it. Documentation (trade journal) is how you learn.

```python
# documentation/trade_journal.py
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DetailedTradeJournal:
    """
    Comprehensive trade journal for learning and improvement.
    Every trade → detailed analysis.
    """
    
    def __init__(self, journal_file: str):
        self.journal_file = journal_file
        self.trades: pd.DataFrame = pd.DataFrame()
        self.load()
    
    def load(self):
        """Load existing journal."""
        try:
            self.trades = pd.read_csv(self.journal_file, parse_dates=['entry_time', 'exit_time'])
        except:
            self.trades = pd.DataFrame()
    
    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        entry_reason: str,
        entry_signal_value: float,
        quantity: int,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        exit_reason: Optional[str] = None,
        max_profit: Optional[float] = None,
        max_loss: Optional[float] = None,
        holding_minutes: Optional[int] = None,
        notes: str = ""
    ):
        """Record a complete trade with analysis."""
        
        pnl = None
        pnl_pct = None
        if exit_price:
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price
        
        trade = {
            'entry_time': entry_time,
            'symbol': symbol,
            'entry_price': entry_price,
            'entry_reason': entry_reason,
            'entry_signal_value': entry_signal_value,
            'quantity': quantity,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'holding_minutes': holding_minutes,
            'notes': notes,
            'recorded_time': datetime.now()
        }
        
        self.trades = pd.concat(
            [self.trades, pd.DataFrame([trade])],
            ignore_index=True
        )
        self.save()
        
        logger.info(
            f"✓ Trade recorded: {symbol} {quantity}sh @ {entry_price:.2f} "
            f"({entry_reason})"
        )
    
    def save(self):
        """Save journal to CSV."""
        self.trades.to_csv(self.journal_file, index=False)
    
    def generate_trade_analysis(self, symbol: Optional[str] = None) -> Dict:
        """Generate analysis of all trades (or specific symbol)."""
        if len(self.trades) == 0:
            return {}
        
        trades = self.trades if symbol is None else self.trades[self.trades['symbol'] == symbol]
        completed = trades[trades['exit_price'].notna()]
        
        if len(completed) == 0:
            return {'status': 'no_completed_trades'}
        
        pnl_values = completed['pnl'].dropna()
        
        analysis = {
            'total_trades': len(trades),
            'completed_trades': len(completed),
            'total_pnl': pnl_values.sum(),
            'avg_pnl': pnl_values.mean(),
            'win_rate': len(pnl_values[pnl_values > 0]) / len(pnl_values),
            'avg_win': pnl_values[pnl_values > 0].mean() if len(pnl_values[pnl_values > 0]) > 0 else 0,
            'avg_loss': pnl_values[pnl_values < 0].mean() if len(pnl_values[pnl_values < 0]) > 0 else 0,
            'best_trade': pnl_values.max(),
            'worst_trade': pnl_values.min(),
            'avg_holding_minutes': completed['holding_minutes'].mean(),
        }
        
        return analysis
    
    def print_journal_summary(self):
        """Print human-readable journal summary."""
        if len(self.trades) == 0:
            logger.info("Trade journal is empty")
            return
        
        logger.info("\n" + "="*70)
        logger.info("TRADE JOURNAL SUMMARY")
        logger.info("="*70)
        
        by_symbol = self.trades.groupby('symbol')
        for symbol, group in by_symbol:
            completed = group[group['exit_price'].notna()]
            if len(completed) > 0:
                pnl = completed['pnl'].sum()
                count = len(completed)
                logger.info(f"\n{symbol}: {count} trades, {pnl:+,.0f} P&L")
        
        logger.info("\n" + "="*70 + "\n")

# JOURNAL BEST PRACTICES
"""
1. Record EVERY trade, good or bad
2. Include reason BEFORE entering trade (not after)
3. Record exit reason honestly (not excuses)
4. Note unusual market conditions
5. Review weekly: what worked, what didn't?
6. Look for patterns:
   - Do you lose more when trading certain times?
   - Do shorts work but longs don't?
   - Do certain entry reasons lead to better trades?
7. Update based on learnings
"""
```

---

## Summary: First 30 Days Playbook

```python
# main_trading_loop.py
"""
Main trading loop: orchestrates pre-market, during-market, post-market routines.
"""

import logging
from datetime import datetime, time
import time as time_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LiveTradingSystem:
    """Main live trading system orchestrator."""
    
    def __init__(
        self,
        broker_connection,
        data_provider,
        signal_generator,
        portfolio_constructor,
        risk_manager,
        config: Dict
    ):
        self.broker = broker_connection
        self.data = data_provider
        self.signals = signal_generator
        self.portfolio = portfolio_constructor
        self.risk = risk_manager
        self.config = config
    
    def run_trading_day(self):
        """Run complete trading day."""
        
        # 1. PRE-MARKET
        pre_market = PreMarketRoutine(
            state_file=self.config['state_file'],
            data_dir=self.config['data_dir'],
            broker_connection=self.broker,
            risk_params=self.config['risk_params']
        )
        pre_market_results = pre_market.run_full_routine()
        
        if not all([
            pre_market_results['time_check'],
            pre_market_results['system_health'],
            pre_market_results['daily_limit']
        ]):
            logger.error("Pre-market checks failed. Exiting.")
            return
        
        # 2. DURING MARKET
        logger.info("Market open. Beginning trading...")
        monitor = MarketMonitor(
            state_file=self.config['state_file'],
            broker_connection=self.broker,
            risk_params=self.config['risk_params']
        )
        
        # Trading loop (9:15 AM - 3:25 PM IST)
        market_open = time(9, 15)
        market_close = time(15, 25)
        
        while True:
            now = datetime.now().time()
            
            if now >= market_close:
                logger.info("Market closing. Stopping live trades.")
                break
            
            if now >= market_open:
                # Get latest data
                latest_data = self.data.get_latest_bars(lookback=100)
                
                # Generate signals
                signals = self.signals.generate(latest_data)
                
                # Construct portfolio
                positions = self.portfolio.construct(signals, latest_data)
                
                # Risk checks
                if self.risk.check_all_limits(positions):
                    # Execute orders
                    self.risk.execute_orders(positions)
                else:
                    logger.warning("Risk checks failed. Not executing.")
                
                # Monitor
                monitor.print_monitoring_summary()
                
                if monitor.should_halt_trading():
                    logger.error("HALT: Daily loss limit exceeded")
                    break
            
            time_module.sleep(60)  # Check every minute
        
        # 3. POST-MARKET
        post_market = PostMarketRoutine(
            state_file=self.config['state_file'],
            broker_connection=self.broker,
            trade_journal_file=self.config['trade_journal']
        )
        daily_metrics = post_market.run_full_routine()
        
        # 4. WEEKLY REVIEW (if Friday)
        if datetime.now().weekday() == 4:  # Friday
            weekly = WeeklyAnalysis(trade_journal_file=self.config['trade_journal'])
            weekly.print_weekly_report()
        
        # 5. MONTHLY REVIEW (if first of month)
        if datetime.now().day == 1:
            monthly = MonthlyReview(
                trade_journal_file=self.config['trade_journal'],
                state_file=self.config['state_file']
            )
            monthly.print_monthly_report(
                backtest_metrics=self.config.get('backtest_metrics', {})
            )

# RUNNING THE SYSTEM
if __name__ == "__main__":
    import json
    
    # Load config
    with open('trading_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize broker connection
    broker = setup_zerodha_connection(config['api_key'])
    
    # Initialize data provider
    data_provider = LiveDataProvider(broker=broker)
    
    # Initialize strategy components
    signal_fn = load_signal_model(config['model_path'])
    portfolio_fn = load_portfolio_constructor(config['portfolio_config'])
    
    # Create main system
    system = LiveTradingSystem(
        broker_connection=broker,
        data_provider=data_provider,
        signal_generator=signal_fn,
        portfolio_constructor=portfolio_fn,
        risk_manager=RiskManager(config['risk_params']),
        config=config
    )
    
    # Run
    try:
        system.run_trading_day()
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")
        logger.error("Attempting emergency shutdown...")
        # Close all positions
        # Disable trading
        # Notify user
```

---

## Conclusion: Your First Month in Production

The first 30 days of live trading are not about profit. They're about learning what your system actually does when real money is at stake.

You'll discover:
- Your model's actual Sharpe ratio (not backtest fiction)
- Where your risk limits actually hurt
- How you behave under pressure
- What you forgot to account for

**By the end of month 1, you should have:**
1. ✓ Complete pre-launch checklist passed
2. ✓ 10+ trading days of live data
3. ✓ Detailed trade journal with analysis
4. ✓ Identified any divergences from backtest
5. ✓ Fixed major bugs and issues
6. ✓ Proven the system runs without human oversight

If you've achieved all this and cumulative P&L is positive, congratulations. You've built something real.

If P&L is negative, that's data too. Review your trade journal, identify patterns, and improve.

The difference between successful quant traders and failed ones isn't intelligence. It's systematic execution, rigorous logging, and disciplined learning from failures.

You now have the playbook. Go execute.

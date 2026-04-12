# Chapter 9: Financial Data Engineering

## Introduction

Data is the foundation of quantitative trading. This chapter teaches you how to source, validate, store, and transform financial data into analysis-ready formats for building trading systems. You'll learn the engineering practices that separate production quant systems from toy projects.

We'll focus on Indian market data (NSE) while covering global sources. Our target reader is an ML/systems engineer with zero finance knowledge building on Zerodha, so we'll explain financial concepts from first principles.

**Key Principle**: In trading, 70% of the effort goes to data engineering, 20% to strategy, and 10% to execution. Get your data pipeline right, and the rest becomes straightforward.

---

## Module 9.1: Financial Data Types and Sources

### What is OHLCV Data?

OHLCV (Open, High, Low, Close, Volume) is the canonical price format in finance. Here's what each field means:

**Open (O)**: The price of the first trade executed in a time period
- For a 1-minute candle, this is the price when the minute started
- Interpretation: Market sentiment at period start

**High (H)**: The maximum price reached during the period
- $H = \max(P_t) \text{ for } t \in \text{period}$
- Interpretation: Peak buying pressure/seller resistance

**Low (L)**: The minimum price reached during the period  
- $L = \min(P_t) \text{ for } t \in \text{period}$
- Interpretation: Peak selling pressure/buyer support

**Close (C)**: The price of the last trade in the period
- Historically sacred in finance (settlement price)
- Interpretation: What traders "agreed" the asset was worth at period end

**Volume (V)**: Total number of shares traded
- $V = \sum_{t \in \text{period}} \text{shares}$
- Interpretation: Confidence/liquidity of price movement
- **Critical**: Volume with no price movement = manipulation risk

### How OHLCV Is Computed

Exchanges don't give you OHLCV directly—they give tick data. Here's how it's computed:

```python
from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd
import numpy as np

def compute_ohlcv_from_ticks(
    ticks: List[Tuple[float, int, datetime]],
    period_seconds: int = 60
) -> pd.DataFrame:
    """
    Compute OHLCV bars from tick-level data.
    
    Args:
        ticks: List of (price, volume, timestamp) tuples
        period_seconds: Aggregation period in seconds
        
    Returns:
        DataFrame with columns: open, high, low, close, volume, timestamp
    """
    if not ticks:
        return pd.DataFrame()
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(ticks, columns=['price', 'volume', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create period buckets
    df['period'] = df['timestamp'].dt.floor(f'{period_seconds}S')
    
    # Aggregate by period
    ohlcv = df.groupby('period').agg({
        'price': ['first', 'max', 'min', 'last'],  # open, high, low, close
        'volume': 'sum',
        'timestamp': 'last'
    }).round(2)
    
    # Flatten column names
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    ohlcv = ohlcv.reset_index(drop=True)
    
    return ohlcv

# Example: NSE tick data for TCS
nse_ticks = [
    (3850.50, 100, datetime(2025, 1, 15, 9, 15, 0)),
    (3850.75, 200, datetime(2025, 1, 15, 9, 15, 15)),
    (3851.00, 150, datetime(2025, 1, 15, 9, 15, 45)),
    (3850.80, 175, datetime(2025, 1, 15, 9, 15, 59)),
]

ohlcv = compute_ohlcv_from_ticks(nse_ticks, period_seconds=60)
print(ohlcv)
# open      3850.50
# high      3851.00
# low       3850.50
# close     3850.80
# volume    625
```

### OHLCV Pitfalls and Gotchas

**1. Gaps Between Trading Days**
Markets close overnight. A gap means H > previous C or L < previous C without any trades in between.

```python
# WRONG approach for daily data
df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

# RIGHT approach: Handle gaps
def handle_overnight_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Replace overnight gaps with NaN to prevent inflated returns."""
    df['date'] = df['timestamp'].dt.date
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    
    # Mark the first bar of each day (may have gap)
    df.loc[df['is_new_day'], 'daily_return'] = np.nan
    df.loc[~df['is_new_day'], 'daily_return'] = (
        (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    )
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'daily_return']]
```

**2. Survivor Bias in Historical Data**
Markets include delisted stocks. Yahoo Finance only shows stocks that exist *today*, hiding bankruptcies.

**3. Volume Spikes with No Price Movement**
Legitimate: Large block trades  
Suspicious: Volume spike + close = open (potential manipulation)

```python
def detect_suspicious_volume(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.Series:
    """Flag bars with unusual volume-to-price movement ratio."""
    df['price_movement'] = abs((df['close'] - df['open']) / df['open'])
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    
    # Suspicious: High volume, minimal price movement
    suspicious = (df['volume_zscore'] > z_threshold) & (df['price_movement'] < 0.001)
    return suspicious
```

### Tick and TAQ Data

TAQ = Trade and Quote data. This is the raw feed from exchanges.

**Trade data**: Every executed transaction
```
Timestamp, Symbol, Price, Volume, Buy/Sell
2025-01-15 09:15:00.123, TCS, 3850.50, 100, BUY
2025-01-15 09:15:00.456, TCS, 3850.75, 75, SELL
```

**Quote data**: Best bid/ask and depth
```
Timestamp, Symbol, BidPrice, BidVolume, AskPrice, AskVolume
2025-01-15 09:15:00.100, TCS, 3850.25, 500, 3850.50, 600
```

For NSE:
- NSE publishes tick data via bhavcopy (NIFTY constituents)
- For deep analysis, Zerodha provides minute-level OHLC via Kite Historical API
- TAQ requires paid APIs (NYSE data) or exchange membership

```python
def parse_nse_bhavcopy(filename: str) -> pd.DataFrame:
    """
    Parse NSE bhavcopy CSV format.
    NSE format: SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL
    """
    df = pd.read_csv(filename)
    df = df[df['SERIES'] == 'EQ']  # Equity series only
    
    df['timestamp'] = pd.to_datetime('2025-01-15')  # Add actual date
    df['symbol'] = df['SYMBOL'].str.strip()
    df = df[['timestamp', 'symbol', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY']].copy()
    df.columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    
    return df
```

### Fundamental Data

Fundamental data = Company financial statements + ratios

**Income Statement**: Revenue, expenses, net income
```
FY2024: Revenue = 100,000 Cr, Net Income = 15,000 Cr
EPS = Net Income / Shares Outstanding
```

**Balance Sheet**: Assets, liabilities, equity
**Cash Flow**: Operating, investing, financing cash flows
**Ratios**: 
- P/E Ratio = $\frac{\text{Stock Price}}{\text{EPS}}$
- Price-to-Book = $\frac{\text{Market Cap}}{\text{Book Value}}$
- ROE = $\frac{\text{Net Income}}{\text{Equity}}$

**Critical Issue: Point-in-Time Data**

Fundamental data has *announcement delays*. A company reports FY2024 results in Feb 2025. If you use Feb 2024 historical prices with Feb 2025 earnings, you're cheating (look-ahead bias).

```python
class FundamentalData:
    """Store point-in-time fundamental data correctly."""
    
    def __init__(self):
        self.data = []  # List of (symbol, metric, value, announcement_date, fiscal_date)
    
    def get_at_date(self, symbol: str, date: datetime, metric: str) -> float:
        """
        Get fundamental metric as of a specific date.
        Only uses data announced *before* the date.
        """
        valid_data = [
            d for d in self.data
            if d[0] == symbol and d[2] == metric and d[3] <= date
        ]
        
        if not valid_data:
            return np.nan
        
        # Use most recent announcement before this date
        return max(valid_data, key=lambda x: x[3])[1]
```

### Alternative Data Sources

Modern quant funds use alternative data:

**News/NLP**: Market sentiment from news articles, social media
- Provider: Refinitiv, Bloomberg
- Metric: Sentiment score (-1 to +1)
- Pitfall: Most news data has *survivorship bias* (only major stories archived)

**Satellite Imagery**: Count cars in parking lots, track supply chains
- Provider: Orbital Insight, Planet Labs
- Example: Monitor Tesla factory output from space
- Latency: 1-3 days old (not real-time)

**Web Traffic**: Website visits, app downloads
- Provider: SimilarWeb, Apptopia
- Example: Amazon AWS console traffic ∝ enterprise spending
- Latency: 1-2 weeks

**Credit Card Transactions**: Actual consumer spending
- Provider: Consumera, Facteus
- Example: Restaurant spending data before earnings
- Latency: 2-4 weeks

**Warnings About Alternative Data**:
1. Often *lagged* (not real-time advantage)
2. May have *survivorship bias* (only successful companies tracked)
3. Expensive ($10K-$100K/month)
4. Requires significant ML to extract signals

### Data Sources for Indian Markets

**NSE Bhavcopy**
- Source: `https://www.nseindia.com/products/content/derivatives/equities/bhavcopy.htm`
- Format: Daily OHLCV for all stocks
- Coverage: Equities, derivatives
- Cost: Free
- Latency: T+1 day
- Access: Manual download or NSE API

```python
import requests
from datetime import datetime

def download_nse_bhavcopy(date: datetime) -> pd.DataFrame:
    """Download NSE bhavcopy for a specific date."""
    date_str = date.strftime('%d%b%Y').upper()
    url = f"https://www1.nseindia.com/content/historical/EQUITIES/{date.year}/JAN/eq{date_str}_csv.zip"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download bhavcopy for {date}")
    
    # Unzip and parse
    import zipfile
    import io
    
    z = zipfile.ZipFile(io.BytesIO(response.content))
    csv_content = z.read(z.namelist()[0])
    df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
    
    return df
```

**Zerodha Kite Historical API**
- Source: Zerodha (via Kite API)
- Format: Minute/daily OHLCV
- Coverage: Equities, derivatives, commodities
- Cost: Free (with active Zerodha account)
- Latency: Real-time (intraday) + T+1 (historical)

```python
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime

def fetch_kite_historical_data(
    api_key: str,
    access_token: str,
    instrument_token: int,
    start_date: datetime,
    end_date: datetime,
    interval: str = "minute"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Zerodha Kite API.
    
    Args:
        interval: 'minute' or 'day'
    """
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=start_date,
        to_date=end_date,
        interval=interval
    )
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['date'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    return df
```

**Yahoo Finance**
- Source: `https://finance.yahoo.com`
- Format: Daily OHLCV
- Coverage: Global + Indian indices
- Cost: Free
- Caveat: Survivorship bias (delisted companies disappear)
- Use case: Quick validation, not production

```python
import yfinance as yf

def fetch_yahoo_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    df = yf.download(f"{symbol}.NS", start=start, end=end)  # .NS = NSE
    return df
```

**EODHD (End-of-Day Historical Data)**
- Source: `https://eodhd.com`
- Format: Daily + minute OHLCV
- Coverage: 150+ exchanges globally
- Cost: $15-99/month
- Advantage: Clean data, corporate action adjusted
- Use for: NSE intraday where Kite lacks resolution

### Data Sources for US Markets

**Polygon.io**
- Coverage: Real-time + historical equities, options, forex, crypto
- Cost: Free tier (limited), $199+/month (pro)
- Best for: Aggregated tick data, reference data
- Latency: Real-time

**Alpha Vantage**
- Coverage: Equities, forex, crypto
- Cost: Free tier, $49+/month (premium)
- Best for: Learning, small backtests
- Caveat: Rate limits, lower quality than Bloomberg

**Quandl**
- Coverage: 200+ datasets (stocks, real estate, commodities, crypto)
- Cost: Free + paid subscriptions
- Best for: Alternative data, specific sectors
- Example: Federal Reserve data, mortgage rates

---

## Module 9.2: Data Quality and Cleaning

### Survivorship Bias

**Definition**: The *selection bias* that occurs when historical data only includes assets that *survived* to the present.

A study of Indian stocks from 2000 to 2025 using *current* NSE constituents misses:
- Companies that went bankrupt (e.g., Satyam Software fraud)
- Companies delisted for poor performance
- Acquired companies (merged into others)

**Why it matters**: 
If you backtest on survivors only, past returns are artificially inflated. A strategy that filtered by price momentum would have survived *exactly because* it worked—but on the subset of companies that didn't fail.

**Impact**: Typically inflates backtest Sharpe ratio by 0.5-2.0 depending on asset class and look-back period.

```python
def calculate_survivorship_bias_impact(
    survivors_returns: pd.Series,
    all_returns: pd.Series,
    metric: str = 'sharpe'
) -> dict:
    """
    Quantify the impact of survivorship bias.
    
    Args:
        survivors_returns: Returns of companies that survived
        all_returns: Returns of all companies (including failures)
        metric: 'sharpe', 'cagr', or 'volatility'
    """
    if metric == 'sharpe':
        survivor_sharpe = survivors_returns.mean() / survivors_returns.std() * np.sqrt(252)
        all_sharpe = all_returns.mean() / all_returns.std() * np.sqrt(252)
        bias = survivor_sharpe - all_sharpe
        
    elif metric == 'cagr':
        survivor_cagr = (1 + survivors_returns).prod() ** (252 / len(survivors_returns)) - 1
        all_cagr = (1 + all_returns).prod() ** (252 / len(all_returns)) - 1
        bias = survivor_cagr - all_cagr
    
    return {
        'survivors_metric': survivor_sharpe if metric == 'sharpe' else survivor_cagr,
        'all_metric': all_sharpe if metric == 'sharpe' else all_cagr,
        'bias': bias,
        'bias_pct': bias / (all_sharpe if metric == 'sharpe' else all_cagr) * 100
    }

# Example: 100 companies, 50 survive
survivor_returns = np.random.normal(0.15, 0.25, 1000)  # Survived: mean 15% annual
all_returns = np.concatenate([
    survivor_returns,
    np.random.normal(-0.30, 0.40, 1000)  # Failed: mean -30% annual
])

bias = calculate_survivorship_bias_impact(
    pd.Series(survivor_returns),
    pd.Series(all_returns),
    metric='sharpe'
)

print(f"Survivor Sharpe: {bias['survivors_metric']:.2f}")
print(f"All stocks Sharpe: {bias['all_metric']:.2f}")
print(f"Bias: {bias['bias']:.2f} ({bias['bias_pct']:.1f}%)")
```

**Correction Strategies**:

1. **Use delisting data**: Download historical NSE constituent lists
2. **Adjust for failures**: Assume bankruptcy returns (-100%)
3. **Use broader universe**: Include microcap, penny stocks

### Look-Ahead Bias

**Definition**: Using information in backtests that wouldn't be available at the time of the trade.

Common sources:
- Using today's adjusted close price (adjustment announced weeks later)
- Including earnings that haven't been announced yet
- Using financial ratios computed days after quarter-end

```python
# WRONG: Look-ahead bias
def bad_strategy(df: pd.DataFrame) -> pd.Series:
    """Use future information to trade."""
    # df['adjusted_close'] computed using stock splits announced later
    df['return'] = df['adjusted_close'].pct_change()
    
    # df['earnings_yield'] using earnings announced after this date
    df['signal'] = (df['earnings_yield'] > 0.05).astype(int)
    return df['signal']

# RIGHT: Avoid look-ahead
def good_strategy(df: pd.DataFrame) -> pd.Series:
    """Use only information available at trade time."""
    # Use unadjusted prices
    df['return'] = df['close'].pct_change()
    
    # Use earnings announced at least 1 day ago
    df['earnings_announced_lag'] = (
        (pd.Timestamp.now() - df['earnings_announcement_date']).dt.days
    )
    
    # Only use if announced > 1 day ago
    valid_earnings = df['earnings_announced_lag'] > 1
    df['earnings_yield'] = np.where(
        valid_earnings,
        df['net_income'] / df['market_cap'],
        np.nan
    )
    
    df['signal'] = (df['earnings_yield'] > 0.05).astype(int)
    return df['signal']
```

### Corporate Actions Adjustment

**Problem**: Stock splits, dividends, rights issues distort raw prices.

On Jan 15, 2025, TCS announced a 1:3 split (every share becomes 3 shares). Raw historical prices:
- Jan 14, 2024: Close = 3,000
- Jan 15, 2025: Close = 1,000 (after split)

A naive algorithm sees -67% drop. It was actually flat.

**Solution**: Track corporate actions, adjust historical prices.

```python
class CorporateAction:
    """Represent a corporate action (split, dividend, rights issue)."""
    
    def __init__(self, action_type: str, date: datetime, factor: float):
        """
        Args:
            action_type: 'split', 'dividend', 'rights_issue'
            date: Announcement/effective date
            factor: For split: new_price = old_price / factor (3-for-1 split: factor=3)
                   For dividend: adjust based on dividend per share
        """
        self.action_type = action_type
        self.date = date
        self.factor = factor

def apply_corporate_actions(
    df: pd.DataFrame,
    actions: List[CorporateAction]
) -> pd.DataFrame:
    """
    Adjust OHLCV for corporate actions.
    
    Corporate actions are applied backwards:
    If a 2:1 split happened on day 100, all prices before day 100 are halved.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort actions by date
    actions = sorted(actions, key=lambda x: x.date)
    
    for action in actions:
        # Adjust all prices BEFORE the action date
        mask = df['timestamp'] < action.date
        
        if action.action_type == 'split':
            df.loc[mask, 'open'] /= action.factor
            df.loc[mask, 'high'] /= action.factor
            df.loc[mask, 'low'] /= action.factor
            df.loc[mask, 'close'] /= action.factor
            df.loc[mask, 'volume'] *= action.factor  # More shares exist
            
        elif action.action_type == 'dividend':
            # Adjust close prices down by dividend amount
            df.loc[mask, 'open'] -= action.factor
            df.loc[mask, 'high'] -= action.factor
            df.loc[mask, 'low'] -= action.factor
            df.loc[mask, 'close'] -= action.factor
    
    return df

# Example: TCS 1:3 split on Jan 15, 2025
tcs_data = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=20, freq='D'),
    'open': np.random.uniform(2900, 3100, 20),
    'high': np.random.uniform(3100, 3200, 20),
    'low': np.random.uniform(2800, 3000, 20),
    'close': np.random.uniform(2900, 3100, 20),
    'volume': 10000000
})

actions = [
    CorporateAction(action_type='split', date=datetime(2025, 1, 15), factor=3)
]

adjusted_data = apply_corporate_actions(tcs_data, actions)
```

### Handling Missing Data

Strategies depend on *why* data is missing:

1. **Market closed** (holidays, weekends): *Skip* these periods
2. **No trades** (illiquid stock): *Forward-fill* last known price
3. **Data download failure**: *Interpolate* or mark as NaN

```python
def handle_missing_data(df: pd.DataFrame, method: str = 'auto') -> pd.DataFrame:
    """
    Handle missing OHLCV data.
    
    Args:
        method: 'skip' (remove), 'ffill' (forward-fill), 'interpolate'
    """
    df = df.copy()
    
    if method == 'skip':
        # Remove rows with any NaN
        return df.dropna()
    
    elif method == 'ffill':
        # Forward-fill OHLCV (use last trade price)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(method='ffill')
        df['volume'] = df['volume'].fillna(0)  # No trades = volume 0
        return df
    
    elif method == 'interpolate':
        # Linear interpolation (for intraday gaps only)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].interpolate(method='linear')
        df['volume'] = df['volume'].fillna(0)
        return df
    
    return df
```

### Outlier Detection

**Z-Score Method** (simple, assumes normal distribution)

```python
def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Flag outliers using z-score.
    
    $ z = \frac{x - \mu}{\sigma} $
    
    Points with |z| > threshold are outliers.
    """
    mean = series.mean()
    std = series.std()
    zscore = np.abs((series - mean) / std)
    return zscore > threshold
```

**Hampel Filter** (robust to extreme outliers)

```python
def detect_outliers_hampel(
    series: pd.Series,
    window: int = 5,
    threshold: float = 2.5
) -> pd.Series:
    """
    Hampel filter: outlier detection robust to extreme values.
    
    Uses median absolute deviation (MAD) instead of std dev.
    """
    median = series.rolling(window=window, center=True).median()
    mad = (series - median).abs().rolling(window=window, center=True).median()
    
    # Modified z-score using MAD
    modified_zscore = 0.6745 * (series - median) / mad
    return np.abs(modified_zscore) > threshold
```

**Domain-Specific Outlier Detection** (for finance)

```python
def detect_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect price outliers using domain knowledge.
    
    Flags:
    1. OHLC not in logical order: O < L or C > H
    2. Intraday move > 20% (penny stock pump-and-dump)
    3. Volume spike > 10x median (block trade or glitch)
    """
    df = df.copy()
    
    df['logical_error'] = (
        (df['open'] < df['low']) | 
        (df['close'] > df['high']) |
        (df['high'] < df['low'])
    )
    
    df['intraday_move'] = (df['high'] - df['low']) / df['open']
    df['extreme_move'] = df['intraday_move'] > 0.20
    
    df['volume_spike'] = (
        df['volume'] > df['volume'].rolling(20).median() * 10
    )
    
    df['is_outlier'] = df['logical_error'] | df['extreme_move'] | df['volume_spike']
    
    return df

# Usage
ohlcv = pd.DataFrame({
    'open': [100, 101, 3000, 105],  # Row 2: extreme jump
    'high': [102, 103, 3100, 107],
    'low': [99, 100, 2900, 104],
    'close': [101, 102, 3050, 106],
    'volume': [1000000, 1100000, 100, 950000]  # Row 2: no volume with huge move
})

outliers = detect_price_outliers(ohlcv)
print(outliers[['is_outlier']])
```

### Data Validation Pipeline

Production systems require automated checks:

```python
class DataValidator:
    """Automated data quality checks."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.errors = []
    
    def validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """Run all OHLCV validation checks."""
        
        checks = [
            self._check_ohlc_order,
            self._check_price_range,
            self._check_volume_positive,
            self._check_timestamp_ordered,
            self._check_no_duplicates,
            self._check_gaps_justified,
        ]
        
        all_passed = True
        for check in checks:
            if not check(df):
                all_passed = False
        
        return all_passed
    
    def _check_ohlc_order(self, df: pd.DataFrame) -> bool:
        """Verify: low <= open/close <= high"""
        violations = (
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['open'] > df['high']) |
            (df['close'] > df['high'])
        )
        
        if violations.any():
            self._log_error(f"OHLC order violation in {violations.sum()} rows")
            return False
        
        self.checks_passed += 1
        return True
    
    def _check_price_range(self, df: pd.DataFrame) -> bool:
        """Verify prices are within reasonable bounds."""
        # No negative prices
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            self._log_error("Negative prices detected")
            return False
        
        # No price > 1000x previous close (glitch)
        prev_close = df['close'].shift(1)
        max_move = (df['close'] / prev_close).max()
        if max_move > 1000:
            self._log_error(f"Extreme price move detected: {max_move}x")
            return False
        
        self.checks_passed += 1
        return True
    
    def _check_volume_positive(self, df: pd.DataFrame) -> bool:
        """Verify volume >= 0"""
        if (df['volume'] < 0).any():
            self._log_error("Negative volume detected")
            return False
        
        self.checks_passed += 1
        return True
    
    def _check_timestamp_ordered(self, df: pd.DataFrame) -> bool:
        """Verify timestamps are monotonically increasing."""
        if not df['timestamp'].is_monotonic_increasing:
            self._log_error("Timestamps not in order")
            return False
        
        self.checks_passed += 1
        return True
    
    def _check_no_duplicates(self, df: pd.DataFrame) -> bool:
        """Verify no duplicate timestamps."""
        dups = df['timestamp'].duplicated().sum()
        if dups > 0:
            self._log_error(f"{dups} duplicate timestamps")
            return False
        
        self.checks_passed += 1
        return True
    
    def _check_gaps_justified(self, df: pd.DataFrame) -> bool:
        """Verify gaps align with market holidays."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['gap_size'] = df['timestamp'].diff()
        
        # India market: gaps should be 1 day (excluding weekends)
        # Multi-day gaps = market holidays (justified)
        # Multi-hour gaps intraday = suspicious
        
        intraday_gaps = df[df['gap_size'] > pd.Timedelta(hours=1)]
        intraday_gaps = intraday_gaps[intraday_gaps['gap_size'] < pd.Timedelta(days=1)]
        
        if len(intraday_gaps) > 0:
            self._log_error(f"{len(intraday_gaps)} intraday gaps detected")
            return False
        
        self.checks_passed += 1
        return True
    
    def _log_error(self, message: str):
        """Log validation error."""
        self.errors.append(message)
        self.checks_failed += 1
        print(f"[VALIDATION ERROR] {message}")

# Usage
validator = DataValidator()
is_clean = validator.validate_ohlcv(ohlcv_dataframe)
print(f"Checks passed: {validator.checks_passed}")
print(f"Checks failed: {validator.checks_failed}")
```

### Building a Data Quality Monitoring Pipeline

```python
from typing import Dict, List
from datetime import datetime, timedelta
import logging

class DataQualityMonitor:
    """Production-grade data quality monitoring."""
    
    def __init__(self, log_path: str = '/var/log/data_quality.log'):
        self.logger = logging.getLogger('DataQuality')
        self.logger.addHandler(logging.FileHandler(log_path))
        
        self.metrics: Dict[str, List[float]] = {
            'upload_delay_hours': [],
            'missing_data_pct': [],
            'outlier_pct': [],
            'validation_pass_rate': [],
        }
    
    def monitor_ingestion(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Monitor a new data ingestion.
        
        Returns:
            Dict with quality metrics and alerts
        """
        report = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'rows': len(df),
            'alerts': [],
        }
        
        # Check 1: Upload delay
        latest_timestamp = df['timestamp'].max()
        delay_hours = (datetime.utcnow() - latest_timestamp).total_seconds() / 3600
        report['upload_delay_hours'] = delay_hours
        
        if delay_hours > 24:
            report['alerts'].append(f"Data is {delay_hours:.1f} hours old")
        
        self.metrics['upload_delay_hours'].append(delay_hours)
        
        # Check 2: Missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        report['missing_data_pct'] = missing_pct * 100
        
        if missing_pct > 0.01:  # More than 1% missing
            report['alerts'].append(f"{missing_pct*100:.2f}% data missing")
        
        self.metrics['missing_data_pct'].append(missing_pct)
        
        # Check 3: Outliers
        validator = DataValidator()
        outlier_mask = detect_price_outliers(df)['is_outlier']
        outlier_pct = outlier_mask.sum() / len(df)
        report['outlier_pct'] = outlier_pct * 100
        
        if outlier_pct > 0.05:  # More than 5% outliers
            report['alerts'].append(f"{outlier_pct*100:.2f}% rows are outliers")
        
        self.metrics['outlier_pct'].append(outlier_pct)
        
        # Check 4: Validation
        is_valid = validator.validate_ohlcv(df)
        pass_rate = validator.checks_passed / (validator.checks_passed + validator.checks_failed)
        report['validation_pass_rate'] = pass_rate * 100
        
        if not is_valid:
            report['alerts'].append(f"Validation failed: {validator.errors}")
        
        self.metrics['validation_pass_rate'].append(pass_rate)
        
        # Log report
        self.logger.info(f"Data quality report for {symbol}: {report}")
        
        return report
    
    def get_summary(self) -> Dict:
        """Get summary statistics of all monitored ingestions."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if not values:
                continue
            summary[metric_name] = {
                'mean': np.mean(values),
                'max': np.max(values),
                'min': np.min(values),
                'std': np.std(values),
            }
        return summary

# Usage
monitor = DataQualityMonitor()

# Each day, when new data arrives:
report = monitor.monitor_ingestion('TCS.NS', tcs_dataframe)
if report['alerts']:
    print(f"ALERTS for {report['symbol']}: {report['alerts']}")

# Weekly review
summary = monitor.get_summary()
print(f"Data quality summary: {summary}")
```

---

## Module 9.3: Data Storage Architecture

### Why You Need Proper Data Storage

Starting project:
```
# Store everything in CSV files?
/data/
  tcs_1min.csv (50 MB)
  tcs_daily.csv (5 MB)
  infy_1min.csv (60 MB)
  nifty_1min.csv (100 MB)
  ...
```

Problems:
1. Can't query efficiently ("Give me TCS closes for Jan 2025" = read entire CSV)
2. Can't update efficiently (rewrite entire file)
3. No point-in-time versioning (overwrites history)
4. Can't handle concurrent access (CSV locks)

### PostgreSQL + TimescaleDB for Time-Series

TimescaleDB is PostgreSQL optimized for *time-series data*. It uses:
- **Hypertables**: Automatically partitioned by time (1 week = 1 partition)
- **Compression**: Stores repeating values once
- **Continuous aggregates**: Pre-computed OHLCV at multiple timeframes

Installation (Ubuntu):

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Install TimescaleDB extension
sudo apt-get install timescaledb-postgresql-13

# Enable TimescaleDB in PostgreSQL
CREATE EXTENSION timescaledb;
```

#### Schema Design

```sql
-- ============================================
-- 1. INSTRUMENTS TABLE (static reference data)
-- ============================================
CREATE TABLE instruments (
    instrument_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    isin VARCHAR(12),
    name VARCHAR(255),
    exchange VARCHAR(10),  -- 'NSE', 'BSE', 'NYSE'
    sector VARCHAR(50),
    industry VARCHAR(50),
    market_cap_crores NUMERIC,
    listed_date DATE,
    status VARCHAR(20),  -- 'ACTIVE', 'SUSPENDED', 'DELISTED'
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_symbol (symbol),
    INDEX idx_exchange (exchange),
    INDEX idx_status (status)
);

-- ============================================
-- 2. OHLCV DATA (time-series, hypertable)
-- ============================================
CREATE TABLE ohlcv (
    time TIMESTAMP NOT NULL,
    instrument_id INTEGER NOT NULL,
    timeframe VARCHAR(10),  -- '1min', '5min', '1day'
    open NUMERIC(10,2) NOT NULL,
    high NUMERIC(10,2) NOT NULL,
    low NUMERIC(10,2) NOT NULL,
    close NUMERIC(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    
    -- Adjusted versions (after corporate actions)
    adj_open NUMERIC(10,2),
    adj_high NUMERIC(10,2),
    adj_low NUMERIC(10,2),
    adj_close NUMERIC(10,2),
    adj_volume BIGINT,
    
    FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id),
    
    INDEX idx_instrument_time (instrument_id, time DESC),
    INDEX idx_timeframe (timeframe)
);

-- Convert to TimescaleDB hypertable (automatic partitioning by week)
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

-- ============================================
-- 3. CORPORATE ACTIONS TABLE
-- ============================================
CREATE TABLE corporate_actions (
    action_id SERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL,
    action_date DATE NOT NULL,
    action_type VARCHAR(50),  -- 'STOCK_SPLIT', 'DIVIDEND', 'BONUS', 'RIGHTS', 'MERGER'
    factor NUMERIC(10,4),  -- For split: 2 (2-for-1) or 0.5 (1-for-2)
    ratio_numerator INTEGER,  -- Alternative: 1:3 split = num=1, denom=3
    ratio_denominator INTEGER,
    announcement_date DATE,
    ex_date DATE,
    record_date DATE,
    effective_date DATE,
    dividend_per_share NUMERIC(10,4),
    
    FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id),
    
    INDEX idx_instrument_date (instrument_id, action_date DESC)
);

-- ============================================
-- 4. UNIVERSE TABLE (which stocks to trade)
-- ============================================
CREATE TABLE universes (
    universe_id SERIAL PRIMARY KEY,
    name VARCHAR(100),  -- 'NIFTY50', 'NIFTY500', 'BANKNIFTY', etc.
    description TEXT,
    rebalance_frequency VARCHAR(50),  -- 'QUARTERLY', 'MONTHLY', 'DAILY'
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE universe_constituents (
    constituent_id SERIAL PRIMARY KEY,
    universe_id INTEGER NOT NULL,
    instrument_id INTEGER NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,  -- NULL = still current
    weight NUMERIC(5,2),  -- As of last rebalance (optional)
    
    FOREIGN KEY (universe_id) REFERENCES universes(universe_id),
    FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id),
    
    INDEX idx_universe_date (universe_id, start_date DESC),
    INDEX idx_instrument_date (instrument_id, start_date DESC)
);

-- ============================================
-- 5. REFERENCE DATA VERSIONS (data lineage)
-- ============================================
CREATE TABLE data_versions (
    version_id SERIAL PRIMARY KEY,
    table_name VARCHAR(100),
    version_number INTEGER,
    description TEXT,
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    is_current BOOLEAN DEFAULT TRUE,
    
    INDEX idx_table_date (table_name, created_at DESC)
);

-- ============================================
-- QUERIES FOR COMMON OPERATIONS
-- ============================================

-- Get latest day OHLCV for all NSE stocks
SELECT 
    i.symbol,
    o.time,
    o.close,
    o.volume
FROM ohlcv o
JOIN instruments i ON o.instrument_id = i.instrument_id
WHERE 
    i.exchange = 'NSE'
    AND o.timeframe = '1day'
    AND o.time = (SELECT MAX(time) FROM ohlcv WHERE timeframe = '1day')
ORDER BY o.close DESC;

-- Get OHLCV for a specific stock and date range
SELECT 
    time,
    open, high, low, close, volume
FROM ohlcv
WHERE 
    instrument_id = (SELECT instrument_id FROM instruments WHERE symbol = 'TCS')
    AND timeframe = '1day'
    AND time BETWEEN '2025-01-01' AND '2025-01-31'
ORDER BY time;

-- Get constituents of NIFTY50 at a specific date
SELECT 
    i.symbol,
    i.name,
    uc.weight
FROM universe_constituents uc
JOIN instruments i ON uc.instrument_id = i.instrument_id
JOIN universes u ON uc.universe_id = u.universe_id
WHERE 
    u.name = 'NIFTY50'
    AND uc.start_date <= '2025-01-15'
    AND (uc.end_date IS NULL OR uc.end_date > '2025-01-15')
ORDER BY uc.weight DESC;
```

### Python Integration with SQLAlchemy

```python
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import pandas as pd

Base = declarative_base()

class Instrument(Base):
    """ORM model for instruments table."""
    __tablename__ = 'instruments'
    
    instrument_id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    isin = Column(String(12))
    name = Column(String(255))
    exchange = Column(String(10))
    sector = Column(String(50))
    market_cap_crores = Column(Float)
    
    def __repr__(self):
        return f"<Instrument {self.symbol}>"

class OHLCV(Base):
    """ORM model for OHLCV table."""
    __tablename__ = 'ohlcv'
    
    time = Column(DateTime, primary_key=True)
    instrument_id = Column(Integer, ForeignKey('instruments.instrument_id'), primary_key=True)
    timeframe = Column(String(10), primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)  # Adjusted for corporate actions
    
    def __repr__(self):
        return f"<OHLCV {self.time} close={self.close}>"

class DataStore:
    """Abstraction layer for data operations."""
    
    def __init__(self, postgres_url: str = "postgresql://user:password@localhost/quant_db"):
        """
        Initialize database connection.
        
        Args:
            postgres_url: PostgreSQL connection string
        """
        self.engine = create_engine(postgres_url, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def insert_ohlcv(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Insert OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'TCS')
            df: DataFrame with columns [timestamp, timeframe, open, high, low, close, volume]
            
        Returns:
            Number of rows inserted
        """
        # Get instrument ID (create if not exists)
        instrument = self.session.query(Instrument).filter_by(symbol=symbol).first()
        if not instrument:
            instrument = Instrument(symbol=symbol, name=symbol)
            self.session.add(instrument)
            self.session.commit()
        
        # Insert OHLCV rows
        rows_inserted = 0
        for _, row in df.iterrows():
            ohlcv = OHLCV(
                time=row['timestamp'],
                instrument_id=instrument.instrument_id,
                timeframe=row.get('timeframe', '1day'),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            self.session.add(ohlcv)
            rows_inserted += 1
        
        self.session.commit()
        return rows_inserted
    
    def get_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1day'
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data for a symbol and date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timeframe: '1min', '5min', '1day', etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        query = self.session.query(OHLCV).join(Instrument).filter(
            Instrument.symbol == symbol,
            OHLCV.timeframe == timeframe,
            OHLCV.time >= start_date,
            OHLCV.time <= end_date
        ).order_by(OHLCV.time)
        
        df = pd.read_sql(query.statement, self.engine)
        return df
    
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest close price for a symbol."""
        latest = self.session.query(OHLCV).join(Instrument).filter(
            Instrument.symbol == symbol
        ).order_by(OHLCV.time.desc()).first()
        
        return latest.close if latest else None
    
    def get_universe_constituents(
        self,
        universe_name: str,
        date: datetime
    ) -> list:
        """Get list of symbols in a universe at a specific date."""
        # Query would join universes, universe_constituents, and instruments
        # Return list of symbol strings
        pass

# Usage
store = DataStore()

# Insert daily OHLCV for TCS
tcs_df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=20, freq='D'),
    'timeframe': '1day',
    'open': np.random.uniform(3000, 3100, 20),
    'high': np.random.uniform(3100, 3200, 20),
    'low': np.random.uniform(2900, 3000, 20),
    'close': np.random.uniform(3000, 3100, 20),
    'volume': 5000000
})

rows = store.insert_ohlcv('TCS', tcs_df)
print(f"Inserted {rows} rows")

# Retrieve data
tcs_data = store.get_ohlcv('TCS', datetime(2025, 1, 1), datetime(2025, 1, 31))
print(tcs_data)

# Get current price
current_price = store.get_latest_price('TCS')
print(f"TCS current price: {current_price}")
```

### Parquet for Research

Parquet is a columnar format optimized for analytics. Much better than CSV for research.

```python
import pyarrow.parquet as pq
import pandas as pd

def save_ohlcv_parquet(
    df: pd.DataFrame,
    output_path: str,
    compression: str = 'snappy'
):
    """
    Save OHLCV data as Parquet.
    
    Args:
        df: OHLCV DataFrame
        output_path: Path to save (e.g., '/data/tcs_daily.parquet')
        compression: 'snappy', 'gzip', or 'none'
    """
    df.to_parquet(
        output_path,
        compression=compression,
        index=False,
        engine='pyarrow'
    )

def load_ohlcv_parquet(path: str) -> pd.DataFrame:
    """Load Parquet file efficiently."""
    return pd.read_parquet(path)

# Usage
save_ohlcv_parquet(tcs_data, '/data/research/tcs_daily.parquet')

# Load subset of columns (Parquet is columnar = efficient)
df = pd.read_parquet(
    '/data/research/tcs_daily.parquet',
    columns=['timestamp', 'close', 'volume']
)
```

### DuckDB for Analytics

DuckDB is an in-process SQL database. Great for quick analytics without spinning up PostgreSQL.

```python
import duckdb

# Create an in-memory DuckDB database
con = duckdb.connect(':memory:')

# Load Parquet directly into SQL
result = con.execute("""
    SELECT 
        DATE_TRUNC('week', timestamp) as week,
        AVG(close) as avg_close,
        MAX(volume) as max_volume
    FROM read_parquet('/data/research/*.parquet')
    WHERE symbol = 'TCS'
    GROUP BY week
    ORDER BY week DESC
""").df()

print(result)
```

### Data Versioning: Immutable Raw + Transformation Layers

Architecture:
```
/data/
  raw/                              # Original, never modified
    nse_bhavcopy/
      2025-01-01_bhavcopy.csv.gz    # Immutable, compressed
      2025-01-02_bhavcopy.csv.gz
      ...
  
  staging/                          # After basic cleaning
    nse_ohlcv_v1/
      2025_01_ohlcv.parquet
      2025_02_ohlcv.parquet
      ...
  
  processed/                        # After corporate actions adjustment
    nse_ohlcv_adjusted_v1/
      2025_01_ohlcv_adj.parquet
      ...
  
  features/                         # Feature engineering (slow to compute)
    technical_indicators_v2/
      2025_01_indicators.parquet
      ...
```

```python
from pathlib import Path
from datetime import datetime
import hashlib

class DataLakeManager:
    """Manage versioned data layers."""
    
    def __init__(self, base_path: str = '/data'):
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create layer directories if they don't exist."""
        for layer in ['raw', 'staging', 'processed', 'features']:
            (self.base_path / layer).mkdir(parents=True, exist_ok=True)
    
    def save_raw(self, data_type: str, date: datetime, df: pd.DataFrame) -> Path:
        """
        Save raw data (immutable).
        
        Args:
            data_type: 'nse_bhavcopy', 'kite_minute', etc.
            date: Date of data
            df: DataFrame
            
        Returns:
            Path where data was saved
        """
        date_str = date.strftime('%Y-%m-%d')
        path = self.base_path / 'raw' / data_type / f'{date_str}_raw.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(path, compression='snappy', index=False)
        
        # Create checksum for integrity
        checksum = self._compute_checksum(path)
        (path.parent / f'{date_str}_raw.parquet.sha256').write_text(checksum)
        
        return path
    
    def save_staging(
        self,
        data_type: str,
        version: int,
        df: pd.DataFrame
    ) -> Path:
        """Save staging data (after basic cleaning)."""
        year_month = df['timestamp'].min().strftime('%Y_%m')
        path = self.base_path / 'staging' / f'{data_type}_v{version}' / f'{year_month}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(path, compression='snappy', index=False)
        return path
    
    def save_processed(
        self,
        data_type: str,
        version: int,
        df: pd.DataFrame
    ) -> Path:
        """Save processed data (final, analysis-ready)."""
        year_month = df['timestamp'].min().strftime('%Y_%m')
        path = self.base_path / 'processed' / f'{data_type}_v{version}' / f'{year_month}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(path, compression='snappy', index=False)
        return path
    
    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def load_processed(self, data_type: str, version: int, year_month: str) -> pd.DataFrame:
        """Load processed data."""
        path = self.base_path / 'processed' / f'{data_type}_v{version}' / f'{year_month}.parquet'
        return pd.read_parquet(path)

# Usage
lake = DataLakeManager('/data/quant')

# Save raw daily NSE data
lake.save_raw('nse_bhavcopy', datetime(2025, 1, 15), raw_bhavcopy_df)

# After cleaning:
lake.save_staging('nse_ohlcv', version=1, df=cleaned_df)

# After corporate action adjustment:
lake.save_processed('nse_ohlcv_adjusted', version=1, df=adjusted_df)
```

### Full Pipeline: Raw NSE → Clean Adjusted

```python
def full_data_pipeline(
    raw_nse_path: str,
    output_version: int = 1
) -> pd.DataFrame:
    """
    Complete pipeline from NSE raw bhavcopy to analysis-ready adjusted data.
    
    Steps:
    1. Load raw bhavcopy
    2. Validate and clean
    3. Apply corporate actions
    4. Save intermediate stages
    5. Return final clean data
    """
    lake = DataLakeManager()
    
    # STEP 1: Load raw
    print("Step 1: Loading raw bhavcopy...")
    raw_df = pd.read_csv(raw_nse_path)
    
    # Parse NSE format
    raw_df = raw_df[raw_df['SERIES'] == 'EQ'].copy()
    raw_df['timestamp'] = pd.to_datetime('2025-01-15')  # Add actual date
    raw_df.columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    
    # Save raw immutable copy
    lake.save_raw('nse_bhavcopy', raw_df['timestamp'].iloc[0], raw_df)
    
    # STEP 2: Validate and clean
    print("Step 2: Validating and cleaning...")
    validator = DataValidator()
    is_valid = validator.validate_ohlcv(raw_df)
    
    # Remove outliers
    outlier_mask = detect_price_outliers(raw_df)['is_outlier']
    clean_df = raw_df[~outlier_mask].copy()
    
    print(f"Removed {outlier_mask.sum()} outlier rows")
    
    # Handle missing data
    clean_df = handle_missing_data(clean_df, method='ffill')
    
    # Save staging
    lake.save_staging('nse_ohlcv', version=output_version, df=clean_df)
    
    # STEP 3: Apply corporate actions
    print("Step 3: Adjusting for corporate actions...")
    
    # In practice, fetch corporate actions from database
    # For now, assume no recent corporate actions
    corporate_actions_by_symbol = {}  # Symbol -> List[CorporateAction]
    
    adjusted_df_list = []
    for symbol in clean_df['symbol'].unique():
        symbol_data = clean_df[clean_df['symbol'] == symbol].copy()
        
        actions = corporate_actions_by_symbol.get(symbol, [])
        if actions:
            symbol_data = apply_corporate_actions(symbol_data, actions)
        
        # Compute adjusted prices
        symbol_data['adj_open'] = symbol_data['open']
        symbol_data['adj_high'] = symbol_data['high']
        symbol_data['adj_low'] = symbol_data['low']
        symbol_data['adj_close'] = symbol_data['close']
        symbol_data['adj_volume'] = symbol_data['volume']
        
        adjusted_df_list.append(symbol_data)
    
    adjusted_df = pd.concat(adjusted_df_list, ignore_index=True)
    
    # Save processed final version
    lake.save_processed('nse_ohlcv_adjusted', version=output_version, df=adjusted_df)
    
    print(f"Pipeline complete: {len(adjusted_df)} rows, {adjusted_df['symbol'].nunique()} symbols")
    return adjusted_df

# Execute pipeline
final_data = full_data_pipeline('/data/raw/eq_26-JAN-2025.csv', output_version=1)
print(final_data.head())
```

---

## Module 9.4: Alternative Bar Types (López de Prado)

### The Problem with Time Bars

Time bars (1-minute, 1-day candles) have a critical flaw: they aggregate trades based on *wall clock time*, not market activity.

Consider a quiet day vs. earnings day:
- **Quiet day**: 10 trades in 1 minute → candle with 10 units of information
- **Earnings day**: 1000 trades in 1 minute → candle with 1000 units of information

Both are bucketed into 1-minute bars, but they have radically different *information content*.

**López de Prado's insight**: Use bars that aggregate based on market activity, not time.

### Time Bars (Baseline)

```python
def time_bars(
    ticks: pd.DataFrame,
    period: str = '1min'
) -> pd.DataFrame:
    """
    Standard time-based bars.
    
    Args:
        ticks: DataFrame with [timestamp, price, volume]
        period: '1min', '5min', '1hour', '1day'
        
    Returns:
        DataFrame with OHLCV bars
    """
    ticks = ticks.copy()
    ticks['period'] = ticks['timestamp'].dt.floor(period)
    
    bars = ticks.groupby('period').agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    
    bars.columns = ['open', 'high', 'low', 'close', 'volume']
    return bars

# Problem: No control over information content per bar
```

**Why time bars are suboptimal**:
- Ignore market microstructure
- Lead to heteroskedasticity (varying volatility across bars)
- Market overreacts to quiet periods, underreacts to busy periods
- Violates implicit uniform information assumption in many models

### Tick Bars

A new bar starts every N trades.

```python
def tick_bars(
    ticks: pd.DataFrame,
    n_ticks: int = 100
) -> pd.DataFrame:
    """
    Generate bars based on number of trades (ticks).
    
    One bar = every n_ticks trades.
    
    Args:
        ticks: DataFrame with [timestamp, price, volume]
        n_ticks: Number of trades per bar
        
    Returns:
        DataFrame with OHLCV bars
    """
    ticks = ticks.copy()
    ticks['tick_count'] = range(len(ticks))
    ticks['bar_id'] = ticks['tick_count'] // n_ticks
    
    bars = ticks.groupby('bar_id').agg({
        'timestamp': 'last',
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    bars = bars.reset_index(drop=True)
    
    return bars

# Each bar has exactly n_ticks trades
# Problem: Ignores volume (100 trades of 1 share = 100 trades of 10 shares)
```

### Volume Bars

A new bar starts every N shares traded.

```python
def volume_bars(
    ticks: pd.DataFrame,
    volume_threshold: int = 1_000_000
) -> pd.DataFrame:
    """
    Generate bars based on cumulative volume.
    
    One bar = every volume_threshold shares traded.
    
    Args:
        ticks: DataFrame with [timestamp, price, volume]
        volume_threshold: Shares per bar
        
    Returns:
        DataFrame with OHLCV bars
    """
    ticks = ticks.copy()
    ticks['cumulative_volume'] = ticks['volume'].cumsum()
    ticks['bar_id'] = ticks['cumulative_volume'] // volume_threshold
    
    bars = ticks.groupby('bar_id').agg({
        'timestamp': 'last',
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    return bars

# Each bar represents volume_threshold shares
# Better than tick bars (accounts for trade size)
# Problem: Ignores price (100 shares @ $1 = different from 100 shares @ $1000)
```

### Dollar Bars

A new bar starts every N dollars worth of trades.

```python
def dollar_bars(
    ticks: pd.DataFrame,
    dollar_threshold: float = 1_000_000.0
) -> pd.DataFrame:
    """
    Generate bars based on cumulative dollar value traded.
    
    One bar = every dollar_threshold in notional value.
    
    Formula: Dollar value = price × volume
    
    Args:
        ticks: DataFrame with [timestamp, price, volume]
        dollar_threshold: Dollar amount per bar
        
    Returns:
        DataFrame with OHLCV bars
    """
    ticks = ticks.copy()
    
    # Compute dollar value of each trade
    ticks['dollar_value'] = ticks['price'] * ticks['volume']
    ticks['cumulative_dollars'] = ticks['dollar_value'].cumsum()
    ticks['bar_id'] = ticks['cumulative_dollars'] // dollar_threshold
    
    bars = ticks.groupby('bar_id').agg({
        'timestamp': 'last',
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum',
        'dollar_value': 'sum'
    })
    
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dollars']
    
    return bars

# Each bar represents dollar_threshold of notional value
# López de Prado recommends this as superior to time/tick/volume bars
```

### Information-Driven Bars

Most advanced: Use *information entropy* to determine bar boundaries.

**Entropy Bars**: New bar when *entropy* of price direction exceeds threshold

```python
def entropy_bars(
    ticks: pd.DataFrame,
    entropy_threshold: float = 1.0
) -> pd.DataFrame:
    """
    Generate bars based on Shannon entropy of price movement direction.
    
    High entropy = high uncertainty = useful information.
    New bar starts when cumulative entropy > entropy_threshold.
    
    Shannon Entropy: $ H = -\sum p_i \log(p_i) $
    where p_i = probability of outcome i
    
    For price movement:
    - Probability of up move
    - Probability of down move
    
    H = -P(up) * log(P(up)) - P(down) * log(P(down))
    H ranges from 0 (certain) to ln(2) ≈ 0.693 (maximum entropy, 50/50)
    """
    ticks = ticks.copy()
    
    # Compute price direction (up=1, down/flat=0)
    ticks['price_change'] = ticks['price'].diff()
    ticks['is_up'] = (ticks['price_change'] > 0).astype(int)
    
    # Compute running entropy
    cumulative_entropy = 0
    bar_ids = []
    current_bar = 0
    
    n_up = 0
    n_down = 0
    
    for idx in range(len(ticks)):
        if ticks.iloc[idx]['is_up']:
            n_up += 1
        else:
            n_down += 1
        
        total = n_up + n_down
        
        if total > 0:
            p_up = n_up / total
            p_down = n_down / total
            
            # Entropy: max when p_up ≈ p_down ≈ 0.5
            entropy = -p_up * np.log(p_up + 1e-8) - p_down * np.log(p_down + 1e-8)
        else:
            entropy = 0
        
        cumulative_entropy += entropy
        
        # New bar when entropy threshold exceeded
        if cumulative_entropy > entropy_threshold:
            current_bar += 1
            cumulative_entropy = 0
            n_up = 0
            n_down = 0
        
        bar_ids.append(current_bar)
    
    ticks['bar_id'] = bar_ids
    
    bars = ticks.groupby('bar_id').agg({
        'timestamp': 'last',
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    return bars
```

**Imbalance Bars**: New bar when buy/sell *imbalance* exceeds threshold

```python
def imbalance_bars(
    ticks: pd.DataFrame,
    imbalance_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Generate bars based on buy/sell volume imbalance.
    
    Imbalance = |V_buy - V_sell| / (V_buy + V_sell)
    
    High imbalance indicates strong directional pressure.
    New bar when cumulative imbalance > threshold.
    
    Args:
        ticks: DataFrame with [timestamp, price, volume, side]
               side = 'BUY' or 'SELL'
        imbalance_threshold: Value between 0 and 1
    """
    ticks = ticks.copy()
    
    cumulative_imbalance = 0
    bar_ids = []
    current_bar = 0
    
    v_buy = 0
    v_sell = 0
    
    for idx in range(len(ticks)):
        side = ticks.iloc[idx]['side']
        volume = ticks.iloc[idx]['volume']
        
        if side == 'BUY':
            v_buy += volume
        else:
            v_sell += volume
        
        total = v_buy + v_sell
        
        if total > 0:
            imbalance = abs(v_buy - v_sell) / total
        else:
            imbalance = 0
        
        cumulative_imbalance += imbalance
        
        # New bar when imbalance threshold exceeded
        if cumulative_imbalance > imbalance_threshold:
            current_bar += 1
            cumulative_imbalance = 0
            v_buy = 0
            v_sell = 0
        
        bar_ids.append(current_bar)
    
    ticks['bar_id'] = bar_ids
    
    bars = ticks.groupby('bar_id').agg({
        'timestamp': 'last',
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    
    bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    return bars
```

### Statistical Properties Comparison

```python
def compare_bar_types(ticks: pd.DataFrame):
    """
    Compare statistical properties of different bar types.
    
    Metrics:
    - Bar count
    - Average bar duration
    - Volatility (std of returns)
    - Autocorrelation of returns
    - Kurtosis (fat tails)
    """
    results = {}
    
    # Time bars
    time_bar_df = time_bars(ticks, period='1min')
    results['time_bars'] = {
        'count': len(time_bar_df),
        'returns': time_bar_df['close'].pct_change(),
        'volatility': time_bar_df['close'].pct_change().std(),
        'kurtosis': time_bar_df['close'].pct_change().kurtosis(),
        'autocorr_lag1': time_bar_df['close'].pct_change().autocorr(lag=1),
    }
    
    # Tick bars
    tick_bar_df = tick_bars(ticks, n_ticks=100)
    results['tick_bars'] = {
        'count': len(tick_bar_df),
        'returns': tick_bar_df['close'].pct_change(),
        'volatility': tick_bar_df['close'].pct_change().std(),
        'kurtosis': tick_bar_df['close'].pct_change().kurtosis(),
        'autocorr_lag1': tick_bar_df['close'].pct_change().autocorr(lag=1),
    }
    
    # Volume bars
    volume_bar_df = volume_bars(ticks, volume_threshold=100_000)
    results['volume_bars'] = {
        'count': len(volume_bar_df),
        'returns': volume_bar_df['close'].pct_change(),
        'volatility': volume_bar_df['close'].pct_change().std(),
        'kurtosis': volume_bar_df['close'].pct_change().kurtosis(),
        'autocorr_lag1': volume_bar_df['close'].pct_change().autocorr(lag=1),
    }
    
    # Dollar bars
    dollar_bar_df = dollar_bars(ticks, dollar_threshold=500_000)
    results['dollar_bars'] = {
        'count': len(dollar_bar_df),
        'returns': dollar_bar_df['close'].pct_change(),
        'volatility': dollar_bar_df['close'].pct_change().std(),
        'kurtosis': dollar_bar_df['close'].pct_change().kurtosis(),
        'autocorr_lag1': dollar_bar_df['close'].pct_change().autocorr(lag=1),
    }
    
    # Print comparison
    print("\n=== BAR TYPE COMPARISON ===\n")
    for bar_type in ['time_bars', 'tick_bars', 'volume_bars', 'dollar_bars']:
        data = results[bar_type]
        print(f"{bar_type}:")
        print(f"  Bar count: {data['count']}")
        print(f"  Volatility: {data['volatility']:.4f}")
        print(f"  Kurtosis: {data['kurtosis']:.4f}")
        print(f"  Autocorr(lag1): {data['autocorr_lag1']:.4f}")
        print()
    
    return results
```

### Comparison Summary

| Aspect | Time Bars | Tick Bars | Volume Bars | Dollar Bars | Entropy Bars |
|--------|-----------|-----------|------------|-------------|--------------|
| Information Homogeneity | ❌ Low | ✓ Medium | ✓ Medium | ✓✓ High | ✓✓✓ Very High |
| Computational Cost | ✓ Low | ✓ Low | ✓ Low | ✓ Low | ❌ Medium |
| Market Microstructure | ❌ Ignores | ✓ Partial | ✓ Accounts for volume | ✓✓ Accounts for $ value | ✓✓✓ Accounts for momentum |
| Backtest Performance | Baseline | +5-10% | +10-20% | +15-30% | +20-40%* |
| Real-World Recommendation | Use for learning | Better | Good for ML | **Best** | Research only |

*Backtests on entropy bars often show *overoptimism* due to look-ahead bias (entropy is computed with full information).

### Production Implementation

```python
class BarFactory:
    """Factory for generating different bar types."""
    
    def __init__(self, ticks: pd.DataFrame):
        """
        Args:
            ticks: DataFrame with [timestamp, price, volume] at minimum
        """
        self.ticks = ticks.copy()
    
    def generate(self, bar_type: str, **kwargs) -> pd.DataFrame:
        """
        Generate bars of specified type.
        
        Args:
            bar_type: 'time', 'tick', 'volume', 'dollar', 'entropy', 'imbalance'
            **kwargs: Type-specific parameters
                - time: period='1min'
                - tick: n_ticks=100
                - volume: volume_threshold=1000000
                - dollar: dollar_threshold=1000000
                - entropy: entropy_threshold=1.0
                - imbalance: imbalance_threshold=0.3
        """
        if bar_type == 'time':
            return time_bars(self.ticks, period=kwargs.get('period', '1min'))
        
        elif bar_type == 'tick':
            return tick_bars(self.ticks, n_ticks=kwargs.get('n_ticks', 100))
        
        elif bar_type == 'volume':
            return volume_bars(self.ticks, volume_threshold=kwargs.get('volume_threshold', 1_000_000))
        
        elif bar_type == 'dollar':
            return dollar_bars(self.ticks, dollar_threshold=kwargs.get('dollar_threshold', 1_000_000))
        
        elif bar_type == 'entropy':
            return entropy_bars(self.ticks, entropy_threshold=kwargs.get('entropy_threshold', 1.0))
        
        elif bar_type == 'imbalance':
            return imbalance_bars(self.ticks, imbalance_threshold=kwargs.get('imbalance_threshold', 0.3))
        
        else:
            raise ValueError(f"Unknown bar type: {bar_type}")

# Usage
factory = BarFactory(nse_ticks)

# Generate different bar types for comparison
time_bars_df = factory.generate('time', period='1min')
dollar_bars_df = factory.generate('dollar', dollar_threshold=500_000)
entropy_bars_df = factory.generate('entropy', entropy_threshold=1.0)

# Use dollar_bars in production ML pipeline
for idx, bar in dollar_bars_df.iterrows():
    features = extract_features(bar)
    prediction = model.predict(features)
    execute_trade(prediction)
```

---

## [VISUALIZATION]: Data Engineering Pipeline

```
RAW DATA SOURCES
      │
      ├─ NSE Bhavcopy (daily)
      ├─ Zerodha Kite API (minute/tick)
      ├─ Yahoo Finance
      └─ Alternative data (satellite, news, etc.)
      │
      ▼
INGESTION & VALIDATION
      │
      ├─ Check OHLCV order (O,C between L,H)
      ├─ Detect outliers (Hampel filter)
      ├─ Validate timestamps
      └─ Data quality monitoring
      │
      ▼
CORPORATE ACTIONS ADJUSTMENT
      │
      ├─ Stock splits (adjust all prices)
      ├─ Dividends (adjust close)
      └─ Rights issues
      │
      ▼
STORAGE LAYERS
      │
      ├─ Raw (immutable): /data/raw/*.parquet
      ├─ Staging (cleaned): PostgreSQL staging tables
      ├─ Processed (adjusted): /data/processed/*.parquet
      └─ Features (engineered): /data/features/*.parquet
      │
      ▼
ALTERNATIVE BAR TYPES
      │
      ├─ Time bars (1-min)
      ├─ Tick bars (N trades)
      ├─ Dollar bars (N dollars)
      └─ Entropy bars (information-driven)
      │
      ▼
ANALYSIS-READY FEATURES
      │
      ├─ Returns
      ├─ Technical indicators
      ├─ Fundamental ratios
      └─ Risk metrics
      │
      ▼
ML MODEL TRAINING
```

---

## [WARNING]: Common Data Pitfalls

**WARNING 1: Look-Ahead Bias**
Using data that wasn't available at trade time inflates backtest returns by 10-50%+.
- ❌ Use adjusted close computed weeks later
- ❌ Use earnings announced yesterday (trade today)
- ✓ Use only unadjusted, previously-published prices
- ✓ Add 1-2 day lag for fundamental data

**WARNING 2: Survivorship Bias**
Testing on only successful companies inflates returns by 5-20%.
- ❌ Download current NSE500 constituents, backtest on 10 years of data
- ✓ Use historical universe definitions (constituents at each point in time)
- ✓ Include delisted stocks with bankruptcy returns

**WARNING 3: Corporate Action Mishandling**
A 2:1 stock split means prices pre-split are double post-split. Ignoring this creates fake 50% returns.
- ❌ Use unadjusted prices directly
- ✓ Maintain and apply corporate action adjustments
- ✓ Document which prices (unadjusted/adjusted) you're using

**WARNING 4: Stale Data**
Using 1-day-old data as "current" in a 1-minute strategy creates execution slippage.
- ❌ Backtest with T+1 delayed data, deploy with real-time data
- ✓ Match latency: backtest with same data freshness as production

**WARNING 5: Data Quality Monitoring Neglect**
Production data pipelines fail silently. Your strategy might be trading on corrupted data.
- ❌ Assume data is always clean
- ✓ Run automated validation checks on every ingestion
- ✓ Log validation metrics (missing %, outlier %, delay hours)
- ✓ Set up alerts for anomalies

---

## Exercises

### Exercise 9.1: Build OHLCV from Ticks (30 minutes)

Download 1 day of NSE tick data for TCS (use Zerodha Kite API or sample data). Implement `compute_ohlcv_from_ticks()` for 1-min, 5-min, and 1-hour bars. Verify:
- O ≤ C ≤ H or O ≥ C ≤ H (OHLC order)
- Returns for each timeframe sum correctly
- Volume aggregation is consistent

**Expected output**: 390 1-min bars, 78 5-min bars, 6-7 1-hour bars for a trading day.

### Exercise 9.2: Corporate Action Adjustment (45 minutes)

Create sample OHLCV data spanning a 2:1 stock split. Implement `apply_corporate_actions()` and verify:
- Pre-split prices are halved
- Pre-split volume is doubled
- Returns across split boundary are continuous (no artificial drops)

**Test case**: Price = 1000 before split, 500 after. Return should be 0%, not -50%.

### Exercise 9.3: Data Quality Pipeline (1 hour)

Build a `DataValidator` class that checks:
1. OHLC logical order
2. Price range (no negatives, no 1000x spikes)
3. Volume ≥ 0
4. Timestamps ordered
5. No duplicates
6. Intraday gaps < 2 hours

Test on a dataset with intentional corruptions (negative prices, out-of-order OHLC, etc.). Ensure your validator catches ≥90% of corruptions.

### Exercise 9.4: PostgreSQL Time-Series Setup (1.5 hours)

1. Install PostgreSQL + TimescaleDB locally
2. Create the schema from Module 9.3 (instruments, OHLCV hypertable, corporate_actions)
3. Insert 100 days of OHLCV for 10 NSE stocks
4. Write 5 SQL queries:
   - Get latest close for all stocks
   - Get OHLCV for TCS in January 2025
   - Get stocks with volume > 10M on latest day
   - Get NIFTY50 constituents at a specific date
   - Get stocks that underwent corporate actions in past 6 months

### Exercise 9.5: Bar Type Comparison (2 hours)

1. Download 1 week of minute-level OHLCV for Nifty 50 index
2. Generate bars using all 4 types (time, tick, volume, dollar)
3. Compute for each:
   - Number of bars
   - Volatility of returns (std of % changes)
   - Kurtosis (fat tails indicator)
   - Autocorrelation at lag-1
4. Create visualization comparing statistics
5. Observation: Which bar type has most homogeneous volatility? Why?

**Expected finding**: Dollar bars typically have lowest autocorrelation (best for ML), entropy bars highest information content (but harder to compute).

---

## Summary

This chapter covered the complete data engineering pipeline:

**Data Types & Sources**: OHLCV computation, TAQ data, fundamental data, alternative data, and sources for Indian/US markets.

**Quality & Cleaning**: Survivorship & look-ahead bias, corporate action adjustment, outlier detection, validation, monitoring.

**Storage**: PostgreSQL+TimescaleDB for time-series (superior to CSV), Parquet for research, DuckDB for analytics, immutable raw + transformation layers.

**Alternative Bars**: López de Prado's insight—bars should normalize information content. Dollar bars in production, entropy bars in research.

In production, expect 70% of development time on data engineering, 20% on strategies, 10% on execution. Get data right, and everything else becomes straightforward.

Next: Chapter 10 covers feature engineering and target creation.

---

## Key Takeaways

1. **OHLCV is not raw**: Exchanges give tick data; you compute OHLCV via aggregation.

2. **Four classes of bias matter**:
   - Survivorship (only successful companies)
   - Look-ahead (using future information)
   - Corporate actions (splits/dividends distort prices)
   - Selection (backtesting only on filtered universes)

3. **Use PostgreSQL+TimescaleDB for production**, not CSV files. You'll query efficiency matters when managing 10 years × 500 stocks × 390 daily bars.

4. **Dollar bars outperform time bars** in most ML applications (better information homogeneity).

5. **Data versioning is non-negotiable**: Separate raw (immutable) → staging (cleaned) → processed (adjusted) layers. Enables reproducibility and auditing.

6. **Automate data quality monitoring**: Log validation metrics daily, set alerts for anomalies. Silent data corruption is worse than no data.


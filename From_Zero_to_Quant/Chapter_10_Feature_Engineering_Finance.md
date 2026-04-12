# Chapter 10: Feature Engineering for Finance

## Introduction

Feature engineering is the **art and science of transforming raw data into meaningful signals** that machine learning models can exploit to predict future price movements. While deep learning practitioners in computer vision or NLP may debate whether hand-crafted features are necessary (the promise of deep learning is to learn representations automatically), in quantitative finance, feature engineering remains absolutely critical.

Why? Because:

1. **Finance markets have deep structural properties** — the physics of supply/demand, volatility clustering, mean reversion, and momentum are well-understood. We should encode this domain knowledge.
2. **Data scarcity is severe** — a Zerodha backtest on NSE with 1-minute bars for 5 years is only ~1.3 million rows. Compare this to ImageNet's 14 million images. Deep learning without careful feature engineering will overfit catastrophically.
3. **Interpretability matters** — regulators, risk managers, and your own due diligence require understanding *why* a model trades. A feature encoding "mean reversion strength" explains itself; a latent representation in a neural network does not.
4. **Real-time constraints** — feature computation must complete in milliseconds. A neural network that takes 500ms to inference is useless for live trading; a well-designed feature set can be computed in microseconds.

This chapter transforms you from "applying scikit-learn's StandardScaler" to designing feature systems that separate profitable trading systems from the 95% that fail. We'll implement 50+ production-grade features across four domains: **price/volume**, **cross-sectional**, **fundamental**, and **pipeline infrastructure**.

---

## Module 10.1: Price and Volume Features

### Why Raw Prices Are Never Features

Before we engineer features, understand why raw prices are fundamentally broken as ML inputs:

**Raw Price Problems:**
1. **Non-stationarity** — A price of ₹2000 in 2015 means something completely different from ₹2000 in 2025 due to inflation, corporate actions, market growth.
2. **Absolute scale dependence** — A stock at ₹100 is treated completely differently from ₹10000, even if their risk/return profiles are identical.
3. **Corporate actions** — Stock splits (Zerodha adjusts automatically) and dividends create discontinuities that confuse models.
4. **Market structure changes** — An NSE stock's regime in 2015 (no FII flows, lower volatility) is fundamentally different from 2025.

**Solution:** Always use transformed features: returns, log-returns, price ratios, and volatility.

### Returns at Multiple Horizons

Returns are the foundation of all quantitative features. They are **forward-looking** (capturing price appreciation) and **scale-invariant**.

#### Simple Returns Formula

$$R_t^{(h)} = \frac{P_{t+h} - P_t}{P_t}$$

Where:
- $R_t^{(h)}$ is the simple return from time $t$ to $t+h$
- $P_t$ is the price at time $t$
- $h$ is the horizon (1 day, 5 days, etc.)

#### Log Returns Formula

$$r_t^{(h)} = \log\left(\frac{P_{t+h}}{P_t}\right) = \log(P_{t+h}) - \log(P_t)$$

**When to use log returns:**
- Modeling continuous compounding (more mathematically correct)
- Volatility estimation (log returns are more symmetric)
- Multi-period aggregation (log returns are additive: $r_1^{(2)} = r_1^{(1)} + r_2^{(1)}$)

**When to use simple returns:**
- Actual P&L calculations (positions × simple returns = actual rupees gained/lost)
- Regulatory reporting
- Portfolio construction

We'll use both, but emphasize log returns for modeling.

#### Implementation: Multi-Horizon Returns

```python
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import timedelta

def compute_returns(
    prices: pd.Series,
    horizons: List[int] = [1, 5, 21, 63, 126, 252],
    return_type: str = 'log'
) -> pd.DataFrame:
    """
    Compute returns at multiple horizons.
    
    Parameters
    ----------
    prices : pd.Series
        Close prices indexed by datetime. Must be sorted ascending.
    horizons : List[int]
        Trading day horizons (1=daily, 5=weekly, 21=monthly, 63=quarterly, 
        126=half-yearly, 252=annual on NSE)
    return_type : str
        'log' for log returns, 'simple' for simple returns
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ret_1d', 'ret_5d', ..., 'ret_252d']
    
    Example
    -------
    >>> prices = pd.Series([100, 101, 102, 103], 
    ...                     index=pd.date_range('2025-01-01', periods=4))
    >>> returns = compute_returns(prices, horizons=[1, 2])
    >>> returns['ret_1d'].iloc[0]  # Next day return from t=0
    0.00995...  # log(101/100)
    """
    returns_df = pd.DataFrame(index=prices.index)
    
    for h in horizons:
        if return_type == 'log':
            # Forward-looking log return
            returns_df[f'ret_{h}d'] = np.log(prices / prices.shift(-h))
        else:
            # Forward-looking simple return
            returns_df[f'ret_{h}d'] = (prices.shift(-h) - prices) / prices
    
    return returns_df


# Validation example for NSE trading
if __name__ == "__main__":
    # Simulate INFY close prices (NSE trading days)
    dates = pd.date_range('2025-01-01', periods=252, freq='B')  # B = business day
    prices = pd.Series(
        np.exp(np.random.randn(252).cumsum() * 0.01) * 1500,  # Log-normal prices
        index=dates
    )
    
    returns = compute_returns(prices, horizons=[1, 5, 21, 63, 126, 252])
    print(f"Returns shape: {returns.shape}")
    print(f"\n1-day returns (first 5):\n{returns['ret_1d'].head()}")
    print(f"\nMean annual return: {returns['ret_252d'].mean():.4f}")
    print(f"Annual return volatility: {returns['ret_252d'].std():.4f}")
```

**Key observations:**
- We use `shift(-h)` to get *forward-looking* returns (predicting future, not past).
- Horizons scale: 252 trading days = 1 year on NSE (India doesn't trade ~13 days/year).
- Returns are NaN at the end (can't compute 252-day return from last day); we'll handle this in the pipeline module.

### Volatility Features

Volatility is the **second moment** of returns and is crucial because:
1. It changes over time (volatility clustering)
2. High volatility → larger price moves → more trading opportunity
3. Volatility mean-reverts (very high vol is unsustainable)

#### Rolling Standard Deviation (Parkinson-free)

$$\sigma_t^{(w)} = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1}(r_{t-i} - \bar{r})^2}$$

Where $w$ is the window size.

#### Parkinson Volatility Estimator

More efficient than standard deviation because it uses high/low (not just close).

$$\sigma_t^{Parkinson} = \sqrt{\frac{1}{4\ln(2)} \cdot \frac{1}{n}\sum_{i=1}^{n}\left[\ln\left(\frac{H_i}{L_i}\right)\right]^2}$$

Where:
- $H_i, L_i$ are high and low on day $i$
- Uses intraday range without depending on opening/closing
- Efficient estimator: requires ~3x fewer days of data than close-based for same accuracy

#### Garman-Klass Volatility Estimator

Combines opens, highs, lows, closes for superior efficiency:

$$\sigma_t^{GK} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left[0.5\ln\left(\frac{H_i}{L_i}\right)^2 - (2\ln(2)-1)\ln\left(\frac{C_i}{O_i}\right)^2\right]}$$

Where $O_i, C_i$ are open and close on day $i$.

#### Yang-Zhang Volatility Estimator

The gold standard, combining overnight gaps and intraday range:

$$\sigma_t^{YZ} = \sqrt{\sigma_{overnight}^2 + \sigma_{scale} \cdot \sigma_{intraday}^2}$$

Where:
- $\sigma_{overnight}^2 = \frac{1}{n}\sum_{i=1}^{n}[\ln(O_i/C_{i-1})]^2$ (gap volatility)
- $\sigma_{intraday}^2 = \frac{1}{n}\sum_{i=1}^{n}[\ln(H_i/C_i)\ln(H_i/O_i) + \ln(L_i/C_i)\ln(L_i/O_i)]$
- $\sigma_{scale}$ balances the two components

#### Implementation: All Volatility Features

```python
def compute_volatility_features(
    ohlc: pd.DataFrame,
    windows: List[int] = [5, 21, 63, 126, 252],
    include_estimators: bool = True
) -> pd.DataFrame:
    """
    Compute rolling volatility and advanced volatility estimators.
    
    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLC data with columns ['open', 'high', 'low', 'close'].
        Must be sorted ascending by date.
    windows : List[int]
        Rolling windows for standard deviation (in days)
    include_estimators : bool
        Whether to compute Parkinson, Garman-Klass, Yang-Zhang
    
    Returns
    -------
    pd.DataFrame
        Volatility features indexed by date
    
    Notes
    -----
    All volatility features are annualized (multiplied by sqrt(252) for NSE trading days)
    """
    
    vol_df = pd.DataFrame(index=ohlc.index)
    
    # 1. Rolling standard deviation of log returns
    log_returns = np.log(ohlc['close'] / ohlc['close'].shift(1))
    
    for w in windows:
        vol_df[f'vol_std_{w}d'] = log_returns.rolling(w).std() * np.sqrt(252)
    
    # 2. Parkinson Volatility
    if include_estimators:
        hl_ratio = np.log(ohlc['high'] / ohlc['low'])
        parkinson_daily = hl_ratio ** 2 / (4 * np.log(2))
        vol_df['vol_parkinson'] = np.sqrt(
            parkinson_daily.rolling(20).mean()
        ) * np.sqrt(252)
        
        # 3. Garman-Klass Volatility
        co_ratio = np.log(ohlc['close'] / ohlc['open'])
        hl_ratio2 = np.log(ohlc['high'] / ohlc['low'])
        
        gk_daily = (
            0.5 * hl_ratio2 ** 2 - 
            (2 * np.log(2) - 1) * co_ratio ** 2
        )
        vol_df['vol_gk'] = np.sqrt(gk_daily.rolling(20).mean()) * np.sqrt(252)
        
        # 4. Yang-Zhang Volatility
        overnight_gap = np.log(ohlc['open'] / ohlc['close'].shift(1))
        overnight_vol = overnight_gap.rolling(20).var()
        
        hl_log = np.log(ohlc['high'] / ohlc['close'])
        ll_log = np.log(ohlc['low'] / ohlc['close'])
        ol_log = np.log(ohlc['open'] / ohlc['close'])
        
        intraday_var = (
            hl_log * (hl_log - ol_log) + 
            ll_log * (ll_log - ol_log)
        ).rolling(20).mean()
        
        # Alpha parameter (optimal is ~0.34 for daily data)
        alpha = 1.34
        
        vol_df['vol_yang_zhang'] = np.sqrt(
            overnight_vol + alpha * intraday_var
        ) * np.sqrt(252)
    
    return vol_df


# Example: Using with Zerodha data structure
if __name__ == "__main__":
    # Simulate 1 year of OHLC (NSE trading days)
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    n = len(dates)
    
    ohlc = pd.DataFrame({
        'open': np.random.uniform(1490, 1510, n),
        'high': np.random.uniform(1510, 1530, n),
        'low': np.random.uniform(1470, 1490, n),
        'close': np.random.uniform(1490, 1510, n),
    }, index=dates)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    ohlc['high'] = ohlc[['open', 'high', 'close']].max(axis=1) + np.random.uniform(0, 5, n)
    ohlc['low'] = ohlc[['open', 'low', 'close']].min(axis=1) - np.random.uniform(0, 5, n)
    
    vol_features = compute_volatility_features(ohlc, windows=[5, 21, 63])
    print(f"Volatility features shape: {vol_features.shape}")
    print(f"\nVolatility features (latest):\n{vol_features.tail()}")
    print(f"\nMean volatility (Yang-Zhang): {vol_features['vol_yang_zhang'].mean():.4f}")
```

### Volume Features

Volume is the **fuel of trading**. High volume indicates conviction; low volume indicates skepticism. Volume features capture:

1. **Relative volume** — Is today's volume abnormal?
2. **Volume-price correlation** — Is volume confirming the price move?

#### Relative Volume

$$RV_t = \frac{V_t}{\text{MA}(V, 20)}$$

Where $V_t$ is volume on day $t$ and MA(V, 20) is the 20-day moving average.

#### Volume Z-Score

$$VZ_t = \frac{V_t - \mu(V)}{\sigma(V)}$$

Detects when volume is statistically unusual.

#### Volume-Price Correlation

$$\text{VPC}_t = \text{Corr}(r_t, \Delta V_t) \text{ over } w \text{ days}$$

If large price moves happen on high volume, the correlation is positive (confirmation); if on low volume (suspicious).

#### Implementation: Volume Features

```python
def compute_volume_features(
    ohlc: pd.DataFrame,
    vol_window: int = 20,
    corr_window: int = 20
) -> pd.DataFrame:
    """
    Compute volume-based features.
    
    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLC data with 'close' and 'volume' columns
    vol_window : int
        Window for moving average volume baseline (in days)
    corr_window : int
        Window for volume-price correlation (in days)
    
    Returns
    -------
    pd.DataFrame
        Volume features
    """
    
    vol_df = pd.DataFrame(index=ohlc.index)
    
    # 1. Relative Volume
    vol_ma = ohlc['volume'].rolling(vol_window).mean()
    vol_df['vol_relative'] = ohlc['volume'] / vol_ma
    
    # 2. Volume Z-Score
    vol_mean = ohlc['volume'].rolling(vol_window * 3).mean()
    vol_std = ohlc['volume'].rolling(vol_window * 3).std()
    vol_df['vol_zscore'] = (ohlc['volume'] - vol_mean) / vol_std
    
    # 3. Price-Volume Correlation
    log_returns = np.log(ohlc['close'] / ohlc['close'].shift(1))
    vol_change = ohlc['volume'].diff()
    
    pvc = log_returns.rolling(corr_window).corr(vol_change)
    vol_df['vol_price_corr'] = pvc
    
    # 4. Volume Rate of Change
    vol_df['vol_roc_5d'] = ohlc['volume'].pct_change(5)
    
    return vol_df
```

### Technical Indicators as Features

Rather than using technical indicators for trading signals (deprecated approach), we use them as **features capturing market microstructure**. We implement from formulas, not libraries (to understand and control them).

#### Relative Strength Index (RSI)

Captures momentum exhaustion:

$$RSI_t = 100 - \frac{100}{1 + RS_t}$$

Where:

$$RS_t = \frac{\text{AvgGain}(w)}{\text{AvgLoss}(w)}$$

And AvgGain and AvgLoss are exponentially smoothed (Wilder's method, not simple MA).

#### MACD (Moving Average Convergence Divergence)

Captures trend changes:

$$\text{MACD}_t = \text{EMA}(close, 12) - \text{EMA}(close, 26)$$
$$\text{Signal}_t = \text{EMA}(\text{MACD}, 9)$$
$$\text{Histogram}_t = \text{MACD}_t - \text{Signal}_t$$

#### Bollinger Bands

Captures volatility-adjusted mean reversion:

$$\text{BB}_{mid} = \text{SMA}(close, 20)$$
$$\text{BB}_{upper} = \text{BB}_{mid} + 2 \cdot \sigma(close, 20)$$
$$\text{BB}_{lower} = \text{BB}_{mid} - 2 \cdot \sigma(close, 20)$$

Feature: $\text{BB\%B} = \frac{close - \text{BB}_{lower}}{\text{BB}_{upper} - \text{BB}_{lower}}$ (position within bands)

#### ADX (Average Directional Index)

Captures trend strength (0=no trend, 100=very strong trend):

$$ADX = 100 \times \frac{\text{True Range Moving Avg}}{\text{Directional Index}}$$

#### Implementation: All Technical Indicators

```python
def exponential_moving_average(
    series: pd.Series,
    span: int,
    adjust: bool = False
) -> pd.Series:
    """
    Compute EMA using Wilder's smoothing (preferred for technical indicators).
    
    Parameters
    ----------
    series : pd.Series
        Input series
    span : int
        Span (half-life in days)
    adjust : bool
        If False, use Wilder's method (alpha = 1/span)
        If True, use standard EMA (alpha = 2/(span+1))
    
    Returns
    -------
    pd.Series
        EMA values
    """
    if adjust:
        return series.ewm(span=span, adjust=False).mean()
    else:
        # Wilder's method: alpha = 1/span
        alpha = 1.0 / span
        return series.ewm(alpha=alpha, adjust=False).mean()


def compute_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute RSI using Wilder's exponential smoothing.
    
    Parameters
    ----------
    close : pd.Series
        Close prices
    period : int
        RSI period (14 is standard)
    
    Returns
    -------
    pd.Series
        RSI values (0-100)
    
    Formula
    -------
    RSI = 100 - (100 / (1 + RS))
    where RS = AvgGain / AvgLoss (Wilder's exponential average)
    """
    
    # Calculate gains and losses
    delta = close.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    
    # Wilder's smoothing
    avg_gain = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD, Signal line, and Histogram.
    
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (MACD line, Signal line, Histogram)
    """
    ema_fast = exponential_moving_average(close, fast, adjust=False)
    ema_slow = exponential_moving_average(close, slow, adjust=False)
    
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal, adjust=False)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands and %B indicator.
    
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        (Upper band, Middle band, Lower band, %B)
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    
    # %B: where in the bands is price?
    bb_percent = (close - lower) / (upper - lower)
    
    return upper, sma, lower, bb_percent


def compute_atr_and_adx(
    ohlc: pd.DataFrame,
    period: int = 14
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Average True Range (ATR) and Average Directional Index (ADX).
    
    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLC data
    period : int
        Lookback period (14 is standard)
    
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (ATR, ADX)
    """
    
    # True Range
    hl = ohlc['high'] - ohlc['low']
    hc = np.abs(ohlc['high'] - ohlc['close'].shift(1))
    lc = np.abs(ohlc['low'] - ohlc['close'].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    
    # ATR (Wilder's smoothing)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    # Directional Movement
    up_move = ohlc['high'].diff()
    down_move = -ohlc['low'].diff()
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed directional movements
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
    
    # ADX
    di_diff = np.abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / (di_sum + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return atr, adx


def compute_technical_features(
    ohlc: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute all technical indicator features.
    
    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLC data with 'open', 'high', 'low', 'close' columns
    
    Returns
    -------
    pd.DataFrame
        Technical features
    """
    
    tech_df = pd.DataFrame(index=ohlc.index)
    
    # RSI
    tech_df['rsi_14'] = compute_rsi(ohlc['close'], period=14)
    tech_df['rsi_7'] = compute_rsi(ohlc['close'], period=7)
    
    # MACD
    macd, signal, histogram = compute_macd(ohlc['close'])
    tech_df['macd'] = macd
    tech_df['macd_signal'] = signal
    tech_df['macd_histogram'] = histogram
    
    # Bollinger Bands
    upper, mid, lower, bb_pct = compute_bollinger_bands(ohlc['close'], period=20)
    tech_df['bb_upper'] = upper
    tech_df['bb_middle'] = mid
    tech_df['bb_lower'] = lower
    tech_df['bb_percent'] = bb_pct
    
    # ATR and ADX
    atr, adx = compute_atr_and_adx(ohlc, period=14)
    tech_df['atr'] = atr
    tech_df['adx'] = adx
    
    # Normalize technical indicators to 0-1 range for modeling
    tech_df['rsi_14_norm'] = tech_df['rsi_14'] / 100.0
    tech_df['adx_norm'] = tech_df['adx'] / 100.0
    
    return tech_df
```

**Summary of Module 10.1:** We've implemented 20+ price/volume/technical features:
- 6 return horizons (1d, 5d, 21d, 63d, 126d, 252d)
- 7 volatility measures (rolling + Parkinson/GK/YZ)
- 4 volume features (relative, zscore, correlation, ROC)
- 8 technical features (RSI, MACD, BB, ADX)

These features are **stationary** (don't trend indefinitely) and **interpretable** (we know what volatility clustering or momentum exhaustion mean).

---

## Module 10.2: Cross-Sectional Features

### The Problem With Time-Series Features Alone

Imagine two scenarios:

**Scenario A:** INFY up 2% today (very good return!)  
**Scenario B:** INFY up 2% today, but NIFTY (index) up 5% (bad relative performance)

A time-series model trained only on INFY's returns cannot distinguish these. Both see a +2% return. But in Scenario B, INFY *underperformed* the market and should be avoided.

**Cross-sectional features fix this** by ranking stocks relative to their peers. They answer: "How does this stock rank among all NSE stocks today?"

### Ranking Features: Cross-Sectional Percentile Rank

$$\text{Rank}_t(\text{stock}) = \frac{\text{# stocks with return} < \text{return}_t}{\text{total # stocks}} \times 100$$

This converts any feature (return, volatility, volume) into a **percentile rank**, making it scale-invariant and market-relative.

#### Implementation: Cross-Sectional Ranking

```python
def compute_cross_sectional_ranks(
    feature_matrix: pd.DataFrame,
    methods: List[str] = ['percentile', 'zscore']
) -> Dict[str, pd.DataFrame]:
    """
    Compute cross-sectional features (rankings) across all stocks.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features by stock (columns) and date (rows).
        Example shape: (252 days, 50 stocks)
    methods : List[str]
        'percentile': 0-100 ranking
        'zscore': z-score relative to cross-section
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of cross-sectional features by method
    
    Notes
    -----
    Cross-sectional features are more stationary than time-series because
    relative rankings are bounded (0-100 for percentile, ~-3 to 3 for z-score).
    
    Example
    -------
    >>> # 252 days, 50 stocks
    >>> returns = pd.DataFrame(
    ...     np.random.randn(252, 50),
    ...     columns=[f'stock_{i}' for i in range(50)]
    ... )
    >>> cs_features = compute_cross_sectional_ranks(returns, methods=['percentile'])
    >>> # Now stock_0 has percentile rank 0-100 relative to peers each day
    """
    
    cs_features = {}
    
    if 'percentile' in methods:
        # Each stock's rank (0-100) relative to all stocks that day
        percentile_ranks = feature_matrix.rank(axis=1, pct=True) * 100
        cs_features['percentile'] = percentile_ranks
    
    if 'zscore' in methods:
        # Each stock's z-score relative to mean/std of all stocks that day
        cs_mean = feature_matrix.mean(axis=1)
        cs_std = feature_matrix.std(axis=1)
        
        # Broadcast to all columns
        z_scores = (feature_matrix.T - cs_mean) / (cs_std + 1e-10)
        cs_features['zscore'] = z_scores.T
    
    return cs_features


# Practical example: Building a cross-sectional momentum factor
if __name__ == "__main__":
    # Simulate 252 days of returns for 50 NSE stocks
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    stocks = [f'stock_{i}' for i in range(50)]
    
    # Each stock has different drift and volatility
    returns_matrix = pd.DataFrame(
        np.random.randn(252, 50) * 0.02 + np.random.uniform(-0.001, 0.001, 50),
        index=dates,
        columns=stocks
    )
    
    # Compute cross-sectional ranks
    cs_features = compute_cross_sectional_ranks(returns_matrix, methods=['percentile', 'zscore'])
    
    percentile_ranks = cs_features['percentile']
    zscore_ranks = cs_features['zscore']
    
    print(f"Percentile ranks shape: {percentile_ranks.shape}")
    print(f"\nTop 5 stocks by percentile rank (today):")
    print(percentile_ranks.iloc[-1].nlargest(5))
    
    print(f"\nBottom 5 stocks by percentile rank (today):")
    print(percentile_ranks.iloc[-1].nsmallest(5))
    
    # Verify: percentile rank should be bounded [0, 100]
    print(f"\nPercentile rank bounds: [{percentile_ranks.min().min():.2f}, {percentile_ranks.max().max():.2f}]")
    print(f"Z-score bounds: [{zscore_ranks.min().min():.2f}, {zscore_ranks.max().max():.2f}]")
```

### Sector/Market Relative Features

Beyond raw rankings, we adjust features **relative to the sector and market**.

#### Sector Adjustment Formula

$$\text{Adj}_{sector} = \text{Feature}_{stock} - \text{Median}(\text{Feature}_{sector})$$

This isolates the stock's unique behavior from sector-wide trends.

#### Market Adjustment Formula

$$\text{Adj}_{market} = \text{Feature}_{stock} - \text{Median}(\text{Feature}_{NSE})$$

Similarly, market-relative features answer: "Is this abnormal compared to all traded stocks?"

#### Implementation: Sector/Market Adjustment

```python
def compute_sector_relative_features(
    feature_matrix: pd.DataFrame,
    sector_mapping: pd.Series
) -> pd.DataFrame:
    """
    Compute sector-relative features (stock feature - sector median).
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features by stock (columns) and date (rows)
    sector_mapping : pd.Series
        Mapping of stock -> sector. Index is stock names, values are sector codes.
    
    Returns
    -------
    pd.DataFrame
        Sector-adjusted features
    
    Example
    -------
    >>> returns = pd.DataFrame(
    ...     np.random.randn(10, 4),
    ...     columns=['TCS', 'INFY', 'RELIANCE', 'HDFC']
    ... )
    >>> sectors = pd.Series({
    ...     'TCS': 'IT',
    ...     'INFY': 'IT',
    ...     'RELIANCE': 'ENERGY',
    ...     'HDFC': 'FINANCE'
    ... })
    >>> adjusted = compute_sector_relative_features(returns, sectors)
    >>> # TCS return is now relative to IT median (not absolute)
    """
    
    sector_adjusted = pd.DataFrame(index=feature_matrix.index, columns=feature_matrix.columns)
    
    for date_idx in range(len(feature_matrix)):
        row = feature_matrix.iloc[date_idx]
        
        # For each stock, subtract its sector median
        for stock in feature_matrix.columns:
            if stock not in sector_mapping.index:
                sector_adjusted.loc[feature_matrix.index[date_idx], stock] = np.nan
                continue
            
            sector = sector_mapping[stock]
            sector_stocks = sector_mapping[sector_mapping == sector].index
            sector_median = row[sector_stocks].median()
            
            sector_adjusted.loc[feature_matrix.index[date_idx], stock] = row[stock] - sector_median
    
    return sector_adjusted


def compute_market_relative_features(
    feature_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute market-relative features (stock feature - NSE median).
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features by stock (columns) and date (rows)
    
    Returns
    -------
    pd.DataFrame
        Market-adjusted features
    """
    
    market_median = feature_matrix.median(axis=1)  # Median across all stocks each day
    
    # Broadcast and subtract
    market_adjusted = feature_matrix.sub(market_median, axis=0)
    
    return market_adjusted
```

### Why Cross-Sectional Features Are More Stationary

**Time-series feature problem:**
$$\text{INFY return}_{2015} \text{ vs } \text{INFY return}_{2025}$$
These are drawn from different distributions (market structure changed, economy evolved).

**Cross-sectional feature advantage:**
$$\text{INFY percentile rank}_{2015} \text{ vs } \text{INFY percentile rank}_{2025}$$
Both are bounded [0, 100]. Even if market returns changed, a stock in the 75th percentile is in the 75th percentile, regardless of year. This makes models more **transferable** across time.

**Example implementation: Demonstrating stationarity**

```python
def compare_stationarity(
    returns_ts: pd.Series,
    returns_cs: pd.DataFrame
) -> Dict[str, float]:
    """
    Compare stationarity of time-series vs cross-sectional features using ADF test.
    
    Returns
    -------
    Dict[str, float]
        ADF test p-values (lower = more stationary)
    """
    from scipy import stats
    
    # Simplified stationarity check: variance should be stable over time
    # Divide into 4 quarters and compute variance ratio
    n = len(returns_ts)
    q_len = n // 4
    
    ts_vars = [
        returns_ts.iloc[i*q_len:(i+1)*q_len].var()
        for i in range(4)
    ]
    ts_instability = np.std(ts_vars) / np.mean(ts_vars)  # Coefficient of variation
    
    # For cross-sectional, compute percentile ranks and check stability
    cs_percentiles = returns_cs.rank(axis=1, pct=True).iloc[:, 0]
    
    cs_vars = [
        cs_percentiles.iloc[i*q_len:(i+1)*q_len].var()
        for i in range(4)
    ]
    cs_instability = np.std(cs_vars) / (np.mean(cs_vars) + 1e-10)
    
    return {
        'time_series_instability': ts_instability,
        'cross_sectional_instability': cs_instability,
        'improvement_ratio': ts_instability / cs_instability
    }
```

**Summary of Module 10.2:** Cross-sectional features convert absolute returns into relative rankings. This improves stationarity and predictiveness because we're modeling *relative* position in the market, not absolute prices.

---

## Module 10.3: Fundamental Features

### Introduction to Fundamental Features

Fundamental features link price to underlying business metrics. They answer: "Is INFY expensive or cheap?" or "Is RELIANCE growing faster than peers?"

Unlike technical features (backward-looking), fundamental features are forward-looking: valuations that are too low often mean the market is pessimistic (a buying opportunity if pessimism is unjustified).

### Valuation Ratios

#### Price-to-Earnings (P/E)

$$\text{P/E} = \frac{\text{Market Cap}}{\text{Annual Earnings}}$$

Or per-share:

$$\text{P/E} = \frac{\text{Stock Price}}{\text{Earnings Per Share (EPS)}}$$

**Interpretation:**
- P/E = 10: Investors pay ₹10 for every ₹1 of earnings (cheap)
- P/E = 30: Investors pay ₹30 for every ₹1 of earnings (expensive)
- Low P/E + growth = value investing opportunity
- High P/E + no growth = value trap (to avoid)

#### Price-to-Book (P/B)

$$\text{P/B} = \frac{\text{Market Cap}}{\text{Book Value of Assets}}$$

Book value = Total Assets - Total Liabilities (tangible value on balance sheet).

**Use case:** Asset-heavy businesses (banks, manufacturing). Shows if stock trades above or below liquidation value.

#### Price-to-Sales (P/S)

$$\text{P/S} = \frac{\text{Market Cap}}{\text{Annual Revenue}}$$

**Advantage:** Can't be manipulated like earnings (revenue is harder to fake).

#### EV/EBITDA

$$\text{EV/EBITDA} = \frac{\text{Enterprise Value}}{\text{EBITDA}}$$

Where:
- Enterprise Value = Market Cap + Net Debt = Market Cap + (Debt - Cash)
- EBITDA = Earnings Before Interest, Taxes, Depreciation, Amortization

**Use case:** Comparing companies with different capital structures (some leveraged, some not).

#### Implementation: Valuation Ratios

```python
def compute_valuation_ratios(
    market_data: pd.DataFrame,
    fundamental_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute valuation ratios from market and fundamental data.
    
    Parameters
    ----------
    market_data : pd.DataFrame
        Columns: 'market_cap' (₹), 'price' (₹), 'net_debt' (₹)
    fundamental_data : pd.DataFrame
        Columns: 'earnings', 'book_value', 'revenue', 'ebitda'
        All in ₹ (annual)
    
    Returns
    -------
    pd.DataFrame
        Valuation ratios (P/E, P/B, P/S, EV/EBITDA)
    
    Notes
    -----
    Handle edge cases: negative earnings (avoid P/E), NaN values, etc.
    """
    
    valuation_df = pd.DataFrame(index=market_data.index)
    
    # P/E Ratio
    # Only compute where earnings > 0 (exclude unprofitable companies)
    pe_ratio = market_data['market_cap'] / fundamental_data['earnings']
    valuation_df['pe_ratio'] = pe_ratio.where(fundamental_data['earnings'] > 0, np.nan)
    
    # P/B Ratio
    valuation_df['pb_ratio'] = market_data['market_cap'] / fundamental_data['book_value']
    
    # P/S Ratio
    valuation_df['ps_ratio'] = market_data['market_cap'] / fundamental_data['revenue']
    
    # EV/EBITDA
    enterprise_value = market_data['market_cap'] + market_data['net_debt']
    valuation_df['ev_ebitda'] = enterprise_value / fundamental_data['ebitda']
    
    return valuation_df
```

### Quality Metrics

#### Return on Equity (ROE)

$$\text{ROE} = \frac{\text{Net Income}}{\text{Shareholders' Equity}}$$

Measures how efficiently the company generates profits from shareholder capital.

**Interpretation:**
- ROE = 20%: Company generates ₹0.20 profit for every ₹1 of equity
- Consistent 20%+ ROE = high-quality business
- Declining ROE = deteriorating fundamentals

#### Return on Assets (ROA)

$$\text{ROA} = \frac{\text{Net Income}}{\text{Total Assets}}$$

Shows profitability relative to all assets (not just equity).

#### Gross Margin

$$\text{Gross Margin} = \frac{\text{Revenue} - \text{Cost of Goods Sold}}{\text{Revenue}}$$

**Interpretation:**
- 60% margin: Company keeps ₹0.60 of every ₹1 revenue before operating expenses
- High margin = pricing power (can raise prices without losing customers)
- Margin expansion = improving profitability (bullish)
- Margin compression = losing pricing power (bearish)

#### Debt-to-Equity Ratio

$$\text{D/E} = \frac{\text{Total Debt}}{\text{Shareholders' Equity}}$$

**Interpretation:**
- 0.5: Company financed 33% by debt, 67% by equity (conservative)
- 2.0: Company financed 67% by debt, 33% by equity (leveraged, risky in downturns)
- Increasing D/E = financial stress (negative signal)

#### Implementation: Quality Metrics

```python
def compute_quality_metrics(
    fundamental_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute quality metrics from fundamental data.
    
    Parameters
    ----------
    fundamental_data : pd.DataFrame
        Columns: 'net_income', 'equity', 'total_assets', 'revenue', 'cogs', 'debt'
    
    Returns
    -------
    pd.DataFrame
        Quality metrics (ROE, ROA, gross margin, D/E)
    """
    
    quality_df = pd.DataFrame(index=fundamental_data.index)
    
    # ROE (%)
    quality_df['roe'] = (fundamental_data['net_income'] / fundamental_data['equity']) * 100
    
    # ROA (%)
    quality_df['roa'] = (fundamental_data['net_income'] / fundamental_data['total_assets']) * 100
    
    # Gross Margin (%)
    gross_profit = fundamental_data['revenue'] - fundamental_data['cogs']
    quality_df['gross_margin'] = (gross_profit / fundamental_data['revenue']) * 100
    
    # Debt-to-Equity
    quality_df['debt_to_equity'] = fundamental_data['debt'] / fundamental_data['equity']
    
    return quality_df
```

### Growth Metrics

#### Earnings Growth Rate

$$\text{Earnings Growth}_t = \frac{\text{EPS}_t - \text{EPS}_{t-1}}{\text{EPS}_{t-1}}$$

**Interpretation:**
- 20% YoY growth: EPS growing at 20% annually (strong growth)
- Negative growth: Earnings declining (warning sign)

#### Revenue Growth Rate

$$\text{Revenue Growth}_t = \frac{\text{Revenue}_t - \text{Revenue}_{t-1}}{\text{Revenue}_{t-1}}$$

#### Margin Expansion

$$\text{Margin Expansion} = \text{Operating Margin}_t - \text{Operating Margin}_{t-4}$$

Measuring change in margins (not absolute level) captures improving profitability.

#### Implementation: Growth Metrics

```python
def compute_growth_metrics(
    fundamental_data: pd.DataFrame,
    periods_back: int = 4  # Quarterly data, so 4 = 1 year
) -> pd.DataFrame:
    """
    Compute growth metrics.
    
    Parameters
    ----------
    fundamental_data : pd.DataFrame
        Columns: 'eps', 'revenue', 'operating_income'
    periods_back : int
        Periods back for YoY comparison (4 for quarterly data)
    
    Returns
    -------
    pd.DataFrame
        Growth metrics
    """
    
    growth_df = pd.DataFrame(index=fundamental_data.index)
    
    # Earnings Growth (YoY)
    growth_df['eps_growth_yoy'] = (
        fundamental_data['eps'].pct_change(periods_back) * 100
    )
    
    # Revenue Growth (YoY)
    growth_df['revenue_growth_yoy'] = (
        fundamental_data['revenue'].pct_change(periods_back) * 100
    )
    
    # Margin Expansion (YoY change)
    operating_margin = fundamental_data['operating_income'] / fundamental_data['revenue'] * 100
    growth_df['margin_expansion_yoy'] = operating_margin.diff(periods_back)
    
    return growth_df
```

### Earnings Surprise (Critical for Predictions)

Earnings surprise is the **actual earnings vs expected earnings**, and it's one of the most predictive signals:

$$\text{Surprise} = \frac{\text{Actual EPS} - \text{Consensus EPS}}{\text{Consensus EPS}}$$

**Logic:** When companies beat expectations, they often rally; when they miss, they drop. This effect persists for weeks (post-earnings drift).

#### Implementation: Earnings Surprise

```python
def compute_earnings_surprise(
    actual_eps: pd.Series,
    consensus_eps: pd.Series
) -> pd.Series:
    """
    Compute earnings surprise (beat or miss relative to consensus).
    
    Parameters
    ----------
    actual_eps : pd.Series
        Actual EPS reported by company
    consensus_eps : pd.Series
        Consensus EPS estimate (analyst average)
    
    Returns
    -------
    pd.Series
        Surprise as percentage
    
    Example
    -------
    >>> actual = pd.Series([100, 110, 95])
    >>> consensus = pd.Series([100, 100, 100])
    >>> surprise = compute_earnings_surprise(actual, consensus)
    >>> surprise[0]  # Beat consensus by 0%
    0.0
    >>> surprise[1]  # Beat consensus by 10%
    10.0
    >>> surprise[2]  # Missed consensus by 5%
    -5.0
    """
    
    surprise = (actual_eps - consensus_eps) / (consensus_eps + 1e-10) * 100
    return surprise
```

### Point-in-Time: Avoiding Look-Ahead Bias

**Critical issue in backtesting:** Fundamental data is released with lags. A company reports Q3 earnings on Jan 20, 2025. If your model "looks at" Q3 earnings on Jan 1, 2025, you have **look-ahead bias** (using future data).

#### Solution: Point-in-Time (PIT) Database

Maintain a **release date** for every piece of fundamental data:

```python
def build_pit_fundamental_data(
    raw_fundamental: pd.DataFrame,
    release_dates: pd.DataFrame
) -> pd.DataFrame:
    """
    Adjust fundamental data to be point-in-time correct.
    
    Parameters
    ----------
    raw_fundamental : pd.DataFrame
        Fundamental data indexed by report date
    release_dates : pd.DataFrame
        When each report was published (release date)
    
    Returns
    -------
    pd.DataFrame
        Data indexed by release date, forward-filled until next release
    
    Example
    -------
    Release Date  |  Q3 Earnings (₹)
    --------
    2025-01-20    |  100  <- Available from Jan 20 onward
    2025-04-20    |  110  <- Q4 earnings, available from Apr 20 onward
    
    Between Jan 20 and Apr 20, model can only see Q3 earnings (100).
    This prevents look-ahead bias.
    """
    
    # Reindex fundamental data by release date, not report date
    pit_data = raw_fundamental.copy()
    pit_data.index = release_dates
    pit_data = pit_data.sort_index()
    
    # Forward-fill: data available from release date until next release
    pit_data = pit_data.fillna(method='ffill')
    
    return pit_data
```

**Summary of Module 10.3:** We've implemented valuation (P/E, P/B, P/S, EV/EBITDA), quality (ROE, ROA, margins, leverage), and growth metrics (earnings/revenue growth). All respect point-in-time constraints to avoid look-ahead bias.

---

## Module 10.4: Feature Processing Pipeline

### Overview: From Raw Data to Model-Ready Features

A production feature pipeline has these stages:

```
Raw Data (OHLC, Fundamentals) 
    ↓ (compute features)
Unprocessed Features (may have NaN, outliers, different scales)
    ↓ (handle missing values)
Features with Imputation (forward-filled, sector-filled)
    ↓ (cap outliers)
Winsorized Features (outliers capped at 1st/99th percentile)
    ↓ (normalize)
Final Features (0-1 range or z-scored)
    ↓ (optional: interactions/polynomials)
ML-Ready Features → Model Training
```

### Handling Missing Values

Missing values are inevitable:
- Stock delisted → no future returns
- Fundamental data not yet released → NaN
- Corporate actions (splits) → gaps

#### Cross-Sectional Median Fill

When a stock is missing a feature, fill with **sector median** or **market median**:

```python
def fill_missing_cross_sectional(
    feature_matrix: pd.DataFrame,
    sector_mapping: pd.Series = None,
    method: str = 'sector'  # or 'market'
) -> pd.DataFrame:
    """
    Fill missing values using cross-sectional information.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features (stocks as columns, dates as rows)
    sector_mapping : pd.Series
        Stock -> sector mapping (required if method='sector')
    method : str
        'sector': use sector median
        'market': use market (NSE) median
    
    Returns
    -------
    pd.DataFrame
        Features with NaN filled
    
    Logic
    -----
    For each NaN at date t, stock s:
    - If method='sector': fill with median of sector s on date t
    - If method='market': fill with median of all stocks on date t
    """
    
    filled = feature_matrix.copy()
    
    if method == 'market':
        for date_idx in filled.index:
            market_median = filled.loc[date_idx].median()
            filled.loc[date_idx] = filled.loc[date_idx].fillna(market_median)
    
    elif method == 'sector' and sector_mapping is not None:
        for date_idx in filled.index:
            for stock in filled.columns:
                if pd.isna(filled.loc[date_idx, stock]):
                    sector = sector_mapping.get(stock, None)
                    if sector is not None:
                        sector_stocks = sector_mapping[sector_mapping == sector].index
                        sector_median = filled.loc[date_idx, sector_stocks].median()
                        filled.loc[date_idx, stock] = sector_median
    
    return filled
```

#### Forward-Fill (for time-series data)

For features that don't change frequently (e.g., quarterly earnings), forward-fill:

```python
def forward_fill_features(
    feature_matrix: pd.DataFrame,
    max_periods: int = 60  # Don't fill gaps > 60 days
) -> pd.DataFrame:
    """
    Forward-fill missing values in time-series features.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features indexed by date
    max_periods : int
        Maximum periods to forward-fill before stopping
    
    Returns
    -------
    pd.DataFrame
        Forward-filled features
    """
    
    return feature_matrix.fillna(method='ffill', limit=max_periods)
```

### Winsorization: Capping Outliers

Outliers (extreme values) can distort models. Rather than deleting rows, **winsorize**: cap extreme values at percentiles.

$$X_{winsorized} = \begin{cases} P_{1\%} & \text{if } X < P_{1\%} \\ X & \text{if } P_{1\%} \leq X \leq P_{99\%} \\ P_{99\%} & \text{if } X > P_{99\%} \end{cases}$$

```python
def winsorize_features(
    feature_matrix: pd.DataFrame,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> pd.DataFrame:
    """
    Winsorize features at specified percentiles.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features to winsorize
    lower_percentile : float
        Lower bound (e.g., 1 = 1st percentile)
    upper_percentile : float
        Upper bound (e.g., 99 = 99th percentile)
    
    Returns
    -------
    pd.DataFrame
        Winsorized features
    
    Example
    -------
    >>> data = pd.DataFrame({'price_change': [0.5, 1000, 0.3, 0.4]})
    >>> winsorized = winsorize_features(data)
    >>> # 1000 is capped at 99th percentile (e.g., 0.5)
    """
    
    lower_bound = feature_matrix.quantile(lower_percentile / 100.0)
    upper_bound = feature_matrix.quantile(upper_percentile / 100.0)
    
    winsorized = feature_matrix.clip(lower=lower_bound, upper=upper_bound)
    
    return winsorized
```

### Normalization: Multiple Strategies

#### Z-Score Normalization (Standardization)

$$X_{zscore} = \frac{X - \mu}{\sigma}$$

Produces features with mean 0, std 1. Good for linear models and neural networks.

#### Min-Max Normalization

$$X_{minmax} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Produces features in [0, 1] range. Good for tree-based models.

#### Rank Normalization

$$X_{rank} = \text{rank}(X) / \text{length}(X)$$

Converts to percentile (0-1). Robust to outliers; useful for cross-sectional ranking.

#### Quantile Normalization

$$X_{quantile} = \Phi^{-1}\left(\text{percentile rank}\right)$$

Maps to normal distribution. Advanced; useful for ensuring normality.

```python
def normalize_features(
    feature_matrix: pd.DataFrame,
    method: str = 'zscore',
    fit_data: pd.DataFrame = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize features using specified method.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Features to normalize
    method : str
        'zscore': (X - mean) / std
        'minmax': (X - min) / (max - min)
        'rank': rank-based percentile
        'quantile': quantile normalization
    fit_data : pd.DataFrame
        Data to compute mean/std (for train/test split). 
        If None, use feature_matrix itself.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (Normalized features, normalization parameters for reproducibility)
    """
    
    if fit_data is None:
        fit_data = feature_matrix
    
    normalized = feature_matrix.copy()
    params = {}
    
    if method == 'zscore':
        mean = fit_data.mean()
        std = fit_data.std()
        normalized = (feature_matrix - mean) / (std + 1e-10)
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = fit_data.min()
        max_val = fit_data.max()
        normalized = (feature_matrix - min_val) / (max_val - min_val + 1e-10)
        params = {'min': min_val, 'max': max_val}
    
    elif method == 'rank':
        # Percentile rank: 0-1
        normalized = feature_matrix.rank(pct=True, axis=0)
        params = {}  # No parameters needed for rank
    
    elif method == 'quantile':
        from scipy.stats import norm
        percentile_rank = feature_matrix.rank(pct=True, axis=0)
        # Map to standard normal
        normalized = norm.ppf(percentile_rank.clip(0.001, 0.999))
        params = {}
    
    return normalized, params
```

### Feature Interactions and Polynomial Features

Sometimes the combination of features is more predictive than individual features.

**Example:** Momentum (price increase) + Volume spike = bullish confirmation
**Feature interaction:** `momentum × volume = momentum_volume_interaction`

#### Implementation: Feature Interactions

```python
def create_feature_interactions(
    feature_matrix: pd.DataFrame,
    feature_pairs: List[Tuple[str, str]] = None,
    max_degree: int = 2  # Polynomial degree
) -> pd.DataFrame:
    """
    Create interaction and polynomial features.
    
    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Input features
    feature_pairs : List[Tuple[str, str]]
        Specific pairs to interact. If None, create all pairwise interactions.
    max_degree : int
        Maximum polynomial degree (1=no interaction, 2=all pairs, 3=triples, etc.)
    
    Returns
    -------
    pd.DataFrame
        Original features + interaction features
    
    Notes
    -----
    High-degree polynomials explode in dimensionality. Avoid > 3.
    Better approach: use domain knowledge to select specific interactions.
    
    Example
    -------
    >>> features = pd.DataFrame({'momentum': [1, 2], 'volume_z': [0.5, 1.5]})
    >>> interactions = create_feature_interactions(
    ...     features,
    ...     feature_pairs=[('momentum', 'volume_z')],
    ...     max_degree=2
    ... )
    >>> # Returns original columns + 'momentum_volume_z' column
    """
    
    interactions = feature_matrix.copy()
    
    if feature_pairs is None:
        # Create all pairwise interactions
        cols = feature_matrix.columns.tolist()
        feature_pairs = [
            (cols[i], cols[j])
            for i in range(len(cols))
            for j in range(i + 1, len(cols))
        ]
    
    for feat1, feat2 in feature_pairs:
        if feat1 in feature_matrix.columns and feat2 in feature_matrix.columns:
            interactions[f'{feat1}_{feat2}_interaction'] = (
                feature_matrix[feat1] * feature_matrix[feat2]
            )
    
    # Optional: polynomial features (squares, cubes)
    if max_degree >= 2:
        for col in feature_matrix.columns:
            interactions[f'{col}_squared'] = feature_matrix[col] ** 2
    
    if max_degree >= 3:
        for col in feature_matrix.columns:
            interactions[f'{col}_cubed'] = feature_matrix[col] ** 3
    
    return interactions
```

### Complete Feature Pipeline: Production-Grade Implementation

Now we integrate everything into a **single reproducible pipeline**:

```python
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FeaturePipelineConfig:
    """Configuration for feature pipeline."""
    
    # Missing value handling
    fill_method: str = 'cross_sectional'  # 'forward_fill' or 'cross_sectional'
    cross_sectional_level: str = 'sector'  # 'sector' or 'market'
    
    # Winsorization
    winsorize: bool = True
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    
    # Normalization
    normalize: bool = True
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'rank', 'quantile'
    
    # Feature interactions
    create_interactions: bool = True
    interaction_pairs: List[Tuple[str, str]] = None


class FeaturePipeline:
    """
    Production-grade feature engineering pipeline.
    
    Usage:
    ------
    config = FeaturePipelineConfig()
    pipeline = FeaturePipeline(config)
    
    # Training phase: fit to train data
    pipeline.fit(X_train)
    
    # Apply to new data
    X_processed_train = pipeline.transform(X_train)
    X_processed_test = pipeline.transform(X_test)
    """
    
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self.normalization_params = {}
        self.is_fitted = False
    
    def fit(
        self,
        feature_matrix: pd.DataFrame,
        sector_mapping: pd.Series = None
    ) -> 'FeaturePipeline':
        """
        Fit pipeline on training data (compute statistics for normalization).
        
        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Training features
        sector_mapping : pd.Series
            Stock -> sector mapping (for cross-sectional fill)
        
        Returns
        -------
        FeaturePipeline
            Self (for chaining)
        """
        
        self.sector_mapping = sector_mapping
        
        # Compute normalization parameters
        if self.config.normalize:
            _, self.normalization_params = normalize_features(
                feature_matrix,
                method=self.config.normalization_method,
                fit_data=feature_matrix
            )
        
        self.is_fitted = True
        return self
    
    def transform(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all pipeline transformations.
        
        Parameters
        ----------
        feature_matrix : pd.DataFrame
            Raw features
        
        Returns
        -------
        pd.DataFrame
            Processed features
        """
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fit before transform. Call fit() first.")
        
        X = feature_matrix.copy()
        
        # Step 1: Fill missing values
        if self.config.fill_method == 'cross_sectional':
            X = fill_missing_cross_sectional(
                X,
                sector_mapping=self.sector_mapping,
                method=self.config.cross_sectional_level
            )
        elif self.config.fill_method == 'forward_fill':
            X = forward_fill_features(X, max_periods=60)
        
        # Step 2: Winsorize outliers
        if self.config.winsorize:
            X = winsorize_features(
                X,
                lower_percentile=self.config.lower_percentile,
                upper_percentile=self.config.upper_percentile
            )
        
        # Step 3: Normalize
        if self.config.normalize:
            X_normalized = X.copy()
            for col in X.columns:
                if self.config.normalization_method == 'zscore':
                    mean = self.normalization_params['mean'].get(col, 0)
                    std = self.normalization_params['std'].get(col, 1)
                    X_normalized[col] = (X[col] - mean) / (std + 1e-10)
                elif self.config.normalization_method == 'minmax':
                    min_val = self.normalization_params['min'].get(col, X[col].min())
                    max_val = self.normalization_params['max'].get(col, X[col].max())
                    X_normalized[col] = (X[col] - min_val) / (max_val - min_val + 1e-10)
                elif self.config.normalization_method == 'rank':
                    X_normalized[col] = X[col].rank(pct=True)
            X = X_normalized
        
        # Step 4: Create interactions (optional)
        if self.config.create_interactions:
            X = create_feature_interactions(
                X,
                feature_pairs=self.config.interaction_pairs,
                max_degree=2
            )
        
        return X
    
    def fit_transform(
        self,
        feature_matrix: pd.DataFrame,
        sector_mapping: pd.Series = None
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(feature_matrix, sector_mapping).transform(feature_matrix)
```

### Efficient Feature Storage for Backtesting

Computing features on-the-fly during backtesting is slow. Better: pre-compute and store.

```python
class FeatureStore:
    """
    Store pre-computed features on disk for fast backtesting access.
    
    Design:
    - One file per feature per stock (efficient, parallel reading)
    - HDF5 format (binary, fast, preserves data types)
    - Metadata: feature name, creation date, processing version
    """
    
    def __init__(self, storage_path: str = './feature_store'):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def save_feature(
        self,
        feature_name: str,
        feature_data: pd.DataFrame,  # Dates (rows) x stocks (cols)
        metadata: Dict = None
    ) -> None:
        """
        Save feature to disk.
        
        Parameters
        ----------
        feature_name : str
            Feature name (e.g., 'ret_1d', 'rsi_14', 'pe_ratio')
        feature_data : pd.DataFrame
            Feature values
        metadata : Dict
            Optional metadata (compute date, version, etc.)
        """
        
        filepath = os.path.join(self.storage_path, f'{feature_name}.h5')
        
        with pd.HDFStore(filepath, mode='w') as store:
            store.put('data', feature_data, format='table')
            
            if metadata:
                store.get_storer('data').attrs['metadata'] = metadata
    
    def load_feature(self, feature_name: str) -> pd.DataFrame:
        """Load feature from disk."""
        filepath = os.path.join(self.storage_path, f'{feature_name}.h5')
        return pd.read_hdf(filepath, key='data')
    
    def save_all_features(
        self,
        features_dict: Dict[str, pd.DataFrame],
        metadata: Dict = None
    ) -> None:
        """Save multiple features at once."""
        for feature_name, feature_data in features_dict.items():
            self.save_feature(feature_name, feature_data, metadata)
    
    def load_all_features(self, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Load multiple features."""
        return {
            name: self.load_feature(name)
            for name in feature_names
        }
```

### Complete Working Example: Building 50+ Features

```python
def build_complete_feature_set(
    ohlc: pd.DataFrame,
    fundamental_data: pd.DataFrame,
    sector_mapping: pd.Series = None
) -> pd.DataFrame:
    """
    Build all 50+ features from raw data.
    
    Parameters
    ----------
    ohlc : pd.DataFrame
        OHLC data (dates x 4 columns)
    fundamental_data : pd.DataFrame
        Fundamental data (dates x N columns)
    sector_mapping : pd.Series
        Stock -> sector mapping
    
    Returns
    -------
    pd.DataFrame
        All features combined and processed
    """
    
    all_features = {}
    
    # MODULE 10.1: Price/Volume/Technical Features
    print("Computing price features...")
    returns_df = compute_returns(ohlc['close'], horizons=[1, 5, 21, 63, 126, 252])
    all_features.update({f'ret_{h}d': returns_df[f'ret_{h}d'] for h in [1, 5, 21, 63, 126, 252]})
    
    print("Computing volatility features...")
    vol_df = compute_volatility_features(ohlc, windows=[5, 21, 63, 126])
    all_features.update(vol_df.to_dict(orient='series'))
    
    print("Computing volume features...")
    vol_features = compute_volume_features(ohlc)
    all_features.update(vol_features.to_dict(orient='series'))
    
    print("Computing technical features...")
    tech_df = compute_technical_features(ohlc)
    all_features.update(tech_df.to_dict(orient='series'))
    
    # Combine all features
    feature_matrix = pd.DataFrame(all_features, index=ohlc.index)
    
    # MODULE 10.4: Processing Pipeline
    print("Applying feature pipeline...")
    config = FeaturePipelineConfig(
        fill_method='cross_sectional',
        winsorize=True,
        normalize=True,
        normalization_method='zscore',
        create_interactions=False  # Too many features already
    )
    
    pipeline = FeaturePipeline(config)
    feature_matrix = pipeline.fit_transform(feature_matrix, sector_mapping)
    
    print(f"Final feature matrix shape: {feature_matrix.shape}")
    print(f"Total features: {feature_matrix.shape[1]}")
    
    return feature_matrix, pipeline


# Example usage
if __name__ == "__main__":
    print("=== Complete Feature Engineering Pipeline ===\n")
    
    # Simulate NSE OHLC data (1 year, daily)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    
    ohlc = pd.DataFrame({
        'open': np.random.uniform(1490, 1510, 252),
        'high': np.random.uniform(1510, 1530, 252),
        'low': np.random.uniform(1470, 1490, 252),
        'close': np.random.uniform(1490, 1510, 252),
        'volume': np.random.uniform(1e6, 5e6, 252)
    }, index=dates)
    
    # Ensure OHLC constraints
    ohlc['high'] = ohlc[['open', 'high', 'close']].max(axis=1) + np.abs(np.random.randn(252))
    ohlc['low'] = ohlc[['open', 'low', 'close']].min(axis=1) - np.abs(np.random.randn(252))
    
    # Simple fundamental data
    fundamental_data = pd.DataFrame({
        'earnings': np.random.exponential(100, 252),
        'book_value': np.random.exponential(1000, 252),
        'revenue': np.random.exponential(500, 252)
    }, index=dates)
    
    # Build features
    features, pipeline = build_complete_feature_set(ohlc, fundamental_data)
    
    print(f"\n=== Feature Summary ===")
    print(f"Shape: {features.shape}")
    print(f"Features: {list(features.columns)[:10]}... (showing first 10)")
    print(f"\nMissing values: {features.isna().sum().sum()}")
    print(f"Feature ranges (should be normalized):")
    print(f"  Min: {features.min().min():.4f}")
    print(f"  Max: {features.max().max():.4f}")
    print(f"  Mean: {features.mean().mean():.4f}")
    print(f"  Std: {features.std().mean():.4f}")
```

---

## Summary: The Complete Feature Engineering Workflow

### Key Principles

1. **Never use raw prices** — They violate stationarity. Use returns, volatility, ratios.

2. **Multi-horizon features** — A single horizon (e.g., 1-day return) misses mean reversion, momentum, seasonality. Use 6+ horizons.

3. **Cross-sectional beats time-series** — Relative rankings are more stable than absolute values. Rank stocks against peers, not against themselves.

4. **Fundamental + Technical** — Technical features capture market microstructure (RSI = exhaustion). Fundamental features link to intrinsic value (P/E = cheapness).

5. **Process before modeling** — Missing values, outliers, and different scales ruin models. Winsorize, normalize, and validate.

6. **Interpretability matters** — You should understand each feature. A feature capturing "earnings surprise" is better than a latent neural network layer.

7. **Look-ahead bias is fatal** — Verify that features use only data available at prediction time. Point-in-time databases are essential.

### Feature Count

In this chapter, we've implemented:

- **Returns**: 6 horizons
- **Volatility**: 7 measures
- **Volume**: 4 features
- **Technical indicators**: 8 features
- **Valuation ratios**: 4 features
- **Quality metrics**: 4 features
- **Growth metrics**: 3 features
- **Cross-sectional adjustments**: 2-3 per feature
- **Interaction features**: unlimited (we control)

**Total: 50+ core features, expandable to 100+ with interactions and cross-sectional variants.**

### For Your NSE Zerodha System

To build a production system:

1. **Data collection**: Use Zerodha's Python API to pull daily OHLC for your NSE universe.
2. **Feature computation**: Run the compute functions on new data daily.
3. **Fundamental data**: Maintain a separate database of quarterly/annual fundamentals with release dates (point-in-time).
4. **Pipeline**: Use `FeaturePipeline` to consistently process training and live data.
5. **Feature store**: Cache computed features in HDF5 for fast backtesting.
6. **Monitoring**: Track feature distributions daily; sharp changes indicate data quality issues.

Next chapter (Ch. 11) will train machine learning models on these features to generate trading signals.

---

## Code Reference: All Functions Summary

### Module 10.1: Price & Volume
- `compute_returns()` — Multi-horizon returns
- `compute_volatility_features()` — Rolling std, Parkinson, Garman-Klass, Yang-Zhang
- `compute_volume_features()` — Relative volume, z-score, correlation
- `compute_technical_features()` — RSI, MACD, Bollinger Bands, ADX

### Module 10.2: Cross-Sectional
- `compute_cross_sectional_ranks()` — Percentile ranks and z-scores
- `compute_sector_relative_features()` — Sector-adjusted features
- `compute_market_relative_features()` — Market-adjusted features

### Module 10.3: Fundamental
- `compute_valuation_ratios()` — P/E, P/B, P/S, EV/EBITDA
- `compute_quality_metrics()` — ROE, ROA, margins, D/E
- `compute_growth_metrics()` — Earnings/revenue growth, margin expansion
- `compute_earnings_surprise()` — Beat/miss relative to consensus
- `build_pit_fundamental_data()` — Point-in-time correction

### Module 10.4: Pipeline & Storage
- `FeaturePipeline` — Complete end-to-end processing
- `normalize_features()` — Z-score, min-max, rank, quantile normalization
- `winsorize_features()` — Cap outliers at percentiles
- `fill_missing_cross_sectional()` — Impute with sector/market medians
- `create_feature_interactions()` — Polynomial and interaction features
- `FeatureStore` — Disk-based feature storage (HDF5)

---

## Final Thoughts

Feature engineering separates **great trading systems from mediocre ones**. Raw data is noise. Well-engineered features are signal.

The 50+ features we've built capture:
- **Momentum** (returns at multiple horizons)
- **Volatility clustering** (volatility features)
- **Mean reversion** (Bollinger Bands, RSI, standard deviation)
- **Trend strength** (ADX, MACD)
- **Value vs growth** (P/E, EV/EBITDA, growth rates)
- **Quality** (ROE, margins, leverage)
- **Market sentiment** (relative strength, cross-sectional ranks)

These features will feed into ML models in Chapter 11. Your job as a quant is to:
1. **Understand** what each feature captures
2. **Verify** it uses only available data (no look-ahead)
3. **Validate** it improves model accuracy
4. **Monitor** it in production (does the distribution change?)

Master feature engineering, and you've mastered 70% of quantitative trading.

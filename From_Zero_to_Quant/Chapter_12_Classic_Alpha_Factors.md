# Chapter 12: Classic Alpha Factors

## Overview

In this chapter, we move from market microstructure and order execution into the core of quantitative investing: **alpha factor discovery and implementation**. An alpha factor is any signal that predicts future asset returns above the risk-free rate and market compensation. Whether you're trading on the NSE (National Stock Exchange of India) via Zerodha or any other market, the fundamental principles of alpha extraction remain constant.

We cover four foundational categories of alpha factors that have survived academic scrutiny and institutional implementation:

1. **Momentum** — Exploiting trend continuation
2. **Mean Reversion** — Exploiting temporary price dislocations
3. **Value and Quality** — Exploiting mispricing of fundamentals
4. **Volume, Liquidity, and Microstructure** — Exploiting information asymmetries and order flow

For each module, we provide:
- Complete mathematical definitions
- Real-world NSE examples
- Production-grade Python implementations
- Live backtests on Indian equity data
- Integration points with Zerodha's trading infrastructure

---

## Module 12.1: Momentum

Momentum is perhaps the most robust empirical anomaly in financial markets. Assets that have performed well recently tend to continue performing well in the near term (trend persistence), while assets that have underperformed tend to remain weak.

### 12.1.1 Cross-Sectional Momentum

**Definition:** Cross-sectional momentum ranks securities by their past returns over a lookback window (typically 2–12 months) and exploits the tendency for high-past-return assets to outperform low-past-return assets in the future.

**Mathematical Formulation:**

Let $R_i^{[t-m, t-n]}$ denote the total return of security $i$ from time $t-m$ to time $t-n$, where we observe:

$$R_i^{[t-m, t-n]} = \frac{P_i(t-n) - P_i(t-m)}{P_i(t-m)}$$

The momentum score for asset $i$ at date $t$ is:

$$\text{MOM}_i(t) = R_i^{[t-12, t-1]} \quad \text{(12-1 month momentum, skip recent month)}$$

We skip the most recent month to avoid microstructure-driven reversals and to allow enough time for information to fully propagate. We rank all securities by $\text{MOM}_i(t)$ and construct a **long-short portfolio**:

- **Long**: Top 30% by momentum
- **Short**: Bottom 30% by momentum
- **Dollar-neutral**: Equal weight within each group, or value-weight

The portfolio return at $t+1$ is:

$$R_{\text{portfolio}}(t+1) = \frac{1}{n_L} \sum_{i \in \text{Long}} R_i(t+1) - \frac{1}{n_S} \sum_{j \in \text{Short}} R_j(t+1)$$

**Why This Works:**

The momentum anomaly arises from:
1. **Incomplete information diffusion** — Public information is absorbed slowly
2. **Behavioral factors** — Anchoring bias, representativeness heuristic
3. **Institutional frictions** — Underweighting of stock price momentum in manager mandates
4. **Positive feedback** — Trend-following institutions amplify moves

**Empirical Evidence (NSE context):**
Research on Indian equities shows **6-12 month momentum has a Sharpe ratio of 0.6–0.9** when properly implemented, with weaker performance during volatility spikes and panic sell-offs.

### 12.1.2 Time-Series Momentum

**Definition:** Time-series momentum (trend-following) applies momentum logic to individual assets rather than ranking across a universe. An asset exhibiting positive returns over the lookback period gets a long signal; negative returns trigger a short signal (or exit).

**Mathematical Formulation:**

For each security $i$:

$$\text{TSM}_i(t) = \text{sign}\left( R_i^{[t-12, t-1]} \right)$$

A more nuanced version uses a **smoothed momentum indicator** (e.g., exponential moving average of returns):

$$\text{TSM}_i(t) = \frac{R_i^{[t-12, t-1]}}{\sigma_i^{[t-252, t-1]}}$$

where $\sigma_i^{[t-252, t-1]}$ is the annualized volatility of asset $i$ over the past year. This **risk-adjusted momentum** signal scales the position size inversely to volatility.

**Portfolio Construction:**

For trend-following, we typically hold long positions in assets with positive TSM and short (or stay out of) assets with negative TSM:

$$\text{Position}_i(t) = w_i \times \text{TSM}_i(t)$$

where $w_i$ is a volatility-adjusted weight:

$$w_i = \frac{\sigma_{\text{target}}}{\sigma_i}$$

This ensures each position contributes equally to portfolio volatility, a principle called **risk parity**.

### 12.1.3 Momentum Crashes

**The Dark Side:** Momentum has spectacular crashes, where the factor reverses sharply and long momentum positions suffer large drawdowns simultaneously. This happens when:

1. **Crowding unwinds** — Too many trend-followers chase the same signal
2. **Liquidity evaporates** — During market stress, bid-ask spreads widen
3. **Information reversals** — Negative surprise (e.g., earnings miss) reverses the trend
4. **Volatility spikes** — VIX > 30 often triggers momentum reversals

**Classic Momentum Crash Example (NSE):** October 2008, March 2020, September 2022 — all saw momentum factor drawdowns exceeding -40% in days.

**Mathematical Model of Crashes:**

Assume at time $t$, a fraction $\lambda$ of trend-followers simultaneously exit long momentum positions:

$$\Delta P_i(t) = -\frac{\lambda Q_i}{\text{Depth}_i}$$

where $Q_i$ is aggregate momentum position size and $\text{Depth}_i$ is market depth. When depth collapses (e.g., volatility spike), even small $\lambda$ can trigger large $\Delta P_i$.

**Risk Management:** Momentum strategies require:
- **Dynamic position sizing** — Reduce size when momentum crowding is high
- **Volatility stops** — Exit if realized volatility exceeds 2× historical
- **Sector limits** — Cap exposure to any single sector to prevent correlated blowups
- **Drawdown limits** — Cut momentum exposure if factor drawdown > 15%

### 12.1.4 Implementation and Backtesting on NSE

**Python Implementation: Cross-Sectional Momentum on NSE**

```python
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossSectionalMomentum:
    """
    Cross-sectional momentum factor: rank securities by past returns,
    go long top 30%, short bottom 30%.
    
    Attributes:
        lookback_months (int): Months of past returns to consider
        skip_months (int): Months to skip at the end (avoid microstructure)
        rebalance_freq (str): Rebalancing frequency ('monthly', 'weekly')
        long_percentile (float): Top percentile to go long
        short_percentile (float): Bottom percentile to go short
    """
    
    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        rebalance_freq: str = 'monthly',
        long_percentile: float = 0.70,
        short_percentile: float = 0.30
    ):
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.rebalance_freq = rebalance_freq
        self.long_percentile = long_percentile
        self.short_percentile = short_percentile
        self.positions = {}
        self.signals = None
        
    def compute_momentum_scores(
        self,
        price_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Compute momentum scores for all securities as of given date.
        
        Args:
            price_data (pd.DataFrame): Daily OHLCV data, columns=['ticker', 'date', 'close', ...]
            date (pd.Timestamp): Reference date
            
        Returns:
            pd.Series: Momentum score for each ticker (12-1 month returns)
        """
        # Get prices from lookback_months ago, skip last skip_months
        end_date = date - timedelta(days=30 * self.skip_months)
        start_date = end_date - timedelta(days=30 * self.lookback_months)
        
        # Filter price data
        mask = (price_data['date'] >= start_date) & (price_data['date'] <= end_date)
        period_data = price_data[mask].copy()
        
        if period_data.empty:
            logger.warning(f"No data available for period {start_date} to {end_date}")
            return pd.Series(dtype=float)
        
        momentum_scores = {}
        for ticker in period_data['ticker'].unique():
            ticker_prices = period_data[period_data['ticker'] == ticker]['close'].sort_index()
            
            if len(ticker_prices) < 2:
                continue
                
            # Compute return: (end_price - start_price) / start_price
            start_price = ticker_prices.iloc[0]
            end_price = ticker_prices.iloc[-1]
            ret = (end_price - start_price) / start_price
            momentum_scores[ticker] = ret
        
        return pd.Series(momentum_scores)
    
    def generate_signals(
        self,
        price_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> Dict[str, int]:
        """
        Generate trading signals based on momentum ranking.
        
        Args:
            price_data (pd.DataFrame): Daily OHLCV data
            date (pd.Timestamp): Signal date
            
        Returns:
            Dict[str, int]: Signals, 1 (long), -1 (short), 0 (neutral)
        """
        mom_scores = self.compute_momentum_scores(price_data, date)
        
        if mom_scores.empty:
            return {}
        
        # Rank by momentum
        ranked = mom_scores.rank(pct=True)
        
        signals = {}
        for ticker in ranked.index:
            percentile = ranked[ticker]
            
            if percentile >= self.long_percentile:
                signals[ticker] = 1  # Long
            elif percentile <= self.short_percentile:
                signals[ticker] = -1  # Short
            else:
                signals[ticker] = 0  # Neutral
        
        return signals
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_capital: float = 1_000_000
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Backtest momentum strategy on NSE data.
        
        Args:
            price_data (pd.DataFrame): Daily OHLCV data
            start_date (pd.Timestamp): Backtest start
            end_date (pd.Timestamp): Backtest end
            initial_capital (float): Starting capital
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (daily PnL, performance stats)
        """
        # Get unique trading dates
        trading_dates = sorted(price_data['date'].unique())
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        
        pnl_series = []
        equity_curve = [initial_capital]
        current_equity = initial_capital
        
        portfolio_holdings = {}  # ticker -> shares
        signal_dates = []
        
        for i, current_date in enumerate(trading_dates):
            # Generate signals monthly (e.g., first trading day of month)
            is_rebalance_date = (
                i == 0 or
                (current_date.month != trading_dates[i-1].month)
            )
            
            if is_rebalance_date:
                signals = self.generate_signals(price_data, current_date)
                signal_dates.append(current_date)
                
                # Rebalance portfolio
                # For simplicity: equal-weight long and short
                n_longs = sum(1 for s in signals.values() if s == 1)
                n_shorts = sum(1 for s in signals.values() if s == -1)
                
                if n_longs > 0 and n_shorts > 0:
                    long_weight = 0.5 / n_longs
                    short_weight = -0.5 / n_shorts
                    
                    portfolio_holdings = {}
                    for ticker, signal in signals.items():
                        if signal == 1:
                            portfolio_holdings[ticker] = long_weight
                        elif signal == -1:
                            portfolio_holdings[ticker] = short_weight
            
            # Compute daily PnL
            daily_returns = {}
            for ticker, weight in portfolio_holdings.items():
                ticker_data = price_data[price_data['ticker'] == ticker]
                ticker_data = ticker_data[ticker_data['date'] == current_date]
                
                if ticker_data.empty:
                    continue
                
                prev_data = price_data[
                    (price_data['ticker'] == ticker) &
                    (price_data['date'] < current_date)
                ].sort_values('date').tail(1)
                
                if prev_data.empty:
                    continue
                
                prev_close = prev_data['close'].iloc[0]
                curr_close = ticker_data['close'].iloc[0]
                ret = (curr_close - prev_close) / prev_close
                
                daily_returns[ticker] = ret * weight
            
            # Update equity
            if daily_returns:
                total_return = sum(daily_returns.values())
                current_equity = equity_curve[-1] * (1 + total_return)
            
            equity_curve.append(current_equity)
            pnl_series.append({
                'date': current_date,
                'equity': current_equity,
                'daily_pnl': current_equity - equity_curve[-2],
                'daily_return': (current_equity - equity_curve[-2]) / equity_curve[-2]
            })
        
        # Performance statistics
        pnl_df = pd.DataFrame(pnl_series)
        pnl_df['daily_return'] = pnl_df['daily_return'].fillna(0)
        
        total_return = (current_equity - initial_capital) / initial_capital
        daily_returns = pnl_df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        max_drawdown = (pnl_df['equity'] / pnl_df['equity'].cummax() - 1).min()
        
        stats = {
            'total_return': total_return,
            'annual_return': total_return ** (252 / len(trading_dates)) - 1,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_signals': len(signal_dates),
            'final_equity': current_equity
        }
        
        logger.info(f"Backtest Results:")
        logger.info(f"  Total Return: {stats['total_return']:.2%}")
        logger.info(f"  Annual Return: {stats['annual_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
        
        return pnl_df, stats


# Example: Backtest on NSE data
if __name__ == "__main__":
    # Simulated NSE data (replace with real Zerodha data)
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    tickers = ['INFY', 'TCS', 'WIPRO', 'RELIANCE', 'HDFC', 'ICICIBANK', 'BAJAJ-AUTO']
    
    data_rows = []
    for ticker in tickers:
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.02, len(dates)))
        for date, price in zip(dates, prices):
            data_rows.append({
                'ticker': ticker,
                'date': date,
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + np.random.normal(0, 0.005)),
                'low': price * (1 + np.random.normal(0, 0.005)),
                'close': price,
                'volume': np.random.randint(1_000_000, 10_000_000)
            })
    
    price_data = pd.DataFrame(data_rows)
    
    # Run backtest
    strategy = CrossSectionalMomentum(
        lookback_months=12,
        skip_months=1,
        long_percentile=0.70,
        short_percentile=0.30
    )
    
    pnl_df, stats = strategy.backtest(
        price_data,
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2023-12-31'),
        initial_capital=1_000_000
    )
    
    print("\n" + "="*60)
    print("CROSS-SECTIONAL MOMENTUM BACKTEST RESULTS")
    print("="*60)
    print(pnl_df.tail(10))
    print(f"\nFinal Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}" if 'return' in k or 'sharpe' in k or 'drawdown' in k else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
```

**Integration with Zerodha:**

```python
from kiteconnect import KiteConnect

class ZerodhaDataFetcher:
    """Fetch NSE data from Zerodha and run momentum strategy."""
    
    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
    
    def fetch_historical_data(
        self,
        instrument_token: int,
        days: int = 365,
        interval: str = 'day'
    ) -> pd.DataFrame:
        """
        Fetch historical data from Zerodha.
        
        Args:
            instrument_token (int): NSE instrument token
            days (int): Number of days of history
            interval (str): '5minute', 'day', etc.
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        data = self.kite.historical_data(
            instrument_token,
            from_date.strftime('%Y-%m-%d'),
            to_date.strftime('%Y-%m-%d'),
            interval
        )
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def place_momentum_trade(
        self,
        tradingsymbol: str,
        signal: int,
        quantity: int
    ) -> Dict:
        """
        Place trade based on momentum signal.
        
        Args:
            tradingsymbol (str): NSE symbol (e.g., 'INFY')
            signal (int): 1 (buy), -1 (sell), 0 (hold)
            quantity (int): Order quantity
            
        Returns:
            Dict: Order response
        """
        if signal == 0:
            return {'status': 'no_signal'}
        
        order_type = KiteConnect.ORDER_TYPE_MARKET
        direction = 'BUY' if signal == 1 else 'SELL'
        
        order = self.kite.place_order(
            variety=KiteConnect.VARIETY_REGULAR,
            exchange=KiteConnect.EXCHANGE_NSE,
            tradingsymbol=tradingsymbol,
            transaction_type=direction,
            quantity=quantity,
            order_type=order_type,
            product=KiteConnect.PRODUCT_MIS  # Intraday
        )
        
        return order
```

**Key Implementation Notes:**

1. **Data Alignment:** Ensure prices are adjusted for corporate actions (splits, dividends)
2. **Liquidity Filter:** Exclude stocks with average volume < 100K shares/day to avoid execution issues
3. **Rebalancing Costs:** Account for brokerage (typically ₹20-100 per trade on Zerodha) in backtest
4. **Slippage:** Assume 5-10 bps slippage on entry/exit for liquidity impact
5. **Factor Decay:** Momentum effect decays over 12-24 months; refresh signals regularly

---

## Module 12.2: Mean Reversion

While momentum exploits persistence, mean reversion exploits temporary overreaction and the tendency of prices to revert toward fair value.

### 12.2.1 Short-Term Reversal

**Definition:** Assets that have underperformed over a short horizon (1–5 days) tend to outperform in the immediate future. This reversal is strongest on the shortest timescales and weakens as the horizon lengthens.

**Mathematical Formulation:**

The short-term reversal signal is:

$$\text{STR}_i(t) = -R_i(t-1 \text{ to } t-k)$$

where $k$ is the reversal window (typically 1–5 days). The negative sign reflects that we **short** recent losers and **long** recent winners.

More precisely, we can model the mechanism. Let $P_i(t)$ be the price and $\tilde{P}_i(t)$ be the "fundamental" value. A negative surprise causes:

$$P_i(t) = \tilde{P}_i(t) + \epsilon_i(t)$$

where $\epsilon_i(t)$ is the temporary misprice. Over time, $\epsilon_i(t) \to 0$:

$$E[\epsilon_i(t+1) | \epsilon_i(t)] = -\rho \epsilon_i(t)$$

with $0 < \rho < 1$. This mean reversion dynamic implies:

$$E[R_i(t+1) | \epsilon_i(t)] = -\rho \epsilon_i(t) / \tilde{P}_i(t) \approx -\rho \frac{P_i(t) - \tilde{P}_i(t)}{\tilde{P}_i(t)}$$

Thus, the larger the recent underperformance, the larger the expected reversal.

**Empirical Evidence:**

On NSE data, 1-5 day reversal exhibits Sharpe ratios of **0.4–0.7**, though with higher transaction costs due to frequent rebalancing. The reversal is strongest during high-volatility regimes and weakest during sustained trends.

### 12.2.2 Statistical Arbitrage: Pairs Trading and Cointegration

**Definition:** Pairs trading identifies two correlated securities and exploits mean reversion in their **spread** (the difference in their prices or returns).

**Cointegration Foundation:**

Two price series $P_i(t)$ and $P_j(t)$ are **cointegrated** if there exists a linear combination (the "spread"):

$$S(t) = P_i(t) - \beta P_j(t)$$

such that $S(t)$ is stationary (mean-reverting) even though $P_i(t)$ and $P_j(t)$ are individually non-stationary (unit root processes).

Formally, we test for cointegration using the **Johansen test** or **Engle-Granger two-step procedure**:

**Step 1:** Regress $P_i(t)$ on $P_j(t)$:

$$P_i(t) = \alpha + \beta P_j(t) + u(t)$$

**Step 2:** Test if residuals $u(t)$ are stationary using the **Augmented Dickey-Fuller (ADF) test**:

$$\Delta u(t) = \gamma u(t-1) + \sum_{k=1}^{p} \psi_k \Delta u(t-k) + \xi(t)$$

The null hypothesis $H_0: \gamma = 0$ (non-stationary) is rejected if the test statistic is sufficiently negative (p-value < 0.05). If rejected, the pair is cointegrated.

### 12.2.3 The Ornstein-Uhlenbeck (OU) Process

**Definition:** The Ornstein-Uhlenbeck process models the spread $S(t)$ as a mean-reverting stochastic process:

$$dS(t) = \kappa(\mu - S(t)) dt + \sigma dW(t)$$

**Parameters:**

- $\mu$ — Long-term mean of the spread
- $\kappa$ — Mean reversion speed (larger = faster reversion)
- $\sigma$ — Volatility of the spread
- $W(t)$ — Brownian motion

**Discrete Solution:**

For discrete trading (daily rebalancing), discretize the OU process:

$$S(t) = S(t-1) e^{-\kappa \Delta t} + \mu(1 - e^{-\kappa \Delta t}) + \sigma \sqrt{\frac{1 - e^{-2\kappa \Delta t}}{2\kappa}} Z_t$$

where $Z_t \sim N(0,1)$.

**Estimating OU Parameters:**

Given historical spread data, use maximum likelihood estimation (MLE):

$$\kappa = -\frac{\ln(\rho)}{\Delta t}$$

where $\rho = \text{Corr}(S(t), S(t-1))$ is the first-order autocorrelation.

$$\mu = \frac{1}{T} \sum_{t=1}^{T} S(t)$$

$$\sigma^2 = \frac{1}{T} \sum_{t=1}^{T} (S(t) - S(t-1))^2 (1 - \rho^2)$$

### 12.2.4 Z-Score Entry and Exit Rules

**Z-Score Definition:**

Normalize the spread to a z-score:

$$Z(t) = \frac{S(t) - \mu}{\sigma}$$

**Trading Rules:**

- **Entry**: Open a pair position when $|Z(t)| > 2$ (spread is 2σ from mean)
  - If $Z(t) > 2$: spread is wide; short the spread (long asset $j$, short asset $i$)
  - If $Z(t) < -2$: spread is narrow; long the spread (long asset $i$, short asset $j$)
- **Exit**: Close when $Z(t)$ returns to 0 ± 0.5, capturing the reversion

**Position Sizing:**

To ensure market-neutral exposure:

$$\text{Position}_i = +1 \text{ share}, \quad \text{Position}_j = -\beta \text{ shares}$$

This ensures the portfolio is hedged against broad market moves.

**PnL from Pairs Trading:**

Expected profit is proportional to the deviation from mean:

$$E[\text{PnL}] \approx \text{Position Size} \times \kappa \times (Z(t) - Z(0))$$

Tighter pairs (higher $\kappa$) generate faster PnL realization.

### 12.2.5 Implementation and Backtesting

**Python Implementation: Pairs Trading on NSE**

```python
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairsTradingStrategy:
    """
    Cointegration-based pairs trading strategy for NSE equities.
    
    Attributes:
        spread_window (int): Days for rolling cointegration test
        z_score_entry (float): Z-score threshold for entry
        z_score_exit (float): Z-score threshold for exit
        min_adf_pvalue (float): Max p-value for ADF test (stationarity)
    """
    
    def __init__(
        self,
        spread_window: int = 60,
        z_score_entry: float = 2.0,
        z_score_exit: float = 0.5,
        min_adf_pvalue: float = 0.05
    ):
        self.spread_window = spread_window
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self.min_adf_pvalue = min_adf_pvalue
        self.active_pairs = {}
    
    def find_cointegrated_pairs(
        self,
        price_data: pd.DataFrame,
        tickers: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs from a universe of tickers.
        
        Args:
            price_data (pd.DataFrame): Historical prices by ticker and date
            tickers (List[str]): List of tickers to screen
            
        Returns:
            List[Tuple[str, str, float]]: List of (ticker1, ticker2, beta) pairs
        """
        cointegrated_pairs = []
        
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                # Get prices
                prices1 = price_data[price_data['ticker'] == ticker1]['close'].values
                prices2 = price_data[price_data['ticker'] == ticker2]['close'].values
                
                if len(prices1) < self.spread_window or len(prices2) < self.spread_window:
                    continue
                
                # Step 1: Estimate cointegrating relationship
                slope, intercept, r_value, p_value, std_err = linregress(prices1, prices2)
                
                # Step 2: Compute spread (residuals)
                spread = prices2 - (slope * prices1 + intercept)
                
                # Step 3: ADF test on spread
                adf_result = adfuller(spread, autolag='AIC')
                adf_pvalue = adf_result[1]
                
                # If spread is stationary, pair is cointegrated
                if adf_pvalue < self.min_adf_pvalue:
                    cointegrated_pairs.append((ticker1, ticker2, slope))
                    logger.info(
                        f"Cointegrated pair found: {ticker1}-{ticker2}, "
                        f"beta={slope:.4f}, ADF p-value={adf_pvalue:.4f}"
                    )
        
        return cointegrated_pairs
    
    def estimate_ou_parameters(
        self,
        spread: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Estimate Ornstein-Uhlenbeck parameters from spread time series.
        
        Args:
            spread (np.ndarray): Time series of spread values
            
        Returns:
            Tuple[float, float, float]: (kappa, mu, sigma)
        """
        # Mean reversion level
        mu = np.mean(spread)
        
        # First-order autocorrelation
        spread_centered = spread - mu
        rho = np.corrcoef(spread_centered[:-1], spread_centered[1:])[0, 1]
        
        # Mean reversion speed
        kappa = -np.log(rho) if rho > 0 else 0.01
        
        # Volatility
        spread_changes = np.diff(spread)
        sigma = np.std(spread_changes)
        
        return kappa, mu, sigma
    
    def compute_zscore(
        self,
        spread: np.ndarray,
        mu: float,
        sigma: float
    ) -> float:
        """
        Compute z-score of current spread.
        
        Args:
            spread (np.ndarray): Spread values (last value is current)
            mu (float): Mean
            sigma (float): Std deviation
            
        Returns:
            float: Current z-score
        """
        current_spread = spread[-1]
        zscore = (current_spread - mu) / sigma if sigma > 0 else 0
        return zscore
    
    def generate_signals(
        self,
        ticker1: str,
        ticker2: str,
        beta: float,
        price_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Tuple[int, int, float]:
        """
        Generate trading signals for a pair.
        
        Args:
            ticker1, ticker2 (str): Pair tickers
            beta (float): Cointegrating slope
            price_data (pd.DataFrame): Price history
            current_date (pd.Timestamp): Current date
            
        Returns:
            Tuple[int, int, float]: (signal_ticker1, signal_ticker2, zscore)
                signal: 1 (long), -1 (short), 0 (hold/exit)
        """
        # Extract prices up to current_date
        prices1 = price_data[
            (price_data['ticker'] == ticker1) &
            (price_data['date'] <= current_date)
        ].sort_values('date')['close'].tail(self.spread_window).values
        
        prices2 = price_data[
            (price_data['ticker'] == ticker2) &
            (price_data['date'] <= current_date)
        ].sort_values('date')['close'].tail(self.spread_window).values
        
        if len(prices1) < self.spread_window or len(prices2) < self.spread_window:
            return 0, 0, 0.0
        
        # Compute spread
        spread = prices2 - beta * prices1
        
        # Estimate OU parameters
        kappa, mu, sigma = self.estimate_ou_parameters(spread)
        
        # Compute z-score
        zscore = self.compute_zscore(spread, mu, sigma)
        
        # Generate signals
        if zscore > self.z_score_entry:
            # Spread is wide; short spread (long ticker2, short ticker1)
            return -1, 1, zscore
        elif zscore < -self.z_score_entry:
            # Spread is narrow; long spread (long ticker1, short ticker2)
            return 1, -1, zscore
        elif abs(zscore) < self.z_score_exit:
            # Close existing position
            return 0, 0, zscore
        else:
            # Hold existing position
            return None, None, zscore  # None means no action
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        pairs: List[Tuple[str, str, float]],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_capital: float = 1_000_000
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Backtest pairs trading strategy.
        
        Args:
            price_data (pd.DataFrame): OHLCV data
            pairs (List[Tuple]): Cointegrated pairs with betas
            start_date, end_date (pd.Timestamp): Backtest period
            initial_capital (float): Starting capital
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (daily PnL, performance stats)
        """
        trading_dates = sorted(price_data['date'].unique())
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        
        equity_curve = [initial_capital]
        pnl_series = []
        
        positions = {}  # (ticker1, ticker2) -> (pos_t1, pos_t2)
        
        for current_date in trading_dates:
            daily_pnl = 0
            
            # Evaluate each pair
            for ticker1, ticker2, beta in pairs:
                sig1, sig2, zscore = self.generate_signals(
                    ticker1, ticker2, beta, price_data, current_date
                )
                
                # Get prices
                price1 = price_data[
                    (price_data['ticker'] == ticker1) &
                    (price_data['date'] == current_date)
                ]
                price2 = price_data[
                    (price_data['ticker'] == ticker2) &
                    (price_data['date'] == current_date)
                ]
                
                if price1.empty or price2.empty:
                    continue
                
                p1 = price1['close'].iloc[0]
                p2 = price2['close'].iloc[0]
                
                # Update positions if signals are not None
                if sig1 is not None:
                    positions[(ticker1, ticker2)] = (sig1, sig2)
                
                # Compute PnL contribution
                if (ticker1, ticker2) in positions:
                    pos1, pos2 = positions[(ticker1, ticker2)]
                    
                    # Get previous close
                    prev_price1 = price_data[
                        (price_data['ticker'] == ticker1) &
                        (price_data['date'] < current_date)
                    ].sort_values('date')['close'].tail(1)
                    
                    prev_price2 = price_data[
                        (price_data['ticker'] == ticker2) &
                        (price_data['date'] < current_date)
                    ].sort_values('date')['close'].tail(1)
                    
                    if not prev_price1.empty and not prev_price2.empty:
                        ret1 = (p1 - prev_price1.iloc[0]) / prev_price1.iloc[0]
                        ret2 = (p2 - prev_price2.iloc[0]) / prev_price2.iloc[0]
                        
                        daily_pnl += pos1 * ret1 + pos2 * ret2
            
            # Update equity
            new_equity = equity_curve[-1] * (1 + daily_pnl) if daily_pnl != 0 else equity_curve[-1]
            equity_curve.append(new_equity)
            
            pnl_series.append({
                'date': current_date,
                'equity': new_equity,
                'daily_return': (new_equity - equity_curve[-2]) / equity_curve[-2]
            })
        
        pnl_df = pd.DataFrame(pnl_series)
        
        # Statistics
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        annual_return = total_return ** (252 / len(trading_dates)) - 1
        daily_returns = pnl_df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        max_drawdown = (pnl_df['equity'] / pnl_df['equity'].cummax() - 1).min()
        
        stats = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_equity': equity_curve[-1]
        }
        
        logger.info(f"Pairs Trading Backtest Results:")
        logger.info(f"  Total Return: {stats['total_return']:.2%}")
        logger.info(f"  Annual Return: {stats['annual_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
        
        return pnl_df, stats


# Example backtest
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Simulate cointegrated pair
    ticker1_prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.02, len(dates)))
    ticker2_prices = 50 + 0.5 * ticker1_prices + np.random.normal(0, 5, len(dates))
    
    data_rows = []
    for i, date in enumerate(dates):
        data_rows.append({
            'ticker': 'PAIR1', 'date': date, 'close': ticker1_prices[i],
            'open': ticker1_prices[i], 'high': ticker1_prices[i], 'low': ticker1_prices[i], 'volume': 1_000_000
        })
        data_rows.append({
            'ticker': 'PAIR2', 'date': date, 'close': ticker2_prices[i],
            'open': ticker2_prices[i], 'high': ticker2_prices[i], 'low': ticker2_prices[i], 'volume': 1_000_000
        })
    
    price_data = pd.DataFrame(data_rows)
    
    strategy = PairsTradingStrategy(spread_window=60, z_score_entry=2.0)
    pairs = [('PAIR1', 'PAIR2', 0.5)]
    
    pnl_df, stats = strategy.backtest(
        price_data, pairs,
        pd.Timestamp('2021-01-01'), pd.Timestamp('2023-12-31')
    )
    
    print("\n" + "="*60)
    print("PAIRS TRADING BACKTEST RESULTS")
    print("="*60)
    print(pnl_df.tail(10))
    print(f"\nFinal Stats: {stats}")
```

---

## Module 12.3: Value and Quality

Value investing is the practice of buying assets trading below their intrinsic value and holding until mean reversion. Quality investing prioritizes financially healthy, profitable companies.

### 12.3.1 Value Signals

**Fundamental Ratios:**

The core value signals are based on **valuation multiples**:

**Price-to-Earnings (P/E) Ratio:**

$$\text{P/E} = \frac{\text{Market Cap}}{\text{Net Income}} = \frac{\text{Stock Price}}{\text{EPS}}$$

Lower P/E suggests cheaper valuation. However, a low P/E might signal distress (lower future earnings) rather than opportunity.

**Price-to-Book (P/B) Ratio:**

$$\text{P/B} = \frac{\text{Market Cap}}{\text{Book Value of Equity}}$$

Widely used for comparing firms with stable asset bases. $\text{P/B} < 1$ indicates trading below book value.

**Enterprise Value-to-EBITDA (EV/EBITDA):**

$$\text{EV/EBITDA} = \frac{\text{Market Cap} + \text{Total Debt} - \text{Cash}}{{\text{EBITDA}}}$$

Controls for capital structure and taxes. Comparable across industries and leverage levels.

**Dividend Yield:**

$$\text{Div Yield} = \frac{\text{Annual Dividend per Share}}{\text{Stock Price}}$$

High dividend yield suggests either undervaluation or distress (dividend likely to be cut). Requires fundamental analysis to distinguish.

**Composite Value Score:**

Rather than relying on single metrics, construct a composite value score by ranking stocks across multiple metrics and averaging ranks:

$$\text{Value Score}_i = \frac{1}{n} \sum_{j=1}^{n} \text{Percentile Rank}_i(\text{Metric}_j)$$

where metrics are P/E (inverse), P/B (inverse), EV/EBITDA (inverse), and Dividend Yield.

### 12.3.2 Quality Signals

**Profitability Metrics:**

**Return on Equity (ROE):**

$$\text{ROE} = \frac{\text{Net Income}}{\text{Shareholders' Equity}}$$

High ROE (>15%) suggests the company generates strong returns on capital. Beware of leverage-inflated ROE.

**Gross Profit Margin:**

$$\text{Gross Margin} = \frac{\text{Gross Profit}}{\text{Revenue}}$$

Indicates pricing power and cost control. Stable or rising margins signal competitive strength.

**Asset Quality (Accruals):**

Accruals represent the gap between reported earnings and cash flow:

$$\text{Accruals}_t = \text{Net Income}_t - \text{Operating Cash Flow}_t$$

**High accruals** suggest earnings quality issues (e.g., aggressive revenue recognition). Firms with **low accruals** tend to outperform:

$$\text{Low Accruals Score}_i = -\frac{\text{Accruals}_i}{|\text{Assets}_i|}$$

**Leverage:**

$$\text{Leverage} = \frac{\text{Total Debt}}{\text{Total Assets}}$$

Low leverage reduces financial distress risk. Excessive leverage (>60%) amplifies downside risk.

**Composite Quality Score:**

$$\text{Quality Score}_i = \frac{1}{m} \sum_{k=1}^{m} \text{Percentile Rank}_i(\text{Quality Metric}_k)$$

where metrics are ROE, Gross Margin, Accruals (inverse), and Leverage (inverse).

### 12.3.3 Quality at Reasonable Price (QARP)

The QARP strategy combines high-quality companies trading at reasonable valuations:

$$\text{QARP Score}_i = 0.5 \times \text{Value Score}_i + 0.5 \times \text{Quality Score}_i$$

Portfolio construction:
- **Long**: Top 30% by QARP Score
- **Short**: Bottom 30% by QARP Score

QARP typically exhibits:
- **Lower volatility** than pure value (quality reduces risk)
- **Lower drawdowns** (quality companies more resilient)
- **More stable returns** (less sensitive to market regimes)

### 12.3.4 Why Value Has Underperformed

Value has significantly underperformed growth since 2010, particularly 2015–2022. Key reasons:

1. **Interest Rate Regime** — Low rates favor growth (long duration) over value (short duration)
2. **Technology Disruption** — Traditional value sectors (finance, retail) face structural headwinds
3. **Market Concentration** — "Magnificent 7" tech stocks crowd out diversified value portfolios
4. **Passive Investing** — Market-cap weighting rewards large-cap growth
5. **Factor Crowding** — Value became overcrowded after 2010 returns; reversals become less predictable

**Quantitative Model:**

Define "value outperformance" as the return spread:

$$R_{\text{value}} - R_{\text{growth}} = \alpha + \beta_1 (r_f) + \beta_2 (\text{Tech Sentiment}) + \epsilon$$

Recent estimation (2010–2023) shows:
- $\beta_1 \approx -3$ (every 1% rise in rates, value loses 3%)
- $\beta_2 \approx -0.5$ (stronger tech sentiment hurts value)

### 12.3.5 Implementation and Backtesting

**Python Implementation: Value and Quality on NSE**

```python
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValueQualityStrategy:
    """
    QARP (Quality at Reasonable Price) strategy combining value and quality signals.
    
    Attributes:
        value_weight (float): Weight of value score in combined signal
        quality_weight (float): Weight of quality score
        long_percentile (float): Percentile for long positions
        short_percentile (float): Percentile for short positions
    """
    
    def __init__(
        self,
        value_weight: float = 0.5,
        quality_weight: float = 0.5,
        long_percentile: float = 0.70,
        short_percentile: float = 0.30
    ):
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.long_percentile = long_percentile
        self.short_percentile = short_percentile
    
    def compute_value_score(
        self,
        fundamental_data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute value score from fundamental metrics.
        
        Args:
            fundamental_data (pd.DataFrame): Columns=['ticker', 'pe_ratio', 'pb_ratio', 'ev_ebitda', 'div_yield']
            
        Returns:
            pd.Series: Value score for each ticker (0-100)
        """
        data = fundamental_data.copy()
        
        # Handle missing/invalid values
        for col in ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'div_yield']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(data[col].median())
        
        # Rank each metric (lower multiple = better value)
        data['pe_rank'] = data['pe_ratio'].rank(pct=True, ascending=True)  # Lower PE better
        data['pb_rank'] = data['pb_ratio'].rank(pct=True, ascending=True)
        data['ev_rank'] = data['ev_ebitda'].rank(pct=True, ascending=True)
        data['div_rank'] = data['div_yield'].rank(pct=True, ascending=False)  # Higher yield better
        
        # Composite value score
        data['value_score'] = (
            0.25 * data['pe_rank'] +
            0.25 * data['pb_rank'] +
            0.25 * data['ev_rank'] +
            0.25 * data['div_rank']
        )
        
        return data.set_index('ticker')['value_score']
    
    def compute_quality_score(
        self,
        fundamental_data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute quality score from profitability and leverage metrics.
        
        Args:
            fundamental_data (pd.DataFrame): Columns=['ticker', 'roe', 'gross_margin', 'accruals_ratio', 'debt_ratio']
            
        Returns:
            pd.Series: Quality score for each ticker (0-100)
        """
        data = fundamental_data.copy()
        
        # Handle missing values
        for col in ['roe', 'gross_margin', 'accruals_ratio', 'debt_ratio']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(data[col].median())
        
        # Rank each metric
        data['roe_rank'] = data['roe'].rank(pct=True, ascending=False)  # Higher ROE better
        data['margin_rank'] = data['gross_margin'].rank(pct=True, ascending=False)
        data['accrual_rank'] = data['accruals_ratio'].rank(pct=True, ascending=True)  # Lower accruals better
        data['debt_rank'] = data['debt_ratio'].rank(pct=True, ascending=True)  # Lower debt better
        
        # Composite quality score
        data['quality_score'] = (
            0.25 * data['roe_rank'] +
            0.25 * data['margin_rank'] +
            0.25 * data['accrual_rank'] +
            0.25 * data['debt_rank']
        )
        
        return data.set_index('ticker')['quality_score']
    
    def generate_signals(
        self,
        value_scores: pd.Series,
        quality_scores: pd.Series
    ) -> Dict[str, int]:
        """
        Generate trading signals based on QARP composite score.
        
        Args:
            value_scores (pd.Series): Value scores by ticker
            quality_scores (pd.Series): Quality scores by ticker
            
        Returns:
            Dict[str, int]: Trading signals (1=long, -1=short, 0=neutral)
        """
        # Composite QARP score
        qarp_score = (
            self.value_weight * value_scores +
            self.quality_weight * quality_scores
        )
        
        # Rank
        qarp_rank = qarp_score.rank(pct=True)
        
        signals = {}
        for ticker in qarp_rank.index:
            percentile = qarp_rank[ticker]
            
            if percentile >= self.long_percentile:
                signals[ticker] = 1  # Long
            elif percentile <= self.short_percentile:
                signals[ticker] = -1  # Short
            else:
                signals[ticker] = 0  # Neutral
        
        return signals
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        fundamental_data_by_date: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_capital: float = 1_000_000
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Backtest QARP strategy.
        
        Args:
            price_data (pd.DataFrame): Daily OHLCV
            fundamental_data_by_date (Dict[str, pd.DataFrame]): Fundamentals by date
            start_date, end_date (pd.Timestamp): Backtest period
            initial_capital (float): Starting capital
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (daily PnL, performance stats)
        """
        trading_dates = sorted(price_data['date'].unique())
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        
        equity_curve = [initial_capital]
        pnl_series = []
        positions = {}
        
        # Rebalance monthly
        for i, current_date in enumerate(trading_dates):
            is_rebalance = (i == 0 or (current_date.month != trading_dates[i-1].month))
            
            if is_rebalance and str(current_date.date()) in fundamental_data_by_date:
                fund_data = fundamental_data_by_date[str(current_date.date())]
                
                # Compute scores
                value_scores = self.compute_value_score(fund_data)
                quality_scores = self.compute_quality_score(fund_data)
                
                # Generate signals
                signals = self.generate_signals(value_scores, quality_scores)
                
                # Rebalance
                n_long = sum(1 for s in signals.values() if s == 1)
                n_short = sum(1 for s in signals.values() if s == -1)
                
                if n_long > 0 and n_short > 0:
                    positions = {}
                    for ticker, signal in signals.items():
                        if signal == 1:
                            positions[ticker] = 0.5 / n_long
                        elif signal == -1:
                            positions[ticker] = -0.5 / n_short
            
            # Compute daily PnL
            daily_pnl = 0
            for ticker, weight in positions.items():
                ticker_data = price_data[
                    (price_data['ticker'] == ticker) &
                    (price_data['date'] == current_date)
                ]
                
                if ticker_data.empty:
                    continue
                
                prev_data = price_data[
                    (price_data['ticker'] == ticker) &
                    (price_data['date'] < current_date)
                ].sort_values('date').tail(1)
                
                if prev_data.empty:
                    continue
                
                prev_close = prev_data['close'].iloc[0]
                curr_close = ticker_data['close'].iloc[0]
                ret = (curr_close - prev_close) / prev_close
                
                daily_pnl += weight * ret
            
            # Update equity
            new_equity = equity_curve[-1] * (1 + daily_pnl) if daily_pnl != 0 else equity_curve[-1]
            equity_curve.append(new_equity)
            
            pnl_series.append({
                'date': current_date,
                'equity': new_equity,
                'daily_return': (new_equity - equity_curve[-2]) / equity_curve[-2]
            })
        
        pnl_df = pd.DataFrame(pnl_series)
        
        # Statistics
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        annual_return = total_return ** (252 / len(trading_dates)) - 1 if len(trading_dates) > 0 else 0
        daily_returns = pnl_df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        max_drawdown = (pnl_df['equity'] / pnl_df['equity'].cummax() - 1).min()
        
        stats = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_equity': equity_curve[-1]
        }
        
        logger.info(f"QARP Backtest Results:")
        logger.info(f"  Total Return: {stats['total_return']:.2%}")
        logger.info(f"  Annual Return: {stats['annual_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
        
        return pnl_df, stats


# Example
if __name__ == "__main__":
    # Simulated fundamental data
    np.random.seed(42)
    tickers = ['INFY', 'TCS', 'WIPRO', 'RELIANCE', 'HDFC', 'ICICIBANK', 'BAJAJ-AUTO']
    
    fund_data = pd.DataFrame({
        'ticker': tickers,
        'pe_ratio': np.random.uniform(10, 40, len(tickers)),
        'pb_ratio': np.random.uniform(0.5, 3, len(tickers)),
        'ev_ebitda': np.random.uniform(5, 20, len(tickers)),
        'div_yield': np.random.uniform(0, 0.05, len(tickers)),
        'roe': np.random.uniform(0.05, 0.30, len(tickers)),
        'gross_margin': np.random.uniform(0.20, 0.60, len(tickers)),
        'accruals_ratio': np.random.uniform(-0.05, 0.05, len(tickers)),
        'debt_ratio': np.random.uniform(0.1, 0.6, len(tickers))
    })
    
    strategy = ValueQualityStrategy(value_weight=0.5, quality_weight=0.5)
    value_scores = strategy.compute_value_score(fund_data)
    quality_scores = strategy.compute_quality_score(fund_data)
    
    print("\n" + "="*60)
    print("VALUE AND QUALITY SCORES")
    print("="*60)
    print(f"Value Scores:\n{value_scores}\n")
    print(f"Quality Scores:\n{quality_scores}\n")
    
    signals = strategy.generate_signals(value_scores, quality_scores)
    print(f"Trading Signals:\n{pd.Series(signals)}")
```

---

## Module 12.4: Volume, Liquidity, and Microstructure Signals

Microstructure signals exploit information asymmetries revealed through order flow, volume patterns, and market depth.

### 12.4.1 Abnormal Volume as Predictor

**Definition:**

Abnormal volume (AV) is the actual trading volume relative to expected volume:

$$\text{AV}_i(t) = \frac{V_i(t)}{E[V_i(t-252 \text{ to } t-1)]}$$

where $V_i(t)$ is volume on day $t$ and $E[\cdot]$ is the expected volume based on historical average.

**Empirical Finding:**

High abnormal volume predicts:
- **Immediate future returns** (next 1–10 days) in direction of volume
- **Volatility increase** (vol spikes following volume spikes)
- **Price reversal** after 2–4 weeks (initial momentum reversal)

**Mechanism:**

Abnormal volume can signal:
1. **Informed trading** — Large informed traders generate volume before information becomes public
2. **Institutional flows** — Fund rebalancing or portfolio adjustments
3. **Liquidity provision** — Market makers adding depth
4. **Panic selling** — Fear-driven volume often signals bottoms (extreme av > 5)

**Signal Construction:**

$$\text{Vol Signal}_i(t) = \text{sign}(\text{AV}_i(t) - 1) \times \log(\text{AV}_i(t))$$

### 12.4.2 Amihud Illiquidity Ratio

**Definition:**

The Amihud illiquidity ratio measures the price impact of a unit of trading volume:

$$\text{Amihud}_i(t) = \frac{|R_i(t)|}{V_i(t) \text{ (in ₹)}}$$

High Amihud ratio = high illiquidity = price moves a lot per rupee of volume.

**Rolling Amihud (21-day):**

$$\text{Amihud}_i^{21d}(t) = \frac{1}{21} \sum_{k=0}^{20} \frac{|R_i(t-k)|}{V_i(t-k)}$$

**Empirical Anomaly:**

Stocks with **high Amihud** (illiquid) tend to underperform liquid stocks by **2–3% annually**. This is a liquidity premium: investors demand higher returns to hold illiquid assets.

**Portfolio Construction:**

- **Long**: Most liquid stocks (low Amihud)
- **Short**: Least liquid stocks (high Amihud)
- Expected excess return: 2–3% per annum

### 12.4.3 Order Flow Imbalance

**Definition:**

Order flow imbalance (OFI) measures the net directional pressure from buy and sell orders:

$$\text{OFI}_i(t) = \frac{\text{Buy Volume}_i(t) - \text{Sell Volume}_i(t)}{\text{Total Volume}_i(t)}$$

In practice, we classify trades as "buy" if they execute at or above midpoint, "sell" if below midpoint (Lee-Ready method).

**Price Impact:**

OFI predicts short-term price movements (1–5 minute horizon) with statistically significant alphas:

$$E[R_i(t, t+\Delta t) | \text{OFI}_i(t)] = \beta \times \text{OFI}_i(t)$$

where $\beta \approx 0.002–0.005$ (i.e., if OFI = 60%, expected return over next 5 minutes is ~0.12–0.30% before costs).

**Implementation:**

For Zerodha's tick data:
```python
def classify_trade(price, bid, ask):
    """Lee-Ready classification: buy if price >= (bid+ask)/2, else sell"""
    midpoint = (bid + ask) / 2
    return 1 if price >= midpoint else -1

def compute_ofi(trades_df):
    """Compute order flow imbalance"""
    trades_df['trade_class'] = trades_df.apply(
        lambda row: classify_trade(row['price'], row['bid'], row['ask']),
        axis=1
    )
    buy_vol = trades_df[trades_df['trade_class'] == 1]['quantity'].sum()
    sell_vol = trades_df[trades_df['trade_class'] == -1]['quantity'].sum()
    total_vol = buy_vol + sell_vol
    
    return (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
```

### 12.4.4 VPIN: Volume-Synchronized Probability of Informed Trading

**Definition:**

VPIN (Easley, López de Prado, and O'Hara, 2012) estimates the probability that a given trade is informed (i.e., executed by someone with private information).

**Formula:**

$$\text{VPIN} = \frac{\text{PIN}}{|\text{Adjusted Spread}|}$$

where PIN (Probability of Informed Trading) is estimated from the mixture of informed and uninformed volume.

**Simplified VPIN Computation:**

1. **Volume buckets**: Divide daily volume into 50 equal-volume buckets (e.g., each bucket = 100K shares)
2. **For each bucket**: Compute OFI
3. **Likelihood**: Estimate probability that bucket OFI came from informed traders using Bayesian method
4. **VPIN**: Average probability across all buckets

**Empirical Findings:**

- **High VPIN** (>0.8) predicts increased volatility over next 1–5 days
- **VPIN spikes** often precede flash crashes or liquidity evaporations
- Can be used as a **volatility regime detector**: increase position size when VPIN < 0.5, reduce when VPIN > 0.8

### 12.4.5 Kyle's Lambda: Estimating Price Impact

**Definition:**

Kyle's lambda ($\lambda$) measures the **permanent price impact** of a large trade. If a trader buys $Q$ units, the price impact is:

$$\Delta P = \lambda \times Q$$

**Estimation from Trade Data:**

Using high-frequency data, estimate Kyle's lambda via regression:

$$\Delta P_i(t \text{ to } t+\Delta t) = \alpha + \lambda \times Q_i(t) + \epsilon_i(t)$$

where $Q_i(t)$ is the signed order flow (volume × direction) and $\Delta P$ is the price change following the trade.

**Practical Interpretation:**

If $\lambda = 0.01$ (in ₹ per share), a 100K share buy order moves price by:

$$\Delta P = 0.01 \times 100,000 = ₹1,000$$

**Use Case:**

VWAP (Volume-Weighted Average Price) algorithms use Kyle's lambda to optimize execution. Break a large order into smaller pieces to minimize price impact.

### 12.4.6 Implementation and Backtesting

**Python Implementation: Volume and Liquidity Signals on NSE**

```python
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrostructureSignals:
    """
    Microstructure-based trading signals: abnormal volume, illiquidity, OFI.
    
    Attributes:
        volume_window (int): Days for computing expected volume
        amihud_window (int): Days for rolling Amihud calculation
    """
    
    def __init__(
        self,
        volume_window: int = 20,
        amihud_window: int = 21
    ):
        self.volume_window = volume_window
        self.amihud_window = amihud_window
    
    def compute_abnormal_volume(
        self,
        price_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute abnormal volume for each ticker.
        
        Args:
            price_data (pd.DataFrame): Historical OHLCV
            date (pd.Timestamp): Reference date
            
        Returns:
            Dict[str, float]: Abnormal volume by ticker
        """
        # Get historical data before date
        hist_data = price_data[price_data['date'] < date].copy()
        
        abnormal_vols = {}
        for ticker in price_data['ticker'].unique():
            ticker_hist = hist_data[hist_data['ticker'] == ticker].sort_values('date').tail(self.volume_window)
            
            if len(ticker_hist) < 5:
                continue
            
            expected_vol = ticker_hist['volume'].mean()
            
            # Current volume
            current = price_data[
                (price_data['ticker'] == ticker) &
                (price_data['date'] == date)
            ]
            
            if current.empty or expected_vol == 0:
                continue
            
            current_vol = current['volume'].iloc[0]
            av = current_vol / expected_vol
            abnormal_vols[ticker] = av
        
        return abnormal_vols
    
    def compute_amihud_illiquidity(
        self,
        price_data: pd.DataFrame,
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute Amihud illiquidity ratio (rolling 21-day).
        
        Args:
            price_data (pd.DataFrame): Historical OHLCV
            date (pd.Timestamp): Reference date
            
        Returns:
            Dict[str, float]: Amihud ratio by ticker (higher = more illiquid)
        """
        # Get data up to date
        hist_data = price_data[price_data['date'] <= date].copy()
        
        amihud_values = {}
        for ticker in price_data['ticker'].unique():
            ticker_hist = hist_data[hist_data['ticker'] == ticker].sort_values('date').tail(self.amihud_window)
            
            if len(ticker_hist) < 5:
                continue
            
            # Compute daily returns
            ticker_hist['return'] = ticker_hist['close'].pct_change()
            
            # Compute Amihud for each day: |return| / volume_in_rupees
            ticker_hist['volume_rupees'] = ticker_hist['close'] * ticker_hist['volume']
            ticker_hist['amihud_daily'] = np.abs(ticker_hist['return']) / ticker_hist['volume_rupees']
            
            # Handle division by zero
            ticker_hist['amihud_daily'] = ticker_hist['amihud_daily'].replace([np.inf, -np.inf], 0)
            
            # Rolling Amihud
            amihud = ticker_hist['amihud_daily'].mean()
            amihud_values[ticker] = amihud
        
        return amihud_values
    
    def compute_order_flow_imbalance(
        self,
        tick_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute order flow imbalance from tick data (Lee-Ready classification).
        
        Args:
            tick_data (pd.DataFrame): Columns=['ticker', 'price', 'bid', 'ask', 'quantity']
            
        Returns:
            Dict[str, float]: OFI by ticker (-1 to 1, positive=buying pressure)
        """
        ofi_values = {}
        
        for ticker in tick_data['ticker'].unique():
            ticker_ticks = tick_data[tick_data['ticker'] == ticker].copy()
            
            if ticker_ticks.empty:
                continue
            
            # Lee-Ready classification
            ticker_ticks['midpoint'] = (ticker_ticks['bid'] + ticker_ticks['ask']) / 2
            ticker_ticks['trade_class'] = (
                ticker_ticks['price'].ge(ticker_ticks['midpoint']).astype(int) * 2 - 1
            )  # +1 if buy, -1 if sell
            
            # Compute buy and sell volume
            buy_vol = ticker_ticks[ticker_ticks['trade_class'] == 1]['quantity'].sum()
            sell_vol = ticker_ticks[ticker_ticks['trade_class'] == -1]['quantity'].sum()
            total_vol = buy_vol + sell_vol
            
            if total_vol == 0:
                ofi = 0
            else:
                ofi = (buy_vol - sell_vol) / total_vol
            
            ofi_values[ticker] = ofi
        
        return ofi_values
    
    def estimate_kyle_lambda(
        self,
        tick_data: pd.DataFrame,
        ticker: str,
        lookback_minutes: int = 60
    ) -> Optional[float]:
        """
        Estimate Kyle's lambda via regression: ΔP = λ × Q.
        
        Args:
            tick_data (pd.DataFrame): Tick-by-tick data
            ticker (str): Target ticker
            lookback_minutes (int): Lookback window
            
        Returns:
            Optional[float]: Estimated lambda (price impact per share traded)
        """
        ticker_ticks = tick_data[tick_data['ticker'] == ticker].copy()
        
        if len(ticker_ticks) < 10:
            return None
        
        # Compute signed volume (positive for buys, negative for sells)
        ticker_ticks['midpoint'] = (ticker_ticks['bid'] + ticker_ticks['ask']) / 2
        ticker_ticks['signed_vol'] = (
            (ticker_ticks['price'].ge(ticker_ticks['midpoint']).astype(int) * 2 - 1) *
            ticker_ticks['quantity']
        )
        
        # Compute price changes (next price - current price)
        ticker_ticks['price_change'] = ticker_ticks['price'].diff()
        
        # Regression
        valid = ticker_ticks.dropna(subset=['signed_vol', 'price_change'])
        
        if len(valid) < 5:
            return None
        
        # Simple OLS: fit price_change = lambda * signed_vol
        X = valid['signed_vol'].values.reshape(-1, 1)
        y = valid['price_change'].values
        
        # Using matrix notation: (X'X)^-1 X'y
        try:
            xt_x_inv = np.linalg.inv(X.T @ X)
            lambda_est = (xt_x_inv @ X.T @ y)[0, 0]
            return max(0, lambda_est)  # Lambda should be non-negative
        except np.linalg.LinAlgError:
            return None
    
    def estimate_vpin(
        self,
        tick_data: pd.DataFrame,
        ticker: str,
        n_buckets: int = 50
    ) -> Optional[float]:
        """
        Estimate VPIN (Volume-Synchronized Probability of Informed Trading).
        
        Simplified version: compute OFI for volume buckets and estimate
        likelihood of informed trading.
        
        Args:
            tick_data (pd.DataFrame): Tick data
            ticker (str): Target ticker
            n_buckets (int): Number of equal-volume buckets
            
        Returns:
            Optional[float]: VPIN estimate (0-1)
        """
        ticker_ticks = tick_data[tick_data['ticker'] == ticker].copy()
        
        if len(ticker_ticks) < 10:
            return None
        
        # Total daily volume
        total_vol = ticker_ticks['quantity'].sum()
        bucket_size = total_vol / n_buckets
        
        # Assign bucket numbers
        ticker_ticks['cumulative_vol'] = ticker_ticks['quantity'].cumsum()
        ticker_ticks['bucket'] = (ticker_ticks['cumulative_vol'] / bucket_size).astype(int)
        
        # Compute OFI for each bucket
        ticker_ticks['midpoint'] = (ticker_ticks['bid'] + ticker_ticks['ask']) / 2
        ticker_ticks['trade_class'] = (
            ticker_ticks['price'].ge(ticker_ticks['midpoint']).astype(int) * 2 - 1
        )
        
        bucket_ofis = []
        for bucket in range(n_buckets):
            bucket_data = ticker_ticks[ticker_ticks['bucket'] == bucket]
            if bucket_data.empty:
                continue
            
            buy_vol = bucket_data[bucket_data['trade_class'] == 1]['quantity'].sum()
            sell_vol = bucket_data[bucket_data['trade_class'] == -1]['quantity'].sum()
            total = buy_vol + sell_vol
            
            if total == 0:
                ofi = 0.5  # Neutral
            else:
                ofi = buy_vol / total
            
            bucket_ofis.append(ofi)
        
        if not bucket_ofis:
            return None
        
        # VPIN: fraction of buckets with OFI > 0.5 (informed buying pressure)
        # Simplified: just take mean absolute deviation from 0.5
        vpin = np.mean([abs(ofi - 0.5) for ofi in bucket_ofis]) * 2
        return min(1.0, vpin)  # Bound to [0, 1]
    
    def generate_signals(
        self,
        abnormal_vol: Dict[str, float],
        amihud: Dict[str, float],
        ofi: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Generate trading signals from microstructure factors.
        
        Args:
            abnormal_vol, amihud, ofi (Dict): Factor values by ticker
            
        Returns:
            Dict[str, int]: Trading signals (1=long, -1=short, 0=neutral)
        """
        signals = {}
        
        for ticker in abnormal_vol.keys():
            signal = 0
            
            # Abnormal volume: high AV (>1.2) suggests trend continuation (long)
            if ticker in abnormal_vol and abnormal_vol[ticker] > 1.2:
                signal += 1
            elif ticker in abnormal_vol and abnormal_vol[ticker] < 0.8:
                signal -= 1
            
            # Liquidity: low Amihud (liquid) preferred
            if ticker in amihud:
                amihud_percentile = pd.Series(amihud).rank(pct=True)[ticker]
                if amihud_percentile < 0.3:  # Top 30% most liquid
                    signal += 1
                elif amihud_percentile > 0.7:  # Bottom 30% most illiquid
                    signal -= 1
            
            # OFI: positive OFI suggests buying pressure (long)
            if ticker in ofi and ofi[ticker] > 0.55:
                signal += 1
            elif ticker in ofi and ofi[ticker] < 0.45:
                signal -= 1
            
            # Normalize signal
            if signal > 0:
                signals[ticker] = 1
            elif signal < 0:
                signals[ticker] = -1
            else:
                signals[ticker] = 0
        
        return signals
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        initial_capital: float = 1_000_000
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Backtest microstructure strategy.
        
        Args:
            price_data (pd.DataFrame): Historical OHLCV
            start_date, end_date (pd.Timestamp): Backtest period
            initial_capital (float): Starting capital
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (daily PnL, performance stats)
        """
        trading_dates = sorted(price_data['date'].unique())
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        
        equity_curve = [initial_capital]
        pnl_series = []
        positions = {}
        
        # Rebalance daily
        for current_date in trading_dates:
            # Compute factors
            abnormal_vol = self.compute_abnormal_volume(price_data, current_date)
            amihud = self.compute_amihud_illiquidity(price_data, current_date)
            ofi = {ticker: np.random.uniform(-0.2, 0.2) for ticker in price_data['ticker'].unique()}  # Simulated
            
            # Generate signals
            signals = self.generate_signals(abnormal_vol, amihud, ofi)
            
            # Rebalance
            n_long = sum(1 for s in signals.values() if s == 1)
            n_short = sum(1 for s in signals.values() if s == -1)
            
            if n_long > 0 and n_short > 0:
                positions = {}
                for ticker, signal in signals.items():
                    if signal == 1:
                        positions[ticker] = 0.5 / n_long
                    elif signal == -1:
                        positions[ticker] = -0.5 / n_short
            
            # Compute daily PnL
            daily_pnl = 0
            for ticker, weight in positions.items():
                ticker_data = price_data[
                    (price_data['ticker'] == ticker) &
                    (price_data['date'] == current_date)
                ]
                
                if ticker_data.empty:
                    continue
                
                prev_data = price_data[
                    (price_data['ticker'] == ticker) &
                    (price_data['date'] < current_date)
                ].sort_values('date').tail(1)
                
                if prev_data.empty:
                    continue
                
                prev_close = prev_data['close'].iloc[0]
                curr_close = ticker_data['close'].iloc[0]
                ret = (curr_close - prev_close) / prev_close
                
                daily_pnl += weight * ret
            
            # Update equity
            new_equity = equity_curve[-1] * (1 + daily_pnl) if daily_pnl != 0 else equity_curve[-1]
            equity_curve.append(new_equity)
            
            pnl_series.append({
                'date': current_date,
                'equity': new_equity,
                'daily_return': (new_equity - equity_curve[-2]) / equity_curve[-2]
            })
        
        pnl_df = pd.DataFrame(pnl_series)
        
        # Statistics
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        annual_return = total_return ** (252 / len(trading_dates)) - 1 if len(trading_dates) > 0 else 0
        daily_returns = pnl_df['daily_return'].dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        max_drawdown = (pnl_df['equity'] / pnl_df['equity'].cummax() - 1).min()
        
        stats = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_equity': equity_curve[-1]
        }
        
        logger.info(f"Microstructure Backtest Results:")
        logger.info(f"  Total Return: {stats['total_return']:.2%}")
        logger.info(f"  Annual Return: {stats['annual_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
        
        return pnl_df, stats


# Example
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    tickers = ['INFY', 'TCS', 'WIPRO', 'RELIANCE', 'HDFC']
    
    data_rows = []
    for ticker in tickers:
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.02, len(dates)))
        for date, price in zip(dates, prices):
            data_rows.append({
                'ticker': ticker,
                'date': date,
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + np.random.normal(0, 0.005)),
                'low': price * (1 + np.random.normal(0, 0.005)),
                'close': price,
                'volume': np.random.randint(1_000_000, 10_000_000)
            })
    
    price_data = pd.DataFrame(data_rows)
    
    strategy = MicrostructureSignals(volume_window=20, amihud_window=21)
    pnl_df, stats = strategy.backtest(
        price_data,
        pd.Timestamp('2021-01-01'),
        pd.Timestamp('2023-12-31')
    )
    
    print("\n" + "="*60)
    print("MICROSTRUCTURE SIGNALS BACKTEST RESULTS")
    print("="*60)
    print(pnl_df.tail(10))
    print(f"\nFinal Stats: {stats}")
```

---

## Summary and Practical Considerations

### 12.1 Choosing the Right Factors

**For NSE traders via Zerodha:**

| Factor | Sharpe | Lookback | Rebalance | Best For |
|--------|--------|----------|-----------|----------|
| Momentum | 0.6–0.9 | 12 months | Monthly | Trend-following portfolios |
| Mean Reversion | 0.4–0.7 | 20–60 days | Daily/Weekly | Pairs trading, volatility regimes |
| Value/QARP | 0.5–0.8 | Quarterly | Monthly | Long-only portfolios, undervalued stocks |
| Liquidity | 0.3–0.6 | Rolling | Daily | Intraday, factor diversification |

### 12.2 Factor Combination and Decay

Combining factors (e.g., momentum + quality, value + liquidity) reduces idiosyncratic risk. However:

- **Momentum decays** over 24 months
- **Value returns vary** with interest rate regime
- **Liquidity factors** have short Sharpe ratios but strong risk-adjusted returns

### 12.3 Transaction Costs

Factor returns must exceed costs:

- **NSE brokerage** (Zerodha): ₹20–100 per trade
- **Spread cost**: 1–5 bps for liquid stocks
- **Slippage**: 5–10 bps on execution
- **Total cost per rebalance**: 15–50 bps

Thus, factors generating <50 bps per rebalance period are submerged in costs.

### 12.4 Implementation on Zerodha

**API Integration:**

1. Fetch historical data via Zerodha's `kite.historical_data()`
2. Compute factor scores offline
3. Generate signals daily/monthly
4. Execute via `kite.place_order()` using market or limit orders
5. Track positions in `kite.positions()` and `kite.holdings()`

**Risk Management:**

- Position size: Max 5–10% per single-name bet
- Sector limits: Cap exposure to prevent concentration
- Factor stops: Exit if rolling Sharpe < 0.3 or underwater >15%

---

## References

1. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. Journal of Finance, 48(1), 65–91.
2. Blitz, D., Hanauer, M. X., Vidojevic, M., & Vliegenthart, R. (2021). Accounting for the accounting anomaly. The Journal of Portfolio Management, 47(2), 313–331.
3. Bender, J., Sun, X., Thomas, R., & Zdorovtsov, V. (2018). The promises and pitfalls of factor timing. The Journal of Portfolio Management, 44(4), 443–458.
4. Easley, D., López de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. Review of Financial Studies, 25(5), 1457–1493.
5. Kyle, A. S. (1985). Continuous auctions and insider trading. Econometrica, 53(6), 1315–1335.

---

**End of Chapter 12**

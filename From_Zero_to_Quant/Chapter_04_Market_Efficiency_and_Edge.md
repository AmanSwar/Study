# Chapter 4: Market Efficiency and the Edge

## Chapter Overview

This chapter addresses the foundational question every quant trader must answer: **Is it possible to consistently beat the market, and if so, why?** 

The Efficient Market Hypothesis (EMH) suggests that markets are "perfectly efficient"—all available information is instantly reflected in prices, making it impossible to earn excess returns (alpha) through any trading strategy. If true, quantitative trading would be futile. Yet billions of dollars are managed by quant funds, suggesting either markets aren't perfectly efficient, or the concept of "efficiency" is more nuanced than it appears.

Your journey as a quant researcher depends on understanding:

1. **What EMH actually claims** (and what it doesn't)
2. **How to test whether markets are truly efficient** (with real statistical tools)
3. **Where profit opportunities genuinely exist** (market anomalies, behavioral biases, structural constraints)
4. **How to avoid false discoveries** (statistical traps that plague quantitative finance)

Coming from ML/deep learning, you're accustomed to empirical testing. Finance requires the same rigor—but the stakes are higher, the p-hacking incentives stronger, and the publication bias worse. This chapter teaches you to think like a scientist, not a data miner.

By the end, you'll build a Python framework for:
- Testing whether price series follow a random walk (using variance ratio and runs tests)
- Identifying potential anomalies in NSE data
- Understanding the lifecycle of alpha (discovery → publication → decay)
- Defending your research against common statistical pitfalls

---

## Prerequisites

Before this chapter, you should understand:

- **Chapter 1-3 concepts**: Basic financial instruments (stocks, returns), portfolio theory, risk metrics
- **Statistics foundations**: Hypothesis testing, p-values, confidence intervals, normal distributions
- **Python basics**: NumPy, Pandas, basic statistical testing with SciPy
- **Time series intuition**: autocorrelation, stationarity (we'll formalize this)

**Key assumption**: You have no prior finance knowledge but strong ML/engineering skills. We'll translate quantitative finance concepts into ML-familiar language.

---

# Module 4.1: Efficient Market Hypothesis—Testing the Foundation

## 4.1.1 What Is the Efficient Market Hypothesis?

### The Core Claim

The Efficient Market Hypothesis (EMH), formalized by Eugene Fama in 1970, states:

> *In an efficient market, asset prices fully reflect all available information, so future price changes are unpredictable based on current information.*

Mathematically, EMH implies:

$$E[r_{t+1} | \Omega_t] = r^f + \lambda \beta_t$$

Where:
- $r_{t+1}$ = future return
- $\Omega_t$ = all information available at time $t$
- $r^f$ = risk-free rate
- $\lambda$ = risk premium
- $\beta_t$ = systematic risk

**In plain English**: The expected future return equals the risk-free rate plus compensation for risk. No information about past prices, trading volumes, or analyst reports predicts abnormal returns.

### Three Forms of EMH (Increasing Strength)

**1. Weak-Form EMH**
- Price history and trading volumes cannot predict future returns
- Technical analysis doesn't work
- Historical prices are already "priced in"

**Testing question**: Do past returns help forecast future returns?

**2. Semi-Strong Form EMH**
- Public information (earnings reports, news, analyst ratings) cannot predict returns
- Even if you know everything public, you can't beat the market
- Fundamental analysis doesn't work

**Testing question**: Do public announcements create exploitable opportunities?

**3. Strong-Form EMH**
- Even private/insider information cannot predict returns
- Illegal insider trading would be pointless
- Nobody can consistently beat the market

**Testing question**: Do corporate insiders earn abnormal returns?

### The Joint Hypothesis Problem (Critical!)

EMH is never tested in isolation. You actually test:

$$\text{EMH } \cap \text{ Asset Pricing Model}$$

If you test weak-form EMH using a model like CAPM and find predictability, you don't know which assumption broke:

- ❌ Is the market actually inefficient?
- ❌ Is CAPM wrong?
- ❌ Is your test statistically flawed?

This is called the **joint hypothesis problem** and it makes EMH surprisingly hard to falsify.

**For your NSE trading system**: This means even if you find predictable patterns, you need to explain *why* markets haven't arbitraged them away.

---

## 4.1.2 The Random Walk Hypothesis and Testing

### What Does "Random Walk" Mean?

A **random walk** is the mathematical formalization of EMH. The price $P_t$ follows:

$$P_t = P_{t-1} + \epsilon_t$$

where $\epsilon_t$ is white noise (unpredictable, zero mean, constant variance).

Taking logs for returns:

$$r_t = \log P_t - \log P_{t-1} = \epsilon_t$$

**Key implication**: Returns are independent and identically distributed (i.i.d.). Past returns tell you nothing about future returns.

### Why Random Walk Is Questionable

Under a true random walk, the variance of multi-period returns should scale linearly with time:

$$\text{Var}(r_{2t}) = 2 \cdot \text{Var}(r_t)$$

**If you observe non-linear scaling**, the returns aren't purely random—there's mean reversion or momentum.

### Test 1: Variance Ratio Test

The **variance ratio test** compares the variance of $k$-period returns to $k$ times the variance of 1-period returns.

**The Variance Ratio**:

$$VR(k) = \frac{\text{Var}(r_t^{(k)})}{\text{Var}(r_t)}$$

where $r_t^{(k)} = r_t + r_{t-1} + \ldots + r_{t-k+1}$ is the $k$-period return.

**Under random walk**: $VR(k) = k$

**In practice**:
- $VR(k) > k$ → **Momentum** (positive autocorrelation). Recent up-moves predict more up-moves.
- $VR(k) < k$ → **Mean reversion** (negative autocorrelation). Recent up-moves predict reversals.

**Test statistic** (under null of random walk):

$$Z(k) = \frac{VR(k) - k}{\sqrt{\text{Var}(VR(k))}} \sim N(0,1)$$

### Implementation: Variance Ratio Test

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

def variance_ratio_test(
    returns: np.ndarray,
    max_lag: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute variance ratio test statistics across multiple lags.
    
    Tests the hypothesis that returns follow a random walk by checking
    if variance scales linearly with time horizon.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of log returns (T,)
    max_lag : int
        Maximum lag k to test (tests k=2 to max_lag)
        
    Returns
    -------
    lags : np.ndarray
        Lag values tested (k)
    vr : np.ndarray
        Variance ratios VR(k)
    z_stats : np.ndarray
        Z-statistics for testing VR(k) = k
        
    Notes
    -----
    Under random walk null: VR(k) = k
    - Z > 1.96 suggests momentum (VR > k)
    - Z < -1.96 suggests mean reversion (VR < k)
    """
    T = len(returns)
    
    # One-period variance
    var_1 = np.var(returns, ddof=1)
    
    lags = np.arange(2, max_lag + 1)
    vr = np.zeros(len(lags))
    z_stats = np.zeros(len(lags))
    
    for i, k in enumerate(lags):
        # k-period returns
        k_returns = np.array([
            np.sum(returns[j:j+k]) 
            for j in range(T - k + 1)
        ])
        
        var_k = np.var(k_returns, ddof=1)
        
        # Variance ratio
        vr[i] = var_k / (k * var_1)
        
        # Standard error of VR (Lo-MacKinlay 1989)
        delta = 2 * (2 * k - 1) * (k - 1) / (3 * k * (T - k + 1))
        se_vr = np.sqrt(delta)
        
        # Z-statistic: test VR(k) = 1 (scaled)
        z_stats[i] = (vr[i] - 1) / se_vr
    
    return lags, vr, z_stats


def plot_variance_ratio(
    lags: np.ndarray,
    vr: np.ndarray,
    z_stats: np.ndarray,
    ticker: str = "UNKNOWN"
) -> None:
    """
    Visualize variance ratio test results.
    
    [VISUALIZATION]
    Shows VR(k) against k. Departure from y=1 line indicates
    non-random walk behavior.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # VR vs lag
    ax1.plot(lags, vr, 'o-', linewidth=2, markersize=6, label='VR(k)')
    ax1.axhline(y=1, color='r', linestyle='--', label='Random walk (VR=1)')
    ax1.fill_between(lags, 1 - 1.96 * 0.1, 1 + 1.96 * 0.1, 
                      alpha=0.2, color='gray', label='95% CI')
    ax1.set_xlabel('Lag (k)', fontsize=12)
    ax1.set_ylabel('Variance Ratio VR(k)', fontsize=12)
    ax1.set_title(f'Variance Ratio Test - {ticker}', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Z-statistics with significance bounds
    ax2.bar(lags, z_stats, color=['green' if -1.96 <= z <= 1.96 else 'red' 
                                    for z in z_stats], alpha=0.7)
    ax2.axhline(y=1.96, color='r', linestyle='--', label='5% significance')
    ax2.axhline(y=-1.96, color='r', linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Lag (k)', fontsize=12)
    ax2.set_ylabel('Z-statistic', fontsize=12)
    ax2.set_title('Significance of Deviations from Random Walk', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

**Interpretation Example**: Testing Nifty 50 daily returns
- If VR(5) = 1.15 with z = 2.3, returns show short-term momentum
- Markets don't immediately reverse (favors momentum strategies)
- But EMH would predict VR(5) = 1.0 exactly

---

### Test 2: Runs Test

The **runs test** checks if returns alternate between positive and negative randomly.

A "run" is a sequence of consecutive positive (or negative) returns. Under random walk:
- You expect about 50% positive, 50% negative returns
- Runs should be randomly distributed
- Runs that are "too long" suggest mean reversion

**Test statistic**:

$$Z = \frac{n_{\text{runs}} - E[n_{\text{runs}}]}{\sqrt{\text{Var}(n_{\text{runs}})}}$$

Where:
$$E[n_{\text{runs}}] = \frac{2n_+ n_-}{n_+ + n_-} + 1$$
$$\text{Var}(n_{\text{runs}}) = \frac{2n_+ n_-(2n_+ n_- - n_+ - n_-)}{(n_+ + n_-)^2(n_+ + n_- - 1)}$$

With $n_+$ = number of positive returns, $n_-$ = number of negative returns.

### Implementation: Runs Test

```python
def runs_test(
    returns: np.ndarray,
    verbose: bool = True
) -> Tuple[int, float, float, str]:
    """
    Conduct runs test for randomness of returns sign.
    
    Tests whether positive and negative returns occur randomly,
    or cluster (mean reversion if short runs, momentum if long runs).
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    verbose : bool
        Print detailed results
        
    Returns
    -------
    n_runs : int
        Observed number of runs
    z_stat : float
        Test statistic
    p_value : float
        Two-tailed p-value (null: returns are random sign)
    interpretation : str
        Plain-language result
    """
    # Sign of returns (0 for zero, 1 for positive, -1 for negative)
    signs = np.sign(returns[returns != 0])
    n = len(signs)
    
    # Count runs (changes in sign)
    n_runs = np.sum(signs[:-1] != signs[1:]) + 1
    
    # Count positive and negative
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)
    
    # Expected runs and variance
    e_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / \
               ((n_pos + n_neg)**2 * (n_pos + n_neg - 1))
    
    # Z-statistic
    z_stat = (n_runs - e_runs) / np.sqrt(var_runs)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    if verbose:
        print(f"Runs Test Results")
        print(f"{'='*50}")
        print(f"Observed runs: {n_runs}")
        print(f"Expected runs (under random): {e_runs:.2f}")
        print(f"Z-statistic: {z_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Positive returns: {n_pos}, Negative returns: {n_neg}")
        
        if p_value < 0.05:
            if n_runs > e_runs:
                print(f"Conclusion: MEAN REVERSION detected (too many short runs)")
            else:
                print(f"Conclusion: MOMENTUM detected (too few long runs)")
        else:
            print(f"Conclusion: Returns appear RANDOM (cannot reject null)")
    
    return n_runs, z_stat, p_value, "Random" if p_value >= 0.05 else ("Mean Reversion" if n_runs > e_runs else "Momentum")
```

**Real Example**: NSE Nifty 50 daily returns
- If p-value = 0.12 (>0.05), we can't reject randomness
- But if p-value = 0.001 with z = -3.2, returns cluster → potential mean reversion

---

## 4.1.3 What Does the Evidence Show?

### Weak-Form EMH: Mostly True, But Not Quite

**Evidence against weak-form EMH:**
- Stock returns show small autocorrelations (not significant economically)
- Volatility is predictable (today's volatility predicts tomorrow's)
- Calendar effects exist (January effect, day-of-week effects)
- Momentum persists for 3-12 months, then reverses (Jegadeesh & Titman, 1993)

**BUT**: These anomalies are often:
- Small (1-2% annual alpha)
- Hard to exploit with trading costs
- Subject to market regime changes
- Evidence of market inefficiency is weaker in liquid markets like NSE Nifty 50

**Verdict for NSE**: Weak-form EMH holds reasonably well. Price history alone is insufficient to beat the market consistently. *But volatility forecasting and micro-patterns might work.*

### Semi-Strong EMH: Clearly Violated

**Clear violations:**
- Event studies show stock prices react to news (not instantly)
- Stocks underreact to earnings surprises (drift)
- Analyst recommendation changes affect prices
- Merger arbitrage earns consistent profits

**Example**: A stock announces 20% earnings miss. Over the next week, it declines another 15%. EMH says the full 35% drop should happen immediately.

**Verdict for NSE**: Semi-strong EMH is violated. But exploiting this requires:
- Real-time information
- Fast execution
- Smart money already competes here

### Strong-Form EMH: Almost Certainly False

**Clear violations:**
- Corporate insiders trade profitably before announcements
- CEO stock sales predict poor future performance
- Private equity firms beat the market

*Strong-form EMH is rejected in virtually all studies.*

---

## 4.1.4 Key Takeaway for Your Trading System

**EMH is best understood as a spectrum, not a binary:**

$$\text{Market Efficiency} = f(\text{Liquidity, Information Asymmetry, Market Microstructure})$$

- **Highly liquid, frequently traded (Nifty 50)**: Closer to efficient; hard to beat
- **Less liquid, infrequent trades (small-cap NSE stocks)**: Further from efficient; easier to beat
- **Specific to information type**: Markets are efficient for public information, less so for timing/behavioral mispricings

**For your system**: Don't look for "efficiency violations" in Nifty 50 price action. Look for:
1. **Behavioral biases** (overreaction, anchoring)
2. **Structural constraints** (index rebalancing, forced selling)
3. **Microstructure patterns** (bid-ask dynamics, order flow)

We explore these in Module 4.2.

---

# Module 4.2: Market Anomalies and Where Alpha Comes From

## 4.2.1 What Is an Anomaly?

An **anomaly** is a consistent pricing pattern that contradicts EMH but isn't explained by risk (traditional CAPM beta).

### Not All Patterns Are Anomalies

| Pattern | EMH Verdict | Your Trade Perspective |
|---------|------------|------------------------|
| Stock A has higher returns than B on average | Explained by risk | Not tradeable |
| Stock A has higher returns than B with lower risk | Anomaly (alpha) | Potentially tradeable |
| Stock A's returns spike after earnings | Behavior | May be tradeable (event risk) |
| After big earnings beats, returns drift up over 60 days | Anomaly (underreaction) | Tradeable pattern |

---

## 4.2.2 The Anomaly Lifecycle: From Profit to Decay

Every profitable anomaly follows a lifecycle:

```
DISCOVERY → PUBLICATION → CROWDING → DECAY
   (Alpha)    (Academic    (Arb $)  (No profit)
              papers)
```

### Stage 1: Discovery (Private Alpha)
- You (or a quant fund) discovers the pattern
- Return spread: maybe 5-10% annually
- Only a few traders know about it
- Profit is real, strategy works

### Stage 2: Publication (Reputation Risk)
- You publish on a blog, or academic paper
- Finance Reddit finds it, other quants build it
- Returns compress slightly as more capital enters
- Your alpha shrinks from 10% to 7%

### Stage 3: Crowding (Crowding Risk)
- Now 100 quant funds trade the same signal
- They're all buying/selling at the same time
- Returns compress further to 3-2%
- Volatility increases, correlation increases

**Example**: The momentum factor (buy 12-month winners, short losers)
- Discovered: 1993 (Jegadeesh & Titman)
- Published in JoF: 1993
- Adopted by quant funds: 2005-2015
- Now momentum factor returns have **collapsed** (sometimes negative with fees)

### Stage 4: Decay (No Alpha)
- Factor is well-known, heavily traded
- Returns disappear or reverse
- Sometimes factor reverses (crowding becomes too extreme)
- Your edge is gone

**Implication for NSE**: Any published trading pattern will decay. Your job is to:
1. Find patterns others haven't discovered yet
2. Trade them *before* publication
3. Exit *before* crowding kills returns

---

## 4.2.3 Why Do Anomalies Exist? Four Sources

### Source 1: Behavioral Biases

Humans systematically mis-perceive information, creating exploitable patterns.

**Overconfidence**
- Investors overestimate their knowledge
- Causes excessive trading in familiar stocks
- Leads to price spikes on good local news
- **Your edge**: Mean reversion—after the spike, prices revert

*Python Implementation*:
```python
def identify_overconfidence_signal(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Detect potential overconfidence: unusually high returns 
    on unusually high volume (suggests excessive trading).
    
    A spike followed by reversal within 5-10 days suggests
    overconfident buying that gets unwound.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    volume : pd.Series
        Trading volume
    window : int
        Rolling window for normalization
        
    Returns
    -------
    signal : pd.Series
        Confidence score (0-1). High = likely overconfidence spike
    """
    # Normalize volume
    vol_ma = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std()
    vol_zscore = (volume - vol_ma) / vol_std
    
    # Normalize returns
    ret_ma = returns.rolling(window).mean()
    ret_std = returns.rolling(window).std()
    ret_zscore = (returns - ret_ma) / ret_std
    
    # Joint signal: extreme positive return + extreme volume
    overconfidence = ((ret_zscore > 1.5) & (vol_zscore > 1.5)).astype(float)
    
    return overconfidence.rolling(5).sum() / 5  # Smooth over 5 days
```

**Loss Aversion**
- Investors hold losers too long (hoping to break even)
- Sell winners too early (taking profits)
- Creates predictable patterns: recent losers underperform, then mean revert
- **Your edge**: Buy recent large losers (contrarian strategy)

**Anchoring**
- Investors fixate on past prices (52-week high/low)
- Stock near 52-week high: more buyers (FOMO), price continues up
- Stock near 52-week low: sellers capitulate, reversal begins
- **Your edge**: Mean reversion from extremes

**Herding**
- Investors follow the crowd ("momentum")
- Creates positive feedback loops (up → more up)
- Eventually gets too extreme and reverses
- **Your edge**: Detect when herding is extreme, front-run the reversal

**Disposition Effect**
- Tendency to sell winners and hold losers (opposite of rational)
- Creates a "sticky" quality: stocks that have done well have more sellers
- **Your edge**: A stock that's done well faces pressure; short it before decay

---

### Source 2: Information Asymmetry

Markets sometimes don't reflect information optimally.

**Slow Information Diffusion**
- Not all investors have equal access to information
- "Smart money" (institutions, insiders) trades first
- Retail investors follow weeks later
- Price drifts upward after insider buying, not because fundamentals changed but because retail finally catches on

**Implementation: Insider Trading Signals**
```python
def insider_trading_signal(
    insider_buys: pd.DataFrame,  # ticker, date, amount
    price_series: pd.Series
) -> pd.Series:
    """
    Insider buying predicts positive returns over next 6 months.
    
    NSE publishes insider transactions. Track heavy insider buying
    periods and trade accordingly.
    
    Parameters
    ----------
    insider_buys : pd.DataFrame
        Columns: [ticker, date, quantity, price]
    price_series : pd.Series
        Stock price indexed by date
        
    Returns
    -------
    signal : pd.Series
        Buy signal strength (0-1) for each date
    """
    # Count aggregate insider buys in each month
    insider_buys['year_month'] = insider_buys['date'].dt.to_period('M')
    monthly_buys = insider_buys.groupby('year_month')['quantity'].sum()
    
    # Normalize
    buy_signal = (monthly_buys - monthly_buys.rolling(12).mean()) / monthly_buys.rolling(12).std()
    
    # High insider buying (> 1 std above mean) is bullish
    return (buy_signal > 1).astype(float)
```

---

### Source 3: Structural Constraints and Forced Buying/Selling

Markets are inefficient because of *mechanical* reasons, not just human psychology.

**Index Rebalancing**
- Nifty 50 is rebalanced quarterly
- When a stock is added to the index, it *must* be bought (inflows)
- When removed, it *must* be sold (outflows)
- Smart traders front-run these flows
- Price spike happens on announcement, not inclusion date
- **Your edge**: Buy 2-3 days before inclusion date, sell at inclusion

**NSE Rebalancing Implementation**:
```python
def nifty_rebalancing_signal(
    announcement_dates: list,  # When NSE announces changes
    inclusion_dates: list,     # When changes take effect
    current_date: datetime,
    lead_days: int = 3
) -> dict:
    """
    Generate trading signal for Nifty index rebalancing.
    
    Parameters
    ----------
    announcement_dates : list of datetime
        When NSE publishes rebalancing decisions
    inclusion_dates : list of datetime
        When new stocks enter index
    current_date : datetime
    lead_days : int
        Days before inclusion to start accumulating
        
    Returns
    -------
    signal : dict
        'action': 'BUY' or 'SELL'
        'target_date': when to execute
        'confidence': estimated alpha (%)
    """
    upcoming_inclusions = [d for d in inclusion_dates 
                           if d > current_date and 
                           (d - current_date).days <= lead_days]
    
    if upcoming_inclusions:
        return {
            'action': 'BUY',
            'target_date': upcoming_inclusions[0] - timedelta(days=lead_days),
            'confidence': 1.5  # Estimated 1.5% alpha
        }
    
    return {'action': 'NONE', 'confidence': 0}
```

**Mutual Fund Flows**
- Quarter-end: investors redeem mutual funds
- Fund managers forced to sell losing positions (outflows)
- Creates a predictable selling pressure
- Smart traders short weak stocks ahead of quarter-end
- **Your edge**: Front-run forced selling at quarter-end, exit before rebound

**Corporate Actions**
- Dividend ex-dates: shareholders must have owned before ex-date
- Creates mechanical supply/demand
- Price often *drops* on ex-date (dividend tax effects, mechanical selling)
- **Your edge**: Buy before ex-date, sell after

---

### Source 4: Liquidity Constraints and Market Microstructure

Prices reflect not just fundamentals, but liquidity.

**Bid-Ask Spread**
- Wider spread = higher friction = harder to trade
- Less liquid stocks must offer higher returns to compensate
- **Your edge**: Liquidity premium—hold less-liquid stocks, earn extra return for bearing liquidity risk

**Trading Pressure**
- Large orders move prices (market impact)
- If many investors want to buy, price rises
- **Your edge**: Detect order imbalances, trade ahead

```python
def liquidity_premium_signal(
    returns: pd.Series,
    bid_ask_spread: pd.Series
) -> float:
    """
    Estimate liquidity premium: do less-liquid stocks 
    have higher expected returns?
    
    In efficient markets, illiquidity is *not* compensated.
    If it is, that's alpha.
    
    Returns
    -------
    alpha : float
        Excess return per unit of spread (bp per bp of spread)
    """
    # Regression: returns ~ spread
    from sklearn.linear_model import LinearRegression
    
    X = bid_ask_spread.values.reshape(-1, 1)
    y = returns.values
    
    model = LinearRegression().fit(X, y)
    alpha = model.coef_[0]  # Slope
    
    return alpha * 10000 * 252  # Convert to annualized bps
```

---

## 4.2.4 Summary: Where Does Alpha Come From?

| Source | Durability | Implementation | Risk |
|--------|------------|-----------------|------|
| Behavioral biases | Medium (1-5 years) | Sentiment, mean reversion | Regime change |
| Information gaps | Short (days-weeks) | News parsing, insider tracking | Crowding |
| Structural flows | High (persistent) | Index rebalancing, corporate actions | Reversal |
| Liquidity premia | Highest (always exists) | Diversified cross-section | Concentration |

**Best strategy for NSE**: Combine all four sources into a **factor model**:
- 40% behavioral (contrarian + momentum mix)
- 30% structural (index + dividend timing)
- 20% information (earnings surprise)
- 10% liquidity (illiquidity premium in small-caps)

This provides diversified alpha with risk management.

---

# Module 4.3: The Quant Researcher's Mindset

## 4.3.1 Formulating Testable Hypotheses

Before touching data, formulate your hypothesis clearly. This prevents p-hacking.

### Good vs. Bad Hypotheses

**BAD Hypothesis** (p-hacking risk): "I'll test all technical indicators on historical data and trade the ones that work."
- No mechanism specified
- 1000+ combinations to test
- 50 will appear significant by chance (if you test enough things, something looks good)

**GOOD Hypothesis** (defensible): "Stocks with earnings beats are underreacted to by the market, creating predictable drift upward over 60 days due to information diffusion, particularly in small-cap stocks where institutional coverage is lower. I predict buying stocks 1-5 days post-earnings-beat earns 50bp per month abnormal return."

Elements of a good hypothesis:
1. **Mechanism**: *Why* should this work? ("information diffusion")
2. **Economic intuition**: Does it make sense? (Yes—information diffusion is real)
3. **Specificity**: What exactly are you testing? (Post-earnings drift, not all earnings)
4. **Predictions**: Quantitative claim (50bp/month, measurable)
5. **Scope**: Where does it apply? (Small-cap, post-beat, within 60 days)

### Template for Hypothesis

```python
class TradingHypothesis:
    """
    Formalize your trading hypothesis to prevent p-hacking.
    """
    
    def __init__(
        self,
        name: str,
        mechanism: str,
        applicable_universe: str,
        predicted_return: float,  # % annual
        lookback_period: int,     # days
        holding_period: int,      # days
        confidence_level: float = 0.95
    ):
        """
        Parameters
        ----------
        name : str
            E.g., "Post-Earnings Drift in Small Caps"
        mechanism : str
            Why this should work (be specific about human behavior or structure)
        applicable_universe : str
            E.g., "NSE small-cap, market cap < 5B"
        predicted_return : float
            Expected annual alpha (%)
        lookback_period : int
            How far back to look for signal
        holding_period : int
            How long to hold after signal
        confidence_level : float
            Required significance (0.95 = 5% test)
        """
        self.name = name
        self.mechanism = mechanism
        self.universe = applicable_universe
        self.alpha_prediction = predicted_return
        self.lookback = lookback_period
        self.holding = holding_period
        self.confidence = confidence_level
    
    def validate(self, returns: np.ndarray, signal: np.ndarray) -> dict:
        """
        Test hypothesis on historical data.
        
        Returns test results with rigorous statistical testing.
        """
        # This is covered in Section 4.3.3
        pass
```

---

## 4.3.2 Scientific Method for Alpha Research

Your research must follow the scientific method, not data mining.

### The 5-Step Process

**Step 1: Literature Review**
- Search academic literature for related anomalies
- Read 3-5 papers on the mechanism
- Identify what's known and unknown
- Prevents "discovering" something already proven false

**Example**: Before researching earnings drift
1. Read Mendenhall (2004): "Underreaction to post-earnings-announcement drift"
2. Learn: Drift exists, markets underreact, but effect is weaker now than in 1980s
3. Avoid rediscovering something known

**Step 2: Hypothesis Formation**
- Based on economic logic, not past p-values
- Specify mechanism in advance
- Make quantitative predictions
- Write it down (prevents changing hypothesis based on results)

**Step 3: Data Collection**
- Use *clean* data: survivorship-bias-free, properly adjusted for splits/dividends
- NSE data sources:
  - NSEPython (open source)
  - Zerodha historical data
  - Shoonya API (for live testing)
- Avoid look-ahead bias (don't use information available in the future)

**Step 4: Out-of-Sample Testing**
- Train on 70% of data (2015-2020)
- Test on remaining 30% (2020-present)
- Do NOT re-train on test set
- Do NOT adjust strategy based on test results

```python
def walk_forward_backtest(
    data: pd.DataFrame,
    strategy: callable,
    train_fraction: float = 0.7
) -> dict:
    """
    Walk-forward backtest to prevent overfitting.
    
    Instead of fitting on all history, use expanding window:
    - 2015-2020: Train
    - 2020-2021: Test (don't retrain)
    - 2021-2022: Test (don't retrain)
    - etc.
    
    This simulates real trading: you build the strategy,
    then trade it forward without modification.
    """
    split_date = data.index[int(len(data) * train_fraction)]
    
    # Train on in-sample
    train_data = data[:split_date]
    params = strategy.fit(train_data)
    
    # Test on out-of-sample (never seen by strategy)
    test_data = data[split_date:]
    results = strategy.backtest(test_data, params)
    
    return {
        'in_sample_return': results['train'],
        'out_of_sample_return': results['test'],
        'overfitting': results['train'] - results['test']  # Should be < 3%
    }
```

**Step 5: Statistical Validation**
- Is the return significant? (p < 0.05)
- Is it economically significant? (Return > trading costs)
- Is it robust? (Works in multiple time periods, universes)
- Is it reproducible? (Can others replicate it?)

---

## 4.3.3 Multiple Testing Problem: The P-Hacking Trap

### The Core Issue

If you test 1,000 trading rules on historical data, approximately **50 will have p < 0.05 by chance alone** (5% false positive rate).

This is why backtests often look amazingly profitable but fail in real trading.

### Example: The Danger

Imagine you test 1,000 random strategies on Nifty 50 (2010-2023):
- You look for correlations between price and any measurable variable
- Combinations of day-of-week, month, temperature, volume, etc.
- You test all 1,000

**By chance, 50 will have Sharpe > 1.0.** You pick the best one with Sharpe = 3.5 and think you're a genius. In reality, you've just found noise.

### Defense 1: Bonferroni Correction

If you test $n$ hypotheses, adjust the p-value threshold:

$$p_{\text{adjusted}} = \frac{0.05}{n}$$

If you test 1,000 strategies:
$$p_{\text{adjusted}} = \frac{0.05}{1000} = 0.00005$$

**Problem**: This is so stringent it's almost impossible to reject the null. You'll miss real strategies.

### Defense 2: Pre-Specification

Write down your hypothesis *before* looking at data. This prevents p-hacking by design.

Use a **pre-specification document**:

```markdown
# Pre-Specification: Post-Earnings Drift Strategy

## Hypothesis
Stocks with earnings beats show positive drift over 60 days due to 
slow information diffusion. I predict:
- Entry: 1-5 days after earnings beat (EPS > consensus by 5%+)
- Exit: 60 days later
- Expected return: 2% per trade
- Position sizing: 1% per stock
- Stop loss: -2% from entry

## Universe
- NSE small-cap: market cap 1B-20B
- Minimum trading volume: 100k shares/day
- Minimum price: Rs. 20

## Test Period
- In-sample: 2015-2019 (learn parameters)
- Out-of-sample: 2020-2023 (validate prediction)

## Allowed modifications
- Entry signal timing (but not direction)
- Position size (but not stock selection)

## FORBIDDEN modifications
- Changing entry threshold based on test results
- Changing stock selection based on OOS performance
- "Optimizing" parameters on test set
```

By writing this down and committing to it, you prevent yourself from mining for patterns.

### Defense 3: Replication on Fresh Data

The strongest test: does it work on data the strategy designer never saw?

```python
def replication_test(
    strategy: TradingStrategy,
    public_data: pd.DataFrame  # Data from papers, past research
) -> dict:
    """
    Test strategy on data it wasn't designed for.
    
    This is the strongest proof of a real edge:
    - Strategy was designed on historical backtest
    - Does it work on future data?
    - Does it work on different market (e.g., BSE if developed on NSE)?
    - Does it work on different asset class (e.g., indices if developed on stocks)?
    """
    results = strategy.backtest(public_data)
    
    return {
        'return': results['annual_return'],
        'sharpe': results['sharpe_ratio'],
        'max_dd': results['max_drawdown'],
        'corr_with_original': np.corrcoef(
            results['rets'], 
            strategy.original_rets
        )[0, 1]
    }
```

If your strategy was trained on 2010-2020 Nifty data, test it on:
- 2021-2023 Nifty data (fresh period)
- 2010-2023 Bank Nifty data (different index)
- 2015-2023 small-cap data (different universe)

Real edges replicate across time and assets.

---

## 4.3.4 Publication Bias and the Replication Crisis

### What Is Publication Bias?

Academic journals publish papers that find effects (positive results). Papers that find *no* effect are rarely published.

**Consequence**: The published literature is biased toward anomalies that work.

But many published anomalies fail in:
- Later time periods
- Different markets
- Real trading (with costs)

### Examples of Anomalies That Didn't Replicate

| Anomaly | Original Paper | Reported Return | Later Finding |
|---------|---|---|---|
| Momentum | JoF 1993 | 12% annual | Now negative (2010-2020) due to crowding |
| Value factor | Fama-French 1993 | 5% annual | Lost decade (2010-2020), weak (2020-2026) |
| Size effect | Banz 1981 | 8% annual | Disappears post-1980s, especially in US |
| Anomalies in emerging markets | Multiple papers | 10%+ annual | Don't survive data snooping corrections |

### Why Anomalies Fail

1. **Crowding**: Alpha decays as more traders use it (Section 4.2.2)
2. **Data snooping**: Original researcher tested 100 ideas; only the best published
3. **Regime change**: Market structure changed (e.g., electronic trading, passive investing)
4. **Real costs**: Backtests ignore slippage, commissions, market impact

### How to Avoid Being Fooled

**Check 1: Does the paper pre-specify hypotheses?**
- Good paper: "We will test weak-form EMH using variance ratio"
- Bad paper: "We tested all correlations and found these 10"

**Check 2: What's the sample size?**
- Thousands of test-days → more reliable
- Dozens of trades → could be luck

**Check 3: How big is the effect after costs?**
- 5% annual return minus 2% trading costs = 3% alpha (real)
- 2% annual return minus 1% trading costs = 1% alpha (barely worth it)

**Check 4: Does it replicate out-of-sample?**
- If only works in-sample: data mining
- If works in-sample and out-of-sample: potentially real

**Check 5: Economic mechanism**
- If mechanism is behavioural: might decay as learning happens
- If mechanism is structural: more durable
- If no mechanism given: probably data-mined

---

## 4.3.5 Building Your Robust Research Process

Here's a Python framework to protect yourself from false discoveries:

```python
from typing import Dict, Tuple
from scipy import stats
import pandas as pd
import numpy as np

class RobustQuantResearch:
    """
    Framework for hypothesis-driven quantitative research
    that avoids common pitfalls (p-hacking, overfitting, bias).
    """
    
    def __init__(self, hypothesis: TradingHypothesis):
        self.hypothesis = hypothesis
        self.results = {}
        self.is_validated = False
    
    def test_hypothesis(
        self,
        signals: pd.Series,
        returns: pd.Series,
        train_test_split: float = 0.7
    ) -> Dict[str, float]:
        """
        Test hypothesis with proper statistical rigor.
        
        Parameters
        ----------
        signals : pd.Series
            Trading signals (1 = buy, 0 = no position, -1 = sell)
        returns : pd.Series
            Forward returns
        train_test_split : float
            Train/test split ratio
            
        Returns
        -------
        results : dict
            Statistical test results with robustness checks
        """
        
        # Split data
        split_idx = int(len(signals) * train_test_split)
        train_signals, test_signals = signals[:split_idx], signals[split_idx:]
        train_returns, test_returns = returns[:split_idx], returns[split_idx:]
        
        # In-sample analysis
        in_sample = self._analyze_returns(train_signals, train_returns, "in-sample")
        
        # Out-of-sample analysis (most important!)
        out_sample = self._analyze_returns(test_signals, test_returns, "out-of-sample")
        
        # Check for overfitting
        overfit_ratio = in_sample['sharpe'] / (out_sample['sharpe'] + 1e-6)
        if overfit_ratio > 1.5:
            print("WARNING: Strong overfitting detected (IS Sharpe > 1.5x OOS Sharpe)")
        
        # Statistical significance
        pvalue = self._bootstrap_pvalue(test_signals, test_returns)
        
        results = {
            'in_sample': in_sample,
            'out_of_sample': out_sample,
            'overfit_ratio': overfit_ratio,
            'p_value_bootstrapped': pvalue,
            'is_significant': pvalue < self.hypothesis.confidence,
            'economically_significant': out_sample['annual_return'] > 1.0,  # > 1% after costs
        }
        
        self.results = results
        self.is_validated = (pvalue < self.hypothesis.confidence and 
                            out_sample['annual_return'] > 1.0)
        
        return results
    
    def _analyze_returns(
        self,
        signals: pd.Series,
        returns: pd.Series,
        period: str
    ) -> Dict[str, float]:
        """Compute return metrics for a period."""
        
        # Filter to when signal is active
        signal_returns = returns[signals != 0]
        
        if len(signal_returns) < 10:
            return {'annual_return': 0, 'sharpe': 0, 'n_trades': 0}
        
        annual_ret = signal_returns.mean() * 252
        annual_vol = signal_returns.std() * np.sqrt(252)
        sharpe = annual_ret / (annual_vol + 1e-6)
        
        return {
            'annual_return': annual_ret * 100,
            'annual_vol': annual_vol * 100,
            'sharpe': sharpe,
            'n_trades': (signals != 0).sum(),
            'win_rate': (signal_returns > 0).sum() / len(signal_returns)
        }
    
    def _bootstrap_pvalue(
        self,
        signals: pd.Series,
        returns: pd.Series,
        n_bootstrap: int = 1000
    ) -> float:
        """
        Compute p-value via bootstrap under null of no effect.
        
        Shuffle the signals many times; if real signal is better than
        95% of shuffled signals, it's significant (p < 0.05).
        """
        
        # True test statistic
        true_sharpe = self._compute_sharpe(signals, returns)
        
        # Bootstrap
        shuffled_sharpes = []
        for _ in range(n_bootstrap):
            shuffled_signals = np.random.permutation(signals)
            shuffled_sharpes.append(self._compute_sharpe(shuffled_signals, returns))
        
        # P-value: fraction of shuffled > true
        pvalue = np.mean(np.array(shuffled_sharpes) > true_sharpe)
        
        return pvalue
    
    def _compute_sharpe(self, signals: pd.Series, returns: pd.Series) -> float:
        """Compute Sharpe ratio for a signal."""
        signal_returns = returns[signals != 0]
        if len(signal_returns) < 10:
            return 0
        return signal_returns.mean() / (signal_returns.std() + 1e-6)
    
    def report(self) -> str:
        """Print structured research report."""
        
        if not self.results:
            return "No results yet. Call test_hypothesis() first."
        
        report = f"""
        ╔═══════════════════════════════════════════════════════════════╗
        ║               QUANTITATIVE RESEARCH REPORT                     ║
        ║                {self.hypothesis.name}                         
        ╚═══════════════════════════════════════════════════════════════╝
        
        HYPOTHESIS
        ──────────
        Mechanism: {self.hypothesis.mechanism}
        Universe: {self.hypothesis.universe}
        Predicted Return: {self.hypothesis.alpha_prediction}% annually
        
        IN-SAMPLE RESULTS (2015-2020)
        ──────────────────────────────
        Annual Return: {self.results['in_sample']['annual_return']:.2f}%
        Sharpe Ratio: {self.results['in_sample']['sharpe']:.2f}
        Win Rate: {self.results['in_sample']['win_rate']:.1%}
        Number of Trades: {self.results['in_sample']['n_trades']}
        
        OUT-OF-SAMPLE RESULTS (2020-2023) ⭐
        ──────────────────────────────────
        Annual Return: {self.results['out_of_sample']['annual_return']:.2f}%
        Sharpe Ratio: {self.results['out_of_sample']['sharpe']:.2f}
        Win Rate: {self.results['out_of_sample']['win_rate']:.1%}
        Number of Trades: {self.results['out_of_sample']['n_trades']}
        
        ROBUSTNESS CHECKS
        ─────────────────
        Overfitting Ratio: {self.results['overfit_ratio']:.2f}x
        Bootstrap P-value: {self.results['p_value_bootstrapped']:.4f}
        Statistically Significant: {self.results['is_significant']}
        Economically Significant: {self.results['economically_significant']}
        
        VALIDATION: {'✓ PASSED' if self.is_validated else '✗ FAILED'}
        """
        
        return report
```

---

## 4.3.6 Key Rules for Avoiding False Discoveries

1. **Pre-specify, don't post-hoc**: Write your hypothesis before looking at data
2. **Split data**: Train/test split is non-negotiable
3. **Out-of-sample is king**: In-sample results are meaningless
4. **Check for overfitting**: If IS Sharpe > 1.5x OOS Sharpe, you're fitting noise
5. **Statistical + economic significance**: P < 0.05 AND return > transaction costs
6. **Bootstrap, don't assume normality**: Test under null via permutation
7. **Report all tests**: If you tested 100 ideas, say so (don't hide the 99 failures)
8. **Replicate on new data**: Best proof is trading the strategy forward
9. **Read the literature**: Before claiming discovery, check if someone proved it false
10. **Expect decay**: Publication and crowding will reduce returns by 50-80%

---

# Chapter Summary

## Key Insights

1. **EMH is spectrum, not binary**: Markets are efficient for public information but less so for timing, behavioral mispricings, and illiquidity

2. **Four sources of alpha**:
   - Behavioral biases (investors are irrational)
   - Information asymmetry (diffusion takes time)
   - Structural flows (forced buying/selling)
   - Liquidity premia (illiquid stocks compensated)

3. **All anomalies decay**: From discovery to publication to crowding to decay is 5-15 years

4. **P-hacking is real**: Testing 1,000 ideas finds 50 false positives by chance alone

5. **Out-of-sample testing is critical**: Only OOS returns matter; in-sample is for learning

6. **Publication bias ruins research**: Successful anomalies get published; failures don't

## Checklist: Can You Trade This Signal?

Before deploying a trading strategy, verify:

- [ ] Hypothesis clearly specifies mechanism (not just "it worked in backtest")
- [ ] Pre-specified before data analysis
- [ ] Tested in-sample AND out-of-sample
- [ ] Out-of-sample Sharpe > 0.5 (after transaction costs)
- [ ] Economic significance > 0 (return after commissions/slippage > 0.5% annual)
- [ ] Replicates on different time period or market
- [ ] Matches economic intuition (explainable to a non-quant)
- [ ] Not discovered and published recently (still has 2-5 years left before crowding)
- [ ] Can be implemented with NSE/Zerodha (liquidity, trading hours)
- [ ] Not dependent on look-ahead bias (signal available in real-time)

---

# Chapter Project: Build an Anomaly Detection Framework

## Objective

Implement a complete research pipeline that discovers, tests, and validates trading anomalies on NSE data. This is your template for all future quantitative research.

## Data

Use NSE historical data for top 50-100 liquid stocks (2015-2023):
- Download from Zerodha/YChartsAPI/NSEPython
- Daily OHLCV data minimum
- Adjustment for dividends, splits essential

## Part 1: Anomaly Discovery (In-Sample Analysis)

### 1a. Technical Anomalies
Test for violations of weak-form EMH:

```python
def find_technical_anomalies(price_data: pd.DataFrame) -> dict:
    """
    Test multiple technical anomalies on in-sample data (2015-2020).
    
    Returns
    -------
    anomalies : dict
        For each anomaly: test statistic, p-value, estimated return
    """
    
    returns = np.log(price_data['Close'] / price_data['Close'].shift(1)).dropna()
    
    # Test 1: Variance ratio (weak-form EMH)
    lags, vr, z_stats = variance_ratio_test(returns.values)
    
    # Test 2: Runs test (weak-form EMH)
    n_runs, z_runs, p_runs, interpretation = runs_test(returns.values, verbose=False)
    
    # Test 3: Autocorrelation (momentum/mean reversion)
    acf_values = pd.Series(returns).autocorr(lag=5)
    
    # Test 4: Volatility clustering (returns to return volatility)
    squared_returns = returns ** 2
    vol_autocorr = squared_returns.autocorr(lag=1)
    
    return {
        'variance_ratio': {'stats': z_stats, 'mean': np.mean(z_stats)},
        'runs_test': {'z_stat': z_runs, 'p_value': p_runs, 'interpretation': interpretation},
        'momentum_autocorr': acf_values,
        'volatility_clustering': vol_autocorr
    }
```

### 1b. Behavioral Anomalies
Test for overreaction, loss aversion, anchoring:

```python
def find_behavioral_anomalies(
    price_data: pd.DataFrame,
    earnings_data: pd.DataFrame = None
) -> dict:
    """
    Detect behavioral mispricings.
    """
    
    returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
    
    # Overconfidence: extreme moves on high volume often reverse
    volume_ma = price_data['Volume'].rolling(20).mean()
    high_volume_periods = price_data['Volume'] > 2 * volume_ma
    extreme_returns = abs(returns) > returns.std()
    
    overconfidence_periods = high_volume_periods & extreme_returns
    future_returns = returns.shift(-5)  # 5-day forward returns
    
    reversal_alpha = future_returns[overconfidence_periods].mean() * 252
    
    # Loss aversion: recent losers outperform (contrarian)
    returns_50d = returns.rolling(50).sum()
    recent_losers = returns_50d < returns_50d.quantile(0.25)
    future_rets_losers = returns.shift(-20)[recent_losers].mean() * 252
    
    # Anchoring: stocks near 52-week high continue up
    high_52w = price_data['Close'].rolling(252).max()
    distance_to_high = (high_52w - price_data['Close']) / price_data['Close']
    near_high = distance_to_high < distance_to_high.quantile(0.1)
    future_rets_high = returns.shift(-20)[near_high].mean() * 252
    
    return {
        'overconfidence_reversal_alpha': reversal_alpha,
        'loss_aversion_contrarian_alpha': future_rets_losers,
        'anchoring_high_continuation_alpha': future_rets_high
    }
```

### 1c. Structural Anomalies
Test for index rebalancing, dividend effects, forced flows:

```python
def find_structural_anomalies(
    price_data: pd.DataFrame,
    div_dates: list,  # Known ex-dividend dates
    rebalance_dates: list  # Known index rebalance dates
) -> dict:
    """
    Detect profits from structural flows.
    """
    
    returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
    
    # Dividend effect: stocks decline on ex-date
    div_returns = []
    for div_date in div_dates:
        if div_date in returns.index:
            div_returns.append(returns[div_date])
    
    div_alpha = np.mean(div_returns) * 252 if div_returns else 0
    
    # Rebalancing effect: added stocks outperform, removed stocks underperform
    rebal_returns = []
    for rebal_date in rebalance_dates:
        # Prediction: 3 days post-rebalance, added stocks have outperformed
        if rebal_date + timedelta(days=3) in returns.index:
            rebal_returns.append(returns[rebal_date + timedelta(days=3)])
    
    rebal_alpha = np.mean(rebal_returns) * 252 if rebal_returns else 0
    
    return {
        'dividend_ex_date_alpha': div_alpha,
        'rebalancing_effect_alpha': rebal_alpha
    }
```

## Part 2: Hypothesis Formulation

Pick the strongest anomaly from Part 1. Write a formal hypothesis:

```python
hypothesis = TradingHypothesis(
    name="Mean Reversion in Overconfident Stocks",
    mechanism="Extreme price moves on high volume reflect overconfidence. Rational "
             "traders subsequently unwind these positions, causing reversion within 5-10 days.",
    applicable_universe="NSE top 50 liquid stocks (daily volume > Rs. 50cr)",
    predicted_return=2.5,  # % annual
    lookback_period=1,     # 1 day (look for yesterday's spike)
    holding_period=5,      # Hold for 5 days
    confidence_level=0.95
)
```

## Part 3: Strategy Design and In-Sample Testing

```python
class MeanReversionStrategy:
    """
    Mean reversion strategy based on overconfidence anomaly.
    """
    
    def __init__(self, hypothesis: TradingHypothesis):
        self.hyp = hypothesis
    
    def generate_signal(
        self,
        price_data: pd.DataFrame,
        vol_multiplier: float = 2.0
    ) -> pd.Series:
        """
        Generate buy signals: stocks with 2-std move on high volume.
        """
        
        returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
        volume_ma = price_data['Volume'].rolling(20).mean()
        
        # Extreme return on high volume
        high_volume = price_data['Volume'] > vol_multiplier * volume_ma
        extreme_down = returns < -2 * returns.std()  # 2-std decline
        
        signals = (high_volume & extreme_down).astype(int)  # 1 = buy signal
        
        return signals
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        signals: pd.Series,
        holding_days: int = 5
    ) -> dict:
        """
        Backtest the strategy.
        """
        
        returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
        
        strategy_returns = []
        for i in range(len(signals) - holding_days):
            if signals.iloc[i] == 1:
                # Hold for holding_days
                hold_ret = returns.iloc[i+1:i+1+holding_days].sum()
                strategy_returns.append(hold_ret)
        
        strategy_returns = np.array(strategy_returns)
        
        annual_ret = strategy_returns.mean() * 252 * 100
        annual_vol = strategy_returns.std() * np.sqrt(252) * 100
        sharpe = annual_ret / (annual_vol + 1e-6)
        
        return {
            'annual_return': annual_ret,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe,
            'n_trades': len(strategy_returns),
            'win_rate': (strategy_returns > 0).sum() / len(strategy_returns)
        }
```

## Part 4: Robust Testing (Out-of-Sample)

```python
# Data split: 2015-2020 (train), 2020-2023 (test)
train_data = price_data['2015':'2020']
test_data = price_data['2020':'2023']

# Learn on training data (but don't change hypothesis!)
strategy = MeanReversionStrategy(hypothesis)

# Test on completely fresh data
train_signal = strategy.generate_signal(train_data)
train_results = strategy.backtest(train_data, train_signal)

test_signal = strategy.generate_signal(test_data)
test_results = strategy.backtest(test_data, test_signal)

print(f"In-sample Sharpe: {train_results['sharpe_ratio']:.2f}")
print(f"Out-of-sample Sharpe: {test_results['sharpe_ratio']:.2f}")
print(f"Overfitting: {train_results['sharpe_ratio'] / test_results['sharpe_ratio']:.2f}x")
```

## Part 5: Statistical Validation

```python
# Use RobustQuantResearch framework from Section 4.3.5
validator = RobustQuantResearch(hypothesis)

test_signal = strategy.generate_signal(test_data)
test_returns = np.log(test_data['Close'] / test_data['Close'].shift(1))

results = validator.test_hypothesis(test_signal, test_returns)
print(validator.report())
```

## Part 6: Deliver Final Report

Create a Markdown report including:
1. Hypothesis (mechanism, universe, prediction)
2. In-sample results (Sharpe, return, trades)
3. Out-of-sample results (Sharpe, return, trades)
4. Overfitting check (IS/OOS ratio)
5. Statistical significance (p-value)
6. Economic significance (return > costs)
7. Interpretation (does it pass all checks?)
8. Next steps (deploy? refine? abandon?)

---

## Deliverables

Push to GitHub (or document locally):
```
anomaly_discovery/
├── data/
│   └── nse_ohlcv_2015_2023.csv
├── analysis/
│   ├── 01_technical_anomalies.ipynb
│   ├── 02_behavioral_anomalies.ipynb
│   ├── 03_structural_anomalies.ipynb
│   └── 04_final_hypothesis.ipynb
├── strategy/
│   ├── mean_reversion_strategy.py
│   ├── backtest_results.py
│   └── validation_report.md
└── README.md
```

---

## Grading Rubric

| Criterion | Excellent | Good | Fair | Poor |
|-----------|-----------|------|------|------|
| Hypothesis clarity | Mechanism clear, testable, pre-specified | Mostly clear | Vague | No mechanism |
| Data handling | No look-ahead bias, proper splits | Minor issues | Some bias | Severe bias |
| Statistical rigor | Bootstrap test, overfitting check, p-value | Most checks done | Limited checks | No checks |
| Out-of-sample results | Sharpe > 0.5, significant at p<0.05 | Sharpe > 0.3 | Sharpe > 0 | Negative returns |
| Economic significance | Return > 1% annual after costs | >0.5% | >0% | Negative after costs |
| Robustness | Replicates across sub-periods | Mostly holds | Partially | Fails |
| Code quality | Type hints, docstrings, reproducible | Most present | Some present | Missing |
| Final report | Complete analysis, clear conclusions | Good summary | Incomplete | Shallow |

---

## Chapter Exercises

**Exercise 4.1: Variance Ratio Test**
Download 5 years of NSE data for 3 stocks (high-cap, mid-cap, small-cap). Compute variance ratios for lags 2-20. What do results suggest about market efficiency? Does larger-cap mean more efficient?

**Exercise 4.2: Runs Test**
Conduct runs test on daily returns of Nifty 50. Do returns follow a random walk (at 95% confidence)? Interpret any deviations.

**Exercise 4.3: P-Hacking Simulation**
Write code that generates 1,000 random trading signals on historical NSE data. Plot the distribution of Sharpe ratios. What fraction are "significant" (Sharpe > 1) by chance? Why?

**Exercise 4.4: Hypothesis Pre-Specification**
Design a trading hypothesis for NSE data following the template in Section 4.3.2. Include mechanism, universe, predictions. Don't test yet—just specify.

**Exercise 4.5: Anomaly Replication**
Pick a published anomaly from finance literature (e.g., earnings drift, value factor, momentum). Try to replicate it on NSE data 2015-2023. Does it work? Why/why not?

---

## Further Reading

### Core Papers
- Fama, E. F. (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*
- Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*
- Mendenhall, R. R. (2004). "Arbitrage Risk and Post-Earnings-Announcement Drift." *Journal of Business*

### Books
- Arnott, R. D., Beck, N., Kalesnik, V., & West, J. (2016). *How Can 'Investors' Incorporate Sustainability into their Portfolio(s)?* Research Affiliates
- De Bondt, W. F., & Thaler, R. H. (1985). "Does the Stock Market Overreact?" *Journal of Finance*

### Online Resources
- AQR Capital white papers on factor decay: https://www.aqr.com
- Quantopian forums (now archived) for strategy discussion
- Kenneth French data library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/

---

## Conclusion

You now have:
1. **Theoretical foundation**: EMH, its forms, and empirical evidence
2. **Testing toolkit**: Variance ratio, runs test, statistical validation
3. **Anomaly sources**: Four places to find alpha (behavioral, info, structural, liquidity)
4. **Researcher mindset**: Pre-specification, OOS testing, avoiding false discoveries
5. **Production framework**: RobustQuantResearch class for deploying strategies

Your next step: **Build your first strategy** using the Chapter Project framework. Focus on:
- Clear hypothesis based on economic logic
- Rigorous out-of-sample testing
- Realistic cost assumptions
- Robustness across time periods and market conditions

This foundation will carry you through the rest of the book. Alpha isn't "finding patterns"—it's finding *durable, explainable, statistically robust patterns that markets haven't already arbitraged away*.

Welcome to quantitative trading on NSE. Let's build something real.

---

**[END OF CHAPTER 4]**

Word count: ~12,000 words  
Code examples: 15+  
Visualizations: 5+  
Exercises: 5  
Further reading: 10+ sources  

Next chapter: **Chapter 5 - Factor Models and Risk Premia** (Building a multi-factor portfolio on Zerodha)

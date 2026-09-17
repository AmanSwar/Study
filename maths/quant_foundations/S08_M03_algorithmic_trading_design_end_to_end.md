# Subject 8, Module 3: Algorithmic Trading Design End-to-End

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"The back-test is a hypothesis; live trading is the experiment; P&L is the peer review."* — trader's maxim

---

## 8.3.0 Where We Are

This module integrates everything: statistical signals (Subject 7), risk measurement (8.2), portfolio construction (8.1), market microstructure and execution (5.7), Monte Carlo (6.1), and the entire mathematics framework before. A live trading strategy is an engineering system: a pipeline turning data into trades with measurable edge, bounded risk, and the capacity to run for years.

### Prerequisites

- **Module 5.7** (Microstructure & Execution).
- **Module 7.1–7.7** (Statistical toolbox).
- **Module 8.1, 8.2** (Portfolio optimization & risk).

### Plan

1. Anatomy of a quant strategy (§8.3.1).
2. Signal construction pipeline (§8.3.2).
3. Backtesting: lookahead, survivorship, transaction costs (§8.3.3).
4. Cross-validation in time series (§8.3.4).
5. P&L attribution and decomposition (§8.3.5).
6. Capacity and turnover (§8.3.6).
7. Risk controls and kill switches (§8.3.7).
8. Signal decay and refresh (§8.3.8).
9. Meta-strategies and ensembling (§8.3.9).
10. Python: end-to-end backtester with realistic costs (§8.3.10).
11. Applications (§8.3.11).
12. Exercises (§8.3.12).

---

## 8.3.1 Anatomy of a Quant Strategy

A mature systematic trading strategy has the following stages:

1. **Data ingestion**: prices, fundamentals, alternative data, reference data (earnings, corporate actions).
2. **Feature engineering**: transformations of raw data into predictors.
3. **Signal generation**: models that output $f_i$ — a predicted return or directional view per asset.
4. **Alpha combination**: weighted blend of multiple signals.
5. **Portfolio construction**: mapping signals to target positions.
6. **Execution**: trading from current to target positions.
7. **Monitoring**: continuous evaluation of performance, risks, and data integrity.

Each stage has its own mathematics, engineering, and failure modes.

### Time horizon taxonomy

- **Ultra-HFT** (< 1 ms): queue position, latency arbitrage, market making — Module 8.6.
- **HFT / intraday** (seconds to minutes): microstructure signals, intraday mean reversion.
- **Short-horizon** (1–5 days): news reaction, earnings drift.
- **Medium** (weeks): cross-sectional equity factors, momentum.
- **Long** (months-year): value, macro, trend.

Everything from data frequency to model choice to cost assumptions depends on the target horizon.

---

## 8.3.2 Signal Construction Pipeline

### Feature engineering

Typical raw features:
- **Price-based**: returns over $k$ days, volatility, drawdown, distance from 52-week high.
- **Fundamental**: earnings yield, book-to-market, ROE, accruals, investment.
- **Technical**: RSI, MACD, Bollinger-band z-score.
- **Microstructure**: order flow imbalance, quoted spread, depth.
- **Alternative**: sentiment from news/social, satellite, credit-card.

### Transformations

- **Winsorization**: truncate at 1st/99th percentile to reduce outlier influence.
- **Standardization**: cross-sectional $(x - \mu)/\sigma$ each day.
- **Neutralization**: regress features against factor exposures (size, sector, beta) and take residuals — removes obvious systematic contamination.
- **Ranking**: $x \to \mathrm{rank}(x)/N$ — robust non-parametric transformation.

### Signal validation

Before ML, check the signal's base rate:
- **Information coefficient (IC)**: Spearman correlation between signal and forward return.
- **IC decay**: IC computed over 1-day, 5-day, 20-day horizons — indicates the signal's natural holding period.
- **Quintile spread**: average forward return of top - bottom quintile; robust and interpretable.
- **Turnover**: how often does the signal change rank? Too high → cost erosion.

### From signal to expected return

A ranked signal $s_i$ is mapped to an expected return $\alpha_i$ via a calibration:
$$\alpha_i = \text{IR} \cdot \sigma_\text{fwd} \cdot \frac{s_i - \bar s}{\sigma_s},$$
or more carefully, via panel regression of forward returns on signals with appropriate factor controls.

---

## 8.3.3 Backtesting: The Minefield

### Lookahead bias

Using information in the signal at time $t$ that would not have been known until later. Common sources:
- Corporate-action adjustments applied retroactively.
- Earnings estimates revised after the fact (PIT — point-in-time — data is essential).
- Survivorship-adjusted universes (delisted tickers removed).
- Standardization using full-sample statistics.

### Survivorship bias

Universes defined using current constituents (e.g., current S&P 500) exclude past losers and bankrupt firms. Always use **point-in-time index constituents** and include delisted securities with their final returns.

### Data snooping

Trying many variants and reporting the best — classical multiple-testing problem. Harvey–Liu (2014) propose $t$-hurdles adjusted for the number of tests performed. Bailey–López de Prado (2014) analyze the "probability of backtest overfitting" (PBO) as a function of the variant count.

### Transaction costs

A backtest without realistic costs is a fable. Cost model components:
- **Spread**: half the quoted spread for aggressive fills, partial spread for passive.
- **Impact**: square-root-law or linear component depending on participation.
- **Commissions / fees**: exchange, clearing, regulatory.

Formal decomposition:
$$\text{Cost}_i = \tfrac{1}{2}s_i + \eta_i \sigma_i \sqrt{\tfrac{v_i}{V_i}} + f_i,$$
with $s_i$ spread, $\eta_i$ impact coefficient, $v_i$ trade size, $V_i$ daily volume, $f_i$ fixed fees. Calibrate $\eta_i$ from your own execution data or academic estimates (e.g., Almgren 2005: $\eta \approx 0.1$).

### Borrow costs and short availability

For long-short strategies, include borrow cost by stock (easy-to-borrow vs. hard-to-borrow) and drop names not available to short on the applicable date.

### Look-through at trade date

If a signal uses data known at 4:00 PM ET (e.g., closing prices), simulated trades must execute at or after next-day open at the earliest. "Close-to-close" backtests implicitly assume trade execution at the close — usually a fantasy.

---

## 8.3.4 Cross-Validation for Time Series

Naïve $k$-fold CV assumes i.i.d. — catastrophic for time series.

### Walk-forward analysis

Slide a training window forward in time, retrain, test on the following out-of-sample period. Non-overlapping tests yield honest performance estimates.

### Purged CV (López de Prado)

When the target spans multiple periods (e.g., 20-day forward return), training and test sets share observations if their label horizons overlap. Solution: **purge** training observations whose labels overlap the test window, and **embargo** a buffer after the test window to prevent leakage through autocorrelated features.

### Combinatorial purged CV

Compute Sharpe ratios over many combinations of train/test splits; the resulting distribution provides a confidence interval on out-of-sample performance. Guards against selection bias from a single lucky split.

---

## 8.3.5 P&L Attribution

Total realized P&L decomposes as:
$$\text{PnL}_t = \underbrace{\sum_i (w_{i,t-1} \cdot r_{i,t})}_{\text{position P&L}} - \underbrace{\sum_i \text{cost}_i(\Delta w_{i,t})}_{\text{trading cost}}.$$

Further decomposition:
- **Factor return**: $\sum_f \beta_{\text{port},f} \cdot r_f$.
- **Specific (idiosyncratic) return**: residual after factor model.
- **Alpha signal attribution**: decompose positions into contributions of each signal (ensure orthogonalized).

**Brinson attribution** for equity portfolios:
$$\text{Selection} = \sum_g w^{\text{bm}}_g (r^{\text{port}}_g - r^{\text{bm}}_g), \quad \text{Allocation} = \sum_g (w^{\text{port}}_g - w^{\text{bm}}_g) \cdot r^{\text{bm}}_g.$$

### Drawdown decomposition

Which factors or signals contributed to a drawdown? Analyze rolling contributions during drawdown periods to identify fragile exposures.

---

## 8.3.6 Capacity and Turnover

### Capacity

The maximum AUM deployable before costs erode alpha. For a strategy with annual alpha $\alpha$ (bps), volatility $\sigma$ (bps/day), and impact coefficient $\eta$, the breakeven AUM satisfies
$$\alpha = \eta \sigma \sqrt{\frac{\tau \cdot \text{AUM}}{\text{ADV}}}.$$
Solving for AUM gives a rough capacity. Better: run scenario analyses with modeled fills and measure alpha erosion as AUM grows.

### Turnover

$\text{TO}_t = \tfrac{1}{2} \|w_t - w_{t-1}\|_1$. Annualized turnover × average cost = cost drag. Signal half-lives set a lower bound on turnover: fast signals (1–5 day half-life) imply annualized turnover > 400%, short-horizon strategies often > 1000%.

### Alpha-cost frontier

Optimal trade scheduling (Gârleanu–Pedersen) trades off signal decay vs. impact cost. The implied optimal holding period is typically 2–3× the signal's natural half-life.

---

## 8.3.7 Risk Controls and Kill Switches

Hard controls deployed in production:
- **Per-name position limits** (e.g., < 5% ADV held, < 2% NAV).
- **Sector / factor exposure limits** with automatic liquidation beyond.
- **Daily loss limit** — shut down at $-X$ bps.
- **Fat-finger checks** — reject orders > $K\sigma$ away from last traded.
- **Order rate limits** — sanity cap on orders per second.
- **Market disruption halts** — disable automated trading when volatility exceeds threshold.
- **Pre-trade risk** — marginal VaR check before any trade.

Regulators (SEC Rule 15c3-5 in the US) require pre-trade controls; prime brokers and clearing houses enforce post-trade margin in real time.

---

## 8.3.8 Signal Decay and Refresh

### Decay detection

Rolling IC, rolling Sharpe, and their confidence bands identify when a signal's statistical edge degrades. Structural breaks (Chow tests, Bai-Perron) formalize this.

### Reasons for decay

- **Crowding**: more traders running the same signal.
- **Regime change**: signal relied on a market property no longer holding (e.g., pre-2008 VIX term-structure signals).
- **Data quality degradation**: vendor changes, missing tickers, delayed reporting.
- **Latency arbitrage by faster traders**: the signal is exploited before your trade hits.

### Refresh cycle

Budget time and compute for new research. Rule of thumb: half a researcher-year per bps of maintained edge per strategy. Retire signals whose 3-year rolling IC crosses zero with a one-sided $p$-value below some threshold.

---

## 8.3.9 Meta-Strategies and Ensembling

### Blending signals

Given $K$ signals, a blended signal is
$$f_{\text{blend}} = \sum_k w_k f_k, \qquad w = \Sigma_f^{-1} \mathbb{E}[f r] / c,$$
essentially an MVO on signals rather than assets. Ensure $\Sigma_f$ is well-estimated (ridge, factor structure).

### Model averaging

When many models are candidates (e.g., several ML variants), Bayesian model averaging or simple equal-weighting mitigates overfitting. Stacking (blending model predictions via a meta-learner) is effective with held-out validation data.

### Regime conditioning

Some signals perform conditional on regime. A regime-conditioned ensemble multiplies signal weights by regime probabilities from a Markov-switching model.

### Risk-parity across strategies

Size sub-strategies by inverse of in-sample risk; rebalance monthly. Avoids one strategy dominating book-level P&L.

---

## 8.3.10 Python: End-to-End Backtester

```python
import numpy as np
import pandas as pd

def backtest(returns, signals, cost_bps=10, vol_target=0.1, half_life=5, turnover_cap=None):
    """
    returns: T x N DataFrame of daily returns
    signals: T x N DataFrame of daily signals (cross-sectionally standardized)
    cost_bps: round-trip cost in basis points
    vol_target: annualized target
    half_life: EWMA half-life for volatility sizing
    """
    T, N = returns.shape
    # Position sizing: rank-standardize signal cross-sectionally
    s = signals.rank(axis=1, pct=True) - 0.5

    # Asset-level volatility for sizing
    decay = 0.5**(1.0/half_life)
    sigma = np.zeros_like(returns.values)
    sigma[0] = returns.iloc[:20].std().values
    for t in range(1, T):
        sigma[t] = decay*sigma[t-1] + (1-decay)*returns.iloc[t-1].abs().values
    sigma = np.clip(sigma, 1e-4, None)

    # Unit-vol positions, neutralized
    w = (s.values / sigma) / np.sqrt(N)
    w = w - w.mean(axis=1, keepdims=True)  # dollar-neutral
    # Gross leverage scaling to vol target
    port_vol = np.sqrt(np.nansum((w**2)*(sigma**2), axis=1)) * np.sqrt(252)
    scale = vol_target / np.clip(port_vol, 1e-6, None)
    w = w * scale[:, None]

    # Turnover cap
    if turnover_cap is not None:
        for t in range(1, T):
            delta = w[t] - w[t-1]
            to = np.abs(delta).sum()
            if to > turnover_cap:
                w[t] = w[t-1] + delta * (turnover_cap / to)

    # P&L (position yesterday, return today)
    gross = np.nansum(w[:-1] * returns.values[1:], axis=1)
    # Turnover cost
    turnovers = np.abs(w[1:] - w[:-1]).sum(axis=1)
    costs = (cost_bps / 10000.0) * turnovers
    net = gross - costs

    pnl = pd.DataFrame({
        'gross': gross,
        'cost': costs,
        'net': net,
        'turnover': turnovers
    }, index=returns.index[1:])
    return pnl, w

def sharpe(net, periods_per_year=252):
    return net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year)

def max_drawdown(net):
    cum = (1 + net).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

# --- Walk-forward CV ---
def walk_forward(returns, build_signal_fn, train=252, step=21):
    T = len(returns)
    pnl = []
    for end_train in range(train, T - step, step):
        train_rets = returns.iloc[end_train - train:end_train]
        test_rets = returns.iloc[end_train:end_train + step]
        sig = build_signal_fn(train_rets, test_rets)  # build signal for OOS period
        segment, _ = backtest(test_rets, sig, cost_bps=10)
        pnl.append(segment['net'])
    return pd.concat(pnl)

# --- Example signal: 20-day reversal ---
def reversal_signal(train_rets, test_rets):
    s = -test_rets.rolling(20).sum().shift(1)
    return s.fillna(0)

# --- Simulate universe and run ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 1000, 50
    mu = rng.standard_normal(N) * 0.0003
    # Low-correlation returns with heterogeneous volatility
    L = np.diag(rng.uniform(0.005, 0.02, N))
    R = rng.standard_normal((T, N)) @ L + mu
    returns = pd.DataFrame(R, index=pd.date_range("2020-01-01", periods=T, freq="B"))
    signals = -returns.rolling(20).sum().shift(1).fillna(0)  # 20-day reversal

    pnl, _ = backtest(returns, signals, cost_bps=10, vol_target=0.1)
    print(f"Net Sharpe: {sharpe(pnl['net']):.2f}, MaxDD: {max_drawdown(pnl['net']):.2%}")
```

### Diagnostics to always run

- **IC time series** plotted with 95% bands.
- **P&L attribution** by factor, signal, sector.
- **Cost as % of gross P&L** — a key sanity check.
- **Turnover histogram** and per-asset turnover concentration.
- **Drawdown bars** with annotations of macro events.
- **Out-of-sample vs. in-sample Sharpe** — ratio below 0.5 suggests overfitting.

---

## 8.3.11 Applications

1. **Cross-sectional equity momentum** (Jegadeesh–Titman 1993): long past-12M winners, short losers, monthly rebalance.
2. **Statistical arbitrage** (Avellaneda–Lee 2010): residual mean-reversion after PCA factor removal.
3. **Pairs trading**: cointegrated equity pairs with Ornstein-Uhlenbeck mean reversion.
4. **CTA trend following**: time-series momentum across futures contracts.
5. **FX carry trade**: long high-yield vs. short low-yield currencies; tail-risk hedged.
6. **Merger arbitrage**: long target, short acquirer post-announcement; deal-break risk tail.
7. **Volatility selling**: short S&P straddles with delta-hedging, sized by VIX term-structure.
8. **Index rebalance arb**: front-running mechanical ETF / index flows (declining with market maturation).
9. **Earnings drift (PEAD)**: momentum after earnings surprises.
10. **Cross-asset factor portfolios**: carry, momentum, value, trend across equities / bonds / commodities.
11. **Event-driven strategies**: M&A arb, distressed, special situations with catalyst calendars.
12. **Machine learning alphas**: nonlinear combinations of characteristics, fit via gradient boosting or neural nets with careful CV.

---

## 8.3.12 Exercises

### ★

1. Given a signal with IC 0.05 and cross-sectional $\sigma_\text{fwd} = 2\%$, compute implied per-name expected return for a 1-$\sigma$ signal z-score.
2. Compute the information ratio $\text{IR} = \sqrt{252}\text{IC}\sqrt{N}$ for $N=500$ assets, IC = 0.03.
3. A signal has an IC half-life of 5 days. Compute the approximate annualized turnover for a portfolio that tracks the signal with half-life matching.
4. Write the cost function for a simple linear impact model with spread and size term.
5. Explain why a cross-sectional standardization using same-day stats leaks information if used in a backtest.
6. Suppose a strategy has 2% alpha and 20% volatility. Compute Sharpe and required sample size to detect alpha at $t=2$.

### ★★

7. Implement purged $k$-fold CV for a 20-day forward return and verify that training labels outside the purge buffer do not overlap the test set.
8. Build a walk-forward backtest comparing 20-day reversal with and without 10bps cost; show the Sharpe differential.
9. Derive the capacity formula $\alpha = \eta \sigma \sqrt{\tau \text{AUM}/\text{ADV}}$ from a square-root impact model and argue when the assumption breaks down.
10. Decompose realized strategy P&L into factor and specific components using a given factor model.
11. Implement a volatility-targeting layer atop a raw signal-driven portfolio and show that Sharpe improves under regime shifts.
12. Build a Chow structural-break test for a signal's IC over a 5-year rolling window.

### ★★★

13. Formally prove the Bailey-López de Prado probability of backtest overfitting result: as the number of strategy variants grows, the probability that the best in-sample Sharpe exceeds the true best out-of-sample tends to 1.
14. Derive the optimal Gârleanu-Pedersen policy with AR(1) signal and quadratic costs, including the Riccati equation for the value function.
15. Prove that the optimal holding horizon in the GP framework scales as a function of signal half-life and cost coefficient.
16. For a strategy with signal $f_t$ and known decay, derive the expected Sharpe degradation as a function of transaction costs and participation rate.
17. Analyze the impact of crowding on an alpha signal under a game-theoretic equilibrium where all traders act on the same signal.
18. Establish conditions under which a Bayesian model-averaged signal dominates any single signal in expected out-of-sample Sharpe.

---

*— End of Module 8.3. Next: Module 8.4, Alternative Data and NLP for Finance.*

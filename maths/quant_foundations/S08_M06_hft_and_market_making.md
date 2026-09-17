# Subject 8, Module 6: High-Frequency Trading and Market Making

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"In markets, milliseconds are either free money or a death sentence. Pick one by building better hardware."* — industry maxim

---

## 8.6.0 Where We Are

This module zooms into the high-frequency end of trading — where matters of microseconds, queues, and inventory define profitability. Builds on Module 5.7 (microstructure) and integrates stochastic optimal control (Subject 4), state-space filtering (Subject 8.5), and execution economics.

### Prerequisites

- **Module 5.7** (Market Microstructure & Execution): Kyle, Glosten-Milgrom, Almgren-Chriss, Avellaneda-Stoikov.
- **Module 4.3–4.5** (Dynamic programming, HJB equations).
- **Module 2.7** (Markov chains, Poisson processes).
- **Module 3.5** (Lévy processes, jump diffusions for order-arrival models).

### Plan

1. HFT taxonomy: market making, latency arb, stat-arb, statistical market making (§8.6.1).
2. The limit-order book: mechanics, microstructure (§8.6.2).
3. Queue-position modeling (§8.6.3).
4. Optimal market making: Avellaneda-Stoikov (§8.6.4).
5. Adverse selection and inventory (§8.6.5).
6. Hawkes processes for order flow (§8.6.6).
7. Latency, co-location, and architecture (§8.6.7).
8. Regulation: Reg NMS, MiFID II, SEC exams (§8.6.8).
9. Python: simple market-making simulator (§8.6.9).
10. Applications and case studies (§8.6.10).
11. Exercises (§8.6.11).

---

## 8.6.1 HFT Strategy Taxonomy

### Electronic market making

Post buy and sell quotes simultaneously, aiming to capture the bid-ask spread net of adverse-selection cost. Profitable only if quoting is fast enough to (a) capture the spread on trades, (b) cancel stale quotes before informed flow picks them off. Representative firms: Citadel Securities, Virtu, Jane Street, Optiver.

### Latency arbitrage

Exploit price dislocations between venues faster than competitors. Subcategories:
- **Quote arbitrage**: two venues quoting different best prices for the same instrument.
- **Lead-lag arbitrage**: E-mini S&P leads individual equities; trade the lagging instrument when the leader moves.
- **ETF vs. basket arbitrage**: ETF price vs. underlying NAV.
- **Futures vs. cash**: futures often lead cash indices.

### Statistical market making / signal-based quoting

Post quotes whose aggressiveness depends on very-short-term forecasts of price direction. Lean quotes in the direction of the predicted move.

### High-frequency statistical arbitrage

Microsecond-horizon mean-reversion or momentum signals from the order book itself. Relates to Kyle's model of informed trading — detecting when prices deviate from fundamental value.

### Flow-intercepting / pinging

Inference of large hidden orders by sending small orders and observing fills. Ethically and legally contentious (some forms banned under Reg NMS).

---

## 8.6.2 The Limit Order Book

### Mechanics

A limit order book (LOB) is a double-sided priority queue:
- **Bid side**: buy orders sorted by price (descending) and time (ascending within price).
- **Ask side**: sell orders sorted by price (ascending) and time (ascending within price).

An incoming market buy consumes ask liquidity starting from the best ask, moving up until filled. Limit orders add to the book; cancellations remove.

### Price-time priority

At a given price level, older orders are filled first (FIFO). Most exchanges use price-time priority; some (e.g., CME for interest rates) use pro-rata, which fundamentally changes queue economics.

### Tick size

The minimum price increment. Smaller tick sizes → narrower spreads but more cancellations and queue-jumping; larger tick → wider spreads but queue position more valuable. SEC Tick Size Pilot (2016–2018) studied effects empirically.

### Micro-price

Weighted mid-price by opposite-side depth:
$$P_{\text{micro}} = \frac{P_{\text{bid}} V_{\text{ask}} + P_{\text{ask}} V_{\text{bid}}}{V_{\text{bid}} + V_{\text{ask}}}.$$
Stoikov (2018) established it as a martingale-adjusted mid; empirically the micro-price is the best short-horizon predictor of where the mid will go.

### Order-flow imbalance (OFI)

$$\mathrm{OFI}_t = \Delta V^{\text{bid}}_t - \Delta V^{\text{ask}}_t,$$
with adjustments for price moves. Cont-Kukanov-Stoikov (2014) showed OFI strongly predicts short-term returns.

---

## 8.6.3 Queue Position Modeling

### The queue game

At the top of the book, your order at position $k$ in a queue of $n$ will be filled when $k$ orders ahead are either consumed by market orders or canceled. Queue position is a *state variable* worth tens of bps per trade for passive market makers.

### Arrival rate model

Assume market orders arrive at rate $\lambda_M$ consuming from the front; cancellations at rate $\lambda_C$ per order, distributed uniformly over queue positions (or modeled with a position-dependent intensity). Expected time to fill at position $k$:
$$\mathbb{E}[\tau_k] \approx \frac{k}{\lambda_M + \lambda_C/2}.$$
More realistically, cancellations are concentrated at the back (later orders cancel faster), making the "effective queue position" smaller than raw $k$.

### Queue value

The option to trade at the current price (queue value) decays as price drifts away:
$$V_{\text{queue}}(k, p) = \mathbb{P}(\text{fill before price moves}) \cdot (\text{spread capture}).$$
Cartea-Jaimungal-Penalva (2015) develop explicit queue models for optimal posting.

### Joining or not joining

If the queue is long and price tends to move before you reach the front, the option is worthless. Smart market makers post deep in the book only when expected price drift is low; when drift is high, they cross to capture the immediate move.

---

## 8.6.4 Avellaneda-Stoikov Optimal Market Making

### Setup

Reference price $S_t$ follows Brownian motion: $dS_t = \sigma dW_t$. Market maker posts bid $b_t = S_t - \delta_b$ and ask $a_t = S_t + \delta_a$. Orders fill at Poisson rate
$$\lambda_\pm(\delta) = A e^{-k \delta},$$
with $A, k$ calibrated to historical fill data. Market maker maximizes terminal expected exponential utility $-e^{-\gamma X_T}$ with inventory $q_T$ marked at $S_T$.

### Value function and HJB

$$u(x, q, s, t) = \sup_{\delta_a, \delta_b} \mathbb{E}\left[-\exp(-\gamma (X_T + q_T S_T))\right].$$
With the ansatz $u = -\exp(-\gamma x)\exp(-\gamma q s) \exp(\theta(q, t))$, the problem reduces to a deterministic ODE for $\theta$, yielding:

**Reservation price:**
$$r(s, q, t) = s - q \gamma \sigma^2 (T - t).$$

**Optimal spread:**
$$\delta_a + \delta_b = \gamma \sigma^2 (T - t) + \frac{2}{\gamma}\ln\left(1 + \frac{\gamma}{k}\right).$$

**Optimal quotes:**
$$\delta_b = (s - r) + \tfrac{1}{2}(\text{spread}), \qquad \delta_a = (r - s) + \tfrac{1}{2}(\text{spread}).$$

### Interpretation

The reservation price shifts linearly with inventory — negative inventory (short) → quote above the mid-price, positive inventory (long) → quote below. This "skews" quotes to bleed off inventory while capturing the spread. The spread has two components: a risk premium $\gamma\sigma^2(T-t)$ for holding inventory, and a fill-rate-based floor $(2/\gamma)\ln(1+\gamma/k)$.

### Enhancements

- **Drift** ($dS_t = \mu dt + \sigma dW$): the reservation price gains $\mu(T-t)$, tilting quotes in the direction of expected drift.
- **Asymmetric fill rates** (toxic flow differentiating bids vs. asks): skew based on expected toxicity.
- **Queue position**: extend the model to explicit queue dynamics (Cartea-Jaimungal).
- **Multiple assets**: Guéant (2016) generalizes to multi-asset inventories.

---

## 8.6.5 Adverse Selection and Inventory

### The toxicity problem

When informed traders lift your ask or hit your bid, you transact at a disadvantageous price. Quantify via:
- **Realized spread**: spread measured against a future midpoint (e.g., 1 minute later). Market makers earn the effective spread but lose the price impact.
- **Effective spread = realized spread + price impact** (see Module 5.7).

### Inventory cost

Holding inventory exposes the market maker to price risk. Risk-averse quoting (Avellaneda-Stoikov) mitigates this; additional tools:
- **Hard inventory caps** triggering automatic unwinding.
- **Dynamic hedging** with correlated instruments (e.g., ETF makers hedge with constituent basket).
- **Cross-venue inventory netting**.

### Adverse-selection detection

Real-time classifiers predict fill toxicity:
- **Features**: book imbalance, recent trade direction, volatility, momentum, opposite-side size, time of day.
- **Labels**: realized adverse P&L on each fill.
- **Models**: logistic regression, gradient boosting, or neural networks; updated intra-day.

### Quote widening

When toxicity score increases, widen quotes or stop posting on the aggressor side. Must be faster than predatory traders' reaction.

---

## 8.6.6 Hawkes Processes for Order Flow

### The Hawkes process

A self-exciting point process with intensity
$$\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)}.$$
Each event boosts future intensity by $\alpha$, decaying exponentially at rate $\beta$. Stable if $\alpha/\beta < 1$; "branching ratio" $\rho = \alpha/\beta$ measures self-excitation.

### Multivariate Hawkes

$$\lambda_i(t) = \mu_i + \sum_j \sum_{t^{(j)}_k < t} \alpha_{ij} e^{-\beta_{ij}(t - t^{(j)}_k)}.$$
Bacry-Delattre-Hoffmann-Muzy (2013) showed empirical Hawkes fits to LOB events: market orders excite market orders, cancellations excite market orders, etc. Typical branching ratios 0.6–0.8 — close to critical, explaining bursts of activity.

### Intraday patterns

Order arrivals exhibit strong diurnal patterns. Include a time-varying baseline $\mu(t)$ — sometimes a Fourier series at daily-frequency components — on top of the Hawkes kernel.

### Signature and realized impact

Jaisson-Rosenbaum (2015) show that in a near-critical Hawkes regime, the price increments follow a rough process — connecting to "rough volatility" models (Module 5.6).

---

## 8.6.7 Latency, Co-location, Architecture

### Latency budget

A round-trip quote-to-quote cycle on a modern exchange (Nasdaq, CME) can be ~10 μs with optimized hardware:
- Wire from co-lo rack to matching engine: ~1 μs.
- FPGA processing on inbound market data: ~1 μs.
- Strategy decision: ~1–5 μs (often on FPGA directly).
- Outbound order submission: ~1 μs.

### Hardware stack

- **FPGAs** for deterministic, hardware-level packet processing — widely used for market data decoding and simple quoting.
- **Custom ASICs** for critical inner loops.
- **GPUs** for batch analytics but rarely in the hot path.
- **Network stacks** bypassing kernel: Solarflare OpenOnload, Exablaze, custom drivers.

### Risk of speed

Flash crashes, self-trade incidents, runaway algorithms. Engineering for fail-safes:
- **Hard kill switches** wired directly to NIC flush.
- **Position monitoring** below strategy level for independent verification.
- **Pre-trade risk gateways** on dedicated hardware.

### Economics

Co-location monthly fees: $1k–$10k per rack. Microwave/laser private networks (e.g., Chicago-NY loop under 4 ms round trip) rent at $100k+ per month per connection. Latency arb economics become tight as technology commodifies.

### Monitoring

Every HFT desk runs a comprehensive post-trade analytics stack: latency histograms, fill quality, inventory PnL, market-impact attribution. Issues are detected in seconds; corrections deployed minutes later in trading hours.

---

## 8.6.8 Regulation

### Reg NMS (US, 2005)

- **Order protection rule** (Rule 611): prevents "trading through" a better price on another venue.
- **Access rule** (Rule 610): fair and non-discriminatory access to quotes; caps access fees.
- **Sub-penny rule**: generally prevents sub-penny quoting for stocks above $1.

### MiFID II (EU, 2018)

- **Market-making obligations**: firms designated as MMs must meet quoting presence thresholds.
- **Algo trading controls**: extensive pre-trade controls, notification of algos to regulators.
- **Tick-size regime**: harmonized across EU.
- **Best execution** (RTS 27/28 reporting).
- **Pre- and post-trade transparency**.

### Compliance and surveillance

Desks need automated surveillance for:
- Layering / spoofing (large orders away from best to influence price, canceled before fill).
- Marking the close (last-minute orders to move closing price).
- Wash trading (self-trades to inflate volume).
- Quote stuffing (excess orders to degrade competitor processing).

Penalties are severe; regulators have demonstrated willingness to pursue both firms and individuals.

### Central clearing and DMA

Most HFT in equities goes through sponsored-access DMA arrangements with a broker-dealer; the broker-dealer bears responsibility for risk controls under SEC Rule 15c3-5.

---

## 8.6.9 Python: Simple Market-Making Simulator

```python
import numpy as np

def as_market_maker_sim(T=1.0, N=500, S0=100.0, sigma=0.5, gamma=0.1, k=1.5, A=140.0,
                        dt_mode='uniform', seed=0):
    """
    Simulate an Avellaneda-Stoikov market maker.
    T: time horizon (days or fraction)
    N: discrete time steps
    sigma: mid volatility
    gamma: risk aversion
    k, A: fill-rate params (lambda = A exp(-k delta))
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    S = np.zeros(N+1); S[0] = S0
    q = 0.0
    x = 0.0
    pnl = []
    inventory = []
    quotes = []
    for t in range(N):
        # Mid-price update
        S[t+1] = S[t] + sigma*np.sqrt(dt)*rng.standard_normal()
        # AS reservation and spread
        tau = T - t*dt
        r = S[t] - q*gamma*sigma**2*tau
        spread = gamma*sigma**2*tau + (2/gamma)*np.log(1 + gamma/k)
        bid = r - spread/2; ask = r + spread/2
        delta_b = S[t] - bid; delta_a = ask - S[t]
        lam_b = A*np.exp(-k*delta_b)
        lam_a = A*np.exp(-k*delta_a)
        # Poisson fills
        filled_b = rng.random() < lam_b*dt
        filled_a = rng.random() < lam_a*dt
        if filled_b:
            q += 1; x -= bid
        if filled_a:
            q -= 1; x += ask
        pnl.append(x + q*S[t+1])
        inventory.append(q)
        quotes.append((bid, ask))
    return np.array(pnl), np.array(inventory), np.array(quotes), S

# Run and report
if __name__ == "__main__":
    pnl, inv, q_, S = as_market_maker_sim(seed=1)
    print(f"Final P&L: {pnl[-1]:.3f}")
    print(f"Max |inventory|: {np.max(np.abs(inv))}")
    print(f"Mean absolute inventory: {np.mean(np.abs(inv)):.2f}")
```

### Calibration checklist

- **$\sigma$**: realized vol at the target horizon (seconds to minutes for HFT).
- **$A, k$**: fit $\log \lambda = \log A - k \delta$ from historical fill-vs-depth data.
- **$\gamma$**: risk aversion; set by desired inventory dispersion (lower $\gamma$ = larger positions).

### Extensions in production

- Discrete tick sizes, not continuous deltas.
- Queue position as explicit state.
- Cancel-on-move logic: cancel when mid shifts > threshold.
- Cross-venue inventory netting.
- Toxicity gating: skip quotes when adverse-selection score is high.
- Dynamic recalibration of fill-rate parameters intra-day.

---

## 8.6.10 Applications and Case Studies

1. **Equity market making**: multi-venue quoting with cross-venue inventory netting.
2. **ETF arbitrage**: ETF vs. underlying basket, NAV vs. traded price.
3. **Futures–cash basis trading**: E-mini S&P vs. SPY or the basket of underlyings.
4. **Options market making**: continuous delta hedging; complex volatility inventory management.
5. **FX HFT**: ECN-fragmented venues, microwave links, ultra-fast quote responses.
6. **Treasury market making**: on/off-the-run spreads, maturity-ladder hedges.
7. **Crypto market making**: centralized exchanges (Binance, OKX, Kraken) plus DEX liquidity provision with AMM mathematics.
8. **Latency arb between exchanges**: CME → NYSE arb via microwave.
9. **Opening/closing auction strategies**: predict auction clear prices.
10. **IPO first-day dynamics**: specialized opening market-making protocols.

### Case study: 2010 Flash Crash

6 May 2010 — S&P fell ~9% in minutes, recovered partially within the hour. Triggered by a large sell algo in E-mini S&P from Waddell & Reed; propagated through cross-market arbitrage; liquidity providers withdrew as inventory limits breached. SEC/CFTC post-mortem: market structure vulnerability to liquidity evaporation in extreme periods. Led to circuit breakers, Limit-Up/Limit-Down (LULD) rules, exchange-wide kill switches.

### Case study: Knight Capital (2012)

A mis-deployed trading algorithm executed erroneous orders for 45 minutes, accumulating $460M in losses, driving Knight into a capital crisis and eventual acquisition. Lesson: deployment infrastructure and kill-switch reliability are as important as strategy research.

### Case study: Virtu's IPO prospectus

Virtu disclosed only one losing trading day over 1238 days — emblematic of modern HFT's diversified-edge, law-of-large-numbers economics. The prospectus remains a famous data point in HFT literature.

---

## 8.6.11 Exercises

### ★

1. For a limit order at queue position 10 in a queue of 50, with market-order arrival rate 1/s and cancellation rate 0.02/s per order, estimate expected time to fill.
2. Derive the micro-price formula and show it equals the mid-price when bid and ask depths are equal.
3. Compute the effective spread and realized spread for a trade at the ask price, with midpoint moving up by half the spread over 1 minute.
4. For $A=140$, $k=1.5$, $\delta = 0.02$, compute the expected fill rate.
5. Show that the Avellaneda-Stoikov reservation price reduces to the mid when inventory is zero.
6. A Hawkes process has $\mu=1, \alpha=0.5, \beta=1$. Compute the branching ratio and stationary intensity.

### ★★

7. Implement the AS market maker simulator and plot the distribution of terminal P&L over 1000 seeds; report Sharpe and inventory dispersion.
8. Derive the HJB equation for the AS market maker and verify the ansatz $u = -\exp(-\gamma x)\exp(-\gamma q s)\exp(\theta(q,t))$ reduces it to an ODE for $\theta$.
9. Simulate a multivariate Hawkes process for market orders, cancellations, and limit orders; estimate cross-excitation parameters.
10. Build a queue-position-aware extension of AS: when deep in the queue, quote further back; when front, quote at spread minimum.
11. Implement a latency-arbitrage simulator: two venues, noisy price feeds with lag $\ell$, execute when price gap > threshold.
12. Build a toxicity classifier (logistic regression) predicting adverse fill probability given book imbalance and recent trade flow; backtest on simulated data.

### ★★★

13. Prove the existence and uniqueness of solutions to the AS ODE for $\theta(q, t)$ under standard regularity conditions.
14. Extend the AS model to include a drift term $dS = \mu dt + \sigma dW$ and derive the modified reservation price.
15. Prove that a Hawkes process with branching ratio $\rho < 1$ is ergodic and derive its stationary intensity.
16. Derive optimal quoting in a pro-rata matching system (vs. price-time priority) and explain why quoted depths become much larger.
17. Formulate the multi-asset market-making problem under correlated inventories and derive the HJB equation; discuss the structure of the resulting value function.
18. Prove that under optimal market making with signal-based mid, the expected PnL per fill equals the spread minus the expected adverse selection, exactly.

---

*— End of Module 8.6. Next: Module 8.7, Systems and MLOps for Quant Research — Subject 8 capstone.*

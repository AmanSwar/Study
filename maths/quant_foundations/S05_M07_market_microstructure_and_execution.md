# Module 5.7 — Market Microstructure and Optimal Execution

*Subject 5 capstone. Where continuous-time asset pricing meets the discrete reality of the limit order book.*

---

## Prerequisites

- **Modules 5.1–5.6**: Arbitrage theory, Black-Scholes, exotics, interest rates, credit, stochastic volatility. We now pivot from the question "what should a claim be worth under a given model?" to "what happens when I actually try to trade it?"
- **Module 3.1 (Brownian motion)** and **Module 3.5 (Lévy and jump processes)**: The mid-price is diffusion-like on long horizons, but tick-by-tick it is a jump process driven by order flow.
- **Module 4.6 (LQG control)**: Almgren-Chriss is the LQ archetype of trading. We reprise and extend it.
- **Module 2.6 (Martingales)**: Fair-game property of the mid under informed-trader equilibrium.
- **Module 0.3.6 (Calculus of variations)**: Euler-Lagrange for execution schedules.
- **Module 3.5 revisited**: Hawkes processes as self-exciting point processes.

---

## 5.7.1 What market microstructure is about

Classical asset pricing assumes a frictionless, infinitely-deep market: you can trade arbitrary size at a single price $S_t$, and every trader sees the same price. Reality is different:

1. **The price is not a number — it's a pair.** The best bid $b_t$ and best ask $a_t$ bracket the "mid" $m_t = (a_t+b_t)/2$. Trading costs at least half the bid-ask spread $s_t = a_t - b_t$.
2. **Depth is finite.** Only $q_t^b$ shares are offered at $b_t$ and $q_t^a$ at $a_t$. Trading more sweeps through deeper levels, paying higher prices for sells-into-bid, lower prices for buys-from-ask. This is **walking the book**.
3. **Participation is strategic.** Some traders know things others don't (informed). Some add liquidity passively (market makers). Some demand liquidity impatiently (aggressive takers). Each is reacting to the others.
4. **Impact.** Your own trades move prices. A buy order today raises the market price, so the *marginal* cost of size increases superlinearly. Impact has a **permanent** component (information revelation) and a **temporary** component (liquidity consumption that relaxes after you stop).
5. **Adverse selection.** When someone lifts your offer, there's a real probability it's because they know something — you just sold cheap. Market makers quote wider spreads to compensate.

Microstructure studies the interplay of these forces. Execution studies how a large agent should split trades in time to minimize cost subject to risk.

**Two distinct horizons.** On the timescale of days, volatility and drift swamp everything and classical pricing works. On the timescale of milliseconds-to-hours — where block orders, ETF rebalances, and market-making live — spreads, impact, and adverse selection dominate expected P\&L.

---

## 5.7.2 The limit order book (LOB)

A **limit order book** is a double-sided queue. Traders post:

- **Limit buy orders** at price $p \le m_t$: "buy up to $x$ shares at $p$ or lower." These stack into the bid side.
- **Limit sell orders** at price $p \ge m_t$: symmetric.
- **Market orders**: "buy $x$ shares immediately at best available." These consume the top of the opposite book.
- **Cancellations**: remove a previously posted limit order.

Each exchange uses **price-time priority**: orders at a better price execute first; at the same price, orders posted earlier execute first.

**State space.** The LOB at time $t$ is a pair of functions $q^b(p,t)$, $q^a(p,t)$ giving the shares at each price. In practice we track the top $K$ levels (often $K=10$). Key summaries:

- **Spread**: $s_t = a_t - b_t$.
- **Mid**: $m_t = (a_t+b_t)/2$.
- **Microprice**: size-weighted mid $m_t^{micro} = \tfrac{a_t q^b_t + b_t q^a_t}{q^b_t + q^a_t}$. When the bid queue is deeper, the microprice tilts toward the ask because sellers have to "cross more road." Stoikov (2018) showed the microprice is a better fair-value estimator than the plain mid.
- **Depth imbalance**: $I_t = (q^b_t - q^a_t)/(q^b_t + q^a_t) \in [-1,1]$. Strong predictor of short-term price direction (Lipton-López de Prado-Pesavento).

**Tick size and queue dynamics.** Exchanges impose a minimum price increment, the **tick**. Stocks with tight effective spreads (spread $\approx$ tick) are **tick-constrained**: most action is queue dynamics at the top of book, not price evolution. Large-tick stocks have high queue priority value; small-tick stocks trade "through the mid" via hidden orders.

**Typical rates (equities, large-cap).** Mid changes ~once per second, trade arrivals ~10/sec, quote updates ~1000/sec. In futures and FX ECNs, rates are an order of magnitude higher.

---

## 5.7.3 Kyle (1985): strategic informed trading

Albert Kyle's 1985 "Continuous Auctions and Insider Trading" is the canonical model of how private information enters price.

**Setup.** One period, single risky asset.
- **Insider** knows the liquidation value $v \sim \mathcal{N}(\mu_0, \Sigma_0)$. Submits order size $x$ to maximize expected profit.
- **Noise trader** submits $u \sim \mathcal{N}(0, \sigma_u^2)$, independent of $v$. Represents liquidity-motivated trades.
- **Market maker** sees only aggregate order flow $y = x + u$ (cannot distinguish) and sets price $p(y) = \mathbb{E}[v \mid y]$ — competitive, zero-profit pricing.

**Linear equilibrium ansatz.**
$$
x = \beta(v - \mu_0), \qquad p = \mu_0 + \lambda y.
$$

The insider's profit: $\pi = (v - p) \cdot x = (v - \mu_0 - \lambda(\beta(v-\mu_0)+u)) \cdot \beta(v-\mu_0)$. Taking conditional expectation given $v$ and maximizing in $\beta$:
$$
\frac{\partial}{\partial x} \mathbb{E}[(v - \mu_0 - \lambda x - \lambda u) x \mid v] = v - \mu_0 - 2\lambda x = 0
\;\;\Rightarrow\;\; x^* = \frac{v - \mu_0}{2\lambda}.
$$
So $\beta = 1/(2\lambda)$.

**Market maker's filter.** With $y = x + u$, $x = \beta(v-\mu_0)$,
$$
p(y) = \mathbb{E}[v \mid y] = \mu_0 + \frac{\text{Cov}(v,y)}{\text{Var}(y)} y = \mu_0 + \frac{\beta \Sigma_0}{\beta^2 \Sigma_0 + \sigma_u^2} y.
$$
So $\lambda = \beta \Sigma_0/(\beta^2 \Sigma_0 + \sigma_u^2)$.

**Solving the fixed point.** Substituting $\beta = 1/(2\lambda)$:
$$
\lambda = \frac{\Sigma_0/(2\lambda)}{\Sigma_0/(4\lambda^2) + \sigma_u^2}
= \frac{2\lambda \Sigma_0}{\Sigma_0 + 4\lambda^2 \sigma_u^2}.
$$
Cancel a $\lambda$ on each side and solve:
$$
\boxed{\lambda = \frac{1}{2}\sqrt{\frac{\Sigma_0}{\sigma_u^2}}, \qquad \beta = \sqrt{\frac{\sigma_u^2}{\Sigma_0}}.}
$$

**Interpretation.**
- **Price impact $\lambda$ is half Kyle's "lambda."** It is the slope of price in order flow. More noise trading $\sigma_u$ → smaller $\lambda$: the insider can hide in more camouflage, so the market maker extracts less per unit of order flow.
- **Insider's trade is proportional to his informational edge.** $\beta$ scales with $\sigma_u/\sqrt{\Sigma_0}$: more camouflage or less information encourages more aggressive trading.
- **Insider's expected profit**: $\mathbb{E}[\pi] = \beta \cdot \Sigma_0 / 2 = \tfrac{1}{2}\sqrt{\Sigma_0 \sigma_u^2}$. Half the information is revealed; the other half is the insider's rent.
- **Price revelation**: $\text{Var}(p) = \lambda^2(\beta^2 \Sigma_0 + \sigma_u^2) = \Sigma_0/2$. Exactly half the insider's information is incorporated.

**Continuous-time Kyle.** In the continuous version (also Kyle 1985), the insider trades continuously over $[0,1]$ until the terminal value is revealed. The equilibrium has:
$$
dx_t = \beta_t(v - p_t) dt, \qquad dp_t = \lambda_t(dx_t + du_t).
$$
Remarkably, $\lambda_t$ is **constant in $t$**: the market maker's price impact doesn't decay, and information is revealed at a constant rate. The insider smooths his trades perfectly.

**Why Kyle matters.** Kyle's $\lambda$ is the microstructure primitive translating information into price. It gives the first-principles derivation of *permanent* impact: when my trade is correlated with the asset's true value, the market rationally updates and the price stays moved.

---

## 5.7.4 Glosten-Milgrom (1985): sequential trade and the spread

Glosten-Milgrom replace Kyle's batch auction with **sequential trade** (one order at a time) and explain the bid-ask spread as pure **adverse selection** cost.

**Setup.** Asset value $V \in \{V_L, V_H\}$ with $V_H > V_L$. Prior $\mathbb{P}(V = V_H) = \theta_0$. On each period a trader arrives; with probability $\alpha$ they are **informed** (know $V$), with probability $1-\alpha$ **uninformed** (buy/sell 50/50).

The market maker posts bid $b$ and ask $a$ before seeing the trade, commits to them for one unit, and must break even in expectation on each side.

**Bayesian updating.** Let $B$ = "next trader buys," $S$ = "next trader sells."
$$
\mathbb{P}(B \mid V_H) = \alpha + (1-\alpha)/2, \qquad \mathbb{P}(B \mid V_L) = (1-\alpha)/2.
$$
By Bayes,
$$
\theta_1^B = \mathbb{P}(V_H \mid B) = \frac{[\alpha + (1-\alpha)/2] \theta_0}{[\alpha + (1-\alpha)/2]\theta_0 + [(1-\alpha)/2](1-\theta_0)}.
$$
The ask equals the conditional mean given a buy:
$$
\boxed{a = \mathbb{E}[V \mid B] = \theta_1^B V_H + (1-\theta_1^B) V_L.}
$$
Symmetrically the bid is $b = \mathbb{E}[V \mid S]$, and the spread
$$
s = a - b > 0
$$
whenever $\alpha > 0$. The market maker loses on informed trades and recovers on uninformed; the spread is exactly the expected loss to adverse selection per trade.

**Asymptotic learning.** Repeated rounds with $\theta_t \to \mathbb{1}_{V = V_H}$ P-a.s. (a martingale converges), and the spread $s_t \to 0$. The informed trader's advantage decays as the market learns.

**Why Glosten-Milgrom matters.**
1. Establishes the **spread as the price of information asymmetry**, not operational cost.
2. Grounds the **tick-by-tick random walk theorem**: under this model, $p_t = \mathbb{E}[V \mid \mathcal{F}_t]$ is a $\mathbb{P}$-martingale. No trading system can profit from the price series alone.
3. Yields the **trade direction signature** — buys push the price up, sells down — with magnitude controlled by $\alpha$.

---

## 5.7.5 Price impact: permanent, temporary, and the square-root law

Empirically, execution a trade of size $Q$ over time $T$ moves the price by $\Delta p$. Practitioners fit:
$$
\Delta p \approx \eta \sigma \sqrt{Q/V_D},
$$
where $\sigma$ is the daily vol and $V_D$ is daily volume. This is the famous **square-root law** (BARRA, Almgren, BNP, Gatheral).

**Components of impact.**
- **Permanent impact** $g(v)$: the mid-price moves because your trade reveals information or commits liquidity; it does not revert. Theory (Huberman-Stanzl 2004) shows that under no-arbitrage, permanent impact must be *linear* in the rate $v = dQ/dt$.
- **Temporary impact** $h(v)$: the price you pay is worse than the mid because you are walking the book. It reverts after you stop (liquidity replenishes).
- **Transient impact** (Gatheral 2010, Obizhaeva-Wang 2013): impact decays with a kernel $G(t)$, so earlier trades still affect current price but less so.

**The Gatheral price-impact model.**
$$
S_t = S_0 + \int_0^t f(v_s) G(t-s) ds + \sigma B_t.
$$
If $G(t) = 1$ (no decay) we get permanent linear impact (Bertsimas-Lo, Almgren-Chriss). If $G$ is exponential we get the Obizhaeva-Wang resilient LOB. If $G$ is power-law $G(t) \sim t^{-\gamma}$ we get empirical transient impact.

**No-dynamic-arbitrage constraint.** Gatheral (2010) proved: for the impact functional $\int f(v) G(t-s) ds$ to be dynamic-arbitrage-free, the product $f$ and $G$ must satisfy $f(v) G$ convex in the sense that round-trip trades cannot make money. This rules out e.g. sublinear $f$ with slow decay.

**Why square-root?** Several theories converge:
1. **LLOB (Latent Limit Order Book, Toth-Lemperiere-Deremble-de Lataillade-Kockelkoren-Bouchaud 2011)**: liquidity resupply is a diffusion equation, and the impact function of a meta-order is a boundary-layer solution scaling as $\sqrt{Q}$.
2. **Adverse-selection bound**: informed trading fraction is bounded; square-root is the Pareto-efficient trade-off.
3. **Empirical fit**: spans 6+ orders of magnitude in trade size across markets, making $\sqrt{Q}$ one of the most robust empirical laws in finance.

---

## 5.7.6 Optimal execution — Almgren-Chriss revisited

Reprise of Module 4.6 with microstructure-aware detail.

**Problem.** Liquidate $X_0$ shares over $[0, T]$. State $X_t$ = shares remaining; trading rate $v_t = -\dot X_t \ge 0$. Execution price:
$$
S_t^{exec} = S_t - h(v_t), \qquad dS_t = -g(v_t) dt + \sigma dB_t,
$$
with $h(v) = \eta v$ (linear temporary impact) and $g(v) = \gamma v$ (linear permanent impact).

Cash received up to time $T$: $\int_0^T S_s^{exec} v_s ds$. **Implementation shortfall** (Perold): expected cost relative to decision price $S_0 X_0$,
$$
\text{IS} = S_0 X_0 - \mathbb{E}\left[\int_0^T S_s^{exec} v_s ds\right] = \int_0^T[g(v_s) X_s + h(v_s) v_s] ds.
$$
**Mean-variance criterion.** With a risk-aversion $\lambda$ penalizing variance from remaining exposure:
$$
J = \mathbb{E}[\text{IS}] + \lambda \text{Var}(\text{IS}) \approx \int_0^T [\eta v_t^2 + \lambda \sigma^2 X_t^2] dt + \frac{\gamma}{2} X_0^2.
$$
(The $\gamma$ term is a constant given total quantity and drops out of the optimization.)

**Euler-Lagrange.** Minimize $\int [\eta \dot X^2 + \lambda \sigma^2 X^2] dt$ with $X(0) = X_0$, $X(T) = 0$:
$$
2\eta \ddot X - 2\lambda\sigma^2 X = 0 \;\;\Rightarrow\;\; \ddot X = \kappa^2 X, \quad \kappa = \sqrt{\lambda \sigma^2/\eta}.
$$
Solution:
$$
\boxed{X_t^* = X_0 \cdot \frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}, \qquad v_t^* = X_0 \kappa \cdot \frac{\cosh(\kappa(T-t))}{\sinh(\kappa T)}.}
$$

**Limits.**
- $\lambda \to 0$: $\kappa \to 0$, $X_t \to X_0(1-t/T)$ — **TWAP** (time-weighted average price), uniform trading.
- $\lambda \to \infty$: $\kappa \to \infty$, front-loaded exponential — trade ASAP to eliminate market risk.
- $\eta \to 0$: instantaneous liquidation at $t=0$. Infinite impact makes it impossible; $\eta$ controls how quickly you dare to trade.

**VWAP vs TWAP.** TWAP trades uniformly in time. VWAP (volume-weighted average price) trades proportionally to market volume $V(t)$ curve. VWAP schedule matches the U-shaped intraday volume profile (deep at open and close), minimizing deviation-from-benchmark risk.

**With signals (Cartea-Jaimungal-Penalva 2015).** Let $dS_t = \mu_t dt + \sigma dB_t$ with $\mu_t$ a tradable alpha signal. The HJB for optimal liquidation adds a drift term; the optimal schedule is Almgren-Chriss + a signal-dependent correction.

**Obizhaeva-Wang with resilient LOB.** When the LOB has finite depth $q$ that replenishes at rate $\rho$, optimal strategies are discrete "chunks" with waiting times — no longer a smooth curve.

---

## 5.7.7 Market making — Avellaneda-Stoikov

Avellaneda-Stoikov (2008) model the optimal quotes of a **market maker** who faces inventory risk.

**Setup.** MM posts bid price $b_t = s_t - \delta^b_t$ and ask $a_t = s_t + \delta^a_t$ around the mid $s_t$, with $ds = \sigma dB_t$. Fill intensities are
$$
\lambda^b(\delta^b) = A e^{-k \delta^b}, \qquad \lambda^a(\delta^a) = A e^{-k \delta^a},
$$
i.e., tighter quotes fill faster; $A$ and $k$ are LOB parameters.

**State.** Inventory $q_t$ (signed shares held by MM) and cash $X_t$. Dynamics:
$$
dq_t = dN^b_t - dN^a_t, \qquad dX_t = (s_t + \delta^a) dN^a_t - (s_t - \delta^b) dN^b_t,
$$
where $N^b, N^a$ are Poisson processes with intensities $\lambda^b, \lambda^a$.

**Objective.** Maximize CARA utility of terminal wealth marked at mid:
$$
V(t, s, q, X) = \sup_{\delta^b, \delta^a} \mathbb{E}[-\exp(-\gamma(X_T + q_T s_T))].
$$

**HJB.**
$$
0 = \partial_t V + \tfrac{1}{2}\sigma^2 \partial_{ss} V + \max_{\delta^a}\lambda^a(\delta^a)[V(s, q-1, X+s+\delta^a) - V] + \max_{\delta^b} \text{(symmetric)}.
$$

**Ansatz**: $V = -\exp(-\gamma X - \gamma q s) \exp(-\gamma\theta(t, q))$. Under the ansatz the optimization reduces to a system of ODEs for $\theta(t,q)$.

**Asymptotic solution (for large $k$ and short horizons).** The **reservation price** (mid at which MM is indifferent to holding $q$ shares):
$$
r(s, q, t) = s - q\gamma\sigma^2(T-t).
$$
MM's **optimal quotes**:
$$
\boxed{\delta^{a,*} + \delta^{b,*} = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln(1 + \gamma/k),}
$$
split around the reservation price:
$$
\delta^{a,*} = \tfrac{1}{2}[\delta^{a,*}+\delta^{b,*}] + \tfrac{1}{2}\gamma\sigma^2 q (T-t), \qquad \delta^{b,*} = \tfrac{1}{2}[\delta^{a,*}+\delta^{b,*}] - \tfrac{1}{2}\gamma\sigma^2 q (T-t).
$$

**Economic content.**
- **Spread widens with inventory risk**: longer $T-t$ or larger $\sigma$ or $\gamma$ means quoting further from mid.
- **Skew with inventory**: long (positive $q$) → bid tight, ask wide, encouraging sell fills to flatten.
- **Symmetry at $q = 0$**: pure spread capture.
- **Baseline spread** $(2/\gamma)\ln(1+\gamma/k)$: reflects trade-off between probability of fill ($k$) and risk aversion ($\gamma$).

**Extensions.** Guéant-Lehalle-Fernandez-Tapia (2013) extended to finite inventory bounds; Cartea-Jaimungal added toxic flow / adverse selection. Avellaneda-Reed used this framework for ETFs.

---

## 5.7.8 High-frequency data, Hawkes processes, trade signs

**Tick data** (trades + quotes) arrive at irregular times and have distinctive statistical properties.

**Trade signs are autocorrelated.** A buy is likelier to be followed by a buy, not because prices are mean-reverting but because **large meta-orders are split into small pieces**. Lillo-Farmer (2004) documented trade-sign autocorrelation decaying as $t^{-\gamma}$ with $\gamma \approx 0.5$, extending over hours.

**Hawkes processes.** A mutually-exciting point process where intensity jumps on each event:
$$
\lambda_t = \lambda_\infty + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)}.
$$
Each arrival increments $\lambda$ by $\alpha$, which decays at rate $\beta$. Stationarity requires the **branching ratio** $n = \alpha/\beta < 1$; the expected total number of offspring per event is $n/(1-n)$.

**Bivariate Hawkes for trade signs.** Let $N^+$, $N^-$ count buys and sells:
$$
\lambda^+_t = \mu^+ + \int \alpha_{++}(t-s) dN^+_s + \int \alpha_{+-}(t-s) dN^-_s.
$$
Jaisson-Rosenbaum (2015) showed that rescaling a near-critical ($n \to 1$) Hawkes process gives rough volatility (Hurst $\approx 0.1$) — a stunning connection between microstructure and rough path theory.

**Trade-sign inference.** Lee-Ready algorithm: classify a trade as buy if $p > m$, sell if $p < m$, tick-test otherwise. Modern approach: EMM (estimation using mid-microprice dynamics).

**Bouchaud-Gefen-Potters-Wyart (2004) "response function"**: average price move conditional on trade sign,
$$
R(\ell) = \mathbb{E}[\epsilon_0 (p_{\ell} - p_0)],
$$
grows slowly with $\ell$ (lags). The autocorrelated sign and transient impact combine so that $\sum \epsilon$ doesn't random-walk linearly.

**Realized volatility and the signature plot.** At tick scale, realized variance depends on sampling frequency — microstructure noise contaminates the signal. Zhang-Mykland-Aït-Sahalia "two-scale realized variance" estimator is consistent.

---

## 5.7.9 Python: a microstructure toolkit

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# 1. Kyle (1985) equilibrium and profit simulation
# ============================================================
Sigma0 = 1.0        # variance of asset value
sigma_u = 1.0       # variance of noise-trader order
lam = 0.5 * np.sqrt(Sigma0 / sigma_u**2)  # Kyle's lambda
beta = np.sqrt(sigma_u**2 / Sigma0)

n = 200_000
v = np.random.normal(0, np.sqrt(Sigma0), n)
u = np.random.normal(0, sigma_u, n)
x = beta * v
y = x + u
p = lam * y
profit = (v - p) * x
print(f"Kyle λ = {lam:.4f}, β = {beta:.4f}")
print(f"Theoretical insider profit = {0.5*np.sqrt(Sigma0*sigma_u**2):.4f}")
print(f"Simulated average profit = {profit.mean():.4f}")
print(f"Price variance {np.var(p):.4f} vs theory Σ₀/2 = {Sigma0/2:.4f}")

# ============================================================
# 2. Glosten-Milgrom sequential Bayesian updating
# ============================================================
V_H, V_L = 101.0, 99.0
alpha = 0.3
theta = 0.5
thetas, mids, asks, bids = [theta], [], [], []
# simulate 100 trades; informed traders reveal V
np.random.seed(42)
V_true = V_H
for _ in range(100):
    informed = np.random.rand() < alpha
    if informed:
        buy = (V_true == V_H)
    else:
        buy = np.random.rand() < 0.5
    # quotes BEFORE trade:
    pB = (alpha + (1-alpha)/2)*theta + ((1-alpha)/2)*(1-theta)
    pS = ((1-alpha)/2)*theta + (alpha + (1-alpha)/2)*(1-theta)
    thetaB = (alpha + (1-alpha)/2)*theta / pB
    thetaS = ((1-alpha)/2)*theta / pS
    a = thetaB*V_H + (1-thetaB)*V_L
    b = thetaS*V_H + (1-thetaS)*V_L
    asks.append(a); bids.append(b); mids.append(0.5*(a+b))
    # update posterior based on observed trade
    theta = thetaB if buy else thetaS
    thetas.append(theta)
print(f"After 100 trades, posterior P(V_H) = {theta:.4f}")
print(f"Final spread = {asks[-1]-bids[-1]:.4f}")

# ============================================================
# 3. Almgren-Chriss optimal liquidation schedule
# ============================================================
X0, T = 1_000_000, 1.0     # 1M shares, 1 trading day
sigma = 0.02               # daily vol
eta = 1e-6                 # temporary impact coefficient
lam_risk = 1e-6            # risk aversion

def almgren_chriss(X0, T, sigma, eta, lam, n_steps=100):
    kappa = np.sqrt(lam*sigma**2/eta)
    t = np.linspace(0, T, n_steps+1)
    X = X0 * np.sinh(kappa*(T-t)) / np.sinh(kappa*T)
    v = X0 * kappa * np.cosh(kappa*(T-t)) / np.sinh(kappa*T)
    return t, X, v, kappa

t, Xs, vs, kappa = almgren_chriss(X0, T, sigma, eta, lam_risk)
t0, Xs0, _, _ = almgren_chriss(X0, T, sigma, eta, 1e-12)  # near-TWAP
print(f"Kappa = {kappa:.2f}")
print(f"First-second trade ratio (risk-averse): {vs[0]/vs[-1]:.2f}")

# ============================================================
# 4. Avellaneda-Stoikov market-making simulation
# ============================================================
def avellaneda_stoikov(T=1.0, dt=0.005, s0=100, sigma=2.0,
                       gamma=0.1, k=1.5, A=140, q0=0):
    N = int(T/dt)
    s = np.zeros(N+1); s[0] = s0
    q = np.zeros(N+1); q[0] = q0
    cash = np.zeros(N+1)
    for i in range(N):
        t = i*dt
        s[i+1] = s[i] + sigma*np.sqrt(dt)*np.random.randn()
        # reservation price and quotes
        r = s[i] - q[i]*gamma*sigma**2*(T-t)
        spread = gamma*sigma**2*(T-t) + (2/gamma)*np.log(1+gamma/k)
        bid = r - spread/2
        ask = r + spread/2
        # fill probabilities
        delta_b = s[i] - bid
        delta_a = ask - s[i]
        p_buy = A*np.exp(-k*delta_b)*dt
        p_sell = A*np.exp(-k*delta_a)*dt
        if np.random.rand() < min(p_buy, 1.0):
            q[i+1] = q[i] + 1
            cash[i+1] = cash[i] - bid
        elif np.random.rand() < min(p_sell, 1.0):
            q[i+1] = q[i] - 1
            cash[i+1] = cash[i] + ask
        else:
            q[i+1] = q[i]; cash[i+1] = cash[i]
    wealth = cash + q*s
    return s, q, wealth

s, q, w = avellaneda_stoikov()
print(f"Terminal P&L = {w[-1]:.2f}, final inventory = {q[-1]}")

# ============================================================
# 5. Hawkes process simulation (Ogata's thinning)
# ============================================================
def simulate_hawkes(T, mu, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    events = []
    t = 0
    lam = mu
    while t < T:
        lam_max = mu + sum(alpha*np.exp(-beta*(t-s)) for s in events)
        if lam_max <= 0: lam_max = mu
        u = rng.exponential(1/max(lam_max, 1e-9))
        t += u
        if t >= T: break
        lam_t = mu + sum(alpha*np.exp(-beta*(t-s)) for s in events)
        if rng.random() < lam_t / lam_max:
            events.append(t)
    return np.array(events)

events = simulate_hawkes(T=100, mu=0.3, alpha=0.8, beta=1.0)
branching_ratio = 0.8/1.0
expected_rate = 0.3/(1 - branching_ratio)
print(f"Hawkes: n_events = {len(events)}, avg rate = {len(events)/100:.3f}, "
      f"theoretical = {expected_rate:.3f}")
```

**What to verify in each demo.**
- **Kyle**: simulated profit $\approx 0.5$ matching $\tfrac{1}{2}\sqrt{\Sigma_0 \sigma_u^2}$; price variance $\approx 0.5$ (half the information revealed).
- **Glosten-Milgrom**: posterior $\theta$ converges to 1 (informed case reveals $V_H$); spread contracts.
- **Almgren-Chriss**: risk-averse schedule front-loads heavily; first trade size $\gg$ last.
- **Avellaneda-Stoikov**: terminal P&L positive, inventory bounded — MM captures spread while managing risk.
- **Hawkes**: empirical rate matches $\mu/(1-n)$; clustering visible in event timestamps.

---

## 5.7.10 [QUANT APPLICATIONS]

1. **Execution desks.** Sell-side algos (VWAP, IS, POV, LIQUID) implement Almgren-Chriss variants. TCA (transaction cost analysis) decomposes realized costs into spread, impact, alpha, and timing, guiding parameter tuning.
2. **HFT market making on equities / futures / crypto.** Avellaneda-Stoikov is the canonical framework; real systems layer flow-toxicity signals (VPIN, Kyle's $\lambda$ estimator) on top to pull quotes when flow is informed.
3. **DEX market making (Uniswap v3, on-chain LPs).** Concentrated liquidity positions are LOB-like; optimal range-setting is an Avellaneda-Stoikov-style problem with impermanent-loss risk replacing inventory cost.
4. **Dark pools and smart order routers.** Optimal split across venues = multi-venue Almgren-Chriss. Information leakage penalty modeled as adverse-selection cost on lit venues.
5. **Block trading / facilitation.** Broker gives client guaranteed VWAP; broker then runs its own AC schedule, pricing the risk using Kyle-style impact models.
6. **ETF arbitrage.** Create-redeem arbitrage between underlying basket and ETF involves executing a basket under liquidity constraints — microstructure-aware sizing matters.
7. **Credit markets.** Dealer-intermediated; Glosten-Milgrom-style adverse selection dominates. RFQ spreads reflect $\alpha$ of informed flow.
8. **FX spot market making.** Large-tick, low-volatility regime with heavy autocorrelated order flow. ECN-style MM uses Hawkes-based flow prediction.
9. **Signal-driven execution.** Cartea-Jaimungal framework: short-term alpha signals (depth imbalance, trade-sign autocorrelation, order-book momentum) modify AC schedule.
10. **Stress testing and XVA.** Liquidation cost becomes crucial in wind-down scenarios; impact-adjusted expected shortfall (Jarrow-Protter 2007, Çetin-Jarrow-Protter 2010) accounts for feedback of forced selling.

---

## 5.7.11 Exercises

**★ (concept drills).**
1. Verify algebraically that $\beta = 1/(2\lambda)$ combined with $\lambda = \beta\Sigma_0/(\beta^2\Sigma_0 + \sigma_u^2)$ yields $\lambda = \tfrac{1}{2}\sqrt{\Sigma_0/\sigma_u^2}$.
2. In Glosten-Milgrom, show that $\mathbb{E}[p_{t+1} \mid \mathcal{F}_t] = p_t$, i.e., the price process is a $\mathbb{P}$-martingale.
3. Derive the limits of Almgren-Chriss as $\lambda \to 0$ (TWAP) and as $\eta \to 0$ (block liquidation at $t=0$).
4. Compute the reservation price in Avellaneda-Stoikov for $q = 10$ long, $T-t = 1$ day, $\sigma = 2\%$, $\gamma = 0.5$. What's the bid-ask skew implied?
5. Show that a Hawkes process with $n = \alpha/\beta < 1$ has stationary rate $\mu/(1-n)$. *Hint:* $\mathbb{E}[\lambda_t]$ satisfies a linear integral equation.
6. Why does a tick-constrained stock have a smaller effective $\lambda$ than a tick-unconstrained one?

**★★ (calculation).**
7. Continuous-time Kyle. Set up the HJB for the insider who maximizes $\mathbb{E}[\int_0^1 (v - p_t) \dot x_t dt]$ subject to $dp_t = \lambda_t(dx_t + du_t)$ with $du = \sigma_u dB_t$. Show that in equilibrium $\lambda_t = \sigma_u/\sqrt{\Sigma_0}$ is constant.
8. Almgren-Chriss with linear permanent impact. Include $g(v) = \gamma v$ and show that the *optimal schedule is unchanged* (only the expected cost level shifts by $\gamma X_0^2/2$). Interpret.
9. Signal-driven AC. Suppose $dS_t = \mu dt + \sigma dB_t$ with constant $\mu > 0$ (alpha signal). Derive the optimal liquidation schedule. How does positive alpha (expecting price to rise) change your selling pace?
10. Avellaneda-Stoikov with inventory penalty. Add $-\phi q^2$ to the running HJB and show that the effective risk-aversion becomes time-dependent, widening quotes even at $t$ far from $T$.
11. **Square-root fit.** Using the daily-vol-and-volume rule $\Delta p = \eta \sigma \sqrt{Q/V_D}$, estimate the price impact of liquidating 5% of daily volume in a stock with $\sigma = 2\%$. What $\eta$ would you use? Where does $\eta$ come from?
12. **Optimal pair execution.** Liquidate $X^A, X^B$ with correlated risk and a spread term. Show that optimal schedules couple: even if you have zero exposure to asset $B$, if $X^A X^B \ne 0$, you trade $B$ to hedge.

**★★★ (open / research).**
13. **Dark pool routing.** Formulate optimal routing across one lit and one dark venue with probability $\pi$ of a fill in the dark. Derive the Hamilton-Jacobi equation and characterize the routing policy.
14. **Obizhaeva-Wang.** Derive the closed-form solution for linear resilient LOB: $\partial_t q(p,t) = -\rho(q(p,t) - q_\infty(p))$. Show the optimal schedule is a discrete chunk at $t=0$, continuous "block rate" during $(0,T)$, and a final chunk at $T$.
15. **Deep hedging of execution.** Parametrize $v(t, X, \text{LOB state})$ with a neural net; train to minimize expected shortfall in simulated LOB. Compare to Almgren-Chriss on the same simulator.
16. **Rough-vol from Hawkes.** Simulate a near-critical bivariate Hawkes for trade signs with $n \to 1$ and rescale time. Verify numerically that realized volatility of the integrated signs has Hurst $\approx 0.1$.
17. **Order-flow toxicity (VPIN).** Implement the Easley-López de Prado-O'Hara VPIN metric on a tick dataset; backtest an MM strategy that withdraws quotes when VPIN exceeds a threshold.
18. **Equilibrium of $N$ executors.** $N$ traders each liquidate $X_0^{(i)}$ simultaneously, with impact coupled via total rate. Formulate the Nash game and derive the equilibrium schedule. Does it collapse to single-executor Almgren-Chriss in the limit $N \to 1$?

---

## Subject 5 — Capstone Summary

Subject 5 built asset pricing as a mathematical theory resting on five stones:

**Module 5.1 — FTAP.** No arbitrage $\Leftrightarrow$ existence of an equivalent martingale measure $\mathbb{Q}$. Completeness $\Leftrightarrow$ uniqueness of $\mathbb{Q}$. Prices are $\mathbb{Q}$-expectations of discounted payoffs.

**Module 5.2 — Black-Scholes.** Five derivations converge on one formula: $C = S_0 N(d_1) - K e^{-rT} N(d_2)$. Implied volatility inverts pricing to give the practitioner's language.

**Module 5.3 — Exotics.** Reflection (barriers), symmetry (Asians), joint distribution (lookbacks), model-free replication (variance swaps). The vocabulary of structured products.

**Module 5.4 — Interest rates.** Short-rate models → HJM → LMM → SABR. The ladder from simplicity to calibration flexibility, each rung adding a dimension of reality.

**Module 5.5 — Credit.** Structural (firm value) vs. reduced-form (intensity), with Gaussian copula as the 2008 cautionary tale. Correlation, contagion, and the limits of tractable modeling.

**Module 5.6 — Stochastic volatility.** Heston's closed-form char function, LSV hybrids, rough volatility's rough path foundations. Smile dynamics is the whole game.

**Module 5.7 — Microstructure.** Kyle, Glosten-Milgrom, Almgren-Chriss, Avellaneda-Stoikov — the mathematics of how trading actually happens. Prices emerge from strategic interaction, and costs scale with $\sqrt{Q}$.

**Bridge to Subject 6.** We have derived every major pricing and execution result under ideal modeling assumptions. Real calibration, greeks, hedging, and risk aggregation demand **numerical methods**: PDE schemes, Monte Carlo with variance reduction, Fourier and tree methods, GPU acceleration, and, increasingly, neural-network surrogates for high-dimensional problems. Subject 6 is where the formulas become functions you can run.

---

*— End of Module 5.7 and Subject 5. Next: Subject 6 — Numerical Methods in Quantitative Finance.*

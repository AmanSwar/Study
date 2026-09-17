# Subject 5, Module 2: Risk-Neutral Pricing and the Black-Scholes Formula

> *"The Black-Scholes formula is derived five different ways in the canonical textbooks, and each derivation teaches a different lesson. Mastering all five is the price of admission to real quant work."*

## Prerequisites

- **Module 3.3**: Itô's formula, Girsanov, martingale representation.
- **Module 3.4**: SDEs, Feynman-Kac, GBM.
- **Module 5.1**: Fundamental theorems, SDF, change of numéraire.
- Helpful: basic options pricing terminology (call, put, strike, payoff).

---

## 5.2.1 The Black-Scholes Model

### Setup

- Stock price $S_t$ under physical measure $\mathbb{P}$:
$$dS_t = \mu S_t \, dt + \sigma S_t \, dB_t.$$
- Risk-free rate $r$ constant, bond $B^0_t = e^{rt}$.
- Single Brownian motion (so $d = 1$, one-factor market).
- Frictionless trading: no transaction costs, perfectly divisible shares, continuous trading.

**European call option** with strike $K$ and maturity $T$ pays $(S_T - K)^+$ at $T$.

### Why Black-Scholes is the canonical model

1. **Closed-form solution.** The call price has an explicit formula in terms of the normal CDF.
2. **Exact replication.** In the BS model, every European claim can be replicated (perfectly hedged) by continuous trading in $(S, B^0)$.
3. **Unique price.** Completeness gives a unique arbitrage-free price.
4. **Model of simplest possible form.** Constant $\mu, \sigma, r$; geometric Brownian motion. Any extension adds complexity.

---

## 5.2.2 Derivation 1: Replication PDE

### The Black-Scholes PDE

Seek a function $V(t, S)$ such that holding a self-financing portfolio of shares and bonds with appropriate weights replicates the option payoff. By Itô on $V(t, S_t)$:
$$dV_t = \left(\partial_t V + \mu S_t \partial_S V + \tfrac{1}{2} \sigma^2 S_t^2 \partial_{SS} V\right) dt + \sigma S_t \partial_S V \, dB_t.$$

Hold $\Delta_t = \partial_S V(t, S_t)$ shares of stock (delta hedge), funded by bonds. The hedging portfolio dynamics:
$$d\Pi_t = \Delta_t \, dS_t + (V_t - \Delta_t S_t) \cdot \dfrac{dB^0_t}{B^0_t}$$
$$= \Delta_t (\mu S_t \, dt + \sigma S_t \, dB_t) + (V_t - \Delta_t S_t) r \, dt.$$

For $\Pi_t = V_t$ (replicating) we need
$$\partial_t V + \mu S_t \Delta_t + \tfrac{1}{2} \sigma^2 S_t^2 \partial_{SS} V = r(V - \Delta_t S_t) + \Delta_t \mu S_t,$$
$$\partial_t V + \tfrac{1}{2} \sigma^2 S^2 \partial_{SS} V + r S \partial_S V - r V = 0.$$

**The Black-Scholes PDE:**
$$\boxed{\partial_t V + \tfrac{1}{2} \sigma^2 S^2 \partial_{SS} V + r S \partial_S V - r V = 0, \quad V(T, S) = \Phi(S),}$$
where $\Phi$ is the terminal payoff (e.g., $\Phi(S) = (S - K)^+$ for a call).

**Key observation:** $\mu$ does not appear! The PDE depends only on $r, \sigma$, not the physical drift. This is a direct consequence of risk-neutral pricing.

### Solving the PDE: transformation to heat equation

Change variables: $x = \ln S, \tau = T - t$. Substitute $V(t, S) = u(\tau, x)$.

Compute:
- $\partial_t V = -\partial_\tau u$.
- $\partial_S V = (1/S) \partial_x u$.
- $\partial_{SS} V = (1/S^2)(\partial_{xx} u - \partial_x u)$.

The PDE becomes:
$$-\partial_\tau u + \tfrac{1}{2}\sigma^2 (\partial_{xx} u - \partial_x u) + r \partial_x u - r u = 0$$
$$\partial_\tau u = \tfrac{1}{2}\sigma^2 \partial_{xx} u + (r - \tfrac{1}{2}\sigma^2) \partial_x u - r u.$$

This is a convection-diffusion equation with first-order "drift" term. Remove the drift by further change: $u(\tau, x) = e^{\alpha \tau + \beta x} w(\tau, y)$ for appropriate $\alpha, \beta$ and $y = x + $ drift. Eventually reduces to the pure heat equation $\partial_\tau w = \tfrac{1}{2}\sigma^2 \partial_{yy} w$, which has explicit Gaussian fundamental solution.

Applying to the call payoff $(S - K)^+$ and inverting the transformations yields the Black-Scholes formula.

---

## 5.2.3 Derivation 2: Risk-Neutral Expectation

### Via Feynman-Kac

By Feynman-Kac (Module 3.3), the solution to the BS PDE is
$$V(t, S_t) = \mathbb{E}\left[ e^{-r(T-t)} \Phi(\tilde S_T) \, \Big| \, \tilde S_t = S_t \right],$$
where $\tilde S$ follows the **risk-neutral dynamics**:
$$d\tilde S_u = r \tilde S_u \, du + \sigma \tilde S_u \, d\tilde B_u.$$

Under $\tilde{\mathbb{P}} = \mathbb{Q}$ (risk-neutral measure), $\tilde S$ has drift $r$ (not $\mu$). This confirms:
$$V_0 = e^{-rT} \mathbb{E}_\mathbb{Q}[\Phi(S_T)].$$

### The Black-Scholes formula

For a European call $\Phi(S) = (S - K)^+$:
$$C(0, S_0) = e^{-rT} \mathbb{E}_\mathbb{Q}[(S_T - K)^+].$$

Under $\mathbb{Q}$, $\ln S_T \sim \mathcal{N}(\ln S_0 + (r - \tfrac{1}{2}\sigma^2) T, \sigma^2 T)$. So
$$\mathbb{E}_\mathbb{Q}[(S_T - K)^+] = S_0 e^{rT} N(d_1) - K N(d_2),$$
where
$$d_1 = \dfrac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2) T}{\sigma \sqrt T}, \quad d_2 = d_1 - \sigma \sqrt T.$$

Discounting:
$$\boxed{C(0, S_0) = S_0 N(d_1) - K e^{-rT} N(d_2).}$$

**Derivation detail.** Write $S_T = S_0 \exp((r - \tfrac{1}{2}\sigma^2) T + \sigma \sqrt T Z)$ with $Z \sim \mathcal{N}(0, 1)$. Then
$$\mathbb{E}_\mathbb{Q}[(S_T - K)^+] = \int_{-\infty}^\infty (S_0 e^{(r - \tfrac{1}{2}\sigma^2) T + \sigma \sqrt T z} - K)^+ \phi(z) dz.$$
Integrand nonzero when $S_0 e^{(r - \tfrac{1}{2}\sigma^2) T + \sigma \sqrt T z} > K$, i.e., $z > -d_2$. Split:
$$= S_0 \int_{-d_2}^\infty e^{(r - \tfrac{1}{2}\sigma^2) T + \sigma \sqrt T z} \phi(z) dz - K \int_{-d_2}^\infty \phi(z) dz.$$
The second integral is $N(d_2)$. For the first, complete the square in the exponent: $-\tfrac{z^2}{2} + \sigma\sqrt T z = -\tfrac{(z - \sigma\sqrt T)^2}{2} + \tfrac{\sigma^2 T}{2}$. So
$$S_0 e^{rT} \int_{-d_2}^\infty \phi(z - \sigma \sqrt T) dz = S_0 e^{rT} N(d_2 + \sigma \sqrt T) = S_0 e^{rT} N(d_1).$$

Combining and discounting:
$$C_0 = S_0 N(d_1) - K e^{-rT} N(d_2). \quad \blacksquare$$

### Put price

By put-call parity $P = C - S_0 + K e^{-rT}$:
$$P(0, S_0) = K e^{-rT} N(-d_2) - S_0 N(-d_1).$$

---

## 5.2.4 Derivation 3: Via Change of Numéraire

Using change of numéraire (Module 5.1), we can derive the BS formula even more slickly.

### The decomposition

Under $\mathbb{Q}$ (bond numéraire):
$$C_0 = e^{-rT} \mathbb{E}_\mathbb{Q}[(S_T - K) \mathbf{1}_{S_T > K}] = e^{-rT} \mathbb{E}_\mathbb{Q}[S_T \mathbf{1}_{S_T > K}] - K e^{-rT} \mathbb{Q}(S_T > K).$$

The second term: $\mathbb{Q}(S_T > K) = N(d_2)$ since $\ln S_T$ is normal under $\mathbb{Q}$.

For the first term, use stock-as-numéraire measure $\mathbb{Q}^S$. Change of measure:
$$\dfrac{d\mathbb{Q}^S}{d\mathbb{Q}} = \dfrac{S_T / S_0}{e^{rT}/1} = \dfrac{S_T}{S_0 e^{rT}}.$$

So
$$e^{-rT} \mathbb{E}_\mathbb{Q}[S_T \mathbf{1}_{S_T > K}] = S_0 \mathbb{E}_{\mathbb{Q}^S}[\mathbf{1}_{S_T > K}] = S_0 \mathbb{Q}^S(S_T > K).$$

Under $\mathbb{Q}^S$, by Girsanov, the process $\ln S_T$ has drift $r + \sigma^2/2$ (versus $r - \sigma^2/2$ under $\mathbb{Q}$). So $\mathbb{Q}^S(S_T > K) = N(d_1)$.

Combining:
$$C_0 = S_0 N(d_1) - K e^{-rT} N(d_2). \quad \blacksquare$$

**Lesson:** $N(d_1)$ is the probability of finishing in-the-money **under the stock measure**, and $N(d_2)$ is the same probability **under the bond measure**. This interpretation is subtle and beautiful.

---

## 5.2.5 Derivation 4: Continuous-Trading Binomial Limit

Take CRR binomial with $n$ steps and $u = e^{\sigma \sqrt{T/n}}, d = 1/u$, $r_n = r T/n$. The binomial price converges to BS price as $n \to \infty$.

### Proof sketch

Risk-neutral probability: $q = (e^{rT/n} - d)/(u - d)$. As $n \to \infty$:
- $q \to \tfrac{1}{2}$.
- $\ln(S_T/S_0) = \sum_{k=1}^n \ln X_k$ where $X_k = u$ or $d$ with prob $q, 1-q$.
- Mean: $nq \ln u + n(1-q) \ln d = nq \sigma \sqrt{T/n} - n(1-q)\sigma \sqrt{T/n} = n \sigma \sqrt{T/n}(2q - 1)$.

Computing $2q - 1$: expansion shows $2q - 1 \approx (r - \tfrac{1}{2}\sigma^2)\sqrt{T/n}/\sigma$, so mean $\to (r - \tfrac{1}{2}\sigma^2) T$. Variance $\to \sigma^2 T$.

By CLT $\ln(S_T/S_0) \to \mathcal{N}((r - \tfrac{1}{2}\sigma^2)T, \sigma^2 T)$, matching the GBM distribution under $\mathbb{Q}$.

So $e^{-rT} \mathbb{E}[\Phi(S_T)]$ in the binomial $\to$ BS expectation.

### The "discrete → continuous" lesson

This derivation shows BS is robust to discretization: any consistent tree-based approximation converges to it. The binomial is the discrete skeleton of Black-Scholes.

---

## 5.2.6 Derivation 5: Girsanov + Martingale Representation

### Setup

Under $\mathbb{P}$, $dS_t = \mu S_t dt + \sigma S_t dB_t$. Define $\theta = (\mu - r)/\sigma$ (market price of risk). By Girsanov, define $\mathbb{Q}$ via
$$\dfrac{d\mathbb{Q}}{d\mathbb{P}}\bigg|_{\mathcal{F}_t} = \exp\left(-\theta B_t - \tfrac{1}{2}\theta^2 t\right).$$

Under $\mathbb{Q}$, $\tilde B_t := B_t + \theta t$ is Brownian. Then
$$dS_t = (\mu - \sigma \theta) S_t \, dt + \sigma S_t \, d\tilde B_t = r S_t \, dt + \sigma S_t \, d\tilde B_t.$$

Discounted stock $\tilde S_t = S_t / B^0_t$ satisfies $d\tilde S_t = \sigma \tilde S_t d\tilde B_t$ — a $\mathbb{Q}$-martingale.

### Martingale representation

Any $\mathcal{F}_T$-measurable bounded random variable $X$ can be written as
$$X = \mathbb{E}_\mathbb{Q}[X] + \int_0^T H_s \, d\tilde S_s$$
for some predictable $H$ (martingale representation theorem). This $H$ is the delta-hedge process.

For the call: $X = (S_T - K)^+/e^{rT}$. Then
$$V_0 = \mathbb{E}_\mathbb{Q}[X] = e^{-rT} \mathbb{E}_\mathbb{Q}[(S_T - K)^+] = S_0 N(d_1) - K e^{-rT} N(d_2),$$
and the hedge is $H_t = \Delta_t = N(d_1(t, S_t))$.

This derivation emphasizes **replicability via martingale representation**, which generalizes to multi-factor and path-dependent settings.

---

## 5.2.7 The Greeks

### Definitions

For $V(t, S) = $ option price:

| Greek | Symbol | Derivative | BS call formula |
|---|---|---|---|
| Delta | $\Delta$ | $\partial V / \partial S$ | $N(d_1)$ |
| Gamma | $\Gamma$ | $\partial^2 V / \partial S^2$ | $\phi(d_1)/(S \sigma \sqrt T)$ |
| Theta | $\Theta$ | $\partial V / \partial t$ | $-S \phi(d_1)\sigma/(2\sqrt T) - r K e^{-rT} N(d_2)$ |
| Vega | $\mathcal{V}$ | $\partial V / \partial \sigma$ | $S \phi(d_1) \sqrt T$ |
| Rho | $\rho$ | $\partial V / \partial r$ | $T K e^{-rT} N(d_2)$ |

### Delta

$\Delta = N(d_1) \in (0, 1)$ for calls. Interpretation: hedge ratio — number of shares to hold.

At-the-money $S_0 \approx K$, large $T$: $\Delta \approx N(\tfrac{1}{2}\sigma\sqrt T)$, close to $0.5$. Deep in-the-money: $\Delta \to 1$. Deep out-of-money: $\Delta \to 0$.

### Gamma

$\Gamma = \phi(d_1)/(S \sigma \sqrt T) > 0$ always for vanilla calls/puts (long gamma). Peaks at the strike, decays as $|S - K|$ grows.

Trader language: "long gamma" = position profits from large moves in either direction.

### Theta

$\Theta < 0$ for long calls (options decay with time). Time decay accelerates near maturity for ATM options ("theta burn").

Theta-vega relationship (at-the-money): $\Theta \approx -\tfrac{1}{2}\sigma^2 S^2 \Gamma$ (from BS PDE).

### Vega

$\mathcal{V} > 0$ for long calls/puts. Higher volatility → higher option price.

**Vega concentration:** vega is highest for ATM options with time to expiry ~1 year. Short-dated vega small; long-dated vega large.

### Vanna, volga, charm

Second-order Greeks: vanna $\partial^2 V/\partial S \partial \sigma$, volga $\partial^2 V/\partial \sigma^2$, charm $\partial^2 V/\partial S \partial t$. Used in volatility trading and smile modeling.

---

## 5.2.8 Implied Volatility

### Definition

Given market price $C^{\text{mkt}}$ of a call, the **implied volatility** $\sigma^{\text{imp}}$ is the value of $\sigma$ satisfying
$$C^{\text{BS}}(t, S; \sigma^{\text{imp}}, K, T, r) = C^{\text{mkt}}.$$

It is found by root-finding (Newton-Raphson, Brent's method) because vega is strictly positive → BS is monotone in $\sigma$.

### The volatility smile / skew

Plot $\sigma^{\text{imp}}$ against strike $K$ (or moneyness $K/S_0$ or log-moneyness $\ln(K/S_0)$) for a fixed expiry. If BS were true, it would be flat. Instead, post-1987 equity markets show:

- **Equity skew:** $\sigma^{\text{imp}}$ is higher for OTM puts (low strikes), decreasing with strike. This is often called the "volatility skew" or "leverage effect."
- **FX smile:** roughly symmetric "smile" shape.
- **Commodity skew:** can go either direction (often "positive skew" due to upward price spikes).

### Why the smile?

The BS model assumes lognormal $S_T$ with constant $\sigma$. The market implicitly prices in:
- **Jump risk**: crashes have fatter left tails.
- **Stochastic volatility**: volatility itself varies randomly.
- **Leverage effect**: $\text{corr}(dS, d\sigma) < 0$.

Implied vol is a "wrong number in a wrong formula giving a right answer" (Rebonato): the market uses BS as a *quoting convention* while it actually prices according to a richer (non-BS) model.

### The volatility surface

Full surface: $\sigma^{\text{imp}}(K, T)$ for all strikes and maturities. Term structure:
- Short-dated: sharper skew (options capture event-driven risk).
- Long-dated: flatter surface (central-limit-like averaging).

At-the-money volatility term structure often shows:
- **ATM variance**: $\sigma^{\text{imp}}_{ATM}(T)^2 \cdot T$ increases with $T$, often approximately linearly.

---

## 5.2.9 Dupire's Local Volatility

### Dupire's formula

Given the implied vol surface $\sigma^{\text{imp}}(K, T)$, the **local volatility function** $\sigma^{\text{loc}}(K, T)$ that makes the model reproduce all market prices is given by **Dupire's formula**:
$$\boxed{\sigma^{\text{loc}}(K, T)^2 = \dfrac{\partial C / \partial T + r K \partial C / \partial K}{\tfrac{1}{2} K^2 \partial^2 C / \partial K^2}.}$$

Derivation: Start from Fokker-Planck in $(S, T)$ for the risk-neutral density $p(T, S)$, relate to call prices via $\partial_{KK} C = e^{-rT} p$, integrate Kolmogorov forward. Result is the Dupire PDE in strike space:
$$\partial_T C = \tfrac{1}{2} \sigma^{\text{loc}}(K, T)^2 K^2 \partial_{KK} C - r K \partial_K C.$$

Rearranging gives Dupire's formula.

### In terms of implied vol

Practically, we observe $\sigma^{\text{imp}}(K, T)$ not $C(K, T)$. Substituting gives a formula in implied-vol derivatives:
$$\sigma^{\text{loc}}(K, T)^2 = \dfrac{\sigma^{\text{imp}}^2 + 2T \sigma^{\text{imp}} \partial_T \sigma^{\text{imp}} + 2rKT\sigma^{\text{imp}} \partial_K \sigma^{\text{imp}}}{(1 - K y_1/\sigma^{\text{imp}} \cdot \partial_K \sigma^{\text{imp}})^2 + K^2 \sigma^{\text{imp}} T \cdot (\partial_{KK} \sigma^{\text{imp}} - \partial_K \sigma^{\text{imp}} \cdot y_1 / \sigma^{\text{imp}})}$$
with $y_1 = d_1 - \sigma^{\text{imp}}\sqrt T$ or similar (various forms). Numerically implemented via finite differences on the implied-vol surface.

### Use in pricing

The local vol model $dS = r S dt + \sigma^{\text{loc}}(S, t) S dB$ reproduces **all** market-observed European prices. It is used to price exotics consistent with the market.

**Caveat:** local vol correctly reproduces the surface at time 0 but dynamics of the smile are unrealistic (smile flattens deterministically with time, contradicting market behavior). For volatility-sensitive exotics, stochastic volatility is needed.

---

## 5.2.10 Python Implementation

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# -----------------------------
# Black-Scholes formula
# -----------------------------
def bs_call(S, K, r, sigma, T):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    return bs_call(S, K, r, sigma, T) + K*np.exp(-r*T) - S

def bs_greeks(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return {
        'delta': norm.cdf(d1),
        'gamma': norm.pdf(d1) / (S*sigma*np.sqrt(T)),
        'theta': -S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2),
        'vega':  S*norm.pdf(d1)*np.sqrt(T),
        'rho':   K*T*np.exp(-r*T)*norm.cdf(d2)
    }

# Sanity check
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
print(f"BS call = {bs_call(S0, K, r, sigma, T):.4f}")
print(f"BS put  = {bs_put(S0, K, r, sigma, T):.4f}")
print(f"Greeks  = {bs_greeks(S0, K, r, sigma, T)}")
print(f"Put-call parity check: C - P = {bs_call(S0,K,r,sigma,T) - bs_put(S0,K,r,sigma,T):.4f}, S_0 - K e^-rT = {S0 - K*np.exp(-r*T):.4f}")

# -----------------------------
# Monte Carlo verification
# -----------------------------
N = 200_000
Z = np.random.randn(N)
ST = S0 * np.exp((r - sigma**2/2)*T + sigma*np.sqrt(T)*Z)
C_mc = np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))
print(f"MC call = {C_mc:.4f} (analytic {bs_call(S0,K,r,sigma,T):.4f})")

# -----------------------------
# Implied volatility inversion
# -----------------------------
def implied_vol(C_mkt, S, K, r, T, tol=1e-8):
    f = lambda sig: bs_call(S, K, r, sig, T) - C_mkt
    return brentq(f, 1e-4, 5.0, xtol=tol)

C_market = bs_call(100, 100, 0.05, 0.25, 1.0)
sig_imp  = implied_vol(C_market, 100, 100, 0.05, 1.0)
print(f"Implied vol = {sig_imp:.6f} (true 0.25)")

# -----------------------------
# Visualize Greeks
# -----------------------------
S_grid = np.linspace(50, 150, 100)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(S_grid, [bs_greeks(s, 100, 0.05, 0.2, 1.0)['delta'] for s in S_grid], label='delta')
ax[0].plot(S_grid, [bs_greeks(s, 100, 0.05, 0.2, 0.1)['delta'] for s in S_grid], label='delta (T=0.1)')
ax[0].set_title('Delta'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(S_grid, [bs_greeks(s, 100, 0.05, 0.2, 1.0)['gamma'] for s in S_grid])
ax[1].plot(S_grid, [bs_greeks(s, 100, 0.05, 0.2, 0.1)['gamma'] for s in S_grid])
ax[1].set_title('Gamma'); ax[1].grid(alpha=0.3)
ax[2].plot(S_grid, [bs_greeks(s, 100, 0.05, 0.2, 1.0)['vega'] for s in S_grid])
ax[2].plot(S_grid, [bs_greeks(s, 100, 0.05, 0.2, 0.1)['vega'] for s in S_grid])
ax[2].set_title('Vega'); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# -----------------------------
# Volatility smile visualization (synthetic)
# -----------------------------
# Generate prices under a stochastic vol model (Heston-lite) and extract implied vols
def heston_mc_prices(S0, r, v0, kappa, theta_v, xi, rho, T, Ks, n_paths=50_000, n_steps=200):
    dt = T / n_steps
    S = np.full(n_paths, S0); v = np.full(n_paths, v0)
    for _ in range(n_steps):
        Z1, Z2 = np.random.randn(n_paths), np.random.randn(n_paths)
        Zv = Z1; Zs = rho*Z1 + np.sqrt(1-rho**2)*Z2
        v_prev = np.maximum(v, 0)
        v = v + kappa*(theta_v - v_prev)*dt + xi*np.sqrt(v_prev*dt)*Zv
        v = np.maximum(v, 1e-8)
        S = S * np.exp((r - v_prev/2)*dt + np.sqrt(v_prev*dt)*Zs)
    prices = [np.exp(-r*T) * np.mean(np.maximum(S - K, 0)) for K in Ks]
    return prices

Ks = np.linspace(70, 130, 15)
h_prices = heston_mc_prices(100, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0, Ks)
iv_smile = [implied_vol(p, 100, K, 0.05, 1.0) for p, K in zip(h_prices, Ks)]

plt.figure(figsize=(9, 5))
plt.plot(Ks/100, iv_smile, 'o-')
plt.xlabel('moneyness K/S0'); plt.ylabel('implied vol')
plt.title('Volatility smile from Heston simulation')
plt.grid(alpha=0.3); plt.show()

# -----------------------------
# Delta hedging simulation
# -----------------------------
def delta_hedge_sim(S0, K, r, sigma, T, n_steps=250):
    dt = T/n_steps
    S = S0
    cash = bs_call(S0, K, r, sigma, T) - bs_greeks(S0, K, r, sigma, T)['delta']*S0
    shares = bs_greeks(S0, K, r, sigma, T)['delta']
    ts = [0]; Ss = [S]; hedge_vals = [cash + shares*S]
    for step in range(1, n_steps+1):
        dB = np.sqrt(dt) * np.random.randn()
        S = S * np.exp((r - sigma**2/2)*dt + sigma*dB)
        cash *= np.exp(r*dt)
        t_rem = T - step*dt
        if t_rem > 0:
            new_delta = bs_greeks(S, K, r, sigma, t_rem)['delta']
            cash -= (new_delta - shares)*S
            shares = new_delta
        ts.append(step*dt); Ss.append(S); hedge_vals.append(cash + shares*S)
    payoff = max(S - K, 0)
    final_cash = cash + shares*S
    return final_cash - payoff

# Statistics over many simulations
np.random.seed(0)
errors = [delta_hedge_sim(100, 100, 0.05, 0.2, 1.0, 250) for _ in range(500)]
print(f"\nDelta hedging error (250 steps): mean={np.mean(errors):.4f}, std={np.std(errors):.4f}")
errors_rough = [delta_hedge_sim(100, 100, 0.05, 0.2, 1.0, 50) for _ in range(500)]
print(f"Delta hedging error ( 50 steps): mean={np.mean(errors_rough):.4f}, std={np.std(errors_rough):.4f}")
# Hedge error std scales like sqrt(1/n_steps); Bertsimas-Kogan-Lo result
```

---

## 5.2.11 [QUANT APPLICATION]

1. **Vanilla trading desks** — every equity, FX, and commodity options desk uses BS as a lingua franca, quoting prices via implied vol. Market makers hedge delta continuously (or rebalance periodically) and manage residual gamma/vega risk.

2. **Implied vol as a state variable.** Option traders think in vol, not price. Indices like VIX are constructed from implied vols; variance swaps replicate directly.

3. **Volatility surface arbitrage.** Smile fitting algorithms (SVI, SSVI) parameterize the surface and arbitrage-check it. Violations (calendar arbitrage, butterfly arbitrage) are systematically exploited or flagged.

4. **Structured products.** Autocallables, principal-protected notes, variance-swap replication — all priced as BS + exotic adjustments, often Monte Carlo'd with a local-vol or stochastic-vol model calibrated to the BS-implied surface.

5. **Model-independent bounds.** Breeden-Litzenberger identity $\partial^2_K C = e^{-rT} p(K, T)$ lets you extract the risk-neutral density from option prices — used in distribution-free pricing of digitals and other exotics.

6. **Risk management — VaR under changing vol.** Vega-weighted portfolio risk, stress-testing under alternative vol surfaces.

7. **Delta hedging in practice.** Discrete rebalancing gives nonzero replication error; Bertsimas-Kogan-Lo (2000) show error std $\propto \sqrt{1/n}$. Transaction costs add dispersion; optimal hedging frequency balances costs vs tracking error.

8. **Employee stock options** are priced via BS with early-exercise adjustments. Fair-value accounting (FAS 123R) requires such valuations.

9. **Real options analysis.** Corporate investment decisions treated as option pricing: expansion = call, abandonment = put. BS or binomial valuation.

10. **Convertible bonds.** Hybrid credit + equity; priced as bond + equity call via BS-like framework with stochastic credit.

---

## 5.2.12 Summary

- **Black-Scholes PDE**: $\partial_t V + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} V + r S \partial_S V - r V = 0$.
- **BS formula**: $C = S N(d_1) - K e^{-rT} N(d_2)$, $P = K e^{-rT} N(-d_2) - S N(-d_1)$.
- **Five derivations**: PDE, risk-neutral expectation, change of numéraire, binomial limit, Girsanov + martingale representation — each illuminates different structure.
- **Greeks**: delta, gamma, theta, vega, rho; closed-form in BS.
- **Implied volatility**: a market-quoting convention; smile/skew reflect non-BS features of the market.
- **Dupire's local vol**: extracts $\sigma^{\text{loc}}(K, T)$ from market prices, reproducing the surface.

### Forward pointers

- **Module 5.3**: path-dependent exotics — barriers, Asians, lookbacks, cliquets; static replication of variance swaps.
- **Module 5.4**: interest-rate models — Vasicek, Hull-White, HJM, LMM.
- **Module 5.5**: credit risk and CDS.
- **Module 5.6**: stochastic volatility — Heston, SABR, rough vol.

---

## Exercises

### Tier 1 (★)

1. Compute $C$ for $S_0 = 100, K = 105, r = 0.03, \sigma = 0.25, T = 0.5$.
2. Verify put-call parity for Ex 1.
3. Implement BS Greeks and verify $\Theta \approx -\tfrac{1}{2}\sigma^2 S^2 \Gamma - rV + rS\Delta$.
4. For $K = S_0 = 100, r = 0, \sigma = 0.2, T = 1$: compute all five Greeks.
5. Derive the delta of a call using the chain rule: $\Delta = \partial C / \partial S$, applying the formula.
6. Show that as $\sigma \to 0$, BS call $\to (S_0 e^{rT} - K)^+ e^{-rT}$ (forward-based payoff).
7. Show that as $\sigma \to \infty$ with $S, K$ fixed, BS call $\to S$ (option becomes stock).
8. Compute vega of a 1-year ATM option with $S_0 = 100, r = 0, \sigma = 0.2$.

### Tier 2 (★★)

9. Prove the BS formula by completing the square, explicitly separating the $N(d_1)$ and $N(d_2)$ terms.
10. Derive BS under continuous dividend yield $q$: $dS = (\mu - q) S dt + \sigma S dB$. Price: $C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$ with $d_1 = [\ln(S_0/K) + (r - q + \tfrac{1}{2}\sigma^2)T]/(\sigma\sqrt T)$.
11. Derive BS in the forward measure: $C = e^{-rT}[F N(d_1) - K N(d_2)]$ with $F = S_0 e^{rT}$.
12. Prove Bertsimas-Kogan-Lo: discrete delta hedging with $n$ equally spaced times has expected squared hedge error $\propto 1/n$.
13. Derive Dupire's formula from the Fokker-Planck equation.
14. Show that in BS, $\Gamma = \text{vega}/(S^2 \sigma T)$ (Greek relationship).
15. Compute the sensitivity of a call to discrete dividends paid at known dates.
16. Prove that implied vol is well-defined (monotonic in $\sigma$) and continuous in market price.

### Tier 3 (★★★)

17. Implement calibration of local vol surface to a given implied vol surface; test on SVI-fitted surface.
18. Prove the symmetry of BS: $P(S, K) = K \cdot C(K, S)/S$ (put-call symmetry under $r = 0, q = 0$).
19. Derive the BS price under time-dependent deterministic $\sigma(t)$ — replace $\sigma^2 T$ with $\int_0^T \sigma(s)^2 ds$.
20. Implement static replication of variance swaps via weighted portfolio of OTM options (Derman-Kamal-Demeterfi-Zou).
21. Derive the BS formula for a "power option" paying $(S_T^\alpha - K)^+$.
22. Implement a "delta-gamma hedge" using two options and the underlying; compute the residual error.
23. Derive the SABR approximate implied vol formula (Hagan-Kumar-Lesniewski-Woodward 2002).
24. Solve the Black-Scholes inverse problem: given option prices, recover the risk-neutral density via Breeden-Litzenberger.
25. Prove that in BS the value of an American call (no dividends) equals the European call (Merton 1973).
26. Derive the forward-start option price: pays $(S_T - \kappa S_t)^+$ at $T$ where strike fixed at time $t < T$ as a fraction $\kappa$ of $S_t$.
27. Numerically solve the full BS PDE via Crank-Nicolson finite difference; compare with analytic.
28. Investigate the Black-Scholes "smile consistency": can you choose $\sigma^{\text{imp}}(K, T)$ arbitrarily, or are there shape constraints?
29. Prove that the local vol function extracted by Dupire is the conditional expectation $\sigma^{\text{loc}}(K, T) = \mathbb{E}_\mathbb{Q}[\sigma^2_T | S_T = K]^{1/2}$ in a stochastic vol model (Gatheral's formula).

---

*Next module:* Exotic options and path-dependent derivatives — barriers, Asians, lookbacks, cliquets, variance swaps, and the methods for pricing and hedging them.

# Subject 5, Module 3: Exotic Options and Path-Dependent Derivatives

> *"Vanillas teach you the framework. Exotics teach you which features of the framework really matter — which approximations hold, which break, and where the risk hides."*

## Prerequisites

- **Modules 5.1-5.2**: FTAP, risk-neutral pricing, Black-Scholes, Greeks, implied vol, Dupire.
- **Module 3.3**: Itô, Feynman-Kac, Girsanov.
- **Module 3.1**: Brownian motion, reflection principle, hitting times.
- Helpful: basic numerical methods (MC, finite differences).

---

## 5.3.1 Taxonomy of Exotics

"Exotic" = not a vanilla call/put. Key classifications:

**Path-dependence:**
- **Vanilla (European):** payoff depends only on $S_T$.
- **Weakly path-dependent:** payoff depends on $S_T$ and one other path statistic (e.g., max, min, average).
- **Strongly path-dependent:** payoff depends on the entire path.

**Exercise features:**
- **European:** exercise only at $T$.
- **American:** exercise any time up to $T$ (Module 4.2).
- **Bermudan:** exercise on a discrete set of dates.

**Payoff structure:**
- **Continuous:** smooth function of state variables.
- **Discontinuous (digital / barrier):** step functions.
- **Capped/floored:** bounded payoffs.

**Major categories covered in this module:**
1. Barrier options (knock-in/knock-out, up/down).
2. Asian options (arithmetic and geometric averages).
3. Lookback options (payoff involving max or min).
4. Digital/binary options.
5. Cliquet options (ratchet of forward-starting options).
6. Variance and volatility swaps.
7. Basket and best-of/worst-of options.

---

## 5.3.2 Barrier Options

### Definitions

A **barrier** is a price level $B$ that triggers activation or termination:

- **Up-and-out call** (UOC): standard call, but if $\max_{t \le T} S_t \ge B$ it knocks out (worthless).
- **Down-and-out call** (DOC): knocks out if $\min_{t \le T} S_t \le B$.
- **Up-and-in call** (UIC): activated only if $S$ crosses $B$ from below.
- **Down-and-in call** (DIC): activated only if $S$ crosses $B$ from above.

**In-out parity:** $\text{Vanilla} = \text{In} + \text{Out}$. So we only need to price one of each pair.

### Pricing: Reflection Principle

For GBM $S_t = S_0 \exp((r - \tfrac{1}{2}\sigma^2)t + \sigma B_t)$, the distribution of $(\max_{t \le T} S_t, S_T)$ follows from the Brownian reflection principle.

**Lemma (Reflection, BM with drift):** For $X_t = \mu t + \sigma B_t$ and $m_T = \max_{t \le T} X_t$,
$$\mathbb{P}(m_T \ge a, X_T \le b) = e^{2\mu a/\sigma^2} \mathbb{P}(X_T \ge 2a - b), \quad a \ge b, a > 0.$$

Applying to $X_t = \ln(S_t/S_0)$ under $\mathbb{Q}$ gives closed-form prices.

### Down-and-out call (DOC) with barrier $B < S_0$, $K$

The standard Merton (1973) formula: assuming $K \ge B$,
$$\text{DOC} = C_{\text{BS}}(S_0, K) - \left(\dfrac{B}{S_0}\right)^{2\lambda} C_{\text{BS}}(B^2/S_0, K),$$
where $\lambda = (r + \tfrac{1}{2}\sigma^2)/\sigma^2$, $C_{\text{BS}}(s, k) = s N(d_1(s,k)) - K e^{-rT} N(d_2(s,k))$ is the vanilla call price.

If $K < B$: need different formula. Case analysis is finicky; see Haug's *Complete Guide to Option Pricing Formulas*.

### General barrier formulas

Merton (1973) and Rubinstein-Reiner (1991) give the full suite of eight barrier option formulas (up/down, in/out, call/put). They all follow the same image-method pattern: "reflect" the payoff across the barrier and subtract/add the image.

### Key insight: static replication (Carr-Chou 1997)

Remarkable: many barrier options can be **statically replicated** (no dynamic hedging) by a portfolio of European options. For DOC:
$$\text{DOC}(S_0, K, B) = C(S_0, K) - \text{(reflection term)}$$
and the reflection term is itself a payout at some effective strike on a barrier-hit trigger. Using put-call parity and the reflection identity, one can construct a *static* replicating portfolio of vanilla puts and calls at different strikes.

Advantage: no dynamic hedging → no transaction costs, less model risk.

### Discrete monitoring

In practice barriers are monitored daily, not continuously. Continuous-barrier approximation systematically underprices knock-outs (the continuous path overshoots the discrete). **Broadie-Glasserman-Kou (1997)** correction:
$$B_{\text{effective}} = B \exp(\beta_1 \sigma \sqrt{\Delta t})$$
with $\beta_1 \approx 0.5826$ (related to Riemann zeta function).

---

## 5.3.3 Asian Options

### Payoffs

- **Arithmetic average call:** $\left(\dfrac{1}{T} \int_0^T S_t dt - K\right)^+$ or discrete analog $\left(\tfrac{1}{n} \sum S_{t_i} - K\right)^+$.
- **Geometric average call:** $\left(\exp\left(\tfrac{1}{T}\int_0^T \ln S_t dt\right) - K\right)^+$.

Asians are cheaper than vanillas and are useful when the buyer wants to reduce impact of terminal price manipulation or single-day spikes (common in commodity markets).

### Geometric Asian: closed form

Under $\mathbb{Q}$, $\ln S_t$ is Gaussian, so $\tfrac{1}{T}\int_0^T \ln S_t dt$ is also Gaussian:
$$Y := \tfrac{1}{T}\int_0^T \ln S_t dt = \ln S_0 + (r - \tfrac{1}{2}\sigma^2) \cdot \tfrac{T}{2} + \tfrac{\sigma}{T} \int_0^T (T - t) dB_t.$$

Mean: $\ln S_0 + (r - \tfrac{1}{2}\sigma^2)T/2$.
Variance: $\sigma^2 T/3$.

So the geometric average option reduces to a BS call with:
- Effective spot: $G_0 = \exp(\text{Mean}) = S_0 \exp((r - \tfrac{1}{2}\sigma^2)T/2 - \sigma^2 T/12)$ ... wait, let me redo.

Actually, $\exp(Y) = \exp(\mathbb{E}[Y] + \tfrac{1}{2}\text{Var}(Y))$ is the log-normal expectation. The price is
$$e^{-rT} \mathbb{E}_\mathbb{Q}[(\exp(Y) - K)^+] = \text{BS}(G_0, K, r_{\text{eff}}, \sigma_{\text{eff}}, T),$$
with effective $\sigma_{\text{eff}} = \sigma/\sqrt 3$ and effective $r_{\text{eff}} = \tfrac{1}{2}(r + \sigma^2/6)$.

### Arithmetic Asian: no closed form

The arithmetic average of log-normals is not log-normal. No exact closed form, but:

**Moment matching (Turnbull-Wakeman 1991):** approximate arithmetic average by log-normal matching first two moments:
$$\mathbb{E}[A] = \dfrac{S_0}{T} \int_0^T e^{rt} dt = S_0 \cdot \dfrac{e^{rT} - 1}{rT}, \quad \mathbb{E}[A^2] = \text{computable by double integral}.$$

Then price as BS with those moments. Accuracy: ~1-2% for $\sigma T < 0.3$.

**Geometric control variate (Kemna-Vorst 1990):** MC estimator with variance reduction:
$$\hat A = \text{MC arith} - \beta (\text{MC geom} - \text{analytic geom}).$$
Optimal $\beta^* = \text{Cov}(A_{\text{arith}}, A_{\text{geom}})/\text{Var}(A_{\text{geom}})$, typically reduces MC variance by 99%+.

**Laplace-transform methods (Geman-Yor 1993):** characteristic function known; numerical inversion.

### Variance reduction example

```python
# Kemna-Vorst control variate for arithmetic Asian
import numpy as np

def asian_mc(S0, K, r, sigma, T, n_steps=252, n_paths=50_000, avg_type='arith'):
    dt = T / n_steps
    Z = np.random.randn(n_paths, n_steps)
    S_path = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
    S_path = np.hstack([np.full((n_paths, 1), S0), S_path])
    if avg_type == 'arith':
        A = S_path.mean(axis=1)
    else:
        A = np.exp(np.log(S_path).mean(axis=1))
    return np.exp(-r*T) * np.maximum(A - K, 0)

S0, K, r, sigma, T = 100, 100, 0.05, 0.3, 1.0
arith_vals = asian_mc(S0, K, r, sigma, T, avg_type='arith')
geom_vals  = asian_mc(S0, K, r, sigma, T, avg_type='geom')  # use same RNG!

# ... compute analytical geometric price and apply control variate adjustment ...
```

---

## 5.3.4 Lookback Options

### Payoffs

- **Floating-strike lookback call:** $S_T - \min_{t \le T} S_t$ (pays the range from lowest to final).
- **Fixed-strike lookback call:** $(\max_{t \le T} S_t - K)^+$ (pays the max price above strike).

Guaranteed to be exercised at the best price - very expensive.

### Pricing

Joint distribution $(\max, S_T)$ for GBM: reflection principle gives joint density. Goldman-Sosin-Gatto (1979) derived closed forms:

**Floating-strike lookback call** (on asset paying no dividend, constant vol):
$$V_0 = S_0 N(a_1) - S_0 e^{-rT} \dfrac{\sigma^2}{2r} N(-a_1) + S_0 e^{-rT} \dfrac{\sigma^2}{2r}\left(\dfrac{S_0}{m}\right)^{-2r/\sigma^2} N(-a_1 + 2r\sqrt T/\sigma) - m e^{-rT} N(a_2)$$
for $m = \min_{t \le 0} S_t$ (already-observed minimum; use $S_0$ if no history), and with appropriate $a_1, a_2$.

The formula involves four normal CDF terms (two for the "nominal" expectation and two from the reflection / boundary correction).

### Discrete monitoring

Similar to barriers, continuous approximation systematically biases discrete-monitoring lookbacks. Andreasen (1998) and Broadie-Glasserman-Kou continuity corrections apply.

---

## 5.3.5 Digital/Binary Options

### Payoffs

- **Cash-or-nothing call:** pays $\mathbf{1}_{S_T > K}$. Price: $e^{-rT} N(d_2)$.
- **Asset-or-nothing call:** pays $S_T \mathbf{1}_{S_T > K}$. Price: $S_0 N(d_1)$.

Vanilla call = asset-or-nothing - $K \times$ cash-or-nothing. This identity is the basis of the BS split $C = S_0 N(d_1) - K e^{-rT} N(d_2)$.

### Hedging challenges

Digital option payoff is discontinuous at $K$. Delta $\to \infty$ near the strike at expiry (infinite gamma at maturity). Makes hedging **very difficult** near maturity for strikes near ATM.

Market solution: **over-replication** by vertical spreads. Replicate digital $\mathbf{1}_{S_T > K}$ by $(1/\epsilon)(\text{call}(K-\epsilon) - \text{call}(K+\epsilon))$ for small $\epsilon$. Cost: slightly higher price, but bounded delta.

---

## 5.3.6 Cliquet Options

### Structure

Cliquet (or ratchet) = series of forward-starting options with periodic reset. Example payoff at maturity $T = n T_1$:
$$V_T = \max\left(C, \min\left(F, \sum_{i=1}^n \max(S_{t_i}/S_{t_{i-1}} - 1, 0)\right)\right)$$
for some cap $F$ and floor $C$.

Pays the cumulative periodic returns (capped / floored). Popular in structured products ("guaranteed minimum accumulation benefits" in variable annuities).

### Pricing

Each period's return is a forward-starting option. Under BS with constant $\sigma$:
$$\mathbb{E}_\mathbb{Q}[\max(S_{t_i}/S_{t_{i-1}} - 1, 0)] = \text{BS call with } S_0 = 1, K = 1, T = t_i - t_{i-1}.$$

So an unbounded cliquet = $n$ × price of ATM BS call.

For capped/floored: MC or analytic with sums of truncated normals.

### Vega structure

Cliquets have interesting vega distribution: positive forward-vol sensitivity but near-zero sensitivity to cumulative variance. This is why they were mispriced in the 2000s by dealers using local-vol models which underestimate forward variance.

---

## 5.3.7 Variance and Volatility Derivatives

### Variance swap

Payoff at $T$: realized variance minus strike $K_{\text{var}}$:
$$V_T = \dfrac{1}{T}\int_0^T \sigma_t^2 dt - K_{\text{var}},$$
scaled by some notional (typically $\$/vol^2$).

In practice uses daily log-return variance: $V_T = \tfrac{252}{n} \sum_{i=1}^n (\ln(S_{t_i}/S_{t_{i-1}}))^2 - K_{\text{var}}$.

### Model-free replication

Here's the magic: variance swaps can be **model-independently** replicated using a portfolio of European options!

**Theorem (Neuberger 1994, Derman-Kamal-Demeterfi-Zou 1999):** The fair strike $K_{\text{var}}$ satisfies
$$K_{\text{var}} = \dfrac{2}{T}\left[ rT - \ln\left(\dfrac{F}{S_0}\right) + \int_0^F \dfrac{P(K)}{K^2} dK + \int_F^\infty \dfrac{C(K)}{K^2} dK \right] e^{rT},$$
where $F = S_0 e^{rT}$ is the forward, $P(K)$ is the put price at strike $K$, $C(K)$ is the call price.

**Proof sketch.** The identity $\int_0^T \sigma^2 dt = 2\int_0^T (dS_t/S_t - d\ln S_t)$ follows from Itô. Take expectation under $\mathbb{Q}$: first term is 0 (martingale), second term is $-2 \mathbb{E}[\ln(S_T/S_0)]$.

Expand $\ln(S_T/S_0)$ around forward $F$ using the Breeden-Litzenberger identity: any twice-differentiable payoff $f(S_T)$ replicates as
$$f(S_T) = f(F) + f'(F)(S_T - F) + \int_0^F f''(K)(K - S_T)^+ dK + \int_F^\infty f''(K)(S_T - K)^+ dK.$$

Applying to $f(S) = \ln S$ (with $f''(K) = -1/K^2$) gives the result.

This is one of the most beautiful results in derivatives pricing — **purely model-free**, just requiring the existence of the option portfolio.

### Volatility swap

Pays $\sqrt{\text{realized var}} - K_{\text{vol}}$. NOT model-free: the square root introduces model dependence (convexity: $\mathbb{E}[\sqrt X] \ne \sqrt{\mathbb{E}[X]}$).

**Convexity adjustment (approximate):** $K_{\text{vol}} \approx \sqrt{K_{\text{var}}} \cdot (1 - \tfrac{1}{8} \text{Var}[V]/K_{\text{var}}^2)$.

### VIX

VIX is CBOE's *volatility index*, defined as the forward-looking 30-day implied variance of SPX (annualized and square-rooted). Formula:
$$\text{VIX}^2 = \dfrac{2}{T}\left[\sum_i \dfrac{\Delta K_i}{K_i^2} e^{rT} Q(K_i) - \dfrac{1}{T}\left(\dfrac{F}{K_0} - 1\right)^2\right]$$
where $Q(K)$ is the OTM option price at strike $K$. Discrete version of Derman-Kamal-Demeterfi-Zou formula.

VIX futures and options trade; the CBOE VIX ETPs (VXX, UVXY, SVXY) have been major drivers of volatility market structure post-2010.

---

## 5.3.8 Basket Options and Best-of/Worst-of

### Basket payoff

European basket call: $(w_1 S_1(T) + \cdots + w_n S_n(T) - K)^+$.

Under multidimensional BS ($S_i$ correlated lognormals), exact pricing is not analytical (sum of lognormals is not lognormal).

**Moment matching:** lognormal approximation, like arithmetic Asian.

**Monte Carlo:** workhorse for basket options. Use Cholesky or PCA for correlation simulation.

### Best-of / worst-of

$V_T = \max(S_1(T), S_2(T), \ldots, S_n(T))$ or $\min$ thereof.

Analytic formulas exist for 2-asset best-of via bivariate normal CDF (Stulz 1982). For $n \ge 3$, use MC or quasi-MC.

**Correlation risk.** Basket/best-of prices are highly sensitive to correlation. Dispersion trades (long vol of components, short vol of basket) exploit this.

### Correlation smile

Analogous to vol surface: observed implied correlations from basket/best-of prices often differ from historical correlations, showing a "smile" (higher implied for OTM strikes). Multi-asset stochastic correlation models (Ma, van Emmerich) address this.

---

## 5.3.9 Pricing Methods for Exotics

### Monte Carlo

**Workhorse.** For any path-dependent payoff $\Phi(S_{t_1}, \ldots, S_{t_n})$:
$$V_0 = e^{-rT} \mathbb{E}_\mathbb{Q}[\Phi] \approx e^{-rT} \cdot \dfrac{1}{N}\sum_{k=1}^N \Phi^{(k)}.$$

- **Variance reduction:** antithetic, control variates, importance sampling, stratified sampling.
- **Quasi-MC (Sobol, Halton):** low-discrepancy sequences give $O(N^{-1+\epsilon})$ convergence, much better than crude MC $O(N^{-1/2})$ — but only for low-to-moderate dimension.
- **Multilevel MC (Giles 2008):** combine different time-step granularities for asymptotically $O(N^{-1})$ cost.

### Finite differences (PDE)

For Markovian path-dependent payoffs, augment the state space. E.g., Asian option: state $(S, A)$ where $A_t = \int_0^t S_s ds$. Solve 2D PDE.

Methods: explicit, Crank-Nicolson, ADI (alternating direction implicit) for multi-D, operator splitting.

### Trees

Binomial/trinomial with path-information: CRR binomial extended to track barrier status, averages, etc. Fast for low-dimensional path-dependent.

### Fourier methods (Carr-Madan, COS)

For any model with tractable characteristic function $\psi(u) = \mathbb{E}[e^{iu \ln S_T}]$:
- **Carr-Madan (1999) FFT:** damp call price and use FFT.
- **COS method (Fang-Oosterlee 2008):** Fourier-cosine series.

These are most useful for vanilla options in non-BS models (Heston, variance gamma, Merton jump-diffusion). Extended versions exist for some exotics.

### Longstaff-Schwartz for Bermudan/path-dependent

Regression-based MC (see Module 4.2) for American and Bermudan exotics.

### Deep learning

Recent: deep BSDE (Han-Jentzen-E), deep hedging (Buehler), neural PDE solvers. Highly flexible for high-dimensional exotics but lack rigorous error bounds.

---

## 5.3.10 Python: Exotic Options Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Simulate GBM paths
# -----------------------------
def gbm_paths(S0, r, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    Z = np.random.randn(n_paths, n_steps)
    log_returns = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    S = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), S])

# -----------------------------
# Barrier option: Down-and-out call
# -----------------------------
def down_and_out_call_mc(S0, K, B, r, sigma, T, n_paths=100_000, n_steps=252):
    paths = gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    min_S = paths.min(axis=1)
    knocked_out = min_S <= B
    payoff = np.where(knocked_out, 0.0, np.maximum(paths[:, -1] - K, 0))
    return np.exp(-r*T) * payoff.mean(), payoff.std()/np.sqrt(n_paths)

def down_and_out_call_analytic(S0, K, B, r, sigma, T):
    # Merton formula (assuming K >= B)
    def C_BS(S, K):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    lam = (r + sigma**2/2) / sigma**2
    return C_BS(S0, K) - (B/S0)**(2*lam) * C_BS(B**2/S0, K)

# Compare
S0, K, B, r, sigma, T = 100, 100, 90, 0.05, 0.20, 1.0
mc_val, mc_se = down_and_out_call_mc(S0, K, B, r, sigma, T)
an_val = down_and_out_call_analytic(S0, K, B, r, sigma, T)
print(f"DOC: MC = {mc_val:.4f} +/- {mc_se*1.96:.4f}, analytic = {an_val:.4f}")

# -----------------------------
# Asian option (geometric analytic, arithmetic MC with control variate)
# -----------------------------
def geom_asian_analytic(S0, K, r, sigma, T):
    sig_eff = sigma / np.sqrt(3)
    r_eff   = 0.5*(r - sigma**2/6)
    d1 = (np.log(S0/K) + (r_eff + sig_eff**2/2)*T) / (sig_eff*np.sqrt(T))
    d2 = d1 - sig_eff*np.sqrt(T)
    return np.exp((r_eff - r)*T) * (S0*norm.cdf(d1) - K*np.exp(-r_eff*T)*norm.cdf(d2))

def asian_options_mc(S0, K, r, sigma, T, n_paths=50_000, n_steps=252):
    np.random.seed(42)
    paths = gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    A_arith = paths.mean(axis=1)
    A_geom  = np.exp(np.log(paths).mean(axis=1))
    p_arith = np.exp(-r*T) * np.maximum(A_arith - K, 0)
    p_geom  = np.exp(-r*T) * np.maximum(A_geom - K, 0)
    G_an    = geom_asian_analytic(S0, K, r, sigma, T)
    # Control variate: use geometric as control
    beta = np.cov(p_arith, p_geom)[0,1] / np.var(p_geom)
    p_arith_cv = p_arith - beta*(p_geom - G_an)
    return {
        'arith_crude':  (p_arith.mean(), p_arith.std()/np.sqrt(n_paths)),
        'arith_cv':     (p_arith_cv.mean(), p_arith_cv.std()/np.sqrt(n_paths)),
        'geom_mc':      (p_geom.mean(), p_geom.std()/np.sqrt(n_paths)),
        'geom_analytic': G_an,
    }

results = asian_options_mc(100, 100, 0.05, 0.3, 1.0)
for key, val in results.items():
    print(f"  {key}: {val}")
# Control variate reduces variance by ~99%

# -----------------------------
# Variance swap: model-free replication
# -----------------------------
def variance_swap_strike(S0, r, T, put_prices, call_prices, put_strikes, call_strikes):
    F = S0 * np.exp(r*T)
    # Put integral for K < F
    put_integrand = put_prices / put_strikes**2
    put_integral = np.trapz(put_integrand, put_strikes)
    # Call integral for K > F
    call_integrand = call_prices / call_strikes**2
    call_integral = np.trapz(call_integrand, call_strikes)
    K_var = 2/T * (r*T - np.log(F/S0) + put_integral + call_integral) * np.exp(r*T)
    return K_var

# -----------------------------
# Lookback call (floating strike) MC
# -----------------------------
def lookback_call_mc(S0, r, sigma, T, n_paths=100_000, n_steps=252):
    paths = gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    min_S = paths.min(axis=1)
    payoff = paths[:, -1] - min_S
    return np.exp(-r*T) * payoff.mean()

print(f"\nLookback call (floating strike): {lookback_call_mc(100, 0.05, 0.2, 1.0):.4f}")

# -----------------------------
# Discrete barrier continuity correction
# -----------------------------
def discrete_barrier_correction(B, sigma, dt):
    beta1 = 0.5826  # approximately -zeta(0.5)/sqrt(2pi)
    return B * np.exp(beta1 * sigma * np.sqrt(dt))

n_monitor = 12  # monthly
dt = 1/n_monitor
B_cont  = 90
B_discr = discrete_barrier_correction(B_cont, 0.2, dt)
print(f"\nContinuous barrier: {B_cont}, discrete (monthly): {B_discr:.4f}")
# Use B_discr in the continuous-barrier formula to get discrete-monitoring price
```

---

## 5.3.11 [QUANT APPLICATION]

1. **Structured retail products** (PPN, autocallables): combine bonds with digital and barrier options. Typical issue: cliquet or Asian payoffs with downside protection.

2. **Variance and volatility trading** is a major business. Variance swaps give pure exposure to realized volatility; VIX futures give forward-looking implied vol. Volatility arbitrage is a major hedge fund strategy.

3. **Dispersion trading**: long vol of components, short vol of index — exposure to implied correlation.

4. **Currency-quanto options**: pay in one currency but payoff referenced to another's asset. Pricing requires quanto-adjusted drift.

5. **FX barriers and one-touches** are the most liquid exotic options class; billions of notional trade daily. Tail-hedging by banks.

6. **Commodity Asian options** dominate in energy, metals (monthly average price contracts).

7. **Equity autocallables** (especially in Europe, Asia): path-dependent knock-out options with coupons. Complex hedging due to barrier and strike discontinuities.

8. **Mortgage-backed securities**: prepayment options are American-style exotics on rates; pricing with Monte Carlo + PDEs.

9. **Convertible bonds**: embedded call option + credit + equity. Priced as multi-factor PDE.

10. **Reinsurance / catastrophe bonds**: payoff triggered on exceedance of loss threshold — barrier-style in claim space.

---

## 5.3.12 Summary

- **Barriers**: closed forms via reflection principle; in-out parity; Broadie-Glasserman-Kou discrete correction.
- **Asians**: geometric closed form; arithmetic needs MC with control variate or moment matching.
- **Lookbacks**: joint distribution of $(\max, S_T)$ gives closed forms.
- **Variance swaps**: model-free replication via vanilla options — the most beautiful result.
- **Cliquets**: forward-starting option series, capped/floored.
- **Pricing methods**: MC (workhorse), PDE (low-dim Markov), Fourier (characteristic functions), trees, deep learning.

### Forward pointers

- **Module 5.4**: interest rate derivatives — caps/floors, swaptions, exotics on curves.
- **Module 5.6**: stochastic volatility — Heston, SABR, their variance-swap implications.
- **Module 9 (future)**: microstructure — how exotics get hedged and traded in real markets.

---

## Exercises

### Tier 1 (★)

1. Derive the reflection-principle formula for $\mathbb{P}(\max_{t \le T} B_t \ge a)$ for standard BM.
2. Price a down-and-out call with $S_0 = 100, K = 100, B = 85, r = 0.05, \sigma = 0.2, T = 1$.
3. Compute the geometric Asian price with parameters $S_0 = K = 100, r = 0.05, \sigma = 0.3, T = 1$.
4. Derive the digital call price $e^{-rT} N(d_2)$.
5. Compute the analytical vanna of a vanilla call.
6. Verify in-out parity numerically.
7. Show that a floating-strike lookback call satisfies $V_0 \ge S_0 - \mathbb{E}[\min_{t \le T} S_t] e^{-rT}$ strict inequality usually.
8. Derive the fair volatility strike for a variance swap with $r = 0$ under BS.

### Tier 2 (★★)

9. Derive the Merton (1973) closed-form DOC price via reflection.
10. Derive the variance swap replication formula from $d\ln S_t = \tfrac{dS}{S} - \tfrac{1}{2}\sigma^2 dt$.
11. Prove the Breeden-Litzenberger density identity.
12. Derive the Broadie-Glasserman-Kou barrier continuity correction.
13. Prove the in-out parity for general barrier options.
14. Derive the geometric Asian closed form in terms of effective parameters.
15. Prove that under BS, max-call - min-call = $S_0 e^{-rT} N(d_1)$ ... hmm let me think — derive an interesting relation between lookback max and min calls.
16. Derive the static replication of a zero-coupon digital via a tight vertical spread; compute the model risk.
17. Show that cliquet total cap reduces the vega significantly compared to uncapped.

### Tier 3 (★★★)

18. Implement Carr-Chou static replication of a down-and-out call.
19. Implement an arithmetic Asian option pricer using Laplace inversion (Geman-Yor).
20. Implement multi-asset MC with Cholesky factorization for a basket option; study convergence vs number of assets.
21. Derive the joint distribution of $(S_T, m_T, M_T)$ for GBM under $\mathbb{Q}$; use it to price double-barrier options.
22. Implement a Crank-Nicolson PDE solver for floating-strike Asian option using $(S, \int S dt)$ state.
23. Derive the VIX formula from first principles starting from the variance-swap replication.
24. Implement Longstaff-Schwartz for a Bermudan Asian option.
25. Prove and numerically demonstrate that local vol correctly prices European options but misprices forward-starting options.
26. Implement the COS method for call pricing under Merton jump-diffusion; compare with MC.
27. For a worst-of put on 3 assets with correlations calibrated from data, study the correlation smile: plot the implied correlation as a function of moneyness.
28. Implement multilevel Monte Carlo for an Asian option; compare cost vs standard MC.
29. Analyze the cliquet pricing puzzle of early-2000s: why did banks misprice using local vol? Demonstrate with a toy model and a stochastic-vol benchmark.

---

*Next module:* Interest rate models — short-rate models (Vasicek, CIR, Hull-White), HJM framework, and the LIBOR market model.

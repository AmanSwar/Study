# Subject 5, Module 4: Interest Rate Models

> *"Equities have one price per company. Rates have an entire curve. The mathematical machinery for rates has to handle an infinite-dimensional object evolving stochastically — and do so consistently across every tenor and every derivative."*

## Prerequisites

- **Module 3.4**: SDEs, Feynman-Kac, OU and CIR processes.
- **Modules 5.1-5.2**: FTAP, change of numéraire, forward measure.
- **Module 3.3**: Girsanov, martingale representation.

---

## 5.4.0 Why Rates are Different

**Stocks:** one stochastic state variable $S_t$, drift $\mu$, diffusion $\sigma$. Pricing via BS requires minimal apparatus.

**Rates:** the state is an entire *yield curve* $P(t, T)$ for all maturities $T > t$. This is an infinite-dimensional object evolving in time. Consistency between different tenors, arbitrage-free dynamics, hedging and calibration all become substantially more involved.

**Three families of models** historically:
1. **Short-rate models** (1970s-80s): specify SDE for the instantaneous rate $r_t$. Simple but can't fit curve shapes perfectly.
2. **HJM framework** (1992): specify dynamics of entire forward rate curve. Maximal flexibility; hard to implement.
3. **LIBOR market model (LMM)** (1997): specify dynamics of discretely-sampled forward rates. Tractable and market-consistent.

Roadmap:
- **5.4.1** Bond math and yield curve basics.
- **5.4.2** Vasicek model.
- **5.4.3** CIR model.
- **5.4.4** Hull-White / extended Vasicek.
- **5.4.5** HJM framework and drift condition.
- **5.4.6** LIBOR market model.
- **5.4.7** SABR model.
- **5.4.8** Calibration and smile.
- **5.4.9** Python and applications.

---

## 5.4.1 Bond Math and Yield Curve Basics

### Zero-coupon bond

$P(t, T)$: price at time $t$ of a contract paying $1 at $T$. $P(T, T) = 1$.

### Instantaneous short rate

$r_t := -\partial_T \ln P(t, T) |_{T = t}$.

Equivalently, $r_t$ is the rate of the risk-free money-market account: $\beta_t = \exp(\int_0^t r_s ds)$.

### Risk-neutral pricing

Fundamental theorem: $P(t, T) = \mathbb{E}_\mathbb{Q}\left[\exp\left(-\int_t^T r_s ds\right) \big| \mathcal{F}_t\right]$.

This is the master formula for all bond pricing.

### Forward rate

Instantaneous forward rate at $t$ for maturity $T$:
$$f(t, T) = -\partial_T \ln P(t, T).$$

So $P(t, T) = \exp(-\int_t^T f(t, u) du)$ and $f(t, t) = r_t$.

### Yield

Continuously-compounded yield: $y(t, T) = -\ln P(t, T)/(T - t)$.

### LIBOR / simple forward rate

For tenor $[T_1, T_2]$, the simple forward rate is
$$L(t, T_1, T_2) = \dfrac{1}{\Delta}\left(\dfrac{P(t, T_1)}{P(t, T_2)} - 1\right), \quad \Delta = T_2 - T_1.$$

At $t = T_1$: $L(T_1, T_1, T_2) = \tfrac{1}{\Delta}(1/P(T_1, T_2) - 1)$ is the $\Delta$-period LIBOR observed at $T_1$.

### Swap rate

Par swap rate $S_t(T_0, T_n)$: fixed-leg coupon that makes a vanilla fixed-for-floating swap zero-value. Formula:
$$S_t(T_0, T_n) = \dfrac{P(t, T_0) - P(t, T_n)}{\sum_{i=1}^n \Delta_i P(t, T_i)}.$$

### Caps and floors

Cap = portfolio of caplets. Each caplet pays $\Delta (L(T_{i-1}, T_{i-1}, T_i) - K)^+$ at $T_i$.

Black-76 convention: quote caplets by implied Black volatility, treating each caplet as BS-option on the forward LIBOR with $T_{i-1}$-forward measure.

### Swaption

European payer swaption: right to enter into a swap paying fixed $K$. Payoff at $T_0$:
$$\left(\sum_{i=1}^n \Delta_i P(T_0, T_i)\right) (S_{T_0}(T_0, T_n) - K)^+.$$

Black-76 quotes swaptions by implied vol on the swap rate.

---

## 5.4.2 Vasicek Model (1977)

### SDE under $\mathbb{Q}$

$$dr_t = \kappa(\theta - r_t) dt + \sigma \, dB^\mathbb{Q}_t.$$

Mean-reverting Ornstein-Uhlenbeck. $\kappa$ = speed of mean reversion, $\theta$ = long-run mean, $\sigma$ = volatility.

### Solution

$$r_t = r_s e^{-\kappa(t-s)} + \theta(1 - e^{-\kappa(t-s)}) + \sigma \int_s^t e^{-\kappa(t-u)} dB^\mathbb{Q}_u.$$

**Distribution:** Gaussian conditional on $r_s$.
- Mean: $r_s e^{-\kappa(t-s)} + \theta(1 - e^{-\kappa(t-s)})$.
- Variance: $\sigma^2 (1 - e^{-2\kappa(t-s)})/(2\kappa)$.

**Stationary distribution:** $\mathcal{N}(\theta, \sigma^2/(2\kappa))$.

### Bond pricing formula

Since $r$ is Gaussian and $\int_t^T r_s ds$ is a linear combination of Gaussians, it's Gaussian. Exponential expectation is computable.

**Affine term structure:**
$$P(t, T) = \exp(A(t, T) - B(t, T) r_t),$$
where
$$B(t, T) = \dfrac{1 - e^{-\kappa(T-t)}}{\kappa},$$
$$A(t, T) = (B(t, T) - (T-t))(\kappa^2 \theta - \sigma^2/2)/\kappa^2 - \sigma^2 B(t, T)^2/(4\kappa).$$

### Options on bonds / caps

Since $r$ is Gaussian and $P(t, T)$ is log-affine in $r$, everything is log-normal. Bond options have closed-form Black-Scholes-like formulas.

**Jamshidian (1989) decomposition:** coupon-bond options decompose into a portfolio of ZCB options → efficient swaption pricing in Vasicek/HW.

### Pros and cons

**Pros:** Gaussian, closed-form, fast.
**Cons:** rates can go negative (historically viewed as flaw; but since 2014 European / Japanese rates actually went negative — Vasicek got a second life). Only one factor → cannot fit the whole curve simultaneously.

---

## 5.4.3 CIR Model (Cox-Ingersoll-Ross 1985)

### SDE

$$dr_t = \kappa(\theta - r_t) dt + \sigma \sqrt{r_t} dB_t.$$

Square-root volatility → rates stay nonneg if $2\kappa\theta \ge \sigma^2$ (Feller condition).

### Distribution

Non-central chi-squared; explicit density.

### Bond pricing

Also affine term structure: $P(t, T) = \exp(A(t, T) - B(t, T) r_t)$ with $A, B$ from Riccati ODEs.

Explicit closed forms:
$$B(t, T) = \dfrac{2(e^{\gamma(T-t)} - 1)}{(\gamma + \kappa)(e^{\gamma(T-t)} - 1) + 2\gamma}, \quad \gamma = \sqrt{\kappa^2 + 2\sigma^2}.$$

### Bond options

Non-Gaussian but closed-form via non-central chi-squared CDF.

### Pros and cons

**Pros:** positive rates (classical appeal), affine, closed form.
**Cons:** only one factor; poor fit to observed caps/swaptions smile.

---

## 5.4.4 Hull-White / Extended Vasicek (1990)

### Setup

$$dr_t = (\theta(t) - \kappa r_t) dt + \sigma \, dB^\mathbb{Q}_t,$$
with time-dependent $\theta(t)$. This lets us fit the initial yield curve **exactly**.

### Calibration to initial curve

Given market forward curve $f^M(0, T)$, solve for $\theta(t)$:
$$\theta(t) = \partial_t f^M(0, t) + \kappa f^M(0, t) + \dfrac{\sigma^2}{2\kappa}(1 - e^{-2\kappa t}).$$

Now Hull-White matches the initial curve perfectly.

### Cap/swaption pricing

Same affine-term-structure analysis; closed forms for bond options; Jamshidian for swaptions.

### Two-factor extension

Often $\sigma$ and $\kappa$ alone can't fit the full curve + cap surface. Two-factor HW:
$$dr_t = (\theta(t) + u_t - \kappa_1 r_t) dt + \sigma_1 dB^1_t,$$
$$du_t = -\kappa_2 u_t dt + \sigma_2 dB^2_t,$$
with correlated Brownians. Still affine.

### Industry use

Hull-White remains the workhorse for counterparty risk / XVA and for bank-book IRR management due to tractability and Gaussian analytics (for American/exotic bond options).

---

## 5.4.5 HJM Framework (Heath-Jarrow-Morton 1992)

### Motivation

Short-rate models derive the whole curve from one state variable. This limits curve shapes. HJM inverts: **specify dynamics of the entire forward-rate curve directly**.

### HJM setup

For each $T$, evolve the forward rate $f(t, T)$:
$$df(t, T) = \alpha(t, T) dt + \sigma(t, T) dB_t,$$
with some drift $\alpha$ and vol $\sigma$ (possibly vector-valued and state-dependent).

### The HJM drift condition

Under $\mathbb{Q}$, the drift is **not free** — it's determined by the volatilities by no-arbitrage:
$$\boxed{\alpha(t, T) = \sigma(t, T) \int_t^T \sigma(t, u) du.}$$

**Proof sketch.** Bond $P(t, T) = \exp(-\int_t^T f(t, u) du)$. By Itô and no-arbitrage, discounted bond $P(t, T)/\beta_t$ must be $\mathbb{Q}$-martingale. Applying Itô + the Fubini theorem for stochastic integrals and matching drifts yields the constraint.

### Examples

- **Constant vol:** $\sigma(t, T) = \sigma$. Gaussian HJM; recovers Hull-White.
- **Exponential vol:** $\sigma(t, T) = \sigma e^{-\kappa(T-t)}$. Also Gaussian, equivalent to Hull-White.
- **Mercurio-Moraleda:** more flexible deterministic volatilities.
- **State-dependent:** $\sigma$ depends on $f$; gives non-Gaussian (e.g. CIR-like) HJM.

### Pros and cons

**Pros:** maximum flexibility; fits any initial curve and any vol structure.
**Cons:** in general non-Markov (infinite state variables); simulation-heavy; hard to price American / Bermudan.

### Markov HJM

Specific choices of $\sigma$ yield Markov HJM (reducible to finite-dim state). Ritchken-Sankarasubramanian (RS) criterion: $\sigma(t, T)$ is "separable" $\sigma(t, T) = g(t) h(T - t)$. Then evolve $(r_t, \phi_t)$ where $\phi$ is an integrated-variance state.

---

## 5.4.6 LIBOR Market Model (Brace-Gatarek-Musiela 1997)

### Motivation

Short-rate and HJM models work with continuous-time instantaneous rates. But markets quote and trade **discrete LIBOR rates** $L(t, T_i, T_{i+1})$ and use **Black-76** for caps. LMM directly models these to get market-consistent pricing.

### LMM setup

Discrete tenor structure $0 = T_0 < T_1 < \cdots < T_N$. Forward LIBOR rates:
$$L_i(t) := L(t, T_i, T_{i+1}).$$

Each $L_i(t)$ is a martingale under the $T_{i+1}$-forward measure $\mathbb{Q}^{i+1}$. Specify lognormal dynamics under this measure:
$$dL_i(t) = \sigma_i(t) L_i(t) \, dW^{i+1}_i(t),$$
where $W^{i+1}$ is Brownian under $\mathbb{Q}^{i+1}$.

**Under $\mathbb{Q}^{i+1}$:** $L_i$ is lognormal → caplet with expiry $T_i$ prices exactly as Black-76:
$$\text{Cpl}_i = \Delta P(0, T_{i+1}) [L_i(0) N(d_1) - K N(d_2)], \quad d_1 = \dfrac{\ln(L_i(0)/K) + \tfrac{1}{2}\bar\sigma^2_i T_i}{\bar\sigma_i \sqrt{T_i}},$$
where $\bar\sigma^2_i T_i = \int_0^{T_i} \sigma_i(t)^2 dt$.

### Drift conversion between forward measures

For simulation of multiple $L_i$, need common measure. Under the **spot measure** (rolling numéraire), $L_i(t)$ has drift
$$\mu_i(t) = -\sum_{j = q(t)}^i \dfrac{\Delta L_j(t) \rho_{ij} \sigma_j(t)}{1 + \Delta L_j(t)} \sigma_i(t),$$
where $q(t) = \min\{j : T_j > t\}$ is the next reset index.

**The "BGM drift terms."**

Under the **terminal measure** $\mathbb{Q}^N$:
$$\mu_i(t) = \sum_{j = i+1}^{N-1} \dfrac{\Delta L_j(t) \rho_{ij} \sigma_j(t) \sigma_i(t)}{1 + \Delta L_j(t)}.$$

### Implementation

Given cap vol surface $\sigma^{\text{cap}}_i$ (one implied vol per caplet) and cap-based calibration plus correlation matrix $\rho$, simulate forward rates on a grid.

**Monte Carlo complexity:** $O(N^2 \cdot n_{\text{paths}})$ per time step due to drift sums.

### Pros and cons

**Pros:** directly models market-quoted rates; caps and floors are exactly log-normal (matching Black-76); swaption approximation reasonable.
**Cons:** high-dimensional state (one per tenor); no exact swaption formula; harder calibration than short-rate; drift corrections are a significant numerical burden.

### Swap Market Model (SMM)

Alternative: model swap rates directly (Jamshidian 1997). Each $S_i$ is lognormal under the appropriate annuity measure. Swaptions exactly Black-76; caps approximate.

SMM and LMM are not both exactly compatible (can't be simultaneously lognormal in both rates). Pick one, approximate the other.

---

## 5.4.7 SABR Model (Hagan-Kumar-Lesniewski-Woodward 2002)

### SDE

For a single forward rate (or price):
$$dF_t = \alpha_t F_t^\beta dB^1_t,$$
$$d\alpha_t = \nu \alpha_t dB^2_t,$$
$$\mathbb{E}[dB^1 dB^2] = \rho dt.$$

Parameters: $\alpha_0$ (initial vol), $\beta \in [0,1]$ (CEV exponent), $\nu$ (vol-of-vol), $\rho$ (correlation).

### The SABR implied vol approximation

Hagan et al.'s famous approximate formula for Black implied vol $\sigma_{\text{imp}}(K, F_0)$:
$$\sigma_{\text{imp}}(K, F_0) \approx \dfrac{\alpha_0}{(F_0 K)^{(1-\beta)/2} \mathcal{Z}(K, F_0)} \cdot \dfrac{z}{\chi(z)} \cdot [1 + \text{higher-order corrections}],$$
where $z = (\nu/\alpha_0)(F_0 K)^{(1-\beta)/2} \ln(F_0/K)$, $\chi(z) = \ln((\sqrt{1 - 2\rho z + z^2} + z - \rho)/(1 - \rho))$, and $\mathcal{Z}(K, F_0)$ is a moneyness correction.

### Calibration

Given market implied vols for various strikes at a fixed expiry, fit $(\alpha_0, \beta, \nu, \rho)$ by minimizing squared error. Typically $\beta$ is fixed by convention ($\beta = 0$ for normal SABR, $\beta = 0.5$ for CIR-like, $\beta = 1$ for lognormal).

### Pros and cons

**Pros:** fits smile shape well; industry standard for swaption and cap pricing; fast due to closed-form approximation.
**Cons:** approximation breaks down for low strikes near zero (nominal "arbitrage" issues near $F = 0$); arbitrage-free version (Hagan 2014) is more complex.

### No-arbitrage SABR

Andreasen-Huge "ZABR" variant and Hagan-Kumar-Lesniewski-Woodward's refined approximation address arbitrage violations near zero strikes.

---

## 5.4.8 Calibration and the Rate Smile

### Market data

Daily market quotes include:
- Zero curve (from OIS, treasury, or swap rates).
- Caps/floors (vol cube: strike × maturity × tenor).
- Swaptions (swaption cube: strike × option expiry × swap tenor).

### Calibration objective

Minimize weighted squared error between model and market implied vols across the cube:
$$\min_\theta \sum_{i} w_i (\sigma^{\text{model}}(\theta, K_i, T_i, \tau_i) - \sigma^{\text{market}}_i)^2.$$

### Rate smile

Like equities, rates exhibit implied-vol smile/skew:
- Negative skew for low strikes (more implied vol for low rates).
- Smile steeper at short maturities.

### Typical calibration workflow

1. Fit yield curve using bootstrapping from liquid instruments.
2. Fit SABR per expiry / tenor; get $(\alpha, \beta, \nu, \rho)$ surface.
3. Use SABR as local parameterization; price exotics with MC or PDE referencing SABR.

Industry software: Numerix, OpenGamma, QuantLib (open source).

---

## 5.4.9 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ncx2
from scipy.integrate import quad

# -----------------------------
# Vasicek bond price and simulation
# -----------------------------
def vasicek_bond_price(r0, kappa, theta, sigma, t, T):
    B = (1 - np.exp(-kappa*(T-t))) / kappa
    A = (theta - sigma**2/(2*kappa**2)) * (B - (T-t)) - sigma**2 * B**2 / (4*kappa)
    return np.exp(A - B*r0)

def simulate_vasicek(r0, kappa, theta, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    r = np.full(n_paths, r0)
    r_path = [r.copy()]
    for _ in range(n_steps):
        dB = np.sqrt(dt) * np.random.randn(n_paths)
        r = r + kappa*(theta - r)*dt + sigma*dB
        r_path.append(r.copy())
    return np.array(r_path).T

r0, kappa, theta, sigma = 0.03, 0.5, 0.05, 0.02
P_calc = vasicek_bond_price(r0, kappa, theta, sigma, 0, 5)
print(f"Vasicek P(0, 5) = {P_calc:.6f}")

# MC check
paths = simulate_vasicek(r0, kappa, theta, sigma, 5, 500, 10_000)
integral_r = np.trapz(paths, dx=5/500, axis=1)
P_mc = np.mean(np.exp(-integral_r))
print(f"Vasicek MC = {P_mc:.6f}")

# -----------------------------
# Yield curve from Vasicek
# -----------------------------
T_grid = np.linspace(0.25, 30, 100)
yields = [-np.log(vasicek_bond_price(r0, kappa, theta, sigma, 0, T))/T for T in T_grid]
plt.figure(figsize=(9, 4))
plt.plot(T_grid, np.array(yields)*100)
plt.xlabel('Maturity T'); plt.ylabel('Zero yield (%)')
plt.title('Vasicek yield curve')
plt.grid(alpha=0.3); plt.show()

# -----------------------------
# CIR simulation (with Feller condition check)
# -----------------------------
def simulate_cir(r0, kappa, theta, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    r = np.full(n_paths, r0)
    r_path = [r.copy()]
    for _ in range(n_steps):
        dB = np.sqrt(dt) * np.random.randn(n_paths)
        r = np.maximum(r + kappa*(theta - r)*dt + sigma*np.sqrt(np.maximum(r, 0))*dB, 0)
        r_path.append(r.copy())
    return np.array(r_path).T

# -----------------------------
# Hull-White calibration example
# -----------------------------
def hw_theta(t, a, sigma, f_M, f_M_deriv):
    return f_M_deriv(t) + a*f_M(t) + (sigma**2/(2*a))*(1 - np.exp(-2*a*t))

# -----------------------------
# SABR implied vol formula
# -----------------------------
def sabr_implied_vol(F, K, T, alpha, beta, nu, rho):
    if abs(F - K) < 1e-12:
        # ATM formula
        FK_beta = F**(1-beta)
        term1 = alpha / FK_beta
        term2 = 1 + (((1-beta)**2 * alpha**2)/(24 * F**(2-2*beta))
                     + (rho*beta*nu*alpha)/(4 * FK_beta)
                     + ((2 - 3*rho**2) * nu**2)/24) * T
        return term1 * term2
    logFK = np.log(F/K)
    FK_beta_half = (F*K)**((1-beta)/2)
    z = (nu / alpha) * FK_beta_half * logFK
    x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho)/(1 - rho))
    numer = alpha / (FK_beta_half * (1 + ((1-beta)**2/24)*logFK**2 + ((1-beta)**4/1920)*logFK**4))
    factor = z / x_z if abs(z) > 1e-12 else 1
    correction = 1 + (((1-beta)**2 * alpha**2)/(24 * (F*K)**(1-beta))
                      + (rho*beta*nu*alpha)/(4 * FK_beta_half)
                      + ((2 - 3*rho**2) * nu**2)/24) * T
    return numer * factor * correction

# Plot SABR smile
F = 0.03; T = 1.0
alpha, beta, nu, rho = 0.01, 0.5, 0.4, -0.3
strikes = np.linspace(0.01, 0.08, 50)
vols = [sabr_implied_vol(F, K, T, alpha, beta, nu, rho) for K in strikes]
plt.figure(figsize=(9, 4))
plt.plot(strikes*100, np.array(vols)*100, label='SABR smile')
plt.xlabel('Strike (%)'); plt.ylabel('Black vol (%)')
plt.title('SABR implied volatility smile')
plt.legend(); plt.grid(alpha=0.3); plt.show()

# -----------------------------
# LMM-lite simulation (single caplet)
# -----------------------------
def lmm_caplet(L0, sigma, T_start, T_end, K, n_paths=50_000):
    # Single forward rate, no drift under its own forward measure
    dt = T_start
    Z = np.random.randn(n_paths)
    L_T = L0 * np.exp(-0.5*sigma**2*T_start + sigma*np.sqrt(T_start)*Z)
    payoff = (T_end - T_start) * np.maximum(L_T - K, 0)
    # Discount with P(0, T_end); user inputs
    return payoff  # expectation × P(0, T_end)

# Compare with Black-76
def black76_caplet(L0, K, T_start, sigma, delta, P_end):
    d1 = (np.log(L0/K) + 0.5*sigma**2*T_start) / (sigma*np.sqrt(T_start))
    d2 = d1 - sigma*np.sqrt(T_start)
    return delta * P_end * (L0*norm.cdf(d1) - K*norm.cdf(d2))

L0, sigma, T, K = 0.04, 0.3, 1.0, 0.04
payoff_sim = lmm_caplet(L0, sigma, T, 1.25, K)
P_end = 0.96
analytical = black76_caplet(L0, K, T, sigma, 0.25, P_end)
simulated  = P_end * np.mean(payoff_sim)
print(f"\nCaplet Black76 = {analytical:.6f}, simulation = {simulated:.6f}")
```

---

## 5.4.10 [QUANT APPLICATION]

1. **Vanilla rates derivatives** (swaps, FRAs, caps, floors, swaptions): all priced within Black-76 or SABR framework. Daily volumes in the trillions.

2. **Bermudan swaptions and callable bonds**: priced via tree or Longstaff-Schwartz within Hull-White or LMM.

3. **Constant-maturity swaps (CMS) and CMS options**: depend on specific tenor of swap rate; complex convexity adjustments.

4. **Credit Valuation Adjustment (CVA)**: risk-neutral expected loss from counterparty default, simulated under HW / LMM joint with credit models.

5. **Structured notes**: range accruals, target redemption notes (TARNs), callable step-ups — all rates exotics priced with LMM.

6. **Mortgage-backed securities and prepayments**: prepayment models driven by rate dynamics + borrower behavior; HW + intensity models.

7. **Insurance liabilities / ALM**: annuities, variable annuities with GMIB / GMAB riders — priced under HW or LMM for hedging.

8. **Central bank and monetary policy**: short-rate models used to forecast yield curves under policy scenarios.

9. **Bank book risk management**: deposits, loans with embedded options, priced with HW.

10. **Inflation-linked products**: real-rate analog of nominal models (Jarrow-Yildirim 2003 framework).

---

## 5.4.11 Summary

- **Short-rate models** (Vasicek, CIR, Hull-White): closed form, tractable, but limited curve flexibility.
- **HJM**: models entire forward curve; drift condition $\alpha = \sigma \int \sigma$; infinite-dim unless Markov.
- **LIBOR market model**: discrete forward rates, each lognormal under its own forward measure; drift corrections under common measure; market-consistent for caps.
- **SABR**: stochastic-vol extension; implied-vol formula; industry standard for smile.

### Forward pointers

- **Module 5.5**: credit risk models — extension of rate modeling to default intensities and structural firm-value dynamics.
- **Module 5.6**: stochastic volatility — Heston for equities; deeper dive into SABR-like dynamics.

---

## Exercises

### Tier 1 (★)

1. Compute $P(0, T)$ under Vasicek for $r_0 = 0.03, \kappa = 0.4, \theta = 0.05, \sigma = 0.015, T = 5$.
2. Compute the yield curve for Ex 1 at $T = 1, 2, 5, 10, 30$.
3. Verify Feller condition for CIR with $\kappa = 0.5, \theta = 0.04, \sigma = 0.1$.
4. Derive the HJM drift condition explicitly.
5. Compute the SABR ATM volatility for $\alpha = 0.01, \beta = 0.5, \nu = 0.3, \rho = -0.2, F = 0.03, T = 1$.
6. Derive the Black-76 caplet formula from LMM.
7. Calibrate a flat-yield curve to observed bond prices.

### Tier 2 (★★)

8. Derive the Vasicek affine term structure from the PDE.
9. Derive the bond-option formula in Vasicek via Jamshidian.
10. Prove the Ritchken-Sankarasubramanian Markov HJM criterion.
11. Derive the drift of $L_i$ under the spot measure in LMM.
12. Calibrate SABR to a stylized caplet smile and assess fit.
13. Show that HJM with exponential vol $\sigma(t, T) = \sigma e^{-\kappa(T-t)}$ recovers Hull-White short-rate dynamics.
14. Derive the LMM swap-rate approximate volatility (Rebonato formula).
15. Compute CMS-swap convexity adjustment under Vasicek.

### Tier 3 (★★★)

16. Implement full LMM calibration + MC for a Bermudan swaption.
17. Implement the Hagan SABR with no-arbitrage correction (SABR-HW 2014).
18. Implement Jarrow-Yildirim inflation model (nominal + real Vasicek-like) and price an inflation cap.
19. Prove the equivalence of HJM with deterministic vol and generalized Hull-White.
20. Implement a Markov-functional model (Hunt-Kennedy-Pelsser) for swaption pricing.
21. Derive and compute the CVA on an interest-rate swap, using HW for rates + intensity model for credit.
22. Implement the displaced-diffusion LMM as an alternative to SABR.
23. Explore the effect of negative rates: Vasicek vs CIR vs shifted-CIR for a modern Bund curve.
24. Implement Stein-Stein or Dufresne-style models that address CIR's poor smile fit.

---

*Next module:* Credit risk modeling — structural (Merton) and reduced-form (Duffie-Singleton) approaches, CDS, CDOs.

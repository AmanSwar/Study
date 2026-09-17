# Subject 5, Module 5: Credit Risk Modeling

> *"Equity tells you what a firm is worth if it survives. Credit tells you what its debt is worth given the possibility it won't. And a CDS is a contract that turns a default probability directly into a traded price."*

## Prerequisites

- **Modules 5.1-5.2**: FTAP, Black-Scholes.
- **Module 5.4**: Interest-rate models.
- **Module 3.5**: Lévy / jump processes (for modeling default arrivals).
- **Module 3.6**: Markov processes and intensities.
- Helpful: basic fixed-income concepts.

---

## 5.5.0 Two Paradigms

**Structural models** (Merton 1974, KMV): treat default as a consequence of firm-value dynamics. A company defaults when its assets fall below its liabilities. Equity is a call option on firm value; debt is a bond minus a put.

**Reduced-form models** (Jarrow-Turnbull 1995, Duffie-Singleton 1999, Lando 1998): treat default as an exogenous Poisson-like event with stochastic intensity $\lambda_t$. Default time is the first jump of a Cox process.

**In practice**, both are used: structural for fundamental analysis and corporate finance, reduced-form for pricing and calibration to market spreads (CDS).

Roadmap:
- **5.5.1** Merton's structural model.
- **5.5.2** Extensions: first-passage, KMV, Black-Cox.
- **5.5.3** Reduced-form intensity models.
- **5.5.4** Cox processes and doubly-stochastic default.
- **5.5.5** CDS pricing and bootstrapping.
- **5.5.6** Copula-based portfolio credit models and CDOs.
- **5.5.7** Python and quant applications.

---

## 5.5.1 Merton's Structural Model (1974)

### Setup

Firm with total asset value $V_t$ following GBM:
$$dV_t = \mu_V V_t \, dt + \sigma_V V_t \, dB_t.$$

Capital structure: single zero-coupon debt with face value $D$ maturing at $T$, plus equity $E_t = V_t - D_t$.

At $T$:
- If $V_T \ge D$: bondholders receive $D$, shareholders receive $V_T - D$.
- If $V_T < D$: bondholders receive $V_T$, shareholders receive $0$.

Default: $\{V_T < D\}$.

### Equity as call option

$$E_T = (V_T - D)^+.$$

So equity is a European call on the firm's assets with strike $D$ and expiry $T$. Under $\mathbb{Q}$:
$$E_0 = V_0 N(d_1) - D e^{-rT} N(d_2),$$
$$d_1 = \dfrac{\ln(V_0/D) + (r + \tfrac{1}{2}\sigma_V^2)T}{\sigma_V \sqrt T}, \quad d_2 = d_1 - \sigma_V\sqrt T.$$

### Debt as bond minus put

Bondholders: $D_T = \min(V_T, D) = D - (D - V_T)^+$.

So debt = riskless bond - put on firm value:
$$D_0 = D e^{-rT} - P_0 = D e^{-rT} - [D e^{-rT} N(-d_2) - V_0 N(-d_1)]$$
$$= V_0 N(-d_1) + D e^{-rT} N(d_2).$$

### Credit spread

Define the yield of the risky bond: $y = -\ln(D_0 / D)/T$. The credit spread is $y - r$.

For short maturities and moderate asset-vol, we can expand and find:
$$y - r \approx \dfrac{\sigma_V^2 T}{2} \cdot \dfrac{N(-d_2)}{...}$$

In particular, **for very short $T$, the spread goes to zero** in the Merton model (given $V_0 > D$, default probability vanishes as $T \to 0^+$). This is a famous *counter-empirical* feature — real short-dated credit spreads are always positive due to jump risk (Merton's model misses jump-to-default).

### Probability of default

Under $\mathbb{Q}$: $\mathbb{P}_\mathbb{Q}(\text{default}) = N(-d_2)$.
Under $\mathbb{P}$: $\mathbb{P}(\text{default}) = N\left(-\dfrac{\ln(V_0/D) + (\mu_V - \tfrac{1}{2}\sigma_V^2)T}{\sigma_V\sqrt T}\right)$.

These differ — physical default probabilities are lower than risk-neutral ones (positive risk premium).

### Distance to default

$DD = d_2$ under $\mathbb{P}$ is the **distance to default**: number of standard deviations of asset value drift needed to reach default. Key KMV metric.

### Limitations

- Can't fit observed credit spread term structures.
- Only single default event at $T$; no early default.
- Asset value not directly observable (implied from equity via iterative calibration).

---

## 5.5.2 First-Passage and KMV Extensions

### Black-Cox (1976) first-passage model

Default at *first time* $V$ hits a barrier $B(t) \le D$ (not just at maturity):
$$\tau = \inf\{t : V_t \le B(t)\}.$$

For exponential barrier $B(t) = B_0 e^{\gamma t}$: closed form via reflection principle.

Allows defaults before $T$; matches empirical data better.

### KMV (Moody's) model

Extended Merton with:
- **DD (Distance-to-Default)** = $(V - D_{\text{default-point}})/(V \sigma_V)$ — proprietary default point (short + half long-term debt).
- Empirical mapping from DD to **Expected Default Frequency (EDF)** — not just normal CDF, but calibrated to historical defaults.

The empirical EDF calibration is key: for a given DD, KMV looks up the historical default frequency over similar firms, bypassing the Gaussian assumption.

### Leland-Toft (1996)

Endogenous default barrier determined by equity-holder optimization (coupons, taxes, bankruptcy costs). Links credit to optimal capital structure.

### Structural models for sovereigns

Can apply similar logic to sovereign debt: GDP as "firm value," external debt as "liabilities." Trickier because government can choose to default strategically (Eaton-Gersovitz).

---

## 5.5.3 Reduced-Form Intensity Models

### Setup

Default time $\tau$ modeled as first jump of a point process with stochastic intensity $\lambda_t$:
$$\mathbb{P}(\tau > t | \mathcal{F}_t) = \exp\left(-\int_0^t \lambda_s ds\right).$$

"Cox process" = doubly-stochastic Poisson (intensity is itself random).

### Conditional survival probability

Given information $\mathcal{G}_t$ (including $\{\tau > t\}$) but not the full intensity history:
$$S(t, T) := \mathbb{P}(\tau > T | \tau > t, \mathcal{F}_t) = \mathbb{E}\left[\exp\left(-\int_t^T \lambda_s ds\right) \big| \mathcal{F}_t\right].$$

### Defaultable zero-coupon bond

Assume zero recovery: bondholder gets $1 \mathbf{1}_{\tau > T}$. Price:
$$P^d(t, T) = \mathbb{E}_\mathbb{Q}\left[e^{-\int_t^T r_s ds} \mathbf{1}_{\tau > T} | \mathcal{F}_t \right] = \mathbb{E}_\mathbb{Q}\left[e^{-\int_t^T (r_s + \lambda_s) ds} | \mathcal{F}_t\right].$$

This is just like a riskless bond but with "effective rate" $r + \lambda$. Crucially:
- If $r_t, \lambda_t$ are modeled as affine (Vasicek, CIR), $P^d$ has affine closed form.
- Credit spread = $\lambda$ if deterministic (in risk-neutral), with corrections for rate-intensity correlation.

### Recovery

Recovery options:
- **Recovery of face value (RFV):** bondholder receives $R$ at $\tau$.
- **Recovery of treasury (RT):** bondholder receives $R P(\tau, T)$ at $\tau$.
- **Recovery of market value (RMV):** bondholder receives $R P^d(\tau^-, T)$ at $\tau$.

RMV leads to a clean "effective rate" form:
$$P^d(t, T) = \mathbb{E}_\mathbb{Q}\left[e^{-\int_t^T (r_s + (1-R)\lambda_s) ds}\right].$$

Thus the effective spread is $(1-R)\lambda$ — the loss-given-default times hazard.

### Poisson, CIR, Vasicek intensities

- **Constant $\lambda$:** exponential default time; $S(t, T) = e^{-\lambda(T-t)}$.
- **CIR $\lambda$:** $d\lambda = \kappa(\theta - \lambda) dt + \sigma\sqrt\lambda dB$; nonneg; affine closed form for survival.
- **Vasicek $\lambda$:** can go negative — unphysical for default intensity (suggests "un-default"). Avoided in practice.

### Jump-intensity models

Add jumps to intensity: $d\lambda = \kappa(\theta - \lambda)dt + \sigma dB + dJ$, for jump process $J$. Captures sudden news (rating downgrades).

### Contagion

$\lambda_t$ can jump in response to other firms' defaults (Davis-Lo, Jarrow-Yu). Models systemic risk.

---

## 5.5.4 Cox Process Framework

### Definition

A **Cox process** has stochastic intensity $\lambda_t$ adapted to a larger filtration $\mathbb{F}$, conditional on which the default time is a Poisson process.

### Cox-Markov formulation

Intensity process $\lambda_t = \lambda(X_t)$ for Markov state $X$. Then survival:
$$S(t, T) = \mathbb{E}[\exp(-\int_t^T \lambda(X_s) ds) | X_t = x] = u(t, T, x),$$
which satisfies Feynman-Kac:
$$\partial_t u + \mathcal{L} u = \lambda(x) u, \quad u(T, T, x) = 1.$$

### Example: CIR intensity

$d\lambda = \kappa(\theta - \lambda) dt + \sigma\sqrt\lambda dB$ → affine → closed form:
$$S(t, T) = A(t, T) e^{-B(t, T) \lambda_t}$$
with $A, B$ from Riccati ODEs.

### Change of measure under default

Under $\mathbb{Q}$, the default intensity might differ from physical $\lambda^\mathbb{P}$: risk premium for default risk. Typically $\lambda^\mathbb{Q} > \lambda^\mathbb{P}$ (investors demand extra yield beyond pure expected loss).

Ratio $\lambda^\mathbb{Q}/\lambda^\mathbb{P}$ ~ 2-10× historically for investment-grade US corporate bonds — the "credit spread puzzle" analogous to the equity premium puzzle.

---

## 5.5.5 Credit Default Swaps (CDS)

### Structure

Protection buyer pays quarterly premium (coupons) to protection seller. In return:
- If default occurs before maturity: seller pays $(1 - R) \cdot $ notional; swap terminates.
- No default: seller receives coupons to maturity.

### Fair spread

Spread $S_{\text{CDS}}$ is chosen so PV(premium leg) = PV(protection leg).

**Premium leg:**
$$\text{PV}_{\text{prem}} = S_{\text{CDS}} \sum_i \Delta_i \cdot \mathbb{E}[e^{-\int_0^{T_i} r ds} \mathbf{1}_{\tau > T_i}] + \text{accrual on default}.$$

**Protection leg:**
$$\text{PV}_{\text{prot}} = (1 - R) \mathbb{E}[e^{-\int_0^\tau r ds} \mathbf{1}_{\tau \le T}].$$

Under assumption of $r$ and $\lambda$ independent (and RMV-like recovery):
$$S_{\text{CDS}} \approx (1 - R) \cdot \lambda,$$
giving a direct mapping from spreads to implied (risk-neutral) default intensity.

More precisely, for piecewise-constant $\lambda$, one *bootstraps* the intensity from a series of CDS spreads at different maturities.

### Bootstrapping

Given market CDS spreads $S_1, S_2, S_3, \ldots$ for 1Y, 2Y, 5Y, 10Y:

1. Fit $\lambda_1$ (intensity from 0 to 1Y) so 1Y CDS prices correctly.
2. Fit $\lambda_2$ (intensity 1Y-2Y) so 2Y CDS prices correctly given $\lambda_1$.
3. Continue for each maturity.

Result: piecewise-constant term structure of hazard rates. Used for marking-to-model all other credit instruments on the same name.

### CDS index products

- **CDX.IG**: 125 investment-grade North American names.
- **iTraxx Europe**: 125 European names.
- **iTraxx XO**: 75 crossover (riskier) names.

These trade as single instruments with fixed coupons; the "index spread" is the weighted average of underlying single-name CDS spreads. Liquid instruments for market-implied aggregate credit conditions.

---

## 5.5.6 Portfolio Credit Models and CDOs

### The portfolio problem

Price products on baskets of defaultable names: CDO tranches, n-th to default swaps, basket credit default swaps.

**Key challenge**: default correlation. Individual default probabilities can be calibrated from CDS spreads, but joint default distribution requires additional modeling.

### Gaussian copula (Li 2000)

Latent variable model. Each name $i$ has a latent variable
$$X_i = \rho_i M + \sqrt{1 - \rho_i^2} \epsilon_i,$$
with $M, \epsilon_i \sim \mathcal{N}(0, 1)$ independent. Default occurs when $X_i < \Phi^{-1}(p_i)$ for marginal default probability $p_i$.

The common factor $M$ drives "systemic" default. $\rho_i$ is "asset correlation"; pair correlation is $\rho_i \rho_j$.

### CDO tranche pricing

CDO tranche with attachment $K_1$ and detachment $K_2$ pays losses in portfolio notional between those levels. Tranche loss:
$$L^{[K_1, K_2]}_t = \min((L_t - K_1)^+, K_2 - K_1),$$
where $L_t = \sum_i (1 - R_i) \mathbf{1}_{\tau_i \le t}$.

Under Gaussian copula: conditional on $M$, defaults are independent with probabilities
$$p_i(M) = \Phi\left(\dfrac{\Phi^{-1}(p_i) - \rho_i M}{\sqrt{1 - \rho_i^2}}\right).$$

Use FFT / recursion to compute loss distribution conditional on $M$, then integrate over $M$ (one-factor conditional-independence framework).

### Implied correlation

Like implied vol: back out $\rho$ that matches traded tranche prices. Result: "correlation skew" — different tranches imply different $\rho$.

### Post-2008 criticism

Gaussian copula failed spectacularly in 2007-2008: implied a structure where senior tranches rarely defaulted (correct under Gaussian), but reality showed correlated mass defaults. Li's model became blamed for the crisis.

### Alternatives

- **Student-$t$ copula:** fatter tails; better extreme-dependence.
- **Random-factor loadings:** $\rho_i$ stochastic.
- **Marshall-Olkin:** common Poisson shocks causing simultaneous defaults.
- **Structural contagion:** firm-value jumps upon defaults.
- **Loss distribution models:** specify loss distribution directly (top-down).

### n-th to default and first-to-default

Pay protection on *first* (or $n$-th) name to default among a basket. Correlation-sensitive: first-to-default is *decreasing* in correlation, $n$-th to default for large $n$ is *increasing* in correlation.

---

## 5.5.7 Python Implementation

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# -----------------------------
# Merton structural model
# -----------------------------
def merton_credit(V0, D, r, sigma_V, T):
    d1 = (np.log(V0/D) + (r + sigma_V**2/2)*T) / (sigma_V*np.sqrt(T))
    d2 = d1 - sigma_V*np.sqrt(T)
    E0 = V0*norm.cdf(d1) - D*np.exp(-r*T)*norm.cdf(d2)
    D0 = V0*norm.cdf(-d1) + D*np.exp(-r*T)*norm.cdf(d2)
    PD = norm.cdf(-d2)  # risk-neutral default prob
    spread = -np.log(D0/D)/T - r
    return {'E0': E0, 'D0': D0, 'PD_Q': PD, 'spread': spread, 'DD': d2}

# Example: firm with V=100, debt D=80 maturing in 5Y, r=3%, asset vol 20%
res = merton_credit(100, 80, 0.03, 0.20, 5)
print(f"Merton: Equity={res['E0']:.4f}, Debt={res['D0']:.4f}")
print(f"        PD_Q={res['PD_Q']:.4f}, DD={res['DD']:.4f}, Spread={res['spread']*10000:.2f} bps")

# Spread vs maturity
T_grid = np.linspace(0.1, 20, 100)
spreads = [merton_credit(100, 80, 0.03, 0.20, T)['spread']*10000 for T in T_grid]
plt.figure(figsize=(9, 4))
plt.plot(T_grid, spreads)
plt.xlabel('Maturity (years)'); plt.ylabel('Credit spread (bps)')
plt.title('Merton credit spread term structure')
plt.grid(alpha=0.3); plt.show()

# Note: spread -> 0 as T -> 0 (known flaw)

# -----------------------------
# Reduced-form: constant intensity
# -----------------------------
def cds_fair_spread_constant(lam, R, T, n_payments, r=0.03):
    dt = T / n_payments
    # Premium leg DV01
    DV01 = sum(dt * np.exp(-(r+lam)*i*dt) for i in range(1, n_payments+1))
    # Protection leg
    prot = (1 - R) * (1 - np.exp(-(r+lam)*T)) * lam/(r+lam)
    return prot / DV01

for lam in [0.005, 0.01, 0.02, 0.05]:
    s = cds_fair_spread_constant(lam, 0.40, 5, 20)
    print(f"Lambda={lam:.3f}: 5Y CDS = {s*10000:.2f} bps (R=40%)")

# -----------------------------
# Bootstrapping hazard rates from CDS spreads
# -----------------------------
def bootstrap_hazard(spreads, maturities, R=0.4, r=0.03):
    """Given CDS spreads at several maturities, solve for piecewise-constant hazard."""
    lambdas = []
    prev_T = 0
    prev_lambdas = []
    for spread, T in zip(spreads, maturities):
        # Find lam s.t. computed CDS equals observed spread, with piecewise-constant
        def cds_price(lam_new):
            # Accumulate premium and protection legs
            # (Simplified: only from prev_T to T with lam_new; before that with prev_lambdas)
            # For brevity, use constant in whole interval
            return cds_fair_spread_constant(lam_new, R, T, 4*T, r=r)
        from scipy.optimize import brentq
        lam_new = brentq(lambda l: cds_price(l) - spread, 1e-6, 1.0)
        lambdas.append(lam_new)
        prev_T = T
        prev_lambdas.append(lam_new)
    return lambdas

spreads_market = [0.005, 0.008, 0.012, 0.020]   # 1Y, 3Y, 5Y, 10Y CDS
maturities     = [1, 3, 5, 10]
# lam_pw = bootstrap_hazard(spreads_market, maturities)

# -----------------------------
# Gaussian copula CDO tranche simulation
# -----------------------------
def gaussian_copula_cdo(p_default, rho, n_paths=100_000, tranche=(0.03, 0.07)):
    n = len(p_default)
    results = []
    for _ in range(n_paths):
        M = np.random.randn()
        eps = np.random.randn(n)
        X = np.sqrt(rho)*M + np.sqrt(1-rho)*eps
        defaults = X < norm.ppf(p_default)
        loss_per_name = 0.6 / n
        total_loss = np.sum(defaults) * loss_per_name  # assume LGD=60%
        tranche_loss = np.clip(total_loss - tranche[0], 0, tranche[1] - tranche[0])
        results.append(tranche_loss)
    return np.mean(results), np.std(results)/np.sqrt(n_paths)

# CDX 5Y: 125 names, uniform PD
p_d = np.full(125, 0.04)
for rho in [0.0, 0.2, 0.5, 0.9]:
    mean, se = gaussian_copula_cdo(p_d, rho)
    print(f"Rho={rho}: equity tranche (0-3%) loss = {mean:.4f}, SE={se:.5f}")

# -----------------------------
# Loss distribution via recursion
# -----------------------------
def loss_dist_recursion(p_default_cond, n_paths=1):
    """Given conditional default probabilities, compute loss distribution."""
    n = len(p_default_cond)
    # Probability of 0, 1, ..., n defaults
    dist = np.zeros(n+1)
    dist[0] = 1.0
    for i in range(n):
        p = p_default_cond[i]
        new_dist = np.zeros(n+1)
        for k in range(n+1):
            new_dist[k] += dist[k] * (1 - p)
            if k < n:
                new_dist[k+1] += dist[k] * p
        dist = new_dist
    return dist

# ...
```

---

## 5.5.8 [QUANT APPLICATION]

1. **Corporate bond pricing and trading** — all banks mark credit spreads daily; Merton-KMV for fundamental analysis, intensity models for mark-to-market.

2. **CDS market** — daily notional in hundreds of billions; essential for bank credit-trading desks. Calibrated intensity models price exotics.

3. **Structured credit (CDO / CLO)** — multi-trillion market; Gaussian copula baseline + refinements.

4. **XVA: CVA, DVA, FVA, MVA** — counterparty value adjustments use stochastic credit + rates models. Every bank has a "XVA desk" computing these for regulatory capital + PnL.

5. **Sovereign credit** — country-level intensity models; contingent sovereign bond valuation.

6. **Credit portfolio management** — banks use structural + intensity models for loan portfolio risk (Basel III regulatory capital).

7. **High-yield and distressed debt** — short-dated structural analysis, recovery valuation.

8. **Insurance-linked credit** (cat bonds, mortality bonds) — intensity-style modeling.

9. **Retail credit: mortgages, auto loans, credit cards** — logistic regression + survival analysis (Cox PH) for default risk.

10. **Credit derivatives dealing** — options on CDS (swaptions), constant-maturity CDS (CMCDS).

---

## 5.5.9 Summary

- **Merton structural** model: equity = call on firm value, debt = bond - put. Gives distance-to-default.
- **Limitations**: spread $\to 0$ as $T \to 0$; can't capture real short-dated spreads (no jump-to-default).
- **Reduced-form intensity** models: default as first jump of Cox process; survival = $\mathbb{E}[e^{-\int \lambda}]$.
- **CDS** fair spread $\approx (1-R) \lambda$; bootstrap piecewise-constant hazard rates from CDS term structure.
- **Gaussian copula** for CDOs: simple, tractable, but notorious failure in 2007-2008.

### Forward pointers

- **Module 5.6**: stochastic volatility — tools overlap with credit (characteristic function pricing of Fourier methods).
- **Module 5.7**: microstructure — many CDS markets are OTC and illiquid; execution algorithms adapted.

---

## Exercises

### Tier 1 (★)

1. Compute Merton equity, debt, credit spread for $V_0 = 100, D = 70, r = 0.04, \sigma_V = 0.30, T = 5$.
2. Verify that $E_0 + D_0 = V_0$ in Ex 1.
3. Compute $\mathbb{P}_\mathbb{Q}(\text{default})$ and distance-to-default.
4. Show that for constant intensity $\lambda$, survival probability is $e^{-\lambda T}$.
5. Derive the fair CDS spread formula with constant $\lambda$.
6. Compute Merton spread for varying leverage $D/V_0$ and plot.
7. Show that CDS fair spread is approximately $(1 - R) \cdot \lambda$ when $\lambda \ll r$.

### Tier 2 (★★)

8. Show that the Merton credit spread behaves like $\sigma \sqrt{T/(2\pi)}$ for very OTM puts at short $T$ — explaining why short-dated spreads go to zero.
9. Derive the Black-Cox formula for first-passage default with constant barrier.
10. Derive the affine-term-structure form of survival probability under CIR intensity.
11. Prove the decomposition of CDS price into premium and protection legs.
12. Derive the bootstrap algorithm for piecewise-constant hazard rates.
13. For the Gaussian copula, derive the conditional default probability $p_i(M)$ given the common factor $M$.
14. Derive the analytic loss distribution recursion for identical marginal conditional default probabilities.

### Tier 3 (★★★)

15. Implement the Longstaff-Schwartz-Santa-Clara (1996) extension of Black-Cox with barrier below asset value.
16. Implement a full KMV-style firm-value extraction: given equity price and vol, iteratively solve for $V_0, \sigma_V$ using $E = f(V, \sigma)$ and the Itô vol relation $E \sigma_E = V N(d_1) \sigma_V$.
17. Implement a Jarrow-Turnbull model for CDS pricing with piecewise-constant intensity; calibrate to market data.
18. Derive the CDS swaption (option on CDS) pricing via a Black-76-like formula with credit-adjusted numéraire.
19. Implement the Duffie-Singleton $(1-R)\lambda$ framework for defaultable bond pricing under CIR intensity; bootstrap from CDS.
20. Implement Gaussian copula + FFT for efficient CDO tranche pricing.
21. Study "base correlation" smile: back out tranche-specific correlations and plot.
22. Implement t-copula and compare with Gaussian copula on CDO pricing, showing the extreme-loss probability.
23. Implement a contagion model (Davis-Lo or Jarrow-Yu) and compare with independent-intensity case for iTraxx.
24. Prove that in the Gaussian copula, senior tranche price is decreasing in correlation.
25. Derive and implement the Leland-Toft endogenous default model.
26. Implement bank portfolio risk / economic capital using Merton-KMV approach with correlated asset returns.
27. Model sovereign debt pricing with strategic default (Eaton-Gersovitz-like); compare with naive Merton-adaptation.

---

*Next module:* Stochastic volatility models — Heston, SABR revisited, rough volatility, Fourier pricing, VIX.

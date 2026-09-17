# Subject 5, Module 6: Stochastic Volatility Models

> *"The world is not log-normal, and volatility is not constant. Once you admit that, the question becomes: which model of volatility buys you the most realism for the least computational pain? That's an engineering question — and the answer has moved over the last thirty years from Heston to SABR to rough."*

## Prerequisites

- **Modules 5.1-5.2**: FTAP, Black-Scholes, implied volatility, Dupire.
- **Module 3.4**: SDEs, especially CIR.
- **Module 3.5**: Lévy processes (for jump extensions).
- **Module 5.3**: exotic options and variance swaps.
- Helpful: Fourier analysis, characteristic functions.

---

## 5.6.0 Why Stochastic Volatility?

**Empirical facts** about equity options markets:
1. The implied-vol surface has a *smile* (FX) or *skew* (equities).
2. Volatility is *mean-reverting* (high-vol days followed by lower-vol days on average).
3. There is a *leverage effect*: negative correlation between returns and volatility changes.
4. Realized variance of log-returns shows *long memory* — autocorrelation decays slower than exponential.
5. The smile has a specific *term structure*: steeper for short expiries, flatter for long.

Black-Scholes with constant $\sigma$ fails all five. **Stochastic volatility** (SV) lets $\sigma_t$ itself be random, capturing:
- Smile/skew via correlated vol-return dynamics.
- Mean reversion via OU-like drift on vol.
- Term structure via the vol-of-vol and mean-reversion parameters.

Roadmap:
- **5.6.1** Heston model.
- **5.6.2** SABR (already covered in 5.4; here revisit for equity use).
- **5.6.3** Other SV models: Hull-White, Stein-Stein, 3/2.
- **5.6.4** Local-stochastic vol (LSV).
- **5.6.5** Rough volatility.
- **5.6.6** Fourier pricing: Carr-Madan and COS methods.
- **5.6.7** VIX and variance derivatives.
- **5.6.8** Python and applications.

---

## 5.6.1 The Heston Model (1993)

### SDE system under $\mathbb{Q}$

$$dS_t = r S_t \, dt + \sqrt{V_t} S_t \, dB^S_t,$$
$$dV_t = \kappa(\theta - V_t) dt + \xi \sqrt{V_t} dB^V_t,$$
$$\mathbb{E}[dB^S dB^V] = \rho \, dt.$$

Parameters:
- $V_0$: initial variance.
- $\kappa$: speed of mean reversion.
- $\theta$: long-run variance.
- $\xi$: vol-of-vol.
- $\rho$: correlation (negative for equities — leverage effect).

Feller condition: $2\kappa\theta \ge \xi^2$ ensures $V_t > 0$ a.s.

### Characteristic function

Heston's key contribution: closed-form characteristic function of $\ln S_T$:
$$\phi(u; T) = \mathbb{E}[e^{iu \ln S_T}] = \exp(A(u, T) + B(u, T) V_0 + iu \ln S_0),$$
where $A, B$ are complex-valued functions with Riccati-form closed solutions.

Specifically:
$$B(u, T) = \dfrac{(b - \rho \xi i u + d)(1 - e^{dT})}{\xi^2 (1 - g e^{dT})},$$
$$A(u, T) = r i u T + \dfrac{\kappa\theta}{\xi^2}\left[(b - \rho \xi i u + d) T - 2 \ln\left(\dfrac{1 - g e^{dT}}{1 - g}\right)\right],$$
where
- $d = \sqrt{(\rho \xi i u - b)^2 + \xi^2 (i u + u^2)}$ (principal branch).
- $g = (b - \rho \xi i u + d) / (b - \rho \xi i u - d)$.
- $b = \kappa + \lambda$ for $\lambda$ the "market price of vol risk" (often taken 0 under $\mathbb{Q}$).

### Option pricing via Fourier

Call price via Heston's original formulation:
$$C = S_0 P_1 - K e^{-rT} P_2,$$
where $P_1, P_2$ are probabilities recovered by inverse-Fourier:
$$P_j = \dfrac{1}{2} + \dfrac{1}{\pi} \int_0^\infty \text{Re}\left(\dfrac{e^{-iu \ln K} \phi_j(u)}{iu}\right) du,$$
with specific $\phi_j$ (Heston's $P_1$ uses stock-measure char. function, $P_2$ uses bond-measure).

Numerical integration: adaptive Gauss quadrature or FFT (Carr-Madan).

### Pros and cons

**Pros:** closed-form characteristic function → fast FFT pricing; captures smile and leverage effect; affine.
**Cons:** Feller parameter restriction sometimes violated in calibration; short-dated smile often not perfectly fit; not always arbitrage-free after calibration.

### Heston calibration

Given market implied-vol surface, minimize squared error between model and market vols across strikes × maturities. Five parameters $(V_0, \kappa, \theta, \xi, \rho)$. Typical calibration uses Levenberg-Marquardt, differential evolution, or SLSQP.

---

## 5.6.2 SABR Revisited (for Equities)

Full SABR (already in Module 5.4):
$$dF = \alpha F^\beta dB^1,$$
$$d\alpha = \nu \alpha dB^2.$$

For equities typically use $\beta = 1$ (lognormal SABR), or $\beta = 0.5$ (CIR-like). The Hagan approximation gives implied-vol formula directly; calibration trivial.

**Industry preference:** short-dated options often use SABR for speed and simplicity; long-dated use Heston for dynamics.

---

## 5.6.3 Other Stochastic Volatility Models

### Hull-White SV (1987)

$$dS_t = rS dt + \sigma_t S \, dB,$$
$$d\sigma_t^2 = \xi \sigma^2 dZ,$$
with lognormal variance (multiplicative rather than CIR-like).

No closed form for char function; priced by MC or series expansion. Less popular than Heston.

### Stein-Stein (1991) / Schöbl-Zhu (1999)

$$d\sigma_t = \kappa(\theta - \sigma_t) dt + \xi dZ.$$

OU process on $\sigma$ (not $\sigma^2$). Closed-form characteristic function possible; similar structure to Heston.

Can go negative (so $\sigma^2$ always positive — just weird if $\sigma$ itself is negative). Mildly less realistic than Heston for equity.

### 3/2 Model (Heston 1997, Lewis 2000)

$$dV_t = V_t(\kappa - \xi V_t) dt + \epsilon V_t^{3/2} dZ.$$

Variance follows "inverse-CIR" style SDE. Captures better the short-dated smile shape. Closed form for vanilla options. Less popular but theoretically elegant.

### Double Heston (2-factor)

Two independent Heston vol factors: captures volatility "term structure." Used for longer-dated VIX derivatives pricing.

### Variance-Gamma and NIG extensions

Combine SV with jumps: Bates model (SVJ) adds Merton jumps to Heston:
$$dS/S = (r - \lambda \bar J) dt + \sqrt V dB + (J-1) dN.$$

### Hybrid models

- **SVJJ** (stochastic vol + jumps in both spot and vol): Duffie-Pan-Singleton (2000).
- **SV with stochastic intensity**: $V$ and $\lambda$ both stochastic.

Affine framework (Duffie-Pan-Singleton) provides unified analytical treatment of all these.

---

## 5.6.4 Local-Stochastic Volatility (LSV)

### Motivation

Local volatility (Dupire) fits today's option surface exactly but has unrealistic dynamics (smile flattens deterministically). Stochastic vol has realistic dynamics but can't fit the surface exactly.

**LSV combines both.** Model:
$$dS_t = r S_t dt + L(t, S_t) \sqrt{V_t} S_t dB^S_t,$$
$$dV_t = \kappa(\theta - V_t) dt + \xi \sqrt{V_t} dB^V_t,$$
with **leverage function** $L(t, S)$ chosen so the model reproduces the market smile.

### Calibration: fixed-point

Given market local-vol $\sigma^{\text{loc}}(K, T)$ and Heston parameters:
$$L(T, K)^2 = \dfrac{\sigma^{\text{loc}}(K, T)^2}{\mathbb{E}_\mathbb{Q}[V_T | S_T = K]}.$$

**Problem:** the conditional expectation depends on $L$. So fixed-point iteration or particle method (Guyon-Henry-Labordère 2012): generate paths, compute empirical conditional expectation, update $L$, repeat.

### Particle method

1. Initialize $L^{(0)}(t, S) \equiv 1$.
2. Simulate Heston + LSV paths.
3. For each $(t_k, S)$ grid point, compute $\mathbb{E}[V_{t_k} | S_{t_k} \approx S]$ empirically.
4. Update $L^{(1)}(t_k, S) = \sigma^{\text{loc}}(S, t_k) / \sqrt{\mathbb{E}[V|\cdot]}$.
5. Repeat until convergence.

Used in industry for exotic pricing consistent with the vanilla market.

---

## 5.6.5 Rough Volatility

### Empirical observation (Gatheral-Jaisson-Rosenbaum 2018)

Realized volatility of many assets exhibits **long memory** with Hurst exponent $H \approx 0.1$ — far below BM's $H = 0.5$. Standard SV models (Heston, SABR) have $H = 0.5$ and can't match this.

### Rough Bergomi (Bayer-Friz-Gatheral 2016)

$$d S_t / S_t = \sqrt{V_t} dB^S_t,$$
$$V_t = V_0 \cdot \exp\left(2\eta \int_0^t (t-s)^{H-1/2} dB^V_s - \eta^2 V_0 \int_0^t \cdots\right),$$
with $H \approx 0.1$.

The vol process is driven by *fractional Brownian motion* — extremely rough (non-semi-martingale) paths.

### Implications for pricing

- Short-dated ATM skew behaves like $T^{H-1/2}$ (steeper as $T \to 0$), matching market data which shows $T^{-0.4}$ scaling.
- Non-Markov: no PDE, must use MC or simulation.
- Characteristic function known semi-analytically.

### Other rough models

- **Rough Heston** (El Euch-Rosenbaum 2019): $V$ satisfies fractional CIR; characteristic function via fractional Riccati.
- **Rough SABR**: similar extension.

### Computational challenges

Fractional Brownian motion simulation is expensive (non-Markov; need full path). Hybrid schemes: Cholesky + truncation; hybrid BSS scheme (Bennedsen-Lunde-Pakkanen 2017).

---

## 5.6.6 Fourier Pricing Methods

### Carr-Madan (1999) FFT

For any model with characteristic function $\phi$ of $\ln S_T$:

1. Damp the call price: $C(K) e^{\alpha \ln K}$ is integrable for some $\alpha > 0$.
2. Fourier transform of damped price yields formula in $\phi$.
3. Inverse via FFT: one FFT call computes $C(K)$ at $N$ log-strikes at once.

Call price formula:
$$C(K) = \dfrac{e^{-\alpha \ln K}}{\pi} \int_0^\infty \text{Re}\left(e^{-i u \ln K} \psi(u) \right) du,$$
$$\psi(u) = \dfrac{e^{-rT} \phi(u - (\alpha+1) i)}{\alpha^2 + \alpha - u^2 + i(2\alpha + 1) u}.$$

Discretize: Simpson's rule on $\{u_k\}$ grid; apply FFT to recover $\{C(K_j)\}$ on log-strike grid.

### COS Method (Fang-Oosterlee 2008)

Expand damped density in Fourier cosine series on $[a, b]$:
$$p(y) \approx \sum_{k=0}^{N-1} F_k \cos(k \pi (y - a)/(b - a)),$$
where $F_k$ are recoverable from the characteristic function.

Call price:
$$C \approx e^{-rT} \sum_k F_k V_k,$$
with $V_k = \int_a^b (e^y - K)^+ \cos(k\pi(y-a)/(b-a)) dy$ having closed form.

**Typical accuracy:** $N = 64$ terms, 10-digit accuracy; much faster than FFT for vanilla options.

### Heston via FFT

The most common use of Fourier methods. For a practitioner: just provide $\phi_{\text{Heston}}(u)$ (5 parameters), and get vanilla prices in milliseconds.

---

## 5.6.7 VIX and Variance Derivatives

### VIX formula

$$\text{VIX}^2 = \dfrac{2}{T}\left[\int_0^{F} \dfrac{P(K)}{K^2} dK + \int_F^\infty \dfrac{C(K)}{K^2} dK\right] \cdot e^{rT}$$
(approximated discretely from CBOE methodology).

### VIX futures

Under $\mathbb{Q}$, the VIX future price at time $t$ for maturity $T$ is:
$$F_t^{VIX}(T) = \mathbb{E}_\mathbb{Q}[\text{VIX}_T | \mathcal{F}_t] = \mathbb{E}_\mathbb{Q}[\sqrt{\mathbb{E}_\mathbb{Q}[\text{var}[t:t+30\text{day}]| \mathcal{F}_T]} | \mathcal{F}_t].$$

Under Heston: can derive semi-closed formula involving the variance dynamics.

### Variance swap

Already discussed in Module 5.3. Under Heston:
$$K_{\text{var}} = \theta + (V_0 - \theta) \dfrac{1 - e^{-\kappa T}}{\kappa T}.$$

### VIX options

Options on VIX futures. Complex because VIX is a nonlinear function of variance. Calibration to SPX smile + VIX smile jointly is a challenge; motivates 2-factor SV models.

---

## 5.6.8 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# -----------------------------
# Heston characteristic function
# -----------------------------
def heston_char(u, S0, T, r, V0, kappa, theta, xi, rho):
    """Heston log-price characteristic function phi(u, T)."""
    d = np.sqrt((rho*xi*1j*u - kappa)**2 + xi**2*(1j*u + u**2))
    g = (kappa - rho*xi*1j*u - d) / (kappa - rho*xi*1j*u + d)
    A = r*1j*u*T + (kappa*theta/xi**2) * (
        (kappa - rho*xi*1j*u - d)*T - 2*np.log((1 - g*np.exp(-d*T))/(1 - g))
    )
    B = (kappa - rho*xi*1j*u - d)/xi**2 * (1 - np.exp(-d*T))/(1 - g*np.exp(-d*T))
    return np.exp(A + B*V0 + 1j*u*np.log(S0))

# -----------------------------
# Carr-Madan call price
# -----------------------------
def carr_madan_call(S0, K, T, r, V0, kappa, theta, xi, rho, alpha=1.5):
    def integrand(u):
        phi = heston_char(u - (alpha+1)*1j, S0, T, r, V0, kappa, theta, xi, rho)
        denom = alpha**2 + alpha - u**2 + 1j*(2*alpha+1)*u
        return np.real(np.exp(-1j*u*np.log(K)) * np.exp(-r*T) * phi / denom)
    integral, _ = quad(integrand, 0, 100, limit=200)
    return np.exp(-alpha*np.log(K)) / np.pi * integral

# Test
S0, K, T = 100, 100, 1.0
r = 0.05
V0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.4, -0.7
C = carr_madan_call(S0, K, T, r, V0, kappa, theta, xi, rho)
print(f"Heston call (Carr-Madan) = {C:.4f}")

# -----------------------------
# Heston Monte Carlo (Euler with full truncation)
# -----------------------------
def heston_mc(S0, K, T, r, V0, kappa, theta, xi, rho, n_paths=100_000, n_steps=200):
    dt = T/n_steps
    S = np.full(n_paths, S0); V = np.full(n_paths, V0)
    for _ in range(n_steps):
        Z1, Z2 = np.random.randn(n_paths), np.random.randn(n_paths)
        dB_V = Z1*np.sqrt(dt)
        dB_S = (rho*Z1 + np.sqrt(1-rho**2)*Z2)*np.sqrt(dt)
        V_prev = np.maximum(V, 0)
        V = V + kappa*(theta - V_prev)*dt + xi*np.sqrt(V_prev)*dB_V
        S = S * np.exp((r - V_prev/2)*dt + np.sqrt(V_prev)*dB_S)
    payoff = np.maximum(S - K, 0)
    return np.exp(-r*T) * payoff.mean(), payoff.std()/np.sqrt(n_paths) * np.exp(-r*T)

price_mc, se = heston_mc(S0, K, T, r, V0, kappa, theta, xi, rho)
print(f"Heston call (MC)        = {price_mc:.4f} +/- {se*1.96:.4f}")

# -----------------------------
# Generate Heston smile
# -----------------------------
def implied_vol_from_price(price, S0, K, r, T, tol=1e-6):
    from scipy.optimize import brentq
    from scipy.stats import norm
    def bs(sig):
        d1 = (np.log(S0/K) + (r + sig**2/2)*T) / (sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return brentq(lambda sig: bs(sig) - price, 1e-4, 3.0, xtol=tol)

strikes = np.linspace(70, 140, 30)
heston_prices = [carr_madan_call(S0, K_, T, r, V0, kappa, theta, xi, rho) for K_ in strikes]
iv_smile = [implied_vol_from_price(p, S0, K_, r, T) for p, K_ in zip(heston_prices, strikes)]
plt.figure(figsize=(9, 4))
plt.plot(strikes, iv_smile, 'o-')
plt.xlabel('Strike'); plt.ylabel('Implied Vol')
plt.title('Heston implied volatility smile')
plt.grid(alpha=0.3); plt.show()

# -----------------------------
# Rough Bergomi Monte Carlo (hybrid scheme stub)
# -----------------------------
def rough_bergomi_mc(S0, K, T, H, eta, rho, V0, n_paths=10_000, n_steps=200):
    """Rough Bergomi via hybrid scheme (simplified)."""
    dt = T/n_steps
    # Generate fBm via Cholesky (O(N^2) memory -- OK for small N)
    # Covariance of fBM: cov(B_t, B_s) = 0.5*(|t|^2H + |s|^2H - |t-s|^2H)
    t = np.arange(1, n_steps+1) * dt
    cov_mat = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(n_steps):
            cov_mat[i, j] = 0.5*(t[i]**(2*H) + t[j]**(2*H) - abs(t[i]-t[j])**(2*H))
    L = np.linalg.cholesky(cov_mat + 1e-10*np.eye(n_steps))

    # Log-payoff MC
    payoffs = []
    for _ in range(n_paths):
        Z_W = np.random.randn(n_steps)
        fBm = L @ Z_W
        # V(t) = V0 * exp(2*eta*fBm - eta^2 * t^(2H))
        V = V0 * np.exp(2*eta*fBm - eta**2 * t**(2*H))
        # S evolution
        Z_S = np.random.randn(n_steps)
        # Correlated Brownian for S: use Z_W and Z_S
        logS = np.log(S0)
        for k in range(n_steps):
            dW = rho*Z_W[k]*np.sqrt(dt) + np.sqrt(1-rho**2)*Z_S[k]*np.sqrt(dt)
            logS += -0.5*V[k]*dt + np.sqrt(V[k])*dW
        S_T = np.exp(logS)
        payoffs.append(max(S_T - K, 0))
    return np.mean(payoffs), np.std(payoffs)/np.sqrt(n_paths)

# Takes a minute or two; small n_paths
# price, se = rough_bergomi_mc(100, 100, 0.5, H=0.1, eta=1.5, rho=-0.7, V0=0.04)
# print(f"Rough Bergomi MC = {price:.4f} +/- {se*1.96:.4f}")
```

---

## 5.6.9 [QUANT APPLICATION]

1. **Equity derivatives desks** — Heston + SABR + LSV for pricing vanilla and exotic equity options; calibrated daily to market.

2. **Variance and vol trading** — variance swaps priced model-free; VIX futures with Heston or Double Heston.

3. **Index options (SPX, NDX)** — deep liquidity; smile structure calibrated to with 2-factor SV or rough vol.

4. **FX options** — similar SV toolkit; Vanna-Volga for smile-adjustments.

5. **Rates smile** — SABR by default; used in swaptions and caps.

6. **Commodity options** — SABR with $\beta = 0.5$ or CEV-like; often with positive skew (inversion of equity).

7. **Autocallables and equity structured notes** — priced with LSV or SLV (stochastic local vol) to match vanilla smile and reflect forward-skew dynamics.

8. **Credit derivatives** — stochastic vol of credit spreads (extension: CVA with SV rates).

9. **High-frequency signal extraction** — rough vol models help estimate conditional variance at short horizons.

10. **Risk management** — VaR under SV captures volatility shock scenarios beyond historical moves.

---

## 5.6.10 Summary

- **Heston**: affine, char. function closed-form, industry workhorse.
- **SABR**: approximate implied vol formula; fast; standard for rates and short-dated equity.
- **Local-stochastic vol (LSV)**: combines Dupire (fit today's surface) with SV (realistic dynamics); calibrated by particle method.
- **Rough volatility**: Hurst parameter ~0.1 matches empirical long memory; handles short-dated skew beautifully; computational cost higher.
- **Fourier pricing (Carr-Madan, COS)**: fast vanilla pricing for any model with tractable characteristic function.
- **Variance and VIX derivatives**: model-free replication of variance; VIX as proxy for implied vol.

### Forward pointers

- **Module 5.7**: market microstructure — how SV models meet limit-order-book reality in trading.
- **Subject 6 (future)**: advanced numerical methods — Fourier, PDE, MC, deep learning for SV models.
- **Subject 8 (future)**: risk management — VaR and ES under SV.

---

## Exercises

### Tier 1 (★)

1. Verify the Feller condition for $\kappa = 3, \theta = 0.04, \xi = 0.6$.
2. Simulate Heston paths with $V_0 = 0.04, \kappa = 2, \theta = 0.04, \xi = 0.3, \rho = -0.7$ and plot.
3. Use Carr-Madan to price a call with Heston parameters from Ex 2 and extract implied vol.
4. Compute the variance swap strike under Heston with the given parameters.
5. Price a VIX future under Heston semi-analytically (for standard parameters).
6. Explain why Heston Euler simulation can produce negative variance and describe full-truncation fix.

### Tier 2 (★★)

7. Derive the Heston characteristic function from the Kolmogorov backward PDE.
8. Prove the variance swap strike formula under Heston: $K_{\text{var}} = \theta + (V_0 - \theta)(1 - e^{-\kappa T})/(\kappa T)$.
9. Prove convergence of the Carr-Madan FFT method.
10. Derive the COS method for call pricing and show the closed-form coefficients.
11. Implement calibration of Heston to a simulated option surface; quantify fit error.
12. Derive the Hull-White SV characteristic function and compare with Heston.
13. Prove the LSV calibration formula: $L(T, K)^2 = \sigma^{\text{loc}}(K, T)^2 / \mathbb{E}[V_T | S_T = K]$.
14. Explain why SV models produce volatility skew in the absence of jumps, using $\rho \ne 0$ analysis.

### Tier 3 (★★★)

15. Implement Andersen's QE scheme for Heston (exact-scheme alternative to Euler).
16. Implement particle LSV calibration and verify that it reproduces the market smile.
17. Implement rough Bergomi with hybrid BSS scheme and compute short-dated ATM skew; verify $T^{H-1/2}$ scaling.
18. Derive the fractional Riccati ODE for the rough Heston characteristic function and implement.
19. Calibrate joint SPX and VIX smile simultaneously with 2-factor SV or rough vol; discuss fit.
20. Prove that for small $T$, rough volatility gives smile skew $\sim T^{H-1/2}$.
21. Implement Bates (SVJ) MC and compare smile with Heston to isolate jump contribution.
22. Derive the Heston delta hedge with vol risk: add a vol-hedging instrument (variance swap) and compute the optimal 2-instrument portfolio.
23. Prove Gatheral's formula $\sigma^{\text{loc}}(K, T)^2 = \mathbb{E}[V_T | S_T = K]$ in an SV model.
24. Implement and compare Heston, SABR, LSV, rough Bergomi on a challenging calibration set (e.g., SPX cascading expiries) and discuss strengths/weaknesses.

---

*Next module:* Market microstructure and execution — limit order books, Kyle and Glosten-Milgrom, execution algorithms, market making.

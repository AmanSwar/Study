# Subject 8, Module 2: Risk Management — VaR, Expected Shortfall, Coherent Measures

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"The role of a risk manager is to be paranoid for a living and make it pay."* — folk wisdom from a Basel committee meeting

---

## 8.2.0 Where We Are

Risk management is where portfolio mathematics meets the regulator, the prime broker, and, most ruthlessly, the market. This module formalizes the apparatus of quantitative risk measurement: what should a good risk measure do, how do we estimate it, and how do we test that our estimates are any good.

### Prerequisites

- **Module 2.5** (Characteristic Functions): for Fourier-based VaR.
- **Module 2.7** (Markov chains, briefly): for regime-switching risk.
- **Module 7.5** (Volatility Modeling): GARCH, realized variance.
- **Module 8.1** (Portfolio Optimization): risk as objective.

### Plan

1. Risk measures: definition, axioms, coherence (§8.2.1).
2. Value-at-Risk: definitions, historical / parametric / Monte Carlo (§8.2.2).
3. Expected Shortfall and spectral risk measures (§8.2.3).
4. Backtesting: Kupiec, Christoffersen, traffic lights (§8.2.4).
5. Stress testing and scenario analysis (§8.2.5).
6. Copulas and dependence modeling (§8.2.6).
7. Extreme Value Theory (§8.2.7).
8. Risk aggregation and allocation (§8.2.8).
9. Regulatory context: Basel FRTB and beyond (§8.2.9).
10. Python implementations (§8.2.10).
11. Applications (§8.2.11).
12. Exercises (§8.2.12).

---

## 8.2.1 Risk Measures and Coherence

A **risk measure** is a functional $\rho: \mathcal{X} \to \mathbb{R}$ mapping random variables (typically losses) to real numbers, interpretable as "how much capital is needed to hold this position."

### Artzner–Delbaen–Eber–Heath (1999) axioms

A coherent risk measure satisfies:
- **Monotonicity**: $X \le Y \Rightarrow \rho(X) \le \rho(Y)$.
- **Subadditivity**: $\rho(X+Y) \le \rho(X) + \rho(Y)$.
- **Positive homogeneity**: $\rho(\lambda X) = \lambda \rho(X)$ for $\lambda \ge 0$.
- **Translation invariance**: $\rho(X + c) = \rho(X) - c$ for $c \in \mathbb{R}$.

Subadditivity formalizes diversification benefit. Monotonicity rules out preferring a worse position. Translation invariance links risk to required capital.

### Convex risk measures

Föllmer–Schied weakened positive homogeneity to **convexity**: $\rho(\lambda X + (1-\lambda)Y) \le \lambda\rho(X) + (1-\lambda)\rho(Y)$. The entropic risk measure
$$\rho(X) = \frac{1}{\gamma}\log\mathbb{E}[e^{-\gamma X}]$$
is convex but not positive homogeneous.

### Representation theorem

Every coherent risk measure admits a robust representation:
$$\rho(X) = \sup_{Q \in \mathcal{Q}}\mathbb{E}_Q[-X]$$
for some family $\mathcal{Q}$ of probability measures. The family encodes the "stress scenarios" implicit in the risk measure.

---

## 8.2.2 Value-at-Risk

### Definition

For a loss $L$ (so positive values are losses) and confidence $\alpha \in (0,1)$,
$$\mathrm{VaR}_\alpha(L) = \inf\{x \in \mathbb{R}: \mathbb{P}(L \le x) \ge \alpha\}.$$
This is the $\alpha$-quantile of $L$. Typical $\alpha$: 0.95 (trading book) or 0.99 (regulatory).

**VaR is not coherent** — it fails subadditivity (except for elliptical distributions). Classic counterexample: two bonds, each defaulting with probability 4% with independent default events. Each has $\mathrm{VaR}_{0.95}=0$, but the combined portfolio has $\mathrm{VaR}_{0.95}>0$.

### Historical VaR

Given a window of $T$ past P&L observations $\{L_t\}$, estimate $\mathrm{VaR}_\alpha$ as the empirical $\alpha$-quantile. Pros: no distributional assumption. Cons: slow to update to changes in volatility (mitigated by age-weighted filtered historical simulation, Boudoukh–Richardson–Whitelaw 1998).

### Parametric VaR

Assume $L \sim \mathcal{N}(\mu, \sigma^2)$:
$$\mathrm{VaR}_\alpha = \mu + \sigma \Phi^{-1}(\alpha).$$
Extensions: $t$-distribution, skew-$t$, Cornish-Fisher expansion correcting for skewness and kurtosis:
$$\mathrm{VaR}_\alpha \approx \mu + \sigma \left[z + \tfrac{1}{6}(z^2-1)s + \tfrac{1}{24}(z^3-3z)k - \tfrac{1}{36}(2z^3-5z)s^2\right],$$
with $z = \Phi^{-1}(\alpha)$, $s$ skewness, $k$ excess kurtosis.

### Monte Carlo VaR

Simulate the portfolio P&L distribution from a specified model (e.g., factor model with GARCH volatilities, copula dependence), then compute the empirical quantile. Slow but flexible.

### Factor-based VaR

For large portfolios, estimate the covariance from a factor model $\Sigma = XFX^\top + D$ (Module 7.6), then use parametric VaR. Industry standard in most bank market-risk systems.

---

## 8.2.3 Expected Shortfall

$$\mathrm{ES}_\alpha(L) = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha(L)] = \frac{1}{1-\alpha}\int_\alpha^1 \mathrm{VaR}_u(L) du.$$

ES is **coherent**. For continuous distributions, $\mathrm{ES}_\alpha \ge \mathrm{VaR}_\alpha$. For normal losses:
$$\mathrm{ES}_\alpha = \mu + \sigma \frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha}.$$

### Why regulators shifted to ES

Basel III's Fundamental Review of the Trading Book (FRTB, 2019) replaced VaR with a 97.5% ES for market risk capital because ES is coherent and captures tail magnitude, not just breach probability.

### Spectral risk measures

A generalization: $\rho_\phi(L) = \int_0^1 \phi(u) \, \mathrm{VaR}_u(L) \, du$ for a weight function $\phi$. ES is a spectral measure with $\phi(u) = \mathbb{1}\{u \ge \alpha\}/(1-\alpha)$. The exponential spectral measure $\phi(u) \propto e^{-\gamma(1-u)}$ is smoother but harder to calibrate.

### ES is hard to backtest

Unlike VaR (binary breach/no breach), ES is a conditional expectation — backtesting requires estimating a continuous quantity under the thin tail. Acerbi-Szekely (2014) proposed three tests:
- **Test 1**: exceedance-sized.
- **Test 2**: conditional-on-breach expectation.
- **Test 3**: likelihood-ratio style.
Kratz–Lok–McNeil (2018) advocate joint VaR-ES backtesting.

---

## 8.2.4 VaR Backtesting

### Kupiec (1995) unconditional coverage

Let $X_t = \mathbb{1}\{L_t > \mathrm{VaR}_{\alpha,t}\}$. Under $H_0: \mathbb{E}[X_t] = 1-\alpha$, the likelihood-ratio test is
$$\mathrm{LR}_{uc} = -2\log\left[\frac{\alpha^{T-k}(1-\alpha)^k}{(1-\hat p)^{T-k}\hat p^k}\right] \sim \chi^2_1,$$
where $k$ is observed breaches and $\hat p = k/T$.

### Christoffersen (1998) independence

Breaches should be i.i.d.; clustering indicates volatility dynamics not captured. Test:
$$\mathrm{LR}_{ind} = -2\log\left[\frac{(1-\hat p_{01})^{n_{00}}\hat p_{01}^{n_{01}}(1-\hat p_{11})^{n_{10}}\hat p_{11}^{n_{11}}}{(1-\hat p)^{n_{00}+n_{10}}\hat p^{n_{01}+n_{11}}}\right] \sim \chi^2_1.$$
Combined: $\mathrm{LR}_{cc} = \mathrm{LR}_{uc} + \mathrm{LR}_{ind} \sim \chi^2_2$.

### Traffic light (Basel)

Over 250 trading days, count breaches:
- Green: 0–4 breaches at 99% (model OK).
- Yellow: 5–9 (investigation; capital multiplier rises 0.4–0.85).
- Red: 10+ (model rejected).

### Duration-based backtest

Christoffersen–Pelletier (2004) test that durations between VaR breaches are exponentially distributed (memoryless), which holds under correct coverage.

---

## 8.2.5 Stress Testing and Scenario Analysis

### Historical scenarios

Run the current portfolio through realized market moves from reference crises: 1987 crash, 1998 LTCM, 2008 GFC, 2011 Eurozone, 2015 Chinese devaluation, 2020 COVID, 2022 rate-shock, etc.

### Hypothetical scenarios

Specify shock vectors (e.g., "equities -20%, credit spreads +200bp, USD +5%, VIX to 50"). Requires a shock-propagation model for unspecified risk factors — commonly a conditional Gaussian:
$$\mathbb{E}[F_{\text{unspec}} \mid F_{\text{spec}} = s] = \mu_{\text{unspec}} + \Sigma_{\text{unspec, spec}}\Sigma_{\text{spec,spec}}^{-1}(s - \mu_{\text{spec}}).$$

### Reverse stress testing

Find the scenario that produces a given loss threshold with minimum unlikelihood:
$$\min_{F} \frac{1}{2}(F-\mu)^\top \Sigma^{-1}(F-\mu) \quad \text{s.t.} \quad w^\top P(F) \le -L^*.$$
This reveals the most-plausible crisis mechanism for the portfolio.

### Liquidity-adjusted VaR

Incorporate the bid-ask spread's multiple of volatility ("liquidity VaR," BIS 2013):
$$\mathrm{VaR}^L_\alpha = \mathrm{VaR}_\alpha + \tfrac{1}{2}\sum_i p_i (\mu_{\text{spread},i} + k \sigma_{\text{spread},i}).$$

---

## 8.2.6 Copulas

### Sklar's theorem

For any joint distribution $F$ with continuous marginals $F_1, \dots, F_d$, there exists a unique copula $C$ such that
$$F(x_1, \dots, x_d) = C(F_1(x_1), \dots, F_d(x_d)).$$
The copula encodes the dependence structure independently of marginals.

### Common copulas

- **Gaussian copula**: $C_\Sigma(u) = \Phi_\Sigma(\Phi^{-1}(u_1), \dots, \Phi^{-1}(u_d))$ — tail-independent.
- **Student-t copula**: retains some tail dependence; widely used in credit.
- **Archimedean**: Clayton (lower-tail dependence), Gumbel (upper-tail dependence), Frank.
- **Vine copulas** (Joe–Bedford–Cooke): decompose multivariate dependence into bivariate pair copulas, allowing flexible tail structure.

### Tail dependence

$$\lambda_U = \lim_{u\to 1^-}\mathbb{P}(U_2 > u \mid U_1 > u), \qquad \lambda_L = \lim_{u\to 0^+}\mathbb{P}(U_2 \le u \mid U_1 \le u).$$
Gaussian copula: $\lambda_U = \lambda_L = 0$. $t$-copula with $\nu$ dof and correlation $\rho$: $\lambda_U = 2 t_{\nu+1}(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)}) > 0$.

The 2008 crisis indicted the Gaussian copula used in CDO pricing — tail independence produced catastrophic underestimation of joint default rates.

### Copula fitting

Two-step IFM: fit marginals, then fit copula parameters via MLE on pseudo-observations $u_{it} = \hat F_i(x_{it})$. Canonical MLE with semiparametric marginals via empirical CDFs is standard.

---

## 8.2.7 Extreme Value Theory

### Fisher–Tippett theorem

If $(M_n - b_n)/a_n \to G$ in distribution for some constants, then $G$ belongs to the Generalized Extreme Value (GEV) family:
$$G(x) = \exp\left(-\left(1 + \xi \tfrac{x-\mu}{\sigma}\right)^{-1/\xi}\right), \qquad 1 + \xi(x-\mu)/\sigma > 0.$$
- $\xi > 0$: Fréchet (heavy tail).
- $\xi = 0$: Gumbel (exponential tail).
- $\xi < 0$: Weibull (bounded tail).

Financial returns are almost always in the Fréchet domain ($\xi > 0$), with estimated $\xi$ around 0.2–0.4 for equities.

### Peaks-over-threshold (Pickands–Balkema–de Haan)

For high threshold $u$, the distribution of excesses $(X - u \mid X > u)$ converges to Generalized Pareto:
$$F_u(y) = 1 - (1 + \xi y/\beta)^{-1/\xi}.$$
Fit $\xi, \beta$ by MLE on observations exceeding $u$; tail VaR:
$$\mathrm{VaR}_\alpha = u + \frac{\beta}{\xi}\left[\left(\frac{1-\alpha}{\bar F(u)}\right)^{-\xi} - 1\right], \qquad \bar F(u) = k/n,$$
where $k$ is the number of exceedances.

### Hill estimator

For heavy-tailed data, the tail index $\xi$ is estimated by
$$\hat\xi_H = \frac{1}{k}\sum_{i=1}^{k}\log X_{(n-i+1)} - \log X_{(n-k)},$$
with $X_{(\cdot)}$ order statistics. Converges as $k \to \infty$, $k/n \to 0$.

### Mean excess function

Empirical plot of $e(u) = \mathbb{E}[X - u \mid X > u]$ against $u$; linear behavior indicates GPD applicability (slope $\xi/(1-\xi)$).

---

## 8.2.8 Risk Aggregation and Allocation

### Aggregation

For a portfolio $\pi = \sum_k \pi_k$ with component risks $\rho(\pi_k)$, subadditivity of $\rho$ means
$$\rho(\pi) \le \sum_k \rho(\pi_k).$$
Basel allows aggregation via summation (conservative) or through internal models (less conservative if validated).

### Euler allocation

For a positively homogeneous $\rho$:
$$\rho(\pi) = \sum_k \pi_k \frac{\partial \rho}{\partial \pi_k}(\pi).$$
The Euler contribution $\mathrm{RC}_k = \pi_k \partial\rho / \partial\pi_k$ allocates total risk to components, linear in positions. For ES:
$$\mathrm{RC}_k = \mathbb{E}[L_k \mid L \ge \mathrm{VaR}_\alpha(L)].$$

### Capital allocation fairness

Denault (2001) showed Euler is the unique allocation rule compatible with a Shapley-value-style fairness axiom when $\rho$ is coherent.

---

## 8.2.9 Regulatory Context

- **Basel II/III (banking)**: minimum capital for credit, market, operational risks.
- **FRTB (Basel III.1)**: ES at 97.5%, liquidity horizons by asset class, Internal Model Approach or Standardized Approach.
- **Solvency II (insurance, EU)**: one-year VaR at 99.5%.
- **UCITS / AIFMD**: leverage and VaR limits on retail funds.
- **CCAR / DFAST (US banks)**: supervisory stress tests with prescribed scenarios.
- **Clearing house margin (SPAN, VaR-based)**: futures and options initial margin.

Risk-management quants spend substantial time reconciling internal models with regulatory prescriptions and running what-if analyses around regulatory capital consumption.

---

## 8.2.10 Python Implementations

```python
import numpy as np
from scipy import stats
import pandas as pd

# ----- VaR and ES from returns -----
def var_historical(returns, alpha=0.99):
    return -np.quantile(returns, 1 - alpha)

def es_historical(returns, alpha=0.99):
    q = np.quantile(returns, 1 - alpha)
    return -returns[returns <= q].mean()

def var_parametric_normal(returns, alpha=0.99):
    mu, sigma = returns.mean(), returns.std(ddof=1)
    return -(mu + sigma * stats.norm.ppf(1 - alpha))

def es_parametric_normal(returns, alpha=0.99):
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))

# ----- Cornish-Fisher VaR -----
def cornish_fisher_var(returns, alpha=0.99):
    mu, sigma = returns.mean(), returns.std(ddof=1)
    s = stats.skew(returns)
    k = stats.kurtosis(returns)  # excess
    z = stats.norm.ppf(1 - alpha)
    z_cf = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24
            - (2*z**3 - 5*z)*s**2/36)
    return -(mu + sigma * z_cf)

# ----- Kupiec test -----
def kupiec_test(hits, alpha=0.99):
    T, k = len(hits), hits.sum()
    p_hat = k / T
    p = 1 - alpha
    ll_null = k*np.log(p) + (T-k)*np.log(1-p) if 0 < p < 1 else 0
    ll_alt = k*np.log(p_hat) + (T-k)*np.log(1-p_hat) if 0 < p_hat < 1 else 0
    LR = -2*(ll_null - ll_alt)
    pval = 1 - stats.chi2.cdf(LR, df=1)
    return LR, pval

# ----- Christoffersen independence test -----
def christoffersen_independence(hits):
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(hits)):
        if hits[i-1]==0 and hits[i]==0: n00 += 1
        elif hits[i-1]==0 and hits[i]==1: n01 += 1
        elif hits[i-1]==1 and hits[i]==0: n10 += 1
        elif hits[i-1]==1 and hits[i]==1: n11 += 1
    p01 = n01/(n00+n01) if n00+n01>0 else 0
    p11 = n11/(n10+n11) if n10+n11>0 else 0
    p = (n01+n11)/(n00+n01+n10+n11)
    ll_null = (n01+n11)*np.log(p) + (n00+n10)*np.log(1-p) if 0<p<1 else 0
    ll_alt = 0
    for (n, q) in [(n01, p01), (n11, p11)]:
        if n>0 and 0<q<1: ll_alt += n*np.log(q)
    for (n, q) in [(n00, p01), (n10, p11)]:
        if n>0 and 0<q<1: ll_alt += n*np.log(1-q)
    LR = -2*(ll_null - ll_alt)
    return LR, 1 - stats.chi2.cdf(LR, df=1)

# ----- Peaks-over-threshold GPD fit -----
def fit_gpd(excesses):
    shape, _, scale = stats.genpareto.fit(excesses, floc=0)
    return shape, scale

def var_es_pot(returns, u, alpha=0.99):
    losses = -returns
    excesses = losses[losses > u] - u
    shape, scale = fit_gpd(excesses)
    n, Nu = len(losses), len(excesses)
    VaR = u + scale/shape * ((((1-alpha)*n/Nu)**(-shape)) - 1)
    ES = (VaR + scale - shape*u)/(1 - shape)
    return VaR, ES

# ----- Gaussian copula simulation -----
def simulate_gaussian_copula(rho, n):
    """2d Gaussian copula with correlation rho, returning uniforms."""
    L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
    z = np.random.standard_normal((n, 2)) @ L.T
    return stats.norm.cdf(z)

# ----- t-copula simulation -----
def simulate_t_copula(rho, nu, n):
    L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
    z = np.random.standard_normal((n, 2)) @ L.T
    s = np.random.chisquare(nu, n) / nu
    t = z / np.sqrt(s)[:, None]
    return stats.t.cdf(t, nu)

# ----- Euler ES allocation -----
def euler_es_allocation(pnl_component, pnl_total, alpha=0.99):
    """pnl_component: T x K array of component P&Ls; pnl_total: T-array of total P&L."""
    q = np.quantile(-pnl_total, alpha)
    breach = (-pnl_total >= q)
    return -pnl_component[breach].mean(axis=0)

# ----- Example -----
rng = np.random.default_rng(0)
R = rng.standard_t(df=5, size=10000) * 0.01
print("Hist VaR(99%):", var_historical(R, 0.99))
print("Hist ES(99%):", es_historical(R, 0.99))
print("Norm VaR(99%):", var_parametric_normal(R, 0.99))
print("CF VaR(99%):", cornish_fisher_var(R, 0.99))
```

---

## 8.2.11 Applications

1. **Trading-book capital**: daily market-risk VaR/ES for regulatory and internal consumption; traffic-light reporting.
2. **Margin models**: CCP initial margin via VaR-like measures (SPAN, IM from central counterparties).
3. **Hedge fund risk reporting**: monthly VaR decomposition by strategy, factor, and trader.
4. **Pre-trade risk checks**: marginal VaR of a proposed trade against current book.
5. **Credit portfolio risk**: copula-based Monte Carlo for CDO tranche losses.
6. **Concentration risk**: diversification ratios and marginal ES.
7. **Liquidity risk**: bid-ask adjusted VaR for illiquid positions.
8. **Counterparty credit risk**: CVA = $\mathbb{E}[(\text{Exposure})^+ \text{default}]$ simulated via Monte Carlo.
9. **Climate stress**: transition- and physical-risk scenarios on long-horizon portfolios.
10. **Insurance solvency**: SCR (Solvency Capital Requirement) under Solvency II's 99.5% 1-year VaR.

---

## 8.2.12 Exercises

### ★

1. Show that expected shortfall is coherent while VaR is not, giving a two-asset counterexample.
2. For $L \sim \mathcal{N}(\mu, \sigma^2)$, derive $\mathrm{ES}_\alpha = \mu + \sigma \phi(z)/(1-\alpha)$.
3. Show that VaR is monotone, positively homogeneous, and translation-invariant — but not subadditive in general.
4. Interpret the Basel traffic-light regions in terms of Kupiec p-values under $\alpha=0.99$.
5. For a Gaussian copula with correlation 0.5, compute $\lambda_U$.
6. Derive the Cornish-Fisher adjusted quantile for skew = 0, excess kurtosis = 3.

### ★★

7. Simulate 1000 days of $t_5$ returns and compare the coverage of historical vs. parametric-normal 99% VaR via Kupiec.
8. Fit a GARCH(1,1) to daily S&P returns and compute conditional 99% VaR; backtest with Christoffersen.
9. Derive the Hill estimator from the MLE of a Pareto tail and prove consistency as $k \to \infty, k/n \to 0$.
10. Simulate a 5-asset portfolio with heavy-tailed marginals and a Clayton copula; show that Gaussian-copula VaR underestimates 99% joint loss.
11. Implement Euler ES allocation for a portfolio of 10 assets and verify that contributions sum to total ES.
12. Build a reverse stress test: find the minimum-unlikelihood scenario producing a 10% portfolio loss, given a Gaussian factor model.

### ★★★

13. Prove the Artzner-Delbaen-Eber-Heath representation theorem: every coherent risk measure admits a dual representation $\rho(X) = \sup_{Q \in \mathcal Q}\mathbb{E}_Q[-X]$.
14. Prove the Fisher-Tippett theorem: the only possible non-degenerate limit distributions of sample maxima are GEV.
15. Derive the Pickands-Balkema-de Haan theorem: the distribution of excesses over a high threshold converges to GPD as the threshold approaches the upper endpoint.
16. Prove that ES is elicitable only jointly with VaR (Fissler-Ziegel 2016), hence the need for joint backtests.
17. Show that under a Gaussian copula with correlation $\rho < 1$, $\lambda_U = 0$, but under a $t$-copula with any $\rho$, $\lambda_U > 0$.
18. Prove that Euler allocation is the unique capital allocation rule satisfying the Denault fairness axiom under a coherent positively homogeneous risk measure.

---

*— End of Module 8.2. Next: Module 8.3, Algorithmic Trading Design End-to-End.*

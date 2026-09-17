# Module 7.5 — Volatility Modeling: ARCH, GARCH, Realized

*Subject 7, Module 5. Engle and Bollerslev revolutionized empirical finance by making volatility itself modelable.*

---

## Prerequisites

- **Module 7.4 (Time series)** — stationarity, ARMA, ACF.
- **Module 5.6 (Stochastic volatility)** — continuous-time analog.
- **Module 2.5 (Characteristic functions)** — normal and $t$ MLE.
- **Module 0.3 (Optimization)** — MLE via numerical methods.

---

## 7.5.1 Stylized facts about returns

Empirical regularities that a volatility model must reproduce:

1. **Heavy tails.** Return distributions have kurtosis $\gg 3$; tails fatter than Gaussian.
2. **Volatility clustering.** Large $|r_t|$ followed by large $|r_{t+1}|$. ACF of $|r_t|$ or $r_t^2$ is slow-decaying (long memory).
3. **Leverage effect.** Negative returns increase future volatility more than positive (Black 1976).
4. **Mean-reverting volatility.** Shocks die out over weeks-months, not instantaneously.
5. **Volatility risk premium.** Options routinely overprice realized vol (IV > RV on average).
6. **Fat-tailed even conditionally.** After normalizing by volatility, returns still have some excess kurtosis ($t$-distributed innovations).

ARCH/GARCH models capture stylized facts 1-2 directly; EGARCH adds 3; SV models (Module 5.6) add 4 smoothly.

---

## 7.5.2 ARCH($q$) (Engle 1982)

$$
r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad \sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2,
$$
with $z_t \sim \text{iid}(0, 1)$.

**Properties.**
- Conditional variance depends on past squared shocks.
- Marginal distribution has heavy tails (even with Gaussian $z_t$).
- $\mathbb{E}[\varepsilon_t^2] = \omega/(1 - \sum\alpha_i)$ if $\sum \alpha_i < 1$ (stationarity condition).

**Estimation.** Conditional MLE: $\ell_t = -\tfrac{1}{2}\log\sigma_t^2 - \tfrac{\varepsilon_t^2}{2\sigma_t^2}$.

**Diagnostic.** Squared standardized residuals $\hat z_t^2 = \varepsilon_t^2/\hat\sigma_t^2$ should be white noise (no residual ARCH). Ljung-Box on $\hat z_t^2$.

**Limitation.** ARCH often needs $q \ge 10$ lags to capture real-world persistence. Too many parameters. GARCH fixes this.

---

## 7.5.3 GARCH($p, q$) (Bollerslev 1986)

$$
\sigma_t^2 = \omega + \sum_{i=1}^p \beta_i \sigma_{t-i}^2 + \sum_{j=1}^q \alpha_j \varepsilon_{t-j}^2.
$$

**GARCH(1,1)** is the workhorse:
$$
\sigma_t^2 = \omega + \beta \sigma_{t-1}^2 + \alpha \varepsilon_{t-1}^2.
$$

**Stationarity**: $\alpha + \beta < 1$.

**Unconditional variance**: $\sigma^2 = \omega/(1 - \alpha - \beta)$.

**Persistence** = $\alpha + \beta$. Typical values on equity indices: 0.95–0.99 ("near-integrated GARCH"). Volatility shocks take months to decay.

**Forecasting.** $h$-step ahead:
$$
\mathbb{E}[\sigma_{t+h}^2 \mid \mathcal{F}_t] = \sigma^2 + (\alpha+\beta)^{h-1}(\sigma_{t+1}^2 - \sigma^2).
$$
Mean-reverts to long-run variance at geometric rate $\alpha + \beta$.

---

## 7.5.4 EGARCH (Nelson 1991) — leverage effect

$$
\log\sigma_t^2 = \omega + \beta \log\sigma_{t-1}^2 + \alpha |z_{t-1}| + \gamma z_{t-1}.
$$
Here $z_{t-1} = \varepsilon_{t-1}/\sigma_{t-1}$ is the standardized residual.

**Asymmetric news impact**. If $\gamma < 0$ (typical for equities), negative shocks ($z_{t-1} < 0$) produce larger volatility increase than positive shocks of same magnitude.

**Log-volatility formulation** ensures $\sigma_t^2 > 0$ without parameter constraints — easier estimation.

**GJR-GARCH** (Glosten-Jagannathan-Runkle 1993). Alternative asymmetric specification:
$$
\sigma_t^2 = \omega + \beta\sigma_{t-1}^2 + \alpha \varepsilon_{t-1}^2 + \gamma \varepsilon_{t-1}^2 \mathbb{1}\{\varepsilon_{t-1} < 0\}.
$$
Simpler interpretation than EGARCH.

**TGARCH, APARCH, FIGARCH** — other variants for specific empirical features (fractional integration, power transformations).

---

## 7.5.5 MLE and innovation distributions

**Gaussian QMLE.** Assume $z_t \sim \mathcal{N}(0, 1)$. Log-likelihood:
$$
\ell(\theta) = -\frac{1}{2}\sum_t \left[\log(2\pi \sigma_t^2) + \frac{\varepsilon_t^2}{\sigma_t^2}\right].
$$
Even if innovations are non-Gaussian, QMLE is consistent and asymptotically normal with correct sandwich SE.

**Student-$t$ QMLE.** Allow $z_t \sim t_\nu$ for $\nu > 2$:
$$
\ell(\theta, \nu) = \sum_t \log f_t(\varepsilon_t/\sigma_t) - \tfrac{1}{2}\log\sigma_t^2.
$$
Often $\nu \in [4, 10]$ for equity returns.

**Skewed-$t$** (Hansen 1994). Allows asymmetric fat tails.

**Generalized error distribution (GED)** is another common choice.

---

## 7.5.6 Multivariate GARCH

For $d$ assets, $\mathbf{r}_t = \boldsymbol\mu + \boldsymbol\varepsilon_t$ with $\boldsymbol\varepsilon_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0, \mathbf{H}_t)$.

**VEC-GARCH.** $\text{vech}(\mathbf{H}_t) = \omega + A \text{vech}(\varepsilon_{t-1}\varepsilon_{t-1}^\top) + B \text{vech}(\mathbf{H}_{t-1})$. Positive definiteness is hard.

**BEKK** (Engle-Kroner 1995). $\mathbf{H}_t = \mathbf{C}\mathbf{C}^\top + A(\varepsilon_{t-1}\varepsilon_{t-1}^\top) A^\top + B \mathbf{H}_{t-1} B^\top$. PSD guaranteed. Many parameters ($O(d^2)$).

**DCC-GARCH** (Engle 2002). Two steps:
1. Univariate GARCH on each series → get standardized residuals $z_{i,t}$.
2. Dynamic correlation: $Q_t = (1-a-b)\bar Q + a z_{t-1} z_{t-1}^\top + b Q_{t-1}$, then correlation $R_t = D_t^{-1} Q_t D_t^{-1}$.

Parsimonious: only 2 extra parameters regardless of $d$. Dominant choice for large portfolios.

---

## 7.5.7 Realized volatility

**High-frequency data paradigm** (Andersen-Bollerslev-Diebold-Labys 2001). For intraday returns $r_{t,j}$ at $n$ evenly-spaced instants:
$$
RV_t = \sum_{j=1}^n r_{t,j}^2.
$$

**Theorem (ABDL 2003).** Under standard SDE for log-price with bounded variation drift and square-integrable volatility, $RV_t \to \int_{t-1}^t \sigma_s^2 ds$ as $n \to \infty$.

**Microstructure noise.** $r_{t,j}^{obs} = r_{t,j}^{true} + \eta_{t,j}$. Sum of squares of noise dominates at ultra-high frequency. Solutions:
- **Sparse sampling**: use 5-minute returns.
- **Two-scale realized variance** (Zhang-Mykland-Aït-Sahalia 2005). Consistent under noise.
- **Realized kernel** (Barndorff-Nielsen-Hansen-Lunde-Shephard 2008). Bias-adjusted kernel estimator.
- **Pre-averaging** (Jacod-Li-Mykland-Podolskij-Vetter 2009).

**HAR model** (Corsi 2009). Heterogeneous Autoregressive:
$$
RV_t = c + \beta_d RV_{t-1} + \beta_w \overline{RV}_{t-5:t-1} + \beta_m \overline{RV}_{t-22:t-1} + \varepsilon_t.
$$
Daily + weekly + monthly component. Captures long memory parsimoniously. Often beats GARCH on forecast RMSE.

**Jump-robust variants.** Bi-power variation $BV$ cancels jumps; $Z$-test for jump detection. Median realized variance (Andersen-Dobrev-Schaumburg 2012).

---

## 7.5.8 Volatility forecasting: GARCH vs HAR vs implied

Empirical horse races (Andersen-Bollerslev, Engle, many others) show:
- **1-day horizon**: HAR on realized volatility wins by 10–30% RMSE over GARCH.
- **1-week horizon**: HAR still best.
- **1-month+**: options-implied volatility often best, especially in regime-changing markets.
- **Combination**: regress future RV on HAR + implied vol — both have independent information.

**Rough volatility insight** (Bayer-Friz-Gatheral 2016, Gatheral-Jaisson-Rosenbaum 2018). Log-RV behaves like fractional Brownian with Hurst $H \approx 0.1$. HAR is an approximate rough-vol model.

---

## 7.5.9 Python: volatility models

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

# ============================================================
# 1. GARCH(1,1) MLE estimation
# ============================================================
def garch11_loglik(params, r):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha+beta >= 1:
        return 1e10
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[0] = r.var()
    for t in range(1, n):
        sigma2[t] = omega + alpha*r[t-1]**2 + beta*sigma2[t-1]
    ll = -0.5*np.sum(np.log(2*np.pi*sigma2) + r**2/sigma2)
    return -ll

def fit_garch11(r):
    init = [0.01*r.var(), 0.1, 0.85]
    res = minimize(garch11_loglik, init, args=(r,), method='Nelder-Mead')
    return res.x

# Simulate GARCH(1,1)
np.random.seed(42)
n = 2000
omega_true, alpha_true, beta_true = 0.05, 0.1, 0.85
sigma2 = np.zeros(n)
r = np.zeros(n)
sigma2[0] = omega_true/(1 - alpha_true - beta_true)
for t in range(1, n):
    sigma2[t] = omega_true + alpha_true*r[t-1]**2 + beta_true*sigma2[t-1]
    r[t] = np.sqrt(sigma2[t])*np.random.randn()

omega_hat, alpha_hat, beta_hat = fit_garch11(r)
print(f"GARCH(1,1) estimates:")
print(f"  ω: true={omega_true:.4f}, est={omega_hat:.4f}")
print(f"  α: true={alpha_true:.4f}, est={alpha_hat:.4f}")
print(f"  β: true={beta_true:.4f}, est={beta_hat:.4f}")
print(f"  persistence: {alpha_hat+beta_hat:.4f}")

# ============================================================
# 2. GJR-GARCH for leverage
# ============================================================
def gjr_garch_loglik(params, r):
    omega, alpha, gamma, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha+gamma/2+beta >= 1:
        return 1e10
    n = len(r)
    sigma2 = np.zeros(n); sigma2[0] = r.var()
    for t in range(1, n):
        neg = (r[t-1] < 0).astype(float)
        sigma2[t] = omega + (alpha + gamma*neg)*r[t-1]**2 + beta*sigma2[t-1]
    return 0.5*np.sum(np.log(2*np.pi*sigma2) + r**2/sigma2)

# ============================================================
# 3. HAR-RV model
# ============================================================
def har_rv(rv_series):
    """Fit HAR on a 1D array of daily RVs."""
    n = len(rv_series)
    y = rv_series[22:]
    rv_d = rv_series[21:n-1]  # previous day
    rv_w = np.array([rv_series[i-5:i].mean() for i in range(21, n-1)])
    rv_m = np.array([rv_series[i-22:i].mean() for i in range(21, n-1)])
    X = np.column_stack([np.ones(len(y)), rv_d, rv_w, rv_m])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, X @ beta

# Simulate RV (approximate rough vol)
np.random.seed(1)
rv_sim = np.abs(np.random.randn(500)) * 0.01  # realistic scale
for i in range(1, 500): rv_sim[i] = 0.5*rv_sim[i-1] + 0.5*rv_sim[i]  # persistence
beta_har, _ = har_rv(rv_sim)
print(f"\nHAR-RV coefficients: {beta_har.round(4)}")
print("  (intercept, daily, weekly, monthly)")

# ============================================================
# 4. DCC-GARCH (2-asset sketch)
# ============================================================
def dcc_univariate_garch(r):
    """Step 1 of DCC: univariate GARCH on each series, return standardized resids."""
    omega, alpha, beta = fit_garch11(r)
    n = len(r)
    sigma2 = np.zeros(n); sigma2[0] = r.var()
    for t in range(1, n):
        sigma2[t] = omega + alpha*r[t-1]**2 + beta*sigma2[t-1]
    z = r/np.sqrt(sigma2)
    return z, sigma2

np.random.seed(2)
n = 1500
r1 = np.random.randn(n)
r2 = 0.5*r1 + np.sqrt(1-0.25)*np.random.randn(n)  # correlated
z1, s21 = dcc_univariate_garch(r1)
z2, s22 = dcc_univariate_garch(r2)
rho_sample = np.corrcoef(z1, z2)[0,1]
print(f"\nDCC-GARCH step 1: standardized residual correlation = {rho_sample:.4f}")
print("  (Step 2 would fit dynamic correlation process)")

# ============================================================
# 5. Realized variance with simple noise robustness
# ============================================================
def realized_variance_simple(intraday_prices):
    log_p = np.log(intraday_prices)
    ret = np.diff(log_p)
    return np.sum(ret**2)

def two_scale_rv(intraday_prices, K=10):
    """Zhang-Mykland-Aït-Sahalia two-scale realized variance."""
    n = len(intraday_prices)
    log_p = np.log(intraday_prices)
    # K subsamples
    rv_sub = np.mean([np.sum(np.diff(log_p[k::K])**2) for k in range(K)])
    # All-data RV (contaminated by noise)
    rv_all = np.sum(np.diff(log_p)**2)
    # Combine to remove bias
    bias_corr = rv_sub - (n/K - 1)/n * rv_all
    return bias_corr

# Simulate: true volatility σ₀ = 0.2 annually, n = 390 5-min intraday bars
np.random.seed(0)
sigma0 = 0.2
dt = 1/(252*78)  # 5-min in 252-day year (78 = 6.5h/5min)
n_intra = 78
true_vol = np.full(n_intra, sigma0*np.sqrt(252))
ret = np.random.randn(n_intra) * true_vol*np.sqrt(dt)
# Add microstructure noise
noise = 0.001*np.random.randn(n_intra + 1)
log_p = np.concatenate([[np.log(100)], np.log(100) + np.cumsum(ret)])
log_p_obs = log_p + noise
p_obs = np.exp(log_p_obs)

rv_simple = realized_variance_simple(p_obs)
rv_two_scale = two_scale_rv(p_obs, K=5)
true_iv = sigma0**2 * 1/252  # daily IV
print(f"\nRealized volatility (true daily IV: {true_iv:.2e}):")
print(f"  Simple RV: {rv_simple:.2e} (biased up by noise)")
print(f"  Two-scale: {rv_two_scale:.2e}")
```

---

## 7.5.10 [QUANT APPLICATIONS]

1. **Risk management.** GARCH VaR for equity portfolios; DCC-GARCH for multi-asset.
2. **Option pricing implied parameters.** Filtered historical simulation uses GARCH-standardized residuals.
3. **Dynamic hedging.** GARCH-forecast volatility feeds into option delta/vega models.
4. **Vol arbitrage.** Systematic selling of overpriced implied vs. realized; GARCH + HAR for realized forecasts.
5. **Asset allocation.** Dynamic Sharpe ratio maximization using time-varying covariance (DCC).
6. **Macro risk budgeting.** Fama-French factor covariance via multivariate GARCH.
7. **VaR backtesting.** Kupiec and conditional coverage tests on GARCH VaR.
8. **Credit risk.** CDS spread volatility modeled via GARCH; used in capital calculation.
9. **Execution risk.** Real-time vol forecasting for trading system throttling.
10. **Stress testing.** Forward-looking scenarios using GARCH-calibrated noise.

---

## 7.5.11 Exercises

**★ (concept drills).**
1. State and prove GARCH(1,1) stationarity condition $\alpha + \beta < 1$.
2. Show that GARCH(1,1) has kurtosis $> 3$ even with Gaussian innovations.
3. Derive the $h$-step variance forecast for GARCH(1,1).
4. Explain the leverage effect. Why does GARCH fail to capture it? How does EGARCH fix it?
5. Why is the realized variance estimator noisy at ultra-high frequency?
6. State the HAR-RV model. Why does it have three components?

**★★ (calculation).**
7. Fit GARCH(1,1) to SPX daily returns. Report $\omega, \alpha, \beta$ and persistence.
8. Compare GARCH vs GJR-GARCH vs EGARCH on SPX. Which has best likelihood? Compute AIC/BIC.
9. Student-$t$ MLE. Fit GARCH(1,1) with $t_\nu$ innovations. Estimate $\nu$; interpret.
10. Forecast variance 5 days ahead using GARCH(1,1). Compare to realized.
11. DCC-GARCH on 3 assets. Implement and fit. Plot time-varying correlations.
12. HAR-RV on SPY 5-min data. Compare to GARCH forecast. Which is better 1-day ahead?
13. Realized variance with microstructure noise. Simulate noisy process; compare simple RV, 5-min RV, two-scale RV.

**★★★ (open / research).**
14. **Rough GARCH** (Bennedsen-Lunde-Pakkanen 2017). Extend HAR with fractional noise. Does it beat HAR on out-of-sample RMSE?
15. **HEAVY model** (Shephard-Sheppard 2010). Combines daily returns and RV in a joint model. Implement and compare.
16. **MIDAS regression for volatility** (Ghysels). Model low-frequency variance as weighted sum of high-frequency squared returns. Apply.
17. **Implied vol vs GARCH for VaR.** Construct VaR using both sources; backtest coverage on SPX. Which fails more often?
18. **GARCH with transition** (Smooth Transition GARCH, Markov-switching GARCH). Test on crisis-period data.

---

*— End of Module 7.5. Next: Module 7.6, Factor Models and Cross-Sectional Regressions.*

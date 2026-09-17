# Module 7.4 — Time Series Foundations

*Subject 7, Module 4. Stationarity, ARMA, spectral analysis, unit roots, cointegration — the grammar of time.*

---

## Prerequisites

- **Module 2.6 (Martingales)** — random walks, difference between martingales and stationary processes.
- **Module 3.1 (Brownian motion)** — limit of random walk.
- **Module 0.5.6 (Fourier transforms)** — spectral density is Fourier transform of autocovariance.
- **Module 2.5 (Characteristic functions)** — spectral representation.

---

## 7.4.1 Stationarity

**Strict stationarity.** $(X_t)$ is strictly stationary if for any $t_1, \ldots, t_k$ and $h$:
$$
(X_{t_1}, \ldots, X_{t_k}) \stackrel{d}{=} (X_{t_1 + h}, \ldots, X_{t_k + h}).
$$

**Weak (covariance, second-order) stationarity.** Mean $\mu = \mathbb{E} X_t$ and autocovariance $\gamma(h) = \text{Cov}(X_t, X_{t+h})$ don't depend on $t$.

**Ergodicity.** Time averages converge to population averages: $\frac{1}{n}\sum X_t \to \mu$. Stronger than stationarity but the right framework for statistical inference.

**Financial returns** are weakly stationary at low to moderate frequencies; **prices** are non-stationary (have unit roots). The first step in any time-series analysis is differencing / return-taking to achieve stationarity.

---

## 7.4.2 Autocorrelation function (ACF) and partial ACF

**Autocorrelation function**: $\rho(h) = \gamma(h)/\gamma(0)$.

**Sample ACF**: $\hat\rho(h) = \hat\gamma(h)/\hat\gamma(0)$, $\hat\gamma(h) = \frac{1}{n}\sum_{t=1}^{n-h}(X_t - \bar X)(X_{t+h} - \bar X)$.

**Bartlett's formula** for ACF confidence bands under white noise: $\hat\rho(h) \sim \mathcal{N}(0, 1/n)$ asymptotically.

**Partial autocorrelation**: correlation between $X_t$ and $X_{t+h}$ after removing linear effect of $X_{t+1}, \ldots, X_{t+h-1}$. Computed via Yule-Walker equations or Durbin-Levinson recursion.

**Diagnostic use.**
- AR($p$): ACF decays exponentially, PACF cuts off at $p$.
- MA($q$): ACF cuts off at $q$, PACF decays.
- ARMA($p, q$): both decay.

---

## 7.4.3 ARMA models

**AR($p$)** (autoregressive): $X_t = \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \varepsilon_t$.

**Causality (stability).** AR roots outside unit circle $\iff$ $X_t = \sum_{j \ge 0} \psi_j \varepsilon_{t-j}$ (one-sided MA representation). Characteristic polynomial $\phi(z) = 1 - \phi_1 z - \cdots - \phi_p z^p$ has all roots $|z| > 1$.

**MA($q$)** (moving average): $X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$.

**Invertibility.** MA roots outside unit circle $\iff$ $\varepsilon_t = \sum_{j \ge 0} \pi_j X_{t-j}$ (can invert to recover shocks).

**ARMA($p, q$)**: $\phi(L) X_t = \theta(L) \varepsilon_t$ where $L$ is the lag operator.

**Wold decomposition.** Every stationary $X_t$ has $X_t = V_t + \sum_{j \ge 0} \psi_j \varepsilon_{t-j}$ where $V_t$ is deterministic and the MA part has orthogonal innovations.

---

## 7.4.4 Estimation of ARMA models

**Yule-Walker for AR(p).** System of equations $\gamma(h) = \phi_1 \gamma(h-1) + \cdots + \phi_p \gamma(h-p)$ for $h = 1, \ldots, p$.

**Maximum Likelihood**. Assuming Gaussian innovations, MLE via Kalman filter (exact for ARMA) or conditional likelihood (ignore first $p$ observations).

**Box-Jenkins methodology** (1970):
1. Identify $(p, q)$ via ACF/PACF.
2. Estimate parameters.
3. Diagnose residuals: should be white noise. Ljung-Box test, ACF of residuals.
4. If diagnostics fail, refine model.

**Information criteria.**
- AIC = $-2\ell + 2k$
- BIC = $-2\ell + k\log n$

Both balance fit against complexity. AIC is asymptotically optimal for prediction; BIC is asymptotically consistent for model selection (picks true $(p,q)$ if true model is in the class).

---

## 7.4.5 Unit roots and random walks

**Random walk**: $X_t = X_{t-1} + \varepsilon_t$. Non-stationary; $\text{Var}(X_t) = t\sigma^2$ grows.

**Unit root** = AR(1) with $\phi = 1$. The characteristic polynomial has a root at $z = 1$, not outside unit circle.

**ARIMA($p, d, q$)**: difference $d$ times, then fit ARMA. $d = 1$ typical for prices/levels, $d = 0$ for returns.

**Dickey-Fuller test.** Test $H_0: \phi = 1$ in $X_t = \phi X_{t-1} + \varepsilon_t$. Non-standard distribution (not Gaussian because limit is functional of Brownian).

**Augmented Dickey-Fuller (ADF).** Test in:
$$
\Delta X_t = \alpha + \delta X_{t-1} + \sum_{j=1}^p \gamma_j \Delta X_{t-j} + \varepsilon_t.
$$
$H_0: \delta = 0$ (unit root). If rejected, stationarity.

**KPSS test.** Reverses hypothesis: $H_0$: stationary, $H_1$: unit root. Used as complementary test.

**Phillips-Perron.** ADF alternative robust to heteroskedasticity and autocorrelation.

---

## 7.4.6 Spectral analysis

**Spectral density** (for weakly stationary $X_t$):
$$
f(\omega) = \frac{1}{2\pi}\sum_{h=-\infty}^\infty \gamma(h) e^{-ih\omega}.
$$
Inverse: $\gamma(h) = \int_{-\pi}^\pi f(\omega) e^{ih\omega} d\omega$.

**Periodogram** (sample spectral density):
$$
I_n(\omega_k) = \frac{1}{n}\left|\sum_{t=1}^n X_t e^{-it\omega_k}\right|^2, \qquad \omega_k = 2\pi k/n.
$$

**Consistency issue.** $I_n(\omega)$ is an unbiased but **not consistent** estimator of $f(\omega)$. Variance is $f^2(\omega)$ regardless of $n$. Smoothing by averaging over nearby frequencies (Bartlett window, Parzen, Daniell) gives consistent estimators.

**Wiener-Khintchine theorem.** For stationary $X_t$, the spectral density is the Fourier transform of the autocovariance.

**Applications.**
- Identify periodicities (business cycles, seasonality).
- Filter design: low-pass, high-pass, band-pass filters.
- Wavelet analysis for non-stationary signals.

---

## 7.4.7 Cointegration (Engle-Granger 1987)

**Setup.** Two nonstationary series $X_t, Y_t$ (both $I(1)$). They are **cointegrated** if there exists $\beta$ such that $Z_t = Y_t - \beta X_t$ is stationary ($I(0)$).

**Economic meaning.** $X$ and $Y$ share a common stochastic trend. Linear combination cancels the trend.

**Example.** Stock prices of two firms in same sector often cointegrate; their spread is mean-reverting.

**Engle-Granger two-step.**
1. Regress $Y_t$ on $X_t$: $\hat\beta$ from OLS.
2. Test whether residuals are stationary (ADF on residuals).

**Error-correction representation.** If $Y_t, X_t$ cointegrate with coefficient $\beta$:
$$
\Delta Y_t = \alpha (Y_{t-1} - \beta X_{t-1}) + \sum \gamma_j \Delta X_{t-j} + \sum \delta_j \Delta Y_{t-j} + \varepsilon_t.
$$
$\alpha < 0$ is the error-correction coefficient: deviations from long-run equilibrium are corrected at rate $|\alpha|$.

**Johansen test.** Multivariate generalization. VAR($p$) can be rewritten as:
$$
\Delta X_t = \Pi X_{t-1} + \sum \Gamma_j \Delta X_{t-j} + \varepsilon_t.
$$
Rank of $\Pi$ = number of cointegrating relationships. Trace and max-eigenvalue tests for rank.

**Cointegration for pairs trading.** If $Y = \beta X + Z$ with $Z$ stationary and mean-reverting, go long $Y$ / short $\beta X$ when $Z < 0$ and reverse when $Z > 0$. Classic stat-arb strategy.

---

## 7.4.8 Vector autoregression (VAR)

Multivariate generalization of AR. VAR($p$):
$$
\mathbf{X}_t = \mathbf{c} + \Phi_1 \mathbf{X}_{t-1} + \cdots + \Phi_p \mathbf{X}_{t-p} + \boldsymbol{\varepsilon}_t.
$$

**Estimation.** OLS equation-by-equation (equivalent to GLS when errors are contemporaneously uncorrelated — and OLS is always consistent).

**Granger causality.** $X$ Granger-causes $Y$ if past $X$ helps predict $Y$ beyond past $Y$'s own information. Standard $F$-test in the VAR context.

**Impulse response function** (IRF): how $\mathbf{X}$ evolves after a one-unit shock to $\varepsilon_j$. Computed by recursion.

**Variance decomposition**: fraction of $k$-step forecast error variance of $Y$ attributable to shocks in $X$ vs its own past.

**Structural VAR.** Identify underlying "structural" shocks via orthogonalization (Cholesky), long-run restrictions (Blanchard-Quah), or sign restrictions.

---

## 7.4.9 Non-stationarity and change points

**Change point detection.** $X_t$ may have a mean or variance shift at unknown $\tau$. CUSUM, Bai-Perron methods detect single or multiple change points.

**Regime switching** (Hamilton 1989). Allow hidden Markov state $S_t$ affecting mean/variance. Viterbi algorithm recovers most-likely state sequence; EM estimates parameters.

**Structural breaks.** Financial crises, policy changes, index reconstitutions. Chow test for known break date; Andrews supremum-of-F test for unknown.

---

## 7.4.10 Python: time series toolkit

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import periodogram

# ============================================================
# 1. Simulate an AR(2) process and plot ACF / PACF
# ============================================================
np.random.seed(42)
n = 500
phi = [0.6, -0.3]
eps = np.random.randn(n)
X = np.zeros(n)
for t in range(2, n):
    X[t] = phi[0]*X[t-1] + phi[1]*X[t-2] + eps[t]

# Fit ARIMA
model = ARIMA(X, order=(2, 0, 0)).fit()
print(f"AR(2) true φ=(0.6, -0.3)")
print(f"      estimated: ({model.params[1]:.3f}, {model.params[2]:.3f})")

# ============================================================
# 2. Unit root test
# ============================================================
# Random walk
rw = np.cumsum(np.random.randn(500))
adf = adfuller(rw)
print(f"\nADF on random walk: stat={adf[0]:.4f}, p-val={adf[1]:.4f}")
print(f"  (expect p-val high, do not reject unit root)")

adf_returns = adfuller(np.diff(rw))
print(f"ADF on differences: stat={adf_returns[0]:.4f}, p-val={adf_returns[1]:.4f}")
print(f"  (expect p-val low, reject unit root)")

# ============================================================
# 3. Periodogram and spectral density
# ============================================================
f, P = periodogram(X, fs=1.0)
# peak frequency
peak = f[np.argmax(P)]
print(f"\nPeak frequency of AR(2): {peak:.3f} (dominant cycle length {1/peak:.1f})")

# ============================================================
# 4. Cointegration test
# ============================================================
np.random.seed(0)
n = 500
Xt = np.cumsum(np.random.randn(n))
Yt = 0.5 * Xt + np.cumsum(0.3*np.random.randn(n)) + np.random.randn(n)

# Engle-Granger: regress Y on X, test residuals for stationarity
from scipy.stats import linregress
slope, intercept, _, _, _ = linregress(Xt, Yt)
resid = Yt - (slope*Xt + intercept)
adf_resid = adfuller(resid)
print(f"\nEngle-Granger cointegration test:")
print(f"  slope = {slope:.4f}, residual ADF p = {adf_resid[1]:.4f}")
if adf_resid[1] < 0.05:
    print("  → Cointegrated")
else:
    print("  → Not cointegrated (residuals still nonstationary)")

# ============================================================
# 5. VAR model for 3 series
# ============================================================
np.random.seed(1)
n = 300
e = np.random.randn(n, 3)
data = np.zeros((n, 3))
for t in range(1, n):
    data[t, 0] = 0.5*data[t-1, 0] + 0.2*data[t-1, 1] + e[t, 0]
    data[t, 1] = -0.3*data[t-1, 0] + 0.4*data[t-1, 1] + 0.1*data[t-1, 2] + e[t, 1]
    data[t, 2] = 0.2*data[t-1, 2] + e[t, 2]

df = pd.DataFrame(data, columns=['a', 'b', 'c'])
var_model = VAR(df).fit(1)
print(f"\nVAR(1) estimates (first row):")
print(var_model.coefs[0].round(3))
# Granger causality
gc = var_model.test_causality('a', ['b'])
print(f"Granger causality b → a: p = {gc.pvalue:.4f}")

# ============================================================
# 6. Forecasting with ARIMA
# ============================================================
forecast = model.get_forecast(steps=20)
conf_int = forecast.conf_int()
print(f"\n20-step forecast (first 5):")
print(np.round(forecast.predicted_mean[:5], 4))
print(f"Conf intervals (first 5):")
print(np.round(conf_int[:5], 4))
```

---

## 7.4.11 [QUANT APPLICATIONS]

1. **Pairs / stat-arb trading.** Cointegrated pairs mean-revert; signal derived from spread z-score.
2. **Yield curve dynamics.** VAR models on principal components (level, slope, curvature) for rate forecasting.
3. **Macro nowcasting.** ARIMA, VAR, and dynamic factor models for GDP, inflation, unemployment.
4. **Volatility forecasting.** ARMA on log-realized-vol (then Module 7.5 GARCH).
5. **Risk factor dynamics.** VAR for Fama-French factors; Granger-causality tests for style drift.
6. **Order imbalance modeling.** ACF of trade signs shows long memory; fit ARFIMA.
7. **Economic regime switching.** Hamilton's 2-state Markov switching for recession/expansion.
8. **Spectral analysis of returns.** Detect long memory, periodicities in strategies.
9. **VIX term-structure modeling.** Cointegration between VIX futures; mean-reverting spread trading.
10. **Treasury/bond spread modeling.** Error-correction dynamics of cross-country spreads, TIPS-nominal breakevens.

---

## 7.4.12 Exercises

**★ (concept drills).**
1. Prove that stationary + ergodic $\implies$ time averages → ensemble averages (SLLN for stationary processes).
2. Derive the Yule-Walker equations for AR($p$).
3. Show that AR(1) is stationary iff $|\phi| < 1$.
4. Derive Wold decomposition in a simple form for AR(1).
5. Explain the intuition behind ADF test. Why is the distribution non-standard?
6. What is cointegration? Why is it different from correlation?
7. Give an example where two series correlate but don't cointegrate, and vice versa.

**★★ (calculation).**
8. Simulate AR(2) with $\phi = (0.7, -0.5)$. Plot theoretical ACF $\rho(h)$ using the Yule-Walker recursion; compare to sample ACF.
9. Fit ARIMA(1,1,1) to a random-walk + MA(1) noise. Does BIC select the correct order?
10. Derive impulse response function for AR(1): $\partial X_{t+h}/\partial\varepsilon_t = \phi^h$.
11. Estimate a VAR on two commodity futures. Test Granger causality in both directions.
12. Engle-Granger on a simulated cointegrated pair. Vary noise-to-signal ratio; when does the test fail?
13. Johansen test for rank determination: simulate 3-dim VAR with rank 1 cointegration; apply Johansen.

**★★★ (open / research).**
14. **ARFIMA** with Hurst parameter estimation. Fit on log-realized-volatility series; compare to rough-vol Hurst estimates.
15. **Multiple structural breaks** (Bai-Perron). Implement for a stock-return series; identify breaks around crises.
16. **Wavelet-based non-stationary analysis.** Apply continuous wavelet transform to VIX; identify regime changes.
17. **Dynamic factor models** (Stock-Watson). Fit to a 50-variable macro dataset; extract common factors.
18. **Regime-switching VAR** (Markov switching VAR). Fit to S&P 500 returns with bull/bear state; forecast expected return conditional on state.

---

*— End of Module 7.4. Next: Module 7.5, Volatility Modeling (ARCH/GARCH/Realized).*

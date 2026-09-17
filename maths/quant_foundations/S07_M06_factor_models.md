# Subject 7, Module 6: Factor Models and Cross-Sectional Regressions

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"There is but one fundamental question: why do some assets earn higher returns than others? Every factor model is a tentative answer."* — Eugene Fama

---

## 7.6.0 Where We Are

This module sits at the intersection of asset pricing theory (Subject 5) and statistical estimation (Modules 7.1–7.5). The central claim of factor models is that expected excess returns are cross-sectionally explained by exposure to a small number of systematic *factors*. Mathematically:

$$\mathbb{E}[R^e_i] = \beta_i^\top \lambda \qquad \text{for all assets } i=1,\dots,N,$$

where $\beta_i \in \mathbb{R}^K$ are factor loadings and $\lambda \in \mathbb{R}^K$ are risk premia. This is a *linear restriction on the cross-section of expected returns* — remarkably strong, and testable.

### Prerequisites

- **Module 5.1** (No-Arbitrage and Fundamental Theorems): SDF representation $\mathbb{E}[MR^e]=0$ and the link between SDF and beta representations.
- **Module 5.4** (Portfolio Theory context): mean-variance efficient portfolio geometry.
- **Module 7.1** (Linear Regression): OLS under heteroskedasticity, HAC standard errors.
- **Module 7.4** (Time Series Foundations): stationarity, AR dynamics for factor returns.
- **Module 0.2** (Linear Algebra): eigendecomposition (for PCA), SVD.
- **Module 0.3** (Optimization): constrained least-squares for portfolio construction.

### Plan

1. CAPM: the single-factor benchmark, derivation, tests (§7.6.1).
2. APT and general multi-factor pricing (§7.6.2).
3. Fama–French 3-factor, 5-factor, Carhart momentum (§7.6.3).
4. Fama–MacBeth two-pass regressions and errors-in-variables (§7.6.4).
5. Statistical factors via PCA and asymptotic PCA (§7.6.5).
6. Dynamic factor models, Bai–Ng factor number (§7.6.6).
7. Barra-style fundamental risk models (§7.6.7).
8. Characteristic-based anomalies and the "factor zoo" (§7.6.8).
9. Machine learning: autoencoders, instrumented PCA, stochastic discount factor via ML (§7.6.9).
10. Python: Fama-French replication, FM two-pass, PCA factors, IPCA sketch (§7.6.10).
11. Quant applications (§7.6.11).
12. Exercises (§7.6.12).

---

## 7.6.1 The Capital Asset Pricing Model

### Setup

Consider $N$ risky assets with excess returns $R^e_i = R_i - r_f$, and a riskless asset with return $r_f$. Suppose all investors are mean-variance optimizers with the same beliefs about the joint distribution of returns. Then the **market portfolio** $w^M$ — the value-weighted portfolio of all assets — is mean-variance efficient.

### Theorem 7.6.1 (Sharpe–Lintner CAPM).

*Under the assumptions above, for every asset $i$:*
$$\mathbb{E}[R^e_i] = \beta_i^M \cdot \mathbb{E}[R^e_M], \qquad \beta_i^M = \frac{\mathrm{Cov}(R^e_i, R^e_M)}{\mathrm{Var}(R^e_M)}.$$

**Proof sketch.** Since $w^M$ is on the tangent portfolio ray from $(0,r_f)$ to the efficient frontier, the first-order condition $\Sigma w^M = c \mu^e$ for some $c>0$ holds (where $\mu^e$ is the vector of expected excess returns and $\Sigma$ is the covariance of excess returns). Multiplying by $e_i^\top$ gives $\mathrm{Cov}(R^e_i, R^e_M) = c \mu^e_i$. For $i=M$: $\mathrm{Var}(R^e_M) = c \mu^e_M$. Dividing: $\mu^e_i = \beta_i^M \mu^e_M$.

### SDF representation

Any linear factor model of the form $\mathbb{E}[R^e_i] = \beta_i^\top \lambda$ has an equivalent SDF representation $M = a - b^\top f$ where $f$ is the vector of factor returns. Specifically (Cochrane 2005):
$$\mathbb{E}[M R^e_i] = 0 \iff \mathbb{E}[R^e_i] = \beta_i^\top \lambda,$$
with $b = \mathrm{Var}(f)^{-1} \lambda / \mathbb{E}[M]$ and $\beta_i = \mathrm{Var}(f)^{-1} \mathrm{Cov}(f, R^e_i)$.

### Testing the CAPM: time-series regression

For each asset $i$ run the time-series regression
$$R^e_{i,t} = \alpha_i + \beta_i R^e_{M,t} + \varepsilon_{i,t}, \qquad t=1,\dots,T.$$
The CAPM predicts $\alpha_i = 0$ for every asset. The **Gibbons–Ross–Shanken (GRS, 1989)** statistic tests this joint hypothesis:
$$\mathrm{GRS} = \frac{T-N-1}{N} \cdot \frac{\hat\alpha^\top \hat\Sigma^{-1} \hat\alpha}{1 + \hat\mu_M^2/\hat\sigma_M^2} \sim F_{N,T-N-1}$$
under normality. Intuitively, the numerator measures the squared distance of the vector of alphas from zero in Mahalanobis geometry; the denominator corrects for the sample Sharpe of the market factor.

### Empirical verdict

The CAPM is empirically rejected: small-cap stocks, value stocks, and momentum winners earn returns unexplained by market beta. Fama–French (1992) documented a flat cross-sectional relation between $\beta^M$ and average returns, with **size** (market cap) and **book-to-market** ratio dominating $\beta^M$ as predictors.

---

## 7.6.2 Arbitrage Pricing Theory

Ross (1976) replaced CAPM's equilibrium derivation with a no-arbitrage argument. Suppose returns follow a $K$-factor structure:
$$R^e_i = \alpha_i + \beta_i^\top f + \varepsilon_i, \qquad \mathbb{E}[\varepsilon_i]=0, \; \mathrm{Cov}(f, \varepsilon_i)=0, \; \mathrm{Cov}(\varepsilon_i, \varepsilon_j)=\sigma_i^2 \delta_{ij}.$$

### Theorem 7.6.2 (APT, approximate).

*If the above factor structure holds and no asymptotic arbitrage exists, then there exist $\lambda_0, \lambda_1,\dots,\lambda_K$ such that*
$$\sum_i (\mathbb{E}[R^e_i] - \lambda_0 - \beta_i^\top \lambda)^2 \le c$$
*for a constant independent of $N$. In particular, pricing errors average to zero.*

**Sketch.** For any zero-net-investment, zero-systematic-risk portfolio $w$ (i.e., $w^\top 1 = 0$, $w^\top \beta = 0$) with $\|w\|_2 = 1$, idiosyncratic risk $w^\top \Sigma_\varepsilon w \le \max_i \sigma_i^2 / N \to 0$. So the portfolio's payoff is approximately riskless, and no-arbitrage forces its expected return to $r_f$. Summing such portfolios over an orthogonal basis yields the APT restriction.

APT is a statement about pricing errors in large economies, not a sharp equilibrium restriction. It motivates *many* factor specifications rather than singling out the market.

### Exact factor pricing vs. approximate

- *Exact APT*: $\alpha_i = 0$ for all $i$ — requires an exact factor structure.
- *Approximate APT*: $\alpha_i$ bounded in $\ell_2$ — holds under mild idiosyncratic conditions.

---

## 7.6.3 Fama–French and Carhart Models

### The 3-factor model (Fama–French 1993)

$$R^e_i = \alpha_i + \beta_i^M \, \text{MKT}_t + \beta_i^{SMB} \, \text{SMB}_t + \beta_i^{HML} \, \text{HML}_t + \varepsilon_{i,t}.$$

- **MKT**: excess return on the value-weighted market portfolio.
- **SMB** (Small Minus Big): return on a long-small, short-big portfolio.
- **HML** (High Minus Low book-to-market): return on a long-value, short-growth portfolio.

Construction: at the end of each June, stocks are sorted into 2 size groups × 3 B/M groups. SMB is the average of the three small-cap portfolios minus the average of the three big-cap portfolios; HML is the average of the two high B/M portfolios minus the average of the two low B/M portfolios.

### Carhart momentum (1997)

Add a fourth factor **UMD** (Up Minus Down), long the past-12-month winners and short the past-12-month losers (formed monthly, skipping the most recent month to avoid short-term reversal).

### 5-factor model (Fama–French 2015)

Extend with:
- **RMW** (Robust Minus Weak profitability).
- **CMA** (Conservative Minus Aggressive investment).

The 5-factor model better absorbs the value premium, which becomes redundant in some specifications once RMW and CMA are included. It, however, fails to capture momentum — practitioners often add UMD to get a 6-factor model.

### Hou–Xue–Zhang q-factor model

An alternative grounded in investment-based asset pricing: market, size, investment, profitability, and (in the q5 version) expected growth.

### Factor spanning tests

To test whether a new factor $g_t$ adds pricing power beyond an existing set $f_t$, regress $g_t$ on $f_t$:
$$g_t = a + b^\top f_t + u_t.$$
If $a=0$, $g_t$ is *spanned* (offers no pricing improvement). Barillas–Shanken (2017) develop a Bayesian framework that directly compares factor models via marginal likelihoods and posterior probabilities.

---

## 7.6.4 Fama–MacBeth Two-Pass Regression

### The method

**Pass 1 (time series).** For each asset $i$, estimate factor loadings:
$$R^e_{i,t} = \alpha_i + \beta_i^\top f_t + \varepsilon_{i,t}, \qquad t=1,\dots,T.$$
Obtain $\hat\beta_i$.

**Pass 2 (cross section).** For each period $t$, run a cross-sectional regression:
$$R^e_{i,t} = \gamma_{0,t} + \hat\beta_i^\top \gamma_t + \eta_{i,t}, \qquad i=1,\dots,N.$$
Obtain $\{\hat\gamma_t\}_{t=1}^T$.

**Estimate premia and standard errors.**
$$\hat\lambda = \frac{1}{T}\sum_t \hat\gamma_t, \qquad \widehat{\mathrm{Var}}(\hat\lambda) = \frac{1}{T^2}\sum_t (\hat\gamma_t - \hat\lambda)(\hat\gamma_t - \hat\lambda)^\top.$$
This Fama–MacBeth variance cleverly avoids the need to model contemporaneous cross-sectional correlation of residuals — temporal independence of $\hat\gamma_t$ is all it needs. If $\hat\gamma_t$ is autocorrelated, Newey–West corrections apply.

### Errors-in-variables and the Shanken correction

Because $\hat\beta_i$ is estimated, the second-pass regression suffers from attenuation bias. Shanken (1992) derived the asymptotic correction:
$$\mathrm{Var}(\hat\lambda) = \frac{1}{T}\left[(1 + \lambda^\top \Sigma_f^{-1} \lambda)(\Sigma_\beta + \Sigma_\eta) + \Sigma_f\right],$$
where $\Sigma_\beta = (B^\top B/N)^{-1} B^\top \Sigma B (B^\top B/N)^{-1}$ is the asymptotic covariance of the second-pass slope absent EIV, $\Sigma_\eta$ reflects residual cross-sectional variance, and the correction factor $(1+\lambda^\top \Sigma_f^{-1}\lambda)$ is bounded by the tangency portfolio's squared Sharpe.

Kan–Robotti–Shanken (2013) extend the corrections to misspecified models; Kleibergen (2009) and Bryzgalova (2016) address weak-factor issues (when $\beta$ is close to zero, inference on $\lambda$ is severely distorted).

### GMM unification

All of this can be cast as GMM with moment conditions
$$g_T(\theta) = \mathbb{E}\left[\begin{pmatrix} R^e_t - \beta f_t - \alpha \\ (R^e_t - \beta f_t - \alpha)\otimes f_t \\ \beta^\top \lambda - \mathbb{E}[R^e] \end{pmatrix}\right] = 0.$$
Hansen's (1982) $J$-statistic tests over-identifying restrictions, the analog of the GRS test.

---

## 7.6.5 Statistical Factors via PCA

### PCA on returns

Let $R$ be the $T \times N$ panel of returns. Let $S = R^\top R / T$ be the sample second moment. The principal components are the eigenvectors of $S$:
$$S v_k = \sigma_k^2 v_k, \qquad \sigma_1^2 \ge \sigma_2^2 \ge \cdots$$

The first $K$ PCs span the factor space *from within returns themselves* — no theoretical motivation needed. Chamberlain–Rothschild (1983) showed that in an approximate factor model where the idiosyncratic covariance has bounded eigenvalues, the top $K$ eigenvalues of $\Sigma$ diverge with $N$, while all others remain bounded. This justifies PCA as an *asymptotic* factor-extraction method.

### Asymptotic PCA (Connor–Korajczyk 1986)

When $T < N$ (typical in asset pricing with many stocks), form the $T \times T$ Gram matrix $G = R R^\top / N$. Eigendecompose $G$ to get factor realizations directly; recover loadings via cross-sectional regression. Computationally this saves work when $N \gg T$.

### Bai–Ng (2002) factor number

Choose $K$ to minimize an information criterion:
$$\mathrm{IC}_p(K) = \log V(K, \hat F^K) + K \cdot p(N,T),$$
where $V(K,F) = (NT)^{-1}\sum_{i,t}(R_{it} - \hat\beta_i^{K\top} F_t^K)^2$. Bai–Ng propose several penalties, e.g.
- $\mathrm{IC}_{p1}$: $p = \frac{N+T}{NT}\log\frac{NT}{N+T}$.
- $\mathrm{IC}_{p2}$: $p = \frac{N+T}{NT}\log C^2_{NT}$, with $C_{NT} = \min(\sqrt N, \sqrt T)$.

Under standard regularity, $\hat K \to K^*$ in probability.

### Rotation indeterminacy

PCA factors $F$ and loadings $\beta$ are identified only up to a nonsingular $K\times K$ rotation $H$: $\tilde F = H F$, $\tilde \beta = \beta H^{-1}$ yield the same returns. Economic interpretation requires rotation (e.g., Varimax, or projection onto named portfolios like MKT).

### Empirical findings

On US equities, the first PC is nearly the market (weights roughly proportional to market cap). The second and third PCs often look like size and value. Higher PCs pick up industries, momentum, and noise.

---

## 7.6.6 Dynamic Factor Models

### Stock–Watson DFM

$$y_t = \Lambda F_t + e_t, \qquad F_t = \Phi_1 F_{t-1} + \cdots + \Phi_p F_{t-p} + u_t.$$
The observation equation loads a high-dimensional panel $y_t \in \mathbb{R}^N$ onto low-dimensional factors $F_t \in \mathbb{R}^K$; the state equation is a VAR on factors. Estimation proceeds via Kalman filter + EM, or by two-step methods (PCA for factors, OLS for VAR).

### Factor-augmented VAR (FAVAR)

Bernanke–Boivin–Eliasz (2005) combine factors extracted from a large macro panel with observed policy variables (e.g., Fed funds rate) to study monetary transmission. The model
$$\begin{pmatrix} F_t \\ Y_t \end{pmatrix} = \Phi \begin{pmatrix} F_{t-1} \\ Y_{t-1} \end{pmatrix} + v_t,$$
with $F_t$ the extracted factors and $Y_t$ policy variables, produces impulse responses that account for full information available to the central bank.

### Large Bayesian VARs

When $N$ is large, MLE breaks down. Bańbura–Giannone–Reichlin (2010) use Minnesota-style priors to shrink VAR coefficients toward a random walk, achieving stable forecasting in 100+ dimensions.

---

## 7.6.7 Fundamental Risk Models (Barra-Style)

Rather than extracting factors statistically, fundamental models *construct* factors from observable firm characteristics: industry dummies, size, value, momentum, volatility, liquidity, quality, etc.

### Model structure

$$R_{i,t} = \sum_{k=1}^{K_{\text{ind}}} X_{i,k,t-1} f^{\text{ind}}_{k,t} + \sum_{j=1}^{K_{\text{style}}} X_{i,j,t-1} f^{\text{style}}_{j,t} + u_{i,t}.$$
Exposures $X_{i,k}$ are known (computed from firm data), factor returns $f_t$ are estimated by cross-sectional regression each period (weighted least squares with market-cap weights typically).

### Risk forecasting

The covariance matrix is
$$\Sigma = X F X^\top + D,$$
where $X$ is the $N \times K$ exposure matrix, $F$ is the $K \times K$ factor covariance, and $D$ is a diagonal matrix of specific risks. Estimation: $F$ from a time-series of factor returns (with EWMA weighting), $D$ from time-series of residuals.

### Usage

- **Portfolio construction**: minimize $w^\top \Sigma w$ subject to return and characteristic targets.
- **Risk decomposition**: total portfolio variance = factor variance + specific variance; factor variance decomposed into individual factor contributions.
- **Attribution**: realized returns decomposed into factor tilts × factor returns plus specific.

### Pros and cons

Pros: intuitive (managers understand "value tilt"), stable (exposures change slowly), captures anomalies out of the box. Cons: characteristic choice subjective; ignores dynamics not captured by observables.

---

## 7.6.8 Characteristics and the Factor Zoo

### Proliferation of factors

Harvey–Liu–Zhu (2016) catalog 316 factors claimed in the academic literature by 2015; Hou–Xue–Zhang (2020) show that roughly two-thirds fail to replicate with reasonable microcap filters.

### Multiple testing

With hundreds of tests at $t=2$, false discoveries are guaranteed. HLZ propose a higher t-hurdle of $\sim 3.0$ after Bonferroni / Benjamini–Hochberg corrections.

### Characteristics vs. covariances

Daniel–Titman (1997): do stocks with high expected returns co-move because of common risk (factor story), or is it pure characteristic premium (mispricing story)? Their finding that *characteristics beat covariances* in predicting returns favored mispricing interpretations; subsequent work (Davis–Fama–French 2000) disputed the econometrics.

### High-dimensional cross-sectional regressions

When $K \approx N$, classical FM regressions are ill-conditioned. Kozak–Nagel–Santosh (2020) advocate for Bayesian shrinkage toward a prior centered on no predictability, operationalized through a ridge penalty:
$$\hat\lambda = \arg\min_\lambda \|R^e - X\lambda\|^2 + \tau \lambda^\top \Omega \lambda,$$
with $\Omega$ informed by a principal-component prior. Freyberger–Neuhierl–Weber (2020) use nonparametric characteristic functions estimated via group-lasso; Chen–Pelger–Zhu (2023) construct an SDF through a deep-learning approach with characteristic-dependent factor weights.

---

## 7.6.9 Machine Learning for Factors

### Autoencoders for latent factors

Gu–Kelly–Xiu (2021) replace linear PCA with a neural-network *conditional autoencoder*:
$$R^e_{i,t} = \beta_\theta(z_{i,t-1})^\top f_t + \varepsilon_{i,t},$$
where $\beta_\theta$ is a deep network mapping firm characteristics $z_{i,t-1}$ to factor loadings, and $f_t$ is a vector of unobserved factor returns estimated jointly. Architecturally, the encoder compresses $R_t \to f_t$, and the decoder maps $(z_{t-1}, f_t) \to \hat R_t$. Training minimizes
$$L = \sum_{i,t}(R^e_{i,t} - \beta_\theta(z_{i,t-1})^\top f_t)^2.$$

### Instrumented PCA (IPCA, Kelly–Pruitt–Su 2019)

A parametric version: loadings are linear in characteristics, $\beta_{i,t} = \Gamma_\beta z_{i,t-1}$, factors $f_t$ free. Estimation alternates between
- solving for $f_t$ given $\Gamma_\beta$ by cross-sectional GLS;
- solving for $\Gamma_\beta$ given $\{f_t\}$ by pooled OLS of stock returns on $z_{i,t-1} f_t^\top$.

IPCA delivers pricing errors dramatically smaller than static characteristic portfolios.

### Deep SDF

Chen–Pelger–Zhu (2023) parameterize the SDF directly:
$$M_t = 1 - \sum_i w_\theta(z_{i,t-1}) R^e_{i,t+1},$$
with $w_\theta$ a neural network. Training: minimize the dual of the Hansen–Jagannathan distance,
$$L(\theta) = \mathbb{E}\left[\max_{R^e} (\mathbb{E}_t[M_{t+1} R^e_{t+1}])^2\right],$$
with a GAN-like adversarial structure — a separate network proposes test assets while the SDF network tries to price them. Reported Sharpe ratios on US equities exceed all previously published benchmarks.

---

## 7.6.10 Python Implementations

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ----- Fama-French-style 3-factor regression -----
# Assume ff is a DataFrame indexed by month with columns
#   ['MKT', 'SMB', 'HML', 'RF'], and rets is a DataFrame of stock monthly returns.

def ff3_regression(rets, ff):
    """For each stock, regress excess return on MKT, SMB, HML; return alpha, betas, t-stats."""
    ex = rets.sub(ff['RF'], axis=0)
    X = sm.add_constant(ff[['MKT', 'SMB', 'HML']])
    out = {}
    for tic in ex.columns:
        y = ex[tic].dropna()
        x = X.loc[y.index]
        m = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        out[tic] = {
            'alpha': m.params['const'], 't_alpha': m.tvalues['const'],
            'beta_mkt': m.params['MKT'], 't_mkt': m.tvalues['MKT'],
            'beta_smb': m.params['SMB'], 't_smb': m.tvalues['SMB'],
            'beta_hml': m.params['HML'], 't_hml': m.tvalues['HML'],
            'R2': m.rsquared
        }
    return pd.DataFrame(out).T

# ----- Fama-MacBeth two-pass -----
def fama_macbeth(rets_excess, factors):
    """
    rets_excess: T x N DataFrame of excess returns
    factors:     T x K DataFrame of factor returns (including a market factor)
    Returns factor risk premia with FM standard errors.
    """
    T, N = rets_excess.shape
    K = factors.shape[1]

    # Pass 1: time-series beta for each asset
    X = sm.add_constant(factors)
    betas = np.zeros((N, K))
    for j, col in enumerate(rets_excess.columns):
        y = rets_excess[col].dropna()
        x = X.loc[y.index]
        m = sm.OLS(y, x).fit()
        betas[j, :] = m.params.iloc[1:].values  # drop intercept

    betas_df = pd.DataFrame(betas, index=rets_excess.columns, columns=factors.columns)

    # Pass 2: cross-sectional regression each period
    gammas = []
    Xb = sm.add_constant(betas_df)
    for t in rets_excess.index:
        y_t = rets_excess.loc[t].dropna()
        x_t = Xb.loc[y_t.index]
        mt = sm.OLS(y_t, x_t).fit()
        gammas.append(mt.params.values)

    gammas = np.array(gammas)
    lam_hat = gammas.mean(axis=0)
    lam_se = gammas.std(axis=0, ddof=1) / np.sqrt(T)
    t_stats = lam_hat / lam_se

    return pd.DataFrame({
        'premia': lam_hat, 'se': lam_se, 't': t_stats
    }, index=['intercept'] + list(factors.columns))

# ----- PCA factors with Bai-Ng IC -----
def pca_factors(rets, K_max=10):
    """Return (factors, loadings, Bai-Ng IC_p2 for each K)."""
    R = (rets - rets.mean()).values  # centered
    T, N = R.shape
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    ic = []
    C_NT = min(np.sqrt(N), np.sqrt(T))
    for K in range(1, K_max + 1):
        # Reconstruct with K components
        R_hat = U[:, :K] @ np.diag(S[:K]) @ Vt[:K, :]
        resid = R - R_hat
        V = (resid**2).sum() / (N * T)
        pen = (N + T) / (N * T) * np.log(C_NT**2)
        ic.append(np.log(V) + K * pen)
    K_star = int(np.argmin(ic) + 1)
    # Scaled factors
    F = U[:, :K_star] * S[:K_star] / np.sqrt(T)
    Lam = Vt[:K_star, :].T * np.sqrt(T)
    return F, Lam, ic, K_star

# ----- Instrumented PCA sketch -----
def ipca(R, Z, K=3, n_iter=20, tol=1e-6):
    """
    R: T x N returns
    Z: T x N x L characteristics (lagged)
    K: number of latent factors
    Estimates Gamma_beta (L x K) and factors F (T x K) by alternating least squares.
    """
    T, N, L = Z.shape
    Gamma = np.random.randn(L, K) * 0.1
    F = np.random.randn(T, K) * 0.1
    for it in range(n_iter):
        # Fix Gamma, solve for F_t by cross-sectional GLS (OLS here for simplicity)
        for t in range(T):
            B_t = Z[t] @ Gamma  # N x K
            # F_t = (B' B)^-1 B' R_t
            F[t] = np.linalg.lstsq(B_t, R[t], rcond=None)[0]
        # Fix F, solve for Gamma by pooling
        # Stack: for each t,i the observation is r_{it} = (z_{it} ⊗ F_t) vec(Gamma)
        lhs = np.zeros((L * K, L * K))
        rhs = np.zeros(L * K)
        for t in range(T):
            for i in range(N):
                zf = np.outer(Z[t, i], F[t]).ravel()  # L*K
                lhs += np.outer(zf, zf)
                rhs += zf * R[t, i]
        Gamma = np.linalg.solve(lhs + 1e-6*np.eye(L*K), rhs).reshape(L, K)
    return Gamma, F
```

### Empirical snapshot

Applying `ff3_regression` to US industry portfolios typically yields:
- $R^2$ in the range 0.80–0.95 (much higher than CAPM's 0.60–0.75).
- Most industry alphas are statistically insignificant once HML and SMB are included.
- Momentum and small-value industries still display significant residual alpha — motivating Carhart and FF5.

---

## 7.6.11 Quant Applications

1. **Risk premia harvesting**: construct diversified long-short portfolios with controlled factor exposures (value, momentum, quality, low-vol). "Smart beta" ETFs operationalize this for retail and institutional investors.
2. **Portfolio construction**: use factor covariance $\Sigma = XFX^\top + D$ as the risk model in Markowitz/Black–Litterman optimization.
3. **Performance attribution**: decompose manager return into market, style, and specific components (Barra attribution).
4. **Risk budgeting**: allocate active risk to intentional bets while neutralizing unintended factor exposures.
5. **Alpha research**: use factors as residualization baselines — new signals must deliver alpha relative to a factor model.
6. **Hedge fund replication**: fit a rolling factor model to a hedge fund's return stream to clone its style at low cost (Hasanhodzic–Lo 2007).
7. **Corporate finance**: cost of equity via multi-factor models (instead of pure CAPM) for capital budgeting.
8. **Statistical arbitrage**: PCA factor residuals drive mean-reversion signals (Avellaneda–Lee 2010).
9. **Credit risk**: multi-factor extensions of KMV/Merton, with industry and macro factors driving asset-value correlations.
10. **Macro trading**: FAVAR-style models inform positioning around monetary policy regimes.
11. **Asset allocation**: risk-parity strategies balance contributions from macro factors (growth, inflation, real rates, credit).
12. **Machine-learning backtests**: IPCA and conditional autoencoders produce SDFs whose implied portfolios deliver out-of-sample Sharpe ratios substantially higher than FF5.

---

## 7.6.12 Exercises

### ★ Exercises

1. Derive the CAPM tangency-portfolio condition from the Lagrangian of the Markowitz problem.
2. Show that the SDF $M = a - b R_M^e$ implies $\mathbb{E}[R_i^e] = \beta_i^M \mathbb{E}[R_M^e]$ with the right choice of $a,b$.
3. Compute $\beta^M$ for an equally-weighted 50/50 portfolio of two assets with known covariances and show the linearity of beta.
4. Derive the formula for SMB as the average of three portfolios minus three, and verify construction weights sum to zero.
5. Show that in a 1-factor model with $\alpha=0$, any zero-beta portfolio must earn $r_f$.
6. Prove that Fama–MacBeth standard errors do not require modeling cross-sectional dependence of $\hat\gamma_t$ across $t$.

### ★★ Exercises

7. Implement the GRS test from scratch and verify on simulated data that its size equals the nominal level.
8. Compare `ff3_regression` on 49 industry portfolios vs. 25 size/BM portfolios; discuss why HML spans more of the latter.
9. Derive the Shanken EIV correction assuming $\hat\beta_i - \beta_i \approx \mathcal{N}(0, V_\beta/T)$.
10. Show that as $N \to \infty$ with an approximate factor structure, the top $K$ eigenvalues of the sample covariance diverge linearly in $N$ while the rest stay bounded. (Chamberlain–Rothschild.)
11. Prove that the Bai–Ng $\mathrm{IC}_{p2}$ penalty is minimal over a grid of $K$ for large $N,T$ at the true $K^*$.
12. Given a fitted IPCA model, derive the implied SDF and show it prices the training characteristic portfolios exactly.

### ★★★ Exercises

13. Formally prove the APT bound $\sum_i (\mathbb{E}[R_i^e] - \beta_i^\top \lambda)^2 \le c$ given Ross's factor assumptions and an asymptotic-arbitrage condition. Specify $c$ in terms of $\max_i \sigma_i^2$ and the efficient Sharpe ratio.
14. Prove consistency of the Fama–MacBeth second-pass estimator $\hat\lambda$ under regularity; derive the asymptotic distribution including the Shanken correction.
15. Analyze the impact of weak factors ($\|\beta\|$ shrinking with $T$) on FM inference; show that standard $t$-statistics can be severely oversized. (Kan–Zhang 1999, Kleibergen 2009.)
16. Derive the Kozak–Nagel–Santosh Bayesian SDF under a ridge prior on factor premia with a PC-aligned covariance; show how the posterior mean is a shrunk version of the OLS estimate.
17. Formulate the conditional autoencoder objective as a variational problem, derive the ELBO under a latent Gaussian factor prior, and compare to the standard PCA objective.
18. Using duality, prove that the Hansen–Jagannathan distance of a candidate SDF equals the maximum pricing error over unit-norm asset returns; explain why minimizing this distance is the motivation for deep SDF training.

---

*— End of Module 7.6. Next: Module 7.7, Causal Inference and Experiments — Subject 7 capstone.*

# Module 7.1 — Linear Regression and Regularization

*Subject 7 — Statistical Learning & Econometrics for Quant, Module 1.*

---

## Prerequisites

- **Module 0.2 (Linear algebra)** — SVD, projection operators.
- **Module 0.3.6 (Calculus of variations / constrained optimization)** — regularization = constrained minimization.
- **Module 2.2 (Expectation and conditional expectation)** — the best $L^2$ predictor is $\mathbb{E}[Y \mid X]$.
- **Module 2.5 (Characteristic functions)** — for normal-theory inference.

---

## 7.1.1 The linear model

Linear regression is the foundational tool of applied statistics. Given response $Y \in \mathbb{R}^n$ and design matrix $X \in \mathbb{R}^{n \times p}$, the linear model assumes
$$
Y = X\beta + \varepsilon,
$$
with $\mathbb{E}[\varepsilon] = 0$, $\text{Cov}(\varepsilon) = \sigma^2 I_n$.

**OLS estimator.** Minimize $\|Y - X\beta\|^2$:
$$
\boxed{\hat\beta_{OLS} = (X^\top X)^{-1} X^\top Y.}
$$

**Geometry.** $\hat Y = X\hat\beta = P_X Y$ where $P_X = X(X^\top X)^{-1} X^\top$ is the orthogonal projection onto $\mathcal{R}(X)$. Residuals $\hat\varepsilon = (I - P_X) Y$ are orthogonal to columns of $X$.

**Unbiasedness and covariance.** Assuming the model is correctly specified:
- $\mathbb{E}[\hat\beta] = \beta$.
- $\text{Cov}(\hat\beta) = \sigma^2 (X^\top X)^{-1}$.

**Gauss-Markov theorem.** Among linear unbiased estimators, OLS has minimum variance. Formally, for any other linear unbiased $\tilde\beta = CY$ with $\mathbb{E}[\tilde\beta] = \beta$:
$$
\text{Cov}(\tilde\beta) - \text{Cov}(\hat\beta_{OLS}) \succeq 0.
$$

**Proof sketch.** Write $C = (X^\top X)^{-1} X^\top + D$; unbiasedness implies $DX = 0$; expand $CC^\top$ and use $D(X(X^\top X)^{-1}) = 0$ to show the cross-term vanishes and the variance of $\tilde\beta$ exceeds OLS by $\sigma^2 DD^\top \succeq 0$.

---

## 7.1.2 Inference under normality

If $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$, then $\hat\beta \sim \mathcal{N}(\beta, \sigma^2 (X^\top X)^{-1})$.

**Unbiased estimator of $\sigma^2$**:
$$
\hat\sigma^2 = \frac{1}{n-p}\|Y - X\hat\beta\|^2.
$$

**$t$-statistic for hypothesis $H_0: \beta_j = 0$**:
$$
t_j = \frac{\hat\beta_j}{\hat\sigma \sqrt{[(X^\top X)^{-1}]_{jj}}} \sim t_{n-p}.
$$

**$F$-statistic for joint hypothesis** $H_0: A\beta = 0$:
$$
F = \frac{(A\hat\beta)^\top [A(X^\top X)^{-1}A^\top]^{-1}(A\hat\beta) / q}{\hat\sigma^2} \sim F_{q, n-p}.
$$

**$R^2$** = $1 - \|Y - X\hat\beta\|^2/\|Y - \bar Y\|^2$ is the fraction of total variance explained. **Adjusted $R^2$** penalizes degrees of freedom: $R^2_{adj} = 1 - (n-1)/(n-p) \cdot (1 - R^2)$.

**Standard errors are wrong when** errors are heteroskedastic or serially correlated. Use **HAC** (Newey-West) estimators in time series.

---

## 7.1.3 Robust standard errors and GLS

**Heteroskedasticity.** If $\text{Cov}(\varepsilon) = \text{diag}(\sigma_i^2)$, OLS is still unbiased but inefficient. Robust **White/Huber/sandwich** covariance:
$$
\text{Cov}(\hat\beta)_{robust} = (X^\top X)^{-1} X^\top \text{diag}(\hat\varepsilon_i^2) X (X^\top X)^{-1}.
$$

**Autocorrelated errors** (Newey-West). With lag length $L$:
$$
S = \sum_{|l|\le L} w_l X_{t-l} \hat\varepsilon_{t-l} \hat\varepsilon_t X_t^\top,
$$
where $w_l = 1 - |l|/(L+1)$ (Bartlett weights). The resulting sandwich estimator is consistent for any lag structure up to $L$.

**Generalized Least Squares.** If $\text{Cov}(\varepsilon) = \sigma^2 \Omega$ known,
$$
\hat\beta_{GLS} = (X^\top \Omega^{-1} X)^{-1} X^\top \Omega^{-1} Y.
$$
GLS is BLUE (best linear unbiased). Feasible GLS estimates $\Omega$ from residuals.

---

## 7.1.4 Bias-variance tradeoff and regularization

**Expected prediction error** (for a new point $(x_*, y_*)$):
$$
\mathbb{E}[(y_* - \hat f(x_*))^2] = \sigma^2 + \text{Bias}^2(\hat f(x_*)) + \text{Var}(\hat f(x_*)).
$$

OLS minimizes the bias component (zero-bias linear estimator) but can have large variance when $p$ is close to $n$ or when $X^\top X$ is ill-conditioned.

**Regularization** trades bias for variance reduction, often reducing MSE.

---

## 7.1.5 Ridge regression (Tikhonov)

$$
\hat\beta_{ridge} = \arg\min_\beta \|Y - X\beta\|^2 + \lambda \|\beta\|^2.
$$

**Closed form:**
$$
\boxed{\hat\beta_{ridge} = (X^\top X + \lambda I)^{-1} X^\top Y.}
$$

**Via SVD.** If $X = U\Sigma V^\top$ with singular values $\sigma_j$:
$$
\hat\beta_{ridge} = V \text{diag}\left(\frac{\sigma_j}{\sigma_j^2 + \lambda}\right) U^\top Y.
$$
Ridge shrinks singular directions with small $\sigma_j$ more aggressively — stabilizes ill-conditioned problems.

**Bias-variance.** Ridge is biased: $\mathbb{E}[\hat\beta_{ridge}] = (I - \lambda(X^\top X + \lambda I)^{-1})\beta$, but variance is strictly lower for $\lambda > 0$. There exists $\lambda^* > 0$ making MSE smaller than OLS for any $\beta \ne 0$.

**Bayesian interpretation.** Ridge is MAP estimate with prior $\beta \sim \mathcal{N}(0, \sigma^2/\lambda \cdot I)$.

**Cross-validation.** Choose $\lambda$ by $k$-fold CV: minimize validation MSE over held-out folds.

---

## 7.1.6 Lasso (Tibshirani 1996)

$$
\hat\beta_{lasso} = \arg\min_\beta \|Y - X\beta\|^2 + \lambda \|\beta\|_1.
$$

**Subgradient condition** at optimum: for each $j$,
$$
-X_j^\top (Y - X\hat\beta) + \lambda \text{sgn}(\hat\beta_j) = 0 \text{ if } \hat\beta_j \ne 0,
$$
$$
|X_j^\top (Y - X\hat\beta)| \le \lambda \text{ if } \hat\beta_j = 0.
$$

**Sparsity.** Unlike ridge, lasso produces exactly-zero coefficients: a $\beta_j$ is zero whenever the corresponding OLS gradient satisfies $|X_j^\top (Y - X\hat\beta_{-j})| < \lambda$.

**Solution path.** $\hat\beta(\lambda)$ is piecewise-linear in $\lambda$ (LARS algorithm, Efron-Hastie-Johnstone-Tibshirani 2004).

**Soft-threshold in orthogonal case** ($X^\top X = I$):
$$
\hat\beta_j = \text{sign}(\hat\beta_j^{OLS}) (|\hat\beta_j^{OLS}| - \lambda/2)_+.
$$

**When does lasso recover the true sparsity pattern?**
- **Mutual incoherence**: max off-diagonal of $|X_S^\top X_{S^c}|$ bounded below 1.
- **Restricted eigenvalue**: signal submatrix well-conditioned.
- **Beta-min condition**: true nonzero coefficients exceed noise threshold $\sim \sigma\sqrt{\log p/n}$.

Under these, lasso is **model-selection consistent** (Zhao-Yu 2006, Wainwright 2009).

**Prediction-error rates.** $\|X\hat\beta_{lasso} - X\beta^*\|^2/n \le O(\sigma^2 s\log p/n)$ where $s$ is the true sparsity. This is the **oracle rate** — as if you knew the true sparsity pattern.

---

## 7.1.7 Elastic net (Zou-Hastie 2005)

$$
\hat\beta = \arg\min \|Y - X\beta\|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|^2.
$$

Combines shrinkage (ridge) with sparsity (lasso). Handles correlated predictors better than pure lasso (which tends to pick only one).

Parameterize as $\alpha \|\beta\|_1 + (1-\alpha)\|\beta\|^2$ for $\alpha \in [0,1]$: $\alpha=0$ is ridge, $\alpha=1$ is lasso.

---

## 7.1.8 Beyond $\ell_1$: SCAD, group lasso, fused lasso

**SCAD** (Fan-Li 2001). Smoothly-clipped absolute deviation. Penalty that's linear for small $|\beta_j|$ (like lasso) but constant for large $|\beta_j|$ (reducing bias). Produces "oracle" estimators.

**Group lasso** (Yuan-Lin 2006). Groups of coefficients share the same fate:
$$
\lambda \sum_g \|\beta_{G_g}\|_2.
$$
If one coefficient in a group is nonzero, all are. Useful for categorical variables with multiple levels.

**Fused lasso** (Tibshirani-Saunders-Rosset-Zhu-Knight 2005).
$$
\lambda_1 \|\beta\|_1 + \lambda_2 \sum_j |\beta_{j+1} - \beta_j|.
$$
Encourages both sparsity and smoothness along an ordering. Used in financial time series where $\beta_j$ = time-varying coefficient.

**Non-convex penalties** (MCP, SCAD) are computationally harder but give less-biased estimates of strong signals.

---

## 7.1.9 The $p \gg n$ regime

Modern finance: 10⁵ tickers × 10⁴ minute bars → $p \gg n$ for stock-level factor modeling on short windows.

**Classical OLS fails**: $X^\top X$ is singular.

**Ridge works** with $\lambda > 0$, always invertible.

**Lasso works** if the true model is sparse with $s \le C \cdot n / \log p$.

**Double descent** (Belkin-Hsu-Ma-Mandal 2019). As $p$ grows through $n$, MSE often **peaks** at the interpolation threshold $p = n$ and then **decreases again** with even higher-dimensional models. Challenges classical bias-variance intuition.

---

## 7.1.10 Kernel ridge regression

Replace $X$ with feature map $\phi(x)$ in a reproducing kernel Hilbert space (RKHS). Ridge:
$$
\hat\beta = \arg\min_\beta \sum_i (y_i - \phi(x_i)^\top \beta)^2 + \lambda \|\beta\|^2.
$$

**Representer theorem**: $\hat\beta = \sum_i \alpha_i \phi(x_i)$. Plugging in:
$$
\hat\alpha = (K + \lambda I)^{-1} Y, \quad K_{ij} = k(x_i, x_j) = \phi(x_i)^\top\phi(x_j).
$$

Prediction: $\hat y(x_*) = k(x_*, X) (K + \lambda I)^{-1} Y$.

Equivalent to Gaussian process regression with kernel $k$ and noise variance $\lambda$.

Cost: $O(n^3)$ — limits to $n \le 10^4$. **Random Fourier features** (Rahimi-Recht 2007) approximate $k$ with finite-dim features for $O(n)$ cost.

---

## 7.1.11 Python: regression toolkit

```python
import numpy as np
from numpy.linalg import lstsq, solve, svd
from scipy.stats import t as student_t
import matplotlib.pyplot as plt

# ============================================================
# 1. OLS with inference
# ============================================================
def ols_with_inference(X, y):
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    sigma2 = (resid @ resid)/(n - p)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov_beta))
    t_stat = beta/se
    p_val = 2*(1 - student_t.cdf(np.abs(t_stat), df=n-p))
    return dict(beta=beta, se=se, t=t_stat, p=p_val,
                sigma2=sigma2, cov=cov_beta)

# Synthetic data
np.random.seed(42)
n, p = 200, 5
X = np.random.randn(n, p)
X = np.column_stack([np.ones(n), X])  # intercept
beta_true = np.array([1, 2, -1, 0.5, 0, 0])  # last two are zero
y = X @ beta_true + 0.5*np.random.randn(n)

res = ols_with_inference(X, y)
print("OLS estimates:")
print(f"  {'True':>8} {'Est':>8} {'SE':>8} {'t':>8} {'p':>8}")
for j in range(X.shape[1]):
    print(f"  {beta_true[j]:8.3f} {res['beta'][j]:8.3f} {res['se'][j]:8.3f} "
          f"{res['t'][j]:8.2f} {res['p'][j]:8.3f}")

# ============================================================
# 2. Ridge with cross-validation
# ============================================================
def ridge_path(X, y, lambdas):
    U, s, Vt = svd(X, full_matrices=False)
    betas = np.zeros((len(lambdas), X.shape[1]))
    for i, lam in enumerate(lambdas):
        d = s/(s**2 + lam)
        betas[i] = Vt.T @ (d * (U.T @ y))
    return betas

def kfold_cv_ridge(X, y, lambdas, k=5):
    n = len(y)
    idx = np.arange(n); np.random.shuffle(idx)
    folds = np.array_split(idx, k)
    mse = np.zeros(len(lambdas))
    for fold in folds:
        train = np.setdiff1d(np.arange(n), fold)
        betas = ridge_path(X[train], y[train], lambdas)
        for i, lam in enumerate(lambdas):
            yhat = X[fold] @ betas[i]
            mse[i] += np.mean((y[fold] - yhat)**2)
    return mse / k

lambdas = np.logspace(-3, 3, 30)
cv_mse = kfold_cv_ridge(X, y, lambdas)
opt_lam = lambdas[np.argmin(cv_mse)]
beta_ridge = ridge_path(X, y, [opt_lam])[0]
print(f"\nRidge CV: optimal λ = {opt_lam:.4f}")
print(f"Ridge estimates: {np.round(beta_ridge, 3)}")

# ============================================================
# 3. Lasso via coordinate descent
# ============================================================
def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

def lasso_coord_descent(X, y, lam, max_iter=1000, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    xx = np.sum(X**2, axis=0)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r = y - X @ beta + X[:, j]*beta[j]
            z_j = X[:, j] @ r
            beta[j] = soft_threshold(z_j, lam * n/2) / xx[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

beta_lasso = lasso_coord_descent(X, y, lam=0.1)
print(f"\nLasso estimates (λ=0.1): {np.round(beta_lasso, 3)}")
print(f"Nonzero: {np.sum(beta_lasso != 0)}")

# ============================================================
# 4. Double descent demonstration
# ============================================================
def double_descent_demo():
    n = 100
    p_grid = np.arange(10, 300, 10)
    test_mse = []
    for p_ in p_grid:
        X = np.random.randn(n, p_)
        beta_true = np.zeros(p_); beta_true[:5] = 1
        y = X @ beta_true + 0.5*np.random.randn(n)
        X_test = np.random.randn(100, p_)
        y_test = X_test @ beta_true + 0.5*np.random.randn(100)
        if p_ < n:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            # Use minimum-norm interpolator (ridgeless)
            U, s, Vt = svd(X, full_matrices=False)
            s_inv = np.where(s > 1e-10, 1/s, 0)
            beta = Vt.T @ (s_inv * (U.T @ y))
        yhat = X_test @ beta
        test_mse.append(np.mean((y_test - yhat)**2))
    return p_grid, test_mse

# Uncomment:
# p_grid, mse = double_descent_demo()
# plt.plot(p_grid/100, mse); plt.axvline(1, color='red', ls='--')
# plt.xlabel('p/n'); plt.ylabel('Test MSE'); plt.title('Double Descent')

# ============================================================
# 5. Kernel ridge with Gaussian kernel
# ============================================================
def gaussian_kernel(X1, X2, gamma=1.0):
    d2 = np.sum(X1**2, axis=1)[:,None] + np.sum(X2**2, axis=1)[None,:] - 2*X1 @ X2.T
    return np.exp(-gamma*d2)

def kernel_ridge_predict(X_train, y_train, X_test, gamma=1.0, lam=0.1):
    K = gaussian_kernel(X_train, X_train, gamma)
    alpha = np.linalg.solve(K + lam*np.eye(len(y_train)), y_train)
    K_test = gaussian_kernel(X_test, X_train, gamma)
    return K_test @ alpha

# Test on a nonlinear example
np.random.seed(0)
X_train = np.sort(np.random.randn(50, 1))
y_train = np.sin(X_train.ravel()*3) + 0.1*np.random.randn(50)
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
yhat = kernel_ridge_predict(X_train, y_train, X_test, gamma=0.5, lam=0.1)
# plt.plot(X_train, y_train, 'o'); plt.plot(X_test, yhat)
print(f"\nKernel ridge fitted on sine; check plot for quality.")
```

---

## 7.1.12 [QUANT APPLICATIONS]

1. **Factor models**. Fama-French, Barra — OLS on excess returns. Ridge for shrinkage toward prior.
2. **Alpha combinations**. Combine 100s of alpha signals via ridge to stabilize weight estimates when alphas are correlated.
3. **Cross-sectional asset pricing**. Fama-MacBeth two-pass regression; time series of coefficients gives pricing of risk.
4. **High-frequency trading signal**. Lasso to select a sparse subset of 1000s of candidate predictors.
5. **Volatility surface smoothing**. Fused lasso along the strike/term structure to produce smooth, arbitrage-free surfaces.
6. **Yield curve fitting**. Nelson-Siegel-Svensson with regularized coefficients to avoid overfitting short-end noise.
7. **Portfolio construction**. Ridge-like shrinkage of covariance (Ledoit-Wolf) for minimum-variance portfolios.
8. **Macro nowcasting**. Ridge / PLS to combine hundreds of mixed-frequency macro indicators into a GDP nowcast.
9. **Credit scoring**. Regularized logistic regression for default probability models.
10. **Execution TCA modeling**. Cross-sectional regression of market-impact on trade characteristics; robust SE for panel autocorrelation.

---

## 7.1.13 Exercises

**★ (concept drills).**
1. Prove Gauss-Markov theorem (following the text above).
2. Derive the closed-form ridge solution from $\partial_\beta (\|Y-X\beta\|^2+\lambda\|\beta\|^2) = 0$.
3. Show that lasso in the orthogonal case reduces to soft-thresholding.
4. Under what prior is ridge the MAP estimate? Under what prior is lasso?
5. Explain mutual incoherence. Why is it necessary for lasso to recover the true support?
6. What is double descent? Why does it not contradict classical bias-variance theory?

**★★ (calculation).**
7. Derive Newey-West HAC variance estimator. Why are Bartlett weights used?
8. Lasso via CVX (duality). Formulate the dual of the lasso primal and discuss.
9. Implement 5-fold CV for lasso. On synthetic data with $n=100, p=50, s=5$, find the optimal $\lambda$. Plot CV curve.
10. Group lasso. Implement proximal gradient for grouped coefficients. Apply to a 3-level categorical variable example.
11. Kernel ridge with polynomial kernel $k(x,y) = (x^\top y + c)^d$. Derive the feature map.
12. Elastic-net CV experiment. On a correlated-predictor dataset, compare lasso, ridge, EN.

**★★★ (open / research).**
13. **SCAD implementation.** Implement coordinate descent for SCAD penalty. Compare oracle property to lasso.
14. **Random Fourier features.** Approximate Gaussian kernel with $m = 500$ random features. Compare prediction to exact kernel ridge on synthetic data.
15. **Double descent verification.** Reproduce Belkin-Hsu-Ma-Mandal experiment for $p/n \in [0.1, 3]$. Plot MSE; verify peak at $p = n$.
16. **Financial lasso study.** Use lasso on 500 predictors (macro + momentum + fundamentals) to forecast S&P 500 excess return. Out-of-sample $R^2$?
17. **Double ML (Chernozhukov et al. 2018).** Implement debiased ridge for treatment-effect estimation. Test on synthetic example.

---

*— End of Module 7.1. Next: Module 7.2, Classification and Kernel Methods.*

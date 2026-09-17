# Subject 8, Module 1: Portfolio Optimization Under Estimation Error

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"Optimization is amplification: good inputs get amplified into great performance, bad inputs amplified into catastrophe."* — paraphrasing Richard Michaud

---

## 8.1.0 Where We Are

Subject 8 turns the rigorous mathematical apparatus built across Subjects 0–7 into the operational tooling of a modern quant desk. Module 8.1 opens that arc with **portfolio construction**: how do we turn signals (from factor models, ML, or discretionary views) into actual position sizes under constraints, costs, and — critically — estimation error?

### Prerequisites

- **Module 0.2** (Linear Algebra): positive-definite matrices, eigendecomposition.
- **Module 0.3** (Optimization): KKT conditions, duality, constrained optimization.
- **Module 5.4** (stylized): Markowitz frontier.
- **Module 7.6** (Factor Models): factor risk model $\Sigma = XFX^\top + D$.
- **Module 7.7** (Causal Inference): for robust estimator design, particularly around structural stability.

### Plan

1. Mean-variance optimization and its fragility (§8.1.1).
2. Robust parameter estimation: Ledoit–Wolf shrinkage, James–Stein (§8.1.2).
3. Bayesian approaches: Black–Litterman (§8.1.3).
4. Resampling: Michaud (§8.1.4).
5. Robust optimization: worst-case MVO, min-variance, max-diversification (§8.1.5).
6. Risk parity, hierarchical risk parity, inverse-volatility (§8.1.6).
7. Transaction costs and turnover control (§8.1.7).
8. Long-only, leverage, and integer constraints (§8.1.8).
9. Multi-period portfolio choice (§8.1.9).
10. Python implementations (§8.1.10).
11. Applications (§8.1.11).
12. Exercises (§8.1.12).

---

## 8.1.1 Mean-Variance Optimization

### The Markowitz problem

$$\max_w \; w^\top \mu - \tfrac{\gamma}{2} w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mathbf 1 = 1.$$

The unconstrained solution is
$$w^* = \frac{1}{\gamma}\Sigma^{-1}(\mu - \lambda^* \mathbf 1), \qquad \lambda^* = \frac{\mathbf 1^\top \Sigma^{-1}\mu - \gamma}{\mathbf 1^\top \Sigma^{-1}\mathbf 1}.$$
Two useful portfolios emerge:
- Minimum-variance: $w_{\text{mv}} = \Sigma^{-1}\mathbf 1 / (\mathbf 1^\top \Sigma^{-1}\mathbf 1)$ (independent of $\mu$).
- Tangency: $w_{\text{tang}} \propto \Sigma^{-1}(\mu - r_f \mathbf 1)$.

### The fragility

Michaud (1989) named MVO "error maximization." Because $w^* \propto \Sigma^{-1}\mu$, even modest sampling error in $\mu$ or $\Sigma$ produces wild swings in $w^*$. Specifically:
- Sample covariance $\hat\Sigma$ has a **spiked eigenvalue** distribution (Marchenko–Pastur when $p/n \to c \in (0,1)$); inverting it amplifies small eigenvalues.
- Sample mean $\hat\mu$ has relative error $\sigma/\sqrt T$ per asset, often comparable to the magnitude of the mean itself.

### Plug-in risk decomposition

For $\hat w = \hat\Sigma^{-1}\hat\mu/\gamma$, define true realized variance:
$$\text{out-of-sample Var}(\hat w^\top R) = \hat w^\top \Sigma \hat w.$$
Under a standard i.i.d. Gaussian world, Kan–Zhou (2007) show the expected out-of-sample utility loss relative to oracle MVO is
$$\Delta U \approx \frac{1}{2\gamma}\text{tr}(\Sigma^{-1}\hat\Sigma^{-1}\Sigma) - \frac{1}{2\gamma N} \cdot \dots,$$
which scales unfavorably in $p/n$.

The empirically observed fact: naïve MVO underperforms the 1/N portfolio across many real datasets (DeMiguel–Garlappi–Uppal 2009) unless estimation error is aggressively controlled.

---

## 8.1.2 Robust Estimation

### James–Stein shrinkage

For estimating $\mu \in \mathbb{R}^p$ with $p \ge 3$ under squared-error loss,
$$\hat\mu_{\text{JS}} = \bar\mu \cdot \mathbf 1 + \left(1 - \frac{(p-2)\sigma^2/n}{\|\hat\mu - \bar\mu \mathbf 1\|^2}\right)(\hat\mu - \bar\mu \mathbf 1)$$
dominates the sample mean (Stein's paradox). In portfolio contexts, Jorion (1986) extended this with a "grand mean" shrinkage target derived from the minimum-variance portfolio.

### Ledoit–Wolf covariance shrinkage

Shrink $\hat\Sigma$ toward a structured target $F$ (e.g., diagonal, constant correlation, or factor model):
$$\hat\Sigma_{\text{LW}} = \alpha F + (1-\alpha) \hat\Sigma,$$
with
$$\alpha^* = \min\left(1, \frac{1}{T}\cdot \frac{\pi - \rho}{\gamma}\right),$$
where $\pi = \sum_{i,j} \text{Var}(\hat\Sigma_{ij})$, $\rho = \sum_{i,j}\text{Cov}(\hat\Sigma_{ij}, F_{ij})$, and $\gamma = \sum_{i,j}(F_{ij}-\Sigma_{ij})^2$. Ledoit–Wolf (2004) give closed-form formulas for each term under the constant-correlation target.

Nonlinear shrinkage (Ledoit–Wolf 2020) modifies individual eigenvalues via a kernel-smoothed estimate of the limiting spectral density, improving performance further in high dimensions.

### Condition number regularization

$\hat\Sigma + \lambda I$ — simple ridge on the covariance — is equivalent to shrinking all eigenvalues up by $\lambda$. Cross-validate on out-of-sample portfolio variance or explicit risk metrics.

### Factor-model covariance

From Module 7.6: $\Sigma = XFX^\top + D$, with $K \ll N$ factors and a diagonal $D$. Factor models automatically impose a low-rank plus sparse structure that is often well-calibrated for equity portfolios.

---

## 8.1.3 Black–Litterman

### Motivation

Allow the investor to combine:
1. **Equilibrium views** (typically reverse-optimized from a market-cap benchmark).
2. **Subjective views** on relative or absolute expected returns.

### The machinery

Suppose the equilibrium mean is $\pi = \gamma \Sigma w_{\text{mkt}}$ (reverse optimization) and the prior is
$$\mu \sim \mathcal{N}(\pi, \tau \Sigma).$$
Subjective views take the form $P\mu = q + \epsilon$, with $\epsilon \sim \mathcal{N}(0, \Omega)$.

Posterior by Bayes' rule:
$$\hat\mu = [(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P]^{-1} [(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} q].$$
$$\hat\Sigma_{\mu} = [(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P]^{-1}.$$

Final portfolio: MVO with $\hat\mu$ and covariance $\Sigma + \hat\Sigma_\mu$ (or just $\Sigma$, depending on implementation).

### Properties

- **No view yields the market**: $P=0$ gives $w^* = w_{\text{mkt}}$.
- **Views move smoothly**: confident views ($\Omega$ small) pull the portfolio firmly; uncertain views ($\Omega$ large) have little effect.
- **Views can be relative**: $P = (0, 1, -1, 0, \ldots)$ says "asset 2 outperforms asset 3 by $q$."

### Choice of $\tau$ and $\Omega$

He–Litterman (2000) recommend $\Omega_{kk} = \tau P_k \Sigma P_k^\top$ — confidence scales with the view's inherent volatility. $\tau$ is typically set to $1/T$ (suggesting the equilibrium prior is worth roughly $T$ observations).

---

## 8.1.4 Michaud Resampling

Richard Michaud (1998):

1. From $(\hat\mu, \hat\Sigma)$, simulate $B$ synthetic return histories.
2. Re-estimate $(\hat\mu^{(b)}, \hat\Sigma^{(b)})$ on each history.
3. Solve the MVO at each simulation to get $w^{(b)}$.
4. Average: $\bar w = \frac{1}{B}\sum_b w^{(b)}$.

Resampled portfolios are smoother and more diversified than the point-estimate MVO. Statistical interpretation: the resampling average approximates the posterior mean under a parametric bootstrap, which is closely related to James–Stein shrinkage in some regimes.

Critique (Scherer 2002): resampled frontiers can be stochastically dominated by direct Bayesian approaches under correct specification.

---

## 8.1.5 Robust Optimization

### Worst-case MVO

Allow $\mu$ to lie in an uncertainty set $U_\mu$ (e.g., an ellipsoid $(\mu-\hat\mu)^\top \Sigma^{-1}(\mu-\hat\mu) \le \kappa^2$) and $\Sigma$ in $U_\Sigma$:
$$\max_w \min_{\mu \in U_\mu, \Sigma \in U_\Sigma} w^\top \mu - \frac{\gamma}{2}w^\top \Sigma w.$$

For ellipsoidal $U_\mu$ and fixed $\Sigma$, Goldfarb–Iyengar (2003) reformulate as a second-order cone program:
$$\max_w w^\top \hat\mu - \kappa \|\Sigma^{1/2} w\| - \frac{\gamma}{2} w^\top \Sigma w.$$
The $\kappa \|\Sigma^{1/2} w\|$ penalty shrinks the portfolio away from corner solutions that concentrate on high-variance directions.

### Tu–Zhou combination

Tu–Zhou (2011) combine the MVO and 1/N portfolios as
$$w^* = \delta \cdot w_{\text{MVO}} + (1-\delta)\cdot w_{1/N},$$
with $\delta$ chosen to minimize expected utility loss. Often dominates both extremes.

### Max-diversification (Choueifaty)

Maximize the diversification ratio
$$\text{DR}(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}},$$
where $\sigma$ is the vector of asset volatilities. The solution is the "most diversified" long-only portfolio; in practice this tilts away from correlated risk sources.

---

## 8.1.6 Risk-Based Portfolios

### Inverse volatility

$$w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}.$$
No correlation structure needed. Common in multi-asset CTAs.

### Risk parity (equal risk contribution)

Each asset contributes the same marginal risk: $w_i (\Sigma w)_i = \text{const}$ for all $i$. With $N$ assets, the condition becomes $w_i (\Sigma w)_i = w^\top \Sigma w / N$. Solve via
$$\min_w \sum_{i,j}\left(w_i (\Sigma w)_i - w_j (\Sigma w)_j\right)^2.$$

Maillard–Roncalli–Teïletche (2010) gave the foundational treatment; Bruder–Roncalli (2012) extended to constrained variants.

### Hierarchical risk parity (López de Prado 2016)

1. Cluster assets by correlation using single-linkage agglomerative clustering on the correlation-distance $d_{ij} = \sqrt{(1 - \rho_{ij})/2}$.
2. Quasi-diagonalize the covariance matrix by reordering per the cluster tree.
3. Recursively bisect the sorted list, allocating inverse-variance across the two halves at each split.

HRP avoids matrix inversion entirely — a substantial robustness gain when $\hat\Sigma$ is near-singular. Out-of-sample performance competitive with MVO and risk parity in high-dimensional settings.

### Critique

Risk parity is **correlation-sensitive**: during crises correlations shoot up, realized risk contributions diverge from targets. Leverage (common in risk-parity funds) magnifies this. The "risk-parity unwind" of 2022 illustrated the hazard of liquidity mismatch in correlation-based leverage.

---

## 8.1.7 Transaction Costs and Turnover

### Quadratic cost

Add a term $-c \|w - w_0\|_{\Lambda}^2$ to the objective, with $\Lambda$ reflecting per-name impact and $w_0$ the current portfolio. Solution: shrink toward current holdings.

### Linear cost (bid–ask style)

$-\sum_i c_i |w_i - w_{0,i}|$ — induces a "no-trade region" (Gârleanu–Pedersen 2013) where the optimal policy is to trade only until the boundary of a target region.

### Intertemporal impact (Almgren–Chriss)

See Module 5.7. Static daily MVO ignores the fact that trading is costly across multiple days. Integrated frameworks (Bouchaud et al.) model both instantaneous and persistent impact.

### Constraints

Common constraints encountered in practice:
- **Leverage cap**: $\|w\|_1 \le L$.
- **Long-only**: $w \ge 0$.
- **Position bounds**: $w_{\min} \le w \le w_{\max}$.
- **Industry / factor neutrality**: $X^\top w = 0$.
- **Turnover cap per period**.
- **Beta target**: $w^\top \beta_M = 1$ or 0.
- **Integer constraints** (position sizes in round lots), solved via branch-and-bound MILPs in production.

Modern solvers (MOSEK, Gurobi, CVXPY) handle all of the above in seconds for $N$ in the thousands.

---

## 8.1.8 Multi-Period Portfolio Choice

Merton (Module 4.5): the continuous-time stochastic-control solution for a log-utility or CRRA investor in a geometric Brownian-motion market yields a constant proportion rule. In discrete time with costs and signal dynamics, the optimal policy becomes state-dependent.

### Gârleanu–Pedersen (2013)

Signal $f_t$ follows an AR process; return has components predictable by $f$. Quadratic costs. Optimal policy is a linear function of $f$ and the current portfolio, with explicit analytic form via a matrix Riccati equation.

### Approximate dynamic programming

For realistic problem features (nonlinear costs, non-Gaussian returns, many factors), numerical ADP (Module 4.7) and reinforcement learning are employed.

---

## 8.1.9 Factor-Risk-Constrained Optimization

A standard production objective:
$$\max_w \; \hat\alpha^\top w - \gamma \, w^\top \Sigma w - c \|w - w_0\|_{\Lambda}^2$$
subject to:
- $\mathbf 1^\top w = 1$ (or $=0$ for long/short).
- $X_f^\top w = b_f$ for each targeted factor exposure $f$ (neutrality or tilt).
- $w \ge 0$ or $|w_i| \le w_{\max}$.
- $\text{TE}(w) = \sqrt{(w - w_B)^\top \Sigma (w - w_B)} \le \sigma_{\text{TE}}$ (tracking error).
- Turnover: $\|w - w_0\|_1 \le \tau$.

Typically solved as a QP / SOCP each day.

---

## 8.1.10 Python Implementations

```python
import numpy as np
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, fcluster

# ----- Mean-variance optimization (CVXPY) -----
def mvo(mu, Sigma, gamma=1.0, long_only=False, turnover_cap=None, w_prev=None):
    n = len(mu)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w - 0.5*gamma*cp.quad_form(w, cp.psd_wrap(Sigma)))
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if turnover_cap is not None and w_prev is not None:
        cons.append(cp.norm1(w - w_prev) <= turnover_cap)
    cp.Problem(obj, cons).solve()
    return w.value

# ----- Ledoit-Wolf shrinkage (constant correlation target) -----
def ledoit_wolf(X):
    T, N = X.shape
    S = np.cov(X, rowvar=False, ddof=1)
    var = np.diag(S)
    sd = np.sqrt(var)
    # Constant correlation target
    r_bar = (np.sum(S / np.outer(sd, sd)) - N) / (N*(N-1))
    F = r_bar * np.outer(sd, sd); np.fill_diagonal(F, var)
    # Asymptotic shrinkage intensity
    Xc = X - X.mean(axis=0, keepdims=True)
    pi_mat = np.zeros((N, N))
    for t in range(T):
        pi_mat += (np.outer(Xc[t], Xc[t]) - S)**2
    pi_mat /= T
    pi = pi_mat.sum()
    rho_diag = np.trace(pi_mat) * N / N  # diagonal simplification
    # For simplicity use a practical bound
    rho = rho_diag
    gamma = np.sum((F - S)**2)
    alpha = max(0, min(1, (pi - rho) / (T*gamma)))
    return alpha*F + (1-alpha)*S, alpha

# ----- Black-Litterman -----
def black_litterman(Sigma, w_mkt, gamma=2.5, tau=0.05, P=None, q=None, Omega=None):
    pi = gamma * Sigma @ w_mkt
    if P is None:
        return pi, Sigma
    if Omega is None:
        Omega = tau * (P @ Sigma @ P.T) * np.eye(P.shape[0])
    A = np.linalg.inv(tau*Sigma) + P.T @ np.linalg.solve(Omega, P)
    b = np.linalg.solve(tau*Sigma, pi) + P.T @ np.linalg.solve(Omega, q)
    mu_bl = np.linalg.solve(A, b)
    Sigma_bl = Sigma + np.linalg.inv(A)
    return mu_bl, Sigma_bl

# ----- Risk parity via root-finding -----
def risk_parity(Sigma, tol=1e-10, max_iter=500):
    n = Sigma.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = w @ Sigma @ w
        mrc = Sigma @ w  # marginal risk contrib (un-normalized)
        rc = w * mrc
        target = port_var / n
        # Gradient-descent style update
        grad = rc - target
        w = w - 0.01 * grad / np.sqrt(port_var)
        w = np.abs(w); w /= w.sum()
        if np.max(np.abs(w*mrc - target)) < tol:
            break
    return w

# ----- Hierarchical Risk Parity -----
def hrp(cov):
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    dist = np.sqrt(0.5*(1 - corr))
    link = linkage((dist[np.triu_indices_from(dist, k=1)]), method='single')
    # Order via quasi-diagonalization
    def get_quasi_diag(link):
        link = link.astype(int); link0 = link[-1, 0]; link1 = link[-1, 1]
        from collections import deque
        sort_ix = deque([link0, link1])
        num_items = link[-1, 3]
        while max(sort_ix) >= num_items:
            for i in range(len(sort_ix)):
                if sort_ix[i] >= num_items:
                    idx = sort_ix[i] - num_items
                    a, b = link[idx, 0], link[idx, 1]
                    sort_ix[i] = a
                    sort_ix.insert(i+1, b)
        return list(sort_ix)
    order = get_quasi_diag(link)
    # Recursive bisection
    w = np.ones(len(order))
    clusters = [order]
    while len(clusters) > 0:
        new_clusters = []
        for c in clusters:
            if len(c) > 1:
                half = len(c)//2
                c_left, c_right = c[:half], c[half:]
                sigma_l = inv_var_port(cov, c_left)
                sigma_r = inv_var_port(cov, c_right)
                alpha = 1 - sigma_l/(sigma_l + sigma_r)
                w[c_left] *= alpha
                w[c_right] *= (1 - alpha)
                new_clusters += [c_left, c_right]
        clusters = new_clusters
    return w

def inv_var_port(cov, idx):
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0/np.diag(sub)
    iv /= iv.sum()
    return iv @ sub @ iv

# ----- Robust MVO (ellipsoidal uncertainty on mu) -----
def robust_mvo(mu_hat, Sigma, kappa=1.0, gamma=1.0, long_only=False):
    n = len(mu_hat)
    w = cp.Variable(n)
    # Worst-case mu lies in ellipsoid {mu : (mu-mu_hat)'Sigma^{-1}(mu-mu_hat) <= kappa^2}
    obj = cp.Maximize(mu_hat @ w - kappa*cp.norm(np.linalg.cholesky(Sigma).T @ w, 2)
                      - 0.5*gamma*cp.quad_form(w, cp.psd_wrap(Sigma)))
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    cp.Problem(obj, cons).solve()
    return w.value

# ----- Example -----
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, T = 20, 100
    true_mu = 0.02 + 0.05*rng.random(N)
    A = rng.standard_normal((N, N))
    true_cov = A @ A.T + np.eye(N)
    R = rng.multivariate_normal(true_mu, true_cov, size=T)

    mu_hat = R.mean(axis=0)
    S = np.cov(R, rowvar=False)
    S_lw, alpha = ledoit_wolf(R); print("Shrinkage alpha:", alpha)

    w_mvo = mvo(mu_hat, S, gamma=10, long_only=True)
    w_robust = robust_mvo(mu_hat, S_lw, kappa=0.5, gamma=10, long_only=True)
    w_rp = risk_parity(S)
    w_hrp = hrp(S)
    print("MVO concentration:", (w_mvo > 0.05).sum())
    print("Robust concentration:", (w_robust > 0.05).sum())
    print("HRP concentration:", (w_hrp > 0.05).sum())
```

---

## 8.1.11 Applications

1. **Equity multi-factor long-short**: factor-risk-constrained MVO on a 3000-stock universe, factor-neutral, volatility-targeted.
2. **Global macro multi-asset**: risk-parity across equities / bonds / commodities / FX, volatility-targeted to 10% annual.
3. **Multi-manager portfolio construction**: Black-Litterman blending bottom-up manager views with top-down risk team priors.
4. **Currency overlays**: robust MVO with an equality constraint $\mathbf 1^\top w = 0$ (zero net FX).
5. **ETF-based robo-advisory**: HRP or risk-parity on a small ETF universe with turnover constraints.
6. **Pension liability-driven investing**: constrained optimization with explicit liability-hedging constraints.
7. **Crypto basket construction**: robust MVO given extreme non-normality; typically shrink heavily toward 1/N.
8. **Systematic CTA sizing**: inverse-volatility sizing of trend signals on commodities and rates.
9. **Factor-timing overlays**: Black-Litterman blending cross-sectional factor premia with regime-dependent views.
10. **Hedge fund tail-risk hedging**: ellipsoidal robust optimization with tail scenarios as constraints.

---

## 8.1.12 Exercises

### ★

1. Derive the unconstrained MVO solution from the Lagrangian and verify the two-fund theorem.
2. Show that adding the constraint $\mathbf 1^\top w = 1$ to min-variance yields $w = \Sigma^{-1}\mathbf 1 / (\mathbf 1^\top \Sigma^{-1}\mathbf 1)$.
3. For inverse-volatility weights, compute each asset's marginal risk contribution and explain why it is generally not equal across assets.
4. Given a 3-asset market with known $\Sigma$, compute the risk-parity weights analytically.
5. Show that for $P = 0$, Black-Litterman recovers the equilibrium weights $w_{\text{mkt}}$.
6. Express the Ledoit-Wolf shrinkage target (constant correlation) in closed form and verify its positive definiteness.

### ★★

7. Simulate a Marchenko–Pastur sample covariance with $p/n = 0.9$ and demonstrate that LW shrinkage dramatically improves the condition number vs. the sample covariance.
8. Implement HRP and verify on simulated factor data that it outperforms inverse-variance under a block-structured covariance.
9. Derive the robust MVO solution for ellipsoidal $U_\mu$ as an SOCP.
10. Prove that risk parity satisfies a KKT condition of a weighted log-utility maximization problem.
11. Implement Gârleanu-Pedersen dynamic portfolio choice for a 1-factor AR(1) signal with quadratic costs; verify the Riccati solution.
12. Run DeMiguel-Garlappi-Uppal's 1/N horse-race on simulated data and characterize when MVO beats 1/N.

### ★★★

13. Prove Stein's paradox: the sample mean in $p \ge 3$ dimensions is inadmissible under squared-error loss.
14. Derive the full Ledoit-Wolf optimal shrinkage intensity formula and prove its asymptotic optimality.
15. Prove that the maximum-diversification portfolio is equivalent to the long-only MVO when expected returns are proportional to volatilities.
16. For Black-Litterman with an ensemble of independent views, derive the posterior distribution explicitly.
17. Establish the conditions under which the Michaud resampling procedure is a consistent approximation of a parametric Bayes estimator.
18. Prove consistency of HRP in the block-correlation model as $p, T \to \infty$ with $p/T \to c \in (0, \infty)$.

---

*— End of Module 8.1. Next: Module 8.2, Risk Management: VaR, Expected Shortfall, Coherent Measures.*

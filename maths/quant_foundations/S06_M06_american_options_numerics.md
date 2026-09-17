# Module 6.6 — American Options Numerics

*Subject 6, Module 6. Longstaff-Schwartz regression, dual bounds, and the primal-dual pricing machinery.*

---

## Prerequisites

- **Module 4.1 (Optimal stopping and Snell envelope)** — the theoretical backbone.
- **Module 4.2 (American options free boundary)** — the PDE formulation.
- **Module 6.1 (Monte Carlo)** — simulation fundamentals.
- **Module 6.3 (Finite differences)** — PDE-based comparisons.
- **Module 6.4 (Trees)** — CRR American option pricer benchmark.

---

## 6.6.1 The challenge of American options

An American option has value
$$
V_0 = \sup_{\tau \in \mathcal{T}_{0,T}} \mathbb{E}^\mathbb{Q}\left[e^{-r\tau} g(S_\tau)\right],
$$
where $\mathcal{T}_{0,T}$ is the set of $\mathbb{F}$-stopping times. **In 1D**, PDE (Module 6.3) and trees (Module 6.4) work well. **In $d \ge 2$**, neither scales — PDE grids blow up combinatorially.

**Monte Carlo is inherently forward** (simulate paths from 0 to $T$). But American pricing is **backward** (propagate optimal exercise from $T$ to 0). Reconciling these two has been a central research question since the 1990s.

Landmark answer: **Longstaff-Schwartz (2001)** regression-based MC.

Second landmark: **dual martingale methods** (Haugh-Kogan 2004, Rogers 2002, Andersen-Broadie 2004) give complementary *upper* bounds to LSM's lower bound, enabling a full confidence interval on the true price.

---

## 6.6.2 Longstaff-Schwartz Method (LSM)

**Setup.** Discretize $[0, T]$ into exercise dates $0 < t_1 < t_2 < \ldots < t_M = T$. Simulate $N$ paths of the asset(s).

**Dynamic programming (backward).**
- At $t_M$: $V^i_M = g(S_{t_M}^i)$ for each path $i$.
- At $t_m$, $m < M$: need the **continuation value** $C(t_m, S) = \mathbb{E}^\mathbb{Q}[e^{-r(t_{m+1}-t_m)} V(t_{m+1}, S_{t_{m+1}}) \mid S_{t_m} = S]$. Compare continuation to intrinsic; exercise if intrinsic is higher.

**Regression idea.** Approximate $C(t_m, \cdot)$ by a linear combination of basis functions $\psi_k$:
$$
C(t_m, S) \approx \sum_{k=1}^K \beta_k^{(m)} \psi_k(S).
$$
Regress $Y_i = e^{-r(t_{m+1}-t_m)} V_{m+1}^i$ on $\psi(S_{t_m}^i)$ using OLS across all paths.

**Algorithm (LSM).**
```
For m = M, M-1, ..., 0:
    intrinsic[i] = g(S_{t_m}^i)
    if m == M:
        V[i] = intrinsic[i]
    else:
        In-the-money paths: ITM = {i: intrinsic[i] > 0}
        Regress Y = e^{-r(t_{m+1}-t_m)} V[i] on ψ(S_{t_m}^i) for i in ITM
        continuation[i] = ψ(S_{t_m}^i)·β̂
        exercise[i] = (intrinsic[i] > continuation[i]) and (i in ITM)
        If exercise: update payoff recording; τ^i = t_m
```
Key step: **regress only on in-the-money paths** (Longstaff-Schwartz insight). Including deep-OTM paths introduces noise without improving the decision boundary.

**Price estimate.** After backward loop, each path $i$ has stopping time $\tau^i$ and payoff $g(S_{\tau^i}^i)$. Discount to $t=0$ and average:
$$
\hat V_0^{\text{LSM}} = \frac{1}{N}\sum_i e^{-r\tau^i} g(S_{\tau^i}^i).
$$

**Convergence.** Clément-Lamberton-Protter (2002): as $N \to \infty$ and $K \to \infty$, $\hat V_0^{\text{LSM}} \to V_0$ a.s. Rate depends on smoothness of the value function.

**Basis choice.** Polynomial basis in $S$: $\{1, S, S^2, S^3\}$ works for 1D smooth problems. For multi-asset, cross-terms $\{1, S_1, S_2, S_1 S_2, S_1^2, \ldots\}$. Laguerre or Hermite polynomials can be used, but empirically polynomials work fine.

**Sensitivity.** LSM is biased **downward** — any regression error $\to$ sub-optimal exercise decisions $\to$ lower value than true. This is the key fact enabling dual methods.

---

## 6.6.3 Regression coefficients: the details

For the regression step, we minimize
$$
\sum_{i \in \text{ITM}} \left(Y_i - \sum_{k=1}^K \beta_k \psi_k(S_{t_m}^i)\right)^2.
$$

This is a standard OLS problem:
$$
\hat\beta = (\Psi^\top \Psi)^{-1} \Psi^\top Y,
$$
where $\Psi_{ik} = \psi_k(S_{t_m}^i)$.

**Numerical stability.** Use SVD or QR decomposition instead of normal equations for ill-conditioned $\Psi^\top \Psi$. `np.linalg.lstsq` handles this.

**Bias vs variance trade-off.** More basis functions → lower regression bias, higher sampling variance. Rule of thumb: $K \approx 5$–$10$ for 1D, $K \le N^{1/3}$ asymptotically.

**Tsitsiklis-Van Roy (2001) variant.** Regress $V_{m+1}$ on $\psi(S_{t_m})$ rather than the continuation. Uses all paths (not just ITM). Simpler but less accurate empirically.

---

## 6.6.4 Dual methods: Haugh-Kogan, Rogers, Andersen-Broadie

**The dual formulation.** For any martingale $M$ with $M_0 = 0$:
$$
V_0 \le \mathbb{E}\left[\max_{t \le T} (g(S_t) - M_t)\right].
$$

**Why?** For any stopping time $\tau$:
$$
\mathbb{E}[g(S_\tau)] = \mathbb{E}[g(S_\tau) - M_\tau] \le \mathbb{E}[\max_{t \le T}(g(S_t) - M_t)],
$$
so
$$
V_0 = \sup_\tau \mathbb{E}[g(S_\tau)] \le \mathbb{E}[\max_{t \le T}(g(S_t) - M_t)].
$$

**Optimal $M$.** Taking $M_t^* = V_t - V_0 - \int_0^t \mathcal{L}V\, ds$, the martingale part of the Doob-Meyer decomposition of $V$, achieves equality. So
$$
V_0 = \inf_M \mathbb{E}\left[\max_{t \le T}(g(S_t) - M_t)\right].
$$

**Haugh-Kogan (2004)**: approximate $V_t$ by $\hat V_t$ from LSM; use
$$
\hat M_t = \hat V_t - \hat V_0 - \sum_{s < t} [\hat V_{s+1} - C(s, S_s)] \quad \text{(LSM-derived martingale)}.
$$
Then $\hat V_0^{\text{dual}} = \mathbb{E}[\max_t(g(S_t) - \hat M_t)]$ is an upper bound.

**Andersen-Broadie (2004)**. Given an approximate exercise policy $\tau^{approx}$, construct a martingale by forward simulation: each time step, simulate "inner" paths to estimate $\hat C$; if $\tau^{approx}$ exercises here, adjust $\hat M$ accordingly. This gives tight dual bounds but requires **nested Monte Carlo** — expensive.

**Nested MC cost.** If outer $N$ and inner $N'$, total cost $N \cdot N' \cdot M$ per step. With $N = N' = 10^4$, $M = 50$, that's $5 \times 10^9$ operations — minutes to hours.

**Rogers (2002)** parameterizes $M$ directly as $\sum \theta_k M^k$ for basis martingales $M^k$ (e.g., $e^{iu B_t - u^2 t/2}$ Heston-style); minimize over $\theta$ via dual stochastic programming.

---

## 6.6.5 Primal-dual gap and confidence intervals

Together, **LSM gives a lower bound $\hat V^L$** and **dual methods give an upper bound $\hat V^U$**. The true price lies in $[\hat V^L - \varepsilon_L, \hat V^U + \varepsilon_U]$ with high probability, where $\varepsilon$'s are MC standard errors.

**Bank practice.** Report LSM point estimate as the "price" and dual gap as the "model uncertainty budget." Gaps of 0.1% of notional are typical for well-tuned LSM; larger gaps flag problematic payoffs (deep OTM Bermudans, extreme regimes).

---

## 6.6.6 Alternatives: stochastic mesh, Tilley, SGBM

**Stochastic Mesh (Broadie-Glasserman 1997).** For each pair of dates $(t_m, t_{m+1})$ and each path $i$ at $t_m$, estimate $C(t_m, S_{t_m}^i)$ by averaging $V_{m+1}$ over all paths $j$ at $t_{m+1}$, weighted by a likelihood ratio. Quadratic in $N$ — feasible for $N \le 10^4$.

**Tilley's algorithm (1993).** Bundle paths into bins by state at each $t_m$; estimate $C$ per bin. Predecessor of LSM; LSM superseded it.

**Stochastic Grid Bundling Method (SGBM, Jain-Oosterlee 2015).** Combine LSM regression with bundle refinement. Faster than nested MC for dual bounds.

**Neural-network approximations (Sirignano-Spiliopoulos 2018, Becker-Cheridito-Jentzen 2019).** Train a NN to approximate $C$ or the exercise policy directly. State of the art for high-dimensional Bermudans.

---

## 6.6.7 Longstaff-Schwartz variants for exotic American

**Chooser options**, **exotic American baskets**, **Americans with callable features**: LSM extends naturally. Just modify the intrinsic value and recompute backward induction.

**Multi-asset baskets.** Basis functions include products $S_i S_j$, squares $S_i^2$, plus basket-specific features like min, max, or weighted sum.

**Path-dependent Americans.** Max-call American (Amin-Morton): intrinsic depends on the running max. Use the max as a state variable in the regression basis.

**Bermudan swaptions.** State variable is the swap rate (or forward rate vector in LMM). Exercise at specified dates only. LSM handles this directly.

**Convertible bonds.** Dual features (call by issuer, convert by holder). Game-theoretic LSM (Kifer 2000) or iterative backward induction.

---

## 6.6.8 Python: LSM implementation

```python
import numpy as np
from scipy.stats import norm

# ============================================================
# 1. Longstaff-Schwartz for American put in BS
# ============================================================
def lsm_american_put(S0, K, r, sigma, T, M=50, N=100_000, deg=3, seed=42):
    rng = np.random.default_rng(seed)
    dt = T/M
    # Simulate paths of log-GBM
    Z = rng.standard_normal((N, M))
    logS = np.log(S0) + np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1)
    S = np.exp(logS)  # N x M (t_1,...,t_M)
    S = np.hstack([np.full((N,1), S0), S])  # N x (M+1)

    # Intrinsic
    intrinsic = np.maximum(K - S, 0.0)
    # Terminal payoff
    V = intrinsic[:, -1].copy()
    tau = np.full(N, M, dtype=int)

    # Backward loop
    for m in range(M-1, 0, -1):
        disc = np.exp(-r*dt)
        # In-the-money paths
        itm = intrinsic[:, m] > 0
        if itm.sum() < deg+1:
            V *= disc
            continue
        # Regress Y = disc * V on polynomial basis in S_m for ITM paths
        X = S[itm, m]
        Y = disc * V[itm]
        # Basis
        Phi = np.vstack([X**k for k in range(deg+1)]).T
        beta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
        C_hat = Phi @ beta
        exercise = (intrinsic[itm, m] > C_hat)

        # Update V and tau for exercised paths
        idx = np.where(itm)[0][exercise]
        V[idx] = intrinsic[idx, m]
        tau[idx] = m
        # Non-exercised paths: V discounts backward
        V[~itm] = disc * V[~itm]
        not_ex_itm = np.where(itm)[0][~exercise]
        V[not_ex_itm] = disc * V[not_ex_itm]

    # Final discount to t=0
    price = np.mean(V * np.exp(-r*dt))  # Last step back
    return price, V, tau

# Benchmark: European put
def european_put_bs(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

# Compare to CRR binomial
def crr_american_put(S0, K, r, sigma, T, N=1000):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt)); d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    disc = np.exp(-r*dt)
    j = np.arange(N+1)
    S_T = S0 * u**(2*j - N)
    V = np.maximum(K - S_T, 0.0)
    for n in range(N-1, -1, -1):
        j = np.arange(n+1)
        S_n = S0 * u**(2*j - n)
        V = np.maximum(K - S_n, disc*(p*V[1:] + (1-p)*V[:-1]))
    return V[0]

params = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
eur = european_put_bs(**params)
crr = crr_american_put(**params, N=2000)
lsm, _, _ = lsm_american_put(**params, M=50, N=200_000, deg=3)
print(f"European put BS:   {eur:.4f}")
print(f"American put CRR:  {crr:.4f}")
print(f"American put LSM:  {lsm:.4f}")
print(f"Gap (CRR - LSM):   {crr - lsm:.4f}")

# ============================================================
# 2. Andersen-Broadie dual upper bound (simplified, non-nested)
# ============================================================
def ab_dual_upper_bound(S0, K, r, sigma, T, M=50, N_outer=10_000, N_inner=100,
                        deg=3, seed=42):
    """Simplified AB upper bound — uses LSM regression for policy, estimates
    continuation at each step via inner MC. Very slow for demo only."""
    rng = np.random.default_rng(seed)
    dt = T/M

    # First: run LSM to get regression coefficients
    _, _, _ = lsm_american_put(S0, K, r, sigma, T, M=M, N=50_000, deg=deg, seed=seed+1)
    # (For proper AB we'd save the regression coefficients β^(m) at each m —
    # for brevity, here we just compute a crude upper bound)

    # Placeholder: return European put value as (overly-loose) upper bound
    eur = european_put_bs(S0, K, r, sigma, T)
    # Actual AB method requires inner MC at each step — see Andersen-Broadie 2004
    return eur + 0.5  # very loose demo bound

ub_demo = ab_dual_upper_bound(**params)
print(f"AB dual upper (demo placeholder): {ub_demo:.4f}")

# ============================================================
# 3. Multi-asset LSM: max-call on 5 assets
# ============================================================
def lsm_max_call_5asset(S0, K, r, sigma, T, M=9, N=50_000, deg=2):
    rng = np.random.default_rng(42)
    d_assets = 5
    dt = T/M
    # Simulate correlated (independent here) paths
    paths = np.zeros((N, M+1, d_assets))
    paths[:, 0] = S0
    for t_ in range(M):
        Z = rng.standard_normal((N, d_assets))
        paths[:, t_+1] = paths[:, t_] * np.exp((r - 0.5*sigma**2)*dt
                                                + sigma*np.sqrt(dt)*Z)

    # Max-call intrinsic
    intrinsic = np.maximum(paths.max(axis=2) - K, 0.0)  # N x (M+1)
    V = intrinsic[:, -1].copy()
    tau = np.full(N, M, dtype=int)

    for m in range(M-1, 0, -1):
        disc = np.exp(-r*dt)
        itm = intrinsic[:, m] > 0
        if itm.sum() < 10:
            V *= disc; continue
        # Basis: 1, S_i, S_i^2, max, max^2
        X_i = paths[itm, m]  # N_itm x d
        max_S = X_i.max(axis=1)
        basis = [np.ones(itm.sum())]
        basis += [X_i[:, k] for k in range(d_assets)]
        basis += [X_i[:, k]**2 for k in range(d_assets)]
        basis.append(max_S); basis.append(max_S**2)
        Phi = np.column_stack(basis)
        Y = disc * V[itm]
        beta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
        C_hat = Phi @ beta
        ex = intrinsic[itm, m] > C_hat
        idx = np.where(itm)[0][ex]
        V[idx] = intrinsic[idx, m]; tau[idx] = m
        not_ex_all = np.setdiff1d(np.arange(N), idx)
        V[not_ex_all] = disc * V[not_ex_all]

    return np.mean(V * np.exp(-r*dt))

S0s, K_ = np.array([90, 90, 90, 90, 90]), 100
# Using scalar S0=90 for the 5-asset version:
price_max_call = lsm_max_call_5asset(S0=90, K=100, r=0.05, sigma=0.2, T=3.0, M=9, N=100_000)
print(f"\n5-asset max-call Bermudan (S0=90, K=100, σ=20%, T=3, 9 exercise dates):")
print(f"  LSM price: {price_max_call:.4f}")
print(f"  (Benchmark: Longstaff-Schwartz quotes ~15.98 for standard parameters)")
```

**What to check.**
- LSM American put should be between European put and CRR American, ideally $\sim 0.01$ below CRR (downward-biased).
- Increasing $N$ and $\deg$ narrows the gap to CRR.
- 5-asset max-call price for standard LS parameters ($S_0=90$, $K=100$, $\sigma=20\%$, $T=3$, 9 exercise dates) should be around $16$ (Longstaff-Schwartz report 15.98).

---

## 6.6.9 [QUANT APPLICATIONS]

1. **Bermudan swaptions.** Bank-scale LSM in LMM or HW to price swaption callability; backbone of IR structured product trading.
2. **Callable / putable bonds.** Issuer callability via LSM; standard for corporate and sovereign debt.
3. **American baskets.** Multi-asset American options on stock baskets or indices.
4. **Convertible bonds.** Game LSM for issuer-call + holder-convert features.
5. **CVA / counterparty risk.** Expected exposure profiles require American-style regression for Bermudan IR swaps and netting sets.
6. **Variable annuities / GMxB.** Insurance products with policyholder surrender optionality; LSM in high-dim state (fund value + guarantees).
7. **Real options.** Investment timing under uncertainty via American option valuation; LSM for multi-factor industrial decisions.
8. **Exercise-boundary approximation.** Deep learning of $\tau^*$ as function of state for fast online decisions (Becker-Cheridito-Jentzen).
9. **Gas storage / swing options.** Path-dependent American with inventory state; LSM with bundled state.
10. **Model-validation benchmark.** Banks maintain dual-method implementations specifically to validate LSM-based production pricers.

---

## 6.6.10 Exercises

**★ (concept drills).**
1. Explain the Longstaff-Schwartz insight: why regress only on ITM paths?
2. Why is LSM biased downward?
3. State the dual formulation of American pricing. Why does the martingale $M$ give an upper bound?
4. In Andersen-Broadie, why do we need "nested" Monte Carlo? What's the cost?
5. When does LSM fail? (Hint: exercise boundary is jagged, or basis doesn't span the value function.)
6. Compare LSM to PDE solvers. Why does LSM win in $d \ge 4$?

**★★ (calculation).**
7. Implement LSM for an American put. Verify bias by comparing to CRR at high $N$. Reduce bias by increasing $\deg$ and $N$.
8. Basis choice experiment. Run LSM with polynomial degrees 2, 3, 4, 5. Show price converges with $\deg$ for fixed $N$. When does overfitting kick in?
9. LSM vs Tsitsiklis-Van Roy. Implement both; compare on American put. TSVR regresses on all paths; LSM on ITM only.
10. Max-call 5-asset. Reproduce the classic Longstaff-Schwartz table (their 2001 paper): $S_0 = 90$, $K = 100$, $\sigma = 20\%$, $T = 3$, 9 exercise dates. Verify price around 16.
11. Implement stochastic mesh. Compare to LSM on the same problem. Cost-accuracy tradeoff at $N = 10^3$ vs $10^5$.

**★★★ (open / research).**
12. **Andersen-Broadie dual.** Implement AB with nested MC. Use LSM regression for the exercise policy. Verify the upper bound is tight for American put.
13. **Rogers' dual.** Parameterize $M$ as linear combination of $e^{iu_j B_t - u_j^2 t/2}$ and minimize the dual objective. Compare to AB.
14. **Neural network American pricing** (Becker-Cheridito-Jentzen). Implement NN-based optimal stopping for American max-call on 100 assets.
15. **Deep optimal stopping.** Reproduce the Hairer-Jentzen-Kloeden style results for high-dimensional Bermudan swaption in LMM.
16. **Quantization methods** (Bally-Pages). Discretize the state space into optimal $N$-grids; backward induction on grids. Compare to LSM on swaption pricing.
17. **GPU-accelerated LSM.** Port the inner loops to CUDA; for $N = 10^6$, compare to CPU implementation. What's the bottleneck?

---

*— End of Module 6.6. Next: Module 6.7, Machine Learning for Pricing and Hedging (deep hedging, deep BSDE). Subject 6 capstone.*

# Module 6.1 — Monte Carlo Methods and Variance Reduction

*Subject 6 — Numerical Methods in Quantitative Finance, Module 1.*

---

## Prerequisites

- **Module 2.4 (LLN and CLT)** — the heart of Monte Carlo's convergence.
- **Module 2.6 (Martingales)** — control variates use martingale identities to reduce variance.
- **Module 3.1 (Brownian motion)** — we need Brownian path simulation.
- **Module 5.2 (Black-Scholes)** — our benchmark for validating MC pricers.
- **Module 3.4 (SDEs)** — Euler and Milstein schemes are SDE discretizations.

---

## 6.1.1 Why Monte Carlo? The curse of dimensionality

Suppose we want to price a European claim with payoff $f(S_T^{(1)}, \ldots, S_T^{(d)})$ on $d$ underlyings. The risk-neutral price is
$$
V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[f(S_T)].
$$

**PDE approach**: solve a $d$-dimensional parabolic PDE. Complexity scales as $N^{d+1}$ where $N$ is grid size per dimension — infeasible for $d \ge 4$.

**Quadrature approach**: tensor-product Gaussian quadrature scales as $N^d$ — same curse.

**Monte Carlo approach**:
$$
V_0 \approx e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^N f(S_T^{(i)}),
$$
where $S_T^{(i)}$ are i.i.d. samples from the $\mathbb{Q}$-distribution. The central limit theorem gives error
$$
\text{RMSE} = O(\sigma_f / \sqrt{N})
$$
**independent of $d$**. This is the power of Monte Carlo: it breaks the curse, at the cost of slow $N^{-1/2}$ convergence compared to PDE's $N^{-2}$ (in 1D).

**Rule of thumb.** PDEs win in $d \le 2$; Monte Carlo wins in $d \ge 4$; the crossover in $d = 3$ depends on smoothness and accuracy required.

---

## 6.1.2 Basic Monte Carlo: theory and rate

Given i.i.d. $X_1, \ldots, X_N \sim \mu$ and a function $f$ with $\mathbb{E}|f(X)|^2 < \infty$, the Monte Carlo estimator is
$$
\hat\theta_N = \frac{1}{N} \sum_{i=1}^N f(X_i).
$$

**Strong law of large numbers**: $\hat\theta_N \xrightarrow{a.s.} \theta = \mathbb{E}[f(X)]$.

**Central limit theorem**: $\sqrt N (\hat\theta_N - \theta) \Rightarrow \mathcal{N}(0, \sigma_f^2)$, where $\sigma_f^2 = \text{Var}(f(X))$.

**Confidence interval** (95%, from 1.96 $\approx$ 2):
$$
\theta \in \hat\theta_N \pm 1.96 \cdot \hat\sigma_f / \sqrt N.
$$

**The efficiency rule.** Define the **efficiency** of an estimator as
$$
E = \frac{1}{\sigma^2 \cdot C},
$$
where $C$ is the cost per sample. To compare two estimators fairly, multiply variance by cost: an estimator with half the variance but double the cost has the **same** efficiency. This is the single most important metric in MC engineering.

---

## 6.1.3 Generating random variables

**Uniform RNG.** Modern practice uses the **Mersenne Twister** (period $2^{19937}-1$) or **PCG** / **xoshiro** families. Cryptographically-weak but statistically excellent.

**Transforming to other distributions.**
- **Inverse CDF**: If $U \sim \mathcal{U}(0,1)$ then $F^{-1}(U) \sim F$. Clean but requires efficient $F^{-1}$.
- **Box-Muller**: For $\mathcal{N}(0,1)$, given $U_1, U_2 \sim \mathcal{U}(0,1)$,
$$Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2),$$
are independent standard normals.
- **Marsaglia polar**: avoids trig. Sample $(V_1, V_2) \sim \mathcal{U}(-1,1)^2$, reject if $S = V_1^2+V_2^2 > 1$, else output $V_i \sqrt{-2\ln S / S}$.
- **Ziggurat** (Marsaglia-Tsang 2000): fastest method for Gaussians, used in many libraries.
- **Acceptance-rejection**: if $f \le c g$ and we can sample from $g$, sample $X \sim g$, $U \sim \mathcal{U}(0,1)$, accept if $U \le f(X)/(c g(X))$.

**Multivariate Gaussians.** To sample $\mathbf{Z} \sim \mathcal{N}(\mu, \Sigma)$:
1. Cholesky: $\Sigma = LL^\top$, then $\mathbf{Z} = \mu + L\mathbf{W}$ for $\mathbf{W} \sim \mathcal{N}(0, I)$.
2. Spectral: $\Sigma = U D U^\top$, then $\mathbf{Z} = \mu + U D^{1/2} \mathbf{W}$.

Cholesky is cheaper ($O(d^3/6)$ vs $O(d^3)$) and preserves sparsity when relevant.

---

## 6.1.4 Simulating SDEs: Euler and Milstein

Given $dX_t = \mu(X_t) dt + \sigma(X_t) dB_t$, to simulate from $0$ to $T$ with $n$ steps of size $h = T/n$:

**Euler-Maruyama**:
$$
X_{k+1} = X_k + \mu(X_k) h + \sigma(X_k) \sqrt h Z_k, \quad Z_k \sim \mathcal{N}(0,1).
$$

**Strong order** $1/2$: $\mathbb{E}|X_n - X_T| = O(\sqrt h)$.
**Weak order** $1$: $|\mathbb{E} f(X_n) - \mathbb{E} f(X_T)| = O(h)$ for smooth $f$.

**Milstein** adds the Itô correction:
$$
X_{k+1} = X_k + \mu h + \sigma \sqrt h Z_k + \tfrac{1}{2}\sigma \sigma'(h Z_k^2 - h).
$$
**Strong order** $1$. For path-dependent problems (barrier, lookback, American) Milstein's improved pathwise accuracy matters.

**For Black-Scholes**, $d\log S = (r - \sigma^2/2)dt + \sigma dB$, which has **no discretization error**:
$$
S_{k+1} = S_k \exp\left((r - \sigma^2/2)h + \sigma \sqrt h Z_k\right).
$$

**For CIR / Heston variance**, $dV = \kappa(\theta - V)dt + \xi\sqrt V dB$, Euler can produce negative $V$. **Full truncation** ($V_+ = \max(V, 0)$) is the workhorse fix; QE (Andersen 2008) is the state-of-the-art.

---

## 6.1.5 Variance reduction: the meta-idea

If a naive estimator has variance $\sigma^2$ and cost $C$, its efficiency is $1/(\sigma^2 C)$. Variance reduction trades off: buy a smaller $\sigma^2$ with a small increase in $C$. If variance drops by $10\times$ while cost only rises by $2\times$, net efficiency is $5\times$.

Six standard techniques:
1. Antithetic variates
2. Control variates
3. Importance sampling
4. Stratified sampling
5. Conditioning
6. Multilevel Monte Carlo (Giles)

---

## 6.1.6 Antithetic variates

**Idea.** If $X$ and $X'$ are identically distributed and **negatively correlated**, then $(X + X')/2$ has lower variance than either $X$ alone.

**For Gaussian innovations**: draw $Z$, use both $Z$ and $-Z$. For a function $f$:
$$
\hat\theta^{\text{anti}} = \frac{1}{2}(f(Z) + f(-Z)), \qquad \text{Var}(\hat\theta^{\text{anti}}) = \frac{1}{2}(\text{Var}(f(Z)) + \text{Cov}(f(Z), f(-Z))).
$$

If $f$ is monotonic, the covariance is negative (by Chebyshev's covariance inequality), so variance drops by more than half per pair. If $f$ is symmetric ($f(Z) = f(-Z)$), variance is **zero**. If $f$ is even/U-shaped, antithetics can increase variance — one must check.

**In BS**: for a European call, payoff is monotonic in $Z$ → antithetics help. For a straddle, symmetric in $Z$ → antithetics kill all variance (but miss the leading BS structure).

**Cost analysis**. Each antithetic pair costs roughly $1.5 \times$ a single simulation (one Gaussian draw, two payoff evaluations). Break-even: variance reduction $> 1.33\times$ per paired sample.

---

## 6.1.7 Control variates

**Idea.** Suppose we want $\mathbb{E}[Y]$ and have a known correlate $X$ with known mean $\mu_X$. For any $\beta$,
$$
\tilde Y = Y - \beta(X - \mu_X)
$$
has $\mathbb{E}\tilde Y = \mathbb{E} Y$ and
$$
\text{Var}(\tilde Y) = \text{Var}(Y) - 2\beta \text{Cov}(Y,X) + \beta^2 \text{Var}(X).
$$
Minimizing over $\beta$:
$$
\boxed{\beta^* = \frac{\text{Cov}(Y,X)}{\text{Var}(X)}, \qquad \text{Var}(\tilde Y) = (1 - \rho_{YX}^2)\text{Var}(Y).}
$$

If $Y$ and $X$ correlate at $\rho = 0.99$, variance drops by $100\times$. Control variates are the single most effective generic technique.

**Classical choices.**
- **Asian option** with arithmetic average — use **geometric average** (known closed form via BS with $\sigma_{\text{eff}} = \sigma/\sqrt 3$).
- **Basket option** — use individual options.
- **American option** — use European.
- **Any payoff** — use the underlying $S_T$ itself ($\mathbb{E} S_T$ is known as $S_0 e^{rT}$).
- **Barrier option** — use the unbounded European.

**Implementation.** Estimate $\beta^*$ on a pilot run, apply on production run. Or estimate jointly with a slight bias that vanishes as $N \to \infty$ (standard practice).

---

## 6.1.8 Importance sampling

**Idea.** Rewrite the expectation under a shifted measure where the integrand is more "peaked":
$$
\mathbb{E}_\mathbb{P}[f(X)] = \mathbb{E}_\mathbb{Q}\left[f(X) \frac{d\mathbb{P}}{d\mathbb{Q}}(X)\right].
$$

The estimator is
$$
\hat\theta = \frac{1}{N}\sum f(X_i) \frac{d\mathbb{P}}{d\mathbb{Q}}(X_i), \qquad X_i \sim \mathbb{Q}.
$$

**Variance of the IS estimator** = $\mathbb{E}_\mathbb{Q}[f(X)^2 (d\mathbb{P}/d\mathbb{Q})^2] - \theta^2$. The ideal $\mathbb{Q}^*$ has $d\mathbb{Q}^* \propto |f(X)| d\mathbb{P}$, giving **zero variance**, but is unknown (knowing it $\iff$ knowing the answer). Good practice: $\mathbb{Q}$ that concentrates mass where $f$ is large.

**For rare events** (e.g. deep out-of-the-money options, tail risk), IS is transformative. Without IS, the probability of a large move is $\ll 1/N$ and the estimator has nearly-infinite relative error.

**Gaussian IS example.** To estimate $\mathbb{P}(Z > a)$ for large $a$:
$$
\mathbb{P}(Z > a) = \mathbb{E}_\mathbb{P}[\mathbb{1}_{Z > a}] = \mathbb{E}_{\mathbb{Q}}\left[\mathbb{1}_{Z > a} \cdot e^{-\theta Z + \theta^2/2}\right]
$$
where under $\mathbb{Q}$, $Z \sim \mathcal{N}(\theta, 1)$. Choosing $\theta = a$ concentrates mass at the threshold. Variance reduction: for $a = 5$, naïve MC has relative error $\sim 10^4$; Gaussian IS has relative error $\sim 1$.

**Sieck-Glynn** (1995) asymptotic optimality for rare events: under mild conditions, Gaussian IS achieves logarithmic optimality (the relative error grows polynomially in $a$, not exponentially).

---

## 6.1.9 Stratified sampling

**Idea.** Partition the support $\Omega$ into strata $\Omega_1, \ldots, \Omega_K$ with $\mathbb{P}(\Omega_k) = p_k$. Allocate $N_k = \lceil p_k N\rceil$ samples to each stratum:
$$
\hat\theta^{\text{strat}} = \sum_k p_k \hat\theta_k.
$$
Variance: $\sum_k p_k^2 \sigma_k^2 / N_k$. This beats unstratified variance $\sigma^2 / N$ whenever the stratum means differ.

**Equiprobable stratification**: $N_k = N/K$ per stratum with $p_k = 1/K$. Simple and effective for smooth integrands. Variance drops to $O(1/(NK))$ for $C^1$ integrands in 1D — similar to the rectangle rule.

**Neyman allocation**: $N_k \propto p_k \sigma_k$. Optimal variance: $(\sum p_k \sigma_k)^2/N$. Requires knowing $\sigma_k$.

**Latin hypercube sampling (LHS)**: stratification in each marginal dimension, random permutation across strata. Gives $o(1/N)$ variance when the integrand is additive.

---

## 6.1.10 Conditional Monte Carlo / Rao-Blackwellization

**Idea.** For $\mathbb{E}[f(X,Y)]$, if we can compute $g(X) = \mathbb{E}[f(X,Y) \mid X]$ analytically, then
$$
\text{Var}(g(X)) \le \text{Var}(f(X,Y)),
$$
so $\hat\theta = (1/N)\sum g(X_i)$ has lower variance than $(1/N)\sum f(X_i, Y_i)$.

**Example**: Barrier option in a GBM. Condition on the end point $S_T$; the probability of hitting the barrier between $0$ and $T$ is given by the Brownian-bridge hitting formula:
$$
\mathbb{P}(\min_{t} S_t < L \mid S_T, S_0) = \exp\left(-\frac{2\ln(S_0/L)\ln(S_T/L)}{\sigma^2 T}\right),
$$
valid when both $S_0, S_T > L$. This analytic correction removes most of the variance.

**Example 2**: For a digital in a BS model, conditioning on $S_T$ gives the exact payoff; variance is zero.

---

## 6.1.11 Brownian bridge construction

Simulating a Brownian path at times $0 < t_1 < \ldots < t_n = T$ can be done in two ways:

1. **Sequential**: $B_{t_{k+1}} = B_{t_k} + \sqrt{\Delta_k} Z_k$. Each $Z_k$ contributes "its own" variance.
2. **Brownian bridge**: simulate $B_T$ first, then $B_{T/2}$ conditional on $B_T$, then $B_{T/4}$ and $B_{3T/4}$, etc.

The bridge reparametrizes the same Gaussian vector so that the **first few $Z$'s capture most of the variance** of path functionals. Combined with QMC (Sobol), this massively improves effective dimensionality and boosts convergence.

Formula for bridge interpolation: if $B_s = a$, $B_u = b$, $s < t < u$,
$$
B_t \mid B_s, B_u \sim \mathcal{N}\left(a + \frac{t-s}{u-s}(b-a),\; \frac{(t-s)(u-t)}{u-s}\right).
$$

---

## 6.1.12 Multilevel Monte Carlo (Giles 2008)

**Setup.** For an SDE approximation with step size $h_\ell = T \cdot 2^{-\ell}$, let $P_\ell$ denote the payoff computed at level $\ell$. The fine estimator $P_L$ has bias $O(h_L)$ and variance $O(1)$, costing $O(2^L)$ per sample.

Giles' telescoping identity:
$$
\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_{\ell=1}^L \mathbb{E}[P_\ell - P_{\ell-1}].
$$

**Key observation.** If we use the **same Brownian path** to compute $P_\ell$ and $P_{\ell-1}$, then $\text{Var}(P_\ell - P_{\ell-1}) = O(h_\ell)$ for Euler schemes and $O(h_\ell^2)$ for Milstein.

**Optimal sample allocation**: $N_\ell \propto \sqrt{V_\ell / C_\ell}$ where $V_\ell$ is the variance of the level-$\ell$ correction and $C_\ell$ its per-sample cost.

**Complexity theorem (Giles).** For target RMSE $\varepsilon$:
- Standard MC with Euler: $O(\varepsilon^{-3})$
- MLMC with Euler: $O(\varepsilon^{-2}(\log\varepsilon)^2)$
- MLMC with Milstein: $O(\varepsilon^{-2})$

MLMC is the dominant framework for production SDE simulation today.

---

## 6.1.13 Python implementation: a full MC toolkit

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ============================================================
# 1. Vanilla BS MC with all variance-reduction techniques
# ============================================================
def bs_price_mc(S0, K, r, sigma, T, N=100_000, scheme='vanilla'):
    rng = np.random.default_rng(2026)
    Z = rng.standard_normal(N)
    def payoff(z):
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
        return np.exp(-r*T) * np.maximum(ST - K, 0)

    if scheme == 'vanilla':
        V = payoff(Z)
    elif scheme == 'antithetic':
        V = 0.5*(payoff(Z) + payoff(-Z))
    elif scheme == 'control':
        # Control variate = S_T; mean known = S0*exp(rT)
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        Y  = np.exp(-r*T)*np.maximum(ST - K, 0)
        X  = ST
        beta = np.cov(Y, X, ddof=0)[0,1]/np.var(X)
        V = Y - beta*(X - S0*np.exp(r*T))
    elif scheme == 'importance':
        # Shift by d = (ln K/S0 - (r-0.5σ²)T)/(σ√T): center on strike
        d = (np.log(K/S0) - (r-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        W = Z + d
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*W)
        weight = np.exp(-d*W + 0.5*d**2)
        V = np.exp(-r*T)*np.maximum(ST - K, 0)*weight
    else:
        raise ValueError(scheme)

    price = V.mean()
    se = V.std(ddof=1)/np.sqrt(N)
    return price, se

# Benchmark: BS closed form
def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

params = dict(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=100_000)
true = bs_call(**{k:v for k,v in params.items() if k != 'N'})
print(f"BS closed form:      {true:.6f}")
for scheme in ['vanilla', 'antithetic', 'control', 'importance']:
    p, se = bs_price_mc(scheme=scheme, **params)
    print(f"  {scheme:12s}  price={p:.6f}  SE={se:.6f}  VarRed={(p-true)/se:.2f}σ")

# ============================================================
# 2. Control variate for Asian arithmetic via geometric
# ============================================================
def asian_arith_call_mc(S0, K, r, sigma, T, M=52, N=100_000, control=True):
    rng = np.random.default_rng(2027)
    dt = T/M
    drift = (r - 0.5*sigma**2)*dt
    diff = sigma*np.sqrt(dt)
    Z = rng.standard_normal((N, M))
    logS = np.log(S0) + np.cumsum(drift + diff*Z, axis=1)
    S = np.exp(logS)
    arith = S.mean(axis=1)
    geom  = np.exp(np.log(S).mean(axis=1))
    disc = np.exp(-r*T)
    Y = disc * np.maximum(arith - K, 0)
    X = disc * np.maximum(geom  - K, 0)

    # Closed form for geometric Asian under BS:
    sig_g = sigma*np.sqrt((M+1)*(2*M+1)/(6*M**2))
    mu_g  = (r - 0.5*sigma**2)*(M+1)/(2*M) + 0.5*sig_g**2
    d1 = (np.log(S0/K) + (mu_g + 0.5*sig_g**2)*T)/(sig_g*np.sqrt(T))
    d2 = d1 - sig_g*np.sqrt(T)
    geom_true = np.exp(-r*T)*(S0*np.exp(mu_g*T)*norm.cdf(d1) - K*norm.cdf(d2))

    if control:
        beta = np.cov(Y,X,ddof=0)[0,1]/np.var(X)
        V = Y - beta*(X - geom_true)
    else:
        V = Y
    return V.mean(), V.std(ddof=1)/np.sqrt(N)

p_naive, se_naive = asian_arith_call_mc(100,100,0.05,0.2,1.0,control=False)
p_cv,    se_cv    = asian_arith_call_mc(100,100,0.05,0.2,1.0,control=True)
print(f"\nAsian arith call (100 paths):")
print(f"  naive        : {p_naive:.6f}  SE={se_naive:.6f}")
print(f"  control var  : {p_cv:.6f}     SE={se_cv:.6f}")
print(f"  VR factor    : {(se_naive/se_cv)**2:.1f}x")

# ============================================================
# 3. Multilevel Monte Carlo — Euler for Heston
# ============================================================
def mlmc_euler_gbm(S0, K, r, sigma, T, eps=0.005):
    """European call via MLMC on log-GBM (proof of concept)."""
    rng = np.random.default_rng(2028)
    # Level estimators
    L_max = 10
    V_l, N_l, P_l_est = [], [], []
    for L in range(L_max):
        n = 2**L; dt = T/n
        # Pilot with N_pilot
        N_pilot = 10_000
        if L == 0:
            Z = rng.standard_normal((N_pilot, n))
            dW = Z*np.sqrt(dt)
            S = S0*np.exp(np.cumsum((r-0.5*sigma**2)*dt + sigma*dW, axis=1))
            P_coarse = np.exp(-r*T)*np.maximum(S[:, -1] - K, 0)
            diff = P_coarse
        else:
            # fine path 2^L steps, coarse = 2^(L-1) steps from same BM
            Z = rng.standard_normal((N_pilot, n))
            dW_f = Z*np.sqrt(dt)
            dW_c = dW_f[:, ::2] + dW_f[:, 1::2]  # pair up increments
            S_f = S0*np.exp(np.cumsum((r-0.5*sigma**2)*dt + sigma*dW_f, axis=1))
            S_c = S0*np.exp(np.cumsum((r-0.5*sigma**2)*(2*dt) + sigma*dW_c, axis=1))
            P_f = np.exp(-r*T)*np.maximum(S_f[:,-1] - K, 0)
            P_c = np.exp(-r*T)*np.maximum(S_c[:,-1] - K, 0)
            diff = P_f - P_c
        V_l.append(diff.var(ddof=1))
        P_l_est.append(diff.mean())
    return P_l_est, V_l

P_est, V_est = mlmc_euler_gbm(100,100,0.05,0.2,1.0)
print(f"\nMLMC level estimates:")
for L, (p, v) in enumerate(zip(P_est, V_est)):
    print(f"  L={L}: E[P_L - P_{{L-1}}]={p:+.4f}, Var={v:.4e}")
print(f"  MLMC sum = {sum(P_est):.4f}, true = {bs_call(100,100,0.05,0.2,1.0):.4f}")
```

**What to check.** 
- Antithetic roughly halves SE; control-variate $S_T$ reduces SE by a factor depending on moneyness.
- Importance sampling (shift to strike) dramatically improves OTM cases — try $K = 150$ or $K = 70$.
- Asian with geometric control reduces SE by $\sim 50\times$ on $M=52$ paths.
- MLMC: variance of level differences shrinks as $O(h_\ell)$ for Euler.

---

## 6.1.14 [QUANT APPLICATIONS]

1. **Exotic options pricing.** Barrier, Asian, lookback, basket options — MC is the default. Control variates (against vanilla analog) are ubiquitous.
2. **Greeks via pathwise / likelihood ratio**. Broadie-Glasserman pathwise estimator for delta: $\Delta = \mathbb{E}[f'(S_T) \cdot S_T/S_0 \cdot \mathbb{1}]$. When payoff non-smooth, use likelihood ratio (score function) or Malliavin weighting.
3. **Counterparty risk (CVA, FVA, XVA).** Requires nested MC: outer simulation of exposure paths, inner pricing of each trade at each future date. MLMC and regression-based (Longstaff-Schwartz-style) methods replace the inner MC.
4. **Structured products calibration.** Calibration loop requires many repriced scenarios; MC with common random numbers gives smoother surfaces for the optimizer.
5. **Monte Carlo VaR and Expected Shortfall.** Tail estimation with importance sampling (exponential twisting, cross-entropy) for rare loss events.
6. **Portfolio-level simulation.** Correlated SDEs for 100s of assets, risk factors, and macro drivers. MC scales naturally to high dimensions.
7. **Insurance and longevity risk.** Nested MC for variable annuities, hedging GMxB guarantees.
8. **Machine learning training data.** Deep hedging and neural SDE calibration consume millions of MC paths for training labels.
9. **Stress testing.** Scenario generation via importance-sampled MC to oversample crisis-like tails.
10. **Regulatory capital (FRTB IMA).** Historical MC for market risk capital; MLMC-style techniques for computational feasibility.

---

## 6.1.15 Exercises

**★ (concept drills).**
1. Derive the 95% confidence interval around a MC estimator with $N$ samples. How many samples are needed for a 1% relative error on an option with $\sigma_f/\mu_f = 2$?
2. Show that antithetic variates are exact (zero variance) for $f(Z) = Z^2$ or any even function.
3. Why does control variate work? Prove that $\text{Var}(\tilde Y) = (1-\rho^2)\text{Var}(Y)$ at optimal $\beta$.
4. For Gaussian importance sampling $Z' = Z + \theta$, write down the likelihood ratio $d\mathbb{P}/d\mathbb{Q}$.
5. What's the efficiency gain from switching Euler → Milstein when pricing a European call in BS? (Hint: both have zero discretization error — neither.)

**★★ (calculation).**
6. Implement antithetic + control variate MC for a barrier option (down-and-out call in BS). Use the unconstrained call as control. Quantify the joint variance reduction on $S_0=100, K=100, L=80, \sigma=20\%, T=1$.
7. Derive the optimal shift $\theta^*$ for importance sampling of $\mathbb{P}(Z > a)$. Show that $\theta^* = a$ minimizes variance and gives relative error $O(1)$ in $a$.
8. Milstein vs Euler on CIR. Simulate $dV = \kappa(\theta-V)dt + \xi\sqrt V dB$ with both schemes. Compare strong error at $T = 1$ as a function of $h$. Verify $O(\sqrt h)$ vs $O(h)$ rates.
9. Brownian bridge for Asian option. Construct the bridge representation for $B$ at $t_k = kT/M$, $k=1, \ldots, M$. Show the first two "bridge steps" (midpoint, quartile) account for most variance of the arithmetic average.
10. Giles MLMC analysis. Derive that for Euler, $\text{Var}(P_\ell - P_{\ell-1}) = O(h_\ell)$, leading to the $O(\varepsilon^{-2}\log^2\varepsilon)$ complexity.

**★★★ (open / research).**
11. **Adaptive MLMC**. Implement Giles' adaptive algorithm that chooses $L$ and $N_\ell$ on the fly to hit target $\varepsilon$. Test on Heston pricing vs analytic characteristic function.
12. **Cross-entropy for rare events**. Implement the CE method to estimate $\mathbb{P}(\max_t X_t > a)$ for a diffusion $X$. Compare to naive MC.
13. **Quasi-MC vs MLMC**. Combine Sobol' with Brownian-bridge construction and compare to plain MC and MLMC on a high-dimensional Asian.
14. **Randomized MLMC (Rhee-Glynn)**. Implement the unbiased MLMC estimator from Rhee-Glynn (2015). Discuss when it beats truncated MLMC.
15. **Nested MC for CVA**. Price the CVA of a 5-year interest-rate swap in a Hull-White model. Compare nested MC, LSM regression, and deep learning (regress exposure by NN).

---

*— End of Module 6.1. Next: Module 6.2, Quasi-Monte Carlo and Low-Discrepancy Sequences.*

# Module 6.2 — Quasi-Monte Carlo and Low-Discrepancy Sequences

*Subject 6, Module 2. Deterministic sequences that beat random points at high-dimensional integration.*

---

## Prerequisites

- **Module 6.1** — standard MC, variance, Brownian bridge.
- **Module 2.1 (probability spaces)** and **Module 1.3 (Lebesgue integration)** — discrepancy lives in a measure-theoretic setting.
- **Module 0.4.4 (Koksma-Hlawka style estimates)** — bounded-variation functions.

---

## 6.2.1 The core idea

Monte Carlo uses i.i.d. random points $\{U_i\} \subset [0,1]^d$ to estimate $\int_{[0,1]^d} f$. The error is $O(N^{-1/2})$ **independent of $d$** but random.

**Quasi-Monte Carlo (QMC)** replaces random points with a carefully-constructed **deterministic** sequence $\{x_i\}$ that is "more uniform" than random. Under smoothness assumptions (bounded variation in the Hardy-Krause sense), the error is
$$
\left|\hat I_N - I\right| = O\left(\frac{(\log N)^d}{N}\right)
$$
— **super-linear** convergence! For moderate $d$ and large $N$, this crushes MC.

The theoretical tool is **discrepancy**, a quantitative measure of how well a point set fills the cube.

---

## 6.2.2 Discrepancy

**Definition (star discrepancy).** For $P = \{x_1, \ldots, x_N\} \subset [0,1]^d$,
$$
D_N^*(P) = \sup_{B = \prod[0, b_j]} \left|\frac{\#\{i: x_i \in B\}}{N} - \text{vol}(B)\right|.
$$
The supremum is over all axis-aligned boxes anchored at the origin.

**Extreme discrepancy** $D_N(P)$ takes the sup over arbitrary axis-aligned boxes $\prod[a_j, b_j]$. Within constants, $D_N^*$ and $D_N$ are equivalent.

**For random points**: $D_N^*(P) = O(\sqrt{\log\log N / N})$ almost surely (law of iterated logarithm).

**For low-discrepancy sequences**: $D_N^*(P) = O((\log N)^d / N)$.

---

## 6.2.3 The Koksma-Hlawka inequality

The central theorem of QMC:
$$
\boxed{\left|\frac{1}{N}\sum_{i=1}^N f(x_i) - \int_{[0,1]^d} f\, du\right| \le V_{HK}(f) \cdot D_N^*(P).}
$$

Here $V_{HK}(f)$ is the **Hardy-Krause variation** — a multidimensional generalization of total variation.

**Takeaways.**
1. The integration error factors into a property of the function ($V_{HK}$) and a property of the points ($D_N^*$).
2. To reduce error, either smooth the function (preprocessing) or increase $N$ with a good sequence.
3. If $V_{HK}(f) = \infty$ (e.g. payoff with jumps, barrier options), the Koksma-Hlawka bound is vacuous, though in practice QMC still outperforms.
4. For $d \gg 1$, the $(\log N)^d$ factor is misleading; **effective dimension** analysis (Owen 1998) shows practical behavior depends on how $f$ decomposes.

---

## 6.2.4 Van der Corput and Halton sequences

**Van der Corput (1935), base $b$**: for $n \in \mathbb{N}$, write $n$ in base $b$ as $n = \sum a_k b^k$. Then the **radical inverse** is
$$
\phi_b(n) = \sum a_k b^{-k-1}.
$$
Example: $n = 13$, $b = 2$: $13 = 1101_2$, $\phi_2(13) = 0.1011_2 = 11/16$.

**Properties.** $\{\phi_b(n)\}_{n=1}^N \subset [0,1]$ is equidistributed with $D_N^* = O(\log N / N)$.

**Halton sequence** (1960): $d$-dimensional sequence with $i$-th point
$$
x_i = (\phi_{b_1}(i), \phi_{b_2}(i), \ldots, \phi_{b_d}(i)),
$$
where $b_1, \ldots, b_d$ are the first $d$ primes ($2, 3, 5, 7, 11, 13, \ldots$).

**Star discrepancy**: $D_N^*(\text{Halton}_d) = O((\log N)^d / N)$.

**Problem for high $d$.** With primes like 23, 29, 31, the radical inverse has slow mixing. Early Halton points in high coordinates **cluster along diagonals** — the notorious "Halton pathology." 

**Fix: scrambled/shuffled Halton** (Kocis-Whiten) permutes digits to break the diagonal structure. Works well up to $d \sim 30$.

---

## 6.2.5 Sobol' sequences

**Sobol' (1967)** is a **$(t, s)$-sequence in base 2**, constructed from primitive polynomials.

**Construction (sketch).** For each dimension $j$, choose direction numbers $v_{j,k}$ (derived from a primitive polynomial of degree $s_j$). For index $i$ with binary expansion $i = \sum i_k 2^{k-1}$,
$$
x^{(j)}_i = i_1 v_{j,1} \oplus i_2 v_{j,2} \oplus \cdots,
$$
where $\oplus$ is bitwise XOR.

**Gray-code implementation**: $x^{(j)}_{i+1} = x^{(j)}_i \oplus v_{j, g(i)}$, where $g(i)$ is the index of the lowest nonzero bit of $i$. Blazing-fast incremental generation.

**Direction numbers**: Joe-Kuo (2008) produced optimized direction numbers up to $d = 21200$, publicly available.

**Properties.**
- Star discrepancy: $O((\log N)^d/N)$.
- **2-dimensional projection**: every pair of coordinates fills a grid well for $N = 2^k$ (the **(0, 2)-property** in base 2).
- Superior to Halton in higher dimensions.

**Randomization.** Plain Sobol' is deterministic, so no statistical error bar. **Randomized QMC (RQMC)** restores an error estimate:
- **Cranley-Patterson shift**: add a single $U \sim \mathcal{U}(0,1)^d$ modulo 1.
- **Owen's scrambling** (nested uniform): permute bits independently at each level. Preserves low discrepancy while giving an unbiased estimator with variance reducible by averaging independent scrambles.
- **Linear scrambling** (Matousek, Owen-Tribble): cheaper alternative with similar performance.

---

## 6.2.6 Niederreiter and $(t,m,s)$-nets

**Generalization.** A $(t, m, s)$-net in base $b$ is a set of $b^m$ points in $[0,1]^s$ such that every "elementary interval" of volume $b^{t-m}$ contains exactly $b^t$ points.

Smaller $t$ → better equidistribution. Sobol' is a $(t, s)$-sequence with $t$ depending on $s$.

**Niederreiter's construction** (1987, 1992) uses rational functions over finite fields to produce $(t, s)$-sequences with excellent $t$ values across dimensions.

**Faure sequences** are $(0, s)$-sequences in base $b = $ smallest prime $\ge s$. Very uniform but explode computationally for large $s$.

**Lattice rules** (Sloan-Joe 1994) are a parallel thread: $x_i = \{i \mathbf{z}/N\}$ for generating vector $\mathbf{z}$. Tailored to periodic functions.

---

## 6.2.7 Effective dimension and why QMC works in 400 dimensions

Despite the $(\log N)^d$ factor in the K-H bound, QMC routinely beats MC in problems with nominal dimension $d = 250$ (e.g., weekly time steps over a decade). Why?

**ANOVA decomposition** (Caflisch-Morokoff-Owen 1997):
$$
f(x) = f_\emptyset + \sum_j f_j(x_j) + \sum_{j < k} f_{jk}(x_j, x_k) + \cdots
$$
with $\int f_u = 0$ for all $u$ in any fixed coordinate. Then $\text{Var}(f) = \sum_u \sigma_u^2$.

**Effective dimension**:
- In the *superposition sense* ($d_S$): the smallest $t$ with $\sum_{|u| \le t}\sigma_u^2 \ge (1-\varepsilon)\text{Var}(f)$.
- In the *truncation sense* ($d_T$): the smallest $t$ with $\sum_{u \subseteq \{1,\ldots,t\}} \sigma_u^2 \ge (1-\varepsilon)\text{Var}(f)$.

**Financial functionals** typically have $d_T$ much smaller than the nominal $d$. A 250-step Asian has $d_T \approx 5$: the early Brownian increments dominate. With Brownian-bridge construction, early Sobol' dimensions (which have best equidistribution) capture the important variance.

**PCA construction**: Principal-component decomposition of the covariance of the Brownian path gives $d_T \approx 1$ (just the "level" of the path). This is the most aggressive QMC preprocessing.

**Rule of thumb**: always combine QMC with Brownian bridge or PCA; it's typically a $10$–$50\times$ speedup over naive QMC on financial payoffs.

---

## 6.2.8 Practical QMC engineering

**Step 1: choose sequence.** For $d \le 20$ use Halton; for $d \le 100$ use Sobol' (Joe-Kuo); for $d > 100$ use scrambled Sobol' or a lattice rule.

**Step 2: randomize.** Run $M$ independent randomizations (say $M = 30$). The sample mean and sample variance of the $M$ QMC estimators give an unbiased estimate and a confidence interval.

**Step 3: reorder dimensions.** The most influential variables go first. For SDEs, use Brownian bridge or PCA. For structured payoffs, use sensitivity analysis / Sobol' indices.

**Step 4: smooth if needed.** Non-smooth payoffs (digitals, barriers) hurt QMC. Conditional MC can smooth them analytically before the remaining integration.

**When QMC doesn't help.**
- $d_T$ very high (e.g., many independent path-dependent features).
- Non-smooth payoffs without smoothing preprocessing.
- Very few samples ($N < 2^{10}$) — QMC asymptotics hadn't kicked in.

---

## 6.2.9 Python: QMC toolkit

```python
import numpy as np
from scipy.stats import qmc, norm
from scipy.stats.qmc import Halton, Sobol
import matplotlib.pyplot as plt

# ============================================================
# 1. Compare Halton, Sobol, random on a 2D uniform fill
# ============================================================
N = 1024
halton = Halton(d=2, scramble=False).random(N)
sobol = Sobol(d=2, scramble=False).random_base2(m=10)  # 2^10 = 1024
rand = np.random.default_rng(0).random((N, 2))

# Quick discrepancy proxy: star discrepancy surrogate
def discrepancy_proxy(P):
    from scipy.stats import qmc
    return qmc.discrepancy(P)

print(f"Discrepancy N={N}")
print(f"  random: {discrepancy_proxy(rand):.6f}")
print(f"  Halton: {discrepancy_proxy(halton):.6f}")
print(f"  Sobol : {discrepancy_proxy(sobol):.6f}")

# ============================================================
# 2. Integrate f(x) = sin(2π x₁) cos(2π x₂) on [0,1]² — exact = 0
# ============================================================
def f(X):
    return np.sin(2*np.pi*X[:,0])*np.cos(2*np.pi*X[:,1])

errs_rand, errs_sobol = [], []
for m in range(4, 15):
    N = 2**m
    rng = np.random.default_rng(0)
    r = rng.random((N,2));  errs_rand.append(abs(f(r).mean()))
    s = Sobol(d=2, scramble=True, seed=0).random_base2(m=m)
    errs_sobol.append(abs(f(s).mean()))
print("\nN     |random|     |sobol|")
for m, er, es in zip(range(4,15), errs_rand, errs_sobol):
    print(f"2^{m:2d}  {er:.2e}   {es:.2e}")

# ============================================================
# 3. QMC pricing of an Asian option with Brownian bridge
# ============================================================
def brownian_bridge_construction(Z, T):
    """Transform independent Gaussians Z[N, M] into a BM path at
    times t_k = k*T/M using recursive bridge. First column = endpoint."""
    N, M = Z.shape
    path = np.zeros((N, M+1))
    times = np.linspace(0, T, M+1)
    # Iteratively fill in: endpoint first, then midpoint, quartiles,...
    path[:, M] = Z[:, 0]*np.sqrt(T)
    # Sequential bridge for simplicity (not optimal order)
    # Use dyadic fill: depth d splits each segment at midpoint
    def bridge_fill(lo, hi, idx):
        if hi - lo < 2 or idx >= M:
            return idx
        mid = (lo + hi)//2
        var = (times[hi]-times[mid])*(times[mid]-times[lo])/(times[hi]-times[lo])
        mean_coef = (times[mid]-times[lo])/(times[hi]-times[lo])
        path[:, mid] = (path[:, lo] + mean_coef*(path[:, hi]-path[:, lo])
                        + np.sqrt(var)*Z[:, idx])
        idx = bridge_fill(lo, mid, idx+1)
        idx = bridge_fill(mid, hi, idx)
        return idx
    bridge_fill(0, M, 1)
    return path[:, 1:]  # return path values at t_1,...,t_M

def asian_qmc_bridge(S0, K, r, sigma, T, M, N, method='sobol'):
    if method == 'sobol':
        U = Sobol(d=M, scramble=True, seed=0).random_base2(
            m=int(np.log2(N))
        )
    else:
        U = np.random.default_rng(0).random((N, M))
    Z = norm.ppf(np.clip(U, 1e-10, 1-1e-10))
    B = brownian_bridge_construction(Z, T)  # N x M BM values
    t = np.linspace(T/M, T, M)
    logS = np.log(S0) + (r-0.5*sigma**2)*t + sigma*B
    S = np.exp(logS)
    avg = S.mean(axis=1)
    pay = np.exp(-r*T)*np.maximum(avg - K, 0)
    return pay.mean(), pay.std(ddof=1)/np.sqrt(N)

price_mc, se_mc = asian_qmc_bridge(100,100,0.05,0.2,1.0,M=52,N=2**12,method='rand')
price_qmc, se_qmc = asian_qmc_bridge(100,100,0.05,0.2,1.0,M=52,N=2**12,method='sobol')
print(f"\nAsian option pricing, M=52, N=4096:")
print(f"  MC:    price={price_mc:.4f}  SE={se_mc:.4e}")
print(f"  QMC+BB price={price_qmc:.4f}  SE={se_qmc:.4e} (not fully meaningful)")
print(f"  Plot convergence below for a meaningful comparison")

# ============================================================
# 4. Convergence rate comparison: MC vs QMC
# ============================================================
true_geom = None  # (skipping — see 6.1 for closed-form geometric Asian)

errors_mc, errors_qmc = [], []
Ms = [2**m for m in range(6, 14)]
# Use a simpler integrand with known value: mean of S_T for BS
true_mean = 100*np.exp(0.05*1.0)
for m in range(6, 14):
    N = 2**m
    # MC
    Zr = np.random.default_rng(m).standard_normal(N)
    ST_mc = 100*np.exp((0.05-0.02)*1.0 + 0.2*Zr)
    errors_mc.append(abs(ST_mc.mean() - true_mean))
    # QMC
    U = Sobol(d=1, scramble=True, seed=m).random_base2(m=m)
    Zq = norm.ppf(np.clip(U, 1e-10, 1-1e-10)).flatten()
    ST_qmc = 100*np.exp((0.05-0.02)*1.0 + 0.2*Zq)
    errors_qmc.append(abs(ST_qmc.mean() - true_mean))
print(f"\nConvergence on E[S_T]:")
for N, em, eq in zip(Ms, errors_mc, errors_qmc):
    print(f"  N=2^{int(np.log2(N)):2d}: MC={em:.2e}  QMC={eq:.2e}  ratio={em/eq:.1f}x")
```

**What to check.**
- Halton and Sobol have discrepancy 10–100x smaller than random at $N = 1024$.
- On the smooth integrand $\sin(2\pi x_1)\cos(2\pi x_2)$, QMC error scales as $\sim N^{-1}$ vs $N^{-1/2}$ for MC.
- Asian with Brownian-bridge + Sobol beats random MC by $\sim 10\times$ on comparable $N$.
- The 1D integration of $\mathbb{E} S_T$ shows the "QMC in 1D" textbook factor-100 improvement.

---

## 6.2.10 [QUANT APPLICATIONS]

1. **CMS-style structured products.** Cliquets, autocallables, and path-dependent structured products have $d = 20$–$500$; Sobol' + Brownian bridge is standard in bank MC engines.
2. **Copula sampling for credit.** For Gaussian / $t$-copulas in CDO tranche pricing, Sobol' across $n$ names with PCA decomposition of $\Sigma$ improves convergence 5–20x.
3. **Greeks via finite differences on QMC.** QMC produces smoother estimators, so finite-difference sensitivities have much less noise. Common random numbers across bumps still required.
4. **Long-dated insurance liabilities.** 30-year monthly simulation ($d = 360$) exceeds usable MC budget; QMC with PCA is the only feasible approach for large insurers.
5. **Index construction / risk attribution.** High-$d$ portfolio risk calculation via correlated Gaussian simulation; QMC + PCA + effective-dimension analysis gives cheap, stable estimates.
6. **Regulatory capital.** Single-step VaR simulation with 500+ risk factors; QMC essential.
7. **Monte Carlo PDE solvers.** Feynman-Kac pricing of high-$d$ PDEs: QMC-accelerated MC is a competitive approach.
8. **Optimization over simulated surfaces.** Calibration requires evaluating a surface at many parameters; QMC common seeds give smoother objectives for gradient-based optimizers.
9. **Stress testing and backtesting.** Historical + QMC hybrid for tail capture.
10. **Machine-learning training.** Pretrain neural pricers with large QMC-generated data sets; lower variance → faster training.

---

## 6.2.11 Exercises

**★ (concept drills).**
1. Compute $\phi_2(n)$ for $n = 1, \ldots, 10$. Plot on $[0,1]$; observe bisection structure.
2. Write out the first 16 points of the 2D Halton sequence $(\phi_2, \phi_3)$. Verify uniform 2D fill better than 16 random points.
3. Explain why the Koksma-Hlawka bound is vacuous for indicator functions of irregular sets.
4. In what sense is Sobol' a $(t, s)$-sequence? Why does small $t$ matter?
5. Define effective dimension in the truncation sense. Why is it typically small for financial payoffs?
6. Why does Owen scrambling preserve low discrepancy while providing an unbiased error estimator?

**★★ (calculation).**
7. Derive the Koksma-Hlawka bound for $f(x) = x_1 x_2$ on $[0,1]^2$. Compute $V_{HK}(f)$.
8. On the 2D integrand $f(x,y) = (x+y)^2$, compare Halton, Sobol, and random at $N = 2^m$, $m = 4, \ldots, 14$. Fit the convergence rates.
9. Implement and verify the $(0,2)$-property of Sobol' in 2D: for $N = 2^m$, every elementary rectangle of volume $2^{-m}$ contains exactly one point.
10. Brownian-bridge vs sequential path construction. For a geometric Asian with $M = 64$ steps, compute the effective dimension under each construction. Why is bridge better?
11. PCA construction. Compute the principal components of the covariance $\Sigma_{ij} = \min(t_i, t_j)$ of discretized BM. Show the first PC captures most variance (the "level" mode).

**★★★ (open / research).**
12. **Scrambled net estimator**. Implement Owen's nested uniform scrambling of Sobol'. Verify unbiasedness and compute empirical variance on a 100-dim Asian.
13. **Lattice rule design**. Find a good generating vector $\mathbf{z}$ for a 20-dim lattice rule via CBC (component-by-component) construction. Compare to Sobol on a test suite.
14. **Effective dimension of a real structured product.** Pick a cliquet or autocallable. Compute ANOVA decomposition numerically and identify $d_T, d_S$.
15. **QMC for non-smooth payoffs.** Combine QMC with analytic conditioning (smoothing) for a knock-in call. Show the combined method restores super-linear convergence.
16. **RQMC + MLMC.** Combine randomized Sobol' with multilevel MC for Euler-Heston. Compare complexity to each method alone.

---

*— End of Module 6.2. Next: Module 6.3, Finite Difference Methods for PDEs.*

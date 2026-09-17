# Module 6.4 — Trees and Lattices

*Subject 6, Module 4. Binomial, trinomial, and Hull-White lattices — the pedagogical and practical scaffolding of derivative pricing.*

---

## Prerequisites

- **Module 5.2 (Black-Scholes)** — continuous-time benchmark.
- **Module 5.4 (Interest rate models)** — Hull-White tree for rates.
- **Module 4.2 (American options)** — early exercise via backward induction.
- **Module 2.4 (CLT)** — tree → diffusion convergence.

---

## 6.4.1 Why trees?

Trees are the most pedagogically-clear approach to derivative pricing: they make **risk-neutral valuation**, **replication**, and **early exercise** explicit through simple backward induction.

But they are also used in production for:
- American options in single-factor models (fast and dense time steps).
- Short-rate models (Hull-White, Black-Karasinski).
- Interest-rate trees calibrated to the yield curve (Ho-Lee, HW) for callable bonds and Bermudans.
- Convergence benchmarks for more complex methods.

---

## 6.4.2 Binomial Tree: Cox-Ross-Rubinstein (1979)

**Setup.** Divide $[0, T]$ into $N$ steps, each of length $\Delta t = T/N$. At each node, the price $S$ can go up to $uS$ or down to $dS$ with risk-neutral probabilities $p$ and $1 - p$.

**Matching moments of GBM.** Under CRR parameterization:
$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad d = 1/u = e^{-\sigma\sqrt{\Delta t}}, \qquad p = \frac{e^{r\Delta t} - d}{u - d}.
$$

**Verification.** Under these choices:
- $\mathbb{E}^p[\log S_{n+1} - \log S_n] = p\sigma\sqrt{\Delta t} + (1-p)(-\sigma\sqrt{\Delta t})$. For small $\Delta t$, $p \approx \tfrac{1}{2} + \tfrac{1}{2}(r-\sigma^2/2)\sqrt{\Delta t}/\sigma$, so the mean is $(r-\sigma^2/2)\Delta t + O(\Delta t^{3/2})$.
- $\text{Var} = \sigma^2\Delta t + O(\Delta t^2)$.

The drift is approximately $(r - \sigma^2/2)$ matching $d\log S$ under risk neutral. ✓

**Pricing a European option.** At maturity, $V_N^j = f(S_0 u^j d^{N-j})$ for $j = 0, \ldots, N$. Backward induction:
$$
V_n^j = e^{-r\Delta t}[p V_{n+1}^{j+1} + (1-p) V_{n+1}^j].
$$
Today's price: $V_0^0$.

**American option.** Add an early-exercise check:
$$
V_n^j = \max\{g(S_0 u^j d^{n-j}), \; e^{-r\Delta t}[p V_{n+1}^{j+1} + (1-p) V_{n+1}^j]\}.
$$

**Convergence.** $V_0^{(N)} \to V_0^{BS}$ as $N \to \infty$. The rate is $O(1/N)$ for smooth payoffs and non-monotone (with oscillations at strikes near grid points).

**Cost.** $O(N^2)$ per pricing. For $N = 1000$, about $5 \times 10^5$ node evaluations — fast in practice.

---

## 6.4.3 Alternative parameterizations

**Jarrow-Rudd (1983).** Match the first two log-moments exactly:
$$
u = e^{(r-\sigma^2/2)\Delta t + \sigma\sqrt{\Delta t}}, \quad d = e^{(r-\sigma^2/2)\Delta t - \sigma\sqrt{\Delta t}}, \quad p = 1/2.
$$
Equal probabilities → slight simplification. Tree is no longer recombining in $\log S$ — still recombining in node indices.

**Leisen-Reimer (1996).** Match the binomial $\mathcal{B}(N, p)$ to the normal CDF via **Peizer-Pratt inversion**. Result: $N$-step error is $O(1/N^2)$ instead of $O(1/N)$, and the oscillations around strikes are dramatically reduced. Preferred for pricing when closed-form benchmarks are needed.

**Trigeorgis.** Match first and second moments of the log-return exactly:
$$
\Delta x = \sqrt{\sigma^2\Delta t + (r-\sigma^2/2)^2 \Delta t^2}, \;\; p = \tfrac{1}{2} + \tfrac{(r-\sigma^2/2)\Delta t}{2\Delta x}.
$$

All binomial parameterizations are equivalent to first order in $\Delta t$; they differ in second-order refinement.

---

## 6.4.4 Trinomial trees and flexibility

**Trinomial.** Each node has three successors: up, middle, down. More degrees of freedom → match more moments → faster convergence, more flexibility with non-uniform time steps or barrier alignment.

**Boyle (1986) trinomial for BS.** With $u = e^{\sigma\sqrt{2\Delta t}}$, $m = 1$, $d = 1/u$:
$$
p_u = \left(\frac{e^{r\Delta t/2} - e^{-\sigma\sqrt{\Delta t/2}}}{e^{\sigma\sqrt{\Delta t/2}} - e^{-\sigma\sqrt{\Delta t/2}}}\right)^2, \quad p_d = \left(\frac{e^{\sigma\sqrt{\Delta t/2}} - e^{r\Delta t/2}}{e^{\sigma\sqrt{\Delta t/2}} - e^{-\sigma\sqrt{\Delta t/2}}}\right)^2, \quad p_m = 1 - p_u - p_d.
$$
Convergence $O(1/N^2)$, smoother than CRR.

**Adapted trinomial for barriers.** Choose $u = e^{\Delta x}$ such that the barrier $L$ lies exactly on a node: $\log(L/S_0) = -k \Delta x$ for integer $k$. Eliminates the dominant error source for barrier pricing — barrier *discretization error*.

---

## 6.4.5 Hull-White tree for short rates

Hull-White's (1993) trinomial lattice prices interest rate derivatives consistent with the observed yield curve.

**Setup.** For short-rate $r_t$ with dynamics $dr = \kappa(\theta(t) - r)dt + \sigma dB$, tree:
- Tree values of $r$ at time step $i$: $r_i^j = r_i^0 + j \Delta r$ for $j \in \{-J_i, \ldots, J_i\}$.
- Choose $\Delta r = \sigma\sqrt{3\Delta t}$ (Hull-White standard).

**Two-stage construction.**

**Stage 1**: build a symmetric "$x$"-tree with $x_{i+1}^j = x_i^j + \Delta x \cdot $(up/mid/down), matching the mean-reverting OU process $dx = -\kappa x dt + \sigma dB$. Probabilities $p_u, p_m, p_d$ chosen to match first and second moments:
- Standard branching at internal nodes.
- Upward-bent branching when $\kappa x > 0$ (to bring branches back toward zero).
- Downward-bent branching when $\kappa x < 0$.

**Stage 2**: shift the tree up by $\alpha_i$ to match zero-coupon bond prices $P(0, t_{i+1})$:
$$
r_i^j = x_i^j + \alpha_i.
$$

Arrow-Debreu state prices $Q_i^j$ are built forward. Matching $\sum_j Q_i^j e^{-r_i^j \Delta t} = P(0, t_{i+1})$ gives $\alpha_i$ by:
$$
\alpha_i = \frac{1}{\Delta t}\ln\frac{\sum_j Q_i^j e^{-x_i^j \Delta t}}{P(0, t_{i+1})}.
$$

**Pricing.** Backward induction using tree probabilities and discount factors $e^{-r\Delta t}$. Works for callable bonds, Bermudan swaptions, and exotic cancelable products.

**Extensions.** Two-factor HW (G2++) requires a 2D tree — exponential growth in nodes unless factor-by-factor decomposition is used.

---

## 6.4.6 Convergence, oscillations, and Richardson extrapolation

**Convergence rate.** CRR: $O(1/N)$ but with **oscillations** — the price jumps as strike crosses grid points. These artifacts make delta and gamma (computed via finite differences on the tree) extremely noisy.

**Smoothing tricks.**
1. **Control variate**. Price European + American on the tree. Take (tree American - tree European) + analytic European. The error cancels.
2. **Leisen-Reimer.** Peizer-Pratt inversion gives $O(1/N^2)$ smooth convergence.
3. **Richardson extrapolation.** If error is $\sim a/N + b/N^2 + \ldots$:
$$
V^{\text{extrap}} = 2V_{2N} - V_N = V_\infty + O(1/N^2).
$$
With Leisen-Reimer, $4V_{2N}/3 - V_N/3$ gives $O(1/N^3)$.
4. **Bermudan interpolation**. For sparse exercise dates, use binomial between exercise dates only; avoid the American-option oscillations.

---

## 6.4.7 Tree-PDE equivalence

Binomial trees are implicit schemes on a non-uniform grid. Specifically:
- CRR binomial $\sim$ explicit finite difference on log-price grid.
- Trinomial $\sim$ explicit finite difference with 3-point spatial stencil (similar to FD).
- $\Delta t = \Delta x^2 / \sigma^2$ gives stability (CFL condition holds automatically in CRR).

**Insight.** Every tree is a finite-difference scheme on a specific grid. Conversely, every explicit FD scheme is a tree. This unifies Modules 6.3 and 6.4.

**Trinomial = Central FD + explicit Euler.** Set $u = e^{\Delta x}$, $d = 1/u$, match moments:
$$
p_u = \tfrac{\Delta t}{2\Delta x^2}(\sigma^2 + (r-\sigma^2/2)\Delta x), \quad p_d = \tfrac{\Delta t}{2\Delta x^2}(\sigma^2 - (r-\sigma^2/2)\Delta x), \quad p_m = 1 - p_u - p_d.
$$
For stability $p_m \ge 0$, i.e. $\Delta t \le \Delta x^2 / \sigma^2$.

---

## 6.4.8 Python: tree pricers

```python
import numpy as np
from scipy.stats import norm

# ============================================================
# 1. CRR binomial tree for European and American call/put
# ============================================================
def binomial_option(S0, K, r, sigma, T, N=500,
                    option_type='call', american=False):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    disc = np.exp(-r*dt)

    # Terminal prices
    j = np.arange(N+1)
    S = S0 * u**(2*j - N)  # S_N^j = S0 * u^j * d^(N-j)

    # Terminal payoff
    if option_type == 'call':
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # Backward induction
    for n in range(N-1, -1, -1):
        j = np.arange(n+1)
        S = S0 * u**(2*j - n)
        V = disc * (p*V[1:] + (1-p)*V[:-1])
        if american:
            intrinsic = np.maximum(S - K, 0.0) if option_type=='call' else np.maximum(K-S, 0.0)
            V = np.maximum(V, intrinsic)
    return V[0]

# ============================================================
# 2. Leisen-Reimer tree (O(1/N²) convergence)
# ============================================================
def peizer_pratt(z, n):
    """Peizer-Pratt method 2 inversion of normal CDF."""
    if n % 2 == 0: n += 1
    return 0.5 + np.sign(z)*np.sqrt(0.25 - 0.25*np.exp(
        -(z/(n + 1/3 + 0.1/(n+1)))**2 * (n + 1/6)))

def leisen_reimer(S0, K, r, sigma, T, N=501, option_type='call', american=False):
    if N % 2 == 0: N += 1  # require odd
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    p  = peizer_pratt(d2, N)
    p_ = peizer_pratt(d1, N)
    u = np.exp(r*T/N)*p_/p
    d = (np.exp(r*T/N) - p*u)/(1-p)
    disc = np.exp(-r*T/N)

    j = np.arange(N+1)
    S = S0 * u**j * d**(N-j)
    V = np.maximum(S-K, 0.0) if option_type=='call' else np.maximum(K-S, 0.0)

    for n in range(N-1, -1, -1):
        j = np.arange(n+1)
        S = S0 * u**j * d**(n-j)
        V = disc * (p*V[1:] + (1-p)*V[:-1])
        if american:
            intrinsic = np.maximum(S-K, 0.0) if option_type=='call' else np.maximum(K-S, 0.0)
            V = np.maximum(V, intrinsic)
    return V[0]

# ============================================================
# Compare to BS closed form
# ============================================================
def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
true = bs_call(S0, K, r, sigma, T)
print(f"BS closed form: {true:.6f}")
for N in [50, 100, 200, 400, 800]:
    crr = binomial_option(S0, K, r, sigma, T, N=N)
    lr  = leisen_reimer(S0, K, r, sigma, T, N=N+1 if N%2==0 else N)
    print(f"N={N:4d}  CRR={crr:.6f}  err={abs(crr-true):.2e}   LR={lr:.6f}  err={abs(lr-true):.2e}")

# ============================================================
# 3. American put comparison
# ============================================================
print(f"\nAmerican put S0={S0}, K={K}, r={r}, σ={sigma}, T={T}:")
for N in [100, 400, 1600]:
    am = binomial_option(S0, K, r, sigma, T, N=N, option_type='put', american=True)
    print(f"  N={N}: American put = {am:.6f}")
eur_put = K*np.exp(-r*T)*norm.cdf(-norm.ppf(0.5)) - bs_call(S0, K, r, sigma, T)
eur_put = K*np.exp(-r*T)*(1-norm.cdf((np.log(S0/K)+(r-sigma**2/2)*T)/(sigma*np.sqrt(T)))) \
         - S0*(1-norm.cdf((np.log(S0/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))))
print(f"European put: {eur_put:.6f}")

# ============================================================
# 4. Hull-White tree for zero-coupon bonds
# ============================================================
def hull_white_tree(kappa, sigma_r, T, N, yield_curve_func):
    """Build Hull-White trinomial tree and price zero-coupon bonds."""
    dt = T/N
    dr = sigma_r*np.sqrt(3*dt)

    # Stage 1: x-tree with jmax = smallest integer ≥ 0.184/(κ dt)
    jmax = int(np.ceil(0.184/(kappa*dt)))
    # Branching probabilities
    def probs(j):
        if j == jmax:
            a = 7/6 + ((kappa*j*dt)**2 + 3*kappa*j*dt)/2
            b = -1/3 - ((kappa*j*dt)**2 + 2*kappa*j*dt)
            c = 1/6 + ((kappa*j*dt)**2 + kappa*j*dt)/2
            return a, b, c  # (j+2, j+1, j)
        elif j == -jmax:
            a = 1/6 + ((kappa*j*dt)**2 - kappa*j*dt)/2
            b = -1/3 - ((kappa*j*dt)**2 - 2*kappa*j*dt)
            c = 7/6 + ((kappa*j*dt)**2 - 3*kappa*j*dt)/2
            return a, b, c  # (j, j-1, j-2)
        else:
            p_up = 1/6 + ((kappa*j*dt)**2 - kappa*j*dt)/2
            p_mid = 2/3 - (kappa*j*dt)**2
            p_dn = 1/6 + ((kappa*j*dt)**2 + kappa*j*dt)/2
            return p_up, p_mid, p_dn  # (j+1, j, j-1)

    # Stage 2: fit α_i to zero curve via Arrow-Debreu forward propagation
    # ... (full implementation beyond this sketch; see Hull's text or QuantLib)
    return "HW tree constructed — see QuantLib for production implementation"

print("\nHull-White tree construction sketch — see QuantLib for production use")

# ============================================================
# 5. Trinomial tree for barrier option (barrier-aligned)
# ============================================================
def trinomial_down_out_call(S0, K, L, r, sigma, T, N=500):
    # Choose dx so that L is on a grid node
    dx_nat = sigma*np.sqrt(3*T/N)
    # Find number of down-steps to reach barrier
    k_bar = int(round(np.log(S0/L)/dx_nat))
    if k_bar <= 0: return 0.0
    dx = np.log(S0/L)/k_bar
    dt = T/N

    # Trinomial probabilities
    nu = r - 0.5*sigma**2
    p_u = 0.5*((sigma**2*dt + nu**2*dt**2)/dx**2 + nu*dt/dx)
    p_d = 0.5*((sigma**2*dt + nu**2*dt**2)/dx**2 - nu*dt/dx)
    p_m = 1 - p_u - p_d

    # Grid of log-prices
    js = np.arange(-N, N+1)
    logS = np.log(S0) + js*dx
    S = np.exp(logS)

    # Payoff at T, zero below barrier
    V = np.where(S <= L, 0.0, np.maximum(S - K, 0.0))

    disc = np.exp(-r*dt)
    for _ in range(N):
        V_new = disc * (p_u*V[2:] + p_m*V[1:-1] + p_d*V[:-2])
        js = js[1:-1]
        S = np.exp(np.log(S0) + js*dx)
        V = np.where(S <= L, 0.0, V_new)
    # Return value at S_0 (j=0)
    mid = len(V)//2
    return V[mid]

barrier_price = trinomial_down_out_call(100, 100, 80, 0.05, 0.2, 1.0, N=500)
print(f"\nTrinomial down-and-out call (S0=100, K=100, L=80): {barrier_price:.4f}")
```

**What to check.**
- CRR converges at $O(1/N)$ with oscillations around strikes.
- Leisen-Reimer converges at $O(1/N^2)$, no oscillations — dominant choice when accuracy matters.
- American put premium > European put, as expected. Premium larger for deeper ITM or shorter $T$ (counter-intuitive but known).
- Trinomial with barrier-aligned grid gives accurate barrier prices even at moderate $N$.

---

## 6.4.9 [QUANT APPLICATIONS]

1. **Early exercise valuation.** Production Bermudan pricing in single-factor HW uses trinomial lattices — fast, well-understood, accurate.
2. **Pedagogical benchmarks.** Every new pricing method is first validated against a dense CRR tree on BS European/American.
3. **Convertible bonds.** Binomial tree with multiple state dimensions (price, parity, conversion ratio) is the standard CB pricer.
4. **Mortgage-backed security prepayment modeling.** Short-rate tree with prepayment rules at each node.
5. **Employee stock options.** SAB valuation uses binomial trees with vesting and exercise modeling.
6. **Real options.** Strategic investment decisions modeled as American options on cash flows, priced via binomial.
7. **FX American options.** Cross-currency rates with foreign and domestic interest rates → modified CRR with dividend yield substitution.
8. **Commodity options.** Forward-curve-aware trees (via Clewlow-Strickland forward curve transformation).
9. **Credit-rating migration.** Tree of credit states with transition probabilities; pricing of rating-dependent payments.
10. **Teaching.** Derivatives courses universally begin with CRR — the clearest illustration of risk-neutral pricing.

---

## 6.4.10 Exercises

**★ (concept drills).**
1. Verify that CRR tree parameters $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$, $p = (e^{r\Delta t}-d)/(u-d)$ match the mean and variance of log-GBM to $O(\Delta t)$.
2. Show via CLT that the CRR binomial distribution of $\log S_N$ converges to Gaussian $\mathcal{N}((r-\sigma^2/2)T, \sigma^2 T)$.
3. Why does LR converge at $O(1/N^2)$ while CRR converges at $O(1/N)$?
4. What's the advantage of trinomial over binomial for barrier options?
5. Explain the two-stage Hull-White construction. Why is the drift adjusted in stage 2?
6. Show that binomial tree pricing with replacement of max($V$, intrinsic) at each node solves the discrete-time LCP for American options.

**★★ (calculation).**
7. Implement CRR and LR; produce a convergence plot for a European call. Fit power-law decay and confirm rates.
8. Add Richardson extrapolation $(2V_{2N} - V_N)$ to CRR. Does the rate improve?
9. Tree delta/gamma. Use finite differences on the tree itself (nodes at $t=0$ and $t=\Delta t$) to compute delta and gamma. Compare to BS closed form.
10. Price an American put via CRR and LR at $S_0 = K$, $r = 5\%$, $\sigma = 30\%$, $T = 1$. Identify the optimal exercise boundary.
11. Construct a Hull-White tree for $\kappa = 0.05, \sigma = 0.01$, $T = 10$, $N = 200$. Verify that zero-coupon bond prices match a flat yield curve of 4%.

**★★★ (open / research).**
12. **Stochastic-vol tree.** Implement a multinomial tree for Hull-White SV (where vol is also a state variable). Compare to Heston ADI.
13. **Linear programming for American options.** Rewrite the LCP as an LP and solve via simplex. Does it handle non-convex payoffs more robustly than PSOR?
14. **Bermudan via trees.** Price a 10-year Bermudan swaption exercisable annually in Hull-White. Compare to Monte Carlo with Longstaff-Schwartz.
15. **Tree discrepancy analysis.** For a digital option, study the convergence of CRR as $N \to \infty$. Show the error oscillates with period depending on $\log(K/S_0)/\log u$.
16. **GPU-accelerated trees.** Implement a CRR tree in CUDA / PyTorch. At what $N$ does GPU overhead pay off? Compare to a threaded CPU implementation.

---

*— End of Module 6.4. Next: Module 6.5, Fourier and Transform Methods.*

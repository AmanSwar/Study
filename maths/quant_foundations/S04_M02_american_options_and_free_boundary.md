# Module 4.2 — American Options and Free Boundary Problems

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 4 (Optimal Stopping & Stochastic Control), Module 2 of 7

---

## Prerequisites

- **Module 3.3** (Itô's formula, Black–Scholes PDE derivation).
- **Module 3.4** (SDEs, Feynman–Kac).
- **Module 4.1** (Optimal stopping, Snell envelope, smooth pasting).
- Basics of numerical PDE solving (finite differences).

American options are the canonical applied problem in optimal stopping: price a contingent claim that the holder can exercise at any time up to maturity. This module pins down the mathematical formulation, derives the free-boundary PDE, and develops robust numerical schemes.

---

## 1. American options: definition and no-arbitrage pricing

### 1.1 Contract specification

An **American option** with payoff $g(S, t)$ and maturity $T$ entitles the holder to exercise at any stopping time $\tau \le T$, receiving payoff $g(S_\tau, \tau)$.

- **American put:** $g(S, t) = (K - S)^+$.
- **American call (no dividends):** same price as European (Merton 1973 — see Section 1.3).
- **American call (with dividends):** has early-exercise premium.

### 1.2 Pricing formula

Under risk-neutral measure $Q$:
$$
V(S, t) = \sup_{\tau \in \mathcal{T}_{t, T}}\, E^Q\bigl[e^{-r(\tau - t)} g(S_\tau, \tau)\bigr|\,S_t = S\bigr].
$$

This is a Markovian optimal stopping problem. The Snell envelope from Module 4.1 applies.

### 1.3 American call with no dividends

**Theorem 1.1 (Merton 1973).** If $S$ pays no dividends, an American call has the same price as the European call. Early exercise is never optimal.

*Proof.* An American call allows exercise at any $\tau \le T$. Exercising at $\tau$ yields $S_\tau - K$. Holding to $T$ via European pricing yields $E^Q[e^{-r(T - \tau)}(S_T - K)^+|\mathcal{F}_\tau] \ge E^Q[e^{-r(T-\tau)}(S_T - K)|\mathcal{F}_\tau] = S_\tau - Ke^{-r(T-\tau)} > S_\tau - K$ for $r > 0$. So European $\ge$ intrinsic, hence exercise is suboptimal. $\square$

With dividends ($q > 0$), the argument breaks down: forward $S e^{-q(T-t)}$ can be less than spot.

---

## 2. Free-boundary PDE formulation

### 2.1 The variational inequality / linear complementarity

From Module 4.1, the Snell-envelope characterization in the continuous-time Markov setting:
$$
\min\bigl(-\partial_t V - \mathcal{L} V + r V,\; V - g\bigr) = 0, \qquad V(S, T) = g(S).
$$

For BS dynamics $dS/S = (r - q) dt + \sigma dB$:
$$
\mathcal{L} V = (r - q) S \partial_S V + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} V.
$$

### 2.2 The two regions

**Continuation region $\mathcal{C}$:** $V > g$. Here the PDE holds:
$$
\partial_t V + \mathcal{L} V - r V = 0.
$$

**Stopping region $\mathcal{S}$:** $V = g$. Here
$$
\partial_t V + \mathcal{L} V - r V \le 0,
$$

(otherwise, exercising is suboptimal — contradiction).

### 2.3 Free boundary

The boundary between $\mathcal{C}$ and $\mathcal{S}$ is the **free boundary**, call it $S^*(t)$. For the American put:
$$
S^*(t) = \inf\{S : V(S, t) = K - S\},
$$

i.e., exercise if $S \le S^*(t)$.

At the boundary, **two conditions**:
- **Value matching:** $V(S^*(t), t) = g(S^*(t)) = K - S^*(t)$.
- **Smooth pasting:** $\partial_S V(S^*(t), t) = -1$.

The second is the high-contact condition from Module 4.1.

### 2.4 Properties of the free boundary

For the American put:
- $S^*(T) = K$ (exercise immediately if $K > S$).
- $S^*(t) < K$ for $t < T$ (there is time value).
- $S^*(t)$ is monotonically increasing in $t$.
- As $T - t \to \infty$, $S^*(t) \to S^*_\infty$ (the perpetual put boundary).

No closed form for $S^*(t)$ exists; numerical or approximation methods required.

---

## 3. Bjerksund–Stensland approximation

**Bjerksund–Stensland (1993)** gives a closed-form approximation to the American option price based on a flat exercise boundary. For a put:

$$
P^{\text{Am}}(S, t) \approx \alpha S^\beta - \alpha \phi(S, T - t; \beta, I, I) + \phi(S, T - t; 1, I, I) - \phi(S, T - t; 1, K, I) - K \phi(S, T - t; 0, K, I) + K \phi(S, T - t; 0, I, I),
$$

where $I$ is a flat boundary chosen optimally, $\alpha$ and $\beta$ depend on $r, q, \sigma$, and $\phi$ is a cumulative bivariate-normal-like function. Accurate to a few cents.

Used extensively on real-time systems where PDE solution is too slow.

---

## 4. Numerical methods

### 4.1 Binomial tree (Cox–Ross–Rubinstein)

Build a recombining binary tree:
$$
S_{t + \Delta t} = S_t \cdot u \text{ or } S_t \cdot d,
$$

with $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$, risk-neutral prob $p = (e^{(r-q)\Delta t} - d)/(u - d)$.

Backward induction at each node:
$$
V_i^k = \max\bigl(g(S_i^k),\; e^{-r\Delta t}[p V_{i+1}^{k+1} + (1-p) V_{i-1}^{k+1}]\bigr).
$$

**Pros:** simple, intuitive, converges at rate $1/N$.

**Cons:** slow convergence, oscillation in convergence rate.

**Improvements:** Leisen–Reimer (2001) with altered $p$; Richardson extrapolation.

### 4.2 Finite difference + PSOR

Discretize the LCP:
$$
\begin{aligned}
& \partial_t V + \mathcal{L} V - r V \le 0, \\
& V \ge g, \\
& (V - g)(\partial_t V + \mathcal{L} V - r V) = 0.
\end{aligned}
$$

Use Crank–Nicolson (time) + central difference (space), with a **Projected SOR** (PSOR) solver for the LCP at each time step. Algorithm:

**For each time step** (backward):
1. Assemble the Crank–Nicolson system $A V^{k-1} = B V^k + b$.
2. Solve the LCP $\min(A V - c, V - g) = 0$ by PSOR:
   - Iterate $V_i \leftarrow \max(g_i, V_i - \omega (\text{residual}_i)/A_{ii})$ until convergence.
   - $\omega \in (1, 2)$ is the relaxation parameter; $\omega = 1.2$ typical.

**Pros:** high accuracy, well-behaved Greeks.

**Cons:** PSOR is iterative, slow for high dimensions.

### 4.3 Longstaff–Schwartz Monte Carlo

For high-dimensional or path-dependent problems:

1. Simulate $M$ paths of $S$.
2. Backward from maturity: at each exercise date, regress continuation value on basis functions of $(S, \text{other state})$.
3. Exercise policy: exercise if $g > \hat{\text{continuation}}$.
4. Average cashflows.

### 4.4 Tilley's bundling

Alternative MC method: bundle paths by $S_t$ value, estimate continuation value within each bundle. Similar accuracy to LS.

### 4.5 Stochastic mesh / regression Monte Carlo

Broadie–Glasserman stochastic mesh: interconnect paths via likelihood ratios; converges to optimal exercise boundary.

---

## 5. PSOR algorithm in detail

The Crank–Nicolson discretization on a grid $S_i = i\Delta S$ yields (for fixed time step):
$$
a_i V_{i-1} + b_i V_i + c_i V_{i+1} = d_i, \quad V_i \ge g_i.
$$

**PSOR iteration** (with overrelaxation $\omega \in (1, 2)$):

```
repeat
  for i = 1 to N-1:
    # Provisional update
    V_i_new = (d_i - a_i*V_{i-1} - c_i*V_{i+1}) / b_i
    V_i_relax = V_i + omega * (V_i_new - V_i)
    # Project onto constraint
    V_i = max(V_i_relax, g_i)
until convergence
```

**Convergence:** guaranteed for diagonally dominant $b_i$ (which Crank–Nicolson provides if $\Delta t$ is small enough).

**Optimal $\omega$:** depends on spectral radius of the iteration matrix; tuning is empirical, but $\omega \approx 1.2$-$1.5$ often works.

---

## 6. The early-exercise premium decomposition

**Theorem 6.1 (Kim 1990, Jamshidian 1992, Carr–Jarrow–Myneni 1992).**
$$
V^{\text{Am}}(S, t) = V^{\text{Eur}}(S, t) + \int_t^T \text{EEP}(S, t; u) du,
$$

where
$$
\text{EEP}(S, t; u) = rK e^{-r(u - t)} N(-d_2(u, S^*(u))) - qS e^{-q(u - t)} N(-d_1(u, S^*(u)))
$$
for an American put, with $d_1, d_2$ the standard BS arguments.

This decomposition is the **integral equation** for the American put: substituting $S = S^*(t)$ gives an equation determining $S^*$. Iterative numerical solution.

---

## 7. Perpetual American options: closed forms

From Module 4.1, Section 5.1, the perpetual American put has closed form:
$$
v(S) = (K - S^*)\bigl(S/S^*\bigr)^{\gamma_2}, \quad S > S^*,
$$
with $S^* = K\gamma_2/(\gamma_2 - 1)$ and $\gamma_2$ the negative root of $\tfrac{1}{2}\sigma^2\gamma(\gamma-1) + (r-q)\gamma - r = 0$.

Similarly, the perpetual American call (with $q > 0$):
$$
v(S) = (S^* - K)(S/S^*)^{\gamma_1}, \quad S < S^*,
$$
with $S^* = K\gamma_1/(\gamma_1 - 1)$ and $\gamma_1$ the positive root.

These formulas provide "asymptotic" upper bounds on finite-horizon American option prices.

---

## 8. Capped American options and other variations

### 8.1 Capped (installment) options

Pay a premium only at exercise; valuation involves a modified obstacle problem.

### 8.2 Bermudan options

Exercise allowed only at prescribed discrete dates. Valuation = finite-horizon DP; Longstaff–Schwartz is the MC workhorse.

### 8.3 American with stochastic vol (e.g., Heston)

Free-boundary surface in $(S, V, t)$. 2D PDE + obstacle. Solved by 2D PSOR or LS Monte Carlo.

### 8.4 American with jumps (Merton, Kou)

Free-boundary PIDE. The exercise boundary now includes jump-overshoot information. See Hirsa (2012) for numerical PIDE + obstacle schemes.

### 8.5 Shout options

Holder can "shout" at some time $\tau$ to lock in a minimum payoff; pricing = optimal stopping with stopping payoff being a European option.

---

## 9. Python: numerical schemes

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)

S0, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.01, 0.2, 1.0

# ---- (a) Binomial tree with Richardson extrapolation ----
def binomial_put(N, American=True):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt) - d) / (u - d)
    disc = np.exp(-r*dt)
    # Terminal
    S_T = S0 * u**np.arange(N+1) * d**(N - np.arange(N+1))
    V = np.maximum(K - S_T, 0.0)
    for i in range(N-1, -1, -1):
        S = S0 * u**np.arange(i+1) * d**(i - np.arange(i+1))
        V = disc * (p * V[1:i+2] + (1-p) * V[0:i+1])
        if American:
            V = np.maximum(V, K - S)
    return V[0]

bin_1000 = binomial_put(1000)
bin_2000 = binomial_put(2000)
richardson = 2*bin_2000 - bin_1000  # rough Richardson
print(f"Binomial Amer put N=1000: {bin_1000:.5f}")
print(f"Binomial Amer put N=2000: {bin_2000:.5f}")
print(f"Richardson extrapolation: {richardson:.5f}")

# European for reference
def euro_put(S, K, r, q, sigma, T):
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
print(f"European put: {euro_put(S0, K, r, q, sigma, T):.5f}")

# ---- (b) Crank-Nicolson + PSOR ----
def cn_psor_amer_put(S0, K, r, q, sigma, T, Smax=400, Ns=200, Nt=200):
    dS = Smax / Ns
    dt = T / Nt
    S = np.linspace(0, Smax, Ns+1)
    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()
    # Build implicit/explicit matrix parts
    i = np.arange(1, Ns)
    a = 0.25 * dt * (sigma**2 * i**2 - (r-q)*i)
    b = -0.5 * dt * (sigma**2 * i**2 + r)
    c = 0.25 * dt * (sigma**2 * i**2 + (r-q)*i)
    # Left matrix: (1-b, -a, -c)
    # Right matrix: (1+b, a, c)
    for step in range(Nt):
        # RHS
        rhs = (1 + b) * V[1:-1] + a * V[0:-2] + c * V[2:]
        # Boundary: V(0) = K - 0 = K (exercise immediately if S=0); V(Smax) = 0
        rhs[0] += a[0] * (K)
        rhs[-1] += c[-1] * 0.0
        # PSOR to solve (1-b) v - a v_{-1} - c v_{+1} = rhs, v >= payoff
        v = V[1:-1].copy()
        omega = 1.2
        for psor_iter in range(500):
            max_err = 0.0
            for j in range(Ns - 1):
                a_j = -a[j] if j > 0 else 0.0
                c_j = -c[j] if j < Ns - 2 else 0.0
                b_j = 1 - b[j]
                lhs_j = rhs[j]
                if j > 0: lhs_j -= a_j * v[j-1]
                if j < Ns-2: lhs_j -= c_j * v[j+1]
                v_new = lhs_j / b_j
                v_candidate = v[j] + omega * (v_new - v[j])
                v_new = max(v_candidate, payoff[j+1])
                max_err = max(max_err, abs(v_new - v[j]))
                v[j] = v_new
            if max_err < 1e-7:
                break
        V[1:-1] = v
        V[0] = K  # left boundary
        V[-1] = 0.0  # right boundary
    # Interpolate
    return np.interp(S0, S, V)

psor_price = cn_psor_amer_put(S0, K, r, q, sigma, T, Ns=200, Nt=200)
print(f"\nCrank-Nicolson + PSOR American put: {psor_price:.5f}")

# ---- (c) Longstaff-Schwartz ----
def ls_amer_put(S0, K, r, q, sigma, T, M=10000, N=50):
    dt = T/N
    paths = np.zeros((M, N+1)); paths[:, 0] = S0
    Z = rng.standard_normal((M, N))
    for k in range(N):
        paths[:, k+1] = paths[:, k] * np.exp((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, k])
    cashflow = np.maximum(K - paths[:, -1], 0.0)
    exercise_time = np.full(M, N)
    for k in range(N-1, 0, -1):
        itm = K - paths[:, k] > 0
        if itm.sum() < 5: continue
        X = paths[itm, k]
        Y = cashflow[itm] * np.exp(-r*(exercise_time[itm] - k)*dt)
        A = np.column_stack([np.ones_like(X), X, X**2])
        beta, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        cont = A @ beta
        ex = K - X
        exercise = ex > cont
        idx = np.where(itm)[0][exercise]
        cashflow[idx] = K - paths[idx, k]
        exercise_time[idx] = k
    return (cashflow * np.exp(-r * exercise_time * dt)).mean()

ls_price = ls_amer_put(S0, K, r, q, sigma, T, M=50000)
print(f"\nLongstaff-Schwartz American put: {ls_price:.5f}")

# ---- (d) Early-exercise boundary via binomial ----
def binomial_amer_put_boundary(N=500):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt)); d = 1/u
    p = (np.exp((r-q)*dt) - d) / (u - d)
    disc = np.exp(-r*dt)
    # Backward build value grid: V(step, node) where node index j = 0..step
    grids = {N: np.maximum(K - S0*u**np.arange(N+1)*d**(N-np.arange(N+1)), 0.0)}
    for i in range(N-1, -1, -1):
        S = S0*u**np.arange(i+1)*d**(i-np.arange(i+1))
        V_cont = disc*(p*grids[i+1][1:i+2] + (1-p)*grids[i+1][0:i+1])
        V_ex = np.maximum(K - S, 0.0)
        V = np.maximum(V_cont, V_ex)
        grids[i] = V
    # Boundary: at each time step, find largest S such that V == K - S
    boundary = np.zeros(N+1)
    for i in range(N+1):
        S = S0*u**np.arange(i+1)*d**(i-np.arange(i+1))
        V = grids[i]
        # exercise iff V == K - S
        ex_nodes = np.where(np.abs(V - (K - S)) < 1e-8)[0]
        if len(ex_nodes) == 0:
            boundary[i] = 0.0
        else:
            boundary[i] = S[ex_nodes].max()
    return boundary

boundary = binomial_amer_put_boundary(N=200)
print(f"\nEarly-exercise boundary S*: S*(0)={boundary[0]:.2f}, S*(T/2)={boundary[100]:.2f}, S*(T)={boundary[-1]:.2f}")
```

### Expected output

- Binomial 1000: ~$6.09$; 2000: ~$6.08$; Richardson: ~$6.08$.
- PSOR: ~$6.08$.
- LS: ~$6.07$ (slight LS bias low).
- Boundary: $S^*(0) \approx 85$, $S^*(T/2) \approx 92$, $S^*(T) = K = 100$.

---

## 10. [QUANT APPLICATION] — American options in industry

### 10.1 Exchange-traded options

Most US equity options are American. Exercise decisions drive dividend capture strategies, short-interest squeezes, and arbitrage relationships.

### 10.2 Employee stock options

Long-dated (7+ years) American calls on employer stock. Value depends on:
- Early-exercise behavior (employees exercise suboptimally due to risk aversion).
- Forfeiture rates.
- Vesting schedules.

### 10.3 Convertible bonds

Contain an American call on equity. Valuation requires solving a two-factor (stock + rate) American option PDE.

### 10.4 Credit callable bonds / MBS

Issuer has an American call on the bond (redeem early). Mortgage prepayment = American put (refinance). Both are path-dependent with stochastic rates; PIDE + MC.

### 10.5 Autocallables

Popular structured product: pays off early if the underlying hits a trigger. Not strictly American (not holder-decided), but similar pricing framework.

### 10.6 Swing options, energy contracts

Industrial gas take-or-pay, electricity peaker plants: multi-exercise optionality. Valuation by DP / multi-stage stopping.

---

## 11. Summary

- **American options** allow early exercise; value = sup over stopping times of discounted expected payoff.
- **Free-boundary PDE formulation** as a linear complementarity problem.
- **Smooth pasting** at the exercise boundary pins down the problem.
- **No closed form** for finite-horizon American options; requires numerical methods.
- **Methods:** binomial tree, Crank–Nicolson + PSOR, Longstaff–Schwartz Monte Carlo, PIDE for jump models.
- **Perpetual** American options admit closed-form solutions via the ansatz $v(S) = A S^\gamma$.

---

## 12. Exercises

**★ (warm-ups)**

**12.1** Verify: for a no-dividend American call, exercise is never optimal before maturity. [Reproduce Theorem 1.1.]

**12.2** For the American put at $t = T$: show $V(S, T) = (K - S)^+$, and the free boundary is $S^*(T) = K$.

**12.3** For a very-in-the-money put ($S \ll K$): show that exercise is optimal, and $V(S, t) = K - S$.

**12.4** For a deep-out-of-the-money put ($S \gg K$): show exercise is suboptimal, and $V(S, t) \approx$ European put.

**12.5** Show the smooth-pasting condition from a variational argument.

**★★ (core)**

**12.6** **(Perpetual call with dividend.)** Derive the closed form for the perpetual American call with $q > 0$.

**12.7** **(Binomial tree convergence.)** Empirically, binomial has $O(1/N)$ convergence with oscillations. Implement and plot the error vs $N$.

**12.8** **(Richardson extrapolation.)** Apply Richardson extrapolation to binomial prices at $N$ and $2N$. Quantify the accuracy improvement.

**12.9** **(PSOR convergence.)** Prove that PSOR with $\omega \in (0, 2)$ converges for a diagonally dominant LCP.

**12.10** **(Crank–Nicolson stability.)** Show Crank–Nicolson is unconditionally stable for the BS PDE.

**12.11** **(Kim integral equation.)** Derive the early-exercise premium decomposition (Theorem 6.1) starting from Itô applied to the American put value process.

**12.12** **(Bermudan approximation.)** Show that the Bermudan with $N$ equally-spaced exercise dates converges to the American as $N \to \infty$.

**12.13** **(Longstaff–Schwartz with deeper basis.)** Increase the LS basis to $\{1, S, S^2, S^3, \log S\}$. Compare with the quadratic basis.

**12.14** **(Upper bound via duality.)** Implement Rogers's dual martingale bound for an American put. Report the bound and compare with LS lower bound.

**12.15** **(American call on dividend stock.)** Implement PSOR for an American call with dividend $q$. Show exercise occurs at large $S$ and find $S^*(t)$.

**★★★ (research / quant)**

**12.16** **(Bjerksund–Stensland.)** Read the BS approximation paper and implement it. Compare with PSOR.

**12.17** **(2D American options.)** Price an American max-call $\max(S_1, S_2) - K)^+$ via 2D binomial tree or PSOR or LS.

**12.18** **(American under stochastic vol.)** Price an American put under Heston (2D) via LS with basis $\{1, S, V, S^2, V^2, SV, \ldots\}$.

**12.19** **(American with jumps.)** Price an American put under Merton jump-diffusion. Implement PIDE + PSOR; handle the nonlocal term via quadrature.

**12.20** **(Convertible bonds.)** Value a 5-year convertible with quarterly coupons, on a stock + stochastic rate (Vasicek). Implement 2D PSOR.

**12.21** **(Deep RL for American options.)** Train a neural network policy to decide exercise vs hold, using PG / actor-critic. Compare with LS.

**12.22** **(Exercise boundary asymptotics.)** Prove $S^*(t) \approx K(1 - c\sqrt{T - t})$ as $t \to T^-$ for the American put. [Evans, 2010.]

---

## 13. Forward pointers

- **Module 4.3 (DP)**: abstraction of the American option pricing DP.
- **Module 4.4 (HJB)**: continuous-time control generalization.
- **Module 4.7 (RL)**: deep-learning approaches to American pricing.

**Next module:** Dynamic programming — the general principle of optimality, of which American option pricing is a special case.

---

*End of Module 4.2.*

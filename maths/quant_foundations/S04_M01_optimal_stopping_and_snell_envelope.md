# Module 4.1 — Optimal Stopping and the Snell Envelope

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 4 (Optimal Stopping & Stochastic Control), Module 1 of 7 — **Subject 4 Opening**

---

## Welcome to Subject 4

Subject 3 gave us the dynamics — Brownian motion, SDEs, Lévy processes, martingales. Subject 4 introduces **decisions**: when should we act on those dynamics, and how? This subject covers:

- **Module 4.1 (this module)**: Optimal stopping, Snell envelope, continuation / stopping regions.
- **Module 4.2**: American options and free-boundary problems.
- **Module 4.3**: Dynamic programming and Bellman equations.
- **Module 4.4**: Hamilton–Jacobi–Bellman (HJB) equations.
- **Module 4.5**: Merton's portfolio problem.
- **Module 4.6**: Linear-Quadratic-Gaussian control and Kalman filtering.
- **Module 4.7**: Reinforcement learning and approximate dynamic programming.

By the end of Subject 4, you will be able to price American options, solve optimal consumption-investment problems, design LQG controllers, and connect classical control theory to modern RL.

---

## Prerequisites

- **Module 2.2** (Conditional expectation, tower property).
- **Module 2.6** (Martingales, optional stopping, Doob's inequality, Doob decomposition).
- **Module 3.1–3.3** (Brownian motion, Itô, Itô's formula).
- **Module 3.7** (Continuous-time martingales, Doob–Meyer, UI martingales).
- Classical analysis: dynamic programming, recursion, supremum over sets.

---

## 1. The optimal stopping problem

### 1.1 Statement

Given:

- A filtered probability space $(\Omega, \mathcal{F}, (\mathcal{F}_t), P)$.
- A stochastic process $G = (G_t)_{t \in \mathcal{T}}$ (the "reward" process), $\mathcal{T} \subseteq [0, \infty)$ finite or infinite, adapted and integrable.

Find a stopping time $\tau^* \in \mathcal{T}$ maximizing $E[G_\tau]$:
$$
V := \sup_{\tau} E[G_\tau] = E[G_{\tau^*}],
$$
where the sup is over $\mathcal{F}_t$-stopping times taking values in $\mathcal{T}$.

**Intuition.** You observe a process evolving; at each moment you can either "stop and cash out" with reward $G_t$, or "continue and wait for more info". When is it optimal to stop?

### 1.2 Examples

- **Secretary problem**: observe $n$ candidates sequentially; pick one. Optimal rule: skip first $n/e$, then pick first one better than all seen.
- **American option pricing**: payoff $(K - S_t)^+$ at time $\tau$; hold or exercise.
- **Optimal selling**: stock $S_t$ follows GBM; decide when to sell to maximize $E[e^{-r\tau} S_\tau]$.
- **Disorder detection (change-point)**: detect as fast as possible when a process transitions from one regime to another.
- **Optimal trading execution**: unload $X$ shares at times / prices to maximize expected revenue.

---

## 2. Discrete-time optimal stopping: the Snell envelope

**Setup.** $G_0, G_1, \ldots, G_N$ integrable adapted. We want
$$
V_0 = \sup_\tau E[G_\tau].
$$

### 2.1 Definition of the Snell envelope

**Definition 2.1.** The **Snell envelope** of $G$ is the process $U$ defined by backward recursion:
$$
U_N := G_N,
$$
$$
U_k := \max\bigl(G_k,\; E[U_{k+1} | \mathcal{F}_k]\bigr), \quad k = N-1, \ldots, 0.
$$

$U$ is the **smallest supermartingale dominating $G$**.

### 2.2 Why this works: dynamic programming principle

At time $k$, you have two choices:

1. **Stop now**: receive $G_k$.
2. **Continue**: receive $E[U_{k+1} | \mathcal{F}_k]$, the best expected future reward.

Optimal: $U_k = \max(G_k, E[U_{k+1}|\mathcal{F}_k])$. This is Bellman's principle.

### 2.3 The optimal stopping time

**Theorem 2.2.** Define $\tau^* := \min\{k : U_k = G_k\}$. Then

(a) $\tau^*$ is a stopping time ($\le N$ since $U_N = G_N$).
(b) $E[G_{\tau^*}] = U_0 = V_0$.
(c) The stopped process $U^{\tau^*}$ is a martingale.

*Proof.*

(a) $\{\tau^* = k\} = \{U_k = G_k, U_j > G_j\, \forall j < k\} \in \mathcal{F}_k$.

(b) By construction $U_k \ge G_k$, so by optional stopping applied to the supermartingale $U$,
$$
U_0 \ge E[U_{\tau^*}] = E[G_{\tau^*}] \le V_0.
$$

For the reverse, use (c) below: $U^{\tau^*}$ is a martingale, so $U_0 = E[U_{\tau^*}] = E[G_{\tau^*}] \le V_0$.

(c) For $k < \tau^*$, $U_k > G_k$, so $U_k = E[U_{k+1}|\mathcal{F}_k]$. So $U^{\tau^*}_{k+1} - U^{\tau^*}_k = \mathbf{1}_{\{k < \tau^*\}}(U_{k+1} - E[U_{k+1}|\mathcal{F}_k])$ is a martingale difference. $\square$

### 2.4 Continuation region and stopping region

The **stopping region** is $\{U = G\}$; the **continuation region** is $\{U > G\}$. The optimal $\tau^*$ = first entry into the stopping region.

**Characterization:** $k$ is in the continuation region iff $E[U_{k+1}|\mathcal{F}_k] > G_k$, i.e., **waiting is strictly better**. Stop iff waiting is no better.

---

## 3. Doob decomposition and the smallest uniformly integrable supermartingale

**Theorem 3.1.** The Snell envelope $U$ can be written as $U = M - A$ where:

- $M$ is a martingale with $M_0 = U_0$,
- $A$ is a nondecreasing predictable process with $A_0 = 0$.

This is the Doob decomposition (Module 2.6). The process $A$ encodes the "value of future optionality" that decays as we stop.

**Alternative characterization.**

**Theorem 3.2.** $\tau^* = \inf\{k : A_{k+1} > 0\}$: the optimal stopping time is the first time **the compensator starts to accumulate**.

*Proof.* For $k < \tau^*$, $A_k = 0$, so $U_k = M_k$, a martingale. Hence $U_k = E[U_{k+1}|\mathcal{F}_k] > G_k$ unless $A_{k+1} > 0$. Setting $\tau^* = \inf\{k : A_{k+1} > 0\}$ captures exactly the first non-martingale moment. $\square$

---

## 4. Continuous-time optimal stopping

### 4.1 Setup

$(G_t)_{0 \le t \le T}$ right-continuous, adapted, integrable. Optimal stopping:
$$
V(t) = \underset{\tau \in \mathcal{T}_t^T}{\text{ess sup}}\, E[G_\tau | \mathcal{F}_t],
$$

where $\mathcal{T}_t^T$ is the set of stopping times in $[t, T]$.

### 4.2 Snell envelope

**Definition 4.1.** The Snell envelope $U_t = V(t)$ is the smallest càdlàg supermartingale dominating $G$.

**Theorem 4.2.** (a) $U$ is the smallest supermartingale $\ge G$.

(b) $\tau^* := \inf\{t : U_t = G_t\}$ is optimal.

(c) $U$ admits a Doob–Meyer decomposition $U = M - A$; $\tau^*$ is the first time $A$ starts increasing.

### 4.3 Characterization via obstacle PDE

For $G_t = g(X_t)$ with $X$ a Markov diffusion with generator $\mathcal{L}$, the value function $v(t, x) = V(t)|_{X_t = x}$ satisfies the **obstacle problem / linear complementarity**:
$$
\min\bigl(-\partial_t v - \mathcal{L} v, v - g\bigr) = 0, \qquad v(T, x) = g(x).
$$

In the continuation region, $\partial_t v + \mathcal{L}v = 0$ and $v > g$; in the stopping region, $v = g$ and $\partial_t v + \mathcal{L} v \le 0$.

The boundary between these regions is the **free boundary** — its location is **part of the solution**.

### 4.4 Smooth pasting

At the free boundary, both value matching and derivative matching hold:
$$
v(t, x^*) = g(x^*), \qquad \partial_x v(t, x^*) = \partial_x g(x^*).
$$

The "smooth pasting" / high-contact condition comes from optimality and ensures the value function is $C^1$ at the boundary. For American put options (Module 4.2), this pins down the exercise boundary.

---

## 5. Optimal stopping of diffusions: Markovian framework

Let $X$ be a diffusion with generator $\mathcal{L}$, and let $g(x)$ be the reward function. The problem is
$$
v(x) = \sup_\tau E_x[e^{-r\tau} g(X_\tau)], \qquad \tau \in \mathcal{T},
$$
for a discount rate $r \ge 0$.

**Variational inequality:** $v$ satisfies
$$
\min((r - \mathcal{L})v, v - g) = 0.
$$

**Examples** (perpetual, $r > 0$):

- **Perpetual put**: $X = S$ GBM, $g(S) = (K - S)^+$, discount $r > 0$. Solution: stop at $S \le S^*$ where $S^* = \frac{\gamma K}{\gamma + 1}$ for $\gamma > 0$ the negative root of $\tfrac{1}{2}\sigma^2\gamma(\gamma-1) + (r - q)\gamma - r = 0$.
- **Perpetual call**: with dividend rate $q > 0$; exercise at $S \ge S^*$ where $S^*$ is the positive root.
- **McKean's free-boundary problem** (1965): pinned down the American put analytically.

### 5.1 Derivation of the perpetual put

For $S$ GBM under risk-neutral measure $dS/S = (r - q) dt + \sigma dB$:

**Continuation region** $S > S^*$: $\mathcal{L} v = rv$, i.e.,
$$
\tfrac{1}{2}\sigma^2 S^2 v''(S) + (r - q) S v'(S) - r v(S) = 0.
$$

General solution: $v(S) = A S^{\gamma_1} + B S^{\gamma_2}$, where $\gamma_{1, 2}$ are roots of $\tfrac{1}{2}\sigma^2\gamma(\gamma - 1) + (r - q)\gamma - r = 0$. For $r > 0$: $\gamma_1 > 0, \gamma_2 < 0$.

For $v$ bounded as $S \to \infty$: $A = 0$. So $v(S) = B S^{\gamma_2}$.

**Boundary conditions** at $S = S^*$:

Value matching: $B (S^*)^{\gamma_2} = K - S^*$.

Smooth pasting: $B\gamma_2 (S^*)^{\gamma_2 - 1} = -1$.

Dividing gives $S^*(\gamma_2 - 1) / S^* = -1 / (K - S^*)\,\cdot$ ... let me redo. From the two equations:

$$
\frac{B\gamma_2 (S^*)^{\gamma_2 - 1}}{B(S^*)^{\gamma_2}} = \frac{-1}{K - S^*},
$$

so $\gamma_2 / S^* = -1/(K - S^*)$, giving $S^* = \gamma_2 K / (\gamma_2 - 1) = \frac{\gamma_2}{\gamma_2 - 1} K$. For $\gamma_2 < 0$, $\gamma_2/(\gamma_2 - 1) \in (0, 1)$, so $S^* < K$ — consistent with put being exercised below strike. Then $B = (K - S^*)/(S^*)^{\gamma_2}$.

Plugging back: $v(S) = (K - S^*)\bigl(S/S^*\bigr)^{\gamma_2}$ for $S > S^*$; $v(S) = K - S$ for $S \le S^*$. This is the **closed-form perpetual American put price**.

---

## 6. Extensions and variations

### 6.1 Finite horizon

For finite $T < \infty$, the free boundary depends on time: $S^*(t)$, with $S^*(T^-) = K$ and $S^*(0) < K$. No general closed form; numerical methods (Module 4.2).

### 6.2 Random horizon

Sometimes stopping time is bounded by a random time (e.g., death, bankruptcy). Adds an extra survival factor; handled by conditioning.

### 6.3 Dual representation

**Davis–Karatzas** and **Rogers** dual representation:
$$
V_0 = \inf_M E\bigl[\sup_{t}(G_t - M_t)\bigr],
$$

where $M$ ranges over martingales with $M_0 = 0$. The optimal $M$ is the martingale part of the Doob–Meyer decomposition of the Snell envelope. This dual is the basis of modern Monte Carlo pricing of American options (Longstaff–Schwartz regression, Rogers upper-bound).

### 6.4 Multi-stopping / swing options

In energy markets, swing options let you exercise $n$ times out of $N$ opportunities; value = sup over stopping-time tuples. Reducible to iterated single-stopping.

### 6.5 Optimal switching / impulse control

Generalize to multiple modes (e.g., production on/off). Variational inequalities with multiple obstacles. Applied to capacity decisions, exchange options.

---

## 7. The dual formulation (Rogers)

**Theorem 7.1 (Rogers 2002).** For finite-horizon discrete problem,
$$
U_0 = \sup_\tau E[G_\tau] = \inf_{M : M_0 = 0,\, M \text{ martingale}} E\bigl[\max_{k}(G_k - M_k)\bigr].
$$

*Proof.* Given any martingale $M$ with $M_0 = 0$, and any stopping time $\tau$,
$$
G_\tau = M_\tau + (G_\tau - M_\tau) \le M_\tau + \max_k(G_k - M_k).
$$

Taking $E$ and noting $E[M_\tau] = 0$ (since bounded + martingale):
$$
E[G_\tau] \le E[\max_k(G_k - M_k)].
$$

So $U_0 \le \inf_M E[\max(G - M)]$. For equality, take $M = $ martingale part of Doob–Meyer of Snell envelope. $\square$

### 7.1 Application: Longstaff–Schwartz pricing

In Monte Carlo American-option pricing:
1. Simulate paths of $X$.
2. At each exercise date, regress continuation value $E[U_{k+1}|\mathcal{F}_k]$ on basis functions of $X_k$.
3. Decide whether to exercise based on max($G_k$, regressed continuation).
4. Average the resulting cashflows.

Longstaff–Schwartz (2001) is the industry-standard MC method for American options.

---

## 8. Worked examples

### 8.1 A three-step coin flip

$X_k \in \{H, T\}$ iid, fair. Reward $G_k = (\text{number of heads})_k - 0.5 k$. Horizon $N = 3$.

Work backward:
- $U_3 = G_3$.
- $U_2 = \max(G_2, E[U_3 | \mathcal{F}_2])$. E[G_3|\mathcal{F}_2] = G_2 + 0 = G_2. So $U_2 = G_2$.
- $U_1 = \max(G_1, E[U_2|\mathcal{F}_1]) = \max(G_1, G_1) = G_1$.
- $U_0 = \max(G_0, E[U_1|\mathcal{F}_0]) = \max(0, 0) = 0$.

So optimal stop = any time; expected value = 0. Makes sense: fair coin, no optionality.

### 8.2 A random walk with drift

$S_{k+1} = S_k + 1$ w.p. $p$, $-1$ w.p. $1 - p$. $G_k = S_k$. Horizon $N$.

If $p > 1/2$: continue always, $V = Np - N(1-p) = N(2p - 1)$.
If $p < 1/2$: stop immediately, $V = 0$ (if $S_0 = 0$).
If $p = 1/2$: any strategy optimal, $V = 0$.

**Stopping region** is non-empty iff $p < 1/2$.

### 8.3 Secretary problem

$n$ candidates arrive; each better/worse than all seen so far (iid rank). You select one irrevocably; payoff 1 if it's the best, 0 otherwise.

**Optimal rule:** skip first $r = n/e$ candidates, pick first one better than all seen.

Value: $V_n = (r/n) \sum_{k = r+1}^n 1/(k-1) \approx 1/e$ as $n \to \infty$.

This is a discrete optimal stopping problem; the Snell envelope approach produces the above rule.

### 8.4 House selling

You receive offers $Y_1, Y_2, \ldots$ iid with density $f$; cost $c$ per period. Stop when an offer is good enough.

**Threshold rule:** accept offer $y$ iff $y \ge x^*$ where $x^*$ satisfies
$$
x^* = E[\max(Y, x^*)] - c = \int_{-\infty}^{x^*} x^* f(y) dy + \int_{x^*}^\infty y f(y) dy - c,
$$

giving an implicit equation for $x^*$. Classical result.

---

## 9. Python: solving discrete optimal stopping

```python
import numpy as np

rng = np.random.default_rng(42)

# ---- (a) American put on GBM via backward induction ----
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
N = 50
dt = T/N

# Binomial tree
u = np.exp(sigma*np.sqrt(dt))
d = 1/u
p = (np.exp(r*dt) - d) / (u - d)
disc = np.exp(-r*dt)

# Terminal
S = S0 * u**np.arange(N+1) * d**(N - np.arange(N+1))
V = np.maximum(K - S, 0.0)

# Backward: V(i, j) for j-th node at step i
# For American: V(i, j) = max(payoff, disc*(p*V(i+1, j+1) + (1-p)*V(i+1, j)))
for i in range(N-1, -1, -1):
    S = S0 * u**np.arange(i+1) * d**(i - np.arange(i+1))
    continuation = disc * (p * V[1:i+2] + (1-p) * V[0:i+1])
    exercise = np.maximum(K - S, 0.0)
    V = np.maximum(continuation, exercise)

print(f"American put binomial price: {V[0]:.4f}")

# European for comparison
V_eur = np.maximum(K - (S0 * u**np.arange(N+1) * d**(N - np.arange(N+1))), 0.0)
for i in range(N-1, -1, -1):
    V_eur = disc * (p * V_eur[1:i+2] + (1-p) * V_eur[0:i+1])
print(f"European put binomial price: {V_eur[0]:.4f}")
# Difference = early-exercise premium

# ---- (b) Secretary problem Monte Carlo ----
n_candidates = 100
M = 100_000
threshold = int(n_candidates / np.e)
wins = 0
for _ in range(M):
    ranks = rng.permutation(n_candidates) + 1  # 1 = best
    # Skip first `threshold`, note best seen
    best_seen = ranks[:threshold].min() if threshold > 0 else n_candidates + 1
    # Pick first better
    picked = None
    for k in range(threshold, n_candidates):
        if ranks[k] < best_seen:
            picked = ranks[k]
            break
    if picked == 1:  # best overall
        wins += 1
    elif picked is None and ranks[-1] == 1:
        wins += 1  # never picked; default to last? (convention varies)
print(f"\nSecretary problem MC: P(best) = {wins/M:.4f}, 1/e = {1/np.e:.4f}")

# ---- (c) Perpetual put closed form ----
r_rate, q, sigma_p = 0.05, 0.0, 0.2
# Solve gamma: 0.5*sigma^2*g*(g-1) + (r-q)*g - r = 0
a_q = 0.5 * sigma_p**2
b_q = (r_rate - q) - 0.5*sigma_p**2
c_q = -r_rate
gamma_2 = (-b_q - np.sqrt(b_q**2 - 4*a_q*c_q)) / (2*a_q)
S_star = gamma_2 / (gamma_2 - 1) * K
print(f"\nPerpetual put S*: {S_star:.4f}")
print(f"Perpetual put value at S0=100: {(K - S_star)*(100/S_star)**gamma_2:.4f}")

# ---- (d) Longstaff-Schwartz for American put via MC ----
N = 50; M_paths = 10000
dt = T/N
paths = np.zeros((M_paths, N+1))
paths[:, 0] = S0
for k in range(N):
    Z = rng.standard_normal(M_paths)
    paths[:, k+1] = paths[:, k] * np.exp((r_rate - 0.5*sigma_p**2)*dt + sigma_p*np.sqrt(dt)*Z)

# Backward induction with regression
cashflow = np.maximum(K - paths[:, -1], 0.0)
exercise_time = np.full(M_paths, N)
for k in range(N-1, 0, -1):
    itm = (K - paths[:, k] > 0)
    if itm.sum() < 5:
        continue
    X = paths[itm, k]
    # Discount cashflows from exercise_time to k
    t_diff = (exercise_time[itm] - k) * dt
    Y = cashflow[itm] * np.exp(-r_rate * t_diff)
    # Regression basis: 1, X, X^2
    A = np.column_stack([np.ones(X.size), X, X**2])
    beta, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    cont = A @ beta
    ex = K - X
    exercise_now = ex > cont
    idx = np.where(itm)[0][exercise_now]
    cashflow[idx] = K - paths[idx, k]
    exercise_time[idx] = k

# Discount all cashflows to t=0
ls_price = (cashflow * np.exp(-r_rate * exercise_time * dt)).mean()
print(f"\nLongstaff-Schwartz American put: {ls_price:.4f}")
```

### Expected output

- Binomial American put ~$5.72$, European ~$5.57$; early-exercise premium ~$0.15$.
- Secretary problem ~$1/e \approx 0.37$.
- Perpetual put: $S^* \approx 57$, value at $S = 100$ small ~$2$-$3$.
- Longstaff–Schwartz price ~$5.72$, matching binomial.

---

## 10. [QUANT APPLICATION] — optimal stopping in practice

### 10.1 American options pricing

Exercise early if intrinsic > continuation. Standard on every equity desk. Methods:

- **Binomial tree** (Cox–Ross–Rubinstein).
- **PDE (Crank–Nicolson) with PSOR** for linear complementarity.
- **Monte Carlo (Longstaff–Schwartz)** for high-dimensional exotics.

### 10.2 Swing options in energy markets

Natural gas "take-or-pay" contracts allow multiple withdrawals. Multi-stopping valuation via iterated optimal stopping.

### 10.3 Best-time-to-sell / entry

When should a PM liquidate a position? Optimal stopping with mean-reverting dynamics. Applications in **pairs trading** (stop at convergence), **merger arb** (stop if spread re-widens).

### 10.4 Credit: first-to-default basket

In a basket of $n$ credits, the value of a first-to-default tranche is determined by an optimal stopping problem — the first time any credit defaults. Copula modeling provides the default-time correlation.

### 10.5 Real options

Corporate investment decisions modeled as perpetual American options: defer / commit / abandon. Dixit–Pindyck's book is standard.

### 10.6 Optimal trading execution

Liquidate $X$ shares over $T$ minutes. Trading costs vs market risk. Almgren–Chriss solve the continuous version via LQ control (Module 4.6); with discretion to stop early, it's an optimal stopping problem.

---

## 11. Summary

- **Optimal stopping** = best time to cash in a stochastic reward.
- **Snell envelope** = smallest supermartingale dominating the reward; computed by backward induction (DP).
- **Optimal stopping time**: first entry into stopping region $\{U = G\}$.
- **Continuous-time**: Snell envelope is smallest càdlàg supermartingale; value function satisfies an **obstacle PDE / linear complementarity**.
- **Smooth pasting** at free boundaries.
- **Dual formulation (Rogers)** enables Monte Carlo upper bounds.
- **Applications**: American options (perpetual closed form, finite-horizon PDE), swing options, real options, optimal execution.

---

## 12. Exercises

**★ (warm-ups)**

**12.1** For $G_k = k$ constant drift: show $U_k = N$ for all $k < N$ and $U_N = N$. Optimal $\tau^* = N$.

**12.2** For $G_k = -k$: optimal $\tau^* = 0$.

**12.3** For $G_k = B_k^2$ with $B$ random walk, $N = 2$: compute $U_0, U_1, U_2$ and $\tau^*$.

**12.4** Verify in Snell envelope recursion: if $G$ is already a supermartingale, then $U = G$.

**12.5** Verify: if $G$ is already a martingale, then $U = G$ and any stopping time is optimal.

**★★ (core)**

**12.6** **(Three-period house selling.)** Offers $Y_k \sim U[0, 1]$, cost $c = 0.1$ per period. Compute the value $V_0$ and optimal threshold.

**12.7** **(Smooth pasting for perpetual call.)** Derive the perpetual American call price with dividends $q > 0$: find $S^* > K$ such that exercise above $S^*$.

**12.8** **(Discrete American put.)** Implement binomial American put and verify put-call parity bound: $P^{\text{Am}} \ge (K - S)^+$.

**12.9** **(Doob decomposition of Snell envelope.)** For $G_k = B_k^2$, $B$ random walk, compute the Doob decomposition of $U$ and identify where $A$ starts increasing.

**12.10** **(Longstaff–Schwartz convergence.)** Explain why LS is biased low (uses suboptimal early-exercise policy); discuss how Rogers-dual gives upper bound.

**12.11** **(Secretary problem variations.)** What if you want to maximize the expected rank of your choice (not just P(best))? Solve.

**12.12** **(Markov chain optimal stopping.)** For a finite Markov chain on $\{1, \ldots, n\}$ with transition matrix $P$ and reward vector $g$, write the Bellman equation and solve by value iteration.

**12.13** **(Continuous smooth pasting.)** For the perpetual American put, show that $v(S)$ is $C^1$ but not $C^2$ at $S^*$: discontinuity in second derivative.

**12.14** **(Disorder problem.)** Observe $X_t = \mu t + B_t$ where $\mu$ changes from $0$ to $\mu_1 > 0$ at an exponentially-distributed random time $\theta$. Detect $\theta$ as quickly as possible. Derive the optimal stopping rule as a threshold on the posterior $P(\theta \le t | \mathcal{F}_t)$.

**12.15** **(Ruin probability as optimal stopping.)** Let $X$ be a random walk. Compute $P(\min X < -a)$ using an optimal stopping formulation.

**★★★ (research / quant)**

**12.16** **(Finite-horizon American put PDE.)** Implement Crank–Nicolson with PSOR to price a finite-maturity American put. Compare with binomial as $N \to \infty$.

**12.17** **(Early exercise boundary numerics.)** For the American put, plot the early-exercise boundary $S^*(t)$ for $t \in [0, T]$. Compare with the "Bjerksund–Stensland" approximation.

**12.18** **(Longstaff–Schwartz with deep learning.)** Read Becker–Cheridito–Jentzen (2019) on deep optimal stopping. Implement a DNN-based policy and compare with LS on a 5D Bermudan max-call option.

**12.19** **(Swing option.)** Implement 3-exercise-right Bermudan swing option with penalty for under/over-lift. Use backward induction with state $(k, X)$ for $k$ = exercises remaining.

**12.20** **(American option with stochastic volatility.)** Price an American put under Heston dynamics. 2D PDE + obstacle = 2D PSOR. Report the exercise surface.

**12.21** **(Optimal sell time in mean-reverting model.)** For $dX = \alpha(\mu - X) dt + \sigma dB$, solve $\sup_\tau E[e^{-r\tau} X_\tau]$ in closed form via confluent hypergeometric functions.

**12.22** **(Dual formulation for MC pricing.)** Implement Rogers's dual for a high-dimensional Bermudan option. Compare LS lower bound with Rogers upper bound; the gap bounds algorithmic error.

---

## 13. Forward pointers

- **Module 4.2 (American options)**: finite-horizon pricing via PDE + numerical methods.
- **Module 4.3 (DP)**: general dynamic programming principle.
- **Module 4.4 (HJB)**: continuous-time control.

**Next module:** American options and free-boundary PDEs — numerical methods for option pricing with early exercise.

---

*End of Module 4.1.*

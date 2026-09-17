# Module 4.3 — Dynamic Programming and Bellman Equations

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 4 (Optimal Stopping & Stochastic Control), Module 3 of 7

---

## Prerequisites

- **Module 2.7** (Markov chains).
- **Module 4.1** (Optimal stopping, Snell envelope).
- **Module 4.2** (American options, DP via backward induction).
- Elementary analysis: fixed-point theorems, Banach contraction.

Dynamic programming (DP) is the general framework for sequential decision-making under uncertainty. This module develops Bellman's principle of optimality, the Bellman equation, and solution algorithms (value iteration, policy iteration).

---

## 1. Markov decision processes (MDPs)

### 1.1 Definition

An MDP consists of:

- **State space** $\mathcal{X}$ (often finite; can be continuous).
- **Action space** $\mathcal{A}$ (actions available at each state; may depend on state).
- **Transition kernel** $P(x' | x, a)$: probability of going to $x'$ from $x$ given action $a$.
- **Reward function** $r(x, a)$: immediate reward from taking action $a$ at state $x$.
- **Discount factor** $\gamma \in (0, 1]$: future rewards discounted by $\gamma^t$.
- **Policy** $\pi : \mathcal{X} \to \mathcal{A}$ (or distribution over actions).

**Value function** for policy $\pi$:
$$
V^\pi(x) := E^\pi\Bigl[\sum_{t = 0}^\infty \gamma^t r(X_t, A_t) \Big| X_0 = x\Bigr].
$$

**Optimal value:**
$$
V^*(x) := \sup_\pi V^\pi(x).
$$

---

## 2. Bellman's principle of optimality

**Theorem 2.1 (Bellman).** The optimal value function satisfies
$$
V^*(x) = \sup_{a \in \mathcal{A}}\bigl[r(x, a) + \gamma\, E_{x' \sim P(\cdot|x, a)}[V^*(x')]\bigr].
$$

This is the **Bellman equation**.

*Proof.*

$(\le)$ For any policy $\pi$, the first action $A_0 = \pi(x)$ yields immediate reward $r(x, A_0)$ and a continuation value of $E[V^{\pi'}(X_1)|X_0 = x]$ for an optimal continuation policy $\pi'$. So
$$
V^\pi(x) \le r(x, A_0) + \gamma E[V^*(X_1)|X_0 = x] \le \sup_a [r(x, a) + \gamma E[V^*(X_1)|X_0 = x, A_0 = a]].
$$

Taking sup over $\pi$: $V^*(x) \le \text{RHS}$.

$(\ge)$ Pick any $a$; define policy that takes $a$ at time 0, then optimal after. Value = $r(x, a) + \gamma E[V^*(X_1)]$. So $V^*(x) \ge \sup_a [\ldots]$. $\square$

### 2.1 Principle of optimality

**Bellman:** "An optimal policy has the property that whatever the initial state and initial decision, the remaining decisions must constitute an optimal policy with respect to the state resulting from the first decision."

Equivalently: **subproblems are nested**. The optimal solution to a big problem contains optimal solutions to its subproblems.

---

## 3. Bellman operator

Define the **Bellman operator** $T : \mathbb{R}^{\mathcal{X}} \to \mathbb{R}^{\mathcal{X}}$:
$$
(TV)(x) := \sup_{a \in \mathcal{A}} [r(x, a) + \gamma\, E_{x'|x, a}[V(x')]].
$$

Bellman's equation: $V^* = T V^*$. The optimal value function is a fixed point of $T$.

**Theorem 3.1 (Banach contraction).** If $\gamma < 1$ and $\|r\|_\infty < \infty$, $T$ is a $\gamma$-contraction on $(\mathbb{R}^{\mathcal{X}}, \|\cdot\|_\infty)$:
$$
\|TV - TV'\|_\infty \le \gamma \|V - V'\|_\infty.
$$

By Banach fixed-point theorem, $T$ has a unique fixed point, and iteration converges:
$$
\|T^n V_0 - V^*\|_\infty \le \gamma^n \|V_0 - V^*\|_\infty.
$$

### 3.1 Proof of contraction

$$
|TV(x) - TV'(x)| = |\sup_a [r(x,a) + \gamma E[V(X_1)|x, a]] - \sup_a [r(x,a) + \gamma E[V'(X_1)|x,a]]|
$$
$$
\le \sup_a |\gamma E[V(X_1) - V'(X_1)|x, a]| \le \gamma \|V - V'\|_\infty.
$$

Using $|\sup f - \sup g| \le \sup |f - g|$. $\square$

---

## 4. Value iteration

**Algorithm.** Initialize $V_0$ (any bounded function). Iterate:
$$
V_{k+1} := T V_k.
$$

By Theorem 3.1, $V_k \to V^*$ exponentially fast.

### 4.1 Greedy policy

Once $V^*$ is computed (or approximated), the **greedy policy**
$$
\pi^*(x) := \arg\max_a [r(x, a) + \gamma E[V^*(X_1)|x, a]]
$$

is optimal.

### 4.2 Convergence rate

$\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty$. For $\gamma = 0.99$, need $\sim 460$ iterations for 0.01 accuracy; for $\gamma = 0.9$, only $\sim 44$.

---

## 5. Policy iteration

**Algorithm.**

1. **Policy evaluation**: given policy $\pi$, compute $V^\pi$ by solving $V^\pi = T^\pi V^\pi$ (a linear system if $\mathcal{X}$ is finite).
2. **Policy improvement**: set $\pi_{\text{new}}(x) := \arg\max_a [r(x, a) + \gamma E[V^\pi(X_1)|x, a]]$.
3. Iterate until policy stabilizes.

**Theorem 5.1.** Policy iteration converges to $\pi^*$ in finitely many steps (finite $\mathcal{X}, \mathcal{A}$).

*Proof idea.* Each iteration strictly improves $V^\pi$ (unless at optimum), and there are finitely many policies. $\square$

### 5.1 Modified policy iteration

Combine: do $k$ steps of value iteration instead of exact policy evaluation. Often faster in practice.

---

## 6. Infinite vs finite horizon

### 6.1 Finite horizon

Reward $\sum_{t=0}^T r(X_t, A_t)$. Value function $V_t(x)$ depends on time:
$$
V_T(x) := \text{terminal reward}, \quad V_{t-1}(x) = \sup_a [r(x, a) + E[V_t(X_1)|x, a]].
$$

Backward induction solves in $T$ steps. No contraction needed.

### 6.2 Infinite horizon

$V^*$ independent of time. Requires $\gamma < 1$ (discounting) or average-reward setup for unique stationary solution.

### 6.3 Average reward

Maximize $\liminf_T \tfrac{1}{T} \sum r(X_t, A_t)$. The relative value function $h$ satisfies
$$
\rho + h(x) = \sup_a [r(x, a) + E[h(X_1)|x, a]],
$$
where $\rho$ is the average reward. Ergodic theory required.

---

## 7. Stochastic shortest path

Special case of MDP: goal state that terminates the process; no discounting. Examples: optimal execution (unload shares), portfolio rebalancing to target.

**Bellman equation** with absorbing terminal state:
$$
V^*(x) = \sup_a [r(x, a) + E[V^*(X_1)|x, a]].
$$

Convergence requires "proper policies" (reach terminal with probability 1) under mild conditions.

---

## 8. Worked examples

### 8.1 Gambler's ruin (DP)

Coin flip with prob $p$ win. Goal: reach $\$N$ starting from $\$i$. Choose stake $a \in \{0, \ldots, \min(i, N - i)\}$ each round. Reward: $1$ if reach $\$N$, $0$ if bust.

Bellman:
$$
V^*(i) = \max_a [p V^*(i + a) + (1 - p) V^*(i - a)], \quad V^*(0) = 0, V^*(N) = 1.
$$

Solution: for $p \le 1/2$, play boldly ($a = \min(i, N-i)$); for $p > 1/2$, play timidly ($a = 1$).

### 8.2 Optimal consumption-savings

Wealth $W_t$; consume $C_t \in [0, W_t]$; save $(1 + r)(W_t - C_t)$. Maximize $\sum \beta^t u(C_t)$ with discount $\beta \in (0, 1)$ and utility $u$.

**Bellman equation:**
$$
V(W) = \sup_{C \in [0, W]} [u(C) + \beta V((1+r)(W - C))].
$$

For CRRA utility $u(c) = c^{1-\gamma}/(1 - \gamma)$, closed form: $V(W) = A W^{1-\gamma}/(1-\gamma)$; $C = \kappa W$ with $\kappa$ explicit.

### 8.3 Secretary problem (DP reformulation)

At stage $k$, state = (rank of current candidate, stage). Decision: accept or reject. $V_k(r) = \max(r, E[V_{k+1}])$. Solution: threshold rule $k > n/e$.

### 8.4 Inventory control

State = current inventory $I_t$. Action = order quantity $a_t$. Demand $D_t \sim F$. Cost: ordering cost + holding + stockout. Optimal policy: $(s, S)$ rule — order up to $S$ if $I < s$, else nothing.

---

## 9. Python: value iteration and policy iteration

```python
import numpy as np

# ---- (a) Simple gridworld MDP ----
# 4x4 grid, actions N/S/E/W, reward -1 per step, terminal corner (3,3)
rows, cols = 4, 4
n_states = rows * cols
n_actions = 4  # N=0, S=1, E=2, W=3
gamma = 0.95

def to_rc(s): return (s // cols, s % cols)
def to_s(r, c): return r * cols + c

terminal = to_s(3, 3)

def next_state(s, a):
    r, c = to_rc(s)
    if a == 0: r = max(0, r-1)   # North
    elif a == 1: r = min(rows-1, r+1)  # South
    elif a == 2: c = min(cols-1, c+1)  # East
    elif a == 3: c = max(0, c-1)  # West
    return to_s(r, c)

def reward(s, a, sp):
    return -1.0 if s != terminal else 0.0

# Build transition tensor
P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions))
for s in range(n_states):
    for a in range(n_actions):
        sp = next_state(s, a)
        P[s, a, sp] = 1.0
        R[s, a] = reward(s, a, sp)

# ---- Value iteration ----
V = np.zeros(n_states)
for iter in range(1000):
    V_new = np.max(R + gamma * P @ V, axis=1)
    V_new[terminal] = 0.0
    if np.max(np.abs(V_new - V)) < 1e-8:
        break
    V = V_new
print(f"Value iteration converged in {iter+1} iterations")
print("V* (grid):")
print(V.reshape(rows, cols).round(2))

# ---- Policy iteration ----
pi = np.zeros(n_states, dtype=int)
for pi_iter in range(100):
    # Evaluate pi
    P_pi = np.array([P[s, pi[s]] for s in range(n_states)])
    R_pi = np.array([R[s, pi[s]] for s in range(n_states)])
    # V_pi = R_pi + gamma * P_pi V_pi  =>  V_pi = (I - gamma P_pi)^{-1} R_pi
    V_pi = np.linalg.solve(np.eye(n_states) - gamma * P_pi, R_pi)
    V_pi[terminal] = 0.0
    # Improve
    pi_new = np.argmax(R + gamma * P @ V_pi, axis=1)
    if np.array_equal(pi_new, pi):
        break
    pi = pi_new
print(f"\nPolicy iteration converged in {pi_iter+1} iterations")
arrows = {0: '↑', 1: '↓', 2: '→', 3: '←'}
print("Optimal policy (grid):")
for r in range(rows):
    print(" ".join(arrows[pi[to_s(r, c)]] for c in range(cols)))

# ---- (b) Gambler's ruin ----
p = 0.4
N = 100
V = np.zeros(N+1)
V[N] = 1.0
for iter in range(10000):
    V_new = V.copy()
    for i in range(1, N):
        best = V[i]
        for a in range(1, min(i, N-i)+1):
            val = p * V[i+a] + (1-p) * V[i-a]
            best = max(best, val)
        V_new[i] = best
    if np.max(np.abs(V_new - V)) < 1e-10:
        break
    V = V_new
print(f"\nGambler's ruin converged in {iter+1} iters")
print(f"P(win from $50 with p=0.4): {V[50]:.5f}")
# Known closed form: [(q/p)^i - 1] / [(q/p)^N - 1]
ratio = (1-p)/p
known = (ratio**50 - 1) / (ratio**N - 1)
print(f"Known closed form:      {known:.5f}")
```

### Expected output

- Gridworld VI converges in ~200 iterations for $\gamma = 0.95$. Optimal V shows negative values increasing toward $0$ at terminal.
- Policy iteration converges in 5-10 iterations (much faster than VI per iteration but more work per iteration).
- Gambler's ruin with $p = 0.4$, bold play: $P(\text{win from } 50) \approx $ something like 0.05; matches closed form.

---

## 10. [QUANT APPLICATION] — DP in finance

### 10.1 Optimal stopping (Module 4.1)

DP backward induction computes the Snell envelope.

### 10.2 American option pricing

DP on finite-state binomial tree or finite-difference grid; in higher dimensions, LS is a DP with regression.

### 10.3 Portfolio choice

Discrete-time Merton: Bellman equation for $V(W_t, t) = \sup_\pi E[\sum \beta^{t'} u(C_{t'})]$. Closed form for CRRA.

### 10.4 Inventory / hedging with transaction costs

Optimal rebalancing of a hedging portfolio in the presence of fixed + proportional transaction costs. Bellman $V(X, S) = \max(V^{\text{no-trade}}, V^{\text{trade}})$, three-region structure (buy / no-trade / sell).

### 10.5 Market making

Bid-ask quoting as a control problem: maximize expected $P\&L = $ spread captured minus inventory risk. Avellaneda–Stoikov (2008) derive closed-form quotes via DP.

### 10.6 Optimal execution

Almgren–Chriss optimal liquidation: minimize cost of trading $X$ shares over $T$ minutes subject to market-impact cost. Continuous-time LQ version in Module 4.6.

### 10.7 Utility maximization with stochastic vol

DP over (wealth, vol) state; Heston setup. Fleming–Hernández-Hernández give semi-closed-form solutions.

---

## 11. Summary

- **Markov Decision Processes** are the framework for sequential decisions under uncertainty.
- **Bellman's principle:** optimal = local optimal + expected optimal continuation.
- **Bellman operator** $T$ is a contraction for discounted MDPs; fixed-point iteration converges.
- **Value iteration** and **policy iteration** are the two primary algorithms.
- Finite-horizon: backward induction. Infinite-horizon: iterative.
- Applications: optimal stopping, American options, portfolio choice, market making, execution.

---

## 12. Exercises

**★ (warm-ups)**

**12.1** Verify that the Bellman operator $T$ is monotone: $V \le V' \Rightarrow TV \le TV'$.

**12.2** Show that $T$ is $\gamma$-contraction (supply the proof).

**12.3** For a 2-state MDP with given $P, R$, write down the Bellman equation and solve it.

**12.4** Show value iteration converges to the optimal $V^*$ from any initial $V_0$.

**12.5** Show policy iteration converges in finitely many steps for finite MDP.

**★★ (core)**

**12.6** **(Weighted sup-norm.)** For unbounded state space, $T$ may not contract in sup-norm. Show it contracts in a weighted sup-norm with weights $w(x) = (1 + |x|)^n$ under linear-growth conditions.

**12.7** **(Gauss–Seidel value iteration.)** Show Gauss–Seidel style in-place updates converge faster than Jacobi.

**12.8** **(Howard's policy iteration convergence.)** Show finite-policy convergence via monotonicity argument.

**12.9** **(Modified policy iteration.)** Implement modified PI with $k$ evaluation steps; empirically compare $k = 1$ (VI), $k = \infty$ (PI), and $k = 10$.

**12.10** **(Optimal consumption CRRA.)** For $dW = rW dt - C dt$, $u(C) = C^{1-\gamma}/(1-\gamma)$, discount $\rho$: derive closed-form $V(W) = A W^{1-\gamma}/(1-\gamma)$, find $A$, and find optimal $C^* = \kappa W$.

**12.11** **(Inventory $(s, S)$ rule.)** Prove that under K-convexity of cost, the optimal inventory policy is of the $(s, S)$ form (Scarf's theorem).

**12.12** **(Bellman for continuous state.)** Extend Bellman equation to continuous state: replace sums with integrals. Show $T$ is contraction on $C_b$.

**12.13** **(LQR via DP.)** For $x_{t+1} = Ax_t + Bu_t + w_t$, minimize $\sum x^T Q x + u^T R u$. Derive the discrete-time Riccati equation.

**12.14** **(Constrained MDP.)** $\max E[\sum \gamma^t r(X, A)]$ subject to $E[\sum \gamma^t g(X, A)] \le C$. Use Lagrangian to reduce to unconstrained.

**12.15** **(Real options.)** Firm can invest $I$ now to receive $V(X) - I$ where $V$ is GBM. Solve optimal investment timing via optimal stopping / DP.

**★★★ (research / quant)**

**12.16** **(Partially Observed MDPs (POMDPs).)** Extend MDP to when state is not directly observed. Belief-state MDP: Bellman on the posterior. Computationally hard; use point-based value iteration.

**12.17** **(Approximate DP.)** For large state spaces, $V^*$ is approximated: $\hat V(x) = \phi(x)^T \theta$. Fitted value iteration projects $T \hat V$ onto basis. Discuss convergence (Bertsekas).

**12.18** **(Mean-variance optimal investment.)** Classical Markowitz as static DP; extended to Bellman's dynamic mean-variance (Li–Zhou 2000). Time-inconsistency issues.

**12.19** **(DP for Bayesian regret.)** In a multi-armed bandit, the Gittins index gives the optimal policy via DP reformulation.

**12.20** **(Optimal hedging with friction.)** Implement DP with proportional transaction costs. Show the no-trade region $(a, b) \ni \text{position}$ and derive its asymptotic width.

**12.21** **(Stochastic shortest path.)** Implement DP for routing / liquidation problems. Show convergence under proper policies.

**12.22** **(Reinforcement learning preview.)** If $P, R$ are unknown but samples $(X, A, X', R)$ available: TD-learning implements incremental Bellman updates. Module 4.7.

---

## 13. Forward pointers

- **Module 4.4 (HJB)**: continuous-time Bellman equation for continuous-state/action MDPs.
- **Module 4.5 (Merton)**: classic application to portfolio / consumption.
- **Module 4.6 (LQG)**: linear dynamics + quadratic cost = analytic DP.
- **Module 4.7 (RL)**: sample-based / model-free DP.

**Next module:** HJB equations — the continuous-time analogue where Bellman's equation becomes a nonlinear PDE.

---

*End of Module 4.3.*

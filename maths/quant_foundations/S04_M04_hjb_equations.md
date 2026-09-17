# Module 4.4 — Hamilton–Jacobi–Bellman Equations

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 4 (Optimal Stopping & Stochastic Control), Module 4 of 7

---

## Prerequisites

- **Module 3.3** (Itô's formula).
- **Module 3.4** (SDEs, Feynman–Kac).
- **Module 4.3** (Dynamic programming, Bellman principle).
- Basic functional analysis (maximum principle, comparison).

This module develops the continuous-time analogue of the Bellman equation: the **Hamilton–Jacobi–Bellman (HJB) equation**, a fully nonlinear parabolic PDE governing stochastic optimal control.

---

## 1. The stochastic optimal control problem

### 1.1 Setup

- **State process** $X = (X_t)_{0 \le t \le T}$ evolving according to
$$
dX_t = \mu(t, X_t, \alpha_t) dt + \sigma(t, X_t, \alpha_t) dB_t, \quad X_0 = x,
$$

where $\alpha_t$ is an adapted control process with values in action set $\mathcal{A}$.

- **Objective:**
$$
J(t, x; \alpha) := E\Bigl[\int_t^T f(s, X_s, \alpha_s) ds + g(X_T)\Big|\, X_t = x\Bigr],
$$

with $f$ the running cost/reward and $g$ the terminal payoff.

- **Value function:**
$$
v(t, x) := \sup_\alpha J(t, x; \alpha).
$$

---

## 2. Heuristic derivation of the HJB equation

### 2.1 Dynamic programming in continuous time

Bellman's principle says: optimal from $(t, x)$ = $\sup_\alpha$ over small time increment $h$ of [running cost + optimal from $(t + h, X_{t+h})$]:
$$
v(t, x) = \sup_{\alpha} E\Bigl[\int_t^{t+h} f(s, X_s, \alpha_s) ds + v(t + h, X_{t+h})\Bigr].
$$

### 2.2 Apply Itô to $v$

For a sufficiently smooth $v$ and a constant control $a \in \mathcal{A}$ over $[t, t + h]$,
$$
v(t + h, X_{t+h}) = v(t, x) + \int_t^{t+h}[\partial_s v + \mathcal{L}^a v] ds + \int_t^{t+h} \sigma \partial_x v\, dB_s,
$$

where
$$
\mathcal{L}^a v := \mu(s, x, a) \partial_x v + \tfrac{1}{2}\sigma^2(s, x, a) \partial_{xx} v.
$$

Taking expectation (the Itô integral has mean zero under regularity):
$$
E[v(t+h, X_{t+h})] = v(t, x) + \int_t^{t+h} E[\partial_s v + \mathcal{L}^a v] ds.
$$

### 2.3 Passing to limits

Bellman's principle gives
$$
0 = \sup_a E\Bigl[\int_t^{t+h}[f(s, X_s, a) + \partial_s v + \mathcal{L}^a v] ds\Bigr].
$$

Dividing by $h$ and taking $h \to 0$:
$$
\boxed{\; \partial_t v(t, x) + \sup_{a \in \mathcal{A}}\bigl[f(t, x, a) + \mathcal{L}^a v(t, x)\bigr] = 0,\quad v(T, x) = g(x). \;}
$$

This is the **Hamilton–Jacobi–Bellman equation**. It is a **fully nonlinear parabolic PDE** due to the $\sup_a$.

---

## 3. The verification theorem

Under regularity, a smooth solution of the HJB equation is the value function.

**Theorem 3.1 (Verification).** Suppose $v \in C^{1, 2}$ solves the HJB equation with terminal condition $v(T, x) = g(x)$, and satisfies polynomial growth. Then:

(a) $v(t, x) \ge J(t, x; \alpha)$ for every admissible $\alpha$.

(b) If $\alpha^* = \alpha^*(t, X_t)$ achieves the supremum in the HJB at each $(t, x)$, then $\alpha^*$ is optimal: $v(t, x) = J(t, x; \alpha^*)$.

*Proof.*

(a) Apply Itô to $v(t, X_t^\alpha)$:
$$
v(T, X_T^\alpha) = v(t, x) + \int_t^T [\partial_s v + \mathcal{L}^\alpha_s v](s, X_s^\alpha) ds + \int_t^T \sigma \partial_x v dB_s.
$$

By HJB, $\partial_s v + \mathcal{L}^\alpha_s v \le -f(s, X_s^\alpha, \alpha_s)$. So
$$
g(X_T^\alpha) = v(T, X_T^\alpha) \le v(t, x) - \int_t^T f(s, X_s^\alpha, \alpha_s) ds + \text{martingale}.
$$

Take $E$: $v(t, x) \ge E[\int_t^T f ds + g(X_T^\alpha)] = J(t, x; \alpha)$.

(b) With $\alpha = \alpha^*$, HJB gives equality, so the above inequality is equality: $v(t, x) = J(t, x; \alpha^*)$. $\square$

### 3.1 Interpretation

- The HJB's $\sup_a$ captures the instantaneous best decision.
- The verification theorem reverses the perspective: if you can guess or construct $v$ solving HJB, you know the optimal control.

---

## 4. Example: Merton's consumption-investment

Wealth $W$; investor allocates fraction $\pi$ to risky asset, $1 - \pi$ to risk-free; consumes at rate $C$. Dynamics:
$$
dW_t = [\pi\mu + (1-\pi) r] W_t dt - C_t dt + \pi\sigma W_t dB_t = [(r + \pi(\mu - r)) W - C] dt + \pi\sigma W dB.
$$

Maximize $\int_0^T e^{-\rho t} u(C_t) dt + B(W_T)$. HJB:
$$
\rho v = \sup_{C \ge 0, \pi} \bigl[u(C) + \partial_t v + (r W + \pi(\mu - r) W - C)\partial_W v + \tfrac{1}{2}\pi^2 \sigma^2 W^2 \partial_{WW} v\bigr].
$$

First-order conditions:

- $\partial_C$: $u'(C) = \partial_W v$, so $C^* = (u')^{-1}(\partial_W v)$.
- $\partial_\pi$: $(\mu - r)W \partial_W v + \pi\sigma^2 W^2 \partial_{WW} v = 0$, so
$$
\pi^* = -\frac{(\mu - r)}{\sigma^2} \cdot \frac{\partial_W v}{W \partial_{WW} v}.
$$

For **CRRA utility** $u(C) = C^{1-\gamma}/(1 - \gamma)$, the ansatz $v(t, W) = h(t) W^{1-\gamma}/(1-\gamma)$ reduces HJB to an ODE for $h(t)$:
$$
h'(t) = -h(t) K + \gamma h(t)^{1 - 1/\gamma} (\text{adjustments}), \qquad h(T) = B_\text{term}.
$$

**Merton's optimal portfolio fraction:**
$$
\pi^* = \frac{\mu - r}{\gamma \sigma^2},
$$

independent of wealth and (for infinite horizon) of time.

**Optimal consumption:** $C^* = \kappa(t) W$ for an explicit $\kappa(t)$.

This is the subject of Module 4.5 in detail.

---

## 5. Viscosity solutions

The HJB equation is nonlinear. Classical solutions may not exist; even when they do, uniqueness requires care.

### 5.1 Why classical solutions fail

The $\sup_a$ makes the equation nonlinear; $v$ may fail to be $C^2$ (e.g., free boundaries in optimal stopping).

### 5.2 Viscosity solution concept (Crandall–Lions 1983)

**Definition 5.1.** A continuous function $v$ is a **viscosity subsolution** of $F(t, x, v, Dv, D^2 v) = 0$ if, whenever $v - \phi$ attains a local max at $(t_0, x_0)$ for $\phi \in C^2$,
$$
F(t_0, x_0, v(t_0, x_0), D\phi(t_0, x_0), D^2\phi(t_0, x_0)) \le 0.
$$

**Viscosity supersolution**: reverse inequality at local min.

**Viscosity solution**: both sub and super.

### 5.3 Comparison principle

Under mild regularity, viscosity sub- and super-solutions satisfying appropriate boundary conditions satisfy a **comparison principle**: subsolution $\le$ supersolution everywhere. This guarantees uniqueness.

### 5.4 Existence

Under Lipschitz and linear-growth conditions on coefficients and costs, the value function is the unique viscosity solution of the HJB equation.

**Reference:** Fleming–Soner (2006) *Controlled Markov Processes and Viscosity Solutions*.

---

## 6. Numerical methods for HJB

### 6.1 Finite differences

Discretize $(t, x)$; approximate $\partial_t, \partial_x, \partial_{xx}$ by finite differences. Compute $\sup_a$ by enumerating actions (if discrete) or Newton (if continuous).

### 6.2 Monotone schemes

Barles–Souganidis (1991): monotone + consistent + stable $\Rightarrow$ converges to viscosity solution. Key property: schemes that preserve the maximum principle.

### 6.3 Semi-Lagrangian schemes

For problems with singular diffusion or near-optimal control degeneracy, semi-Lagrangian (discretize along characteristics) is more stable than finite differences.

### 6.4 Markov chain approximation (Kushner–Dupuis)

Replace SDE by a Markov chain on a grid; DP on the chain converges to HJB solution. Standard workhorse.

### 6.5 Deep learning (Han–Jentzen–E 2017)

Parameterize $v$ by a neural network; minimize HJB residual via backward SDE formulation. Scales to high-dimensional problems.

---

## 7. HJB for optimal stopping

Revisit optimal stopping: action is "stop or continue". HJB becomes:
$$
\min\bigl(-\partial_t v - \mathcal{L} v + r v,\; v - g\bigr) = 0.
$$

This is the **variational inequality** from Module 4.2, a specific form of HJB with binary action set.

### 7.1 Impulse control

Occasional discrete actions (e.g., rebalancing with fixed cost). HJB + quasi-variational inequality:
$$
\min\bigl(-\partial_t v - \mathcal{L} v,\; v - \mathcal{M} v\bigr) = 0,
$$
where $\mathcal{M} v(x) = \sup_\xi [v(x + \xi) - K(\xi)]$ is the impulse operator ($K$ = intervention cost).

---

## 8. HJB for ergodic / infinite-horizon control

For average-cost infinite-horizon control, HJB takes the form:
$$
\rho = \sup_a [f(x, a) + \mathcal{L}^a v(x)],
$$

where $\rho$ is the optimal long-run average cost, $v$ is the "relative value function". Unique $v$ up to constant.

### 8.1 Example: ergodic market making

Avellaneda–Stoikov (2008): market maker quotes bid/ask to maximize utility over long horizon; HJB on inventory state reduces to an ergodic equation with explicit solution.

---

## 9. Stochastic maximum principle (Pontryagin)

Alternative to HJB: characterize optimal control via a **backward SDE for the adjoint process** $Y_t$:

- **Adjoint equation:** $-dY_t = [\partial_x f(t, X, \alpha^*) + (\partial_x \mu) Y + (\partial_x \sigma) Z] dt - Z_t dB_t$, $Y_T = \partial_x g(X_T)$.
- **Hamiltonian:** $H(t, x, a, y, z) = f(t, x, a) + \mu(t, x, a) y + \sigma(t, x, a) z$.
- **Maximality:** $\alpha^*(t) = \arg\max_a H(t, X_t, a, Y_t, Z_t)$.

This is Pontryagin's principle in the stochastic setting.

**Relationship to HJB:** $Y_t = \partial_x v(t, X_t)$, $Z_t = \sigma \partial_{xx} v(t, X_t)$. So adjoints are the gradient of value function.

---

## 10. Python: HJB solution for Merton

```python
import numpy as np

# ---- Merton's consumption-investment problem with CRRA utility ----
T = 10.0
gamma = 3.0  # CRRA parameter
rho = 0.04  # discount
r = 0.03  # risk-free
mu = 0.08  # risky drift
sigma = 0.2  # risky vol

# Closed-form Merton: pi* = (mu - r)/(gamma sigma^2)
pi_star = (mu - r) / (gamma * sigma**2)
print(f"Merton optimal pi*: {pi_star:.4f}")

# Consumption rate: C* = kappa W, kappa satisfies
# kappa^(1/gamma) + (rho - (1-gamma)[r + 0.5*(mu-r)^2/(gamma*sigma^2)])/gamma = kappa
# For infinite horizon, kappa*^gamma = (rho - (1-gamma)*nu)/gamma where nu = r + 0.5*(mu-r)^2/(gamma*sigma^2)
nu_infty = r + 0.5 * (mu - r)**2 / (gamma * sigma**2)
# For existence of stationary, need rho > (1-gamma)*nu
denom = rho - (1 - gamma) * nu_infty
print(f"Admissibility: rho - (1-gamma)*nu = {denom:.5f} (need > 0)")
kappa_infty = denom / gamma
print(f"Infinite-horizon kappa* = {kappa_infty:.5f}")

# ---- Finite horizon: solve ODE for h(t) ----
# v(t,W) = h(t) W^(1-gamma)/(1-gamma)
# h solves: h' = -((1-gamma)*nu - rho)*h + gamma * h^(1 - 1/gamma)
# with h(T) = 0 (no bequest) -- but that's singular; use h(T) = epsilon.
# Terminal with B(W) = eps W^(1-gamma)/(1-gamma) ==> h(T) = eps.

# Numerical ODE solution backward
N = 1000
dt = T / N
h = np.zeros(N + 1)
h[-1] = 1e-6  # small terminal condition (no bequest)
for k in range(N - 1, -1, -1):
    # Backward Euler for stability
    A = -((1 - gamma) * nu_infty - rho)
    # h' = A h + gamma h^(1 - 1/gamma)
    # Euler backward
    h_guess = h[k + 1]
    for _ in range(50):
        f_val = A * h_guess + gamma * h_guess**(1 - 1/gamma) if h_guess > 0 else 0
        h_guess = h[k + 1] + dt * f_val
        if h_guess < 0: h_guess = 1e-10
    h[k] = h_guess

kappa_t = h ** (-1/gamma)
print(f"\nFinite-horizon Merton: kappa(0) = {kappa_t[0]:.4f}")
print(f"                      kappa(T/2) = {kappa_t[N//2]:.4f}")
print(f"                      kappa(T-eps) approaches infinity (consume all near end)")

# ---- Monte Carlo verification ----
M = 10000
dt_mc = 0.01
N_mc = int(T / dt_mc)
W_sim = np.zeros((M, N_mc + 1))
W_sim[:, 0] = 100.0
rng = np.random.default_rng(0)
consumption = np.zeros((M, N_mc))
for k in range(N_mc):
    t_curr = k * dt_mc
    t_idx = int(t_curr / dt * N)
    kappa_curr = kappa_t[min(t_idx, N)]
    C_t = kappa_curr * W_sim[:, k]
    dW_brownian = rng.standard_normal(M) * np.sqrt(dt_mc)
    W_sim[:, k + 1] = W_sim[:, k] + ((r + pi_star * (mu - r)) * W_sim[:, k] - C_t) * dt_mc + pi_star * sigma * W_sim[:, k] * dW_brownian
    consumption[:, k] = C_t
# Cumulative utility
u_C = np.power(np.maximum(consumption, 1e-10), 1 - gamma) / (1 - gamma)
discount = np.exp(-rho * np.arange(N_mc) * dt_mc)
J_mc = (u_C * discount).sum(axis=1) * dt_mc
print(f"\nMonte Carlo total utility: {J_mc.mean():.4f}")
# Value function at t=0, W=100
V_0 = h[0] * 100.0**(1 - gamma) / (1 - gamma)
print(f"Analytic v(0, 100) = h(0)*100^(1-gamma)/(1-gamma) = {V_0:.4f}")
```

---

## 11. [QUANT APPLICATION] — HJB in finance

### 11.1 Merton's problem (Module 4.5)

Closed-form HJB solution; foundational.

### 11.2 Optimal execution (Almgren–Chriss; Module 4.6)

Continuous-time liquidation with market impact: LQ-type HJB with closed form.

### 11.3 Market making (Avellaneda–Stoikov)

HJB on inventory: bid-ask quotes balance profit vs inventory risk. Reduces to a linear PDE under suitable ansatz.

### 11.4 Optimal hedging with transaction costs

Davis–Panas–Zariphopoulou (1993): HJB for hedging an option portfolio with proportional transaction costs. No-trade region in position space.

### 11.5 Portfolio choice with habit formation

Utility depends on consumption relative to habit: state-dependent HJB with two-dim state.

### 11.6 Optimal stopping + control (mixed)

Merton's portfolio with random time horizon / ruin: HJB with boundary at wealth = 0.

### 11.7 Robust / model-uncertain control

Maximize worst-case over models: **second-order HJB**, related to G-expectations.

---

## 12. Summary

- **HJB equation** generalizes Bellman's equation to continuous time and continuous state.
- Derivation via Itô + dynamic programming.
- **Verification theorem** confirms optimality of HJB solutions.
- **Viscosity solutions** provide rigorous interpretation when classical $C^{1, 2}$ solutions fail.
- **Numerical methods**: monotone FD, Markov chain approximation, semi-Lagrangian, deep learning.
- **Special forms**: variational inequality (stopping), QVI (impulse).
- **Stochastic maximum principle** is the dual approach.
- **Finance applications**: Merton, Almgren–Chriss, Avellaneda–Stoikov, hedging with TCs.

---

## 13. Exercises

**★ (warm-ups)**

**13.1** Derive HJB for a deterministic system $dx/dt = f(x, a)$. Show it reduces to the Hamilton–Jacobi equation from classical mechanics.

**13.2** For a linear state $dX = (AX + Ba) dt + C dB$ and quadratic cost $X^T Q X + a^T R a$, set up the HJB.

**13.3** For a one-state Merton with CARA utility $u(C) = -e^{-\alpha C}/\alpha$: solve HJB with ansatz $v(t, W) = -e^{-\alpha W h(t)}/\alpha$.

**13.4** Verify: HJB for optimal stopping $\min(-\partial_t v - \mathcal{L}v, v - g) = 0$.

**13.5** In discrete time, HJB reduces to the Bellman equation. Verify via $h \to 0$.

**★★ (core)**

**13.6** **(Merton with CRRA.)** Complete the derivation: solve the ODE for $h(t)$; give closed form for finite horizon.

**13.7** **(Merton with bequest.)** Introduce $B(W_T) = \varepsilon W_T^{1 - \gamma}/(1 - \gamma)$. Solve HJB and identify the effect on $C^*$.

**13.8** **(Optimal portfolio with random income.)** Add stochastic labor income $Y_t$ to Merton. Derive HJB, recognize non-closedform.

**13.9** **(Almgren–Chriss.)** For $dX = -ax dt + \sigma dB$ (liquidating $X$ shares with price impact $a$), minimize $E[\int (a x)^2 dt] + \lambda \int x^2 dt$. Derive HJB; get linear solution.

**13.10** **(Avellaneda–Stoikov.)** Set up HJB for market maker quoting bid/ask; inventory $q$; reservation price $S - q\gamma\sigma^2(T-t)$.

**13.11** **(HJB verification, example.)** For Merton's $v(t, W)$ explicitly, verify the HJB is satisfied.

**13.12** **(Viscosity solutions, heat equation.)** Verify that classical solutions of $u_t = u_{xx}$ are viscosity solutions.

**13.13** **(Comparison principle.)** State and sketch the proof of the comparison principle for first-order HJ equations.

**13.14** **(Barles–Souganidis convergence.)** State the monotone + consistent + stable theorem for numerical schemes.

**13.15** **(Policy iteration in continuous time.)** For control-affine systems, policy iteration is: pick $\alpha$, solve linear PDE $\partial_t v + \mathcal{L}^\alpha v + f = 0$, improve. Explain convergence.

**★★★ (research / quant)**

**13.16** **(Optimal investment under stochastic vol.)** For Heston volatility, HJB is 2D. Derive via Fleming–Hernández and give quasi-closed-form.

**13.17** **(Merton with transaction costs.)** Davis–Norman three-region structure (buy/hold/sell). Implement numerically.

**13.18** **(Robust control via G-expectations.)** Peng's G-Brownian motion; HJB becomes a Hamilton–Jacobi–Bellman–Isaacs (HJBI) with $\sup\inf$.

**13.19** **(BSDE approach to HJB.)** For semilinear HJB, $v(t, X_t) = Y_t$ for a BSDE with generator related to Hamiltonian. Implement via deep BSDE (Han–Jentzen–E 2018).

**13.20** **(Optimal liquidation with market impact.)** Almgren–Chriss full model with permanent + temporary impact; solve HJB for LQ-Gaussian case.

**13.21** **(Pontryagin for jump diffusions.)** Extend stochastic maximum principle to jump processes. Identify the jump term in the adjoint BSDE.

**13.22** **(Ergodic HJB for pairs trading.)** Cointegrated spread $Z_t$; maximize long-run utility of trading. Derive ergodic HJB; solve for optimal buy/sell thresholds.

---

## 14. Forward pointers

- **Module 4.5 (Merton)**: detailed solution of HJB for Merton's portfolio-consumption problem.
- **Module 4.6 (LQG)**: linear-quadratic case — HJB admits closed-form.
- **Module 4.7 (RL)**: approximate HJB via sampling.

**Next module:** Merton's portfolio problem — the canonical HJB solution in continuous-time finance.

---

*End of Module 4.4.*

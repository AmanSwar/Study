# Subject 4, Module 6: LQG Control and the Kalman-Bucy Filter

> *"The linear-quadratic-Gaussian problem is to optimal control what the Black-Scholes model is to derivatives pricing: a special case so tractable, and so generative of intuition, that understanding it deeply is more useful than knowing ten weaker general results."*

## Prerequisites

- **Module 3.1** (Brownian motion), **Module 3.4** (linear SDEs, OU process).
- **Module 4.3** (DP, Bellman).
- **Module 4.4** (HJB, verification theorem).
- **Module 4.5** (the spirit of "quadratic cost + linear dynamics + Gaussian noise" generalizes Merton).
- Helpful: linear algebra (matrix eigenvalues, Lyapunov and Riccati equations), basic control theory.

---

## 4.6.0 Why LQG?

In the 1950s Kalman, Bucy, Bellman, and Athans worked out that a very specific class of control problems — **linear** dynamics, **quadratic** cost, **Gaussian** noise — admits **fully explicit** solutions via matrix Riccati ODEs. The breakthrough is that even though LQG sounds like a toy, three deep theorems make it the backbone of applied stochastic control:

1. **Closed-form via Riccati.** The value function is quadratic in state, with coefficient a symmetric positive-definite matrix solving a matrix Riccati ODE. The optimal policy is *linear* in state.

2. **Separation principle.** When state is partially observed and observation noise is Gaussian, the optimal controller is: (a) **estimate** the state via Kalman filter, then (b) use the LQG controller as if the estimate were the true state. Estimation and control decouple.

3. **Universality through linearization.** Any smooth control problem linearized around its optimum becomes locally LQG. So LQG theory provides the *local* structure of every nonlinear control problem — in particular, second-order approximations used in differential dynamic programming (DDP) and iLQR.

Applications in quant finance and beyond:

- **Optimal execution** (Almgren-Chriss) is pure LQ.
- **Portfolio choice with mean-reverting returns** (Kim-Omberg) reduces to LQG under quadratic approximation.
- **Stochastic volatility portfolio problems** often have LQG inner layers.
- **Signal-based alpha** uses Kalman filtering to extract tradeable signals from noisy observations.
- **Robot control**, **autopilot**, **missile guidance** are canonical LQG applications.
- **Recursive least squares** and **adaptive filtering** are special cases of Kalman.

Roadmap:
- **4.6.1** The deterministic LQ problem.
- **4.6.2** Stochastic LQ (LQG state-feedback).
- **4.6.3** The matrix Riccati equation.
- **4.6.4** Infinite-horizon LQR and algebraic Riccati.
- **4.6.5** Discrete-time LQ and DLQR.
- **4.6.6** Kalman-Bucy filter derivation.
- **4.6.7** Separation principle.
- **4.6.8** Almgren-Chriss optimal execution.
- **4.6.9** Python implementations and quant applications.

---

## 4.6.1 The Deterministic LQ Problem

### Problem

State $X_t \in \mathbb{R}^n$, control $u_t \in \mathbb{R}^m$. Linear dynamics:
$$\dot X_t = A X_t + B u_t, \qquad X_0 = x.$$

Quadratic cost:
$$J(x; u) = \int_0^T \left( X_t^\top Q X_t + u_t^\top R u_t \right) dt + X_T^\top F X_T,$$
with $Q \succeq 0, R \succ 0, F \succeq 0$ (state penalties PSD, control penalty PD, terminal penalty PSD).

Goal: find $u^*(\cdot)$ minimizing $J$.

### HJB

Value function $v(t, x) = \inf_u J$. HJB:
$$\partial_t v + \inf_u \left\{ x^\top Q x + u^\top R u + (A x + B u)^\top \nabla_x v \right\} = 0, \quad v(T, x) = x^\top F x.$$

**Ansatz:** $v(t, x) = x^\top P(t) x$ with $P(t)$ symmetric. Then $\nabla_x v = 2 P x$ and $\partial_t v = x^\top \dot P x$.

**Pointwise min over $u$:** $\partial_u [u^\top R u + 2 u^\top B^\top P x] = 2 R u + 2 B^\top P x = 0$, so
$$\boxed{u^*(t, x) = -R^{-1} B^\top P(t) x \equiv -K(t) x.}$$

Optimal control is **linear state feedback** with gain $K(t) = R^{-1} B^\top P(t)$.

### Riccati ODE

Substitute $u^*$ back into HJB:
$$x^\top \dot P x + x^\top Q x + (-R^{-1} B^\top P x)^\top R (-R^{-1} B^\top P x) + (Ax - B R^{-1} B^\top P x)^\top (2 P x) = 0.$$

Simplify:
- $u^{*\top} R u^* = x^\top P B R^{-1} B^\top P x$.
- $(Ax - B u^*)^\top (2 P x) = 2 x^\top A^\top P x - 2 x^\top P B R^{-1} B^\top P x$.

Sum:
$$x^\top [\dot P + Q + P B R^{-1} B^\top P + A^\top P + P A - 2 P B R^{-1} B^\top P] x = 0.$$

Using $P B R^{-1} B^\top P - 2 P B R^{-1} B^\top P = -P B R^{-1} B^\top P$, and symmetrizing $A^\top P + PA$:

$$\boxed{\dot P(t) + A^\top P(t) + P(t) A - P(t) B R^{-1} B^\top P(t) + Q = 0, \quad P(T) = F.}$$

This is the **(differential) matrix Riccati equation** for LQ control. It is a matrix ODE with quadratic nonlinearity.

### Theorem (LQR existence/uniqueness)

If $(A, B)$ is **stabilizable** and $(A, Q^{1/2})$ is **detectable**, then the Riccati ODE has a unique symmetric PSD solution on $[0, T]$, the optimal value $v(0, x) = x^\top P(0) x$ is finite, and the feedback control $u^* = -R^{-1} B^\top P(t) x$ is optimal.

The detectability and stabilizability conditions are classical control-theory conditions ensuring the system can be driven to zero (stabilizable) and that the state contributes observably to the cost (detectable).

### Worked example: scalar LQ

Take $n = m = 1$, $A = a, B = b, Q = q, R = r_0$, $F = f$. The Riccati becomes
$$\dot P + 2 a P - \dfrac{b^2}{r_0} P^2 + q = 0, \quad P(T) = f.$$

This is a scalar Bernoulli-type ODE (Riccati in the classical scalar sense). Substitute $P = -r_0 \dot \phi / (b^2 \phi)$ to convert to a second-order linear ODE for $\phi$ with constant coefficients; solve explicitly. Alternatively, for the **infinite-horizon** case $\dot P = 0$: algebraic Riccati gives
$$\boxed{P_\infty = \dfrac{a r_0 + \sqrt{a^2 r_0^2 + q b^2 r_0}}{b^2 / 1}} ... \text{let me redo.}$$

Algebraic version with $\dot P = 0$:
$$-\dfrac{b^2}{r_0} P^2 + 2 a P + q = 0 \ \Longrightarrow \ P = \dfrac{2 a \pm \sqrt{4 a^2 + 4 q b^2 / r_0}}{2 b^2 / r_0} = \dfrac{r_0}{b^2}\left(a \pm \sqrt{a^2 + q b^2/r_0}\right).$$

Take the positive root. Optimal gain: $K_\infty = b P_\infty / r_0 = a/1 \pm \sqrt{a^2 + q b^2/r_0}$, ensuring $A - B K_\infty < 0$ (closed-loop stability).

---

## 4.6.2 Stochastic LQ (LQG State-Feedback)

Now add Gaussian noise to the dynamics:
$$dX_t = (A X_t + B u_t) \, dt + \Sigma \, dB_t, \quad X_0 = x,$$
with $B_t \in \mathbb{R}^d$ Brownian, $\Sigma \in \mathbb{R}^{n \times d}$.

Cost (unchanged):
$$J(x; u) = \mathbb{E}\left[ \int_0^T (X_t^\top Q X_t + u_t^\top R u_t) dt + X_T^\top F X_T \right].$$

### HJB

$$\partial_t v + \inf_u \left\{ x^\top Q x + u^\top R u + (Ax + Bu)^\top \nabla_x v + \tfrac{1}{2} \text{tr}(\Sigma \Sigma^\top \nabla^2_x v) \right\} = 0.$$

Ansatz $v(t, x) = x^\top P(t) x + q(t)$ (with an additive function of $t$ to absorb the trace term).

Same optimization as in deterministic case gives $u^* = -R^{-1} B^\top P(t) x$, and substituting yields:

$$\dot P + A^\top P + PA - P B R^{-1} B^\top P + Q = 0 \quad \text{(same as deterministic!)}$$
$$\dot q + \text{tr}(\Sigma \Sigma^\top P(t)) = 0, \quad q(T) = 0.$$

So
$$q(t) = \int_t^T \text{tr}(\Sigma \Sigma^\top P(s)) \, ds.$$

The **optimal feedback law is identical to the deterministic case** — this is a manifestation of the **certainty equivalence principle**: for LQ control with additive Gaussian noise, the optimal control depends only on the mean of the state, not on its covariance. The noise only affects the *value* (via $q(t)$), not the *policy*.

### Theorem (certainty equivalence)

For LQG state-feedback problems, the optimal control is the same as the deterministic LQ optimal control evaluated at the current state. Noise enters the value function only through the additive $q(t)$ term.

This is a huge simplification. In non-quadratic or non-linear problems certainty equivalence fails, and noise influences the policy.

---

## 4.6.3 The Matrix Riccati Equation

### Structure

$$\dot P = -A^\top P - P A + P B R^{-1} B^\top P - Q.$$

Symmetry-preserving: if $P(T) = F$ symmetric, then $P(t)$ is symmetric for all $t$.

Positive-semidefiniteness preserving: if $P(T) \succeq 0, Q \succeq 0, R \succ 0$, then $P(t) \succeq 0$ for all $t \le T$.

### Solution via matrix linearization

The quadratic Riccati has a classical trick: introduce the Hamiltonian matrix
$$H = \begin{pmatrix} A & -B R^{-1} B^\top \\ -Q & -A^\top \end{pmatrix} \in \mathbb{R}^{2n \times 2n}.$$

Linear system:
$$\dfrac{d}{dt} \begin{pmatrix} X \\ Y \end{pmatrix} = H \begin{pmatrix} X \\ Y \end{pmatrix}.$$

Then $P = Y X^{-1}$ solves the Riccati. Explicitly, solve the linear ODE via matrix exponential:
$$\begin{pmatrix} X(t) \\ Y(t) \end{pmatrix} = e^{H(t-T)} \begin{pmatrix} I \\ F \end{pmatrix}.$$

This gives a semi-explicit formula for $P(t)$ via matrix exponentials and inversions — fundamental in numerical LQG solvers.

### Chandrasekhar form

For sparse or large-scale problems, the Chandrasekhar equations reduce the Riccati ODE to a system of lower-rank ODEs, often more numerically stable.

---

## 4.6.4 Infinite-Horizon LQR

### Setup

$$J_\infty(x; u) = \mathbb{E}\left[ \int_0^\infty (X_t^\top Q X_t + u_t^\top R u_t) dt \right], \quad dX = (AX + Bu)dt + \Sigma dB.$$

With the ergodic setup, stationary policies are optimal. Set $\dot P = 0$:

$$\boxed{A^\top P + P A - P B R^{-1} B^\top P + Q = 0.}$$

This is the **algebraic Riccati equation (ARE)**. 

**Theorem:** If $(A, B)$ is stabilizable and $(A, Q^{1/2})$ is detectable, the ARE has a unique PSD solution $P_\infty$ with $A - B R^{-1} B^\top P_\infty$ stable (Hurwitz, eigenvalues with negative real parts).

Optimal feedback: $u^* = -K x$ with $K = R^{-1} B^\top P_\infty$.

Optimal cost (for deterministic):
$$V_\infty(x) = x^\top P_\infty x.$$

Stochastic adds the "steady-state noise cost":
$$V_\infty^{\text{stoch}}(x) = x^\top P_\infty x + \dfrac{\text{tr}(\Sigma \Sigma^\top P_\infty)}{1}.$$

Wait — infinite-horizon stochastic has divergent integral unless we use **average cost** formulation:
$$J^{\text{avg}}(u) = \limsup_{T \to \infty} \dfrac{1}{T} \mathbb{E}\left[ \int_0^T (X^\top Q X + u^\top R u) dt \right].$$

Under the optimal stationary policy, the closed-loop system has a Gaussian stationary distribution $\mathcal{N}(0, \Xi)$ where $\Xi$ solves the Lyapunov equation:
$$(A - BK) \Xi + \Xi (A - BK)^\top + \Sigma \Sigma^\top = 0.$$

Steady-state average cost:
$$J^{\text{avg}} = \text{tr}(Q \Xi) + \text{tr}(K^\top R K \Xi) = \text{tr}(\Sigma \Sigma^\top P_\infty).$$

### Solving the ARE

**Numerical methods:**
1. **Schur method** (Laub 1979): eigendecomposition of Hamiltonian $H$.
2. **Newton iteration:** $P_{k+1}$ from $P_k$ by Kleinman iteration, where each step is a Lyapunov equation.
3. **Doubling algorithm:** $O(\log N)$ convergence.

In Python: `scipy.linalg.solve_continuous_are(A, B, Q, R)`.

---

## 4.6.5 Discrete-Time LQ

### Problem

$$X_{k+1} = A X_k + B u_k + w_k, \quad w_k \sim \mathcal{N}(0, \Sigma_w) \text{ iid},$$
$$J = \mathbb{E}\left[ \sum_{k=0}^{N-1} (X_k^\top Q X_k + u_k^\top R u_k) + X_N^\top F X_N \right].$$

### Discrete Riccati

$$P_k = A^\top P_{k+1} A - A^\top P_{k+1} B (R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A + Q,$$
with $P_N = F$.

Optimal control:
$$u_k^* = -(R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A X_k.$$

Value function: $v_k(x) = x^\top P_k x + c_k$ with $c_k = \sum_{j=k+1}^{N-1} \text{tr}(\Sigma_w P_j)$.

### Discrete Algebraic Riccati (DARE)

Set $P_{k+1} = P_k = P$:
$$P = A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A + Q.$$

Scipy: `scipy.linalg.solve_discrete_are(A, B, Q, R)`.

---

## 4.6.6 The Kalman-Bucy Filter

### Partial-observation problem

State: $dX_t = (A X_t + B u_t) dt + \Sigma \, dB_t$.

Observation: we see not $X_t$ directly but a noisy linear function:
$$dZ_t = C X_t \, dt + D \, dW_t,$$
with $W_t$ another Brownian (independent of $B_t$), $C \in \mathbb{R}^{p \times n}$, $D \in \mathbb{R}^{p \times p}$ invertible.

Equivalently, $Y_t = \int_0^t dZ_s = C \int X_s ds + D W_t$ — observation integral.

### The filtering problem

Given observations $\mathcal{Y}_t = \sigma(Z_s : s \le t)$, compute $\hat X_t := \mathbb{E}[X_t | \mathcal{Y}_t]$ — the best linear estimate of $X_t$.

**Theorem (Kalman-Bucy, 1961):** Given Gaussian initial condition $X_0 \sim \mathcal{N}(\hat X_0, \Pi_0)$ and jointly Gaussian dynamics + observations, the conditional distribution is Gaussian $X_t | \mathcal{Y}_t \sim \mathcal{N}(\hat X_t, \Pi_t)$ with:

**Innovations SDE for the mean:**
$$\boxed{d\hat X_t = (A \hat X_t + B u_t) \, dt + \Pi_t C^\top (D D^\top)^{-1} (dZ_t - C \hat X_t \, dt).}$$

**Matrix Riccati ODE for the covariance:**
$$\boxed{\dot \Pi_t = A \Pi_t + \Pi_t A^\top + \Sigma \Sigma^\top - \Pi_t C^\top (D D^\top)^{-1} C \Pi_t.}$$

### Proof sketch

The proof uses the **innovation process**
$$dI_t := dZ_t - C \hat X_t \, dt.$$

Key properties:
1. $I_t$ is a Brownian motion (rescaled by $D$) with respect to $\mathcal{Y}_t$.
2. $I_t$ is independent of the past of $\hat X$ and of $\Pi$.
3. The "new information" in $Z$ past time $t$ is captured in $I_t$.

The filter equation says: "predict" via dynamics ($A \hat X + Bu$) then "correct" using the innovation weighted by the **Kalman gain** $K_t := \Pi_t C^\top (DD^\top)^{-1}$.

### Duality with LQR

Observation: the Kalman-Bucy Riccati ODE for $\Pi_t$ is formally **identical** to the LQR Riccati after a substitution:
$$\text{LQR: } \dot P = -A^\top P - PA + P B R^{-1} B^\top P - Q.$$
$$\text{Kalman: } \dot \Pi = A \Pi + \Pi A^\top - \Pi C^\top (DD^\top)^{-1} C \Pi + \Sigma \Sigma^\top.$$

Substitute $A \leftrightarrow A^\top$, $B R^{-1} B^\top \leftrightarrow C^\top (DD^\top)^{-1} C$, $Q \leftrightarrow \Sigma\Sigma^\top$, and reverse time. This is the **LQR-Kalman duality**: control and filtering are dual problems.

Consequence: software that solves LQR also solves Kalman filter covariance (flip signs and transpose).

### Infinite-horizon Kalman

If $(A, \Sigma)$ is stabilizable and $(A, C)$ is detectable, the Riccati ODE converges to a unique PSD steady-state covariance $\Pi_\infty$ solving
$$A \Pi_\infty + \Pi_\infty A^\top - \Pi_\infty C^\top (DD^\top)^{-1} C \Pi_\infty + \Sigma \Sigma^\top = 0.$$

The steady-state filter is a linear system with gain $K_\infty$.

---

## 4.6.7 The Separation Principle

### Setup (LQG with partial observation)

$$dX_t = (A X_t + B u_t) dt + \Sigma dB_t, \quad dZ_t = C X_t \, dt + D \, dW_t.$$

Cost:
$$J = \mathbb{E}\left[ \int_0^T (X^\top Q X + u^\top R u) dt + X_T^\top F X_T \right],$$
where $u_t$ must be $\mathcal{Y}_t$-adapted (based only on observed $Z$).

### Theorem (Separation Principle)

The optimal control is
$$\boxed{u^*_t = -K(t) \hat X_t,}$$
where:
- $K(t) = R^{-1} B^\top P(t)$ is the LQR feedback gain (as if $X$ were observable),
- $\hat X_t$ is the Kalman-Bucy estimate.

Furthermore, the optimal cost is
$$V(x, \Pi_0) = x^\top P(0) x + \int_0^T \text{tr}(\Sigma \Sigma^\top P(s)) ds + \int_0^T \text{tr}(P(s) (dE_s / ds)) ds,$$

where the last integral involves the estimation error covariance evolution... concisely:
$$V = \text{LQR cost} + \text{tr}(P_0 \Pi_0) + \text{extra terms involving } \Pi_t.$$

### Proof idea

Use the decomposition $X_t = \hat X_t + \tilde X_t$ where $\tilde X_t$ is the estimation error (Gaussian, $\mathcal{Y}_t$-independent conditional on initial distribution). Then $\mathbb{E}[X_t^\top Q X_t | \mathcal{Y}_t] = \hat X_t^\top Q \hat X_t + \text{tr}(Q \Pi_t)$. Since $\Pi_t$ is deterministic (does not depend on controls in LQG), the control optimization reduces to the LQR problem for $\hat X_t$ — and $\hat X_t$ follows LQ dynamics.

### Significance

The separation principle has enormous practical importance. It says:

1. You can design the **estimator** (Kalman filter) without knowing the controller.
2. You can design the **controller** (LQR) assuming full observation.
3. Plugging in $\hat X$ for $X$ gives the optimal LQG output-feedback controller.

This decouples two hard problems into two easy ones.

**Caveat:** Separation holds exactly only for LQG. For non-linear or non-Gaussian problems, it fails — the estimator and controller are entangled (dual control problem; see Feldbaum, Bellman 1961).

---

## 4.6.8 Almgren-Chriss Optimal Execution

### The problem

Trader needs to liquidate $X$ shares over $[0, T]$. Trade rate $v_t = -\dot X_t$. Price impact: selling rate $v_t$ depresses price by $\eta v_t$ temporary + $\gamma \int_0^t v_s ds$ permanent. Reference price $P_t = P_0 - \gamma (X_0 - X_t) + \sigma B_t$ (arithmetic Brownian reference).

Realized price: $\tilde P_t = P_t - \eta v_t$. Execution cost over $[0,T]$:
$$\text{cost} = -\int_0^T \tilde P_t v_t dt + X_0 P_0 = \text{(drift)} + \eta \int_0^T v_t^2 dt + \sigma \int \cdots dB_t.$$

Objective: minimize expected cost **plus $\lambda$ × variance** of cost (mean-variance):
$$\min \mathbb{E}[\text{cost}] + \lambda \, \text{Var}[\text{cost}].$$

### LQ formulation

State $X_t$, control $v_t$. Dynamics $dX = -v \, dt$, deterministic! But the cost has Brownian term:
- Expected cost: $\eta \int v^2 dt + \tfrac{1}{2} \gamma X_0^2$ (permanent impact integral).
- Variance of cost: $\sigma^2 \int X_t^2 dt$ (from holding inventory during the Brownian-driven price moves).

So mean-variance cost:
$$\min_v \int_0^T \left( \eta v_t^2 + \lambda \sigma^2 X_t^2 \right) dt.$$

Plus boundary: $X_0 = X_\text{start}, X_T = 0$.

This is a **pure deterministic LQ problem** (stochasticity enters only through variance of cost) with state penalty $\lambda \sigma^2 X^2$, control penalty $\eta v^2$. The Euler-Lagrange equation is:
$$\eta \ddot X_t = \lambda \sigma^2 X_t,$$
with BC $X_0$ and $X_T = 0$. Solution: hyperbolic cosine / sinh:
$$X_t^* = X_0 \dfrac{\sinh(\kappa (T-t))}{\sinh(\kappa T)}, \quad \kappa = \sqrt{\lambda \sigma^2 / \eta}.$$

Trade rate:
$$v_t^* = X_0 \kappa \dfrac{\cosh(\kappa(T-t))}{\sinh(\kappa T)}.$$

### Interpretation

- **Risk neutral** ($\lambda = 0$): $\kappa = 0$, $X_t^* = X_0 (T-t)/T$, **TWAP** (uniform trading).
- **Risk averse** ($\lambda \to \infty$): $\kappa \to \infty$, almost all trades at $t = 0$ (front-load to eliminate inventory risk).

The Almgren-Chriss trajectory is used in essentially every bank's execution algo, parameterized by a "risk aversion" knob.

### Generalizations

- **Stochastic volatility:** $\sigma \to \sigma_t$. HJB still solvable with quadratic-in-$X$ ansatz and an ODE for the time-varying coefficient.
- **Drift/signal:** add $\alpha_t$ to the price dynamics. Optimal trajectory shifts to buy when $\alpha > 0$ and sell faster when $\alpha < 0$.
- **Multiple assets, correlated:** full matrix LQ.
- **Market-maker extension** (Avellaneda-Stoikov): dual problem of quoting bid/ask to maximize expected P&L under inventory risk; solvable via HJB with quadratic structure.

---

## 4.6.9 Python: Solving LQG and Kalman

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are, expm

# -----------------------------
# Example 1: Scalar LQR
# -----------------------------
A = np.array([[1.0]])
B = np.array([[1.0]])
Q = np.array([[1.0]])
R = np.array([[1.0]])

P_inf = solve_continuous_are(A, B, Q, R)
K_inf = np.linalg.solve(R, B.T @ P_inf)
print(f"Scalar LQR: P_inf = {P_inf[0,0]:.4f}, K_inf = {K_inf[0,0]:.4f}")
# Analytical: P = 1 + sqrt(2), K = sqrt(2) approx 1.414

# Simulate closed loop
T = 10.0; dt = 0.01; n = int(T/dt)
x = np.zeros(n+1); x[0] = 1.0
t = np.arange(n+1) * dt
for i in range(n):
    u = -K_inf @ np.array([x[i]])
    x[i+1] = x[i] + (A[0,0] * x[i] + B[0,0] * u) * dt

plt.figure(figsize=(8,4))
plt.plot(t, x)
plt.xlabel('t'); plt.ylabel('x')
plt.title('LQR stabilization'); plt.grid(alpha=0.3); plt.show()

# -----------------------------
# Example 2: 2D LQG with noise
# -----------------------------
A = np.array([[0., 1.], [-1., 0.]])    # harmonic oscillator
B = np.array([[0.], [1.]])              # control on velocity
Sigma = np.array([[0.1, 0.], [0., 0.1]])
Q = np.eye(2); R = np.array([[1.]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)
print(f"\n2D LQR gain K = {K}")

# MC simulation
np.random.seed(0)
N_paths = 500
T = 10.0; dt = 0.01; n = int(T/dt)
X = np.random.randn(N_paths, 2) * 0.5  # random initial conditions
cost = np.zeros(N_paths)
for step in range(n):
    U = -X @ K.T
    cost += (np.sum(X * (X @ Q.T), axis=1) + np.sum(U * (U @ R.T), axis=1)) * dt
    dB = np.sqrt(dt) * np.random.randn(N_paths, 2)
    X = X + (X @ A.T + U @ B.T) * dt + dB @ Sigma.T

print(f"Average realized cost: {cost.mean():.4f}")

# -----------------------------
# Example 3: Kalman-Bucy filter
# -----------------------------
# True state: 1D OU process
a = -0.5; sigma = 0.3
c = 1.0; d = 0.5       # observation: noisy measurement of x
x_true = 0.0
x_hat  = 0.0
Pi     = 1.0           # initial posterior variance

# Infinite-horizon filter covariance
Pi_inf = solve_continuous_are(
    A=np.array([[a]]).T, B=np.array([[c]]).T,
    Q=np.array([[sigma**2]]), R=np.array([[d**2]])
)[0,0]
K_inf  = Pi_inf * c / d**2
print(f"\nKalman: Pi_inf = {Pi_inf:.4f}, K_inf = {K_inf:.4f}")

T = 20.0; dt = 0.01; n = int(T/dt)
t = np.arange(n+1)*dt
X_true = np.zeros(n+1); X_hat = np.zeros(n+1); Pi_t = np.zeros(n+1)
X_true[0] = 0; X_hat[0] = 1.0; Pi_t[0] = 1.0

for i in range(n):
    # True dynamics
    dB = np.sqrt(dt) * np.random.randn()
    X_true[i+1] = X_true[i] + a * X_true[i] * dt + sigma * dB
    # Observation
    dW = np.sqrt(dt) * np.random.randn()
    dZ = c * X_true[i] * dt + d * dW
    # Kalman update
    K_t = Pi_t[i] * c / d**2
    X_hat[i+1] = X_hat[i] + a * X_hat[i] * dt + K_t * (dZ - c * X_hat[i] * dt)
    Pi_t[i+1] = Pi_t[i] + (2*a*Pi_t[i] + sigma**2 - Pi_t[i]**2 * c**2 / d**2) * dt

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(t, X_true, label='true', alpha=0.7)
ax[0].plot(t, X_hat, label='Kalman estimate')
ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title('Kalman-Bucy filter')
ax[1].plot(t, Pi_t)
ax[1].axhline(Pi_inf, color='red', linestyle='--', label=f'Pi_inf={Pi_inf:.3f}')
ax[1].set_xlabel('t'); ax[1].set_ylabel('posterior variance')
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# -----------------------------
# Example 4: Almgren-Chriss execution
# -----------------------------
X0 = 100_000      # shares to sell
T  = 1.0          # one day
eta = 2e-6        # temp impact
sigma_p = 0.3     # price vol
lam = 1e-6        # risk aversion

kappa = np.sqrt(lam * sigma_p**2 / eta)
t_grid = np.linspace(0, T, 100)
X_opt = X0 * np.sinh(kappa * (T - t_grid)) / np.sinh(kappa * T)
v_opt = X0 * kappa * np.cosh(kappa * (T - t_grid)) / np.sinh(kappa * T)

# TWAP for comparison
X_twap = X0 * (1 - t_grid / T)

plt.figure(figsize=(10, 5))
plt.plot(t_grid, X_opt, label=f'Almgren-Chriss (kappa={kappa:.2f})')
plt.plot(t_grid, X_twap, '--', label='TWAP (lambda=0)')
plt.xlabel('t (days)'); plt.ylabel('shares remaining')
plt.legend(); plt.grid(alpha=0.3); plt.title('Optimal execution')
plt.show()
```

---

## 4.6.10 [QUANT APPLICATION] LQG in Quant Finance

1. **Optimal execution (Almgren-Chriss and descendants).** The canonical LQ problem in trading. Every sell-side algo desk implements variants.

2. **Market making (Avellaneda-Stoikov).** Dual LQG — set quote prices given inventory penalty. Used in HFT market-maker code.

3. **Kalman filter for signal extraction.** Noisy alpha signals, momentum indicators, order-flow features are filtered with Kalman to extract the "true" underlying signal.

4. **State-space equity factor models.** Risk factors following linear Gaussian dynamics with noisy observations of returns → Kalman-filter estimation (Fama-MacBeth extensions).

5. **Pairs trading.** Model spread as OU: $dS_t = -\lambda(S_t - \mu) dt + \sigma dB_t$. LQ optimal trading rule: $v_t = K(S_t - \mu)$ with $K$ from Riccati.

6. **Interest rate model calibration.** Gaussian HJM / Hull-White parameters are estimated via Kalman filter on yield curve observations.

7. **Portfolio optimization with predictable returns.** Kim-Omberg model's AR(1) risk premium leads to LQG-like HJB under quadratic approximation of power utility.

8. **High-frequency inventory management.** Designate cash & inventory as state, LQ cost of holding inventory → optimal rebalancing rates.

9. **Latency-jitter hedging.** Trading systems with measurement delays fit LQG with observation delays; the delay Kalman filter extends standard Kalman.

10. **Machine-learning-based control.** Modern deep RL approaches often linearize around a baseline LQG controller (iLQR) and then add nonlinear corrections.

---

## 4.6.11 Summary

- LQ problems have **closed-form quadratic value functions** and **linear feedback policies**, determined by the matrix **Riccati ODE** (finite horizon) or **ARE** (infinite horizon).
- **LQG** (adding Gaussian noise) satisfies **certainty equivalence**: optimal policy unchanged from deterministic LQ.
- **Kalman-Bucy filter** gives optimal linear Gaussian state estimation via a Riccati ODE for the error covariance.
- **LQR-Kalman duality**: control and estimation are dual problems; same math, different sign conventions.
- **Separation principle**: optimal LQG output-feedback = Kalman filter + LQR of the estimate.
- **Almgren-Chriss** optimal execution is the canonical quant-finance LQ application.

### Forward pointers

- **Module 4.7 (RL / ADP)** will tackle the case where model parameters $(A, B, Q, R)$ are unknown or the problem is non-LQG. LQR-based warm starts and iLQR / DDP are heavily used.
- **Subject 5 (Asset pricing)** will use Kalman filtering for estimating latent factors and stochastic discount factor.
- **Subject 8 (Risk management)** uses Kalman filter for online estimation of volatility and correlations.

---

## Exercises

### Tier 1 (★) — Computation

1. Solve the scalar ARE $2aP - b^2 P^2 / r + q = 0$ for $a = -1, b = 1, q = 1, r = 1$.
2. For the 2D system $A = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}, B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, Q = I, R = 1$, numerically compute the optimal gain $K$.
3. Derive Almgren-Chriss explicit formula from the Euler-Lagrange equation.
4. Verify that the Kalman gain $K_\infty = \Pi_\infty C^\top (DD^\top)^{-1}$ satisfies a scalar form of ARE.
5. For a random walk $dX = \sigma dB$ with noisy observations $dZ = X dt + dW$, compute the steady-state filter variance.
6. Compute the average cost under LQR with noise: $J^{avg} = \text{tr}(\Sigma \Sigma^\top P_\infty)$.

### Tier 2 (★★) — Derivations and proofs

7. Prove the certainty equivalence principle for LQG by computing $\mathbb{E}[X^\top Q X | \mathcal{Y}] = \hat X^\top Q \hat X + \text{tr}(Q \Pi)$.
8. Prove the LQR-Kalman duality: show the Kalman Riccati is the LQR Riccati with time reversed and appropriate substitutions.
9. Derive the innovations representation: $Z_t = C \int \hat X_s ds + (\text{Brownian})$.
10. Show that for the infinite-horizon LQR, the closed-loop matrix $A - BR^{-1}B^\top P_\infty$ is Hurwitz.
11. Prove the separation principle in full: given a Kalman filter and LQR controller, the combined output-feedback controller is optimal for LQG.
12. Derive the information (inverse) form of Kalman filter and explain when it's computationally preferable.
13. Extend Kalman-Bucy to correlated observation and process noise: $\mathbb{E}[dB dW^\top] = S dt \ne 0$.

### Tier 3 (★★★) — Open-ended

14. Implement iLQR (iterative LQR) for a pendulum swing-up problem. Linearize around the current trajectory, solve LQR, update, iterate. Compare with direct trajectory optimization.
15. Implement optimal execution for a multi-asset portfolio with cross-impact matrix $\Lambda$. Solve the Riccati ODE and compare with single-asset Almgren-Chriss.
16. Implement an Avellaneda-Stoikov market maker. Derive the HJB for inventory-aware quoting and solve numerically.
17. Prove existence-uniqueness of Riccati solutions under stabilizability + detectability using Kleinman iteration.
18. Derive the Kalman filter for **non-Gaussian** heavy-tailed noise using Variational Bayes approximation. Compare with particle filter on a simple example.
19. Implement an **extended Kalman filter** (EKF) for a nonlinear SDE with Gaussian noise. Test on a Van der Pol oscillator with sparse observations.
20. For a cointegration model (pairs trading), derive the optimal trading strategy as LQG: states are prices, observations are prices + noise, control is portfolio weight. Compare with simple Z-score rules.
21. **Dual control / Bayesian adaptive control:** when $A, B$ are unknown, show that the optimal controller is no longer certainty-equivalent — the "probing" effect leads to exploration-exploitation tradeoffs. Implement on a scalar LQ with unknown $A$.
22. Derive and implement the **Linear-Exponential-Quadratic-Gaussian (LEQG) / risk-sensitive** control, replacing $\min \mathbb{E}[J]$ with $\min (1/\theta) \ln \mathbb{E}[\exp(\theta J)]$. Show how Riccati generalizes.

---

*Next module:* Reinforcement learning and approximate dynamic programming — solving control problems when the model is unknown.

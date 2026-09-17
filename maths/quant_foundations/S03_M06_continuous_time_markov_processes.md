# Module 3.6 — Continuous-Time Markov Processes: Semigroups, Generators, and Feller Theory

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 3 (Stochastic Processes), Module 6 of 7

---

## Prerequisites

- **Module 2.7** (Discrete-time Markov chains, transition matrices, ergodic theorem).
- **Module 3.1–3.5** (Brownian motion, Itô calculus, SDEs, Lévy processes).
- **Module 1.5** (Hilbert / Banach spaces, bounded linear operators).
- Elementary functional analysis (bounded linear operators, closed operators, spectrum).

This module establishes the **semigroup-generator calculus** for Markov processes: the abstract operator-theoretic viewpoint that unifies Brownian motion, Lévy processes, and diffusions under one roof.

---

## 1. Markov processes: definition

**Definition 1.1.** A **Markov process** is a family of random variables $(X_t)_{t \ge 0}$ on $(\Omega, \mathcal{F}, P)$ with values in $(E, \mathcal{E})$ (the state space) such that
$$
P(X_t \in B | \mathcal{F}_s) = P(X_t \in B | X_s), \quad \text{for all } s \le t, B \in \mathcal{E}.
$$

In other words: **conditional on the present, the future is independent of the past**.

The **transition probability** is $P_{s, t}(x, B) := P(X_t \in B | X_s = x)$ (when the RHS is well-defined). For homogeneous processes (our focus), $P_{s, t}$ depends only on $t - s$: $P_{s, t} = P_{t - s}$, and we write $P_t(x, B)$.

### 1.1 Examples

- **Brownian motion**: $P_t(x, B) = \int_B \frac{1}{\sqrt{2\pi t}} e^{-(y-x)^2/(2t)} dy$.
- **Geometric BM**: lognormal transition.
- **Ornstein–Uhlenbeck**: Gaussian transition with decaying mean.
- **Poisson process**: $P_t(n, \cdot)$ shifts by Poisson$(\lambda t)$.
- **Compound Poisson**: convolution semigroup of jump distributions.
- **Lévy process**: transition depends only on $t$ (stationary increments).
- **Finite-state CTMC**: generator matrix $Q$, transition $P(t) = e^{tQ}$.

---

## 2. Markov semigroups on function spaces

**Definition 2.1.** The **Markov semigroup** $(P_t)_{t \ge 0}$ acting on bounded measurable functions $f$ is
$$
(P_t f)(x) := E[f(X_t) | X_0 = x] = \int f(y) P_t(x, dy).
$$

**Properties.**

- **Semigroup:** $P_{t + s} = P_t P_s$ (Chapman–Kolmogorov).
- **Positivity:** $f \ge 0 \Rightarrow P_t f \ge 0$.
- **Markov:** $P_t \mathbf{1} = \mathbf{1}$ (conservation of probability).
- **Contraction on $L^\infty$:** $\|P_t f\|_\infty \le \|f\|_\infty$.

### 2.1 Semigroup on $C_0$: Feller processes

We often restrict to $C_0(E) := \{f \in C(E) : f\to 0 \text{ at } \infty\}$ (continuous functions vanishing at infinity). A Markov semigroup is **Feller** if:

- $P_t : C_0 \to C_0$ for each $t$.
- $P_t f \to f$ in $\|\cdot\|_\infty$ as $t \to 0$ for each $f \in C_0$.

**Feller processes** are the most well-behaved Markov processes — their transition semigroups act nicely on $C_0$.

### 2.2 Chapman–Kolmogorov and Feller property

- Brownian motion is Feller.
- Lévy processes are Feller.
- SDEs with Lipschitz coefficients and bounded derivatives produce Feller diffusions.
- Finite-state CTMCs trivially Feller (finite state space).

---

## 3. Infinitesimal generator

**Definition 3.1.** The **infinitesimal generator** of $(P_t)$ on $C_0$ is
$$
\mathcal{L} f := \lim_{t \downarrow 0} \frac{P_t f - f}{t},
$$

with domain $D(\mathcal{L}) = \{f \in C_0 : \text{limit exists in } \|\cdot\|_\infty\}$.

**Interpretation.** $\mathcal{L} f(x)$ is the expected instantaneous rate of change of $f(X_t)$ started at $x$:
$$
\mathcal{L} f(x) = \lim_{t \to 0}\frac{E[f(X_t)|X_0 = x] - f(x)}{t}.
$$

### 3.1 Generators of standard processes

- **Brownian motion**: $\mathcal{L} f = \tfrac{1}{2}f''$.
- **Brownian with drift** $dX = \mu dt + \sigma dB$: $\mathcal{L} f = \mu f' + \tfrac{1}{2}\sigma^2 f''$.
- **Multi-D Brownian**: $\mathcal{L} f = \tfrac{1}{2}\Delta f$ (Laplacian).
- **SDE** $dX = \mu(x) dt + \sigma(x) dB$: $\mathcal{L} f = \mu(x) f' + \tfrac{1}{2}\sigma^2(x) f''$.
- **Poisson process** rate $\lambda$: $\mathcal{L} f(n) = \lambda(f(n+1) - f(n))$.
- **Compound Poisson**: $\mathcal{L} f(x) = \lambda \int [f(x + y) - f(x)] \nu_Y(dy)$.
- **Lévy process** with triplet $(b, A, \nu)$:
$$
\mathcal{L} f(x) = b f'(x) + \tfrac{1}{2}A f''(x) + \int[f(x + y) - f(x) - y f'(x)\mathbf{1}_{|y|\le 1}]\nu(dy).
$$

---

## 4. The fundamental identity: Kolmogorov equations

**Theorem 4.1 (Backward equation).** For $f \in D(\mathcal{L})$,
$$
\frac{d}{dt}P_t f = \mathcal{L} P_t f = P_t \mathcal{L} f, \qquad P_0 f = f.
$$

This is the functional-analytic version of Kolmogorov's backward equation. It says: $P_t = e^{t\mathcal{L}}$ (formally), the exponential of the generator.

**Forward equation.** The adjoint $\mathcal{L}^*$ (acting on measures or densities) governs
$$
\frac{\partial}{\partial t}p_t = \mathcal{L}^* p_t,
$$

where $p_t$ is the density of $X_t$. For a diffusion, this is the Fokker–Planck PDE; for a Lévy process, a PIDE.

---

## 5. The Hille–Yosida theorem

Which linear operators $\mathcal{L}$ can be generators of Feller semigroups? The answer is classical functional analysis.

**Theorem 5.1 (Hille–Yosida).** A closed densely defined operator $\mathcal{L}$ on a Banach space is the generator of a strongly-continuous contraction semigroup iff:

1. $(0, \infty) \subset \rho(\mathcal{L})$ (resolvent set).
2. $\|\lambda(\lambda - \mathcal{L})^{-1}\| \le 1$ for all $\lambda > 0$.

*Proof sketch.*

$\Rightarrow$: if $\mathcal{L}$ generates $(P_t)$, then the resolvent $(\lambda - \mathcal{L})^{-1} f = \int_0^\infty e^{-\lambda t} P_t f dt$ exists for $\lambda > 0$ and has norm $\le 1/\lambda$.

$\Leftarrow$: define Yosida approximations $\mathcal{L}_\lambda = \lambda \mathcal{L}(\lambda - \mathcal{L})^{-1}$, which are bounded; let $P_t^\lambda = e^{t\mathcal{L}_\lambda}$; show this converges to a semigroup as $\lambda \to \infty$. Full proof in Pazzy (1983) *Semigroups of Linear Operators*. $\square$

**Corollary 5.2 (Markov Hille–Yosida).** A closed densely defined operator is the generator of a Feller Markov semigroup iff Hille–Yosida conditions hold plus:

- **Maximum principle:** if $f \in D(\mathcal{L})$ attains its max at $x$, then $\mathcal{L} f(x) \le 0$.

### 5.1 When does an SDE produce a Feller generator?

For $dX_t = \mu(X_t) dt + \sigma(X_t) dB_t$ with Lipschitz + linear-growth coefficients, the generator
$$
\mathcal{L} f = \mu f' + \tfrac{1}{2}\sigma^2 f''
$$

on $C_0^2(\mathbb{R})$ satisfies Hille–Yosida + maximum principle, and generates a Feller semigroup.

---

## 6. Martingale problems and Dynkin's formula

### 6.1 Dynkin's formula

**Theorem 6.1 (Dynkin).** Let $\mathcal{L}$ be the generator of the Feller process $X$. For any $f \in D(\mathcal{L})$ and stopping time $\tau$ with $E_x[\tau] < \infty$,
$$
E_x[f(X_\tau)] = f(x) + E_x\Bigl[\int_0^\tau \mathcal{L} f(X_s) ds\Bigr].
$$

*Proof.* The process $M_t := f(X_t) - f(X_0) - \int_0^t \mathcal{L} f(X_s) ds$ is a martingale (using the semigroup and Markov property). Applying optional stopping to $M_\tau$ gives the formula. $\square$

### 6.2 Martingale problem

**Definition 6.2 (Stroock–Varadhan).** The martingale problem for $\mathcal{L}$ with initial distribution $\mu$ asks for a probability measure $P$ on $C([0, \infty), E)$ such that $X_0$ has law $\mu$ under $P$ and, for every $f \in D(\mathcal{L})$, the process
$$
f(X_t) - f(X_0) - \int_0^t \mathcal{L} f(X_s) ds
$$
is a $P$-martingale.

Uniqueness of the martingale problem for $\mathcal{L}$ is **equivalent to** existence and uniqueness of the Markov process with generator $\mathcal{L}$. This is the foundation of **weak solutions to SDEs** (Module 3.4).

### 6.3 Application: Harmonic functions

A function $h : E \to \mathbb{R}$ is **$\mathcal{L}$-harmonic** if $\mathcal{L} h = 0$. Dynkin's formula then gives
$$
E_x[h(X_\tau)] = h(x)
$$

for any stopping time with finite expectation. This recovers classical potential-theoretic results:

- For Brownian motion in $\mathbb{R}^d$, $\mathcal{L} = \tfrac{1}{2}\Delta$, so harmonic = Laplace-harmonic. $u(x) = E_x[g(B_\tau)]$ with $\tau$ = exit from a bounded domain gives the solution to the Dirichlet problem.
- For an SDE diffusion, $\mathcal{L}$-harmonic functions are used to compute hitting probabilities.

---

## 7. Invariant measures and ergodic theorem

**Definition 7.1.** A probability measure $\pi$ on $E$ is **invariant** (stationary) for $(P_t)$ if
$$
\int (P_t f)(x) \pi(dx) = \int f(x) \pi(dx), \quad \text{for all } f \in C_b, t \ge 0.
$$

Equivalently (under regularity), $\mathcal{L}^* \pi = 0$.

### 7.1 Existence: Krylov–Bogoliubov

If $E$ is compact (or the process has some Lyapunov function ensuring tightness of $(P_t \delta_x)$), an invariant measure exists.

### 7.2 Uniqueness and convergence: ergodic theorems

**Theorem 7.2 (Birkhoff / ergodic for Markov).** If $\pi$ is the unique invariant measure and the semigroup is **irreducible** (connection between any two points in positive time), then for $\pi$-a.e. $x$,
$$
\frac{1}{T}\int_0^T f(X_t) dt \xrightarrow{T\to\infty} \int f d\pi \quad \text{a.s.}
$$

### 7.3 Explicit stationary measures

- **OU**: $\pi = N(\theta, \sigma^2/(2\kappa))$ (Gaussian).
- **CIR**: $\pi = \text{Gamma}(2\kappa\theta/\xi^2, \xi^2/(2\kappa))$.
- **Reflected BM on $[0, \infty)$**: exponential distribution.
- **Overdamped Langevin $dX = -\nabla U(X) dt + \sqrt{2} dB$**: $\pi \propto e^{-U}$ (Boltzmann–Gibbs).

The Langevin connection is crucial in MCMC: to sample from $e^{-U}$, run Langevin SDE to equilibrium (or discretize via MALA / HMC).

---

## 8. Spectral theory of generators

For self-adjoint $\mathcal{L}$ on $L^2(\pi)$ (e.g., reversible diffusions), we get spectral decomposition:
$$
\mathcal{L} = \sum_k -\lambda_k P_{\psi_k}, \qquad 0 = \lambda_0 < \lambda_1 \le \lambda_2 \le \cdots
$$

where $\psi_k$ are eigenfunctions, $\lambda_k$ eigenvalues (non-negative for generators of reversible processes). The **spectral gap** $\lambda_1$ controls the convergence rate to equilibrium:
$$
\|P_t f - \int f d\pi\|_{L^2(\pi)} \le e^{-\lambda_1 t} \|f - \int f d\pi\|_{L^2(\pi)}.
$$

### 8.1 Examples

- **OU generator $\mathcal{L} = -\alpha x \partial_x + \tfrac{1}{2}\sigma^2 \partial_{xx}$**: eigenfunctions are Hermite polynomials; eigenvalues $\lambda_k = k\alpha$.
- **1D Brownian on $[0, 1]$ reflected**: eigenfunctions are cosines; $\lambda_k = (\pi k)^2 / 2$.
- **CIR generator**: eigenfunctions are Laguerre polynomials.

The spectral gap theorem quantifies mixing time and is central to MCMC convergence analysis (Module 2.7 in discrete time).

---

## 9. Feller diffusions and Ray–Knight theorems

A **Feller diffusion** is a one-dimensional Markov process $X$ on an interval $I$ with continuous paths whose generator is second-order:
$$
\mathcal{L} f = a(x) f''(x) + b(x) f'(x), \qquad a(x) > 0.
$$

Feller's classification of boundaries ($x$ can be *natural*, *entrance*, *exit*, *regular*) determines whether $X$ is extended through the boundary, killed, or reflected.

### 9.1 Application: Local time at zero

The **Ray–Knight theorem** gives the law of local time $L^x_T$ of Brownian motion as a certain squared Bessel process in $x$ — a remarkable connection between Brownian path integrals and diffusions.

---

## 10. Python: numerical stationary distributions

```python
import numpy as np
from scipy.stats import norm, gamma

rng = np.random.default_rng(0)

# ---- (a) OU stationary via simulation ----
alpha, sigma = 1.5, 1.0
T_sim = 200.0
dt = 0.001
n = int(T_sim/dt)
X = np.zeros(n+1)
X[0] = 3.0  # far from mean
for k in range(n):
    X[k+1] = X[k] + (-alpha*X[k])*dt + sigma*np.sqrt(dt)*rng.standard_normal()

# Empirical stationary distribution from second half
X_stationary = X[n//2:]
print(f"OU empirical mean: {X_stationary.mean():.4f}, theory: 0")
print(f"OU empirical var : {X_stationary.var():.4f}, theory: {sigma**2/(2*alpha):.4f}")

# ---- (b) CIR stationary: Gamma ----
kappa, theta, xi = 2.0, 0.04, 0.2
T_sim = 500.0
dt = 0.001
n = int(T_sim/dt)
V = np.zeros(n+1)
V[0] = theta
for k in range(n):
    V_pos = max(V[k], 0.0)
    V[k+1] = V[k] + kappa*(theta - V[k])*dt + xi*np.sqrt(V_pos*dt)*rng.standard_normal()

V_stationary = V[n//2:]
# Theory: Gamma(2*kappa*theta/xi^2, xi^2/(2*kappa))
shape_th = 2*kappa*theta / xi**2
scale_th = xi**2 / (2*kappa)
print(f"\nCIR empirical mean: {V_stationary.mean():.5f}, theory: {shape_th*scale_th:.5f}")
print(f"CIR empirical var : {V_stationary.var():.6f}, theory: {shape_th*scale_th**2:.6f}")

# ---- (c) Langevin sampler for bimodal target ----
def neg_log_posterior(x):
    # target = exp(-(x^2 - 2)^2 / 2)  -- bimodal with modes at ±sqrt(2)
    return (x**2 - 2)**2 / 2.0

def grad_U(x):
    return 2 * (x**2 - 2) * 2 * x  # gradient

T_sim = 500.0
dt = 0.005
n = int(T_sim/dt)
x = np.zeros(n+1)
x[0] = 0.0
for k in range(n):
    x[k+1] = x[k] - grad_U(x[k])*dt + np.sqrt(2*dt)*rng.standard_normal()

# Plot or summary
print(f"\nLangevin: P(x > 0): {(x > 0).mean():.4f}, theory ~ 0.5 (bimodal symmetric)")

# ---- (d) Spectral gap via Monte Carlo of OU ----
M = 5000
T_sim = 5.0
dt = 0.01
n = int(T_sim/dt)
f0_value = 0.0  # f(X) = X at X_0 = x_0 = 3.0
x_start = 3.0
# Simulate OU from x_start
X_paths = np.zeros((M, n+1))
X_paths[:, 0] = x_start
for k in range(n):
    X_paths[:, k+1] = X_paths[:, k] + (-alpha*X_paths[:, k])*dt + sigma*np.sqrt(dt)*rng.standard_normal(M)

# f(x) = x, invariant mean = 0, so |P_t f (x_0)| decays like e^{-alpha t}
mean_path = X_paths.mean(axis=0)
t_grid = np.linspace(0, T_sim, n+1)
decay_theory = x_start * np.exp(-alpha * t_grid)
print(f"\nOU decay at t=T: simulated={mean_path[-1]:.4f}, theory={decay_theory[-1]:.4f}")
```

### Expected output

- OU stationary: mean ~0, variance ~$\sigma^2/(2\alpha) = 1/3$.
- CIR stationary: mean ~0.04, variance matches Gamma theory.
- Langevin: roughly 50/50 between modes (bimodal sampling).
- OU decay rate matches $e^{-\alpha t}$ exponential.

---

## 11. [QUANT APPLICATION] — semigroups in finance

### 11.1 Pricing as semigroup

Option price $V(t, x) = E[g(X_T)|X_t = x] = (P_{T-t}^{r} g)(x)$, where $P_s^r$ is the **killed semigroup** corresponding to discounting: $(P_s^r f)(x) = E[e^{-\int_0^s r(X_u) du} f(X_s) | X_0 = x]$. Feynman–Kac says $V$ satisfies $(\partial_t + \mathcal{L} - r)V = 0$.

### 11.2 Credit risk: reduced-form models

Default intensity $\lambda_t$ defines the survival semigroup $P_t = e^{-\int_0^t \lambda_s ds}$. Bond prices, CDS spreads, etc., are all semigroup expressions. Affine structure of $\lambda$ gives tractable closed forms.

### 11.3 Mixing times and MCMC in finance

The spectral gap of a Gibbs sampler (Module 2.7) controls how fast Bayesian posteriors can be sampled. Slower mixing = longer MC; faster mixing = faster convergence. Critical for calibration of high-dimensional models.

### 11.4 Regime-switching models

$$
X_t = X_t^{(i_t)}, \quad i_t \in \{1, \ldots, K\}\text{ (regime)},
$$

where regime $i_t$ is itself a CTMC with generator $Q$. The joint $(X, i)$ is a higher-dim Markov process whose generator decomposes into SDE + regime-switching parts. Fang–Oosterlee (2008) give Fourier methods; Elliott et al. (1994) give pricing formulas.

### 11.5 Ergodicity and long-run growth

For a stationary GBM-like model with ergodic components, the long-run growth rate (Lyapunov exponent) is computable via the stationary measure: $\lim_t \tfrac{1}{t}\log S_t = E_\pi[\text{drift}]$. Relevant for long-horizon asset allocation.

---

## 12. Summary

- **Markov processes** have memoryless structure: future depends on present only.
- **Semigroup** $P_t f(x) = E[f(X_t)|X_0 = x]$ captures the dynamics on test functions.
- **Generator** $\mathcal{L} = \frac{d}{dt}P_t|_{t=0}$ is the infinitesimal driver; differential / integro-differential operator.
- **Hille–Yosida** characterizes generators of Feller semigroups.
- **Dynkin's formula**: $E_x[f(X_\tau)] = f(x) + E_x[\int_0^\tau \mathcal{L}f ds]$.
- **Martingale problem** (Stroock–Varadhan) establishes equivalence between semigroups and Markov processes.
- **Invariant measures and ergodic theorems** govern long-run behavior.
- **Spectral gap** controls convergence rate to equilibrium.
- **Workhorse examples**: OU (Gaussian), CIR (Gamma), reflected BM (exponential), Langevin ($e^{-U}$).

---

## 13. Exercises

**★ (warm-ups)**

**13.1** Verify the semigroup property $P_{t+s} f = P_t P_s f$ for Brownian motion via the density formula.

**13.2** Compute the generator of GBM $dS/S = \mu dt + \sigma dB$: show $\mathcal{L} f = \mu x f'(x) + \tfrac{1}{2}\sigma^2 x^2 f''(x)$.

**13.3** Verify that $f(x) = e^{\alpha x}$ is in the domain of the generator of OU, and compute $\mathcal{L} f$.

**13.4** Derive the generator of a finite-state CTMC directly from the transition matrix $P(t) = e^{tQ}$.

**13.5** Derive the generator of a Poisson process with rate $\lambda$ on state space $\mathbb{N}$.

**★★ (core)**

**13.6** **(Dynkin's formula.)** Prove Dynkin's formula by first showing $M_t := f(X_t) - f(X_0) - \int_0^t \mathcal{L} f(X_s) ds$ is a martingale, then applying optional stopping.

**13.7** **(OU eigenfunctions.)** For OU with generator $\mathcal{L} f = -\alpha x f' + \tfrac{1}{2}\sigma^2 f''$, verify that the Hermite polynomials $H_k(x\sqrt{\alpha}/\sigma)$ are eigenfunctions with eigenvalues $-k\alpha$.

**13.8** **(Feller boundary classification.)** For $dX = \sigma\sqrt{X} dB$ (squared Bessel), classify the boundary $0$ as natural, entrance, exit, or regular.

**13.9** **(Stationary OU.)** Derive the stationary density of OU from Fokker–Planck: solve $\mathcal{L}^* p = 0$ in the form $\partial_x[\alpha x p + \tfrac{1}{2}\sigma^2 \partial_x p] = 0$.

**13.10** **(Stationary CIR.)** Derive the stationary density of CIR similarly, and show it is Gamma.

**13.11** **(Langevin equilibrium.)** Show that $dX = -\nabla U(X) dt + \sqrt{2\varepsilon} dB$ has stationary density $\propto e^{-U/\varepsilon}$.

**13.12** **(Krylov–Bogoliubov.)** If $E$ is compact and $X$ is Feller, show an invariant measure exists.

**13.13** **(Harmonic function.)** In 1D OU, find $h : \mathbb{R} \to \mathbb{R}$ with $\mathcal{L} h = 0$: show $h$ is a polynomial of degree $\le 1$.

**13.14** **(Spectral gap for OU.)** Use the Hermite eigenfunction decomposition to show the spectral gap is $\alpha$; verify decay of $(P_t - \pi)f$ is $e^{-\alpha t}$.

**13.15** **(Dirichlet problem.)** Let $D \subset \mathbb{R}^d$ bounded, and $\tau = $ exit time of $B$ from $D$. Show $u(x) = E_x[g(B_\tau)]$ satisfies $\Delta u = 0$ in $D$ with boundary value $g$.

**★★★ (research / quant)**

**13.16** **(Affine generator.)** For an affine diffusion $dX = (\mu_0 + \mu_1 X) dt + \sqrt{\sigma_0 + \sigma_1 X} dB$, write the generator and show its action on exponentials $e^{uX}$ closes (the CF is a product of exponentials with affine coefficients — Duffie, Filipovic, Schachermayer 2003).

**13.17** **(HMC derivation.)** Using the Hamilton equations generator, show that HMC's generator is close to Langevin's generator (up to a momentum-flip step) and that invariant distribution is $\propto e^{-U(x)}$.

**13.18** **(Infinitesimal transition of CTMC.)** For a finite-state CTMC with generator $Q$, derive the forward Kolmogorov equation $\dot P(t) = P(t) Q$ and the backward $\dot P(t) = Q P(t)$.

**13.19** **(Stroock–Varadhan martingale problem.)** Read Stroock–Varadhan's classic paper on the martingale problem. Apply it to a diffusion with non-smooth coefficients.

**13.20** **(Regime switching generator.)** For $dX = \mu(i_t, X) dt + \sigma(i_t, X) dB + dJ$ with $i_t$ a CTMC on $\{1, 2, 3\}$, write the joint generator as a block $3 \times 3$ operator.

**13.21** **(BSDEs and semigroups.)** For a BSDE $-dY_t = f(Y_t, Z_t) dt - Z_t dB_t$, $Y_T = g(X_T)$, show that $Y_t = u(t, X_t)$ for $u$ solving the semilinear PDE $\partial_t u + \mathcal{L}u + f(u, \sigma \partial_x u) = 0$.

**13.22** **(Feynman–Kac with nonzero source.)** Solve $\partial_t u + \tfrac{1}{2}\Delta u + f = 0$ on $[0, T] \times \mathbb{R}^d$ with $u(T, x) = 0$ using Feynman–Kac: show $u(t, x) = E[\int_t^T f(s, B_s) ds | B_t = x]$.

---

## 14. Forward pointers

- **Module 3.7 (Continuous-time martingales).** Continuous-time martingale theory: Doob–Meyer, BDG, semi-martingale representation, local time.
- **Subjects 4+**: optimal stopping (American options), HJB equations, Merton's portfolio problem, linear quadratic control, reinforcement learning.

Semigroup theory is the **abstract underpinning** of all of continuous-time financial mathematics: pricing, hedging, risk measures, filtering, and optimal control are all semigroup evaluations.

**Next module:** Continuous-time martingales — the final pillar: Doob–Meyer decomposition, BDG inequalities, semi-martingale integration, local time.

---

*End of Module 3.6.*

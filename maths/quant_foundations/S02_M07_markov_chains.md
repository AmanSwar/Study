# Module 2.7: Markov Chains

**Subject 2: Probability Theory** · Module 7 of 7

---

## 0. Prerequisites and Position

- **Module 1.3** (Lebesgue integration), **Module 1.5** ($L^p$), **Module 1.6** (Radon-Nikodym).
- **Module 2.1** (probability spaces, 0–1 laws), **Module 2.2** (conditional expectation), **Module 2.3** (modes of convergence), **Module 2.4** (LLN, CLT), **Module 2.6** (martingales).

A Markov chain is a stochastic process with the **memoryless property**: the future is conditionally independent of the past given the present. This simple structural assumption unlocks an enormous analytic theory — state classification, existence and uniqueness of stationary distributions, ergodic theorems, and the computational methods of MCMC that have become indispensable to modern Bayesian statistics and computational finance.

This is the *final module of Subject 2*, Probability Theory. It synthesizes everything we have built: conditional expectation, martingales for hitting-time analysis, LLN/ergodic theorems for stationary chains, CLT for chains via Poisson equations.

We cover:
- Discrete-time Markov chains: transition kernels, Chapman-Kolmogorov, strong Markov property.
- State classification: irreducibility, periodicity, transience, recurrence.
- Stationary distributions: existence, uniqueness, reversibility.
- Ergodic theorem: time averages $=$ space averages for positive recurrent chains.
- $n$-step convergence: total variation and spectral gap.
- Markov chain CLT via the Poisson equation.
- MCMC: Metropolis-Hastings and Gibbs samplers, rigorous convergence.
- Quant applications: interest rate chains, credit transition matrices, MCMC in Bayesian asset allocation.

---

## 1. Discrete-Time Markov Chains

### 1.1 Definition

Let $S$ be a countable state space (finite or denumerably infinite). A **transition matrix** on $S$ is $P = (p(x, y))_{x, y \in S}$ with $p(x, y) \ge 0$ and $\sum_y p(x, y) = 1$ for every $x$. We write $Pf(x) := \sum_y p(x, y) f(y)$ for bounded $f$.

**Definition 1.1.** $(X_0, X_1, X_2, \dots)$ is a **Markov chain** with transition matrix $P$ and initial distribution $\mu$ if $X_0 \sim \mu$ and
$$
P(X_{n+1} = y | X_0 = x_0, \dots, X_n = x_n) = p(x_n, y) \quad \text{whenever } P(X_n = x_n, \dots) > 0.
$$

Equivalently, the **Markov property**: $P(X_{n+1} = y | \mathcal{F}_n) = p(X_n, y)$ where $\mathcal{F}_n := \sigma(X_0, \dots, X_n)$.

**Existence (Kolmogorov extension, Module 2.1).** Given $\mu$ and $P$, a Markov chain with these characteristics exists on $S^{\mathbb{N}}$ with the cylinder $\sigma$-algebra.

### 1.2 Chapman-Kolmogorov

$n$-step transitions: $p^{(n)}(x, y) := P(X_n = y | X_0 = x)$. By the Markov property,
$$
p^{(n+m)}(x, y) = \sum_z p^{(n)}(x, z) p^{(m)}(z, y).
$$
This is matrix multiplication: $P^{(n+m)} = P^{(n)} P^{(m)}$, so $P^{(n)} = P^n$.

### 1.3 Strong Markov property

**Theorem 1.2** (Strong Markov)**.** For a Markov chain $X$ with transition $P$ and any stopping time $\tau < \infty$ a.s., conditional on $\mathcal{F}_\tau$ and $X_\tau = x$, the post-$\tau$ process $(X_{\tau + n})_{n \ge 0}$ is a Markov chain with transition $P$ starting from $x$, independent of $\mathcal{F}_\tau$.

*Proof.* Use the tower + partition $\{\tau = k\}$:
$$
P(X_{\tau+n} = y, \tau = k | \mathcal{F}_k) \cdot \mathbf{1}_{\tau = k} = p^{(n)}(X_k, y) \mathbf{1}_{\tau = k}
$$
by the (weak) Markov property applied on $\{\tau = k\} \in \mathcal{F}_k$. Sum over $k$. $\square$

This is essential for recurrence/transience analysis: once the chain hits a state $x$, it "restarts" with the same dynamics.

---

## 2. Classification of States

### 2.1 Accessibility and communication

**Definition 2.1.**
- $y$ is **accessible** from $x$ (written $x \to y$) if $p^{(n)}(x, y) > 0$ for some $n \ge 0$.
- $x$ and $y$ **communicate** ($x \leftrightarrow y$) if $x \to y$ and $y \to x$.
- Communication is an equivalence relation, partitioning $S$ into **communicating classes**.
- A chain is **irreducible** if there is a single class: all states communicate.

### 2.2 Recurrence and transience

Let $T_y := \inf\{n \ge 1 : X_n = y\}$ (first return/hitting time; $= \infty$ if never).

**Definition 2.2.** $y$ is **recurrent** if $P_y(T_y < \infty) = 1$, else **transient**. Write $\rho_y := P_y(T_y < \infty)$ (return probability starting from $y$).

**Theorem 2.3.** $y$ recurrent iff $\sum_n p^{(n)}(y, y) = \infty$ iff $P_y(\text{visit } y \text{ i.o.}) = 1$.

*Proof.* Let $V_y = \sum_n \mathbf{1}_{X_n = y}$ (number of visits). By the strong Markov property, $V_y$ under $P_y$ has a geometric-like structure: $P_y(V_y > k) = \rho_y^k$. Hence
$$
E_y[V_y] = \sum_{k \ge 0} P_y(V_y > k) = \sum_{k \ge 0} \rho_y^k = \begin{cases} \infty & \rho_y = 1 \\ 1/(1-\rho_y) & \rho_y < 1. \end{cases}
$$
And $E_y[V_y] = \sum_{n \ge 0} P_y(X_n = y) = \sum_n p^{(n)}(y, y)$. Thus $\rho_y = 1$ iff $\sum_n p^{(n)}(y, y) = \infty$. Also $P_y(V_y = \infty) = \lim_k P_y(V_y > k) = 1$ if $\rho_y = 1$. $\square$

**Corollary 2.4** (Recurrence is a class property)**.** If $x \leftrightarrow y$, then $x$ recurrent iff $y$ recurrent. *Proof.* $\sum_n p^{(n)}(x, x) \ge p^{(k)}(x, y) p^{(m)}(y, x) \sum_n p^{(n)}(y, y)$ for suitable $k, m$ with $p^{(k)}(x,y), p^{(m)}(y,x) > 0$.

### 2.3 Positive recurrence and null recurrence

**Definition 2.5.** A recurrent state $y$ is **positive recurrent** if $E_y[T_y] < \infty$, **null recurrent** otherwise.

**Positive recurrence** is also a class property (proof via renewal theory).

### 2.4 Examples

**Example 2.6** (Finite-state)**.** For a finite-state chain, every state is either transient or positive recurrent (no null recurrence). An irreducible finite-state chain is positive recurrent.

**Example 2.7** (SRW on $\mathbb{Z}^d$)**.** Simple symmetric random walk on $\mathbb{Z}^d$:
- $d = 1, 2$: recurrent (Pólya).
- $d \ge 3$: transient.

Proof via $\sum p^{(2n)}(0, 0) \sim c/n^{d/2}$ (Stirling). Sum converges iff $d/2 > 1$.

**Example 2.8** (Asymmetric 1-D SRW)**.** $X_i$ iid with $P(X_i = 1) = p$, $P(X_i = -1) = q = 1 - p$. If $p \ne q$, $S_n = \sum X_i$ is transient (drifts to $\pm \infty$).

### 2.5 Periodicity

**Definition 2.9.** The **period** of state $x$ is $d(x) := \gcd\{n \ge 1 : p^{(n)}(x, x) > 0\}$. A state is **aperiodic** if $d(x) = 1$. Periodicity is a class property.

**Example.** SRW on $\mathbb{Z}$: period 2 (returns only at even times).

**Lemma 2.10.** For aperiodic irreducible chain, for every $x, y$ there exists $N = N(x, y)$ such that $p^{(n)}(x, y) > 0$ for all $n \ge N$.

---

## 3. Stationary Distributions

### 3.1 Definition

**Definition 3.1.** A probability measure $\pi$ on $S$ is **stationary** (or **invariant**) for $P$ if $\pi P = \pi$, i.e., $\sum_x \pi(x) p(x, y) = \pi(y)$ for every $y$. Equivalently, if $X_0 \sim \pi$, then $X_n \sim \pi$ for all $n$.

The stationary distribution captures the "long-run" fraction of time the chain spends in each state.

### 3.2 Existence and uniqueness

**Theorem 3.2.** For an irreducible chain:
(a) If positive recurrent, there is a unique stationary distribution, given by
$$
\pi(y) = \frac{1}{E_y[T_y]}.
$$
(b) If null recurrent or transient, no stationary distribution exists.

*Proof sketch of (a).* Fix a reference state $a$. For each $y$, let
$$
\tilde\pi(y) := E_a\Big[\sum_{n=0}^{T_a - 1} \mathbf{1}_{X_n = y}\Big] = \text{expected visits to } y \text{ between consecutive visits to } a.
$$
$\tilde\pi$ is a non-trivial invariant *measure* (not necessarily probability): $\tilde\pi P = \tilde\pi$. Its total mass is $\tilde\pi(S) = E_a[T_a]$. If $E_a[T_a] < \infty$ (positive recurrent), normalize: $\pi := \tilde\pi / E_a[T_a]$ is the stationary distribution with $\pi(a) = 1/E_a[T_a]$.

For uniqueness: the normalized $\tilde\pi$ depends on the reference state $a$, but all such normalizations coincide on the full space (unique up to scaling, and both hit mass 1 with the given normalization). $\square$

### 3.3 Reversibility

**Definition 3.3.** A chain is **reversible** with respect to $\pi$ if the detailed balance equations hold:
$$
\pi(x) p(x, y) = \pi(y) p(y, x) \quad \text{for all } x, y.
$$

Detailed balance $\Rightarrow$ $\pi$ is stationary ($\sum_x \pi(x) p(x, y) = \sum_x \pi(y) p(y, x) = \pi(y)$). Reversibility is much stronger than stationarity.

**Example 3.4** (Random walk on a graph)**.** For a connected undirected graph with degree $d(x)$, the random walk $p(x, y) = \mathbf{1}_{x \sim y}/d(x)$ is reversible with $\pi(x) \propto d(x)$.

**Reversible chains in MCMC.** Most MCMC algorithms (Metropolis-Hastings, Gibbs) are designed to satisfy detailed balance with respect to the target $\pi$ — this guarantees $\pi$ is stationary.

---

## 4. Ergodic Theorem

### 4.1 Pointwise ergodic theorem

**Theorem 4.1** (Markov chain ergodic theorem)**.** Let $X$ be an irreducible positive recurrent Markov chain with stationary distribution $\pi$. For any $f: S \to \mathbb{R}$ with $\sum_x \pi(x) |f(x)| < \infty$, and any initial distribution $\mu$,
$$
\frac{1}{n} \sum_{k=0}^{n-1} f(X_k) \xrightarrow{\text{a.s.}} \sum_x \pi(x) f(x) =: \pi(f) \quad \text{as } n \to \infty.
$$

*Proof.* Fix a state $a$. Let $T_a^{(k)}$ be the $k$-th return to $a$ (with $T_a^{(0)} = 0$ if $X_0 = a$, else first hit). By strong Markov, the **cycles** $(X_{T_a^{(k-1)}}, \dots, X_{T_a^{(k)} - 1})$ are iid across $k \ge 1$.

Define $W_k := \sum_{i = T_a^{(k-1)}}^{T_a^{(k)} - 1} f(X_i)$ (cycle sum), $L_k := T_a^{(k)} - T_a^{(k-1)}$ (cycle length). By iid + SLLN:
$$
\frac{1}{N} \sum_{k=1}^N W_k \xrightarrow{\text{a.s.}} E_a[W_1] = E_a\Big[\sum_{i=0}^{T_a - 1} f(X_i)\Big] = \sum_y f(y) \tilde\pi(y) = E_a[T_a] \cdot \pi(f),
$$
using the Kac-style representation of the invariant measure. Similarly $N^{-1} T_a^{(N)} \to E_a[T_a]$.

For general $n$, find $N(n)$ with $T_a^{(N(n))} \le n < T_a^{(N(n)+1)}$. Sandwich:
$$
\frac{1}{T_a^{(N(n)+1)}} \sum_{k=1}^{N(n)} W_k \le \frac{1}{n} \sum_{i=0}^{n-1} f(X_i) \le \frac{1}{T_a^{(N(n))}} \sum_{k=1}^{N(n)+1} W_k + \text{boundary}.
$$
(For $f \ge 0$; general $f = f^+ - f^-$.) The sandwich bounds both converge a.s. to $\pi(f)$. $\square$

### 4.2 Interpretation

Time averages equal space averages:
$$
\underbrace{\frac{1}{n}\sum_{k<n} f(X_k)}_{\text{time average}} \to \underbrace{\int f \, d\pi}_{\text{space average under stationary distribution}}.
$$

This is the SLLN for Markov chains. For iid, $\pi$ is the common distribution and we recover the classical SLLN.

---

## 5. Convergence to Stationarity

### 5.1 Total variation distance

$\|\mu - \nu\|_{TV} := \frac{1}{2} \sum_x |\mu(x) - \nu(x)| = \sup_{A \subseteq S} |\mu(A) - \nu(A)|$.

**Theorem 5.1** (Convergence theorem)**.** For an irreducible, aperiodic, positive recurrent chain with stationary $\pi$, and any initial $\mu$:
$$
\|\mu P^n - \pi\|_{TV} \to 0 \quad \text{as } n \to \infty.
$$

*Proof via coupling.* Run two copies $(X_n, Y_n)$ independently with $X_0 \sim \mu$, $Y_0 \sim \pi$, both with transition $P$. Define the coupling time $\tau := \inf\{n : X_n = Y_n\}$, and set $Y'_n := X_n$ for $n \ge \tau$, $Y'_n := Y_n$ for $n < \tau$. The coupled $(X, Y')$ has the same marginal law ($X \sim \mu P^n$, $Y' \sim \pi$ for each $n$ by strong Markov). The **coupling inequality**:
$$
\|\mu P^n - \pi\|_{TV} \le P(\tau > n).
$$
By irreducibility + aperiodicity, $\tau < \infty$ a.s.; hence $P(\tau > n) \to 0$. $\square$

### 5.2 Spectral gap

Suppose $P$ is reversible w.r.t. $\pi$. Then $P$ is self-adjoint on $L^2(\pi)$, with real eigenvalues $1 = \lambda_1 > \lambda_2 \ge \dots \ge \lambda_{|S|} \ge -1$. The **spectral gap** is $\gamma := 1 - \lambda_2$ (if $\lambda_2 < 1$).

**Theorem 5.2.** For a reversible, irreducible, aperiodic chain,
$$
\|\mu P^n - \pi\|_{TV} \le \frac{1}{2\sqrt{\pi_*}} (1 - \gamma)^n
$$
where $\pi_* := \min_x \pi(x)$.

*Proof.* Spectral expansion of the density $d\mu P^n / d\pi$ in the eigenbasis of $P$. $(1 - \gamma)^n = \lambda_2^n$ controls the decay. $\square$

**Significance.** The spectral gap $\gamma$ is the *mixing rate*. A large $\gamma$ means fast convergence; a small $\gamma$ means slow mixing. Essential for MCMC efficiency.

### 5.3 Cheeger inequality

For reversible chains, the spectral gap is related to geometric quantities:
$$
\frac{\Phi^2}{2} \le \gamma \le 2 \Phi,
$$
where $\Phi := \min_{A : \pi(A) \le 1/2} \pi(A)^{-1} \sum_{x \in A, y \in A^c} \pi(x) p(x, y)$ is the **conductance**. Small conductance (bottlenecks) $\Rightarrow$ small spectral gap $\Rightarrow$ slow mixing.

---

## 6. Poisson Equation and Markov Chain CLT

### 6.1 Poisson equation

For $f: S \to \mathbb{R}$ with $\pi(f) = 0$, the **Poisson equation** is
$$
(I - P) g = f, \quad \text{i.e., } g(x) - \sum_y p(x, y) g(y) = f(x).
$$

**Theorem 6.1.** If the chain is irreducible, positive recurrent, aperiodic with sufficiently fast mixing (e.g., spectral gap), then $g(x) := \sum_{n \ge 0} E_x[f(X_n)]$ (or $g(x) = E_x[\sum_{k=0}^{T_a - 1} f(X_k)]$ for a reference state $a$) solves $(I - P)g = f$.

### 6.2 Martingale from Poisson equation

Given $g$ solving $(I - P) g = f$, define
$$
M_n := g(X_n) - g(X_0) + \sum_{k=0}^{n-1} f(X_k).
$$
Then $M_n$ is a martingale:
$$
E[M_{n+1} - M_n | \mathcal{F}_n] = E[g(X_{n+1}) | \mathcal{F}_n] - g(X_n) + f(X_n) = (Pg)(X_n) - g(X_n) + f(X_n) = 0.
$$
So $S_n^f := \sum_{k=0}^{n-1} f(X_k) = g(X_0) - g(X_n) + M_n$.

### 6.3 Markov chain CLT

**Theorem 6.2** (Markov chain CLT)**.** Under suitable mixing conditions + $\pi(g^2) < \infty$, for an irreducible, positive recurrent, aperiodic chain started from any $x$ or from $\pi$:
$$
\frac{S_n^f}{\sqrt{n}} = \frac{1}{\sqrt{n}} \sum_{k=0}^{n-1} f(X_k) \xRightarrow{d} \mathcal{N}(0, \sigma_f^2),
$$
where the asymptotic variance is
$$
\sigma_f^2 = \pi\big((g + f)^2 - (Pg)^2\big) = \text{Var}_\pi(f) + 2 \sum_{k \ge 1} \text{Cov}_\pi(f(X_0), f(X_k)).
$$

*Proof sketch.* Use the martingale decomposition $S_n^f = g(X_0) - g(X_n) + M_n$. The boundary terms $g(X_0) - g(X_n)$ are $O_P(1)$, negligible under $\sqrt{n}$ scaling. The martingale $M_n$ has increments $\Delta_k = g(X_k) - (Pg)(X_{k-1}) + f(X_{k-1})$; apply martingale CLT (Module 2.6, Theorem 8.1) with conditional variance $E[\Delta_k^2 | \mathcal{F}_{k-1}] = (Pg^2)(X_{k-1}) - (Pg)^2(X_{k-1}) + \cdots$, converging (ergodic theorem!) to $\sigma_f^2 \cdot k$. $\square$

**Significance.** Asymptotic variance $\sigma_f^2$ is not just the stationary variance $\pi(f^2)$ (as in iid), but includes **autocorrelation corrections**. This is why MCMC with slow mixing gives noisy estimates — the effective sample size is $n / \tau_{int}$ where $\tau_{int} \sim \sigma_f^2 / \text{Var}_\pi(f)$ is the integrated autocorrelation time.

---

## 7. MCMC: Metropolis-Hastings and Gibbs

### 7.1 Goal

Sample from a target distribution $\pi$ on a space $S$ where $\pi$ is only known up to normalization (e.g., Bayesian posterior $\pi(\theta) \propto L(\text{data} | \theta) p(\theta)$, or Boltzmann distribution $\pi(x) \propto e^{-\beta H(x)}$).

**Strategy.** Construct a Markov chain with stationary distribution $\pi$; run it long enough; use the samples as approximate draws from $\pi$.

### 7.2 Metropolis-Hastings

**Algorithm.** Pick a **proposal kernel** $q(x, y)$ (easy to sample). From current state $x$:
1. Propose $y \sim q(x, \cdot)$.
2. Accept with probability $\alpha(x, y) := \min(1, \frac{\pi(y) q(y, x)}{\pi(x) q(x, y)})$.
3. If accepted, $X_{n+1} = y$; else $X_{n+1} = x$.

**Claim 7.1.** This chain is reversible w.r.t. $\pi$.

*Proof.* Transition kernel: $p(x, y) = q(x, y) \alpha(x, y) + \mathbf{1}_{y = x} (1 - \sum_{z \ne x} q(x,z) \alpha(x,z))$. Detailed balance for $x \ne y$:
$$
\pi(x) p(x, y) = \pi(x) q(x, y) \alpha(x, y) = \min(\pi(x) q(x, y), \pi(y) q(y, x)).
$$
Symmetric in $(x, y)$. $\square$

**Why this works.** Only need $\pi(y)/\pi(x)$ — normalizing constants cancel. So we can sample from non-normalized targets.

### 7.3 Random-walk Metropolis (RWM)

For $S = \mathbb{R}^d$ and $\pi$ with a density, use proposal $y = x + \varepsilon Z$, $Z \sim \mathcal{N}(0, I)$, $\varepsilon$ step size. Acceptance: $\min(1, \pi(y)/\pi(x))$ (proposal is symmetric).

**Optimal scaling.** Roberts-Gelman-Gilks (1997): for high-$d$ target with iid components, optimal $\varepsilon$ is $2.38/\sqrt{d}$, achieving acceptance $\approx 0.234$. Beyond-iid: tune to match this.

### 7.4 Gibbs sampling

For $\vec{X} = (X_1, \dots, X_d)$, update one component at a time:
$$
X_i^{(n+1)} \sim \pi(X_i | X_{-i} = x_{-i}^{(n)}),
$$
using the **full conditional** distribution. Cycling through $i = 1, \dots, d$, then repeating.

**Claim 7.2.** Gibbs is a special case of Metropolis-Hastings with proposal $= $ full conditional, giving acceptance probability 1. $\pi$ is stationary.

**Blocked Gibbs.** Update groups of coordinates at a time for better mixing.

### 7.5 Hamiltonian Monte Carlo (HMC)

For $\pi(x) \propto e^{-U(x)}$, introduce momentum $p$, sample from the joint $\pi(x, p) \propto e^{-U(x) - |p|^2/2}$. Use leapfrog integration of Hamilton's equations to propose far-reaching moves with high acceptance. Used in Stan, PyMC — the workhorse for Bayesian modeling. Mixing can be orders of magnitude faster than RWM in high dimensions.

### 7.6 Convergence diagnostics

MCMC in practice is tricky: it's easy to be fooled by slow mixing.
- **Burn-in**: discard first $n_0$ samples.
- **Gelman-Rubin $\hat R$**: run multiple chains; compare within-chain and between-chain variances. $\hat R \approx 1$ = converged.
- **Effective sample size (ESS)**: $n / (1 + 2 \sum_k \rho_k)$ where $\rho_k$ is lag-$k$ autocorrelation. Quantifies the "iid-equivalent" sample count.
- **Traceplots, autocorrelation plots, posterior predictive checks.**

---

## 8. Continuous-Time Markov Chains

Brief preview (full theory in Module 3 / Subject 3).

**Definition 8.1.** A continuous-time Markov chain (CTMC) on $S$ has an infinitesimal **generator matrix** $Q = (q(x, y))_{x, y \in S}$ with $q(x, y) \ge 0$ for $x \ne y$ and $\sum_y q(x, y) = 0$ (rows sum to zero). The $P$-matrix at time $t$ is $P_t = e^{tQ}$.

**Stationary distribution.** $\pi$ is stationary iff $\pi Q = 0$. Detailed balance: $\pi(x) q(x, y) = \pi(y) q(y, x)$.

**Example** (birth-death process)**.** $q(x, x+1) = \lambda_x$, $q(x, x-1) = \mu_x$, $q(x, x) = -(\lambda_x + \mu_x)$. Stationary: $\pi(x) = \pi(0) \prod_{k=1}^x \lambda_{k-1}/\mu_k$. M/M/1 queue is this with constants.

Applications in finance: credit rating transitions (see §9.4), order book dynamics, interest rate dynamics (CIR as a diffusive CTMC limit).

---

## 9. Python: MCMC in Action

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# ================================================================
# 1. Finite Markov chain: simulate and estimate stationary distribution
# ================================================================
# Random 3-state chain
P = np.array([
    [0.5, 0.3, 0.2],
    [0.1, 0.7, 0.2],
    [0.3, 0.3, 0.4],
])

def simulate_chain(P, x0, n):
    S = P.shape[0]
    states = [x0]
    for _ in range(n):
        states.append(rng.choice(S, p=P[states[-1]]))
    return np.array(states)

path = simulate_chain(P, 0, 100000)
print("Finite-state chain empirical distribution:")
for s in range(3):
    print(f"  π̂({s}) = {np.mean(path == s):.4f}")

# Exact stationary: solve πP = π
eigvals, eigvecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigvals - 1))
pi = eigvecs[:, idx].real
pi /= pi.sum()
print(f"\nExact π: {pi}")

# ================================================================
# 2. Metropolis-Hastings for a mixture of Gaussians
# ================================================================
def target_log_density(x):
    return np.log(0.3 * np.exp(-(x - 2)**2 / 2) +
                  0.7 * np.exp(-(x + 1)**2 / (2 * 0.25)))

def metropolis(target_log, x0, n_steps, step_size=1.0):
    samples = [x0]
    accepts = 0
    for _ in range(n_steps):
        y = samples[-1] + rng.normal() * step_size
        log_alpha = target_log(y) - target_log(samples[-1])
        if np.log(rng.uniform()) < log_alpha:
            samples.append(y)
            accepts += 1
        else:
            samples.append(samples[-1])
    return np.array(samples), accepts / n_steps

samples, rate = metropolis(target_log_density, 0.0, 50000, step_size=2.0)
print(f"\nMetropolis on Gaussian mixture:")
print(f"  Acceptance rate: {rate:.3f}")
print(f"  Empirical mean: {samples[5000:].mean():.3f} (target: 0.3*2+0.7*(-1)=-0.1)")
print(f"  Empirical std: {samples[5000:].std():.3f}")

# ================================================================
# 3. Gibbs sampler for bivariate normal
# ================================================================
def gibbs_bivariate(n, rho=0.7):
    """Sample from N(0, [[1, rho], [rho, 1]])."""
    samples = np.zeros((n, 2))
    x, y = 0.0, 0.0
    for i in range(n):
        x = rng.normal(rho * y, np.sqrt(1 - rho**2))
        y = rng.normal(rho * x, np.sqrt(1 - rho**2))
        samples[i] = [x, y]
    return samples

gibbs = gibbs_bivariate(50000)
print(f"\nGibbs on bivariate N with ρ=0.7:")
print(f"  Sample covariance:\n{np.cov(gibbs.T)}")
print(f"  (target: [[1, 0.7], [0.7, 1]])")

# ================================================================
# 4. MCMC autocorrelation and effective sample size
# ================================================================
def autocorr(x, max_lag=100):
    x = x - x.mean()
    result = np.correlate(x, x, mode='full')[len(x)-1:]
    return result[:max_lag] / result[0]

ac = autocorr(samples[5000:], max_lag=100)
# Integrated autocorrelation time
tau_int = 1 + 2 * np.sum(ac[1:50])
ess = len(samples[5000:]) / tau_int
print(f"\nAutocorrelation analysis:")
print(f"  τ_int ≈ {tau_int:.2f}")
print(f"  ESS ≈ {ess:.0f} out of {len(samples[5000:])} samples")

# ================================================================
# 5. Simulate SRW on Z^2 and show recurrence (returns to origin)
# ================================================================
def srw_2d(n):
    steps_x = rng.choice([-1, 1, 0, 0], n)
    steps_y = rng.choice([0, 0, -1, 1], n)
    # Adjust: at each step, choose one of 4 directions
    # Use a cleaner version:
    dirs = rng.integers(0, 4, n)
    dx = np.where(dirs == 0, 1, np.where(dirs == 1, -1, 0))
    dy = np.where(dirs == 2, 1, np.where(dirs == 3, -1, 0))
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    returns = np.sum((x == 0) & (y == 0))
    return returns

n_steps = 100000
returns = srw_2d(n_steps)
print(f"\nSRW on Z² returned to origin {returns} times in {n_steps} steps")

# ================================================================
# 6. Markov chain CLT: autocorrelation-aware variance
# ================================================================
# For an AR(1) process X_{n+1} = phi X_n + epsilon, stationary var = sigma^2/(1-phi^2),
# long-run variance for sample mean = sigma^2 / (1-phi)^2
phi = 0.8
sigma_eps = 1.0
stat_var = sigma_eps**2 / (1 - phi**2)
lr_var_formula = sigma_eps**2 / (1 - phi)**2

def ar1(n, phi, sigma):
    X = np.zeros(n)
    X[0] = rng.normal(0, np.sqrt(sigma**2/(1-phi**2)))
    for i in range(1, n):
        X[i] = phi * X[i-1] + rng.normal(0, sigma)
    return X

X = ar1(10000, phi, sigma_eps)
# Batch-mean variance estimate of mean
batches = X.reshape(-1, 100).mean(axis=1)
empirical_lr_var = 100 * batches.var()  # batch × batch size
print(f"\nMarkov chain CLT (AR(1), φ=0.8):")
print(f"  Stationary var = {stat_var:.4f}")
print(f"  Naive iid variance of mean (n=10000) = σ_stat²/10000 = {stat_var/10000:.5f}")
print(f"  Long-run variance (CLT) / n = {lr_var_formula/10000:.5f}")
print(f"  Empirical batch variance / n = {empirical_lr_var/10000:.5f}")
# The AR(1) has much larger variance than iid due to correlation
```

---

## 10. [QUANT APPLICATION] Markov Chains in Quantitative Finance

### 10.1 Credit rating transitions

Credit ratings (AAA, AA, A, ..., D) are modeled as a discrete-state Markov chain. Transition matrix $T$ is estimated from historical data:
$$
T = \begin{pmatrix}
P(\text{AAA} \to \text{AAA}) & P(\text{AAA} \to \text{AA}) & \cdots \\
P(\text{AA} \to \text{AAA}) & P(\text{AA} \to \text{AA}) & \cdots \\
\vdots & & \ddots
\end{pmatrix}.
$$
Applications:
- **Default probability** of a bond rated $X$ after $k$ years: $(T^k)_{X, D}$.
- **Risk-neutral transition matrix** (JLT model, Jarrow-Lando-Turnbull 1997): scale the physical $T$ to match observed credit spreads.
- **Portfolio credit risk**: simulate transitions for a correlated portfolio, compute loss distribution.

### 10.2 Interest rate chains (regime-switching)

Hamilton (1989) introduced Markov regime-switching. Interest rate $r_t$ follows different dynamics in "high-vol" vs "low-vol" regimes, with a hidden Markov chain governing the regime. Calibration via EM / Kalman filter (Module 2.2).

Applications:
- Yield curve modeling with regime changes.
- Option pricing under regime-switching (Naik 1993, Bollen 1998).
- Default intensity modeling.

### 10.3 MCMC for Bayesian asset allocation

Black-Litterman + full Bayesian: posterior on expected returns $\mu$ given historical data + views is high-dimensional and non-Gaussian. MCMC (often HMC in Stan/PyMC) is the standard tool.

Sample posterior $\pi(\mu, \Sigma | \text{data}, \text{views})$; for each posterior draw $(\mu, \Sigma)$, compute optimal portfolio weights; average. This gives a **posterior predictive** portfolio that accounts for parameter uncertainty.

### 10.4 Hidden Markov models for high-frequency trading

Microstructure: observed prices follow a mixture model with hidden Markov regime (bull / bear / neutral). HMM fitted via Baum-Welch (EM) gives regime probabilities; trading signals based on regime detection.

### 10.5 Markov-functional models for interest rate derivatives

Hunt-Kennedy-Pelsser (2000): model the short rate as a function of a driver Markov process. Combines tractability (of the driver) with flexibility (of the functional). Used for Bermudan swaptions pricing.

### 10.6 MCMC pricing of exotic options

For a high-dimensional path-dependent payoff, direct MC is slow. MCMC on the joint path space with target $\pi(\text{path}) \propto \text{payoff}(\text{path}) \cdot P(\text{path})$ gives variance reduction via importance sampling (if $\text{payoff}$ is concentrated on rare paths, weighting accordingly).

### 10.7 Stationary distributions and long-term portfolio behavior

For an ergodic return process (GARCH with stationary volatility), the portfolio value $W_t$ under a constant-mix strategy has a stationary distribution. Long-run growth rate is $\lim_{t \to \infty} t^{-1} \log W_t \xrightarrow{\text{a.s.}} g_\pi$ by ergodic theorem (Kelly-optimal growth rate under stationarity).

### 10.8 Monte Carlo in stochastic control (ADP, Q-learning)

For an optimal trading / execution problem, Bellman's equation is
$$
V(x) = \max_a \{r(x, a) + \gamma \sum_y p(y | x, a) V(y)\}.
$$
Approximate Dynamic Programming (ADP) + MCMC: simulate the chain under various policies, estimate $V$ via temporal-difference learning. Convergence of TD($\lambda$): Robbins-Monro stochastic approximation (Module 2.6 exercise) applied to a Markov chain. Foundation of reinforcement learning for algorithmic trading (Hambly-Xu 2023).

---

## 11. Worked Examples

### Example 11.1 (Ehrenfest urn)

$N$ balls distributed between two urns. At each step, pick one ball uniformly and move it to the other urn. State = number of balls in urn 1, $S = \{0, 1, \dots, N\}$. Transition: $p(k, k+1) = (N-k)/N$, $p(k, k-1) = k/N$. Stationary distribution: Binomial$(N, 1/2)$ (check detailed balance). Irreducible, period 2 (not aperiodic, so no convergence in TV — instead use time-$(2n)$ marginals).

### Example 11.2 (Birth-death process / M/M/1 queue)

State $= $ number in queue. $\lambda_k = \lambda$, $\mu_k = \mu$ for $k \ge 1$. Stationary: $\pi(k) = (1 - \rho) \rho^k$ for $\rho := \lambda/\mu < 1$ (geometric). Positive recurrent iff $\rho < 1$, null recurrent at $\rho = 1$, transient at $\rho > 1$.

### Example 11.3 (PageRank)

Web graph: random walk + damping with probability $1 - d$ of jumping to a uniform page. Transition matrix is $d P + (1-d) (1/N) \mathbf{1} \mathbf{1}^\top$. Always irreducible (due to teleportation), with unique stationary $\pi = $ PageRank vector. Google's original algorithm.

### Example 11.4 (Reversible chain on a graph for spectral analysis)

Random walk on graph $G$: $p(x, y) = \mathbf{1}_{x \sim y}/\deg(x)$. Reversible w.r.t. $\pi(x) \propto \deg(x)$. Eigenvalues of $P$ are related to those of the **normalized graph Laplacian** $L = I - D^{-1/2} A D^{-1/2}$. Spectral gap of $P$ = $1 - \lambda_2$ where $\lambda_2$ is the second-largest eigenvalue. Expanders = graphs with spectral gap bounded away from 0.

### Example 11.5 (Ising model MCMC)

Target $\pi(\sigma) \propto e^{-\beta H(\sigma)}$ on spin configurations $\sigma \in \{-1, +1\}^L$. Glauber dynamics (single-spin-flip Metropolis): at high $T$ (high $\beta^{-1}$), mixing is fast; at low $T$, below critical $\beta_c$, exponentially slow (multiple modes). Illustrates how physical phase transitions affect MCMC convergence.

---

## 12. Exercises

### Tier ★

**2.7.E1.** For the 2-state chain $p(1, 1) = 1 - a$, $p(1, 2) = a$, $p(2, 1) = b$, $p(2, 2) = 1 - b$ ($a, b \in (0, 1)$), find $\pi$ and verify it's stationary. Eigenvalues of $P$?

**2.7.E2.** Show: an irreducible chain on a finite state space is always positive recurrent.

**2.7.E3.** For SRW on $\mathbb{Z}$: compute $p^{(2n)}(0, 0) = \binom{2n}{n} 2^{-2n}$. Using Stirling, show $\sum_n p^{(2n)}(0, 0) = \infty$, hence recurrence.

**2.7.E4.** Show that irreducibility + aperiodicity + finite state space implies that $P^n$ converges to the stationary projection $\mathbf{1}\pi^\top$ at geometric rate.

**2.7.E5.** Simulate a Markov chain and verify the ergodic theorem for a specific $f$. Compare empirical $n^{-1} \sum f(X_k)$ to $\pi(f)$ for several $n$.

**2.7.E6.** Prove: if $\pi_1, \pi_2$ are stationary distributions for an irreducible chain, then $\pi_1 = \pi_2$. (Hence uniqueness.)

**2.7.E7.** For the Ehrenfest urn with $N = 4$, write down the full transition matrix, find eigenvalues, and compute mixing time to within $\varepsilon = 0.01$ TV.

**2.7.E8.** Given a 3-state chain: set up Metropolis-Hastings with a symmetric proposal to sample from $\pi = (0.2, 0.5, 0.3)$. Verify detailed balance.

### Tier ★★

**2.7.E9** (Harris recurrence)**.** For a general state space chain (not just countable), define Harris recurrence: $P_x(X \text{ visits } A \text{ i.o.}) = 1$ for every $x$ and every $A$ with $\pi(A) > 0$. Give conditions for Harris recurrence (minorization / drift).

**2.7.E10** (Lyapunov drift and positive recurrence)**.** Let $V: S \to [0, \infty)$ with $V(x) \to \infty$ as $x$ leaves a finite set. If $(PV)(x) \le V(x) - 1 + b \mathbf{1}_C(x)$ for a finite set $C$ and constant $b > 0$, then the chain is positive recurrent with $\sum_x \pi(x) V(x) < \infty$. (Foster-Lyapunov theorem.)

**2.7.E11** (Kemeny-Snell fundamental matrix)**.** For a finite irreducible chain, define $Z := (I - P + \mathbf{1}\pi^\top)^{-1}$. Prove: $E_x[T_y] = (Z_{yy} - Z_{xy})/\pi(y)$.

**2.7.E12** (Doeblin's condition)**.** Assume there is $\varepsilon > 0$, $n_0 \ge 1$, and a probability measure $\nu$ with $p^{(n_0)}(x, \cdot) \ge \varepsilon \nu(\cdot)$ for all $x$. Prove: $\|P^n(\mu, \cdot) - \pi\|_{TV} \le (1 - \varepsilon)^{\lfloor n/n_0 \rfloor}$. (Doeblin's classical mixing rate.)

**2.7.E13** (Minorization condition for MCMC)**.** For the RWM with Gaussian proposals on a target with compactly-supported log-concave density, show Doeblin's condition holds. Deduce geometric ergodicity.

**2.7.E14** (Markov chain CLT via Poisson equation)**.** For an AR(1) $X_{n+1} = \phi X_n + \varepsilon_n$ with $\varepsilon_n \sim \mathcal{N}(0, 1)$ iid, $|\phi| < 1$, find the stationary distribution, solve Poisson equation for $f(x) = x$, and compute $\sigma_f^2 = $ long-run variance of sample mean.

**2.7.E15** (Birkhoff ergodic theorem)**.** State and prove the Birkhoff ergodic theorem: for a measure-preserving transformation $T$ on $(\Omega, \mathcal{F}, \mu)$ with $\mu$ a probability measure and $T$ ergodic, $n^{-1} \sum_{k=0}^{n-1} f(T^k \omega) \to \int f d\mu$ a.e. for $f \in L^1$. (Our Markov chain ergodic theorem is a special case.)

**2.7.E16** (Stationary Gaussian AR(p))**.** For an AR(p) process $X_n = \phi_1 X_{n-1} + \dots + \phi_p X_{n-p} + \varepsilon_n$, show it is a Markov chain on $\mathbb{R}^p$ (vector state). Find conditions on $\phi$ for stationarity (roots of characteristic polynomial outside unit disk). Compute stationary covariance matrix via Lyapunov equation.

### Tier ★★★

**2.7.E17** (Rating transition matrix calibration)**.** Given historical rating data, estimate a Markov transition matrix $T$ via maximum likelihood. Then calibrate a risk-neutral transition matrix $\tilde{T}$ such that implied bond prices $P(X \to \text{D}, t)$ match market spreads — JLT model.

**2.7.E18** (HMC implementation)**.** Implement Hamiltonian Monte Carlo for a multivariate Gaussian target. Compare to RWM and show significantly faster mixing in high dimensions.

**2.7.E19** (Metropolis-adjusted Langevin, MALA)**.** The proposal is $y = x + (\varepsilon^2/2) \nabla \log \pi(x) + \varepsilon Z$ with $Z \sim \mathcal{N}(0, I)$. Show: accept-reject step corrects for discretization of Langevin dynamics $dX = \frac{1}{2}\nabla\log\pi \, dt + dW$. Show mixing is faster than RWM for smooth log-concave targets.

**2.7.E20** (Regime-switching Black-Scholes)**.** Let $r_t, \mu_t, \sigma_t$ follow a continuous-time Markov chain between two regimes. Derive the PDE for the European call price under regime-switching BS (system of coupled PDEs, one per regime). Solve numerically and compare to constant-regime BS.

**2.7.E21** (Perfect sampling / Coupling From The Past)**.** Implement Propp-Wilson's CFTP for a 2-state chain. CFTP produces exact samples from $\pi$ in finite time (no "how long is burn-in?" question). Verify empirically.

**2.7.E22** (MCMC for Bayesian SV model)**.** Implement a Gibbs sampler for a stochastic volatility model: $r_t = \sqrt{h_t} \varepsilon_t$, $\log h_t = \alpha + \beta \log h_{t-1} + \eta_t$, with $\varepsilon_t, \eta_t$ iid standard normal. Full conditionals: $(\alpha, \beta | \text{rest})$ conjugate normal, $(\log h_{1:T} | \text{rest})$ via Kim-Shephard-Chib mixture approximation. Apply to SPX daily returns, compare to GARCH.

---

## 13. Summary and Forward Pointers (Subject Capstone)

**What we proved in Module 2.7.**
- Markov chain formalism: transition kernels, Chapman-Kolmogorov, strong Markov property.
- State classification: irreducibility, periodicity, transience, recurrence, positive/null recurrence.
- Existence & uniqueness of stationary distributions (positive recurrent case).
- Reversibility and detailed balance.
- Ergodic theorem: time averages = space averages.
- Convergence to stationarity via coupling + spectral gap.
- Markov chain CLT via Poisson equation + martingale methods.
- MCMC: Metropolis-Hastings, Gibbs, HMC; convergence diagnostics.
- Continuous-time chains (preview).

**Subject 2 capstone recap.**
Over seven modules we built probability theory from the ground up:
- **2.1**: Probability spaces, distributions, independence, Borel-Cantelli, 0-1 laws.
- **2.2**: Expectation, variance, moment inequalities, conditional expectation via Radon-Nikodym.
- **2.3**: Four modes of convergence, UI, Scheffé, Kolmogorov's three-series.
- **2.4**: LLN (WLLN, SLLN), CLT (Lindeberg-Lévy, Lindeberg-Feller, Berry-Esseen).
- **2.5**: Characteristic functions, Lévy inversion and continuity, Bochner, stable/inf. divisible.
- **2.6**: Martingales, optional stopping, Doob's inequalities, martingale convergence.
- **2.7**: Markov chains, stationary distributions, ergodic theorem, MCMC.

These seven modules together form the mathematical foundation of all of stochastic analysis, statistical inference, and quantitative finance.

**Forward pointers to Subject 3: Stochastic Processes.**
- **3.1**: Brownian motion — continuous-time martingale + functional CLT limit (Donsker).
- **3.2**: Stochastic integration (Itô integral).
- **3.3**: Itô's lemma, Girsanov's theorem.
- **3.4**: SDEs: existence/uniqueness, PDE connections (Feynman-Kac).
- **3.5**: Lévy processes — continuous-time analog of Module 2.5.
- **3.6**: Markov processes in continuous time — strong Markov, semigroups, generators.
- **3.7**: Continuous-time martingale theory (Doob-Meyer, martingale representation).

The tools of Subject 2 are exactly the tools needed for Subject 3 — Brownian motion is the limit of random walks (Donsker), Itô integration relies on martingale decomposition (Doob), Girsanov's theorem is a change of measure (Radon-Nikodym + exponential martingale), Feynman-Kac is conditional expectation + PDE.

**Onward to Subject 3.** We now have all the probability foundations we need to build stochastic calculus, the language of derivative pricing, hedging, and modern continuous-time finance.

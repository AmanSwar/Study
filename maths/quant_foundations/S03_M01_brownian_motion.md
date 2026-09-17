# Module 3.1: Brownian Motion — Construction and Properties

**Subject 3: Stochastic Processes** · Module 1 of 7

---

## 0. Prerequisites and Position

- **Subject 1** (Measure theory): Lebesgue integration, product measures, $L^p$ spaces, Radon-Nikodym.
- **Subject 2** (Probability): random variables, modes of convergence, LLN/CLT, characteristic functions, martingales (2.6), Markov chains (2.7).
- Specifically: Kolmogorov extension theorem (2.1), martingale convergence + Doob (2.6), CFs and Lévy inversion (2.5).

Welcome to Subject 3 — **Stochastic Processes**, the continuous-time probability theory that forms the mathematical backbone of modern derivative pricing, statistical physics, stochastic control, and machine learning's continuous-time analyses.

**Brownian motion** (also called the **Wiener process**) is the central object. It is the continuous-time limit of random walks (Donsker's invariance principle), the prototypical continuous martingale, the only Lévy process with continuous paths, the Gaussian process with stationary independent increments, and the "noise" in every Itô SDE. Getting Brownian motion right — existence, properties, path regularity — is the foundation on which the rest of Subject 3 is built.

This module:
- **Definition** (finite-dimensional distributions + path continuity).
- **Existence**: three constructions — Lévy's iterative (Schauder-like), Kolmogorov extension + Kolmogorov-Chentsov regularity, Donsker's functional CLT.
- **Path properties**: continuous nowhere-differentiable, Hölder of order $< 1/2$, quadratic variation, local times.
- **Markov and strong Markov property**, reflection principle.
- **Martingale characterization** (Lévy's characterization).
- **Multidimensional and geometric Brownian motion**.
- Applications: Black-Scholes derivation, hitting times, Brownian bridge.

---

## 1. Definition

**Definition 1.1** (Standard Brownian motion)**.** A stochastic process $B = (B_t)_{t \ge 0}$ on $(\Omega, \mathcal{F}, P)$ is a **standard Brownian motion** (or **Wiener process**) if:

(B1) $B_0 = 0$ a.s.

(B2) For $0 = t_0 < t_1 < \dots < t_n$, the increments $B_{t_1} - B_{t_0}, B_{t_2} - B_{t_1}, \dots, B_{t_n} - B_{t_{n-1}}$ are **independent**.

(B3) For $s < t$, $B_t - B_s \sim \mathcal{N}(0, t - s)$.

(B4) $t \mapsto B_t(\omega)$ is **continuous** for $P$-a.e. $\omega$.

If (B1)-(B3) hold but not necessarily (B4), we call $B$ a **Brownian motion in distribution** (or a process with the BM *finite-dimensional distributions*).

**Equivalent formulations.**
- Gaussian process with $E[B_t] = 0$, $\text{Cov}(B_s, B_t) = s \wedge t$, continuous paths.
- Continuous martingale with quadratic variation $\langle B\rangle_t = t$ (Lévy's characterization — §6).
- Functional CLT limit of rescaled simple random walks (Donsker — §5).
- Unique Lévy process with continuous paths (Subject 3 Module 5).

The essential content: BM has Gaussian increments (scaling with $\sqrt{t-s}$), independent increments, and the technical miracle (B4): its paths are continuous functions of time.

### 1.1 Why existence is nontrivial

The finite-dimensional distributions (fdd's) specified by (B1)-(B3) are all Gaussian and consistent, so by Kolmogorov's extension theorem (Module 2.1) there exists a probability measure on $\mathbb{R}^{[0, \infty)}$ giving these fdd's. The difficulty: the set of continuous paths $C[0, \infty)$ is **not measurable** in the product $\sigma$-algebra (this $\sigma$-algebra sees only countably many coordinates at a time). So we cannot assert path-continuity from fdd's alone — we must modify the process on a null set of each time slice to obtain a continuous version.

This is the **path regularization problem**, and it has two classical solutions: Lévy's direct construction on dyadic rationals (§2), and Kolmogorov's continuity criterion (§3) applied to the Kolmogorov-extension process.

---

## 2. Lévy's Construction via Haar / Schauder Basis

This is the most elegant existence proof: we build BM explicitly on $[0, 1]$ as a random series in a wavelet basis. It gives continuity for free (uniform convergence) and exposes why BM exists.

### 2.1 Schauder functions

Enumerate the dyadic rationals $\{k/2^n : 0 \le k \le 2^n\}$ and define Haar / Schauder functions supported on shrinking dyadic intervals.

For $n \ge 0$ and $1 \le k \le 2^n$ with $k$ odd, define the **Schauder function**
$$
s_{n, k}(t) := \begin{cases}
2^{(n-1)/2} \cdot (t - (k-1)/2^n) & (k-1)/2^n \le t \le k/2^n \\
2^{(n-1)/2} \cdot ((k+1)/2^n - t) & k/2^n \le t \le (k+1)/2^n \\
0 & \text{otherwise.}
\end{cases}
$$

Also include the linear function $s_0(t) := t$ and the "boundary" function. Together with appropriate normalization, these form a complete orthonormal basis (via integration) of a Hilbert space giving BM structure.

### 2.2 Construction

Let $Z_0, Z_{n, k}$ be iid standard normal random variables (on some probability space). Define:
$$
B_t := Z_0 \cdot t + \sum_{n = 0}^\infty \sum_{k \text{ odd}, 1 \le k \le 2^n} Z_{n, k} s_{n, k}(t).
$$

**Theorem 2.1** (Lévy's construction)**.** This series converges uniformly in $t \in [0, 1]$ almost surely, and the limit $(B_t)_{t \in [0, 1]}$ is a standard Brownian motion on $[0, 1]$.

*Proof sketch.*

**Step 1: Uniform convergence.** Partial sum over level $n$:
$$
U_n(t) := \sum_{k \text{ odd}, 1 \le k \le 2^n} Z_{n, k} s_{n, k}(t).
$$
The Schauder functions at level $n$ have disjoint supports (up to a measure-zero overlap), each with peak value $2^{(n-1)/2} \cdot 2^{-n-1} = 2^{-(n+3)/2}$... no wait, peak value is $2^{(n-1)/2} \cdot 2^{-n} \cdot (\text{something})$... recomputing: at the peak $t = k/2^n$, $s_{n,k}(k/2^n) = 2^{(n-1)/2} \cdot 1/2^n = 2^{-n/2 - 1/2}$. So $\|s_{n,k}\|_\infty = 2^{-(n+1)/2}$.

Disjoint supports at each level mean $\|U_n\|_\infty = \max_k |Z_{n,k}| \cdot 2^{-(n+1)/2}$. There are $2^{n-1}$ Schauder functions at level $n$ (odd $k$ in $[1, 2^n]$), so the expected maximum of $2^{n-1}$ standard Gaussians is $\sim \sqrt{2 \log 2^{n-1}} = \sqrt{2(n-1) \log 2}$. By Borel-Cantelli, $\max_k |Z_{n,k}| \le C \sqrt{n}$ eventually almost surely, so
$$
\|U_n\|_\infty \le C \sqrt{n} \cdot 2^{-(n+1)/2} \to 0
$$
geometrically. The series $\sum_n U_n$ converges absolutely and uniformly a.s.

**Step 2: Gaussian increments.** Finite-dimensional distributions of partial sums are joint Gaussian (linear combinations of iid Gaussians). The covariance structure at dyadic points can be directly computed: for $s, t$ dyadic,
$$
E[B_s B_t] = st + \sum_n \sum_k s_{n,k}(s) s_{n,k}(t) = s \wedge t
$$
by the reproducing-kernel property of the Schauder basis for $L^2$ integration — specifically, $\int_0^1 (\mathbf{1}_{[0,s]})' (\mathbf{1}_{[0,t]})' \, du = s \wedge t$, and the Schauder basis diagonalizes this in $L^2$.

**Step 3: Continuity extends from dyadics.** Uniform convergence of continuous partial sums gives a continuous limit. By density of dyadics in $[0, 1]$ + continuity, the Gaussian increment structure extends.

$\square$

### 2.3 Extension to $[0, \infty)$

Patch together independent BM copies on each $[n, n+1]$: $B_t := B_n^{(n)} + \tilde{B}_{t - n}^{(n+1)}$ for $t \in [n, n+1]$, conditioning on $\tilde{B}^{(n+1)}$ being independent BM restarted at $B_n^{(n)}$.

---

## 3. Kolmogorov-Chentsov Continuity Criterion

Alternative (and more general) route: start from Kolmogorov extension + Kolmogorov-Chentsov to get a Hölder-continuous version.

**Theorem 3.1** (Kolmogorov-Chentsov)**.** Let $(X_t)_{t \in [0, T]}$ be a process such that
$$
E[|X_t - X_s|^\alpha] \le C |t - s|^{1 + \beta}
$$
for some $\alpha, \beta, C > 0$ and all $s, t$. Then $X$ has a continuous modification, and in fact a modification with Hölder paths of any exponent $\gamma < \beta/\alpha$.

*Proof sketch.* Work on dyadic rationals $D_n := \{k/2^n\}$. Let $M_n := \max_k |X_{k/2^n} - X_{(k-1)/2^n}|$. By Markov + union bound,
$$
P(M_n > 2^{-n\gamma}) \le 2^n \cdot \frac{E|X_{k/2^n} - X_{(k-1)/2^n}|^\alpha}{2^{-n\gamma\alpha}} \le 2^n \cdot \frac{C \cdot 2^{-n(1+\beta)}}{2^{-n\gamma\alpha}} = C \cdot 2^{-n(\beta - \gamma\alpha)}.
$$
For $\gamma < \beta/\alpha$, this is summable in $n$. Borel-Cantelli: $M_n \le 2^{-n\gamma}$ eventually a.s. This implies $X$ is Hölder-$\gamma$ on the dyadics, hence extends continuously to $[0, T]$. $\square$

### 3.1 Application to BM

For BM, $B_t - B_s \sim \mathcal{N}(0, t-s)$, so
$$
E|B_t - B_s|^{2m} = (2m - 1)!! \cdot (t - s)^m = O(|t - s|^m).
$$
With $\alpha = 2m$, $\beta = m - 1$: Kolmogorov-Chentsov gives Hölder exponent $\gamma < (m-1)/(2m) = 1/2 - 1/(2m)$. Letting $m \to \infty$:

**Corollary 3.2.** Brownian paths are a.s. Hölder-$\gamma$ for every $\gamma < 1/2$.

But they are *not* Hölder-$1/2$ — see §4.3 for the sharp result (modulus of continuity).

---

## 4. Path Properties

### 4.1 Continuous nowhere-differentiable

**Theorem 4.1** (Paley-Wiener-Zygmund)**.** Almost every Brownian path is differentiable at no point:
$$
P(\exists t : B'_t \text{ exists and is finite}) = 0.
$$

*Proof sketch.* Fix $\varepsilon > 0$ and for $t \in [0, 1]$ and each integer $n$, define the event
$$
A_n(t) := \{|B_{t + k/n} - B_t| \le \varepsilon \cdot k/n \text{ for } k = 1, 2, 3\}.
$$
If $B'_t$ exists and is bounded by $\varepsilon$, then for large $n$, $A_n(t)$ holds. The probability of the three constraints: $|Z_1|, |Z_2 - Z_1|, |Z_3 - Z_2| \le \varepsilon/\sqrt{n}$ where $Z_k \sim \mathcal{N}(0, k)$; bounded by $(\varepsilon/\sqrt{n})^3$. Intersect over $t$ in $[0, 1]$: the event of some $t \in [0, 1]$ satisfying the condition has probability $\le n \cdot (\varepsilon/\sqrt{n})^3 = \varepsilon^3/\sqrt{n} \to 0$.

Union over $\varepsilon \in \mathbb{N}^{-1}$ bounds and $t \in [0, T]$ rationals gives the result. $\square$

**Intuition.** BM has infinitely many oscillations at every time scale; the paths are "rough" enough that zooming in reveals more wiggles. Quantitatively: $B_{t+h} - B_t \sim \sqrt{h}$, not $\sim h$, so $(B_{t+h} - B_t)/h \sim 1/\sqrt{h} \to \infty$.

### 4.2 Quadratic variation

For a partition $\pi = \{0 = t_0 < t_1 < \dots < t_n = T\}$ of $[0, T]$, define
$$
V^2_\pi(B) := \sum_{k=0}^{n-1} (B_{t_{k+1}} - B_{t_k})^2.
$$
The **mesh** is $|\pi| := \max_k (t_{k+1} - t_k)$.

**Theorem 4.2.** For any sequence of partitions $\pi_n$ with $|\pi_n| \to 0$, $V^2_{\pi_n}(B) \xrightarrow{P} T$. If $\sum_n |\pi_n| < \infty$ (e.g., dyadic partitions with $|\pi_n| = T/2^n$), then $V^2_{\pi_n}(B) \xrightarrow{\text{a.s.}} T$.

*Proof.* $E[V^2_\pi(B)] = \sum_k E[(B_{t_{k+1}} - B_{t_k})^2] = \sum_k (t_{k+1} - t_k) = T$. Variance: increments are independent Gaussian, so
$$
\text{Var}(V^2_\pi(B)) = \sum_k \text{Var}((B_{t_{k+1}} - B_{t_k})^2) = \sum_k 2(t_{k+1} - t_k)^2 \le 2 |\pi| T \to 0.
$$
Chebyshev: $V^2_{\pi_n}(B) \xrightarrow{P} T$. For a.s. convergence along a fast sequence of partitions: Borel-Cantelli on $P(|V^2_{\pi_n} - T| > \varepsilon) \le 2 |\pi_n| T / \varepsilon^2$. Summable if $\sum |\pi_n| < \infty$. $\square$

**Consequence: bounded variation fails.** The quadratic variation is $T > 0$, but the **total variation** $V^1(B) = \sup_\pi \sum_k |B_{t_{k+1}} - B_{t_k}|$ is infinite for BM. Key inequality:
$$
V^2_\pi(B) \le V^1(B) \cdot \max_k |B_{t_{k+1}} - B_{t_k}|.
$$
If $V^1 < \infty$, then by path-continuity $\max_k |B_{t_{k+1}} - B_{t_k}| \to 0$, so $V^2 \to 0 \ne T$. Contradiction; hence $V^1 = \infty$ a.s.

**Quadratic variation is the key non-triviality.** It means you cannot define a Riemann-Stieltjes integral $\int f(t) \, dB_t$ in the classical sense — you need Itô's stochastic integral (Module 3.2).

### 4.3 Law of iterated logarithm (LIL)

**Theorem 4.3** (Khinchin LIL)**.** $\limsup_{t \downarrow 0} \frac{B_t}{\sqrt{2 t \log \log(1/t)}} = 1$ a.s., and $\liminf = -1$ a.s.

This is the sharp modulus of continuity near $t = 0$ (and similarly at infinity). It says paths fluctuate by roughly $\sqrt{t \log \log(1/t)}$ near any time.

**Theorem 4.4** (Lévy's modulus of continuity)**.** $\limsup_{h \downarrow 0} \frac{\max_{0 \le t \le 1 - h} |B_{t+h} - B_t|}{\sqrt{2 h \log(1/h)}} = 1$ a.s.

This is the *uniform* modulus — paths are $\sqrt{2h \log(1/h)}$-Hölder uniformly but not better.

### 4.4 Arcsine laws

**Theorem 4.5** (Arcsine law for last zero)**.** Let $L_t := \sup\{s \le t : B_s = 0\}$. Then $L_1/1 \sim \text{Arcsine}(0, 1)$, i.e., density $1/(\pi \sqrt{x(1-x)})$.

**Theorem 4.6** (Arcsine law for occupation time)**.** Let $A_t := \int_0^t \mathbf{1}_{B_s > 0} ds$ (time spent positive). Then $A_1 / 1 \sim \text{Arcsine}(0, 1)$.

**Interpretation.** Counter-intuitively, BM is most likely to spend nearly *all* its time on one side of zero — the mode of the arcsine is at 0 and 1, not 1/2. Paths are not "balanced" around zero.

---

## 5. Donsker's Invariance Principle

Brownian motion is the *functional CLT limit* of simple random walks.

### 5.1 Setup

Let $X_1, X_2, \dots$ be iid with $E[X_1] = 0$, $\text{Var}(X_1) = 1$. Define the polygonal process
$$
W^{(n)}(t) := \frac{1}{\sqrt{n}}\left(S_{\lfloor nt \rfloor} + (nt - \lfloor nt\rfloor)(S_{\lfloor nt\rfloor + 1} - S_{\lfloor nt\rfloor})\right), \quad t \in [0, 1],
$$
where $S_k = X_1 + \dots + X_k$. $W^{(n)} \in C[0, 1]$ (continuous piecewise linear).

**Theorem 5.1** (Donsker's theorem)**.** $W^{(n)} \Rightarrow B$ in $C[0, 1]$, where $B$ is standard Brownian motion and $\Rightarrow$ is weak convergence of probability measures on $C[0, 1]$ equipped with uniform topology.

*Proof sketch.*
**Step 1: Finite-dimensional convergence.** Use multivariate CLT (Module 2.4, Exercise 12): for any $0 < t_1 < \dots < t_k \le 1$,
$$
(W^{(n)}(t_1), \dots, W^{(n)}(t_k)) \xRightarrow{d} (B_{t_1}, \dots, B_{t_k}).
$$

**Step 2: Tightness in $C[0, 1]$.** Show for every $\varepsilon > 0$, there exists $\delta$ with
$$
\sup_n P\Big(\sup_{|s - t| < \delta} |W^{(n)}(t) - W^{(n)}(s)| > \varepsilon\Big) < \varepsilon.
$$
This is the Prokhorov condition for tightness on $C[0, 1]$. Apply Doob's maximal inequality or direct estimates using $E|X_1|^4 < \infty$ (or $L^2$ with care).

**Step 3: Apply Prokhorov.** Tightness + fdd convergence → weak convergence. $\square$

### 5.2 Consequence for BM existence

Run Donsker in reverse: if you believe iid random walks exist (obvious), Donsker plus tightness give you a probability measure on $C[0, 1]$ giving BM — a third construction.

### 5.3 Applications via continuous mapping

Donsker + continuous mapping theorem (Module 2.3, Theorem 8.3) gives:

**Corollary 5.2.**
- $\max_{0 \le t \le 1} W^{(n)}(t) \xRightarrow{d} \max_{0 \le t \le 1} B_t$.
- $\int_0^1 W^{(n)}(t)^2 \, dt \xRightarrow{d} \int_0^1 B_t^2 \, dt$.

Explicit limit laws: $\max_{0 \le t \le 1} B_t \sim |\mathcal{N}(0, 1)|$ (reflection principle, §7).

---

## 6. Lévy's Characterization

A beautiful converse: continuous martingales with the right quadratic variation are Brownian motions.

**Theorem 6.1** (Lévy's characterization)**.** Let $M = (M_t)$ be a continuous process with $M_0 = 0$ such that both $M$ and $M_t^2 - t$ are martingales (w.r.t. some filtration $\mathbb{F}$). Then $M$ is a standard Brownian motion.

*Proof.* We show $M$ has Gaussian increments $\mathcal{N}(0, t-s)$ independent of $\mathcal{F}_s$.

Fix $s < t$, $u \in \mathbb{R}$. Consider the complex exponential $f_t(u) := E[e^{iu M_t}]$. By Itô's lemma (Module 3.3), $e^{iuM_t} = 1 + iu \int_0^t e^{iu M_r} dM_r - \frac{u^2}{2} \int_0^t e^{iuM_r} d\langle M\rangle_r$. Given $\langle M\rangle_t = t$ (the second martingale condition),
$$
f_t(u) = 1 - \frac{u^2}{2} \int_0^t f_r(u) dr \implies f_t(u) = e^{-u^2 t / 2}.
$$
CF of $\mathcal{N}(0, t)$. More generally, the conditional CF $E[e^{iu(M_t - M_s)} | \mathcal{F}_s] = e^{-u^2(t-s)/2}$ by the same argument applied to $M_{s + \cdot} - M_s$, which is again a continuous martingale with $\langle\cdot\rangle = \cdot$. This gives conditional Gaussian increments independent of $\mathcal{F}_s$. $\square$

**Significance.** Lévy's theorem is extraordinarily useful: it lets you *recognize* Brownian motion without having to check Gaussianity directly. Many SDEs and random time changes produce processes that turn out to be Brownian by this characterization.

### 6.1 Example: Dambis-Dubins-Schwarz

**Theorem 6.2** (Dambis-Dubins-Schwarz, DDS)**.** Every continuous local martingale $M$ with $\langle M\rangle_\infty = \infty$ a.s. is a time-changed Brownian motion:
$$
M_t = B_{\langle M\rangle_t}
$$
for some Brownian motion $B$ (on a possibly extended filtration).

*Proof sketch.* Define $\tau_s := \inf\{t : \langle M\rangle_t > s\}$. Then $B_s := M_{\tau_s}$ is a continuous martingale (by OST for continuous martingales) with $\langle B\rangle_s = s$. Lévy: $B$ is a BM. And $M_t = B_{\langle M\rangle_t}$ by construction. $\square$

---

## 7. Markov Property and Reflection Principle

### 7.1 Markov property

**Theorem 7.1.** BM is a Markov process: $E[f(B_{t+s}) | \mathcal{F}_s] = P_t f(B_s)$ where $P_t f(x) := E[f(x + B_t)]$ is the **heat semigroup**:
$$
P_t f(x) = \int_\mathbb{R} f(x + y) \frac{e^{-y^2/(2t)}}{\sqrt{2\pi t}} dy.
$$

This follows directly from independent increments. Applying $\partial/\partial t$: $P_t f$ solves the **heat equation** $\partial_t u = \frac{1}{2} \partial_{xx} u$ with $u(0, x) = f(x)$. So BM is the Markov process associated with the Laplacian — the classical connection.

### 7.2 Strong Markov property

**Theorem 7.2** (Strong Markov)**.** For any $\mathbb{F}$-stopping time $\tau < \infty$ a.s., the shifted process $(B_{\tau + t} - B_\tau)_{t \ge 0}$ is a standard BM independent of $\mathcal{F}_\tau$.

*Proof sketch.* First for stopping times taking countably many values: direct from the (weak) Markov property. General case: approximate by $\tau_n := 2^{-n} \lceil 2^n \tau \rceil$, apply countable Markov, pass to the limit using path continuity. $\square$

### 7.3 Reflection principle

Let $M_t := \sup_{0 \le s \le t} B_s$ (running maximum).

**Theorem 7.3** (Reflection principle)**.** For $a > 0$ and $b \le a$,
$$
P(M_t \ge a, B_t \le b) = P(B_t \ge 2a - b).
$$

*Proof.* Let $\tau_a := \inf\{s : B_s = a\}$. By continuity, $\{M_t \ge a\} = \{\tau_a \le t\}$. By strong Markov at $\tau_a$: given $\tau_a$, the post-$\tau_a$ process $\tilde{B}_s := B_{\tau_a + s} - B_{\tau_a}$ is independent BM. By symmetry, $-\tilde{B}$ has the same distribution as $\tilde{B}$, so
$$
P(B_t \le b, \tau_a \le t) = P(B_t \le b, \tau_a \le t, \tilde{B}_{t - \tau_a} \le b - a)
$$
which by reflection ($-\tilde{B}$ in place of $\tilde{B}$) equals $P(B_t \ge 2a - b, \tau_a \le t)$. On $\{\tau_a \le t\}$, $M_t \ge a$ automatically. And $B_t \ge 2a - b \ge a$ implies $\tau_a \le t$. $\square$

**Corollaries.**
- $P(M_t \ge a) = 2 P(B_t \ge a) = 2(1 - \Phi(a/\sqrt{t}))$ (take $b = a$).
- Density of $M_t$: $f_{M_t}(a) = \sqrt{2/(\pi t)} e^{-a^2/(2t)}$ for $a > 0$.
- $M_t \sim |B_t| \sim |\mathcal{N}(0, t)|$ (in distribution).
- Hitting time density: $f_{\tau_a}(t) = a/\sqrt{2\pi t^3} e^{-a^2/(2t)}$ for $a > 0$. Note $\tau_a$ is NOT integrable: $E[\tau_a] = \infty$.

### 7.4 Joint distribution of $(B_t, M_t)$

**Theorem 7.4.** For $0 \le b \le a$,
$$
P(M_t \ge a, B_t \in db) = \frac{2(2a - b)}{\sqrt{2\pi t^3}} e^{-(2a-b)^2/(2t)} db \cdot \mathbf{1}_{b \le a, a \ge 0}.
$$

*Proof.* Differentiate the reflection-principle identity. $\square$

Used in pricing barrier options.

---

## 8. Martingales Built from BM

**Examples.** The following are martingales w.r.t. the natural filtration $\mathcal{F}_t^B$:

1. $B_t$ itself (mean zero, $L^2$, independent increments).
2. $B_t^2 - t$: check $E[B_t^2 - t | \mathcal{F}_s] = B_s^2 + (t - s) - t = B_s^2 - s$.
3. $\exp(\sigma B_t - \sigma^2 t / 2)$: **exponential martingale**, the building block of Girsanov.
4. $B_t^3 - 3 t B_t$: cubic martingale.
5. $\exp(iu B_t + u^2 t / 2)$: complex exponential, CF form.

**Exponential martingale computation.**
$$
E[\exp(\sigma B_t - \sigma^2 t/2) | \mathcal{F}_s] = \exp(\sigma B_s - \sigma^2 t/2) E[\exp(\sigma(B_t - B_s)) | \mathcal{F}_s] = \exp(\sigma B_s - \sigma^2 t/2) \cdot e^{\sigma^2(t-s)/2} = \exp(\sigma B_s - \sigma^2 s/2).
$$
Uses MGF of Gaussian: $E[e^{\sigma Z}] = e^{\sigma^2/2}$ for $Z \sim \mathcal{N}(0, 1)$.

---

## 9. Multi-Dimensional Brownian Motion

### 9.1 Definition

$\vec{B} = (B^1, \dots, B^d): [0, \infty) \to \mathbb{R}^d$ is a **$d$-dimensional standard BM** if each coordinate $B^i$ is a 1D standard BM and $B^1, \dots, B^d$ are mutually independent.

Equivalently: $\vec{B}_t \sim \mathcal{N}_d(0, t I)$ with independent increments + continuous paths.

**Covariation.** $\langle B^i, B^j\rangle_t = \delta_{ij} t$ (Kronecker delta). This is the "orthogonality" of independent components.

### 9.2 Rotational invariance

$\vec{B}$ is rotationally invariant: for any orthogonal $R$, $R \vec{B}$ is again a $d$-dim BM (since linear combinations of independent Gaussians with the right covariance give independent Gaussians with the same covariance).

### 9.3 Recurrence / transience

Using the generator $\frac{1}{2}\Delta$ (Laplacian) and Green's function:
- $d = 1, 2$: BM is recurrent (visits every neighborhood infinitely often).
- $d \ge 3$: BM is transient, $|\vec{B}_t| \to \infty$ a.s. The Green's function $G(x) = c_d |x|^{2-d}$ is finite for $d \ge 3$.

### 9.4 Correlated BMs

$\vec{W} = \Sigma^{1/2} \vec{B}$ with $\Sigma$ positive-definite gives a BM with $E[W^i_t W^j_t] = \Sigma_{ij} t$. Correlation $\rho_{ij} = \Sigma_{ij}/\sqrt{\Sigma_{ii}\Sigma_{jj}}$.

Used in multi-asset Black-Scholes: $S^i_t$ with correlated diffusions $dS^i_t = \mu_i S^i_t dt + \sigma_i S^i_t dW^i_t$, $d\langle W^i, W^j\rangle_t = \rho_{ij} dt$.

---

## 10. Geometric Brownian Motion

**Definition 10.1.** $S_t := S_0 \exp(\mu t + \sigma B_t)$ is **geometric Brownian motion** (GBM) with drift $\mu$ and volatility $\sigma$.

**Distribution.** $\log S_t \sim \mathcal{N}(\log S_0 + \mu t, \sigma^2 t)$, so $S_t$ is log-normal.

**Moments.** $E[S_t] = S_0 e^{(\mu + \sigma^2/2) t}$, $\text{Var}(S_t) = S_0^2 e^{(2\mu + \sigma^2)t}(e^{\sigma^2 t} - 1)$.

**SDE form** (anticipating Module 3.3 Itô's lemma):
$$
dS_t = S_t (\tilde{\mu} dt + \sigma dB_t), \quad \tilde{\mu} = \mu + \sigma^2/2.
$$
The extra $\sigma^2/2$ is the "Itô correction" — classical calculus would give $\mu$, but Itô's lemma adds this term.

**Black-Scholes setup.** Under risk-neutral $Q$, $dS_t = r S_t dt + \sigma S_t dW_t$, so $S_t = S_0 e^{(r - \sigma^2/2) t + \sigma W_t}$. European call price:
$$
C = E^Q[e^{-rT}(S_T - K)^+] = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2),
$$
where $d_{1,2} = (\log(S_0/K) + (r \pm \sigma^2/2) T)/(\sigma\sqrt{T})$. **Black-Scholes formula** — derived in Module 3.4.

---

## 11. Brownian Bridge

**Definition 11.1.** Brownian bridge on $[0, 1]$: $X_t := B_t - t B_1$, $t \in [0, 1]$.

Equivalently, $X$ is BM conditioned to return to $0$ at $t = 1$.

**Properties.**
- Gaussian process with $E[X_t] = 0$, $\text{Cov}(X_s, X_t) = s(1 - t)$ for $s \le t$.
- $X_0 = X_1 = 0$ a.s., continuous paths.
- Self-similar: $(X_{as + b(1-s)})_{s \in [0, 1]}$ for fixed $a, b$ with $a + b \le 1$ is rescaled bridge.

**Applications.** Kolmogorov-Smirnov statistic; exact simulation of BM on a time grid; conditional sampling.

---

## 12. Python Computational Demonstrations

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2024)

# ================================================================
# 1. Generate Brownian motion paths
# ================================================================
def simulate_bm(n, T=1.0):
    """Standard Brownian motion on [0, T] with n+1 points."""
    dt = T / n
    dB = rng.standard_normal(n) * np.sqrt(dt)
    B = np.concatenate([[0], np.cumsum(dB)])
    t = np.linspace(0, T, n+1)
    return t, B

t, B = simulate_bm(10000, T=1)
print(f"BM path: B_1 = {B[-1]:.4f} (theoretical: N(0,1))")

# ================================================================
# 2. Verify variance / quadratic variation scaling
# ================================================================
# E[B_t^2] = t: sample many paths
N_paths = 10000
BT_samples = np.array([simulate_bm(1000, T=1.0)[1][-1] for _ in range(N_paths)])
print(f"\nE[B_1^2]: empirical = {np.mean(BT_samples**2):.4f}, theory = 1.0")
print(f"Var(B_1): empirical = {np.var(BT_samples):.4f}, theory = 1.0")

# Quadratic variation
def quadratic_variation(B):
    return np.sum(np.diff(B)**2)

# For n = 10, 100, 1000, 10000 subdivisions of [0, 1]:
print("\nQuadratic variation (should → 1.0):")
for n in [10, 100, 1000, 10000]:
    qv_samples = []
    for _ in range(100):
        _, B = simulate_bm(n, T=1.0)
        qv_samples.append(quadratic_variation(B))
    print(f"  n={n:>5}: mean = {np.mean(qv_samples):.4f}, std = {np.std(qv_samples):.4f}")

# Total variation DIVERGES
print("\nTotal variation (should → ∞):")
for n in [10, 100, 1000, 10000]:
    _, B = simulate_bm(n, T=1.0)
    tv = np.sum(np.abs(np.diff(B)))
    print(f"  n={n:>5}: TV = {tv:.4f}")

# ================================================================
# 3. Reflection principle
# ================================================================
# P(M_1 ≥ a) = 2 P(B_1 ≥ a) = 2(1 - Φ(a))
from scipy.stats import norm
N = 50000
BMs = np.array([simulate_bm(1000, T=1.0)[1] for _ in range(N)])
M_values = BMs.max(axis=1)
print("\nReflection principle P(M_1 ≥ a) ≈ 2(1 - Φ(a)):")
for a in [0.5, 1.0, 1.5, 2.0]:
    emp = np.mean(M_values >= a)
    theory = 2 * (1 - norm.cdf(a))
    print(f"  a={a}: emp = {emp:.4f}, theory = {theory:.4f}")

# ================================================================
# 4. Exponential martingale verification
# ================================================================
# E[exp(σ B_t - σ²t/2)] = 1 for all t
sigma = 0.5
T = 1.0
exp_mg_samples = []
for _ in range(10000):
    _, B = simulate_bm(100, T=T)
    exp_mg_samples.append(np.exp(sigma * B[-1] - sigma**2 * T / 2))
print(f"\nE[exp(σB_T - σ²T/2)]: empirical = {np.mean(exp_mg_samples):.4f}, theory = 1.0")

# ================================================================
# 5. Donsker's theorem: normalized random walks → BM
# ================================================================
# Simulate a random walk scaled to converge to BM
def donsker_walk(n_steps, N_paths=1):
    """n_steps steps scaled to [0, 1]."""
    steps = rng.choice([-1, 1], (N_paths, n_steps))
    t = np.linspace(0, 1, n_steps+1)
    paths = np.zeros((N_paths, n_steps+1))
    paths[:, 1:] = np.cumsum(steps, axis=1) / np.sqrt(n_steps)
    return t, paths

# Verify KS distance between donsker_walk(n)[0, T=1] and N(0,1)
from scipy.stats import kstest
for n in [10, 100, 1000]:
    _, paths = donsker_walk(n, N_paths=10000)
    B1_samples = paths[:, -1]
    ks = kstest(B1_samples, 'norm').statistic
    print(f"Donsker n={n:>4}: KS dist of W^(n)(1) to N(0,1) = {ks:.4f}")

# ================================================================
# 6. Hitting time distribution
# ================================================================
# τ_a = inf{t : B_t = a}; f_{τ_a}(t) = a/√(2πt³) exp(-a²/(2t))
a = 1.0
N_trials = 5000
tau_samples = []
for _ in range(N_trials):
    _, B = simulate_bm(10000, T=10.0)
    t_grid = np.linspace(0, 10, 10001)
    idx = np.argmax(B >= a) if np.any(B >= a) else -1
    if idx > 0:
        tau_samples.append(t_grid[idx])
tau_samples = np.array(tau_samples)
print(f"\nHitting times at a={a} (N={len(tau_samples)} samples out of {N_trials}):")
print(f"  Median τ_a: {np.median(tau_samples):.4f}")
print(f"  Note E[τ_a] = ∞, only median makes sense")

# ================================================================
# 7. Geometric Brownian motion
# ================================================================
# S_t = S_0 exp((μ-σ²/2)t + σ B_t); check log-normal
S0 = 100
mu = 0.05
sigma = 0.2
T = 1.0
N = 20000
_, B = simulate_bm(N, T=T)
# actually need many paths:
BTs = np.array([simulate_bm(100, T=T)[1][-1] for _ in range(20000)])
ST_samples = S0 * np.exp((mu - sigma**2/2)*T + sigma * BTs)
print(f"\nGBM S_T = S_0 exp((μ-σ²/2)T + σB_T):")
print(f"  E[S_T]: empirical = {np.mean(ST_samples):.4f}, "
      f"theory = {S0 * np.exp(mu * T):.4f}")
print(f"  E[log S_T]: empirical = {np.mean(np.log(ST_samples)):.4f}, "
      f"theory = {np.log(S0) + (mu - sigma**2/2) * T:.4f}")

# ================================================================
# 8. Black-Scholes Monte Carlo call option pricing
# ================================================================
def bs_mc_call(S0, K, r, sigma, T, N=100000):
    BT = rng.standard_normal(N) * np.sqrt(T)
    ST = S0 * np.exp((r - sigma**2/2)*T + sigma*BT)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff), np.exp(-r*T) * np.std(payoff) / np.sqrt(N)

def bs_formula(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 1.0
mc_price, mc_se = bs_mc_call(S0, K, r, sigma, T)
bs_price = bs_formula(S0, K, r, sigma, T)
print(f"\nEuropean call price (S0={S0}, K={K}, T={T}, r={r}, σ={sigma}):")
print(f"  BS formula: {bs_price:.4f}")
print(f"  Monte Carlo: {mc_price:.4f} ± {mc_se:.4f}")

# ================================================================
# 9. Brownian bridge
# ================================================================
def brownian_bridge(n, T=1.0):
    t = np.linspace(0, T, n+1)
    B = simulate_bm(n, T=T)[1]
    X = B - t * B[-1] / T  # X_t = B_t - (t/T) B_T
    return t, X

t, X = brownian_bridge(1000, T=1.0)
print(f"\nBrownian bridge: X_0 = {X[0]:.4f}, X_1 = {X[-1]:.4f} (both should be 0)")

# Covariance of bridge: Cov(X_s, X_t) = s(1-t) for s ≤ t
# For s=0.3, t=0.7: Cov = 0.3 * 0.3 = 0.09
s_idx, t_idx = 300, 700  # out of 1000
X_paths = np.array([brownian_bridge(1000, T=1.0)[1] for _ in range(5000)])
cov_emp = np.cov(X_paths[:, s_idx], X_paths[:, t_idx])[0, 1]
print(f"  Cov(X_0.3, X_0.7) emp = {cov_emp:.4f}, theory = 0.09")
```

---

## 13. [QUANT APPLICATION] Brownian Motion in Finance

### 13.1 Black-Scholes model

Under risk-neutral $Q$: $S_t = S_0 \exp((r - \sigma^2/2)t + \sigma W_t)$. European options priced via $C_0 = E^Q[e^{-rT}(S_T - K)^+]$.

### 13.2 Vasicek short rate

$dr_t = \kappa(\theta - r_t) dt + \sigma dW_t$ — Ornstein-Uhlenbeck driven by BM. Bond prices $P(0, T) = E^Q[e^{-\int_0^T r_s ds}]$ computed via Gaussian integrals (Vasicek formula).

### 13.3 Barrier options

For a knock-out call with barrier $H > S_0$, reflection principle gives closed-form prices in GBM. Let $M_T^B := \max_t B_t$; then
$$
C_{KO} = E^Q[e^{-rT}(S_T - K)^+ \mathbf{1}_{M_T^B < \ldots}]
$$
computed from the joint distribution of $(B_T, M_T^B)$ (Theorem 7.4).

### 13.4 Discrete hedging error

Approximate continuous hedging with $n$ rebalancings. Hedging error converges in distribution to a Gaussian with variance $\propto \int_0^T (\Gamma_t S_t \sigma_t)^2 dt / n$ (Bertsimas-Kogan-Lo 2000). Uses functional CLT + Itô.

### 13.5 Feynman-Kac for American options

The value $V(t, x) = \sup_\tau E^Q[e^{-r(\tau - t)} h(S_\tau) | S_t = x]$ solves an optimal stopping problem on BM-driven GBM. The free boundary (exercise region) is characterized by variational inequalities from the Laplacian + BM.

### 13.6 Rough paths and rough volatility

Recent (2014+): fractional BM with Hurst $H < 1/2$ is a better fit to implied vol dynamics than standard BM. Gatheral-Jaisson-Rosenbaum showed $H \approx 0.1$ in practice. This requires the "rough path" theory of stochastic integration, beyond classical Itô.

### 13.7 Quadratic variation as realized volatility

If $\log S_t$ is Itô semi-martingale, $\langle\log S\rangle_T = \int_0^T \sigma_t^2 dt$. Realized variance $\text{RV}_T := \sum_i (\log S_{t_i} - \log S_{t_{i-1}})^2 \xrightarrow{P} \langle\log S\rangle_T$ as mesh $\to 0$. Used in VIX-like realized variance swaps.

### 13.8 Exotic pricing via reflection and strong Markov

Lookback option: $h(S) = S_T - \min_t S_t$ or similar. Under GBM, change variables $X_t = \log(S_t/S_0)/\sigma$ to get drifted BM, apply reflection principle. All four "single-barrier" options (down-and-in / out, up-and-in / out, calls and puts) have closed forms.

---

## 14. Worked Examples

### Example 14.1 (Computing $E[B_t^4]$ via characteristic function)

$B_t \sim \mathcal{N}(0, t)$. $\varphi_{B_t}(u) = e^{-u^2 t/2}$. Moments from Taylor: $E[B_t^n] = i^{-n} \varphi^{(n)}(0)$. Compute: $\varphi^{(4)}(0) = 3 t^2$, so $E[B_t^4] = 3 t^2$.

Alternative: Wick's theorem / Isserlis' theorem for Gaussians: $E[X^{2k}] = (2k-1)!! \sigma^{2k}$ for $X \sim \mathcal{N}(0, \sigma^2)$. $k = 2$: $E[X^4] = 3 \sigma^4 = 3 t^2$.

### Example 14.2 (Sub-martingale $|B_t|$ and bounds)

$|B_t|$ is a submartingale (convex fn of martingale). $E|B_t| = \sqrt{2t/\pi}$ (half-normal). Doob: $E[\sup_{s \le T} |B_s|] \le 2 E|B_T| = 2\sqrt{2T/\pi}$. Sharp: $E[\sup_{s \le T} |B_s|] = ?$... actually $\sup |B|$ has a specific distribution, via reflection on $(B_t, M_t)$. Computing gives: $\sup_{s \le T} B_s \sim |B_T|$ so $E[\sup] = \sqrt{2T/\pi}$, and $\sup |B_s|$ is slightly larger.

### Example 14.3 (Hitting probability for drifted BM)

For $X_t = B_t + \mu t$ with $\mu > 0$ and barrier $a > 0$: $P(\tau_a < \infty) = ?$. Use the exponential martingale $M_t := \exp(-2\mu X_t)$ (check it's a martingale via Itô for $X$). OST on $\tau_a \wedge N$: $E[M_{\tau_a \wedge N}] = M_0 = 1$. As $N \to \infty$: on $\{\tau_a < \infty\}$, $M_{\tau_a} = e^{-2\mu a}$; on $\{\tau_a = \infty\}$, $M_N = e^{-2\mu X_N}$ which $\to 0$ a.s. (since $X_N \to \infty$). So $1 = E[M_{\tau_a}] = e^{-2\mu a} P(\tau_a < \infty)$, giving $P(\tau_a < \infty) = e^{-2\mu a}$.

For negative-drift downward walk this gives ruin probability in insurance / Cramér-Lundberg.

### Example 14.4 (Variance of $\int_0^T B_t^2 dt$)

$E[\int_0^T B_t^2 dt] = \int_0^T t \, dt = T^2/2$. For variance, need $E[B_s^2 B_t^2]$. By Wick for $(B_s, B_t) \sim \mathcal{N}_2$:
$$
E[B_s^2 B_t^2] = \text{Cov}(B_s^2, B_t^2) + E[B_s^2] E[B_t^2] = 2 (\text{Cov}(B_s, B_t))^2 + st = 2(s\wedge t)^2 + st.
$$
Then $\text{Var}(\int_0^T B_t^2 dt) = 2 \int_0^T \int_0^T (s \wedge t)^2 ds dt = \ldots = 2 T^4/3 - T^4/2 = T^4/6$... Let me redo: $\int\int (s \wedge t)^2 ds dt = 2\int_0^T \int_0^t s^2 ds dt = 2 \int T^3/3 = T^4/6$. So $\text{Var} = 2 \cdot T^4/6 = T^4/3$.

---

## 15. Exercises

### Tier ★ (Foundational)

**3.1.E1.** Show that if $(B_t)$ is a BM, then so are $(-B_t)$, $(B_{t+s} - B_s)$ (for fixed $s$), and $(cB_{t/c^2})$ (for $c > 0$, "Brownian scaling").

**3.1.E2.** Verify directly that $B_t^2 - t$ is a martingale w.r.t. $\mathcal{F}_t^B$.

**3.1.E3.** Compute $E[B_s B_t B_u B_v]$ for $s \le t \le u \le v$ using Wick's theorem.

**3.1.E4.** For fixed $a > 0$, compute $P(B_T > a)$ and $P(\max_{t \le T} B_t > a)$.

**3.1.E5.** Prove: $t \mapsto B_t/t$ is a martingale (on $(0, \infty)$) iff... wait, is it? $E[B_t/t | \mathcal{F}_s] = (1/t) E[B_t | \mathcal{F}_s] = B_s/t \ne B_s/s$ unless $t = s$. So no. Find the correct rescaling to make it a martingale.

**3.1.E6.** Generate 10 BM paths on $[0, 1]$ with $n = 1000$ steps each. Plot them. Compute max, min, and $\int_0^1 B_t dt$ for each path.

**3.1.E7.** For $X_t := \int_0^t B_s ds$, show $X_t \sim \mathcal{N}(0, t^3/3)$. Is $X_t$ Markovian on its own? Is $(X_t, B_t)$ Markovian?

**3.1.E8.** Prove the Brownian scaling $B_{t \cdot c^2}/c \stackrel{d}{=} B_t$ for $c > 0$. Use it to express $\sup_{t \le T} B_t$ as a function of $T$ and a unit-time supremum.

### Tier ★★ (Intermediate)

**3.1.E9** (Kolmogorov-Chentsov proof details)**.** Fill in the proof of Theorem 3.1. Show the modification $\tilde{X}$ agrees with $X$ at each fixed $t$ a.s., but their joint distribution on $[0, T]$ has continuous paths.

**3.1.E10** (Lévy's construction details)**.** Verify that the BM from Lévy's construction (§2) has $E[B_s B_t] = s \wedge t$. Compute for $s = 1/2, t = 3/4$.

**3.1.E11** (Time inversion)**.** Show that $\tilde{B}_t := t B_{1/t}$ (with $\tilde{B}_0 := 0$) is also a BM on $[0, \infty)$. Use time inversion to prove $\limsup_{t \to \infty} B_t / \sqrt{2 t \log \log t} = 1$ a.s. (from the LIL at $0$).

**3.1.E12** (BM non-differentiability via Paley-Wiener-Zygmund)**.** Complete the proof of Theorem 4.1: show the probability of having $|B'_t| \le K$ at *some* $t \in [0, 1]$ is $0$, for any $K$.

**3.1.E13** (Reflection principle details)**.** Derive the joint density of $(M_T, B_T)$ from the reflection principle. Integrate to get the density of $M_T$.

**3.1.E14** (Arcsine law for last zero)**.** Prove $P(L_1 \le x) = (2/\pi) \arcsin(\sqrt{x})$ using the reflection principle and $P(\text{no zero in } [x, 1]) = P(B_x B_{x + \cdots} > 0 \ldots)$... more cleanly via Feller or direct computation.

**3.1.E15** (Brownian bridge as conditional BM)**.** Show that $(B_t)_{t \in [0,1]}$ conditioned on $B_1 = 0$ has the same law as the Brownian bridge $X_t := B_t - t B_1$. Hint: compute joint density and marginal.

**3.1.E16** (Donsker tightness)**.** Prove the tightness step in Donsker's theorem. Use $E|S_n|^4 = O(n^2)$ and Doob's maximal inequality on the $L^4$-submartingale $S_n^4$.

### Tier ★★★ (Advanced / Quant)

**3.1.E17** (Black-Scholes via reflection for down-and-out barrier)**.** For a down-and-out call with barrier $H < S_0$ and strike $K > H$, derive the closed-form price under GBM using reflection. Show
$$
C_{DO} = C_{BS}(S_0) - \left(\frac{H}{S_0}\right)^{2(r-\sigma^2/2)/\sigma^2} C_{BS}(H^2/S_0),
$$
where $C_{BS}$ is the Black-Scholes call.

**3.1.E18** (Girsanov preview)**.** Let $\theta \in \mathbb{R}$. Define $dQ/dP|_{\mathcal{F}_T} := Z_T := \exp(-\theta B_T - \theta^2 T/2)$. Show $Z$ is a $P$-martingale, $E_P[Z_T] = 1$, and (by change of measure) $\tilde{B}_t := B_t + \theta t$ is a $Q$-Brownian motion on $[0, T]$.

**3.1.E19** (Brownian motion and the heat equation)**.** Let $u(t, x) := E[f(B_t^x)]$ where $B_t^x = x + B_t$. Show $u$ solves $\partial_t u = \frac{1}{2} \partial_{xx} u$ with $u(0, x) = f(x)$. Use the explicit heat kernel to verify directly for $f(x) = e^{ikx}$.

**3.1.E20** (Functional CLT for stochastic integrals)**.** Let $\xi_n$ be iid mean-zero with unit variance. Define the Riemann-like sum $T_n := n^{-1} \sum_{k=1}^n f(S_{k-1}/\sqrt{n}) (S_k - S_{k-1})/\sqrt{n}$ where $S_k = \sum_{i \le k} \xi_i$. Show $T_n \xRightarrow{d} \int_0^1 f(B_s) dB_s$ (Itô integral, Module 3.2).

**3.1.E21** (Rough volatility)**.** Simulate a fractional BM with Hurst $H = 0.1$ using the Cholesky method. Compute sample autocorrelation of increments $B_{t+h} - B_t$ and show it does not decay like standard BM (which has zero correlation).

**3.1.E22** (Quadratic variation for stochastic volatility)**.** Let $\sigma_t$ be a positive process (e.g., CIR) and $X_t = \int_0^t \sigma_s dB_s$. Show $\langle X\rangle_t = \int_0^t \sigma_s^2 ds$ via quadratic variation. Implement a realized variance estimator and compare to $\int \sigma_s^2 ds$ for simulated Heston paths.

---

## 16. Summary and Forward Pointers

**What we proved.**
- Definition of BM (3 axioms + path continuity).
- Existence via Lévy's Haar/Schauder construction and via Kolmogorov-Chentsov on extension process.
- Path regularity: Hölder-$\gamma$ for $\gamma < 1/2$, but nowhere differentiable.
- Quadratic variation $\langle B\rangle_t = t$; total variation $= \infty$.
- LIL, arcsine laws.
- Donsker's invariance principle.
- Lévy's martingale characterization of BM + Dambis-Dubins-Schwarz.
- Markov, strong Markov, reflection principle.
- Martingales built from BM (exponential, polynomial).
- Multi-D BM, GBM, Brownian bridge.

**The big picture.** BM is "the Gaussian process of all times" — it's the limit of random walks, the unique continuous-path Lévy process, the canonical martingale with unit quadratic variation, and the solution of the "simplest" SDE $dX = dW$. Everything in Subject 3 builds on this.

**Forward pointers.**
- **Module 3.2** (Stochastic integration): define $\int_0^T H_s dB_s$ as an isometry from $L^2(\text{predictable})$ to $L^2(P)$. This is the Itô integral, the cornerstone of continuous-time finance.
- **Module 3.3** (Itô calculus): Itô's lemma for $f(B_t)$, Girsanov, martingale representation — the computational toolkit.
- **Module 3.4** (SDEs and PDEs): solve $dX = b(X) dt + \sigma(X) dW$; Feynman-Kac connects SDE expectations to PDEs.
- **Module 3.5** (Lévy processes): generalize BM to jumps — compound Poisson, subordinators, Lévy-Itô.
- **Module 3.6** (Markov processes): abstract generator theory (Hille-Yosida), invariant measures, ergodicity for diffusions.
- **Module 3.7** (Continuous-time martingales): Doob-Meyer decomposition, BDG inequalities, martingale representation theorem.

**Next module:** The Itô integral — how to integrate against a Brownian motion when the paths aren't differentiable.

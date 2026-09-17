# Module 1.6 — Signed Measures, Radon-Nikodym, and Lebesgue Decomposition

> *"Every σ-finite absolutely continuous measure is an integral."* — the Radon-Nikodym theorem, 1913-1930.

## Prerequisites

- **Modules 1.1-1.5** — full measure, integration, and $L^p$ theory.
- **Module 0.4** — inner products / Hilbert spaces (we use von Neumann's beautiful Hilbert-space proof of R-N).

## Overview

We have met measures as non-negative, countably additive set functions. Signed measures allow negative values: think of the net flow of a probability shift, or an expected value that can be positive or negative. This module builds the three central decomposition theorems:

1. **Hahn-Jordan decomposition** — every signed measure splits as $\nu = \nu^+ - \nu^-$ with $\nu^\pm$ positive measures on disjoint sets; the total variation $|\nu| = \nu^+ + \nu^-$ makes signed measures into a Banach space.

2. **Radon-Nikodym theorem** — if $\nu \ll \mu$ (absolutely continuous) and both are $\sigma$-finite, then $\nu$ has a density with respect to $\mu$: $d\nu/d\mu = f$ with $f \ge 0$ measurable, and $\nu(E) = \int_E f \, d\mu$.

3. **Lebesgue decomposition** — every $\sigma$-finite $\nu$ decomposes uniquely as $\nu = \nu_a + \nu_s$ with $\nu_a \ll \mu$ and $\nu_s \perp \mu$ (singular with respect to $\mu$).

These three theorems are the analytic foundation of:
- **Probability**: change of measure, Girsanov, likelihood ratios, the market price of risk.
- **Stochastic calculus**: equivalent martingale measures, risk-neutral pricing.
- **Statistics**: likelihood ratios, Fisher information, Neyman-Pearson.
- **PDE**: weak derivatives, distributional derivatives, Sobolev spaces.

We give a self-contained proof of R-N using von Neumann's Hilbert-space argument (Riesz representation on $L^2$), which is short and elegant.

---

## Topic 1.6.1 — Signed and Complex Measures

### Definition 1.6.1 (Signed measure)

A **signed measure** on $(X, \mathcal F)$ is a function $\nu: \mathcal F \to [-\infty, +\infty]$ such that:

1. $\nu(\emptyset) = 0$;
2. $\nu$ takes at most one of the values $-\infty, +\infty$ (never both);
3. **Countable additivity**: for any disjoint sequence $\{E_n\} \subset \mathcal F$,
$$\nu\!\left(\bigsqcup_n E_n\right) = \sum_n \nu(E_n),$$
with the sum converging absolutely when $\nu(\bigsqcup E_n) \in \mathbb R$.

The condition that at most one of $\pm\infty$ is attained prevents $\infty - \infty$ ambiguities.

### Definition 1.6.2 (Complex measure)

A **complex measure** $\nu: \mathcal F \to \mathbb C$ is a countably additive function with finite complex values (no infinities allowed), so $\nu(E) \in \mathbb C$ for every $E$.

### Examples

- **Difference of two finite positive measures**: $\nu = \mu_1 - \mu_2$ with both $\mu_i$ finite is a signed measure.
- **Density against a measure**: $\nu(E) = \int_E f \, d\mu$ for $f \in L^1(\mu)$ (real or complex) gives a signed/complex measure.
- **Probability displacement**: in a portfolio context, $\nu(E) = \mathbb P(E) - \mathbb Q(E)$ captures the mismatch between physical and risk-neutral measures.

### Proposition 1.6.3 (Hahn decomposition)

For any signed measure $\nu$ on $(X, \mathcal F)$, there exist disjoint $P, N \in \mathcal F$ with $X = P \sqcup N$ such that $\nu(A) \ge 0$ for every $A \subset P$ and $\nu(A) \le 0$ for every $A \subset N$. We call $P$ a **positive set** and $N$ a **negative set** for $\nu$.

**Proof.** WLOG $\nu$ does not attain $+\infty$ (otherwise consider $-\nu$). Set $M = \sup\{\nu(E) : E \in \mathcal F\} \in [0, +\infty)$ (finite since $\nu$ is bounded above).

Pick sets $E_n$ with $\nu(E_n) \ge M - 1/n$. We construct a maximizing set $P$ out of the $E_n$:

*Lemma (refinement).* If $\nu(E) > -\infty$, then $E$ contains a **positive subset** $P_E$ with $\nu(P_E) \ge \nu(E)$.

*Proof of lemma.* If $E$ contains no subset of strictly negative measure, $E$ itself is positive, take $P_E = E$. Otherwise, let $\delta_1 = -\inf\{\nu(F) : F \subset E\} > 0$; pick $F_1 \subset E$ with $\nu(F_1) \le -\delta_1/2 < 0$. Set $E_1 = E \setminus F_1$; then $\nu(E_1) = \nu(E) - \nu(F_1) \ge \nu(E) + \delta_1/2 > \nu(E)$. Iterate: at stage $k$, take $E_{k-1}$, and either (a) $E_{k-1}$ contains no subset of strictly negative measure, in which case $E_{k-1}$ is positive and we stop, or (b) pick $F_k \subset E_{k-1}$ with $\nu(F_k) \le -\delta_k/2$, $\delta_k = -\inf\{\nu(F) : F \subset E_{k-1}\}$; set $E_k = E_{k-1} \setminus F_k$.

If the process terminates at finite step $k$, take $P_E = E_k$: it is positive and $\nu(P_E) \ge \nu(E) + \sum \delta_i / 2$.

If the process doesn't terminate, $\sum \delta_i/2 \le |\nu(E)|$ (since each step increases $\nu(E_i)$ by at least $\delta_i/2$, but $\nu$ is bounded), so $\delta_i \to 0$. Set $P_E = E \setminus \bigcup_k F_k$. For any $F \subset P_E$, $F \subset E_k$ for every $k$, so $\nu(F) \ge -\delta_{k+1} \to 0$ as $k \to \infty$. Hence $\nu(F) \ge 0$ for every $F \subset P_E$; i.e. $P_E$ is positive. And $\nu(P_E) = \nu(E) - \sum \nu(F_k) \ge \nu(E)$ since each $\nu(F_k) < 0$. Lemma proved.

*Back to the main argument.* For each $n$, apply the lemma to $E_n$: get $P_n \subset E_n$ positive with $\nu(P_n) \ge \nu(E_n) \ge M - 1/n$. Set $P = \bigcup_n P_n$. Since finite (or countable) unions of positive sets are positive (exercise; use that for $A \subset \bigcup P_n$, $A = \bigsqcup_n (A \cap P_n \setminus \bigcup_{k<n} P_k)$, each piece in a positive set), $P$ is positive.

Also $\nu(P) \ge \nu(P_n) \ge M - 1/n$ for all $n$, so $\nu(P) = M$. Let $N = X \setminus P$.

*Claim: $N$ is negative.* If not, there's $A \subset N$ with $\nu(A) > 0$. Then $\nu(P \cup A) = \nu(P) + \nu(A) > M$, contradicting $M = \sup$. So $N$ is negative. $\blacksquare$

### Theorem 1.6.4 (Jordan decomposition)

For any signed measure $\nu$, write $P \sqcup N$ as in Hahn. Define
$$\nu^+(E) := \nu(E \cap P), \qquad \nu^-(E) := -\nu(E \cap N).$$
Both are positive measures, **mutually singular** (supported on disjoint $P$ and $N$), and $\nu = \nu^+ - \nu^-$. This decomposition is unique among all such decompositions into mutually singular positive measures.

**Total variation**: $|\nu| := \nu^+ + \nu^-$ is a positive measure. For a complex measure $\nu$, define $|\nu|$ via a similar construction (via decomposing real and imaginary parts).

**Proof (existence).** Immediate from Hahn: $\nu^+, \nu^-$ positive, disjoint support, $\nu = \nu^+ - \nu^-$.

**Uniqueness.** Suppose $\nu = \mu_1 - \mu_2$ with $\mu_1 \perp \mu_2$ (supported on disjoint measurable sets $A, B$ with $X = A \sqcup B$). Then on $A$: $\nu \ge 0$, on $B$: $\nu \le 0$. So $A$ is a positive Hahn set and $B$ a negative. Any two Hahn decompositions $(P, N)$ and $(A, B)$ differ by a null set for $|\nu|$ (since $P \triangle A \subset (P \cap B) \cup (N \cap A)$, and $\nu$ is both non-negative and non-positive on each piece, so $|\nu|$ is zero there). Hence $\mu_1(E) = \nu(E \cap A) = \nu(E \cap P) = \nu^+(E)$, similarly $\mu_2 = \nu^-$. $\blacksquare$

### Theorem 1.6.5 (Signed measures form a Banach space)

The space of finite signed measures $M(X, \mathcal F)$, with norm $\|\nu\| := |\nu|(X)$, is a Banach space.

**Proof.** $\|\cdot\|$ is a norm: positivity, homogeneity, and the triangle inequality $|\nu_1 + \nu_2|(E) \le |\nu_1|(E) + |\nu_2|(E)$ (a classical argument: the total variation is the sup of $\sum |\nu(E_i)|$ over partitions of $E$, and sums respect the triangle inequality). Completeness follows from completeness of $L^1$: identify $M(X) \cong L^1(|\nu|)$... this is a non-trivial argument. (Proof in Rudin RCA or Folland.) $\blacksquare$

### Connection to $L^1$

Given $\mu$ $\sigma$-finite, the map $f \mapsto \nu_f$, $\nu_f(E) := \int_E f \, d\mu$ is an isometric embedding $L^1(\mu) \hookrightarrow M(X)$ with $\|\nu_f\| = \|f\|_1$. The image is exactly the set of measures absolutely continuous w.r.t. $\mu$ (Radon-Nikodym).

---

## Topic 1.6.2 — Absolute Continuity and Singularity

### Definition 1.6.6 (Absolute continuity; mutual singularity)

Let $\mu$ and $\nu$ be measures on $(X, \mathcal F)$, with $\nu$ signed (or complex).

- **$\nu \ll \mu$** ($\nu$ is absolutely continuous w.r.t. $\mu$): $\mu(E) = 0 \Rightarrow \nu(E) = 0$.
- **$\nu \perp \mu$** ($\nu$ is singular w.r.t. $\mu$): there exist disjoint $A, B \in \mathcal F$ with $X = A \sqcup B$, $\nu$ supported on $A$ (i.e. $\nu(E) = \nu(E \cap A)$ for all $E$), $\mu$ supported on $B$. Equivalently, they live on disjoint sets.
- **$\nu \approx \mu$** (equivalent): $\nu \ll \mu$ and $\mu \ll \nu$.

### Proposition 1.6.7 ($\varepsilon$-$\delta$ version of absolute continuity for finite measures)

Let $\nu$ be a **finite** signed measure and $\mu$ a positive measure. Then $\nu \ll \mu$ iff for every $\varepsilon > 0$ there exists $\delta > 0$ such that $\mu(E) < \delta \Rightarrow |\nu(E)| < \varepsilon$.

**Proof.**

*$\Leftarrow$ (ε-δ implies ≪).* If $\mu(E) = 0$, then for any $\varepsilon$, $|\nu(E)| < \varepsilon$, so $\nu(E) = 0$.

*$\Rightarrow$.* Suppose ε-δ fails. Then there exist $\varepsilon_0 > 0$ and $E_n$ with $\mu(E_n) < 2^{-n}$ and $|\nu(E_n)| \ge \varepsilon_0$. Set $E = \limsup_n E_n = \bigcap_m \bigcup_{n \ge m} E_n$. Then $\mu(E) \le \lim_m \mu(\bigcup_{n \ge m} E_n) \le \lim_m \sum_{n \ge m} 2^{-n} = 0$. But $|\nu|(E) \ge \limsup_n |\nu|(E_n) \ge \varepsilon_0$ by continuity of measure from above (valid since $|\nu|$ is finite), contradicting $\nu \ll \mu$. $\blacksquare$

**Warning.** The ε-δ characterization fails for infinite $\nu$. For example on $(\mathbb R, \lambda)$, $d\nu = e^x \, dx$ has $\nu \ll \lambda$ but $\lambda(\{x > n\}) < \infty$ yet $\nu(\{x > n\}) = \infty$ — ε-δ cannot hold at tail.

### Remark. The ε-δ is in fact the defining property in quant contexts

When pricing, absolute continuity between real-world $\mathbb P$ and risk-neutral $\mathbb Q$ is the statement that no event of positive probability under one is impossible under the other. The ε-δ says: arbitrarily unlikely events under $\mathbb P$ are arbitrarily unlikely under $\mathbb Q$. This is the "no arbitrage" compatibility condition.

---

## Topic 1.6.3 — The Radon-Nikodym Theorem

### Theorem 1.6.8 (Radon-Nikodym)

Let $\mu$ and $\nu$ be $\sigma$-finite positive measures on $(X, \mathcal F)$ with $\nu \ll \mu$. Then there exists a non-negative measurable $f: X \to [0, \infty]$, unique up to $\mu$-a.e. equality, such that
$$\nu(E) = \int_E f \, d\mu \quad \text{for all } E \in \mathcal F.$$
The function $f$ is called the **Radon-Nikodym derivative** and is written $f = d\nu/d\mu$.

**Proof (von Neumann's Hilbert-space proof for finite measures).** We first do the case where $\mu$ and $\nu$ are both finite. Extension to $\sigma$-finite is by a piecewise argument.

*Step 1: auxiliary measure.* Let $\rho := \mu + \nu$, a finite positive measure. For $f \in L^2(\rho)$, define
$$\Lambda(f) := \int_X f \, d\nu.$$
By Cauchy-Schwarz, $|\Lambda(f)| \le \int |f| \, d\nu \le \int |f| \, d\rho \le \rho(X)^{1/2} \|f\|_{L^2(\rho)}$ (the last uses Cauchy-Schwarz: $\int |f| \cdot 1 \, d\rho \le \|f\|_2 \|1\|_2 = \rho(X)^{1/2} \|f\|_2$). So $\Lambda$ is a continuous linear functional on $L^2(\rho)$.

*Step 2: Riesz representation.* By the Riesz representation theorem in Hilbert spaces (Module 0.4), there exists $g \in L^2(\rho)$ with
$$\Lambda(f) = \int f g \, d\rho \quad \text{for all } f \in L^2(\rho).$$
That is, $\int f \, d\nu = \int f g \, d\rho = \int f g \, d\mu + \int f g \, d\nu$ for all $f \in L^2(\rho)$.

Rearranging: $\int f (1 - g) \, d\nu = \int f g \, d\mu$ for all $f \in L^2(\rho)$.

*Step 3: $0 \le g < 1$ $\rho$-a.e.* Take $f = \mathbf 1_E$ with $E = \{g < 0\}$:
$$0 \le \nu(E) = \int_E g \, d\rho \le 0$$
(since $g < 0$ on $E$ and $\rho(E) \ge 0$). So $\nu(E) = 0$, and by the same sign analysis $\int_E g \, d\rho = \nu(E) - \mu(E) \cdot 0$... wait let me redo:

Take $f = \mathbf 1_E$: $\int_E d\nu = \int_E g \, d\mu + \int_E g \, d\nu$, i.e. $\nu(E) = \int_E g \, d\rho$.

If $E = \{g < 0\}$ has $\rho(E) > 0$ then $\int_E g \, d\rho < 0$, contradiction. So $g \ge 0$ $\rho$-a.e.

If $E = \{g > 1\}$ has $\rho(E) > 0$: $\int_E g \, d\rho > \rho(E) = \mu(E) + \nu(E) \ge \nu(E)$, contradiction (since $\nu(E) = \int_E g \, d\rho$). So $g \le 1$ $\rho$-a.e.

If $E = \{g = 1\}$ has $\mu(E) > 0$: $\mu(E) = \int_E (1 - g) \, d\rho/ \text{something}$... actually, $\int_E (1 - g) \, d\nu = \int_E g \, d\mu$ becomes $0 = \mu(E)$ on $\{g = 1\}$, so $\mu(\{g = 1\}) = 0$. Since $\nu \ll \mu$, $\nu(\{g = 1\}) = 0$ too. So redefine $g$ on $\{g = 1\}$ to be $0$ if needed; doesn't affect any integral.

Hence after modification on a null set, $0 \le g < 1$ everywhere.

*Step 4: construct the density.* Define $f := g / (1 - g)$. We claim $\nu(E) = \int_E f \, d\mu$ for all $E$.

For non-negative simple $s$: $\int s \, d\nu = \int s g \, d\rho = \int s g \, d\mu + \int s g \, d\nu$, so $\int s (1 - g) \, d\nu = \int s g \, d\mu$.

Take $s = \mathbf 1_E / (1 - g) \cdot (1 - g) = \mathbf 1_E$: wait, we want to divide. Instead substitute $s = \mathbf 1_E$ and then work backwards:
$$\int_E (1 - g) \, d\nu = \int_E g \, d\mu.$$
This gives $\nu(E) - \int_E g \, d\nu = \int_E g \, d\mu$, so $\nu(E) = \int_E g \, d\mu + \int_E g \, d\nu$, which is just Step 2 again.

To extract $\nu(E) = \int_E f \, d\mu$ with $f = g/(1-g)$, replace $s = \mathbf 1_E \cdot \mathbf 1_{\{g < 1\}}$ by $s_n = \mathbf 1_E \cdot \sum_{k=0}^n g^k = \mathbf 1_E (1 - g^{n+1})/(1 - g)$. Then on $\{0 \le g < 1\}$, $s_n \uparrow \mathbf 1_E / (1 - g)$. Now $s_n (1 - g) = \mathbf 1_E (1 - g^{n+1}) \uparrow \mathbf 1_E$. By MCT,
$$\int s_n (1 - g) \, d\nu = \int \mathbf 1_E (1 - g^{n+1}) \, d\nu \uparrow \nu(E).$$
Also $s_n g = \mathbf 1_E g (1 - g^{n+1})/(1-g)$, which $\uparrow \mathbf 1_E g/(1-g) = \mathbf 1_E f$ a.e. By MCT,
$$\int s_n g \, d\mu \uparrow \int_E f \, d\mu.$$
Since $\int s_n (1-g) \, d\nu = \int s_n g \, d\mu$ for every $n$ (from the earlier identity, applied to $s = s_n$), passing to limits:
$$\nu(E) = \int_E f \, d\mu. \qquad \blacksquare$$

**Uniqueness.** If $\int_E f \, d\mu = \int_E f' \, d\mu$ for all $E$, then $\int_E (f - f') \, d\mu = 0$ for all $E$. Take $E = \{f > f'\}$: $\int (f - f')^+ \, d\mu = 0$, so $(f - f')^+ = 0$ a.e., i.e. $f \le f'$ a.e. Symmetrically $f' \le f$ a.e. Hence $f = f'$ $\mu$-a.e.

**Extension to $\sigma$-finite.** Write $X = \bigsqcup X_n$ with $\mu(X_n), \nu(X_n) < \infty$. Apply the finite-case theorem on each $X_n$ to get $f_n$; paste $f = \sum f_n \mathbf 1_{X_n}$. Then $\nu(E) = \sum \nu(E \cap X_n) = \sum \int_{E \cap X_n} f_n \, d\mu = \int_E f \, d\mu$ (the last by MCT). $\blacksquare$

### Corollary 1.6.9 (R-N for signed/complex measures)

Let $\nu$ be a signed (or complex) measure with $\nu \ll \mu$ and $|\nu|$ $\sigma$-finite. Then there exists a signed (or complex) measurable $f$ with $\nu(E) = \int_E f \, d\mu$; $f \in L^1(\mu)$ iff $\nu$ is finite.

**Proof.** Apply R-N to $\nu^+, \nu^-$ separately (both $\ll \mu$ because $\nu \ll \mu$), get $f^+, f^- \ge 0$, take $f = f^+ - f^-$. For complex: decompose into real + imaginary. $\blacksquare$

### Chain rule and other properties

**Proposition 1.6.10.** For $\nu \ll \mu \ll \rho$ (all $\sigma$-finite):
$$\frac{d\nu}{d\rho} = \frac{d\nu}{d\mu} \cdot \frac{d\mu}{d\rho} \quad \rho\text{-a.e.}$$

**Proof.** $\nu(E) = \int_E \frac{d\nu}{d\mu} d\mu = \int_E \frac{d\nu}{d\mu} \frac{d\mu}{d\rho} d\rho$. Uniqueness of R-N density gives the chain rule. $\blacksquare$

**Proposition 1.6.11 (Change of variable).** For $\nu \ll \mu$ and $g: X \to \overline{\mathbb R}$ measurable:
$$\int g \, d\nu = \int g \cdot \frac{d\nu}{d\mu} \, d\mu,$$
whenever either side makes sense (both non-negative, or $g \cdot d\nu/d\mu \in L^1(\mu)$).

**Proof.** Standard machine: indicator → simple → non-negative measurable → general. $\blacksquare$

---

## Topic 1.6.4 — Lebesgue Decomposition Theorem

### Theorem 1.6.12 (Lebesgue decomposition)

Let $\mu$ and $\nu$ be $\sigma$-finite positive measures on $(X, \mathcal F)$. Then there exist unique measures $\nu_a, \nu_s$ such that
$$\nu = \nu_a + \nu_s, \quad \nu_a \ll \mu, \quad \nu_s \perp \mu.$$
Moreover, $\nu_a$ has a density $f = d\nu_a/d\mu \ge 0$ by Radon-Nikodym.

**Proof (via Hilbert space again).** The argument closely follows the R-N proof.

Finite case: work with $\rho = \mu + \nu$. Let $g \in L^2(\rho)$ be the Riesz-representing function for $f \mapsto \int f \, d\nu$ on $L^2(\rho)$, as before. Steps 1-3 give $0 \le g \le 1$ and $\nu(E) = \int_E g \, d\rho$.

Set $A := \{g < 1\}$ and $B := \{g = 1\}$; $X = A \sqcup B$.

Define $\nu_a(E) := \nu(E \cap A)$ and $\nu_s(E) := \nu(E \cap B)$.

*$\nu_a \ll \mu$.* On $A$, $g < 1$, so the construction in R-N produces a density $f = g/(1-g)$ with $\nu_a(E) = \int_E f \, d\mu$. So $\nu_a \ll \mu$.

*$\nu_s \perp \mu$.* We show $\mu(B) = 0$. From $\int f (1 - g) \, d\nu = \int f g \, d\mu$ with $f = \mathbf 1_B$:
$$0 = \int_B (1 - g) \, d\nu = \int_B g \, d\mu = \int_B 1 \, d\mu = \mu(B),$$
using $g = 1$ on $B$. So $\mu(B) = 0$, and $\nu_s$ is supported on $B$ while $\mu$ is supported on $A = X \setminus B$. Hence $\nu_s \perp \mu$.

*Uniqueness.* If $\nu = \nu_a + \nu_s = \nu'_a + \nu'_s$ are two such decompositions, then $\nu_a - \nu'_a = \nu'_s - \nu_s$; the LHS is absolutely continuous w.r.t. $\mu$ (difference of two a.c. measures), the RHS is singular (difference of two measures both singular to $\mu$, supported on respective singular sets, the difference is supported on the union which is a $\mu$-null set). A measure that is both $\ll \mu$ and $\perp \mu$ must be zero: if it is supported on a $\mu$-null set $N$, then for any $E$, $\nu(E) = \nu(E \cap N) = 0$ because $\mu(E \cap N) = 0$ and $\nu \ll \mu$. Hence $\nu_a = \nu'_a$, $\nu_s = \nu'_s$.

Extend to $\sigma$-finite by pasting. $\blacksquare$

### Example 1.6.13 (Cantor distribution)

Let $F_C: [0, 1] \to [0, 1]$ be the Cantor devil's staircase — continuous, non-decreasing, $F_C(0) = 0$, $F_C(1) = 1$, constant on each interval removed from $[0, 1]$ to form the Cantor set, and derivative $0$ a.e. The associated Lebesgue-Stieltjes measure $\nu_C$ on $[0, 1]$ satisfies $\nu_C([a, b]) = F_C(b) - F_C(a)$.

$\nu_C$ is a probability measure but $\nu_C \perp \lambda$: $\nu_C$ is supported on the Cantor set (Lebesgue measure zero), so $\lambda$ and $\nu_C$ live on disjoint supports.

Thus for the Cantor distribution:
- $\nu_C$ is singular w.r.t. $\lambda$.
- $\nu_C$ has no density w.r.t. Lebesgue (no $\lambda$-a.e. derivative).
- R-N fails because the hypothesis $\nu_C \ll \lambda$ fails.

The Lebesgue decomposition of $\nu_C$ w.r.t. $\lambda$ is $\nu_C = 0 + \nu_C$: the absolutely continuous part is $0$, the singular part is $\nu_C$ itself.

### Example 1.6.14 (Dirac mass)

$\delta_{x_0}$ is singular w.r.t. Lebesgue: supported at a point of measure zero. Lebesgue decomposition of $d\nu = e^{-x} dx + 3 \delta_0$ w.r.t. Lebesgue: $\nu_a = e^{-x} dx$ (with density $f = e^{-x}$), $\nu_s = 3\delta_0$.

### Example 1.6.15 (Discrete + continuous random variable)

A random variable $X$ that is uniform on $[0, 1]$ with probability $1/2$ and equals $0$ with probability $1/2$ has distribution $\mu_X = \frac{1}{2} \lambda|_{[0, 1]} + \frac{1}{2} \delta_0$. Lebesgue decomposition: absolutely continuous part $\frac{1}{2} \lambda|_{[0, 1]}$ (with density $\frac{1}{2} \mathbf 1_{[0, 1]}$), singular part $\frac{1}{2}\delta_0$.

In finance: jump models (jump-diffusion) have singular measures (jumps) alongside continuous densities, and R-N/Lebesgue decomposition is the right framework.

---

## Topic 1.6.5 — Conditional Expectation as R-N Derivative

### Definition 1.6.16 (Conditional expectation)

Let $(\Omega, \mathcal F, \mathbb P)$ be a probability space, $X \in L^1(\mathbb P)$ a random variable, and $\mathcal G \subset \mathcal F$ a sub-σ-algebra. The **conditional expectation** $\mathbb E[X | \mathcal G]$ is the unique (up to $\mathbb P$-null set) $\mathcal G$-measurable random variable $Y$ such that
$$\int_A Y \, d\mathbb P = \int_A X \, d\mathbb P \quad \text{for all } A \in \mathcal G. \tag{*}$$

### Theorem 1.6.17 (Existence via R-N)

For $X \in L^1(\mathbb P)$, $\mathbb E[X | \mathcal G]$ exists and is unique (a.s.).

**Proof.** WLOG $X \ge 0$ (decompose $X = X^+ - X^-$, apply to each).

Define the measure $\nu$ on $(\Omega, \mathcal G)$ by $\nu(A) := \int_A X \, d\mathbb P$ for $A \in \mathcal G$. Check:

- $\nu$ is a positive measure (non-negativity, countable additivity from DCT/MCT).
- $\nu$ is finite: $\nu(\Omega) = \int X \, d\mathbb P = \mathbb E X < \infty$.
- $\nu \ll \mathbb P|_\mathcal G$: if $\mathbb P(A) = 0$ for $A \in \mathcal G$, then $\int_A X \, d\mathbb P = 0$.

By R-N on $(\Omega, \mathcal G, \mathbb P|_\mathcal G)$, there exists a $\mathcal G$-measurable $Y \ge 0$ with $\nu(A) = \int_A Y \, d\mathbb P|_\mathcal G$. This $Y$ is the conditional expectation: $(*)$ is satisfied. $\blacksquare$

### Properties of conditional expectation

- **Linearity**: $\mathbb E[\alpha X + \beta Y | \mathcal G] = \alpha \mathbb E[X|\mathcal G] + \beta \mathbb E[Y|\mathcal G]$ a.s.
- **Tower**: for $\mathcal H \subset \mathcal G$, $\mathbb E[\mathbb E[X | \mathcal G] | \mathcal H] = \mathbb E[X | \mathcal H]$ a.s.
- **Pulling out known factors**: if $Z$ is $\mathcal G$-measurable and bounded (or $XZ \in L^1$), then $\mathbb E[XZ | \mathcal G] = Z \cdot \mathbb E[X | \mathcal G]$ a.s.
- **Jensen**: for convex $\varphi$, $\varphi(\mathbb E[X | \mathcal G]) \le \mathbb E[\varphi(X) | \mathcal G]$ a.s.

The full treatment is Module 2.2. Here we note only that it rests on R-N.

---

## Topic 1.6.6 — Python: Radon-Nikodym Derivatives in Practice

```python
"""
Numerical examples of Radon-Nikodym densities (a.k.a. likelihood ratios).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------- Likelihood ratio: N(mu, 1) vs N(0, 1) on R ----------
# d P_mu / d P_0 = exp(mu x - mu^2 / 2)
mu = 0.5
xs = np.linspace(-5, 5, 1000)
likelihood_ratio = np.exp(mu * xs - mu**2 / 2)
density_P_mu = norm.pdf(xs, loc=mu, scale=1)
density_P_0 = norm.pdf(xs, loc=0, scale=1)
ratio_numerical = density_P_mu / density_P_0
print(f"Max abs error in LR formula: {np.max(np.abs(likelihood_ratio - ratio_numerical)):.2e}")
# The explicit formula matches the density ratio exactly.

# Girsanov: Under P_0, W_T ~ N(0, T); under P_mu, W_T - mu*T ~ N(0, T) (drift removed).
# d P_mu / d P_0 = exp(mu W_T - mu^2 T / 2).

# ---------- Singular measure: Cantor distribution (approximation) ----------
def cantor_cdf(x, depth=10):
    """Approximate Cantor CDF by iterating tent-map-like folding."""
    x = np.atleast_1d(x).astype(float)
    result = np.zeros_like(x)
    for i in range(depth):
        low = (x < 1/3)
        high = (x >= 2/3)
        mid = ~(low | high)
        # On [0, 1/3]: F(x) = F(3x) / 2
        # On [1/3, 2/3]: F(x) = 1/2
        # On [2/3, 1]: F(x) = 1/2 + F(3x - 2)/2
        result_new = np.where(low, 0.0, np.where(mid, 0.5, 0.5))
        x_new = np.where(low, 3*x, np.where(mid, 1.0, 3*x - 2))
        result = np.where(low, result + 0 * (0.5**i), result)  # partial update; crude
        x = x_new
    return result
# (Full Cantor CDF construction is standard; here just a sketch.)

# ---------- Radon-Nikodym and change of measure for option pricing ----------
# Under real-world P: dS = mu S dt + sigma S dW (drift mu)
# Under risk-neutral Q: dS = r S dt + sigma S dW (drift r)
# dQ/dP = exp(-theta W_T - theta^2 T / 2) with theta = (mu - r)/sigma
mu_real, r, sigma, T = 0.10, 0.02, 0.25, 1.0
theta = (mu_real - r) / sigma
# Sample W_T under P
n_samples = 10000
W_T = np.random.normal(0, np.sqrt(T), n_samples)
dQdP = np.exp(-theta * W_T - 0.5 * theta**2 * T)
# Verify E_P [dQ/dP] = 1
print(f"E_P [dQ/dP] ≈ {dQdP.mean():.4f} (should be 1)")
# Stock price at T under P
S_T = 100 * np.exp((mu_real - 0.5 * sigma**2) * T + sigma * W_T)
# Option payoff
K = 100
payoff = np.maximum(S_T - K, 0)
# Under-Q price = E_P [dQdP * payoff] / E_P [dQdP] * exp(-r T)
price_Q_from_P = np.exp(-r * T) * np.mean(dQdP * payoff) / np.mean(dQdP)
# Direct under-Q sampling for comparison:
W_T_Q = np.random.normal(0, np.sqrt(T), n_samples)
S_T_Q = 100 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T_Q)
payoff_Q = np.maximum(S_T_Q - K, 0)
price_direct = np.exp(-r * T) * np.mean(payoff_Q)
# Black-Scholes closed form
d1 = (np.log(100 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
price_BS = 100 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
print(f"Price via change of measure: {price_Q_from_P:.4f}")
print(f"Price via direct Q-sampling: {price_direct:.4f}")
print(f"Black-Scholes closed form:   {price_BS:.4f}")
```

---

## Topic 1.6.7 — [QUANT APPLICATION] Change of Measure in Finance

### Application 1. Girsanov's theorem

Let $W_t$ be a Brownian motion under $\mathbb P$, and $\theta$ a suitable process. Under the change of measure
$$\frac{d\mathbb Q}{d\mathbb P}\bigg|_{\mathcal F_T} = \exp\!\left(-\int_0^T \theta_s \, dW_s - \frac{1}{2}\int_0^T \theta_s^2 \, ds\right),$$
the process $\widetilde W_t = W_t + \int_0^t \theta_s ds$ is a $\mathbb Q$-Brownian motion. This changes the drift of diffusions without changing the volatility.

R-N derivative makes sense *because* $\mathbb Q \ll \mathbb P$ (and in fact $\mathbb Q \approx \mathbb P$, i.e. equivalent, under Novikov's condition $\mathbb E e^{\frac{1}{2}\int_0^T \theta_s^2 ds} < \infty$).

### Application 2. Risk-neutral pricing

Under the risk-neutral measure $\mathbb Q$, the discounted stock $\widetilde S_t = S_t / B_t$ is a $\mathbb Q$-martingale. The price of a claim $H$ is
$$V_0 = B_0 \mathbb E^{\mathbb Q}[H / B_T] = \mathbb E^{\mathbb P}\left[\frac{d\mathbb Q}{d\mathbb P} \cdot H/B_T\right],$$
so pricing is an expectation under $\mathbb Q$, or equivalently an expectation under $\mathbb P$ weighted by the R-N derivative.

### Application 3. Fundamental theorem of asset pricing (First FTAP)

A market is arbitrage-free iff there exists an equivalent martingale measure $\mathbb Q \approx \mathbb P$ such that discounted prices are $\mathbb Q$-martingales. The equivalence $\mathbb Q \approx \mathbb P$ is exactly "positive events under $\mathbb P$ have positive probability under $\mathbb Q$ and vice versa" — a direct R-N statement. (FTAP is covered in Module 3.5.)

### Application 4. Likelihood ratio tests in statistics

For two probability models $\mathbb P_0, \mathbb P_1$ with $\mathbb P_1 \ll \mathbb P_0$, the **likelihood ratio** is $\Lambda = d\mathbb P_1/d\mathbb P_0$. The Neyman-Pearson lemma says: the most powerful test for size $\alpha$ rejects $H_0$ when $\Lambda > k_\alpha$ for the critical value $k_\alpha$.

### Application 5. Importance sampling

To compute $\mathbb E^{\mathbb P} h(X)$, if the integrand is small except on a rare event, draw samples from a different measure $\mathbb Q$ (emphasising the rare event) and reweight:
$$\mathbb E^{\mathbb P} h(X) = \mathbb E^{\mathbb Q}[h(X) \cdot d\mathbb P/d\mathbb Q].$$
This reduces variance dramatically when $d\mathbb P/d\mathbb Q$ can be controlled. In finance: simulating deep-out-of-the-money option payoffs, credit defaults, VaR exceedances.

### Application 6. Heath-Jarrow-Morton and forward measures

Under the forward measure $\mathbb Q^{T}$ (using the $T$-maturity bond as numeraire), forward rates are martingales. Change of measure from spot $\mathbb Q$ to $\mathbb Q^T$ is given by an R-N derivative involving the bond discounting. This simplifies pricing of caplets, swaptions, etc.

### Application 7. Malliavin weights

Greeks can be computed as $\partial V / \partial \theta = \mathbb E^{\mathbb Q}[H(X) \cdot W(\theta)]$ where $W(\theta)$ is a Malliavin weight — an R-N derivative w.r.t. a perturbation of the parameter $\theta$. This bypasses differentiation of the payoff (useful for discontinuous $H$).

### Application 8. Minimal martingale measure and incomplete markets

In incomplete markets there are infinitely many equivalent martingale measures. The **minimal martingale measure** is the $\mathbb Q$ that is closest to $\mathbb P$ in an $L^2$ sense: $\mathbb Q = \operatorname{argmin}_{\mathbb Q' \approx \mathbb P \text{ mart}} \mathbb E^{\mathbb P}[(d\mathbb Q'/d\mathbb P - 1)^2]$. This is a convex optimization in the space of R-N densities (an $L^2$ ball projected onto the martingale constraint). Föllmer-Schweizer and Föllmer-Sondermann hedging use this measure.

---

## Topic 1.6.8 — Exercises

### ★ (Foundational)

**1.6.E1.** Show: if $\nu$ is signed and $\mu$ positive, $\nu \ll \mu \iff \nu^+ \ll \mu$ and $\nu^- \ll \mu$.

**1.6.E2.** Give an example of a signed measure on $\mathbb R$ that is neither $\ll \lambda$ nor $\perp \lambda$, and show its Lebesgue decomposition.

**1.6.E3.** Compute the R-N derivative $d\nu/d\lambda$ for $\nu$ with CDF $F(x) = x^2 \mathbf 1_{[0, 1]}(x) + \mathbf 1_{(1, \infty)}(x)$.

**1.6.E4.** Let $\nu$ be the Dirac measure at $0$ and $\mu = \lambda$ on $\mathbb R$. Show $\nu \perp \mu$ via a Hahn-type split.

**1.6.E5.** Show: the sum of two absolutely continuous measures (w.r.t. $\mu$) is absolutely continuous, with density = sum of densities.

**1.6.E6.** Show: if $\mu \ll \nu$ and $\nu \ll \mu$, then $d\mu/d\nu \cdot d\nu/d\mu = 1$ $\mu$-a.e.

**1.6.E7.** In the Black-Scholes setup of Application 1, verify $\mathbb E^{\mathbb P}[d\mathbb Q/d\mathbb P] = 1$ directly from the formula for $dQ/dP$.

**1.6.E8.** Let $X \sim N(0, 1)$, $\mathbb P_0 = N(0, 1)$, $\mathbb P_1 = N(\mu, 1)$. Compute $d\mathbb P_1/d\mathbb P_0$ as a function of $x$.

### ★★ (Core)

**1.6.E9.** **Radon-Nikodym chain rule.** If $\nu \ll \mu \ll \rho$ with all $\sigma$-finite, prove $d\nu/d\rho = (d\nu/d\mu)(d\mu/d\rho)$ $\rho$-a.e. Use the standard machine.

**1.6.E10.** Prove Hahn-Jordan decomposition **uniqueness**: if $\nu = \mu_1 - \mu_2 = \mu'_1 - \mu'_2$ with $\mu_i, \mu'_i$ positive and mutually singular pairs, then $\mu_i = \mu'_i$.

**1.6.E11.** Show: if $\mu \ll \nu \ll \rho$, $\mu \ll \rho$.

**1.6.E12.** For the Cantor distribution $\nu_C$, prove directly that $\nu_C \perp \lambda$ by constructing the Hahn pair.

**1.6.E13.** **Lebesgue decomposition of a mixture.** Let $\nu = \alpha \lambda + \beta \nu_C + \gamma \delta_0$ with $\alpha, \beta, \gamma > 0$. Find the Lebesgue decomposition of $\nu$ w.r.t. $\lambda$.

**1.6.E14.** **Total variation inequality.** Show $|\nu|(E) = \sup \{ \sum_i |\nu(E_i)| : \{E_i\} \text{ finite measurable partition of } E\}$.

**1.6.E15.** Using the L'Hospital-style Lebesgue differentiation theorem (Module 1.7 preview), prove that for a $\sigma$-finite $\nu \ll \lambda$ on $\mathbb R^n$, $d\nu/d\lambda(x) = \lim_{r \to 0} \frac{\nu(B_r(x))}{\lambda(B_r(x))}$ for $\lambda$-a.e. $x$.

**1.6.E16.** Prove: $\nu_a$ and $\nu_s$ in Lebesgue decomposition satisfy $\nu_a \ll \mu$ and $\nu_s \perp \mu$ are the unique way to split if we require the two pieces to be absolutely continuous / singular w.r.t. $\mu$ respectively.

### ★★★ (Challenging / quant-relevant)

**1.6.E17.** **Riesz decomposition of supermartingales.** Use R-N and conditional expectation to prove: every non-negative supermartingale $X_t$ decomposes as $X_t = M_t - A_t$ where $M_t$ is a martingale and $A_t$ is a non-decreasing predictable process with $A_0 = 0$. (This is the Doob-Meyer decomposition, stated in Module 2.6.)

**1.6.E18.** **Kakutani's theorem (almost).** Let $\mathbb P, \mathbb Q$ be product measures $\prod_n \mathbb P_n, \prod_n \mathbb Q_n$ with $\mathbb P_n \approx \mathbb Q_n$ for each $n$. Show: $\mathbb P \approx \mathbb Q$ or $\mathbb P \perp \mathbb Q$, with a concrete criterion (involving $\prod \int \sqrt{d\mathbb Q_n/d\mathbb P_n} d\mathbb P_n$).

**1.6.E19.** **Novikov's condition and Girsanov.** For $\theta$ adapted with $\mathbb E e^{\frac{1}{2}\int_0^T \theta_s^2 ds} < \infty$, prove the exponential martingale $Z_t = \exp(-\int_0^t \theta_s dW_s - \frac{1}{2}\int_0^t \theta_s^2 ds)$ is a martingale, so $dQ/dP = Z_T$ defines an equivalent measure.

**1.6.E20.** **Quant: Black-Scholes via Girsanov.** Using the R-N derivative for the change from physical to risk-neutral in Black-Scholes, derive the $\mathbb E^{\mathbb Q} (S_T - K)^+$ formula and show it equals $S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$.

**1.6.E21.** **Quant: The minimal entropy martingale measure.** In an incomplete market, among equivalent martingale measures $\mathbb Q$, the minimal entropy one minimizes the relative entropy $H(\mathbb Q | \mathbb P) = \mathbb E^{\mathbb Q} \log(d\mathbb Q/d\mathbb P)$. Express this in terms of R-N derivatives and show the solution has a specific exponential form in the martingale representation.

**1.6.E22.** **Quant: HJM drift condition.** Given a forward rate $f(t, T)$ with dynamics $df = \alpha dt + \sigma dW$, use the R-N/Girsanov framework to derive the Heath-Jarrow-Morton no-arbitrage drift condition $\alpha(t, T) = \sigma(t, T) \int_t^T \sigma(t, s) ds$ under the risk-neutral measure.

---

## Module Summary

- Signed / complex measures and their Hahn-Jordan decomposition into positive + negative parts.
- **Absolute continuity** ($\nu \ll \mu$): null sets of $\mu$ are null for $\nu$. Has ε-δ characterization for finite measures.
- **Radon-Nikodym theorem**: for $\sigma$-finite $\mu, \nu$ with $\nu \ll \mu$, $\nu$ has a density $f = d\nu/d\mu$ w.r.t. $\mu$. Proof via von Neumann's Hilbert-space argument.
- **Chain rule and change of variable** for R-N derivatives.
- **Lebesgue decomposition**: every $\sigma$-finite $\nu$ splits uniquely as $\nu_a + \nu_s$ (absolutely continuous + singular).
- **Conditional expectation** is an R-N derivative on the sub-σ-algebra, laying the foundation of probability.
- Quant applications: Girsanov, risk-neutral measure, FTAP, likelihood ratios, importance sampling, HJM, Malliavin weights.

**Forward pointers:**

This module closes Subject 1 — the measure-theoretic foundation. We now have:
- Spaces to integrate over (Module 1.1)
- Functions we can integrate (Module 1.2)
- Values we can assign to those integrals (Module 1.3)
- Multi-dimensional integrals (Module 1.4)
- Spaces $L^p$ where those integrals live (Module 1.5)
- Ways to change measures and extract densities (Module 1.6)

**Subject 2 — Probability Theory** starts here:
- Module 2.1: Probability spaces, random variables, distributions, independence.
- Module 2.2: Expectation, variance, conditional expectation (using Module 1.6).
- Module 2.3: Modes of convergence, Borel-Cantelli.
- Module 2.4: Law of large numbers, central limit theorem.
- Module 2.5: Characteristic functions (fleshing out Module 0.5).
- Module 2.6: Martingales and filtrations.
- Module 2.7: Markov chains.

**Recommended reading:**
- Folland, *Real Analysis*, Chapter 3.
- Rudin, *Real and Complex Analysis*, Chapters 6-7.
- Billingsley, *Probability and Measure*, Chapter 32 (R-N & conditional expectation).
- Williams, *Probability with Martingales*, Chapter 14 (R-N as 'the' theorem of probability).
- Karatzas & Shreve, *Brownian Motion and Stochastic Calculus*, Chapter 3 for Girsanov.

**Closing thought on Subject 1.** Everything we have built — from the humble σ-algebra to the grand Lebesgue decomposition — is a preparation for probability theory. The next subject takes $(\Omega, \mathcal F, \mathbb P)$ as given and studies random variables, their distributions, and their limiting behavior with the tools we have now established.

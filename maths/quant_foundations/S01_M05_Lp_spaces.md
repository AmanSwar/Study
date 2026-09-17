# Module 1.5 — $L^p$ Spaces

> *"The class $L^p$ is for functions what $\ell^p$ is for sequences: a complete normed vector space where norm convergence controls integration."*

## Prerequisites

- **Module 1.1–1.4** — full Lebesgue integration theory, Fubini, null sets and completion.
- **Module 0.1–0.3** — sequences and series in $\mathbb R$, uniform convergence, basics of metric spaces.
- **Module 0.4** — normed spaces, inner products, Banach and Hilbert space basics (foundational; we develop everything here anyway).

## Overview

Today we promote the Lebesgue integral from a tool for assigning numbers to functions, to a tool for *measuring distance between functions*. The resulting spaces $L^p$ — for $p \in [1, \infty]$ — are the universal function spaces of analysis. Specifically:

- **$L^1$**: integrable functions, with norm $\|f\|_1 = \int |f| \, d\mu$. Home of the fundamental convergence theorems; natural space for convolutions, characteristic functions, density functions.
- **$L^2$**: square-integrable functions, with norm $\|f\|_2 = (\int |f|^2 d\mu)^{1/2}$. A *Hilbert space* (inner product $\langle f, g\rangle = \int fg$); the home of Fourier analysis and quantum mechanics.
- **$L^\infty$**: essentially bounded functions, with norm $\|f\|_\infty = \operatorname{ess\,sup} |f|$. Dual of $L^1$ (for $\sigma$-finite $\mu$).
- **General $L^p$**: interpolates between these, with Hölder $\|fg\|_1 \le \|f\|_p \|g\|_q$ and Minkowski $\|f+g\|_p \le \|f\|_p + \|g\|_p$.

Four theorems drive the module:

1. **Hölder's inequality** (and its corollary, Cauchy-Schwarz for $p = q = 2$).
2. **Minkowski's inequality** — the triangle inequality for $\|\cdot\|_p$.
3. **Riesz-Fischer theorem** — completeness: $L^p$ is a Banach space for all $p \in [1, \infty]$.
4. **Density theorems** — simple functions, continuous compactly-supported functions, and Schwartz functions are dense in $L^p(\mathbb R^n, \lambda)$ for $1 \le p < \infty$.

We will also cover the dual space of $L^p$ for $1 \le p < \infty$ (which is $L^q$ for $1/p + 1/q = 1$), weak convergence, and quant-relevant applications including $L^2$-projections for hedging, mean-variance optimization, and $L^\infty$ for essentially bounded payoffs.

Throughout $(X, \mathcal F, \mu)$ is a measure space (assumed $\sigma$-finite when we need it); functions are measurable $X \to \mathbb R$ or $X \to \mathbb C$ unless stated.

---

## Topic 1.5.1 — Definition of $L^p$

### Definition 1.5.1 (The space $\mathcal L^p$, before quotient)

For $p \in [1, \infty)$, let
$$\mathcal L^p(X, \mathcal F, \mu) := \left\{ f: X \to \mathbb C \text{ measurable} : \|f\|_p := \left(\int_X |f|^p \, d\mu\right)^{1/p} < \infty \right\}.$$

For $p = \infty$, define the **essential supremum**
$$\|f\|_\infty := \operatorname{ess\,sup} |f| := \inf\{M \ge 0 : \mu(\{|f| > M\}) = 0\}.$$
Note $\|f\|_\infty = \infty$ is possible. Let
$$\mathcal L^\infty(X, \mathcal F, \mu) := \{f : \|f\|_\infty < \infty\}.$$

### Lemma 1.5.2 (Essential sup is achieved a.e.)

For $f$ measurable, $|f(x)| \le \|f\|_\infty$ for $\mu$-a.e. $x$ (when $\|f\|_\infty < \infty$).

**Proof.** Let $M_n = \|f\|_\infty + 1/n$; by definition of the infimum, $\mu(\{|f| > \|f\|_\infty\}) \le \mu(\{|f| > M_n - 1/n\})$... actually let me argue directly. Let $E = \{|f| > \|f\|_\infty\}$; we want $\mu(E) = 0$. Write $E = \bigcup_n E_n$ where $E_n = \{|f| > \|f\|_\infty + 1/n\}$. Each $E_n$ has the property that for any $M < \|f\|_\infty + 1/n$ we have $\mu(\{|f| > M\}) > 0$, but we want $\mu(E_n)$ itself. Take $M = \|f\|_\infty + 1/(2n)$: then $\mu(\{|f| > M\}) = 0$ (because the infimum in the essential sup's definition is achieved by any $M > \|f\|_\infty$; the inf $\alpha$ of a set of "$M$ with $\mu \{|f| > M\} = 0$" satisfies that all $M > \alpha$ are in the set, so $\mu \{|f| > \alpha + 1/(2n)\} = 0$). Since $E_n \subset \{|f| > \|f\|_\infty + 1/(2n)\}$, $\mu(E_n) = 0$. Hence $\mu(E) \le \sum \mu(E_n) = 0$. $\blacksquare$

### Definition 1.5.3 (The Banach space $L^p$)

Two measurable functions $f, g$ are **equivalent** ($f \sim g$) if $f = g$ $\mu$-a.e. This is an equivalence relation on measurable functions. Define
$$L^p(X, \mathcal F, \mu) := \mathcal L^p / \sim,$$
the quotient space of $\mathcal L^p$ modulo a.e. equality. Elements of $L^p$ are equivalence classes, but we conventionally write $f \in L^p$ and treat $f$ as a function (understanding that a.e.-equal functions are identified).

The quotient is necessary because $\|f\|_p = 0$ does not imply $f = 0$ pointwise (only a.e.): e.g. $f = \mathbf 1_{\{x_0\}}$ has $\|f\|_p = 0$ under Lebesgue. Without the quotient, $\|\cdot\|_p$ would be a semi-norm, not a norm.

### Remark. $L^p$ is typically a function space, not a space of pointwise values

Many classical formulas like "$f(x_0)$" don't make sense for an $L^p$ equivalence class. However, for a specific equivalence class representative, point values are defined, and theorems like **Lebesgue's differentiation theorem** (Module 1.7 preview) identify "canonical" pointwise values for $L^p$ functions via averages over shrinking balls.

---

## Topic 1.5.2 — Hölder's and Minkowski's Inequalities

### Definition 1.5.4 (Conjugate exponents)

For $p \in [1, \infty]$, the **conjugate exponent** $q$ satisfies
$$\frac{1}{p} + \frac{1}{q} = 1, \qquad \text{with conventions } \frac{1}{\infty} = 0, \; p=1 \iff q=\infty.$$

### Lemma 1.5.5 (Young's inequality)

For $a, b \ge 0$ and conjugate $p, q \in (1, \infty)$,
$$ab \le \frac{a^p}{p} + \frac{b^q}{q}.$$
Equality iff $a^p = b^q$.

**Proof.** If $a = 0$ or $b = 0$ the inequality is trivial. Otherwise, consider the concave function $\log t$. The inequality is $\log(ab) \le \log\left(\frac{a^p}{p} + \frac{b^q}{q}\right)$... actually, easier to prove by writing
$$\log(ab) = \log a + \log b = \frac{1}{p}\log a^p + \frac{1}{q} \log b^q \le \log\!\left(\frac{a^p}{p} + \frac{b^q}{q}\right),$$
by concavity of $\log$ (Jensen for two points with weights $1/p, 1/q$). Exponentiating gives $ab \le \frac{a^p}{p} + \frac{b^q}{q}$. $\blacksquare$

### Theorem 1.5.6 (Hölder's inequality)

For $p, q \in [1, \infty]$ conjugate, $f \in L^p(\mu)$, $g \in L^q(\mu)$:
$$\|fg\|_1 = \int_X |fg| \, d\mu \le \|f\|_p \|g\|_q.$$

**Proof.**

*Case $p = 1, q = \infty$.* $|fg| \le \|g\|_\infty |f|$ a.e., so $\int |fg| \le \|g\|_\infty \int |f| = \|f\|_1 \|g\|_\infty$.

*Case $p \in (1, \infty)$.* If $\|f\|_p = 0$ then $f = 0$ a.e. and the inequality is $0 \le 0$; similarly for $\|g\|_q = 0$. So assume both are positive and finite.

Normalize: define $\widetilde f = f / \|f\|_p$ and $\widetilde g = g / \|g\|_q$, so $\|\widetilde f\|_p = \|\widetilde g\|_q = 1$. Apply Young's inequality pointwise to $|\widetilde f|, |\widetilde g|$:
$$|\widetilde f(x) \widetilde g(x)| \le \frac{|\widetilde f(x)|^p}{p} + \frac{|\widetilde g(x)|^q}{q}.$$
Integrate: $\int |\widetilde f \widetilde g| \, d\mu \le \frac{1}{p} \|\widetilde f\|_p^p + \frac{1}{q} \|\widetilde g\|_q^q = \frac{1}{p} + \frac{1}{q} = 1$. So $\int |fg| \le \|f\|_p \|g\|_q$. $\blacksquare$

### Corollary 1.5.7 (Cauchy-Schwarz)

$p = q = 2$: $\int |fg| \, d\mu \le \|f\|_2 \|g\|_2$. Equivalently, $|\langle f, g\rangle| \le \|f\|_2 \|g\|_2$ where $\langle f, g\rangle = \int f \bar g \, d\mu$.

### Corollary 1.5.8 (Hölder gives inclusion on finite measure spaces)

If $\mu(X) < \infty$ and $1 \le p \le r \le \infty$, then $L^r(\mu) \subset L^p(\mu)$ with $\|f\|_p \le \mu(X)^{1/p - 1/r} \|f\|_r$.

**Proof.** If $r = \infty$: $|f|^p \le \|f\|_\infty^p$ a.e., so $\int |f|^p \le \|f\|_\infty^p \mu(X)$, and taking $p$-th roots, $\|f\|_p \le \mu(X)^{1/p} \|f\|_\infty$.

If $r < \infty$: apply Hölder with exponents $r/p$ and $(r/p)' = r/(r-p)$ to the product $|f|^p \cdot 1$:
$$\int |f|^p \, d\mu = \int |f|^p \cdot 1 \, d\mu \le (\int |f|^r \, d\mu)^{p/r} (\mu(X))^{(r-p)/r} = \|f\|_r^p \mu(X)^{(r-p)/r}.$$
Taking $p$-th roots gives $\|f\|_p \le \|f\|_r \mu(X)^{1/p - 1/r}$. $\blacksquare$

**Warning.** The inclusion $L^r \subset L^p$ **fails** on infinite measure spaces. On $(\mathbb R, \lambda)$, $\mathbf 1_{[0, \infty)}(x) / (1 + x^2)$ is in $L^2$ but not $L^1$; conversely, $x^{-1/2} \mathbf 1_{(0, 1)}$ is in $L^1$ but not $L^2$.

### Theorem 1.5.9 (Minkowski's inequality)

For $p \in [1, \infty]$ and $f, g \in L^p(\mu)$:
$$\|f + g\|_p \le \|f\|_p + \|g\|_p.$$

**Proof.**

*Case $p = 1$.* $|f + g| \le |f| + |g|$, integrate.

*Case $p = \infty$.* $|f + g| \le \|f\|_\infty + \|g\|_\infty$ a.e., so $\|f + g\|_\infty \le \|f\|_\infty + \|g\|_\infty$.

*Case $p \in (1, \infty)$.* Write
$$|f + g|^p = |f + g| \cdot |f + g|^{p-1} \le (|f| + |g|) \cdot |f + g|^{p-1}.$$
Integrate, then apply Hölder with conjugate exponents $p$ and $q = p/(p-1)$ to each term on the right:
$$\int |f| \cdot |f + g|^{p-1} \, d\mu \le \|f\|_p \|(f+g)^{p-1}\|_q = \|f\|_p \left(\int |f+g|^{(p-1)q} \, d\mu\right)^{1/q} = \|f\|_p \|f+g\|_p^{p-1},$$
using $(p-1)q = p$. Similarly for the $|g|$ term:
$$\int |f+g|^p \, d\mu \le (\|f\|_p + \|g\|_p) \|f+g\|_p^{p-1}.$$
If $\|f+g\|_p = 0$, the inequality is trivial. Otherwise divide by $\|f+g\|_p^{p-1}$ (finite because $|f+g| \le 2 \max(|f|, |g|)$ gives $|f+g|^p \le 2^p (|f|^p + |g|^p)$, so $\|f+g\|_p < \infty$):
$$\|f+g\|_p = \|f+g\|_p^p / \|f+g\|_p^{p-1} \le \|f\|_p + \|g\|_p. \qquad \blacksquare$$

### Remark. $\|\cdot\|_p$ is a norm on $L^p$

Properties: (i) $\|f\|_p \ge 0$ with equality iff $f = 0$ in $L^p$ (i.e. $f = 0$ a.e., which is exactly the zero equivalence class). (ii) $\|\alpha f\|_p = |\alpha| \|f\|_p$. (iii) Triangle inequality = Minkowski.

For $p \in (0, 1)$ Minkowski fails; $L^p$ is only a *quasi-norm* space (with $\|f+g\|_p \le 2^{1/p - 1}(\|f\|_p + \|g\|_p)$) or equivalently, metric space under $d(f, g) = \int |f - g|^p$.

---

## Topic 1.5.3 — Completeness: the Riesz-Fischer Theorem

### Theorem 1.5.10 (Riesz-Fischer)

For any measure space $(X, \mathcal F, \mu)$ and $p \in [1, \infty]$, $L^p(\mu)$ is a Banach space (i.e., complete under $\|\cdot\|_p$).

**Proof.**

*Case $p \in [1, \infty)$.* Let $\{f_n\}$ be Cauchy in $L^p$. We need to produce a limit $f \in L^p$ with $\|f_n - f\|_p \to 0$.

*Step 1: extract a fast-convergent subsequence.* For each $k \ge 1$, pick $n_k$ so that $m, n \ge n_k$ implies $\|f_m - f_n\|_p < 2^{-k}$ (possible by Cauchy; choose $n_{k+1} > n_k$). Let $g_k = f_{n_k}$. Then $\|g_{k+1} - g_k\|_p < 2^{-k}$.

*Step 2: construct a pointwise limit.* Let
$$S(x) := \sum_{k=1}^\infty |g_{k+1}(x) - g_k(x)|, \qquad G_N(x) := \sum_{k=1}^N |g_{k+1}(x) - g_k(x)|.$$
$G_N$ is measurable, non-negative, and $G_N \uparrow S$. By Minkowski (repeated application) or MCT:
$$\|G_N\|_p \le \sum_{k=1}^N \|g_{k+1} - g_k\|_p < \sum_{k=1}^\infty 2^{-k} = 1.$$
By MCT applied to $|G_N|^p$ (non-negative, increasing in $N$), $\int |S|^p \, d\mu \le 1 < \infty$. In particular, $S < \infty$ $\mu$-a.e., so the series $\sum_k (g_{k+1}(x) - g_k(x))$ converges absolutely for a.e. $x$. Hence the telescoping partial sums $g_N(x) = g_1(x) + \sum_{k=1}^{N-1}(g_{k+1}(x) - g_k(x))$ converge, i.e. $g_N(x) \to f(x)$ for some $f$ defined a.e.

Set $f = 0$ on the exceptional null set.

*Step 3: $f \in L^p$ and $g_N \to f$ in $L^p$.* $|f - g_N| \le \sum_{k \ge N} |g_{k+1} - g_k| \le S$, so $|f - g_N|^p \le S^p \in L^1$. Moreover $|f - g_N| \to 0$ a.e. By DCT,
$$\|f - g_N\|_p^p = \int |f - g_N|^p \, d\mu \to 0.$$
In particular $f - g_N \in L^p$, so $f = g_N + (f - g_N) \in L^p$.

*Step 4: the full Cauchy sequence converges.* $\|f_n - f\|_p \le \|f_n - g_k\|_p + \|g_k - f\|_p$; both terms go to $0$ (the second by Step 3, the first by the Cauchy property since $n_k \ge n$ for $k$ large). $\blacksquare$

*Case $p = \infty$.* Let $\{f_n\}$ be Cauchy in $L^\infty$. For each pair $m, n$, set $E_{mn} = \{|f_m - f_n| > \|f_m - f_n\|_\infty\}$; this is a null set. Let $E = \bigcup_{m, n} E_{mn}$, also null. Off $E$, $\{f_n(x)\}$ is a Cauchy sequence in $\mathbb C$ (uniform in $x$, since $|f_m(x) - f_n(x)| \le \|f_m - f_n\|_\infty \to 0$). Define $f(x)$ as the pointwise limit (and $0$ on $E$). Then $|f_n - f|(x) \le \sup_{m \ge n} |f_n - f_m|(x) \le \sup_{m \ge n} \|f_n - f_m\|_\infty \to 0$ uniformly off $E$, so $\|f_n - f\|_\infty \to 0$. $\blacksquare$

### Corollary 1.5.11 (Cauchy in $L^p$ implies a subsequence converges a.e.)

In the $p \in [1, \infty)$ case, Step 2 above shows: if $\{f_n\}$ is Cauchy in $L^p$, there is a subsequence $\{f_{n_k}\}$ converging to the $L^p$-limit $f$ $\mu$-almost everywhere.

**Remark.** The *full* sequence need not converge a.e. The classic counterexample is the **typewriter sequence** in Module 1.2, where $f_n \to 0$ in $L^p$ but pointwise convergence fails at every point.

### Theorem 1.5.12 ($L^2$ is a Hilbert space)

$L^2(\mu)$ with inner product $\langle f, g\rangle := \int f \bar g \, d\mu$ is a Hilbert space (complete inner product space). The norm is $\|f\|_2 = \sqrt{\langle f, f\rangle}$.

**Proof.** Inner-product axioms (conjugate bilinearity, positive definiteness modulo a.e.) are immediate. Completeness is Riesz-Fischer for $p = 2$. $\blacksquare$

This is the most important special case: Hilbert space machinery (orthogonal projection, Riesz representation, adjoints, orthonormal bases, spectral theory) applies to $L^2$.

---

## Topic 1.5.4 — Density Theorems

### Theorem 1.5.13 (Simple functions dense in $L^p$)

For $1 \le p < \infty$, the set of **simple functions vanishing outside a set of finite measure** is dense in $L^p(\mu)$.

**Proof.** Let $f \in L^p$, WLOG $f \ge 0$ (treat real and imaginary, positive and negative parts separately). By Module 1.2 approximation theorem, there exist simple $0 \le \varphi_n \uparrow f$ with $\varphi_n \le f$ everywhere. Then $|f - \varphi_n|^p \le f^p \in L^1$ (since $f \in L^p$), and $|f - \varphi_n|^p \to 0$ pointwise. DCT gives $\|f - \varphi_n\|_p \to 0$.

For finiteness of support: each $\varphi_n$ has only finitely many nonzero values $a_1, \ldots, a_k$, supported on $A_1, \ldots, A_k \subset \{f > 0\}$. If some $\mu(A_i) = \infty$, that contradicts $\int \varphi_n^p \le \int f^p < \infty$. So $\mu(A_i) < \infty$. $\blacksquare$

### Theorem 1.5.14 ($C_c$ dense in $L^p(\mathbb R^n, \lambda)$)

For $1 \le p < \infty$, continuous compactly supported functions $C_c(\mathbb R^n)$ are dense in $L^p(\mathbb R^n, \lambda)$.

**Proof.** By Theorem 1.5.13 it suffices to approximate each finite-support simple function $\varphi = \sum_i a_i \mathbf 1_{A_i}$ (with $A_i$ of finite measure) in $L^p$ by a $C_c$ function. By linearity, it suffices to approximate $\mathbf 1_A$ with $\lambda(A) < \infty$.

**Step 1: replace $A$ by an open bounded set.** By outer regularity of Lebesgue measure (Module 1.1), for $\varepsilon > 0$ there is an open $U \supset A$ with $\lambda(U \setminus A) < \varepsilon^p / 2$. Restricting further, $U$ can be chosen bounded (intersect with a large ball of finite measure containing $A$ up to a small correction).

**Step 2: replace $\mathbf 1_U$ by a $C_c$ function.** For an open bounded $U$, let $K$ be a compact subset with $\lambda(U \setminus K) < \varepsilon^p / 2$ (by inner regularity). Construct a continuous function $g \in C_c$ with $g = 1$ on $K$, $g = 0$ outside $U$, $0 \le g \le 1$. Standard constructions: $g(x) = \max(0, 1 - n \cdot d(x, K))$ for $n$ large enough, or Urysohn's lemma.

**Step 3: combine.** $\|\mathbf 1_A - g\|_p^p \le \int |\mathbf 1_A - \mathbf 1_U|^p + \int |\mathbf 1_U - g|^p \le \lambda(U \setminus A) + \lambda(U \setminus K) < \varepsilon^p$. So $\|\mathbf 1_A - g\|_p < \varepsilon$. $\blacksquare$

### Theorem 1.5.15 (Schwartz functions dense in $L^p(\mathbb R^n, \lambda)$)

For $1 \le p < \infty$, the Schwartz space $\mathcal S(\mathbb R^n)$ (smooth functions with all derivatives decaying faster than polynomially) is dense in $L^p(\mathbb R^n, \lambda)$. In particular, $C_c^\infty$ is dense.

**Proof.** By Theorem 1.5.14, $C_c$ is dense, so it suffices to approximate $f \in C_c$ by Schwartz functions. Let $\eta \in C_c^\infty$ be a **mollifier**: $\eta \ge 0$, $\int \eta = 1$, $\operatorname{supp} \eta \subset B_1(0)$. Define $\eta_\delta(x) = \delta^{-n} \eta(x/\delta)$. Then $f_\delta := f * \eta_\delta \in C_c^\infty \subset \mathcal S$ (convolution inherits smoothness).

As $\delta \to 0$, $f_\delta \to f$ uniformly (by uniform continuity of $f$), hence in $L^p$ (since both are supported in a bounded set, uniform convergence implies $L^p$ convergence). $\blacksquare$

### Proposition 1.5.16 (Continuity of translation)

For $1 \le p < \infty$, translation is continuous on $L^p(\mathbb R^n, \lambda)$: for $f \in L^p$ and $h \in \mathbb R^n$, denote $\tau_h f(x) := f(x - h)$. Then $\|\tau_h f - f\|_p \to 0$ as $h \to 0$.

**Proof.** By density of $C_c$, approximate: given $\varepsilon > 0$, pick $g \in C_c$ with $\|f - g\|_p < \varepsilon$. For $h$ small, $\|\tau_h g - g\|_p < \varepsilon$ (uniform continuity of $g$ on compact support gives pointwise convergence uniformly, and $L^p$ convergence on bounded support). Then
$$\|\tau_h f - f\|_p \le \|\tau_h f - \tau_h g\|_p + \|\tau_h g - g\|_p + \|g - f\|_p = 2\|f - g\|_p + \|\tau_h g - g\|_p < 3\varepsilon,$$
using translation invariance of Lebesgue measure: $\|\tau_h f - \tau_h g\|_p = \|f - g\|_p$. $\blacksquare$

**Warning.** Translation is NOT continuous on $L^\infty$. Counterexample: $f = \mathbf 1_{[0, 1]}$, $\|\tau_h f - f\|_\infty = 1$ for all $h \ne 0$.

---

## Topic 1.5.5 — The Dual of $L^p$

### Definition 1.5.17 (Continuous linear functional)

A **continuous linear functional** on $L^p$ is a linear map $\Lambda: L^p \to \mathbb C$ with $|\Lambda f| \le C \|f\|_p$ for some constant $C$. The smallest such $C$ is $\|\Lambda\|_{(L^p)^*}$.

### Theorem 1.5.18 (Riesz representation for $L^p$, $1 \le p < \infty$, $\sigma$-finite)

Let $(X, \mathcal F, \mu)$ be $\sigma$-finite, $1 \le p < \infty$, and $q$ the conjugate exponent. For each $g \in L^q(\mu)$, the map
$$\Lambda_g: L^p \to \mathbb C, \qquad \Lambda_g(f) := \int f g \, d\mu$$
is a continuous linear functional with $\|\Lambda_g\| = \|g\|_q$. Conversely, every continuous linear functional on $L^p$ arises this way, and the map $g \mapsto \Lambda_g$ is an isometric isomorphism $L^q \cong (L^p)^*$.

**Proof (sketch).**

*$\|\Lambda_g\| \le \|g\|_q$.* Hölder: $|\Lambda_g(f)| \le \|f\|_p \|g\|_q$.

*$\|\Lambda_g\| \ge \|g\|_q$.* For $q < \infty$: let $h = |g|^{q-1} \operatorname{sgn} \bar g$, so $gh = |g|^q$ and $h \in L^p$ with $\|h\|_p = \|g\|_q^{q/p}$. Then $\Lambda_g(h) = \|g\|_q^q = \|h\|_p \|g\|_q$, so $\|\Lambda_g\| \ge \|g\|_q$.

For $q = \infty$: for any $M < \|g\|_\infty$, the set $E = \{|g| > M\}$ has $\mu(E) > 0$ (and finite by $\sigma$-finiteness). Take $f = \mathbf 1_E \operatorname{sgn} \bar g \cdot (1/\mu(E))$, so $\|f\|_1 = 1$ and $\Lambda_g(f) = \int_E |g|/\mu(E) \, d\mu \ge M$. So $\|\Lambda_g\| \ge M$, hence $\ge \|g\|_\infty$.

*Surjectivity (harder, omitted details — uses Radon-Nikodym).* Given $\Lambda \in (L^p)^*$, define the signed measure $\nu(E) := \Lambda(\mathbf 1_E)$ (for $\mu(E) < \infty$). Show $\nu \ll \mu$ with $|\nu(E)| \le \|\Lambda\| \mu(E)^{1/p}$; by Radon-Nikodym (Module 1.6) $\nu = g \cdot \mu$ for some $g$. Check $g \in L^q$ and $\Lambda = \Lambda_g$. The full proof is Folland Theorem 6.15 or Rudin RCA Theorem 6.16. $\blacksquare$

### Corollary 1.5.19 (Reflexivity of $L^p$, $1 < p < \infty$)

For $1 < p < \infty$, $(L^p)^{**} \cong L^p$ isometrically — $L^p$ is **reflexive**.

$L^1$ and $L^\infty$ are **not reflexive** in general. $(L^1)^* = L^\infty$ but $(L^\infty)^*$ is strictly larger than $L^1$ (contains exotic finitely-additive measures).

### Theorem 1.5.20 (Weak convergence in $L^p$)

$f_n \to f$ *weakly* in $L^p$ (for $1 \le p < \infty$) iff $\int f_n g \to \int f g$ for every $g \in L^q$. Weak convergence allows things like $\sin(nx) \to 0$ weakly in $L^2([0, 2\pi])$ even though $\sin(nx) \not\to 0$ pointwise or in norm.

**Banach-Alaoglu + reflexivity:** the unit ball of $L^p$ is weakly (sequentially) compact for $1 < p < \infty$. This is the mathematical foundation of calculus of variations / optimization (e.g. finding a minimum via weak limits).

---

## Topic 1.5.6 — Python: Computations in $L^p$

```python
"""
Numerical experiments illustrating Hölder, Minkowski, completeness, and density.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- Verify Hölder for sequences (L^p on N with counting measure is l^p) ----------
def holder_check(a, b, p, q):
    """Verify sum |a_n b_n| <= (sum |a|^p)^{1/p} (sum |b|^q)^{1/q}"""
    lhs = np.sum(np.abs(a * b))
    rhs = (np.sum(np.abs(a)**p))**(1/p) * (np.sum(np.abs(b)**q))**(1/q)
    return lhs, rhs

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])
for (p, q) in [(2, 2), (3, 1.5), (4, 4/3), (10, 10/9)]:
    lhs, rhs = holder_check(a, b, p, q)
    print(f"p={p}, q={q}: LHS={lhs:.4f}, RHS={rhs:.4f}, ratio={lhs/rhs:.4f}")

# ---------- Minkowski ----------
def minkowski_check(a, b, p):
    lhs = (np.sum(np.abs(a + b)**p))**(1/p)
    rhs = (np.sum(np.abs(a)**p))**(1/p) + (np.sum(np.abs(b)**p))**(1/p)
    return lhs, rhs

for p in [1, 1.5, 2, 3, 10]:
    lhs, rhs = minkowski_check(a, b, p)
    print(f"p={p}: ||a+b||_p = {lhs:.4f}, ||a||_p + ||b||_p = {rhs:.4f}")

# ---------- Typewriter sequence: L^p convergence without a.e. convergence ----------
def typewriter_sequence(N, x_grid):
    """Returns f_n(x) where f_n is indicator of I_n = [(n - 2^k)/2^k, (n - 2^k + 1)/2^k] for n in [2^k, 2^{k+1})"""
    fs = []
    n = 1
    while n <= N:
        k = int(np.floor(np.log2(n)))
        # sub-interval indices within [0, 1]:
        j = n - 2**k  # 0 <= j < 2^k
        left = j / 2**k
        right = (j + 1) / 2**k
        f = ((x_grid >= left) & (x_grid < right)).astype(float)
        fs.append(f)
        n += 1
    return fs

x = np.linspace(0, 1, 2000)
fs = typewriter_sequence(32, x)
lp_norms = [np.trapz(f**2, x) for f in fs]  # L^2 norm squared
print("L^2 norms squared of typewriter:", lp_norms[:10], "...")
# Norms go to 0, but for each x, f_n(x) = 1 infinitely often (the sequence doesn't converge pointwise).

# ---------- Fourier coefficients of characteristic function: 1_A in L^2([0, 2pi]) ----------
# A = [pi/4, 3pi/4], compute coefficients c_n = <1_A, e^{in x}/sqrt(2pi)>
from scipy.integrate import quad
def char_int(n, a=np.pi/4, b=3*np.pi/4):
    re, _ = quad(lambda x: np.cos(n*x), a, b)
    im, _ = quad(lambda x: -np.sin(n*x), a, b)  # e^{-inx}
    return (re + 1j*im) / np.sqrt(2*np.pi)

coeffs = np.array([char_int(n) for n in range(-50, 51)])
# Parseval: sum |c_n|^2 = ||1_A||_2^2 = (3pi/4 - pi/4) = pi/2
print(f"sum |c_n|^2 = {np.sum(np.abs(coeffs)**2):.4f}, expected pi/2 = {np.pi/2:.4f}")
# ---------- Mollification / density of smooth functions ----------
from scipy.ndimage import gaussian_filter1d

f = ((x > 0.3) & (x < 0.7)).astype(float)  # indicator function, not smooth
for sigma in [0.001, 0.01, 0.05, 0.1]:
    f_smooth = gaussian_filter1d(f, sigma=sigma / (x[1] - x[0]))
    diff = np.trapz((f - f_smooth)**2, x)
    print(f"sigma={sigma}: ||f - f_smooth||_2^2 = {diff:.4f}")
# As sigma -> 0, the smoothed function converges to the indicator in L^2.
```

---

## Topic 1.5.7 — [QUANT APPLICATION] $L^2$ Projection and Hedging

### Application 1. $L^2$ projection = best linear estimator

Let $(\Omega, \mathcal F, \mathbb P)$ be a probability space, $\mathcal G \subset \mathcal F$ a sub-σ-algebra, $X \in L^2(\mathbb P)$. The **conditional expectation** $\mathbb E[X | \mathcal G]$ is the $L^2$-orthogonal projection of $X$ onto $L^2(\mathcal G, \mathbb P)$:
$$\mathbb E[X | \mathcal G] = \mathop{\operatorname{argmin}}_{Y \in L^2(\mathcal G)} \|X - Y\|_2^2.$$

Proof: by Hilbert space orthogonal projection (Module 4.1), the projection exists and is characterized by $\langle X - Y, Z\rangle = 0$ for all $Z \in L^2(\mathcal G)$, i.e. $\mathbb E[(X - Y) Z] = 0$ for all $\mathcal G$-measurable $Z$ in $L^2$. This is the defining property of conditional expectation.

### Application 2. Variance minimization and Sharpe ratios

For a portfolio of returns $R = \sum w_i R_i$ with mean $\mu = w^\top m$ and variance $\sigma^2 = w^\top \Sigma w$, the **Sharpe ratio** is $\mathrm{SR} = (\mu - r)/\sigma$. Minimum-variance portfolios (Markowitz, 1952) solve:
$$\min_w w^\top \Sigma w \quad \text{subject to } w^\top m = \mu^*, \; w^\top \mathbf 1 = 1.$$
Lagrangian → linear system → closed-form solution. The entire theory is Hilbert-space geometry in $L^2$ of the (vector-valued) return random variable.

### Application 3. Delta hedging as $L^2$ minimization

In the Black-Scholes setting, the optimal self-financing hedge $\{\Delta_t\}$ for a claim $H$ minimizes the terminal tracking error $\|H - V_T\|_2$ where $V_T = V_0 + \int_0^T \Delta_t \, dS_t$. By Hilbert-space projection (Clark-Ocone formula in Module 4.7), this optimal $\Delta_t$ is the Malliavin derivative (a.k.a. the hedging strategy from the martingale representation theorem).

### Application 4. Cauchy-Schwarz and Sharpe bounds

For two portfolios $P_1, P_2$ with returns $R_1, R_2 \in L^2$,
$$\operatorname{Cov}(R_1, R_2) \le \sqrt{\operatorname{Var}(R_1) \operatorname{Var}(R_2)}$$
by Cauchy-Schwarz. This bounds the correlation coefficient $|\rho| \le 1$ and is the basis of factor model constraints.

### Application 5. $L^\infty$ and essentially bounded payoffs

Bounded payoffs like a call spread $[K_1, K_2]$ with payoff $H \in [0, K_2 - K_1]$ lie in $L^\infty$. The dual $(L^1)^* = L^\infty$ identifies pricing functionals (positive linear functionals on $L^1$ that send $\mathbf 1 \mapsto 1$) with risk-neutral probability measures: $\Lambda(H) = \mathbb E^{\mathbb Q} H$.

### Application 6. $L^p$-convergence of Monte Carlo estimators

For $X \in L^p(\mathbb P)$, $p \ge 2$, the MC estimator $\widehat \mu_N = \frac{1}{N}\sum X_i$ satisfies
$$\|\widehat \mu_N - \mathbb E X\|_p = O(N^{-1/2})$$
by the Marcinkiewicz-Zygmund inequalities (a generalization of CLT-scale bounds to $L^p$). Control of variance ($p = 2$) gives the familiar $1/\sqrt N$; control of higher moments gives concentration (sub-Gaussian, sub-exponential) under stronger assumptions.

### Application 7. Weak convergence and martingale limits

The martingale convergence theorem (Module 2.6) produces an $L^1$-limit $X_\infty$ of $X_n$; weak $L^1$-convergence is often what one has, even when strong $L^p$ fails. Example: Lévy's upward theorem says $\mathbb E[X | \mathcal F_n] \to \mathbb E[X | \mathcal F_\infty]$ strongly in $L^p$ for $X \in L^p$.

### Application 8. Orthogonal decomposition and Wold representation

A stationary $L^2$ process $X_t$ admits a Wold decomposition $X_t = \sum_{k \ge 0} \psi_k \varepsilon_{t-k} + V_t$ (linear predictable part + deterministic residual). This is a Hilbert-space orthogonal decomposition in $L^2(\mathbb P)$, the cornerstone of time-series econometrics (ARMA, state-space models, Kalman filter).

---

## Topic 1.5.8 — Exercises

### ★ (Foundational)

**1.5.E1.** Show Hölder fails for $p < 1$. *Hint: $f = g = \mathbf 1_{[0,1]}$ on Lebesgue; compare $\int fg$ with the RHS for $p = 1/2$.*

**1.5.E2.** Prove: for $\mu(X) < \infty$ and $f \in L^\infty$, $\|f\|_p \to \|f\|_\infty$ as $p \to \infty$.

**1.5.E3.** Find $p$ such that $x^{-1/2} \in L^p((0, 1), \lambda)$. (Answer: $p < 2$.)

**1.5.E4.** Find $p$ such that $(1 + x^2)^{-1} \in L^p(\mathbb R, \lambda)$. (Answer: $p > 1/2$, so all $p \ge 1$.)

**1.5.E5.** Show $L^p \cap L^q \subset L^r$ for $p \le r \le q$, with $\|f\|_r \le \|f\|_p^\theta \|f\|_q^{1-\theta}$ where $1/r = \theta/p + (1-\theta)/q$.

**1.5.E6.** Verify Minkowski's inequality becomes equality iff $f$ and $g$ are a.e. non-negative multiples of each other (for $p > 1$).

**1.5.E7.** Compute $\|e^{-|x|}\|_p$ for $p \in [1, \infty]$ on $(\mathbb R, \lambda)$.

**1.5.E8.** Find a sequence in $L^1([0, 1])$ that converges to $0$ in $L^1$ but not a.e., using a variant of the typewriter construction.

### ★★ (Core)

**1.5.E9.** Prove: if $f_n \to f$ in $L^p$ and $g_n \to g$ in $L^q$ with $1/p + 1/q = 1/r \le 1$, then $f_n g_n \to fg$ in $L^r$.

**1.5.E10.** (**Generalised Hölder / $n$-factor form.**) For $1/p_1 + \ldots + 1/p_k = 1$ and $f_i \in L^{p_i}$, prove $\int |f_1 \cdots f_k| \le \prod \|f_i\|_{p_i}$.

**1.5.E11.** Prove: if $1 \le p < \infty$ and $f_n \to f$ in $L^p$, then $\|f_n\|_p \to \|f\|_p$. Give a counterexample on the weak convergence side.

**1.5.E12.** For $1 \le p < \infty$, prove $L^p(\mathbb R^n, \lambda)$ is **separable** (has a countable dense subset). *Hint: simple functions with rational coefficients and rectangular supports with rational endpoints.*

**1.5.E13.** Show $L^\infty(\mathbb R, \lambda)$ is **not separable**. *Hint: the uncountable family $\{\mathbf 1_{(-\infty, a)} : a \in \mathbb R\}$ is $2$-separated.*

**1.5.E14.** **Minkowski's integral inequality.** For $f(x, y) \ge 0$ measurable, show
$$\left(\int \left|\int f(x, y) \, d\nu(y)\right|^p \, d\mu(x)\right)^{1/p} \le \int \left(\int |f(x, y)|^p \, d\mu(x)\right)^{1/p} d\nu(y).$$
This is the "continuous Minkowski" — a key tool in Fubini-style arguments in analysis.

**1.5.E15.** Prove the **Clarkson inequality** for $p \ge 2$:
$$\left\|\frac{f+g}{2}\right\|_p^p + \left\|\frac{f-g}{2}\right\|_p^p \le \frac{\|f\|_p^p + \|g\|_p^p}{2}.$$
Deduce uniform convexity of $L^p$ for $p \ge 2$.

**1.5.E16.** Show the embedding $L^p \hookrightarrow L^q$ for $p \ge q$ on $\mu(X) < \infty$ is **continuous** (bounded linear map) but **not compact** in general.

### ★★★ (Challenging / quant-relevant)

**1.5.E17.** **Rellich-Kondrachov compactness.** On a bounded domain $U \subset \mathbb R^n$, prove the Sobolev embedding $H^1(U) \hookrightarrow L^2(U)$ is *compact*. (Uses mollification and Arzelà-Ascoli.) This is essential for the calculus of variations.

**1.5.E18.** Prove $(L^\infty)^* \supsetneq L^1$ in general, by exhibiting a bounded linear functional on $L^\infty([0,1])$ that is not representable as $f \mapsto \int fg$ for any $g \in L^1$. *Hint: Hahn-Banach extension of the functional "$f \mapsto \lim_{x \to 0^+} f(x)$" from the subspace of continuous functions (use an invariant Banach limit).*

**1.5.E19.** **Riesz-Thorin interpolation.** Let $T: L^{p_0} + L^{p_1} \to L^{q_0} + L^{q_1}$ be a linear operator with $\|Tf\|_{q_i} \le M_i \|f\|_{p_i}$ for $i = 0, 1$. For $\theta \in [0, 1]$ and $1/p_\theta = (1-\theta)/p_0 + \theta/p_1$, $1/q_\theta = (1-\theta)/q_0 + \theta/q_1$, prove $\|Tf\|_{q_\theta} \le M_0^{1-\theta} M_1^{\theta} \|f\|_{p_\theta}$. (Uses complex analysis of the three-lines theorem.)

**1.5.E20.** **Quant: Mean-variance frontier.** Given $n$ assets with expected returns $m \in \mathbb R^n$ and covariance $\Sigma \succ 0$, show that the minimum-variance portfolio for a target expected return $\mu^*$ is
$$w^* = \Sigma^{-1} \frac{(\mu^* C - B) \mathbf 1 + (A - \mu^* B) m}{AC - B^2},$$
where $A = m^\top \Sigma^{-1} m$, $B = m^\top \Sigma^{-1} \mathbf 1$, $C = \mathbf 1^\top \Sigma^{-1} \mathbf 1$. Derive the **efficient frontier** as a hyperbola in the $(\sigma, \mu)$ plane.

**1.5.E21.** **Quant: Vasicek model $L^2$-projection.** In the Vasicek short-rate model $dr_t = \kappa(\theta - r_t) dt + \sigma dW_t$, show that $\int_0^T r_s \, ds$ is Gaussian and compute its $L^2$-projection onto $\mathcal F_t = \sigma(W_s : s \le t)$. *Hint: use Itô's formula to write $r_T = $ affine in $r_t$ plus independent noise.*

**1.5.E22.** **Quant: FFT and $L^2$ convergence.** Show the Carr-Madan option price formula converges in $L^2$ (in the damping parameter $\alpha$) to the true price as the numerical FFT grid refines. *Hint: Parseval's theorem applied to the Fourier transform of the damped payoff, plus classical approximation of integrals.*

---

## Module Summary

- $L^p(\mu)$ is the Banach space of (equivalence classes a.e. of) measurable functions with $\int |f|^p \, d\mu < \infty$; $L^\infty$ is the space of essentially bounded functions.
- **Hölder**: $\int |fg| \le \|f\|_p \|g\|_q$ with $1/p + 1/q = 1$; **Minkowski**: $\|f+g\|_p \le \|f\|_p + \|g\|_p$; both proved via Young's inequality and the standard $|f+g|^p \le (|f|+|g|)|f+g|^{p-1}$ trick.
- **Riesz-Fischer**: $L^p$ is complete. Proof uses a fast-Cauchy subsequence and MCT/DCT.
- **Density theorems**: Simple functions, $C_c$, and $C_c^\infty$ (or Schwartz) are dense in $L^p(\mathbb R^n)$ for $1 \le p < \infty$. Not true for $p = \infty$.
- **Dual space**: $(L^p)^* \cong L^q$ isometrically for $1 \le p < \infty$ on $\sigma$-finite spaces; $L^p$ is reflexive for $1 < p < \infty$; $L^1$ and $L^\infty$ have more exotic duals.
- **$L^2$ is a Hilbert space**; conditional expectation is an orthogonal projection; the basis of mean-variance optimization and martingale theory.
- **Translation continuity** holds in $L^p$ for $1 \le p < \infty$ but not in $L^\infty$.

**Forward pointers:**
- Module 1.6 (Radon-Nikodym) justifies the "converse" of absolute continuity used to establish $(L^p)^* = L^q$.
- Subject 2 (Probability) uses $L^1, L^2$ relentlessly: expectations, $L^2$-martingale convergence, Hilbert-space projections as conditional expectations, characteristic functions as Fourier transforms of distributions.
- Subject 3 (Stochastic Calculus) builds Itô integrals as $L^2(\mathbb P \times [0, T])$-valued stochastic processes, with the Itô isometry as an instance of $L^2$ orthogonality.
- Subject 4 (Functional Analysis) extends $L^p$ ideas to abstract Banach/Hilbert spaces, spectral theory, compact/Fredholm operators, and the unbounded operator theory used in PDEs.

**Recommended reading:**
- Folland, *Real Analysis*, Chapter 6.
- Rudin, *Real and Complex Analysis*, Chapter 3.
- Lieb & Loss, *Analysis*, Chapters 1-2 (Gourmet treatment of inequalities).
- Reed & Simon, *Methods of Mathematical Physics*, Vol. 1 (operator-theoretic perspective).

**Next module:** Signed measures, Hahn-Jordan decomposition, Radon-Nikodym theorem, Lebesgue decomposition — completing the edifice of measure theory before we enter abstract probability in Subject 2.

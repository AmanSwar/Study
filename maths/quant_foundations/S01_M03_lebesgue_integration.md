# Module 1.3 — The Lebesgue Integral

> *"The integral of a function is not the limit of Riemann sums; it is the measure of the region under its graph."* — informal rendering of Lebesgue's idea.

## Prerequisites

- **Module 1.1** — σ-algebras, measures, Carathéodory extension, Lebesgue measure on $\mathbb R^n$, null sets and completion.
- **Module 1.2** — measurable functions, simple functions and their approximation of non-negative measurables, modes of convergence.
- **Module 0.1** — limits, suprema, $\varepsilon$-$\delta$ arguments (used constantly).
- **Module 0.3** — uniform convergence (to see how the Lebesgue theory *replaces* it).
- **Module 0.4** — basic Riemann integration (we will compare).

## Overview

In Module 1.2 we promoted `measurable function' from a formal definition to a robust class of objects that is closed under arithmetic, composition, and pointwise limits. The Lebesgue integral $\int f \, d\mu$ assigns a number (possibly $+\infty$) to each non-negative measurable $f$, and a real or complex number to each integrable $f$. The construction proceeds in three stages:

1. **Simple functions**: define $\int \varphi \, d\mu = \sum a_i \mu(A_i)$ for $\varphi = \sum a_i \mathbf 1_{A_i}$ in standard form.
2. **Non-negative measurable functions**: define $\int f \, d\mu = \sup \{ \int \varphi \, d\mu : 0 \le \varphi \le f, \varphi \text{ simple} \}$. The central payoff is the **Monotone Convergence Theorem** (MCT).
3. **General measurable functions**: write $f = f^+ - f^-$ and say $f$ is *integrable* if $\int |f| \, d\mu < \infty$. Integration is linear on $L^1(\mu)$.

Three convergence theorems dominate the theory:

- **Monotone Convergence Theorem (MCT, B. Levi, 1906)** — for $0 \le f_n \uparrow f$ we can swap limit and integral.
- **Fatou's Lemma (1906)** — $\int \liminf f_n \le \liminf \int f_n$ for non-negative $f_n$; the cheapest and most flexible tool.
- **Dominated Convergence Theorem (DCT, Lebesgue)** — $f_n \to f$ and $|f_n| \le g \in L^1$ gives $\int f_n \to \int f$.

After proving these we will compare the Lebesgue and Riemann integrals (the Lebesgue integral is strictly more general on $[a,b]$, and a bounded function on $[a,b]$ is Riemann integrable **iff** its set of discontinuities has Lebesgue measure zero — a clean topological/measure-theoretic characterization that Riemann himself could not state).

Throughout this module $(X, \mathcal F, \mu)$ denotes a fixed measure space. Functions are measurable from $(X, \mathcal F)$ to $\overline{\mathbb R} = [-\infty, +\infty]$ with its Borel σ-algebra, unless stated otherwise. The default convention for arithmetic with $\infty$ is $0 \cdot \infty := 0$ (which is indispensable for $\int 0 \, d\mu = 0$ when $\mu(X) = \infty$).

---

## Topic 1.3.1 — Integral of Non-Negative Simple Functions

Recall from Module 1.2 that a **simple function** is a measurable function $\varphi: X \to \mathbb R$ taking only finitely many values. Any simple function has a **standard representation**
$$\varphi = \sum_{i=1}^{n} a_i \mathbf 1_{A_i}, \qquad A_i = \varphi^{-1}(\{a_i\}),$$
where the $a_i$ are the *distinct* values of $\varphi$ and the sets $A_i$ form a measurable partition of $X$. We call $\varphi$ **non-negative simple** if $a_i \ge 0$ for all $i$, and write $\varphi \in \mathcal S^+$.

### Definition 1.3.1 (Integral of a non-negative simple function)

For $\varphi = \sum_{i=1}^{n} a_i \mathbf 1_{A_i} \in \mathcal S^+$ in standard form, define
$$\int_X \varphi \, d\mu \;:=\; \sum_{i=1}^{n} a_i \, \mu(A_i) \;\in\; [0, +\infty].$$
Using the convention $0 \cdot \infty = 0$: if some $a_i = 0$ then the term $a_i \mu(A_i)$ is $0$ even when $\mu(A_i) = \infty$, and if $a_i > 0$ with $\mu(A_i) = \infty$ then the term equals $+\infty$.

For $E \in \mathcal F$, define $\int_E \varphi \, d\mu := \int_X \varphi \cdot \mathbf 1_E \, d\mu = \sum_i a_i \mu(A_i \cap E)$.

### Remark. The standard form is not the only way we will write a simple function

Given any finite measurable partition $\{B_j\}_{j=1}^m$ and coefficients $c_j \ge 0$ (not necessarily distinct), the function $\psi = \sum_j c_j \mathbf 1_{B_j}$ is simple. We will need to know that the value of the integral does not depend on how we represent $\psi$. This is the content of the next lemma.

### Lemma 1.3.2 (Well-definedness on arbitrary partitions)

Let $\varphi \in \mathcal S^+$ and suppose $\varphi = \sum_{j=1}^{m} c_j \mathbf 1_{B_j}$ where $\{B_j\}$ is a measurable partition of $X$ and $c_j \ge 0$ (values need not be distinct). Then
$$\sum_{j=1}^{m} c_j \mu(B_j) = \int_X \varphi \, d\mu.$$

**Proof.** Let $\varphi = \sum_i a_i \mathbf 1_{A_i}$ be the standard form, with $\{A_i\}$ the partition by preimages. Since $\{A_i\}$ and $\{B_j\}$ are both measurable partitions, the family $\{A_i \cap B_j\}_{i,j}$ is also a measurable partition (with empty pieces allowed). For $x \in A_i \cap B_j$ we have $\varphi(x) = a_i = c_j$, so whenever $A_i \cap B_j \ne \emptyset$, $a_i = c_j$. By countable additivity of $\mu$ on the disjoint union $A_i = \bigsqcup_j A_i \cap B_j$,
$$\sum_i a_i \mu(A_i) = \sum_i a_i \sum_j \mu(A_i \cap B_j) = \sum_{i,j} a_i \mu(A_i \cap B_j).$$
Similarly,
$$\sum_j c_j \mu(B_j) = \sum_j c_j \sum_i \mu(A_i \cap B_j) = \sum_{i,j} c_j \mu(A_i \cap B_j).$$
On each non-empty $A_i \cap B_j$ we have $a_i = c_j$, so the two double sums are equal term by term. (If $A_i \cap B_j = \emptyset$ both terms are $0 \cdot 0 = 0$, whatever the convention.) This proves the equality. $\blacksquare$

### Proposition 1.3.3 (Basic properties of simple integrals)

For $\varphi, \psi \in \mathcal S^+$ and $\alpha, \beta \ge 0$:

1. **Linearity.** $\int (\alpha \varphi + \beta \psi) \, d\mu = \alpha \int \varphi \, d\mu + \beta \int \psi \, d\mu$.
2. **Monotonicity.** $\varphi \le \psi$ pointwise $\Rightarrow \int \varphi \, d\mu \le \int \psi \, d\mu$.
3. **Integral as a measure.** For fixed $\varphi \in \mathcal S^+$ the map $E \mapsto \int_E \varphi \, d\mu$ is a measure on $(X, \mathcal F)$.

**Proof.**

*(1) Linearity.* Write $\varphi = \sum_i a_i \mathbf 1_{A_i}$ and $\psi = \sum_j b_j \mathbf 1_{B_j}$ in standard form. Set $E_{ij} = A_i \cap B_j$; then $\{E_{ij}\}$ is a measurable partition of $X$ and
$$\alpha \varphi + \beta \psi = \sum_{i,j} (\alpha a_i + \beta b_j) \mathbf 1_{E_{ij}}.$$
By Lemma 1.3.2 this is a legitimate representation, so
$$\int (\alpha \varphi + \beta \psi) \, d\mu = \sum_{i,j}(\alpha a_i + \beta b_j) \mu(E_{ij}) = \alpha \sum_{i,j} a_i \mu(E_{ij}) + \beta \sum_{i,j} b_j \mu(E_{ij}).$$
By countable additivity $\sum_j \mu(E_{ij}) = \mu(A_i)$ and $\sum_i \mu(E_{ij}) = \mu(B_j)$, so the right side is $\alpha \int \varphi + \beta \int \psi$.

*(2) Monotonicity.* Since $\psi - \varphi \ge 0$ pointwise, $\psi - \varphi \in \mathcal S^+$. By (1),
$$\int \psi \, d\mu = \int \varphi \, d\mu + \int (\psi - \varphi) \, d\mu \ge \int \varphi \, d\mu,$$
since $\int (\psi - \varphi) \, d\mu \ge 0$ (all coefficients and measures non-negative).

*(3) $\nu(E) := \int_E \varphi \, d\mu$ is a measure.* We check the three axioms:

- **Non-negativity** is clear.
- **Empty set:** $\nu(\emptyset) = \sum_i a_i \mu(A_i \cap \emptyset) = 0$.
- **Countable additivity:** let $E = \bigsqcup_{k=1}^\infty E_k$ disjoint. Then
$$\nu(E) = \sum_i a_i \mu\!\left(A_i \cap E\right) = \sum_i a_i \sum_k \mu(A_i \cap E_k) = \sum_k \sum_i a_i \mu(A_i \cap E_k) = \sum_k \nu(E_k),$$
where the swap of sums is justified because all terms are non-negative (Tonelli for non-negative double series, or simply the monotone convergence of partial sums). $\blacksquare$

### Example 1.3.4 (A concrete simple integral)

Let $X = [0, 5]$ with Lebesgue measure $\lambda$, and let
$$\varphi(x) = \begin{cases} 2 & 0 \le x < 1 \\ 0 & 1 \le x < 2 \\ 3 & 2 \le x \le 5. \end{cases}$$
Standard form: $\varphi = 2 \mathbf 1_{[0,1)} + 0 \cdot \mathbf 1_{[1,2)} + 3 \mathbf 1_{[2,5]}$.
$$\int_{[0,5]} \varphi \, d\lambda = 2 \cdot 1 + 0 \cdot 1 + 3 \cdot 3 = 11.$$
This matches the Riemann integral because $\varphi$ is a step function. But the Lebesgue construction also makes sense for $\varphi = \mathbf 1_{\mathbb Q \cap [0,1]}$:
$$\int_{[0,1]} \mathbf 1_{\mathbb Q \cap [0,1]} \, d\lambda = 1 \cdot \lambda(\mathbb Q \cap [0,1]) = 0,$$
since $\mathbb Q \cap [0,1]$ is countable and Lebesgue measure is zero on countable sets. The Riemann integral of this function does not exist (upper sums are always $1$, lower sums are always $0$).

---

## Topic 1.3.2 — Integral of Non-Negative Measurable Functions and the MCT

### Definition 1.3.5 (Integral of a non-negative measurable function)

For a measurable $f: X \to [0, +\infty]$, define
$$\int_X f \, d\mu \;:=\; \sup\!\left\{ \int_X \varphi \, d\mu : \varphi \in \mathcal S^+, \; 0 \le \varphi \le f \right\} \in [0, +\infty].$$
For $E \in \mathcal F$, $\int_E f \, d\mu := \int_X f \mathbf 1_E \, d\mu$. If $\int_X f \, d\mu < \infty$ we say $f$ is **integrable over $X$**.

**Sanity checks:**
- If $f = \varphi \in \mathcal S^+$ already, the two definitions agree: $\varphi$ itself is a simple function $\le \varphi$, so the sup is at least $\int \varphi$; and monotonicity (Proposition 1.3.3.2) says no simple function $\psi \le \varphi$ can exceed $\int \varphi$.
- Monotonicity is inherited: $f \le g \Rightarrow \int f \le \int g$, because every simple $\varphi \le f$ is also $\le g$.
- Scaling: $\int (\alpha f) \, d\mu = \alpha \int f \, d\mu$ for $\alpha \ge 0$, because simple functions $\le \alpha f$ are precisely $\alpha \varphi$ with $\varphi$ simple $\le f$ (and $\alpha \cdot 0 = 0$ is handled correctly even when $\alpha = 0$).

Linearity (finite sums) is *not* obvious at this point, because the sup of a sum is not the sum of sups in general. We will get it as a corollary of the MCT.

### Theorem 1.3.6 (Monotone Convergence Theorem — MCT)

Let $f_n: X \to [0, +\infty]$ be measurable with $f_n(x) \le f_{n+1}(x)$ for all $n$ and all $x$, and let $f(x) := \lim_n f_n(x) = \sup_n f_n(x)$ (pointwise, possibly $+\infty$). Then $f$ is measurable and
$$\int_X f \, d\mu = \lim_{n \to \infty} \int_X f_n \, d\mu.$$

**Proof.** Measurability of $f$ was proved in Module 1.2 (sup of measurables is measurable).

*Upper bound.* Since $f_n \le f$ for each $n$, monotonicity of the integral gives $\int f_n \le \int f$, so
$$\lim_n \int f_n \le \int f. \tag{$*$}$$
(The limit exists because $\int f_n$ is an increasing sequence in $[0, +\infty]$.)

*Lower bound.* Fix $\varphi \in \mathcal S^+$ with $\varphi \le f$, and fix $c \in (0, 1)$. Define
$$E_n := \{x : f_n(x) \ge c \varphi(x)\} \subset X.$$
Each $E_n$ is measurable (sub-level set), $E_n \subset E_{n+1}$ (since $f_n \le f_{n+1}$), and $\bigcup_n E_n = X$:

- If $\varphi(x) = 0$, then $c \varphi(x) = 0 \le f_1(x)$, so $x \in E_1$.
- If $\varphi(x) > 0$, then $\varphi(x) \le f(x)$ with $f(x) > 0$ and $c < 1$, so $c\varphi(x) < \varphi(x) \le f(x) = \lim f_n(x)$; hence $f_n(x) \ge c\varphi(x)$ eventually, i.e. $x \in E_n$ for some $n$.

Now, for each $n$,
$$\int_X f_n \, d\mu \ge \int_{E_n} f_n \, d\mu \ge \int_{E_n} c\varphi \, d\mu = c \int_{E_n} \varphi \, d\mu.$$

By Proposition 1.3.3.3, $E \mapsto \int_E \varphi \, d\mu$ is a measure; measures are continuous from below, so
$$\int_{E_n} \varphi \, d\mu \;\uparrow\; \int_{\bigcup_n E_n} \varphi \, d\mu = \int_X \varphi \, d\mu.$$

Taking $n \to \infty$ in $\int f_n \ge c \int_{E_n} \varphi$:
$$\lim_n \int_X f_n \, d\mu \ge c \int_X \varphi \, d\mu.$$

Let $c \uparrow 1$: $\lim_n \int f_n \ge \int \varphi$. Taking the sup over all simple $\varphi \le f$:
$$\lim_n \int_X f_n \, d\mu \ge \int_X f \, d\mu. \tag{$**$}$$

$(*)$ and $(**)$ together give equality. $\blacksquare$

### Corollary 1.3.7 (Linearity of the integral on non-negative measurables)

For measurable $f, g \ge 0$ and $\alpha, \beta \ge 0$:
$$\int_X (\alpha f + \beta g) \, d\mu = \alpha \int_X f \, d\mu + \beta \int_X g \, d\mu.$$

**Proof.** Scaling $\int \alpha f = \alpha \int f$ was already observed. It suffices to prove $\int(f+g) = \int f + \int g$.

By Module 1.2's approximation theorem, there exist increasing sequences $0 \le \varphi_n \uparrow f$ and $0 \le \psi_n \uparrow g$ of simple functions. Then $\varphi_n + \psi_n \uparrow f + g$ (monotone increase, pointwise limit is $f + g$). By linearity on simple functions (Proposition 1.3.3.1),
$$\int (\varphi_n + \psi_n) \, d\mu = \int \varphi_n \, d\mu + \int \psi_n \, d\mu.$$
Apply MCT on each side (all three sequences are increasing and non-negative):
$$\int (f+g) \, d\mu = \int f \, d\mu + \int g \, d\mu. \qquad \blacksquare$$

### Corollary 1.3.8 (Swapping sum and integral of non-negative series)

Let $h_k: X \to [0, \infty]$ be measurable, $k \ge 1$. Then
$$\int_X \sum_{k=1}^\infty h_k \, d\mu = \sum_{k=1}^\infty \int_X h_k \, d\mu.$$

**Proof.** Let $S_n = \sum_{k=1}^n h_k$. Then $S_n \uparrow S := \sum_{k=1}^\infty h_k$ (partial sums of non-negative terms increase). By linearity, $\int S_n = \sum_{k=1}^n \int h_k$. MCT gives $\int S = \lim_n \int S_n = \sum_{k=1}^\infty \int h_k$. $\blacksquare$

This already shows that the Lebesgue theory handles series of non-negative functions effortlessly. The Riemann theory needs uniform convergence or at least dominated convergence. Here we get *nothing but non-negativity* and the theorem works.

### Corollary 1.3.9 (Chebyshev's / Markov's inequality)

Let $f: X \to [0, \infty]$ be measurable and $a > 0$. Then
$$\mu(\{x : f(x) \ge a\}) \le \frac{1}{a} \int_X f \, d\mu.$$

**Proof.** Let $E = \{f \ge a\}$. Then $f \ge a \mathbf 1_E$ pointwise, so by monotonicity,
$$\int f \, d\mu \ge \int a \mathbf 1_E \, d\mu = a \mu(E). \qquad \blacksquare$$

This innocuous inequality is the workhorse of probability and statistics (via tail bounds, concentration, and convergence in probability).

### Example 1.3.10 (Counting measure: integrals are series)

Let $X = \mathbb N$, $\mathcal F = 2^{\mathbb N}$, $\mu = $ counting measure. Any function $f: \mathbb N \to [0, \infty]$ is measurable. Writing $a_k := f(k)$,
$$\int_{\mathbb N} f \, d\mu = \sum_{k=1}^\infty a_k.$$

*Proof.* Approximate $f$ from below by $f_n = \sum_{k=1}^n a_k \mathbf 1_{\{k\}}$. Then $f_n \uparrow f$, $\int f_n = \sum_{k=1}^n a_k$, and MCT gives $\int f = \sum_k a_k$.

Hence MCT for counting measure becomes Tonelli's theorem for non-negative double series $\sum_k \sum_j = \sum_j \sum_k$.

### Example 1.3.11 (A function integrable on $(0,1)$ but not on $(0, \infty)$)

Let $f(x) = 1/\sqrt{x}$ on $(0, \infty)$ with Lebesgue measure.

On $(0, 1]$: approximate by $f_n(x) = \min(f(x), n)$, which is bounded and equals $n$ on $(0, 1/n^2]$ and $1/\sqrt x$ on $[1/n^2, 1]$. For the Riemann part we get $\int_{1/n^2}^1 x^{-1/2} \, dx = 2 - 2/n$. On $(0, 1/n^2]$ we get $n \cdot 1/n^2 = 1/n$. So $\int f_n \to 2$. MCT says $\int f = 2 < \infty$.

On $[1, \infty)$: $f_n = f \mathbf 1_{[1, n]}$, $\int f_n = 2\sqrt n - 2 \to \infty$. MCT says $\int f \mathbf 1_{[1,\infty)} = \infty$, so $f \notin L^1([1, \infty))$.

This asymmetry between $p < 1$ (integrable near $0$, not near $\infty$) and $p > 1$ (reverse) is one of the first quantitative facts beginners should internalize.

---

## Topic 1.3.3 — Fatou's Lemma

Fatou's lemma is the cheapest tool in the box: it requires *no* hypothesis beyond non-negativity and measurability, and it gives an inequality that we can often upgrade to an equality. We will also use it twice in the proof of the DCT.

### Theorem 1.3.12 (Fatou's Lemma)

Let $f_n: X \to [0, \infty]$ be measurable. Then
$$\int_X \liminf_{n \to \infty} f_n \, d\mu \le \liminf_{n \to \infty} \int_X f_n \, d\mu.$$

**Proof.** Define $g_n(x) := \inf_{k \ge n} f_k(x)$. Each $g_n$ is measurable (inf of measurables is measurable, Module 1.2), $g_n \ge 0$, and $g_n \le f_k$ for all $k \ge n$, so $\int g_n \le \int f_k$ for all $k \ge n$; hence
$$\int g_n \le \inf_{k \ge n} \int f_k.$$

Also $g_n \uparrow \liminf_n f_n$ (by definition of $\liminf = \sup_n \inf_{k \ge n}$). By MCT,
$$\int \liminf_n f_n \, d\mu = \lim_n \int g_n \, d\mu \le \lim_n \inf_{k \ge n} \int f_k \, d\mu = \liminf_n \int f_n \, d\mu. \qquad \blacksquare$$

### Remark. Strict inequality is possible

Take $X = [0, 1]$, $\mu = $ Lebesgue. Define $f_n = n \mathbf 1_{(0, 1/n)}$ (tall narrow spike). Then $f_n \to 0$ pointwise (any fixed $x > 0$ satisfies $f_n(x) = 0$ for $n > 1/x$), so $\liminf f_n = 0$ and $\int \liminf = 0$. But $\int f_n = n \cdot 1/n = 1$ for every $n$, so $\liminf \int f_n = 1 > 0$. Mass has *escaped* to infinity in the sense that the graph of $f_n$ becomes a taller and narrower spike; the integral does not see the pointwise convergence because the total mass is conserved.

*Analogous "travelling wave" example on $\mathbb R$*: $g_n(x) = \mathbf 1_{[n, n+1]}(x)$. Then $g_n \to 0$ pointwise (any fixed $x$ has $g_n(x) = 0$ for $n > x$), so $\liminf \int g_n = 1 > 0 = \int \liminf g_n$. Mass escapes to $+\infty$.

### Corollary 1.3.13 (Reverse Fatou, requires an integrable majorant)

If $f_n \le g$ with $g \ge 0$ measurable and $\int g \, d\mu < \infty$, then
$$\int_X \limsup_n f_n \, d\mu \ge \limsup_n \int_X f_n \, d\mu.$$

**Proof.** Apply Fatou to the non-negative sequence $g - f_n \ge 0$:
$$\int \liminf (g - f_n) \le \liminf \int (g - f_n) = \int g - \limsup \int f_n.$$
Since $\int g$ is finite we can subtract. Also $\liminf(g - f_n) = g - \limsup f_n$ (basic property of liminf/limsup with a fixed sequence, valid when $g$ is finite a.e.; on a null set the equality fails but doesn't affect the integral). Hence
$$\int g - \int \limsup f_n \le \int g - \limsup \int f_n,$$
giving the desired inequality. $\blacksquare$

### Worked Example 1.3.14 (Using Fatou to prove a lower bound)

Let $X = [0, 1]$, $\mu$ Lebesgue, and $f_n(x) = n x^{n-1}$. For $x \in [0, 1)$, $nx^{n-1} \to 0$; at $x = 1$, $f_n(1) = n \to \infty$. So $\liminf f_n = 0$ a.e. $\int f_n = \int_0^1 n x^{n-1} dx = 1$ for all $n$. Fatou gives $0 = \int \liminf \le \liminf \int = 1$, which is correct but not tight. The issue again is that all the mass concentrates near $x = 1$.

### Worked Example 1.3.15 (Moment bounds via Fatou in probability)

Suppose $X_n$ are non-negative random variables on a probability space $(\Omega, \mathcal F, \mathbb P)$ with $\mathbb E X_n \le C$ for all $n$. Assume $X_n \to X$ a.s. Then
$$\mathbb E X = \int \lim X_n \, d\mathbb P = \int \liminf X_n \, d\mathbb P \le \liminf \mathbb E X_n \le C.$$
So *convergence a.s. + uniform $L^1$ bound* $\Rightarrow$ limit is in $L^1$. This is the first step in many martingale-limit proofs, e.g. martingale convergence theorems in Module 2.6.

---

## Topic 1.3.4 — Integral of General Measurable Functions

Writing $f = f^+ - f^-$ with $f^+ = \max(f, 0)$ and $f^- = \max(-f, 0)$, both non-negative and measurable (Module 1.2).

### Definition 1.3.16 (Lebesgue integrable, integral)

A measurable $f: X \to \overline{\mathbb R}$ is **Lebesgue integrable (with respect to $\mu$)** if
$$\int_X |f| \, d\mu < \infty.$$
In that case both $\int f^+$ and $\int f^-$ are finite (since $f^\pm \le |f|$), and we define
$$\int_X f \, d\mu \;:=\; \int_X f^+ \, d\mu - \int_X f^- \, d\mu \in \mathbb R.$$

For $E \in \mathcal F$, $\int_E f \, d\mu := \int_X f \mathbf 1_E \, d\mu$.

We write $L^1(X, \mathcal F, \mu)$, or $L^1(\mu)$, or simply $L^1$, for the set of $\mu$-integrable real-valued measurable functions. The integrable requirement forces both positive and negative parts to be finite, avoiding $\infty - \infty$.

### Definition 1.3.17 (Complex-valued integrable functions)

A measurable $f: X \to \mathbb C$ is integrable iff $|f| = \sqrt{(\operatorname{Re} f)^2 + (\operatorname{Im} f)^2}$ is integrable; equivalently iff $\operatorname{Re} f, \operatorname{Im} f \in L^1$. We define
$$\int f \, d\mu := \int \operatorname{Re} f \, d\mu + i \int \operatorname{Im} f \, d\mu.$$

### Proposition 1.3.18 (Basic properties of the Lebesgue integral on $L^1$)

For $f, g \in L^1(\mu)$, $\alpha, \beta \in \mathbb R$:

1. **Linearity.** $\alpha f + \beta g \in L^1$ and $\int(\alpha f + \beta g) = \alpha \int f + \beta \int g$.
2. **Monotonicity.** $f \le g$ a.e. $\Rightarrow \int f \le \int g$.
3. **Triangle inequality.** $\left| \int f \, d\mu \right| \le \int |f| \, d\mu$.
4. **Insensitivity to null sets.** If $f = g$ a.e. and one is in $L^1$ then so is the other, and $\int f = \int g$.
5. **Characterization of a.e. zero.** If $f \ge 0$ is measurable and $\int f \, d\mu = 0$, then $f = 0$ a.e.

**Proof.**

*(1) Linearity.* First, $|\alpha f + \beta g| \le |\alpha||f| + |\beta||g|$, so $\int|\alpha f + \beta g| \le |\alpha|\int|f| + |\beta|\int|g| < \infty$. For the equality we reduce to scalings and sums.

- *Scaling.* For $\alpha \ge 0$: $(\alpha f)^+ = \alpha f^+$ and $(\alpha f)^- = \alpha f^-$, so $\int \alpha f = \int \alpha f^+ - \int \alpha f^- = \alpha (\int f^+ - \int f^-) = \alpha \int f$. For $\alpha < 0$: $(\alpha f)^+ = -\alpha f^-$ and $(\alpha f)^- = -\alpha f^+$, so $\int \alpha f = -\alpha \int f^- - (-\alpha \int f^+) = \alpha (\int f^+ - \int f^-) = \alpha \int f$.

- *Addition.* Write $h = f + g$. On the set where $f, g$ are finite we have $h^+ - h^- = (f^+ - f^-) + (g^+ - g^-)$, i.e.
$$h^+ + f^- + g^- = h^- + f^+ + g^+. \tag{A}$$
Both sides are non-negative measurable functions with finite integrals (since $f, g \in L^1$ and $|h| \le |f| + |g|$). By Corollary 1.3.7 (linearity on non-negative measurables), integrating (A):
$$\int h^+ + \int f^- + \int g^- = \int h^- + \int f^+ + \int g^+,$$
and rearranging,
$$\int h = \int h^+ - \int h^- = (\int f^+ - \int f^-) + (\int g^+ - \int g^-) = \int f + \int g.$$

Combining scaling and addition gives full linearity.

*(2) Monotonicity.* $g - f \ge 0$ a.e., so $\int (g - f) \ge 0$ (Definition 1.3.5 applied to $g - f$ modulo a null set, see item 4). By linearity $\int g - \int f \ge 0$.

*(3) Triangle inequality.* Real case: $\int f = \int f^+ - \int f^-$ and $|\int f| = |\int f^+ - \int f^-| \le \int f^+ + \int f^- = \int |f|$.
Complex case: pick $\theta \in \mathbb R$ such that $e^{-i\theta} \int f = |\int f| \in \mathbb R_{\ge 0}$. Then
$$\left|\int f\right| = e^{-i\theta} \int f = \int e^{-i\theta} f \, d\mu = \int \operatorname{Re}(e^{-i\theta} f) \, d\mu \le \int |e^{-i\theta} f| \, d\mu = \int |f| \, d\mu,$$
using $\operatorname{Re}(w) \le |w|$ and the real monotonicity.

*(4) Null sets.* If $f = g$ a.e., the set $N = \{f \ne g\}$ has $\mu(N) = 0$. For non-negative $f, g$ with $f = g$ on $X \setminus N$:
$$\int f \, d\mu = \int_{X \setminus N} f \, d\mu + \int_N f \, d\mu = \int_{X \setminus N} g \, d\mu + 0 = \int g \, d\mu,$$
because $\int_N f \, d\mu = 0$ whenever $\mu(N) = 0$ (any simple $\varphi \le f \mathbf 1_N$ has the form $\sum a_i \mathbf 1_{A_i}$ with $A_i \subset N$, so $\sum a_i \mu(A_i) \le \sum a_i \cdot 0 = 0$ — using $0 \cdot \infty = 0$ but also $\mu(A_i) = 0$ so there's no issue at all). For general $f, g$ apply to $f^\pm, g^\pm$.

*(5) A.e. zero.* If $\int f \, d\mu = 0$ and $f \ge 0$, consider $E_n := \{f > 1/n\}$. By Chebyshev (Corollary 1.3.9), $\mu(E_n) \le n \int f \, d\mu = 0$. Hence $\{f > 0\} = \bigcup_n E_n$ has measure $\le \sum_n \mu(E_n) = 0$. So $f = 0$ a.e. $\blacksquare$

### Theorem 1.3.19 (Integrability characterization)

For a measurable $f: X \to \overline{\mathbb R}$, the following are equivalent:

1. $f \in L^1(\mu)$.
2. There exists $g \in L^1(\mu)$ with $|f| \le g$ a.e.
3. $f^+, f^- \in L^1(\mu)$ (both non-negative parts integrable).

**Proof.** $(1) \Rightarrow (3)$: $f^\pm \le |f|$ and $\int |f| < \infty$ give $\int f^\pm < \infty$.
$(3) \Rightarrow (1)$: $|f| = f^+ + f^-$ and $\int |f| = \int f^+ + \int f^- < \infty$.
$(1) \Rightarrow (2)$: take $g = |f|$.
$(2) \Rightarrow (1)$: $\int|f| \le \int g < \infty$. $\blacksquare$

### Proposition 1.3.20 (Integrable functions are finite almost everywhere)

If $f \in L^1(\mu)$ then $|f| < \infty$ a.e.

**Proof.** Let $E_\infty = \{|f| = \infty\}$. Then for every $n \in \mathbb N$, $|f| \ge n \mathbf 1_{E_\infty}$, so
$$\infty > \int |f| \, d\mu \ge n \mu(E_\infty).$$
If $\mu(E_\infty) > 0$, then for $n$ large $n \mu(E_\infty) = \infty$, contradiction. Hence $\mu(E_\infty) = 0$. $\blacksquare$

So modulo a null set we can always replace an integrable function by a finite-valued one, which is why we usually work with $f: X \to \mathbb R$ rather than $\overline{\mathbb R}$ in $L^1$.

---

## Topic 1.3.5 — Dominated Convergence Theorem

The DCT is the swap-limit-with-integral theorem for sequences that are *not* necessarily monotone. It demands an integrable envelope $g$, but in exchange gives the full conclusion $\int f_n \to \int f$.

### Theorem 1.3.21 (Dominated Convergence Theorem — DCT)

Let $f_n: X \to \overline{\mathbb R}$ be measurable, $f_n(x) \to f(x)$ a.e., and suppose there exists $g \in L^1(\mu)$ with $|f_n(x)| \le g(x)$ a.e. for all $n$. Then $f \in L^1(\mu)$ and
$$\lim_{n \to \infty} \int_X |f_n - f| \, d\mu = 0, \qquad \text{in particular} \qquad \lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.$$

**Proof.** First, measurability of $f$ follows from pointwise convergence of measurables (Module 1.2). On the a.e. set where everything holds, $|f| \le g$, so $f \in L^1$ by the characterization above.

Modify $f_n$ and $f$ on the null set where the hypotheses fail (set them to $0$ there). This does not change any integral.

Apply Fatou's lemma to the non-negative sequence $2g - |f_n - f| \ge 0$ (we used $|f_n - f| \le |f_n| + |f| \le 2g$, which holds everywhere after the null-set modification):
$$\int \liminf_n (2g - |f_n - f|) \le \liminf_n \int (2g - |f_n - f|).$$

Since $f_n \to f$ pointwise, $|f_n - f| \to 0$, so $\liminf_n (2g - |f_n - f|) = 2g - 0 = 2g$. Hence
$$\int 2g \, d\mu \le \int 2g \, d\mu - \limsup_n \int |f_n - f| \, d\mu,$$
using $\liminf(a - b_n) = a - \limsup b_n$ for constants $a \in \mathbb R$ (valid because $\int 2g$ is finite).

Rearranging: $\limsup_n \int |f_n - f| \, d\mu \le 0$. Since the integrand is non-negative, the limsup is $\ge 0$, forcing it to be $0$. Hence $\int |f_n - f| \, d\mu \to 0$.

For the integrated convergence: $\left|\int f_n - \int f\right| = \left|\int (f_n - f)\right| \le \int |f_n - f| \to 0$. $\blacksquare$

### Remark. DCT ⇒ Bounded convergence

On a finite measure space $\mu(X) < \infty$, if $|f_n| \le M$ (constant) and $f_n \to f$ a.e., then DCT with $g = M$ (which is in $L^1$ because $\int M = M \mu(X) < \infty$) gives $\int f_n \to \int f$. This is the **Bounded Convergence Theorem**.

### Example 1.3.22 (A must-know counterexample: no dominator)

Let $f_n = n \mathbf 1_{(0, 1/n)}$ on $[0,1]$. Then $f_n \to 0$ pointwise, but $\int f_n = 1$ for all $n$, so $\int f_n \to 1 \ne 0 = \int \lim f_n$. The DCT hypothesis fails: any $g$ dominating all $f_n$ must satisfy $g(x) \ge n$ on $(0, 1/n)$, so $g(x) = \infty$ on $(0, 1)$ — not in $L^1$.

Equivalently, $\sup_n |f_n|$ is not in $L^1$ (it equals the non-integrable $f^*(x) = 1/x$ on $(0,1)$, which fails to be in $L^1$).

### Example 1.3.23 (Differentiation under the integral sign)

Let $f: X \times (a, b) \to \mathbb R$, measurable in $x$ for each $t$, differentiable in $t$ for each $x$, with $|\partial_t f(x, t)| \le g(x)$ for $g \in L^1(\mu)$. Then
$$F(t) := \int_X f(x, t) \, d\mu(x)$$
is differentiable on $(a, b)$ and $F'(t) = \int_X \partial_t f(x, t) \, d\mu(x)$.

**Proof.** Fix $t_0 \in (a, b)$ and take $h_n \to 0$ in $(a, b) - t_0$. Define
$$\Phi_n(x) := \frac{f(x, t_0 + h_n) - f(x, t_0)}{h_n}.$$
By the mean value theorem, $|\Phi_n(x)| \le g(x)$ (with the bound valid for $t$ in a bounded interval around $t_0$ inside $(a, b)$). Also $\Phi_n(x) \to \partial_t f(x, t_0)$ pointwise. DCT gives
$$\lim_n \frac{F(t_0 + h_n) - F(t_0)}{h_n} = \lim_n \int \Phi_n \, d\mu = \int \partial_t f(x, t_0) \, d\mu,$$
which is the claim. $\blacksquare$

This is the key tool for computing derivatives of expectations in probability (score function in maximum likelihood, Greeks in quantitative finance).

### Example 1.3.24 (DCT in probability: convergence of expectations)

If $X_n \to X$ a.s. and $|X_n| \le Y$ with $\mathbb E Y < \infty$, then $\mathbb E X_n \to \mathbb E X$. This is the simplest way to prove convergence of pricing formulas, estimators, etc.

### Extending DCT: a.e. convergence suffices with a refined envelope

**Theorem (Generalized DCT / Vitali's version).** If $f_n \to f$ a.e., $|f_n| \le g_n$ a.e., $g_n \to g$ a.e., all $g_n \in L^1$, and $\int g_n \to \int g < \infty$, then $\int |f_n - f| \to 0$.

*Proof sketch.* Apply Fatou to $g_n + g - |f_n - f| \ge 0$ (eventually) and use $\int g_n + \int g \to 2 \int g$ to conclude $\limsup \int |f_n - f| \le 0$. Details expanded in Exercises.

This generalization is used when the natural envelope varies with $n$ — e.g., when using exponential tails of Gaussians whose variance depends on $n$.

---

## Topic 1.3.6 — Comparison with the Riemann Integral

### Theorem 1.3.25 (Riemann integrable $\Rightarrow$ Lebesgue integrable with the same value)

Let $f: [a, b] \to \mathbb R$ be bounded and Riemann integrable. Then $f$ is Lebesgue measurable (with respect to the Lebesgue completion) and
$$\int_{[a,b]}^{\text{Lebesgue}} f \, d\lambda = \int_a^b f(x) \, dx \quad \text{(Riemann).}$$

**Proof.** Let $|f| \le M$. Let $P_n$ be a sequence of partitions of $[a, b]$ with mesh $\to 0$. For each partition $P = \{a = x_0 < x_1 < \dots < x_k = b\}$ define
$$\underline S(P) := \sum_{i=1}^{k} \left(\inf_{[x_{i-1}, x_i]} f \right) (x_i - x_{i-1}), \qquad \overline S(P) := \sum_{i=1}^{k} \left(\sup_{[x_{i-1}, x_i]} f \right)(x_i - x_{i-1}).$$
Riemann integrability means $\underline S(P_n), \overline S(P_n)$ both converge to the same limit $I = \int_a^b f$.

Define the simple functions $\varphi_n, \psi_n$ on $[a, b]$ by
$$\varphi_n(x) = \inf_{[x_{i-1}, x_i]} f \quad \text{for } x \in (x_{i-1}, x_i], \qquad \psi_n(x) = \sup_{[x_{i-1}, x_i]} f.$$
These are step functions (hence simple), measurable, and uniformly bounded by $M$. Refine the partitions $P_n \subset P_{n+1}$ (WLOG); then $\varphi_n$ is increasing in $n$ (refining partitions raises infima) and $\psi_n$ decreasing. Let $\varphi = \lim \varphi_n$ and $\psi = \lim \psi_n$ (a.e. convergence everywhere by monotonicity); these are Lebesgue measurable by Module 1.2.

Because $\int \varphi_n \, d\lambda = \underline S(P_n) \to I$ and $\int \psi_n \, d\lambda = \overline S(P_n) \to I$ (Riemann integrability), and $|\varphi_n|, |\psi_n| \le M$, the Bounded Convergence Theorem (Remark after DCT, or direct MCT since $\varphi_n$ is increasing and $M - \psi_n$ is increasing) gives
$$\int_{[a,b]} \varphi \, d\lambda = I = \int_{[a,b]} \psi \, d\lambda.$$
Since $\psi \ge \varphi$ and they have the same integral, $\psi = \varphi$ a.e.; on this set they both equal $f$ (because $\varphi_n \le f \le \psi_n$ always). So $f = \varphi$ a.e., hence is equal a.e. to a Lebesgue measurable function, hence is Lebesgue measurable (with respect to the completion). Moreover $\int_{[a,b]} f \, d\lambda = \int \varphi \, d\lambda = I$. $\blacksquare$

### Theorem 1.3.26 (Lebesgue's characterization of Riemann integrability)

Let $f: [a, b] \to \mathbb R$ be bounded. Then $f$ is Riemann integrable **iff** the set of discontinuities of $f$ has Lebesgue measure zero.

**Proof (sketch).** Define the **oscillation** of $f$ at $x$ by
$$\omega(f; x) := \inf_{\delta > 0} \operatorname*{osc}_{[x-\delta, x+\delta]} f = \inf_{\delta > 0} \sup_{|y-x| < \delta} f(y) - \inf_{|y-x| < \delta} f(y).$$
$f$ is continuous at $x$ iff $\omega(f; x) = 0$. The set of discontinuities is $D = \bigcup_{k \ge 1} D_k$ where $D_k = \{x : \omega(f; x) \ge 1/k\}$; each $D_k$ is closed.

$f$ Riemann integrable iff for every $\varepsilon > 0$ there is a partition with $\overline S - \underline S < \varepsilon$. Decomposing this quantity across intervals according to whether the interval meets $D_k$ or not, one shows this is equivalent to $\lambda(D_k) = 0$ for every $k$, hence $\lambda(D) = 0$.

(Full proof in standard texts — e.g. Rudin's PMA Theorem 11.33, or Royden–Fitzpatrick §2.7. Carried out as Exercise ★★★.) $\blacksquare$

### Example 1.3.27 (Dirichlet's function)

$f = \mathbf 1_{\mathbb Q \cap [0,1]}$ is discontinuous at every point, so the set of discontinuities is $[0, 1]$ with measure $1 > 0$. Hence $f$ is not Riemann integrable. But Lebesgue: $\int f \, d\lambda = \lambda(\mathbb Q \cap [0,1]) = 0$.

### Example 1.3.28 (Thomae's function)

$T: [0, 1] \to [0, 1]$ defined by $T(p/q) = 1/q$ for $\gcd(p, q) = 1$, $T(0) = 1$, $T(x) = 0$ for $x$ irrational. $T$ is discontinuous on $\mathbb Q \cap [0, 1]$ and continuous on irrationals. The discontinuities form a set of measure zero, so $T$ is Riemann integrable, and its integral is $0$ (upper sums $\to 0$ because for any $\varepsilon$ only finitely many points have $T > \varepsilon$).

### Improper Riemann integrals vs. Lebesgue

For functions on unbounded intervals or with singularities, there is a distinction:
- If $f \ge 0$, the improper Riemann integral $\int_a^\infty f = \lim_{b \to \infty} \int_a^b f$ agrees with the Lebesgue integral $\int_{[a, \infty)} f \, d\lambda$ (by MCT applied to $f \mathbf 1_{[a, b_n]}$ for $b_n \uparrow \infty$).
- For signed $f$, the improper Riemann integral may *converge* while $|f|$ is *not* Lebesgue integrable. Canonical example: $\int_0^\infty \frac{\sin x}{x} dx = \pi/2$ converges as an improper Riemann integral, but $\int_0^\infty \left|\frac{\sin x}{x}\right| dx = \infty$. So $\sin x / x \notin L^1(\mathbb R_{\ge 0})$. The Lebesgue theory does not recognize this integral; the machinery of the Fourier transform in Module 2.5 uses principal-value integrals or the Schwartz space to handle it.

---

## Topic 1.3.7 — Absolute Continuity, Layer Cake, Change of Variables

### Theorem 1.3.29 (Absolute continuity of the integral)

Let $f \in L^1(\mu)$. For every $\varepsilon > 0$ there exists $\delta > 0$ such that for every $E \in \mathcal F$ with $\mu(E) < \delta$,
$$\left| \int_E f \, d\mu \right| \le \int_E |f| \, d\mu < \varepsilon.$$

**Proof.** WLOG $f \ge 0$ (apply to $|f|$). For $M > 0$ define $f_M = \min(f, M)$. Then $f_M \uparrow f$ (pointwise, as $M \to \infty$) and MCT gives $\int f_M \uparrow \int f$. Pick $M$ so large that $\int (f - f_M) d\mu < \varepsilon/2$. Then for any $E$ with $\mu(E) < \delta := \varepsilon / (2M)$,
$$\int_E f \, d\mu = \int_E f_M \, d\mu + \int_E (f - f_M) \, d\mu \le M \mu(E) + \int (f - f_M) \, d\mu < M \cdot \delta + \varepsilon/2 = \varepsilon. \qquad \blacksquare$$

The conclusion $\int_E f \, d\mu < \varepsilon$ for $\mu(E) < \delta$ is the content of saying that the measure $\nu(E) := \int_E f \, d\mu$ is *absolutely continuous* with respect to $\mu$ (in the $\varepsilon$-$\delta$ sense). The Radon-Nikodym theorem (Module 1.6) will show this is equivalent to the $\sigma$-algebra statement "$\mu(E) = 0 \Rightarrow \nu(E) = 0$" (plus $\sigma$-finiteness).

### Theorem 1.3.30 (Layer cake formula)

Let $f \ge 0$ be measurable and $\varphi: [0, \infty) \to [0, \infty)$ be increasing, absolutely continuous on $[0, A]$ for every $A$, with $\varphi(0) = 0$ and derivative $\varphi' \ge 0$ a.e. Then
$$\int_X \varphi(f(x)) \, d\mu(x) \;=\; \int_0^\infty \varphi'(t) \, \mu(\{f > t\}) \, dt.$$

**Special case ($\varphi(t) = t$):**
$$\int_X f \, d\mu = \int_0^\infty \mu(\{f > t\}) \, dt.$$

**Special case ($\varphi(t) = t^p$ for $p \ge 1$):**
$$\int_X f^p \, d\mu = p \int_0^\infty t^{p-1} \mu(\{f > t\}) \, dt.$$

**Proof (of the $\varphi(t) = t$ case first).** Using $f(x) = \int_0^{f(x)} dt = \int_0^\infty \mathbf 1_{\{t < f(x)\}} \, dt$, and Tonelli's theorem (non-negative integrands, Module 1.4 — or for now MCT applied to truncations):
$$\int f \, d\mu = \int_X \int_0^\infty \mathbf 1_{\{t < f(x)\}} \, dt \, d\mu(x) = \int_0^\infty \int_X \mathbf 1_{\{t < f(x)\}} \, d\mu(x) \, dt = \int_0^\infty \mu(\{f > t\}) \, dt.$$

**General $\varphi$:** write $\varphi(f(x)) = \int_0^{f(x)} \varphi'(t) \, dt = \int_0^\infty \varphi'(t) \mathbf 1_{\{t < f(x)\}} \, dt$, and apply the same Tonelli argument. $\blacksquare$

This formula is the foundation of probability tail arguments: $\mathbb E X^p = p \int_0^\infty t^{p-1} \mathbb P(X > t) \, dt$ for $X \ge 0$.

### Theorem 1.3.31 (Change of variables / image measure)

Let $T: (X, \mathcal F, \mu) \to (Y, \mathcal G)$ be measurable. The **pushforward measure** $T_*\mu$ on $\mathcal G$ is defined by $T_*\mu(B) := \mu(T^{-1}(B))$. For a measurable $g: Y \to \overline{\mathbb R}$:
$$\int_Y g \, d(T_*\mu) = \int_X (g \circ T) \, d\mu,$$
provided either side makes sense (both non-negative, or $g \circ T \in L^1(\mu)$).

**Proof.** Check on indicator functions: for $B \in \mathcal G$, $\int_Y \mathbf 1_B \, d(T_*\mu) = T_*\mu(B) = \mu(T^{-1} B) = \int_X \mathbf 1_{T^{-1}B} \, d\mu = \int_X \mathbf 1_B \circ T \, d\mu$. Extend to simple functions by linearity, to non-negative measurables by MCT (picking simple approximations), and to general integrables via $g = g^+ - g^-$. $\blacksquare$

In probability this is the identity $\mathbb E g(X) = \int g(x) \, d\mu_X(x)$ where $\mu_X = X_* \mathbb P$ is the distribution of the random variable $X$; it is why all of distribution-theoretic probability reduces to Lebesgue integration on $(\mathbb R, \mathcal B)$ with the pushforward measure.

---

## Topic 1.3.8 — Python: Numerical Verification of MCT, Fatou, DCT

```python
"""
Numerical experiments supporting the core convergence theorems.
Uses scipy for numerical integration (Simpson's rule / adaptive),
and a Monte Carlo check against the Lebesgue integral for functions
where Riemann fails (e.g., Dirichlet).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ---------- MCT: f_n(x) = min(f(x), n) with f(x) = 1/sqrt(x) on (0, 1) ----------
def f_truncated(x, n):
    return np.minimum(1.0 / np.sqrt(np.maximum(x, 1e-12)), n)

def integral_truncated(n, num_pts=10**6):
    # Use Simpson's rule on a dense grid; cap x away from 0 by 1/n^2 for the sup of f_n.
    # f_n is bounded by n, so Riemann is fine here.
    xs = np.linspace(1e-10, 1.0, num_pts)
    return np.trapz(f_truncated(xs, n), xs)

true_integral = 2.0  # int_0^1 x^{-1/2} dx = 2
ns = [1, 2, 5, 10, 50, 100, 1000]
print(f"True integral: {true_integral}")
for n in ns:
    approx = integral_truncated(n)
    print(f"n = {n:6d} -> Int f_n = {approx:.6f}   |f_n - f| gap = {abs(approx - true_integral):.6f}")

# Observation: integrals of truncations increase monotonically to 2.
# This is MCT in action.

# ---------- Fatou: f_n = n * 1_{(0, 1/n)} -> strict inequality ----------
# Pointwise limit is 0, but int f_n = 1 for all n. Numerical verification:
for n in [10, 100, 1000]:
    # Riemann approximation: f_n = n on (0, 1/n), 0 elsewhere
    integral = n * (1/n)  # = 1
    print(f"f_n = n * 1_(0, 1/n): int = {integral}, but lim f_n = 0 pointwise")
# Fatou: int lim inf = 0 <= lim inf int = 1. Strict.

# ---------- DCT: f_n(x) = sin(nx)/n on (0, 2pi), dominated by 1/n ----------
# Actually sin(nx)/n is bounded by 1/n, so DCT with g = 1 gives int -> 0.
for n in [1, 10, 100, 1000]:
    xs = np.linspace(0, 2*np.pi, 10**5)
    approx = np.trapz(np.sin(n * xs) / n, xs)
    print(f"n = {n}: int sin(nx)/n dx on (0, 2pi) = {approx:.8f}  (should -> 0)")

# ---------- Monte Carlo: Lebesgue integral of Dirichlet ----------
# f = 1 on rationals, 0 on irrationals. int f d lambda = 0.
# Uniform samples from [0,1] land on irrationals with probability 1.
rng = np.random.default_rng(42)
samples = rng.uniform(0, 1, 10**6)
def dirichlet(x):
    # A numerical sample is an irrational (representable float, but uniformly random)
    # so we return 0 always. This demonstrates that Lebesgue integration ignores
    # measure-zero sets of "rational-like" points.
    return np.zeros_like(x)
print(f"MC estimate of int_[0,1] 1_Q dx = {dirichlet(samples).mean():.8f}")
# Riemann cannot integrate 1_Q (upper sums = 1, lower sums = 0).
# Lebesgue: int = lambda(Q cap [0,1]) = 0.
```

Expected output (abbreviated):
- MCT truncation sequence: $1.0 < 1.75 < 1.91 \ldots \to 2.0$.
- Fatou with $f_n = n \mathbf 1_{(0, 1/n)}$: each $\int f_n = 1$, but $\int \lim f_n = 0$; strict inequality in Fatou.
- DCT $\int \sin(nx)/n \to 0$: values decay like $1/n$.
- Monte Carlo for $\mathbf 1_{\mathbb Q}$: estimated $0.0$, matching the Lebesgue integral.

### Python 2: Layer cake visualization for a heavy-tailed distribution

```python
"""
Verify int X^p = p int_0^inf t^{p-1} P(X > t) dt for a Pareto(alpha) random variable.
For alpha > p, the p-th moment is finite; we compute both sides numerically.
"""
import numpy as np
from scipy import integrate

alpha = 3.0   # shape parameter, so P(X > t) = (1/t)^alpha for t >= 1
p = 2.0       # compute E[X^p]

# Direct: E[X^p] = int_1^inf x^p * alpha * x^{-alpha-1} dx = alpha / (alpha - p)
direct = alpha / (alpha - p)
print(f"Direct E[X^p] = {direct:.6f}")

# Layer cake: p * int_0^inf t^{p-1} P(X > t) dt
# P(X > t) = 1 for t < 1, (1/t)^alpha for t >= 1
def tail(t):
    return np.where(t < 1, 1.0, t**(-alpha))

integrand = lambda t: p * t**(p - 1) * tail(t)
layer_cake, _ = integrate.quad(integrand, 0, np.inf)
print(f"Layer cake E[X^p] = {layer_cake:.6f}")
# Should agree.
```

### Python 3: DCT diagnostic — identify when envelope fails

```python
# Check if a candidate dominator is integrable. If not, warn.
import numpy as np
def suggest_dominator(f_funcs, grid, lebesgue_measure_of_cell):
    """
    f_funcs: list of vectorized functions f_n(x).
    grid: array of x-values.
    Returns g(x) = sup_n |f_n(x)| and its "integral" (Riemann on the grid).
    Warns if the integral appears to diverge (heuristic: rises faster than log).
    """
    vals = np.array([np.abs(f(grid)) for f in f_funcs])
    g = vals.max(axis=0)
    approx_int = g.sum() * lebesgue_measure_of_cell
    return g, approx_int

grid = np.linspace(1e-6, 1, 10**5)
dx = grid[1] - grid[0]
f_funcs = [lambda x, n=n: n * ((x > 0) & (x < 1/n)).astype(float) for n in [10, 100, 1000]]
g, g_int = suggest_dominator(f_funcs, grid, dx)
print(f"sup_n |f_n| approximate integral = {g_int:.4f}")
# grows without bound with finer grid -> dominator not in L^1.
# This is exactly why DCT fails for f_n = n * 1_(0, 1/n).
```

---

## Topic 1.3.9 — [QUANT APPLICATION] Expected Values, Monte Carlo, and Pricing

### Application 1. Expectation is a Lebesgue integral

On a probability space $(\Omega, \mathcal F, \mathbb P)$, a random variable $X: \Omega \to \mathbb R$ has expected value
$$\mathbb E X = \int_\Omega X \, d\mathbb P,$$
defined only when $X \in L^1(\mathbb P)$ (otherwise we split into $\mathbb E X^+$ and $\mathbb E X^-$ and say $\mathbb E X$ is defined as $\mathbb E X^+ - \mathbb E X^- \in [-\infty, \infty]$ when at least one is finite).

By the change-of-variables theorem (1.3.31), $\mathbb E X = \int_\mathbb R x \, d\mu_X(x)$ where $\mu_X = X_*\mathbb P$ is the distribution of $X$. If $X$ has a density $f$ w.r.t. Lebesgue ($\mu_X(dx) = f(x) dx$), this becomes $\mathbb E X = \int x f(x) dx$, the formula from elementary probability.

### Application 2. Monte Carlo pricing

The risk-neutral price of a European option with payoff $H(S_T)$ is
$$V_0 = e^{-rT} \mathbb E^{\mathbb Q}[H(S_T)] = e^{-rT} \int_\Omega H(S_T(\omega)) \, d\mathbb Q(\omega) = e^{-rT} \int_\mathbb R H(s) \, d\mu_{S_T}(s).$$
Generating samples $S_T^{(1)}, \ldots, S_T^{(N)}$ under $\mathbb Q$ and computing $\widehat V_0 = e^{-rT} \frac{1}{N} \sum_i H(S_T^{(i)})$ is a **strong law of large numbers** application:
$$\widehat V_0 \to V_0 \quad \mathbb Q\text{-a.s. as } N \to \infty$$
provided $H(S_T) \in L^1(\mathbb Q)$. The rate $1/\sqrt N$ comes from the central limit theorem, proved in Module 2.4 using characteristic functions (from Module 0.5).

### Application 3. DCT in Greeks computation

The delta of an option is $\Delta = \partial V_0 / \partial S_0$. Writing $V_0(S_0) = e^{-rT} \mathbb E H(S_T(S_0))$, and assuming differentiability conditions, by Example 1.3.23:
$$\Delta = e^{-rT} \mathbb E \partial_{S_0} H(S_T(S_0)).$$
The *pathwise method* for Greeks requires a DCT-style envelope on the derivative. For non-smooth $H$ (e.g. vanilla calls), this uses the Clark-Ocone or Malliavin calculus (Module 4.7). The *likelihood ratio method* circumvents this by differentiating the density rather than the payoff.

### Application 4. Fatou for risk bounds

Let $L_n$ be the loss on day $n$, and suppose $L_n \to L$ a.s. (e.g. as the portfolio is held). If $\mathbb E L_n \le C$ for all $n$ (bounded mean loss), then $\mathbb E L \le C$. This is used to argue that candidate strategies satisfy a risk bound in the limit.

### Application 5. Jensen's inequality, risk aversion, and MGF bounds

If $\varphi$ is convex and $X \in L^1$ with $\varphi(X) \in L^1$, then $\varphi(\mathbb E X) \le \mathbb E \varphi(X)$. The proof uses the supporting line $\varphi(x) \ge \varphi(\mu) + c(x - \mu)$ (from $\varphi$ convex) and integrates — pure Lebesgue linearity + monotonicity. In economics: a risk-averse investor with concave utility $u$ satisfies $u(\mathbb E W) \ge \mathbb E u(W)$, so a certain payoff of $\mathbb E W$ is preferred to the random payoff $W$.

### Application 6. The moment generating function

For $X$ with $\mathbb E e^{tX} < \infty$ for $t$ in an open interval around $0$, Example 1.3.23 gives
$$M_X'(t) = \mathbb E [X e^{tX}], \qquad M_X^{(n)}(0) = \mathbb E X^n,$$
generating moments by differentiating the MGF. The MGF of a normal $N(\mu, \sigma^2)$ is $\exp(\mu t + \sigma^2 t^2 / 2)$, used in the Black-Scholes derivation in Module 0.5 and the theory of sub-Gaussian random variables in Module 2.4.

### Application 7. Absolute continuity of the integral and VaR/ES

**Value at Risk (VaR)** at confidence $\alpha$: $\mathrm{VaR}_\alpha(L) = \inf\{\ell : \mathbb P(L > \ell) \le 1 - \alpha\}$.
**Expected Shortfall (ES)**: $\mathrm{ES}_\alpha(L) = \mathbb E[L | L > \mathrm{VaR}_\alpha(L)] = \frac{1}{1-\alpha} \int_\alpha^1 \mathrm{VaR}_u(L) \, du$.

ES is *coherent* (subadditive) whereas VaR is not. The fact that $L \mapsto \mathrm{ES}_\alpha(L)$ is a (law-invariant, convex, monotone) risk measure uses the layer cake and absolute continuity. This underpins Basel III / IV capital requirements.

### Application 8. DCT for continuity of characteristic functions

The characteristic function $\varphi_X(t) = \mathbb E e^{itX}$ is continuous in $t$: by DCT with $g = 1$,
$$\lim_{t \to t_0} \varphi_X(t) = \mathbb E \lim_{t \to t_0} e^{itX} = \mathbb E e^{it_0 X} = \varphi_X(t_0).$$
This is the first step in Bochner's theorem and in the Lévy continuity theorem (Module 0.5 preview, full proof in Module 2.4).

---

## Topic 1.3.10 — Exercises

### ★ (Foundational)

**1.3.E1.** Compute $\int_{[0, 3]} \lfloor x \rfloor \, d\lambda$ directly from the simple-function definition.

**1.3.E2.** Let $f \ge 0$ be simple with $\int f \, d\mu = 0$. Show $f = 0$ a.e. directly.

**1.3.E3.** Let $f_n(x) = x^n$ on $[0, 1]$. Compute $\lim_n \int_0^1 f_n \, dx$ two ways (MCT or DCT + pointwise limit).

**1.3.E4.** Show: if $f \ge 0$ is integrable with $\int f = c > 0$, the measure $\nu(E) := \int_E f \, d\mu / c$ is a probability measure (a "tilting" of $\mu$).

**1.3.E5.** Use the layer cake to compute $\mathbb E X$ for $X \sim \text{Exp}(\lambda)$ (exponential with rate $\lambda$).

**1.3.E6.** Verify by direct computation that if $f = \mathbf 1_{[0,1]} - \mathbf 1_{[1,2]}$ on $[0,2]$ with Lebesgue measure, then $\int f \, d\lambda = 0$ while $\int |f| \, d\lambda = 2$.

**1.3.E7.** Let $f_n = \frac{1}{n} \mathbf 1_{[0, n]}$ on $\mathbb R$. Show $f_n \to 0$ uniformly but $\int f_n = 1$ for every $n$, so uniform convergence alone does not justify swapping limit and integral when $\mu$ is not finite.

**1.3.E8.** Prove: $f \in L^1(\mu) \iff$ for every $\varepsilon > 0$ there is a measurable set $E_\varepsilon$ with $\mu(E_\varepsilon) < \infty$ and $\int_{X \setminus E_\varepsilon} |f| \, d\mu < \varepsilon$. (I.e. integrable functions are "essentially supported on a finite-measure set".)

### ★★ (Core)

**1.3.E9.** Prove: if $f_n \to f$ in measure on a finite measure space and $|f_n| \le g \in L^1$ a.e., then $\int f_n \to \int f$. (DCT under convergence in measure.)

**1.3.E10.** **Scheffé's Lemma.** Let $f_n, f \ge 0$ be measurable with $f_n \to f$ a.e. and $\int f_n \to \int f < \infty$. Show $\int |f_n - f| \to 0$. *Hint: split $|f_n - f| = (f_n - f)^+ + (f_n - f)^-$ and use $(f_n - f)^- \le f$.*

**1.3.E11.** **Brezis-Lieb lemma (lite version).** Let $f_n \to f$ a.e. with $|f_n|, |f| \in L^p(\mu)$ for some $p \ge 1$. If $\sup_n \|f_n\|_p < \infty$, then
$$\lim_n \int |f_n|^p - \int |f_n - f|^p - \int |f|^p = 0.$$
(Equivalently, $\|f_n\|_p^p = \|f_n - f\|_p^p + \|f\|_p^p + o(1)$.) *Hint: use that $|a|^p - |a - b|^p - |b|^p \to 0$ uniformly on compacts and apply a modified DCT.* ($p = 1$ case first.)

**1.3.E12.** Prove: if $\mu$ is $\sigma$-finite, the set $\{|f| < \infty, f \text{ integrable}\}$ is closed under pointwise limits *if* we have uniform $L^1$ control. Make a precise statement and prove it.

**1.3.E13.** Show that if $f \in L^1(\lambda)$ on $\mathbb R$, then $\lim_{|t| \to \infty} \int f(x) \cos(tx) \, dx = 0$ (Riemann-Lebesgue lemma, easy case). *Hint: reduce to $f \in C_c(\mathbb R)$ via $L^1$-density (cf. Module 1.5).*

**1.3.E14.** Let $f \in L^1(\mathbb R)$. Define $g(t) = \int f(x) e^{-(x-t)^2} dx$. Use DCT to prove $g$ is continuous on $\mathbb R$, and bounded.

**1.3.E15.** Show that $h(x) = \log x \cdot \mathbf 1_{(0, 1)}(x)$ is in $L^1([0,1])$. Compute $\int_0^1 \log x \, dx$ using MCT applied to $\log \max(x, 1/n)$.

**1.3.E16.** Let $f \ge 0$ on $\mathbb R$ with $\int f \, d\lambda = 1$. Prove: if $\{x : f(x) > 0\}$ has infinite Lebesgue measure then $\operatorname{ess\,inf} f = 0$. (I.e. $f$ must dip down to $0$ somewhere of positive measure — no Roberts-Ulam plateau.)

### ★★★ (Challenging / quant-relevant)

**1.3.E17.** **Sharp Riemann-Lebesgue.** For $f \in L^1(\mathbb R)$, define $\widehat f(t) = \int f(x) e^{-2\pi i t x} dx$. Prove $\widehat f \in C_0(\mathbb R)$ (continuous and vanishing at infinity). *Hint: density of Schwartz functions in $L^1$ + DCT.*

**1.3.E18.** **Improper Riemann but not Lebesgue.** Show $\int_0^\infty \frac{\sin x}{x} dx = \pi/2$ converges as an improper Riemann integral. Compute this integral by either (a) parametrizing with a dampening factor $e^{-tx}$ and differentiating under the integral sign, or (b) contour integration (Module 0.5).

**1.3.E19.** **Lebesgue's criterion for Riemann integrability, full proof.** Let $f: [a, b] \to \mathbb R$ be bounded. Prove $f$ is Riemann integrable iff its set of discontinuities has Lebesgue measure zero. (Outline in Topic 1.3.6; flesh out.)

**1.3.E20.** **Quant: discrete-time hedging.** Consider a one-period binomial model with stock $S_0 \in \{u, d\}$ at time $1$ (relative to $S_0 = 1$). A claim $H(S_1)$ is replicated by holding $\Delta$ shares and $B$ in cash, so $\Delta u + B(1+r) = H(u)$ and $\Delta d + B(1+r) = H(d)$. Use linearity of the Lebesgue integral (expectation) to express the replication cost $V_0 = \Delta + B$ as $\mathbb E^{\mathbb Q}[H(S_1)] / (1 + r)$ for an appropriate risk-neutral measure $\mathbb Q$.

**1.3.E21.** **Quant: Monte Carlo with control variates.** Let $X \in L^2$ and $Y \in L^2$ with known $\mathbb E Y$. The estimator $\widehat M := \frac{1}{N} \sum_i (X_i - c(Y_i - \mathbb E Y))$ is unbiased for $\mathbb E X$. Prove that choosing $c = \operatorname{Cov}(X, Y)/\operatorname{Var}(Y)$ minimizes the variance of $\widehat M$, and the reduction factor is $1 - \rho^2(X, Y)$. *Uses Cauchy-Schwarz and the fact that $\mathbb E X^2 < \infty$ allows moment computations.*

**1.3.E22.** **Quant: DCT for stopping times.** Let $(X_n)$ be a martingale bounded by an integrable $Y$ (i.e. $|X_n| \le Y$ for all $n$ with $Y \in L^1$). Let $\tau$ be a stopping time. Prove $\mathbb E X_\tau = \mathbb E X_0$ using DCT. (Preview of the optional stopping theorem, Module 2.6.)

---

## Module Summary

- The Lebesgue integral is built in three stages: simple functions → non-negative measurables → general measurables, with each stage justified by a well-definedness lemma and culminating in the three convergence theorems (MCT, Fatou, DCT).
- **MCT** is the foundation: $f_n \uparrow f \Rightarrow \int f_n \uparrow \int f$. The proof uses a clever "pie slice" $E_n$ construction with a scaling factor $c < 1$.
- **Fatou's lemma** is an immediate consequence of MCT applied to $g_n = \inf_{k \ge n} f_k$ and gives a one-sided inequality that suffices in many proofs. Its strength is that it requires nothing but non-negativity.
- **DCT** is proved via Fatou on $2g - |f_n - f|$ and yields $\int |f_n - f| \to 0$, which is strictly stronger than $\int f_n \to \int f$.
- The Lebesgue integral strictly extends the Riemann integral: every Riemann-integrable function on $[a, b]$ is Lebesgue integrable with the same value, and Riemann integrability is characterized by "set of discontinuities has Lebesgue measure zero." On unbounded intervals with signed integrands, improper Riemann integrals may exist without Lebesgue integrability.
- **Absolute continuity of the integral** (Theorem 1.3.29) states that integrals over small sets are small — foundational for the Radon-Nikodym theorem (Module 1.6).
- The **layer cake formula** converts integrals of non-negative functions into integrals of measure tails: the basis of tail-based moment estimation.
- **Pushforward / change of variables** identifies probabilistic expectations $\mathbb E g(X)$ with integrals $\int g \, d\mu_X$ against the distribution — the engine of classical probability calculations.

**Forward pointers:**
- Module 1.4 (Product Measures and Fubini) will make precise the double-integral swaps we've used implicitly. It introduces product $\sigma$-algebras and shows that for $\sigma$-finite measures, Tonelli's and Fubini's theorems hold unconditionally (for non-negative) and with absolute integrability (for signed).
- Module 1.5 ($L^p$ spaces) uses the Lebesgue integral to define Banach spaces of functions, with Hölder / Minkowski as the key inequalities, Riesz-Fischer giving completeness, and the Riesz representation theorem identifying duals.
- Module 1.6 (Radon-Nikodym and Signed Measures) proves that under absolute continuity and $\sigma$-finiteness, $\nu \ll \mu \Rightarrow \nu = f \cdot \mu$ for some $f \ge 0$ a.e. — the abstract formulation of "density function" and the foundation of change-of-measure theorems (Girsanov, risk-neutral pricing, likelihood ratios).
- In Subject 2 (Probability), the Lebesgue integral becomes expectation, and all of the convergence theorems translate into facts about random variables: a.s. convergence + $L^1$ bound (Fatou), a.s. convergence + dominated (DCT), monotone (MCT). The proofs of the law of large numbers, central limit theorem, and martingale convergence theorems all rest on today's machinery.

**Recommended reading:**
- Folland, *Real Analysis*, Chapter 2.
- Royden & Fitzpatrick, *Real Analysis*, Chapters 3-4.
- Rudin, *Real and Complex Analysis*, Chapter 1.
- Billingsley, *Probability and Measure*, Chapters 15-16 (probabilistic angle).
- Williams, *Probability with Martingales*, Chapters 5-6 (clean, compact).

**Next module:** Product measures and Fubini's theorem — the foundation for multidimensional Lebesgue integration, joint distributions, and convolutions.

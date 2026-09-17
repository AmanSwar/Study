# Subject 1 — Measure Theory
# Module 1.1 — σ-Algebras and Measures

> *"The difference between a good mathematician and a great one is measure theory."* — attributed (variously) to Kolmogorov and Feller.

## Prerequisites

- **Module 0.1** (Logic, sets, functions, cardinality, countability, Cantor's diagonal).
- **Module 0.4** (Metric spaces, open/closed sets in $\mathbb{R}^n$, compactness, Borel σ-algebra is foreshadowed by the topology of $\mathbb{R}^n$).
- **Module 0.5** (A bit of complex analysis — not strictly required for Subject 1, but comfort with rigorous proofs of uniqueness theorems is assumed).

The two most important prerequisite facts:

1. **Countable unions and intersections:** A countable union of countable sets is countable; every open set in $\mathbb{R}$ is a countable union of open intervals.
2. **Cardinality of $\mathbb{R}$:** $|\mathbb{R}| = 2^{\aleph_0}$, strictly greater than $|\mathbb{Q}| = \aleph_0$ — hence, intuitively, "most" subsets of $\mathbb{R}$ are not measurable.

## Why Measure Theory?

Riemann integration suffices for most of calculus but breaks down in three important settings:

**1. Limits of integrals.** If $f_n \to f$ pointwise on $[0, 1]$ with $f_n$ Riemann integrable, is $f$ Riemann integrable, and if so, does $\int f_n \to \int f$? In general, **no**: the indicator $\mathbf 1_\mathbb{Q}$ on $[0, 1]$ is a pointwise limit of Riemann-integrable functions (indicators of finite rational sets) yet is itself not Riemann integrable.

**2. Function spaces.** The space of Riemann-integrable functions is *not complete* under the $L^2$ norm — Cauchy sequences can fail to converge to Riemann-integrable functions. This breaks Hilbert-space methods, which are central to Fourier analysis, spectral theory, and $L^2$-probability.

**3. Probability.** To talk rigorously about a probability measure on $\mathbb{R}^d$ (or on path space, e.g., Wiener measure on $C[0, 1]$), we need a framework that assigns "probability" to a class of events closed under countable union and intersection — a σ-algebra.

Measure theory resolves all three: the Lebesgue integral has strong convergence theorems (MCT, Fatou, DCT), $L^p$-spaces are complete, and probability theory rests on measure-theoretic foundations (Kolmogorov 1933).

Quant relevance: **every single model of quant finance**, from Black-Scholes to Heston to rough volatility, is built on a probability space $(\Omega, \mathcal F, \mathbb P)$. Stochastic integrals are integrals against martingales (measure-theoretic). Risk measures are expectations. Girsanov's theorem is a change of measure (Radon-Nikodym). You cannot do rigorous quant research without measure theory.

## Module Table of Contents

- **Topic 1.1.1** — Set systems: rings, algebras, σ-algebras, monotone classes.
- **Topic 1.1.2** — Measures: definition, properties, examples (counting, Dirac, Lebesgue, probability).
- **Topic 1.1.3** — Outer measures and Carathéodory extension.
- **Topic 1.1.4** — Borel σ-algebra and construction of Lebesgue measure.
- **Topic 1.1.5** — Null sets, complete measures, and completion.
- **Topic 1.1.6** — Non-measurable sets: Vitali construction, Banach-Tarski preview.

---

## Topic 1.1.1 — Set Systems

### 1.1.1.1 Motivation

Given a "ground space" $\Omega$ (e.g., $\mathbb{R}$, $\mathbb{R}^d$, a sample space in probability), we want to assign *sizes* or *probabilities* to subsets. But the set-theoretic zoo is vast — pathological subsets of $\mathbb{R}$ resist any reasonable size assignment (Vitali sets, etc.). The resolution is to restrict attention to a well-behaved family of "measurable" subsets, closed under countable set operations.

### 1.1.1.2 Definitions

Let $\Omega$ be a set.

**Definition 1.1.1.1 (Algebra).** A family $\mathcal A \subseteq 2^\Omega$ is an **algebra** (or **field**) on $\Omega$ if:

(A1) $\Omega \in \mathcal A$.

(A2) $A \in \mathcal A \Rightarrow A^c \in \mathcal A$ (closed under complement).

(A3) $A, B \in \mathcal A \Rightarrow A \cup B \in \mathcal A$ (closed under finite union).

From (A2) and (A3) plus De Morgan, an algebra is closed under finite intersections and set differences.

**Definition 1.1.1.2 (σ-Algebra).** A family $\mathcal F \subseteq 2^\Omega$ is a **σ-algebra** (or **σ-field**) on $\Omega$ if it is an algebra and additionally:

(σA3) $A_1, A_2, \ldots \in \mathcal F \Rightarrow \bigcup_{n=1}^\infty A_n \in \mathcal F$ (closed under countable unions).

By De Morgan, σ-algebras are closed under countable intersections and set differences as well.

**Definition 1.1.1.3 (Ring, σ-Ring).** A **ring** on $\Omega$ is a family closed under finite unions and set differences (but not necessarily containing $\Omega$). A **σ-ring** adds closure under countable unions. Rings are useful when $\Omega$ is "infinite" in a way that makes $\Omega$ itself unwieldy (e.g., $\sigma$-finite measures on $\mathbb{R}$).

**Elements of a σ-algebra** are called **measurable sets** (or, in probability, **events**).

### 1.1.1.3 Examples

**Example 1: The trivial σ-algebra.** $\mathcal F = \{\emptyset, \Omega\}$. Smallest σ-algebra on $\Omega$.

**Example 2: The discrete σ-algebra.** $\mathcal F = 2^\Omega$ (all subsets). Largest σ-algebra.

**Example 3: The co-countable σ-algebra.** $\mathcal F = \{A \subseteq \Omega : A \text{ or } A^c \text{ is countable}\}$. Smallest σ-algebra containing all singletons when $\Omega$ is uncountable (and not equal to $2^\Omega$).

**Example 4: The Borel σ-algebra.** $\mathcal B(\mathbb{R})$ is the smallest σ-algebra containing all open intervals (equivalently, all open sets). This is the quintessential σ-algebra for analysis and probability on $\mathbb{R}$. Same for $\mathcal B(\mathbb{R}^n)$ using open balls or open rectangles.

**Example 5: σ-Algebra generated by a partition.** If $\{A_i\}_{i \in I}$ is a (countable) partition of $\Omega$, the σ-algebra they generate consists of all unions $\bigcup_{i \in J} A_i$ for $J \subseteq I$.

**Example 6: Product σ-algebra.** If $\mathcal F_1, \mathcal F_2$ are σ-algebras on $\Omega_1, \Omega_2$, the **product σ-algebra** $\mathcal F_1 \otimes \mathcal F_2$ on $\Omega_1 \times \Omega_2$ is generated by "rectangles" $A_1 \times A_2$ with $A_i \in \mathcal F_i$.

**Example 7: Cylinder σ-algebra on path space.** $\Omega = \mathbb{R}^{[0, T]}$ (functions $[0, T] \to \mathbb{R}$). The cylinder σ-algebra is generated by finite-dimensional "cylinder sets" $\{f : (f(t_1), \ldots, f(t_n)) \in B\}$ for $B \in \mathcal B(\mathbb{R}^n)$. This is the ambient σ-algebra on which Wiener measure is constructed.

### 1.1.1.4 Generated σ-Algebras

**Theorem 1.1.1.4.** The intersection of any family of σ-algebras on $\Omega$ is a σ-algebra.

*Proof.* Let $\{\mathcal F_\alpha\}_{\alpha \in I}$ be a family of σ-algebras, and let $\mathcal F = \bigcap_\alpha \mathcal F_\alpha$.

- $\Omega \in \mathcal F_\alpha$ for each $\alpha$, so $\Omega \in \mathcal F$.
- If $A \in \mathcal F$, then $A \in \mathcal F_\alpha$ for each $\alpha$, so $A^c \in \mathcal F_\alpha$ for each $\alpha$, hence $A^c \in \mathcal F$.
- If $A_1, A_2, \ldots \in \mathcal F$, then each $A_n \in \mathcal F_\alpha$ for each $\alpha$, so $\bigcup A_n \in \mathcal F_\alpha$ for each $\alpha$, hence $\bigcup A_n \in \mathcal F$. $\blacksquare$

**Corollary 1.1.1.5 (Generated σ-algebra).** For any $\mathcal E \subseteq 2^\Omega$, there is a *smallest* σ-algebra containing $\mathcal E$, denoted $\sigma(\mathcal E)$. It equals the intersection of all σ-algebras containing $\mathcal E$ (this intersection is nonempty since $2^\Omega$ is one such σ-algebra).

**Caution.** $\sigma(\mathcal E)$ is defined abstractly as an intersection; there is no *constructive* description of $\sigma(\mathcal E)$ in general. Trying to build it by taking all countable unions, then complements, then countable unions, etc., does not terminate — you need to iterate transfinitely (up to $\omega_1$, the first uncountable ordinal). This non-constructiveness is fundamental.

### 1.1.1.5 Monotone Class and π-λ Theorem

In practice we often want to prove a σ-algebra-level statement by checking it on a simpler generator. The two tools for this are the **monotone class theorem** and the **π-λ theorem**.

**Definition 1.1.1.6 (π-system, λ-system, monotone class).**

- A **π-system** on $\Omega$ is a family closed under finite intersections.
- A **λ-system** (or **Dynkin system**) on $\Omega$ is a family $\mathcal L$ with: (i) $\Omega \in \mathcal L$; (ii) $A, B \in \mathcal L, A \subseteq B \Rightarrow B \setminus A \in \mathcal L$; (iii) $A_n \in \mathcal L, A_n \uparrow A \Rightarrow A \in \mathcal L$ (closed under increasing countable unions).
- A **monotone class** is a family closed under increasing countable unions and decreasing countable intersections.

**Theorem 1.1.1.7 (Dynkin's π-λ theorem).** If $\mathcal P$ is a π-system and $\mathcal L$ is a λ-system on $\Omega$ with $\mathcal P \subseteq \mathcal L$, then $\sigma(\mathcal P) \subseteq \mathcal L$.

*Proof.* Let $\lambda(\mathcal P)$ denote the smallest λ-system containing $\mathcal P$. It's straightforward that $\lambda(\mathcal P) \subseteq \mathcal L$; we show $\lambda(\mathcal P) = \sigma(\mathcal P)$.

Since every σ-algebra is a λ-system, $\lambda(\mathcal P) \subseteq \sigma(\mathcal P)$. For the reverse, it suffices to show $\lambda(\mathcal P)$ is a σ-algebra, i.e., closed under finite (and hence countable, by complement + intersection + λ-system) intersections.

Define, for each $A \in \lambda(\mathcal P)$,
$$
\mathcal G_A := \{B \in \lambda(\mathcal P) : A \cap B \in \lambda(\mathcal P)\}.
$$
$\mathcal G_A$ is a λ-system (easy check using the λ-system axioms for $\lambda(\mathcal P)$ and distributivity of intersection over set difference / countable union).

For $A \in \mathcal P$: $B \in \mathcal P \Rightarrow A \cap B \in \mathcal P$ (π-system) $\subseteq \lambda(\mathcal P)$, so $\mathcal P \subseteq \mathcal G_A$. Hence $\lambda(\mathcal P) \subseteq \mathcal G_A$, i.e., $A \cap B \in \lambda(\mathcal P)$ for all $B \in \lambda(\mathcal P)$.

Now for arbitrary $A \in \lambda(\mathcal P)$: by the above, $\mathcal P \subseteq \mathcal G_A$ (since $B \in \mathcal P \Rightarrow A \cap B \in \lambda(\mathcal P)$ by the previous paragraph). Again $\lambda(\mathcal P) \subseteq \mathcal G_A$. So $\lambda(\mathcal P)$ is closed under intersection, hence a σ-algebra. $\blacksquare$

**Application.** If two finite measures $\mu, \nu$ agree on a π-system $\mathcal P$, they agree on $\sigma(\mathcal P)$. (Let $\mathcal L = \{A : \mu(A) = \nu(A)\}$; this is a λ-system containing $\mathcal P$; by π-λ, $\sigma(\mathcal P) \subseteq \mathcal L$.)

**Theorem 1.1.1.8 (Monotone class theorem).** Let $\mathcal A$ be an algebra and $\mathcal M$ a monotone class with $\mathcal A \subseteq \mathcal M$. Then $\sigma(\mathcal A) \subseteq \mathcal M$.

Proof similar in spirit to the π-λ theorem: show the smallest monotone class containing $\mathcal A$ equals $\sigma(\mathcal A)$.

### 1.1.1.6 Measurable Space

**Definition 1.1.1.9 (Measurable space).** A **measurable space** is a pair $(\Omega, \mathcal F)$ where $\Omega$ is a set and $\mathcal F$ is a σ-algebra on $\Omega$.

---

## Topic 1.1.2 — Measures

### 1.1.2.1 Definition and Basic Properties

**Definition 1.1.2.1 (Measure).** Let $(\Omega, \mathcal F)$ be a measurable space. A **measure** on $(\Omega, \mathcal F)$ is a function $\mu: \mathcal F \to [0, \infty]$ such that:

(M1) $\mu(\emptyset) = 0$.

(M2) **Countable additivity:** For any disjoint sequence $A_1, A_2, \ldots \in \mathcal F$,
$$
\mu\left(\bigsqcup_{n=1}^\infty A_n\right) = \sum_{n=1}^\infty \mu(A_n).
$$

A **measure space** is a triple $(\Omega, \mathcal F, \mu)$. A **probability measure** is a measure with $\mu(\Omega) = 1$; a **probability space** is then $(\Omega, \mathcal F, \mathbb P)$.

**Proposition 1.1.2.2 (Basic properties).** Let $(\Omega, \mathcal F, \mu)$ be a measure space.

(a) **Finite additivity:** $\mu(A \sqcup B) = \mu(A) + \mu(B)$ for disjoint $A, B$.

(b) **Monotonicity:** $A \subseteq B \Rightarrow \mu(A) \leq \mu(B)$.

(c) **Subtractivity:** $A \subseteq B$ and $\mu(A) < \infty \Rightarrow \mu(B \setminus A) = \mu(B) - \mu(A)$.

(d) **Countable subadditivity:** $\mu(\bigcup_n A_n) \leq \sum_n \mu(A_n)$ for any sequence $A_n \in \mathcal F$ (not necessarily disjoint).

(e) **Continuity from below:** $A_n \uparrow A \Rightarrow \mu(A_n) \uparrow \mu(A)$.

(f) **Continuity from above:** $A_n \downarrow A$ with $\mu(A_1) < \infty \Rightarrow \mu(A_n) \downarrow \mu(A)$.

*Proof.*
**(a):** Take $A_1 = A, A_2 = B, A_n = \emptyset$ for $n \geq 3$; apply (M2).

**(b):** $B = A \sqcup (B \setminus A)$, so $\mu(B) = \mu(A) + \mu(B \setminus A) \geq \mu(A)$.

**(c):** Subtract using (b), allowed since $\mu(A) < \infty$.

**(d):** Write $\bigcup_n A_n = \bigsqcup_n B_n$ where $B_n = A_n \setminus \bigcup_{k < n} A_k$. Then $B_n \subseteq A_n$, so $\mu(B_n) \leq \mu(A_n)$, and $\mu(\bigcup A_n) = \sum \mu(B_n) \leq \sum \mu(A_n)$.

**(e):** Write $A = A_1 \sqcup (A_2 \setminus A_1) \sqcup (A_3 \setminus A_2) \sqcup \cdots$. By (M2), $\mu(A) = \mu(A_1) + \sum_{n \geq 2} \mu(A_n \setminus A_{n-1})$. Telescoping partial sums (valid because each term is finite by $A_n \subseteq A_{n+1}$ and positivity): $\sum_{k=1}^n \mu(A_k \setminus A_{k-1}) = \mu(A_n)$ (with $A_0 := \emptyset$). So $\mu(A) = \lim_{n} \mu(A_n)$.

**(f):** Apply (e) to $B_n := A_1 \setminus A_n$ (increasing to $A_1 \setminus A$); using $\mu(A_1) < \infty$ to subtract: $\mu(A_1) - \mu(A_n) = \mu(B_n) \uparrow \mu(A_1 \setminus A) = \mu(A_1) - \mu(A)$, hence $\mu(A_n) \downarrow \mu(A)$. $\blacksquare$

**Remark (counterexample to (f) without finiteness).** Let $\mu$ = Lebesgue measure on $\mathbb{R}$ and $A_n = [n, \infty)$. Then $A_n \downarrow \emptyset$, $\mu(A_n) = \infty$ for all $n$, but $\mu(\emptyset) = 0$. So "continuity from above" fails without the finiteness hypothesis.

### 1.1.2.2 σ-Finiteness and Localization

**Definition 1.1.2.3 (σ-finite measure).** A measure $\mu$ is **σ-finite** if $\Omega = \bigcup_{n=1}^\infty \Omega_n$ for some $\Omega_n \in \mathcal F$ with $\mu(\Omega_n) < \infty$ for each $n$.

Most measures in practice are σ-finite: Lebesgue measure on $\mathbb{R}^d$ ($\Omega_n = [-n, n]^d$), counting measure on $\mathbb{N}$, probability measures (trivially σ-finite with $\Omega_1 = \Omega$). Non-σ-finite examples are exotic: counting measure on $\mathbb{R}$, the measure "number of ends a set has" on general spaces.

σ-Finiteness is the regularity condition that makes most theorems (Radon-Nikodym, Fubini) work. Without it, these theorems can fail in weird ways.

### 1.1.2.3 Examples of Measures

**Example 1: Counting measure.** On any measurable space $(\Omega, 2^\Omega)$,
$$
\mu_c(A) := |A| \quad \text{(number of elements, or $\infty$)}.
$$
On $\mathbb{N}$ this is σ-finite; on $\mathbb{R}$ it is not.

**Example 2: Dirac measure.** For $\omega_0 \in \Omega$,
$$
\delta_{\omega_0}(A) := \mathbf 1_{A}(\omega_0) = \begin{cases} 1 & \omega_0 \in A, \\ 0 & \omega_0 \notin A. \end{cases}
$$
This is a probability measure.

**Example 3: Atomic (discrete) measures.** Given countable $\{\omega_n\} \subset \Omega$ and weights $\{p_n\} \geq 0$,
$$
\mu = \sum_n p_n \delta_{\omega_n}.
$$
A probability measure iff $\sum p_n = 1$.

**Example 4: Lebesgue measure on $\mathbb{R}$.** The unique (up to scaling) translation-invariant measure on $\mathcal B(\mathbb{R})$ with $\mu([0, 1]) = 1$. Construction via Carathéodory extension — Topic 1.1.3-4.

**Example 5: Lebesgue-Stieltjes measures.** For any non-decreasing right-continuous $F: \mathbb{R} \to \mathbb{R}$, there is a unique measure $\mu_F$ on $\mathcal B(\mathbb{R})$ with $\mu_F((a, b]) = F(b) - F(a)$. When $F$ is a probability distribution function, $\mu_F$ is a probability measure on $\mathbb{R}$.

**Example 6: Haar measure on a locally compact group.** A translation-invariant measure, unique up to scaling. Generalizes Lebesgue measure from $\mathbb{R}$ to topological groups. Used in representation theory.

**Example 7: Wiener measure.** The unique probability measure on $C[0, T]$ (continuous paths) such that the canonical process $W_t(\omega) = \omega(t)$ is a Brownian motion. Constructed via Kolmogorov extension from finite-dimensional Gaussian distributions — but this requires the full machinery of Subjects 1 and 2.

### 1.1.2.4 Pre-Measures

When constructing measures, we often start with a function defined on an algebra $\mathcal A$, not yet a σ-algebra:

**Definition 1.1.2.4 (Pre-measure).** Let $\mathcal A$ be an algebra on $\Omega$. A **pre-measure** on $\mathcal A$ is a function $\mu_0: \mathcal A \to [0, \infty]$ with:

(PM1) $\mu_0(\emptyset) = 0$.

(PM2) **Countable additivity whenever possible:** For disjoint $A_1, A_2, \ldots \in \mathcal A$ with $\bigsqcup A_n \in \mathcal A$, $\mu_0(\bigsqcup A_n) = \sum \mu_0(A_n)$.

The catch: a countable disjoint union of algebra elements may not be in the algebra, so (PM2) is a non-trivial condition only when it applies.

**Key example.** Define $\mu_0((a, b]) := b - a$ on the algebra $\mathcal A$ of finite disjoint unions of half-open intervals in $\mathbb{R}$. $\mu_0$ is a pre-measure, and Carathéodory extension produces Lebesgue measure on $\mathcal B(\mathbb{R})$.

---

## Topic 1.1.3 — Outer Measures and Carathéodory Extension

### 1.1.3.1 Outer Measures

The Carathéodory construction extends a pre-measure on an algebra to a measure on a σ-algebra, via an intermediate object called an outer measure.

**Definition 1.1.3.1 (Outer measure).** An **outer measure** on $\Omega$ is a function $\mu^*: 2^\Omega \to [0, \infty]$ satisfying:

(OM1) $\mu^*(\emptyset) = 0$.

(OM2) **Monotonicity:** $A \subseteq B \Rightarrow \mu^*(A) \leq \mu^*(B)$.

(OM3) **Countable subadditivity:** $\mu^*(\bigcup A_n) \leq \sum \mu^*(A_n)$.

Note: outer measures are defined on **all** subsets, but they are only subadditive, not additive.

**Construction (outer measure from a pre-measure).** Given a pre-measure $\mu_0$ on an algebra $\mathcal A$, define
$$
\mu^*(E) := \inf\left\{ \sum_n \mu_0(A_n) : A_n \in \mathcal A, \; E \subseteq \bigcup_n A_n \right\}
$$
for $E \subseteq \Omega$.

**Proposition 1.1.3.2.** $\mu^*$ defined above is an outer measure.

*Proof.*
**(OM1):** Take $A_n = \emptyset$; $\mu^*(\emptyset) \leq 0$, hence $= 0$.

**(OM2):** Any cover of $B$ is also a cover of $A$, so the infimum for $A$ is over a larger set, hence smaller.

**(OM3):** Given $\epsilon > 0$ and sets $E_1, E_2, \ldots$, pick covers $\{A_{n,k}\}_k \subseteq \mathcal A$ of $E_n$ with $\sum_k \mu_0(A_{n,k}) \leq \mu^*(E_n) + \epsilon / 2^n$. Then $\{A_{n,k}\}_{n,k}$ covers $\bigcup E_n$, and $\sum_{n,k} \mu_0(A_{n,k}) \leq \sum_n \mu^*(E_n) + \epsilon$. Letting $\epsilon \to 0$, $\mu^*(\bigcup E_n) \leq \sum \mu^*(E_n)$. $\blacksquare$

### 1.1.3.2 Carathéodory Measurability

Outer measures are *not* countably additive on $2^\Omega$ in general. To extract a σ-algebra on which $\mu^*$ *is* countably additive, Carathéodory devised the following:

**Definition 1.1.3.3 (Carathéodory measurable set).** A set $E \subseteq \Omega$ is **μ*-measurable** (or **Carathéodory measurable**) if for every $T \subseteq \Omega$,
$$
\mu^*(T) = \mu^*(T \cap E) + \mu^*(T \cap E^c).
$$

Equivalently (using subadditivity which is automatic): $E$ is measurable iff $\mu^*(T) \geq \mu^*(T \cap E) + \mu^*(T \cap E^c)$ for all $T$ with $\mu^*(T) < \infty$.

### 1.1.3.3 Carathéodory's Extension Theorem

**Theorem 1.1.3.4 (Carathéodory).** Let $\mu^*$ be an outer measure on $\Omega$, and let $\mathcal M$ be the collection of $\mu^*$-measurable sets. Then:

(a) $\mathcal M$ is a σ-algebra.

(b) $\mu := \mu^*|_{\mathcal M}$ is a measure on $(\Omega, \mathcal M)$.

(c) $\mathcal M$ is **complete**: if $N \subseteq E$ with $\mu^*(E) = 0$, then $N \in \mathcal M$ and $\mu(N) = 0$.

*Proof.*

**(a), Step 1 — $\mathcal M$ is an algebra.**

$\emptyset \in \mathcal M$: $\mu^*(T) = \mu^*(T \cap \emptyset) + \mu^*(T \cap \Omega) = 0 + \mu^*(T)$. ✓

Closure under complement: if $E \in \mathcal M$, the defining equation is symmetric in $E$ and $E^c$, so $E^c \in \mathcal M$.

Closure under union: suppose $E_1, E_2 \in \mathcal M$. For any $T$,
$$
\mu^*(T) = \mu^*(T \cap E_1) + \mu^*(T \cap E_1^c) \text{ (using } E_1 \in \mathcal M)
$$
and
$$
\mu^*(T \cap E_1^c) = \mu^*(T \cap E_1^c \cap E_2) + \mu^*(T \cap E_1^c \cap E_2^c) \text{ (using } E_2 \in \mathcal M).
$$
Substituting:
$$
\mu^*(T) = \mu^*(T \cap E_1) + \mu^*(T \cap E_1^c \cap E_2) + \mu^*(T \cap E_1^c \cap E_2^c).
$$
Now $T \cap (E_1 \cup E_2) = (T \cap E_1) \sqcup (T \cap E_1^c \cap E_2)$ (disjoint), so by subadditivity $\mu^*(T \cap (E_1 \cup E_2)) \leq \mu^*(T \cap E_1) + \mu^*(T \cap E_1^c \cap E_2)$. Thus
$$
\mu^*(T) \geq \mu^*(T \cap (E_1 \cup E_2)) + \mu^*(T \cap (E_1 \cup E_2)^c),
$$
which with $\leq$ from subadditivity gives equality. So $E_1 \cup E_2 \in \mathcal M$.

**(a), Step 2 — $\mathcal M$ is closed under countable disjoint unions.**

Let $E_1, E_2, \ldots \in \mathcal M$ disjoint, $E := \bigsqcup E_n$. Finite disjoint union induction from Step 1 gives $F_N := \bigsqcup_{n=1}^N E_n \in \mathcal M$.

**Sub-claim (finite additivity on $\mathcal M$):** For disjoint $E, F \in \mathcal M$ and any $T$, $\mu^*(T \cap (E \sqcup F)) = \mu^*(T \cap E) + \mu^*(T \cap F)$. [Apply the measurability of $E$ to test set $T \cap (E \sqcup F)$: $\mu^*(T \cap (E \sqcup F)) = \mu^*(T \cap (E \sqcup F) \cap E) + \mu^*(T \cap (E \sqcup F) \cap E^c) = \mu^*(T \cap E) + \mu^*(T \cap F)$.] Iterating, for $F_N = \bigsqcup_{n \leq N} E_n$:
$$
\mu^*(T \cap F_N) = \sum_{n=1}^N \mu^*(T \cap E_n).
$$

Apply $F_N$-measurability: $\mu^*(T) = \mu^*(T \cap F_N) + \mu^*(T \cap F_N^c) \geq \sum_{n \leq N} \mu^*(T \cap E_n) + \mu^*(T \cap E^c)$ (using $F_N \subseteq E$, so $F_N^c \supseteq E^c$, hence $\mu^*(T \cap F_N^c) \geq \mu^*(T \cap E^c)$). Let $N \to \infty$:
$$
\mu^*(T) \geq \sum_n \mu^*(T \cap E_n) + \mu^*(T \cap E^c) \geq \mu^*(T \cap E) + \mu^*(T \cap E^c),
$$
using subadditivity $\sum_n \mu^*(T \cap E_n) \geq \mu^*(T \cap E)$. By subadditivity in the other direction, equality holds, so $E \in \mathcal M$.

**Countable (not necessarily disjoint) unions:** Given $E_1, E_2, \ldots \in \mathcal M$, let $\tilde E_n := E_n \setminus \bigcup_{k < n} E_k = E_n \cap (E_1 \cup \ldots \cup E_{n-1})^c \in \mathcal M$ (by algebra closure); $\tilde E_n$ disjoint, $\bigcup \tilde E_n = \bigcup E_n$. Apply disjoint union case.

**(b) — $\mu := \mu^*|_{\mathcal M}$ is a measure.** Take $T = E = \bigsqcup E_n$ in the above: $\mu^*(E) \geq \sum_n \mu^*(E_n) + \mu^*(E \cap E^c) = \sum \mu^*(E_n) + 0$. Combined with subadditivity, $\mu^*(E) = \sum \mu^*(E_n)$. So $\mu$ is countably additive on $\mathcal M$, and $\mu(\emptyset) = \mu^*(\emptyset) = 0$.

**(c) — Completeness.** If $N \subseteq E$ with $\mu^*(E) = 0$, then $\mu^*(N) \leq \mu^*(E) = 0$ by monotonicity, so $\mu^*(N) = 0$. For any $T$: $\mu^*(T \cap N) + \mu^*(T \cap N^c) \leq \mu^*(N) + \mu^*(T) = \mu^*(T)$. Combined with subadditivity, equality holds, so $N \in \mathcal M$. $\blacksquare$

**Theorem 1.1.3.5 (Extension of a pre-measure).** Let $\mu_0$ be a pre-measure on an algebra $\mathcal A$, and let $\mu^*$ be the induced outer measure. Then:

(a) $\mathcal A \subseteq \mathcal M$, the σ-algebra of $\mu^*$-measurable sets.

(b) $\mu^*|_{\mathcal A} = \mu_0$.

(c) $\mu^*$ restricted to $\sigma(\mathcal A)$ is a measure extending $\mu_0$.

(d) **Uniqueness of extension when $\mu_0$ is σ-finite:** if $\mu_0$ is σ-finite on $\mathcal A$, then there is at most one measure on $\sigma(\mathcal A)$ extending $\mu_0$.

*Proof sketch.*
**(a):** Let $E \in \mathcal A$. Given $T \subseteq \Omega$ and cover $\{A_n\} \subseteq \mathcal A$ of $T$ with $\sum \mu_0(A_n) \leq \mu^*(T) + \epsilon$: each $A_n = (A_n \cap E) \sqcup (A_n \cap E^c)$, disjoint, both in $\mathcal A$. Additivity of $\mu_0$: $\mu_0(A_n) = \mu_0(A_n \cap E) + \mu_0(A_n \cap E^c)$. The former is a cover of $T \cap E$ by $\mathcal A$-sets, with sum $\leq \mu^*(T) + \epsilon$; similarly for $E^c$. Take infimum, letting $\epsilon \to 0$: $\mu^*(T \cap E) + \mu^*(T \cap E^c) \leq \mu^*(T)$, so $E \in \mathcal M$.

**(b):** For $A \in \mathcal A$: $\mu^*(A) \leq \mu_0(A)$ using trivial cover $\{A\}$. Reverse: for any cover $\{A_n\} \subseteq \mathcal A$ of $A$, $A = \bigcup_n (A \cap A_n) = \bigcup_n (A_n \cap A)$ (a subset of $\bigcup A_n$); using $\mu_0 \leq \mu_0$-countable-subadditivity (follows from pre-measure axioms applied to disjointification), $\mu_0(A) \leq \sum \mu_0(A_n)$. So $\mu^*(A) \geq \mu_0(A)$.

**(c):** By (a) and Carathéodory, $\mathcal A \subseteq \mathcal M$, so $\sigma(\mathcal A) \subseteq \mathcal M$. $\mu := \mu^*|_{\mathcal M}$ is a measure, so $\mu|_{\sigma(\mathcal A)}$ is a measure, agreeing with $\mu_0$ on $\mathcal A$ by (b).

**(d):** Uniqueness. Let $\mu, \nu$ be two measures on $\sigma(\mathcal A)$ extending $\mu_0$; assume σ-finite. By σ-finiteness, $\Omega = \bigsqcup \Omega_n$ with $\Omega_n \in \mathcal A$, $\mu_0(\Omega_n) < \infty$. It suffices to show $\mu = \nu$ on $\{A \cap \Omega_n : A \in \sigma(\mathcal A)\}$ for each $n$. Fix $n$; let $\mathcal L := \{A \in \sigma(\mathcal A) : \mu(A \cap \Omega_n) = \nu(A \cap \Omega_n)\}$. $\mathcal L$ is a λ-system (the finiteness is used for subtraction), and $\mathcal A \subseteq \mathcal L$ (a π-system). By π-λ, $\sigma(\mathcal A) \subseteq \mathcal L$. $\blacksquare$

---

## Topic 1.1.4 — Borel σ-Algebra and Lebesgue Measure

### 1.1.4.1 The Borel σ-Algebra

**Definition 1.1.4.1 (Borel σ-algebra).** Let $X$ be a topological space. The **Borel σ-algebra** $\mathcal B(X)$ is $\sigma(\mathcal T)$ where $\mathcal T$ is the collection of open sets.

Equivalently, $\mathcal B(X)$ is generated by closed sets, since closure complements turn open to closed. For $X = \mathbb R$, equivalent generators include:

- All open intervals $(a, b)$, $a, b \in \mathbb R$.
- All open intervals with rational endpoints (the open sets form a basis, and every open set is a countable union of rational-endpoint basis elements).
- All half-lines $(-\infty, a]$, or $(a, \infty)$, or $[a, \infty)$, etc.
- All half-open intervals $(a, b]$, $[a, b)$, etc.

**Proposition 1.1.4.2.** The Borel σ-algebra on $\mathbb R$ has cardinality continuum $\mathfrak c = 2^{\aleph_0}$.

*Proof sketch.* Each Borel set is built by transfinite iteration from countable collections of open intervals; there are at most $(2^{\aleph_0})^{\aleph_0} = 2^{\aleph_0}$ such Borel sets. Conversely, every singleton $\{x\} = \bigcap_n (x - 1/n, x + 1/n)$ is Borel, and there are $2^{\aleph_0}$ such singletons, giving at least $2^{\aleph_0}$ Borel sets. $\blacksquare$

Compare to $|2^{\mathbb{R}}| = 2^{2^{\aleph_0}}$: the power set is *strictly larger* than the Borel σ-algebra. So **most subsets of $\mathbb R$ are not Borel**. However, concretely exhibiting a non-Borel set requires (in effect) the axiom of choice or some equivalent non-constructive principle.

### 1.1.4.2 Construction of Lebesgue Measure

**Step 1.** Let $\mathcal A$ be the algebra of **elementary sets** on $\mathbb R$: finite disjoint unions of intervals of the form $(a, b]$ (with $-\infty < a \leq b < \infty$) or $(-\infty, b]$ or $(a, \infty)$. This is an algebra (exercise: check closure under complement, finite union, finite intersection).

**Step 2.** Define $\mu_0$ on $\mathcal A$ by additivity extended from $\mu_0((a, b]) := b - a$.

**Step 3.** Verify $\mu_0$ is a pre-measure (countably additive on $\mathcal A$ for disjoint unions that remain in $\mathcal A$). The critical step: suppose $(a, b] = \bigsqcup (a_n, b_n]$ (disjoint, countable). Then $b - a = \sum (b_n - a_n)$. Proof via compactness of $[a + \epsilon, b]$: finite sub-cover of the open intervals $(a_n, b_n + \epsilon/2^n)$, finite-additivity comparison, let $\epsilon \to 0$.

**Step 4.** Apply Carathéodory: extend $\mu_0$ to an outer measure $\mu^*$ on $2^{\mathbb R}$, restrict to the σ-algebra $\mathcal L$ of $\mu^*$-measurable sets. Call the result **Lebesgue measure**, denoted $\mu$ or $m$ or $\lambda$.

**Step 5.** Verify $\mathcal B(\mathbb R) \subseteq \mathcal L$: every Borel set is Lebesgue measurable.

**Step 6.** Verify uniqueness: any translation-invariant measure on $\mathcal B(\mathbb R)$ with $\mu([0, 1]) = 1$ equals Lebesgue measure.

**Theorem 1.1.4.3 (Lebesgue measure on $\mathbb R^d$).** Analogous construction on $\mathbb R^d$ gives a measure $\mu$ on the Lebesgue σ-algebra $\mathcal L(\mathbb R^d)$ with:

- $\mu([a_1, b_1] \times \cdots \times [a_d, b_d]) = \prod (b_i - a_i)$.
- $\mu$ is translation-invariant.
- $\mu$ is the completion of the Borel measure on $\mathbb R^d$ (see Topic 1.1.5).

### 1.1.4.3 The Borel Hierarchy

Not every σ-algebra element is obtained by a "nice" countable process; however, the Borel σ-algebra admits a natural hierarchy:

- $\Sigma^0_1$ = open sets.
- $\Pi^0_1$ = closed sets.
- $\Sigma^0_2$ = countable unions of closed sets (= $F_\sigma$).
- $\Pi^0_2$ = countable intersections of open sets (= $G_\delta$).
- $\Sigma^0_3$ = $G_{\delta\sigma}$, $\Pi^0_3$ = $F_{\sigma\delta}$, etc.

This is the **Borel hierarchy**. Every Borel set lies at some level of this hierarchy — but the level can be any countable ordinal (not just finite). The union over all countable ordinals gives the Borel σ-algebra.

**Consequence.** Closed sets, $G_\delta$'s, and $F_\sigma$'s are Borel. In particular, all singletons, countable sets, compact sets are Borel.

### 1.1.4.4 Regularity of Lebesgue Measure

**Theorem 1.1.4.4 (Regularity).** For every Lebesgue-measurable set $E \subseteq \mathbb R^d$:

(a) **Outer regularity:** $\mu(E) = \inf\{\mu(U) : U \supseteq E, U \text{ open}\}$.

(b) **Inner regularity:** $\mu(E) = \sup\{\mu(K) : K \subseteq E, K \text{ compact}\}$.

*Proof idea.* Outer regularity follows from the definition of $\mu^*$ (cover $E$ by countable unions of intervals = open sets). For compact inner regularity: approximate $E$ by a closed set from inside (using complementarity: $E = (E^c)^c$, and $E^c$ can be approximated from outside by an open set), then intersect with large compact boxes. $\blacksquare$

### 1.1.4.5 Example: Cantor Set

Let $C \subseteq [0, 1]$ be the Cantor middle-thirds set: $C = [0, 1] \setminus \bigcup_{n} \text{middle thirds}$. Removed mass: $\sum_n 2^{n-1}/3^n = 1$. So $\mu(C) = 0$.

But $C$ is uncountable ($|C| = 2^{\aleph_0}$, via ternary expansions with no "1" digit). So we have an uncountable set of Lebesgue measure zero — a "measure-theoretically null but cardinality-rich" object. Cantor-like sets feature in:

- Construction of non-Borel sets (via the Cantor set's uncountable cardinality).
- Analytic continuation (strange sets where continuation fails).
- Financial models (certain fractal path properties, e.g., of Brownian motion).

**Fat Cantor sets.** By removing smaller and smaller "middle chunks" (e.g., middle $1/4^n$ instead of $1/3$), we obtain Cantor-like sets of *positive* Lebesgue measure — another counterintuitive feature of measure theory.

---

## Topic 1.1.5 — Null Sets, Complete Measures, and Completion

### 1.1.5.1 Null Sets

**Definition 1.1.5.1.** A set $N \in \mathcal F$ is **null** (w.r.t. $\mu$) if $\mu(N) = 0$.

**Properties:**
- Countable unions of null sets are null.
- Any subset of a null set is a subset — but not necessarily measurable!

**Definition 1.1.5.2 (Complete measure).** A measure space $(\Omega, \mathcal F, \mu)$ is **complete** if every subset of a null set is measurable (and hence null).

**Why completeness matters.** A function $f$ is "a.e. equal" to a measurable function means $f = g$ outside a null set $N$. Without completeness, $\{f \neq g\}$ might not be measurable, making "a.e." notions awkward.

The Carathéodory construction produces complete measures. But the Borel σ-algebra on $\mathbb R^d$ is *not complete* under Lebesgue measure: there exist subsets of Borel null sets that are not Borel.

### 1.1.5.2 Completion

**Theorem 1.1.5.3 (Completion).** Let $(\Omega, \mathcal F, \mu)$ be a measure space. Define
$$
\bar{\mathcal F} := \{E \cup N : E \in \mathcal F, N \subseteq M \text{ for some } M \in \mathcal F \text{ with } \mu(M) = 0\},
$$
and extend $\bar\mu(E \cup N) := \mu(E)$.

Then $(\Omega, \bar{\mathcal F}, \bar\mu)$ is a complete measure space, and $\bar{\mathcal F}$ is the smallest σ-algebra containing $\mathcal F$ such that the extended measure is complete.

*Proof sketch.* Check $\bar{\mathcal F}$ is a σ-algebra: closed under countable unions (union of $E_n \cup N_n$'s rearranges), closed under complement (requires care — uses $E^c = (E \cup N)^c \sqcup N' = (E \cup M)^c \cup (M \setminus N)$ or similar). Check $\bar\mu$ is well-defined (the decomposition $E \cup N$ is not unique, but $\mu(E)$ is). Completeness and minimality: if a σ-algebra $\mathcal F'$ contains $\mathcal F$ and is $\mu'$-complete for any extension $\mu'$ of $\mu$, then $\mathcal F' \supseteq \bar{\mathcal F}$ (null subsets must be included). $\blacksquare$

**Terminology.** The completion of the Borel σ-algebra w.r.t. Lebesgue measure is precisely the **Lebesgue σ-algebra** $\mathcal L(\mathbb R^d)$. So Lebesgue-measurable sets = Borel sets + their null subsets. This is why we distinguish "Borel measurable" from "Lebesgue measurable" — the latter is broader but the former is canonical (depends only on the topology, not on the measure).

### 1.1.5.3 "Almost Everywhere" Statements

A property $P(\omega)$ holds **almost everywhere** (a.e.) if $\{\omega : P(\omega) \text{ fails}\}$ is null.

**Example.** $f = g$ a.e. means $\{\omega : f(\omega) \neq g(\omega)\}$ has measure $0$. On a complete measure space, $\{f \neq g\}$ is automatically measurable whenever $f, g$ are.

In probability theory, "almost surely" (a.s.) is synonymous with "almost everywhere" — "with probability $1$."

---

## Topic 1.1.6 — Non-Measurable Sets

### 1.1.6.1 The Vitali Construction

Not every subset of $\mathbb R$ is Lebesgue measurable. Here is the classical construction showing so:

**Theorem 1.1.6.1 (Vitali, 1905).** There is a subset $V \subseteq [0, 1]$ that is not Lebesgue measurable.

*Proof.* Define an equivalence relation on $[0, 1]$: $x \sim y \iff x - y \in \mathbb Q$. The equivalence classes partition $[0, 1]$ into uncountably many classes, each countable.

By the **axiom of choice**, select a representative $v$ from each class; let $V$ be the set of all representatives.

**Claim:** $V$ is not measurable.

For rational $q \in (-1, 1)$, let $V_q := V + q$. The $V_q$ are pairwise disjoint: if $v_1 + q_1 = v_2 + q_2$ then $v_1 - v_2 = q_2 - q_1 \in \mathbb Q$, so $v_1 \sim v_2$, so $v_1 = v_2$ (same representative).

Also, every $x \in [0, 1]$ is $x = v + q$ for some $v \in V$ (its representative) and $q = x - v \in \mathbb Q$. If $v \in [0, 1]$ and $x \in [0, 1]$, then $q \in [-1, 1] \cap \mathbb Q$. So
$$
[0, 1] \subseteq \bigcup_{q \in \mathbb Q \cap [-1, 1]} V_q \subseteq [-1, 2].
$$

If $V$ were measurable with $\mu(V) = c$, then by translation invariance $\mu(V_q) = c$ for all $q$; by countable additivity,
$$
1 = \mu([0, 1]) \leq \sum_{q \in \mathbb Q \cap [-1, 1]} \mu(V_q) = \sum_{q} c \leq \mu([-1, 2]) = 3.
$$
The sum of countably many copies of $c$ is either $0$ (if $c = 0$) or $\infty$ (if $c > 0$). Neither $0 \leq 1$ nor $\infty \leq 3$ is a contradiction... wait, $0 \leq 1$ is fine, so $c = 0$ would mean $\mu([0, 1]) \leq 0$, contradiction; $c > 0$ would mean $\infty \leq 3$, contradiction. Either way contradiction. $\blacksquare$

**Remark.** The proof uses the axiom of choice essentially. Solovay (1970) showed that without AC (in a model with different set-theoretic foundations), "every subset of $\mathbb R$ is Lebesgue measurable" can be consistent. So the existence of non-measurable sets is a *choice-dependent phenomenon*.

### 1.1.6.2 Consequences

The Vitali construction shows:

1. **No translation-invariant measure on all of $2^{\mathbb R}$** with $\mu([0, 1]) = 1$ exists (extending Lebesgue measure to all subsets is impossible while keeping countable additivity).

2. **The Banach-Tarski paradox** (which we won't prove): in $\mathbb R^3$, a ball can be decomposed into finitely many pieces, rearranged via isometries, to form two balls. This relies on AC + the structure of $SO(3)$. The pieces are necessarily non-measurable.

3. **Practical implication:** measure theory cannot just be "define measures on $2^\Omega$." We need the σ-algebra to be a proper subclass, and the Carathéodory construction gives us the right subclass.

### 1.1.6.3 Python: Constructing Lebesgue Measure (conceptually)

```python
"""
Conceptual demonstration of Lebesgue outer measure via interval covering.
We approximate μ*(E) = inf{Σ (b_n - a_n) : E ⊆ ∪(a_n, b_n]}.

Naturally we can't compute this for arbitrary E; but for simple sets
like unions of intervals, the approximation is exact.
"""
import numpy as np

def outer_measure_of_interval_union(intervals, tol=1e-9):
    """Compute Lebesgue measure of a finite union of intervals [a_i, b_i].
    
    Uses the sweep-line algorithm: sort endpoints, merge overlapping."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a > cur_b + tol:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    total += cur_b - cur_a
    return total

# Examples:
print(outer_measure_of_interval_union([(0, 1)]))            # 1.0
print(outer_measure_of_interval_union([(0, 1), (2, 3)]))    # 2.0
print(outer_measure_of_interval_union([(0, 1), (0.5, 2)]))  # 2.0 (merged: (0, 2))

# Cantor set approximation: remove middle thirds iteratively.
def cantor_approximation(n_levels):
    """Generate the intervals comprising the n-th approximation of the Cantor set."""
    intervals = [(0.0, 1.0)]
    for _ in range(n_levels):
        new_intervals = []
        for (a, b) in intervals:
            w = (b - a) / 3
            new_intervals.append((a, a + w))
            new_intervals.append((b - w, b))
        intervals = new_intervals
    return intervals

for n in [0, 1, 2, 3, 5, 10]:
    cs = cantor_approximation(n)
    m = outer_measure_of_interval_union(cs)
    print(f"Cantor level {n}: {len(cs)} intervals, measure = {m:.6f}, expected (2/3)^n = {(2/3)**n:.6f}")

# Vitali set: cannot be constructed explicitly without the axiom of choice!
# But we can illustrate the rational equivalence classes at finite precision.
import fractions
def vitali_representative_example(n_rationals=50):
    """Pick representatives from each rational-equivalence class of a fine grid.
    This doesn't give a non-measurable set, but illustrates the coset idea."""
    grid_points = np.linspace(0, 1, 1000)
    classes = {}  # key: fractional part after subtracting minimum rational
    rationals = [fractions.Fraction(p, q) for p in range(1, 10) for q in range(1, 10)]
    # This is toy: just illustrates that equivalence classes have many points.
    return rationals

# Illustrate σ-additivity failure for indicator of rationals:
# Lebesgue measure of Q∩[0,1] is 0 because Q is countable.
# Any attempt to Riemann-integrate the indicator of Q fails — 
# Lebesgue measure gives 0, Riemann integral doesn't exist.
print("\nμ(Q ∩ [0,1]) = 0 via Lebesgue (countable subset of [0,1])")
print("The indicator of Q is not Riemann integrable but is Lebesgue integrable.")
```

---

### 1.1.7 [QUANT APPLICATION]

**1. Probability spaces.** Every quant model lives on a probability space $(\Omega, \mathcal F, \mathbb P)$. Examples:

- **Coin flip:** $\Omega = \{H, T\}^{\mathbb N}$, $\mathcal F$ generated by cylinder sets, $\mathbb P$ = product Bernoulli.
- **Black-Scholes:** $\Omega = C[0, T]$ (paths), $\mathcal F$ = Borel σ-algebra on $C[0, T]$, $\mathbb P$ = Wiener measure.
- **Jump-diffusion:** $\Omega = D[0, T]$ (càdlàg paths), $\mathcal F$ = Skorokhod σ-algebra.

Without measure theory, these setups are informal; with it, every statement about Brownian motion or jump processes is rigorously well-defined.

**2. Filtrations and the "information at time $t$"** are just sub-σ-algebras $\mathcal F_t \subseteq \mathcal F$. The σ-algebra $\mathcal F_t$ encodes all measurable events observable by time $t$. In mathematical finance, the "no insider trading" condition is "random variables must be $\mathcal F_t$-measurable to be traded at time $t$" — a direct application of σ-algebra theory.

**3. Regular conditional probabilities.** For events $A, B \in \mathcal F$ with $\mathbb P(B) > 0$, $\mathbb P(A | B) = \mathbb P(A \cap B)/\mathbb P(B)$. For conditioning on a σ-algebra $\mathcal G$, the conditional probability $\mathbb P(A | \mathcal G)$ is $\mathcal G$-measurable and constructed via Radon-Nikodym (Module 1.6). This is the mathematical object underlying "given information I know so far, what is my posterior?"

**4. Null sets in arbitrage theory.** A claim $X$ **dominates** another $Y$ if $X \geq Y$ a.s. and $X > Y$ on a set of positive measure. Arbitrage-free pricing relies crucially on the distinction between "a.s. equal" and "identically equal" — many strategies differ on null sets but are treated as identical. Without a well-defined notion of null sets (= completeness of the σ-algebra), pricing theory collapses.

**5. Change of measure.** Girsanov's theorem (Module 6.X) says: under a new probability measure $\mathbb Q \ll \mathbb P$, a Brownian motion under $\mathbb P$ becomes a Brownian motion under $\mathbb Q$ with a drift. The rigorous statement involves Radon-Nikodym derivatives (Module 1.6), which require the measure-theoretic framework of σ-algebras.

**6. Kolmogorov extension theorem.** This constructs stochastic processes from their finite-dimensional distributions. The machinery is σ-algebras, product measures, and projective limits. Brownian motion, Lévy processes, and Gaussian processes are all constructed via this theorem — Subject 2.

---

### 1.1.8 Exercises

#### ★ (Warm-up)

**E1.1.1.** Which of the following is a σ-algebra on $\mathbb N$?

(a) $\{A \subseteq \mathbb N : A \text{ finite or } A^c \text{ finite}\}$.
(b) $\{A \subseteq \mathbb N : A \text{ countable or } A^c \text{ countable}\}$.
(c) $\{A \subseteq \mathbb N : A \cap \{1, 2, 3\} \in \{\emptyset, \{1\}\}\}$.
(d) $2^{\mathbb N}$.

**E1.1.2.** Prove that an arbitrary intersection of σ-algebras is a σ-algebra, but an arbitrary intersection of **generating** sets is not necessarily one.

**E1.1.3.** Show that $\sigma(\{(a, b) : a < b \in \mathbb Q\}) = \mathcal B(\mathbb R)$.

**E1.1.4.** Compute $\mu([0, 1] \cap \mathbb Q)$ using Lebesgue measure. (Hint: $\mathbb Q$ is countable.)

**E1.1.5.** Show that $\mu([0, 1]) = 1$ for Lebesgue measure, directly from the construction.

**E1.1.6.** Is the counting measure on $\mathbb R$ σ-finite? What about the Dirac measure?

**E1.1.7.** Describe the smallest σ-algebra containing the singletons in $\mathbb R$.

#### ★★ (Standard)

**E1.1.8.** Prove: every σ-algebra $\mathcal F$ on a set $\Omega$ is either finite or uncountable. (Hint: if $|\mathcal F| = \aleph_0$, pick a countable "atomization" partition and derive a contradiction.)

**E1.1.9.** Let $\mathcal A$ be the algebra of finite unions of half-open intervals in $\mathbb R$. Show $\sigma(\mathcal A) = \mathcal B(\mathbb R)$.

**E1.1.10.** Describe the Borel σ-algebra on $\mathbb Q$ (with the subspace topology). Show it equals $2^{\mathbb Q}$.

**E1.1.11 (Continuity from above without finiteness fails).** Give an example in Lebesgue measure on $\mathbb R$ of $A_n \downarrow A$ with $\mu(A_n) = \infty$ for all $n$ and $\mu(A) = 0$.

**E1.1.12 (Fat Cantor).** Construct a Cantor-like set in $[0, 1]$ of Lebesgue measure $1/2$ by removing middle intervals of length $2^{-n-1}$ at level $n$.

**E1.1.13 (Borel is not complete).** Exhibit (assuming the existence of a non-Borel subset of the Cantor set) a Lebesgue-null subset of $\mathbb R$ that is not Borel. Why is the Cantor set the right starting point?

**E1.1.14 (Outer regularity of Lebesgue measure).** Prove: for any Lebesgue-measurable $E$, $\mu(E) = \inf\{\mu(U) : U \supseteq E, U \text{ open}\}$.

**E1.1.15 (Borel ⊃ $G_\delta$'s, $F_\sigma$'s).** Prove every Borel set in $\mathbb R$ is of the form $G_\delta$ or $F_\sigma$ or a more complex construction. In fact, produce an example of a $G_{\delta\sigma}$ set that is not $G_\delta$.

**E1.1.16 (π-λ in practice).** Let $\mu, \nu$ be finite measures on $\mathcal B(\mathbb R)$ with $\mu((-\infty, t]) = \nu((-\infty, t])$ for all $t \in \mathbb R$. Show $\mu = \nu$.

#### ★★★ (Challenge)

**E1.1.17 (Non-measurable set, assuming AC).** Complete the details of the Vitali construction: describe the equivalence classes, the selection, and the measure contradiction.

**E1.1.18 (Borel hierarchy is strict).** Show that there is a Borel set that is $\Sigma^0_3$ but not $\Sigma^0_2$ (i.e., $G_{\delta\sigma}$ but not $F_\sigma$). (Hint: Baire category arguments; see Kechris, *Classical Descriptive Set Theory*.)

**E1.1.19 (Invariance of Lebesgue measure).** Prove that Lebesgue measure is invariant under rotations and reflections as well as translations. Derive that it is invariant under all isometries of $\mathbb R^d$.

**E1.1.20 (Haar measure existence).** State and sketch a proof of the existence of Haar measure on a locally compact topological group.

**E1.1.21 (Kolmogorov extension).** State the Kolmogorov extension theorem for stochastic processes. Sketch how it is used to construct Brownian motion.

**E1.1.22 (Banach-Tarski sketch).** Describe the paradoxical decomposition of a ball in $\mathbb R^3$ using the free subgroup of $SO(3)$. What step uses AC? Why does this not extend to $\mathbb R^2$?

---

## Module 1.1 Summary

### Key Theorems

1. **Intersection of σ-algebras is a σ-algebra** → generated σ-algebra exists.
2. **π-λ theorem** → checking equality of measures on a π-system suffices to conclude on $\sigma(\mathcal P)$.
3. **Carathéodory extension theorem** → a pre-measure on an algebra extends to a (complete) measure on the σ-algebra of Carathéodory-measurable sets.
4. **Uniqueness of σ-finite extension** → Lebesgue measure is the unique translation-invariant Borel measure with $\mu([0,1]) = 1$.
5. **Regularity of Lebesgue measure** → outer regularity via open sets, inner regularity via compact sets.
6. **Existence of non-measurable sets (Vitali)** → requires AC; shows $\mu$ cannot be extended translation-invariantly to all of $2^{\mathbb R}$.

### Forward Pointers

- **Module 1.2 (Measurable functions)** will build on σ-algebras to define measurable functions, the carriers of probability theory (random variables).
- **Module 1.3 (Lebesgue integration)** defines integrals $\int f \, d\mu$ against a measure — the generalization of the Riemann integral that plays well with limits.
- **Module 1.4 (Product measures)** extends to $\Omega_1 \times \Omega_2$ and proves Fubini's theorem.
- **Module 1.5 ($L^p$ spaces)** shows the completeness that makes Hilbert-space methods available.
- **Module 1.6 (Radon-Nikodym)** treats absolute continuity and change of measure — the mathematical tool underlying Girsanov's theorem.

### The Big Picture

The construction so far can be summarized:
```
Algebra  --(pre-measure)-->  Algebra  
   |                            |
   | σ(·)                       | Carathéodory
   ↓                            ↓
σ-algebra  <----------- μ*-measurable σ-algebra
                                 |
                             (complete)
```

Lebesgue measure is the completion of the Borel measure on $\mathbb R$. This paradigm — define on an algebra, extend to σ-algebra, complete — is the template for constructing virtually all measures in analysis (Hausdorff measures, Haar measures, Wiener measure via Kolmogorov).

**Next module:** Measurable functions and random variables.

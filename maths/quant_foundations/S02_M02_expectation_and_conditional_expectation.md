# Module 2.2 — Expectation and Conditional Expectation

> *"Conditional expectation is the best $L^2$-predictor of $X$ given $\mathcal G$."* — the Hilbert-space characterization.

## Prerequisites

- **Modules 1.3-1.6** — Lebesgue integral, $L^p$ spaces, Radon-Nikodym.
- **Module 2.1** — probability spaces, random variables, independence.

## Overview

Expectation is the Lebesgue integral on a probability space. Variance, covariance, and higher moments are derived quantities. The foundational inequalities of probability (Markov, Chebyshev, Jensen, Cauchy-Schwarz, Hölder) are direct specializations of the $L^p$ results in Module 1.5. This module focuses on:

1. **Moment inequalities** — the quant-trader's bread and butter for tail bounds.
2. **Conditional expectation** $\mathbb E[X | \mathcal G]$ — the central construction of modern probability and finance.
3. **Jensen's inequality** — convexity meets expectation; the theoretical basis of risk aversion and utility.

Conditional expectation is the most important object in Subject 2. Every martingale is defined via conditional expectations; every no-arbitrage pricing formula is an expectation against a risk-neutral measure; every Bayesian estimator is a conditional expectation. This module develops its theory rigorously via Radon-Nikodym (Module 1.6) and verifies all the standard properties.

---

## Topic 2.2.1 — Expectation

### Definition 2.2.1 (Expectation)

For a random variable $X$ on $(\Omega, \mathcal F, \mathbb P)$, the **expectation** (or **expected value**, or **mean**) is the Lebesgue integral
$$\mathbb E X := \int_\Omega X \, d\mathbb P,$$
defined whenever at least one of $\mathbb E X^+, \mathbb E X^-$ is finite. $X$ is **integrable** if $\mathbb E|X| < \infty$, written $X \in L^1(\mathbb P)$.

By LOTUS (Proposition 2.1.14),
$$\mathbb E g(X) = \int_\mathbb R g(x) \, d\mu_X(x) = \begin{cases} \sum_k g(x_k) \mathbb P(X = x_k) & \text{if } X \text{ discrete} \\ \int g(x) f_X(x) \, dx & \text{if } X \text{ has density } f_X. \end{cases}$$

### Proposition 2.2.2 (Properties of expectation)

For $X, Y \in L^1$ and $\alpha, \beta \in \mathbb R$:
1. **Linearity**: $\mathbb E(\alpha X + \beta Y) = \alpha \mathbb E X + \beta \mathbb E Y$.
2. **Monotonicity**: $X \le Y$ a.s. $\Rightarrow \mathbb E X \le \mathbb E Y$; equality iff $X = Y$ a.s.
3. **Triangle**: $|\mathbb E X| \le \mathbb E |X|$.
4. **$\mathbb E \mathbf 1_A = \mathbb P(A)$**.
5. **Fatou, MCT, DCT** (from Module 1.3) apply directly.

Proofs are immediate from Module 1.3.

### Definition 2.2.3 (Variance and moments)

For $X$ with $\mathbb E X^2 < \infty$:
$$\operatorname{Var}(X) := \mathbb E[(X - \mathbb E X)^2] = \mathbb E X^2 - (\mathbb E X)^2.$$
Standard deviation: $\sigma(X) := \sqrt{\operatorname{Var}(X)}$.

**$p$-th moment**: $\mathbb E X^p$ (if exists). **Central $p$-th moment**: $\mathbb E[(X - \mathbb E X)^p]$.

**Skewness**: $\mathbb E[(X - \mathbb E X)^3]/\sigma^3$. **Kurtosis**: $\mathbb E[(X - \mathbb E X)^4]/\sigma^4$ (excess = kurtosis $- 3$).

### Definition 2.2.4 (Covariance and correlation)

For $X, Y \in L^2$:
$$\operatorname{Cov}(X, Y) := \mathbb E[(X - \mathbb E X)(Y - \mathbb E Y)] = \mathbb E XY - \mathbb E X \mathbb E Y.$$
Correlation: $\rho(X, Y) := \operatorname{Cov}(X, Y)/(\sigma(X) \sigma(Y)) \in [-1, 1]$.

$X, Y$ are **uncorrelated** if $\operatorname{Cov}(X, Y) = 0$. Independent $\Rightarrow$ uncorrelated (when both are $L^2$); converse fails.

### Proposition 2.2.5 (Variance of sums)

For $X_1, \ldots, X_n \in L^2$:
$$\operatorname{Var}\!\left(\sum X_i\right) = \sum_i \operatorname{Var}(X_i) + 2 \sum_{i < j} \operatorname{Cov}(X_i, X_j).$$
If $X_i$ are **pairwise uncorrelated** (e.g. independent): $\operatorname{Var}(\sum X_i) = \sum \operatorname{Var}(X_i)$.

**Proof.** Expand $(X - \mathbb E X)^2 = (\sum (X_i - \mathbb E X_i))^2$ and take expectations. $\blacksquare$

### Standard computations of moments

| Distribution | $\mathbb E X$ | $\operatorname{Var}(X)$ | Notes |
|---|---|---|---|
| Bernoulli$(p)$ | $p$ | $p(1-p)$ | |
| Binomial$(n, p)$ | $np$ | $np(1-p)$ | Sum of $n$ i.i.d. Bernoullis |
| Poisson$(\lambda)$ | $\lambda$ | $\lambda$ | |
| Geometric$(p)$ | $1/p$ | $(1-p)/p^2$ | Number of trials until first success |
| Uniform$(a, b)$ | $(a+b)/2$ | $(b-a)^2/12$ | |
| Exponential$(\lambda)$ | $1/\lambda$ | $1/\lambda^2$ | Memoryless |
| Normal$(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ | |
| Gamma$(\alpha, \beta)$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Sum of $\alpha$ (integer) Exp$(\beta)$ |
| Cauchy | undefined | undefined | No moments |
| Student-$t_\nu$ | $0$ for $\nu > 1$ | $\nu/(\nu-2)$ for $\nu > 2$ | Heavy tails |

---

## Topic 2.2.2 — Moment Inequalities

### Theorem 2.2.6 (Markov's inequality)

For $X \ge 0$ and $a > 0$: $\mathbb P(X \ge a) \le \mathbb E X / a$.

**Proof.** $a \mathbf 1_{X \ge a} \le X$, take expectations. $\blacksquare$

### Theorem 2.2.7 (Chebyshev's inequality)

For $X \in L^2$ and $a > 0$: $\mathbb P(|X - \mathbb E X| \ge a) \le \operatorname{Var}(X)/a^2$.

**Proof.** Apply Markov to $(X - \mathbb E X)^2$: $\mathbb P((X - \mathbb E X)^2 \ge a^2) \le \mathbb E(X - \mathbb E X)^2/a^2$. $\blacksquare$

### Theorem 2.2.8 (Generalized Markov / Chernoff)

For $X$ and any increasing $\varphi \ge 0$:
$$\mathbb P(X \ge a) \le \frac{\mathbb E \varphi(X)}{\varphi(a)}.$$

Taking $\varphi(x) = e^{tx}$ with $t > 0$ and optimizing:
$$\mathbb P(X \ge a) \le \inf_{t > 0} e^{-ta} M_X(t), \qquad M_X(t) := \mathbb E e^{tX}.$$

**Proof.** $\varphi(X) \ge \varphi(a) \mathbf 1_{X \ge a}$; take expectations and apply Markov. $\blacksquare$

Chernoff bounds are exponentially tight and are the main tool of concentration inequalities.

### Theorem 2.2.9 (Cauchy-Schwarz)

$|\mathbb E XY| \le \sqrt{\mathbb E X^2} \sqrt{\mathbb E Y^2}$.

**Proof.** Hölder with $p = q = 2$. $\blacksquare$

### Theorem 2.2.10 (Jensen's inequality)

For $\varphi: \mathbb R \to \mathbb R$ convex and $X \in L^1$ with $\varphi(X) \in L^1$:
$$\varphi(\mathbb E X) \le \mathbb E \varphi(X).$$

If $\varphi$ is strictly convex and $X$ is non-constant, the inequality is strict.

**Proof.** Since $\varphi$ is convex, for any $x_0$ there exists a **subdifferential** $c \in \mathbb R$ (a supporting line slope) with
$$\varphi(x) \ge \varphi(x_0) + c(x - x_0) \quad \forall x.$$

Apply with $x_0 = \mathbb E X$, $x = X$:
$$\varphi(X) \ge \varphi(\mathbb E X) + c(X - \mathbb E X) \quad \text{a.s.}$$

Take expectations: $\mathbb E \varphi(X) \ge \varphi(\mathbb E X) + c(\mathbb E X - \mathbb E X) = \varphi(\mathbb E X)$. $\blacksquare$

### Corollary 2.2.11 (Common Jensen consequences)

- $\varphi(x) = x^2$: $(\mathbb E X)^2 \le \mathbb E X^2$ (variance is non-negative).
- $\varphi(x) = |x|^p$ for $p \ge 1$: $|\mathbb E X|^p \le \mathbb E |X|^p$, so $\|X\|_1 \le \|X\|_p$ on a probability space.
- $\varphi(x) = e^x$: $e^{\mathbb E X} \le \mathbb E e^X$.
- $\varphi(x) = -\log x$ for $x > 0$: $\log(\mathbb E X) \ge \mathbb E \log X$ (AM-GM-like).

### Corollary 2.2.12 ($L^p$ inclusion on probability spaces)

For $1 \le p \le q \le \infty$: $L^q(\mathbb P) \subset L^p(\mathbb P)$ with $\|X\|_p \le \|X\|_q$.

**Proof.** Jensen on $|X|^{p/q}$ with the convex function $x \mapsto x^{q/p}$ (since $q/p \ge 1$). $\blacksquare$

This is the key *structural* consequence of $\mathbb P$ being a probability measure: moment spaces are nested. On infinite measure, $L^p$ spaces are in general incomparable.

### Worked Example 2.2.13 (Hoeffding's inequality — preview)

For $X_1, \ldots, X_n$ independent with $a_i \le X_i \le b_i$ a.s., let $S_n = \sum X_i$:
$$\mathbb P(|S_n - \mathbb E S_n| \ge t) \le 2 \exp\!\left(-\frac{2 t^2}{\sum (b_i - a_i)^2}\right).$$

**Proof sketch.** Chernoff: $\mathbb P(S_n - \mathbb E S_n \ge t) \le e^{-\lambda t} \mathbb E e^{\lambda(S_n - \mathbb E S_n)} = e^{-\lambda t} \prod_i \mathbb E e^{\lambda (X_i - \mathbb E X_i)}$. Hoeffding's lemma: $\mathbb E e^{\lambda(X_i - \mathbb E X_i)} \le e^{\lambda^2(b_i - a_i)^2/8}$. Optimize $\lambda$: $\lambda^* = 4t / \sum (b_i - a_i)^2$. $\blacksquare$

This is the workhorse of statistical learning theory and empirical process theory.

---

## Topic 2.2.3 — Conditional Expectation: Definition and Existence

### Motivation

If $X$ is a random variable and we observe a related quantity $Y$ (or more generally, the information in a sub-σ-algebra $\mathcal G$), we want to form the "best prediction" of $X$. The answer is $\mathbb E[X | \mathcal G]$, defined formally below. It is a $\mathcal G$-measurable random variable — the randomness it carries is only that present in $\mathcal G$.

### Definition 2.2.14 (Conditional expectation)

Let $X \in L^1(\Omega, \mathcal F, \mathbb P)$ and $\mathcal G \subset \mathcal F$ a sub-σ-algebra. A **conditional expectation** of $X$ given $\mathcal G$ is any $\mathcal G$-measurable random variable $Y$ such that
$$\int_A Y \, d\mathbb P = \int_A X \, d\mathbb P \quad \text{for all } A \in \mathcal G. \tag{CE}$$

### Theorem 2.2.15 (Existence and uniqueness)

For $X \in L^1$, a conditional expectation $Y = \mathbb E[X | \mathcal G]$ exists and is unique up to $\mathbb P$-null sets. If $X \in L^2$, $\mathbb E[X | \mathcal G] \in L^2$ and equals the $L^2$-orthogonal projection of $X$ onto $L^2(\mathcal G)$.

**Proof (via Radon-Nikodym).** Assume $X \ge 0$ (decompose). Define $\nu$ on $(\Omega, \mathcal G)$ by $\nu(A) := \int_A X \, d\mathbb P$ for $A \in \mathcal G$. Then $\nu$ is a finite positive measure ($\nu(\Omega) = \mathbb E X < \infty$), and $\nu \ll \mathbb P|_\mathcal G$: if $\mathbb P(A) = 0$ then $\int_A X \, d\mathbb P = 0$. By R-N (Module 1.6), there exists $Y \ge 0$ $\mathcal G$-measurable with $\nu(A) = \int_A Y \, d\mathbb P$, which is exactly (CE).

*Uniqueness.* If $Y_1, Y_2$ both satisfy (CE), then $\int_A (Y_1 - Y_2) d\mathbb P = 0$ for all $A \in \mathcal G$. Taking $A = \{Y_1 > Y_2\} \in \mathcal G$ gives $\int (Y_1 - Y_2)^+ d\mathbb P = 0$, so $Y_1 \le Y_2$ a.s. Symmetrically $Y_2 \le Y_1$. Hence $Y_1 = Y_2$ a.s.

*$L^2$ case.* For $X \in L^2$, the condition $\mathbb E[(X - Y) Z] = 0$ for all $Z \in L^2(\mathcal G)$ is the definition of orthogonal projection. Setting $Z = \mathbf 1_A$ recovers (CE). The $L^2$ orthogonal projection exists by the Hilbert space projection theorem (Module 0.4). $\blacksquare$

### Remark. Conditional expectation is defined up to null sets

We will often abuse notation and write $\mathbb E[X | \mathcal G]$ for any representative. All identities hold "a.s."; we drop the caveat when clear.

### Definition 2.2.16 (Conditioning on random variables and events)

- **$\mathbb E[X | Y]$** := $\mathbb E[X | \sigma(Y)]$.
- **$\mathbb E[X | Y = y]$**: a function of $y$; rigorously defined via regular conditional probabilities (Module 2.3/2.7).
- **$\mathbb P(A | B)$** for $\mathbb P(B) > 0$: elementary definition $= \mathbb P(A \cap B)/\mathbb P(B)$.
- **$\mathbb P(A | \mathcal G)$** := $\mathbb E[\mathbf 1_A | \mathcal G]$, a $\mathcal G$-measurable random variable.

---

## Topic 2.2.4 — Properties of Conditional Expectation

Throughout, $X, Y \in L^1$ and $\mathcal G, \mathcal H$ are sub-σ-algebras. All equalities are a.s.

### Proposition 2.2.17 (Basic properties)

1. **Linearity**: $\mathbb E[aX + bY | \mathcal G] = a \mathbb E[X | \mathcal G] + b \mathbb E[Y | \mathcal G]$.
2. **Monotonicity**: $X \le Y \Rightarrow \mathbb E[X | \mathcal G] \le \mathbb E[Y | \mathcal G]$.
3. **Trivial σ-algebras**:
   - $\mathbb E[X | \{\emptyset, \Omega\}] = \mathbb E X$.
   - If $X$ is $\mathcal G$-measurable, $\mathbb E[X | \mathcal G] = X$.
4. **Total expectation**: $\mathbb E[\mathbb E[X | \mathcal G]] = \mathbb E X$. (Take $A = \Omega$ in (CE).)
5. **Triangle**: $|\mathbb E[X | \mathcal G]| \le \mathbb E[|X| \; | \mathcal G]$.
6. **MCT, Fatou, DCT for conditional expectation**: direct analogs hold.

*Proofs.* Items 1-5 are immediate from (CE) and uniqueness. Items 6 are verified by applying the full MCT/Fatou/DCT inside the defining equality (CE). $\blacksquare$

### Theorem 2.2.18 (Tower property)

If $\mathcal H \subset \mathcal G \subset \mathcal F$:
$$\mathbb E[\mathbb E[X | \mathcal G] | \mathcal H] = \mathbb E[X | \mathcal H].$$

**Proof.** Both sides are $\mathcal H$-measurable. For any $A \in \mathcal H \subset \mathcal G$,
$$\int_A \mathbb E[\mathbb E[X | \mathcal G] | \mathcal H] d\mathbb P = \int_A \mathbb E[X | \mathcal G] d\mathbb P \quad (\text{by (CE) for } \mathbb E[X | \mathcal G])$$
$$= \int_A X \, d\mathbb P \quad (\text{by (CE) for } \mathbb E[X | \mathcal G], \text{ using } A \in \mathcal G)$$
$$= \int_A \mathbb E[X | \mathcal H] d\mathbb P.$$
Hence both sides equal $\mathbb E[X | \mathcal H]$ by uniqueness. $\blacksquare$

### Theorem 2.2.19 (Pulling out known factors)

If $Z$ is $\mathcal G$-measurable and $ZX \in L^1$:
$$\mathbb E[ZX | \mathcal G] = Z \cdot \mathbb E[X | \mathcal G].$$

**Proof.** *Step 1: $Z$ indicator.* $Z = \mathbf 1_B, B \in \mathcal G$. For any $A \in \mathcal G$, $A \cap B \in \mathcal G$:
$$\int_A \mathbf 1_B X \, d\mathbb P = \int_{A \cap B} X \, d\mathbb P = \int_{A \cap B} \mathbb E[X | \mathcal G] d\mathbb P = \int_A \mathbf 1_B \mathbb E[X | \mathcal G] d\mathbb P.$$
By uniqueness, $\mathbb E[\mathbf 1_B X | \mathcal G] = \mathbf 1_B \mathbb E[X | \mathcal G]$.

*Step 2: $Z$ simple $\mathcal G$-measurable.* By linearity.

*Step 3: $Z \ge 0$ $\mathcal G$-measurable (not necessarily bounded).* Approximate by simple $Z_n \uparrow Z$ (Module 1.2); apply MCT in (CE). Caveat: need $Z X \in L^1$; in that case $Z_n X \to ZX$ in $L^1$ by DCT (dominated by $|ZX|$), and both sides converge.

*Step 4: general $Z$.* Decompose $Z = Z^+ - Z^-$. $\blacksquare$

### Theorem 2.2.20 (Independence)

If $\mathcal G$ and $\sigma(X)$ are independent, $\mathbb E[X | \mathcal G] = \mathbb E X$.

**Proof.** For any $A \in \mathcal G$, $\mathbf 1_A$ and $X$ are independent, so $\mathbb E[\mathbf 1_A X] = \mathbb E \mathbf 1_A \cdot \mathbb E X = \mathbb P(A) \mathbb E X = \int_A (\mathbb E X) d\mathbb P$. So the constant $\mathbb E X$ satisfies (CE), and by uniqueness it's the conditional expectation. $\blacksquare$

### Theorem 2.2.21 (Jensen for conditional expectation)

For $\varphi$ convex and $X, \varphi(X) \in L^1$:
$$\varphi(\mathbb E[X | \mathcal G]) \le \mathbb E[\varphi(X) | \mathcal G] \quad \text{a.s.}$$

**Proof.** Subdifferential: for each $y_0$, there is $c(y_0)$ with $\varphi(x) \ge \varphi(y_0) + c(y_0)(x - y_0)$ for all $x$. Set $y_0 = \mathbb E[X | \mathcal G]$, a $\mathcal G$-measurable r.v. Then $c(y_0)$ can be chosen $\mathcal G$-measurable (take an appropriate measurable selection of subdifferentials; for $\varphi$ convex and finite on $\mathbb R$, any Borel selection of a subdifferential works).
$$\varphi(X) \ge \varphi(\mathbb E[X | \mathcal G]) + c(\mathbb E[X | \mathcal G])(X - \mathbb E[X | \mathcal G]).$$
Take $\mathbb E[\cdot | \mathcal G]$ on both sides, using pull-out (Theorem 2.2.19) on the $\mathcal G$-measurable factor $c(\mathbb E[X | \mathcal G])$:
$$\mathbb E[\varphi(X) | \mathcal G] \ge \varphi(\mathbb E[X | \mathcal G]) + c(\mathbb E[X | \mathcal G]) \cdot (\mathbb E[X | \mathcal G] - \mathbb E[X | \mathcal G]) = \varphi(\mathbb E[X | \mathcal G]). \qquad \blacksquare$$

### Corollary 2.2.22 ($L^p$ contraction)

For $p \ge 1$, $\|\mathbb E[X | \mathcal G]\|_p \le \|X\|_p$.

**Proof.** Jensen with $\varphi(x) = |x|^p$: $|\mathbb E[X | \mathcal G]|^p \le \mathbb E[|X|^p | \mathcal G]$. Take full expectation: $\mathbb E |\mathbb E[X | \mathcal G]|^p \le \mathbb E |X|^p$. Take $p$-th root. $\blacksquare$

So conditional expectation is a bounded linear operator on $L^p$ with norm $\le 1$ (a **contraction**). On $L^2$ it is an orthogonal projection.

### Theorem 2.2.23 (Conditional MCT, Fatou, DCT)

- **MCT**: If $0 \le X_n \uparrow X$ a.s., $\mathbb E[X_n | \mathcal G] \uparrow \mathbb E[X | \mathcal G]$ a.s.
- **Fatou**: If $X_n \ge 0$, $\mathbb E[\liminf X_n | \mathcal G] \le \liminf \mathbb E[X_n | \mathcal G]$ a.s.
- **DCT**: If $X_n \to X$ a.s., $|X_n| \le Y$ with $\mathbb E Y < \infty$, $\mathbb E[X_n | \mathcal G] \to \mathbb E[X | \mathcal G]$ a.s. and in $L^1$.

**Proof.** MCT: let $Y_n = \mathbb E[X_n | \mathcal G]$, monotone increasing a.s. (by monotonicity 2.2.17.2). Let $Y = \lim Y_n$; then $Y$ is $\mathcal G$-measurable. For $A \in \mathcal G$, by MCT (Module 1.3),
$$\int_A Y \, d\mathbb P = \lim \int_A Y_n \, d\mathbb P = \lim \int_A X_n \, d\mathbb P = \int_A X \, d\mathbb P.$$
Uniqueness: $Y = \mathbb E[X | \mathcal G]$.

Fatou and DCT follow analogously using the unconditional versions. $\blacksquare$

---

## Topic 2.2.5 — Conditional Variance and Law of Total Variance

### Definition 2.2.24 (Conditional variance)

For $X \in L^2$: $\operatorname{Var}(X | \mathcal G) := \mathbb E[(X - \mathbb E[X | \mathcal G])^2 | \mathcal G]$.

### Theorem 2.2.25 (Law of total variance / Pythagorean decomposition)

$$\operatorname{Var}(X) = \mathbb E[\operatorname{Var}(X | \mathcal G)] + \operatorname{Var}(\mathbb E[X | \mathcal G]).$$

**Proof.** Expand:
$$\operatorname{Var}(X) = \mathbb E X^2 - (\mathbb E X)^2 = \mathbb E \mathbb E[X^2 | \mathcal G] - (\mathbb E X)^2.$$
Now $\mathbb E[X^2 | \mathcal G] = \operatorname{Var}(X | \mathcal G) + \mathbb E[X | \mathcal G]^2$ (by the unconditional variance formula, applied conditionally). So
$$\mathbb E X^2 = \mathbb E \operatorname{Var}(X | \mathcal G) + \mathbb E \mathbb E[X | \mathcal G]^2.$$
Subtract $(\mathbb E X)^2 = (\mathbb E \mathbb E[X | \mathcal G])^2$:
$$\operatorname{Var}(X) = \mathbb E \operatorname{Var}(X | \mathcal G) + (\mathbb E \mathbb E[X | \mathcal G]^2 - (\mathbb E \mathbb E[X | \mathcal G])^2) = \mathbb E \operatorname{Var}(X | \mathcal G) + \operatorname{Var}(\mathbb E[X | \mathcal G]). \qquad \blacksquare$$

**Interpretation.** The unconditional variance of $X$ decomposes into:
- $\mathbb E[\operatorname{Var}(X | \mathcal G)]$: the *residual* variance not explained by $\mathcal G$.
- $\operatorname{Var}(\mathbb E[X | \mathcal G])$: the *explained* variance, i.e. the variation in the conditional mean.

This is the ANOVA identity, the foundation of regression's $R^2$.

### Corollary 2.2.26 (Variance reduction)

$\operatorname{Var}(\mathbb E[X | \mathcal G]) \le \operatorname{Var}(X)$: conditioning reduces variance (projection onto a smaller space).

---

## Topic 2.2.6 — Explicit Computations

### Example 2.2.27 (Discrete case: conditional mass)

If $(X, Y)$ is discrete with joint mass $p(x, y)$, then
$$\mathbb E[X | Y = y] = \frac{\sum_x x p(x, y)}{\mathbb P(Y = y)} = \sum_x x \cdot p(x | y),$$
where $p(x | y) = p(x, y)/p_Y(y)$.

So $\mathbb E[X | Y](\omega) = g(Y(\omega))$ for $g(y) = \sum_x x p(x | y)$.

### Example 2.2.28 (Continuous case: conditional density)

If $(X, Y)$ has joint density $f(x, y)$, then
$$f_{X | Y}(x | y) = \frac{f(x, y)}{f_Y(y)}, \qquad \mathbb E[X | Y = y] = \int x f_{X | Y}(x | y) dx.$$

### Example 2.2.29 (Bivariate normal)

$(X, Y) \sim N_2(\mu_X, \mu_Y, \sigma_X^2, \sigma_Y^2, \rho)$. Then
$$\mathbb E[X | Y = y] = \mu_X + \rho \frac{\sigma_X}{\sigma_Y}(y - \mu_Y), \qquad \operatorname{Var}(X | Y = y) = \sigma_X^2(1 - \rho^2).$$

The conditional mean is *linear* in $y$; the conditional variance is *constant* (doesn't depend on $y$). These properties characterize the Gaussian family among all joint distributions with the same marginals and correlation (up to technical conditions).

### Example 2.2.30 (Conditional expectation with information flow — filtrations)

Let $X_1, X_2, \ldots$ be i.i.d. with $\mathbb E X_i = 0$ and $\operatorname{Var} X_i = 1$. Let $S_n = \sum_{k \le n} X_k$ and $\mathcal F_n = \sigma(X_1, \ldots, X_n)$. Then for $m \le n$:
$$\mathbb E[S_n | \mathcal F_m] = S_m + \mathbb E[X_{m+1} + \ldots + X_n | \mathcal F_m] = S_m + 0 = S_m.$$
So $(S_n, \mathcal F_n)$ is a **martingale** (Module 2.6). This is the starting point of martingale theory.

### Example 2.2.31 (Black-Scholes as a conditional expectation)

Under the risk-neutral measure $\mathbb Q$, the time-$t$ price of a call option is
$$V_t = \mathbb E^{\mathbb Q}[e^{-r(T-t)}(S_T - K)^+ | \mathcal F_t].$$
By the Markov property of geometric Brownian motion, this is a function $V_t = C(t, S_t)$ — the Black-Scholes PDE solution. The conditional expectation reduces to a deterministic function of the current price.

---

## Topic 2.2.7 — Python: Conditional Expectation and ANOVA

```python
"""
Estimating conditional expectations via regression and Monte Carlo.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- Linear regression: E[Y | X] = alpha + beta X for bivariate normal ----------
N = 10**4
rho = 0.6
X = rng.standard_normal(N)
Y = rho * X + np.sqrt(1 - rho**2) * rng.standard_normal(N)
# Conditional expectation E[Y | X = x] = rho * x for standard bivariate normal
# Empirical: regress Y on X
beta = np.cov(X, Y)[0, 1] / np.var(X)
print(f"Estimated beta: {beta:.4f}, theoretical rho: {rho}")
# Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
emp_VarY = np.var(Y)
VarEY_X = np.var(rho * X)  # theoretical
EVarY_X = 1 - rho**2  # Var(Y | X) = 1 - rho^2 (constant for bivariate normal)
print(f"Var(Y) = {emp_VarY:.4f}, Var(E[Y|X]) + E[Var(Y|X)] = {VarEY_X + EVarY_X:.4f}")
# These should be close to 1.

# ---------- Tower property: E[E[X | Y] | Z] = E[X | Z] for Z = sigma(g(Y)) subset sigma(Y) ----------
# Let X = Y^2 + noise, Y ~ N(0, 1), Z = sign(Y)
Y = rng.standard_normal(N)
noise = rng.standard_normal(N) * 0.5
X = Y**2 + noise
# E[X | Y] = Y^2 (in expectation)
# E[X | Z = sign(Y)] = E[Y^2 + noise | sign(Y)] = E[Y^2] = 1 (by symmetry)
emp_E_X_given_positive = X[Y > 0].mean()
emp_E_X_given_negative = X[Y < 0].mean()
print(f"E[X | Y > 0] = {emp_E_X_given_positive:.4f}, E[X | Y < 0] = {emp_E_X_given_negative:.4f}, both should be ~1")

# ---------- Jensen's inequality: E[e^X] > e^{E[X]} for X random ----------
X = rng.standard_normal(N)
print(f"E[e^X] = {np.mean(np.exp(X)):.4f}")
print(f"e^{{E[X]}} = {np.exp(np.mean(X)):.4f}")
# The first should be e^{1/2} ≈ 1.648 (MGF of N(0,1) at t=1), second ≈ 1.

# ---------- Conditional MC: variance reduction via conditioning ----------
# E[f(X, Y)] = E[E[f(X, Y) | Y]]. If we can compute the inner E analytically, var reduces.
# Example: E[X Y] where X, Y independent, X ~ N(0, 1), Y ~ Exp(1).
# Direct MC: estimate from samples of X, Y.
N_MC = 10**4
X_sim = rng.standard_normal(N_MC)
Y_sim = rng.exponential(1, N_MC)
direct = np.mean(X_sim * Y_sim)
print(f"Direct MC: E[XY] ≈ {direct:.4f}, SE ≈ {np.std(X_sim * Y_sim) / np.sqrt(N_MC):.4f}")

# Conditional: E[XY | Y] = Y E[X] = 0, so E[XY] = 0. Zero variance!
print(f"Conditional MC: E[XY] = 0 exactly (using E[X | Y] = E[X] = 0).")

# ---------- Markov & Chebyshev tightness ----------
# X = 0 with prob 1 - 1/10, 10 with prob 1/10. Mean = 1, Var = 10 - 1 = 9.
# Markov: P(X >= 10) <= E[X]/10 = 1/10. Tight: P(X >= 10) = 1/10.
# Chebyshev: P(|X - 1| >= 9) <= 9/81 = 1/9. Not tight: actual = 1/10.
p = 0.1
samples = rng.choice([0, 10], size=10**5, p=[1-p, p])
print(f"P(X >= 10) = {np.mean(samples >= 10):.4f}, Markov bound: {np.mean(samples)/10:.4f}")
print(f"P(|X - 1| >= 9) = {np.mean(np.abs(samples - 1) >= 9):.4f}, Chebyshev bound: {np.var(samples)/81:.4f}")
```

---

## Topic 2.2.8 — [QUANT APPLICATION]

### Application 1. Expected return and Sharpe ratio

The expected portfolio return $\mathbb E R_p = w^\top m$ and variance $\operatorname{Var}(R_p) = w^\top \Sigma w$ (from Module 1.5). The Sharpe ratio $(\mathbb E R_p - r_f)/\sigma$ is maximized by the **tangent portfolio** $w^* \propto \Sigma^{-1}(m - r_f \mathbf 1)$.

### Application 2. Markov property and path dependence

A stochastic process $X_t$ is **Markov** if $\mathbb E[f(X_{t+h}) | \mathcal F_t] = \mathbb E[f(X_{t+h}) | X_t]$. Under the Markov property, the *path history* collapses to the *current state*: pricing reduces to solving PDEs.

### Application 3. No-arbitrage pricing and martingale measures

Under the risk-neutral measure $\mathbb Q$, discounted asset prices are $\mathbb Q$-martingales: $\widetilde S_t = e^{-rt} S_t$ satisfies $\mathbb E^{\mathbb Q}[\widetilde S_T | \mathcal F_t] = \widetilde S_t$. This is a pure conditional-expectation statement; all pricing is built on it.

### Application 4. Bayesian inference and posterior distributions

Bayesian posterior $\pi(\theta | x) \propto \ell(x | \theta) \pi(\theta)$ is the conditional density of parameter given data. Posterior mean $\mathbb E[\theta | x]$ is the MSE-optimal point estimate (by $L^2$-projection / conditional expectation).

### Application 5. Kalman filter as iterated conditional expectation

In a Gaussian state-space model $X_{t+1} = A X_t + \varepsilon_t$, $Y_t = C X_t + \eta_t$, the Kalman filter recursively computes $\mathbb E[X_t | \mathcal F_t^Y]$ using two steps:

1. *Predict*: $\hat X_{t | t-1} = A \hat X_{t-1 | t-1}$ (Markov evolution of the conditional mean).
2. *Update*: $\hat X_{t | t} = \hat X_{t | t-1} + K_t (Y_t - C \hat X_{t | t-1})$ (projection onto new information).

This is pure Hilbert space machinery on $L^2(\mathbb P)$.

### Application 6. Law of total variance in portfolio risk decomposition

Decompose return $R = \alpha + \beta F + \varepsilon$ (factor + idiosyncratic). By total variance:
$$\operatorname{Var}(R) = \beta^2 \operatorname{Var}(F) + \operatorname{Var}(\varepsilon).$$
Same identity powers principal component analysis, factor models, and attribution reports.

### Application 7. Conditional VaR and coherent risk

$\text{ES}_\alpha(L) = \mathbb E[L | L > \text{VaR}_\alpha(L)]$ is literally a conditional expectation. Its subadditivity is a consequence of the structure of conditional expectation and tail quantiles (given appropriate regularity).

### Application 8. Jensen and utility theory

If $u$ is concave (risk-averse), Jensen gives $\mathbb E u(W) \le u(\mathbb E W)$: the investor prefers the certain payoff $\mathbb E W$ to the risky $W$. The **certainty equivalent** $CE := u^{-1}(\mathbb E u(W)) \le \mathbb E W$, with difference $\mathbb E W - CE$ = **risk premium**. Portfolio optimization typically maximizes $CE$.

---

## Topic 2.2.9 — Exercises

### ★ (Foundational)

**2.2.E1.** Compute $\mathbb E, \operatorname{Var}$ of a $\text{Binomial}(n, p)$, two ways: using moment generating functions and by direct computation.

**2.2.E2.** For $X \sim \text{Exp}(\lambda)$, compute $\mathbb E X^k$ for $k \ge 1$. Deduce $\operatorname{Var} X = 1/\lambda^2$.

**2.2.E3.** Prove Chebyshev from Markov directly.

**2.2.E4.** Show: for $X \in L^2$, $\mathbb E X^2 \ge (\mathbb E X)^2$ with equality iff $X$ is a.s. constant.

**2.2.E5.** If $X, Y$ are independent and both in $L^2$, $\operatorname{Cov}(X, Y) = 0$. Converse? Construct a counterexample (hint: $Y = X^2$ for $X \sim N(0, 1)$).

**2.2.E6.** Compute $\mathbb E[X | Y]$ and $\mathbb E[Y | X]$ for $(X, Y)$ uniform on the unit disk.

**2.2.E7.** Let $Y_1, Y_2 \sim \text{Exp}(\lambda)$ independent. Compute $\mathbb E[Y_1 | Y_1 + Y_2 = s]$ and $\mathbb E[Y_1 \wedge Y_2 | Y_1 + Y_2 = s]$.

**2.2.E8.** Verify all the properties in Proposition 2.2.17 by direct computation for a discrete $X, Y$.

### ★★ (Core)

**2.2.E9.** Prove: $\operatorname{Var}(\mathbb E[X | \mathcal G]) = \operatorname{Var}(X) - \mathbb E[\operatorname{Var}(X | \mathcal G)]$ via expansion, and use it to prove the variance-reducing property of conditioning.

**2.2.E10.** Show: if $X$ is $\mathcal G$-measurable, $\mathbb E[XY | \mathcal G] = X \mathbb E[Y | \mathcal G]$ (extend to $X \ge 0$ unbounded via MCT).

**2.2.E11.** **Conditional Cauchy-Schwarz**: $|\mathbb E[XY | \mathcal G]|^2 \le \mathbb E[X^2 | \mathcal G] \mathbb E[Y^2 | \mathcal G]$.

**2.2.E12.** Show: if $\mathbb E X^2 < \infty$ and $Z \in L^2(\mathcal G)$, then $\mathbb E[(X - \mathbb E[X | \mathcal G])^2] \le \mathbb E[(X - Z)^2]$. I.e. conditional expectation minimizes MSE over $L^2(\mathcal G)$.

**2.2.E13.** **Iterated substitution lemma.** For $h$ Borel and $Y$ random: $\mathbb E[h(X, Y) | Y](\omega) = g(Y(\omega))$ where $g(y) := \mathbb E[h(X, y)]$ (for $X, Y$ independent). Prove this via standard machine.

**2.2.E14.** **Radon-Nikodym chain for conditional expectation.** If $\mathbb P \ll \mathbb Q$ with $d\mathbb P/d\mathbb Q = L$, then $\mathbb E^{\mathbb P}[X | \mathcal G] = \mathbb E^{\mathbb Q}[XL | \mathcal G]/\mathbb E^{\mathbb Q}[L | \mathcal G]$.

**2.2.E15.** For $X, Y$ with joint density $f(x, y)$, prove $\mathbb E[X | Y = y]$ is given by $\int x f(x, y) dx / f_Y(y)$ using the definition (CE).

**2.2.E16.** **Bayes rule for σ-algebras.** If $A \in \mathcal F$ has $\mathbb P(A) > 0$ and $\mathcal G \subset \mathcal F$:
$$\mathbb P(A | \mathcal G) = \frac{\mathbb E[\mathbf 1_A | \mathcal G]}{1}, \qquad \mathbb E[X | A, \mathcal G] = \frac{\mathbb E[X \mathbf 1_A | \mathcal G]}{\mathbb P(A | \mathcal G)}.$$

### ★★★ (Challenging / quant-relevant)

**2.2.E17.** **Doob's $L^p$-inequality (preview).** For a non-negative submartingale $X_n$ and $p > 1$:
$$\left\|\max_{k \le n} X_k\right\|_p \le \frac{p}{p-1} \|X_n\|_p.$$
*Uses conditional expectation + Hölder.* Full proof in Module 2.6.

**2.2.E18.** **Hoeffding's inequality.** For independent $X_i \in [a_i, b_i]$ and $t > 0$:
$$\mathbb P(|S_n - \mathbb E S_n| \ge t) \le 2 \exp(-2 t^2 / \sum (b_i - a_i)^2).$$

**2.2.E19.** **Quant: Black-Scholes as iterated conditional expectation.** In a discrete-time binomial model with $n$ steps, $S_{k+1} = S_k \cdot U$ or $S_k \cdot D$ with risk-neutral probabilities $p = (1+r-D)/(U-D)$. Show the option price $V_0 = \mathbb E^{\mathbb Q}[(S_n - K)^+/(1+r)^n]$ satisfies $V_k = \mathbb E^{\mathbb Q}[V_{k+1}/(1+r) | \mathcal F_k]$ (backward recursion).

**2.2.E20.** **Quant: CARA utility and the certainty equivalent.** Exponential utility $u(w) = -e^{-\gamma w}$. Show the certainty equivalent of $W \sim N(\mu, \sigma^2)$ is $\mu - \gamma \sigma^2/2$. (So CARA maximizes *mean minus penalty times variance*, the Markowitz form.)

**2.2.E21.** **Quant: Kalman filter in closed form.** State-space $X_{t+1} = a X_t + \varepsilon_t$, $Y_t = X_t + \eta_t$, all Gaussian. Derive the Kalman recursion $\hat X_{t|t} = \hat X_{t | t-1} + K_t(Y_t - \hat X_{t | t-1})$ with $K_t = P_{t | t-1}/(P_{t|t-1} + \sigma_\eta^2)$, and show $P_{t|t} = P_{t|t-1} - K_t P_{t|t-1}$.

**2.2.E22.** **Quant: conditional variance as instantaneous volatility.** For a diffusion $dX_t = \mu(X_t) dt + \sigma(X_t) dW_t$, show
$$\operatorname{Var}(X_{t+h} - X_t | \mathcal F_t) = \sigma^2(X_t) h + o(h).$$
This gives $\sigma^2(X_t)$ as the instantaneous variance per unit time — the *local volatility*.

---

## Module Summary

- Expectation is the Lebesgue integral on $(\Omega, \mathcal F, \mathbb P)$; variance, moments, covariance are derived.
- Moment inequalities: **Markov**, **Chebyshev**, **Chernoff**, **Cauchy-Schwarz**, **Jensen**. Jensen gives $\varphi(\mathbb E X) \le \mathbb E \varphi(X)$ for $\varphi$ convex.
- On a probability space, $L^p$-norms are nested: $\|X\|_p \le \|X\|_q$ for $p \le q$.
- **Conditional expectation** $\mathbb E[X | \mathcal G]$: unique (a.s.) $\mathcal G$-measurable r.v. satisfying $\int_A \mathbb E[X | \mathcal G] = \int_A X$ for all $A \in \mathcal G$. Exists by Radon-Nikodym.
- Properties: linearity, tower, pulling out known factors, independence → constant, Jensen, $L^p$-contraction (hence $L^2$-projection).
- Conditional variance and **law of total variance**: $\operatorname{Var}(X) = \mathbb E \operatorname{Var}(X | \mathcal G) + \operatorname{Var}(\mathbb E[X | \mathcal G])$.

**Forward pointers:**
- Module 2.3: four modes of convergence, Borel-Cantelli in action, Kolmogorov three-series.
- Module 2.6: martingales defined via conditional expectation; optional stopping, Doob's inequalities.
- Subject 3: stochastic integration and Itô calculus; Itô isometry is an $L^2$ statement, Feynman-Kac is a conditional expectation.

**Recommended reading:**
- Williams, *Probability with Martingales*, Chapters 5-9.
- Durrett, *Probability: Theory and Examples*, Chapters 4-5.
- Rogers & Williams, *Diffusions, Markov Processes, and Martingales*, Vol. 1.

**Next module:** Modes of convergence — a.s., in probability, $L^p$, in distribution — and the fundamental results that relate them.

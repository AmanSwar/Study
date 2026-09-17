# Module 0.4 — Real Analysis and Metric Spaces

*Part of "Mathematical Foundations for Quantitative Research: From JEE to Jane Street" — Subject 0 (Quant Foundations), Module 4.*

---

## Table of Contents

- **Topic 0.4.1** — Metric Spaces
- **Topic 0.4.2** — Completeness and Banach Spaces
- **Topic 0.4.3** — Compactness
- **Topic 0.4.4** — Continuous Functions and Uniform Convergence
- **Topic 0.4.5** — Connectedness

---

## Module Overview

Multivariable calculus (Module 0.3) rested on topological bedrock it did not prove: that continuous functions on closed bounded sets of $\mathbb{R}^n$ attain their extrema; that $C^1$ bijections with non-vanishing derivative have inverses; that Cauchy sequences in $\mathbb{R}^n$ converge. This module supplies that bedrock, in a level of generality that will underpin everything that follows — probability (Subject 2), stochastic calculus (Subject 3), functional analysis in optimization (Subject 5), and ergodic theory.

The central objects are **metric spaces** and their three flagship properties: **completeness**, **compactness**, and **connectedness**. Each property corresponds to a fundamental theorem:

- **Completeness** ⇔ Banach fixed-point theorem, Picard-Lindelöf (ODE existence), Baire category.
- **Compactness** ⇔ Extreme value theorem, Bolzano-Weierstrass, Arzela-Ascoli (compactness in function spaces).
- **Connectedness** ⇔ Intermediate value theorem, monodromy in complex analysis.

Where Module 0.3 spoke the language of $\mathbb{R}^n$, this module speaks the universal language that the same arguments work in $\ell^p$ spaces, in $C[0,1]$, in path space for Brownian motion, and in any abstract metric setting we may encounter. The payoff is massive: the Banach fixed-point theorem proven once buys us ODE existence, SDE existence, dynamic programming contractions, Bellman equation solvability, and more.

*Prerequisites: Modules 0.1 (Logic/Proofs), 0.2 (Linear Algebra), 0.3 (Multivariable Calculus).*

---


## Topic 0.4.1 — Metric Spaces

### Motivation

The $\varepsilon$-$\delta$ definition of continuity in $\mathbb{R}$ depends only on the absolute-value distance $|x - y|$. The same proof works for any distance $d$ satisfying a few axioms, so we abstract: a **metric space** is a set with a well-behaved notion of distance. This abstraction pays off twice. First, the same theorems apply in diverse settings — $\mathbb{R}^n$, infinite sequences $\ell^p$, spaces of continuous functions $C[a, b]$, probability measures (Wasserstein distance), graphs (shortest path), DNA (Hamming distance). Second, the *process* of proving in the abstract forces us to identify exactly which properties a statement uses, stripping away accidental features of $\mathbb{R}$.

Most of modern analysis, from the dominated convergence theorem to the martingale convergence theorem to the Picard-Lindelöf theorem on ODE existence, requires only the vocabulary introduced here. In quant work, metric-space topology underpins:

- **Convergence of numerical methods** — discretizations must converge in a suitable metric.
- **Path-space topology** for diffusion processes (continuous paths equipped with sup-metric / Skorokhod metric).
- **Wasserstein distance** between probability distributions (fundamental in robust optimization, GANs, and optimal transport).
- **Hilbert projection arguments** (Module 0.2 / Hilbert-space machinery) that need completeness to guarantee unique orthogonal projections.

### Prerequisites

- Sets, functions (Module 0.1).
- Inner products, norms (Module 0.2).
- Real numbers: completeness axiom (least-upper-bound property) taken as given.

### Definitions

**Definition 0.4.1.1 (Metric).** A **metric** on a set $X$ is a function $d: X \times X \to \mathbb{R}_{\geq 0}$ satisfying, for all $x, y, z \in X$:

(M1) $d(x, y) = 0 \iff x = y$ (point-separation);
(M2) $d(x, y) = d(y, x)$ (symmetry);
(M3) $d(x, z) \leq d(x, y) + d(y, z)$ (triangle inequality).

The pair $(X, d)$ is called a **metric space**.

**Definition 0.4.1.2 (Pseudometric, semimetric).** Dropping axiom (M1) — allowing $d(x, y) = 0$ with $x \neq y$ — gives a **pseudometric**. Most analysis still works (after quotienting by the equivalence $x \sim y \iff d(x, y) = 0$).

**Examples 0.4.1.3.**

- *Real line.* $X = \mathbb{R}$, $d(x, y) = |x - y|$.
- *Euclidean space.* $X = \mathbb{R}^n$, $d_2(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$. The triangle inequality is Cauchy-Schwarz in disguise (Topic 0.2.3).
- *$\ell^p$-metrics on $\mathbb{R}^n$.* $d_p(\mathbf{x}, \mathbf{y}) = \left(\sum_i |x_i - y_i|^p\right)^{1/p}$ for $1 \leq p < \infty$; $d_\infty(\mathbf{x}, \mathbf{y}) = \max_i |x_i - y_i|$. Triangle inequality is Minkowski's inequality.
- *Discrete metric.* On any set $X$, $d(x, y) = 0$ if $x = y$, $1$ otherwise. All axioms trivial. Every function out of $(X, d_{\mathrm{disc}})$ is continuous.
- *Sup metric on $C[a, b]$.* The set of continuous functions $f: [a, b] \to \mathbb{R}$; $d_\infty(f, g) = \sup_{t \in [a, b]} |f(t) - g(t)|$. The triangle inequality follows from the pointwise version.
- *$L^p$-metrics on function spaces.* $d_p(f, g) = \left(\int_a^b |f(t) - g(t)|^p\, dt\right)^{1/p}$. Complete setup needs measure theory (Subject 1); we treat $L^p$ provisionally via Riemann integration.
- *Hamming distance.* On binary strings $\{0, 1\}^n$, $d_H(\mathbf{x}, \mathbf{y}) = \#\{i : x_i \neq y_i\}$. Foundational in coding theory.
- *Edit distance (Levenshtein).* On strings of varying length, $d(s, t)$ = minimum single-character insertions, deletions, substitutions to transform $s$ into $t$.
- *Wasserstein / earth-mover distance.* On probability measures on $(\mathbb{R}^n, d_2)$: $W_p(\mu, \nu) = \big(\inf_{\gamma \in \Pi(\mu, \nu)} \int d(x,y)^p\, d\gamma\big)^{1/p}$ where $\Pi(\mu, \nu)$ is the set of joint distributions ("couplings") with marginals $\mu, \nu$. Essential in optimal transport and robust statistics.

### Induced topology

**Definition 0.4.1.4 (Open ball).** For $x \in X$, $r > 0$, the **open ball** is
$$B_r(x) = \{y \in X : d(x, y) < r\}.$$
The **closed ball** is $\overline{B_r(x)} = \{y : d(x, y) \leq r\}$ (careful — this is *not* always the topological closure of $B_r(x)$ in general metric spaces!).

**Definition 0.4.1.5 (Open set).** A set $U \subseteq X$ is **open** iff for every $x \in U$ there exists $r > 0$ with $B_r(x) \subseteq U$.

**Proposition 0.4.1.6 (Topology axioms for metric spaces).** The collection $\mathcal{T}_d$ of open sets satisfies:
(T1) $\varnothing, X \in \mathcal{T}_d$.
(T2) Arbitrary unions of open sets are open.
(T3) Finite intersections of open sets are open.

**Proof.** (T1) trivial. (T2): if $U = \bigcup_\alpha U_\alpha$ and $x \in U$, then $x \in U_\alpha$ for some $\alpha$, and $\exists r > 0$ with $B_r(x) \subseteq U_\alpha \subseteq U$. (T3): if $U = \bigcap_{i=1}^n U_i$ and $x \in U$, each $U_i$ gives $r_i > 0$ with $B_{r_i}(x) \subseteq U_i$; take $r = \min r_i > 0$, then $B_r(x) \subseteq U$. $\blacksquare$

*Why not infinite intersections?* Consider $\bigcap_{n=1}^\infty (-1/n, 1/n) = \{0\}$ in $\mathbb{R}$, which is *not* open.

**Definition 0.4.1.7 (Closed set).** $F \subseteq X$ is **closed** iff $X \setminus F$ is open.

**Definition 0.4.1.8 (Interior, closure, boundary).** For $A \subseteq X$:

- Interior: $\mathrm{int}(A) = \bigcup \{U \subseteq A : U \text{ open}\}$, the largest open set $\subseteq A$.
- Closure: $\overline{A} = \bigcap \{F \supseteq A : F \text{ closed}\}$, the smallest closed set $\supseteq A$.
- Boundary: $\partial A = \overline{A} \setminus \mathrm{int}(A) = \overline{A} \cap \overline{X \setminus A}$.

**Proposition 0.4.1.9 (Sequential characterization of closure).** $\overline{A} = \{x \in X : \exists (a_n) \subseteq A \text{ with } a_n \to x\}$.

**Proof.** ($\supseteq$) If $a_n \in A$ and $a_n \to x$, then any open $U \ni x$ contains some $a_n \in A$, so $x$ cannot be in the interior of $X \setminus A$ (which would force a ball around $x$ disjoint from $A$). Thus $x \in \overline{A}$.

($\subseteq$) If $x \in \overline{A}$, then for every $n \geq 1$, $B_{1/n}(x) \cap A \neq \varnothing$: otherwise $B_{1/n}(x) \subseteq X \setminus A$ is an open set of $X \setminus A$ around $x$, contradicting $x \in \overline{X \setminus (X \setminus A)} \supseteq \overline{A}$. Wait — that's not quite right. Redo: if $B_{1/n}(x) \cap A = \varnothing$, then $B_{1/n}(x) \subseteq X \setminus A$. The complement of $\overline{A}$ is $\mathrm{int}(X \setminus A)$ by general-topology duality (interior of complement = complement of closure), so $B_{1/n}(x) \subseteq X \setminus \overline{A}$, which would contradict $x \in \overline{A}$. Pick $a_n \in B_{1/n}(x) \cap A$; then $d(x, a_n) < 1/n \to 0$, so $a_n \to x$. $\blacksquare$

### Convergence in Metric Spaces

**Definition 0.4.1.10 (Convergence).** A sequence $(x_n) \subseteq X$ **converges** to $x \in X$ (written $x_n \to x$) iff
$$\forall \varepsilon > 0 \; \exists N \in \mathbb{N} \; \forall n \geq N: \; d(x_n, x) < \varepsilon.$$

Equivalently, the real sequence $d(x_n, x) \to 0$.

**Proposition 0.4.1.11 (Uniqueness of limits).** Limits in metric spaces are unique: if $x_n \to x$ and $x_n \to y$, then $x = y$.

**Proof.** $d(x, y) \leq d(x, x_n) + d(x_n, y) \to 0$, so $d(x, y) = 0$, hence $x = y$ by (M1). $\blacksquare$

**Definition 0.4.1.12 (Cauchy sequence).** $(x_n)$ is a **Cauchy sequence** iff
$$\forall \varepsilon > 0 \; \exists N \; \forall m, n \geq N: \; d(x_m, x_n) < \varepsilon.$$

**Proposition 0.4.1.13 (Convergent ⇒ Cauchy).** Every convergent sequence is Cauchy.

**Proof.** If $x_n \to x$, given $\varepsilon > 0$ pick $N$ with $d(x_n, x) < \varepsilon/2$ for $n \geq N$; then $d(x_m, x_n) \leq d(x_m, x) + d(x, x_n) < \varepsilon$ for $m, n \geq N$. $\blacksquare$

The converse (Cauchy ⇒ convergent) is **completeness**, the subject of Topic 0.4.2. It holds in $\mathbb{R}^n$ (essentially by the supremum axiom) but *fails* in general: e.g., the rationals $\mathbb{Q}$ under $|\cdot - \cdot|$ have Cauchy sequences (a decimal expansion of $\sqrt 2$) that do not converge in $\mathbb{Q}$.

### Continuity

**Definition 0.4.1.14 (Continuous function).** $f: (X, d_X) \to (Y, d_Y)$ is **continuous at $x_0 \in X$** iff
$$\forall \varepsilon > 0 \; \exists \delta > 0 \; \forall x: \; d_X(x, x_0) < \delta \implies d_Y(f(x), f(x_0)) < \varepsilon.$$
$f$ is **continuous on $X$** iff continuous at every $x_0 \in X$.

**Theorem 0.4.1.15 (Three equivalent characterizations of continuity).** For $f: X \to Y$ between metric spaces, TFAE:

(a) $f$ is continuous on $X$ (Definition 0.4.1.14).
(b) For every open $V \subseteq Y$, $f^{-1}(V)$ is open in $X$.
(c) (Sequential continuity) For every sequence $x_n \to x$ in $X$, $f(x_n) \to f(x)$ in $Y$.

**Proof.**

(a) $\Rightarrow$ (b). Let $V \subseteq Y$ be open; let $x \in f^{-1}(V)$. Then $f(x) \in V$, and openness of $V$ gives $\varepsilon > 0$ with $B_\varepsilon(f(x)) \subseteq V$. By (a), $\exists \delta > 0$ with $f(B_\delta(x)) \subseteq B_\varepsilon(f(x)) \subseteq V$, so $B_\delta(x) \subseteq f^{-1}(V)$. Hence $f^{-1}(V)$ is open.

(b) $\Rightarrow$ (a). Fix $x_0 \in X$ and $\varepsilon > 0$. Then $V = B_\varepsilon(f(x_0))$ is open, so $f^{-1}(V)$ is open and contains $x_0$. Pick $\delta > 0$ with $B_\delta(x_0) \subseteq f^{-1}(V)$; then $d_X(x, x_0) < \delta \Rightarrow f(x) \in V \Rightarrow d_Y(f(x), f(x_0)) < \varepsilon$.

(a) $\Rightarrow$ (c). Let $x_n \to x$. Fix $\varepsilon > 0$; (a) gives $\delta > 0$ with $f(B_\delta(x)) \subseteq B_\varepsilon(f(x))$. Pick $N$ with $d_X(x_n, x) < \delta$ for $n \geq N$; then $d_Y(f(x_n), f(x)) < \varepsilon$ for $n \geq N$.

(c) $\Rightarrow$ (a). Contrapositive: suppose $f$ is not continuous at $x$. Then $\exists \varepsilon_0 > 0$ such that $\forall \delta > 0 \; \exists x' \in B_\delta(x)$ with $d_Y(f(x'), f(x)) \geq \varepsilon_0$. Take $\delta = 1/n$ and pick such an $x_n$; then $x_n \to x$ but $d_Y(f(x_n), f(x)) \geq \varepsilon_0$, so $f(x_n) \not\to f(x)$. $\blacksquare$

**Corollary 0.4.1.16.** Continuity is a topological property: it depends only on the induced topology (collection of open sets), not on the specific metric that induced it.

**Example 0.4.1.17 (Equivalent metrics).** Two metrics $d_1, d_2$ on $X$ are **equivalent** iff they induce the same topology. Sufficient condition: $\exists c, C > 0$ with $c\, d_1(x,y) \leq d_2(x,y) \leq C\, d_1(x,y)$ for all $x, y$ (called **Lipschitz equivalence**). All $\ell^p$-metrics on $\mathbb{R}^n$ are Lipschitz-equivalent (dimension-dependent constants, but constants all the same). All norms on a finite-dimensional vector space are equivalent (Topic 0.2.9 / proof via compactness in Topic 0.4.3).

### Worked Examples

#### Example 0.4.1.18 — Verifying the triangle inequality for $\ell^p$

For $p \geq 1$ and $\mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathbb{R}^n$:
$$\|\mathbf{x} - \mathbf{z}\|_p \leq \|\mathbf{x} - \mathbf{y}\|_p + \|\mathbf{y} - \mathbf{z}\|_p.$$
This is **Minkowski's inequality** applied to $\mathbf{a} = \mathbf{x} - \mathbf{y}$, $\mathbf{b} = \mathbf{y} - \mathbf{z}$:
$$\left(\sum (a_i + b_i)^p\right)^{1/p} \leq \left(\sum a_i^p\right)^{1/p} + \left(\sum b_i^p\right)^{1/p} \quad (a_i, b_i \geq 0).$$

**Proof of Minkowski.** For $p = 1$: $|a_i + b_i| \leq |a_i| + |b_i|$ is trivial. For $p > 1$, use Hölder's inequality with conjugate exponent $q = p/(p-1)$: $\sum |u_i v_i| \leq (\sum |u_i|^p)^{1/p} (\sum |v_i|^q)^{1/q}$. Write:
$$\sum (a_i + b_i)^p = \sum (a_i + b_i)^{p-1} a_i + \sum (a_i + b_i)^{p-1} b_i.$$
Apply Hölder to each piece with the "$v_i$" being $(a_i + b_i)^{p-1}$, exponent $q$:
$$\left[\sum (a_i + b_i)^{(p-1)q}\right]^{1/q} = \left[\sum (a_i + b_i)^p\right]^{1/q}.$$
So
$$\sum(a_i + b_i)^p \leq \left(\sum a_i^p\right)^{1/p} \left[\sum (a_i + b_i)^p\right]^{1/q} + \left(\sum b_i^p\right)^{1/p} \left[\sum (a_i + b_i)^p\right]^{1/q}.$$
Divide both sides by $[\sum (a_i + b_i)^p]^{1/q}$; the remaining exponent is $1 - 1/q = 1/p$:
$$\left[\sum (a_i + b_i)^p\right]^{1/p} \leq \left(\sum a_i^p\right)^{1/p} + \left(\sum b_i^p\right)^{1/p}. \qquad \blacksquare$$

#### Example 0.4.1.19 — The sup metric on $C[0, 1]$ and uniform convergence

On $X = C[0, 1]$ (continuous real-valued functions on $[0, 1]$), $d_\infty(f, g) = \sup_{t \in [0, 1]} |f(t) - g(t)|$.

**Claim.** $d_\infty$ is a metric.
- (M1): $d_\infty(f, g) = 0 \iff f(t) = g(t)$ for all $t$, i.e., $f = g$.
- (M2): symmetric.
- (M3): $|f(t) - h(t)| \leq |f(t) - g(t)| + |g(t) - h(t)|$ pointwise; taking sup gives (M3). (Caution: $\sup(A + B) \leq \sup A + \sup B$, not equality.)

Convergence in $d_\infty$ is **uniform convergence**: $f_n \to f$ uniformly iff $\sup_t |f_n(t) - f(t)| \to 0$. This metric captures exactly the notion of convergence under which the limit of continuous functions remains continuous (an essential fact we will prove in Topic 0.4.4).

#### Example 0.4.1.20 — A non-sequentially-closed set in an exotic topology

*This example shows why Definition 0.4.1.7 matters.* In a metric space, closed = sequentially closed (Proposition 0.4.1.9). In a general topological space they can differ, which is one reason metric spaces are friendlier. We flag this for contrast.

#### Example 0.4.1.21 — The Wasserstein-$1$ distance between empirical measures

Let $\mu_n = n^{-1}\sum_{i=1}^n \delta_{x_i}$ and $\mu = n^{-1}\sum \delta_{y_i}$ be empirical measures on $\mathbb{R}$. Then
$$W_1(\mu_n, \mu) = \frac{1}{n} \sum_{i=1}^n |x_{(i)} - y_{(i)}|,$$
where $x_{(i)}, y_{(i)}$ are the sorted values (in increasing order). This quantifies distributional difference and is routinely used in distributionally robust optimization.

### Computational Implementation

```python
import numpy as np

# ---------- Metric verification helpers ----------
def is_metric(d_func, sample_points, atol=1e-10):
    """Verify the three metric axioms on a small sample."""
    ok = True
    for x in sample_points:
        if abs(d_func(x, x)) > atol:
            print(f"  M1 fail: d({x}, {x}) = {d_func(x, x)} != 0"); ok = False
    for x in sample_points:
        for y in sample_points:
            if x != y and d_func(x, y) <= atol:
                print(f"  M1 fail: d({x}, {y}) = {d_func(x, y)} but x != y"); ok = False
            if abs(d_func(x, y) - d_func(y, x)) > atol:
                print(f"  M2 fail: d({x}, {y}) != d({y}, {x})"); ok = False
    for x in sample_points:
        for y in sample_points:
            for z in sample_points:
                if d_func(x, z) > d_func(x, y) + d_func(y, z) + atol:
                    print(f"  M3 fail at ({x}, {y}, {z})"); ok = False
    return ok

# ---------- L^p metrics ----------
def lp_metric(p):
    def d(x, y):
        x, y = np.asarray(x), np.asarray(y)
        if p == np.inf:
            return np.max(np.abs(x - y))
        return np.sum(np.abs(x - y)**p)**(1/p)
    return d

# Random test vectors
np.random.seed(7)
pts = [tuple(np.random.randn(3).round(2)) for _ in range(5)]
for p in [1, 2, 3, np.inf]:
    d = lp_metric(p)
    print(f"Verifying L^{p} metric on R^3:")
    assert is_metric(d, pts)
    print(f"  ✓ metric axioms satisfied")

# ---------- Lipschitz equivalence of L^p metrics ----------
# On R^n: ||x||_inf <= ||x||_p <= n^{1/p} ||x||_inf for p >= 1
x = np.random.randn(100)
for p in [1, 2, 3, 10, np.inf]:
    norm_p = np.linalg.norm(x, p) if p != np.inf else np.max(np.abs(x))
    print(f"  ||x||_{p} = {norm_p:.4f}")
print(f"  n^{{1/p}} * ||x||_inf for p=2: {np.sqrt(100) * np.max(np.abs(x)):.4f}")

# ---------- Wasserstein-1 distance between empirical measures on R ----------
def wasserstein1_1d(xs, ys):
    """W_1 between two empirical measures of the same size on R."""
    xs_sorted = np.sort(xs)
    ys_sorted = np.sort(ys)
    return np.mean(np.abs(xs_sorted - ys_sorted))

# Example: two samples from slightly different Gaussians
np.random.seed(0)
xs = np.random.randn(1000)
ys = np.random.randn(1000) + 0.5  # mean-shifted
print(f"\nW_1 between N(0,1) and N(0.5, 1) empirical measures: {wasserstein1_1d(xs, ys):.4f}")
print(f"Theoretical W_1 = |mean shift| = 0.5")

# ---------- Open-set check in (R^2, L^2) ----------
def open_ball(center, radius, metric):
    def membership(point):
        return metric(center, point) < radius
    return membership

disk = open_ball(np.zeros(2), 1.0, lp_metric(2))
print(f"\nIs (0.5, 0) in B_1(0)? {disk(np.array([0.5, 0]))}")
print(f"Is (1, 0) in B_1(0)? {disk(np.array([1.0, 0]))}")  # boundary, so NO
print(f"Is (0.99, 0) in B_1(0)? {disk(np.array([0.99, 0]))}")

# ---------- Checking triangle inequality on C[0,1] with sup metric ----------
# Use functions on a dense grid
t = np.linspace(0, 1, 1001)
f = lambda t: np.sin(3*t)
g = lambda t: np.cos(2*t)
h = lambda t: t**2 - 0.3

d_sup = lambda u, v: np.max(np.abs(u - v))
vals_f, vals_g, vals_h = f(t), g(t), h(t)
fh = d_sup(vals_f, vals_h); fg = d_sup(vals_f, vals_g); gh = d_sup(vals_g, vals_h)
print(f"\nSup metric on C[0,1]:")
print(f"  d(f, h) = {fh:.4f}, d(f, g) + d(g, h) = {fg + gh:.4f}")
print(f"  Triangle OK? {fh <= fg + gh + 1e-12}")

# ---------- Sequential characterization of closure ----------
# Show that closure of Q ∩ [0, 1] in R is [0, 1]: approximate any real by rationals.
def approx_irrational_by_rational(x, n_digits=10):
    return round(x * 10**n_digits) / 10**n_digits

sqrt2_half = 0.5 * np.sqrt(2)
for n in [1, 3, 6, 10]:
    q = approx_irrational_by_rational(sqrt2_half, n)
    print(f"  rational approx to sqrt(2)/2 at {n} digits: {q}, error: {abs(q - sqrt2_half):.2e}")
```

### [QUANT APPLICATION] Metrics in Quant Finance and ML

**(A) Wasserstein-robust portfolios.** Modern robust optimization minimizes the worst-case expected loss over all distributions within Wasserstein distance $\varepsilon$ of the nominal. Concretely,
$$\min_\mathbf{w} \max_{\nu: W_p(\nu, \hat\mu) \leq \varepsilon} \mathbb{E}_\nu[\ell(\mathbf{w}, X)].$$
Kuhn, Esfahani, and Mohajerin Esfahani (2018) showed this dual-reduces to a tractable convex program — the foundation of "Wasserstein distributionally robust optimization" which is widely used in finance.

**(B) Sup-metric in option pricing.** The Black-Scholes pricing function is Lipschitz in the underlying with respect to the sup metric: $\sup_S |C(S, K) - C(S', K)| \leq \sup_S |S - S'|$, giving stability bounds on the pricing operator essential for numerical schemes.

**(C) Total variation and KL-divergence.** TV distance $d_{TV}(\mu, \nu) = \sup_A |\mu(A) - \nu(A)|$ is a metric on probability measures. KL-divergence is *not* a metric (neither symmetric nor satisfying triangle), but it bounds TV via Pinsker's inequality $d_{TV} \leq \sqrt{\tfrac{1}{2} \mathrm{KL}}$. Both appear in information-theoretic bounds, MLE consistency, and variational inference.

**(D) Path-space topology.** For SDEs, the space $C([0, T], \mathbb{R}^n)$ with sup metric is the natural home for continuous paths; jump processes live in $D([0, T], \mathbb{R}^n)$ with the Skorokhod $J_1$ topology. These function-space metrics are used to show weak convergence of discretization schemes (Euler-Maruyama → true SDE).

**(E) Graph / network distances.** Counterparty credit risk modeling uses network metrics on a weighted graph of counterparty exposures; the shortest-path distance is a metric, and stability of risk measures under small graph perturbations requires a metric setup.

**(F) Lasso and proximal operators.** Proximal-gradient methods (ISTA, FISTA) solve $\min_\mathbf{x} f(\mathbf{x}) + g(\mathbf{x})$ iteratively using proximal operators, which are contractions in suitable metrics. Convergence theory rests on metric-space fixed-point theory.

### Exercises

#### ★ (Foundation)

**E0.4.1.1.** Prove the **reverse triangle inequality** $|d(x, z) - d(y, z)| \leq d(x, y)$.

**E0.4.1.2.** Show that $d(x, y) = |f(x) - f(y)|$ is a *pseudo*metric on any set $X$ given $f: X \to \mathbb{R}$. Under what condition is it a metric?

**E0.4.1.3.** Prove that the closed ball $\overline{B_r(x)} = \{y : d(x, y) \leq r\}$ is closed and that it contains $\overline{B_r(x)}$ (the closure of the open ball). Give an example in an exotic metric space where the inclusion is strict.

**E0.4.1.4.** Prove that a subset $A \subseteq X$ is closed iff every convergent sequence $(a_n) \subseteq A$ has its limit in $A$.

**E0.4.1.5.** In $\mathbb{R}$, show that the set $\{1/n : n \in \mathbb{N}\}$ has closure $\{0\} \cup \{1/n : n \in \mathbb{N}\}$. What is its interior? Its boundary?

#### ★★ (Intermediate)

**E0.4.1.6 (Equivalent metrics on $\mathbb{R}^n$).** Prove that all norms on a finite-dimensional vector space are equivalent. (Hint: use compactness of the unit sphere, which we prove in Topic 0.4.3.)

**E0.4.1.7 (Ultrametric).** A metric $d$ is an **ultrametric** iff $d(x, z) \leq \max(d(x, y), d(y, z))$ — a stronger triangle inequality. Show: in an ultrametric space, every triangle is isosceles with the unique shortest side being the base. Example: the $p$-adic metric on $\mathbb{Z}$: $d_p(n, m) = p^{-v_p(n - m)}$ where $v_p$ is the $p$-adic valuation.

**E0.4.1.8 (Hausdorff distance).** On the collection of non-empty compact subsets of a metric space, the **Hausdorff distance** $d_H(A, B) = \max\{\sup_{a \in A} d(a, B), \sup_{b \in B} d(b, A)\}$. Verify it is a metric.

**E0.4.1.9 (Lipschitz functions preserve Cauchy).** If $f: (X, d_X) \to (Y, d_Y)$ is Lipschitz ($d_Y(f(x), f(x')) \leq L d_X(x, x')$) and $(x_n)$ is Cauchy in $X$, then $(f(x_n))$ is Cauchy in $Y$.

**E0.4.1.10 (Continuity from the metric directly).** Prove that $x \mapsto d(x, a)$ is Lipschitz (hence continuous) for any fixed $a \in X$. Hint: reverse triangle.

#### ★★★ (Challenge)

**E0.4.1.11 (Metrization theorems).** Prove that a topological space is metrizable iff it is "regular" (separates points from closed sets), Hausdorff, and has a countable basis (**Urysohn's metrization theorem**). This shows which topological spaces *can* come from a metric.

**E0.4.1.12 (Complete sigma-algebras and metrization).** Let $(X, \mathcal{A})$ be a measurable space with countably generated sigma-algebra. Show there is a metric on $X$ (up to a completion) such that $\mathcal{A}$ is the Borel sigma-algebra — a fact used in "standard Borel spaces" in probability.

**E0.4.1.13 (Skorokhod metric).** Define the Skorokhod $J_1$ metric on $D([0, T], \mathbb{R})$ (càdlàg functions). Prove it is a metric and that its topology differs from the sup-metric topology (continuity at the jump times).

**E0.4.1.14 (Wasserstein duality).** Prove the **Kantorovich-Rubinstein duality**: for $p = 1$,
$$W_1(\mu, \nu) = \sup_{f \text{ 1-Lipschitz}} \left|\int f\, d\mu - \int f\, d\nu\right|.$$
This is one of the most useful computational identities in modern probability.

---


## Topic 0.4.2 — Completeness and Banach Spaces

### Motivation

Completeness is the single most consequential property a metric space can possess. It is what separates $\mathbb{R}$ from $\mathbb{Q}$: both are densely ordered, both support the same arithmetic, but in $\mathbb{R}$ every Cauchy sequence converges and in $\mathbb{Q}$ it does not. Nearly every existence theorem in analysis — for limits, for fixed points, for solutions of equations, for optimal control — ultimately relies on completeness.

Concretely:
- Solutions to ODEs are constructed as Cauchy sequences of approximate solutions; Picard-Lindelöf needs completeness of $C(I, \mathbb{R}^n)$.
- Bellman equations in dynamic programming are solved by contraction iteration; the iteration converges because the value-function space is complete.
- The Hilbert space $L^2$ is complete, which is why orthogonal projections exist, which is why conditional expectations exist.
- Neural network training loss minimization constructs iterates $\theta_{k+1} = \theta_k - \eta \nabla L$; convergence of these iterates requires completeness of parameter space.

This topic introduces completeness, the parallel notion of **Banach spaces** (complete normed vector spaces), and the **Baire category theorem** — a surprisingly powerful consequence with wide-ranging applications in functional analysis.

### Prerequisites

- Metric spaces and Cauchy sequences (Topic 0.4.1).
- Inner products, norms (Module 0.2).

### Completeness

**Definition 0.4.2.1 (Complete metric space).** A metric space $(X, d)$ is **complete** iff every Cauchy sequence in $X$ converges to a limit in $X$.

**Examples 0.4.2.2.**

- $\mathbb{R}$ with usual metric is complete (follows from the supremum axiom).
- $\mathbb{Q}$ is *not* complete: the sequence $(1, 1.4, 1.41, 1.414, \ldots)$ approximating $\sqrt 2$ is Cauchy in $\mathbb{Q}$ but has no rational limit.
- $\mathbb{R}^n$ with any $\ell^p$-metric is complete (coordinate-wise; $\mathbb{R}$ complete $\Rightarrow$ $\mathbb{R}^n$ complete).
- A discrete metric space $(X, d_{\mathrm{disc}})$ is complete (only eventually constant sequences are Cauchy).
- $C[a, b]$ with sup metric is complete (uniform limit of continuous functions is continuous — proved below).
- $C[a, b]$ with the $L^1$-metric is *not* complete: the sequence of continuous approximations to the indicator of a half-interval is Cauchy but its $L^1$-limit is discontinuous.

**Theorem 0.4.2.3 (Completeness of $\mathbb{R}$).** Every Cauchy sequence of real numbers converges.

**Proof.** Let $(x_n)$ be Cauchy in $\mathbb{R}$.

*Step 1: Cauchy sequences are bounded.* Choose $N$ so $|x_n - x_N| < 1$ for $n \geq N$; then $|x_n| \leq |x_N| + 1$ for $n \geq N$, and $|x_n|$ is finite for $n < N$. So $\{x_n\}$ is bounded.

*Step 2: Extract a monotone subsequence.* Either (a) for every $n$ there is $m > n$ with $x_m \geq x_n$, or (b) there is $n_0$ such that for all $n \geq n_0$ we have $x_m < x_n$ for all $m > n$. In case (a), build a weakly increasing subsequence; in case (b), a strictly decreasing one. Either way, we have a monotone subsequence $(x_{n_k})$.

*Step 3: Monotone bounded sequences converge.* The supremum $L = \sup x_{n_k}$ (case (a)) exists by the supremum axiom and is the limit: for any $\varepsilon > 0$, there's some $x_{n_k} > L - \varepsilon$, and monotonicity gives $x_{n_\ell} \in (L - \varepsilon, L]$ for all $\ell \geq k$. Similarly in case (b) with infimum.

*Step 4: Cauchy + convergent subsequence $\Rightarrow$ convergent.* Suppose $x_{n_k} \to L$. Given $\varepsilon > 0$, pick $N_1$ with $|x_m - x_n| < \varepsilon/2$ for $m, n \geq N_1$ (Cauchy), and $K$ with $|x_{n_k} - L| < \varepsilon/2$ for $k \geq K$. For $n \geq N_1$, pick $k \geq K$ with $n_k \geq N_1$ as well; then $|x_n - L| \leq |x_n - x_{n_k}| + |x_{n_k} - L| < \varepsilon$. $\blacksquare$

*This proof uses the supremum axiom in step 3. The supremum axiom is logically equivalent to completeness of $\mathbb{R}$.*

**Proposition 0.4.2.4.** $\mathbb{R}^n$ with any $\ell^p$-metric is complete.

**Proof.** All $\ell^p$-metrics on $\mathbb{R}^n$ are equivalent, so it suffices to show it for $d_\infty$. A sequence $(\mathbf{x}^{(k)})$ is Cauchy in $d_\infty$ iff each coordinate sequence $(x_i^{(k)})_k$ is Cauchy in $\mathbb{R}$. By Theorem 0.4.2.3, each coordinate sequence converges to some $x_i^* \in \mathbb{R}$. Let $\mathbf{x}^* = (x_1^*, \ldots, x_n^*)$. Then $\|\mathbf{x}^{(k)} - \mathbf{x}^*\|_\infty = \max_i |x_i^{(k)} - x_i^*| \to 0$. $\blacksquare$

### Closed Subsets of Complete Spaces

**Proposition 0.4.2.5.** *A subset $A$ of a complete metric space $(X, d)$ is complete (in the induced metric) iff $A$ is closed in $X$.*

**Proof.** ($\Rightarrow$) If $A$ is complete, let $x \in \overline{A}$. By the sequential characterization, $\exists (a_n) \subseteq A$ with $a_n \to x$. The sequence $(a_n)$ is convergent in $X$, hence Cauchy. Completeness of $A$ gives a limit $a \in A$; uniqueness of limits in $X$ (Prop 0.4.1.11) forces $a = x$, so $x \in A$. Thus $A = \overline A$, i.e., $A$ is closed.

($\Leftarrow$) If $A$ is closed, let $(a_n) \subseteq A$ be Cauchy. Then $(a_n)$ is Cauchy in the complete space $X$, hence converges to some $x \in X$. By the sequential characterization of closure, $x \in \overline A = A$. $\blacksquare$

### Banach Spaces

A normed vector space is a natural metric space: the metric $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$ satisfies (M1)-(M3) automatically. When such a space is complete, we call it a **Banach space**, after the Polish mathematician Stefan Banach.

**Definition 0.4.2.6 (Banach space).** A **Banach space** is a complete normed vector space.

**Examples 0.4.2.7.**

- $\mathbb{R}^n$ with any norm (all equivalent in finite dimensions).
- $\ell^p$ for $1 \leq p \leq \infty$: sequences $\mathbf{x} = (x_n)$ with $\|\mathbf{x}\|_p = (\sum_n |x_n|^p)^{1/p} < \infty$ (or sup for $p = \infty$). Complete.
- $C([a, b], \mathbb{R}^n)$ with sup norm $\|f\|_\infty = \sup_t \|f(t)\|$. Complete.
- $C^k([a, b])$ with norm $\|f\|_{C^k} = \sum_{j=0}^k \sup |f^{(j)}|$. Complete.
- $L^p([a, b], \mu)$ for $1 \leq p \leq \infty$ — complete only after the measure-theoretic definition of integration (Subject 1). With Riemann integration, $L^p$ is *not* complete.
- Space of bounded linear operators $\mathcal{L}(X, Y)$ between Banach spaces, with operator norm. Complete if $Y$ is complete.

**Proposition 0.4.2.8 (Completeness of $C(X, \mathbb{R})$ for compact $X$).** *If $X$ is a compact metric space, then $(C(X, \mathbb{R}), \|\cdot\|_\infty)$ is a Banach space.*

**Proof.** Let $(f_n) \subseteq C(X, \mathbb{R})$ be Cauchy in sup norm. For each $x \in X$, $(f_n(x))$ is Cauchy in $\mathbb{R}$ (since $|f_n(x) - f_m(x)| \leq \|f_n - f_m\|_\infty$), hence converges. Define $f(x) := \lim f_n(x)$ pointwise.

*Uniform convergence.* For $\varepsilon > 0$, choose $N$ so $\|f_m - f_n\|_\infty < \varepsilon$ for $m, n \geq N$. Let $m \to \infty$ in $|f_m(x) - f_n(x)| < \varepsilon$: by continuity of the metric (or pointwise convergence),
$$|f(x) - f_n(x)| \leq \varepsilon \quad \forall x \in X, \; n \geq N.$$
Thus $\|f - f_n\|_\infty \leq \varepsilon$.

*Continuity of $f$.* Fix $x_0 \in X$ and $\varepsilon > 0$. Pick $n \geq N$ with $\|f - f_n\|_\infty < \varepsilon/3$; by continuity of $f_n$, $\exists \delta > 0$ with $|f_n(x) - f_n(x_0)| < \varepsilon/3$ for $d(x, x_0) < \delta$. Then
$$|f(x) - f(x_0)| \leq |f(x) - f_n(x)| + |f_n(x) - f_n(x_0)| + |f_n(x_0) - f(x_0)| < \varepsilon.$$
$\blacksquare$

### Completion of Metric Spaces

Every metric space $X$ embeds isometrically into a complete metric space $\tilde X$ as a dense subset; $\tilde X$ is called the **completion** of $X$. Morally, $\mathbb{R}$ is the completion of $\mathbb{Q}$; $L^p$ is the completion of step functions in the $p$-metric.

**Theorem 0.4.2.9 (Completion theorem).** *Every metric space $(X, d)$ has a completion $(\tilde X, \tilde d)$: a complete metric space and an isometric embedding $\iota: X \to \tilde X$ with $\iota(X)$ dense in $\tilde X$. The completion is unique up to isometry.*

**Proof (sketch).** Define $\tilde X$ as the set of equivalence classes of Cauchy sequences in $X$: $(x_n) \sim (y_n)$ iff $d(x_n, y_n) \to 0$. Define $\tilde d([x_n], [y_n]) = \lim d(x_n, y_n)$ (the limit exists and is well-defined using triangle inequality). The inclusion $x \mapsto [(x, x, x, \ldots)]$ embeds $X$ isometrically into $\tilde X$.

*Denseness.* Given any Cauchy equivalence class $[x_n]$, the constant sequences $[(x_k, x_k, \ldots)]$ (for large $k$) are in $\iota(X)$ and converge in $\tilde d$ to $[x_n]$.

*Completeness.* A Cauchy sequence of equivalence classes can be approximated by a Cauchy "diagonal" sequence of representatives, and one shows its equivalence class is the limit.

*Uniqueness up to isometry.* If $(\tilde X_1, \iota_1)$ and $(\tilde X_2, \iota_2)$ are two completions, the isometric identification $\iota_2 \circ \iota_1^{-1}$ on $\iota_1(X)$ extends uniquely to an isometry of the completions (extending a uniformly continuous map from a dense subset of a complete metric space to its completion is automatic). $\blacksquare$

### The Banach Fixed-Point Theorem (Contraction Mapping Principle)

This is the workhorse theorem built on completeness. We proved it in Module 0.3 (Topic 0.3.3, en route to IFT); here we elevate it to the central tool of this topic.

**Theorem 0.4.2.10 (Banach fixed-point theorem).** *Let $(X, d)$ be a non-empty complete metric space and $T: X \to X$ a **contraction**: there exists $L \in [0, 1)$ with $d(T(x), T(y)) \leq L\, d(x, y)$ for all $x, y \in X$. Then:*

- *$T$ has a unique fixed point $x^* \in X$;*
- *for every $x_0 \in X$, the iteration $x_{n+1} = T(x_n)$ converges to $x^*$;*
- *error estimate: $d(x_n, x^*) \leq \frac{L^n}{1 - L} d(x_1, x_0)$.*

**Proof.** *Uniqueness.* If $T(x^*) = x^*$ and $T(y^*) = y^*$, then $d(x^*, y^*) = d(T(x^*), T(y^*)) \leq L\, d(x^*, y^*)$, so $(1 - L) d(x^*, y^*) \leq 0$; since $L < 1$, $d(x^*, y^*) = 0$.

*Existence via iteration.* Let $x_0 \in X$ be arbitrary; $x_{n+1} = T(x_n)$. By induction, $d(x_{n+1}, x_n) \leq L^n d(x_1, x_0)$. For $m > n$, use the triangle inequality:
$$d(x_m, x_n) \leq \sum_{k=n}^{m-1} d(x_{k+1}, x_k) \leq \sum_{k=n}^{m-1} L^k d(x_1, x_0) \leq \frac{L^n}{1 - L} d(x_1, x_0).$$

This tends to $0$ as $n \to \infty$, so $(x_n)$ is Cauchy. By completeness, $x_n \to x^*$ for some $x^* \in X$.

*$x^*$ is a fixed point.* $T$ is continuous (Lipschitz with constant $L < 1$), so $T(x_n) \to T(x^*)$; but $T(x_n) = x_{n+1} \to x^*$ as well, giving $T(x^*) = x^*$.

*Error bound.* Let $m \to \infty$ in $d(x_m, x_n) \leq L^n d(x_1, x_0) / (1 - L)$. $\blacksquare$

**Application preview.** Picard-Lindelöf ODE existence: the solution operator on $C([0, h], \mathbb{R}^n)$ is a contraction for small $h$; Banach gives the unique solution.

### Baire Category Theorem

**Theorem 0.4.2.11 (Baire category theorem).** *Let $(X, d)$ be a complete metric space. If $(U_n)_{n \in \mathbb{N}}$ is a countable family of open dense subsets, then $\bigcap_n U_n$ is dense.*

*Equivalently:* $X$ is *not* a countable union of closed nowhere-dense subsets.

**Proof.** Let $V \subseteq X$ be a non-empty open set; we will show $V \cap \bigcap_n U_n \neq \varnothing$. Inductively construct nested closed balls $\overline{B_n}$ with:

- $\overline{B_1} \subseteq V \cap U_1$, radius $< 1$;
- $\overline{B_{n+1}} \subseteq B_n \cap U_{n+1}$, radius $< 1/(n+1)$.

Such balls exist because $U_1$ is dense in $X$ (so $V \cap U_1$ is non-empty open, and contains a closed ball of small radius), and $U_{n+1}$ is dense in $X$ (so $B_n \cap U_{n+1}$ is non-empty open).

The centers $x_n$ of the balls form a Cauchy sequence (since $x_n, x_m \in \overline{B_n}$ for $m \geq n$, so $d(x_n, x_m) < 2/n$). By completeness, $x_n \to x^*$. Since $x^* \in \overline{B_n} \subseteq U_n$ for every $n$ (the balls are nested and closed) and $x^* \in \overline{B_1} \subseteq V$, we have $x^* \in V \cap \bigcap_n U_n$. $\blacksquare$

**Definition 0.4.2.12 (Category).** In a topological space:

- A **nowhere-dense** set is one whose closure has empty interior.
- A **meager** (or **first-category**) set is a countable union of nowhere-dense sets.
- A set that is not meager is **non-meager** (or of **second category**).
- A **residual** (or **comeager**) set is the complement of a meager set.

Baire's theorem says: *in a complete metric space, the whole space is of second category* — it is not meager. Equivalently, residual sets are dense, so properties that hold on residual sets are "generic."

**Consequences.**

- **Banach-Steinhaus / Uniform Boundedness Principle.** A pointwise-bounded family of continuous linear operators between Banach spaces is uniformly bounded. (Pivotal in functional analysis.)
- **Open Mapping Theorem.** A surjective bounded linear map between Banach spaces is open.
- **Closed Graph Theorem.** A linear map between Banach spaces with closed graph is continuous.

These three theorems are the "three pillars" of linear functional analysis; they rest on Baire. We will prove the uniform boundedness principle after Topic 0.4.3 (compactness); full details belong in a functional analysis module.

**Example 0.4.2.13 (Baire + density).** The set of irrationals $\mathbb{R} \setminus \mathbb{Q}$ is residual in $\mathbb{R}$. Indeed, $\mathbb{Q}$ is countable, hence a countable union of singletons (nowhere dense in $\mathbb{R}$), hence meager. So $\mathbb{R} \setminus \mathbb{Q} = \bigcap_q (\mathbb{R} \setminus \{q\})$ is an intersection of open dense sets — Baire says it's dense.

### Nowhere-Differentiable Functions

An impressive Baire-theoretic result: **most** continuous functions on $[0, 1]$ are nowhere-differentiable. Formally, the set of nowhere-differentiable functions is residual in $C([0, 1])$ equipped with the sup metric.

**Theorem 0.4.2.14 (Baire-generic nowhere differentiability).** *In $C([0, 1])$, the set $\{f : f \text{ is differentiable at some } t\}$ is meager.*

**Proof sketch.** For each $n$, let
$$F_n = \{f \in C[0, 1] : \exists t_0 \in [0, 1 - 1/n] \text{ with } |f(t) - f(t_0)| \leq n|t - t_0| \text{ for all } t \in [t_0, t_0 + 1/n]\}.$$
Differentiable functions (at some point) lie in $\bigcup_n F_n$. One shows (i) each $F_n$ is closed in $C[0, 1]$, (ii) each $F_n$ has empty interior (a small perturbation by a sawtooth function kicks you out of $F_n$). So $\bigcup_n F_n$ is meager. $\blacksquare$

This gives a measure-theoretic flavor to *genericity* of analytic pathology — fitting preparation for Brownian motion, whose paths are a.s. nowhere differentiable (a probabilistic genericity complementing Baire's topological one).

### Worked Examples

#### Example 0.4.2.15 — $\mathbb{Q}$ is not complete

Define recursively $x_1 = 1$, $x_{n+1} = (x_n + 2/x_n)/2 \in \mathbb{Q}$. Each $x_n$ rational (ratio of rational). $(x_n)$ converges in $\mathbb{R}$ to $\sqrt 2$ (Newton's iteration), hence is Cauchy in $\mathbb{R}$ and thus in $\mathbb{Q}$. But $\sqrt 2 \notin \mathbb{Q}$, so no limit in $\mathbb{Q}$.

#### Example 0.4.2.16 — $C[0,1]$ with $L^2$-norm is not complete

Let $f_n(t) = \min(n, 1/\sqrt{t})$ for $t > 0$, $f_n(0) = n$. Each $f_n \in C[0, 1]$. In $L^2$-norm,
$$\int_0^1 |f_n - f_m|^2\, dt \to 0 \text{ as } m, n \to \infty$$
(Cauchy). But the pointwise limit $f(t) = 1/\sqrt t$ is not in $C[0, 1]$ (it blows up at $0$). So $(f_n)$ has no $C[0,1]$-limit in $L^2$; $C[0,1]$ under $L^2$-norm is incomplete. The completion (via measure theory) is $L^2[0, 1]$ — a proper superset of $C[0, 1]$.

#### Example 0.4.2.17 — Picard iteration for $y' = y$

Solve $y' = y$, $y(0) = 1$ by Banach iteration. Rewrite as $y(t) = 1 + \int_0^t y(s)\, ds =: T(y)(t)$.

On $X = C([0, T], \mathbb{R})$ with sup norm, $T$ maps $X \to X$, and
$$|T(y)(t) - T(z)(t)| \leq \int_0^t |y(s) - z(s)|\, ds \leq T \|y - z\|_\infty,$$
so $T$ is a contraction for $T < 1$ with Lipschitz constant $T$.

Start with $y_0 \equiv 1$:
- $y_1(t) = 1 + t$,
- $y_2(t) = 1 + t + t^2/2$,
- $y_n(t) = \sum_{k=0}^n t^k/k!$,

which converges uniformly on $[0, T]$ to $e^t$. Banach confirms $y(t) = e^t$ is the unique fixed point, i.e., the unique solution.

#### Example 0.4.2.18 — Banach iteration for Bellman equation

In dynamic programming, the **Bellman operator** $T: V \mapsto \max_a [r(s, a) + \gamma \mathbb{E} V(s')]$ acts on value functions $V: \mathcal{S} \to \mathbb{R}$ (for states $\mathcal{S}$ finite or continuous). If $0 \leq \gamma < 1$ (discount factor), then $T$ is a contraction with constant $\gamma$ in the sup norm:
$$\|TV - TV'\|_\infty \leq \gamma \|V - V'\|_\infty.$$
Banach fixed-point gives a unique optimal value function $V^*$ with $V^* = TV^*$, and value iteration $V_{k+1} = TV_k$ converges geometrically. This is the theoretical backbone of reinforcement learning.

### Computational Implementation

```python
import numpy as np

# ---------- Newton / Banach iteration for sqrt 2 ----------
x = 1.0
for k in range(10):
    x = 0.5 * (x + 2/x)
    print(f"Step {k+1}: x = {x:.15f}, |x^2 - 2| = {abs(x**2 - 2):.2e}")
print()

# ---------- Picard iteration for y' = y, y(0) = 1 ----------
import sympy as sp
t = sp.Symbol('t')
y = sp.Integer(1)  # start at y_0(t) = 1
for k in range(6):
    y = 1 + sp.integrate(y, (t, 0, t))
    print(f"y_{k+1}(t) = {sp.series(y, t, 0, 7).removeO()}")
print()

# ---------- Bellman iteration for a 2-state MDP ----------
# States: 0, 1; Actions: 'stay', 'move'; Rewards: r, Transitions: P
gamma = 0.9
# r[s, a], P[s, a, s']
r = np.array([[0, 1], [1, 0]])  # reward for (state, action)
P = np.zeros((2, 2, 2))
P[0, 0, 0] = 1.0  # stay in state 0
P[0, 1, 1] = 1.0  # move from 0 to 1
P[1, 0, 1] = 1.0  # stay in state 1
P[1, 1, 0] = 1.0  # move from 1 to 0

V = np.zeros(2)
for k in range(50):
    V_new = np.zeros(2)
    for s in range(2):
        V_new[s] = np.max([r[s, a] + gamma * np.sum(P[s, a, :] * V) for a in range(2)])
    diff = np.max(np.abs(V_new - V))
    V = V_new
    if k % 10 == 0 or diff < 1e-10:
        print(f"Step {k+1}: V = {V}, sup change = {diff:.3e}")
    if diff < 1e-12: break

print(f"\nBellman fixed point: V* = {V}")
print(f"Analytical: V*(0) = V*(1) = 1/(1 - gamma) = {1/(1-gamma):.4f}? Actually it's 10 since we alternate.")
print(f"Check: r + gamma V = {[np.max([r[s, a] + gamma*np.sum(P[s,a,:]*V) for a in range(2)]) for s in range(2)]}")

# ---------- Demonstration that Banach iteration fails if L = 1 ----------
# T(x) = x + 1 has no fixed point but d(T(x), T(y)) = d(x, y): L = 1, not < 1.
x = 0.0
for k in range(5):
    x = x + 1
    print(f"L = 1, iter {k+1}: x = {x} (diverges; no fixed point)")

# ---------- Approximation of nowhere-differentiable function (Weierstrass) ----------
import matplotlib  # noqa  (only needed if plotting)
def weierstrass(x, N=100, a=0.5, b=7):
    """W(x) = sum_{n=0}^{N-1} a^n cos(b^n pi x), nowhere differentiable if ab > 1 + 3pi/2."""
    n = np.arange(N)
    return np.sum(a**n * np.cos(b**n * np.pi * x[:, None]), axis=1)

xs = np.linspace(0, 1, 1000)
W = weierstrass(xs)
print(f"\nWeierstrass function: range = [{W.min():.4f}, {W.max():.4f}]")
print(f"Continuous but nowhere differentiable (generic by Baire theorem).")

# ---------- Cauchy-but-not-convergent sequence in C[0,1] with L^1 norm ----------
# f_n(t) = step function approximation sharpening around t = 0.5
def f_n(t, n):
    return 1.0 / (1.0 + np.exp(-n * (t - 0.5)))  # sigmoid sharpening to step at 0.5

ts = np.linspace(0, 1, 1001)
for n in [5, 20, 50]:
    print(f"  f_{n}(0.4) = {f_n(0.4, n):.4f}, f_{n}(0.6) = {f_n(0.6, n):.4f}")
# L^1 distance between f_n and f_m shrinks; pointwise limit is discontinuous step.
L1_10_100 = np.trapezoid(np.abs(f_n(ts, 10) - f_n(ts, 100)), ts)
print(f"  ||f_10 - f_100||_1 = {L1_10_100:.4f}  (small; they are Cauchy in L^1)")
```

### [QUANT APPLICATION] Completeness in Quant Algorithms

**(A) Picard-Lindelöf for ODE-based models.** Short-rate models (Vasicek, CIR) are ODEs/SDEs; existence of solutions requires completeness of the function space. The Picard-Lindelöf construction (Banach fixed-point on integral-equation form) is the constructive existence proof.

**(B) Bellman / value iteration.** Every dynamic programming algorithm — from Merton's optimal consumption to deep Q-networks — is a Banach fixed-point iteration on a function space. Discount $\gamma < 1$ ensures contraction; completeness of the value-function space ensures the limit exists.

**(C) Black-Scholes PDE and fixed-point methods.** Solving the Black-Scholes PDE with free boundary (American options) uses iterative schemes (e.g., penalty methods) that converge by contraction arguments on complete spaces of candidate prices. Completeness of $L^2$ plus projection gives optimal stopping solutions.

**(D) Kalman filter convergence.** The Riccati equation for the covariance in the Kalman filter has a fixed-point form $P_{\infty} = AP_\infty A^\top + Q - AP_\infty H^\top (HP_\infty H^\top + R)^{-1} HP_\infty A^\top$ (discrete) which is a contraction under detectability/stabilizability conditions; the steady-state covariance is a Banach fixed-point in the space of positive semidefinite matrices.

**(E) Neural tangent kernel convergence.** At the overparametrized limit, gradient descent on wide neural networks converges to the fixed point of a contraction in function space (the NTK regime). Jacot-Gabriel-Hongler (2018) proved this using completeness of $L^2$ and a generalized Banach argument.

**(F) Risk-neutral measure existence.** Constructed as an equivalent martingale measure via Radon-Nikodym derivative; the Radon-Nikodym theorem relies on the completeness of $L^1$ and Hilbert projection (completeness of $L^2$).

**(G) Cubature formulas for Bayesian inference.** Many sequential Monte Carlo / particle filter schemes are iterative Bayesian update operators. Convergence of the particle approximation to the true posterior uses contraction arguments in the Wasserstein-metric-complete space of probability measures.

**(H) Robustness to floating-point errors.** Numerical stability of iterative algorithms (Newton, Krylov, SVD updates) is analyzed via perturbation bounds that rely on contraction properties in a complete normed space — the rounding errors are bounded and the iteration stays close to the exact path.

### Exercises

#### ★ (Foundation)

**E0.4.2.1.** Prove that every discrete metric space is complete.

**E0.4.2.2.** Show that if $(X, d)$ is complete, so is $X \times Y$ with product metric $d_{X \times Y}((x_1, y_1), (x_2, y_2)) = d_X(x_1, x_2) + d_Y(y_1, y_2)$, provided $(Y, d_Y)$ is complete.

**E0.4.2.3.** Prove that $\ell^\infty$ (bounded sequences with sup norm) is a Banach space.

**E0.4.2.4 (Fixed point of affine contraction).** Show that $T(x) = ax + b$ on $\mathbb{R}$ is a contraction iff $|a| < 1$; find the fixed point explicitly.

**E0.4.2.5.** Prove: if $T: X \to X$ is a contraction with constant $L$ and $T^k$ (the $k$-fold composition) has a fixed point $x^*$, then so does $T$.

#### ★★ (Intermediate)

**E0.4.2.6 (Completeness of $\ell^p$).** Prove that $\ell^p$ for $1 \leq p < \infty$ is complete.

**E0.4.2.7 (Complete iff closed in completion).** Show that $X$ is complete iff $X = \tilde X$ under the embedding $\iota$ into the completion.

**E0.4.2.8 (Picard-Lindelöf in full generality).** Let $f: [0, T] \times \mathbb{R}^n \to \mathbb{R}^n$ be continuous and $L$-Lipschitz in the second argument. Use Banach fixed-point on $C([0, T], \mathbb{R}^n)$ to prove existence and uniqueness of a solution to $y'(t) = f(t, y(t))$, $y(0) = y_0$. (Hint: iterate with weighted sup norm $\|y\|_\lambda = \sup_t e^{-\lambda t}|y(t)|$ for large $\lambda$.)

**E0.4.2.9 (Baire corollary).** Show: in a complete metric space, the complement of a countable union of closed sets with empty interior is dense.

**E0.4.2.10 (Baire applied to polynomials).** Prove: any continuous $f: \mathbb{R} \to \mathbb{R}$ that agrees with *some* polynomial on each point (i.e., for each $x$ there is a polynomial $P_x$ and a neighborhood of $x$ on which $f = P_x$) is globally a polynomial.

#### ★★★ (Challenge)

**E0.4.2.11 (Banach-Steinhaus).** Prove: if $\{T_\alpha\}$ is a family of bounded linear operators between Banach spaces $X \to Y$ with $\sup_\alpha \|T_\alpha x\| < \infty$ for each $x \in X$, then $\sup_\alpha \|T_\alpha\|_{\mathrm{op}} < \infty$. Use Baire on the sets $E_n = \{x : \sup_\alpha \|T_\alpha x\| \leq n\}$.

**E0.4.2.12 (Open Mapping Theorem).** Prove: a surjective bounded linear operator between Banach spaces is open. Use Baire.

**E0.4.2.13 (Closed Graph Theorem).** Prove: a linear operator $T: X \to Y$ between Banach spaces is continuous iff its graph $\{(x, Tx)\}$ is closed in $X \times Y$.

**E0.4.2.14 (Nowhere-differentiable generic).** Prove Theorem 0.4.2.14 in full (continuous functions are generically nowhere differentiable).

**E0.4.2.15 (Completion of $C[0, 1]$ in $L^p$ is $L^p$).** Show that the completion of $(C[0, 1], \|\cdot\|_{L^p})$ is (isometric to) $L^p[0, 1]$. This is the abstract route to constructing $L^p$ without measure theory — but the measure-theoretic construction is more natural and gives more tools. (Forward reference to Subject 1.)

**E0.4.2.16 (Meager set of everywhere-unbounded sequences).** In $\ell^\infty$, show that the set of sequences $(x_n)$ whose "oscillation" at infinity exceeds $1$ is meager. Interpret.

---


## Topic 0.4.3 — Compactness

### Motivation

Compactness is the topological abstraction of "finite" — a compact set is one that behaves like a finite set for many analytical purposes. The extreme value theorem (continuous functions attain max/min) and the Bolzano-Weierstrass theorem (every bounded sequence has a convergent subsequence) are compactness manifestations; they fail on non-compact sets.

For quantitative work, compactness is *the* guarantee that optimizations have solutions:
- An objective continuous on a compact set attains its minimum.
- A probability distribution on a compact state space has a normalizing constant (finite total mass).
- A family of uniformly bounded, equicontinuous functions has a uniformly convergent subsequence (Arzela-Ascoli) — this is the basis of weak compactness arguments in calculus of variations, infinite-horizon control, and PDE existence proofs.

In finite dimensions, compactness boils down to Heine-Borel: compact = closed and bounded. In infinite dimensions, the connection is subtler — the closed unit ball of an infinite-dimensional Banach space is *not* compact — and one needs Arzela-Ascoli or Rellich-Kondrachov-type theorems.

### Prerequisites

- Metric spaces (Topic 0.4.1), Cauchy sequences and completeness (Topic 0.4.2).
- Finite/infinite set distinctions (Topic 0.1.6).

### Three Definitions of Compactness

There are three notions of compactness used in metric spaces, and they are equivalent.

**Definition 0.4.3.1.** Let $(X, d)$ be a metric space.

- (OC) **Compact** (open-cover): every open cover $\{U_\alpha\}_{\alpha \in A}$ of $X$ has a finite subcover (there exist $\alpha_1, \ldots, \alpha_N$ with $X \subseteq \bigcup_i U_{\alpha_i}$).
- (SC) **Sequentially compact**: every sequence $(x_n)$ in $X$ has a convergent subsequence (with limit in $X$).
- (TB) **Totally bounded and complete**: $X$ is complete, and for every $\varepsilon > 0$, $X$ is covered by finitely many balls of radius $\varepsilon$.

A set $A \subseteq X$ is compact / sequentially compact / etc. iff $A$ is so when regarded as a metric space in its own right (with induced metric).

**Theorem 0.4.3.2.** *For a metric space $(X, d)$, the following are equivalent: (OC), (SC), (TB).*

**Proof.** We prove (OC) ⇒ (SC) ⇒ (TB) ⇒ (OC).

**(OC) ⇒ (SC).** Suppose $(x_n)$ has no convergent subsequence. Then for each $x \in X$, $x$ is not a limit of any subsequence, so there is an $\varepsilon_x > 0$ and $N_x \in \mathbb{N}$ with $x_n \notin B_{\varepsilon_x}(x)$ for $n \geq N_x$ *except possibly finitely many $n$*. (More precisely, $B_{\varepsilon_x}(x) \cap \{x_n\}$ is finite.)

The open cover $\{B_{\varepsilon_x}(x) : x \in X\}$ has a finite subcover $B_{\varepsilon_1}(y_1), \ldots, B_{\varepsilon_N}(y_N)$ by (OC). But each of these balls contains only finitely many of the $x_n$, so $\bigcup_i B_{\varepsilon_i}(y_i)$ contains only finitely many $x_n$ — contradicting $X = \bigcup_i B_{\varepsilon_i}(y_i)$ and $(x_n) \subseteq X$ infinite (or if $(x_n)$ is eventually constant, take a constant subsequence and we're done).

**(SC) ⇒ (TB).** *Complete:* if $(x_n)$ is Cauchy, (SC) gives a convergent subsequence $x_{n_k} \to x^*$; Cauchy + convergent subsequence ⇒ convergent (as in Theorem 0.4.2.3 step 4).

*Totally bounded:* suppose not. Then some $\varepsilon > 0$ admits no finite cover by $\varepsilon$-balls. Inductively construct $(x_n)$ with $d(x_n, x_m) \geq \varepsilon$ for $m \neq n$: choose $x_1$ arbitrary; given $x_1, \ldots, x_n$, by failure of finite covering there is $x_{n+1} \notin \bigcup_i B_\varepsilon(x_i)$, so $d(x_{n+1}, x_i) \geq \varepsilon$. No subsequence of $(x_n)$ can be Cauchy (since any two terms are $\geq \varepsilon$ apart), contradicting (SC).

**(TB) ⇒ (OC).** Let $\mathcal{U} = \{U_\alpha\}$ be an open cover. Suppose no finite subcollection covers $X$. We derive a contradiction using total boundedness + completeness.

*Construction.* For $n = 1, 2, \ldots$, use total boundedness: $X$ is covered by finitely many $1/n$-balls. At least one of these balls, call it $B_n$, cannot be covered by finitely many $U_\alpha$'s (otherwise finitely many suffice to cover $X$). Moreover, we can arrange $B_n \cap B_{n-1} \neq \varnothing$ by choosing $B_{n}$ inside $B_{n-1}$: replace "cover $X$ by $1/n$-balls" with "cover $B_{n-1}$ by $1/n$-balls" at the inductive step.

Pick any $x_n \in B_n$. Then $d(x_n, x_{n-1}) \leq 2/n \to 0$, so $(x_n)$ is Cauchy. By completeness, $x_n \to x^*$. Pick $\alpha_0$ with $x^* \in U_{\alpha_0}$; openness gives $r > 0$ with $B_r(x^*) \subseteq U_{\alpha_0}$. For $n$ large enough, $B_n \subseteq B_r(x^*) \subseteq U_{\alpha_0}$, so $B_n$ is covered by a single $U_{\alpha_0}$ — contradicting its un-coverability. $\blacksquare$

### Heine-Borel Theorem

**Theorem 0.4.3.3 (Heine-Borel in $\mathbb{R}^n$).** *A subset $A \subseteq \mathbb{R}^n$ is compact iff it is closed and bounded.*

**Proof.** *($\Rightarrow$)* Compact metric spaces are complete (from (TB)), hence closed (as subspaces of complete $\mathbb{R}^n$; compactness implies closedness). Compact ⇒ totally bounded ⇒ bounded.

*($\Leftarrow$)* Suppose $A$ is closed and bounded; $A \subseteq [-M, M]^n$ for some $M$. The closed cube $[-M, M]^n$ is sequentially compact: a sequence in the cube is bounded, hence has a subsequence converging coordinate-wise (Bolzano-Weierstrass in $\mathbb{R}$, applied $n$ times). The limit lies in the closed cube. So the cube is (SC) hence compact.

$A$ is closed in $\mathbb{R}^n$, hence closed in $[-M, M]^n$, hence a closed subset of a compact space. Any sequence in $A$ has a convergent subsequence (in the cube) whose limit, by closedness of $A$, lies in $A$. So $A$ is (SC), hence compact. $\blacksquare$

**Bolzano-Weierstrass.** An immediate corollary: every bounded sequence in $\mathbb{R}^n$ has a convergent subsequence.

### Extreme Value Theorem

**Theorem 0.4.3.4 (Extreme value theorem).** *Let $(X, d_X)$ and $(Y, d_Y)$ be metric spaces, $X$ compact, $f: X \to Y$ continuous. Then $f(X)$ is compact in $Y$. In particular, if $Y = \mathbb{R}$, $f$ attains its supremum and infimum.*

**Proof.** We show (SC). Let $(y_n) \subseteq f(X)$; pick $x_n \in X$ with $f(x_n) = y_n$. By compactness (SC) of $X$, a subsequence $x_{n_k} \to x^* \in X$. By continuity, $y_{n_k} = f(x_{n_k}) \to f(x^*) \in f(X)$. So $(y_n)$ has a convergent subsequence; $f(X)$ is (SC), hence compact.

When $Y = \mathbb{R}$, $f(X)$ is compact in $\mathbb{R}$, hence closed and bounded; a closed bounded non-empty subset of $\mathbb{R}$ contains its supremum and infimum. $\blacksquare$

### Uniform Continuity on Compact Sets

**Definition 0.4.3.5 (Uniform continuity).** $f: (X, d_X) \to (Y, d_Y)$ is **uniformly continuous** iff
$$\forall \varepsilon > 0 \; \exists \delta > 0 \; \forall x, x' \in X: \; d_X(x, x') < \delta \Rightarrow d_Y(f(x), f(x')) < \varepsilon.$$
Note that $\delta$ does *not* depend on $x$, unlike ordinary continuity.

**Example.** $f(x) = x^2$ is continuous on $\mathbb{R}$ but not uniformly continuous (large $x$ makes derivative large, so $\delta$ must shrink). It is uniformly continuous on any bounded set.

**Theorem 0.4.3.6 (Uniform continuity on compact domain).** *A continuous function from a compact metric space $X$ to any metric space is uniformly continuous.*

**Proof.** Fix $\varepsilon > 0$. For each $x \in X$, continuity at $x$ gives $\delta_x > 0$ with $f(B_{\delta_x}(x)) \subseteq B_{\varepsilon/2}(f(x))$.

The collection $\{B_{\delta_x/2}(x) : x \in X\}$ is an open cover of $X$; compactness gives a finite subcover $B_{\delta_{x_1}/2}(x_1), \ldots, B_{\delta_{x_N}/2}(x_N)$. Let $\delta = \tfrac{1}{2} \min_i \delta_{x_i} > 0$.

Suppose $d(x, x') < \delta$. Pick $i$ with $x \in B_{\delta_{x_i}/2}(x_i)$; then $d(x, x_i) < \delta_{x_i}/2$, and $d(x', x_i) \leq d(x', x) + d(x, x_i) < \delta + \delta_{x_i}/2 \leq \delta_{x_i}$. Thus $x, x' \in B_{\delta_{x_i}}(x_i)$, so
$$d(f(x), f(x')) \leq d(f(x), f(x_i)) + d(f(x_i), f(x')) < \varepsilon/2 + \varepsilon/2 = \varepsilon. \qquad \blacksquare$$

### Finite-Dimensional Normed Spaces

**Theorem 0.4.3.7 (All norms on $\mathbb{R}^n$ are equivalent).** *Any two norms on a finite-dimensional real vector space induce the same topology, hence are Lipschitz equivalent.*

**Proof.** Let $\|\cdot\|, \|\cdot\|'$ be two norms on $\mathbb{R}^n$ (WLOG work in a basis). The map $N: (\mathbb{R}^n, \|\cdot\|_2) \to \mathbb{R}$, $N(\mathbf{x}) = \|\mathbf{x}\|$, is continuous:
$$|N(\mathbf{x}) - N(\mathbf{y})| \leq \|\mathbf{x} - \mathbf{y}\| \leq C_1 \|\mathbf{x} - \mathbf{y}\|_2,$$
where $C_1 = \sum_i \|\mathbf{e}_i\|$ (from the triangle inequality applied to $\mathbf{x} - \mathbf{y} = \sum (x_i - y_i)\mathbf{e}_i$). Similarly $N'$ is continuous.

The unit sphere $S = \{\mathbf{x} : \|\mathbf{x}\|_2 = 1\}$ is closed and bounded, hence compact by Heine-Borel. The continuous $N$ attains its min $m > 0$ (positive because $\|\mathbf{x}\|_2 = 1 \Rightarrow \mathbf{x} \neq 0 \Rightarrow N(\mathbf{x}) > 0$) and max $M < \infty$ on $S$. For any $\mathbf{x} \neq 0$, $\mathbf{x}/\|\mathbf{x}\|_2 \in S$ so $m \leq N(\mathbf{x}/\|\mathbf{x}\|_2) \leq M$, giving
$$m \|\mathbf{x}\|_2 \leq \|\mathbf{x}\| \leq M \|\mathbf{x}\|_2.$$

Apply the same argument to $N'$ to get $m' \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|' \leq M' \|\mathbf{x}\|_2$. Combining:
$$\frac{m}{M'} \|\mathbf{x}\|' \leq \|\mathbf{x}\| \leq \frac{M}{m'} \|\mathbf{x}\|'. \qquad \blacksquare$$

**Corollary 0.4.3.8.** Every finite-dimensional normed vector space is a Banach space and is linearly homeomorphic to $\mathbb{R}^n$.

**Corollary 0.4.3.9 (Riesz's lemma / Infinite-dimensional ball is non-compact).** *The closed unit ball of an infinite-dimensional normed space is not compact.*

**Proof of Corollary 0.4.3.9.** *Riesz's lemma:* in any proper closed subspace $Y \subsetneq X$ and $\varepsilon > 0$, there is $x_\varepsilon \in X$ with $\|x_\varepsilon\| = 1$ and $d(x_\varepsilon, Y) > 1 - \varepsilon$. (Proof: pick $x \notin Y$, let $\delta = d(x, Y) > 0$; there's $y_0 \in Y$ with $\|x - y_0\| \leq \delta/(1 - \varepsilon)$. Set $x_\varepsilon = (x - y_0)/\|x - y_0\|$; for any $y \in Y$, $\|x_\varepsilon - y\| = \|x - y_0\|^{-1} \|x - (y_0 + \|x - y_0\| y)\| \geq \|x - y_0\|^{-1} \delta \geq 1 - \varepsilon$.)

In an infinite-dimensional space, iteratively choose unit vectors $x_1, x_2, \ldots$ with $x_{n+1}$ at distance $> 1/2$ from $\mathrm{span}\{x_1, \ldots, x_n\}$ (Riesz with $\varepsilon = 1/2$). The sequence $(x_n)$ lies in the unit ball but $\|x_m - x_n\| \geq 1/2$ for $m \neq n$, so no convergent subsequence; unit ball is not (SC). $\blacksquare$

This is a big deal: in infinite dimensions, "bounded" is much weaker than "has compact closure." We need additional hypotheses (equicontinuity, tightness) to recover compactness in function spaces.

### Arzelà-Ascoli Theorem

The decisive compactness theorem in function spaces.

**Definition 0.4.3.10 (Equicontinuity).** A family $\mathcal{F} \subseteq C(X, Y)$ is **equicontinuous at $x_0$** iff
$$\forall \varepsilon > 0 \; \exists \delta > 0: \; d_X(x, x_0) < \delta \Rightarrow d_Y(f(x), f(x_0)) < \varepsilon \quad \forall f \in \mathcal{F}.$$
$\mathcal{F}$ is **equicontinuous** iff equicontinuous at every $x_0 \in X$.

$\mathcal{F}$ is **uniformly equicontinuous** iff the same $\delta$ works for all $x_0$ (so the $\delta$ depends only on $\varepsilon$, not on $f$ *or* on $x_0$).

**Example.** A family of functions with $\|f\|_\infty \leq 1$ and $\|f'\|_\infty \leq 10$ on $[0, 1]$ is equicontinuous with $\delta = \varepsilon/10$.

**Theorem 0.4.3.11 (Arzelà-Ascoli).** *Let $(X, d_X)$ be a compact metric space and $(f_n) \subseteq C(X, \mathbb{R}^m)$. Suppose*

- *$(f_n)$ is **pointwise bounded**: for each $x$, $\sup_n \|f_n(x)\| < \infty$.*
- *$(f_n)$ is **equicontinuous**.*

*Then there is a subsequence $(f_{n_k})$ that converges uniformly on $X$ to some $f \in C(X, \mathbb{R}^m)$.*

*Equivalently:* a closed, bounded, equicontinuous family in $C(X, \mathbb{R}^m)$ is compact.

**Proof.** We use (SC) characterization of compactness.

*Step 1: Uniform equicontinuity.* $X$ compact + pointwise equicontinuity ⇒ uniform equicontinuity (same proof as Theorem 0.4.3.6). So for each $k \in \mathbb{N}$, there is $\delta_k > 0$ with $d_X(x, x') < \delta_k \Rightarrow \|f_n(x) - f_n(x')\| < 1/k$ for all $n$.

*Step 2: Dense countable subset.* $X$ is totally bounded (by (TB) characterization); cover $X$ by finitely many $1/k$-balls $\{B_{1/k}(y_i^k)\}$ for each $k$; the union $D = \bigcup_k \{y_i^k\}$ is a countable dense subset of $X$.

*Step 3: Diagonal subsequence for pointwise convergence.* Enumerate $D = \{y_1, y_2, \ldots\}$. $(f_n(y_1))$ is bounded in $\mathbb{R}^m$; by Bolzano-Weierstrass, extract a subsequence $(f_n^{(1)})$ with $f_n^{(1)}(y_1)$ convergent. From $(f_n^{(1)})$, extract $(f_n^{(2)})$ with $f_n^{(2)}(y_2)$ convergent (note $f_n^{(2)}(y_1)$ still converges too). Iterate. The diagonal $(g_n) = (f_n^{(n)})$ converges at every $y_k$: for $n \geq k$, $g_n \in (f_n^{(k)})$, so $g_n(y_k) \to \lim_m f_m^{(k)}(y_k)$.

Define $g(y_k) = \lim_n g_n(y_k)$ for each $k$.

*Step 4: Uniform convergence of $(g_n)$ on $X$.* Fix $\varepsilon > 0$; pick $k$ with $1/k < \varepsilon/3$, and use the $\delta_k$-$1/k$ equicontinuity from Step 1. Cover $X$ by finitely many $\delta_k$-balls around points $y_{j_1}, \ldots, y_{j_M} \in D$. Pointwise convergence at each $y_{j_i}$ gives $N$ with $\|g_n(y_{j_i}) - g_m(y_{j_i})\| < \varepsilon/3$ for $m, n \geq N$, $i = 1, \ldots, M$.

For any $x \in X$, pick $y_{j_i}$ with $d(x, y_{j_i}) < \delta_k$; then
$$\|g_n(x) - g_m(x)\| \leq \|g_n(x) - g_n(y_{j_i})\| + \|g_n(y_{j_i}) - g_m(y_{j_i})\| + \|g_m(y_{j_i}) - g_m(x)\| < 1/k + \varepsilon/3 + 1/k < \varepsilon.$$
So $(g_n)$ is Cauchy in sup-norm on $X$. Since $C(X, \mathbb{R}^m)$ is complete (Proposition 0.4.2.8), $(g_n) \to f$ uniformly for some continuous $f$. $\blacksquare$

**Remark.** The converse also holds: if $\mathcal{F} \subseteq C(X, \mathbb{R}^m)$ is relatively compact (its closure is compact), then $\mathcal{F}$ is uniformly bounded and uniformly equicontinuous.

### Worked Examples

#### Example 0.4.3.12 — A continuous function on $(0, 1)$ without a maximum

$f(x) = x$ on $(0, 1)$: continuous, bounded by $1$, but $\sup = 1 \notin f((0, 1)) = (0, 1)$. The domain is not compact (not closed in $\mathbb{R}$), so EVT does not apply.

#### Example 0.4.3.13 — Proving all norms on $\mathbb{R}^n$ are equivalent via explicit bounds

For $\mathbf{x} \in \mathbb{R}^n$, $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n} \|\mathbf{x}\|_\infty$, and $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_1 \leq n \|\mathbf{x}\|_\infty$. These give Lipschitz equivalence of $\ell^1, \ell^2, \ell^\infty$ on $\mathbb{R}^n$ with dimension-dependent constants. (The *existence* of some constants is what Theorem 0.4.3.7 guarantees abstractly; the *explicit* constants come from direct inequality manipulation.)

#### Example 0.4.3.14 — Application of Arzelà-Ascoli: Peano's ODE existence

Peano's theorem: if $f: [0, T] \times \mathbb{R}^n \to \mathbb{R}^n$ is continuous and bounded, then the ODE $y' = f(t, y)$ with $y(0) = y_0$ has at least one solution on $[0, T]$ (uniqueness not guaranteed without Lipschitz).

*Proof sketch.* Define approximate solutions $y_n(t)$ by Euler's method with step $T/n$; these satisfy $|y_n(t) - y_n(s)| \leq M|t - s|$ where $M = \sup \|f\|$ (equicontinuity) and $|y_n(t)| \leq |y_0| + MT$ (uniform boundedness). By Arzelà-Ascoli, a subsequence $y_{n_k} \to y$ uniformly on $[0, T]$. The limit $y$ satisfies $y(t) = y_0 + \int_0^t f(s, y(s))\, ds$ (by passing to the limit in the integral form of Euler's approximation), so $y$ is a solution.

#### Example 0.4.3.15 — Compactness of Gaussian kernels

The integral operator $(Kf)(x) = \int_0^1 k(x, y) f(y)\, dy$ with $k \in C([0, 1]^2)$ maps bounded sequences in $C[0, 1]$ to uniformly bounded, uniformly equicontinuous sequences; by Arzelà-Ascoli, $K$ is a *compact operator* on $C[0, 1]$ (i.e., sends bounded sets to relatively compact sets). Compactness of integral operators is the foundation of the Fredholm alternative and spectral theory of integral equations.

### Computational Implementation

```python
import numpy as np

# ---------- Verifying Heine-Borel: extracting convergent subsequence ----------
np.random.seed(1)
# Bounded sequence in R^2 on [-1, 1]^2
xs = np.random.uniform(-1, 1, size=(1000, 2))

# Extract Cauchy subsequence: sort by first coordinate, take the sorted order
idx_sorted = np.argsort(xs[:, 0])
xs_sorted = xs[idx_sorted]
# Consecutive differences should have small first-coordinate differences
diffs = np.linalg.norm(np.diff(xs_sorted, axis=0), axis=1)
print(f"Max consecutive difference (after sort): {diffs.max():.4f}")
# But this is not a Cauchy subsequence per se; need a different extraction.
# Use a bisection argument to find a convergent subsequence (skipped for brevity).

# ---------- Uniform continuity on [0, 1] ----------
# f(x) = x^2 is uniformly continuous on [0, 1] (closed, bounded -> compact)
# but not on R. Let's check epsilon-delta.
f = lambda x: x**2
epsilon = 0.01
# On [0, 1]: |f(x) - f(y)| = |x - y||x + y| <= 2|x - y|; so delta = epsilon / 2 suffices.
delta = epsilon / 2
print(f"f(x)=x^2 on [0,1], epsilon={epsilon}, delta={delta}")
# On R: needs delta -> 0 at infinity
for x0 in [0, 1, 10, 100]:
    # Near x0 = 100, |f(100 + h) - f(100)| = 200h + h^2 ≈ 200h
    # For epsilon = 0.01, need delta < epsilon/200 = 5e-5
    print(f"  Near x0={x0}, local delta = {epsilon/(2*x0 + 0.001):.2e}")

# ---------- Extreme Value Theorem on [0, 1] ----------
from scipy.optimize import minimize_scalar
g = lambda x: x * np.sin(10 * x) + (x - 0.5)**2
res_min = minimize_scalar(g, bounds=(0, 1), method='bounded')
res_max = minimize_scalar(lambda x: -g(x), bounds=(0, 1), method='bounded')
print(f"\nEVT on [0,1]: min of g = {res_min.fun:.4f} at x = {res_min.x:.4f}")
print(f"              max of g = {-res_max.fun:.4f} at x = {res_max.x:.4f}")

# ---------- Arzelà-Ascoli demonstration ----------
# Family {f_n(x) = sin(nx)/n} is uniformly bounded and equicontinuous:
# |sin(nx)/n - sin(ny)/n| = |sin(nx) - sin(ny)|/n <= (n|x - y|)/n = |x - y|
# so delta = epsilon suffices for the whole family.
# Extract uniformly convergent subsequence (converges to 0).
xs = np.linspace(0, 1, 100)
for n in [1, 5, 10, 50, 100]:
    f_n = np.sin(n * xs) / n
    print(f"  n={n}: ||f_n||_inf = {np.max(np.abs(f_n)):.4f}")

# Non-equicontinuous family: g_n(x) = sin(nx) (bounded but oscillating)
# No uniformly convergent subsequence.
print(f"\nBounded but non-equicontinuous family {{sin(nx)}}: no AA subsequence")
for n in [1, 10, 100]:
    print(f"  n={n}, |g_n(0.01) - g_n(0)| = {abs(np.sin(n*0.01) - np.sin(0)):.4f}")

# ---------- Riesz's lemma: unit ball in C[0,1] not compact ----------
# Sequence f_n(x) = max(0, 1 - n|x - 1/n|): bumps getting thinner and moving.
# All have ||f_n||_inf = 1, but no uniformly convergent subsequence.
def bump(x, n):
    return np.maximum(0, 1 - n * np.abs(x - 1.0/n))

xs = np.linspace(0, 1, 500)
for n in [5, 20, 100]:
    vals = bump(xs, n)
    print(f"  ||bump_{n}||_inf = {vals.max():.4f}, support width ≈ {2/n:.3f}")
# No pointwise limit at x > 0 (bumps pass by); no uniform limit.

# ---------- Compactness of integral operator ----------
# K(x, y) = exp(-(x-y)^2 / 0.1), f -> (Kf)(x) = int K(x, y) f(y) dy
from scipy.integrate import quad

def K(x, y):
    return np.exp(-(x - y)**2 / 0.1)

def Kf(x, f):
    return quad(lambda y: K(x, y) * f(y), 0, 1)[0]

# Image of several input functions
ys = np.linspace(0, 1, 50)
for name, f in [('f=1', lambda y: 1.0),
                ('f=y', lambda y: y),
                ('f=sin(10y)', lambda y: np.sin(10*y))]:
    vals = np.array([Kf(y, f) for y in ys])
    print(f"  (K{name})(x=0.5) = {Kf(0.5, f):.4f}, range: [{vals.min():.3f}, {vals.max():.3f}]")
# Kernel smooths: outputs are much smoother than inputs. This is the compactness of K.
```

### [QUANT APPLICATION] Compactness in Optimization, PDE, and Statistics

**(A) Portfolio optimization has a solution.** A convex compact feasible set (e.g., the simplex with box bounds) + continuous objective ⇒ EVT guarantees a minimizer exists. This is the deep reason optimization solvers can be expected to converge to *something*.

**(B) Existence of Nash equilibria.** Brouwer's fixed-point theorem (each best-response is a continuous self-map on a compact convex set) relies on compactness; Nash's theorem is an equilibrium existence result that would fail on non-compact strategy spaces.

**(C) Robust / minimax optimization.** Value function $V(\theta) = \min_\pi \max_\omega L(\pi, \theta, \omega)$ is well-defined when $\omega$ ranges over compact uncertainty sets; ML robustness guarantees (distributionally robust, adversarial training) rest on compactness of the adversary's ambiguity set.

**(D) Stochastic control / optimal stopping.** HJB equations are solved as fixed points on compact subsets of value-function spaces; without compactness (e.g., infinite-horizon problems with unbounded state), one needs extra regularity (convex duality, Girsanov-transformed spaces).

**(E) Tightness in probability.** A family of probability measures $\{\mu_n\}$ on a metric space is **tight** iff for every $\varepsilon > 0$, there is a compact set $K$ with $\mu_n(K) > 1 - \varepsilon$ for all $n$. Tightness is the probabilistic analogue of compactness — Prokhorov's theorem says tight families are relatively weakly compact.

**(F) Weak convergence of Euler-Maruyama to SDEs.** Tightness of the discretized paths, combined with a martingale identification of the limit, gives convergence of the approximations. The entire framework of diffusion limits (Donsker's invariance principle, Stroock-Varadhan) is a compactness-in-path-space theorem.

**(G) Consistency of estimators.** MLE is consistent under regularity + compactness of the parameter space: if the parameter space is compact and the log-likelihood is continuous, EVT guarantees the maximum exists and its limit (by Arzelà-Ascoli-style arguments on the log-likelihood process) identifies the true parameter.

**(H) Arzelà-Ascoli in policy function approximation.** In reinforcement learning with continuous state spaces, proving convergence of policy gradient or approximate value iteration involves showing the iterates are equicontinuous and pointwise bounded — then Arzelà-Ascoli produces uniform convergence of a subsequence.

**(I) Rellich-Kondrachov.** The Sobolev embedding $H^1 \hookrightarrow L^2$ on a bounded domain is compact (by Rellich's theorem, an infinite-dimensional Arzelà-Ascoli). Used heavily in variational PDE and in the theory of regression with smoothness penalties.

**(J) Numerical analysis.** Many finite-element and spectral methods converge because their discretization operators are compact perturbations of the identity, enabling Fredholm theory.

### Exercises

#### ★ (Foundation)

**E0.4.3.1.** Prove: a compact metric space is bounded (i.e., has finite diameter).

**E0.4.3.2.** Show that a finite union of compact sets is compact. Show that an arbitrary intersection of compact sets is compact.

**E0.4.3.3.** Prove: continuous image of a connected compact set is connected and compact.

**E0.4.3.4.** Show that in $\mathbb{R}$, a set is compact iff it is closed and bounded — directly, without invoking the abstract equivalence.

**E0.4.3.5.** Is the set $\{1/n : n \in \mathbb{N}\} \cup \{0\}$ compact? Is $\{1/n : n \in \mathbb{N}\}$? Explain.

**E0.4.3.6 (Uniform continuity of $\sqrt{x}$).** Prove $f(x) = \sqrt{x}$ is uniformly continuous on $[0, 1]$ (directly from EVT + continuity, or from Theorem 0.4.3.6).

#### ★★ (Intermediate)

**E0.4.3.7 (Cantor's intersection theorem).** Let $X$ be compact and $(K_n)$ a nested sequence of non-empty closed subsets. Show $\bigcap K_n \neq \varnothing$.

**E0.4.3.8 (Total boundedness in $\ell^p$).** Prove a subset of $\ell^p$ is totally bounded iff it is bounded and "tight at infinity": $\forall \varepsilon > 0, \exists N, \forall \mathbf{x} \in A: \sum_{n > N} |x_n|^p < \varepsilon^p$.

**E0.4.3.9 (Riesz's lemma in full).** Carry out the details of Riesz's lemma from Corollary 0.4.3.9.

**E0.4.3.10 (Locally compact spaces).** A metric space is **locally compact** iff every point has a compact neighborhood. Prove that $\mathbb{R}^n$ is locally compact but $\ell^2$ is not.

**E0.4.3.11 (Arzelà-Ascoli with modulus of continuity).** State and prove the variant of Arzelà-Ascoli where the equicontinuity is quantified by a **common modulus of continuity** $\omega(\delta)$ with $\omega(\delta) \to 0$ as $\delta \to 0$.

**E0.4.3.12 (Compact operators preserve compactness).** Let $X, Y$ be normed spaces. A linear $T: X \to Y$ is **compact** iff $T(\text{bounded})$ is relatively compact in $Y$. Prove that the composition of a compact operator and a bounded one (in either order) is compact.

#### ★★★ (Challenge)

**E0.4.3.13 (Stone-Čech compactification).** Read the construction of the Stone-Čech compactification $\beta X$ of a non-compact metric space $X$: it is the largest compactification of $X$. This is used in functional analysis to reduce non-compact problems to compact ones.

**E0.4.3.14 (Tychonoff's theorem).** State Tychonoff's theorem: an arbitrary product of compact spaces is compact. Note: the proof uses the axiom of choice (equivalent to it for general products). Try the proof in the countable case.

**E0.4.3.15 (Rellich-Kondrachov).** Read and state Rellich's theorem: for a bounded Lipschitz domain $\Omega \subseteq \mathbb{R}^n$, the Sobolev embedding $H^1(\Omega) \hookrightarrow L^2(\Omega)$ is compact. Interpret as an infinite-dimensional Arzelà-Ascoli.

**E0.4.3.16 (Prokhorov's theorem).** Let $\mathcal{P}(\mathbb{R})$ be probability measures on $\mathbb{R}$ with weak topology. Prove: $\mathcal{F} \subseteq \mathcal{P}(\mathbb{R})$ is weakly sequentially compact iff $\mathcal{F}$ is tight (uniformly concentrated on a compact set in a quantified sense).

**E0.4.3.17 (Brouwer's fixed-point theorem).** Prove Brouwer's fixed-point theorem: any continuous map $f: \overline{B^n} \to \overline{B^n}$ has a fixed point. (Requires either degree theory or a retract argument; read a proof.)

**E0.4.3.18 (Schauder's fixed-point theorem).** Generalize Brouwer to infinite dimensions: a continuous map from a compact convex subset of a Banach space to itself has a fixed point. Use to prove Peano's ODE existence cleanly.

---


## Topic 0.4.4 — Continuous Functions and Uniform Convergence

### Motivation

Topic 0.4.1 introduced continuity abstractly; Topic 0.4.3 refined this to uniform continuity on compact domains and Arzelà-Ascoli for families of continuous functions. This topic confronts the gap between two modes of convergence for sequences of functions: **pointwise** and **uniform**. The distinction is far from pedantic — it determines whether term-by-term differentiation and integration are legitimate, whether limits of continuous functions stay continuous, and whether a given numerical scheme is stable under refinement.

In quant practice, uniform convergence is what makes numerical methods "trustworthy":
- Monte Carlo estimators converge pointwise (in probability, almost surely) but not uniformly in the parameter — which is exactly why variance-reduction techniques (control variates, stratification, antithetic) matter.
- A finite-difference PDE scheme is convergent iff the scheme's error shrinks uniformly on the domain.
- Series representations of pricing functions (e.g., binomial approximations converging to Black-Scholes) require uniform convergence to justify taking derivatives or integrals under the limit.

The two pillars of this topic are the **Weierstrass M-test** (a sufficient condition for uniform convergence of series) and the **Stone-Weierstrass theorem** (density of polynomials and other algebras in $C(X)$ — a massively generalizable result pivotal in approximation theory).

### Prerequisites

- Metric spaces and continuity (Topic 0.4.1).
- Completeness of $C(X)$ for compact $X$ (Topic 0.4.2).
- Compactness (Topic 0.4.3).

### Pointwise vs Uniform Convergence

**Definition 0.4.4.1 (Pointwise convergence).** $(f_n) \subseteq C(X, \mathbb{R})$ **converges pointwise** to $f$ iff for every $x \in X$, $f_n(x) \to f(x)$ as $n \to \infty$.

**Definition 0.4.4.2 (Uniform convergence).** $(f_n) \subseteq B(X, \mathbb{R})$ (bounded functions) **converges uniformly** to $f$ iff $\sup_{x \in X} |f_n(x) - f(x)| \to 0$. This is convergence in the sup-metric.

**Example 0.4.4.3 (Classic pointwise vs uniform).** $f_n: [0, 1] \to \mathbb{R}$, $f_n(x) = x^n$.

- Pointwise limit: $f(x) = 0$ for $x \in [0, 1)$, $f(1) = 1$. Discontinuous at $1$.
- $\sup_{x \in [0, 1]} |f_n(x) - f(x)| = 1$ for every $n$ (taking $x = 1$ or approaching it). So $(f_n)$ does *not* converge uniformly.

This example shows the key takeaway: *pointwise limits of continuous functions need not be continuous*.

**Theorem 0.4.4.4 (Uniform limit is continuous).** *If $f_n \in C(X, Y)$ and $f_n \to f$ uniformly on $X$ (both metric spaces), then $f \in C(X, Y)$.*

**Proof.** Fix $x_0 \in X$ and $\varepsilon > 0$. Uniform convergence: pick $N$ with $d_Y(f(x), f_N(x)) < \varepsilon/3$ for all $x$. Continuity of $f_N$ at $x_0$: pick $\delta > 0$ with $d_X(x, x_0) < \delta \Rightarrow d_Y(f_N(x), f_N(x_0)) < \varepsilon/3$. Then for such $x$:
$$d_Y(f(x), f(x_0)) \leq d_Y(f(x), f_N(x)) + d_Y(f_N(x), f_N(x_0)) + d_Y(f_N(x_0), f(x_0)) < \varepsilon. \qquad \blacksquare$$

This is often called the "$\varepsilon/3$ argument" and is the canonical template for uniform-convergence proofs.

### Uniform Convergence and Interchange Theorems

**Theorem 0.4.4.5 (Uniform + Riemann integrable → interchange).** *If $f_n: [a, b] \to \mathbb{R}$ are Riemann integrable and $f_n \to f$ uniformly, then $f$ is Riemann integrable and*
$$\int_a^b f\, dx = \lim_{n \to \infty} \int_a^b f_n\, dx.$$

**Proof.** $f$ is Riemann integrable because it is a uniform limit of Riemann integrable functions (the upper-lower sum gap stays small under uniform approximation). The integral interchange follows from
$$\left|\int_a^b f - \int_a^b f_n\right| \leq \int_a^b |f - f_n|\, dx \leq (b - a) \|f - f_n\|_\infty \to 0. \qquad \blacksquare$$

**Theorem 0.4.4.6 (Uniform convergence of derivatives).** *Let $f_n \in C^1([a, b])$ with $f_n(x_0)$ convergent for some $x_0 \in [a, b]$ and $(f_n')$ convergent uniformly on $[a, b]$ (to some $g$). Then $(f_n)$ converges uniformly on $[a, b]$ to a $C^1$-function $f$ with $f'(x) = g(x)$.*

**Proof.** Define $L = \lim f_n(x_0)$. By the fundamental theorem of calculus, $f_n(x) = f_n(x_0) + \int_{x_0}^x f_n'(s)\, ds$. The right-hand side converges as $n \to \infty$ to $L + \int_{x_0}^x g(s)\, ds$ uniformly (by Theorem 0.4.4.5 applied on each sub-interval; uniform convergence of $f_n'$ and convergence of $f_n(x_0)$ combine).

So $f_n \to f$ uniformly where $f(x) = L + \int_{x_0}^x g(s)\, ds$. By continuity of $g$ (uniform limit of continuous) and the fundamental theorem of calculus, $f'(x) = g(x)$. $\blacksquare$

This is the interchange of $\lim$ and $\frac{d}{dx}$: $\lim_n f_n'(x) = (\lim_n f_n)'(x)$, *provided* the derivatives converge uniformly.

### Weierstrass M-test

**Theorem 0.4.4.7 (Weierstrass M-test).** *Let $(g_n)$ be functions on $X$ with $|g_n(x)| \leq M_n$ for all $x, n$, and $\sum_n M_n < \infty$. Then $\sum_n g_n$ converges uniformly (and absolutely) on $X$.*

**Proof.** Partial sums $S_N = \sum_{n \leq N} g_n$: for $M > N$,
$$|S_M(x) - S_N(x)| \leq \sum_{n = N+1}^M |g_n(x)| \leq \sum_{n = N+1}^M M_n \leq \sum_{n > N} M_n.$$
The right-hand side is independent of $x$ and tends to $0$ as $N \to \infty$ (tail of a convergent series). So $(S_N)$ is uniformly Cauchy; by completeness of $B(X, \mathbb{R})$, converges uniformly. $\blacksquare$

**Application.** Power series $\sum a_n x^n$ on a closed interval strictly inside its radius of convergence: $|a_n x^n| \leq |a_n| r^n$ for $|x| \leq r < R$, and $\sum |a_n| r^n$ converges; M-test gives uniform convergence, hence the series is continuous and can be integrated term by term on compact subintervals.

### Stone-Weierstrass Theorem

A cornerstone of approximation theory: polynomials are dense in $C([a, b])$ under the sup norm. Weierstrass proved the base case; Stone generalized it to arbitrary compact Hausdorff spaces with any point-separating unital subalgebra.

**Theorem 0.4.4.8 (Weierstrass approximation theorem).** *For every $f \in C([a, b])$ and $\varepsilon > 0$, there exists a polynomial $P$ with $\|f - P\|_\infty < \varepsilon$.*

**Proof via Bernstein polynomials.** WLOG $[a, b] = [0, 1]$. For $f \in C([0, 1])$ define the **$n$-th Bernstein polynomial**
$$B_n(f)(x) = \sum_{k=0}^n \binom{n}{k} x^k (1 - x)^{n - k} f(k/n).$$
$B_n(f)$ is a polynomial in $x$ of degree $\leq n$. We claim $B_n(f) \to f$ uniformly on $[0, 1]$.

**Probabilistic interpretation.** $B_n(f)(x) = \mathbb{E}[f(X_n/n)]$ where $X_n \sim \mathrm{Binomial}(n, x)$. By LLN, $X_n/n \to x$ in probability; by boundedness of $f$ and continuity, $\mathbb{E}[f(X_n/n)] \to f(x)$. Uniform convergence follows from uniform continuity of $f$ on compact $[0, 1]$.

**Quantitative proof.** Let $\omega(\delta) = \sup\{|f(x) - f(y)| : |x - y| < \delta\}$ (modulus of continuity). By uniform continuity, $\omega(\delta) \to 0$.
$$|f(x) - B_n(f)(x)| = \left|\sum_k \binom{n}{k} x^k (1-x)^{n-k} (f(x) - f(k/n))\right| \leq \sum_k \binom{n}{k} x^k (1-x)^{n-k} |f(x) - f(k/n)|.$$
Split the sum into $|k/n - x| < \delta$ and $|k/n - x| \geq \delta$:
- First part: $\leq \omega(\delta) \sum_k \binom{n}{k} x^k (1-x)^{n-k} = \omega(\delta)$.
- Second part: $\leq 2 \|f\|_\infty \cdot \mathbb{P}(|X_n/n - x| \geq \delta) \leq 2 \|f\|_\infty \cdot \mathrm{Var}(X_n/n)/\delta^2 = 2 \|f\|_\infty \cdot \frac{x(1-x)}{n \delta^2} \leq \frac{\|f\|_\infty}{2 n \delta^2}$
 (Chebyshev plus $x(1-x) \leq 1/4$).

Total: $|f(x) - B_n(f)(x)| \leq \omega(\delta) + \|f\|_\infty/(2n\delta^2)$. Choose $\delta = n^{-1/3}$: both terms $\to 0$. Independent of $x$, so uniform. $\blacksquare$

**Theorem 0.4.4.9 (Stone-Weierstrass).** *Let $X$ be a compact metric space and $\mathcal{A} \subseteq C(X, \mathbb{R})$ a subalgebra that (i) contains constants, (ii) separates points (for every $x \neq y$ in $X$, some $f \in \mathcal{A}$ has $f(x) \neq f(y)$). Then $\mathcal{A}$ is dense in $C(X, \mathbb{R})$ under sup norm.*

**Consequences.**

- Polynomials are dense in $C([a, b])$ (Weierstrass): the algebra of polynomials separates points and contains constants.
- Trigonometric polynomials are dense in $C(\mathbb{T}) = $ continuous functions on the circle (under Stone-Weierstrass for circle-valued functions, with complex coefficients — the theorem generalizes to $\mathbb{C}$-valued functions *provided* the subalgebra is closed under complex conjugation).
- Products $p(x, y)$ are dense in $C([a, b] \times [c, d])$.

**Proof of Stone-Weierstrass (outline).** The delicate steps:

- **Step 1.** The closure $\overline{\mathcal{A}}$ in sup norm is a subalgebra (continuity of addition, multiplication, scalar multiplication on $C(X)$).
- **Step 2.** If $f \in \overline{\mathcal{A}}$, then $|f| \in \overline{\mathcal{A}}$. Proof: approximate $t \mapsto |t|$ on $[-\|f\|_\infty, \|f\|_\infty]$ by a polynomial uniformly (by Weierstrass!), and note that polynomials in $f$ lie in $\overline{\mathcal{A}}$ (since $\mathcal{A}$ is an algebra).
- **Step 3.** Consequently, $\max(f, g) = \tfrac{1}{2}(f + g + |f - g|)$ and $\min$ are in $\overline{\mathcal{A}}$ for $f, g \in \overline{\mathcal{A}}$. So $\overline{\mathcal{A}}$ is a **lattice**.
- **Step 4.** Using point-separation, for any $x \neq y$ and any $a, b \in \mathbb{R}$, there is $f_{xy} \in \mathcal{A}$ with $f_{xy}(x) = a$, $f_{xy}(y) = b$ (by combining linearly a constant and a separator).
- **Step 5.** A closed lattice of continuous functions on a compact space containing, for every pair $(x, y)$ and target $(a, b)$, a function matching those values, coincides with all of $C(X)$. Proof: given target $g \in C(X)$ and $\varepsilon$, build at each point $x$ a local $g_x \in \overline{\mathcal{A}}$ with $g_x(x) = g(x)$ and $g_x(y) < g(y) + \varepsilon$ on a neighborhood of $y$ varying with $x$; take finite minima and maxima to get uniform $\varepsilon$-approximation. $\blacksquare$

### Equicontinuity and Compactness in Function Spaces (Recap)

We re-emphasize Arzelà-Ascoli (Topic 0.4.3) as the fundamental tool:

*On compact $X$, a family $\mathcal{F} \subseteq C(X, \mathbb{R}^m)$ is relatively compact in sup norm iff it is pointwise bounded and equicontinuous.*

Because of this, whenever a family arises from bounded-derivative assumptions (Lipschitz, $C^k$-bounded, Sobolev-bounded), Arzelà-Ascoli automatically gives uniformly convergent subsequences — the analytical workhorse of existence theorems in calculus of variations, PDE theory, and optimal control.

### Worked Examples

#### Example 0.4.4.10 — A pointwise-but-not-uniform convergent sequence

$f_n(x) = n x (1 - x)^n$ on $[0, 1]$.
- Pointwise limit: $\lim_n n x (1-x)^n = 0$ for $x \in [0, 1]$ (exponential decay beats polynomial growth).
- Maximum: $f_n'(x) = n(1 - x)^{n-1}(1 - (n+1)x) = 0$ at $x = 1/(n+1)$, giving $f_n(1/(n+1)) = n/(n+1) (1 - 1/(n+1))^n \to 1/e$.
- So $\|f_n\|_\infty \to 1/e$, not to $0$. No uniform convergence.

Note: $\int_0^1 f_n(x)\, dx = n \int_0^1 x(1-x)^n\, dx = n \cdot B(2, n+1) = n/[(n+1)(n+2)] \to 0$. Here $\int \lim = \lim \int$ accidentally holds despite non-uniform convergence (it required dominated convergence, a measure-theoretic tool we will develop in Subject 1).

#### Example 0.4.4.11 — Weierstrass-$M$ for Riemann $\zeta$

$\zeta(s) = \sum_n 1/n^s$ on $\mathrm{Re}(s) > 1$. For $s = \sigma + it$ with $\sigma \geq \sigma_0 > 1$:
$$|1/n^s| = 1/n^\sigma \leq 1/n^{\sigma_0},$$
and $\sum 1/n^{\sigma_0} < \infty$. By M-test, $\zeta$ converges uniformly on $\{\mathrm{Re}(s) \geq \sigma_0\}$ and is holomorphic there (uniform convergence + termwise differentiability for power series → limit is holomorphic).

#### Example 0.4.4.12 — Approximation of $|x|$ by polynomials

$|x|$ is not a polynomial but, by Weierstrass, is uniformly approximable by polynomials on $[-1, 1]$. A well-known explicit sequence:
$$P_n(x) = P_{n-1}(x) + \tfrac{1}{2}(x^2 - P_{n-1}(x)^2), \quad P_0(x) = 0.$$
Then $P_n(x) \to |x|$ uniformly on $[-1, 1]$. Used in Stone-Weierstrass step 2.

#### Example 0.4.4.13 — Stone-Weierstrass gives density of $\{\text{linear combinations of } e^{-\lambda x}\}$ in $C[0, \infty)$

This is a variant: the algebra generated by $\{x, 1, e^{-\lambda x}\}$ for $\lambda > 0$ separates points on any bounded interval, so its closure contains all continuous functions (on the compactified half-line including $\infty$). This is the analytical basis of **Laplace transform inversion** techniques.

### Computational Implementation

```python
import numpy as np
import matplotlib  # only referenced for comment

# ---------- Pointwise vs uniform: x^n on [0, 1] ----------
xs = np.linspace(0, 1, 1000)
f = lambda x, n: x**n
sup_diffs = []
for n in [1, 10, 50, 200, 1000]:
    vals = f(xs, n)
    # pointwise limit is 0 for x < 1, 1 at x = 1
    pointwise = np.where(xs < 1, 0.0, 1.0)
    sup_diff = np.max(np.abs(vals - pointwise))
    sup_diffs.append((n, sup_diff))
    print(f"  n={n}: sup|f_n - f_pointwise| = {sup_diff:.4f}")
# sup diff never goes to 0 -- no uniform convergence

# ---------- Weierstrass M-test: uniform convergence of sum 1/n^s for Re(s) > 1 ----------
def zeta_partial(s, N):
    return np.sum(1.0/np.arange(1, N+1)**s)

for s in [1.5, 2, 3, 5]:
    for N in [10, 100, 1000, 10000]:
        print(f"  zeta({s}), N={N}: {zeta_partial(s, N):.6f}")

# ---------- Bernstein approximation to |x| on [-1, 1] ----------
from scipy.special import comb

def bernstein_approx(f, n, x):
    """B_n(f)(x) = sum_{k=0}^n C(n,k) x^k (1-x)^{n-k} f(k/n), on [0, 1]."""
    result = np.zeros_like(x)
    for k in range(n+1):
        result += comb(n, k) * x**k * (1 - x)**(n-k) * f(k/n)
    return result

# Map [-1, 1] to [0, 1] via y = (x+1)/2
def bern_on_minus11(f, n, x):
    y = (x + 1) / 2
    g = lambda u: f(2*u - 1)
    return bernstein_approx(g, n, y)

xs = np.linspace(-1, 1, 200)
abs_fn = lambda x: np.abs(x)
for n in [5, 20, 100]:
    approx = bern_on_minus11(abs_fn, n, xs)
    err = np.max(np.abs(approx - abs_fn(xs)))
    print(f"  Bernstein n={n}: max error approximating |x| = {err:.4f}")

# ---------- Uniform convergence of derivatives ----------
# f_n(x) = sin(nx)/n^2 on [0, 2pi]:
# f_n(x) -> 0 uniformly (|sin(nx)/n^2| <= 1/n^2)
# f_n'(x) = cos(nx)/n -> 0 uniformly (|cos(nx)/n| <= 1/n)
# f_n''(x) = -sin(nx) does NOT converge uniformly (|sin(nx)| doesn't go to 0)
import numpy as np
xs = np.linspace(0, 2*np.pi, 1000)
for n in [1, 10, 100]:
    fn = np.sin(n*xs)/n**2
    fn_prime = np.cos(n*xs)/n
    fn_double_prime = -np.sin(n*xs)
    print(f"  n={n}: ||f_n||_inf = {np.max(np.abs(fn)):.4f}, "
          f"||f_n'||_inf = {np.max(np.abs(fn_prime)):.4f}, "
          f"||f_n''||_inf = {np.max(np.abs(fn_double_prime)):.4f}")

# ---------- Weierstrass nowhere-differentiable function ----------
# W(x) = sum_n (1/2)^n cos(7^n pi x)
# This is uniformly convergent by M-test (|term| <= (1/2)^n) with continuous limit
# But the derivative series sum_n (1/2)^n * 7^n pi sin(7^n pi x) = sum_n (7/2)^n pi sin(...) diverges
# so W is continuous but nowhere differentiable.
def weier(x, N=50, a=0.5, b=7):
    n = np.arange(N)
    return np.sum(a**n * np.cos(b**n * np.pi * x[:, None]), axis=1)

xs = np.linspace(0, 1, 2000)
W = weier(xs)
print(f"\nWeierstrass W: range = [{W.min():.4f}, {W.max():.4f}]")
# Numerical derivative: blows up on fine scales
dx = xs[1] - xs[0]
dW = (W[1:] - W[:-1]) / dx
print(f"Weierstrass numerical derivative: range = [{dW.min():.2f}, {dW.max():.2f}]")
print(f"At finer resolution, these will grow — reflecting nowhere-differentiability.")
```

### [QUANT APPLICATION] Uniform Convergence in Numerical Finance

**(A) Binomial tree → Black-Scholes.** The Cox-Ross-Rubinstein binomial tree converges to Black-Scholes as the time step shrinks. Convergence is *uniform* in the option price on compact sets of (spot, time), validated via uniform convergence theorems applied to the pricing functional. Without uniform convergence, computing Greeks by finite differences on the tree would be unstable.

**(B) Weierstrass approximation in option pricing.** Any continuous payoff $\varphi(S_T)$ can be uniformly approximated by polynomials on a compact price range; this underlies the "replication by vanilla options" intuition — every continuous payoff is approximately a linear combination of call and put payoffs (or polynomials, which are integrals of call payoffs against a discrete measure).

**(C) Chebyshev polynomials.** A specific polynomial family whose interpolants minimize the sup-norm error. Used in pricing when a closed-form is unavailable: approximate the price function by Chebyshev series, evaluate pricelets quickly. Essential for calibration and scenario analysis where the same pricing kernel is called millions of times.

**(D) Spectral methods for PDE.** Solve the Black-Scholes PDE by expanding solutions in a basis that is dense by Stone-Weierstrass; uniform convergence justifies truncation error estimates. FFT-based pricing (Carr-Madan, Lewis) is a frequency-domain Stone-Weierstrass in disguise.

**(E) Uniform laws of large numbers.** In statistical learning, Rademacher complexity / VC theory quantify how fast empirical averages $\hat{\mathbb{E}}[f]$ converge to $\mathbb{E}[f]$ uniformly over a function class $\mathcal{F}$. The uniform LLN lets us guarantee that empirical risk minimization (ERM) produces a near-optimal hypothesis.

**(F) Calibration stability.** If calibrated model parameters $\theta^*$ depend continuously on market prices $p$, calibration is "stable"; discontinuity signals non-identifiability or singularity. Uniform continuity (via EVT + compact parameter space) gives quantitative bounds on stability.

**(G) Numerical stability of SDEs.** The Euler-Maruyama scheme $X^{(n)}$ converges in law to the true SDE $X$. Uniform convergence in weak topology (via tightness + finite-dimensional distribution matching) is the foundation of modern scientific ML (PINNs, neural SDEs).

**(H) Interpolation of yield curves / implied vol surfaces.** Cubic spline + sup-norm control of the smoothing parameter: Stone-Weierstrass-type density results justify using smooth interpolants as universal approximators.

### Exercises

#### ★ (Foundation)

**E0.4.4.1.** Show that $f_n(x) = \arctan(nx)/n$ converges uniformly to $0$ on $\mathbb{R}$; hence $f_n'(x) = 1/(1 + n^2 x^2)$ must converge somewhere non-uniformly — describe where.

**E0.4.4.2.** Prove: the uniform limit of bounded functions on $X$ is bounded.

**E0.4.4.3.** Show that $\sum_n \sin(nx)/n^2$ converges uniformly on $\mathbb{R}$ (use M-test). Is $\sum_n \sin(nx)/n$ uniformly convergent on all of $\mathbb{R}$? (Hint: Dirichlet's test, convergence but not absolute.)

**E0.4.4.4.** Prove (using pointwise / uniform analysis): the series $\sum_{n=0}^\infty x^n/n!$ converges uniformly on any compact $[-R, R]$, hence $e^x$ is continuous.

**E0.4.4.5.** Construct a sequence of continuous $f_n: [0, 1] \to \mathbb{R}$ with $f_n \to 0$ pointwise but $\int f_n \not\to 0$. (Possible because pointwise convergence is weaker than uniform; not a contradiction with Theorem 0.4.4.5.)

#### ★★ (Intermediate)

**E0.4.4.6 (Dini's theorem).** Prove: if $(f_n)$ is a *monotone* sequence of continuous functions on compact $X$ converging pointwise to a *continuous* $f$, then convergence is uniform.

**E0.4.4.7 (Abel's test for uniform convergence).** State and prove Abel's uniform test (if $\sum a_n(x)$ has uniformly bounded partial sums and $b_n(x)$ is monotone uniformly bounded decreasing to $0$, then $\sum a_n(x) b_n(x)$ converges uniformly).

**E0.4.4.8 (Approximation by convolution).** Let $\varphi \geq 0$ be smooth compact-support with $\int \varphi = 1$ and $\varphi_n(x) = n \varphi(nx)$. Prove: for $f \in C_c(\mathbb{R})$, $f * \varphi_n \to f$ uniformly. Bonus: $f * \varphi_n$ is smooth, so smooth functions are dense in $C_c$.

**E0.4.4.9 (Stone-Weierstrass with complex-valued functions).** Prove: a subalgebra $\mathcal{A} \subseteq C(X, \mathbb{C})$ separating points, containing constants, *and closed under complex conjugation* is dense in $C(X, \mathbb{C})$. (The conjugate-closure condition is essential; find a counterexample when it fails.)

**E0.4.4.10 (Mercer's theorem preview).** Read Mercer's theorem: for a continuous, symmetric, positive-definite kernel $K \in C([a, b]^2)$, $K(x, y) = \sum_n \lambda_n \phi_n(x) \phi_n(y)$ where $\lambda_n \geq 0$ and $\{\phi_n\}$ is an orthonormal basis of $L^2[a, b]$, with uniform convergence of the series on $[a, b]^2$. (This is the continuous analogue of spectral decomposition for kernel operators.)

**E0.4.4.11 (Chebyshev approximation).** Prove: among all polynomials of degree $\leq n$ approximating a given $f \in C([-1, 1])$ in sup norm, the minimum-error one is unique and is characterized by equi-oscillation (the error alternates between $\pm\|f - P_n\|_\infty$ at least $n+2$ times). This is the **Chebyshev alternation theorem**.

#### ★★★ (Challenge)

**E0.4.4.12 (Müntz-Szász theorem).** Prove: a sequence of powers $\{x^{\lambda_n}\}$ with $\lambda_0 = 0$ and $\lambda_n \to \infty$ has its linear span dense in $C([0, 1])$ iff $\sum 1/\lambda_n = \infty$. (Explains why $\{1, x, x^2, \ldots\}$ is dense but $\{1, x, x^4, x^9, \ldots\}$ is not.)

**E0.4.4.13 (Hahn-Banach extension).** Read and state the Hahn-Banach extension theorem: a bounded linear functional on a subspace extends to a bounded functional on the whole Banach space with the same norm. This is a non-trivial existence theorem that uses Zorn's lemma; with H-B, one can *construct* continuous linear functionals to separate closed sets, giving the existence of support hyperplanes etc.

**E0.4.4.14 (Weierstrass nowhere-differentiable, quantitative).** Prove that $\sum_n a^n \cos(b^n \pi x)$ with $0 < a < 1$, $ab > 1 + 3\pi/2$, is continuous and nowhere differentiable. Compute an upper bound on the Hölder modulus.

**E0.4.4.15 (Approximating by neural networks).** Prove the **universal approximation theorem**: single-hidden-layer neural networks with a non-polynomial continuous activation are dense in $C(K)$ for compact $K \subseteq \mathbb{R}^n$. (Cybenko 1989 for sigmoidal; Hornik 1991 in generality.)

**E0.4.4.16 (Kolmogorov-Arnold theorem).** Read: any continuous $f: [0, 1]^n \to \mathbb{R}$ is a finite superposition of continuous functions of *single variables*. (Contrast with Hilbert's 13th problem's failure for algebraic functions.) This is the abstract basis of Kolmogorov-Arnold networks (KANs).

---


## Topic 0.4.5 — Connectedness

### Motivation

The intermediate value theorem (IVT) — a continuous function on $[a, b]$ takes every value between $f(a)$ and $f(b)$ — is the archetypal connectedness theorem. It rests not on the metric structure of $[a, b]$ but on its topological **connectedness**: $[a, b]$ cannot be partitioned into two nonempty open pieces.

For quantitative work, connectedness is the mathematical ingredient behind:

- **Root-finding by bisection.** IVT guarantees a root; bisection exploits it iteratively. Every numerical root-finder builds on this.
- **Continuation methods.** Path-continuity in parameter space (homotopy methods for nonlinear equations, arc-length continuation for bifurcations) depends on the connectedness of the parameter-space manifold.
- **Monodromy and analytic continuation.** In complex analysis (Module 0.5), extending a holomorphic function along paths gives consistent values only if the domain is simply connected — a strong connectedness condition.
- **No-arbitrage pricing.** The fundamental theorem of asset pricing (existence of a risk-neutral measure under no-arbitrage) uses separation of *convex sets* (Hahn-Banach), but the path-connectedness of the positive orthant's interior plays a supporting role in proving continuity of pricing functionals.

The topic is brief — connectedness is foundational but has few deep theorems to prove; its power comes from its many applications.

### Prerequisites

- Metric spaces, open/closed sets, continuity (Topic 0.4.1).
- Compactness (Topic 0.4.3).

### Definitions

**Definition 0.4.5.1 (Connected, disconnected).** A topological space $X$ is **disconnected** iff it is the disjoint union of two non-empty open sets: $X = U \sqcup V$ with $U, V$ open, $U, V \neq \varnothing$, $U \cap V = \varnothing$. Otherwise $X$ is **connected**.

Equivalent formulations:

- $X$ has no non-trivial "clopen" (simultaneously closed and open) subsets: $A \subseteq X$ clopen $\Rightarrow A = \varnothing$ or $A = X$.
- Every continuous $f: X \to \{0, 1\}$ (discrete 2-point space) is constant.

A subset $A \subseteq X$ is **connected** iff it is connected as a subspace (relative topology).

**Example 0.4.5.2.**

- $\{0, 1\}$ with discrete metric is disconnected.
- $(-1, 0) \cup (0, 1)$ in $\mathbb{R}$ is disconnected; each open piece is a clopen subset.
- $\mathbb{R}$ is connected (Theorem 0.4.5.4 below).
- $\mathbb{Q}$ in $\mathbb{R}$ is *totally disconnected* — every connected subset is a single point. ($\mathbb{Q} \cap (-\sqrt 2, \sqrt 2)$ and its complement in $\mathbb{Q}$ show how irrational cuts disconnect $\mathbb{Q}$.)

### Key Theorems

**Theorem 0.4.5.3 (Continuous image of connected is connected).** *If $X$ is connected and $f: X \to Y$ is continuous, then $f(X) \subseteq Y$ is connected.*

**Proof.** Suppose $f(X)$ is disconnected: $f(X) = U' \sqcup V'$ with $U', V'$ non-empty, open in $f(X)$. Let $U = f^{-1}(U'), V = f^{-1}(V')$: open in $X$ by continuity, non-empty (preimages of non-empty), disjoint, union = $X$. So $X = U \sqcup V$ is a disconnection of $X$, contradiction. $\blacksquare$

**Theorem 0.4.5.4 (Connected subsets of $\mathbb{R}$).** *A subset $A \subseteq \mathbb{R}$ is connected iff it is an interval (possibly unbounded, possibly degenerate).*

**Proof.** ($\Leftarrow$) If $A$ is not an interval, there exist $a, b \in A$ and $c \notin A$ with $a < c < b$. Then $A = (A \cap (-\infty, c)) \sqcup (A \cap (c, \infty))$ is a disconnection (both pieces non-empty since $a, b \in A$, and open in $A$). So a connected subset of $\mathbb{R}$ is an interval.

($\Rightarrow$) Let $I$ be an interval. Suppose $I = U \sqcup V$ with $U, V$ non-empty open in $I$ and disjoint. Take $u \in U, v \in V$, WLOG $u < v$. Let $s = \sup(U \cap [u, v])$. Then $s \in I$ (since $I$ is an interval and $s \in [u, v]$). Also $s \in \overline{U \cap [u, v]} \subseteq \overline{U}$. By openness of $V$ in $I$ and $v \in V$, some small interval around $v$ lies in $V$, so $s < v$ (we can't have $s = v$ because then $v \in \overline U$ would contradict $V$ open and disjoint from $U$). Now:

- If $s \in U$: $U$ is open in $I$, so some $\varepsilon > 0$ has $(s - \varepsilon, s + \varepsilon) \cap I \subseteq U$; but $s + \varepsilon/2 \leq v$ gives an element of $U$ greater than $s$, contradicting $s = \sup$.
- If $s \in V$: $V$ is open in $I$, so some $\varepsilon > 0$ has $(s - \varepsilon, s + \varepsilon) \cap I \subseteq V$; but then no element of $U \cap [u, v]$ exceeds $s - \varepsilon$, contradicting $s = \sup$.

Contradiction in both cases, so no such disconnection exists. $\blacksquare$

**Theorem 0.4.5.5 (Intermediate Value Theorem).** *Let $f: [a, b] \to \mathbb{R}$ be continuous. For any $y$ between $f(a)$ and $f(b)$, there exists $c \in [a, b]$ with $f(c) = y$.*

**Proof.** $[a, b]$ is connected (interval), so $f([a, b])$ is connected (Theorem 0.4.5.3), hence an interval in $\mathbb{R}$ (Theorem 0.4.5.4). This interval contains $f(a)$ and $f(b)$, so it contains every value between them. $\blacksquare$

**Corollary 0.4.5.6 (Borsuk-Ulam preview).** Any continuous $f: S^n \to \mathbb{R}$ has a pair of antipodes with the same image ($f(-x) = f(x)$). Sketch: $g(x) = f(x) - f(-x)$ is continuous, odd, and $S^n$ is connected (and path-connected), so $g$ takes both positive and negative values (as $g(x) = -g(-x)$), hence takes $0$ somewhere.

### Path-Connectedness

**Definition 0.4.5.7 (Path-connected).** $X$ is **path-connected** iff for every $x, y \in X$ there is a continuous $\gamma: [0, 1] \to X$ with $\gamma(0) = x$, $\gamma(1) = y$.

**Proposition 0.4.5.8.** *Path-connected implies connected.*

**Proof.** Suppose $X$ is path-connected but $X = U \sqcup V$ is a disconnection. Pick $u \in U$, $v \in V$ and a path $\gamma: [0, 1] \to X$ with $\gamma(0) = u, \gamma(1) = v$. Then $\gamma^{-1}(U), \gamma^{-1}(V)$ disconnects $[0, 1]$ — contradicting connectedness of the interval. $\blacksquare$

**Proposition 0.4.5.9 (Partial converse: open sets in $\mathbb{R}^n$).** *Connected open subsets of $\mathbb{R}^n$ are path-connected.*

**Proof.** Let $U \subseteq \mathbb{R}^n$ be open and connected; fix $a \in U$. Define $A = \{x \in U : \exists \text{ path in } U \text{ from } a \text{ to } x\}$.

*$A$ is open:* if $x \in A$, there's an open ball $B_r(x) \subseteq U$; for every $y \in B_r(x)$, concatenate the path from $a$ to $x$ with the straight line $[x, y]$ (which lies in $B_r(x) \subseteq U$). So $y \in A$, proving $B_r(x) \subseteq A$.

*$U \setminus A$ is open:* similarly, if $x \in U \setminus A$, any $y$ in a small ball $B_r(x) \subseteq U$ cannot be in $A$ (otherwise concatenating gives a path from $a$ to $x$, contradicting $x \notin A$). So $B_r(x) \subseteq U \setminus A$.

Thus $U = A \sqcup (U \setminus A)$ as open sets, and connectedness of $U$ forces $U \setminus A = \varnothing$ (since $a \in A$ so $A \neq \varnothing$), i.e., $A = U$. $\blacksquare$

**Counterexample (connected but not path-connected):** the **topologist's sine curve** $T = \{(x, \sin(1/x)) : 0 < x \leq 1\} \cup (\{0\} \times [-1, 1]) \subseteq \mathbb{R}^2$. Connected (closure of path-connected set is connected), but not path-connected (no continuous path from the $y$-axis piece to the sinusoidal piece — too much oscillation).

### Components

**Definition 0.4.5.10 (Connected component).** The **connected component** of a point $x \in X$ is the union of all connected subsets of $X$ containing $x$. Equivalently, the largest connected subset containing $x$.

**Proposition 0.4.5.11.** *Connected components are closed and partition $X$.*

**Proof.** *Closed:* the closure of a connected set is connected (if $C$ is connected and $C \subseteq D \subseteq \overline C$, then $D$ is connected; details: any disconnection of $D$ by two open sets would restrict to a disconnection of $C$). So the component of $x$, which is a maximal connected subset, must be closed.

*Partition:* if $C_1, C_2$ are components and $C_1 \cap C_2 \neq \varnothing$, then $C_1 \cup C_2$ is connected (union of two connected sets with a common point is connected, as any disconnection of the union would disconnect each piece) and strictly larger than each, contradicting maximality. So components are either equal or disjoint; their union equals $X$. $\blacksquare$

**Example 0.4.5.12.** $\mathbb{R} \setminus \mathbb{Q}$ has uncountably many connected components (each a single irrational). This is the totally disconnected nature of the irrationals (in the subspace topology).

### Simply Connected (Preview)

In complex analysis and algebraic topology, **simply connected** means path-connected plus "every loop is contractible" — no holes. $\mathbb{R}^2 \setminus \{0\}$ is path-connected but not simply connected (a loop around origin cannot shrink to a point). We defer the formal definition to Module 0.5 (complex analysis) and Subject 4 (differential geometry), where it is essential for Cauchy's theorem and for understanding monodromy.

### Worked Examples

#### Example 0.4.5.13 — Root-finding via bisection

Let $f: [a, b] \to \mathbb{R}$ be continuous with $f(a) < 0 < f(b)$. By IVT, there's a root $c \in (a, b)$. *Bisection*: let $m = (a + b)/2$. If $f(m) = 0$, done. If $f(m) > 0$, root lies in $(a, m)$; else in $(m, b)$. Iterate. After $n$ iterations, the interval has length $(b - a)/2^n$; linear (quadratic-in-bits) convergence. Bisection is robust (no derivative needed) but slow compared to Newton.

#### Example 0.4.5.14 — Homotopy continuation for nonlinear systems

To solve $F(x) = 0$ for a hard $F$, construct a homotopy $H(x, t) = (1 - t) G(x) + t F(x)$ where $G$ has easy roots. Start with a root $x_0$ of $G$; trace the path $t \mapsto x(t)$ of roots as $t: 0 \to 1$. Connectedness of the path gives existence (under regularity); the path's terminal point is a root of $F$. Implemented in packages like HomotopyContinuation.jl, used in computer algebra and algebraic geometry-based finance.

#### Example 0.4.5.15 — Arbitrage-free price intervals

For an illiquid derivative, the range of *arbitrage-free* prices forms a connected interval (often a singleton in complete markets, a proper interval in incomplete markets). Connectedness is what allows "fair-value" interpolation between bid and ask.

#### Example 0.4.5.16 — IVT-based existence of a critical point in 1D

Given a $C^1$ function $f: [a, b] \to \mathbb{R}$ with $f(a) = f(b)$, by Rolle's theorem (a version of IVT applied to $f'$), there is $c \in (a, b)$ with $f'(c) = 0$. The argument: $f$ on compact $[a, b]$ attains min and max; if both are at the endpoints (and equal), $f$ is constant (trivial); else an interior extremum is a critical point.

### Computational Implementation

```python
import numpy as np

# ---------- Bisection root-finder ----------
def bisect(f, a, b, tol=1e-12, max_iter=100):
    assert f(a) * f(b) < 0, "f must have opposite signs at endpoints"
    for _ in range(max_iter):
        m = (a + b) / 2
        if f(m) == 0 or (b - a) / 2 < tol:
            return m
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return (a + b) / 2

# Find root of cos(x) - x in (0, 1)
root = bisect(lambda x: np.cos(x) - x, 0, 1)
print(f"Root of cos(x) = x: {root:.12f}")
print(f"Verification: cos({root:.6f}) - {root:.6f} = {np.cos(root) - root:.3e}")

# ---------- IVT-based existence of fixed points ----------
# If g: [0, 1] -> [0, 1] is continuous, then f(x) = g(x) - x has f(0) = g(0) >= 0
# and f(1) = g(1) - 1 <= 0, so IVT gives a fixed point.
g = lambda x: 0.5 + 0.25*np.sin(5*x)  # maps [0,1] into [0.25, 0.75]
fixed_pt = bisect(lambda x: g(x) - x, 0, 1)
print(f"\nFixed point of g: {fixed_pt:.8f}, g({fixed_pt:.4f}) = {g(fixed_pt):.8f}")

# ---------- Homotopy continuation for f(x) = 0 ----------
# Easy system: G(x) = x - 1 (root at 1)
# Hard system: F(x) = x^3 - 2x - 5 (root near 2.0946)
# Trace: H(x, t) = (1-t)(x - 1) + t(x^3 - 2x - 5) = 0
# dx/dt = -(dH/dt) / (dH/dx) at the current x

from scipy.integrate import solve_ivp

def homotopy_rhs(t, x):
    # dx/dt = -(dH/dt) / (dH/dx)
    dHdt = (x**3 - 2*x - 5) - (x - 1)
    dHdx = (1-t) + t*(3*x**2 - 2)
    return -dHdt / dHdx

# Start at x0 = 1 (root of G)
sol = solve_ivp(homotopy_rhs, [0, 1], [1.0], rtol=1e-10, atol=1e-12)
x_final = sol.y[0, -1]
print(f"\nHomotopy continuation: x(t=1) = {x_final:.8f}")
print(f"Verify: f({x_final:.4f}) = {x_final**3 - 2*x_final - 5:.3e}")

# ---------- Path-connectedness vs. connectedness ----------
# Topologist's sine curve: visualize and sample 'paths'
def tscurve_oscillating(x): return np.sin(1/x)
xs = np.linspace(0.01, 1, 5000)
sample = tscurve_oscillating(xs)
print(f"\nTopologist's sine curve: oscillations between -1 and 1")
print(f"  Near x=0.01: sin(1/0.01) = {tscurve_oscillating(0.01):.4f}")
print(f"  Near x=0.001: sin(1/0.001) = {tscurve_oscillating(0.001):.4f}")
print(f"These oscillations prevent path-connection to the y-axis part.")

# ---------- Connected components of R \ Q approximation (finite sample) ----------
# Approximate "irrational" as Q-complement up to tolerance
qs_sample = np.arange(1, 101) / 50  # rationals in (0, 2)
print(f"\nRational points in (0, 2): {len(qs_sample)}")
print(f"Each gap between consecutive rationals is a connected component of (R \\ Q) restricted to this sample")
gaps = np.diff(np.sort(qs_sample))
print(f"Gap sizes: min {gaps.min():.4f}, max {gaps.max():.4f}")

# ---------- IVT in optimization: continuity of argmax ----------
# For continuous f on [0, 1], the function x* = argmax_{x in [0, c]} f(x) is continuous in c
# (for c varying over a range where the argmax is unique). This is a connectedness-of-level-sets argument.
def argmax_up_to(c, f=lambda x: -(x - 0.7)**2 + np.sin(8*x)):
    xs = np.linspace(0, c, 1001)
    return xs[np.argmax(f(xs))]

cs = np.linspace(0.1, 1.0, 20)
argmaxes = [argmax_up_to(c) for c in cs]
print(f"\nContinuity of argmax in c:")
for c, x in zip(cs, argmaxes):
    print(f"  c={c:.3f}: x* = {x:.4f}")
```

### [QUANT APPLICATION] Connectedness in Finance and Numerical Methods

**(A) Bisection in implied volatility solvers.** The classical implied-vol solver uses bisection on $\sigma \mapsto C_{BS}(\sigma) - C_{\mathrm{mkt}} = 0$, exploiting the strict monotonicity (vega $> 0$) of $C_{BS}$ in $\sigma$. IVT guarantees existence; strict monotonicity gives uniqueness. Every options desk runs millions of these.

**(B) Continuation of calibrated parameters.** When market data moves incrementally, calibrated parameters trace a continuous path in parameter space. Homotopy methods solve for $\theta$ as market data varies; connectedness of the valid-parameter region ensures no "jumps" in calibration as long as the path stays in the region.

**(C) Arbitrage-free price intervals (super- and sub-replication).** In incomplete markets, the set of arbitrage-free prices for a contingent claim is a closed interval $[\pi_{\mathrm{sub}}(X), \pi_{\mathrm{sup}}(X)]$; the interval's connectedness is what allows traders to quote any price within the range.

**(D) Ergodic theory of recurrent processes.** Connectedness of the state space (or strongly connected graph for discrete chains) is a standard assumption to ensure irreducibility and uniqueness of the invariant measure, foundational for stationary analysis of time series.

**(E) Simply connected domains and option pricing via complex analysis.** Hilbert-transform-based inversion of option prices to implied distributions (Breeden-Litzenberger) relies on analytic continuations that require simply connected complex domains.

**(F) Convergence of Monte Carlo continuation schemes.** Path-connected acceptance regions in MCMC (Metropolis-Hastings) enable ergodic chains; failure of connectedness leads to "mode collapse" (the chain samples only one of several disconnected modes). Techniques like parallel tempering overcome this.

**(G) Connected components in credit risk graphs.** Default cascade models on counterparty networks use graph connectedness; the propagation of defaults is exactly a connected-component analysis. Systemic risk measures (DebtRank, contagion measures) rely on this.

**(H) Graph-based machine learning.** Spectral clustering finds connected (or nearly connected) components via the graph Laplacian; the Fiedler eigenvector reveals the structure. This is used in market-regime detection and in factor discovery.

### Exercises

#### ★ (Foundation)

**E0.4.5.1.** Prove: the empty set and any single-point set are connected.

**E0.4.5.2.** Show that the union of two connected sets with non-empty intersection is connected.

**E0.4.5.3.** Prove: a continuous $f: X \to \mathbb{Z}$ (with $\mathbb{Z}$ discrete) on a connected $X$ is constant.

**E0.4.5.4.** Use IVT to prove: every odd-degree polynomial with real coefficients has a real root.

**E0.4.5.5.** Show that $[0, 1] \cup [2, 3]$ is disconnected in $\mathbb{R}$.

#### ★★ (Intermediate)

**E0.4.5.6 (Connected components of $GL_n(\mathbb{R})$).** Prove: $GL_n(\mathbb{R})$ has exactly two connected components: $\{\det > 0\}$ and $\{\det < 0\}$. (Use that any matrix with $\det \neq 0$ can be continuously deformed to $\pm I$ via elementary row operations.)

**E0.4.5.7 (Path-connected $\Rightarrow$ connected; converse fails).** Give a full proof that path-connectedness implies connectedness. Construct the topologist's sine curve rigorously and prove it is connected but not path-connected.

**E0.4.5.8 (Connected components of open sets in $\mathbb{R}^n$).** Show that an open subset of $\mathbb{R}^n$ has at most countably many connected components.

**E0.4.5.9 (IVT for monotonic functions).** Prove: a continuous strictly increasing $f: [a, b] \to \mathbb{R}$ is a homeomorphism onto $[f(a), f(b)]$. Deduce that $e^x$, $\log x$, and power functions are well-defined continuous bijections on their natural domains.

**E0.4.5.10 (Fixed-point theorem on intervals).** Prove: every continuous $f: [0, 1] \to [0, 1]$ has a fixed point. (Baby Brouwer in 1D.) Show by example this fails for $f: (0, 1) \to (0, 1)$.

#### ★★★ (Challenge)

**E0.4.5.11 (Poincaré-Miranda theorem).** A 2-d generalization of IVT: let $\mathbf{f}: [-1, 1]^n \to \mathbb{R}^n$ be continuous with $f_i|_{x_i = -1} \leq 0$ and $f_i|_{x_i = 1} \geq 0$ for each $i$. Then $\mathbf{f}$ has a zero in $[-1, 1]^n$. (This is equivalent to Brouwer's fixed-point theorem.)

**E0.4.5.12 (Borsuk-Ulam theorem).** Prove the 1D case of Borsuk-Ulam: every continuous $f: S^1 \to \mathbb{R}$ has a pair of antipodes with the same image. Extend to $S^n \to \mathbb{R}^n$ (significantly harder — requires some degree theory or homology).

**E0.4.5.13 (Ham Sandwich theorem).** Read and state: for any three (measurable, finite) bounded sets in $\mathbb{R}^3$, there is a single plane simultaneously bisecting each. Prove via Borsuk-Ulam. Generalize to $n$ sets in $\mathbb{R}^n$.

**E0.4.5.14 (Simply connected open sets in $\mathbb{R}^2$).** State the Riemann mapping theorem: every simply connected open proper subset of $\mathbb{C}$ is conformally equivalent to the open disk. (We will revisit in Module 0.5.)

**E0.4.5.15 (Connectedness of Julia sets).** Read: the Julia set of $z \mapsto z^2 + c$ is either connected (for $c$ in the Mandelbrot set) or totally disconnected (Cantor set). This dichotomy is connectedness appearing in complex dynamics.

**E0.4.5.16 (Topological genericity of connected components).** Read and understand Whitney's theorem that every open subset of $\mathbb{R}^n$ is a disjoint union of countably many *path-connected* open components.

---

## Module 0.4 Summary and Forward Pointers

We assembled the **topological bedrock** of all modern analysis.

### What we covered

- **Metric spaces** (0.4.1): distance axioms; open/closed sets; induced topology; convergence, Cauchy sequences; continuity with three equivalent characterizations; equivalent metrics.
- **Completeness and Banach spaces** (0.4.2): completeness as the Cauchy-convergent axiom; $\mathbb{R}^n$ complete; closed subsets of complete spaces are complete; Banach spaces; Banach fixed-point theorem (full proof, restated with bounds); completion theorem; Baire category theorem with consequences (uniform boundedness, open mapping, closed graph — sketched) and nowhere-differentiable functions.
- **Compactness** (0.4.3): three equivalent definitions (open cover, sequential, totally bounded + complete), with the proof of equivalence; Heine-Borel in $\mathbb{R}^n$; extreme value theorem; uniform continuity on compact domains; all norms on finite-dimensional spaces equivalent; Riesz's lemma (infinite-dimensional unit ball non-compact); Arzelà-Ascoli theorem with full proof (diagonal subsequence + uniform continuity argument).
- **Continuous functions and uniform convergence** (0.4.4): pointwise vs uniform; uniform limit of continuous is continuous ($\varepsilon/3$ argument); interchange of limit with integral / derivative; Weierstrass M-test; Stone-Weierstrass theorem; Bernstein approximation of $|x|$; universal approximation themes.
- **Connectedness** (0.4.5): connected / path-connected / components; connected subsets of $\mathbb{R}$ are intervals; IVT via connectedness; topologist's sine curve as the connected-but-not-path-connected example; applications to root-finding, homotopy methods, and arbitrage-free price intervals.

### Synthesis

The three headline properties — completeness, compactness, connectedness — each give an existence theorem:

- Completeness → Cauchy sequences converge (Banach fixed-point, Picard-Lindelöf).
- Compactness → continuous functions attain their extrema (EVT, Arzelà-Ascoli).
- Connectedness → continuous functions realize all intermediate values (IVT).

Any complete metric space or Banach space you encounter later in the curriculum has been given the machinery to prove existence theorems. Any compact set has been given the apparatus to extract convergent subsequences and bound uniform approximants. Any connected set gives IVT for free.

### Quant applications we touched

- Wasserstein distance and distributionally robust optimization.
- Bellman fixed-point iterations (value iteration, policy iteration) in dynamic programming.
- Picard-Lindelöf for ODE/SDE existence underlying all diffusion models.
- Compactness in tightness arguments (Donsker, Kalman, MCMC).
- Arzelà-Ascoli underlying calculus of variations and infinite-horizon control.
- Stone-Weierstrass in universal approximation theorems for neural networks.
- IVT in bisection solvers for implied vol.
- Continuation methods in calibration.
- Baire category in functional analysis (uniform boundedness principle, open mapping, closed graph).

### What we did *not* cover (and where it's picked up)

- **General topological spaces** (non-metric): not needed for our applications; we stuck with metric.
- **Lebesgue measure and integration**: assumed at working level, fully developed in Subject 1.
- **Hilbert spaces** beyond basics: Module 0.2 gave enough; full Hilbert-space theory (projections, dual spaces, adjoints) comes in Subject 1 or a dedicated functional analysis module.
- **Weak topologies on Banach spaces**: relevant for probability (weak convergence of measures), to be developed in Subject 2.

### Forward pointers

- **Module 0.5 (Complex Analysis)**: metric-space topology on $\mathbb{C}$; simply connected domains; Cauchy's theorem; residues. Essential for Fourier transforms and characteristic functions.
- **Subject 1 (Measure and Integration)**: proper construction of $L^p$ spaces (Banach, even Hilbert for $p = 2$); Carathéodory's extension; Radon-Nikodym theorem.
- **Subject 2 (Probability)**: weak convergence of measures (Prokhorov's theorem, using tightness = compactness in measures); Skorokhod metric on càdlàg paths.
- **Subject 3 (Stochastic Calculus)**: Itô integration relies on $L^2$ completeness; pathwise properties of Brownian motion (continuity but non-differentiability, Hölder regularity) are concrete instances of genericity results from this module.
- **Subject 5 (Optimization)**: convex analysis in Banach spaces; fixed-point methods for Bellman / HJB equations.

*Next up: Module 0.5 — Complex Analysis.*

---


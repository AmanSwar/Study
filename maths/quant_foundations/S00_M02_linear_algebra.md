# Subject 0, Module 0.2 — Linear Algebra (Beyond JEE)

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street.*

---

## How to read this module

JEE linear algebra is matrix algebra: invert a 3×3 matrix, find the eigenvalues of a 2×2 matrix, compute a determinant, maybe diagonalize a small example. This module does not *replace* that — the computational fluency matters — but it re-founds the subject on the abstract structure of vector spaces and inner product spaces, proves the two most important decomposition theorems in all of applied mathematics (the spectral theorem and the singular value decomposition), develops matrix calculus as a working tool, and closes with the theory of norms. By the end you should be able to:

- State the axioms of a vector space and recognize non-obvious examples (function spaces, polynomial spaces).
- Prove the rank-nullity theorem without looking it up.
- Prove the Cauchy–Schwarz inequality from the inner-product axioms.
- State and prove the spectral theorem for real symmetric matrices.
- Derive the SVD from the spectral theorem.
- Differentiate scalar-of-matrix expressions like $f(W) = \mathrm{tr}(W^\top A W) + \lambda \|W\|_F^2$ without consulting a cheat sheet.
- Derive the closed-form mean-variance-optimal portfolio weights from scratch using matrix calculus and Lagrange multipliers.

The motivation for the depth: everything in Subject 1 (measure theory on function spaces), Subject 2 (covariance structure in probability), Subject 7 (regularized learning), Subject 8 (random matrix theory for covariance denoising), and Subject 9 (Banach/Hilbert spaces, RKHS) is built on this foundation. An approximate grasp of the spectral theorem, unnoticed in an ML course, becomes an active liability in RMT.

## Table of Contents

- Topic 0.2.1 — Vector Spaces
- Topic 0.2.2 — Linear Transformations and Rank–Nullity
- Topic 0.2.3 — Inner Product Spaces
- Topic 0.2.4 — Eigendecomposition
- Topic 0.2.5 — The Spectral Theorem
- Topic 0.2.6 — The Singular Value Decomposition
- Topic 0.2.7 — Positive (Semi-)Definite Matrices and Cholesky
- Topic 0.2.8 — Matrix Calculus
- Topic 0.2.9 — Norms

---

# Topic 0.2.1 — Vector Spaces

## Motivation

In JEE you used $\mathbb{R}^n$ as a container for row-vectors and column-vectors. But a *vector space* is much more general: it is any set with an addition and a scaling operation that satisfies eight axioms. Once you see the axioms, you realize that polynomials, continuous functions, random variables (modulo a.s. equality), and matrices are all examples. Theorems about vector spaces — existence of a basis, dimension, quotient construction — apply to all of these at once. This is the first serious taste of abstraction in the curriculum: proving things about *structures*, not about specific representations.

## Prerequisites

Topics 0.1.1–0.1.5.

## Definitions

**Definition 0.2.1.1 (Field).** A *field* $\mathbb{F}$ is a set with two operations $+$ (addition) and $\cdot$ (multiplication) satisfying: both operations are commutative and associative; both distribute over each other appropriately; there are additive and multiplicative identities $0 \neq 1$; every element has an additive inverse; every non-zero element has a multiplicative inverse.

The fields we use in this curriculum are almost always $\mathbb{R}$ (real numbers) and occasionally $\mathbb{C}$ (complex numbers). You can treat $\mathbb{F}$ as "the reals" on first reading; everything generalizes cleanly.

**Definition 0.2.1.2 (Vector space).** A *vector space* over $\mathbb{F}$ is a set $V$ together with two operations — addition $+: V \times V \to V$ and scalar multiplication $\cdot: \mathbb{F} \times V \to V$ — satisfying:

1. $(V, +)$ is an abelian group: associative, commutative, has a zero element $\mathbf{0} \in V$, every $\mathbf{v}$ has a negative $-\mathbf{v}$.
2. $1 \cdot \mathbf{v} = \mathbf{v}$ for all $\mathbf{v} \in V$.
3. $(\alpha\beta)\mathbf{v} = \alpha(\beta\mathbf{v})$.
4. $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$.
5. $\alpha(\mathbf{v} + \mathbf{w}) = \alpha\mathbf{v} + \alpha\mathbf{w}$.

Elements of $V$ are *vectors*; elements of $\mathbb{F}$ are *scalars*.

*Intuition.* A vector space is *anything* that you can add and scale in a way compatible with ordinary addition and multiplication. The eight (or so — some authors combine a few) axioms are a minimal list of compatibility conditions.

**Examples.** The following are all vector spaces over $\mathbb{R}$:

1. $\mathbb{R}^n$ — the usual one.
2. $\mathbb{F}^n$ for any field $\mathbb{F}$ — column vectors with entries in $\mathbb{F}$.
3. $M_{m \times n}(\mathbb{R})$ — $m \times n$ real matrices with componentwise addition.
4. $\mathcal{P}_n$ — polynomials of degree $\leq n$ with real coefficients.
5. $\mathcal{P}$ — all polynomials (no degree bound). This is infinite-dimensional.
6. $C([0, 1])$ — continuous functions $[0, 1] \to \mathbb{R}$ with pointwise addition.
7. $L^2(\Omega, P)$ — square-integrable random variables on a probability space, modulo almost-sure equality (Subject 1).
8. The solutions to a homogeneous linear ODE $y'' + p(x)y' + q(x)y = 0$ form a 2-dimensional subspace of $C^2(\mathbb{R})$.

**Definition 0.2.1.3 (Subspace).** $W \subseteq V$ is a *subspace* if:
- $\mathbf{0} \in W$;
- $\mathbf{v}, \mathbf{w} \in W \Rightarrow \mathbf{v} + \mathbf{w} \in W$ (closed under addition);
- $\mathbf{v} \in W, \alpha \in \mathbb{F} \Rightarrow \alpha\mathbf{v} \in W$ (closed under scaling).

Equivalently, $W$ is non-empty and closed under all linear combinations.

*Sanity check.* The second and third conditions together can be phrased as: $\alpha \mathbf{v} + \beta \mathbf{w} \in W$ for all $\mathbf{v}, \mathbf{w} \in W$ and $\alpha, \beta \in \mathbb{F}$. The requirement $\mathbf{0} \in W$ follows from non-emptiness and closure ($\mathbf{v} \in W \Rightarrow 0 \cdot \mathbf{v} = \mathbf{0} \in W$), so the three conditions are not minimal; they are stated for clarity.

**Definition 0.2.1.4 (Linear combination, span).** A *linear combination* of $\mathbf{v}_1, \ldots, \mathbf{v}_k \in V$ is any vector $\sum_{i=1}^k \alpha_i \mathbf{v}_i$ with $\alpha_i \in \mathbb{F}$. The *span* of a set $S \subseteq V$ is
$$\mathrm{span}(S) = \{\text{all (finite) linear combinations of elements of } S\}.$$
The span of $S$ is always a subspace of $V$ — the smallest subspace containing $S$.

**Definition 0.2.1.5 (Linear independence).** A finite set $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is *linearly independent* if the only scalars $\alpha_1, \ldots, \alpha_k$ with $\sum_i \alpha_i \mathbf{v}_i = \mathbf{0}$ are $\alpha_1 = \cdots = \alpha_k = 0$. A set $S$ (possibly infinite) is linearly independent iff every finite subset is.

*Intuition.* Linear independence means no vector in the set can be expressed as a linear combination of the others. Dependence means there is redundancy — some vector is a combination of the others.

**Definition 0.2.1.6 (Basis, dimension).** A *basis* of $V$ is a linearly independent spanning set. The *dimension* of $V$, $\dim V$, is the cardinality of any basis. (We will prove below that all bases have the same cardinality — this is what makes dimension well-defined.)

*Examples.*
- $\dim \mathbb{R}^n = n$: standard basis $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$.
- $\dim M_{m \times n}(\mathbb{R}) = mn$: basis is $\{E_{ij}\}_{1 \leq i \leq m, 1 \leq j \leq n}$, where $E_{ij}$ has a $1$ in position $(i,j)$ and zeros elsewhere.
- $\dim \mathcal{P}_n = n + 1$: basis $\{1, x, x^2, \ldots, x^n\}$.
- $\dim \mathcal{P} = \aleph_0$: infinite (countable) basis $\{1, x, x^2, x^3, \ldots\}$.
- $\dim C([0, 1]) = \mathfrak{c}$ (uncountable).

## Key Results

### Theorem 0.2.1.7 (Uniqueness of representation in a basis)

Let $B = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ be a basis of $V$ (or more generally, a linearly independent set). Then every $\mathbf{v} \in \mathrm{span}(B)$ has a *unique* representation $\mathbf{v} = \sum_i \alpha_i \mathbf{b}_i$.

**Proof.** Existence follows from $\mathbf{v} \in \mathrm{span}(B)$. Uniqueness: suppose $\sum_i \alpha_i \mathbf{b}_i = \sum_i \beta_i \mathbf{b}_i$. Subtract: $\sum_i (\alpha_i - \beta_i) \mathbf{b}_i = \mathbf{0}$. By linear independence, $\alpha_i - \beta_i = 0$ for all $i$, so $\alpha_i = \beta_i$. $\square$

### Theorem 0.2.1.8 (Exchange Lemma — Steinitz)

Let $V$ be a vector space, $L = \{\mathbf{v}_1, \ldots, \mathbf{v}_m\}$ a linearly independent set, and $S = \{\mathbf{w}_1, \ldots, \mathbf{w}_n\}$ a spanning set of $V$. Then $m \leq n$, and one can replace $m$ elements of $S$ by the elements of $L$ to obtain a new spanning set.

**Proof (by induction on $m$).**

*Base case $m = 0$:* $L$ is empty; nothing to exchange; $0 \leq n$.

*Inductive step:* Assume true for $m - 1$. Given $L = \{\mathbf{v}_1, \ldots, \mathbf{v}_m\}$ linearly independent and $S$ spanning with $|S| = n$. Apply the inductive hypothesis to $\{\mathbf{v}_1, \ldots, \mathbf{v}_{m-1}\}$ and $S$: we get $m - 1 \leq n$ and can exchange $m - 1$ elements of $S$ with these vectors. Relabel so that after exchange, $S' = \{\mathbf{v}_1, \ldots, \mathbf{v}_{m-1}, \mathbf{w}_m, \ldots, \mathbf{w}_n\}$ is spanning.

Since $S'$ spans $V$ and $\mathbf{v}_m \in V$:
$$\mathbf{v}_m = \sum_{i=1}^{m-1} \alpha_i \mathbf{v}_i + \sum_{j=m}^{n} \beta_j \mathbf{w}_j.$$

Some $\beta_j \neq 0$: if all $\beta_j = 0$, we would have $\mathbf{v}_m = \sum \alpha_i \mathbf{v}_i$, contradicting linear independence of $L$. Without loss of generality, $\beta_m \neq 0$ (relabel).

In particular, $m \leq n$ (we need at least one $\mathbf{w}_j$ with $j \in \{m, \ldots, n\}$, so $n \geq m$).

Solve for $\mathbf{w}_m$:
$$\mathbf{w}_m = \frac{1}{\beta_m}\left(\mathbf{v}_m - \sum_{i=1}^{m-1} \alpha_i \mathbf{v}_i - \sum_{j=m+1}^{n} \beta_j \mathbf{w}_j\right).$$

Hence $\mathbf{w}_m \in \mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_m, \mathbf{w}_{m+1}, \ldots, \mathbf{w}_n\}$. Replace $\mathbf{w}_m$ in $S'$ by $\mathbf{v}_m$:
$$S'' = \{\mathbf{v}_1, \ldots, \mathbf{v}_m, \mathbf{w}_{m+1}, \ldots, \mathbf{w}_n\}.$$

$S''$ spans $V$: any vector is a linear combination of $S'$, and $\mathbf{w}_m$ in $S'$ is expressible via $S''$, so the linear combination can be rewritten using $S''$. $\square$

### Theorem 0.2.1.9 (Dimension is well-defined)

Any two (finite) bases of the same vector space $V$ have the same number of elements.

**Proof.** Let $B_1, B_2$ be bases with $|B_1| = m, |B_2| = n$. Since $B_1$ is linearly independent and $B_2$ spans, by exchange lemma $m \leq n$. By symmetry $n \leq m$. So $m = n$. $\square$

The infinite-dimensional version requires a different argument (typically Zorn's lemma to produce a basis in the first place), but the conclusion generalizes: any two bases have the same cardinality.

### Theorem 0.2.1.10 (Finite-dimensional subspace criterion)

Let $V$ be finite-dimensional and $W \subseteq V$ a subspace. Then $\dim W \leq \dim V$, with equality iff $W = V$.

**Proof.** A basis of $W$ is a linearly independent set in $V$, hence by exchange lemma has at most $\dim V$ elements. If $\dim W = \dim V$, a basis of $W$ is a linearly independent set of size $\dim V$ in $V$; we show it is a basis of $V$. It is linearly independent; we show it spans. If not, some $\mathbf{v} \in V \setminus \mathrm{span}(\text{basis of } W)$, and adding $\mathbf{v}$ produces a linearly independent set of size $\dim V + 1$ — impossible by exchange. So the basis of $W$ spans $V$, hence $W = V$. $\square$

## Worked Examples

### Example 0.2.1.11 (Direct: polynomials of degree $\leq 2$)

*Show that $\{1, x - 1, (x-1)^2\}$ is a basis of $\mathcal{P}_2$.*

**Solution.** We show linear independence and spanning.

*Independence.* Suppose $\alpha + \beta(x - 1) + \gamma(x - 1)^2 = 0$ for all $x$. Expand: $\alpha + \beta x - \beta + \gamma x^2 - 2\gamma x + \gamma = 0$, i.e.,
$$\gamma x^2 + (\beta - 2\gamma) x + (\alpha - \beta + \gamma) = 0.$$
Equating coefficients to zero: $\gamma = 0$, $\beta - 2\gamma = 0 \Rightarrow \beta = 0$, $\alpha - \beta + \gamma = 0 \Rightarrow \alpha = 0$.

*Spanning.* We have $\dim \mathcal{P}_2 = 3$, and an independent set of three elements in a 3-dimensional space is automatically spanning (Theorem 0.2.1.10). $\square$

*Observation.* This is the Taylor basis around $x = 1$: polynomials re-expressed in powers of $(x - 1)$. A change of basis in $\mathcal{P}_2$.

### Example 0.2.1.12 (Subtle: continuous functions are not finite-dimensional)

*Show that $\{1, x, x^2, x^3, \ldots\}$ is a linearly independent subset of $C([0, 1])$. Conclude $\dim C([0, 1]) = \infty$.*

**Solution.** Suppose $\sum_{i=0}^n \alpha_i x^i = 0$ (as a function on $[0, 1]$). A polynomial that is zero on a set of positive measure (in fact, on more than $n$ points) must have all coefficients zero. Hence $\alpha_0 = \cdots = \alpha_n = 0$. So $\{x^i\}_{i \geq 0}$ is linearly independent. An independent set in $C([0,1])$ of arbitrary finite size means $\dim C([0,1]) \geq n$ for every $n$; hence infinite. $\square$

**Remark.** In fact $\dim C([0,1]) = \mathfrak{c}$ (uncountable). This does not matter much for computation — in practice we work with *finite-dimensional approximations* (e.g., polynomial bases up to some degree, or finite-dimensional projections onto eigenfunctions of a kernel). But it matters for theory: the rank–nullity theorem, which we prove next, is a *finite-dimensional* statement; the infinite-dimensional version requires Banach space methods (Subject 9).

## Computational Implementation

```python
import numpy as np
from itertools import product

def is_linearly_independent(vectors):
    """Check if a list of numpy vectors is linearly independent."""
    M = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(M)
    return rank == len(vectors)

def span_contains(vectors, target, atol=1e-10):
    """Check if target is in span(vectors) by solving least-squares and comparing."""
    M = np.column_stack(vectors)
    coeffs, *_ = np.linalg.lstsq(M, target, rcond=None)
    return np.allclose(M @ coeffs, target, atol=atol)

# Example: check {1, x-1, (x-1)^2} at evaluation points
xs = np.linspace(0, 1, 50)
v1 = np.ones_like(xs)
v2 = xs - 1
v3 = (xs - 1) ** 2
print("Independent:", is_linearly_independent([v1, v2, v3]))  # True

# Target: x^2 should be in the span
target = xs ** 2
print("x^2 in span:", span_contains([v1, v2, v3], target))  # True

# Dimension computation via rank
A = np.random.randn(5, 3)
print("Rank (should be 3):", np.linalg.matrix_rank(A))
```

## [QUANT APPLICATION]

**Factor models and portfolio subspaces.** In a $k$-factor model for $N$ stocks, asset returns are approximated as
$$\mathbf{r} = \mathbf{B} \mathbf{f} + \boldsymbol\varepsilon, \qquad \mathbf{B} \in \mathbb{R}^{N \times k}, \; \mathbf{f} \in \mathbb{R}^k.$$
The column space of $\mathbf{B}$ is a $k$-dimensional subspace of $\mathbb{R}^N$ — the subspace of returns explained by the factors. The orthogonal complement is the *idiosyncratic* subspace. The *intended* portfolio allocations live in $\mathrm{col}(\mathbf{B})$ (factor bets) or its complement (market-neutral, pure idiosyncratic bets). Knowing the dimension of the subspace (the number of factors) is not just a fitting choice; it is a claim about the dimensionality of your signal. You will meet this again in Subject 8 where RMT tells you how to *test* whether an extracted factor is real or noise.

## Exercises

### ★ (Foundation)

**E0.2.1.1.** Verify that the set of diagonal $2 \times 2$ real matrices forms a vector subspace of $M_{2\times2}(\mathbb{R})$. Give a basis and find its dimension.

*Solution.* Contains $\mathbf{0}$ (zero matrix is diagonal). Closed under sum (sum of diagonals is diagonal). Closed under scaling (scalar times diagonal is diagonal). Basis: $\{E_{11}, E_{22}\}$. Dimension 2.

**E0.2.1.2.** Show that $\{(1, 2, 3), (4, 5, 6), (7, 8, 9)\}$ is linearly dependent in $\mathbb{R}^3$.

*Solution.* $(7, 8, 9) = 2 \cdot (4, 5, 6) - (1, 2, 3)$. Equivalently, determinant of the matrix with these as rows is zero.

**E0.2.1.3.** Show that the subset $W = \{(x, y, z) \in \mathbb{R}^3 : x + y + z = 0\}$ is a subspace of $\mathbb{R}^3$, and give a basis.

*Solution.* Contains origin, closed under sum and scaling. Basis: $\{(1, -1, 0), (1, 0, -1)\}$ (two linearly independent solutions of the plane equation). Dimension 2.

### ★★ (Intermediate)

**E0.2.1.4.** Let $U, W$ be subspaces of $V$. Prove $\dim(U + W) + \dim(U \cap W) = \dim U + \dim W$ (the *dimension formula*).

*Solution.* Let $\{\mathbf{b}_1, \ldots, \mathbf{b}_k\}$ be a basis of $U \cap W$. Extend to a basis $\{\mathbf{b}_1, \ldots, \mathbf{b}_k, \mathbf{u}_1, \ldots, \mathbf{u}_r\}$ of $U$ (possible because $U \cap W \subseteq U$ and we can extend an independent set to a basis). Extend the same base to a basis $\{\mathbf{b}_1, \ldots, \mathbf{b}_k, \mathbf{w}_1, \ldots, \mathbf{w}_s\}$ of $W$. So $\dim U = k + r, \dim W = k + s$.

Claim: $\mathcal{B} = \{\mathbf{b}_i\} \cup \{\mathbf{u}_j\} \cup \{\mathbf{w}_l\}$ is a basis of $U + W$.

*Spanning.* Any $\mathbf{u} + \mathbf{w} \in U + W$ is a linear combination of elements of $\mathcal{B}$ (from the two bases). Done.

*Independence.* Suppose $\sum \alpha_i \mathbf{b}_i + \sum \beta_j \mathbf{u}_j + \sum \gamma_l \mathbf{w}_l = \mathbf{0}$. Rearrange: $\sum \beta_j \mathbf{u}_j = -\sum \alpha_i \mathbf{b}_i - \sum \gamma_l \mathbf{w}_l$. LHS is in $U$; RHS is in $W$. So LHS is in $U \cap W$. But LHS is a linear combination of $\{\mathbf{u}_j\}$, which are linearly independent of $U \cap W$ (they were added to extend $\{\mathbf{b}_i\}$ to a basis of $U$). The only way LHS can be in $U \cap W$ is if LHS $= \mathbf{0}$, which forces $\beta_j = 0$ for all $j$ (since $\{\mathbf{b}_i\} \cup \{\mathbf{u}_j\}$ is a basis of $U$). Similarly, the remaining equation $\sum \alpha_i \mathbf{b}_i + \sum \gamma_l \mathbf{w}_l = \mathbf{0}$ inside $W$'s basis forces $\alpha_i = 0, \gamma_l = 0$.

Hence $|\mathcal{B}| = k + r + s$, giving
$$\dim(U + W) = k + r + s = (k + r) + (k + s) - k = \dim U + \dim W - \dim(U \cap W). \square$$

**E0.2.1.5.** Let $\mathcal{P}_n$ denote polynomials of degree $\leq n$. For a fixed real $a$, define $T: \mathcal{P}_n \to \mathcal{P}_n$ by $T(p)(x) = p(x + a) - p(x)$. Show that $T(\mathcal{P}_n) \subseteq \mathcal{P}_{n-1}$ and use this to argue that $T$ is not injective.

*Solution.* If $p(x) = c_n x^n + \text{lower}$, then $p(x + a) - p(x) = c_n[(x+a)^n - x^n] + \text{lower} = c_n \cdot n a x^{n-1} + \text{lower}$, degree $\leq n - 1$. So $T$ maps $\mathcal{P}_n$ into $\mathcal{P}_{n-1}$, a strictly smaller space when $a \neq 0$ (it has smaller dimension). $T$ cannot be injective because its image has dimension $\leq n < n + 1 = \dim \mathcal{P}_n$ (rank-nullity, Topic 0.2.2, forces a nonzero kernel). Explicit kernel: constants, since $T(c) = c - c = 0$.

**E0.2.1.6.** Let $V$ be a 4-dimensional vector space and $U, W$ subspaces of $V$ with $\dim U = \dim W = 3$. Prove $\dim(U \cap W) \geq 2$.

*Solution.* By the dimension formula (E0.2.1.4), $\dim(U + W) = 3 + 3 - \dim(U \cap W)$. Since $U + W \subseteq V$, $\dim(U + W) \leq 4$. So $\dim(U \cap W) \geq 3 + 3 - 4 = 2$.

### ★★★ (Challenge)

**E0.2.1.7.** Prove that every vector space has a basis (Hamel basis), assuming the axiom of choice. Formally: for any vector space $V$, there is a linearly independent $B \subseteq V$ that spans $V$.

*Hint.* Apply Zorn's lemma to the poset of linearly independent subsets of $V$ ordered by inclusion. The union of a chain of linearly independent sets is linearly independent (check this — a finite relation can involve only finitely many vectors, so only one element of the chain, by the chain property). The maximal element is a basis.

**E0.2.1.8.** Let $V = \mathbb{R}^\mathbb{N}$ (sequences of reals). Show that the obvious "basis" $\{\mathbf{e}_i\}_{i \in \mathbb{N}}$ — where $\mathbf{e}_i$ has a $1$ in position $i$ and zeros elsewhere — *does not* span $V$. Deduce that the Hamel basis of $V$ is uncountable.

*Hint.* The vector $(1, 1, 1, \ldots) \in V$ is not in the span of the $\mathbf{e}_i$'s, because a linear combination of the $\mathbf{e}_i$'s is by definition a *finite* linear combination. Any Hamel basis must therefore be uncountable — a non-constructive fact that underscores the distinction between "algebraic span" and "topological closure of span" (the latter is what Hilbert/Banach space theory studies in Subject 9).

**E0.2.1.9.** Consider the vector space $V = C([-1, 1])$ of continuous real-valued functions. Define the inner product $\langle f, g \rangle = \int_{-1}^1 f g \, dx$. The Legendre polynomials $\{P_n\}_{n \geq 0}$ — defined by orthogonalizing $\{1, x, x^2, \ldots\}$ under this inner product — form a "basis" in what sense? Make precise the difference between (a) they form a Hamel basis of the polynomial subspace $\mathcal{P} \subseteq V$, and (b) they form an orthonormal *Schauder* basis of $V$ (in a completed space — the $L^2$ space, which you will meet in Subject 1).

*Hint.* A Hamel basis requires every element to be a *finite* linear combination of basis elements; a Schauder basis requires the existence of a convergent *infinite series* in the norm. The latter is a topological notion, the former an algebraic one.

---

# Topic 0.2.2 — Linear Transformations and Rank–Nullity

## Motivation

Once you have vector spaces, the next question is what maps between them preserve the structure. These are the *linear transformations*. They are the abstract objects matrices represent. Matrices are *representations* of linear transformations in specific bases; linear transformations are the structural objects. This distinction matters when (a) you change bases, (b) you work on infinite-dimensional spaces where there is no finite matrix, and (c) you want to state theorems — like rank-nullity — in a basis-independent way.

## Prerequisites

Topic 0.2.1.

## Definitions

**Definition 0.2.2.1 (Linear transformation).** Let $V, W$ be vector spaces over the same field $\mathbb{F}$. A *linear transformation* (or *linear map*) $T: V \to W$ is a function satisfying
$$T(\alpha \mathbf{v}_1 + \beta \mathbf{v}_2) = \alpha T(\mathbf{v}_1) + \beta T(\mathbf{v}_2)$$
for all $\mathbf{v}_1, \mathbf{v}_2 \in V, \alpha, \beta \in \mathbb{F}$.

Equivalently: $T$ preserves addition ($T(\mathbf{v}_1 + \mathbf{v}_2) = T(\mathbf{v}_1) + T(\mathbf{v}_2)$) and scaling ($T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})$).

*Immediate consequences.* $T(\mathbf{0}) = \mathbf{0}$ (set $\alpha = \beta = 0$). $T$ is determined by its values on a basis (since every $\mathbf{v}$ is a linear combination of basis vectors).

**Examples.**
- Left-multiplication by a matrix $A \in \mathbb{R}^{m \times n}$: $T(\mathbf{v}) = A\mathbf{v}$ is a linear map $\mathbb{R}^n \to \mathbb{R}^m$.
- Differentiation $D: \mathcal{P}_n \to \mathcal{P}_{n-1}$, $D(p) = p'$.
- Integration $I: C([0, 1]) \to \mathbb{R}$, $I(f) = \int_0^1 f \, dx$.
- Expectation $E: L^1(\Omega, P) \to \mathbb{R}$, $E(X) = \int X \, dP$. (Linearity of expectation is the statement that $E$ is a linear map.)
- Fourier transform, Laplace transform — linear maps on function spaces.

**Definition 0.2.2.2 (Kernel, image).**
$$\ker T = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}, \qquad \mathrm{im}\, T = T(V) = \{T(\mathbf{v}) : \mathbf{v} \in V\}.$$

Both $\ker T \subseteq V$ and $\mathrm{im}\, T \subseteq W$ are subspaces (direct verification from linearity).

The *rank* of $T$ is $\mathrm{rank}(T) = \dim(\mathrm{im}\, T)$, and the *nullity* is $\mathrm{nullity}(T) = \dim(\ker T)$.

**Definition 0.2.2.3 (Matrix representation).** Let $T: V \to W$ be linear between finite-dimensional spaces with bases $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ of $V$ and $\mathcal{C} = \{\mathbf{c}_1, \ldots, \mathbf{c}_m\}$ of $W$. The matrix $[T]_{\mathcal{B}}^{\mathcal{C}} \in M_{m \times n}(\mathbb{F})$ has $j$-th column equal to the coordinate vector of $T(\mathbf{b}_j)$ in basis $\mathcal{C}$:
$$T(\mathbf{b}_j) = \sum_{i=1}^m A_{ij} \mathbf{c}_i, \quad [T]_{\mathcal{B}}^{\mathcal{C}} = (A_{ij}).$$

*Intuition.* The matrix reads off how each basis vector of $V$ lands in the basis of $W$. Once you know those, linearity extends the map to all of $V$.

## Key Results

### Theorem 0.2.2.4 (Rank–Nullity)

Let $V$ be finite-dimensional and $T: V \to W$ linear. Then
$$\dim V = \mathrm{rank}(T) + \mathrm{nullity}(T).$$

**Proof.** Let $\dim V = n, \dim \ker T = k$. Choose a basis $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ of $\ker T$, and extend to a basis $\{\mathbf{v}_1, \ldots, \mathbf{v}_k, \mathbf{u}_1, \ldots, \mathbf{u}_{n-k}\}$ of $V$ (possible by extending an independent set; this was developed in Topic 0.2.1). We show $\{T(\mathbf{u}_1), \ldots, T(\mathbf{u}_{n-k})\}$ is a basis of $\mathrm{im}\, T$; hence $\mathrm{rank}(T) = n - k$, giving $\mathrm{rank}(T) + \mathrm{nullity}(T) = (n - k) + k = n$.

*Spanning.* Any element of $\mathrm{im}\, T$ is $T(\mathbf{v})$ for some $\mathbf{v} \in V$. Write $\mathbf{v} = \sum_i \alpha_i \mathbf{v}_i + \sum_j \beta_j \mathbf{u}_j$. Apply $T$:
$$T(\mathbf{v}) = \sum_i \alpha_i T(\mathbf{v}_i) + \sum_j \beta_j T(\mathbf{u}_j) = \sum_j \beta_j T(\mathbf{u}_j),$$
since $\mathbf{v}_i \in \ker T$ makes $T(\mathbf{v}_i) = \mathbf{0}$. So $T(\mathbf{v}) \in \mathrm{span}\{T(\mathbf{u}_j)\}$.

*Independence.* Suppose $\sum_j \gamma_j T(\mathbf{u}_j) = \mathbf{0}$. By linearity $T(\sum_j \gamma_j \mathbf{u}_j) = \mathbf{0}$, so $\sum_j \gamma_j \mathbf{u}_j \in \ker T$. Write $\sum_j \gamma_j \mathbf{u}_j = \sum_i \delta_i \mathbf{v}_i$ for some scalars $\delta_i$. Rearrange: $\sum_j \gamma_j \mathbf{u}_j - \sum_i \delta_i \mathbf{v}_i = \mathbf{0}$. Since $\{\mathbf{v}_i, \mathbf{u}_j\}$ is a basis of $V$, all coefficients are zero: $\gamma_j = 0, \delta_i = 0$.

Hence $\{T(\mathbf{u}_j)\}$ is a basis of $\mathrm{im}\, T$, and $|\{T(\mathbf{u}_j)\}| = n - k$. $\square$

**Why this matters.** Rank-nullity is the single most-used structural theorem in finite-dimensional linear algebra. It says: *the domain decomposes into a kernel (where $T$ kills everything) and a complement (on which $T$ acts as a bijection onto $\mathrm{im}\, T$).* Many questions — solvability of linear systems, dimensionality of solution spaces of ODEs, surjectivity of a linear map — reduce to a rank-nullity count. In a quant setting, when you check whether a signal matrix $S \in \mathbb{R}^{T \times K}$ (T days, K signals) has full column rank (i.e., rank K), rank-nullity tells you that a deficient rank corresponds to a non-trivial kernel — a linear combination of your signals that is *identically zero on your data*. That identifies redundant signals. Same math, different dress.

### Theorem 0.2.2.5 (Matrix of composition)

Let $T: V \to W, S: W \to U$ linear, with bases $\mathcal{B}, \mathcal{C}, \mathcal{D}$ of $V, W, U$ respectively. Then
$$[S \circ T]_{\mathcal{B}}^{\mathcal{D}} = [S]_{\mathcal{C}}^{\mathcal{D}} \cdot [T]_{\mathcal{B}}^{\mathcal{C}}.$$

**Proof.** Let $A = [T]_{\mathcal{B}}^{\mathcal{C}} = (A_{ij})$ and $B = [S]_{\mathcal{C}}^{\mathcal{D}} = (B_{ki})$. Then for basis vector $\mathbf{b}_j$ of $V$:
$$S(T(\mathbf{b}_j)) = S\left(\sum_i A_{ij} \mathbf{c}_i\right) = \sum_i A_{ij} S(\mathbf{c}_i) = \sum_i A_{ij} \sum_k B_{ki} \mathbf{d}_k = \sum_k \left(\sum_i B_{ki} A_{ij}\right) \mathbf{d}_k.$$
The $(k, j)$ entry of $[S \circ T]$ is $\sum_i B_{ki} A_{ij} = (BA)_{kj}$. $\square$

This is the origin of matrix multiplication. Matrix multiplication is *defined* so that the matrix of a composition is the product of matrices.

### Theorem 0.2.2.6 (Change of basis)

Let $T: V \to V$ be linear, with bases $\mathcal{B}, \mathcal{B}'$ of $V$, and let $P = [\mathrm{id}]_{\mathcal{B}'}^{\mathcal{B}}$ be the change-of-basis matrix (columns are $\mathcal{B}'$-vectors expressed in the $\mathcal{B}$-basis). Then
$$[T]_{\mathcal{B}'}^{\mathcal{B}'} = P^{-1} [T]_{\mathcal{B}}^{\mathcal{B}} P.$$

**Proof.** Use $[S \circ T]_{\mathcal{B}_1}^{\mathcal{B}_3} = [S]_{\mathcal{B}_2}^{\mathcal{B}_3} [T]_{\mathcal{B}_1}^{\mathcal{B}_2}$ repeatedly:
$$[T]_{\mathcal{B}'}^{\mathcal{B}'} = [\mathrm{id}]_{\mathcal{B}}^{\mathcal{B}'} [T]_{\mathcal{B}}^{\mathcal{B}} [\mathrm{id}]_{\mathcal{B}'}^{\mathcal{B}} = P^{-1} [T]_{\mathcal{B}}^{\mathcal{B}} P,$$
noting $[\mathrm{id}]_{\mathcal{B}}^{\mathcal{B}'} = P^{-1}$ (inverses of change-of-basis). $\square$

This similarity transformation is the central computational fact for diagonalization (Topic 0.2.4).

## Worked Examples

### Example 0.2.2.7 (Direct: kernel and image of a matrix)

*Let $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix}$. Find $\ker A$, $\mathrm{im}\, A$, and verify rank-nullity.*

**Solution.** Row-reduce: row 2 is $2 \times$ row 1, row 3 is $3 \times$ row 1. So $A$ has rank 1. Row-reduced form: $\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$.

$\ker A$: solve $x + 2y + 3z = 0$. Two free parameters: $y, z$. $x = -2y - 3z$. Basis: $\{(-2, 1, 0), (-3, 0, 1)\}$. Dimension 2.

$\mathrm{im}\, A$: column space. Every column is a multiple of $(1, 2, 3)^\top$. Basis: $\{(1, 2, 3)^\top\}$. Dimension 1.

Rank-nullity check: $\dim(\ker) + \dim(\mathrm{im}) = 2 + 1 = 3 = \dim \mathbb{R}^3$. ✓

### Example 0.2.2.8 (Subtle: rank of $A$ equals rank of $A^\top$)

*Prove $\mathrm{rank}(A) = \mathrm{rank}(A^\top)$.*

**Solution.** This is the statement "row rank = column rank", a non-obvious fact. Here is a clean argument.

Let $A \in \mathbb{R}^{m \times n}$ have column rank $r$ — the dimension of the column space. Choose a basis $\mathbf{c}_1, \ldots, \mathbf{c}_r$ of the column space (each $\mathbf{c}_i \in \mathbb{R}^m$). Form $C \in \mathbb{R}^{m \times r}$ with these as columns. Each column $\mathbf{a}_j$ of $A$ is a linear combination of $\mathbf{c}_i$'s:
$$\mathbf{a}_j = \sum_{i=1}^r R_{ij} \mathbf{c}_i, \quad R \in \mathbb{R}^{r \times n}.$$
So $A = CR$. Now transpose: $A^\top = R^\top C^\top$. The column space of $A^\top = R^\top C^\top$ is contained in the column space of $R^\top$ (since every column of $R^\top C^\top$ is $R^\top$ times a column of $C^\top$, which is $r$-dimensional...). Actually, we need $\mathrm{rank}(A^\top) \leq \mathrm{rank}(R^\top) \leq r$.

Wait, a direct dimension argument: $A^\top = R^\top C^\top$, so every column of $A^\top$ is in the column space of $R^\top$, which has at most $r$ columns. Hence column rank of $A^\top \leq r$.

Symmetric argument (apply the same logic with $A$ replaced by $A^\top$): column rank of $A \leq $ column rank of $A^\top$, i.e., $r \leq$ column rank of $A^\top$.

Combining, column rank of $A$ = column rank of $A^\top$. $\square$

**Moral.** Every matrix has the same rank as its transpose. This lets us compute rank either by reducing rows or reducing columns; both give the same answer.

## Computational Implementation

```python
import numpy as np

# Rank-nullity verification
def kernel_basis(A, tol=1e-10):
    """Return a matrix whose columns form a basis for the kernel of A."""
    # SVD-based: kernel is right singular vectors corresponding to zero singular values
    U, s, Vt = np.linalg.svd(A)
    # Keep columns of V (rows of Vt) where singular value < tol
    null_mask = s < tol
    # In case A has more columns than s, need padding
    if len(s) < Vt.shape[0]:
        null_mask = np.concatenate([null_mask, [True] * (Vt.shape[0] - len(s))])
    return Vt[null_mask].T

A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]], dtype=float)

rank = np.linalg.matrix_rank(A)
N = kernel_basis(A)
nullity = N.shape[1]

print("Rank:", rank)                  # 1
print("Nullity:", nullity)            # 2
print("Rank + Nullity:", rank + nullity)  # 3, matches dim domain

# Verify that A @ N is zero
print("A @ ker_basis:\n", A @ N)      # Should be ~zero
```

## [QUANT APPLICATION]

**Collinearity diagnosis in signal matrices.** You have $K$ candidate signals over $T$ days: matrix $S \in \mathbb{R}^{T \times K}$. Regressing forward returns $\mathbf{r} \in \mathbb{R}^T$ on these signals requires inverting $S^\top S$. If $S$ has rank $< K$, then $S^\top S$ is singular and the regression is ill-posed.

Rank-nullity tells you: $\mathrm{rank}(S) < K$ iff $\dim \ker S > 0$, i.e., there exists a non-trivial linear combination of your signals that is identically zero on your data. The SVD (Topic 0.2.6) will give you exactly that combination — it appears as the right singular vector associated with the smallest singular value. Identifying and pruning such collinear combinations is standard pre-regression hygiene; the theoretical guarantee that rank-deficiency ↔ nontrivial kernel is the rank-nullity theorem.

## Exercises

### ★ (Foundation)

**E0.2.2.1.** Let $T: \mathbb{R}^3 \to \mathbb{R}^2$, $T(x, y, z) = (x + y, y + z)$. Find $\ker T$ and $\mathrm{im}\, T$.

*Solution.* $T$ has matrix $\begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix}$. $\ker T$: $x + y = 0, y + z = 0$. One free parameter ($z$), giving $y = -z, x = z$. Basis $\{(1, -1, 1)\}$. Dimension 1. $\mathrm{im}\, T$: spanned by columns $(1, 0), (1, 1), (0, 1)$, which span $\mathbb{R}^2$. So $\mathrm{im}\, T = \mathbb{R}^2$, dimension 2. Rank-nullity: $2 + 1 = 3 = \dim \mathbb{R}^3$. ✓

**E0.2.2.2.** Let $D: \mathcal{P}_3 \to \mathcal{P}_3$ be differentiation. Find $\ker D$, $\mathrm{im}\, D$, and verify rank-nullity.

*Solution.* $\ker D$: polynomials with zero derivative, i.e., constants. Basis $\{1\}$, dim 1. $\mathrm{im}\, D \subseteq \mathcal{P}_2$: actually $\mathrm{im}\, D = \mathcal{P}_2$ (every polynomial of degree $\leq 2$ is the derivative of one of degree $\leq 3$, up to constant). Dimension 3. Sum: $1 + 3 = 4 = \dim \mathcal{P}_3$. ✓

**E0.2.2.3.** Prove that if $T: V \to W$ is linear and $\dim V < \dim W$, then $T$ cannot be surjective.

*Solution.* By rank-nullity, $\mathrm{rank}(T) = \dim V - \mathrm{nullity}(T) \leq \dim V < \dim W$. But surjective means $\mathrm{rank}(T) = \dim W$. Contradiction. $\square$

### ★★ (Intermediate)

**E0.2.2.4.** Let $V$ be a finite-dimensional vector space. Prove that $T: V \to V$ linear is injective iff it is surjective.

*Solution.* $T$ injective iff $\ker T = \{\mathbf{0}\}$ iff $\mathrm{nullity}(T) = 0$. By rank-nullity, this is iff $\mathrm{rank}(T) = \dim V$ iff $\mathrm{im}\, T = V$ (since $\mathrm{im}\, T \subseteq V$ has dim $\dim V$ iff it is all of $V$) iff $T$ surjective. $\square$

**Remark.** This fails in infinite dimensions. Take $V = \mathbb{R}^\mathbb{N}$, and $T(a_1, a_2, \ldots) = (0, a_1, a_2, \ldots)$ (right-shift). Injective (kernel trivial), not surjective (no element maps to $(1, 0, 0, \ldots)$). This is one of the places finite-dimensional intuition breaks down; you will meet the full-fledged theory in Subject 9.

**E0.2.2.5.** Let $T: \mathbb{R}^n \to \mathbb{R}^n$ linear, with matrix $A$ in the standard basis. Show that $\det(A) \neq 0$ iff $T$ is bijective.

*Solution.* $T$ bijective iff $\ker T = \{\mathbf{0}\}$ iff the columns of $A$ are linearly independent iff $\mathrm{rank}(A) = n$ iff $A$ is non-singular iff $\det(A) \neq 0$.

**E0.2.2.6.** Let $A \in \mathbb{R}^{m \times n}$ with $\mathrm{rank}(A) = r$. Show $A$ can be factored as $A = BC$ with $B \in \mathbb{R}^{m \times r}$ and $C \in \mathbb{R}^{r \times n}$, both of rank $r$.

*Solution.* Let $\{\mathbf{c}_1, \ldots, \mathbf{c}_r\}$ be a basis of the column space of $A$. Form $B = [\mathbf{c}_1 | \cdots | \mathbf{c}_r] \in \mathbb{R}^{m \times r}$. Each column $\mathbf{a}_j$ of $A$ is a unique linear combination $\mathbf{a}_j = \sum_{i=1}^r C_{ij} \mathbf{c}_i$, so $A = BC$ where $C = (C_{ij}) \in \mathbb{R}^{r \times n}$. By construction, $B$ has rank $r$ (its columns are independent); $C$ has rank $r$ (else $A = BC$ would have rank $< r$). This factorization is called a *rank factorization* of $A$. $\square$

### ★★★ (Challenge)

**E0.2.2.7.** Let $V, W$ finite-dimensional. Show that the set $\mathrm{Hom}(V, W)$ of linear maps $V \to W$ is itself a vector space with $\dim \mathrm{Hom}(V, W) = \dim V \cdot \dim W$.

*Hint.* Fix bases and identify $\mathrm{Hom}(V, W)$ with $M_{m \times n}$ where $m = \dim W, n = \dim V$.

**E0.2.2.8.** Let $T: V \to V$ linear, $V$ finite-dimensional. Prove $V = \ker T^k \oplus \mathrm{im}\, T^k$ for sufficiently large $k$. (This is the *Fitting decomposition*, a precursor to the Jordan canonical form.)

*Hint.* Consider the decreasing chain $\mathrm{im}\, T \supseteq \mathrm{im}\, T^2 \supseteq \cdots$ and the increasing chain $\ker T \subseteq \ker T^2 \subseteq \cdots$. They stabilize at some $k$. Show $\ker T^k \cap \mathrm{im}\, T^k = \{\mathbf{0}\}$ and sum to $V$ by dimensions.

**E0.2.2.9.** Let $T: V \to W$ linear, $V$ finite-dimensional. Define the *dual* map $T^*: W^* \to V^*$ by $T^*(\phi) = \phi \circ T$ for $\phi \in W^* = \mathrm{Hom}(W, \mathbb{F})$. Prove $\mathrm{rank}(T^*) = \mathrm{rank}(T)$ and interpret this in terms of row rank versus column rank of the matrix.

*Hint.* Identify the matrix of $T^*$ in dual bases; it is $A^\top$. Then rank equality becomes row rank = column rank.

---

# Topic 0.2.3 — Inner Product Spaces

## Motivation

A vector space gives you addition and scaling. An *inner product* adds geometry: lengths, angles, orthogonality, projections. This is the structure you need to talk about "closest approximation", "orthogonal basis", and "projection onto a subspace" — concepts that show up everywhere from regression (the regression coefficient vector is the orthogonal projection of $\mathbf{y}$ onto the column space of $X$) to the measure-theoretic definition of conditional expectation in Subject 1. Cauchy–Schwarz and Gram–Schmidt are both here; you have seen them in JEE in specific cases; we now prove them in full generality.

## Prerequisites

Topics 0.2.1, 0.2.2.

## Definitions

**Definition 0.2.3.1 (Inner product).** Let $V$ be a vector space over $\mathbb{R}$. An *inner product* is a map $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfying, for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $\alpha \in \mathbb{R}$:

1. *Symmetry*: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$.
2. *Linearity in first argument*: $\langle \alpha \mathbf{u} + \mathbf{v}, \mathbf{w} \rangle = \alpha \langle \mathbf{u}, \mathbf{w} \rangle + \langle \mathbf{v}, \mathbf{w} \rangle$. (By symmetry, linearity in the second argument follows.)
3. *Positive-definiteness*: $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$, with equality iff $\mathbf{v} = \mathbf{0}$.

A vector space with an inner product is an *inner product space*.

Over $\mathbb{C}$: replace symmetry with conjugate-symmetry $\langle \mathbf{u}, \mathbf{v} \rangle = \overline{\langle \mathbf{v}, \mathbf{u} \rangle}$; linearity is in the first argument, conjugate-linearity in the second.

**Examples.**
- $\mathbb{R}^n$ with $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_i x_i y_i$.
- $\mathbb{R}^n$ with $\langle \mathbf{x}, \mathbf{y} \rangle_A = \mathbf{x}^\top A \mathbf{y}$ for any symmetric positive-definite $A$.
- $C([a, b])$ with $\langle f, g \rangle = \int_a^b f(x) g(x) \, dx$.
- $L^2(\Omega, P)$ with $\langle X, Y \rangle = \mathbb{E}[XY]$. This is *the* inner product on square-integrable random variables.
- Matrix space $M_{m \times n}$ with Frobenius inner product $\langle A, B \rangle_F = \mathrm{tr}(A^\top B) = \sum_{i, j} A_{ij} B_{ij}$.

**Definition 0.2.3.2 (Norm induced by inner product).** $\|\mathbf{v}\| := \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$.

This satisfies the norm axioms (Topic 0.2.9), but we will establish the triangle inequality shortly, which is the non-trivial one.

**Definition 0.2.3.3 (Orthogonality).** $\mathbf{u} \perp \mathbf{v}$ iff $\langle \mathbf{u}, \mathbf{v} \rangle = 0$. A set is *orthogonal* if its elements are pairwise orthogonal; *orthonormal* if in addition each has norm 1.

**Definition 0.2.3.4 (Orthogonal complement).** For $S \subseteq V$, $S^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{s} \rangle = 0 \;\forall \mathbf{s} \in S\}$. Always a subspace of $V$.

## Key Results

### Theorem 0.2.3.5 (Cauchy–Schwarz inequality)

For any inner product space $V$ and $\mathbf{u}, \mathbf{v} \in V$:
$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|,$$
with equality iff $\mathbf{u}, \mathbf{v}$ are linearly dependent.

**Proof.** If $\mathbf{v} = \mathbf{0}$, both sides are 0; trivial. Assume $\mathbf{v} \neq \mathbf{0}$. For any $t \in \mathbb{R}$ consider
$$0 \leq \|\mathbf{u} - t\mathbf{v}\|^2 = \langle \mathbf{u} - t\mathbf{v}, \mathbf{u} - t\mathbf{v} \rangle = \|\mathbf{u}\|^2 - 2t\langle \mathbf{u}, \mathbf{v} \rangle + t^2 \|\mathbf{v}\|^2.$$
The RHS is a quadratic in $t$, non-negative for all $t$, so its discriminant is non-positive:
$$4\langle \mathbf{u}, \mathbf{v} \rangle^2 - 4 \|\mathbf{u}\|^2 \|\mathbf{v}\|^2 \leq 0,$$
i.e., $\langle \mathbf{u}, \mathbf{v} \rangle^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$. Take square roots: $|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \|\mathbf{v}\|$.

*Equality case.* Equality in Cauchy–Schwarz holds iff the quadratic in $t$ has a real root, iff $\mathbf{u} - t_0 \mathbf{v} = \mathbf{0}$ for some $t_0$, iff $\mathbf{u}$ is a scalar multiple of $\mathbf{v}$. (Together with the trivial case $\mathbf{v} = \mathbf{0}$, we can say: equality iff $\mathbf{u}, \mathbf{v}$ linearly dependent.) $\square$

*Alternative proof (one-liner).* Let $t_* = \langle \mathbf{u}, \mathbf{v} \rangle / \|\mathbf{v}\|^2$. Plug in:
$$0 \leq \|\mathbf{u} - t_* \mathbf{v}\|^2 = \|\mathbf{u}\|^2 - 2 t_* \langle \mathbf{u}, \mathbf{v} \rangle + t_*^2 \|\mathbf{v}\|^2 = \|\mathbf{u}\|^2 - \frac{\langle \mathbf{u}, \mathbf{v} \rangle^2}{\|\mathbf{v}\|^2}.$$
Multiply by $\|\mathbf{v}\|^2$: $\langle \mathbf{u}, \mathbf{v} \rangle^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$. This is the same result, obtained by choosing $t$ to minimize the quadratic — which is the *projection of $\mathbf{u}$ onto $\mathbf{v}$*.

**Intuition.** Cauchy–Schwarz is "correlation is at most one": the angle $\theta$ defined by $\cos \theta = \langle \mathbf{u}, \mathbf{v} \rangle / (\|\mathbf{u}\| \|\mathbf{v}\|)$ is a real number in $[-1, 1]$; equivalently, $|\cos \theta| \leq 1$. In the $L^2$ space of random variables, Cauchy–Schwarz says $|\mathrm{Cov}(X, Y)| \leq \sigma_X \sigma_Y$, which is the statement that correlation is in $[-1, 1]$. You use this every time you look at a correlation matrix.

### Theorem 0.2.3.6 (Triangle inequality)

For any inner product space and $\mathbf{u}, \mathbf{v} \in V$: $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$.

**Proof.**
$$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\langle \mathbf{u}, \mathbf{v} \rangle + \|\mathbf{v}\|^2 \leq \|\mathbf{u}\|^2 + 2 \|\mathbf{u}\| \|\mathbf{v}\| + \|\mathbf{v}\|^2 = (\|\mathbf{u}\| + \|\mathbf{v}\|)^2,$$
using Cauchy–Schwarz. Take square roots. $\square$

### Theorem 0.2.3.7 (Pythagorean theorem)

If $\mathbf{u} \perp \mathbf{v}$, then $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$.

**Proof.** $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\langle \mathbf{u}, \mathbf{v} \rangle + \|\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 0 + \|\mathbf{v}\|^2$. $\square$

More generally, for pairwise orthogonal $\mathbf{v}_1, \ldots, \mathbf{v}_n$:
$$\left\|\sum_i \mathbf{v}_i\right\|^2 = \sum_i \|\mathbf{v}_i\|^2.$$

### Theorem 0.2.3.8 (Orthogonal projection)

Let $W \subseteq V$ be a finite-dimensional subspace of an inner product space. For any $\mathbf{v} \in V$ there is a unique $\mathbf{p} \in W$ with $\mathbf{v} - \mathbf{p} \perp W$. This $\mathbf{p}$ is the *orthogonal projection* of $\mathbf{v}$ onto $W$, and it is the unique minimizer of $\|\mathbf{v} - \mathbf{w}\|$ over $\mathbf{w} \in W$.

**Proof.** Let $\{\mathbf{e}_1, \ldots, \mathbf{e}_k\}$ be an orthonormal basis of $W$ (possible by Gram–Schmidt, Theorem 0.2.3.10 below). Define
$$\mathbf{p} := \sum_{i=1}^k \langle \mathbf{v}, \mathbf{e}_i \rangle \mathbf{e}_i.$$

*Perpendicularity.* For each $j$:
$$\langle \mathbf{v} - \mathbf{p}, \mathbf{e}_j \rangle = \langle \mathbf{v}, \mathbf{e}_j \rangle - \sum_i \langle \mathbf{v}, \mathbf{e}_i \rangle \langle \mathbf{e}_i, \mathbf{e}_j \rangle = \langle \mathbf{v}, \mathbf{e}_j \rangle - \langle \mathbf{v}, \mathbf{e}_j \rangle = 0,$$
using $\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \delta_{ij}$. Since $\mathbf{v} - \mathbf{p}$ is orthogonal to every $\mathbf{e}_j$, it is orthogonal to their span $W$.

*Uniqueness.* Suppose $\mathbf{p}'$ also satisfies $\mathbf{p}' \in W, \mathbf{v} - \mathbf{p}' \perp W$. Then $\mathbf{p} - \mathbf{p}' = (\mathbf{v} - \mathbf{p}') - (\mathbf{v} - \mathbf{p}) \in W$ (both summands have LHS in $V$, and the difference is $\mathbf{p} - \mathbf{p}' \in W$). Also $\mathbf{p} - \mathbf{p}'$ is orthogonal to every element of $W$ (linearity of orthogonality). In particular, $\mathbf{p} - \mathbf{p}'$ is orthogonal to itself, so $\|\mathbf{p} - \mathbf{p}'\|^2 = \langle \mathbf{p} - \mathbf{p}', \mathbf{p} - \mathbf{p}' \rangle = 0$, giving $\mathbf{p} = \mathbf{p}'$.

*Minimization.* For any $\mathbf{w} \in W$,
$$\|\mathbf{v} - \mathbf{w}\|^2 = \|(\mathbf{v} - \mathbf{p}) + (\mathbf{p} - \mathbf{w})\|^2 = \|\mathbf{v} - \mathbf{p}\|^2 + \|\mathbf{p} - \mathbf{w}\|^2,$$
by Pythagoras ($\mathbf{v} - \mathbf{p} \perp W$ and $\mathbf{p} - \mathbf{w} \in W$). The RHS is minimized over $\mathbf{w} \in W$ exactly when $\|\mathbf{p} - \mathbf{w}\| = 0$, i.e., $\mathbf{w} = \mathbf{p}$. $\square$

**Intuition.** $\mathbf{p}$ is the "shadow" of $\mathbf{v}$ onto $W$. It is the closest point in $W$ to $\mathbf{v}$. This is the geometric heart of:
- Ordinary least squares: $\hat\beta = \arg\min \|\mathbf{y} - X\beta\|^2$ finds the projection of $\mathbf{y}$ onto the column space of $X$.
- Conditional expectation: $\mathbb{E}[Y | \mathcal{G}]$ is the orthogonal projection of $Y$ onto the subspace of $\mathcal{G}$-measurable random variables in $L^2$ (Subject 2).

### Theorem 0.2.3.9 (Orthogonal decomposition)

If $W \subseteq V$ is a finite-dimensional subspace, then $V = W \oplus W^\perp$.

**Proof.** Every $\mathbf{v} \in V$ decomposes as $\mathbf{v} = \mathbf{p} + (\mathbf{v} - \mathbf{p})$ with $\mathbf{p} \in W$ and $\mathbf{v} - \mathbf{p} \in W^\perp$ (Theorem 0.2.3.8). Uniqueness: if $\mathbf{v} = \mathbf{p}_1 + \mathbf{q}_1 = \mathbf{p}_2 + \mathbf{q}_2$, then $\mathbf{p}_1 - \mathbf{p}_2 = \mathbf{q}_2 - \mathbf{q}_1 \in W \cap W^\perp = \{\mathbf{0}\}$ (since any vector orthogonal to itself has norm 0). $\square$

### Theorem 0.2.3.10 (Gram–Schmidt orthogonalization)

Given linearly independent $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$, construct an orthonormal set $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ with $\mathrm{span}\{\mathbf{e}_1, \ldots, \mathbf{e}_k\} = \mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ for each $k$.

**Construction.**
$$\mathbf{u}_1 = \mathbf{v}_1, \quad \mathbf{e}_1 = \mathbf{u}_1 / \|\mathbf{u}_1\|.$$
For $k = 2, \ldots, n$:
$$\mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \langle \mathbf{v}_k, \mathbf{e}_j \rangle \mathbf{e}_j, \quad \mathbf{e}_k = \mathbf{u}_k / \|\mathbf{u}_k\|.$$

**Proof of correctness.**

*$\mathbf{u}_k \neq \mathbf{0}$ for each $k$.* If $\mathbf{u}_k = 0$, then $\mathbf{v}_k \in \mathrm{span}\{\mathbf{e}_1, \ldots, \mathbf{e}_{k-1}\} = \mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}\}$ (by induction on $k$), contradicting linear independence of $\{\mathbf{v}_i\}$.

*$\{\mathbf{e}_j\}_{j \leq k}$ is orthonormal.* Induct on $k$. Base: $k = 1$ trivial. Step: assume $\{\mathbf{e}_1, \ldots, \mathbf{e}_{k-1}\}$ orthonormal. For $j < k$:
$$\langle \mathbf{u}_k, \mathbf{e}_j \rangle = \langle \mathbf{v}_k, \mathbf{e}_j \rangle - \sum_{l=1}^{k-1} \langle \mathbf{v}_k, \mathbf{e}_l \rangle \langle \mathbf{e}_l, \mathbf{e}_j \rangle = \langle \mathbf{v}_k, \mathbf{e}_j \rangle - \langle \mathbf{v}_k, \mathbf{e}_j \rangle = 0.$$
Normalization gives $\|\mathbf{e}_k\| = 1$ and $\langle \mathbf{e}_k, \mathbf{e}_j \rangle = 0$ for $j < k$.

*Span property.* $\mathbf{e}_k = \mathbf{u}_k / \|\mathbf{u}_k\|$, and $\mathbf{u}_k$ is $\mathbf{v}_k$ minus a linear combination of $\mathbf{e}_1, \ldots, \mathbf{e}_{k-1}$, which in turn are in $\mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}\}$. So $\mathbf{e}_k \in \mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$. Conversely, $\mathbf{v}_k = \mathbf{u}_k + (\text{linear combo of } \mathbf{e}_1, \ldots, \mathbf{e}_{k-1}) = \|\mathbf{u}_k\| \mathbf{e}_k + (\text{linear combo of earlier } \mathbf{e}_j)$. So $\mathbf{v}_k \in \mathrm{span}\{\mathbf{e}_1, \ldots, \mathbf{e}_k\}$. The two spans match.

$\square$

**Numerical warning.** Classical Gram–Schmidt is numerically unstable — the orthogonality of the $\mathbf{e}_j$'s degrades due to cancellation. In practice, use *modified Gram–Schmidt* (orthogonalize against each $\mathbf{e}_j$ as you go, not against the original $\mathbf{v}_k$) or better, QR decomposition via Householder reflections. Any serious numerical library (numpy's `qr`) uses Householder.

## Worked Examples

### Example 0.2.3.11 (Direct: Gram–Schmidt on $\mathbb{R}^3$)

*Apply Gram–Schmidt to $\mathbf{v}_1 = (1, 1, 0), \mathbf{v}_2 = (1, 0, 1), \mathbf{v}_3 = (0, 1, 1)$.*

**Solution.**

$\mathbf{u}_1 = (1, 1, 0), \|\mathbf{u}_1\| = \sqrt{2}, \mathbf{e}_1 = (1, 1, 0)/\sqrt{2}$.

$\langle \mathbf{v}_2, \mathbf{e}_1 \rangle = (1 + 0 + 0)/\sqrt{2} = 1/\sqrt{2}$.

$\mathbf{u}_2 = (1, 0, 1) - (1/\sqrt{2}) \cdot (1, 1, 0)/\sqrt{2} = (1, 0, 1) - (1/2, 1/2, 0) = (1/2, -1/2, 1)$.

$\|\mathbf{u}_2\| = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2}$. $\mathbf{e}_2 = (1/2, -1/2, 1)/\sqrt{3/2} = (1, -1, 2)/\sqrt{6}$.

$\langle \mathbf{v}_3, \mathbf{e}_1 \rangle = (0 + 1 + 0)/\sqrt{2} = 1/\sqrt{2}$. $\langle \mathbf{v}_3, \mathbf{e}_2 \rangle = (0 - 1 + 2)/\sqrt{6} = 1/\sqrt{6}$.

$\mathbf{u}_3 = (0, 1, 1) - (1/\sqrt{2}) \cdot (1, 1, 0)/\sqrt{2} - (1/\sqrt{6}) \cdot (1, -1, 2)/\sqrt{6}$
$= (0, 1, 1) - (1/2, 1/2, 0) - (1/6, -1/6, 2/6)$
$= (0 - 1/2 - 1/6, 1 - 1/2 + 1/6, 1 - 0 - 1/3)$
$= (-2/3, 2/3, 2/3)$.

$\|\mathbf{u}_3\| = \sqrt{4/9 + 4/9 + 4/9} = 2/\sqrt{3}$. $\mathbf{e}_3 = (-1, 1, 1)/\sqrt{3}$.

Verify: $\langle \mathbf{e}_1, \mathbf{e}_2 \rangle = (1 - 1 + 0)/\sqrt{12} = 0$. $\langle \mathbf{e}_1, \mathbf{e}_3 \rangle = (-1 + 1 + 0)/\sqrt{6} = 0$. $\langle \mathbf{e}_2, \mathbf{e}_3 \rangle = (-1 - 1 + 2)/\sqrt{18} = 0$. All orthogonal. $\square$

### Example 0.2.3.12 (Subtle: correlation as cosine)

*In $L^2(\Omega, P)$, let $X, Y$ be mean-zero random variables. Verify that the correlation $\rho = \mathbb{E}[XY]/(\sigma_X \sigma_Y)$ is the cosine of the angle between $X$ and $Y$ viewed as vectors.*

**Solution.** The inner product on $L^2$ is $\langle X, Y \rangle = \mathbb{E}[XY]$. The induced norm is $\|X\| = \sqrt{\mathbb{E}[X^2]} = \sigma_X$ (since $\mathbb{E}[X] = 0$). Hence
$$\cos \theta := \frac{\langle X, Y \rangle}{\|X\| \|Y\|} = \frac{\mathbb{E}[XY]}{\sigma_X \sigma_Y} = \rho.$$
Cauchy–Schwarz: $|\rho| \leq 1$, with equality iff $X, Y$ are linearly dependent (i.e., $Y = aX$ for some scalar $a$ almost surely).

**Lesson.** Correlation IS a cosine. Two random variables are "perpendicular" iff their correlation is zero iff their covariance is zero. Orthogonality in the $L^2$ sense is uncorrelatedness — not independence, unless additional structure (Gaussian) is imposed. This geometric picture carries over wholesale to regression, conditional expectation, and factor decomposition.

## Computational Implementation

```python
import numpy as np

def gram_schmidt(V):
    """Classical Gram-Schmidt. V has vectors as columns. Returns Q with orthonormal columns."""
    n = V.shape[1]
    Q = np.zeros_like(V, dtype=float)
    for k in range(n):
        u = V[:, k].astype(float)
        for j in range(k):
            u -= np.dot(V[:, k], Q[:, j]) * Q[:, j]
        Q[:, k] = u / np.linalg.norm(u)
    return Q

V = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float).T
Q = gram_schmidt(V)
print("Q:\n", Q)
print("Q^T Q (should be ~I):\n", Q.T @ Q)

# Projection onto a subspace
def project(v, W_basis):
    """Project v onto the span of columns of W_basis (assumed orthonormal)."""
    coeffs = W_basis.T @ v
    return W_basis @ coeffs

v = np.array([1.0, 2.0, 3.0])
W = Q[:, :2]  # span of first two orthonormal vectors
p = project(v, W)
print("Projection p:", p)
print("Residual v - p (should be perp to W):", v - p)
print("(v - p) . W (should be zero):", W.T @ (v - p))
```

## [QUANT APPLICATION]

**Regression as projection.** Given $\mathbf{y} \in \mathbb{R}^T$ and design matrix $X \in \mathbb{R}^{T \times K}$, OLS solves $\hat\beta = \arg\min_\beta \|\mathbf{y} - X\beta\|^2$. Geometrically, $X\hat\beta$ is the orthogonal projection of $\mathbf{y}$ onto the column space of $X$. The residual $\mathbf{y} - X\hat\beta$ is orthogonal to every column of $X$ — this is the *normal equation* $X^\top(\mathbf{y} - X\hat\beta) = 0$, i.e., $X^\top \mathbf{y} = X^\top X \hat\beta$. Solving: $\hat\beta = (X^\top X)^{-1} X^\top \mathbf{y}$.

This is not a formula to memorize. It is the explicit form of "project $\mathbf{y}$ onto $\mathrm{col}(X)$": Theorem 0.2.3.8 gives $\hat\beta$ once you choose an orthonormal basis of $\mathrm{col}(X)$ (which is why QR decomposition is the standard way to solve least-squares — it Gram–Schmidts the columns of $X$ and projects explicitly).

Same structure, in $L^2$: the conditional expectation $\mathbb{E}[Y | \mathcal{G}]$ is the projection of $Y$ onto the subspace of $\mathcal{G}$-measurable random variables in $L^2$. This is why $\mathbb{E}[Y | \mathcal{G}]$ is the $L^2$-best predictor of $Y$ given the information $\mathcal{G}$. Subject 2 will formalize this. Get comfortable with projection now and the measure-theoretic conditional expectation will feel like a familiar friend in a new context.

## Exercises

### ★ (Foundation)

**E0.2.3.1.** Verify that $\langle f, g \rangle = \int_0^1 f(x) g(x) \, dx$ is an inner product on $C([0, 1])$.

*Solution.* Symmetry (trivial, $fg = gf$). Linearity (linearity of the integral). Positive-definiteness: $\int_0^1 f(x)^2 dx \geq 0$; equals zero iff $f \equiv 0$ on $[0, 1]$ (since $f$ is continuous, $f^2 \geq 0$ has zero integral iff $f \equiv 0$). $\square$

**E0.2.3.2.** Compute $\langle \mathbf{x}, \mathbf{y} \rangle, \|\mathbf{x}\|, \|\mathbf{y}\|$ for $\mathbf{x} = (1, 2, 3), \mathbf{y} = (2, 0, -1)$. Verify Cauchy–Schwarz.

*Solution.* $\langle \mathbf{x}, \mathbf{y} \rangle = 2 + 0 - 3 = -1$. $\|\mathbf{x}\| = \sqrt{14}$. $\|\mathbf{y}\| = \sqrt{5}$. $|\langle \mathbf{x}, \mathbf{y} \rangle| = 1$. $\|\mathbf{x}\| \|\mathbf{y}\| = \sqrt{70} \approx 8.37$. $1 \leq 8.37$. ✓

**E0.2.3.3.** Show that in $\mathbb{R}^n$, the vectors $\mathbf{e}_i = (0, \ldots, 1, \ldots, 0)$ (1 in the $i$-th position) form an orthonormal basis with respect to the standard inner product.

*Solution.* $\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \sum_k (\mathbf{e}_i)_k (\mathbf{e}_j)_k = \delta_{ij}$. Orthonormal. Span: every $\mathbf{x} = \sum_i x_i \mathbf{e}_i$.

### ★★ (Intermediate)

**E0.2.3.4.** Show that in any inner product space, $\|\mathbf{u} + \mathbf{v}\|^2 + \|\mathbf{u} - \mathbf{v}\|^2 = 2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$ (the *parallelogram law*).

*Solution.* Expand: $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\langle \mathbf{u}, \mathbf{v} \rangle + \|\mathbf{v}\|^2$. $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 - 2\langle \mathbf{u}, \mathbf{v} \rangle + \|\mathbf{v}\|^2$. Sum: $2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$. $\square$

**Remark.** The parallelogram law is a *characterizing* property of norms that come from inner products. In Subject 9 we will meet norms that don't satisfy it (e.g., $\ell^1$); those are Banach spaces but not Hilbert spaces.

**E0.2.3.5.** Show $W^\perp \cap W = \{\mathbf{0}\}$ and deduce that the orthogonal decomposition $V = W \oplus W^\perp$ is a direct sum.

*Solution.* If $\mathbf{v} \in W \cap W^\perp$, then $\mathbf{v}$ is orthogonal to itself, so $\|\mathbf{v}\|^2 = \langle \mathbf{v}, \mathbf{v} \rangle = 0$, giving $\mathbf{v} = \mathbf{0}$. Combined with existence of the decomposition (Theorem 0.2.3.9), we have a direct sum.

**E0.2.3.6.** Prove Bessel's inequality: for orthonormal $\mathbf{e}_1, \ldots, \mathbf{e}_n$ in $V$ and any $\mathbf{v} \in V$,
$$\sum_{i=1}^n |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2 \leq \|\mathbf{v}\|^2,$$
with equality iff $\mathbf{v} \in \mathrm{span}\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$.

*Solution.* Let $\mathbf{p} = \sum_i \langle \mathbf{v}, \mathbf{e}_i \rangle \mathbf{e}_i$ (the projection onto the span). By the orthogonal decomposition (Theorem 0.2.3.9), $\mathbf{v} = \mathbf{p} + (\mathbf{v} - \mathbf{p})$ with $(\mathbf{v} - \mathbf{p}) \perp \mathrm{span}\{\mathbf{e}_i\}$. By Pythagoras, $\|\mathbf{v}\|^2 = \|\mathbf{p}\|^2 + \|\mathbf{v} - \mathbf{p}\|^2 \geq \|\mathbf{p}\|^2$. And $\|\mathbf{p}\|^2 = \sum_i |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2$ (orthonormality). Equality iff $\mathbf{v} - \mathbf{p} = 0$, iff $\mathbf{v} \in \mathrm{span}\{\mathbf{e}_i\}$. $\square$

### ★★★ (Challenge)

**E0.2.3.7.** Prove: in a finite-dimensional inner product space, every subspace $W$ is closed, and $(W^\perp)^\perp = W$.

*Hint.* Closedness is automatic in finite dimensions (all subspaces are closed). For $(W^\perp)^\perp = W$: clearly $W \subseteq (W^\perp)^\perp$. Use the dimension formula and $\dim V = \dim W + \dim W^\perp$.

**E0.2.3.8.** Let $V$ be a real finite-dimensional inner product space. A linear map $T: V \to V$ is *self-adjoint* if $\langle T\mathbf{v}, \mathbf{w} \rangle = \langle \mathbf{v}, T\mathbf{w} \rangle$ for all $\mathbf{v}, \mathbf{w}$. Show that if $\{\mathbf{e}_i\}$ is an orthonormal basis of $V$ and $A$ is the matrix of $T$ in this basis, then $T$ is self-adjoint iff $A$ is symmetric. (You will use this a lot in Topic 0.2.5.)

*Hint.* $\langle T\mathbf{e}_i, \mathbf{e}_j \rangle = A_{ji}$ and $\langle \mathbf{e}_i, T\mathbf{e}_j \rangle = A_{ij}$.

**E0.2.3.9.** Let $\{\mathbf{e}_i\}_{i \geq 1}$ be an orthonormal sequence in a Hilbert space $H$. Prove that if $\sum_i |\alpha_i|^2 < \infty$ then $\sum_i \alpha_i \mathbf{e}_i$ converges in $H$ (Riesz–Fischer). Deduce Parseval's identity $\|\sum_i \alpha_i \mathbf{e}_i\|^2 = \sum_i |\alpha_i|^2$.

*Hint.* Show the partial sums form a Cauchy sequence, then use completeness of $H$. This is preparing ground for Subject 1's completeness of $L^2$.

---

## Topic 0.2.4 — Eigendecomposition

### Motivation

A linear map $T: V \to V$ may look complicated in one basis and embarrassingly simple in another. An **eigenvector** is a direction that $T$ preserves: $T\mathbf{v} = \lambda \mathbf{v}$ for some scalar $\lambda$. Along that direction, $T$ just multiplies by $\lambda$. If we can find enough eigenvectors to form a basis, $T$ becomes diagonal — a pile of independent 1-dimensional multiplications. That transforms questions about iterated dynamics $(T^k \mathbf{x})$, matrix exponentials $(e^{tA})$, long-run covariance structures, spectral gaps of Markov chains, and principal component analysis into questions about plain scalars.

This topic is the workhorse of the rest of the course. PCA (factor models) is diagonalization of the covariance matrix. Stationary distributions of Markov chains come from the left eigenvector with eigenvalue $1$. Stability of linear recursions $x_{k+1} = Ax_k$ is spectral radius $\rho(A) < 1$. Continuous-time linear ODEs $\dot{\mathbf{x}} = A\mathbf{x}$ are solved by $e^{tA}$, which diagonalization computes in closed form. Ornstein–Uhlenbeck processes (Subject 3), Kalman filters, and the VAR models in time series all live and die by eigenvalues.

### Prerequisites

Linear transformations and matrix representation (Topic 0.2.2), polynomials over a field, determinants (we will briefly axiomatize them below — a quick review suffices), complex numbers.

### Determinants (Quick Review)

We take the determinant as a known object but let us state what we need. The determinant is a function $\det: M_n(\mathbb{F}) \to \mathbb{F}$ characterized by three properties:

1. **Multilinearity in columns (and rows):** $\det$ is linear in each column when the others are fixed.
2. **Alternating:** Swapping two columns changes the sign of $\det$.
3. **Normalization:** $\det(I_n) = 1$.

These three properties uniquely determine $\det$ and yield the Leibniz formula
$$\det(A) = \sum_{\sigma \in S_n} \mathrm{sgn}(\sigma) \prod_{i=1}^n A_{i, \sigma(i)}.$$

Key consequences we will use without re-deriving:

- $\det(AB) = \det(A)\det(B)$;
- $\det(A^\top) = \det(A)$;
- $A$ is invertible iff $\det(A) \neq 0$;
- If $A$ is block triangular, $\det(A) = \prod_i \det(A_{ii})$ (determinants of diagonal blocks);
- $\det(A)$ is a polynomial of degree $n$ in the entries.

(For a rigorous development starting from the three axioms, see Axler's *Linear Algebra Done Right* or Hoffman–Kunze; we will rebuild determinants from scratch in Topic 0.2.7 via permanents when we discuss positive definite matrices.)

### Definitions

**Definition 0.2.4.1 (Eigenvalue, eigenvector).** Let $V$ be a vector space over $\mathbb{F}$ and $T: V \to V$ linear. A scalar $\lambda \in \mathbb{F}$ is an **eigenvalue** of $T$ if there exists a nonzero $\mathbf{v} \in V$ with
$$T\mathbf{v} = \lambda \mathbf{v}.$$
The vector $\mathbf{v}$ is called an **eigenvector** corresponding to $\lambda$. The **eigenspace**
$$E_\lambda(T) \;=\; \ker(T - \lambda I) \;=\; \{\mathbf{v} \in V : T\mathbf{v} = \lambda \mathbf{v}\}$$
is a subspace (including $\mathbf{0}$). Its dimension is the **geometric multiplicity** of $\lambda$, denoted $g(\lambda)$.

For a matrix $A \in M_n(\mathbb{F})$, we apply the same definitions to the linear map $\mathbf{x} \mapsto A\mathbf{x}$.

**Remark.** The zero vector is excluded as an eigenvector because otherwise every $\lambda$ would be an "eigenvalue" trivially. But $\mathbf{0} \in E_\lambda$ — eigenspaces are subspaces.

**Definition 0.2.4.2 (Characteristic polynomial).** For $A \in M_n(\mathbb{F})$, the **characteristic polynomial** is
$$p_A(t) \;=\; \det(tI_n - A) \;\in\; \mathbb{F}[t].$$

**Proposition 0.2.4.3 (Eigenvalues are roots of $p_A$).** $\lambda \in \mathbb{F}$ is an eigenvalue of $A$ iff $p_A(\lambda) = 0$.

*Proof.* $\lambda$ is an eigenvalue iff there is $\mathbf{v} \neq \mathbf{0}$ with $(A - \lambda I)\mathbf{v} = \mathbf{0}$, iff $\ker(A - \lambda I) \neq \{\mathbf{0}\}$, iff $A - \lambda I$ is not injective, iff (in finite dimension) $A - \lambda I$ is not invertible, iff $\det(A - \lambda I) = 0$, iff $\det(\lambda I - A) = 0$ (since $\det(-X) = (-1)^n \det(X)$), iff $p_A(\lambda) = 0$. $\square$

**Definition 0.2.4.4 (Algebraic multiplicity).** If $\lambda$ is a root of $p_A$, its **algebraic multiplicity** $a(\lambda)$ is the largest integer $k$ with $(t - \lambda)^k \mid p_A(t)$.

**Definition 0.2.4.5 (Spectrum).** The **spectrum** of $A$ is $\sigma(A) = \{\lambda \in \mathbb{F} : \lambda \text{ is an eigenvalue of } A\}$. The **spectral radius** is $\rho(A) = \max_{\lambda \in \sigma(A)} |\lambda|$ (over $\mathbb{C}$ typically).

### Key Results

**Theorem 0.2.4.6 (Existence of eigenvalues over $\mathbb{C}$).** Every matrix $A \in M_n(\mathbb{C})$ with $n \geq 1$ has at least one eigenvalue in $\mathbb{C}$.

*Proof.* $p_A(t)$ is a polynomial of degree exactly $n \geq 1$ in $t$ (the leading term is $t^n$ from the product $\prod_i (t - A_{ii})$ in the Leibniz formula; all other terms have lower degree). By the Fundamental Theorem of Algebra, $p_A$ has a root in $\mathbb{C}$. $\square$

**Remark (real matrices).** Over $\mathbb{R}$, eigenvalues may fail to exist: the rotation $R_{\pi/2} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ has $p_R(t) = t^2 + 1$, with no real roots. Its complex eigenvalues are $\pm i$.

**Theorem 0.2.4.7 (Eigenvectors for distinct eigenvalues are independent).** Let $T: V \to V$ be linear and let $\mathbf{v}_1, \ldots, \mathbf{v}_k$ be eigenvectors of $T$ corresponding to *distinct* eigenvalues $\lambda_1, \ldots, \lambda_k$. Then $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is linearly independent.

*Proof.* Induction on $k$. Base $k = 1$: a single nonzero vector is independent.

Inductive step: assume the result for $k - 1$. Suppose
$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k = \mathbf{0}. \tag{$\ast$}$$

Apply $T$:
$$c_1 \lambda_1 \mathbf{v}_1 + c_2 \lambda_2 \mathbf{v}_2 + \cdots + c_k \lambda_k \mathbf{v}_k = \mathbf{0}. \tag{$\ast\ast$}$$

Multiply $(\ast)$ by $\lambda_k$ and subtract from $(\ast\ast)$:
$$c_1 (\lambda_1 - \lambda_k) \mathbf{v}_1 + c_2 (\lambda_2 - \lambda_k) \mathbf{v}_2 + \cdots + c_{k-1} (\lambda_{k-1} - \lambda_k) \mathbf{v}_{k-1} = \mathbf{0}.$$

By the inductive hypothesis, $\mathbf{v}_1, \ldots, \mathbf{v}_{k-1}$ are independent, so $c_i (\lambda_i - \lambda_k) = 0$ for $i = 1, \ldots, k-1$. Since $\lambda_i \neq \lambda_k$ for $i < k$ (distinct eigenvalues), we get $c_i = 0$ for $i = 1, \ldots, k-1$. Substituting back into $(\ast)$: $c_k \mathbf{v}_k = \mathbf{0}$, and since $\mathbf{v}_k \neq \mathbf{0}$, $c_k = 0$. All coefficients vanish. $\square$

**Corollary 0.2.4.8.** If $A \in M_n(\mathbb{F})$ has $n$ distinct eigenvalues, it has a basis of eigenvectors.

*Proof.* Pick one eigenvector per eigenvalue; by Theorem 0.2.4.7 they are independent, and there are $n$ of them in an $n$-dimensional space, so they form a basis. $\square$

**Theorem 0.2.4.9 (Geometric $\leq$ Algebraic multiplicity).** For every eigenvalue $\lambda$ of $A \in M_n(\mathbb{F})$,
$$1 \leq g(\lambda) \leq a(\lambda).$$

*Proof.* The lower bound $g(\lambda) \geq 1$ holds because $\lambda$ being an eigenvalue means $E_\lambda \neq \{\mathbf{0}\}$.

For $g(\lambda) \leq a(\lambda)$: let $g = g(\lambda)$ and choose a basis $\mathbf{v}_1, \ldots, \mathbf{v}_g$ of $E_\lambda$. Extend to a basis $\mathbf{v}_1, \ldots, \mathbf{v}_g, \mathbf{w}_1, \ldots, \mathbf{w}_{n-g}$ of $\mathbb{F}^n$. Let $P$ be the change-of-basis matrix with these columns. Then
$$P^{-1} A P = \begin{pmatrix} \lambda I_g & B \\ 0 & C \end{pmatrix}$$
for some blocks $B$ (size $g \times (n-g)$) and $C$ (size $(n-g) \times (n-g)$). The upper-left block is $\lambda I_g$ because $A\mathbf{v}_i = \lambda \mathbf{v}_i$ for $i = 1, \ldots, g$, so the first $g$ columns of $AP$ are $\lambda \mathbf{v}_i$, and expressing these in the new basis gives coefficients $\lambda$ on $\mathbf{v}_i$ and $0$ elsewhere.

Now $p_A(t) = \det(tI - A) = \det(P^{-1}(tI - A)P) = \det(tI - P^{-1}AP)$ (since $\det(P)\det(P^{-1}) = 1$). So
$$p_A(t) = \det \begin{pmatrix} (t - \lambda) I_g & -B \\ 0 & tI_{n-g} - C \end{pmatrix} = (t - \lambda)^g \det(tI_{n-g} - C)$$
by block-triangular determinant. Therefore $(t - \lambda)^g \mid p_A(t)$, giving $a(\lambda) \geq g = g(\lambda)$. $\square$

**Example (strict inequality).** $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ has $p_A(t) = (t - 1)^2$, so $\lambda = 1$ has $a(1) = 2$. But $A - I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ has kernel spanned by $\mathbf{e}_1$ alone; so $g(1) = 1 < 2$. $A$ is **defective**.

### Diagonalization

**Definition 0.2.4.10.** A matrix $A \in M_n(\mathbb{F})$ is **diagonalizable** over $\mathbb{F}$ if there exists an invertible $P \in M_n(\mathbb{F})$ and a diagonal $D \in M_n(\mathbb{F})$ with $A = P D P^{-1}$.

Equivalently, a linear map $T: V \to V$ is diagonalizable if $V$ has a basis consisting of eigenvectors of $T$.

**Theorem 0.2.4.11 (Diagonalizability criterion).** Let $A \in M_n(\mathbb{F})$. The following are equivalent:

(i) $A$ is diagonalizable over $\mathbb{F}$.

(ii) $\mathbb{F}^n$ has a basis of eigenvectors of $A$.

(iii) $\sum_\lambda g(\lambda) = n$, where the sum is over eigenvalues of $A$ in $\mathbb{F}$.

(iv) The characteristic polynomial $p_A$ splits over $\mathbb{F}$ (factors into linear factors) *and* $g(\lambda) = a(\lambda)$ for every eigenvalue $\lambda$.

*Proof.* (i) $\Leftrightarrow$ (ii): If $A = PDP^{-1}$, write $P = [\mathbf{p}_1 | \cdots | \mathbf{p}_n]$ (columns). Then $AP = PD$ means $A\mathbf{p}_i = D_{ii} \mathbf{p}_i$, so each $\mathbf{p}_i$ is an eigenvector with eigenvalue $D_{ii}$, and they are independent (columns of an invertible matrix). Conversely, if $\mathbf{p}_1, \ldots, \mathbf{p}_n$ are independent eigenvectors with $A\mathbf{p}_i = \lambda_i \mathbf{p}_i$, let $P$ have these as columns and $D = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$; then $AP = PD$, so $A = PDP^{-1}$.

(ii) $\Leftrightarrow$ (iii): Eigenvectors from *different* eigenspaces are independent (Theorem 0.2.4.7 extended: eigenspaces for distinct eigenvalues are independent in the sense that their sum is a direct sum — see Lemma below). Inside each eigenspace we can pick a basis of size $g(\lambda)$. The union is independent and has size $\sum_\lambda g(\lambda)$. This equals $n$ iff they form a basis of $\mathbb{F}^n$.

**Lemma.** Eigenspaces for distinct eigenvalues are independent: if $\mathbf{u}_1 + \cdots + \mathbf{u}_k = \mathbf{0}$ with $\mathbf{u}_i \in E_{\lambda_i}$ and the $\lambda_i$ distinct, then each $\mathbf{u}_i = \mathbf{0}$.

*Proof of Lemma.* If not, the nonzero $\mathbf{u}_i$ would be eigenvectors for distinct eigenvalues summing to $\mathbf{0}$, contradicting Theorem 0.2.4.7. $\square$

(iii) $\Leftrightarrow$ (iv): We have $\sum_\lambda a(\lambda) \leq n$ always (sum of degrees of linear factors in the factorization of $p_A$ over $\mathbb{F}$ is at most $\deg p_A = n$), with equality iff $p_A$ splits over $\mathbb{F}$. And $g(\lambda) \leq a(\lambda)$. So
$$\sum_\lambda g(\lambda) \leq \sum_\lambda a(\lambda) \leq n,$$
with both equalities iff $p_A$ splits *and* $g(\lambda) = a(\lambda)$ for every $\lambda$. $\square$

**Computational implication.** If $A = PDP^{-1}$, then for any $k$,
$$A^k = PD^kP^{-1}, \qquad e^{A} = P e^{D} P^{-1},$$
and $e^D = \mathrm{diag}(e^{\lambda_1}, \ldots, e^{\lambda_n})$ is trivial to compute. Diagonalization turns matrix power series into scalar power series on each eigenvalue.

### When Diagonalization Fails: Jordan Form (Overview)

When $A$ is not diagonalizable, we have defective eigenvalues ($g(\lambda) < a(\lambda)$). The correct replacement is the **Jordan canonical form** (over an algebraically closed field, e.g., $\mathbb{C}$):

$$J = \begin{pmatrix} J_{k_1}(\lambda_1) & & & \\ & J_{k_2}(\lambda_2) & & \\ & & \ddots & \\ & & & J_{k_r}(\lambda_r) \end{pmatrix}, \qquad J_k(\lambda) = \begin{pmatrix} \lambda & 1 & & \\ & \lambda & 1 & \\ & & \ddots & 1 \\ & & & \lambda \end{pmatrix} \in M_k.$$

Every $A \in M_n(\mathbb{C})$ is similar to a unique (up to ordering of blocks) Jordan matrix $J$: $A = P J P^{-1}$. Inside a Jordan block $J_k(\lambda)$, we have generalized eigenvectors: $\mathbf{v}_1$ satisfies $(A - \lambda I)\mathbf{v}_1 = \mathbf{0}$, $\mathbf{v}_2$ satisfies $(A - \lambda I)\mathbf{v}_2 = \mathbf{v}_1$, etc. — a chain of length $k$. The total size of Jordan blocks with eigenvalue $\lambda$ equals $a(\lambda)$; the number of such blocks equals $g(\lambda)$.

We will not prove Jordan form here (the clean proof uses modules over a PID or the primary decomposition theorem). For quant applications, two things matter:

1. **$A^k$ for defective $A$:** $(J_k(\lambda))^m$ has entries involving $\binom{m}{j} \lambda^{m-j}$ — polynomial-in-$m$ times $\lambda^m$. So trajectories of iterated dynamics can have polynomial growth along a direction of eigenvalue on the unit circle. This is why stability analysis requires checking that *all* eigenvalues with $|\lambda| = 1$ have $g(\lambda) = a(\lambda)$ (semisimple) — otherwise you get polynomial blow-up.

2. **Perturbation:** Non-diagonalizable matrices are a measure-zero set. Any $A$ can be perturbed to a diagonalizable one. So generic matrices are diagonalizable. But in structured problems (e.g., systems with repeated eigenvalues by symmetry), defectiveness arises naturally.

**Theorem 0.2.4.12 (Cayley–Hamilton).** For any $A \in M_n(\mathbb{F})$, $p_A(A) = 0$, i.e., $A$ satisfies its own characteristic polynomial.

*Proof sketch (complex case, via density).* Diagonalizable matrices are dense in $M_n(\mathbb{C})$. For diagonalizable $A = PDP^{-1}$ with diagonal entries $\lambda_i$:
$$p_A(A) = P\, p_A(D) \, P^{-1} = P \cdot \mathrm{diag}(p_A(\lambda_1), \ldots, p_A(\lambda_n)) \cdot P^{-1} = 0$$
since each $\lambda_i$ is a root of $p_A$. Now $p_A(A)$ depends polynomially (hence continuously) on $A$, and it vanishes on a dense set; so it vanishes everywhere. Over $\mathbb{R}$, extend scalars to $\mathbb{C}$ and restrict. $\square$

A more direct proof uses the adjugate matrix identity $(tI - A)\,\mathrm{adj}(tI - A) = p_A(t) I$ and comparing coefficients; see Hoffman–Kunze for details.

### Worked Examples

**Example 0.2.4.13 (Diagonalizing a $2\times 2$).** Let $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$.

*Step 1. Characteristic polynomial.* $p_A(t) = \det(tI - A) = (t-4)(t-3) - (-1)(-2) = t^2 - 7t + 12 - 2 = t^2 - 7t + 10 = (t-5)(t-2)$. Eigenvalues: $\lambda_1 = 5$, $\lambda_2 = 2$.

*Step 2. Eigenvectors.* For $\lambda_1 = 5$: $A - 5I = \begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix}$. Kernel: $-v_1 + v_2 = 0$, so $\mathbf{v}_1 = (1, 1)^\top$.

For $\lambda_2 = 2$: $A - 2I = \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix}$. Kernel: $2v_1 + v_2 = 0$, so $\mathbf{v}_2 = (1, -2)^\top$.

*Step 3. Diagonalization.* $P = \begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix}$, $D = \mathrm{diag}(5, 2)$. Then $\det(P) = -3$, $P^{-1} = -\frac{1}{3}\begin{pmatrix} -2 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 2 & 1 \\ 1 & -1 \end{pmatrix}$, and $A = PDP^{-1}$.

*Sanity.* Trace: $4 + 3 = 7 = 5 + 2$. ✓ Determinant: $12 - 2 = 10 = 5 \cdot 2$. ✓

**Example 0.2.4.14 (A defective matrix).** $A = \begin{pmatrix} 3 & 1 \\ 0 & 3 \end{pmatrix}$. $p_A(t) = (t-3)^2$, so $\lambda = 3$ with $a(3) = 2$. But $A - 3I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, kernel is $\mathrm{span}\{\mathbf{e}_1\}$, $g(3) = 1$. Not diagonalizable. The Jordan form is $A$ itself (it is already a Jordan block).

To compute $A^n$ we use the binomial expansion: $A = 3I + N$ where $N = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, $N^2 = 0$. Since $I$ and $N$ commute,
$$A^n = (3I + N)^n = \sum_{k=0}^n \binom{n}{k} 3^{n-k} N^k = 3^n I + n \cdot 3^{n-1} N = \begin{pmatrix} 3^n & n \cdot 3^{n-1} \\ 0 & 3^n \end{pmatrix}.$$

Notice the $n \cdot 3^{n-1}$ entry: polynomial-in-$n$ times $3^n$ — the hallmark of Jordan-block dynamics.

**Example 0.2.4.15 (Real matrix with complex eigenvalues).** $R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$, rotation by $\theta$. $p_R(t) = t^2 - 2\cos\theta \cdot t + 1$, eigenvalues $e^{\pm i\theta}$.

Not diagonalizable over $\mathbb{R}$ (unless $\theta \in \{0, \pi\}$). Over $\mathbb{C}$: eigenvectors are $\mathbf{v}_\pm = (1, \mp i)^\top / \sqrt 2$. So $R = U \mathrm{diag}(e^{i\theta}, e^{-i\theta}) U^{-1}$.

*Why this matters for quant.* Complex eigenvalues of a real dynamical system $x_{k+1} = Ax_k$ correspond to oscillatory modes with period $2\pi/\theta$ and amplitude growth $|\lambda| = 1$. Think AR(2) models with complex roots — they generate damped sinusoidal autocorrelations.

### Computational Implementation

```python
import numpy as np
from numpy.linalg import eig, eigvals, matrix_rank

# Example 0.2.4.13
A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, eigenvectors = eig(A)
print("Eigenvalues:", eigenvalues)           # [5. 2.]
print("Eigenvectors (columns):\n", eigenvectors)

# Verify A @ v = lambda * v
for i, lam in enumerate(eigenvalues):
    v = eigenvectors[:, i]
    print(f"A @ v - lambda * v = {A @ v - lam * v}")  # ~0

# Diagonalization check
D = np.diag(eigenvalues)
P = eigenvectors
assert np.allclose(A, P @ D @ np.linalg.inv(P))
print("Diagonalization verified.")

# Fast matrix power via diagonalization
def matrix_power_diag(A, k):
    """Compute A^k when A is diagonalizable."""
    lam, V = eig(A)
    return (V @ np.diag(lam**k) @ np.linalg.inv(V)).real

print("A^10 via eig  =\n", matrix_power_diag(A, 10))
print("A^10 direct   =\n", np.linalg.matrix_power(A, 10))

# Geometric vs algebraic multiplicity
def multiplicities(A, tol=1e-9):
    """Return {lambda: (alg_mult, geom_mult)} for numerical A."""
    eigs = eigvals(A)
    # Group near-equal eigenvalues
    unique_eigs = []
    for lam in eigs:
        placed = False
        for u in unique_eigs:
            if abs(lam - u) < tol:
                placed = True
                break
        if not placed:
            unique_eigs.append(lam)
    result = {}
    n = A.shape[0]
    for lam in unique_eigs:
        alg = sum(1 for e in eigs if abs(e - lam) < tol)
        geom = n - matrix_rank(A - lam * np.eye(n), tol=tol)
        result[lam] = (alg, geom)
    return result

A_def = np.array([[3., 1.],
                  [0., 3.]])
print("Defective matrix multiplicities:", multiplicities(A_def))
# {3.0: (2, 1)} — defective

A_diag = np.array([[4., 1.],
                   [2., 3.]])
print("Diagonalizable multiplicities:", multiplicities(A_diag))
# {5.0: (1, 1), 2.0: (1, 1)}

# Spectral radius and stability test
def is_stable(A, tol=1e-12):
    """Check x_{k+1} = A x_k is stable: rho(A) < 1."""
    return np.max(np.abs(eigvals(A))) < 1 - tol

print("Is A=0.5*I stable?", is_stable(0.5 * np.eye(3)))   # True
print("Is rotation stable?", is_stable(np.array([[0., -1.], [1., 0.]])))  # False (boundary)
```

**Numerical warning.** `numpy.linalg.eig` works in general but for non-symmetric matrices can be numerically sensitive when eigenvalues are nearly equal (ill-conditioned eigenvector problem). For symmetric/Hermitian matrices use `eigh` (stable, faster, real eigenvalues). For power methods and long-run dynamics, Schur decomposition (`scipy.linalg.schur`) is more numerically stable than direct eigendecomposition.

### [QUANT APPLICATION] — Markov Chain Stationary Distribution, Mean Reversion, Covariance Matrix Powers

**(A) Markov Chains.** A discrete-time Markov chain with transition matrix $P \in [0,1]^{n \times n}$ (rows sum to $1$, so $P\mathbf{1} = \mathbf{1}$, meaning $1$ is always an eigenvalue) has distribution at time $k$ given by
$$\pi_k^\top = \pi_0^\top P^k.$$
By Perron–Frobenius (proved in random matrix theory, Subject 8; accept here), if $P$ is irreducible and aperiodic, $\rho(P) = 1$ is a simple eigenvalue and all others have $|\lambda| < 1$. Diagonalize $P = V\Lambda V^{-1}$ (assume diagonalizable for cleanness); write the initial distribution $\pi_0^\top = \sum_i c_i \ell_i^\top$ in the basis of left eigenvectors. Then
$$\pi_k^\top = \sum_i c_i \lambda_i^k \ell_i^\top.$$
The $\lambda_1 = 1$ term survives: $\lim_{k\to\infty} \pi_k = \pi_\infty$, the stationary distribution. The *rate of convergence* is governed by the **spectral gap** $1 - |\lambda_2|$ where $\lambda_2$ is the second-largest eigenvalue:
$$\|\pi_k - \pi_\infty\|_1 \leq C\, |\lambda_2|^k.$$
In credit-rating transition modeling, $P$ encodes one-year rating transitions and $P^k$ gives $k$-year transitions. Eigendecomposition exposes the long-run composition.

**(B) Mean reversion and AR(1).** The AR(1) process $x_{t+1} = \phi x_t + \varepsilon_{t+1}$ is the scalar case. The VAR(1) generalization $\mathbf{x}_{t+1} = A \mathbf{x}_t + \varepsilon_{t+1}$ is stationary iff $\rho(A) < 1$. Half-life of mean reversion along the $\lambda_i$-eigenmode is $-\log 2 / \log |\lambda_i|$. Quant traders ask: is the mean-reversion speed of my pair-trading signal along the correct principal direction? The answer is literally eigendecomposition of $A$.

**(C) Covariance matrix powers in risk forecasting.** Long-horizon covariance under a GARCH or VAR model involves $A^k \Sigma (A^\top)^k$ terms. Via $A = V\Lambda V^{-1}$, these reduce to elementwise scaling of a transformed covariance — closed-form long-horizon variance instead of simulation.

**(D) PCA.** Covariance matrix $\Sigma$ is symmetric positive semidefinite (Topic 0.2.7). Its eigenvalues are nonnegative and its eigenvectors are orthogonal (Spectral Theorem, Topic 0.2.5). PCA = eigendecomposition of $\Sigma$; principal components = eigenvectors; explained variance = eigenvalues. A single factor model with $\lambda_1 \gg \lambda_2$ means one dominant eigenvalue absorbs most variance.

### Exercises

#### ★ (Foundation)

**E0.2.4.1.** Compute eigenvalues and eigenvectors of $A = \begin{pmatrix} 2 & 0 \\ 0 & 5 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $C = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$.

**E0.2.4.2.** Show that eigenvalues of an upper triangular matrix are its diagonal entries.

**E0.2.4.3.** Let $A$ be invertible with eigenvalue $\lambda$ and eigenvector $\mathbf{v}$. Show $\lambda \neq 0$ and $A^{-1}$ has eigenvalue $1/\lambda$ with the same eigenvector $\mathbf{v}$.

**E0.2.4.4.** Prove: $\mathrm{tr}(A) = \sum_\lambda a(\lambda) \lambda$ and $\det(A) = \prod_\lambda \lambda^{a(\lambda)}$ (eigenvalues counted with algebraic multiplicity).

*Hint.* Expand $p_A(t) = \prod_i (t - \lambda_i)$ and compare coefficients of $t^{n-1}$ (trace) and $t^0$ (determinant).

#### ★★ (Intermediate)

**E0.2.4.5.** Let $A, B \in M_n(\mathbb{F})$. Show $AB$ and $BA$ have the same nonzero eigenvalues with the same algebraic multiplicities (not necessarily the same $0$ multiplicities).

*Hint.* Show that if $\lambda \neq 0$ and $\mathbf{v}$ satisfies $AB\mathbf{v} = \lambda \mathbf{v}$, then $B\mathbf{v} \neq \mathbf{0}$ and $BA(B\mathbf{v}) = \lambda (B\mathbf{v})$. For multiplicities, use the identity $\det(\lambda I - AB) = \det(\lambda I - BA) \cdot \lambda^?$ or a matrix identity like $\begin{pmatrix} \lambda I & A \\ B & I \end{pmatrix}$ evaluated two ways.

**E0.2.4.6.** Let $A \in M_n(\mathbb{C})$ satisfy $A^k = I$ for some $k \geq 1$. Prove $A$ is diagonalizable and its eigenvalues are $k$-th roots of unity.

*Hint.* The minimal polynomial of $A$ divides $t^k - 1$, which has distinct roots, so $A$ is diagonalizable (minimal polynomial has no repeated roots $\Leftrightarrow$ diagonalizable).

**E0.2.4.7.** Let $A \in M_n(\mathbb{R})$ have all eigenvalues with $|\lambda_i| < 1$. Prove $\sum_{k=0}^\infty A^k = (I - A)^{-1}$ (series converges).

*Hint.* Diagonalize (or Jordanize). Note the geometric sum: $(I - A)(\sum_{k=0}^N A^k) = I - A^{N+1} \to I$.

**E0.2.4.8.** Let $P$ be an $n \times n$ stochastic matrix (rows nonnegative, sum to $1$). Prove $1$ is an eigenvalue and every eigenvalue $\lambda$ satisfies $|\lambda| \leq 1$.

*Hint.* For the first: $P\mathbf{1} = \mathbf{1}$. For the bound: if $P\mathbf{v} = \lambda \mathbf{v}$, pick the index $i$ maximizing $|v_i|$ and use triangle inequality on the $i$-th row.

#### ★★★ (Challenge)

**E0.2.4.9 (Simultaneous diagonalization).** Let $A, B \in M_n(\mathbb{C})$ be diagonalizable. Show that $A$ and $B$ are *simultaneously* diagonalizable ($\exists P$ with both $P^{-1}AP$ and $P^{-1}BP$ diagonal) iff $AB = BA$.

*Hint.* Forward: commuting diagonal matrices commute trivially. Backward: $B$ preserves each eigenspace of $A$ (if $A\mathbf{v} = \lambda \mathbf{v}$ and $BA = AB$, then $A(B\mathbf{v}) = BA\mathbf{v} = B(\lambda \mathbf{v}) = \lambda B\mathbf{v}$, so $B\mathbf{v} \in E_\lambda$). Restrict $B$ to each eigenspace of $A$; it is still diagonalizable there; diagonalize jointly.

**E0.2.4.10 (Variational characterization for symmetric — preview of Topic 0.2.5).** For a real symmetric $A \in M_n(\mathbb{R})$, prove
$$\lambda_{\max}(A) = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}.$$

*Hint.* Diagonalize $A = Q \Lambda Q^\top$ (spectral theorem — coming in Topic 0.2.5). Change variables $\mathbf{y} = Q^\top \mathbf{x}$. The quotient becomes $\sum \lambda_i y_i^2 / \sum y_i^2$, which is a convex combination of eigenvalues.

**E0.2.4.11 (Matrix exponential).** Define $e^A = \sum_{k=0}^\infty A^k / k!$ (we will prove convergence in Topic 0.2.9 via matrix norms). Show: if $A = PDP^{-1}$ with $D$ diagonal, then $e^A = Pe^DP^{-1}$ and $e^D = \mathrm{diag}(e^{\lambda_i})$. Deduce $\det(e^A) = e^{\mathrm{tr}(A)}$.

*Hint.* The first part follows from $A^k = PD^kP^{-1}$ and termwise summing. For the determinant identity: $\det(PDP^{-1}) = \det(D)$ and $\mathrm{tr}(PDP^{-1}) = \mathrm{tr}(D)$; then $\det(e^D) = \prod e^{\lambda_i} = e^{\sum \lambda_i}$.

---

## Topic 0.2.5 — The Spectral Theorem

### Motivation

For general matrices, eigendecomposition requires complex eigenvalues, may have non-orthogonal eigenvectors, and can fail (Jordan blocks). For a special but ubiquitous class — real symmetric matrices and their complex cousins, Hermitian matrices, and more generally **normal** matrices — eigendecomposition is maximally clean:

- All eigenvalues are real (for symmetric/Hermitian).
- Eigenvectors can be chosen orthonormal.
- Diagonalization is achieved by an orthogonal/unitary matrix.

This is the **Spectral Theorem**, and it is the mathematical foundation of PCA, the singular value decomposition (next topic), positive definite programming, optimization over quadratic forms, Mercer's theorem in kernel methods, and spectral methods in graph theory. The theorem is so clean that in applied work, *"symmetric"* is shorthand for *"I can diagonalize this beautifully and interpret the eigenvalues/eigenvectors geometrically."*

Crucially for quant: a **covariance matrix is always symmetric PSD**. Principal components literally *are* the orthonormal eigenvectors given by the spectral theorem.

### Prerequisites

Eigendecomposition (Topic 0.2.4), inner product spaces (Topic 0.2.3), complex conjugate and conjugate transpose $A^* = \overline{A^\top}$.

### Definitions

Throughout this topic, the ground field is $\mathbb{R}$ or $\mathbb{C}$, with $\langle \cdot, \cdot \rangle$ the standard inner product on $\mathbb{F}^n$: $\langle \mathbf{x}, \mathbf{y} \rangle = \sum_i x_i \overline{y_i}$ (complex) or $\sum_i x_i y_i$ (real).

**Definition 0.2.5.1 (Adjoint / conjugate transpose).** For $A \in M_{m \times n}(\mathbb{C})$, the **adjoint** is $A^* = \overline{A}^\top \in M_{n \times m}(\mathbb{C})$. For real matrices, $A^* = A^\top$.

**Fundamental property (adjoint as inner-product dual):** $\langle A\mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{x}, A^* \mathbf{y} \rangle$ for all $\mathbf{x} \in \mathbb{C}^n$, $\mathbf{y} \in \mathbb{C}^m$.

*Verification.* $\langle A\mathbf{x}, \mathbf{y} \rangle = \sum_i (A\mathbf{x})_i \overline{y_i} = \sum_{i,j} A_{ij} x_j \overline{y_i} = \sum_j x_j \overline{\sum_i \overline{A_{ij}} y_i} = \sum_j x_j \overline{(A^*\mathbf{y})_j} = \langle \mathbf{x}, A^*\mathbf{y}\rangle$. $\square$

**Definition 0.2.5.2 (Symmetric, Hermitian, orthogonal, unitary, normal).** A matrix $A \in M_n$:
- is **symmetric** (real case) if $A^\top = A$;
- is **Hermitian** (complex case) if $A^* = A$;
- is **orthogonal** (real) if $A^\top A = I$, equivalently $A^{-1} = A^\top$;
- is **unitary** (complex) if $A^* A = I$, equivalently $A^{-1} = A^*$;
- is **normal** if $A A^* = A^* A$.

Orthogonal/unitary matrices are **isometries**: $\|A\mathbf{x}\| = \|\mathbf{x}\|$ for all $\mathbf{x}$, since $\|A\mathbf{x}\|^2 = \langle A\mathbf{x}, A\mathbf{x}\rangle = \langle \mathbf{x}, A^*A\mathbf{x}\rangle = \langle \mathbf{x}, \mathbf{x}\rangle = \|\mathbf{x}\|^2$. Columns of a unitary are orthonormal (as are rows).

Every Hermitian matrix is normal. Every unitary matrix is normal. Both facts are immediate from the definitions.

### Key Lemmas

**Lemma 0.2.5.3 (Eigenvalues of Hermitian are real).** Let $A \in M_n(\mathbb{C})$ be Hermitian and let $\lambda \in \mathbb{C}$ be an eigenvalue. Then $\lambda \in \mathbb{R}$.

*Proof.* Let $A\mathbf{v} = \lambda \mathbf{v}$ with $\mathbf{v} \neq \mathbf{0}$. Then
$$\lambda \langle \mathbf{v}, \mathbf{v}\rangle = \langle \lambda \mathbf{v}, \mathbf{v}\rangle = \langle A\mathbf{v}, \mathbf{v}\rangle = \langle \mathbf{v}, A^*\mathbf{v}\rangle = \langle \mathbf{v}, A\mathbf{v}\rangle = \langle \mathbf{v}, \lambda \mathbf{v}\rangle = \overline{\lambda} \langle \mathbf{v}, \mathbf{v}\rangle.$$
Since $\langle \mathbf{v}, \mathbf{v}\rangle \neq 0$, $\lambda = \overline{\lambda}$, so $\lambda \in \mathbb{R}$. $\square$

**Lemma 0.2.5.4 (Eigenvectors of Hermitian for distinct eigenvalues are orthogonal).** If $A$ is Hermitian with $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$ and $\lambda_1 \neq \lambda_2$, then $\langle \mathbf{v}_1, \mathbf{v}_2\rangle = 0$.

*Proof.* Eigenvalues are real by Lemma 0.2.5.3. Compute
$$\lambda_1 \langle \mathbf{v}_1, \mathbf{v}_2\rangle = \langle \lambda_1 \mathbf{v}_1, \mathbf{v}_2\rangle = \langle A\mathbf{v}_1, \mathbf{v}_2\rangle = \langle \mathbf{v}_1, A\mathbf{v}_2\rangle = \langle \mathbf{v}_1, \lambda_2 \mathbf{v}_2\rangle = \overline{\lambda_2}\langle \mathbf{v}_1, \mathbf{v}_2\rangle = \lambda_2 \langle \mathbf{v}_1, \mathbf{v}_2\rangle.$$
So $(\lambda_1 - \lambda_2)\langle \mathbf{v}_1, \mathbf{v}_2\rangle = 0$, and since $\lambda_1 \neq \lambda_2$, $\langle \mathbf{v}_1, \mathbf{v}_2\rangle = 0$. $\square$

**Lemma 0.2.5.5 (Invariant orthogonal complement).** Let $T: \mathbb{C}^n \to \mathbb{C}^n$ be linear and $W \subseteq \mathbb{C}^n$ a subspace. If $T(W) \subseteq W$, then $T^*(W^\perp) \subseteq W^\perp$.

*Proof.* Let $\mathbf{u} \in W^\perp$ and $\mathbf{w} \in W$. Then $\langle T^*\mathbf{u}, \mathbf{w}\rangle = \langle \mathbf{u}, T\mathbf{w}\rangle = 0$ since $T\mathbf{w} \in W$. So $T^*\mathbf{u} \in W^\perp$. $\square$

**Corollary.** If $A$ is Hermitian and $W$ is $A$-invariant, then $W^\perp$ is also $A$-invariant (since $A^* = A$).

### The Spectral Theorem

**Theorem 0.2.5.6 (Spectral theorem for Hermitian matrices).** Let $A \in M_n(\mathbb{C})$ be Hermitian. Then there exists a unitary $U \in M_n(\mathbb{C})$ and a real diagonal $\Lambda \in M_n(\mathbb{R})$ with
$$A = U \Lambda U^*.$$
Equivalently, $\mathbb{C}^n$ has an orthonormal basis of eigenvectors of $A$, and all eigenvalues are real.

*Proof (by induction on $n$).*

*Base case $n = 1$.* Any $1 \times 1$ Hermitian matrix is just a real number $\lambda$. Take $U = (1)$, $\Lambda = (\lambda)$.

*Inductive step.* Assume the theorem holds for Hermitian matrices of size $< n$. Let $A \in M_n(\mathbb{C})$ Hermitian.

*Step 1. Pick an eigenvector.* By the Fundamental Theorem of Algebra, $p_A$ has a root $\lambda \in \mathbb{C}$. By Lemma 0.2.5.3, $\lambda \in \mathbb{R}$. Pick an eigenvector $\mathbf{u}_1 \neq \mathbf{0}$ and normalize: $\|\mathbf{u}_1\| = 1$. Set $W = \mathrm{span}\{\mathbf{u}_1\}$, $W^\perp = \{\mathbf{v} : \langle \mathbf{v}, \mathbf{u}_1\rangle = 0\}$. Then $\dim W^\perp = n - 1$.

*Step 2. Restrict $A$ to $W^\perp$.* $W$ is $A$-invariant (it's an eigenspace). By Lemma 0.2.5.5 (applied with $T = A$, $T^* = A$), $W^\perp$ is also $A$-invariant. Let $A' = A|_{W^\perp}$ be the restriction.

*Step 3. $A'$ is Hermitian.* For $\mathbf{v}, \mathbf{w} \in W^\perp$: $\langle A'\mathbf{v}, \mathbf{w}\rangle = \langle A\mathbf{v}, \mathbf{w}\rangle = \langle \mathbf{v}, A\mathbf{w}\rangle = \langle \mathbf{v}, A'\mathbf{w}\rangle$ using that $A$ is Hermitian and $A\mathbf{v}, A\mathbf{w} \in W^\perp$. So the restriction to an orthonormal basis of $W^\perp$ gives a Hermitian $(n-1) \times (n-1)$ matrix.

*Step 4. Apply induction.* By the inductive hypothesis, $W^\perp$ has an orthonormal basis $\mathbf{u}_2, \ldots, \mathbf{u}_n$ of eigenvectors of $A'$ (equivalently, of $A$, since $A|_{W^\perp} = A'$), with real eigenvalues.

*Step 5. Assemble.* $\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_n$ is an orthonormal basis of $\mathbb{C}^n$ (orthogonality of $\mathbf{u}_1$ to the rest is by construction; the others are mutually orthonormal by induction), all eigenvectors of $A$. Form $U = [\mathbf{u}_1 | \cdots | \mathbf{u}_n]$; then $U$ is unitary (orthonormal columns) and $AU = U\Lambda$ with $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$ real. Thus $A = U\Lambda U^{-1} = U\Lambda U^*$. $\square$

**Theorem 0.2.5.7 (Spectral theorem for real symmetric).** If $A \in M_n(\mathbb{R})$ with $A^\top = A$, then there exist an orthogonal $Q \in M_n(\mathbb{R})$ and diagonal $\Lambda \in M_n(\mathbb{R})$ with $A = Q \Lambda Q^\top$.

*Proof.* Over $\mathbb{C}$, $A$ is Hermitian (since $A^* = \overline{A^\top} = A$ as $A$ is real). By Theorem 0.2.5.6, eigenvalues are real. To get *real* eigenvectors: if $\lambda \in \mathbb{R}$ is an eigenvalue, then $A - \lambda I \in M_n(\mathbb{R})$ is a real matrix with nontrivial kernel; that kernel has a nonzero real vector (since its rank over $\mathbb{R}$ equals its rank over $\mathbb{C}$ — any rational/real-entry matrix has the same rank over any extension field, as rank equals the size of the largest nonvanishing minor). Repeat the inductive construction using real eigenvectors. $\square$

**Theorem 0.2.5.8 (Spectral theorem for normal matrices).** $A \in M_n(\mathbb{C})$ is normal ($AA^* = A^*A$) iff $A = U\Lambda U^*$ for some unitary $U$ and diagonal $\Lambda \in M_n(\mathbb{C})$ (eigenvalues now complex in general).

*Proof.* ($\Leftarrow$) If $A = U\Lambda U^*$ with diagonal $\Lambda$, then $A^* = U\overline{\Lambda}U^*$ and
$$AA^* = U\Lambda \overline{\Lambda}U^* = U|\Lambda|^2 U^*, \qquad A^*A = U\overline{\Lambda}\Lambda U^* = U|\Lambda|^2 U^*.$$
So $AA^* = A^*A$.

($\Rightarrow$) By **Schur's theorem** (proved below as Theorem 0.2.5.9), every $A \in M_n(\mathbb{C})$ is unitarily similar to an *upper triangular* matrix $T$: $A = UTU^*$. If additionally $A$ is normal, so is $T$ (normality is preserved under unitary similarity: $TT^* = U^*AA^*U = U^*A^*AU = T^*T$). An upper triangular normal matrix must be diagonal (see Lemma 0.2.5.10 below). So $A$ is unitarily diagonalizable. $\square$

**Theorem 0.2.5.9 (Schur's theorem).** For every $A \in M_n(\mathbb{C})$ there exist unitary $U$ and upper triangular $T$ with $A = U T U^*$.

*Proof (by induction on $n$).* Base $n = 1$: trivial.

Step: pick an eigenvalue $\lambda_1$ and a unit eigenvector $\mathbf{u}_1 \in \mathbb{C}^n$. Extend $\{\mathbf{u}_1\}$ to an orthonormal basis $\{\mathbf{u}_1, \mathbf{w}_2, \ldots, \mathbf{w}_n\}$ of $\mathbb{C}^n$ (Gram–Schmidt on any extension). Let $U_1 = [\mathbf{u}_1 | \mathbf{w}_2 | \cdots | \mathbf{w}_n]$ unitary. Then
$$U_1^* A U_1 = \begin{pmatrix} \lambda_1 & * \\ \mathbf{0} & A' \end{pmatrix}$$
with $A' \in M_{n-1}(\mathbb{C})$. By induction, $A' = V T' V^*$ with $V$ unitary and $T'$ upper triangular. Let $U_2 = \begin{pmatrix} 1 & 0 \\ 0 & V \end{pmatrix}$ (block unitary) and $U = U_1 U_2$. Then
$$U^* A U = U_2^* (U_1^* A U_1) U_2 = \begin{pmatrix} \lambda_1 & * \cdot V \\ 0 & V^* A' V \end{pmatrix} = \begin{pmatrix} \lambda_1 & * \\ 0 & T' \end{pmatrix} = T$$
upper triangular. $\square$

**Lemma 0.2.5.10.** An upper-triangular normal matrix is diagonal.

*Proof.* Let $T$ be upper triangular and normal. Compare diagonal entries of $TT^*$ and $T^*T$:
$$(TT^*)_{ii} = \sum_j |T_{ij}|^2 = \sum_{j \geq i} |T_{ij}|^2 \qquad \text{(upper triangular: } T_{ij} = 0 \text{ for } j < i\text{)}.$$
$$(T^*T)_{ii} = \sum_j |T_{ji}|^2 = \sum_{j \leq i} |T_{ji}|^2.$$
Equating for $i = 1$: $(TT^*)_{11} = \sum_{j \geq 1} |T_{1j}|^2$ vs $(T^*T)_{11} = |T_{11}|^2$. Equality forces $T_{1j} = 0$ for $j > 1$, i.e., the first row has only a diagonal entry. Then comparing $i = 2$: $(TT^*)_{22} = \sum_{j \geq 2} |T_{2j}|^2$ vs $(T^*T)_{22} = |T_{22}|^2$ (since $T_{12} = 0$ by Step 1). Same conclusion. By induction, all off-diagonal entries vanish. $\square$

### Spectral Decomposition as a Sum of Projectors

**Corollary 0.2.5.11 (Spectral decomposition).** If $A \in M_n(\mathbb{R})$ is symmetric with eigenvalues $\lambda_1, \ldots, \lambda_n$ (with multiplicity) and orthonormal eigenvectors $\mathbf{q}_1, \ldots, \mathbf{q}_n$, then
$$A = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^\top.$$
Each $P_i = \mathbf{q}_i \mathbf{q}_i^\top$ is the orthogonal projector onto $\mathrm{span}\{\mathbf{q}_i\}$. These projectors satisfy $P_i^2 = P_i$, $P_i^\top = P_i$, $P_i P_j = 0$ for $i \neq j$, and $\sum_i P_i = I$.

*Proof.* $A = Q\Lambda Q^\top = Q \left(\sum_i \lambda_i \mathbf{e}_i \mathbf{e}_i^\top\right) Q^\top = \sum_i \lambda_i (Q\mathbf{e}_i)(Q\mathbf{e}_i)^\top = \sum_i \lambda_i \mathbf{q}_i \mathbf{q}_i^\top$. The projector properties follow from orthonormality: $P_i P_j = \mathbf{q}_i \mathbf{q}_i^\top \mathbf{q}_j \mathbf{q}_j^\top = \mathbf{q}_i (\mathbf{q}_i^\top \mathbf{q}_j) \mathbf{q}_j^\top = \delta_{ij} \mathbf{q}_i \mathbf{q}_j^\top$. $\square$

**Interpretation.** A symmetric matrix acts by projecting onto each orthogonal eigendirection and scaling by $\lambda_i$. $A^k = \sum_i \lambda_i^k \mathbf{q}_i \mathbf{q}_i^\top$ — matrix powers are painless.

### Variational Characterization (Courant–Fischer)

Order the eigenvalues of symmetric $A$ as $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$.

**Theorem 0.2.5.12 (Rayleigh quotient; extremal eigenvalues).** For symmetric $A \in M_n(\mathbb{R})$,
$$\lambda_1 = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}, \qquad \lambda_n = \min_{\mathbf{x} \neq \mathbf{0}} \frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}.$$
Maximizer: $\mathbf{q}_1$. Minimizer: $\mathbf{q}_n$.

*Proof.* Diagonalize $A = Q\Lambda Q^\top$. Let $\mathbf{y} = Q^\top \mathbf{x}$; then $\|\mathbf{y}\| = \|\mathbf{x}\|$ and $\mathbf{x}^\top A \mathbf{x} = \mathbf{y}^\top \Lambda \mathbf{y} = \sum_i \lambda_i y_i^2$. So
$$\frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}} = \frac{\sum_i \lambda_i y_i^2}{\sum_i y_i^2}$$
is a convex combination of eigenvalues with weights $y_i^2 / \|\mathbf{y}\|^2$. This is bounded between $\lambda_n$ and $\lambda_1$, achieved at $\mathbf{y} = \mathbf{e}_1$ (giving $\mathbf{x} = \mathbf{q}_1$) and $\mathbf{y} = \mathbf{e}_n$ (giving $\mathbf{x} = \mathbf{q}_n$) respectively. $\square$

**Theorem 0.2.5.13 (Courant–Fischer min-max).** For symmetric $A \in M_n(\mathbb{R})$ with $\lambda_1 \geq \cdots \geq \lambda_n$ and $1 \leq k \leq n$,
$$\lambda_k = \max_{\substack{S \subseteq \mathbb{R}^n \\ \dim S = k}} \min_{\substack{\mathbf{x} \in S \\ \mathbf{x} \neq \mathbf{0}}} \frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}} = \min_{\substack{S \subseteq \mathbb{R}^n \\ \dim S = n - k + 1}} \max_{\substack{\mathbf{x} \in S \\ \mathbf{x} \neq \mathbf{0}}} \frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}.$$

*Proof (of the first equality).* Let $V_k = \mathrm{span}\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$. On $V_k$, $\mathbf{x}^\top A \mathbf{x}/\|\mathbf{x}\|^2 = \sum_{i \leq k} \lambda_i y_i^2 / \sum_{i \leq k} y_i^2 \geq \lambda_k$ (weighted average of $\lambda_1, \ldots, \lambda_k$, all $\geq \lambda_k$). So the $\min$ on $V_k$ is $\geq \lambda_k$, achieved at $\mathbf{q}_k$. Hence $\max_S \min_{\mathbf{x} \in S} \geq \lambda_k$.

Conversely, let $S$ be any $k$-dimensional subspace. Consider $W = \mathrm{span}\{\mathbf{q}_k, \mathbf{q}_{k+1}, \ldots, \mathbf{q}_n\}$, of dimension $n - k + 1$. By the dimension formula (Exercise 0.2.1.9):
$$\dim(S \cap W) \geq \dim S + \dim W - n = k + (n-k+1) - n = 1.$$
So there is a nonzero $\mathbf{x} \in S \cap W$. Writing $\mathbf{x} = \sum_{i \geq k} c_i \mathbf{q}_i$:
$$\frac{\mathbf{x}^\top A \mathbf{x}}{\|\mathbf{x}\|^2} = \frac{\sum_{i \geq k} \lambda_i c_i^2}{\sum_{i \geq k} c_i^2} \leq \lambda_k.$$
So $\min_{\mathbf{x} \in S} \mathbf{x}^\top A \mathbf{x}/\|\mathbf{x}\|^2 \leq \lambda_k$ for every $k$-dim $S$. Taking $\max$ over $S$, the value is $\leq \lambda_k$. Combined with the $\geq$ direction, equality. $\square$

**Corollary 0.2.5.14 (Weyl's inequality — eigenvalue perturbation).** For symmetric $A, B \in M_n(\mathbb{R})$,
$$\lambda_k(A+B) \leq \lambda_k(A) + \lambda_1(B), \qquad \lambda_k(A+B) \geq \lambda_k(A) + \lambda_n(B).$$
In particular, $|\lambda_k(A+B) - \lambda_k(A)| \leq \|B\|_{op}$ where $\|B\|_{op}$ is the operator norm (= largest eigenvalue magnitude for symmetric $B$, see Topic 0.2.9).

*Proof.* Apply Courant–Fischer. Let $V_k$ achieve the max for $A$:
$$\lambda_k(A+B) \geq \min_{\mathbf{x} \in V_k} \frac{\mathbf{x}^\top (A+B) \mathbf{x}}{\|\mathbf{x}\|^2} \geq \min_{\mathbf{x} \in V_k} \frac{\mathbf{x}^\top A \mathbf{x}}{\|\mathbf{x}\|^2} + \min_{\mathbf{x}} \frac{\mathbf{x}^\top B \mathbf{x}}{\|\mathbf{x}\|^2} = \lambda_k(A) + \lambda_n(B).$$
The other side is symmetric. $\square$

This result is deep for quant: it says that if you perturb a covariance matrix a little (in spectral norm), eigenvalues move only a little. This underlies the stability of PCA to noise.

### Worked Examples

**Example 0.2.5.15 (Spectral decomposition).** $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ (symmetric). $p_A(t) = t^2 - 4t + 3 = (t-1)(t-3)$, eigenvalues $1, 3$.

Eigenvector for $\lambda = 3$: $A - 3I = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}$, kernel $(1,1)^\top$. Normalize: $\mathbf{q}_1 = (1,1)^\top/\sqrt 2$.

Eigenvector for $\lambda = 1$: $A - I = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, kernel $(1,-1)^\top$. Normalize: $\mathbf{q}_2 = (1,-1)^\top/\sqrt 2$.

Check orthogonality: $\mathbf{q}_1 \cdot \mathbf{q}_2 = (1-1)/2 = 0$. ✓

Spectral decomposition:
$$A = 3 \cdot \frac{1}{2}\begin{pmatrix} 1 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix} + 1 \cdot \frac{1}{2}\begin{pmatrix} 1 \\ -1 \end{pmatrix}\begin{pmatrix} 1 & -1 \end{pmatrix} = \frac{3}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1\end{pmatrix} + \frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1\end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}.\ \checkmark$$

**Example 0.2.5.16 (Symmetric matrix with repeated eigenvalue).** $A = 2I + \mathbf{v}\mathbf{v}^\top$ where $\mathbf{v} = (1, 1, 1)^\top$. This is rank-1 perturbation of $2I$. Eigenvalues of $\mathbf{v}\mathbf{v}^\top$: $\|\mathbf{v}\|^2 = 3$ (with eigenvector $\mathbf{v}$), and $0$ (twice, on $\mathbf{v}^\perp$). So eigenvalues of $A$ are $2 + 3 = 5$ (once) and $2$ (twice). The spectral theorem gives orthonormal basis: $\mathbf{q}_1 = \mathbf{v}/\sqrt 3$, and any orthonormal basis of $\{\mathbf{v}\}^\perp$, e.g., $\mathbf{q}_2 = (1,-1,0)^\top/\sqrt 2$, $\mathbf{q}_3 = (1,1,-2)^\top/\sqrt 6$.

**Example 0.2.5.17 (A non-normal matrix).** $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$. $A^* A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}$ vs $A A^* = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$. Different. So $A$ is not normal; the spectral theorem does not apply. (Topic 0.2.6 will handle it via SVD.)

### Computational Implementation

```python
import numpy as np
from numpy.linalg import eigh, eig

# Example 0.2.5.15
A = np.array([[2., 1.],
              [1., 2.]])

# Use eigh for symmetric/Hermitian — returns real eigenvalues, orthonormal eigenvectors
eigvals, Q = eigh(A)  # eigvals ascending
print("Eigenvalues:", eigvals)     # [1. 3.]
print("Q:\n", Q)

# Verify orthonormality
assert np.allclose(Q.T @ Q, np.eye(2))
# Verify diagonalization
Lambda = np.diag(eigvals)
assert np.allclose(A, Q @ Lambda @ Q.T)
print("Spectral decomposition verified.")

# Spectral decomposition as sum of rank-1 projectors
def spectral_reconstruct(eigvals, Q):
    n = len(eigvals)
    A = np.zeros((n, n))
    for i in range(n):
        q_i = Q[:, i:i+1]  # column vector
        A += eigvals[i] * (q_i @ q_i.T)
    return A

assert np.allclose(A, spectral_reconstruct(eigvals, Q))

# Rayleigh quotient: sampling gives lambda_1 and lambda_n
rng = np.random.default_rng(0)
xs = rng.standard_normal((2, 10000))
rayleigh = np.sum(xs * (A @ xs), axis=0) / np.sum(xs**2, axis=0)
print(f"Rayleigh samples: min={rayleigh.min():.4f}, max={rayleigh.max():.4f}")
print(f"True lambda_min={eigvals.min():.4f}, lambda_max={eigvals.max():.4f}")

# Schur decomposition (for non-normal matrices)
from scipy.linalg import schur
B = np.array([[1., 1.],
              [0., 1.]])
T, U = schur(B, output='complex')
print("Schur T:\n", T)              # upper triangular
print("Unitary U:\n", U)
assert np.allclose(B, U @ T @ U.conj().T)

# PCA directly via eigh of covariance
n_samples, n_features = 500, 3
X = rng.standard_normal((n_samples, n_features)) @ np.array([[2, 1, 0.5],
                                                             [0, 1.5, 0.3],
                                                             [0, 0, 1]])
X = X - X.mean(axis=0)                       # center
Sigma = X.T @ X / (n_samples - 1)            # sample covariance
var_explained, components = eigh(Sigma)       # ascending
# Reverse to get descending
var_explained = var_explained[::-1]
components = components[:, ::-1]
print(f"Variance explained per PC: {var_explained}")
print(f"Proportion: {var_explained / var_explained.sum()}")
```

**Numerical note.** `numpy.linalg.eigh` exploits symmetry: faster, more accurate than `eig`, returns real eigenvalues. *Always* use `eigh` for symmetric/Hermitian inputs. For sparse/large symmetric matrices, `scipy.sparse.linalg.eigsh` computes a few extreme eigenvalues via Lanczos iteration.

### [QUANT APPLICATION] — PCA on Correlation Matrices, Factor Models, Robust Covariance

**(A) PCA in depth.** Given a centered data matrix $X \in \mathbb{R}^{n \times d}$ (rows = observations, columns = features), the sample covariance is $\Sigma = X^\top X / (n-1)$. By the spectral theorem, $\Sigma = Q\Lambda Q^\top$ with orthonormal eigenvectors (the **principal components**, or "factors") and ordered eigenvalues $\lambda_1 \geq \cdots \geq \lambda_d \geq 0$ (variances along each component).

The $k$-th PC score of observation $\mathbf{x}_i$ is $\mathbf{q}_k^\top \mathbf{x}_i$. Reconstruction with $k$ PCs: $\hat{\mathbf{x}}_i = \sum_{j=1}^k (\mathbf{q}_j^\top \mathbf{x}_i) \mathbf{q}_j$. The truncation error in mean squared norm is $\sum_{j > k} \lambda_j$ (Mirsky/Eckart–Young — the "best rank-$k$ approximation" result, proven via SVD in Topic 0.2.6).

For equity returns: PC1 typically explains $\sim 40$–$60\%$ of variance and represents the *market factor*. PC2, PC3, ... are style/sector factors.

**(B) Factor models are spectral decompositions.** The single-factor model $\mathbf{r}_t = \alpha + \beta \, m_t + \varepsilon_t$ implies $\Sigma = \sigma_m^2 \beta \beta^\top + \Omega$ (idiosyncratic). The $\beta \beta^\top$ is rank-1; PCA on $\Sigma$ recovers $\beta$ as the leading eigenvector (up to scale). Multi-factor models $\Sigma = B \Phi B^\top + \Omega$ give low-rank + diagonal structure; eigendecomposition reveals how many factors are needed.

**(C) Spectral clipping for robust covariance.** Sample covariance $\hat\Sigma$ is noisy for $n \sim d$. A common robust estimator clips eigenvalues: replace $\lambda_i \to \max(\lambda_i, c)$ for some floor $c > 0$. This preserves eigenvectors (directions) but stabilizes eigenvalues against small-sample noise. Mathematically: $\hat\Sigma_{\mathrm{robust}} = \sum_i \max(\lambda_i, c) \mathbf{q}_i \mathbf{q}_i^\top$.

**(D) Marchenko–Pastur denoising.** When $d/n$ is a finite ratio, the bulk of small eigenvalues of $\hat\Sigma$ spreads following the Marchenko–Pastur distribution (random matrix theory, Subject 8). Eigenvalues inside the MP bulk are statistical noise; those above are signal. Keep only the "signal" eigenvalues in the spectral decomposition.

**(E) Spectral gap and mixing.** For Markov transition matrix $P$ symmetrized by detailed balance, the second-largest eigenvalue $\lambda_2$ controls mixing time: $t_{\mathrm{mix}} \sim \log(1/\varepsilon) / (1 - \lambda_2)$. Credit-rating mixing, portfolio rebalancing dynamics, and MCMC simulations all care about spectral gaps.

### Exercises

#### ★ (Foundation)

**E0.2.5.1.** Prove: if $A$ is real symmetric, $\mathrm{tr}(A^k) = \sum_i \lambda_i^k$.

*Hint.* $A = Q\Lambda Q^\top$, trace is cyclic: $\mathrm{tr}(Q\Lambda^k Q^\top) = \mathrm{tr}(\Lambda^k Q^\top Q) = \mathrm{tr}(\Lambda^k)$.

**E0.2.5.2.** Show: a real symmetric matrix $A$ is invertible iff all eigenvalues are nonzero, and $A^{-1}$ has eigenvalues $1/\lambda_i$ with the same eigenvectors.

**E0.2.5.3.** Verify the spectral theorem numerically on $A = \begin{pmatrix} 5 & 4 & 2 \\ 4 & 5 & 2 \\ 2 & 2 & 2 \end{pmatrix}$: compute eigenvalues via `eigh`, confirm orthogonality of eigenvectors, and reconstruct $A$ from the spectral decomposition.

**E0.2.5.4.** For symmetric $A$ with eigenvalues $\lambda_1 \geq \cdots \geq \lambda_n$, prove $\|A\|_F^2 = \sum_i \lambda_i^2$ where $\|A\|_F^2 = \sum_{ij} A_{ij}^2$.

*Hint.* $\|A\|_F^2 = \mathrm{tr}(A^\top A) = \mathrm{tr}(A^2)$ for symmetric $A$.

#### ★★ (Intermediate)

**E0.2.5.5 (Simultaneous diagonalization of commuting symmetrics).** If $A, B \in M_n(\mathbb{R})$ are symmetric and $AB = BA$, show they are simultaneously diagonalizable by a *single* orthogonal $Q$.

*Hint.* $B$ preserves each eigenspace of $A$. Restrict $B$ to each eigenspace; it is symmetric there; diagonalize by an orthonormal basis of that eigenspace.

**E0.2.5.6 (Square root).** Let $A$ be symmetric positive semidefinite (eigenvalues $\geq 0$). Show there is a unique symmetric PSD $B$ with $B^2 = A$, and $B = Q\Lambda^{1/2} Q^\top$ where $A = Q\Lambda Q^\top$ and $\Lambda^{1/2} = \mathrm{diag}(\sqrt{\lambda_i})$.

*Hint.* Existence: construct from diagonalization. Uniqueness: any PSD square root commutes with $A$ (because $B = f(A)$ for a polynomial $f$, once you know $B$ commutes with $A$; use simultaneous diagonalization).

**E0.2.5.7 (Rayleigh iteration convergence).** Implement the **power iteration** to find the largest eigenvalue of a symmetric matrix: $\mathbf{x}_{k+1} = A\mathbf{x}_k / \|A\mathbf{x}_k\|$. Show (with a proof and with code) that if $|\lambda_1| > |\lambda_2|$, then $\mathbf{x}_k$ converges to (a scalar multiple of) $\mathbf{q}_1$ geometrically, with error $\|\mathbf{x}_k - c\mathbf{q}_1\| \leq C |\lambda_2/\lambda_1|^k$.

**E0.2.5.8 (Sylvester's law of inertia — preview).** The **inertia** of symmetric $A$ is the triple $(n_+, n_-, n_0)$ of (positive, negative, zero) eigenvalue counts. Show: for any invertible $P$, $PAP^\top$ has the same inertia as $A$. (This is the "congruence" transformation, not similarity; eigenvalues change but signs are preserved.)

*Hint.* Use the variational characterization: $n_+$ = max dim of a subspace where $\mathbf{x}^\top A \mathbf{x} > 0$ on nonzero vectors. Show this is invariant under $A \mapsto P A P^\top$.

#### ★★★ (Challenge)

**E0.2.5.9 (Weyl monotonicity).** Prove: for symmetric $A, B$ with $B \succeq 0$ (PSD), $\lambda_k(A + B) \geq \lambda_k(A)$ for all $k$. (Adding PSD only increases eigenvalues.)

*Hint.* Rayleigh on the subspace achieving max for $A$.

**E0.2.5.10 (Simultaneous diagonalization of a generalized eigenvalue problem).** Let $A$ be symmetric and $B$ symmetric positive definite. Show: there exists an invertible $V$ with $V^\top A V = \Lambda$ (diagonal) and $V^\top B V = I$. The diagonal entries of $\Lambda$ are the *generalized eigenvalues* $A\mathbf{v} = \lambda B \mathbf{v}$.

*Hint.* Let $B = L L^\top$ (Cholesky — Topic 0.2.7). Transform to $L^{-1} A L^{-\top}$, which is symmetric; apply spectral theorem; undo the transform.

This is the linear algebra behind **LDA** (Fisher's linear discriminant) and **generalized PCA**.

**E0.2.5.11 (Hoffman–Wielandt inequality).** For symmetric $A, B$ with sorted eigenvalues,
$$\sum_i (\lambda_i(A) - \lambda_i(B))^2 \leq \|A - B\|_F^2.$$
This strengthens Weyl's inequality: not only do individual eigenvalues not move far, but the whole sorted spectrum is Lipschitz in Frobenius norm.

*Hint.* Write $A = Q\Lambda Q^\top$, $B = P \mathrm{M} P^\top$, reduce to the trace-matching doubly stochastic optimization and use the Birkhoff–von Neumann theorem. (This is hard — it is a full research-level exercise. Attempt a proof for $2 \times 2$ first.)

---

## Topic 0.2.6 — Singular Value Decomposition (SVD)

### Motivation

The spectral theorem handles symmetric matrices elegantly. But most matrices in practice are **not** square and certainly not symmetric: a data matrix $X \in \mathbb{R}^{n \times d}$, a regression design matrix, a time-by-asset return matrix. The **singular value decomposition** is the spectral theorem's generalization to *arbitrary* rectangular matrices. It says:

> Every matrix $A \in \mathbb{R}^{m \times n}$ can be written as $A = U \Sigma V^\top$, with orthogonal $U \in \mathbb{R}^{m \times m}$, orthogonal $V \in \mathbb{R}^{n \times n}$, and "diagonal" $\Sigma \in \mathbb{R}^{m \times n}$ with nonnegative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$.

The SVD tells a geometric story: $A$ acts on $\mathbb{R}^n$ by (1) rotating/reflecting via $V^\top$, (2) stretching each coordinate axis by the singular values, (3) rotating/reflecting the result via $U$. Every linear map is a rotation, a stretch, and another rotation. That's it.

The SVD is arguably the most useful decomposition in applied linear algebra. It unlocks: least squares and the pseudoinverse, low-rank approximation (Eckart–Young), PCA (cleanest formulation), matrix completion, recommendation systems, total least squares, procrustes analysis, spectral graph embedding, image compression, and — in finance — factor modeling, trading signal decomposition, and robust PCA.

### Prerequisites

Spectral theorem (Topic 0.2.5), orthogonal complements, rank-nullity (Topic 0.2.2).

### Definitions and Construction

**Theorem 0.2.6.1 (SVD — existence).** Let $A \in \mathbb{R}^{m \times n}$ with rank $r$. There exist orthogonal $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$, and a "diagonal" matrix $\Sigma \in \mathbb{R}^{m \times n}$ (entries zero except $\Sigma_{ii} = \sigma_i$ for $i = 1, \ldots, \min(m, n)$) with
$$A = U \Sigma V^\top, \qquad \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots = \sigma_{\min(m,n)}.$$

*Proof (constructive).*

*Step 1. Consider $A^\top A$.* This is an $n \times n$ real symmetric matrix, and it is **positive semidefinite**: $\mathbf{x}^\top (A^\top A) \mathbf{x} = \|A\mathbf{x}\|^2 \geq 0$. By the spectral theorem (0.2.5.7), $A^\top A = V \Lambda V^\top$ with $V$ orthogonal and $\Lambda = \mathrm{diag}(\mu_1, \ldots, \mu_n)$, $\mu_1 \geq \cdots \geq \mu_n \geq 0$.

*Step 2. Note $\mathrm{rank}(A^\top A) = \mathrm{rank}(A)$.* We proved this in Topic 0.2.2 (consequence of $\ker(A^\top A) = \ker(A)$): if $A^\top A \mathbf{x} = 0$, then $\mathbf{x}^\top A^\top A \mathbf{x} = \|A\mathbf{x}\|^2 = 0$, so $A\mathbf{x} = 0$. Conversely, $A\mathbf{x} = 0 \Rightarrow A^\top A \mathbf{x} = 0$. So $\ker(A^\top A) = \ker(A)$, and by rank-nullity, $\mathrm{rank}(A^\top A) = n - \dim\ker(A) = \mathrm{rank}(A) = r$.

Therefore exactly $r$ of the $\mu_i$ are positive and the rest are zero. Order $V$'s columns so $\mu_1 \geq \cdots \geq \mu_r > 0 = \mu_{r+1} = \cdots = \mu_n$. Set $\sigma_i = \sqrt{\mu_i}$ for $i = 1, \ldots, \min(m,n)$, with the convention $\sigma_i = 0$ for $i > r$.

*Step 3. Build $U$'s first $r$ columns.* Let $\mathbf{v}_1, \ldots, \mathbf{v}_n$ be the columns of $V$. Define
$$\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i \quad \text{for } i = 1, \ldots, r.$$
Then:

(a) $\|\mathbf{u}_i\| = 1$: $\|\mathbf{u}_i\|^2 = \frac{1}{\sigma_i^2} \mathbf{v}_i^\top A^\top A \mathbf{v}_i = \frac{1}{\sigma_i^2} \mu_i = 1$.

(b) $\mathbf{u}_i \perp \mathbf{u}_j$ for $i \neq j$: $\mathbf{u}_i^\top \mathbf{u}_j = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^\top A^\top A \mathbf{v}_j = \frac{\mu_j}{\sigma_i \sigma_j} \mathbf{v}_i^\top \mathbf{v}_j = 0$ since $\{\mathbf{v}_i\}$ orthonormal.

*Step 4. Extend $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$ to an orthonormal basis of $\mathbb{R}^m$.* Use Gram–Schmidt on any extension (or equivalently, pick an orthonormal basis of $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}^\perp$). Call the resulting basis $\mathbf{u}_1, \ldots, \mathbf{u}_m$ and set $U = [\mathbf{u}_1 | \cdots | \mathbf{u}_m]$.

*Step 5. Verify $A = U\Sigma V^\top$.* It suffices to show $AV = U\Sigma$. Column by column:

- For $i \leq r$: $(AV)_{\cdot i} = A\mathbf{v}_i = \sigma_i \mathbf{u}_i = (U\Sigma)_{\cdot i}$ since $(U\Sigma)_{\cdot i} = \sum_j \Sigma_{ji} \mathbf{u}_j = \sigma_i \mathbf{u}_i$.

- For $i > r$: $A\mathbf{v}_i$? Well, $A^\top A \mathbf{v}_i = \mu_i \mathbf{v}_i = 0$, so $\|A\mathbf{v}_i\|^2 = \mathbf{v}_i^\top A^\top A \mathbf{v}_i = 0$, giving $A\mathbf{v}_i = \mathbf{0}$. And $(U\Sigma)_{\cdot i} = \sum_j \Sigma_{ji} \mathbf{u}_j = 0$ since column $i$ of $\Sigma$ is zero (given $i > r \geq \min(m,n) - $  well, need to be careful: $\Sigma_{ii} = \sigma_i = 0$ for $i > r$, other entries of column $i$ are zero). ✓

So $AV = U\Sigma$, hence $A = U\Sigma V^\top$. $\square$

**Definition 0.2.6.2.** $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ are the **singular values** of $A$. Columns of $V$ are **right singular vectors**; columns of $U$ are **left singular vectors**. We have

- $A \mathbf{v}_i = \sigma_i \mathbf{u}_i$ for $i \leq r$ (and $= \mathbf{0}$ for $i > r$, on kernel)
- $A^\top \mathbf{u}_i = \sigma_i \mathbf{v}_i$ for $i \leq r$

**Remark (uniqueness).** Singular values are unique (positive square roots of eigenvalues of $A^\top A$). Singular vectors are unique up to sign when all singular values are distinct; degenerate when there are repeated singular values (rotation inside eigenspace).

### Reduced (Thin) SVD

Writing out the full $U \Sigma V^\top$ carries a lot of zero-padded structure. The **reduced SVD** (also called thin SVD) keeps only the nonzero columns:
$$A = U_r \Sigma_r V_r^\top, \qquad U_r \in \mathbb{R}^{m \times r},\ \Sigma_r = \mathrm{diag}(\sigma_1, \ldots, \sigma_r),\ V_r \in \mathbb{R}^{n \times r}.$$

Here $U_r$ has orthonormal columns ($U_r^\top U_r = I_r$ but $U_r U_r^\top \neq I_m$ — it is the projector onto $\mathrm{range}(A)$), similarly $V_r$.

Equivalently (outer-product form):
$$A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top.$$
Each term is a rank-$1$ matrix; $A$ is their weighted sum.

### Connection to Eigendecomposition

The SVD of $A$ produces simultaneously:
- Eigendecomposition of $A^\top A$: eigenvalues $\sigma_i^2$, eigenvectors $\mathbf{v}_i$ (right singular vectors).
- Eigendecomposition of $A A^\top$: eigenvalues $\sigma_i^2$ (same!), eigenvectors $\mathbf{u}_i$ (left singular vectors).
- $A^\top A$ and $A A^\top$ have the **same nonzero eigenvalues** with the same multiplicities. This is a special case of E0.2.4.5.

### The Four Fundamental Subspaces

From the SVD, we can read off the four fundamental subspaces of $A$:

- $\mathrm{range}(A) = \mathrm{span}\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$ (column space of $A$).
- $\ker(A^\top) = \mathrm{span}\{\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m\}$ (left null space).
- $\mathrm{range}(A^\top) = \mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_r\}$ (row space).
- $\ker(A) = \mathrm{span}\{\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n\}$.

These are orthogonal pairs: $\mathrm{range}(A) \perp \ker(A^\top)$ in $\mathbb{R}^m$ and $\mathrm{range}(A^\top) \perp \ker(A)$ in $\mathbb{R}^n$ — this is the "fundamental theorem of linear algebra" (Gilbert Strang).

### Pseudoinverse

**Definition 0.2.6.3 (Moore–Penrose pseudoinverse).** For $A = U\Sigma V^\top$, the **pseudoinverse** is
$$A^+ = V \Sigma^+ U^\top,$$
where $\Sigma^+$ is obtained from $\Sigma$ by transposing and replacing each nonzero $\sigma_i$ with $1/\sigma_i$ (and leaving zeros as zeros).

**Theorem 0.2.6.4 (Properties of pseudoinverse).** $A^+$ is the unique matrix satisfying the **Moore–Penrose conditions**:

1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^\top = A A^+$
4. $(A^+ A)^\top = A^+ A$

*Proof of existence (satisfying the four conditions).* Using $A = U\Sigma V^\top$ and $A^+ = V\Sigma^+ U^\top$:
- $AA^+ = U\Sigma V^\top V \Sigma^+ U^\top = U\Sigma \Sigma^+ U^\top = U \begin{pmatrix} I_r & 0 \\ 0 & 0 \end{pmatrix} U^\top = U_r U_r^\top$, orthogonal projector onto $\mathrm{range}(A)$.
- $A^+A = V \Sigma^+ \Sigma V^\top = V_r V_r^\top$, orthogonal projector onto $\mathrm{range}(A^\top)$.
- $AA^+A = U_r U_r^\top U\Sigma V^\top = U_r (U_r^\top U) \Sigma V^\top$; since $U_r^\top U = [I_r | 0]$ and applying to $\Sigma$ keeps the top-left $r \times r$ block, we recover $U \Sigma V^\top = A$.

(2) is symmetric. (3), (4): $AA^+ = U_r U_r^\top$ and $A^+A = V_r V_r^\top$ are symmetric projectors. $\square$

*Uniqueness:* Suppose $B$ satisfies all four. Then (using the four identities manipulation) $B = A^+$. Standard exercise in Hoffman–Kunze.

**Corollary.** When $A$ is invertible, $A^+ = A^{-1}$. When $A$ has full column rank ($r = n$, so $m \geq n$), $A^+ = (A^\top A)^{-1} A^\top$. When $A$ has full row rank ($r = m$), $A^+ = A^\top (AA^\top)^{-1}$.

**Linear least squares.** The solution to $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|^2$ of minimum norm is $\mathbf{x}^* = A^+ \mathbf{b}$. When $A$ has full column rank, this reduces to the normal equation $(A^\top A)\mathbf{x} = A^\top \mathbf{b}$.

### Low-Rank Approximation — Eckart–Young–Mirsky

**Theorem 0.2.6.5 (Eckart–Young).** Let $A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ be the SVD of $A \in \mathbb{R}^{m \times n}$, and for $k < r$ let
$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top.$$
Then $A_k$ is the **best rank-$k$ approximation** of $A$ in:

(a) **Spectral (operator) norm:** $\min_{\mathrm{rank}(B) \leq k} \|A - B\|_{op} = \|A - A_k\|_{op} = \sigma_{k+1}$.

(b) **Frobenius norm:** $\min_{\mathrm{rank}(B) \leq k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i > k} \sigma_i^2}$.

*Proof of (b).* Write $B \in \mathbb{R}^{m \times n}$ with $\mathrm{rank}(B) \leq k$. We want to show $\|A - B\|_F^2 \geq \sum_{i > k} \sigma_i^2$.

$\dim\ker(B) \geq n - k$. And $\dim V_{k+1} = k+1$ where $V_{k+1} = \mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{k+1}\}$. By the dimension formula, $\dim(V_{k+1} \cap \ker B) \geq (k+1) + (n-k) - n = 1$. Pick a unit vector $\mathbf{w} \in V_{k+1} \cap \ker B$. Write $\mathbf{w} = \sum_{i=1}^{k+1} c_i \mathbf{v}_i$ with $\sum c_i^2 = 1$.

$$\|A\mathbf{w}\|^2 = \sum_{i=1}^{k+1} c_i^2 \sigma_i^2 \geq \sigma_{k+1}^2.$$

And $B\mathbf{w} = 0$, so $\|(A - B)\mathbf{w}\| = \|A\mathbf{w}\| \geq \sigma_{k+1}$.

This gives $\|A - B\|_{op} \geq \sigma_{k+1}$, proving (a).

For Frobenius, the argument is more intricate. One approach: use the singular-value analogue of Weyl's / Hoffman–Wielandt. Let $\tau_1 \geq \tau_2 \geq \cdots$ be the singular values of $B$ (at most $k$ nonzero) and $\nu_i$ be the singular values of $A - B$. The singular-value Mirsky inequality states
$$\sum_i (\sigma_i(A) - \tau_i)^2 \leq \|A - B\|_F^2.$$
Since $\tau_{k+1} = \tau_{k+2} = \cdots = 0$, this gives
$$\|A - B\|_F^2 \geq \sum_{i > k} \sigma_i^2,$$
with equality for $B = A_k$. The Mirsky inequality itself uses Hoffman–Wielandt applied to the symmetric $\begin{pmatrix} 0 & A \\ A^\top & 0\end{pmatrix}$; see Horn–Johnson, *Matrix Analysis*. $\square$

**Takeaway:** The best way to compress a matrix is to keep its top-$k$ singular triples. Truncation error is precisely the tail of the singular value spectrum.

### Worked Examples

**Example 0.2.6.6 (SVD of a simple matrix).** $A = \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix}$.

*Step 1.* $A^\top A = \begin{pmatrix} 4 & 3 \\ 0 & -5\end{pmatrix}\begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix} = \begin{pmatrix} 25 & -15 \\ -15 & 25\end{pmatrix}$.

*Step 2.* Eigenvalues of $A^\top A$: $p(t) = (t-25)^2 - 225 = t^2 - 50t + 625 - 225 = t^2 - 50t + 400 = (t - 40)(t - 10)$. So $\mu_1 = 40$, $\mu_2 = 10$, and $\sigma_1 = 2\sqrt{10}$, $\sigma_2 = \sqrt{10}$.

*Step 3.* Eigenvectors: $A^\top A - 40I = \begin{pmatrix} -15 & -15 \\ -15 & -15 \end{pmatrix}$, $\mathbf{v}_1 = (1,-1)^\top/\sqrt 2$. $A^\top A - 10I = \begin{pmatrix} 15 & -15 \\ -15 & 15\end{pmatrix}$, $\mathbf{v}_2 = (1,1)^\top/\sqrt 2$.

*Step 4.* Left singular vectors: $\mathbf{u}_1 = A\mathbf{v}_1 / \sigma_1 = \frac{1}{\sqrt 2}\begin{pmatrix} 4 \\ 3+5\end{pmatrix}/ (2\sqrt{10}) = \frac{1}{2\sqrt{20}}\begin{pmatrix} 4 \\ 8\end{pmatrix} = \frac{1}{\sqrt{20}}\begin{pmatrix} 2 \\ 4 \end{pmatrix} = \frac{1}{\sqrt 5}\begin{pmatrix} 1 \\ 2\end{pmatrix}$. Similarly $\mathbf{u}_2 = A\mathbf{v}_2 / \sigma_2 = \frac{1}{\sqrt 2}\begin{pmatrix} 4 \\ 3 - 5\end{pmatrix} / \sqrt{10} = \frac{1}{\sqrt{20}} \begin{pmatrix} 4 \\ -2\end{pmatrix} = \frac{1}{\sqrt 5}\begin{pmatrix} 2 \\ -1\end{pmatrix}$.

Check orthogonality: $\mathbf{u}_1 \cdot \mathbf{u}_2 = (1 \cdot 2 + 2 \cdot -1)/5 = 0$. ✓

*Assembly.*
$$U = \frac{1}{\sqrt 5}\begin{pmatrix} 1 & 2 \\ 2 & -1 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 2\sqrt{10} & 0 \\ 0 & \sqrt{10} \end{pmatrix}, \quad V = \frac{1}{\sqrt 2}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}.$$

*Sanity.* $\det(A) = 4 \cdot (-5) - 0 \cdot 3 = -20$. $\sigma_1 \sigma_2 = 2\sqrt{10} \cdot \sqrt{10} = 20 = |\det A|$. ✓

**Example 0.2.6.7 (Rank-1 approximation).** Consider
$$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \\ 1 & 1 \end{pmatrix}.$$

Numerically (computed in code below): singular values $\sigma_1 \approx 4.216$, $\sigma_2 \approx 2.000$. Rank-1 approximation retains $\sigma_1$:
$$A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top.$$
Interpretation: replace the column-wise structure of $A$ by its best single-factor summary. In finance: a single-factor PCA of a return matrix.

**Example 0.2.6.8 (SVD for least squares with rank deficiency).** Suppose the design matrix $X \in \mathbb{R}^{n \times p}$ is rank deficient (multicollinearity: some predictors are nearly linear combinations of others). The normal equation $(X^\top X)^{-1} X^\top \mathbf{y}$ fails — $X^\top X$ is singular. SVD saves us:
$$X = U\Sigma V^\top \implies \hat\beta_{\mathrm{LS}} = X^+ \mathbf{y} = V \Sigma^+ U^\top \mathbf{y} = \sum_{i: \sigma_i > 0} \frac{\mathbf{u}_i^\top \mathbf{y}}{\sigma_i} \mathbf{v}_i.$$
The pseudoinverse picks the minimum-norm solution. In practice, one also truncates singular values below a threshold (discarding noisy small-$\sigma$ directions) — this is **truncated SVD regression** or **Tikhonov regularization** via SVD.

### Computational Implementation

```python
import numpy as np
from numpy.linalg import svd, pinv, norm, matrix_rank

# Example 0.2.6.6
A = np.array([[4., 0.],
              [3., -5.]])
U, s, Vt = svd(A)
print("Singular values:", s)         # [6.324..., 3.162...] = 2*sqrt(10), sqrt(10)
print("U:\n", U)
print("V^T:\n", Vt)

# Reconstruct
Sigma = np.zeros_like(A); np.fill_diagonal(Sigma, s)
assert np.allclose(A, U @ Sigma @ Vt)
print("|det(A)|:", abs(np.linalg.det(A)), "= prod(s):", np.prod(s))  # Both 20

# Reduced (thin) SVD
U_r, s_r, Vt_r = svd(A, full_matrices=False)
print("Thin U shape:", U_r.shape)       # (m, min(m,n))
assert np.allclose(A, (U_r * s_r) @ Vt_r)

# Pseudoinverse via SVD
def svd_pinv(A, tol=None):
    U, s, Vt = svd(A, full_matrices=False)
    if tol is None:
        tol = max(A.shape) * np.finfo(A.dtype).eps * s.max()
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return Vt.T @ np.diag(s_inv) @ U.T

A_pinv = svd_pinv(A)
print("A A+ A = A:", np.allclose(A @ A_pinv @ A, A))
print("A+ A A+ = A+:", np.allclose(A_pinv @ A @ A_pinv, A_pinv))
print("Compare with numpy pinv:", np.allclose(A_pinv, pinv(A)))

# Low-rank approximation
def truncated_svd(A, k):
    """Best rank-k approximation in Frobenius norm."""
    U, s, Vt = svd(A, full_matrices=False)
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    return (U_k * s_k) @ Vt_k

# Example: random matrix, compare rank-k errors
rng = np.random.default_rng(42)
M = rng.standard_normal((20, 15))
_, s, _ = svd(M, full_matrices=False)
for k in [1, 5, 10, 14]:
    Mk = truncated_svd(M, k)
    err = norm(M - Mk, 'fro')
    theoretical = np.sqrt(np.sum(s[k:]**2))
    print(f"k={k:3d}  ||M - M_k||_F = {err:.4f}  theoretical = {theoretical:.4f}")
# Errors should match exactly: Eckart-Young verified

# SVD-based least squares with rank deficiency
p, n = 4, 3
# Construct rank-deficient X: 4 columns but rank 3 (one column is sum of others)
X_full = rng.standard_normal((10, p))
X_full[:, 3] = X_full[:, 0] + X_full[:, 1]  # create collinearity
y = rng.standard_normal(10)
print("rank(X)=", matrix_rank(X_full))          # 3, not 4
try:
    beta_normal = np.linalg.solve(X_full.T @ X_full, X_full.T @ y)
except np.linalg.LinAlgError as e:
    print(f"Normal equation failed: {e}")
beta_pinv = pinv(X_full) @ y
print("Pseudoinverse solution:", beta_pinv)
print("Norm of beta_pinv:", norm(beta_pinv))
print("Residual:", norm(X_full @ beta_pinv - y))

# Image compression (singular value decay)
# (conceptual demo — swap in a real image if desired)
n_img = 100
# "Image" with low-rank structure + noise
rank_true = 5
L = rng.standard_normal((n_img, rank_true))
R = rng.standard_normal((rank_true, n_img))
img = L @ R + 0.1 * rng.standard_normal((n_img, n_img))
_, s_img, _ = svd(img, full_matrices=False)
print(f"First 10 singular values of 'image': {s_img[:10]}")
# Observe sharp drop after k=5: low-rank signal plus small noise tail
```

**Numerical tip.** `numpy.linalg.svd(A, full_matrices=False)` gives the thin SVD; faster and uses less memory than full. For very large matrices, `scipy.sparse.linalg.svds` computes only the top $k$ singular triples via Lanczos.

### [QUANT APPLICATION] — Factor Models via SVD, Recommendation Systems, Robust PCA, Matrix Completion

**(A) SVD = PCA (up to centering).** For a centered data matrix $X$ (rows = observations, columns = features), $X = U\Sigma V^\top$ gives:
- Principal components (loadings) = columns of $V$ (right singular vectors).
- Principal component scores = $U\Sigma$ (projections of observations onto PCs).
- Variance explained by PC$_i$ = $\sigma_i^2 / (n-1)$.

SVD-based PCA avoids explicitly forming $X^\top X$ (which doubles condition number). For a $n \times d$ matrix with $n \gg d$ or $n \ll d$, this is a meaningful numerical advantage.

**(B) Factor model regression.** Decompose a returns matrix $R \in \mathbb{R}^{T \times N}$ (time × assets) as $R = U\Sigma V^\top$. The top-$k$ truncation $R_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ is a rank-$k$ factor model: $\mathbf{u}_i$ (after normalization) represent $k$ time-series "factors," and $\mathbf{v}_i$ are loadings. This is the Connor–Korajczyk approach — principal components of the returns matrix directly.

**(C) Recommendation systems (Netflix Prize).** A user-item rating matrix $R$ is mostly missing. With the known entries, seek a low-rank approximation $R \approx U V^\top$. Classic SVD does not handle missing values, but matrix completion algorithms (soft-impute, ALS) iterate SVD on imputed matrices. The low-rank assumption is equivalent to: "users are characterized by $k$ latent factors, items by $k$ latent factors, and ratings are their inner product."

**(D) Robust PCA.** Given $M = L + S$ where $L$ is low-rank (normal market structure) and $S$ is sparse (idiosyncratic shocks, gross errors, fraud), decompose by convex optimization:
$$\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.}\quad L + S = M,$$
where $\|L\|_* = \sum_i \sigma_i(L)$ is the nuclear norm (sum of singular values — a convex relaxation of rank). This is hugely useful in separating regime shifts from persistent factor structure. Candès–Li–Ma–Wright (2011) established the recovery theory.

**(E) Signal-to-noise via singular value gap.** The effective rank of an empirically estimated matrix is the number of singular values above a threshold. In quant, selecting the number of PCA factors often reduces to "count eigenvalues above the Marchenko–Pastur bulk edge." Random matrix theory gives you the threshold explicitly (Subject 8).

**(F) Total least squares.** Ordinary least squares assumes noise only in $\mathbf{y}$: $\mathbf{y} = X\beta + \varepsilon$. When $X$ has noise too, the SVD-based Total Least Squares estimator is $\hat\beta_{TLS} = -V_{12}/V_{22}$ where the SVD of $[X | \mathbf{y}]$ is partitioned accordingly. Classic in errors-in-variables regression.

### Exercises

#### ★ (Foundation)

**E0.2.6.1.** Compute the SVD of $A = \begin{pmatrix} 1 & 1 \\ 1 & 1\end{pmatrix}$ by hand. What is its rank?

**E0.2.6.2.** Let $A \in \mathbb{R}^{m \times n}$ with $m \geq n$. Prove: $A$ has full column rank iff all singular values are positive.

**E0.2.6.3.** Prove $\|A\|_F^2 = \sum_i \sigma_i^2$ and $\|A\|_{op} = \sigma_1$ (operator norm = largest singular value).

*Hint.* Frobenius: use $\|A\|_F^2 = \mathrm{tr}(A^\top A)$. Operator norm: use Rayleigh + spectral theorem on $A^\top A$.

**E0.2.6.4.** Show: if $A$ is symmetric with eigenvalues $\lambda_1, \ldots, \lambda_n$ (real), then singular values are $|\lambda_1|, \ldots, |\lambda_n|$.

#### ★★ (Intermediate)

**E0.2.6.5 (Polar decomposition).** Prove that every $A \in \mathbb{R}^{n \times n}$ admits a *polar decomposition* $A = QS$ where $Q$ is orthogonal and $S$ is symmetric PSD. When is this decomposition unique?

*Hint.* From SVD $A = U\Sigma V^\top$, take $Q = UV^\top$ and $S = V\Sigma V^\top$. Uniqueness: $S = \sqrt{A^\top A}$ is unique; $Q$ is unique when $A$ is invertible.

**E0.2.6.6 (Procrustes problem).** Given $X, Y \in \mathbb{R}^{n \times d}$, find the orthogonal $Q \in O(d)$ minimizing $\|XQ - Y\|_F$.

*Hint.* The minimizer is $Q = UV^\top$ where $X^\top Y = U\Sigma V^\top$. Prove it by expanding the Frobenius norm and using the trace inequality $\mathrm{tr}(QM) \leq \sum_i \sigma_i(M)$ for any orthogonal $Q$.

**E0.2.6.7 (Stability of pseudoinverse).** Let $A_\varepsilon = A + \varepsilon E$ for a small perturbation $E$. Show that if $A$ has full column rank, $\|A_\varepsilon^+ - A^+\|_{op} = O(\varepsilon)$. However, if $A$ is rank-deficient with smallest nonzero singular value $\sigma_r$, a perturbation $\varepsilon$ with $\varepsilon < \sigma_r$ keeps rank the same, but $\|A_\varepsilon^+\|_{op}$ can blow up as $1/\sigma_r$.

*Commentary.* This is why **truncated SVD** is essential in rank-deficient least squares: tiny singular values get enormous in the pseudoinverse, amplifying noise.

**E0.2.6.8 (Courant–Fischer for singular values).** Prove: for $A \in \mathbb{R}^{m \times n}$,
$$\sigma_k(A) = \max_{\substack{S \subseteq \mathbb{R}^n \\ \dim S = k}} \min_{\substack{\mathbf{x} \in S \\ \|\mathbf{x}\| = 1}} \|A\mathbf{x}\|.$$

*Hint.* Apply Courant–Fischer for $A^\top A$.

#### ★★★ (Challenge)

**E0.2.6.9 (Eckart–Young in operator norm).** Complete the proof of Eckart–Young in operator norm: show $\|A - A_k\|_{op} = \sigma_{k+1}$.

*Hint.* Upper bound: $\|A - A_k\|_{op} = \|\sum_{i > k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top\|_{op} = \sigma_{k+1}$ (top singular value of the tail). Lower bound: done in the proof above.

**E0.2.6.10 (Nuclear norm).** The **nuclear norm** of $A$ is $\|A\|_* = \sum_i \sigma_i$. Show:
(a) $\|A\|_*$ is a norm on $\mathbb{R}^{m \times n}$.
(b) Its dual norm (under the trace inner product $\langle A, B\rangle = \mathrm{tr}(A^\top B)$) is the operator norm.
(c) $\|A\|_* = \max\{\mathrm{tr}(U^\top A) : U \in \mathbb{R}^{m \times n}, \|U\|_{op} \leq 1\}$.

These facts underpin nuclear norm minimization in matrix completion.

**E0.2.6.11 (Perturbation of singular vectors — Wedin's theorem, statement only).** Let $A, B \in \mathbb{R}^{m \times n}$, and consider the leading $k$ left singular vectors (columns of $U_k$ and $\tilde U_k$). Prove the *Davis–Kahan $\sin\theta$ theorem* for a simple case:
$$\|\sin\Theta(U_k, \tilde U_k)\|_F \leq \frac{\|A - B\|_F}{\sigma_k(A) - \sigma_{k+1}(A)}.$$
(This is research-level; familiarize yourself with the statement — it governs PCA stability in finite samples.)

---

## Topic 0.2.7 — Positive Definite Matrices

### Motivation

A **positive (semi)definite** matrix is the linear-algebraic analog of a positive real number. If scalars $> 0$ let you compute square roots, ratios, and logs, then positive definite matrices let you compute $A^{1/2}$, $A^{-1}$, $\log A$, and — most importantly — meaningful Cholesky factors $A = LL^\top$. They provide the rigorous backbone for:

- **Covariance matrices:** always positive semidefinite. A "valid" covariance matrix *is* a PSD matrix. When we estimate $\hat\Sigma$ from data, ensuring positive semidefiniteness (and often *definiteness*) is a nontrivial sanity check.
- **Kernel methods:** a Mercer kernel produces a PSD Gram matrix; inner products in a feature space are encoded as PSD matrices.
- **Convex optimization:** the Hessian of a (strictly) convex function is PSD (resp. PD). Semidefinite programming (SDP) is optimization over the cone of PSD matrices.
- **Gaussian distributions:** the density $\mathcal{N}(\mu, \Sigma)$ has $\Sigma$ PD; sampling via $\mathbf{x} = \mu + L\mathbf{z}$ where $LL^\top = \Sigma$ is Cholesky's killer app.
- **Stability of dynamical systems:** Lyapunov equations — $A^\top P + PA = -Q$ for PD $P, Q$ — certify stability.

This topic is pure bread-and-butter for any quant. Every covariance matrix, every kernel matrix, every Hessian of a quadratic program is PSD.

### Prerequisites

Spectral theorem (Topic 0.2.5), inner products (Topic 0.2.3), SVD (Topic 0.2.6).

### Definitions and Basic Properties

**Definition 0.2.7.1 (Positive (semi)definite).** Let $A \in \mathbb{R}^{n \times n}$ be symmetric. $A$ is:
- **positive semidefinite** (PSD), written $A \succeq 0$, if $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x} \in \mathbb{R}^n$;
- **positive definite** (PD), written $A \succ 0$, if $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$.

For Hermitian $A \in \mathbb{C}^{n \times n}$ (so $A^* = A$), the same definitions apply with $\mathbf{x}^* A \mathbf{x}$ (which is real by Hermitian symmetry).

**Convention.** We restrict to symmetric/Hermitian matrices. Some authors define PD/PSD without requiring symmetry, but this is non-standard and leads to algebraic complications. The quadratic form $\mathbf{x}^\top A \mathbf{x}$ depends only on the symmetric part $(A + A^\top)/2$, so the asymmetric case adds nothing new.

**Partial order.** For symmetric $A, B$, we write $A \succeq B$ iff $A - B \succeq 0$ and $A \succ B$ iff $A - B \succ 0$. This is a genuine partial order (the *Loewner order*) on symmetric matrices.

**Theorem 0.2.7.2 (Characterizations of PD).** For symmetric $A \in \mathbb{R}^{n \times n}$, TFAE:

(i) $A \succ 0$ (i.e., $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$).

(ii) All eigenvalues of $A$ are strictly positive.

(iii) There exists an invertible $B$ with $A = B^\top B$.

(iv) There exists an invertible *lower triangular* $L$ with positive diagonal and $A = LL^\top$. (Cholesky factorization.)

(v) All leading principal minors $\det(A_{[1:k, 1:k]}) > 0$ for $k = 1, \ldots, n$. (Sylvester's criterion.)

*Proof.*

(i) $\Rightarrow$ (ii): By the spectral theorem, $A = Q\Lambda Q^\top$ with eigenvalues $\lambda_i$. For the $i$-th eigenvector $\mathbf{q}_i$: $\mathbf{q}_i^\top A \mathbf{q}_i = \lambda_i > 0$ (from (i)). So all $\lambda_i > 0$.

(ii) $\Rightarrow$ (iii): $A = Q\Lambda Q^\top$ with $\Lambda = \mathrm{diag}(\lambda_i)$, $\lambda_i > 0$. Set $B = \Lambda^{1/2} Q^\top = \mathrm{diag}(\sqrt{\lambda_i}) Q^\top$. Then $B^\top B = Q \Lambda^{1/2} \Lambda^{1/2} Q^\top = Q\Lambda Q^\top = A$, and $B$ is invertible (product of invertibles).

(iii) $\Rightarrow$ (i): $\mathbf{x}^\top A \mathbf{x} = \mathbf{x}^\top B^\top B \mathbf{x} = \|B\mathbf{x}\|^2 \geq 0$, with equality iff $B\mathbf{x} = \mathbf{0}$ iff $\mathbf{x} = \mathbf{0}$ (invertibility). So $\mathbf{x} \neq \mathbf{0}$ gives $> 0$.

(ii) $\Leftrightarrow$ (iv): We prove this via Gaussian elimination. Assume (ii), i.e., $A \succ 0$. Proceed by induction on $n$.

Base $n = 1$: $A = (a_{11})$ with $a_{11} > 0$; set $L = (\sqrt{a_{11}})$. ✓

Inductive step: partition $A = \begin{pmatrix} a_{11} & \mathbf{b}^\top \\ \mathbf{b} & A' \end{pmatrix}$. Since $A \succ 0$, we have $a_{11} = \mathbf{e}_1^\top A \mathbf{e}_1 > 0$. Perform symmetric Gaussian elimination:
$$A = \begin{pmatrix} 1 & 0 \\ \mathbf{b}/a_{11} & I \end{pmatrix}\begin{pmatrix} a_{11} & 0 \\ 0 & A' - \mathbf{b}\mathbf{b}^\top/a_{11} \end{pmatrix}\begin{pmatrix} 1 & \mathbf{b}^\top/a_{11} \\ 0 & I \end{pmatrix}.$$
Set $S = A' - \mathbf{b}\mathbf{b}^\top/a_{11}$ (the **Schur complement**). We claim $S \succ 0$. Suppose $\mathbf{y} \neq \mathbf{0}$ in $\mathbb{R}^{n-1}$. Set $\mathbf{x} = (-\mathbf{b}^\top \mathbf{y}/a_{11}, \mathbf{y})^\top$. Then
$$\mathbf{x}^\top A \mathbf{x} = \frac{a_{11}(\mathbf{b}^\top \mathbf{y})^2}{a_{11}^2} - 2\cdot \frac{\mathbf{b}^\top \mathbf{y}}{a_{11}} \mathbf{b}^\top \mathbf{y} + \mathbf{y}^\top A' \mathbf{y} = \mathbf{y}^\top A' \mathbf{y} - \frac{(\mathbf{b}^\top \mathbf{y})^2}{a_{11}} = \mathbf{y}^\top S \mathbf{y}.$$
Since $\mathbf{x} \neq \mathbf{0}$ (last $n-1$ components are $\mathbf{y} \neq \mathbf{0}$) and $A \succ 0$, $\mathbf{y}^\top S \mathbf{y} > 0$. So $S \succ 0$.

Apply the inductive hypothesis: $S = L' (L')^\top$ with $L'$ lower triangular, positive diagonal. Then
$$A = \begin{pmatrix} 1 & 0 \\ \mathbf{b}/a_{11} & I \end{pmatrix}\begin{pmatrix} \sqrt{a_{11}} & 0 \\ 0 & L' \end{pmatrix}\begin{pmatrix} \sqrt{a_{11}} & 0 \\ 0 & L' \end{pmatrix}^\top \begin{pmatrix} 1 & \mathbf{b}^\top/a_{11} \\ 0 & I \end{pmatrix}.$$
Combining: $A = LL^\top$ with
$$L = \begin{pmatrix} \sqrt{a_{11}} & 0 \\ \mathbf{b}/\sqrt{a_{11}} & L' \end{pmatrix},$$
lower triangular with positive diagonal.

For (iv) $\Rightarrow$ (ii): $A = LL^\top$ invertible implies $A$ invertible (no zero eigenvalues). Also $\mathbf{x}^\top A \mathbf{x} = \|L^\top \mathbf{x}\|^2 \geq 0$. Equality iff $L^\top \mathbf{x} = 0$ iff (invertibility) $\mathbf{x} = 0$. So $A \succ 0$, hence all eigenvalues $> 0$.

(ii) $\Leftrightarrow$ (v) (Sylvester's criterion): The forward direction uses that every leading principal submatrix of $A$ is itself positive definite (restriction of the quadratic form to $\mathbb{R}^k \hookrightarrow \mathbb{R}^n$), hence has positive determinant (product of positive eigenvalues).

The backward direction requires showing that positive leading principal minors imply $A \succ 0$. Induct on $n$. Base $n=1$: $a_{11} > 0$ gives $A \succ 0$ immediately. Step: assume true for $n-1$. The top-left $(n-1)\times(n-1)$ block $A_{n-1}$ has positive leading principal minors (subset), so by induction is positive definite. Perform the same block elimination as in (ii)⇒(iv). The Schur complement $s = a_{nn} - \mathbf{b}^\top A_{n-1}^{-1} \mathbf{b}$ satisfies $\det(A) = \det(A_{n-1}) \cdot s$. Given $\det(A) > 0$ and $\det(A_{n-1}) > 0$, we have $s > 0$. Then the block diagonal matrix $\mathrm{diag}(A_{n-1}, s)$ is PD, and $A$ is congruent to it, so $A \succ 0$. $\square$

**Theorem 0.2.7.3 (PSD analogs).** For symmetric $A$, $A \succeq 0$ iff all eigenvalues $\geq 0$ iff $A = B^\top B$ for some (not necessarily invertible) $B$.

*Proof.* Same pattern; drop the invertibility. $\square$

**Cholesky uniqueness.** The Cholesky factor $L$ (with positive diagonal) is *unique*. Indeed if $A = L_1 L_1^\top = L_2 L_2^\top$ with both lower triangular and positive diagonals, then $L_2^{-1} L_1 = L_2^\top L_1^{-\top}$. LHS is lower triangular; RHS is upper triangular; both equal, so diagonal. And the product is orthogonal (equating $L_2^{-1} L_1 (L_2^{-1} L_1)^\top = L_2^{-1} L_1 L_1^\top L_2^{-\top} = L_2^{-1} A L_2^{-\top} = I$). Diagonal orthogonal with positive diagonal = identity. So $L_1 = L_2$.

### Further Properties and Operations

**Proposition 0.2.7.4 (Closure).** If $A, B \succeq 0$ and $c \geq 0$, then $A + B \succeq 0$ and $cA \succeq 0$. The PSD cone is a convex cone in $\mathrm{Sym}_n(\mathbb{R})$.

*Proof.* $\mathbf{x}^\top (A + B) \mathbf{x} = \mathbf{x}^\top A \mathbf{x} + \mathbf{x}^\top B \mathbf{x} \geq 0$. $\mathbf{x}^\top (cA) \mathbf{x} = c \mathbf{x}^\top A \mathbf{x} \geq 0$. $\square$

**Proposition 0.2.7.5 (Compression/congruence).** If $A \succeq 0$ and $B$ is any real matrix (not necessarily square), then $B^\top A B \succeq 0$. If additionally $A \succ 0$ and $B$ has full column rank, then $B^\top A B \succ 0$.

*Proof.* $\mathbf{x}^\top B^\top A B \mathbf{x} = (B\mathbf{x})^\top A (B\mathbf{x}) \geq 0$. Equality iff $A(B\mathbf{x}) = 0$ (given $A \succeq 0$), and under $A \succ 0$ this requires $B\mathbf{x} = 0$, which under full column rank requires $\mathbf{x} = 0$. $\square$

**Corollary.** If $A \succ 0$ and $P$ is invertible, $P^\top A P \succ 0$. PD is preserved by congruence.

**Proposition 0.2.7.6 (Square root).** Every PSD matrix $A$ has a unique PSD square root $A^{1/2}$: a symmetric PSD matrix $B$ with $B^2 = A$. It is $B = Q\Lambda^{1/2} Q^\top$ where $A = Q\Lambda Q^\top$ is the spectral decomposition.

*Proof.* Existence by construction. Uniqueness: any PSD $B$ with $B^2 = A$ commutes with $A$ (since $BA = B \cdot B^2 = B^3 = B^2 \cdot B = AB$), so by simultaneous diagonalization (E0.2.5.5), $B$ is diagonal in the same basis $Q$, and then $B = Q\mathrm{diag}(\sqrt{\lambda_i}) Q^\top$ forced by $B \succeq 0$ and $B^2 = A$. $\square$

**Proposition 0.2.7.7 (Inverse).** If $A \succ 0$, then $A^{-1} \succ 0$ with eigenvalues $1/\lambda_i$.

### The Schur Complement

**Definition 0.2.7.8.** For a block matrix $M = \begin{pmatrix} A & B \\ B^\top & D\end{pmatrix}$ with $A$ invertible, the **Schur complement of $A$** is $M/A := D - B^\top A^{-1} B$.

**Theorem 0.2.7.9 (Schur complement theorem).** Let $M = \begin{pmatrix} A & B \\ B^\top & D\end{pmatrix}$ be symmetric with $A \succ 0$. Then
$$M \succ 0 \iff D - B^\top A^{-1} B \succ 0, \qquad M \succeq 0 \iff D - B^\top A^{-1} B \succeq 0.$$

*Proof.* Symmetric block-congruence:
$$\begin{pmatrix} I & 0 \\ -B^\top A^{-1} & I\end{pmatrix} \begin{pmatrix} A & B \\ B^\top & D\end{pmatrix}\begin{pmatrix} I & -A^{-1} B \\ 0 & I\end{pmatrix} = \begin{pmatrix} A & 0 \\ 0 & D - B^\top A^{-1} B\end{pmatrix}.$$

The left multiplier is invertible (lower triangular, unit diagonal) and the right is its transpose. So $M$ is congruent (via an invertible matrix) to the block diagonal $\mathrm{diag}(A, D - B^\top A^{-1} B)$. Congruence preserves positive definiteness (Prop 0.2.7.5 applied with invertible $B$, and its converse). Since $A \succ 0$, $M \succ 0$ iff the other block $\succ 0$. $\square$

**Quant application: conditional covariance.** If $(X, Y)$ is jointly Gaussian with $\mathrm{Cov}\begin{pmatrix} X \\ Y\end{pmatrix} = \begin{pmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY}\end{pmatrix}$, the conditional covariance
$$\mathrm{Cov}(Y \mid X) = \Sigma_{YY} - \Sigma_{YX} \Sigma_{XX}^{-1} \Sigma_{XY}$$
is precisely the Schur complement of $\Sigma_{XX}$. It is always PSD by Theorem 0.2.7.9 — the residual uncertainty after conditioning is nonnegative, as it must be. The Kalman filter update formula is exactly this.

### Quadratic Forms and Level Sets

For symmetric $A \succ 0$, the quadratic form $q(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ is a strictly convex function with a unique minimum at $\mathbf{x} = \mathbf{0}$. Its level sets $\{\mathbf{x} : \mathbf{x}^\top A \mathbf{x} = c\}$ for $c > 0$ are ellipsoids with semi-axes $\sqrt{c/\lambda_i}$ along eigenvectors. The Mahalanobis distance from $\mathbf{0}$ (or any center $\mu$) is
$$d_A(\mathbf{x}) = \sqrt{\mathbf{x}^\top A^{-1} \mathbf{x}} \quad \text{or in full: } d(\mathbf{x}, \mu)^2 = (\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu)$$
— a covariance-aware distance. Mahalanobis distance in risk management: detecting outliers.

### Worked Examples

**Example 0.2.7.10 (Cholesky by hand).** Let $A = \begin{pmatrix} 4 & 12 & -16 \\ 12 & 37 & -43 \\ -16 & -43 & 98\end{pmatrix}$.

*Sylvester check.* Leading minors: $4, \det\begin{pmatrix} 4 & 12 \\ 12 & 37\end{pmatrix} = 148 - 144 = 4, \det(A)$. Compute $\det(A)$ via cofactor: $4 \cdot (37 \cdot 98 - 43^2) - 12\cdot(12 \cdot 98 - (-43)(-16)) + (-16)\cdot(12 \cdot -43 - 37 \cdot -16)$. $37 \cdot 98 = 3626, 43^2 = 1849$, first cofactor: $4 \cdot (3626 - 1849) = 4 \cdot 1777 = 7108$. $12 \cdot 98 = 1176, 43 \cdot 16 = 688$, second: $-12 \cdot (1176 - 688) = -12 \cdot 488 = -5856$. $12 \cdot 43 = 516, 37 \cdot 16 = 592$, third: $-16 \cdot (-516 + 592) = -16 \cdot 76 = -1216$. Sum: $7108 - 5856 - 1216 = 36$. All $> 0$. ✓

*Cholesky.* Walk through the algorithm:
- $L_{11} = \sqrt{4} = 2$.
- $L_{21} = 12/2 = 6$, $L_{31} = -16/2 = -8$.
- $L_{22} = \sqrt{37 - 6^2} = \sqrt{37 - 36} = 1$.
- $L_{32} = (−43 − L_{31} L_{21})/L_{22} = (-43 - (-8)(6))/1 = (-43 + 48)/1 = 5$.
- $L_{33} = \sqrt{98 - L_{31}^2 - L_{32}^2} = \sqrt{98 - 64 - 25} = \sqrt{9} = 3$.

$L = \begin{pmatrix} 2 & 0 & 0 \\ 6 & 1 & 0 \\ -8 & 5 & 3\end{pmatrix}$. Verify: $LL^\top = \begin{pmatrix} 4 & 12 & -16 \\ 12 & 37 & -43 \\ -16 & -43 & 98\end{pmatrix}$. ✓

**Example 0.2.7.11 (Checking PSD via eigenvalues).** Is $A = \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2\end{pmatrix}$ PSD?

It is symmetric. Eigenvalues of this tridiagonal Toeplitz: $\lambda_k = 2 - 2\cos(k\pi/4)$ for $k = 1, 2, 3$, giving $\lambda_1 = 2 - \sqrt 2$, $\lambda_2 = 2$, $\lambda_3 = 2 + \sqrt 2$. All positive; so $A \succ 0$.

(This matrix is the discrete Laplacian on 3 nodes with Dirichlet boundary — crops up in finite-difference PDE discretizations for option pricing.)

**Example 0.2.7.12 (Sample covariance is PSD).** Given data $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$, centered ($\bar{\mathbf{x}} = 0$), the sample covariance $\hat\Sigma = \frac{1}{n}\sum_i \mathbf{x}_i \mathbf{x}_i^\top$ is PSD.

*Proof.* $\mathbf{y}^\top \hat\Sigma \mathbf{y} = \frac{1}{n}\sum_i (\mathbf{y}^\top \mathbf{x}_i)^2 \geq 0$. ✓

It is PD iff $\mathrm{span}\{\mathbf{x}_i\} = \mathbb{R}^d$ (need $d$ linearly independent samples). If $n < d$, $\hat\Sigma$ is necessarily rank-deficient.

**Example 0.2.7.13 (Gram matrix).** Given vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ in an inner product space, the **Gram matrix** $G_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$ is PSD. It is PD iff $\{\mathbf{v}_i\}$ are linearly independent.

*Proof.* For $\mathbf{c} \in \mathbb{R}^k$: $\mathbf{c}^\top G \mathbf{c} = \sum_{ij} c_i c_j \langle \mathbf{v}_i, \mathbf{v}_j\rangle = \|\sum_i c_i \mathbf{v}_i\|^2 \geq 0$. Equality iff $\sum c_i \mathbf{v}_i = 0$, iff (independence) $\mathbf{c} = 0$. $\square$

This underpins **kernel methods**: a kernel $k(x, x')$ is valid iff the Gram matrix $K_{ij} = k(x_i, x_j)$ is always PSD (Mercer's condition).

### Computational Implementation

```python
import numpy as np
from numpy.linalg import cholesky, eigvalsh, LinAlgError

# Example 0.2.7.10
A = np.array([[ 4., 12., -16.],
              [12., 37., -43.],
              [-16., -43., 98.]])
L = cholesky(A)
print("Cholesky L:\n", L)
assert np.allclose(A, L @ L.T)

# Checking PD via Cholesky (faster than eigendecomposition)
def is_positive_definite(M, tol=1e-10):
    """Return True iff M is symmetric PD."""
    M = np.asarray(M, dtype=float)
    if not np.allclose(M, M.T, atol=tol):
        return False
    try:
        cholesky(M)
        return True
    except LinAlgError:
        return False

print(is_positive_definite(A))                        # True
print(is_positive_definite(np.array([[1., 2.], [2., 1.]])))  # False (eigenvalues -1, 3)

# Alternative: check eigenvalues
def is_positive_semidefinite(M, tol=1e-10):
    M = np.asarray(M, dtype=float)
    if not np.allclose(M, M.T, atol=tol):
        return False
    return np.all(eigvalsh(M) >= -tol)

# Matrix square root via spectral decomposition
def matrix_sqrt_psd(A):
    """PSD square root."""
    vals, V = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)  # clip small negatives from roundoff
    return V @ np.diag(np.sqrt(vals)) @ V.T

B = matrix_sqrt_psd(A)
assert np.allclose(B @ B, A)

# Sample covariance and its Cholesky-based sampling
rng = np.random.default_rng(0)
# True covariance
Sigma = np.array([[ 2.0, 0.5,  0.0],
                  [ 0.5, 1.0, -0.3],
                  [ 0.0, -0.3, 0.8]])
L = cholesky(Sigma)
# Generate correlated Gaussian samples x = L @ z
n_samples = 50_000
Z = rng.standard_normal((3, n_samples))
X = L @ Z
Sigma_hat = X @ X.T / n_samples
print("||Sigma_hat - Sigma||_F =", np.linalg.norm(Sigma_hat - Sigma, 'fro'))

# Schur complement for conditional covariance
# P( Y | X ) with Gaussian (X, Y)
Sigma_XX = Sigma[:2, :2]
Sigma_XY = Sigma[:2, 2:3]
Sigma_YX = Sigma_XY.T
Sigma_YY = Sigma[2:3, 2:3]
cond_cov = Sigma_YY - Sigma_YX @ np.linalg.solve(Sigma_XX, Sigma_XY)
print("Conditional variance of Y given X:", cond_cov)
print("Is PSD?", is_positive_semidefinite(cond_cov))

# Nearest PSD matrix (important for correcting broken sample covariances)
def nearest_psd(A, epsilon=0.0):
    """Project symmetric matrix to PSD cone (Higham 1988 simplified)."""
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, epsilon)
    return vecs @ np.diag(vals) @ vecs.T

# Demo: a perturbed covariance with spurious negative eigenvalue
bad = np.array([[1.0, 0.9, 0.85],
                [0.9, 1.0, 0.92],
                [0.85, 0.92, 1.0]]) - 0.02 * np.eye(3)
bad[0, 1] = bad[1, 0] = 1.01   # introduce inconsistency
print("Eigenvalues of bad matrix:", eigvalsh(bad))
good = nearest_psd(bad, epsilon=1e-6)
print("Eigenvalues after projection:", eigvalsh(good))
```

**Numerical tips.**
- `np.linalg.cholesky` is the workhorse for PD matrices; both fastest and numerically stable.
- Use `scipy.linalg.solve_triangular` for solves involving the Cholesky factor.
- `scipy.linalg.cho_solve` solves $A\mathbf{x} = \mathbf{b}$ given $L$ directly.
- For rank-deficient or near-PSD matrices (common with sample covariances when $n \sim d$), use nearest-PSD projection or shrinkage estimators (Ledoit–Wolf).

### [QUANT APPLICATION] — Covariance Matrices, Portfolio Optimization, Kalman Filter

**(A) Every covariance matrix is PSD.** If $\mathbf{X} = (X_1, \ldots, X_d)$ is a random vector with finite second moments, $\Sigma = \mathrm{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \mu)(\mathbf{X} - \mu)^\top]$ satisfies $\mathbf{a}^\top \Sigma \mathbf{a} = \mathrm{Var}(\mathbf{a}^\top \mathbf{X}) \geq 0$. This is PSD almost tautologically: variances are nonnegative.

It is PD iff no nontrivial linear combination of the $X_i$ is almost-surely constant, iff the random vector is *truly* $d$-dimensional (not concentrated on a lower-dimensional subspace). In finance: $\Sigma$ is singular precisely when one asset is a linear combination of others (a "redundant" asset — cannot happen with genuinely different securities, except in ETF / index-constituent settings).

**(B) Mean-variance portfolio optimization (Markowitz).** Minimize $\mathbf{w}^\top \Sigma \mathbf{w}$ subject to $\mathbb{E}[R]^\top \mathbf{w} = r$ and $\mathbf{1}^\top \mathbf{w} = 1$. Lagrangian gives closed-form $\mathbf{w}^* = \Sigma^{-1}(\alpha \mathbb{E}[R] + \beta \mathbf{1})$, requiring $\Sigma^{-1}$, which exists iff $\Sigma \succ 0$.

When $\hat\Sigma$ is ill-conditioned (noisy sample covariance on many assets), the inverse amplifies noise → absurd portfolios. Remedies:

1. **Shrinkage** (Ledoit–Wolf): $\hat\Sigma_{\mathrm{shrunk}} = \alpha \hat\Sigma + (1 - \alpha) F$ where $F$ is a well-conditioned target (identity-scaled, or single-factor). $\hat\Sigma_{\mathrm{shrunk}} \succ 0$ even if $\hat\Sigma \succeq 0$.

2. **Factor-model regularization:** $\Sigma = B\Phi B^\top + \mathrm{diag}(\sigma^2_i)$. Guaranteed PD if $\Phi \succ 0$ and all $\sigma_i^2 > 0$.

3. **Robust optimization:** min-max over an uncertainty set of covariances.

**(C) Kalman filter update = Schur complement.** The update step in a linear-Gaussian state-space model computes posterior covariance as
$$P_{t|t} = P_{t|t-1} - K_t H P_{t|t-1} = P_{t|t-1} - P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1} H P_{t|t-1},$$
which is the Schur complement of $(H P_{t|t-1} H^\top + R)$ in the joint $(X, Y)$ covariance. PSD-ness is preserved, as it must be.

**(D) Cholesky for fast Gaussian sampling.** To sample $\mathbf{x} \sim \mathcal{N}(\mu, \Sigma)$: compute $L = \mathrm{chol}(\Sigma)$ once, then $\mathbf{x} = \mu + L\mathbf{z}$ with $\mathbf{z} \sim \mathcal{N}(0, I)$. Cost: one $O(d^3)$ Cholesky, then $O(d^2)$ per sample. Used in Monte Carlo simulations for multi-asset pricing, copula methods, and VaR computation.

**(E) Quadratic programming and SDP.** Quadratic programs $\min \frac{1}{2}\mathbf{x}^\top Q \mathbf{x} + \mathbf{c}^\top \mathbf{x}$ s.t. $A\mathbf{x} \leq \mathbf{b}$ are convex iff $Q \succeq 0$. Semidefinite programs (SDP) are convex optimizations over matrix variables constrained to the PSD cone; used in robust portfolio optimization, spectral graph algorithms, and shape-constrained regression.

### Exercises

#### ★ (Foundation)

**E0.2.7.1.** Is $A = \begin{pmatrix} 3 & 1 \\ 1 & 2\end{pmatrix}$ positive definite? Verify using eigenvalues *and* Sylvester.

**E0.2.7.2.** Compute the Cholesky of $B = \begin{pmatrix} 1 & -1 \\ -1 & 5\end{pmatrix}$.

**E0.2.7.3.** Prove: if $A \succ 0$, all diagonal entries $A_{ii} > 0$.

*Hint.* $A_{ii} = \mathbf{e}_i^\top A \mathbf{e}_i > 0$.

**E0.2.7.4.** Prove: if $A, B \succeq 0$, then $\mathrm{tr}(AB) \geq 0$.

*Hint.* Write $A = C^\top C$; use cyclic property of trace.

**E0.2.7.5.** Let $D = \mathrm{diag}(d_1, \ldots, d_n)$. When is $D$ PD? PSD? Use this to write down the square root and Cholesky of a diagonal PD matrix.

#### ★★ (Intermediate)

**E0.2.7.6 (Hadamard's inequality).** Prove: for $A \succeq 0$, $\det(A) \leq \prod_i A_{ii}$, with equality iff $A$ is diagonal (or has a zero diagonal entry).

*Hint.* Factor $A = B^\top B$ and apply the geometric-arithmetic mean inequality to $\det(A) = \det(B)^2$ bounded by products of column norms of $B$.

**E0.2.7.7 (Loewner monotonicity of inverse).** Prove: if $0 \prec A \preceq B$, then $B^{-1} \preceq A^{-1}$.

*Hint.* Reduce to diagonal case via congruence: find $L$ with $L^\top A L = I$ and $L^\top B L = \mathrm{diag}(\mu_i)$ with $\mu_i \geq 1$ (generalized eigenvalue problem — see E0.2.5.10). Then work out the inverses.

**E0.2.7.8 (Concavity of $\log\det$).** Show: on the positive definite cone, $A \mapsto \log\det(A)$ is concave.

*Hint.* Prove $\log\det(\alpha A + (1-\alpha)B) \geq \alpha \log\det A + (1-\alpha)\log\det B$ using the fact that $\log$ is concave and the Hadamard–Minkowski inequality.

*Relevance.* This is used in Gaussian maximum likelihood (the log-likelihood contains $\log\det\Sigma$) and in determinantal point processes.

**E0.2.7.9 (Kronecker product of PSD matrices).** Show: if $A, B \succeq 0$, then $A \otimes B \succeq 0$.

*Hint.* Eigendecompose both and observe $\lambda(A \otimes B) = \{\lambda_i(A) \lambda_j(B)\}$.

#### ★★★ (Challenge)

**E0.2.7.10 (Matrix inequality — Cauchy–Schwarz).** For PSD $A$, prove $(\mathbf{x}^\top A \mathbf{y})^2 \leq (\mathbf{x}^\top A \mathbf{x})(\mathbf{y}^\top A \mathbf{y})$. This is the Cauchy–Schwarz in the semi-inner-product induced by $A$.

*Hint.* Apply the standard Cauchy–Schwarz argument with the degenerate inner product $\langle \mathbf{x}, \mathbf{y}\rangle_A = \mathbf{x}^\top A \mathbf{y}$.

**E0.2.7.11 (Löwner–Heinz).** Prove: for $A, B \succeq 0$, $A \preceq B$ implies $A^{1/2} \preceq B^{1/2}$, but **not** in general $A^2 \preceq B^2$. Construct a $2 \times 2$ counterexample to the squared version.

*Hint for counterexample.* $A = \begin{pmatrix} 1 & 0 \\ 0 & 0\end{pmatrix}$, $B$ slightly larger but with off-diagonal to break squaring.

**E0.2.7.12 (Positive semidefinite programming duality — statement).** Consider the SDP $\min_{X \succeq 0} \mathrm{tr}(CX)$ s.t. $\mathrm{tr}(A_i X) = b_i$. Its dual is $\max \mathbf{b}^\top \mathbf{y}$ s.t. $C - \sum_i y_i A_i \succeq 0$. Strong duality holds under Slater's condition. Work through the Lagrangian derivation.

*Relevance.* This is the framework behind robust portfolio optimization and spectral relaxations.

**E0.2.7.13 (Sample covariance bias — Marchenko–Pastur preview).** Suppose true $\Sigma = I$ and we have $n$ iid samples $\mathbf{x}_i \sim \mathcal{N}(0, I)$ in $\mathbb{R}^d$ with $c = d/n \in (0, 1)$. Prove that the expected smallest eigenvalue of $\hat\Sigma = \frac{1}{n}\sum \mathbf{x}_i \mathbf{x}_i^\top$ is approximately $(1 - \sqrt{c})^2$ for large $n, d$. (Heuristic and/or simulation-based proof acceptable — rigorous proof awaits Subject 8.)

---

## Topic 0.2.8 — Matrix Calculus

### Motivation

In machine learning, optimization, econometrics, and quantitative finance, we routinely compute derivatives of *scalar-valued functions of matrices* (log-likelihoods, loss functions) and *matrix-valued functions of matrices* (Jacobians of transformations, matrix-valued derivatives in stochastic calculus). Examples:

- Gradient of $\log \det(\Sigma)$ with respect to $\Sigma$ (Gaussian MLE).
- Gradient of $\|X\beta - \mathbf{y}\|^2$ with respect to $\beta$ (linear regression).
- Jacobian of softmax, cross-entropy, or forward pass of a neural network.
- Gradient of portfolio Sharpe ratio with respect to weights.
- Derivative of $A^{-1}$ with respect to $A$ (used in implicit function theorem, backpropagation through solve).

These are solvable by two approaches:

1. **Component-wise** calculus — tedious and error-prone.
2. **Matrix calculus** — a concise notation and a set of identities that reduce such derivations to one- or two-line algebra.

This topic gives you the toolkit. It's not deep mathematics — it's bookkeeping elevated to ergonomics. But it is used *constantly* in quant: any time you optimize a likelihood, fit a regression, compute a Greek via adjoint methods, or backprop through a custom layer, you are doing matrix calculus.

### Prerequisites

Differentiability in multiple variables (review below), basic linear algebra, $O(\cdot)$ notation.

### Notational Conventions (Choose One and Stick With It)

There are two conventions for the derivative of a scalar $f: \mathbb{R}^n \to \mathbb{R}$ with respect to $\mathbf{x}$:

- **Numerator layout** (Jacobian-style): $\partial f/\partial \mathbf{x}$ is a **row** vector: $(\partial f/\partial x_1, \ldots, \partial f/\partial x_n)$.
- **Denominator layout** (Gradient-style): $\partial f/\partial \mathbf{x}$ is a **column** vector: $(\partial f/\partial x_1, \ldots, \partial f/\partial x_n)^\top$.

Both conventions exist in the literature. We will use **denominator layout for gradients of scalars** (so $\nabla_\mathbf{x} f$ is a column vector, matching the convention in optimization) and **numerator layout for Jacobians of vector functions** (so $\partial \mathbf{f}/\partial \mathbf{x}$ is a matrix with rows for each component of $\mathbf{f}$, which matches the matrix product rule). We will be explicit when there is any ambiguity. This is the convention of Magnus–Neudecker's *Matrix Differential Calculus*.

### The Fréchet Derivative

**Definition 0.2.8.1.** Let $U \subseteq \mathbb{R}^n$ open and $f: U \to \mathbb{R}^m$. $f$ is **differentiable** at $\mathbf{x}_0 \in U$ if there exists a linear map $Df(\mathbf{x}_0): \mathbb{R}^n \to \mathbb{R}^m$ with
$$f(\mathbf{x}_0 + \mathbf{h}) = f(\mathbf{x}_0) + Df(\mathbf{x}_0) \mathbf{h} + o(\|\mathbf{h}\|), \quad \mathbf{h} \to \mathbf{0}.$$

The matrix representation of $Df(\mathbf{x}_0)$ in the standard bases is the **Jacobian matrix** $J_f(\mathbf{x}_0) \in \mathbb{R}^{m \times n}$ with
$$(J_f)_{ij} = \frac{\partial f_i}{\partial x_j}(\mathbf{x}_0).$$

When $m = 1$, the **gradient** is the column vector $\nabla f(\mathbf{x}_0) = J_f(\mathbf{x}_0)^\top \in \mathbb{R}^n$.

**Theorem 0.2.8.2 (Chain rule).** If $g: U \to V$ is differentiable at $\mathbf{x}$ and $f: V \to W$ is differentiable at $g(\mathbf{x})$, then $f \circ g$ is differentiable at $\mathbf{x}$ with
$$D(f \circ g)(\mathbf{x}) = Df(g(\mathbf{x})) \cdot Dg(\mathbf{x}).$$
In Jacobian form: $J_{f \circ g}(\mathbf{x}) = J_f(g(\mathbf{x})) J_g(\mathbf{x})$.

This is the cornerstone identity — all of matrix calculus is applying the chain rule repeatedly to compositions.

### The Differential (`d`) Notation

The most powerful approach to matrix calculus is the **differential calculus** of Magnus–Neudecker. Given a matrix-valued function $F(X)$, we compute the **differential** $dF$ by treating $X \mapsto X + dX$ and expanding to first order in $dX$. Once we have $dF$ in the form $dF = A(X)\, dX\, B(X) + C(X)\, dX + \ldots$, we identify the derivative.

**Rules for the differential** (identical to single-variable calculus but applied to matrices):
1. $d(A + B) = dA + dB$.
2. $d(AB) = (dA)B + A\,dB$ (matrix product rule — *order matters*; no commutation).
3. $d(A^\top) = (dA)^\top$.
4. $d\,\mathrm{tr}(A) = \mathrm{tr}(dA)$.
5. $d\,\det(A) = \det(A) \mathrm{tr}(A^{-1} dA)$, for invertible $A$ (proven below).
6. $d(A^{-1}) = -A^{-1} (dA) A^{-1}$, for invertible $A$ (proven below).
7. $d(A \otimes B) = dA \otimes B + A \otimes dB$ (Kronecker product).

**The Identification Theorem.** If for all $dX$ (appropriate space),
$$df = \mathrm{tr}(G^\top dX), \quad f \text{ scalar},$$
then $\partial f / \partial X = G$ (denominator layout: same shape as $X$).

If $dF = A (dX) B$ with $F$ matrix-valued, then using **vectorization** ($\mathrm{vec}(F)$ stacks columns of $F$ into a column vector) and the identity $\mathrm{vec}(AXB) = (B^\top \otimes A) \mathrm{vec}(X)$:
$$d\,\mathrm{vec}(F) = (B^\top \otimes A)\, d\,\mathrm{vec}(X).$$
So the Jacobian $\partial \mathrm{vec}(F) / \partial \mathrm{vec}(X) = B^\top \otimes A$.

### Core Identities (Derivations)

We now derive the identities that everyone memorizes (and that you should *also* know how to re-derive).

**Identity 1: Gradient of $\mathbf{a}^\top \mathbf{x}$.**

$f = \mathbf{a}^\top \mathbf{x} = \sum_i a_i x_i$. $df = \mathbf{a}^\top d\mathbf{x}$, so $\partial f / \partial \mathbf{x} = \mathbf{a}$.

**Identity 2: Gradient of $\mathbf{x}^\top A \mathbf{x}$.**

$df = d(\mathbf{x}^\top A \mathbf{x}) = (d\mathbf{x})^\top A \mathbf{x} + \mathbf{x}^\top A (d\mathbf{x}) = \mathbf{x}^\top A^\top d\mathbf{x} + \mathbf{x}^\top A d\mathbf{x}$ (using $(d\mathbf{x})^\top A \mathbf{x} = (A^\top \mathbf{x})^\top d\mathbf{x}$). So
$$df = \mathbf{x}^\top (A + A^\top) d\mathbf{x} \implies \partial f/\partial \mathbf{x} = (A + A^\top)\mathbf{x}.$$
If $A$ is symmetric, this reduces to $2A\mathbf{x}$.

**Identity 3: Gradient of $\|X\beta - \mathbf{y}\|^2$ with respect to $\beta$.**

$f = (X\beta - \mathbf{y})^\top (X\beta - \mathbf{y})$. Let $\mathbf{r} = X\beta - \mathbf{y}$. $d\mathbf{r} = X d\beta$. $df = d(\mathbf{r}^\top \mathbf{r}) = 2\mathbf{r}^\top d\mathbf{r} = 2\mathbf{r}^\top X d\beta = 2(X\beta - \mathbf{y})^\top X d\beta$. So
$$\nabla_\beta f = 2 X^\top (X\beta - \mathbf{y}).$$
Setting this to zero gives the **normal equations**: $X^\top X \beta = X^\top \mathbf{y}$.

**Identity 4: Derivative of $\det(A)$.**

*Proposition.* For invertible $A$, $d\,\det(A) = \det(A)\, \mathrm{tr}(A^{-1} dA)$. Equivalently, $\partial \det(A)/\partial A = \det(A)\,A^{-\top}$.

*Proof.* Start from $\det(A + dA) = \det(A(I + A^{-1} dA)) = \det(A) \det(I + A^{-1} dA)$. For any matrix $M$ with small entries, $\det(I + M) = \prod_i (1 + \lambda_i(M)) = 1 + \mathrm{tr}(M) + O(\|M\|^2)$. So
$$\det(A + dA) = \det(A) (1 + \mathrm{tr}(A^{-1} dA) + O(\|dA\|^2)).$$
Subtracting $\det(A)$: $d\det(A) = \det(A) \mathrm{tr}(A^{-1} dA)$. Via the identification $df = \mathrm{tr}(G^\top dA)$, we read $G = \det(A) A^{-\top}$. $\square$

**Identity 5: Derivative of $\log\det(A)$.**

$d\log\det(A) = \frac{d\det(A)}{\det(A)} = \mathrm{tr}(A^{-1} dA)$. Hence $\partial \log\det(A)/\partial A = A^{-\top}$. For symmetric PD $A$, $A^{-\top} = A^{-1}$.

**Identity 6: Derivative of $A^{-1}$.**

$I = A A^{-1}$. Differentiate: $0 = dI = (dA) A^{-1} + A (dA^{-1})$. Rearrange: $dA^{-1} = -A^{-1} (dA) A^{-1}$.

**Identity 7: Derivative of $\mathrm{tr}(A^{-1} B)$ with respect to $A$.**

$d \mathrm{tr}(A^{-1} B) = \mathrm{tr}(dA^{-1} \cdot B) = \mathrm{tr}(-A^{-1} (dA) A^{-1} B) = -\mathrm{tr}(A^{-1} B A^{-1} dA)$. So $\partial f/\partial A = -A^{-\top} B^\top A^{-\top} = -(A^{-1} B A^{-1})^\top$.

**Identity 8: Derivative of $\mathrm{tr}(AB)$ with respect to $A$.**

$d\mathrm{tr}(AB) = \mathrm{tr}((dA) B) = \mathrm{tr}(B dA)$. By the identification, $\partial f/\partial A = B^\top$.

**Identity 9: Derivative of $\mathrm{tr}(A X^\top B X C)$ with respect to $X$.** (A common pattern.)

$d\mathrm{tr}(A X^\top B X C) = \mathrm{tr}(A (dX)^\top B X C) + \mathrm{tr}(A X^\top B (dX) C)$.

First term: $\mathrm{tr}(A (dX)^\top B X C) = \mathrm{tr}((BXC A)^\top dX^\top) = \mathrm{tr}(dX^\top BXCA) = \mathrm{tr}((BXCA)^\top dX) = \mathrm{tr}(A^\top C^\top X^\top B^\top dX)$. (Using cyclic and transpose-of-trace.)

Second term: $\mathrm{tr}(A X^\top B (dX) C) = \mathrm{tr}(CA X^\top B dX)$.

Sum: $df = \mathrm{tr}((A^\top C^\top X^\top B^\top + CA X^\top B) dX)$.

Identification: $\partial f/\partial X = (A^\top C^\top X^\top B^\top + CA X^\top B)^\top = B X C A + B^\top X C^\top A^\top$.

*(This is tedious but mechanical — follow the bookkeeping.)*

**Identity 10: Jacobian of matrix inverse $\partial \mathrm{vec}(A^{-1}) / \partial \mathrm{vec}(A)$.**

From $dA^{-1} = -A^{-1} dA \cdot A^{-1}$:
$$d\mathrm{vec}(A^{-1}) = -\mathrm{vec}(A^{-1} dA A^{-1}) = -(A^{-\top} \otimes A^{-1}) \mathrm{vec}(dA).$$
So the Jacobian is $-(A^{-\top} \otimes A^{-1})$.

### Gradient, Hessian, and Second-Order Information

For $f: \mathbb{R}^n \to \mathbb{R}$ twice differentiable, the **Hessian** is $H_f(\mathbf{x}) = \nabla^2 f(\mathbf{x}) \in \mathbb{R}^{n \times n}$ with $(H_f)_{ij} = \partial^2 f/(\partial x_i \partial x_j)$. Symmetric (by Schwarz's theorem, given continuity of second partials).

**Second-order Taylor:**
$$f(\mathbf{x}_0 + \mathbf{h}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \mathbf{h} + \tfrac{1}{2} \mathbf{h}^\top H_f(\mathbf{x}_0) \mathbf{h} + o(\|\mathbf{h}\|^2).$$

**Convexity and Hessian.** $f$ is convex on an open convex set iff $H_f \succeq 0$ everywhere. Strictly convex iff $H_f \succ 0$ (locally). This connects Topic 0.2.7 to optimization: PD Hessian means strictly convex local minimum.

**Example.** For $f(\beta) = \|X\beta - \mathbf{y}\|^2$: $\nabla f = 2X^\top (X\beta - \mathbf{y})$, $H_f = 2 X^\top X \succeq 0$. Convex always; strictly convex iff $X$ has full column rank.

### Vectorization and Kronecker Products (Essentials)

**Definition 0.2.8.3.** For $A \in \mathbb{R}^{m \times n}$, $\mathrm{vec}(A) \in \mathbb{R}^{mn}$ stacks columns of $A$ vertically.

**Definition 0.2.8.4.** For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$, the **Kronecker product** $A \otimes B \in \mathbb{R}^{mp \times nq}$ has block structure $(A \otimes B)_{ij} = A_{ij} B$.

**Key identities:**
- $\mathrm{vec}(AXB) = (B^\top \otimes A) \mathrm{vec}(X)$.
- $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$ when the products are defined.
- $(A \otimes B)^\top = A^\top \otimes B^\top$.
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$ (if both invertible).
- $\det(A \otimes B) = \det(A)^q \det(B)^p$ for $A \in M_p, B \in M_q$.
- Eigenvalues: $\lambda(A \otimes B) = \{\lambda_i(A) \lambda_j(B)\}$.

Vectorization translates linear-matrix equations into linear-vector equations. For example, the Lyapunov equation $AX + XA^\top = Q$ becomes $(I \otimes A + A \otimes I) \mathrm{vec}(X) = \mathrm{vec}(Q)$.

### Worked Examples

**Example 0.2.8.5 (MLE of Gaussian covariance).** Suppose $\mathbf{x}_1, \ldots, \mathbf{x}_n$ iid $\mathcal{N}(\mathbf{0}, \Sigma)$. The log-likelihood (ignoring constants) is
$$\ell(\Sigma) = -\frac{n}{2}\log\det\Sigma - \frac{1}{2}\sum_{i=1}^n \mathbf{x}_i^\top \Sigma^{-1} \mathbf{x}_i = -\frac{n}{2}\log\det\Sigma - \frac{n}{2}\mathrm{tr}(\Sigma^{-1} S)$$
where $S = \frac{1}{n}\sum_i \mathbf{x}_i \mathbf{x}_i^\top$ is the empirical covariance.

Differential: $d\ell = -\frac{n}{2}\mathrm{tr}(\Sigma^{-1} d\Sigma) - \frac{n}{2}\mathrm{tr}(-\Sigma^{-1}(d\Sigma)\Sigma^{-1} S) = -\frac{n}{2}\mathrm{tr}((\Sigma^{-1} - \Sigma^{-1} S \Sigma^{-1}) d\Sigma)$.

(Used $d\Sigma^{-1} = -\Sigma^{-1}(d\Sigma)\Sigma^{-1}$ and cyclic trace.)

Identification: $\partial \ell/\partial \Sigma = -\frac{n}{2}(\Sigma^{-1} - \Sigma^{-1} S \Sigma^{-1})$. Setting to zero: $\Sigma^{-1} = \Sigma^{-1} S \Sigma^{-1}$, i.e., $\Sigma = S$. So the MLE is $\hat\Sigma = S$. ✓

**Example 0.2.8.6 (Ridge regression gradient).** $L(\beta) = \|X\beta - \mathbf{y}\|^2 + \lambda \|\beta\|^2$.

$\nabla L = 2X^\top(X\beta - \mathbf{y}) + 2\lambda \beta$. Setting to zero: $(X^\top X + \lambda I)\beta = X^\top \mathbf{y}$. Closed form: $\hat\beta = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$.

Note: $X^\top X + \lambda I \succ 0$ for $\lambda > 0$ (even if $X^\top X$ is rank-deficient), guaranteeing a unique solution. Ridge regression = regularized pseudoinverse.

**Example 0.2.8.7 (Sharpe ratio gradient).** Portfolio Sharpe is $S(\mathbf{w}) = (\mathbf{w}^\top \mu) / \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}$.

Let $a = \mathbf{w}^\top \mu$, $b = \mathbf{w}^\top \Sigma \mathbf{w}$. $da = \mu^\top d\mathbf{w}$, $db = 2 \mathbf{w}^\top \Sigma d\mathbf{w}$ (using Identity 2 with symmetric $\Sigma$). 

$dS = d(a/b^{1/2}) = \frac{b^{1/2} da - a \cdot \frac{1}{2} b^{-1/2} db}{b} = \frac{1}{\sqrt b} da - \frac{a}{2 b^{3/2}} db$.

Plug in: $dS = \frac{\mu^\top d\mathbf{w}}{\sqrt b} - \frac{a \mathbf{w}^\top \Sigma d\mathbf{w}}{b^{3/2}} = \left(\frac{\mu^\top}{\sqrt b} - \frac{a \mathbf{w}^\top \Sigma}{b^{3/2}}\right) d\mathbf{w}$.

So $\nabla_\mathbf{w} S = \frac{\mu}{\sqrt b} - \frac{a \Sigma \mathbf{w}}{b^{3/2}}$.

Setting to zero: $\sqrt b \mu = a \Sigma \mathbf{w} / \sqrt b$, so $b \mu = a \Sigma \mathbf{w}$, i.e., $\mathbf{w} \propto \Sigma^{-1} \mu$. This is the classical **tangent portfolio** — mean-variance optimal direction.

**Example 0.2.8.8 (Backprop through a linear layer).** In a neural network, a layer computes $\mathbf{h} = W\mathbf{x} + \mathbf{b}$ where $W \in \mathbb{R}^{m \times n}$, and subsequent layers produce a scalar loss $L$. Suppose the upstream gradient $\partial L/\partial \mathbf{h} \in \mathbb{R}^m$ is known. What are the gradients $\partial L/\partial W$ and $\partial L/\partial \mathbf{x}$?

$dL = \mathbf{g}^\top d\mathbf{h}$ where $\mathbf{g} = \partial L/\partial \mathbf{h}$. $d\mathbf{h} = (dW)\mathbf{x} + W d\mathbf{x}$. So:
- $dL|_{W} = \mathbf{g}^\top (dW)\mathbf{x} = \mathrm{tr}(\mathbf{x} \mathbf{g}^\top dW) \implies \partial L/\partial W = \mathbf{g} \mathbf{x}^\top$. (Outer product!)
- $dL|_{\mathbf{x}} = \mathbf{g}^\top W d\mathbf{x} \implies \partial L/\partial \mathbf{x} = W^\top \mathbf{g}$.

These are the backprop equations for a linear layer. Deep learning frameworks chain these automatically.

### Computational Implementation

```python
import numpy as np
from numpy.linalg import solve, slogdet, inv

# Verify gradient formulas numerically using finite differences
def numerical_gradient(f, x, eps=1e-6):
    """Compute gradient of scalar f: x -> R by central differences."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad

# Test Identity 2: grad(x^T A x) = (A + A^T) x
rng = np.random.default_rng(0)
A = rng.standard_normal((4, 4))
x = rng.standard_normal(4)
f = lambda x: x @ A @ x
analytic = (A + A.T) @ x
numeric = numerical_gradient(f, x.copy())
print("||analytic - numeric||:", np.linalg.norm(analytic - numeric))  # ~1e-9

# Test Identity 3: grad of ||X beta - y||^2
X = rng.standard_normal((20, 5))
y = rng.standard_normal(20)
beta = rng.standard_normal(5)
f = lambda b: np.sum((X @ b - y)**2)
analytic = 2 * X.T @ (X @ beta - y)
numeric = numerical_gradient(f, beta.copy())
print("||analytic - numeric||:", np.linalg.norm(analytic - numeric))

# Test Identity 5: grad of log det(A)
def logdet(A):
    sign, logabs = slogdet(A)
    if sign <= 0:
        return -np.inf
    return logabs

A_sym = A @ A.T + np.eye(4)          # symmetric PD
f = lambda S: logdet(S)
analytic = inv(A_sym).T               # = inv(A_sym) since symmetric
numeric = numerical_gradient(f, A_sym.copy())
print("||analytic - numeric||:", np.linalg.norm(analytic - numeric))

# Test MLE gradient: grad of log likelihood wrt Sigma
def gaussian_ll(Sigma, X):
    """Log-likelihood -n/2 log det Sigma - (n/2) tr(Sigma^-1 S) where S = X^T X / n."""
    n = X.shape[0]
    S = X.T @ X / n
    sign, ldet = slogdet(Sigma)
    if sign <= 0:
        return -np.inf
    return -0.5 * n * ldet - 0.5 * n * np.trace(solve(Sigma, S))

n = 1000
Sigma_true = np.array([[2., 0.3],
                       [0.3, 1.]])
X = rng.standard_normal((n, 2)) @ np.linalg.cholesky(Sigma_true).T
f = lambda S: gaussian_ll(S, X)
Sigma_test = np.eye(2) + 0.1 * rng.standard_normal((2, 2))
Sigma_test = (Sigma_test + Sigma_test.T) / 2 + 2 * np.eye(2)   # make PD
S = X.T @ X / n
analytic = -0.5 * n * (inv(Sigma_test) - inv(Sigma_test) @ S @ inv(Sigma_test))
numeric = numerical_gradient(f, Sigma_test.copy())
print("||analytic - numeric||:", np.linalg.norm(analytic - numeric))
# Note: symmetric matrix gradient has a subtlety — off-diagonals in analytic include
# both the (i,j) and (j,i) terms "summed". For strict comparison use vech(), but
# for MLE setting Sigma = argmax, the gradient being zero is what matters, and
# Sigma_hat = S makes both representations vanish simultaneously.

# MLE solution check
Sigma_hat = S
print("Gradient at Sigma_hat:", -0.5 * n * (inv(Sigma_hat) - inv(Sigma_hat) @ S @ inv(Sigma_hat)))
# ≈ 0

# Kronecker and vec
def vec(A): return A.T.reshape(-1)     # column-stacking
def unvec(v, shape): return v.reshape(shape[1], shape[0]).T

A_mat = rng.standard_normal((3, 4))
B_mat = rng.standard_normal((5, 2))
X_mat = rng.standard_normal((4, 5))
lhs = vec(A_mat @ X_mat @ B_mat)
rhs = np.kron(B_mat.T, A_mat) @ vec(X_mat)
print("vec(AXB) == (B^T kron A) vec(X):", np.allclose(lhs, rhs))
```

### [QUANT APPLICATION] — Gradient-Based Optimization, Adjoint Methods for Greeks, Backprop in Deep Nets

**(A) Gradient descent & Newton's method.** Gradient descent updates $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$. Newton's method uses $\mathbf{x}_{k+1} = \mathbf{x}_k - H_f^{-1} \nabla f$. Convergence analysis uses matrix calculus to bound Hessian changes. In portfolio optimization, quasi-Newton methods (L-BFGS) are the workhorse.

**(B) Adjoint methods for sensitivities (Greeks).** Given a derivative pricer $P(\theta)$ where $\theta$ is a high-dimensional parameter vector (e.g., entire forward-rate curve), computing each sensitivity $\partial P/\partial \theta_i$ by finite differences costs $d+1$ evaluations. The **adjoint** method computes *all* sensitivities in **one backward pass** at a cost comparable to a single forward pass — same complexity as forward pricing. This is matrix calculus applied to the pricer's computational graph, and it is what makes risk management tractable for exotic derivatives with thousands of parameters. Uses the same math as neural-network backpropagation.

**(C) Backprop.** A neural network is a chain of parameterized linear maps and nonlinearities. Backpropagation computes $\partial L/\partial \theta$ for all parameters $\theta$ by chaining the Jacobians of each layer. We derived the linear-layer gradient in Example 0.2.8.8. Autograd packages (PyTorch, JAX, TensorFlow) automate this. Quant applications: deep hedging, neural SDE calibration, reinforcement learning for execution.

**(D) Implicit differentiation.** Suppose $\mathbf{x}^*(\theta)$ is the solution to $g(\mathbf{x}, \theta) = \mathbf{0}$ (e.g., equilibrium, market-clearing, or KKT conditions). Implicit function theorem gives
$$\frac{\partial \mathbf{x}^*}{\partial \theta} = -\left(\frac{\partial g}{\partial \mathbf{x}}\right)^{-1} \frac{\partial g}{\partial \theta}.$$
This lets you differentiate through optimization solvers, KKT conditions, and equilibrium models — essential for calibration.

**(E) Fisher information and information geometry.** The Fisher information matrix is $\mathcal{I}(\theta) = -\mathbb{E}[\partial^2 \log p_\theta / \partial \theta \partial \theta^\top]$, a key quantity in MLE asymptotics ($\hat\theta \sim \mathcal{N}(\theta, \mathcal{I}(\theta)^{-1}/n)$). Computing it requires second-order matrix calculus.

### Exercises

#### ★ (Foundation)

**E0.2.8.1.** Using the differential, derive the gradient of $f(\mathbf{x}) = \|\mathbf{x}\|^4 = (\mathbf{x}^\top \mathbf{x})^2$.

**E0.2.8.2.** Show: $\partial \mathrm{tr}(A)/\partial A = I$, $\partial \mathrm{tr}(A^\top A)/\partial A = 2A$.

**E0.2.8.3.** Compute the Hessian of $f(\mathbf{x}) = \tfrac{1}{2}\mathbf{x}^\top Q \mathbf{x} + \mathbf{c}^\top \mathbf{x}$ for symmetric $Q$. Confirm it equals $Q$.

**E0.2.8.4.** Verify numerically that $\partial \log\det \Sigma/\partial \Sigma = \Sigma^{-1}$ when $\Sigma$ is symmetric. (For symmetric parameterizations, the off-diagonals require care: the gradient is symmetric, and the "vectorized" gradient uses the `vech` operator.)

#### ★★ (Intermediate)

**E0.2.8.5 (Lasso gradient).** Derive the gradient (or subgradient) of $L(\beta) = \|X\beta - \mathbf{y}\|^2 + \lambda \|\beta\|_1$ and write down the soft-thresholding coordinate-descent update.

*Hint.* $\partial |\beta_j|/\partial \beta_j = \mathrm{sgn}(\beta_j)$ for $\beta_j \neq 0$, the subgradient interval $[-1, 1]$ at $\beta_j = 0$.

**E0.2.8.6 (Softmax Jacobian).** For $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$, $p_i = e^{z_i}/\sum_j e^{z_j}$, compute the Jacobian $\partial \mathbf{p}/\partial \mathbf{z}$. Show it equals $\mathrm{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ (which is PSD and rank $n-1$).

**E0.2.8.7 (Cross-entropy gradient).** For $L = -\sum_i y_i \log p_i$ with $\mathbf{p} = \mathrm{softmax}(\mathbf{z})$ and $\mathbf{y}$ one-hot, show $\partial L/\partial \mathbf{z} = \mathbf{p} - \mathbf{y}$. This is the famously clean softmax + cross-entropy gradient.

**E0.2.8.8 (Stochastic gradient of SGD variance).** For quadratic $f(\mathbf{x}) = \tfrac{1}{2}\mathbf{x}^\top Q \mathbf{x}$, analyze the second moment of the SGD step $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \hat g_k$ where $\hat g_k = Q\mathbf{x}_k + \boldsymbol\xi_k$ with $\boldsymbol\xi_k \sim \mathcal{N}(0, \sigma^2 I)$. Derive the fixed point of the covariance of $\mathbf{x}_k$ and relate it to eigenvalues of $Q$.

#### ★★★ (Challenge)

**E0.2.8.9 (Magnus's theorem / vec of Jacobians).** Define the Jacobian of matrix-valued $F: \mathbb{R}^{p \times q} \to \mathbb{R}^{m \times n}$ as $DF = \partial \mathrm{vec}(F)/\partial \mathrm{vec}(X)^\top \in \mathbb{R}^{mn \times pq}$. Using the chain rule and vectorization identities, compute $DF$ for $F(X) = X^{-1}$ (with $p = q = m = n$).

**E0.2.8.10 (Differential of matrix exponential).** Show
$$d\,e^A = \int_0^1 e^{sA} (dA) e^{(1-s)A}\, ds.$$
(Useful in Lie group calculus, continuous-time optimization, and stochastic exponential martingales.)

*Hint.* Differentiate the power series termwise, using the identity $\sum_{k \geq 0} \frac{1}{k!}\sum_{j=0}^{k-1} A^j (dA) A^{k-1-j} = \int_0^1 e^{sA} (dA) e^{(1-s)A} ds$.

**E0.2.8.11 (Natural gradient).** Let $p_\theta(x)$ be a family of distributions parameterized by $\theta \in \Theta \subseteq \mathbb{R}^d$ and $\mathcal{I}(\theta)$ its Fisher information matrix. The **natural gradient** of a function $f(\theta)$ is $\tilde\nabla f(\theta) = \mathcal{I}(\theta)^{-1} \nabla f(\theta)$. Show that natural gradient descent is invariant under smooth reparameterizations of $\theta$ (unlike standard gradient descent).

*Relevance.* Amari's natural gradient underlies modern optimization for probabilistic models.

---

## Topic 0.2.9 — Norms on Vector Spaces and Matrices

### Motivation

A norm measures "size." In analysis, probability, and numerical work, choosing the right norm and understanding how operators behave under it is essential. For example:

- To say a sequence of random variables converges, we need a notion of "distance" — $L^p$ norms, total variation, Wasserstein, etc.
- To analyze iterative algorithms (gradient descent, power iteration, Krylov methods), we need to bound $\|A^k\|$; this depends on the spectral radius but also on the condition number.
- To define the matrix exponential $e^A = \sum A^k/k!$, we need to prove *convergence* of the series, which requires a submultiplicative norm.
- To regularize optimization problems, we use norm balls (ridge uses $\ell^2$, lasso uses $\ell^1$, nuclear norm for matrix completion).
- To talk about perturbation theory (how much an eigenvalue moves when the matrix is perturbed), we need operator norms.

In *infinite dimensions*, norms become more delicate: different norms need not be equivalent, and completeness becomes a substantive condition (Banach spaces). Here we treat the finite-dimensional case rigorously, paving the way for the measure-theoretic $L^p$ spaces we meet in Subject 1.

### Prerequisites

Real analysis basics (limits, continuity), inner product spaces (Topic 0.2.3), SVD (Topic 0.2.6).

### Definitions and Examples

**Definition 0.2.9.1 (Norm).** A **norm** on a real (or complex) vector space $V$ is a function $\|\cdot\|: V \to [0, \infty)$ satisfying:

(N1) **Positive definiteness:** $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$.

(N2) **Absolute homogeneity:** $\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$ for all scalars $\alpha$.

(N3) **Triangle inequality:** $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$.

A **seminorm** drops (N1) — we allow $\|\mathbf{x}\| = 0$ for $\mathbf{x} \neq \mathbf{0}$.

**Definition 0.2.9.2 ($p$-norms on $\mathbb{R}^n$).** For $1 \leq p < \infty$:
$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}.$$
For $p = \infty$:
$$\|\mathbf{x}\|_\infty = \max_i |x_i|.$$

**Theorem 0.2.9.3 (The $\ell^p$ norms are norms).** For every $p \in [1, \infty]$, $\|\cdot\|_p$ satisfies (N1)–(N3).

*Proof.* (N1) and (N2) are clear by inspection. (N3) for $p = 1, \infty$ is elementary (triangle inequality for absolute values).

For $p \in (1, \infty)$, (N3) is the **Minkowski inequality**:
$$\left(\sum_i |x_i + y_i|^p\right)^{1/p} \leq \left(\sum_i |x_i|^p\right)^{1/p} + \left(\sum_i |y_i|^p\right)^{1/p}.$$

*Proof of Minkowski.* Need **Hölder's inequality** first: for $1 < p, q < \infty$ with $1/p + 1/q = 1$,
$$\sum_i |a_i b_i| \leq \|\mathbf{a}\|_p \|\mathbf{b}\|_q.$$

*Proof of Hölder.* By homogeneity we may assume $\|\mathbf{a}\|_p = \|\mathbf{b}\|_q = 1$. Young's inequality: $\alpha\beta \leq \alpha^p/p + \beta^q/q$ for $\alpha, \beta \geq 0$ (since $\log$ is concave: $\log(\alpha^p/p + \beta^q/q) \geq (1/p)\log \alpha^p + (1/q)\log\beta^q = \log\alpha + \log\beta = \log(\alpha\beta)$). Apply with $\alpha = |a_i|$, $\beta = |b_i|$:
$$\sum_i |a_i b_i| \leq \sum_i \left(\frac{|a_i|^p}{p} + \frac{|b_i|^q}{q}\right) = \frac{\|\mathbf{a}\|_p^p}{p} + \frac{\|\mathbf{b}\|_q^q}{q} = \frac{1}{p} + \frac{1}{q} = 1.$$ $\square$

*Proof of Minkowski via Hölder.* $|x_i + y_i|^p = |x_i + y_i|\cdot |x_i + y_i|^{p-1} \leq (|x_i| + |y_i|)|x_i + y_i|^{p-1}$. Sum:
$$\sum_i |x_i + y_i|^p \leq \sum_i |x_i||x_i + y_i|^{p-1} + \sum_i |y_i||x_i + y_i|^{p-1}.$$
Apply Hölder to each term with exponent $p$ on the first factor and $q = p/(p-1)$ on the second:
$$\sum_i |x_i||x_i + y_i|^{p-1} \leq \|\mathbf{x}\|_p \left(\sum_i |x_i + y_i|^{(p-1)q}\right)^{1/q} = \|\mathbf{x}\|_p \|\mathbf{x} + \mathbf{y}\|_p^{p-1}.$$
Similarly for the $\mathbf{y}$ term. Summing and dividing by $\|\mathbf{x} + \mathbf{y}\|_p^{p-1}$ (assume nonzero; else trivial):
$$\|\mathbf{x} + \mathbf{y}\|_p \leq \|\mathbf{x}\|_p + \|\mathbf{y}\|_p.\ \square$$

### Equivalence of Norms (Finite Dimensions)

**Definition 0.2.9.4.** Two norms $\|\cdot\|_a$ and $\|\cdot\|_b$ on $V$ are **equivalent** if there exist $0 < c \leq C < \infty$ with
$$c \|\mathbf{x}\|_a \leq \|\mathbf{x}\|_b \leq C \|\mathbf{x}\|_a \quad \text{for all } \mathbf{x} \in V.$$

Equivalent norms induce the same topology (open sets, convergent sequences, continuity).

**Theorem 0.2.9.5 (All norms on $\mathbb{R}^n$ are equivalent).** For any two norms $\|\cdot\|_a, \|\cdot\|_b$ on $\mathbb{R}^n$, they are equivalent.

*Proof.* By transitivity it suffices to show every norm is equivalent to $\|\cdot\|_\infty$.

Let $\mathbf{e}_1, \ldots, \mathbf{e}_n$ be the standard basis and $M = \max_i \|\mathbf{e}_i\|_a$. For any $\mathbf{x} = \sum x_i \mathbf{e}_i$:
$$\|\mathbf{x}\|_a \leq \sum_i |x_i| \|\mathbf{e}_i\|_a \leq n M \|\mathbf{x}\|_\infty. \qquad (*)$$

For the other direction, consider $\|\cdot\|_a$ as a function on the unit sphere $S = \{\mathbf{x} : \|\mathbf{x}\|_\infty = 1\}$ (compact in $\mathbb{R}^n$, since closed and bounded).

*Claim:* $\|\cdot\|_a$ is continuous as a function from $(\mathbb{R}^n, \|\cdot\|_\infty)$ to $\mathbb{R}$.

*Proof of claim.* $|\|\mathbf{x}\|_a - \|\mathbf{y}\|_a| \leq \|\mathbf{x} - \mathbf{y}\|_a \leq nM \|\mathbf{x} - \mathbf{y}\|_\infty$ by reverse triangle inequality and $(*)$. So $\|\cdot\|_a$ is Lipschitz, hence continuous. $\square$ (claim)

By the extreme value theorem, $\|\cdot\|_a$ attains its minimum $m$ and maximum $M'$ on the compact $S$. Since $\|\mathbf{x}\|_a = 0 \iff \mathbf{x} = 0 \notin S$, $m > 0$. By homogeneity, for any $\mathbf{x} \neq 0$, $\mathbf{x}/\|\mathbf{x}\|_\infty \in S$, so $\|\mathbf{x}/\|\mathbf{x}\|_\infty\|_a \in [m, M']$, giving
$$m \|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_a \leq M' \|\mathbf{x}\|_\infty.$$

So all norms are sandwiched between constants times $\|\cdot\|_\infty$. Combined with the same for any other norm, all norms are equivalent. $\square$

**Remark.** In infinite dimensions, this theorem fails: on $C([0,1])$, the $L^1$ and $L^\infty$ norms are not equivalent (one dominates the other but not conversely). This is a key technicality in functional analysis.

### Induced Matrix (Operator) Norms

**Definition 0.2.9.6.** Given norms $\|\cdot\|_{\mathrm{src}}$ on $\mathbb{R}^n$ and $\|\cdot\|_{\mathrm{tgt}}$ on $\mathbb{R}^m$, the **induced (operator) norm** of $A \in \mathbb{R}^{m \times n}$ is
$$\|A\|_{\mathrm{op}} = \sup_{\mathbf{x} \neq 0} \frac{\|A\mathbf{x}\|_{\mathrm{tgt}}}{\|\mathbf{x}\|_{\mathrm{src}}} = \sup_{\|\mathbf{x}\|_{\mathrm{src}} = 1} \|A\mathbf{x}\|_{\mathrm{tgt}}.$$

The supremum is attained (unit sphere compact in finite dim, $\|A\mathbf{x}\|_{\mathrm{tgt}}$ continuous). When source and target norms are both $\ell^p$, we write $\|A\|_p$.

**Theorem 0.2.9.7 (Formulas for $\ell^p$ operator norms).**
- $\|A\|_1 = \max_j \sum_i |A_{ij}|$ (maximum column sum).
- $\|A\|_\infty = \max_i \sum_j |A_{ij}|$ (maximum row sum).
- $\|A\|_2 = \sigma_1(A)$ (largest singular value).

*Proof.*

($\|A\|_1$): $\|A\mathbf{x}\|_1 = \sum_i |\sum_j A_{ij} x_j| \leq \sum_i \sum_j |A_{ij}||x_j| = \sum_j |x_j| \sum_i |A_{ij}| \leq (\max_j c_j) \|\mathbf{x}\|_1$ where $c_j = \sum_i |A_{ij}|$. So $\|A\|_1 \leq \max_j c_j$. Equality achieved at $\mathbf{x} = \mathbf{e}_{j^*}$ where $j^*$ maximizes $c_j$: $\|A \mathbf{e}_{j^*}\|_1 = c_{j^*} = \max_j c_j$.

($\|A\|_\infty$): $\|A\mathbf{x}\|_\infty = \max_i |\sum_j A_{ij} x_j| \leq \max_i \sum_j |A_{ij}| \|\mathbf{x}\|_\infty = (\max_i r_i)\|\mathbf{x}\|_\infty$. Equality at a suitable $\mathbf{x}$ with $x_j = \mathrm{sgn}(A_{i^* j})$ where $i^*$ maximizes $r_i$.

($\|A\|_2$): $\|A\mathbf{x}\|_2^2 = \mathbf{x}^\top A^\top A \mathbf{x}$. By Rayleigh, max over unit $\mathbf{x}$ is $\lambda_{\max}(A^\top A) = \sigma_1^2$. So $\|A\|_2 = \sigma_1$. $\square$

### Submultiplicativity

**Theorem 0.2.9.8 (Submultiplicativity of operator norms).** For induced operator norms, $\|AB\| \leq \|A\|\|B\|$.

*Proof.* For any $\mathbf{x}$: $\|AB\mathbf{x}\| \leq \|A\|\|B\mathbf{x}\| \leq \|A\|\|B\|\|\mathbf{x}\|$. Divide by $\|\mathbf{x}\|$ and take sup. $\square$

**Corollary.** $\|A^k\| \leq \|A\|^k$.

### Frobenius Norm and Trace Inner Product

**Definition 0.2.9.9.** The **Frobenius norm** of $A \in \mathbb{R}^{m \times n}$ is
$$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\mathrm{tr}(A^\top A)} = \sqrt{\sum_i \sigma_i^2}.$$

It is the norm induced by the **trace (Hilbert–Schmidt) inner product** $\langle A, B\rangle_F = \mathrm{tr}(A^\top B)$.

**Submultiplicativity:** $\|AB\|_F \leq \|A\|_F \|B\|_F$. *Proof.* Columns of $AB$: $(AB)_{\cdot j} = A B_{\cdot j}$, so $\|AB\|_F^2 = \sum_j \|A B_{\cdot j}\|_2^2 \leq \sum_j \|A\|_2^2 \|B_{\cdot j}\|_2^2 = \|A\|_2^2 \|B\|_F^2 \leq \|A\|_F^2 \|B\|_F^2$ (using $\|A\|_2 \leq \|A\|_F$). $\square$

**Comparison to operator norm:** $\|A\|_2 \leq \|A\|_F \leq \sqrt{\mathrm{rank}(A)} \|A\|_2$. The first is immediate ($\sigma_1 \leq \sqrt{\sum \sigma_i^2}$); the second since $\sum_{i \leq r} \sigma_i^2 \leq r \sigma_1^2$.

**Unitary invariance:** $\|UAV\|_F = \|A\|_F$ for orthogonal $U, V$. Similarly $\|UAV\|_2 = \|A\|_2$. *Proof.* For Frobenius: $\mathrm{tr}((UAV)^\top UAV) = \mathrm{tr}(V^\top A^\top U^\top U A V) = \mathrm{tr}(V^\top A^\top A V) = \mathrm{tr}(A^\top A)$ (cyclic). For operator: singular values are invariant under orthogonal transformations. $\square$

### Spectral Radius and the Gelfand Formula

**Definition 0.2.9.10.** The **spectral radius** of $A \in \mathbb{C}^{n \times n}$ is $\rho(A) = \max_{\lambda \in \sigma(A)} |\lambda|$.

**Theorem 0.2.9.11 (Basic bound).** For any submultiplicative norm $\|\cdot\|$, $\rho(A) \leq \|A\|$.

*Proof.* Let $\lambda$ be an eigenvalue with maximum $|\lambda|$ and $\mathbf{v}$ a corresponding eigenvector. For the *vector norm* $\|\cdot\|_v$ that induces the matrix norm $\|\cdot\|$: $|\lambda|\|\mathbf{v}\|_v = \|\lambda \mathbf{v}\|_v = \|A\mathbf{v}\|_v \leq \|A\|\|\mathbf{v}\|_v$. Divide: $|\lambda| \leq \|A\|$.

For norms not induced by a vector norm (e.g., Frobenius without unit-norm convention issues, but actually Frobenius is induced), the argument is: $\|A^k\| \geq |\text{trace of }A^k / n^?|$ — subtle. A clean route: choose a vector norm $\|\cdot\|_v$ that realizes $\rho(A)$ within $\varepsilon$, show its induced matrix norm gives the bound. Formally, one shows there exists a vector norm with $\|A\|_{\mathrm{ind}} \leq \rho(A) + \varepsilon$ (see Theorem 0.2.9.13). $\square$

**Theorem 0.2.9.12 (Gelfand's spectral radius formula).** For any matrix norm and any $A \in \mathbb{C}^{n\times n}$,
$$\rho(A) = \lim_{k \to \infty} \|A^k\|^{1/k}.$$

*Proof sketch (assumes Jordan form for clean argument; a full elementary proof is in Horn–Johnson).*

Write $A = PJP^{-1}$ with $J$ Jordan. Then $A^k = P J^k P^{-1}$ and $\|A^k\| \leq \|P\|\|P^{-1}\|\|J^k\|$. By bounding the entries of $J^k$ (each block's entries behave like $\binom{k}{j}\lambda^{k-j}$), $\|J^k\| \leq C \cdot k^{n-1} \rho(A)^k$ for a constant $C$.

So $\|A^k\|^{1/k} \leq (\|P\|\|P^{-1}\|C)^{1/k} \cdot k^{(n-1)/k} \rho(A) \to \rho(A)$ as $k \to \infty$.

For the other direction, pick an eigenvalue $\lambda$ with $|\lambda| = \rho(A)$ and an eigenvector $\mathbf{v}$. Consider any vector norm $\|\cdot\|_v$. Then $\|A^k \mathbf{v}\|_v = \rho(A)^k \|\mathbf{v}\|_v$, so $\|A^k\| \geq \rho(A)^k \|\mathbf{v}\|_v / \|\mathbf{v}\|_v = \rho(A)^k$ (for the operator norm induced by $\|\cdot\|_v$). Thus $\|A^k\|^{1/k} \geq \rho(A)$. Combined, $\|A^k\|^{1/k} \to \rho(A)$.

For a *general* matrix norm (not necessarily operator-induced): by equivalence of norms, any matrix norm is sandwiched by operator norms, so the limit is the same. $\square$

**Theorem 0.2.9.13 (Existence of a close-to-spectral norm).** For every $\varepsilon > 0$ and $A \in \mathbb{C}^{n \times n}$, there exists a vector norm $\|\cdot\|_v$ whose induced operator norm satisfies $\rho(A) \leq \|A\|_{\mathrm{op}} \leq \rho(A) + \varepsilon$.

*Proof (sketch).* Schur triangularize $A = UTU^*$. For a diagonal weight matrix $D_s = \mathrm{diag}(1, s, s^2, \ldots, s^{n-1})$ with small $s > 0$, $D_s^{-1} T D_s$ has its off-diagonal upper entries scaled by factors $s^{i-j}$ for $i < j$ (so very small for small $s$). Define $\|\mathbf{x}\|_v = \|D_s^{-1} U^* \mathbf{x}\|_\infty$. The induced operator norm of $A$ equals the max row sum of $D_s^{-1} T D_s$, which for small $s$ is close to $\max_i |T_{ii}| = \rho(A)$. $\square$

### Condition Number

**Definition 0.2.9.14.** For invertible $A$, the **condition number** (with respect to a norm $\|\cdot\|$) is $\kappa(A) = \|A\|\|A^{-1}\|$.

In the 2-norm: $\kappa_2(A) = \sigma_1(A)/\sigma_n(A)$ (ratio of largest to smallest singular value).

**Interpretation.** $\kappa(A)$ is the factor by which errors in $\mathbf{b}$ propagate to errors in the solution of $A\mathbf{x} = \mathbf{b}$. If $A\mathbf{x} = \mathbf{b}$ and $A\tilde{\mathbf{x}} = \mathbf{b} + \delta\mathbf{b}$:
$$\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}.$$

*Proof.* $\tilde{\mathbf{x}} - \mathbf{x} = A^{-1}\delta\mathbf{b}$, so $\|\tilde{\mathbf{x}} - \mathbf{x}\| \leq \|A^{-1}\|\|\delta\mathbf{b}\|$. And $\|\mathbf{b}\| = \|A\mathbf{x}\| \leq \|A\|\|\mathbf{x}\|$. Divide. $\square$

### Convergence of Matrix Series

**Theorem 0.2.9.15 (Neumann series).** If $\|A\| < 1$ in some submultiplicative norm, then $I - A$ is invertible and
$$(I - A)^{-1} = \sum_{k=0}^\infty A^k, \qquad \|(I - A)^{-1}\| \leq \frac{1}{1 - \|A\|}.$$

*Proof.* $\|A^k\| \leq \|A\|^k$ (submultiplicativity), so the partial sums $S_N = \sum_{k=0}^N A^k$ form a Cauchy sequence in $M_n$ (complete in any matrix norm — finite-dim, so all norms are equivalent to a complete one). Hence the limit $S = \sum_k A^k$ exists. Computing: $(I - A) S_N = I - A^{N+1} \to I$. So $S = (I - A)^{-1}$.

$\|S\| \leq \sum_{k=0}^\infty \|A^k\| \leq \sum_k \|A\|^k = 1/(1 - \|A\|)$. $\square$

**Theorem 0.2.9.16 (Matrix exponential).** For any $A \in M_n(\mathbb{C})$, the series $e^A = \sum_k A^k/k!$ converges in any matrix norm, and $\|e^A\| \leq e^{\|A\|}$.

*Proof.* $\|A^k/k!\| \leq \|A\|^k / k!$, and $\sum_k \|A\|^k/k! = e^{\|A\|} < \infty$. Absolute convergence in finite-dim implies convergence. $\square$

### Worked Examples

**Example 0.2.9.17 (Comparing $\ell^p$ balls).** Unit balls $\{\mathbf{x} : \|\mathbf{x}\|_p \leq 1\}$ in $\mathbb{R}^2$:
- $p = 1$: diamond.
- $p = 2$: circle.
- $p = \infty$: square.
- $p \to \infty$ shrinks to unit square; $p = 0.5$ (not a norm — fails triangle) gives a star.

The numeric inequalities: $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1$ and $\|\mathbf{x}\|_1 \leq \sqrt n \|\mathbf{x}\|_2 \leq n \|\mathbf{x}\|_\infty$.

**Example 0.2.9.18 (Condition number and numerical issues).** Hilbert matrix $H_n$ with $H_{ij} = 1/(i + j - 1)$. $H_5$ has $\kappa_2(H_5) \approx 4.77 \times 10^5$. Solving $H_5 \mathbf{x} = \mathbf{b}$ with single precision loses $\sim \log_{10}\kappa_2 \approx 5.7$ digits — almost half of single precision!

This is why in least squares fitting of polynomials over wide ranges, people use orthogonal polynomials (Legendre, Chebyshev) rather than raw monomials — the Vandermonde matrix is notoriously ill-conditioned.

**Example 0.2.9.19 (Power iteration convergence).** For $A$ with $|\lambda_1| > |\lambda_2|$: $\mathbf{x}_{k+1} = A\mathbf{x}_k/\|A\mathbf{x}_k\|$ converges to $\mathbf{q}_1$ with rate $|\lambda_2/\lambda_1|^k$. Slower when $\kappa_2(A)$ is small (eigenvalues cluster).

### Computational Implementation

```python
import numpy as np
from numpy.linalg import norm, cond, matrix_power

rng = np.random.default_rng(1)
A = rng.standard_normal((4, 4))
x = rng.standard_normal(4)

# Various norms
print(f"||x||_1 = {norm(x, 1):.4f}")
print(f"||x||_2 = {norm(x, 2):.4f}")
print(f"||x||_inf = {norm(x, np.inf):.4f}")
print(f"||A||_1 = {norm(A, 1):.4f}")             # max column sum
print(f"||A||_2 = {norm(A, 2):.4f}")             # largest singular value
print(f"||A||_inf = {norm(A, np.inf):.4f}")      # max row sum
print(f"||A||_F = {norm(A, 'fro'):.4f}")
print(f"||A||_* (nuclear) = {np.sum(np.linalg.svd(A, compute_uv=False)):.4f}")

# Condition number
print(f"cond_2(A) = {cond(A, 2):.4f}")
print(f"cond_2(Hilbert 5) = {cond(1/(np.add.outer(np.arange(5), np.arange(5))+1), 2):.2e}")

# Equivalence of norms: compute constants
n = 4
xs = rng.standard_normal((n, 10000))
ratios_1_inf = norm(xs, 1, axis=0) / norm(xs, np.inf, axis=0)
print(f"||x||_1/||x||_inf: min={ratios_1_inf.min():.4f}, max={ratios_1_inf.max():.4f}, theory=[1, {n}]")
ratios_2_inf = norm(xs, 2, axis=0) / norm(xs, np.inf, axis=0)
print(f"||x||_2/||x||_inf: min={ratios_2_inf.min():.4f}, max={ratios_2_inf.max():.4f}, theory=[1, sqrt({n})={np.sqrt(n):.4f}]")

# Gelfand's formula: spectral radius = lim ||A^k||^(1/k)
A_small = rng.standard_normal((5, 5))
rho = np.max(np.abs(np.linalg.eigvals(A_small)))
for k in [1, 5, 10, 50, 100]:
    A_k = matrix_power(A_small, k)
    approx = norm(A_k, 2)**(1/k)
    print(f"k={k:4d}  ||A^k||_2^(1/k) = {approx:.6f}  rho(A) = {rho:.6f}")

# Neumann series: compute (I - A)^(-1) via series when ||A|| < 1
A_small = 0.5 * rng.standard_normal((3, 3)) / 3   # make ||A|| < 1
assert norm(A_small, 2) < 1
partial = np.eye(3)
sum_so_far = np.eye(3)
for k in range(1, 20):
    partial = partial @ A_small
    sum_so_far = sum_so_far + partial
exact = np.linalg.inv(np.eye(3) - A_small)
print("||series - exact||:", norm(sum_so_far - exact))

# Matrix exponential convergence
from scipy.linalg import expm
A2 = rng.standard_normal((4, 4))
exp_series = np.eye(4)
term = np.eye(4)
for k in range(1, 40):
    term = term @ A2 / k
    exp_series = exp_series + term
exp_ref = expm(A2)
print("||series - scipy expm||:", norm(exp_series - exp_ref))
```

### [QUANT APPLICATION] — Numerical Stability, Regularization Norm Balls, Algorithm Convergence Rates

**(A) Condition numbers in covariance inversion.** The mean-variance optimal weights are $\mathbf{w}^* \propto \Sigma^{-1}\mu$. If $\hat\Sigma$ is ill-conditioned ($\kappa_2(\hat\Sigma) \gg 1$), small estimation errors in $\hat\Sigma$ or $\hat\mu$ produce wildly unstable portfolios. This is why **shrinkage** (Ledoit–Wolf) and **factor models** (which force low-rank + diagonal structure with good conditioning) dominate practice.

**(B) Regularization as norm balls.** Ridge regression $\min \|X\beta - \mathbf{y}\|^2 + \lambda\|\beta\|_2^2$ is equivalent to $\min \|X\beta - \mathbf{y}\|^2$ s.t. $\|\beta\|_2 \leq t$. Lasso uses $\|\beta\|_1 \leq t$ (corners of the $\ell^1$ ball promote sparsity). Elastic net combines both. Nuclear norm $\|X\|_*$ promotes low-rank. The *geometry* of the norm ball determines the structural inductive bias.

**(C) Gradient descent convergence rates.** For quadratic $f(\mathbf{x}) = \tfrac{1}{2}\mathbf{x}^\top Q\mathbf{x}$ with $0 \prec mI \preceq Q \preceq MI$, gradient descent with step size $\eta = 1/M$ converges:
$$\|\mathbf{x}_k - \mathbf{x}^*\|_2 \leq \left(1 - \frac{m}{M}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|_2.$$
The rate depends on the condition number $\kappa = M/m$. Preconditioning = change norms to make $\kappa$ smaller.

**(D) Operator norm for robust optimization.** Robust portfolio optimization might ask: $\min_\mathbf{w} \max_{\|\delta\Sigma\|_2 \leq \rho} \mathbf{w}^\top (\Sigma + \delta\Sigma)\mathbf{w}$. The inner max depends on operator norms of perturbation; solving gives a tractable reformulation via SDP.

**(E) Spectral radius and stability.** In dynamical systems $\mathbf{x}_{t+1} = A\mathbf{x}_t + \boldsymbol\varepsilon_t$, stationarity requires $\rho(A) < 1$. For numerical stability of iterative schemes (explicit finite-difference PDE solvers), CFL conditions are spectral-radius constraints.

**(F) Total variation distance and coupling.** For probability measures, total variation $\|\mu - \nu\|_{TV} = \frac{1}{2}\sum|p_i - q_i|$ is an $\ell^1$ norm on the density differences. Spectral gap of a Markov chain times time step upper-bounds TV distance to stationary — this is the norm-driven analysis of mixing times, central to MCMC pricing.

### Exercises

#### ★ (Foundation)

**E0.2.9.1.** Verify: $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt n \|\mathbf{x}\|_\infty$. When does equality hold on each side?

**E0.2.9.2.** For $A \in \mathbb{R}^{2\times 2}$ diagonal with entries $\lambda_1, \lambda_2$, compute $\|A\|_1, \|A\|_2, \|A\|_\infty, \|A\|_F$.

**E0.2.9.3.** Show: $\|A\mathbf{x}\|_p \leq \|A\|_p \|\mathbf{x}\|_p$ (definition of induced norm).

**E0.2.9.4.** Prove the Frobenius norm equals $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$ (using SVD).

**E0.2.9.5.** Show: condition number of a unitary/orthogonal matrix is $1$ in 2-norm. Why does this make unitary transformations "perfect" numerically?

#### ★★ (Intermediate)

**E0.2.9.6 (Hölder's conjugate in operator norms).** For $A \in \mathbb{R}^{m \times n}$, show $\|A\|_p = \|A^\top\|_q$ where $1/p + 1/q = 1$. In particular $\|A\|_1 = \|A^\top\|_\infty$, $\|A\|_2 = \|A^\top\|_2$.

*Hint.* Use the duality $\|A\mathbf{x}\|_p = \sup_{\|\mathbf{y}\|_q = 1} \mathbf{y}^\top A \mathbf{x}$.

**E0.2.9.7 (Von Neumann's trace inequality).** For $A, B \in \mathbb{R}^{n \times n}$, prove
$$|\mathrm{tr}(AB)| \leq \sum_i \sigma_i(A) \sigma_i(B).$$
Deduce: $|\mathrm{tr}(AB)| \leq \|A\|_F \|B\|_F$ (Cauchy–Schwarz for trace inner product, which also follows directly from Cauchy–Schwarz).

*Hint.* Use SVD on both matrices; reduce to doubly stochastic optimization.

**E0.2.9.8 (Spectral radius sub-additivity fails).** Give an example of matrices $A, B$ with $\rho(A + B) > \rho(A) + \rho(B)$, showing $\rho$ is *not* a norm.

*Hint.* Take nilpotent $A, B$ with $\rho = 0$ each but $AB \neq 0$. Try $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$.

**E0.2.9.9 (Condition number & perturbation of inverse).** Prove: if $A$ is invertible and $\|\delta A\| < 1/\|A^{-1}\|$, then $A + \delta A$ is invertible and
$$\|(A + \delta A)^{-1} - A^{-1}\| \leq \frac{\|A^{-1}\|^2 \|\delta A\|}{1 - \|A^{-1}\|\|\delta A\|}.$$

*Hint.* Write $A + \delta A = A(I + A^{-1}\delta A)$ and use Neumann series.

#### ★★★ (Challenge)

**E0.2.9.10 (Equivalence constants for $\ell^p$).** For $1 \leq p \leq q \leq \infty$ on $\mathbb{R}^n$:
$$\|\mathbf{x}\|_q \leq \|\mathbf{x}\|_p \leq n^{1/p - 1/q} \|\mathbf{x}\|_q.$$
Prove both inequalities and identify equality cases.

*Hint.* The first uses concavity (power-mean inequality); the second uses Hölder.

**E0.2.9.11 (Lyapunov function for stability).** Prove: $\rho(A) < 1$ iff there exists $P \succ 0$ with $A^\top P A - P \prec 0$ (discrete Lyapunov).

*Hint.* ($\Leftarrow$) Use $\mathbf{x}_k^\top P \mathbf{x}_k$ as a decreasing "energy." ($\Rightarrow$) Construct $P = \sum_{k=0}^\infty (A^\top)^k A^k$ (converges since $\rho(A) < 1$ by Gelfand).

**E0.2.9.12 (Horn–Schur majorization).** Let $A \in \mathbb{C}^{n\times n}$ with eigenvalues $\lambda_i$ and singular values $\sigma_i$, both sorted in decreasing order of modulus. Prove the majorization: for every $k$,
$$\sum_{i=1}^k |\lambda_i| \leq \sum_{i=1}^k \sigma_i, \qquad \prod_{i=1}^k |\lambda_i| \leq \prod_{i=1}^k \sigma_i.$$

*Hint.* Use Schur triangularization for the sum; determinants of restrictions for the product.

**E0.2.9.13 (Matrix norm inequalities for convergence).** Show that the matrix exponential map $A \mapsto e^A$ is Lipschitz on bounded sets: for any $R > 0$,
$$\|e^A - e^B\| \leq e^R \|A - B\| \quad \text{when } \|A\|, \|B\| \leq R.$$

*Hint.* Use the integral representation $e^A - e^B = \int_0^1 e^{sA}(A - B) e^{(1-s)B}\,ds$ or induction on powers.

---

## Module 0.2 Summary

We have rebuilt linear algebra from axioms, keeping JEE-level computational fluency but adding the structural machinery that quant work demands.

**The arc of the module:**

Topics 0.2.1–0.2.3 laid the foundations. **Vector spaces** gave us the axiomatic language: basis, dimension, subspace, direct sum. **Linear transformations** gave us rank-nullity and the matrix of a map — the bridge between abstract algebra and computation. **Inner product spaces** added geometry: orthogonal projection (the cleanest version of "best approximation"), Gram–Schmidt, and the key fact that $V = W \oplus W^\perp$ in finite-dimensional inner product spaces. These three topics collectively give the workspace in which everything else happens.

Topics 0.2.4–0.2.6 were the decomposition theorems. **Eigendecomposition** revealed the hidden simple structure: linear maps that preserve certain directions can be diagonalized (when enough eigenvectors exist). **The Spectral Theorem** upgraded this for symmetric/Hermitian/normal matrices: unitary diagonalization with real eigenvalues — the mathematical heart of PCA, factor models, quadratic forms, and Mercer kernels. **SVD** extended decomposition to arbitrary rectangular matrices, connecting to rank, the pseudoinverse, low-rank approximation (Eckart–Young), and the four fundamental subspaces. Three decompositions, one structural story: every linear map is a rotate–stretch–rotate.

Topics 0.2.7–0.2.9 brought us to the analytic/computational layer. **Positive definiteness** characterized the "nonnegative reals" analogue of matrices — the world of covariance, kernels, Hessians, and Cholesky. **Matrix calculus** gave us the tools to differentiate through all this cleanly: gradients of quadratic forms, $\log\det$, matrix inverse, adjoint/backprop identities. **Norms** completed the picture: the right notion of size on both vectors and operators, the spectral radius, condition number, and the convergence machinery for Neumann series, matrix exponentials, and iterative algorithms.

**Why this matters for Subject 1 (Measure Theory):**
- Inner product and norm structure generalize to $L^p$ spaces. The geometric intuitions you built here (projection, orthogonal decomposition, Cauchy–Schwarz) extend almost verbatim to $L^2(\Omega, P)$ — and random variables become vectors, covariance becomes an inner product, regression becomes orthogonal projection.
- The spectral theorem for compact symmetric operators on a Hilbert space is the infinite-dimensional generalization of Topic 0.2.5, central to kernel methods and functional data analysis.
- Convergence of sequences (Topic 0.2.9) is the prototype for almost-sure convergence, $L^p$ convergence, and convergence in distribution in Subject 1.

**Why this matters for Subject 2 (Probability Theory):**
- Covariance matrices (always PSD) are the algebraic form of second-moment information.
- Multivariate Gaussians are entirely characterized by mean and covariance; Cholesky gives you a sampling algorithm.
- The law of large numbers and CLT are statements about convergence in various norms.

**Why this matters for Subject 5 (Stochastic Calculus):**
- Itô's formula uses matrix calculus for multi-dimensional semimartingales.
- Quadratic variation of multi-dim Brownian motion is a covariance-matrix-valued process.
- Matrix exponentials appear in closed-form solutions of linear SDEs (Ornstein–Uhlenbeck and Gaussian processes).

**Where we punted:**
- Full Jordan form proof — we used its existence (in 0.2.4) without proving it from modules over a PID; see Dummit–Foote or Hoffman–Kunze.
- Hoffman–Wielandt and Davis–Kahan theorems — stated but proofs deferred to advanced matrix analysis (Horn–Johnson).
- Duals and tensor products — not needed for the current track; will appear in functional analysis (Subject 9).
- Infinite-dimensional issues (Banach spaces, closed operators) — deferred to Subject 9.

### Forward Pointers

- **Module 0.3 (Multivariable Calculus Formalized):** Gradients, Jacobians, Hessians rigorously via the Fréchet derivative on normed spaces. Implicit function theorem, inverse function theorem. Lagrangian optimization with constraints.

- **Module 0.4 (Real Analysis / Metric Spaces):** Completeness, compactness, uniform convergence. Sets up the topology needed for measure theory.

- **Module 0.5 (Complex Analysis):** Holomorphic functions, Cauchy's theorem, residues. Needed for characteristic functions (Subject 2) and Fourier-based option pricing.

- **Subject 1 (Measure Theory):** Sigma-algebras, Lebesgue integral, $L^p$ spaces. Generalization of all the finite-dimensional geometry we built.

- **Subject 2 (Probability Theory):** Built atop measure theory. Uses linear algebra for multivariate Gaussians, covariance structure, and characteristic functions.

- **Subject 4 (Stochastic Processes) and Subject 5 (Stochastic Calculus):** Brownian motion, Itô integration, SDEs. Linear algebra lives inside every multi-dimensional SDE calculation.

*Next up: Module 0.3 — Multivariable Calculus Formalized.*


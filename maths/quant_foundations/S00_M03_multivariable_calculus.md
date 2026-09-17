# Module 0.3 — Multivariable Calculus Formalized

> *Subject 0 — Prerequisites, Module 3 of 5.*
>
> *Prerequisites:* Module 0.1 (Logic and Proof), Module 0.2 (Linear Algebra).
>
> *What this module delivers:* The rigorous scaffolding of multivariable calculus — differentiability in normed spaces, the chain rule, Taylor's theorem, the Inverse and Implicit Function Theorems with complete proofs, constrained optimization (Lagrange and KKT), and the multivariate change-of-variables formula. This is the language of optimization, econometrics, stochastic calculus, and every "take the derivative" argument you will ever make in finance.
>
> *What makes this different from an engineering multivariable calc course:* We state and prove everything. The IFT is proved via Banach's fixed-point theorem, not waved at. KKT is derived from scratch via the Lagrangian. Change of variables gets a real (though sketched) justification via the area formula. By the end, you should be able to write down *and justify* the derivative of any composite map involving vectors, matrices, and constraints.

---

## Table of Contents

- **Topic 0.3.1** — Fréchet and Gâteaux Derivatives
- **Topic 0.3.2** — Chain Rule and Taylor's Theorem
- **Topic 0.3.3** — Inverse Function Theorem
- **Topic 0.3.4** — Implicit Function Theorem
- **Topic 0.3.5** — Constrained Optimization: Lagrange and KKT
- **Topic 0.3.6** — Change of Variables in Multiple Integrals

---

## Topic 0.3.1 — Fréchet and Gâteaux Derivatives

### Motivation

In single-variable calculus, the derivative $f'(x)$ is a *number* — the slope of the tangent line. In several variables, this generalizes in two different ways that agree in nice cases but can disagree in pathological examples:

- The **directional derivative** $D_\mathbf{v} f(\mathbf{x})$ measures rate of change along a specific direction $\mathbf{v}$.
- The **Fréchet derivative** $Df(\mathbf{x})$ is a single *linear map* that simultaneously captures rates of change in *all* directions.

The Fréchet notion is the correct one for almost every serious application: it's the notion under which the chain rule works cleanly, under which differentiable functions are continuous, and under which Taylor's theorem makes sense. This topic develops it carefully, distinguishes it from the weaker Gâteaux derivative, and gives sufficient conditions (continuous partials) that make verification tractable.

This is the foundation for literally every later "compute a derivative" argument: implicit differentiation for calibration, backpropagation, adjoint methods for Greeks, first- and second-order optimality conditions, and Itô's formula in stochastic calculus.

### Prerequisites

Normed spaces (Topic 0.2.9), linear maps (Topic 0.2.2), basic real analysis (limits, continuity).

### Setting and Notation

Throughout, $X$ and $Y$ are real finite-dimensional normed spaces (you can think $X = \mathbb{R}^n$, $Y = \mathbb{R}^m$ in examples). $\mathcal{L}(X, Y)$ denotes the space of continuous linear maps $X \to Y$; in finite dimensions all linear maps are continuous and $\mathcal{L}(X, Y) = \mathbb{R}^{m \times n}$ after choosing bases.

$U \subseteq X$ will always denote an *open* subset. The operator norm on $\mathcal{L}(X, Y)$ is $\|T\| = \sup_{\|\mathbf{x}\| = 1} \|T\mathbf{x}\|$.

### Definitions

**Definition 0.3.1.1 (Directional / Gâteaux derivative).** Let $f: U \to Y$, $\mathbf{x} \in U$, $\mathbf{v} \in X$. The **directional derivative** of $f$ at $\mathbf{x}$ in direction $\mathbf{v}$ is
$$D_\mathbf{v} f(\mathbf{x}) \;=\; \lim_{t \to 0}\; \frac{f(\mathbf{x} + t\mathbf{v}) - f(\mathbf{x})}{t}$$
when the limit exists. If $D_\mathbf{v} f(\mathbf{x})$ exists for every $\mathbf{v} \in X$ and the map $\mathbf{v} \mapsto D_\mathbf{v} f(\mathbf{x})$ is linear, we say $f$ is **Gâteaux differentiable** at $\mathbf{x}$ and write $D_G f(\mathbf{x}): X \to Y$ for this linear map.

**Definition 0.3.1.2 (Fréchet derivative).** $f: U \to Y$ is **Fréchet differentiable** at $\mathbf{x} \in U$ if there exists a linear map $A: X \to Y$ with
$$\lim_{\mathbf{h} \to \mathbf{0}} \frac{\|f(\mathbf{x} + \mathbf{h}) - f(\mathbf{x}) - A\mathbf{h}\|_Y}{\|\mathbf{h}\|_X} \;=\; 0.$$

Equivalently: $f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + A\mathbf{h} + r(\mathbf{h})$ with $r(\mathbf{h}) = o(\|\mathbf{h}\|)$ as $\mathbf{h} \to \mathbf{0}$.

The linear map $A$, when it exists, is called the **Fréchet derivative** (or the **total derivative**, or the **differential**) of $f$ at $\mathbf{x}$, and is denoted $Df(\mathbf{x})$ (or $f'(\mathbf{x})$). Its matrix in standard bases is the **Jacobian** $J_f(\mathbf{x})$.

**Proposition 0.3.1.3 (Uniqueness of the Fréchet derivative).** If $A, \tilde A$ both satisfy Definition 0.3.1.2, then $A = \tilde A$.

*Proof.* Subtract the two expansions:
$$A\mathbf{h} - \tilde A \mathbf{h} = o(\|\mathbf{h}\|).$$
So $\|(A - \tilde A)\mathbf{h}\| = o(\|\mathbf{h}\|)$. But $A - \tilde A$ is linear, so $\|(A - \tilde A)(t\mathbf{v})\| = t\|(A - \tilde A)\mathbf{v}\|$ for $t > 0$. The $o$ condition then forces $\|(A - \tilde A)\mathbf{v}\| = 0$ for every $\mathbf{v}$, so $A = \tilde A$. $\square$

### Immediate Consequences

**Proposition 0.3.1.4 (Fréchet $\Rightarrow$ continuous).** If $f$ is Fréchet differentiable at $\mathbf{x}$, then $f$ is continuous at $\mathbf{x}$.

*Proof.* $\|f(\mathbf{x} + \mathbf{h}) - f(\mathbf{x})\| \leq \|Df(\mathbf{x})\mathbf{h}\| + \|r(\mathbf{h})\| \leq \|Df(\mathbf{x})\|\|\mathbf{h}\| + o(\|\mathbf{h}\|) \to 0$ as $\mathbf{h} \to 0$. $\square$

**Proposition 0.3.1.5 (Fréchet $\Rightarrow$ Gâteaux, with $D_G f = Df$).** If $f$ is Fréchet differentiable at $\mathbf{x}$, then $f$ is Gâteaux differentiable there and $D_G f(\mathbf{x}) = Df(\mathbf{x})$.

*Proof.* For any $\mathbf{v} \in X$ and $t \to 0$:
$$\frac{f(\mathbf{x} + t\mathbf{v}) - f(\mathbf{x})}{t} = \frac{Df(\mathbf{x})(t\mathbf{v}) + r(t\mathbf{v})}{t} = Df(\mathbf{x})\mathbf{v} + \frac{r(t\mathbf{v})}{t}.$$
Since $r(t\mathbf{v}) = o(\|t\mathbf{v}\|) = o(|t|\|\mathbf{v}\|)$, $r(t\mathbf{v})/t \to 0$. So $D_\mathbf{v} f(\mathbf{x}) = Df(\mathbf{x})\mathbf{v}$. Linearity in $\mathbf{v}$ is inherited from $Df(\mathbf{x})$. $\square$

**Warning.** The converse is **false**. There exist functions that are Gâteaux differentiable (directional derivative in every direction, linear in direction) yet not Fréchet differentiable — and not even continuous. Example below.

**Example 0.3.1.6 (Gâteaux without Fréchet).** Define $f: \mathbb{R}^2 \to \mathbb{R}$ by
$$f(x, y) = \begin{cases} \dfrac{x^3 y}{x^4 + y^2} & (x, y) \neq (0,0), \\ 0 & (x, y) = (0, 0). \end{cases}$$

Along any line $(x, y) = t\mathbf{v}$ with $\mathbf{v} = (a, b)$: if $b \neq 0$, $f(ta, tb) = \frac{t^4 a^3 b}{t^4 a^4 + t^2 b^2} = \frac{t^2 a^3 b}{t^2 a^4 + b^2} \to 0$ as $t \to 0$. If $b = 0$, $f(ta, 0) = 0$. So $D_\mathbf{v} f(0) = 0$ for every $\mathbf{v}$, and the map $\mathbf{v} \mapsto D_\mathbf{v} f(0) \equiv 0$ is linear. $f$ is Gâteaux differentiable at $(0,0)$ with $D_G f(0) = 0$.

Yet $f$ is *not continuous* at $(0, 0)$: along the parabola $y = x^2$, $f(x, x^2) = x^5/(x^4 + x^4) = x/2$, which tends to $0$ as $x \to 0$ but the parabolic-trail gives a *different value of the difference quotient*:
$$\frac{f(x, x^2)}{\|(x, x^2)\|} \approx \frac{x/2}{|x|} \not\to 0.$$

So there is no linear $A$ approximating $f$ near $0$ in the Fréchet sense. By Proposition 0.3.1.4, Fréchet differentiability fails.

### Partial Derivatives

**Definition 0.3.1.7.** For $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ and $\mathbf{x} \in U$, the **partial derivative** $\partial f/\partial x_j(\mathbf{x}) = D_{\mathbf{e}_j} f(\mathbf{x})$ is the directional derivative in the $j$-th coordinate direction.

When $f$ is Fréchet differentiable, the Jacobian $J_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ has $(J_f)_{ij} = \partial f_i/\partial x_j(\mathbf{x})$: the $j$-th *column* of the Jacobian is $\partial f/\partial x_j$.

**Warning.** Existence of all partial derivatives does **not** imply Fréchet differentiability (as Example 0.3.1.6 shows — its partials at $0$ are both $0$, yet $f$ is not even continuous there). We need a stronger hypothesis:

**Theorem 0.3.1.8 (Continuously differentiable $\Rightarrow$ Fréchet differentiable).** Let $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$. Suppose all partial derivatives $\partial f_i/\partial x_j$ exist in a neighborhood of $\mathbf{x}_0 \in U$ and are continuous at $\mathbf{x}_0$. Then $f$ is Fréchet differentiable at $\mathbf{x}_0$ with Jacobian $J_f(\mathbf{x}_0) = (\partial f_i/\partial x_j(\mathbf{x}_0))$.

*Proof (for $m = 1$; vector case is componentwise).* Write $\mathbf{x}_0 = (a_1, \ldots, a_n)$ and $\mathbf{h} = (h_1, \ldots, h_n)$. Define intermediate points $\mathbf{z}_k = (a_1 + h_1, \ldots, a_k + h_k, a_{k+1}, \ldots, a_n)$ for $k = 0, 1, \ldots, n$; so $\mathbf{z}_0 = \mathbf{x}_0$ and $\mathbf{z}_n = \mathbf{x}_0 + \mathbf{h}$.

Telescope:
$$f(\mathbf{x}_0 + \mathbf{h}) - f(\mathbf{x}_0) = \sum_{k=1}^n [f(\mathbf{z}_k) - f(\mathbf{z}_{k-1})].$$

The $k$-th difference changes only the $k$-th coordinate (from $a_k$ to $a_k + h_k$), keeping the others fixed. Apply the one-variable mean value theorem to the single-variable function $\varphi_k(t) = f(a_1 + h_1, \ldots, a_{k-1} + h_{k-1}, t, a_{k+1}, \ldots, a_n)$ on $[a_k, a_k + h_k]$:
$$f(\mathbf{z}_k) - f(\mathbf{z}_{k-1}) = \varphi_k(a_k + h_k) - \varphi_k(a_k) = \varphi_k'(c_k) h_k = \frac{\partial f}{\partial x_k}(\mathbf{z}_{k-1}')\, h_k,$$
where $\mathbf{z}_{k-1}'$ is $\mathbf{z}_{k-1}$ but with the $k$-th coordinate replaced by some $c_k \in [a_k, a_k + h_k]$.

Summing:
$$f(\mathbf{x}_0 + \mathbf{h}) - f(\mathbf{x}_0) = \sum_{k=1}^n \frac{\partial f}{\partial x_k}(\mathbf{z}_{k-1}') h_k.$$

Compare to the candidate linear map $A\mathbf{h} = \sum_k \frac{\partial f}{\partial x_k}(\mathbf{x}_0) h_k$:
$$f(\mathbf{x}_0 + \mathbf{h}) - f(\mathbf{x}_0) - A\mathbf{h} = \sum_{k=1}^n \left[\frac{\partial f}{\partial x_k}(\mathbf{z}_{k-1}') - \frac{\partial f}{\partial x_k}(\mathbf{x}_0)\right] h_k.$$

By continuity of the partials at $\mathbf{x}_0$, for every $\varepsilon > 0$ there is $\delta > 0$ such that $\|\mathbf{h}\| < \delta$ implies $|\partial f/\partial x_k(\mathbf{z}_{k-1}') - \partial f/\partial x_k(\mathbf{x}_0)| < \varepsilon$ for all $k$ (all $\mathbf{z}_{k-1}'$ are within $\|\mathbf{h}\|$ of $\mathbf{x}_0$). Then
$$|f(\mathbf{x}_0 + \mathbf{h}) - f(\mathbf{x}_0) - A\mathbf{h}| \leq \varepsilon \sum_k |h_k| \leq \varepsilon \sqrt n \|\mathbf{h}\|_2.$$

Dividing by $\|\mathbf{h}\|$: the ratio is at most $\varepsilon\sqrt n$, which $\to 0$ as $\varepsilon \to 0$. So the Fréchet definition is satisfied with $A = J_f(\mathbf{x}_0)$. $\square$

**Definition 0.3.1.9 ($C^k$ functions).** $f: U \to \mathbb{R}^m$ is of class $C^1$ on $U$ if all partial derivatives exist and are continuous on $U$. Inductively, $f$ is $C^k$ if all partial derivatives up to order $k$ exist and are continuous. $f$ is $C^\infty$ (or **smooth**) if it is $C^k$ for every $k$.

**Corollary.** $C^1$ $\Rightarrow$ Fréchet differentiable with continuous derivative.

### Gradient, Jacobian, Hessian — Summary Table

For $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ at an interior point where it is Fréchet differentiable:

- Scalar case ($m = 1$): $Df(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}$ is a linear functional represented by the **gradient** column vector $\nabla f(\mathbf{x}) \in \mathbb{R}^n$: $Df(\mathbf{x})\mathbf{h} = \nabla f(\mathbf{x})^\top \mathbf{h}$.
- Vector case: $Df(\mathbf{x})$ represented by the $m \times n$ **Jacobian matrix** $J_f(\mathbf{x})$: $Df(\mathbf{x})\mathbf{h} = J_f(\mathbf{x})\mathbf{h}$.
- Second derivative of scalar $f$: $D^2 f(\mathbf{x})$ is a symmetric bilinear form represented by the $n \times n$ **Hessian matrix** $H_f(\mathbf{x})$: $D^2 f(\mathbf{x})(\mathbf{h}, \mathbf{k}) = \mathbf{h}^\top H_f(\mathbf{x}) \mathbf{k}$.

### Worked Examples

**Example 0.3.1.10 (Linear map).** For $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ with $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$:
$$f(\mathbf{x} + \mathbf{h}) - f(\mathbf{x}) = A\mathbf{h}.$$
Exact, no remainder. $Df(\mathbf{x}) = A$ for every $\mathbf{x}$.

**Example 0.3.1.11 (Quadratic form).** $f(\mathbf{x}) = \mathbf{x}^\top Q \mathbf{x}$ with $Q \in \mathbb{R}^{n \times n}$. Expanding:
$$f(\mathbf{x} + \mathbf{h}) = (\mathbf{x} + \mathbf{h})^\top Q (\mathbf{x} + \mathbf{h}) = \mathbf{x}^\top Q \mathbf{x} + \mathbf{x}^\top Q \mathbf{h} + \mathbf{h}^\top Q \mathbf{x} + \mathbf{h}^\top Q \mathbf{h}.$$
The linear-in-$\mathbf{h}$ part is $\mathbf{x}^\top Q \mathbf{h} + \mathbf{h}^\top Q \mathbf{x} = \mathbf{h}^\top(Q + Q^\top)\mathbf{x}$. The $O(\|\mathbf{h}\|^2)$ part is $\mathbf{h}^\top Q \mathbf{h}$.

So $\nabla f(\mathbf{x}) = (Q + Q^\top)\mathbf{x}$ (as in Topic 0.2.8). If $Q$ is symmetric, $\nabla f = 2Q\mathbf{x}$. The Hessian is constant: $H_f = Q + Q^\top$.

**Example 0.3.1.12 (Norm squared).** $f(\mathbf{x}) = \|\mathbf{x}\|^2 = \sum x_i^2$. Partials: $\partial f/\partial x_i = 2x_i$, continuous everywhere. So $\nabla f(\mathbf{x}) = 2\mathbf{x}$, $H_f = 2I$.

**Example 0.3.1.13 ($\|\mathbf{x}\|$ at origin).** $f(\mathbf{x}) = \|\mathbf{x}\|_2$ is *not* differentiable at $\mathbf{x} = \mathbf{0}$. (Any candidate $A$ would satisfy $\|t\mathbf{v}\| - A(t\mathbf{v}) = o(t)$, i.e., $|t|\|\mathbf{v}\| - t A\mathbf{v} = o(t)$ — dividing by $t$ and letting $t \to 0^\pm$ gives contradictory limits.) Away from origin, $\nabla f(\mathbf{x}) = \mathbf{x}/\|\mathbf{x}\|$ (the unit vector along $\mathbf{x}$).

**Example 0.3.1.14 (Matrix function).** $f(X) = \mathrm{tr}(X^\top X)$ for $X \in \mathbb{R}^{m \times n}$.

Expand: $f(X + H) = \mathrm{tr}((X + H)^\top (X + H)) = \mathrm{tr}(X^\top X) + \mathrm{tr}(X^\top H) + \mathrm{tr}(H^\top X) + \mathrm{tr}(H^\top H) = f(X) + 2\mathrm{tr}(X^\top H) + \|H\|_F^2$.

Linear part: $2\mathrm{tr}(X^\top H) = \langle 2X, H\rangle_F$. So $Df(X)(H) = 2\mathrm{tr}(X^\top H)$, and the gradient (matrix-shaped, under trace inner product) is $\nabla f(X) = 2X$.

### Computational Implementation

```python
import numpy as np
from numpy.linalg import norm

# Check Fréchet differentiability numerically:
#   residual(h) = f(x+h) - f(x) - J @ h  should be o(||h||)

def check_frechet(f, J, x, n_samples=100, scales=None, rng=None):
    """Verify that J is the Fréchet derivative of f at x by checking
    ||f(x+h) - f(x) - J @ h|| / ||h|| -> 0 as ||h|| -> 0.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if scales is None:
        scales = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    x = np.asarray(x, dtype=float)
    results = []
    for s in scales:
        ratios = []
        for _ in range(n_samples):
            h = s * rng.standard_normal(x.shape)
            num = norm(f(x + h) - f(x) - J @ h)
            den = norm(h)
            ratios.append(num / den)
        results.append((s, np.mean(ratios), np.max(ratios)))
    return results

# Example: f(x) = Q x + sin(x) [componentwise]
rng = np.random.default_rng(42)
Q = rng.standard_normal((4, 4))
f = lambda x: Q @ x + np.sin(x)
J = lambda x: Q + np.diag(np.cos(x))

x0 = rng.standard_normal(4)
for (s, mean, mx) in check_frechet(f, J(x0), x0, rng=rng):
    print(f"||h|| ~ {s:.0e}: mean ratio = {mean:.2e}, max ratio = {mx:.2e}")
# Ratio should decrease proportionally to s (first-order remainder)

# Counterexample: Gâteaux but not Fréchet
def gateaux_only(xy):
    x, y = xy
    if x == 0 and y == 0:
        return 0.0
    return x**3 * y / (x**4 + y**2)

# Directional derivatives at 0 are all 0 (check along any direction)
for v in [np.array([1, 0]), np.array([0, 1]), np.array([1, 1]), np.array([1, 2])]:
    ts = np.array([1e-3, 1e-4, 1e-5])
    diff_quot = [gateaux_only(t * v) / t for t in ts]
    print(f"Direction {v}: diff quotients = {diff_quot}")  # all -> 0

# But approaching along parabola y = x^2:
for x in [1e-2, 1e-3, 1e-4]:
    val = gateaux_only(np.array([x, x**2])) / norm(np.array([x, x**2]))
    print(f"Parabola x={x}: f(x,x^2)/||(x,x^2)|| = {val}")  # ~0.5, does NOT -> 0
# So f is not Fréchet differentiable at origin.
```

### [QUANT APPLICATION] — When Partial Derivatives Suffice, Numerical Gradients, Automatic Differentiation

**(A) Smoothness assumptions in finance.** Most payoffs we encounter are piecewise $C^1$: vanilla options have a kink at strike, barriers jump at the barrier. Greeks (Delta, Gamma) are derivatives *where they exist*; at kinks, the second derivative can be a Dirac measure (see pricing-theory treatments of Gamma for barrier options). The distinction between Fréchet differentiability and distributional derivatives matters for hedging arguments.

**(B) Finite-difference gradients and their pitfalls.** Computing $\partial f/\partial x_i \approx (f(\mathbf{x} + \varepsilon \mathbf{e}_i) - f(\mathbf{x} - \varepsilon \mathbf{e}_i))/(2\varepsilon)$ converges as $O(\varepsilon^2)$ in smooth functions, but suffers from roundoff for small $\varepsilon$. The optimal $\varepsilon$ trades off truncation error ($\propto \varepsilon^2$) vs. roundoff ($\propto \varepsilon_{\mathrm{mach}}/\varepsilon$), typically around $\varepsilon \sim \varepsilon_{\mathrm{mach}}^{1/3}$.

**(C) Complex-step differentiation.** For analytic $f: \mathbb{R} \to \mathbb{R}$ that extends to $\mathbb{C}$: $f'(x) \approx \mathrm{Im}[f(x + i\varepsilon)] / \varepsilon$ with no subtraction, avoiding roundoff. Used in option pricing Greeks.

**(D) Automatic differentiation.** Forward-mode AD propagates derivatives through arithmetic. Reverse-mode AD (backprop) computes gradients of scalar outputs with respect to all inputs at a cost comparable to one function evaluation — the adjoint methods mentioned in Topic 0.2.8. JAX, PyTorch, and Zygote implement AD for quant research pipelines.

**(E) Differentiable constraints and the Implicit Function Theorem.** Many quant problems require differentiating through an optimization solver (calibration: solve for model parameters given market prices). IFT (Topic 0.3.4) is the theoretical basis.

### Exercises

#### ★ (Foundation)

**E0.3.1.1.** For $f(x, y) = x^2 y + e^{xy}$, compute $\nabla f(x, y)$ and verify continuity of partials. Conclude $f$ is Fréchet differentiable everywhere.

**E0.3.1.2.** Let $f: \mathbb{R}^n \to \mathbb{R}$ be $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$. Show $Df(\mathbf{x}) \mathbf{h} = \mathbf{a}^\top \mathbf{h}$, so $\nabla f \equiv \mathbf{a}$.

**E0.3.1.3.** For $f: \mathbb{R}^n \to \mathbb{R}^m$, prove: if all components $f_i$ are Fréchet differentiable at $\mathbf{x}$, so is $f$, and $J_f = (J_{f_1}; \ldots; J_{f_m})^\top$ (stacked rows).

**E0.3.1.4.** Compute the Jacobian of the polar-to-Cartesian map $(r, \theta) \mapsto (r\cos\theta, r\sin\theta)$ and its determinant.

**E0.3.1.5.** Compute $\nabla f$ for $f(\mathbf{x}) = \|\mathbf{x}\|^3$ (for $\mathbf{x} \neq \mathbf{0}$).

*Hint.* Let $r = \|\mathbf{x}\|$; then $f = r^3$, $\nabla r = \mathbf{x}/r$, chain rule.

#### ★★ (Intermediate)

**E0.3.1.6.** Show that Fréchet differentiability at a point is a local property: if $f = g$ on an open neighborhood of $\mathbf{x}_0$, then $Df(\mathbf{x}_0) = Dg(\mathbf{x}_0)$ (provided one exists).

**E0.3.1.7.** Prove: if $f: U \to \mathbb{R}$ is Fréchet differentiable at $\mathbf{x}_0$ and achieves a local minimum at $\mathbf{x}_0$, then $\nabla f(\mathbf{x}_0) = \mathbf{0}$.

*Hint.* Directional derivative in direction $\mathbf{v}$ and $-\mathbf{v}$ must both be $\geq 0$.

**E0.3.1.8.** Let $f(x, y) = xy(x^2 - y^2)/(x^2 + y^2)$ for $(x,y) \neq (0,0)$, $f(0,0) = 0$. Show that both partial derivatives exist everywhere, but the mixed partials $f_{xy}(0,0) = -1 \neq 1 = f_{yx}(0,0)$ at the origin. Explain why this doesn't violate Clairaut's / Schwarz's theorem.

*Hint.* Continuity of mixed partials is required for Schwarz. Check (dis)continuity here.

**E0.3.1.9.** Let $A: X \to Y$ be linear and continuous. Show $DA(\mathbf{x}) = A$ for every $\mathbf{x}$.

#### ★★★ (Challenge)

**E0.3.1.10 (Fréchet differentiability of the determinant).** Let $f: M_n(\mathbb{R}) \to \mathbb{R}$, $f(X) = \det X$. Show $f$ is Fréchet differentiable everywhere, with $Df(X)(H) = \det(X) \cdot \mathrm{tr}(X^{-1} H)$ when $X$ is invertible. What is $Df(X)$ when $X$ is singular?

*Hint (singular case).* $Df(X)(H) = \mathrm{tr}(\mathrm{adj}(X) H)$ — uses the adjugate (classical adjoint) matrix.

**E0.3.1.11 (Darboux's theorem — derivatives are Darboux even if not continuous).** If $f: [a, b] \to \mathbb{R}$ is differentiable and $f'(a) < c < f'(b)$, there exists $\xi \in (a, b)$ with $f'(\xi) = c$. (Like the intermediate value property, but for $f'$, which need not itself be continuous.)

*Hint.* Consider $g(x) = f(x) - cx$; apply the extreme value theorem on $[a, b]$; the extremum is interior, giving $g'(\xi) = 0$.

**E0.3.1.12 (Smoothness under composition).** If $f: U \to V$ and $g: V \to W$ are both $C^k$, show $g \circ f$ is $C^k$. (Requires the multivariate chain rule, Topic 0.3.2, plus induction.)

---

## Topic 0.3.2 — Chain Rule and Taylor's Theorem

### Motivation

The **chain rule** is the engine behind every serious derivative computation in quant. It lets us:

- Break big derivatives into compositions of simple ones (what backpropagation does mechanically).
- Transport derivatives through reparameterizations (change of variables in probability densities, in SDEs, in Greeks).
- State optimality conditions for constrained problems, since Lagrange and KKT are just the chain rule applied to parameterized curves on feasible sets.

**Taylor's theorem** gives us the "polynomial approximation with remainder bounds" that underlies:

- Second-order optimization (Newton's method, trust regions).
- Error analysis for numerical methods (finite differences, quadrature, ODE solvers).
- The second-order conditions for minima (Hessian $\succeq 0$ at a local min).
- Asymptotic expansions (Laplace's method, saddle-point approximation for option pricing).
- The Stratonovich–Itô correction via the quadratic term in the Taylor expansion of a smooth function of a semimartingale.

### Prerequisites

Fréchet differentiability (Topic 0.3.1), operator norms (Topic 0.2.9).

### The Chain Rule

**Theorem 0.3.2.1 (Chain rule).** Let $f: U \to Y$ be Fréchet differentiable at $\mathbf{x} \in U \subseteq X$, and $g: V \to Z$ Fréchet differentiable at $\mathbf{y} = f(\mathbf{x}) \in V \subseteq Y$, with $f(U) \subseteq V$. Then $g \circ f$ is Fréchet differentiable at $\mathbf{x}$, with
$$D(g \circ f)(\mathbf{x}) = Dg(f(\mathbf{x})) \circ Df(\mathbf{x}).$$
In matrix form: $J_{g \circ f}(\mathbf{x}) = J_g(f(\mathbf{x})) \cdot J_f(\mathbf{x})$.

*Proof.* Write $A = Df(\mathbf{x})$, $B = Dg(\mathbf{y})$. We have
$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + A\mathbf{h} + r_1(\mathbf{h}), \quad r_1(\mathbf{h}) = o(\|\mathbf{h}\|).$$
$$g(\mathbf{y} + \mathbf{k}) = g(\mathbf{y}) + B\mathbf{k} + r_2(\mathbf{k}), \quad r_2(\mathbf{k}) = o(\|\mathbf{k}\|).$$

Set $\mathbf{k} = f(\mathbf{x} + \mathbf{h}) - \mathbf{y} = A\mathbf{h} + r_1(\mathbf{h})$. Then
$$g(f(\mathbf{x} + \mathbf{h})) = g(\mathbf{y}) + B(A\mathbf{h} + r_1(\mathbf{h})) + r_2(A\mathbf{h} + r_1(\mathbf{h})) = g(f(\mathbf{x})) + BA\mathbf{h} + B r_1(\mathbf{h}) + r_2(\mathbf{k}).$$

We need to check that $Br_1(\mathbf{h}) + r_2(\mathbf{k}) = o(\|\mathbf{h}\|)$:

- $\|Br_1(\mathbf{h})\| \leq \|B\|\|r_1(\mathbf{h})\| = \|B\| \cdot o(\|\mathbf{h}\|) = o(\|\mathbf{h}\|)$. ✓

- For $r_2(\mathbf{k})$: $\|\mathbf{k}\| \leq \|A\|\|\mathbf{h}\| + \|r_1(\mathbf{h})\| = O(\|\mathbf{h}\|)$. So as $\mathbf{h} \to 0$, $\mathbf{k} \to 0$ and $r_2(\mathbf{k}) = o(\|\mathbf{k}\|) \leq o(\|\mathbf{h}\| \cdot O(1)) = o(\|\mathbf{h}\|)$. ✓

Therefore $g(f(\mathbf{x} + \mathbf{h})) = g(f(\mathbf{x})) + BA\mathbf{h} + o(\|\mathbf{h}\|)$, so $D(g \circ f)(\mathbf{x}) = BA$. $\square$

**Corollary (component form).** For $f: \mathbb{R}^n \to \mathbb{R}^m$, $g: \mathbb{R}^m \to \mathbb{R}^p$:
$$\frac{\partial (g \circ f)_i}{\partial x_k}(\mathbf{x}) = \sum_{j=1}^m \frac{\partial g_i}{\partial y_j}(f(\mathbf{x})) \cdot \frac{\partial f_j}{\partial x_k}(\mathbf{x}).$$

### Mean Value Inequality

In one variable, the mean value theorem says $f(b) - f(a) = f'(\xi)(b - a)$ for some $\xi \in (a, b)$. In higher dimensions, *no single point* $\xi$ works in general (for vector-valued $f$). We instead have a *bound*:

**Theorem 0.3.2.2 (Mean value inequality).** Let $f: U \to Y$ with $U \subseteq X$ open convex, and $f$ Fréchet differentiable on $U$. For any $\mathbf{a}, \mathbf{b} \in U$,
$$\|f(\mathbf{b}) - f(\mathbf{a})\|_Y \leq \sup_{t \in [0,1]} \|Df(\mathbf{a} + t(\mathbf{b} - \mathbf{a}))\|_{\mathrm{op}} \cdot \|\mathbf{b} - \mathbf{a}\|_X.$$

*Proof.* Let $\gamma(t) = \mathbf{a} + t(\mathbf{b} - \mathbf{a})$ and $\varphi(t) = f(\gamma(t))$. By the chain rule, $\varphi'(t) = Df(\gamma(t))(\mathbf{b} - \mathbf{a})$.

Pick any continuous linear functional $\ell$ on $Y$ with $\|\ell\|_{Y^*} = 1$ and $\ell(f(\mathbf{b}) - f(\mathbf{a})) = \|f(\mathbf{b}) - f(\mathbf{a})\|$ (exists by Hahn–Banach in infinite dim; in finite dim, take $\ell(\mathbf{y}) = \mathbf{y}^\top \mathbf{u}$ where $\mathbf{u} = (f(\mathbf{b}) - f(\mathbf{a}))/\|f(\mathbf{b}) - f(\mathbf{a})\|$).

Then $\psi(t) = \ell(f(\gamma(t)))$ is a real-valued function on $[0, 1]$, differentiable with $\psi'(t) = \ell(\varphi'(t))$. By the one-dimensional mean value theorem applied to $\psi$:
$$\psi(1) - \psi(0) = \psi'(\xi) \quad \text{for some } \xi \in (0, 1).$$

But $\psi(1) - \psi(0) = \ell(f(\mathbf{b}) - f(\mathbf{a})) = \|f(\mathbf{b}) - f(\mathbf{a})\|$. And
$$|\psi'(\xi)| = |\ell(\varphi'(\xi))| \leq \|\ell\|_{Y^*} \|\varphi'(\xi)\|_Y = \|\varphi'(\xi)\|_Y = \|Df(\gamma(\xi))(\mathbf{b} - \mathbf{a})\| \leq \|Df(\gamma(\xi))\|_{\mathrm{op}} \|\mathbf{b} - \mathbf{a}\|.$$

Combining gives the claim. $\square$

**Corollary 0.3.2.3 (Lipschitz from bounded derivative).** If $f: U \to Y$ is $C^1$ on a convex open set $U$ with $\sup_U \|Df\|_{\mathrm{op}} \leq M$, then $f$ is $M$-Lipschitz on $U$:
$$\|f(\mathbf{x}) - f(\mathbf{y})\| \leq M \|\mathbf{x} - \mathbf{y}\|.$$

### Higher-Order Derivatives

**Definition 0.3.2.4.** Suppose $f: U \to Y$ is Fréchet differentiable on $U$, so $Df: U \to \mathcal{L}(X, Y)$ is a map. If $Df$ is itself Fréchet differentiable at $\mathbf{x}$, we get the **second derivative** $D^2 f(\mathbf{x}) \in \mathcal{L}(X, \mathcal{L}(X, Y))$.

A linear map $X \to \mathcal{L}(X, Y)$ is canonically identified with a *bilinear* map $X \times X \to Y$ via $D^2 f(\mathbf{x})(\mathbf{h}, \mathbf{k}) = (D^2 f(\mathbf{x})(\mathbf{h}))(\mathbf{k})$.

Inductively, the $k$-th derivative $D^k f(\mathbf{x})$ is identified with a **$k$-linear map** $X^k \to Y$.

**Theorem 0.3.2.5 (Schwarz / symmetry of second derivatives).** If $f: U \to Y$ is $C^2$ (i.e., $D^2 f$ exists and is continuous), then $D^2 f(\mathbf{x})$ is a *symmetric* bilinear form:
$$D^2 f(\mathbf{x})(\mathbf{h}, \mathbf{k}) = D^2 f(\mathbf{x})(\mathbf{k}, \mathbf{h}).$$

*Proof (for $f: \mathbb{R}^2 \to \mathbb{R}$; general case reduces to this).* Consider the symmetric difference
$$\Delta(s, t) = f(\mathbf{x} + s\mathbf{e}_1 + t\mathbf{e}_2) - f(\mathbf{x} + s\mathbf{e}_1) - f(\mathbf{x} + t\mathbf{e}_2) + f(\mathbf{x}).$$
Apply the one-variable MVT twice: first in $t$, then in $s$:
$$\Delta(s, t) = t \partial_2 f(\mathbf{x} + s\mathbf{e}_1 + \tau_1 \mathbf{e}_2) - t\partial_2 f(\mathbf{x} + \tau_2 \mathbf{e}_2) = st \partial_1 \partial_2 f(\mathbf{x} + \sigma \mathbf{e}_1 + \tau \mathbf{e}_2) + o(st).$$
By continuity of second partials, as $(s, t) \to 0$: $\Delta(s, t)/(st) \to \partial_1 \partial_2 f(\mathbf{x})$. Reversing the order of MVT gives $\Delta(s, t)/(st) \to \partial_2 \partial_1 f(\mathbf{x})$. Since the limit is unique, $\partial_1 \partial_2 f(\mathbf{x}) = \partial_2 \partial_1 f(\mathbf{x})$. $\square$

**Corollary.** In matrix form, the Hessian $H_f(\mathbf{x}) = (\partial_i \partial_j f)$ is *symmetric* when $f \in C^2$.

### Taylor's Theorem

**Theorem 0.3.2.6 (Taylor's theorem, Lagrange remainder — scalar case).** Let $U \subseteq \mathbb{R}^n$ open, $f: U \to \mathbb{R}$ of class $C^{k+1}$, and $[\mathbf{x}, \mathbf{x} + \mathbf{h}] \subseteq U$. Then there exists $\xi \in (0, 1)$ with
$$f(\mathbf{x} + \mathbf{h}) = \sum_{j=0}^k \frac{1}{j!} D^j f(\mathbf{x})(\mathbf{h}, \ldots, \mathbf{h}) + \frac{1}{(k+1)!} D^{k+1} f(\mathbf{x} + \xi\mathbf{h})(\mathbf{h}, \ldots, \mathbf{h}).$$

Here $D^j f(\mathbf{x})(\mathbf{h}, \ldots, \mathbf{h})$ means the $j$-linear form $D^j f(\mathbf{x})$ evaluated on $j$ copies of $\mathbf{h}$.

*Proof.* Let $\varphi(t) = f(\mathbf{x} + t\mathbf{h})$. By the chain rule (iterated), $\varphi \in C^{k+1}([0, 1])$ with $\varphi^{(j)}(t) = D^j f(\mathbf{x} + t\mathbf{h})(\mathbf{h}, \ldots, \mathbf{h})$.

By one-variable Taylor with Lagrange remainder, there exists $\xi \in (0, 1)$ with
$$\varphi(1) = \sum_{j=0}^k \frac{\varphi^{(j)}(0)}{j!} + \frac{\varphi^{(k+1)}(\xi)}{(k+1)!}.$$

Substituting gives the claim. $\square$

**Theorem 0.3.2.7 (Taylor with integral remainder).** Under the same hypotheses:
$$f(\mathbf{x} + \mathbf{h}) = \sum_{j=0}^k \frac{1}{j!} D^j f(\mathbf{x})(\mathbf{h}, \ldots, \mathbf{h}) + \int_0^1 \frac{(1-t)^k}{k!} D^{k+1} f(\mathbf{x} + t\mathbf{h})(\mathbf{h}, \ldots, \mathbf{h})\, dt.$$

*Proof.* Integration by parts in the integral form of one-variable Taylor's theorem. $\square$

**Theorem 0.3.2.8 (Taylor with Peano remainder).** If $f \in C^k$ near $\mathbf{x}$:
$$f(\mathbf{x} + \mathbf{h}) = \sum_{j=0}^k \frac{1}{j!} D^j f(\mathbf{x})(\mathbf{h}, \ldots, \mathbf{h}) + o(\|\mathbf{h}\|^k).$$

This is the softest form (only needs $C^k$, not $C^{k+1}$) and suffices for most proofs of optimality conditions.

### Second-Order Taylor and Optimality

In multivariate calculus, first-order Taylor says
$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h} + o(\|\mathbf{h}\|),$$
and second-order Taylor says
$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h} + \tfrac{1}{2} \mathbf{h}^\top H_f(\mathbf{x}) \mathbf{h} + o(\|\mathbf{h}\|^2).$$

**Theorem 0.3.2.9 (First-order necessary condition).** If $f: U \to \mathbb{R}$ is differentiable and $\mathbf{x}^* \in U$ is a local minimum, then $\nabla f(\mathbf{x}^*) = \mathbf{0}$.

*Proof.* If $\nabla f(\mathbf{x}^*) \neq \mathbf{0}$, taking $\mathbf{h} = -t\nabla f(\mathbf{x}^*)$ for small $t > 0$: $f(\mathbf{x}^* + \mathbf{h}) - f(\mathbf{x}^*) = -t\|\nabla f(\mathbf{x}^*)\|^2 + o(t) < 0$ — contradiction. $\square$

**Theorem 0.3.2.10 (Second-order conditions).** Let $f \in C^2$ near $\mathbf{x}^*$.
- *Necessary:* If $\mathbf{x}^*$ is a local minimum, then $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $H_f(\mathbf{x}^*) \succeq 0$ (PSD).
- *Sufficient:* If $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $H_f(\mathbf{x}^*) \succ 0$ (PD), then $\mathbf{x}^*$ is a strict local minimum.

*Proof.* Necessary: from 2nd-order Taylor with $\nabla f(\mathbf{x}^*) = 0$: $f(\mathbf{x}^* + \mathbf{h}) - f(\mathbf{x}^*) = \tfrac{1}{2}\mathbf{h}^\top H_f(\mathbf{x}^*)\mathbf{h} + o(\|\mathbf{h}\|^2)$. If $\mathbf{h}^\top H_f(\mathbf{x}^*)\mathbf{h} < 0$ for some $\mathbf{h}$, then $f(\mathbf{x}^* + t\mathbf{h}) - f(\mathbf{x}^*) < 0$ for small $t$ — contradicting local min. So $H_f(\mathbf{x}^*) \succeq 0$.

Sufficient: let $m = \lambda_{\min}(H_f(\mathbf{x}^*)) > 0$. Then $\mathbf{h}^\top H_f(\mathbf{x}^*) \mathbf{h} \geq m\|\mathbf{h}\|^2$. With $o(\|\mathbf{h}\|^2) \leq \tfrac{m}{4}\|\mathbf{h}\|^2$ for small $\mathbf{h}$:
$$f(\mathbf{x}^* + \mathbf{h}) - f(\mathbf{x}^*) \geq \tfrac{m}{2}\|\mathbf{h}\|^2 - \tfrac{m}{4}\|\mathbf{h}\|^2 = \tfrac{m}{4}\|\mathbf{h}\|^2 > 0$$
for $\mathbf{h} \neq \mathbf{0}$ small. So $\mathbf{x}^*$ is a strict local minimum. $\square$

**Saddle point test.** If $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $H_f(\mathbf{x}^*)$ has both positive and negative eigenvalues, $\mathbf{x}^*$ is a **saddle point** (neither min nor max).

### Worked Examples

**Example 0.3.2.11 (Chain rule for gradient of composition).** Let $g: \mathbb{R}^n \to \mathbb{R}$ and $f: \mathbb{R}^m \to \mathbb{R}^n$, define $h = g \circ f: \mathbb{R}^m \to \mathbb{R}$. Chain rule:
$$\nabla h(\mathbf{x}) = J_f(\mathbf{x})^\top \nabla g(f(\mathbf{x})).$$
(Transpose because $Dh(\mathbf{x}) = Dg(f(\mathbf{x})) J_f(\mathbf{x})$ is a row vector; the gradient is its transpose.)

**Example 0.3.2.12 (Hessian of composition).** For $h(\mathbf{x}) = g(A\mathbf{x} + \mathbf{b})$ with linear inner map:
$$\nabla h(\mathbf{x}) = A^\top \nabla g(A\mathbf{x} + \mathbf{b}), \qquad H_h(\mathbf{x}) = A^\top H_g(A\mathbf{x} + \mathbf{b}) A.$$

**Example 0.3.2.13 (Taylor expansion of $\log(1 + \mathbf{x}^\top \mathbf{x})$).** Let $f(\mathbf{x}) = \log(1 + \|\mathbf{x}\|^2)$. At $\mathbf{x} = 0$: $f(0) = 0$; $\nabla f(0) = 0$; $H_f(\mathbf{x}) = \frac{2I}{1 + \|\mathbf{x}\|^2} - \frac{4\mathbf{x}\mathbf{x}^\top}{(1 + \|\mathbf{x}\|^2)^2}$, so $H_f(0) = 2I$. Taylor:
$$f(\mathbf{h}) = \|\mathbf{h}\|^2 + o(\|\mathbf{h}\|^2).$$
Matches the scalar expansion $\log(1 + u) = u - u^2/2 + \ldots$ with $u = \|\mathbf{h}\|^2$.

**Example 0.3.2.14 (Saddle point).** $f(x, y) = x^2 - y^2$. $\nabla f = (2x, -2y)$, critical point $(0, 0)$. $H_f = \mathrm{diag}(2, -2)$, indefinite — saddle.

### Computational Implementation

```python
import numpy as np
from numpy.linalg import eigvalsh, norm

# Verify chain rule numerically
rng = np.random.default_rng(0)
n, m = 4, 3
A = rng.standard_normal((m, n))
b = rng.standard_normal(m)

f = lambda x: A @ x + b                     # f: R^n -> R^m
g = lambda y: np.sum(y**3)                   # g: R^m -> R
h = lambda x: g(f(x))                        # composition

x0 = rng.standard_normal(n)
y0 = f(x0)
grad_g = 3 * y0**2                           # nabla g at y0
grad_h_analytic = A.T @ grad_g               # chain rule

# Compare with finite-difference gradient of h
eps = 1e-6
grad_h_numeric = np.zeros(n)
for i in range(n):
    e = np.zeros(n); e[i] = 1
    grad_h_numeric[i] = (h(x0 + eps*e) - h(x0 - eps*e)) / (2*eps)

print("Chain rule error:", norm(grad_h_analytic - grad_h_numeric))

# Second-order Taylor approximation
def taylor2(f, grad_f, hess_f, x0, h):
    return f(x0) + grad_f(x0) @ h + 0.5 * h @ hess_f(x0) @ h

# Function: f(x) = log(1 + ||x||^2)
f_scalar = lambda x: np.log1p(x @ x)
grad_scalar = lambda x: 2*x / (1 + x @ x)
def hess_scalar(x):
    n = len(x)
    s = 1 + x @ x
    return 2/s * np.eye(n) - 4 * np.outer(x, x) / s**2

x0 = np.array([0.1, -0.2, 0.15])
h = 0.01 * rng.standard_normal(3)
exact = f_scalar(x0 + h)
approx = taylor2(f_scalar, grad_scalar, hess_scalar, x0, h)
print(f"Exact: {exact:.8f}, Taylor2: {approx:.8f}, diff: {abs(exact - approx):.2e}")
# diff ~ O(||h||^3)

# Second-order conditions check: is origin a local min of f(x,y) = x^4 + y^4?
def classify_critical(H):
    eigs = eigvalsh(H)
    if all(eigs > 1e-10): return "local min"
    if all(eigs < -1e-10): return "local max"
    if all(abs(eigs) < 1e-10): return "degenerate (Hessian test inconclusive)"
    return "saddle point"

# Saddle: f(x,y) = x^2 - y^2
print("f=x^2-y^2:", classify_critical(np.array([[2, 0], [0, -2]])))
# Min: f(x,y) = x^2 + y^2
print("f=x^2+y^2:", classify_critical(np.array([[2, 0], [0, 2]])))
# Degenerate: f(x,y) = x^4 + y^4  (Hessian at origin is zero)
print("f=x^4+y^4 at 0:", classify_critical(np.zeros((2, 2))))
```

### [QUANT APPLICATION] — Newton's Method, Greeks via Chain Rule, Second-Order Conditions in Portfolio Problems

**(A) Newton's method.** To minimize $f$, Newton's update is $\mathbf{x}_{k+1} = \mathbf{x}_k - H_f(\mathbf{x}_k)^{-1} \nabla f(\mathbf{x}_k)$. Derived from second-order Taylor: minimize the quadratic approximation at each step. Converges quadratically near a local minimum with $H_f \succ 0$; fails or diverges at saddles or when $H_f$ is indefinite. Practical variants: Levenberg–Marquardt, trust region, quasi-Newton (BFGS, L-BFGS).

**(B) Greeks via chain rule.** A derivative $V(S, \sigma, r, T, K)$ depends on many variables. The Greeks $\Delta = \partial V/\partial S$, $\Gamma = \partial^2 V/\partial S^2$, $\mathcal{V} = \partial V/\partial \sigma$ (vega), etc. come from partial derivatives. When $V$ is computed via a pipeline of transformations (e.g., implied-vol → Black-Scholes → market-adjusted → portfolio), chain rule (backprop / adjoint) gives all Greeks at a cost comparable to one forward pricing. This is the foundation of **Adjoint Algorithmic Differentiation** (AAD) in quant libraries.

**(C) Convexity for portfolio optimization.** A function is convex iff its Hessian is PSD (global version of Theorem 0.3.2.10). The Markowitz objective $\mathbf{w}^\top \Sigma \mathbf{w}$ is convex (Hessian $2\Sigma \succeq 0$). Any unconstrained critical point is the global minimum. This is why mean-variance problems have unique (up to degeneracies) solutions and why convex solvers (CVXPY, MOSEK) can reliably find them.

**(D) Saddle-point method in asymptotic analysis.** For computing tail probabilities and option prices via Fourier inversion, integrals of the form $\int e^{-T\psi(\theta)}d\theta$ are dominated by the saddle point $\theta^*$ where $\psi'(\theta^*) = 0$. The second-order expansion around $\theta^*$ gives the leading asymptotic term (Laplace's method).

**(E) Itô's formula via Taylor expansion.** For smooth $f$ of a semimartingale $X_t$, Itô expands to second order:
$$f(X_t) = f(X_0) + \int_0^t f'(X_s)\,dX_s + \tfrac{1}{2}\int_0^t f''(X_s)\,d\langle X\rangle_s.$$
The "extra" $\tfrac{1}{2}f''\,d\langle X\rangle$ is the quadratic Taylor term — identically what would appear in multivariate Taylor but applied to stochastic processes. This is the subject of Module 5.

### Exercises

#### ★ (Foundation)

**E0.3.2.1.** Compute the Jacobian of $f(x, y) = (e^x \cos y, e^x \sin y)$. Verify $\det J_f = e^{2x}$ ("conformal" at every point).

**E0.3.2.2.** Apply the chain rule: if $f(\mathbf{x}) = g(\|\mathbf{x}\|^2)$ for $g: \mathbb{R} \to \mathbb{R}$, compute $\nabla f$ and $H_f$.

**E0.3.2.3.** Write the second-order Taylor expansion of $f(x, y) = e^{xy}$ around $(0, 0)$.

**E0.3.2.4.** Use second-order conditions to classify the critical points of $f(x, y) = x^3 - 3x + y^2$.

#### ★★ (Intermediate)

**E0.3.2.5 (Bound on matrix derivative).** Let $\Phi(t) = e^{tA}$ for $A \in M_n$. Show $\Phi$ is $C^\infty$ in $t$ with $\Phi'(t) = A e^{tA} = e^{tA} A$ and $\|\Phi(t) - I - tA\| \leq \|A\|^2 t^2/2 \cdot e^{\|A\| t}$ for $t \geq 0$.

**E0.3.2.6 (Constrained linearization).** Let $g: U \to \mathbb{R}^k$ be $C^1$ with rank $k$ everywhere. Show that the level set $M = \{g = 0\}$ has well-defined tangent spaces $T_\mathbf{x} M = \ker Dg(\mathbf{x})$ at each point. (Proof deferred to IFT in Topic 0.3.4; work out the tangent-space picture here.)

**E0.3.2.7 (Taylor for matrix inverse).** For invertible $A$ and small $H$:
$$(A + H)^{-1} = A^{-1} - A^{-1} H A^{-1} + A^{-1} H A^{-1} H A^{-1} - \cdots$$
Prove this via Neumann series (Topic 0.2.9). Interpret as Taylor expansion in $H$.

**E0.3.2.8.** Prove: if $f: \mathbb{R}^n \to \mathbb{R}$ is $C^2$ and $H_f(\mathbf{x}) \succeq 0$ for every $\mathbf{x}$, then $f$ is convex (i.e., $f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})$ for all $\lambda \in [0,1]$).

*Hint.* The restriction to a line is a one-variable $C^2$ function with nonneg second derivative; hence convex; hence $f$ convex.

#### ★★★ (Challenge)

**E0.3.2.9 (Taylor with integral remainder via IBP).** Derive Theorem 0.3.2.7 from Theorem 0.3.2.6 using integration by parts $k$ times.

**E0.3.2.10 (Convergence rate of Newton's method).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be $C^3$ with minimum at $\mathbf{x}^*$, $H_f(\mathbf{x}^*) \succ 0$. Show: for $\mathbf{x}_k$ close enough to $\mathbf{x}^*$, Newton's iteration satisfies
$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C \|\mathbf{x}_k - \mathbf{x}^*\|^2$$
for some constant $C$ depending on $f$.

*Hint.* Use Taylor to write $\nabla f(\mathbf{x}_k) = H_f(\mathbf{x}^*)(\mathbf{x}_k - \mathbf{x}^*) + R$ with $R = O(\|\mathbf{x}_k - \mathbf{x}^*\|^2)$, and similarly for $H_f(\mathbf{x}_k)$.

**E0.3.2.11 (Faà di Bruno's formula).** Derive the combinatorial formula for the $k$-th derivative of a composition $g \circ f$ in one variable; state (without full proof) the multivariate analogue.

---

## Topic 0.3.3 — The Inverse Function Theorem

### Motivation

If $f: \mathbb{R}^n \to \mathbb{R}^n$ is $C^1$ and $Df(\mathbf{x}_0)$ is invertible, then *locally* near $\mathbf{x}_0$, $f$ has a $C^1$ inverse. The linearization being invertible propagates to the nonlinear map. This is the **Inverse Function Theorem (IFT)** — arguably the single most important theorem in this module.

Consequences / applications:

- **Implicit Function Theorem** (next topic) is an immediate corollary.
- **Change of variables** in integration (Topic 0.3.6) requires local diffeomorphism, which IFT provides.
- **Submersion and immersion theorems** (manifold theory) follow from IFT.
- **Existence of solutions to nonlinear systems**: if you have a good starting guess, Newton's method converges — and IFT is the theoretical guarantee that local inversion is well-defined.
- **Calibration** in finance: given market prices $\mathbf{p} = f(\boldsymbol\theta)$ for model parameters $\boldsymbol\theta$, IFT gives a local unique inverse $\boldsymbol\theta(\mathbf{p})$, guaranteeing the calibration problem is well-posed near any regular point.

The proof is a beautiful application of Banach's fixed-point theorem — it's the canonical example of how "nonlinear problem, contractible iteration" can be done rigorously.

### Prerequisites

Completeness of $\mathbb{R}^n$, operator norms (Topic 0.2.9), Fréchet differentiability (Topic 0.3.1), chain rule (Topic 0.3.2).

### Banach's Fixed-Point Theorem

We first prove the central analytic tool.

**Definition 0.3.3.1.** Let $(M, d)$ be a metric space. A map $T: M \to M$ is a **contraction** (with constant $c \in [0, 1)$) if
$$d(T(x), T(y)) \leq c\, d(x, y) \quad \text{for all } x, y \in M.$$

**Theorem 0.3.3.2 (Banach fixed-point theorem).** Let $(M, d)$ be a **complete** metric space and $T: M \to M$ a contraction with constant $c$. Then:

(i) $T$ has a unique fixed point $x^* \in M$ (i.e., $T(x^*) = x^*$).

(ii) For any $x_0 \in M$, the sequence $x_{n+1} = T(x_n)$ converges to $x^*$.

(iii) $d(x_n, x^*) \leq \frac{c^n}{1 - c} d(x_0, x_1)$ (geometric convergence).

*Proof.*

*Step 1. Cauchy sequence.* $d(x_{n+1}, x_n) = d(T(x_n), T(x_{n-1})) \leq c \cdot d(x_n, x_{n-1})$. By induction, $d(x_{n+1}, x_n) \leq c^n d(x_1, x_0)$.

For $m > n$:
$$d(x_m, x_n) \leq \sum_{k=n}^{m-1} d(x_{k+1}, x_k) \leq \sum_{k=n}^{m-1} c^k d(x_1, x_0) \leq \frac{c^n}{1-c} d(x_1, x_0).$$

As $n \to \infty$, $c^n \to 0$, so $\{x_n\}$ is Cauchy.

*Step 2. Limit and fixed-point property.* Since $M$ is complete, $x_n \to x^*$ for some $x^* \in M$. By continuity of $T$ (contractions are Lipschitz, hence continuous), $T(x^*) = T(\lim x_n) = \lim T(x_n) = \lim x_{n+1} = x^*$.

*Step 3. Uniqueness.* If $T(y^*) = y^*$ and $T(x^*) = x^*$, then $d(x^*, y^*) = d(T(x^*), T(y^*)) \leq c\, d(x^*, y^*)$. Since $c < 1$, $d(x^*, y^*) = 0$, i.e., $x^* = y^*$.

*Step 4. Error bound.* From Step 1, letting $m \to \infty$: $d(x_n, x^*) \leq c^n d(x_1, x_0)/(1 - c)$. $\square$

### The Inverse Function Theorem

**Theorem 0.3.3.3 (Inverse Function Theorem).** Let $U \subseteq \mathbb{R}^n$ open, $f: U \to \mathbb{R}^n$ be $C^1$. Suppose $\mathbf{a} \in U$ and $Df(\mathbf{a})$ is invertible. Then there exist open neighborhoods $V \subseteq U$ of $\mathbf{a}$ and $W$ of $f(\mathbf{a})$ such that:

1. $f|_V: V \to W$ is a bijection.
2. The inverse $f^{-1}: W \to V$ is $C^1$.
3. For each $\mathbf{y} \in W$, $D(f^{-1})(\mathbf{y}) = Df(f^{-1}(\mathbf{y}))^{-1}$.

Moreover, if $f$ is $C^k$ for $k \geq 1$, so is $f^{-1}$.

*Proof.* Without loss of generality, by translating, assume $\mathbf{a} = \mathbf{0}$ and $f(\mathbf{0}) = \mathbf{0}$. Let $A = Df(\mathbf{0})$; by hypothesis $A$ is invertible. By composing with $A^{-1}$, i.e., considering $\tilde f = A^{-1} \circ f$ instead of $f$, we may further assume $Df(\mathbf{0}) = I$.

*Step 1. Setup for fixed point.* Given a target $\mathbf{y}$ near $\mathbf{0}$, we want to solve $f(\mathbf{x}) = \mathbf{y}$ for $\mathbf{x}$ near $\mathbf{0}$. Rewrite as fixed-point equation:
$$\mathbf{x} = \mathbf{x} - f(\mathbf{x}) + \mathbf{y} =: T_\mathbf{y}(\mathbf{x}).$$
If we can show $T_\mathbf{y}$ is a contraction on a suitable closed set, Banach's theorem gives existence and uniqueness.

*Step 2. Contraction bound.* Let $g(\mathbf{x}) = \mathbf{x} - f(\mathbf{x})$. Then $Dg(\mathbf{x}) = I - Df(\mathbf{x})$, so $Dg(\mathbf{0}) = I - I = 0$.

By continuity of $Df$, there exists $r > 0$ with $\bar B_r(\mathbf{0}) \subseteq U$ and $\|Dg(\mathbf{x})\|_{\mathrm{op}} \leq 1/2$ for all $\mathbf{x} \in \bar B_r(\mathbf{0})$.

By the mean value inequality (Theorem 0.3.2.2) on the convex ball:
$$\|g(\mathbf{x}_1) - g(\mathbf{x}_2)\| \leq \tfrac{1}{2}\|\mathbf{x}_1 - \mathbf{x}_2\|, \quad \mathbf{x}_1, \mathbf{x}_2 \in \bar B_r(\mathbf{0}).$$

Equivalently:
$$\|T_\mathbf{y}(\mathbf{x}_1) - T_\mathbf{y}(\mathbf{x}_2)\| = \|g(\mathbf{x}_1) - g(\mathbf{x}_2)\| \leq \tfrac{1}{2}\|\mathbf{x}_1 - \mathbf{x}_2\|. \quad (*)$$

So $T_\mathbf{y}$ is a $\tfrac{1}{2}$-contraction.

*Step 3. $T_\mathbf{y}$ maps $\bar B_r(\mathbf{0})$ into itself for $\mathbf{y} \in \bar B_{r/2}(\mathbf{0})$.*

For $\mathbf{x} \in \bar B_r(\mathbf{0})$ and $\|\mathbf{y}\| \leq r/2$:
$$\|T_\mathbf{y}(\mathbf{x})\| \leq \|g(\mathbf{x}) - g(\mathbf{0})\| + \|g(\mathbf{0})\| + \|\mathbf{y}\| \leq \tfrac{1}{2}\|\mathbf{x}\| + 0 + r/2 \leq r/2 + r/2 = r.$$
(Using $g(\mathbf{0}) = \mathbf{0} - f(\mathbf{0}) = \mathbf{0}$.)

*Step 4. Apply Banach.* For each $\mathbf{y} \in \bar B_{r/2}(\mathbf{0})$, $T_\mathbf{y}: \bar B_r(\mathbf{0}) \to \bar B_r(\mathbf{0})$ is a contraction on a complete metric space (closed ball in $\mathbb{R}^n$). So it has a unique fixed point $\mathbf{x}^* = \mathbf{x}^*(\mathbf{y}) \in \bar B_r(\mathbf{0})$ with $f(\mathbf{x}^*) = \mathbf{y}$.

Define $f^{-1}(\mathbf{y}) = \mathbf{x}^*(\mathbf{y})$ for $\mathbf{y} \in \bar B_{r/2}(\mathbf{0})$.

*Step 5. $f^{-1}$ is Lipschitz.* For $\mathbf{y}_1, \mathbf{y}_2 \in \bar B_{r/2}$ with inverses $\mathbf{x}_1, \mathbf{x}_2$:
$$\mathbf{x}_1 - \mathbf{x}_2 = (\mathbf{x}_1 - f(\mathbf{x}_1)) - (\mathbf{x}_2 - f(\mathbf{x}_2)) + (\mathbf{y}_1 - \mathbf{y}_2) = g(\mathbf{x}_1) - g(\mathbf{x}_2) + (\mathbf{y}_1 - \mathbf{y}_2).$$

Taking norms and using $(*)$:
$$\|\mathbf{x}_1 - \mathbf{x}_2\| \leq \tfrac{1}{2}\|\mathbf{x}_1 - \mathbf{x}_2\| + \|\mathbf{y}_1 - \mathbf{y}_2\|,$$
so $\|\mathbf{x}_1 - \mathbf{x}_2\| \leq 2\|\mathbf{y}_1 - \mathbf{y}_2\|$. Hence $f^{-1}$ is 2-Lipschitz.

*Step 6. Open neighborhoods.* Take $V = f^{-1}(B_{r/2}(\mathbf{0})) \cap B_r(\mathbf{0})$ (open, by continuity of $f$ and by the definition), $W = f(V)$. (Claim: $W$ is open — by continuity of $f^{-1}$ on $\bar B_{r/2}$, $W = (f^{-1})^{-1}(V) \cap B_{r/2}(\mathbf{0})$, open.) Then $f|_V: V \to W$ is a bijection.

*Step 7. Differentiability of $f^{-1}$.* Fix $\mathbf{y}_0 \in W$, with $\mathbf{x}_0 = f^{-1}(\mathbf{y}_0)$. Set $B = Df(\mathbf{x}_0)$; invertible (by continuity of $Df$ and invertibility at $\mathbf{0}$ — shrink $r$ if needed to ensure $Df$ invertible throughout $\bar B_r$).

We claim $D(f^{-1})(\mathbf{y}_0) = B^{-1}$.

Let $\mathbf{y} = \mathbf{y}_0 + \mathbf{k}$ and $\mathbf{x} = f^{-1}(\mathbf{y})$. Then $\mathbf{x} - \mathbf{x}_0 =: \mathbf{h}$ and $\mathbf{y} - \mathbf{y}_0 = f(\mathbf{x}) - f(\mathbf{x}_0) = B\mathbf{h} + r(\mathbf{h})$ with $r(\mathbf{h}) = o(\|\mathbf{h}\|)$.

From $\mathbf{k} = B\mathbf{h} + r(\mathbf{h})$: $\mathbf{h} = B^{-1}\mathbf{k} - B^{-1} r(\mathbf{h})$.

We need to show $\mathbf{h} = B^{-1} \mathbf{k} + o(\|\mathbf{k}\|)$. That is, $B^{-1} r(\mathbf{h}) = o(\|\mathbf{k}\|)$.

From Lipschitz $f^{-1}$: $\|\mathbf{h}\| \leq 2\|\mathbf{k}\|$. As $\mathbf{k} \to 0$, $\mathbf{h} \to 0$; $\|r(\mathbf{h})\| = o(\|\mathbf{h}\|) = o(\|\mathbf{k}\|)$; $\|B^{-1} r(\mathbf{h})\| \leq \|B^{-1}\| \cdot o(\|\mathbf{k}\|) = o(\|\mathbf{k}\|)$. ✓

So $D(f^{-1})(\mathbf{y}_0) = B^{-1} = [Df(f^{-1}(\mathbf{y}_0))]^{-1}$.

*Step 8. Continuous differentiability.* The map $\mathbf{y} \mapsto D(f^{-1})(\mathbf{y}) = [Df(f^{-1}(\mathbf{y}))]^{-1}$ is a composition: $\mathbf{y} \to f^{-1}(\mathbf{y}) \to Df(f^{-1}(\mathbf{y})) \to [Df(f^{-1}(\mathbf{y}))]^{-1}$. Each step is continuous ($f^{-1}$ continuous by Step 5, $Df$ continuous by hypothesis, matrix inversion continuous on $GL_n(\mathbb{R})$ by Cramer's rule or Neumann series). So $f^{-1}$ is $C^1$.

*Step 9. Higher regularity.* If $f$ is $C^k$, $Df$ is $C^{k-1}$, so $D(f^{-1}) = (Df)^{-1} \circ f^{-1}$ is the composition of a $C^{k-1}$ map with $f^{-1}$ (which we're inductively establishing is $C^{k-1}$). Induction: $f^{-1}$ is $C^k$. $\square$

### The Derivative Formula

**Corollary 0.3.3.4.** When $f$ is invertible with $C^1$ inverse:
$$D(f^{-1})(\mathbf{y}) = [Df(f^{-1}(\mathbf{y}))]^{-1}.$$

*Proof.* From $f^{-1} \circ f = \mathrm{id}$, differentiate via chain rule: $D(f^{-1})(f(\mathbf{x})) \cdot Df(\mathbf{x}) = I$, so $D(f^{-1})(f(\mathbf{x})) = [Df(\mathbf{x})]^{-1}$. Substituting $\mathbf{x} = f^{-1}(\mathbf{y})$ gives the formula. $\square$

### Diffeomorphisms and Global Inversion

**Definition 0.3.3.5.** A $C^k$-**diffeomorphism** from $U$ to $V$ is a bijection $f: U \to V$ that is $C^k$ with $C^k$ inverse. **IFT gives local diffeomorphisms at points where the derivative is invertible.**

**Warning: local ≠ global.** A map can be a local diffeomorphism everywhere yet not injective globally. Classic example: $f(x, y) = (e^x \cos y, e^x \sin y)$, the complex exponential in real form. $\det J_f = e^{2x} > 0$ everywhere, so $Df$ is invertible everywhere. But $f(x, y + 2\pi) = f(x, y)$ — not injective. IFT only gives a local inverse.

**Theorem 0.3.3.6 (Hadamard's global inverse).** If $f: \mathbb{R}^n \to \mathbb{R}^n$ is $C^1$, $Df(\mathbf{x})$ is invertible for every $\mathbf{x}$, *and* $\|f(\mathbf{x})\| \to \infty$ as $\|\mathbf{x}\| \to \infty$ (proper), then $f$ is a global diffeomorphism.

(Proof via covering spaces and properness; see Milnor's *Topology from the Differentiable Viewpoint*.)

### Worked Examples

**Example 0.3.3.7 (Polar coordinates).** $f(r, \theta) = (r\cos\theta, r\sin\theta)$. Jacobian $J_f = \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta\end{pmatrix}$, $\det = r$. Invertible iff $r \neq 0$. So polar coordinates are a local diffeomorphism on $\{r > 0\}$. But they're not injective globally (periodicity in $\theta$). IFT gives local inverses.

**Example 0.3.3.8 (Nonlinear system).** Consider $f: \mathbb{R}^2 \to \mathbb{R}^2$,
$$f(x, y) = (x + \sin(xy),\, y + \cos(xy)).$$
At $(0, 0)$: $f(0, 0) = (0, 1)$, and $Df(0, 0) = I$. IFT: there is a neighborhood of $(0, 1)$ on which $f^{-1}$ exists and is $C^\infty$. Concretely: for $\mathbf{y}$ close to $(0, 1)$, the equation $f(\mathbf{x}) = \mathbf{y}$ has a unique solution $\mathbf{x}$ near $\mathbf{0}$, computable by Newton iteration with quadratic convergence.

**Example 0.3.3.9 (Change of variable in probability).** If $Y = f(X)$ where $X$ has density $p_X$ on $\mathbb{R}^n$ and $f$ is a $C^1$ bijection, then $Y$ has density
$$p_Y(\mathbf{y}) = p_X(f^{-1}(\mathbf{y})) \cdot |\det J_{f^{-1}}(\mathbf{y})| = \frac{p_X(f^{-1}(\mathbf{y}))}{|\det J_f(f^{-1}(\mathbf{y}))|}.$$
This is the **change of variables formula** for densities. IFT guarantees $f^{-1}$ exists locally; Theorem 0.3.6 (Topic 0.3.6) establishes the full formula.

### Computational Implementation

```python
import numpy as np
from numpy.linalg import solve, norm

def newton_solve(f, Jf, y, x0, tol=1e-12, max_iter=50):
    """Solve f(x) = y starting from x0 using Newton's method."""
    x = x0.copy()
    for k in range(max_iter):
        r = f(x) - y
        if norm(r) < tol:
            return x, k
        dx = solve(Jf(x), -r)
        x = x + dx
    return x, max_iter

# Example 0.3.3.8: invert f locally near (0,0) -> (0,1)
def f(x):
    a, b = x
    return np.array([a + np.sin(a*b), b + np.cos(a*b)])

def Jf(x):
    a, b = x
    c = np.cos(a*b); s = np.sin(a*b)
    return np.array([[1 + b*c, a*c],
                     [-b*s, 1 - a*s]])

# Solve f(x) = (0.1, 1.05)
target = np.array([0.1, 1.05])
x_inv, iters = newton_solve(f, Jf, target, x0=np.array([0., 1.]))
print(f"Inverse after {iters} iters: {x_inv}, residual: {f(x_inv) - target}")

# Verify D(f^-1)(y) = Df(x)^(-1)
Df_at_x = Jf(x_inv)
Df_inv = solve(Df_at_x, np.eye(2))
# Numerical Jacobian of f^-1 via finite differences
eps = 1e-5
Jfi_num = np.zeros((2, 2))
for j in range(2):
    e = np.zeros(2); e[j] = eps
    xp, _ = newton_solve(f, Jf, target + e, x0=x_inv)
    xm, _ = newton_solve(f, Jf, target - e, x0=x_inv)
    Jfi_num[:, j] = (xp - xm) / (2*eps)
print("D(f^-1) analytic:", Df_inv.flatten())
print("D(f^-1) numeric: ", Jfi_num.flatten())
print("Difference:", norm(Df_inv - Jfi_num))

# Local vs global: polar coordinates
def polar_to_cart(r_theta):
    r, theta = r_theta
    return np.array([r*np.cos(theta), r*np.sin(theta)])

print("f(1, 0) =", polar_to_cart([1, 0]))
print("f(1, 2*pi) =", polar_to_cart([1, 2*np.pi]))
# Same output — not globally injective.
```

### [QUANT APPLICATION] — Model Calibration and the Well-Posedness of Implied Parameters

**(A) Calibration as inversion.** Given a pricing model with parameters $\boldsymbol\theta$ (e.g., Black-Scholes with $\sigma$; Heston with $(\kappa, \theta, \sigma, \rho, v_0)$), the model outputs prices $\mathbf{p} = M(\boldsymbol\theta)$ for a vector of instruments (e.g., option prices at various strikes/maturities). Market calibration: given observed $\mathbf{p}_{\mathrm{mkt}}$, find $\boldsymbol\theta$ with $M(\boldsymbol\theta) = \mathbf{p}_{\mathrm{mkt}}$.

IFT says: **if the Jacobian $DM(\boldsymbol\theta_0)$ has full rank at a calibrated point $\boldsymbol\theta_0$, then calibration is a well-defined locally unique inverse.** The Jacobian here is the matrix of *model sensitivities* of prices to parameters — closely related to Greeks.

**(B) Implied volatility.** The Black-Scholes price $C_{BS}(S, K, T, r, \sigma)$ is strictly increasing in $\sigma$ (vega $\partial C/\partial \sigma > 0$). IFT (scalar version): for each market price $C_{\mathrm{mkt}}$ in the valid range, there is a unique $\sigma^* = \sigma_{\mathrm{impl}}(C_{\mathrm{mkt}})$ — implied volatility exists and is differentiable.

**(C) Newton's method for calibration.** Once IFT guarantees the inverse exists, Newton's method computes it:
$$\boldsymbol\theta_{k+1} = \boldsymbol\theta_k - [DM(\boldsymbol\theta_k)]^{-1} (M(\boldsymbol\theta_k) - \mathbf{p}_{\mathrm{mkt}}),$$
converging quadratically near the solution. For over-determined systems (more market instruments than parameters), one uses Levenberg-Marquardt with the pseudoinverse. The *local* well-posedness comes from IFT; *global* identifiability is harder (example: SABR has partial identifiability issues).

**(D) Differentiating through calibrated parameters.** If market conditions change by $d\mathbf{p}_{\mathrm{mkt}}$, IFT's derivative formula says
$$d\boldsymbol\theta^* = [DM(\boldsymbol\theta^*)]^{-1} d\mathbf{p}_{\mathrm{mkt}}.$$
This lets you compute risk sensitivities with respect to *market observations* rather than model parameters — crucial for hedging portfolios with market-consistent Greeks.

### Exercises

#### ★ (Foundation)

**E0.3.3.1.** Verify that $f(x) = x + x^3$ is a global diffeomorphism of $\mathbb{R}$ (by IFT pointwise and a direct argument for globality).

**E0.3.3.2.** For $f(x, y) = (x + y, xy)$, at which points does IFT apply? Find and describe the inverse near a point where it does.

**E0.3.3.3.** Prove: if $f: U \to V$ is a $C^1$ bijection with $Df(\mathbf{x})$ invertible for every $\mathbf{x}$, then $f^{-1}$ is $C^1$. (No need for open/closed subtlety here since bijectivity is given.)

#### ★★ (Intermediate)

**E0.3.3.4 (IFT in Banach spaces).** State the IFT in Banach spaces (both $X, Y$ Banach, $Df(\mathbf{a})$ a bounded invertible linear map). Which step of the proof uses completeness of $Y$? Which uses completeness of $X$?

*Commentary.* The Banach IFT is the backbone of infinite-dimensional analysis (PDE, calculus of variations).

**E0.3.3.5 (Quantitative IFT).** Refine Theorem 0.3.3.3: show that if $\|Df(\mathbf{x}) - A\|_{\mathrm{op}} \leq \delta \|A^{-1}\|^{-1}/2$ for all $\mathbf{x}$ in a ball $B_r(\mathbf{a})$ with $A = Df(\mathbf{a})$, then $f$ is invertible on a ball of radius proportional to $r$ around $\mathbf{a}$ with quantitative Lipschitz estimates.

**E0.3.3.6 (Kantorovich's theorem).** Read and understand the statement of Kantorovich's theorem (a quantitative version of Newton's method convergence): given $\|[Df(\mathbf{x}_0)]^{-1}\|, \|[Df(\mathbf{x}_0)]^{-1} f(\mathbf{x}_0)\|$, and a Lipschitz bound on $Df$, conditions on these quantities guarantee Newton converges quadratically to a root. Compare to the IFT.

#### ★★★ (Challenge)

**E0.3.3.7 (Failure of IFT at degeneracy).** Consider $f(x) = x^3$. Show $f'(0) = 0$, IFT does not apply, yet $f$ is a bijection of $\mathbb{R}$ with inverse $f^{-1}(y) = y^{1/3}$ which is *not* differentiable at $0$. Conclusion: IFT's hypothesis of invertible derivative is not necessary for bijectivity, but it **is** necessary for smooth invertibility.

**E0.3.3.8 (Nash–Moser).** Read the statement of the Nash–Moser inverse function theorem (used for smooth inverses in non-Banach scales, e.g., isometric embedding problems). This is far outside the scope of this module, but good to know it exists.

**E0.3.3.9 (IFT and ODE solutions).** Use IFT to show: given $F: \mathbb{R} \times \mathbb{R}^n \to \mathbb{R}^n$ $C^1$ with $F(0, \mathbf{x}) = \mathbf{x}$, for each initial condition $\mathbf{x}_0$ there is a local $C^1$ flow $\varphi_t(\mathbf{x}_0)$ satisfying $\partial_t \varphi_t = F(t, \varphi_t)$, smoothly depending on $\mathbf{x}_0$.

---


## Topic 0.3.4 — Implicit Function Theorem

### Motivation

The **Inverse Function Theorem** answers: when can we solve $f(\mathbf{x}) = \mathbf{y}$ for $\mathbf{x}$ as a function of $\mathbf{y}$? The **Implicit Function Theorem (ImFT)** answers a richer question: given a relation $F(\mathbf{x}, \mathbf{y}) = \mathbf{0}$, when can we solve for $\mathbf{y}$ as a function of $\mathbf{x}$ locally?

This is the *central* theorem of smooth geometry, differential topology, optimization, equilibrium theory, and comparative statics. Whenever an object is defined by an equation $F = 0$ rather than by an explicit formula, the ImFT tells you whether the object is well-defined as a function (locally) and how it varies with parameters.

Examples pervading quantitative work:
- **Yield curve construction.** Bond prices $P_i$ are given; spot rates $r(T_i)$ are defined implicitly by $P_i = \sum e^{-r(T_j) T_j} c_j$. ImFT guarantees rates vary smoothly with prices.
- **Market equilibrium.** The price vector $\mathbf{p}^*$ solving $\mathrm{Supply}(\mathbf{p}) = \mathrm{Demand}(\mathbf{p}, \boldsymbol\alpha)$ depends on exogenous parameters $\boldsymbol\alpha$ (tax rate, income). ImFT's formula gives comparative-statics derivatives $\partial \mathbf{p}^*/\partial \boldsymbol\alpha$ without ever solving the equilibrium explicitly.
- **Calibrated volatility surfaces.** The local-vol function $\sigma_{\mathrm{loc}}(S, t)$ solving Dupire's equation is defined *implicitly* from market option prices; ImFT says it varies smoothly with the inputs.
- **Optimality conditions.** The KKT system for a constrained optimum is $F(\mathbf{x}^*, \boldsymbol\lambda^*, \boldsymbol\alpha) = 0$ where $\boldsymbol\alpha$ is a parameter vector (weights, bounds, ...). ImFT gives the *envelope theorem* and sensitivity of optimizers.

### Prerequisites

- Inverse Function Theorem (Topic 0.3.3).
- Fréchet derivatives and chain rule (Topics 0.3.1, 0.3.2).
- Block matrix inverses (Schur complement, Topic 0.2.7).

### Statement of the Theorem

Write points in $\mathbb{R}^n \times \mathbb{R}^m$ as pairs $(\mathbf{x}, \mathbf{y})$ with $\mathbf{x} \in \mathbb{R}^n$ and $\mathbf{y} \in \mathbb{R}^m$. For a $C^1$ map $F: U \to \mathbb{R}^m$ defined on an open set $U \subseteq \mathbb{R}^n \times \mathbb{R}^m$, its Jacobian $DF$ at $(\mathbf{a}, \mathbf{b})$ decomposes into blocks
$$DF(\mathbf{a}, \mathbf{b}) = \big[\, D_{\mathbf{x}} F(\mathbf{a}, \mathbf{b}) \;\big|\; D_{\mathbf{y}} F(\mathbf{a}, \mathbf{b}) \,\big],$$
an $m \times n$ block (derivatives with respect to $\mathbf{x}$) followed by an $m \times m$ block (derivatives with respect to $\mathbf{y}$).

**Theorem 0.3.4.1 (Implicit Function Theorem).** *Let $F: U \to \mathbb{R}^m$ be $C^1$ on an open set $U \subseteq \mathbb{R}^n \times \mathbb{R}^m$. Suppose $(\mathbf{a}, \mathbf{b}) \in U$ satisfies*
$$F(\mathbf{a}, \mathbf{b}) = \mathbf{0}, \qquad D_{\mathbf{y}} F(\mathbf{a}, \mathbf{b}) \in \mathbb{R}^{m\times m} \text{ is invertible.}$$
*Then there exist open neighborhoods $V \ni \mathbf{a}$ in $\mathbb{R}^n$ and $W \ni \mathbf{b}$ in $\mathbb{R}^m$ with $V \times W \subseteq U$, and a unique $C^1$ function $\mathbf{g}: V \to W$ such that*
- *$\mathbf{g}(\mathbf{a}) = \mathbf{b}$;*
- *for every $\mathbf{x} \in V$, $\mathbf{y} = \mathbf{g}(\mathbf{x})$ is the **only** solution of $F(\mathbf{x}, \mathbf{y}) = 0$ with $\mathbf{y} \in W$;*
- *the derivative is given by*
$$D\mathbf{g}(\mathbf{x}) = - \big[ D_{\mathbf{y}} F(\mathbf{x}, \mathbf{g}(\mathbf{x})) \big]^{-1} D_{\mathbf{x}} F(\mathbf{x}, \mathbf{g}(\mathbf{x})). \qquad \square$$

*If $F$ is $C^k$ ($k \geq 1$) or $C^\infty$ or real-analytic, then so is $\mathbf{g}$.*

#### The derivative formula, heuristically

Differentiating the identity $F(\mathbf{x}, \mathbf{g}(\mathbf{x})) = 0$ with respect to $\mathbf{x}$ and applying the chain rule:
$$D_{\mathbf{x}} F(\mathbf{x}, \mathbf{g}(\mathbf{x})) + D_{\mathbf{y}} F(\mathbf{x}, \mathbf{g}(\mathbf{x})) \cdot D\mathbf{g}(\mathbf{x}) = 0,$$
and solving for $D\mathbf{g}$ gives the stated formula. This is how one uses the ImFT in practice — the existence is guaranteed; the *computation* of the derivative is routine once you trust it.

### Proof via the Inverse Function Theorem

The ImFT and IFT are equivalent: one implies the other in a few lines. Here we derive ImFT from IFT.

#### Step 1 — The auxiliary map

Define $\Phi: U \to \mathbb{R}^n \times \mathbb{R}^m$ by
$$\Phi(\mathbf{x}, \mathbf{y}) = (\mathbf{x}, F(\mathbf{x}, \mathbf{y})).$$
So $\Phi$ leaves the first $n$ coordinates untouched and replaces the last $m$ by the value of $F$. Clearly $\Phi$ is $C^1$ since $F$ is.

At $(\mathbf{a}, \mathbf{b})$:
$$\Phi(\mathbf{a}, \mathbf{b}) = (\mathbf{a}, F(\mathbf{a}, \mathbf{b})) = (\mathbf{a}, \mathbf{0}).$$

#### Step 2 — Invertibility of $D\Phi(\mathbf{a}, \mathbf{b})$

The Jacobian of $\Phi$ has the block form
$$D\Phi(\mathbf{a}, \mathbf{b}) = \begin{pmatrix} I_n & \mathbf{0} \\ D_{\mathbf{x}} F(\mathbf{a}, \mathbf{b}) & D_{\mathbf{y}} F(\mathbf{a}, \mathbf{b}) \end{pmatrix}.$$
This is a block lower-triangular matrix; its determinant is $\det(I_n) \det(D_{\mathbf{y}} F(\mathbf{a}, \mathbf{b})) = \det(D_{\mathbf{y}} F(\mathbf{a}, \mathbf{b})) \neq 0$ by hypothesis. Therefore $D\Phi(\mathbf{a}, \mathbf{b})$ is invertible.

#### Step 3 — Apply IFT to $\Phi$

By Theorem 0.3.3.3 (IFT), there are open neighborhoods $U_0 \ni (\mathbf{a}, \mathbf{b})$ and $\tilde V \ni (\mathbf{a}, \mathbf{0})$ such that $\Phi: U_0 \to \tilde V$ is a $C^1$-diffeomorphism. Write its inverse as
$$\Phi^{-1}(\mathbf{x}, \mathbf{z}) = (\mathbf{x}, H(\mathbf{x}, \mathbf{z}))$$
for some $C^1$ function $H: \tilde V \to \mathbb{R}^m$. (The first component of $\Phi^{-1}$ is forced to be $\mathbf{x}$ because the first component of $\Phi$ is the identity; the second component, which we call $H$, is defined by the requirement that $\Phi(\Phi^{-1}) = \mathrm{id}$.)

#### Step 4 — Define the implicit function

By shrinking if necessary, choose open sets $V \ni \mathbf{a}$ in $\mathbb{R}^n$ and $W \ni \mathbf{b}$ in $\mathbb{R}^m$ such that $V \times W \subseteq U_0$ and $V \times \{\mathbf{0}\} \subseteq \tilde V$. Define
$$\mathbf{g}(\mathbf{x}) := H(\mathbf{x}, \mathbf{0}) \quad \text{for } \mathbf{x} \in V.$$
This is $C^1$ as a composition of $C^1$ functions, and $\mathbf{g}(\mathbf{a}) = H(\mathbf{a}, \mathbf{0}) = \mathbf{b}$ because $\Phi^{-1}(\mathbf{a}, \mathbf{0}) = (\mathbf{a}, \mathbf{b})$.

#### Step 5 — Verify $F(\mathbf{x}, \mathbf{g}(\mathbf{x})) = 0$

For $\mathbf{x} \in V$:
$$\Phi(\mathbf{x}, \mathbf{g}(\mathbf{x})) = \Phi(\Phi^{-1}(\mathbf{x}, \mathbf{0})) = (\mathbf{x}, \mathbf{0}).$$
The second component of $\Phi(\mathbf{x}, \mathbf{g}(\mathbf{x}))$ is $F(\mathbf{x}, \mathbf{g}(\mathbf{x}))$, so $F(\mathbf{x}, \mathbf{g}(\mathbf{x})) = \mathbf{0}$ as required.

#### Step 6 — Uniqueness in $W$

Suppose $\mathbf{x} \in V$ and $\mathbf{y} \in W$ with $F(\mathbf{x}, \mathbf{y}) = 0$. Then $\Phi(\mathbf{x}, \mathbf{y}) = (\mathbf{x}, 0) = \Phi(\mathbf{x}, \mathbf{g}(\mathbf{x}))$. Since $\Phi$ is injective on $U_0 \supseteq V \times W$, we conclude $\mathbf{y} = \mathbf{g}(\mathbf{x})$.

#### Step 7 — The derivative formula

Since $F(\mathbf{x}, \mathbf{g}(\mathbf{x})) = 0$ on $V$, differentiate both sides with respect to $\mathbf{x}$ using the multivariate chain rule (Theorem 0.3.2.1 applied to $\mathbf{x} \mapsto (\mathbf{x}, \mathbf{g}(\mathbf{x})) \mapsto F(\mathbf{x}, \mathbf{g}(\mathbf{x}))$):
$$D_{\mathbf{x}} F(\mathbf{x}, \mathbf{g}(\mathbf{x})) \cdot I_n + D_{\mathbf{y}} F(\mathbf{x}, \mathbf{g}(\mathbf{x})) \cdot D\mathbf{g}(\mathbf{x}) = 0.$$
Since $D_{\mathbf{y}} F(\mathbf{a}, \mathbf{b})$ is invertible and $DF$ is continuous, $D_{\mathbf{y}} F(\mathbf{x}, \mathbf{g}(\mathbf{x}))$ remains invertible in a neighborhood of $\mathbf{a}$ (invertibility is an open condition on $GL_m(\mathbb{R})$). Shrink $V$ if needed so this holds throughout $V$. Then
$$D\mathbf{g}(\mathbf{x}) = - [D_{\mathbf{y}} F(\mathbf{x}, \mathbf{g}(\mathbf{x}))]^{-1} D_{\mathbf{x}} F(\mathbf{x}, \mathbf{g}(\mathbf{x})). \qquad \blacksquare$$

### The Jacobian matrix in coordinates

Writing $F = (F_1, \ldots, F_m)$, $\mathbf{x} = (x_1, \ldots, x_n)$, $\mathbf{y} = (y_1, \ldots, y_m)$:

- $D_{\mathbf{x}} F$ is the $m \times n$ matrix $[\partial F_i/\partial x_j]_{i,j}$.
- $D_{\mathbf{y}} F$ is the $m \times m$ matrix $[\partial F_i/\partial y_k]_{i,k}$.
- $D\mathbf{g}$ is the $m \times n$ matrix $[\partial g_i/\partial x_j]$.

The derivative formula reads componentwise:
$$\frac{\partial g_i}{\partial x_j} = - \sum_{k=1}^m \Big([D_{\mathbf{y}} F]^{-1}\Big)_{ik} \frac{\partial F_k}{\partial x_j}.$$

#### Scalar case ($n = m = 1$)

If $F(x, y) = 0$ defines $y = g(x)$, and $\partial F/\partial y \neq 0$, then
$$g'(x) = -\frac{\partial F/\partial x}{\partial F/\partial y}.$$
This is the formula every calculus student learns (often rotely; the ImFT says *why* it's legitimate).

### Worked Examples

#### Example 0.3.4.1 — The unit circle

Let $F(x, y) = x^2 + y^2 - 1$. Fix a point $(a, b)$ on the circle ($F(a,b) = 0$).

$\partial F/\partial y = 2y$. This is invertible ($\neq 0$) iff $b \neq 0$, i.e., we are *not* at the left or right extreme points $(\pm 1, 0)$.

For $b \neq 0$, ImFT gives a local $C^\infty$ function $y = g(x)$ with $g(a) = b$ (explicitly $g(x) = \pm\sqrt{1-x^2}$ with the sign matching $b$), and
$$g'(x) = -\frac{\partial F/\partial x}{\partial F/\partial y} = -\frac{2x}{2y} = -\frac{x}{g(x)},$$
which agrees with direct differentiation of $\pm\sqrt{1-x^2}$.

At $(\pm 1, 0)$ the theorem fails: near these points one cannot write $y$ as a function of $x$ — vertical tangent. (However, one *can* write $x$ as a function of $y$ there, by swapping the roles in ImFT.)

#### Example 0.3.4.2 — A nonlinear system

Solve $F(x, y, u, v) = 0$ where $F: \mathbb{R}^2 \times \mathbb{R}^2 \to \mathbb{R}^2$,
$$F_1(x, y, u, v) = u^2 + v^2 + x - 2, \qquad F_2(x, y, u, v) = uv + y - 1,$$
near the point $(x, y, u, v) = (0, 0, 1, 1)$, where indeed $F_1 = 1 + 1 + 0 - 2 = 0$ and $F_2 = 1 + 0 - 1 = 0$.

The Jacobian blocks at this point:
$$D_{\mathbf{y}} F = \begin{pmatrix} \partial F_1/\partial u & \partial F_1/\partial v \\ \partial F_2/\partial u & \partial F_2/\partial v \end{pmatrix} = \begin{pmatrix} 2u & 2v \\ v & u \end{pmatrix}_{(1,1)} = \begin{pmatrix} 2 & 2 \\ 1 & 1 \end{pmatrix}.$$

$\det = 2\cdot 1 - 2\cdot 1 = 0$. *Uh-oh.* $D_{\mathbf{y}} F$ is not invertible. ImFT fails at this point; we cannot solve for $(u, v)$ as $C^1$ functions of $(x, y)$ near $(1, 1)$.

Let's try a different base point, say $(0, 0, 1, -1)$ (if it satisfies $F = 0$): $F_1 = 1 + 1 + 0 - 2 = 0$, $F_2 = -1 + 0 - 1 = -2 \neq 0$. Not on the zero set.

Try $(x_0, y_0, u_0, v_0) = (0, 0, \sqrt{2}, 0)$: $F_1 = 2 + 0 + 0 - 2 = 0$, $F_2 = 0 + 0 - 1 = -1 \neq 0$. Not on the zero set.

Try $(x_0, y_0, u_0, v_0) = (0, 1, 1, 1)$: $F_1 = 1 + 1 + 0 - 2 = 0$, $F_2 = 1 + 1 - 1 = 1 \neq 0$. Not on the zero set.

Consider instead a base with $u \neq v$ on the zero set. The system $u^2 + v^2 = 2 - x$, $uv = 1 - y$ with $x = y = 0$ gives $u^2 + v^2 = 2$, $uv = 1$, so $(u, v) = (1, 1)$ (double root) or $(-1, -1)$; the solution set is discrete and tangency at double roots causes the Jacobian to drop rank. This is the "bifurcation" phenomenon: the implicit function theorem's failure detects *critical* points of the equation system.

For $x = 0, y = -1/2$: $u^2 + v^2 = 2$, $uv = 3/2$. Then $u, v$ are roots of $t^2 - (u+v)t + 3/2 = 0$ with $u + v = \sqrt{u^2 + v^2 + 2uv} = \sqrt{2 + 3} = \sqrt{5}$ (taking positive sum). So $(u, v) = ((\sqrt{5}+1)/2, (\sqrt{5}-1)/2)$ and permutation. At this point,
$$D_{\mathbf{y}}F = \begin{pmatrix} 2u & 2v \\ v & u \end{pmatrix}, \quad \det = 2(u^2 - v^2) = 2(u-v)(u+v) = 2 \sqrt{5} \cdot 1 = 2\sqrt{5} \neq 0.$$
ImFT applies; locally $(u, v)$ is a $C^\infty$ function of $(x, y)$ with derivative
$$D\mathbf{g}(0, -1/2) = -\frac{1}{\det}\begin{pmatrix} u & -2v \\ -v & 2u \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = -\frac{1}{2\sqrt{5}}\begin{pmatrix} u & -2v \\ -v & 2u \end{pmatrix}.$$

#### Example 0.3.4.3 — Level sets as manifolds

If $F: \mathbb{R}^N \to \mathbb{R}^m$ is $C^1$ and $DF(\mathbf{p})$ has full row rank $m$ at every $\mathbf{p} \in F^{-1}(\mathbf{0})$, then $F^{-1}(\mathbf{0})$ is a **$C^1$-manifold** of dimension $N - m$.

*Proof.* At each $\mathbf{p}$, full rank means we can relabel coordinates $(\mathbf{x}, \mathbf{y})$ with $\mathbf{x} \in \mathbb{R}^{N-m}, \mathbf{y} \in \mathbb{R}^m$ such that $D_{\mathbf{y}} F(\mathbf{p})$ is invertible (pick $m$ independent columns of $DF$ and call those the $\mathbf{y}$-coordinates). ImFT gives a local $C^1$ graph parametrization $\mathbf{y} = \mathbf{g}(\mathbf{x})$ of $F^{-1}(\mathbf{0})$ near $\mathbf{p}$ over an open set $V \subseteq \mathbb{R}^{N-m}$. These graphs constitute a manifold atlas. $\blacksquare$

This is how every algebraic and differential-geometric object you will meet later — the sphere $S^{n-1}$, the orthogonal group $O(n)$, the Stiefel manifold, the Lie group $SL_n(\mathbb{R})$, the solution set of an equilibrium equation — is manufactured.

#### Example 0.3.4.4 — Comparative statics

Let $f: \mathbb{R}^n \times \mathbb{R}^p \to \mathbb{R}$, $(\mathbf{x}, \boldsymbol\alpha) \mapsto f(\mathbf{x}, \boldsymbol\alpha)$, be a $C^2$ objective. For each parameter vector $\boldsymbol\alpha$, suppose $\mathbf{x}^*(\boldsymbol\alpha)$ is an interior maximizer, so the first-order condition is
$$F(\mathbf{x}, \boldsymbol\alpha) := \nabla_{\mathbf{x}} f(\mathbf{x}, \boldsymbol\alpha) = \mathbf{0}.$$

If at $\boldsymbol\alpha_0$ the **Hessian** $D_{\mathbf{x}} F = \nabla_{\mathbf{x}}^2 f$ is invertible (in particular, negative definite at a strict local max), then ImFT says $\mathbf{x}^*(\boldsymbol\alpha)$ exists and is $C^1$ in $\boldsymbol\alpha$ near $\boldsymbol\alpha_0$, with
$$D_{\boldsymbol\alpha} \mathbf{x}^*(\boldsymbol\alpha_0) = -[\nabla_{\mathbf{x}}^2 f]^{-1} \nabla_{\mathbf{x}\boldsymbol\alpha}^2 f,$$
the mixed Hessian. In economics this is called **comparative statics**; in optimization, **sensitivity analysis**; in statistics, **influence functions**. The invertibility of the Hessian is exactly the second-order sufficient condition that also guarantees the extremum is non-degenerate.

### Computational Implementation

```python
import numpy as np
from scipy.optimize import fsolve

# ImFT demonstration: F(x, y) = 0 defines y = g(x)
# Pick F(x, y) where y is (u, v) and x = (p, q)
# Solve x^2 + y^2 + p = 2, xy + q = 1 ... let's use clean notation below.

def F(x, y):
    """F: R^2 x R^2 -> R^2.  x = (p, q) (parameter), y = (u, v) (implicit)."""
    p, q = x
    u, v = y
    return np.array([u**2 + v**2 + p - 2.0, u*v + q - 1.0])

def DF_x(x, y):
    """Partial derivatives of F wrt x."""
    return np.eye(2)  # F_1 has p with coeff 1; F_2 has q with coeff 1.

def DF_y(x, y):
    """Partial derivatives of F wrt y."""
    _, _ = x
    u, v = y
    return np.array([[2*u, 2*v], [v, u]])

# Base point (p, q) = (0, -0.5) where we computed u = (√5+1)/2, v = (√5-1)/2
x0 = np.array([0.0, -0.5])
u0 = (np.sqrt(5) + 1) / 2
v0 = (np.sqrt(5) - 1) / 2
y0 = np.array([u0, v0])

# 1. Verify F(x0, y0) = 0
print("F(x0, y0) =", F(x0, y0))  # should be (0, 0)

# 2. Solve F(x0 + dx, y) = 0 numerically for small dx; compare with ImFT prediction
# ImFT: g(x0 + dx) ≈ y0 + Dg(x0) dx where Dg(x0) = -DF_y^{-1} DF_x
Dg = -np.linalg.solve(DF_y(x0, y0), DF_x(x0, y0))
print("Dg(x0) predicted by ImFT:\n", Dg)

# Numerical verification: perturb x and solve for y
dx = np.array([0.01, 0.02])
x_pert = x0 + dx
y_pert = fsolve(lambda y: F(x_pert, y), y0)
print("y solved numerically:", y_pert)
print("y0 + Dg*dx (linear ImFT):", y0 + Dg @ dx)
print("difference (should be O(|dx|^2)):", np.linalg.norm(y_pert - (y0 + Dg @ dx)))

# 3. Verify chain rule: d/dx F(x, g(x)) = 0
# Use secant approximation of Dg
def solve_g(x, y_guess):
    return fsolve(lambda y: F(x, y), y_guess)

eps = 1e-5
Dg_numeric = np.zeros((2, 2))
for j in range(2):
    e = np.zeros(2)
    e[j] = eps
    Dg_numeric[:, j] = (solve_g(x0 + e, y0) - solve_g(x0 - e, y0)) / (2*eps)
print("Dg(x0) via finite differences:\n", Dg_numeric)
print("max |difference|:", np.max(np.abs(Dg - Dg_numeric)))
```

**Output analysis.** The ImFT prediction and the finite-difference approximation agree to within truncation error $O(\varepsilon^2)$; the second-order residual $y_{\mathrm{num}} - (y_0 + Dg\, dx)$ scales like $\|dx\|^2$ (this is the quadratic Taylor remainder of $g$).

#### A general-purpose ImFT solver / sensitivity engine

```python
def implicit_derivative(F, DF_x, DF_y, x, y):
    """Given a root (x, y) of F(x, y) = 0, return Dg(x) = -DF_y^{-1} DF_x."""
    return -np.linalg.solve(DF_y(x, y), DF_x(x, y))

def implicit_sensitivity(F_fun, x_base, y_base, x_new, jac_fn_y=None):
    """
    Track y along a one-parameter family x(t) = x_base + t*(x_new - x_base), t ∈ [0, 1].
    Returns the trajectory y(t) at a grid of t-values and the ImFT-predicted initial slope.
    """
    from scipy.integrate import solve_ivp
    d = x_new - x_base
    # dy/dt = -DF_y^{-1} DF_x * dx/dt = Dg * d
    def rhs(t, y):
        x_t = x_base + t * d
        return -np.linalg.solve(DF_y(x_t, y), DF_x(x_t, y)) @ d
    sol = solve_ivp(rhs, (0, 1), y_base, dense_output=True, rtol=1e-9, atol=1e-11)
    return sol

# Apply it
sol = implicit_sensitivity(F, x0, y0, x0 + np.array([0.1, -0.05]))
print("y(t=1) via continuation:", sol.y[:, -1])
print("Check F at endpoint:", F(x0 + np.array([0.1, -0.05]), sol.y[:, -1]))
```

This "continuation" method (also called *numerical homotopy*) uses the ImFT derivative formula as the vector field of an ODE whose integration traces the implicit manifold from $(x_0, y_0)$ to the new parameter. It is the industrial method for following calibrated parameters as market conditions change.

### [QUANT APPLICATION] ImFT in Calibration, Risk, and Equilibrium

**(A) The calibration derivative.** Model parameters $\boldsymbol\theta \in \mathbb{R}^p$ are fitted to market prices $\mathbf{p} \in \mathbb{R}^m$ by solving
$$F(\boldsymbol\theta, \mathbf{p}) := M(\boldsymbol\theta) - \mathbf{p} = \mathbf{0}.$$
ImFT (applied to $F$ as a function of $(\mathbf{p}, \boldsymbol\theta)$, treating $\mathbf{p}$ as input and $\boldsymbol\theta$ as implicit) gives
$$D_{\mathbf{p}} \boldsymbol\theta^*(\mathbf{p}) = [DM(\boldsymbol\theta^*)]^{-1}$$
(at square systems where $p = m$). This is exactly the **calibration Jacobian** — how calibrated parameters move when market prices move. It is the central object in *risk bucketing* (decomposing P&L in terms of market moves) and in *counterparty credit valuation adjustment* (CVA) sensitivities.

**(B) Greeks through the calibration.** Let $V(\boldsymbol\theta)$ be a derivative price that depends on calibrated parameters. The sensitivity to a market price $p_i$ is
$$\frac{\partial V}{\partial p_i} = \sum_{k} \frac{\partial V}{\partial \theta_k} \cdot \frac{\partial \theta_k^*}{\partial p_i} = \nabla_{\boldsymbol\theta} V \cdot [DM]^{-1} \mathbf{e}_i.$$
This is called **market-consistent Greeks**: instead of shocking an abstract model parameter (meaningless — who cares about the "SABR $\nu$"?), you shock an observable market instrument. Banks spend millions on the infrastructure to compute these ImFT-derived Greeks efficiently.

**(C) Envelope theorem.** Consider a value function $V(\boldsymbol\alpha) = \max_{\mathbf{x}} f(\mathbf{x}, \boldsymbol\alpha)$ and the maximizer $\mathbf{x}^*(\boldsymbol\alpha)$ assumed $C^1$ via ImFT (from the FOC). Then
$$\nabla_{\boldsymbol\alpha} V(\boldsymbol\alpha) = \nabla_{\boldsymbol\alpha} f(\mathbf{x}^*(\boldsymbol\alpha), \boldsymbol\alpha) + \underbrace{\nabla_{\mathbf{x}} f(\mathbf{x}^*, \boldsymbol\alpha)}_{= 0 \text{ at optimum}} \cdot \nabla_{\boldsymbol\alpha} \mathbf{x}^*(\boldsymbol\alpha) = \nabla_{\boldsymbol\alpha} f(\mathbf{x}^*, \boldsymbol\alpha).$$
The **envelope theorem** says: to differentiate a value function with respect to a parameter, fix the optimizer and differentiate the objective directly — you don't need $\nabla \mathbf{x}^*$. In finance this appears as the Hamilton–Jacobi–Bellman equation of stochastic control.

**(D) Equilibrium models.** In general equilibrium, $\mathbf{p}^*$ solves the market clearing equation $Z(\mathbf{p}, \boldsymbol\alpha) = 0$ (excess demand = 0). ImFT's derivative $D_{\boldsymbol\alpha} \mathbf{p}^* = -[D_{\mathbf{p}} Z]^{-1} D_{\boldsymbol\alpha} Z$ is the **comparative statics** of an equilibrium — how prices move when exogenous variables (policy, preferences, endowments) change. Central to macroeconomics and auction theory.

**(E) Yield curve bootstrap.** Bond prices $B_i(\mathbf{r})$ depend nonlinearly on spot rates $\mathbf{r}$. Bootstrap solves $B_i(\mathbf{r}) = B_i^{\mathrm{mkt}}$ for all $i$ iteratively. ImFT's derivative formula gives $\partial r_j/\partial B_i^{\mathrm{mkt}}$ — the **Key Rate Durations** — quantifying how shocking a single market price propagates through the curve.

**(F) Implicit volatility.** The equation $C_{BS}(S, K, T, r, \sigma) = C_{\mathrm{mkt}}$ defines $\sigma_{\mathrm{impl}}$ as an implicit function; ImFT says $d\sigma_{\mathrm{impl}}/dC_{\mathrm{mkt}} = 1/\mathrm{vega}$, used when converting between price and vol quotes on every options desk.

### Exercises

#### ★ (Foundation)

**E0.3.4.1.** Apply ImFT to $F(x, y) = x^2 y + xy^2 - 2 = 0$ at $(1, 1)$: verify the hypothesis and compute $g'(1)$ for $y$ as a function of $x$.

**E0.3.4.2.** Consider $F(x, y, z) = x^2 + y^2 + z^2 - 3 = 0$ at $(1, 1, 1)$. Which of $x, y, z$ can be solved for as a function of the other two? Write the formula for $\partial z/\partial x$ and $\partial z/\partial y$.

**E0.3.4.3.** Show that the ImFT and IFT are logically equivalent: prove IFT from ImFT. (Hint: for $f: \mathbb{R}^n \to \mathbb{R}^n$, consider $F(\mathbf{x}, \mathbf{y}) = f(\mathbf{y}) - \mathbf{x}$.)

**E0.3.4.4.** The **implicit function theorem for linear systems.** Given a linear system $A\mathbf{x} + B\mathbf{y} = 0$ with $B \in \mathbb{R}^{m\times m}$ invertible, show directly that $\mathbf{y} = -B^{-1} A \mathbf{x}$. Observe that this is exactly the ImFT formula; the theorem is the nonlinear generalization.

#### ★★ (Intermediate)

**E0.3.4.5 (Manifold structure of $O(n)$).** Define $F: \mathbb{R}^{n \times n} \to \mathrm{Sym}_n(\mathbb{R})$, $F(A) = A^\top A - I$. Show $DF(A)[H] = A^\top H + H^\top A$ and compute its image; conclude that $O(n) = F^{-1}(0)$ is a $C^\infty$-manifold of dimension $n(n-1)/2$. (This is the orthogonal group. The analogous argument for $A^\top A - I$ restricted to square matrices with $\det = 1$ gives $SO(n)$.)

**E0.3.4.6 (Level sets and regular values).** Let $F: \mathbb{R}^n \to \mathbb{R}^m$ be $C^1$. A value $\mathbf{c} \in \mathbb{R}^m$ is *regular* if $DF(\mathbf{x})$ has full row rank at every $\mathbf{x} \in F^{-1}(\mathbf{c})$. Prove: if $\mathbf{c}$ is regular, $F^{-1}(\mathbf{c})$ is a $C^1$-manifold of dimension $n - m$. This is **Sard's regular value theorem**; almost every $\mathbf{c}$ is regular (Sard's theorem).

**E0.3.4.7 (Tangent space via ImFT).** Let $M = F^{-1}(\mathbf{c}) \subseteq \mathbb{R}^n$ be the manifold from E0.3.4.6. Prove that the tangent space at a point $\mathbf{p} \in M$ is $T_\mathbf{p} M = \ker DF(\mathbf{p})$.

**E0.3.4.8 (Lagrange multipliers via ImFT).** Let $f, g: \mathbb{R}^n \to \mathbb{R}$ be $C^1$ and consider optimizing $f$ subject to $g(\mathbf{x}) = 0$. Use ImFT on the constraint to parametrize the level set locally as $x_n = h(x_1, \ldots, x_{n-1})$ (assuming $\partial g/\partial x_n \neq 0$), substitute into $f$, and derive the first-order condition $\nabla f = \lambda \nabla g$. This is the pedagogical derivation of Lagrange multipliers we will formalize in Topic 0.3.5.

#### ★★★ (Challenge)

**E0.3.4.9 (Higher-regularity).** Show by induction that if $F \in C^k$, then the implicit function $\mathbf{g} \in C^k$. (Hint: differentiate the formula $D\mathbf{g} = -(D_{\mathbf{y}} F)^{-1} D_{\mathbf{x}} F$ and bound derivatives by a linear recursion.)

**E0.3.4.10 (Analytic ImFT).** Prove (outline): if $F$ is real-analytic, so is $\mathbf{g}$. (One clean approach: extend everything to complex variables and apply the holomorphic implicit function theorem, which follows from the same proof with $\mathbb{C}$ in place of $\mathbb{R}$.)

**E0.3.4.11 (Bifurcation / degenerate ImFT).** Consider $F(x, y) = y^2 - x^2(x+1)$ at $(0, 0)$. Show $\partial F/\partial y = 0$ there, so ImFT fails; describe the zero set near $(0, 0)$ geometrically (it is a figure-eight, the **nodal curve**). This is a basic example in singularity theory.

**E0.3.4.12 (Implicit functions in Banach spaces).** State and prove the infinite-dimensional ImFT: for $X, Y, Z$ Banach and $F: X \times Y \to Z$ a $C^k$-map with $D_\mathbf{y} F(\mathbf{a}, \mathbf{b}) \in \mathcal{L}(Y, Z)$ invertible, one has a unique $C^k$ implicit function. Consequence: short-time existence of solutions to nonlinear PDEs near a known solution.

**E0.3.4.13 (Global ImFT).** Find conditions under which the implicit function exists globally (not just locally). Hint: a condition involving properness of $F$ together with a monodromy-type argument.

---


## Topic 0.3.5 — Constrained Optimization: Lagrange Multipliers and KKT Conditions

### Motivation

Unconstrained optimization asks: find $\mathbf{x}^*$ minimizing $f(\mathbf{x})$ over all of $\mathbb{R}^n$. Reality always constrains us. A portfolio cannot have negative cash in a margin account; a probability distribution cannot have negative entries; a covariance matrix must be positive semidefinite; the weight vector of a long-only equity portfolio lives in the simplex $\{\mathbf{w} \geq 0 : \sum w_i = 1\}$.

The **Lagrange multiplier theorem** (for equality constraints) and its generalization the **Karush-Kuhn-Tucker (KKT) conditions** (for inequality constraints) are the first-order optimality conditions under constraints. They are the workhorse of:
- **Portfolio optimization** (Markowitz mean-variance; Black-Litterman; risk parity).
- **Convex optimization** (SVMs, LASSO, cone programming).
- **Mechanism design and auction theory** (incentive compatibility as constraints).
- **Optimal control** (Pontryagin's principle is KKT in infinite dimensions).
- **Economic theory** (utility maximization subject to budget; firm profit maximization subject to production).

### Prerequisites

- Implicit Function Theorem (Topic 0.3.4).
- Gradients, directional derivatives, tangent spaces (Topic 0.3.2).
- Inner products and orthogonality (Topic 0.2.3).

### The Problem Setup

The standard nonlinear program is
$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) \quad \text{subject to} \quad \begin{cases} g_i(\mathbf{x}) = 0, & i = 1, \ldots, m \text{ (equality)} \\ h_j(\mathbf{x}) \leq 0, & j = 1, \ldots, p \text{ (inequality)} \end{cases}$$
with $f, g_i, h_j \in C^1$. The **feasible set** is
$$\Omega = \{\mathbf{x} : g_i(\mathbf{x}) = 0 \text{ for all } i, \; h_j(\mathbf{x}) \leq 0 \text{ for all } j\}.$$

We begin with the equality-constrained case, then add inequalities.

### Lagrange Multipliers (Equality Constraints)

#### Geometric intuition

At a constrained minimizer $\mathbf{x}^*$, no feasible direction decreases $f$. A direction $\mathbf{v}$ is *feasible to first order* along the surface $\{g = 0\}$ iff it is tangent, i.e., $Dg(\mathbf{x}^*) \mathbf{v} = \mathbf{0}$. The condition that $\nabla f(\mathbf{x}^*)^\top \mathbf{v} \geq 0$ for all such $\mathbf{v}$ and $-\mathbf{v}$ forces $\nabla f(\mathbf{x}^*) \perp \ker Dg(\mathbf{x}^*)$. By orthogonal decomposition, $\nabla f(\mathbf{x}^*) \in (\ker Dg(\mathbf{x}^*))^\perp = \mathrm{range}(Dg(\mathbf{x}^*)^\top) = \mathrm{span}\{\nabla g_1, \ldots, \nabla g_m\}$. So there exist scalars $\lambda_i$ with
$$\nabla f(\mathbf{x}^*) = \sum_{i=1}^m \lambda_i \nabla g_i(\mathbf{x}^*). \quad(*)$$
The $\lambda_i$ are the **Lagrange multipliers**.

#### Formal theorem

**Theorem 0.3.5.1 (Lagrange multiplier rule).** *Let $f, g_1, \ldots, g_m: U \to \mathbb{R}$ be $C^1$ on an open set $U \subseteq \mathbb{R}^n$ with $n > m$. Let $\mathbf{x}^* \in U$ be a local extremum of $f$ restricted to $M = \{\mathbf{x} : g_i(\mathbf{x}) = 0 \text{ for all } i\}$. If the* **constraint qualification** *$\nabla g_1(\mathbf{x}^*), \ldots, \nabla g_m(\mathbf{x}^*)$ are linearly independent holds, then there exist unique $\boldsymbol\lambda = (\lambda_1, \ldots, \lambda_m) \in \mathbb{R}^m$ such that*
$$\nabla f(\mathbf{x}^*) = \sum_{i=1}^m \lambda_i \nabla g_i(\mathbf{x}^*). \qquad \square$$

*Equivalently, define the* **Lagrangian** *$L(\mathbf{x}, \boldsymbol\lambda) = f(\mathbf{x}) - \sum_i \lambda_i g_i(\mathbf{x})$; then $\nabla_\mathbf{x} L = 0$ at $(\mathbf{x}^*, \boldsymbol\lambda^*)$.*

**Proof.** Let $\mathbf{g} = (g_1, \ldots, g_m)^\top$. By the constraint qualification, $D\mathbf{g}(\mathbf{x}^*) \in \mathbb{R}^{m \times n}$ has rank $m$. By column pivoting, relabel coordinates so that the last $m$ columns form an invertible $m \times m$ block; write $\mathbf{x} = (\mathbf{u}, \mathbf{v})$ with $\mathbf{u} \in \mathbb{R}^{n-m}, \mathbf{v} \in \mathbb{R}^m$, so $D_{\mathbf{v}} \mathbf{g}(\mathbf{x}^*)$ is invertible.

Apply the Implicit Function Theorem (0.3.4.1) to $\mathbf{g}(\mathbf{u}, \mathbf{v}) = 0$: there exist open neighborhoods $V_0 \ni \mathbf{u}^*$ and $W_0 \ni \mathbf{v}^*$ and a $C^1$ function $\mathbf{v} = \boldsymbol\varphi(\mathbf{u})$ with $\boldsymbol\varphi(\mathbf{u}^*) = \mathbf{v}^*$ and $\mathbf{g}(\mathbf{u}, \boldsymbol\varphi(\mathbf{u})) = 0$ on $V_0$, with derivative
$$D\boldsymbol\varphi(\mathbf{u}^*) = -[D_\mathbf{v} \mathbf{g}(\mathbf{x}^*)]^{-1} D_\mathbf{u} \mathbf{g}(\mathbf{x}^*).$$

Define the reduced objective $\tilde f(\mathbf{u}) = f(\mathbf{u}, \boldsymbol\varphi(\mathbf{u}))$ on $V_0$. Since $\mathbf{x}^*$ is a local extremum of $f|_M$ and $(\mathbf{u}, \boldsymbol\varphi(\mathbf{u}))$ parametrizes $M$ near $\mathbf{x}^*$, $\mathbf{u}^*$ is a local extremum of $\tilde f$ on the **open** set $V_0$. Hence $\nabla \tilde f(\mathbf{u}^*) = 0$.

By the chain rule:
$$\nabla \tilde f(\mathbf{u}^*) = \nabla_\mathbf{u} f(\mathbf{x}^*) + [D\boldsymbol\varphi(\mathbf{u}^*)]^\top \nabla_\mathbf{v} f(\mathbf{x}^*) = 0,$$
i.e.,
$$\nabla_\mathbf{u} f(\mathbf{x}^*) = [D_\mathbf{u} \mathbf{g}(\mathbf{x}^*)]^\top [D_\mathbf{v} \mathbf{g}(\mathbf{x}^*)]^{-\top} \nabla_\mathbf{v} f(\mathbf{x}^*).$$

Define $\boldsymbol\lambda^* = [D_\mathbf{v} \mathbf{g}(\mathbf{x}^*)]^{-\top} \nabla_\mathbf{v} f(\mathbf{x}^*) \in \mathbb{R}^m$. Then:
- $\nabla_\mathbf{v} f(\mathbf{x}^*) = [D_\mathbf{v} \mathbf{g}(\mathbf{x}^*)]^\top \boldsymbol\lambda^* = \sum_i \lambda_i^* \nabla_\mathbf{v} g_i(\mathbf{x}^*)$;
- $\nabla_\mathbf{u} f(\mathbf{x}^*) = [D_\mathbf{u} \mathbf{g}(\mathbf{x}^*)]^\top \boldsymbol\lambda^* = \sum_i \lambda_i^* \nabla_\mathbf{u} g_i(\mathbf{x}^*)$.

Stacking the two blocks, $\nabla f(\mathbf{x}^*) = \sum_i \lambda_i^* \nabla g_i(\mathbf{x}^*)$. Uniqueness of $\boldsymbol\lambda^*$ follows from linear independence of the $\nabla g_i(\mathbf{x}^*)$. $\blacksquare$

**Remark on constraint qualification.** Without linear independence, Lagrange multipliers may fail. Classic counterexample: minimize $f(x, y) = x$ subject to $g(x, y) = y^2 - x^3 = 0$ (the cuspidal cubic). The constraint set has $(0,0)$ as a cusp; $\nabla g(0,0) = 0$, so the rank condition fails. Here $f$ is minimized at $(0, 0)$ but $\nabla f(0, 0) = (1, 0)^\top \neq \lambda \nabla g(0, 0) = \mathbf{0}$ for any $\lambda$. Lagrange multipliers do not exist. The Fritz John form of the theorem absorbs this by allowing a multiplier on $\nabla f$: $\lambda_0 \nabla f = \sum \lambda_i \nabla g_i$ with $\lambda_0 \geq 0$ and $(\lambda_0, \boldsymbol\lambda) \neq 0$. Constraint qualification simply forces $\lambda_0 > 0$ (so we can normalize to $\lambda_0 = 1$).

### Second-Order Conditions

First order only says $\mathbf{x}^*$ is a *critical point* of the constrained problem. To distinguish minima from saddles / maxima, we need second-order information along the tangent space.

**Theorem 0.3.5.2 (Second-order necessary condition).** *If $\mathbf{x}^*$ is a local minimum of $f$ on $M = \{\mathbf{g} = 0\}$, constraint qualification holds, and $\boldsymbol\lambda^*$ is the Lagrange multiplier, then the* **bordered Hessian** *$\nabla^2_\mathbf{x} L(\mathbf{x}^*, \boldsymbol\lambda^*)$ satisfies $\mathbf{v}^\top \nabla^2_\mathbf{x} L(\mathbf{x}^*, \boldsymbol\lambda^*) \mathbf{v} \geq 0$ for all $\mathbf{v} \in T_{\mathbf{x}^*} M = \ker D\mathbf{g}(\mathbf{x}^*)$.*

**Theorem 0.3.5.3 (Second-order sufficient condition).** *If the first-order Lagrange condition holds and $\mathbf{v}^\top \nabla^2_\mathbf{x} L(\mathbf{x}^*, \boldsymbol\lambda^*) \mathbf{v} > 0$ for all nonzero $\mathbf{v} \in T_{\mathbf{x}^*} M$, then $\mathbf{x}^*$ is a strict local minimum of $f$ on $M$.*

**Proof sketch (sufficient).** Parametrize $M$ near $\mathbf{x}^*$ as in the proof of Theorem 0.3.5.1: $\mathbf{x} = (\mathbf{u}, \boldsymbol\varphi(\mathbf{u}))$. The reduced Hessian of $\tilde f(\mathbf{u}) = f(\mathbf{u}, \boldsymbol\varphi(\mathbf{u}))$ at $\mathbf{u}^*$ equals (by direct computation) the restriction of $\nabla^2_\mathbf{x} L$ to $T_{\mathbf{x}^*} M$. Strict positivity of this restricted form is exactly the unconstrained second-order sufficient condition for $\tilde f$, whence $\mathbf{u}^*$ (and therefore $\mathbf{x}^*$) is a strict local min. $\blacksquare$

#### Why the Lagrangian Hessian, not $\nabla^2 f$?

A common student mistake: at an equality-constrained critical point, check whether $\nabla^2 f(\mathbf{x}^*)$ is positive definite on $T_{\mathbf{x}^*} M$. This is **wrong**. The correct object is $\nabla_\mathbf{x}^2 L(\mathbf{x}^*, \boldsymbol\lambda^*) = \nabla^2 f(\mathbf{x}^*) - \sum_i \lambda_i^* \nabla^2 g_i(\mathbf{x}^*)$. Why? The curvature of $M$ itself contributes: a flat function $f$ on a curved constraint set can have a minimum because of the constraint's curvature. The Lagrange multiplier captures exactly the "cost of curvature."

### KKT Conditions (Inequality Constraints)

Now add $h_j(\mathbf{x}) \leq 0$ for $j = 1, \ldots, p$. A constraint is **active** at $\mathbf{x}^*$ iff $h_j(\mathbf{x}^*) = 0$; the set of active indices is $\mathcal{A}(\mathbf{x}^*)$.

**Theorem 0.3.5.4 (KKT necessary conditions).** *Let $f, g_i, h_j \in C^1$ near $\mathbf{x}^* \in \Omega$. Suppose $\mathbf{x}^*$ is a local minimum of $f$ over $\Omega$, and suppose the* **LICQ (Linear Independence Constraint Qualification)** *holds: the gradients $\{\nabla g_i(\mathbf{x}^*)\}_{i=1}^m \cup \{\nabla h_j(\mathbf{x}^*)\}_{j \in \mathcal{A}(\mathbf{x}^*)}$ are linearly independent. Then there exist unique multipliers $\boldsymbol\lambda^* \in \mathbb{R}^m$, $\boldsymbol\mu^* \in \mathbb{R}^p$ such that*
1. (Stationarity) $\nabla f(\mathbf{x}^*) = \sum_i \lambda_i^* \nabla g_i(\mathbf{x}^*) + \sum_j \mu_j^* \nabla h_j(\mathbf{x}^*)$;
2. (Primal feasibility) $g_i(\mathbf{x}^*) = 0$ for all $i$; $h_j(\mathbf{x}^*) \leq 0$ for all $j$;
3. (Dual feasibility) $\mu_j^* \geq 0$ for all $j$;
4. (Complementary slackness) $\mu_j^* h_j(\mathbf{x}^*) = 0$ for all $j$. $\qquad \square$

The four conditions (1)–(4) are the **Karush-Kuhn-Tucker (KKT) conditions**.

#### Interpretation of each condition

- *Stationarity* is the generalized Lagrange condition.
- *Dual feasibility* $\mu_j \geq 0$ encodes that inequality constraints only "push inward": the multiplier cannot encourage a direction that violates $h_j$.
- *Complementary slackness* $\mu_j h_j = 0$ is the crucial novelty: either $h_j(\mathbf{x}^*) = 0$ (active constraint, $\mu_j$ can be nonzero) or $h_j(\mathbf{x}^*) < 0$ (inactive, multiplier = 0). So inactive constraints contribute nothing to the stationarity relation.

**Proof of Theorem 0.3.5.4.** Partition $\{1, \ldots, p\} = \mathcal{A} \sqcup \mathcal{I}$ where $\mathcal{A}$ is the active set at $\mathbf{x}^*$. The *inactive* constraints $h_j(\mathbf{x}^*) < 0$ are locally irrelevant: a small neighborhood of $\mathbf{x}^*$ has all inactive constraints satisfied strictly. Thus $\mathbf{x}^*$ is a local min of $f$ over the reduced problem
$$\min f \quad \text{s.t.} \quad g_i = 0 \; (\forall i), \quad h_j = 0 \; (j \in \mathcal{A}).$$
(Why $h_j = 0$ not $h_j \leq 0$? Because if we allowed $h_j < 0$, we'd get a richer feasible set, and $\mathbf{x}^*$ is still a minimum on the smaller set where $h_j = 0$. We need to verify the multipliers for inactive constraints are $0$, and that the multipliers for active ones are $\geq 0$.)

By Theorem 0.3.5.1 (Lagrange) applied with the enlarged equality set $\{g_i = 0\} \cup \{h_j = 0 : j \in \mathcal{A}\}$ (LICQ gives independence), there exist unique $\lambda_i^*$, $\mu_j^*$ ($j \in \mathcal{A}$) with
$$\nabla f(\mathbf{x}^*) = \sum_i \lambda_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j \in \mathcal{A}} \mu_j^* \nabla h_j(\mathbf{x}^*). \quad (*)$$
Set $\mu_j^* = 0$ for $j \in \mathcal{I}$; then $(*)$ is exactly the stationarity condition, and complementary slackness $\mu_j^* h_j(\mathbf{x}^*) = 0$ holds automatically (either $\mu_j^* = 0$ or $h_j(\mathbf{x}^*) = 0$).

It remains to show $\mu_j^* \geq 0$ for $j \in \mathcal{A}$. Suppose, for contradiction, $\mu_{j_0}^* < 0$ for some $j_0 \in \mathcal{A}$. By LICQ, the gradients $\{\nabla g_i\} \cup \{\nabla h_j\}_{j \in \mathcal{A}}$ are independent; pick a vector $\mathbf{v}$ with $Dg \cdot \mathbf{v} = 0$, $\nabla h_j \cdot \mathbf{v} = 0$ for $j \in \mathcal{A} \setminus \{j_0\}$, and $\nabla h_{j_0}(\mathbf{x}^*)^\top \mathbf{v} = -1 < 0$ (such $\mathbf{v}$ exists by linear algebra: the target vector $(-1) \cdot e_{j_0}$ in the constraint gradient basis is achievable since the gradients are independent).

Along a smooth curve $\gamma(t)$ with $\gamma(0) = \mathbf{x}^*$, $\gamma'(0) = \mathbf{v}$, lying on $M = \{g_i = 0, h_j = 0 \text{ for } j \in \mathcal{A} \setminus \{j_0\}\}$ (constructed via ImFT), we have:
- $h_{j_0}(\gamma(t)) = h_{j_0}(\mathbf{x}^*) + t \nabla h_{j_0} \cdot \mathbf{v} + O(t^2) = -t + O(t^2) < 0$ for small $t > 0$; so $\gamma(t)$ is *feasible* for the original problem.
- $f(\gamma(t)) = f(\mathbf{x}^*) + t \nabla f \cdot \mathbf{v} + O(t^2)$. Using $(*)$: $\nabla f \cdot \mathbf{v} = \sum_i \lambda_i (\nabla g_i \cdot \mathbf{v}) + \sum_{j \in \mathcal{A}} \mu_j (\nabla h_j \cdot \mathbf{v}) = 0 + \mu_{j_0}^* \cdot (-1) = -\mu_{j_0}^* > 0$.

Wait, that's the wrong sign; let me redo. $\nabla f \cdot \mathbf{v} = \mu_{j_0}^* \cdot (-1) = -\mu_{j_0}^*$. If $\mu_{j_0}^* < 0$, then $-\mu_{j_0}^* > 0$, so $f(\gamma(t)) > f(\mathbf{x}^*)$ for small $t > 0$ — this doesn't contradict anything.

Let me reverse $\mathbf{v}$: instead choose $\mathbf{v}$ with $\nabla h_{j_0} \cdot \mathbf{v} = +1$. Then $h_{j_0}(\gamma(t)) = t + O(t^2) > 0$ for small $t > 0$ — *not* feasible.

Let me redo the argument correctly. Pick $\mathbf{v}$ with $Dg \cdot \mathbf{v} = 0$, $\nabla h_j \cdot \mathbf{v} = 0$ for $j \in \mathcal{A} \setminus \{j_0\}$, and $\nabla h_{j_0} \cdot \mathbf{v} = -1$. Along $\gamma(t)$: $h_{j_0}(\gamma(t)) = -t + O(t^2) < 0$ feasible. Now:
$$\nabla f \cdot \mathbf{v} = \mu_{j_0}^* \cdot (-1) = -\mu_{j_0}^* .$$
If $\mu_{j_0}^* < 0$, then $\nabla f \cdot \mathbf{v} = -\mu_{j_0}^* > 0$, so $f(\gamma(t)) - f(\mathbf{x}^*) = t (-\mu_{j_0}^*) + O(t^2) > 0$ for small $t$. This says $\gamma(t)$ is feasible but $f(\gamma(t)) > f(\mathbf{x}^*)$, which is *consistent* with $\mathbf{x}^*$ being a local minimum. No contradiction here.

Let me reverse the $\mathbf{v}$: choose $\mathbf{v}$ with $\nabla h_{j_0} \cdot \mathbf{v} = +1$, all other conditions intact. Then $\gamma(t)$ satisfies $h_{j_0}(\gamma(t)) = t + O(t^2) > 0$ — *not feasible*. Try $t < 0$: $\gamma(-|t|)$ has $h_{j_0} = -|t| + O(t^2) < 0$ feasible, and $f(\gamma(-|t|)) = f(\mathbf{x}^*) + (-|t|)(\mu_{j_0}^* \cdot 1) + O(t^2) = f(\mathbf{x}^*) - |t|\mu_{j_0}^* + O(t^2)$.

If $\mu_{j_0}^* < 0$, then $-|t|\mu_{j_0}^* > 0$, again consistent with a local min.

Hmm — the sign of $\mu_{j_0}^*$ isn't coming out naturally. The real argument is more subtle. Redo:

*Correct argument.* Assume $\mu_{j_0}^* < 0$. Choose $\mathbf{v}$ with:
- $\nabla g_i \cdot \mathbf{v} = 0$ for all $i$,
- $\nabla h_j \cdot \mathbf{v} = 0$ for $j \in \mathcal{A} \setminus \{j_0\}$,
- $\nabla h_{j_0} \cdot \mathbf{v} = -1$ (this sign will push into feasibility).

Then $\mathbf{v}$ is a "feasible direction": walking along $\gamma(t)$ for small $t > 0$ keeps us on $M' := \{g_i = 0, h_j = 0 \text{ for } j \in \mathcal{A} \setminus \{j_0\}\}$ and makes $h_{j_0}$ decrease, so all constraints remain satisfied.

Compute the linear change in $f$:
$$\frac{d}{dt}\Big|_{t=0} f(\gamma(t)) = \nabla f \cdot \mathbf{v} = \sum_i \lambda_i \nabla g_i \cdot \mathbf{v} + \sum_{j \in \mathcal{A}} \mu_j \nabla h_j \cdot \mathbf{v} = 0 + \mu_{j_0}^* \cdot (-1) = -\mu_{j_0}^*.$$
Since $\mu_{j_0}^* < 0$, we have $-\mu_{j_0}^* > 0$. Hmm, $f$ increases along $\mathbf{v}$, still consistent with minimum. The minimum is not violated this way.

Try $-\mathbf{v}$: $\nabla h_{j_0} \cdot (-\mathbf{v}) = +1$, so $h_{j_0}$ *increases* along $-\mathbf{v}$, exiting feasibility. We need a direction that (i) stays feasible, and (ii) strictly decreases $f$. The direction $\mathbf{v}$ (with $\nabla h_{j_0} \cdot \mathbf{v} = -1$) stays feasible but *increases* $f$ when $\mu_{j_0}^* < 0$. So feasibility and $f$-increase are linked — no contradiction from this $\mathbf{v}$.

Actually I had the correct insight but mis-tracked. The problematic direction is $-\mathbf{v}$ (which makes $h_{j_0}$ grow, infeasible) — there's no contradiction from within the feasible set for this $\mathbf{v}$.

Let me switch to the **standard clean proof**: it uses Farkas' lemma or a variant.

**Clean proof via Farkas' lemma.** Since $\mathbf{x}^*$ is a local minimizer of $f$ over $\Omega$, by standard first-order analysis there is no feasible descent direction. A direction $\mathbf{d}$ is "feasible and descent" iff (linearized):
- $\nabla g_i(\mathbf{x}^*)^\top \mathbf{d} = 0$ for all $i$,
- $\nabla h_j(\mathbf{x}^*)^\top \mathbf{d} \leq 0$ for all $j \in \mathcal{A}$,
- $\nabla f(\mathbf{x}^*)^\top \mathbf{d} < 0$.

Nonexistence of such $\mathbf{d}$ (combined with the LICQ, which ensures linearized feasibility matches true feasibility) is exactly Farkas-type alternative: there exist $\lambda_i \in \mathbb{R}$, $\mu_j \geq 0$ (for $j \in \mathcal{A}$) with
$$\nabla f(\mathbf{x}^*) = \sum_i \lambda_i \nabla g_i(\mathbf{x}^*) + \sum_{j \in \mathcal{A}} \mu_j \nabla h_j(\mathbf{x}^*).$$

**Farkas' Lemma** (recall): exactly one of the following holds for $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times n}$, $\mathbf{c} \in \mathbb{R}^n$:

(i) There is $\mathbf{d} \in \mathbb{R}^n$ with $A\mathbf{d} = 0$, $B\mathbf{d} \leq 0$, $\mathbf{c}^\top \mathbf{d} < 0$.
(ii) There exist $\boldsymbol\lambda \in \mathbb{R}^m$, $\boldsymbol\mu \in \mathbb{R}^p$ with $\boldsymbol\mu \geq 0$ and $\mathbf{c} = A^\top \boldsymbol\lambda + B^\top \boldsymbol\mu$.

Apply with $A = [\nabla g_i^\top]_i$, $B = [\nabla h_j^\top]_{j \in \mathcal{A}}$, $\mathbf{c} = \nabla f(\mathbf{x}^*)$: no feasible descent $\Leftrightarrow$ (i) fails $\Leftrightarrow$ (ii) holds, giving the KKT multipliers with $\mu_j \geq 0$. Set $\mu_j = 0$ for $j \notin \mathcal{A}$. Uniqueness follows from LICQ. $\blacksquare$

*(The linearization-matches-feasibility step is where LICQ is used; weaker constraint qualifications like Slater's condition work in the convex case.)*

### Convex Optimization: KKT Become Sufficient

A central result: for **convex** problems, the KKT conditions are *sufficient* for a global optimum, not merely necessary.

**Theorem 0.3.5.5 (Sufficiency for convex programs).** *Consider the problem $\min f(\mathbf{x})$ subject to $g_i(\mathbf{x}) = 0$ (affine) and $h_j(\mathbf{x}) \leq 0$ (convex), with $f$ convex and $C^1$. If $(\mathbf{x}^*, \boldsymbol\lambda^*, \boldsymbol\mu^*)$ satisfies the KKT conditions, then $\mathbf{x}^*$ is a global minimum.*

**Proof.** For any feasible $\mathbf{x}$,
$$f(\mathbf{x}) \geq f(\mathbf{x}^*) + \nabla f(\mathbf{x}^*)^\top (\mathbf{x} - \mathbf{x}^*) \quad \text{(convexity of $f$)}$$
$$= f(\mathbf{x}^*) + \Big(\sum_i \lambda_i^* \nabla g_i + \sum_j \mu_j^* \nabla h_j\Big)^\top (\mathbf{x} - \mathbf{x}^*) \quad \text{(KKT stationarity)}$$
$$\geq f(\mathbf{x}^*) + \sum_i \lambda_i^* (g_i(\mathbf{x}) - g_i(\mathbf{x}^*)) + \sum_j \mu_j^* (h_j(\mathbf{x}) - h_j(\mathbf{x}^*))$$
using affine equality and convexity of $h_j$ for the inequalities (and $\mu_j^* \geq 0$). Now $g_i(\mathbf{x}) - g_i(\mathbf{x}^*) = 0$ (both feasible), and $\mu_j^* h_j(\mathbf{x}) \leq 0$ while $\mu_j^* h_j(\mathbf{x}^*) = 0$ (complementary slackness). Thus $f(\mathbf{x}) \geq f(\mathbf{x}^*)$. $\blacksquare$

### Duality

The Lagrangian $L(\mathbf{x}, \boldsymbol\lambda, \boldsymbol\mu) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x}) + \sum_j \mu_j h_j(\mathbf{x})$ (with multipliers absorbing signs conventionally; many texts use $-\lambda_i$ instead — choose your sign convention and stick with it) leads to the **dual function**
$$q(\boldsymbol\lambda, \boldsymbol\mu) = \inf_\mathbf{x} L(\mathbf{x}, \boldsymbol\lambda, \boldsymbol\mu),$$
and the **dual problem**
$$\max_{\boldsymbol\lambda \in \mathbb{R}^m, \; \boldsymbol\mu \geq 0} q(\boldsymbol\lambda, \boldsymbol\mu).$$

**Weak duality:** $q(\boldsymbol\lambda, \boldsymbol\mu) \leq f(\mathbf{x}^*)$ for any feasible $\mathbf{x}^*$ and any $\boldsymbol\mu \geq 0$. (Immediate.)

**Strong duality:** for convex problems under Slater's condition, $\sup q = \min f$, and the supremum is attained. The dual optimum multipliers are exactly the KKT multipliers. Strong duality is the theoretical basis for *dual methods* in numerical optimization (ADMM, dual decomposition, etc.) and the economic interpretation of multipliers as shadow prices (see below).

### Worked Examples

#### Example 0.3.5.1 — Closest point to a hyperplane

Minimize $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{x}\|^2$ subject to $\mathbf{a}^\top \mathbf{x} = b$ (with $\mathbf{a} \neq 0$, so LICQ trivially holds).

Lagrangian: $L = \frac{1}{2}\|\mathbf{x}\|^2 - \lambda(\mathbf{a}^\top \mathbf{x} - b)$. Stationarity: $\mathbf{x} - \lambda \mathbf{a} = 0$, so $\mathbf{x}^* = \lambda \mathbf{a}$. Primal feasibility: $\mathbf{a}^\top (\lambda \mathbf{a}) = b \Rightarrow \lambda = b/\|\mathbf{a}\|^2$. Answer: $\mathbf{x}^* = (b/\|\mathbf{a}\|^2) \mathbf{a}$, $f(\mathbf{x}^*) = b^2/(2\|\mathbf{a}\|^2)$.

This is the formula for orthogonal projection onto a hyperplane — derived in three lines by Lagrange.

#### Example 0.3.5.2 — Markowitz mean-variance portfolio

Minimize $\frac{1}{2} \mathbf{w}^\top \Sigma \mathbf{w}$ subject to $\boldsymbol\mu^\top \mathbf{w} = r_0$ (target return), $\mathbf{1}^\top \mathbf{w} = 1$ (fully invested). Here $\Sigma \succ 0$, $\boldsymbol\mu, \mathbf{1}$ linearly independent (required for feasibility to be nondegenerate).

Lagrangian: $L = \frac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w} - \lambda_1 (\boldsymbol\mu^\top \mathbf{w} - r_0) - \lambda_2 (\mathbf{1}^\top \mathbf{w} - 1)$.

Stationarity: $\Sigma \mathbf{w} = \lambda_1 \boldsymbol\mu + \lambda_2 \mathbf{1}$, so $\mathbf{w}^* = \Sigma^{-1}(\lambda_1 \boldsymbol\mu + \lambda_2 \mathbf{1})$.

Plug into constraints:
$$\boldsymbol\mu^\top \Sigma^{-1} \boldsymbol\mu \cdot \lambda_1 + \boldsymbol\mu^\top \Sigma^{-1} \mathbf{1} \cdot \lambda_2 = r_0,$$
$$\mathbf{1}^\top \Sigma^{-1} \boldsymbol\mu \cdot \lambda_1 + \mathbf{1}^\top \Sigma^{-1} \mathbf{1} \cdot \lambda_2 = 1.$$
Define $A = \boldsymbol\mu^\top \Sigma^{-1} \boldsymbol\mu$, $B = \boldsymbol\mu^\top \Sigma^{-1} \mathbf{1}$, $C = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}$. Solve this $2 \times 2$ system:
$$\lambda_1 = \frac{C r_0 - B}{AC - B^2}, \quad \lambda_2 = \frac{A - B r_0}{AC - B^2}.$$
Denominator $AC - B^2 > 0$ by Cauchy-Schwarz (applied in the $\Sigma^{-1}$-inner-product). Minimum variance: $\mathrm{Var}(\mathbf{w}^*) = \mathbf{w}^{*\top} \Sigma \mathbf{w}^* = \lambda_1 r_0 + \lambda_2$ (substitute and simplify using stationarity). This parabolic relationship between target return $r_0$ and optimal variance is the **efficient frontier**.

#### Example 0.3.5.3 — Log-barrier utility maximization (KKT with inequality)

Maximize $u(\mathbf{x}) = \sum_i \log x_i$ subject to $\mathbf{p}^\top \mathbf{x} \leq W$ (budget), $\mathbf{x} \geq 0$. (This is consumption theory: log utility subject to budget and nonnegativity.)

Equivalently, minimize $-\sum_i \log x_i$ subject to $\mathbf{p}^\top \mathbf{x} - W \leq 0$ and $-x_i \leq 0$.

KKT with $\mu$ for budget and $\nu_i$ for nonnegativity:
- Stationarity: $-1/x_i + \mu p_i - \nu_i = 0 \Rightarrow 1/x_i = \mu p_i - \nu_i$ for each $i$.
- Complementary slackness: $\nu_i x_i = 0$.

Since $-\log 0 = +\infty$, any minimizer must have $x_i > 0$; thus $\nu_i = 0$ for all $i$, and $1/x_i = \mu p_i$ so $x_i^* = 1/(\mu p_i)$.

Budget: $\sum_i p_i x_i^* = \sum_i 1/\mu = n/\mu \leq W$. At the interior optimum, the budget is active: $\mu^* = n/W$ and $x_i^* = W/(n p_i)$.

Economic interpretation: optimal consumption allocates the same *dollar amount* ($W/n$) to each good, so quantities are $W/(n p_i)$. The multiplier $\mu^* = n/W$ is the **marginal utility of income**. This is the log-utility consumer's Walrasian demand.

#### Example 0.3.5.4 — Support Vector Machine (KKT in action)

The soft-margin SVM solves
$$\min_{\mathbf{w}, b, \boldsymbol\xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0.$$

KKT gives multipliers $\alpha_i \geq 0$ (margin) and $\beta_i \geq 0$ (slack). Stationarity in $\mathbf{w}$:
$$\mathbf{w}^* = \sum_i \alpha_i^* y_i \mathbf{x}_i.$$
Stationarity in $\xi_i$: $C = \alpha_i + \beta_i$. Complementary slackness: $\alpha_i [y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i] = 0$ and $\beta_i \xi_i = 0$.

These conditions classify training points:
- $\alpha_i = 0$: point inactive (lies outside margin or correctly outside), contributes nothing to $\mathbf{w}^*$.
- $0 < \alpha_i < C$: point lies exactly on the margin, $\xi_i = 0$.
- $\alpha_i = C$: point lies strictly inside the margin (or misclassified), $\xi_i > 0$.

The **support vectors** are the points with $\alpha_i > 0$. The KKT sparsity structure is what makes SVMs computationally tractable and is the origin of the **kernel trick**: $\mathbf{w}^*$ is expressed purely via inner products with data, so replacing $\mathbf{x}_i^\top \mathbf{x}_j$ by a positive-definite kernel $K(\mathbf{x}_i, \mathbf{x}_j)$ lifts SVMs to infinite-dimensional feature spaces.

### Computational Implementation

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # for efficient frontier only

# Example 1: Closest point to hyperplane, verify Lagrange analytically vs numerically
a = np.array([1.0, 2.0, 3.0])
b = 7.0
n = 3

# Analytical
lam = b / (a @ a)
x_analytical = lam * a

# Numerical (SLSQP)
def f(x): return 0.5 * x @ x
def eq_constraint(x): return a @ x - b
result = minimize(f, np.zeros(n), method='SLSQP', constraints={'type': 'eq', 'fun': eq_constraint})
print("Analytical x* =", x_analytical)
print("Numerical  x* =", result.x)
print("Error:", np.linalg.norm(x_analytical - result.x))

# Example 2: Markowitz efficient frontier
np.random.seed(0)
n_assets = 5
Sigma = np.random.randn(n_assets, n_assets)
Sigma = Sigma @ Sigma.T + 0.05 * np.eye(n_assets)  # PD covariance
mu = np.random.randn(n_assets) * 0.05 + 0.1

one = np.ones(n_assets)
Sigma_inv = np.linalg.inv(Sigma)
A = mu @ Sigma_inv @ mu
B = mu @ Sigma_inv @ one
C = one @ Sigma_inv @ one
D = A * C - B**2

def markowitz(r0):
    lam1 = (C*r0 - B) / D
    lam2 = (A - B*r0) / D
    w = Sigma_inv @ (lam1 * mu + lam2 * one)
    var = w @ Sigma @ w
    return w, var, lam1, lam2

r_targets = np.linspace(min(mu)*0.8, max(mu)*1.2, 25)
variances = [markowitz(r)[1] for r in r_targets]

# Verify Lagrange: gradient of L at optimum should vanish
w_mid, _, lam1, lam2 = markowitz(0.1)
stationarity_residual = Sigma @ w_mid - lam1 * mu - lam2 * one
print("Stationarity residual (should be ~0):", np.max(np.abs(stationarity_residual)))
print("Primal feasibility 1 (return  = 0.1):", mu @ w_mid)
print("Primal feasibility 2 (sum w = 1):", w_mid @ one)

# Example 3: KKT for log-utility consumer
# Max sum log(x_i) subject to p @ x <= W, x >= 0
p = np.array([2.0, 5.0, 1.0, 3.0])
W = 20.0
n_goods = len(p)
# Analytical: x* = W / (n p)
x_star_analytical = W / (n_goods * p)
mu_star = n_goods / W
print("\nLog utility demand (analytical):", x_star_analytical)
print("Check budget binds:", p @ x_star_analytical, "==", W)

# Numerical
def neg_u(x): return -np.sum(np.log(x))
def budget(x): return W - p @ x  # >= 0
constraints = [{'type': 'ineq', 'fun': budget}]
bounds = [(1e-8, None)] * n_goods
res = minimize(neg_u, np.ones(n_goods)*W/(n_goods*p.mean()), method='SLSQP', 
               bounds=bounds, constraints=constraints)
print("Log utility demand (numerical):", res.x)
print("Multiplier via gradient: mu* =", 1/(res.x * p))  # all should equal mu*

# Example 4: KKT multiplier as a "shadow price"
# Solve min x^2 + y^2 subject to x + y >= c, for varying c
def min_sq_with_constraint(c):
    def f(v): return v[0]**2 + v[1]**2
    cons = [{'type': 'ineq', 'fun': lambda v, c=c: v[0] + v[1] - c}]
    return minimize(f, [0,0], method='SLSQP', constraints=cons)

cs = np.linspace(-2, 2, 21)
fstars = [min_sq_with_constraint(c).fun for c in cs]
# Analytical: if c > 0, x = y = c/2, f* = c^2/2, dual mu* = c; if c <= 0, x = y = 0, f* = 0, mu* = 0.
dfdc_numerical = np.gradient(fstars, cs)
print("\nKKT multiplier (shadow price) matches d f*/dc:")
for c, mu_shadow in zip(cs, dfdc_numerical):
    expected = max(c, 0)
    print(f"  c = {c:+.2f}: observed df*/dc = {mu_shadow:+.3f}, expected mu* = {expected:+.3f}")
```

The final block demonstrates the **shadow-price interpretation of multipliers**: the KKT multiplier $\mu^*$ for a constraint $h(\mathbf{x}) \leq b$ equals $\partial V(b)/\partial b$ where $V(b) = \min f$ subject to the constraint — the *sensitivity of the optimal value to a relaxation of the constraint*. This is why multipliers are economically interpretable as prices.

### [QUANT APPLICATION] Markowitz, Risk Parity, and KKT in Portfolio Construction

**(A) Markowitz optimal portfolios.** As shown above, the efficient frontier is traced out by varying $r_0$. The closed-form multipliers $(\lambda_1, \lambda_2)$ translate directly to **risk aversion** parameters in the related "mean-variance utility" problem $\max \mathbf{w}^\top \boldsymbol\mu - \tfrac{\gamma}{2} \mathbf{w}^\top \Sigma \mathbf{w}$ subject to $\mathbf{1}^\top \mathbf{w} = 1$.

**(B) Risk parity.** The risk-parity constraint "each asset contributes equally to portfolio variance" is $w_i \cdot (\Sigma \mathbf{w})_i = c$ for all $i$. KKT on $\min \mathbf{w}^\top \Sigma \mathbf{w}$ with this constraint yields a nonlinear system solvable via Newton's method on the Lagrangian. Risk parity is the theoretical foundation of the well-known Bridgewater All-Weather portfolio.

**(C) Long-only and leverage constraints.** Adding $\mathbf{w} \geq 0$ (long-only) creates inequality constraints; complementary slackness tells you that assets with $w_i = 0$ have "shadow negative expected return" — the multiplier reveals which assets *would* be shorted if allowed. Leverage constraint $\|\mathbf{w}\|_1 \leq L$ is handled similarly; its KKT multiplier is the **marginal cost of leverage**.

**(D) Cone programming.** Modern portfolio optimization uses second-order cone programs (SOCP) and semidefinite programs (SDP), which are KKT-solvable convex problems. Examples: robust portfolio optimization (cost is worst-case over an uncertainty set), minimum-CVaR (Conditional VaR) portfolios. Solvers like CVXPY, MOSEK, ECOS all exploit the KKT structure.

**(E) Options pricing as constrained optimization.** The upper-Snell envelope — price of an American option — is the value function of an **optimal stopping** problem, which can be formulated as a linear program (LP) with KKT conditions characterizing the exercise boundary. The multipliers give the early-exercise premium.

**(F) Utility maximization.** In stochastic control / consumption-investment problems (Merton, Karatzas-Shreve), the HJB equation is derived by applying Lagrange / KKT to a constrained dynamic programming problem. The "dual" problem in utility maximization (Karatzas-Kou-Lehoczky-Shreve) uses the Legendre-Fenchel transform, a KKT cousin via convex conjugacy.

**(G) Linear pricing and no arbitrage.** The **Fundamental Theorem of Asset Pricing** can be phrased as: absence of arbitrage $\Leftrightarrow$ existence of a strictly positive linear pricing functional $\Leftrightarrow$ a KKT-like Farkas alternative in the finite-state model. Risk-neutral probabilities are KKT multipliers of the no-arbitrage LP.

### Exercises

#### ★ (Foundation)

**E0.3.5.1.** Minimize $f(x, y) = x + y$ subject to $x^2 + y^2 = 1$. Find both critical points (min and max) using Lagrange; verify with substitution $y = \pm\sqrt{1 - x^2}$.

**E0.3.5.2.** Derive the formula for the closest point on the sphere $\|\mathbf{x}\| = r$ to a given point $\mathbf{p}$. What if $\mathbf{p} = 0$?

**E0.3.5.3.** Verify the KKT conditions for: $\min x^2 + y^2$ subject to $x + y \geq 1$, $x \geq 0$, $y \geq 0$. Identify active constraints, multipliers, complementary slackness.

**E0.3.5.4 (The Lagrangian is saddle-shaped).** For a convex problem with strong duality, show that $(\mathbf{x}^*, \boldsymbol\lambda^*, \boldsymbol\mu^*)$ is a saddle point of the Lagrangian: $L(\mathbf{x}^*, \boldsymbol\lambda, \boldsymbol\mu) \leq L(\mathbf{x}^*, \boldsymbol\lambda^*, \boldsymbol\mu^*) \leq L(\mathbf{x}, \boldsymbol\lambda^*, \boldsymbol\mu^*)$.

#### ★★ (Intermediate)

**E0.3.5.5 (Rayleigh-Ritz via Lagrange).** Show that $\max \mathbf{x}^\top A \mathbf{x}$ subject to $\mathbf{x}^\top \mathbf{x} = 1$ (for symmetric $A$) has Lagrangian stationarity $A \mathbf{x} = \lambda \mathbf{x}$, recovering the eigenvalue equation. Conclude that the maximum value is the largest eigenvalue. This is the variational characterization from Topic 0.2.5.

**E0.3.5.6 (Entropy maximization).** Maximize $H(\mathbf{p}) = -\sum_i p_i \log p_i$ subject to $\sum p_i = 1$ and $\sum p_i E_i = \bar E$ (expected energy fixed). Show that the optimal distribution is the Gibbs / Boltzmann distribution $p_i \propto e^{-\beta E_i}$ where $\beta$ is a Lagrange multiplier. (This is how statistical mechanics derives the canonical ensemble.)

**E0.3.5.7 (Cauchy-Schwarz via Lagrange).** Prove Cauchy-Schwarz: maximize $\mathbf{a}^\top \mathbf{x}$ subject to $\|\mathbf{x}\|^2 = 1$ using Lagrange. Conclude $|\mathbf{a}^\top \mathbf{x}| \leq \|\mathbf{a}\| \|\mathbf{x}\|$.

**E0.3.5.8 (Slater's condition).** Read about Slater's condition for strong duality (convex problem + existence of a strictly interior feasible point). Explain why it implies LICQ-like constraint qualifications fail to be needed in the convex setting.

**E0.3.5.9 (Dual of the LP).** Given $\min \mathbf{c}^\top \mathbf{x}$ subject to $A\mathbf{x} \leq \mathbf{b}$, $\mathbf{x} \geq 0$, derive the dual LP $\max \mathbf{b}^\top \mathbf{y}$ subject to $A^\top \mathbf{y} \leq \mathbf{c}$, $\mathbf{y} \leq 0$ via KKT and Lagrangian duality. Interpret the multipliers.

#### ★★★ (Challenge)

**E0.3.5.10 (Fritz John conditions).** Prove the Fritz John form of KKT (without constraint qualification): there exist $\mu_0 \geq 0$, $\boldsymbol\lambda \in \mathbb{R}^m$, $\boldsymbol\mu \geq 0$, not all zero, with $\mu_0 \nabla f = \sum \lambda_i \nabla g_i + \sum \mu_j \nabla h_j$. Use a limiting argument on a sequence of problems with perturbed constraints.

**E0.3.5.11 (Mangasarian-Fromovitz CQ).** State and prove the Mangasarian-Fromovitz constraint qualification (a weaker CQ than LICQ) implies existence of KKT multipliers.

**E0.3.5.12 (Sensitivity theorem).** Prove (outline): if $V(b)$ is the optimal value of $\min f(\mathbf{x})$ subject to $g(\mathbf{x}) \leq b$ (parametrized by $b$), and the problem is convex with strong duality, then $V$ is convex in $b$ and $\mu^*(b_0) \in \partial V(b_0)$ — the KKT multiplier lies in the subdifferential of the value function. This generalizes the envelope theorem.

**E0.3.5.13 (Duality and game theory).** Read the connection between Lagrangian duality and zero-sum games: $\min_\mathbf{x} \max_\mathbf{y} L(\mathbf{x}, \mathbf{y}) \geq \max_\mathbf{y} \min_\mathbf{x} L$, with equality exactly at saddle points (von Neumann's minimax theorem). Derive strong duality for convex programs from this.

**E0.3.5.14 (KKT in ∞ dimensions).** Generalize the KKT conditions to functional optimization: minimize $\int_0^T L(x(t), \dot x(t), t)\, dt$ subject to $x(0) = 0$, $\int g(x(t))\, dt \leq 0$. The Euler-Lagrange equation is the stationarity condition; the isoperimetric multiplier is Lagrange's. (The full KKT calculus in Banach spaces underlies Pontryagin's principle in control theory.)

---


## Topic 0.3.6 — Change of Variables in Multiple Integrals

### Motivation

The single-variable substitution formula
$$\int_a^b f(\varphi(t)) \varphi'(t)\, dt = \int_{\varphi(a)}^{\varphi(b)} f(u)\, du$$
generalizes to multiple integrals with a striking replacement: the derivative $\varphi'(t)$ becomes the **Jacobian determinant** $|\det D\varphi|$. This is not mere notation: it expresses how an infinitesimal volume element scales under a smooth transformation, and it is the cornerstone of almost every computation involving densities, measure transformation, or geometric integration.

In quantitative work, change of variables appears ubiquitously:
- **Density transformation.** If $X$ has density $f_X$ and $Y = g(X)$ with $g$ a diffeomorphism, then $f_Y(\mathbf{y}) = f_X(g^{-1}(\mathbf{y})) |\det Dg^{-1}(\mathbf{y})|$. This is the starting point of all nonlinear filtering and Monte Carlo importance sampling.
- **Girsanov / change of measure.** Radon-Nikodym derivatives on infinite-dimensional path spaces generalize the Jacobian — with formidable consequences for risk-neutral pricing, optimal filtering, and large deviations.
- **Normalizing flows.** Modern deep generative models (NICE, RealNVP, Glow, Neural Spline Flows) are parametric diffeomorphisms trained by maximizing likelihood via change-of-variables $\log p(\mathbf{x}) = \log p_Z(g^{-1}(\mathbf{x})) + \log|\det Dg^{-1}|$. Architectures are *designed* so the Jacobian determinant has a tractable form (triangular, low-rank, etc.).
- **Polar / spherical / cylindrical computations.** Gaussian integrals, surface areas, inertia tensors, and most physical applications reduce to change-of-variable evaluations.

### Prerequisites

- Inverse Function Theorem (0.3.3), Diffeomorphisms.
- Basic Riemann/Lebesgue integration (the precise setting we adopt).
- Determinants (Topic 0.2.2).

### Statement of the Theorem

**Theorem 0.3.6.1 (Change of variables).** *Let $U, V \subseteq \mathbb{R}^n$ be open, $\varphi: U \to V$ a $C^1$-diffeomorphism. Let $f: V \to \mathbb{R}$ be Lebesgue-integrable over $V$. Then $f \circ \varphi \cdot |\det D\varphi|$ is Lebesgue-integrable over $U$, and*
$$\int_V f(\mathbf{y})\, d\mathbf{y} = \int_U f(\varphi(\mathbf{x})) \, |\det D\varphi(\mathbf{x})|\, d\mathbf{x}. \qquad \square$$

*More generally, if $A \subseteq U$ is measurable, $B = \varphi(A) \subseteq V$, and $f: B \to \mathbb{R}$ is integrable over $B$, then*
$$\int_B f(\mathbf{y})\, d\mathbf{y} = \int_A f(\varphi(\mathbf{x})) \, |\det D\varphi(\mathbf{x})|\, d\mathbf{x}.$$

The quantity $|\det D\varphi(\mathbf{x})|$ is the **Jacobian** of the transformation (at $\mathbf{x}$); it is the local volume-scaling factor.

The proof is nontrivial (it uses covering arguments, partitions of unity, careful approximation of general diffeomorphisms by affine maps on small cubes). We give the argument in outline since it is foundational and clarifies *why* the determinant — rather than some other quantity — is the correct scaling factor.

### Proof Outline

#### Step 1 — Affine case: linear change of variables

For $\varphi(\mathbf{x}) = L\mathbf{x} + \mathbf{c}$ with $L \in GL_n(\mathbb{R})$, the theorem reduces to the statement that Lebesgue measure satisfies
$$\mathrm{vol}(L(A)) = |\det L| \cdot \mathrm{vol}(A) \qquad (*)$$
for every measurable $A$. This is a *geometric* fact: the volume of a parallelepiped spanned by the columns of $L$ equals $|\det L|$.

**Proof of $(*)$.** Every $L \in GL_n(\mathbb{R})$ factors into elementary row operations:
- **Scaling** $L_i(\lambda): \mathbf{e}_i \mapsto \lambda \mathbf{e}_i$, $\det = \lambda$. Scales the $i$-th axis by $|\lambda|$; Cavalieri's principle gives $\mathrm{vol}(L_i(\lambda) A) = |\lambda| \mathrm{vol}(A)$.
- **Shear** $S_{ij}(\tau): \mathbf{e}_i \mapsto \mathbf{e}_i + \tau \mathbf{e}_j$, $\det = 1$. Preserves volume (shears a parallelepiped into another of the same base and height).
- **Swap** $P_{ij}$: $\det = -1$. Preserves volume (reflection).

By induction on the number of elementary matrices in a factorization and the multiplicativity $\det(AB) = \det(A)\det(B)$, $(*)$ holds for arbitrary invertible $L$. For translations $\mathbf{x} \mapsto \mathbf{x} + \mathbf{c}$, Lebesgue measure is translation-invariant. Combining the two covers affine transformations. $\blacksquare$

#### Step 2 — Nonlinear case: local linear approximation

For a $C^1$-diffeomorphism $\varphi$, the differentiability of $\varphi$ at $\mathbf{x}_0$ says
$$\varphi(\mathbf{x}) = \varphi(\mathbf{x}_0) + D\varphi(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0) + o(\|\mathbf{x} - \mathbf{x}_0\|).$$
So $\varphi$ looks affine on small scales; the affine case gives the correct volume scaling to leading order.

**Formal statement (volume scaling in the small).** For a $C^1$-diffeomorphism $\varphi$, any $\mathbf{x}_0 \in U$, and any $\varepsilon > 0$, there exists $\delta > 0$ such that for every measurable set $E \subseteq B_\delta(\mathbf{x}_0)$,
$$\big| \mathrm{vol}(\varphi(E)) - |\det D\varphi(\mathbf{x}_0)| \cdot \mathrm{vol}(E) \big| \leq \varepsilon \cdot |\det D\varphi(\mathbf{x}_0)| \cdot \mathrm{vol}(E).$$

*Sketch.* Let $L = D\varphi(\mathbf{x}_0)$. Factor $\varphi = (L \cdot ) + r$ where $r(\mathbf{x}) = \varphi(\mathbf{x}) - \varphi(\mathbf{x}_0) - L(\mathbf{x} - \mathbf{x}_0)$; by Fréchet differentiability, $\|r(\mathbf{x})\| = o(\|\mathbf{x} - \mathbf{x}_0\|)$. For small enough $\delta$, the nonlinear map $\varphi$ is squeezed between two affine maps with Jacobians $(1 \pm \varepsilon) L$, and each of these has a fixed volume-scaling factor $(1 \pm \varepsilon)^n |\det L|$. Take $\varepsilon$ small enough so $(1 \pm \varepsilon)^n$ is within the tolerance. $\blacksquare$

#### Step 3 — Global via partition of unity

Cover $U$ by small open balls $B_{r_k}(\mathbf{x}_k)$ on each of which the local approximation error is below $\varepsilon/2^k$ relative to the local Jacobian. Choose a $C^\infty$ partition of unity $\{\eta_k\}$ subordinate to this cover. For continuous $f$ with compact support:
$$\int_V f(\mathbf{y})\, d\mathbf{y} = \sum_k \int_V f(\mathbf{y}) \eta_k(\varphi^{-1}(\mathbf{y}))\, d\mathbf{y}.$$
Each piece is an integral over the image of a small ball where $\varphi$ is near-affine; apply Step 2 with estimates uniform over the ball. Sum and let $\varepsilon \to 0$:
$$\int_V f\, d\mathbf{y} = \sum_k \int_U (f \circ \varphi) \eta_k |\det D\varphi|\, d\mathbf{x} = \int_U (f \circ \varphi) |\det D\varphi|\, d\mathbf{x}.$$

Extend from continuous compactly supported $f$ (which are dense in $L^1$) to general integrable $f$ by monotone/dominated convergence. $\blacksquare$

*(A more modern proof uses the Vitali-Carathéodory covering lemma and the product-of-shears decomposition; the argument we gave follows Rudin's Real and Complex Analysis.)*

### Why the Determinant?

The determinant's appearance is not accidental: it is *uniquely* characterized as the signed volume-scaling factor for linear maps. A clean axiomatic characterization:

**Proposition 0.3.6.2.** *The function $L \mapsto |\det L|$ is the unique continuous function $GL_n(\mathbb{R}) \to \mathbb{R}_{>0}$ that is (i) multiplicative: $J(LM) = J(L)J(M)$; (ii) equals $1$ on the orthogonal group $O(n)$.*

So: volume scaling must be multiplicative under composition (immediate) and unchanged by rotation/reflection (geometric). These two conditions force the scaling factor to be the absolute value of the determinant.

### Worked Examples

#### Example 0.3.6.1 — Polar coordinates

Let $\varphi(r, \theta) = (r\cos\theta, r\sin\theta)$ on $(0, \infty) \times (0, 2\pi)$ onto $\mathbb{R}^2 \setminus \{\text{positive } x\text{-axis}\}$.
$$D\varphi = \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{pmatrix}, \quad \det D\varphi = r\cos^2\theta + r\sin^2\theta = r.$$
The Jacobian is $r$. Hence $dx\, dy = r\, dr\, d\theta$.

Application: the classic Gaussian integral.
$$\left(\int_{\mathbb{R}} e^{-x^2}\, dx\right)^2 = \int_{\mathbb{R}^2} e^{-x^2 - y^2}\, dx\, dy = \int_0^{2\pi} \int_0^\infty e^{-r^2} r\, dr\, d\theta = 2\pi \cdot \tfrac{1}{2} = \pi,$$
so $\int e^{-x^2} dx = \sqrt{\pi}$. This is the standard polar trick.

#### Example 0.3.6.2 — Spherical coordinates in $\mathbb{R}^3$

$\varphi(r, \theta, \varphi) = (r\sin\theta\cos\varphi, r\sin\theta\sin\varphi, r\cos\theta)$ with $r > 0$, $\theta \in (0, \pi)$, $\varphi \in (0, 2\pi)$.

Computing $D\varphi$ and its determinant (straightforward expansion):
$$\det D\varphi = r^2 \sin\theta.$$
Hence $dx\, dy\, dz = r^2 \sin\theta\, dr\, d\theta\, d\varphi$.

Application: volume of the unit ball.
$$\mathrm{vol}(B_1^3) = \int_0^1 \int_0^\pi \int_0^{2\pi} r^2 \sin\theta\, d\varphi\, d\theta\, dr = \tfrac{1}{3} \cdot 2 \cdot 2\pi = \frac{4\pi}{3}.$$

#### Example 0.3.6.3 — Volume of the unit ball in $\mathbb{R}^n$

Generalize: $\mathrm{vol}(B_1^n) = \pi^{n/2}/\Gamma(n/2 + 1)$. One slick derivation uses the polar-like decomposition $\mathbb{R}^n = \mathbb{R}_+ \times S^{n-1}$ where $d^n \mathbf{x} = r^{n-1}\, dr\, d\sigma$ ($\sigma$ = surface measure on the sphere). Compute
$$\int_{\mathbb{R}^n} e^{-\|\mathbf{x}\|^2}\, d^n\mathbf{x} = \pi^{n/2} \quad \text{(separability)}$$
$$= \int_{S^{n-1}}\!\! \int_0^\infty e^{-r^2} r^{n-1}\, dr\, d\sigma = \tfrac{1}{2}\Gamma(n/2) \cdot \mathrm{vol}(S^{n-1}).$$
So $\mathrm{vol}(S^{n-1}) = 2\pi^{n/2}/\Gamma(n/2)$, and $\mathrm{vol}(B_1^n) = \mathrm{vol}(S^{n-1})/n = \pi^{n/2}/\Gamma(n/2 + 1)$.

#### Example 0.3.6.4 — Affine transformation of a Gaussian

Let $X \sim \mathcal{N}(\mathbf{0}, I_n)$ with density $f_X(\mathbf{x}) = (2\pi)^{-n/2} e^{-\|\mathbf{x}\|^2/2}$. Let $Y = AX + \boldsymbol\mu$ with $A \in GL_n(\mathbb{R})$. The change of variable $\mathbf{x} = A^{-1}(\mathbf{y} - \boldsymbol\mu)$ has $|\det D(A^{-1})| = |\det A|^{-1}$. So the density of $Y$ is
$$f_Y(\mathbf{y}) = f_X(A^{-1}(\mathbf{y} - \boldsymbol\mu)) \cdot |\det A|^{-1} = \frac{1}{(2\pi)^{n/2} |\det A|} \exp\!\left(-\tfrac{1}{2} (\mathbf{y} - \boldsymbol\mu)^\top (AA^\top)^{-1} (\mathbf{y} - \boldsymbol\mu)\right).$$
Writing $\Sigma = AA^\top$ (covariance of $Y$), $|\det A| = |\det \Sigma|^{1/2}$, we recover the multivariate Gaussian density
$$f_Y(\mathbf{y}) = (2\pi)^{-n/2} |\det \Sigma|^{-1/2} \exp\!\left(-\tfrac{1}{2}(\mathbf{y} - \boldsymbol\mu)^\top \Sigma^{-1} (\mathbf{y} - \boldsymbol\mu)\right).$$
This is the *construction* of multivariate normals via affine maps — a core use of change of variables in probability.

#### Example 0.3.6.5 — Box-Muller transform

Sampling $\mathcal{N}(0, 1)$ from uniform random numbers. Let $U_1, U_2 \sim \mathrm{Unif}(0, 1)$. Define
$$Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2), \quad Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2).$$
Claim: $(Z_1, Z_2)$ is a standard 2D Gaussian.

*Proof by change of variables.* In polar form, $R = \sqrt{-2 \ln U_1}$, $\Theta = 2\pi U_2$, so $U_1 = e^{-R^2/2}$ and $U_2 = \Theta/(2\pi)$. The Jacobian:
$$\det\frac{\partial(U_1, U_2)}{\partial(R, \Theta)} = \det\begin{pmatrix} -R e^{-R^2/2} & 0 \\ 0 & 1/(2\pi) \end{pmatrix} = -\frac{R e^{-R^2/2}}{2\pi}.$$
The density of $(R, \Theta)$ is (in absolute value) $(R/(2\pi)) e^{-R^2/2}$ for $R > 0, \Theta \in (0, 2\pi)$. Converting to $(Z_1, Z_2) = (R\cos\Theta, R\sin\Theta)$ with polar Jacobian $dz_1\, dz_2 = R\, dr\, d\theta$:
$$f_{Z_1, Z_2}(z_1, z_2) = \frac{(1/(2\pi)) e^{-R^2/2} \cdot R}{R} = \frac{1}{2\pi} e^{-(z_1^2 + z_2^2)/2},$$
the standard bivariate normal. $\blacksquare$

This **Box-Muller transform** is how every random-number generator in scientific computing constructs Gaussians from uniforms.

#### Example 0.3.6.6 — Volume of a simplex

The simplex $\Delta_n = \{\mathbf{x} \in \mathbb{R}_{\geq 0}^n : \sum x_i \leq 1\}$. Use the recursive formula with the substitution $x_n = t$, $0 \leq t \leq 1$ and $(x_1, \ldots, x_{n-1}) \in (1-t)\Delta_{n-1}$:
$$\mathrm{vol}(\Delta_n) = \int_0^1 \mathrm{vol}((1-t)\Delta_{n-1})\, dt = \int_0^1 (1-t)^{n-1} \mathrm{vol}(\Delta_{n-1})\, dt = \frac{1}{n} \mathrm{vol}(\Delta_{n-1}).$$
With $\mathrm{vol}(\Delta_0) = 1$: $\mathrm{vol}(\Delta_n) = 1/n!$. (The scaling step uses affine change of variables: $\mathbf{u} = \mathbf{x}/(1-t)$ has Jacobian $(1-t)^{n-1}$.)

### Computational Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Example 1: Verify polar-coordinate Jacobian numerically
# Monte Carlo: sample uniformly in a square, reject outside disk of radius R
np.random.seed(42)
R = 2.0

# Integral of e^{-x^2-y^2} over disk of radius R
# Analytical: int_0^{2pi} int_0^R e^{-r^2} r dr dtheta = 2pi * (1/2)(1 - e^{-R^2}) = pi(1 - e^{-R^2})
analytical = np.pi * (1 - np.exp(-R**2))

# MC sampling in the disk in Cartesian
N = 200_000
theta = np.random.uniform(0, 2*np.pi, N)
r = R * np.sqrt(np.random.uniform(0, 1, N))  # so that area element is uniform in disk
x, y = r*np.cos(theta), r*np.sin(theta)
integrand = np.exp(-x**2 - y**2)
mc_cartesian = integrand.mean() * (np.pi * R**2)
print(f"Integral (analytical): {analytical:.6f}")
print(f"Integral (MC Cartesian): {mc_cartesian:.6f}")

# MC sampling in polar: draw (r, theta) uniform, weight by Jacobian r
r_p = np.random.uniform(0, R, N)
theta_p = np.random.uniform(0, 2*np.pi, N)
integrand_polar = np.exp(-r_p**2) * r_p  # includes Jacobian
mc_polar = integrand_polar.mean() * (R * 2*np.pi)
print(f"Integral (MC polar w/ Jacobian): {mc_polar:.6f}")

# Example 2: Box-Muller as a test of change-of-variable
U1, U2 = np.random.uniform(size=(2, 100_000))
Z1 = np.sqrt(-2*np.log(U1)) * np.cos(2*np.pi*U2)
Z2 = np.sqrt(-2*np.log(U1)) * np.sin(2*np.pi*U2)
print(f"\nBox-Muller: mean(Z1) = {Z1.mean():.4f}, std(Z1) = {Z1.std():.4f}")
print(f"mean(Z2) = {Z2.mean():.4f}, std(Z2) = {Z2.std():.4f}")
print(f"correlation(Z1, Z2) = {np.corrcoef(Z1, Z2)[0,1]:.4f}")

# Example 3: Density transformation under affine map
# Y = A X + mu, X ~ N(0, I). Verify that f_Y matches the formula.
A = np.array([[1.5, 0.3], [-0.5, 2.0]])
mu = np.array([1.0, -0.5])
n_samples = 100_000
X = np.random.randn(n_samples, 2)
Y = X @ A.T + mu

# Theoretical density at a grid
x_grid = np.linspace(-4, 6, 80)
y_grid = np.linspace(-6, 4, 80)
Xg, Yg = np.meshgrid(x_grid, y_grid)
coords = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
Sigma = A @ A.T
Sigma_inv = np.linalg.inv(Sigma)
det_Sigma = np.linalg.det(Sigma)
diff = coords - mu
mahalanobis = np.einsum('ij,jk,ik->i', diff, Sigma_inv, diff)
f_theory = (1/(2*np.pi*np.sqrt(det_Sigma))) * np.exp(-0.5*mahalanobis)
f_theory = f_theory.reshape(Xg.shape)

# Empirical histogram
H, xe, ye = np.histogram2d(Y[:,0], Y[:,1], bins=[x_grid, y_grid], density=True)
# H is on bin centers; approximate comparison
print(f"\nDensity transformation check:")
print(f"  det(A) = {np.linalg.det(A):.4f}, |det A|^{-1} = {1/abs(np.linalg.det(A)):.4f}")
print(f"  det(Sigma)^{{1/2}} = {np.sqrt(det_Sigma):.4f} (should equal |det A|)")
print(f"  Peak empirical density: {H.max():.4f}; peak theoretical: {f_theory.max():.4f}")

# Example 4: Normalizing flow snippet (density via stacked affine-coupling)
# A minimal illustrative example: Y = x^3 + x (bijection of R)
# f_Y(y) = f_X(g^{-1}(y)) / g'(g^{-1}(y))
def g(x): return x**3 + x
def g_inv(y, x0=None):
    # Solve x^3 + x - y = 0 for real x (cubic with positive discriminant)
    # Using formula: x = (y/2 + sqrt(y^2/4 + 1/27))^{1/3} + (y/2 - sqrt(y^2/4 + 1/27))^{1/3}
    disc = y**2/4 + 1/27
    u = y/2 + np.sqrt(disc)
    v = y/2 - np.sqrt(disc)
    return np.sign(u)*np.abs(u)**(1/3) + np.sign(v)*np.abs(v)**(1/3)

# Verify: if X ~ N(0,1), what is f_Y?
x_test = np.linspace(-3, 3, 100)
y_test = g(x_test)
# Density: f_X(x) / g'(x) where y = g(x)
g_prime = 3*x_test**2 + 1
f_X = (1/np.sqrt(2*np.pi)) * np.exp(-x_test**2/2)
f_Y_analytical = f_X / g_prime

# Compare with MC
X_samples = np.random.randn(200_000)
Y_samples = g(X_samples)
hist, bins = np.histogram(Y_samples, bins=100, range=(y_test.min(), y_test.max()), density=True)
print(f"\nNormalizing flow Y = X^3 + X, X ~ N(0,1)")
print(f"  MC density peak: {hist.max():.4f}")
print(f"  Analytical density peak: {f_Y_analytical.max():.4f}")
```

### [QUANT APPLICATION] Measure Change, Normalizing Flows, and Girsanov

**(A) Importance sampling with density ratios.** To estimate $\mathbb{E}_p[f(X)]$ when $p$ is hard to sample, draw from a proposal $q$ and reweight: $\mathbb{E}_p[f(X)] = \mathbb{E}_q[f(X) \cdot p(X)/q(X)]$. The ratio $p/q$ is a special case of the change-of-variable Jacobian when both are densities over the same space. In path-space (SDEs), this is **Girsanov's theorem**.

**(B) Normalizing flows.** Given a learnable invertible map $g_\theta: \mathbb{R}^n \to \mathbb{R}^n$ with a known (easy) base density $p_Z$, the pullback density is
$$p_X(\mathbf{x}) = p_Z(g_\theta^{-1}(\mathbf{x})) \cdot |\det D g_\theta^{-1}(\mathbf{x})|.$$
For training, maximize log-likelihood $\sum_i \log p_X(\mathbf{x}_i)$, which requires computing $\log |\det Dg^{-1}|$ efficiently. Three standard architectures:
- **Affine coupling (RealNVP)**: split $\mathbf{x} = (\mathbf{x}_A, \mathbf{x}_B)$; $\mathbf{y}_A = \mathbf{x}_A$, $\mathbf{y}_B = \mathbf{x}_B \odot e^{s(\mathbf{x}_A)} + t(\mathbf{x}_A)$. The Jacobian is lower triangular with determinant $\prod e^{s_i}$. Compute in $O(n)$.
- **Autoregressive (MAF, IAF)**: $y_i = x_i \cdot e^{s_i(\mathbf{x}_{<i})} + t_i(\mathbf{x}_{<i})$. Triangular Jacobian, $O(n)$ det.
- **Continuous normalizing flows (Neural ODE / FFJORD)**: $\mathbf{y}(T) = \mathbf{x}(0) + \int_0^T h_\theta(\mathbf{x}(t), t)\, dt$. The log-det satisfies a trace ODE: $\partial_t \log|\det| = \mathrm{tr}(\nabla_\mathbf{x} h_\theta)$. Hutchinson's stochastic trace estimator enables $O(n)$ estimation.

**(C) Risk-neutral pricing.** Under the risk-neutral measure $\mathbb{Q}$, $\mathbb{E}^\mathbb{Q}[e^{-rT} f(S_T)] = \mathbb{E}^\mathbb{P}\!\left[ e^{-rT} f(S_T) \frac{d\mathbb{Q}}{d\mathbb{P}} \right]$. The Radon-Nikodym derivative $d\mathbb{Q}/d\mathbb{P}$ is the Girsanov density — an infinite-dimensional cousin of the Jacobian, giving a multiplicative scaling of path-space measure when the drift is changed. This is the mechanism that converts a *real-world* Brownian motion into a *risk-neutral* one for pricing.

**(D) Copulas and dependence modeling.** If $(U_1, \ldots, U_n)$ is a copula (uniform marginals), and $F_i^{-1}$ are the inverse CDFs of desired marginals, then $X_i = F_i^{-1}(U_i)$ gives a multivariate distribution with specified marginals and the copula's dependence structure. The joint density involves the copula density (a Jacobian computation):
$$f_{\mathbf{X}}(\mathbf{x}) = c(F_1(x_1), \ldots, F_n(x_n)) \prod_i f_i(x_i).$$
Gaussian copulas (CDO pricing), $t$-copulas (tail risk), Archimedean copulas — all are density transformations.

**(E) Stochastic volatility and volatility surface generation.** Some models express the joint density of (spot, vol) after an appropriate nonlinear change of variable; the Jacobian appears in local-vol calibration (Dupire's formula involves $\partial_T C/\partial_K^2 C$, a derivative ratio that is morally a change-of-measure Radon-Nikodym).

**(F) Bayesian posteriors and reparametrization tricks.** Variational autoencoders sample latent variables $\mathbf{z} = \mu + \sigma \odot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$. The reparametrization is an affine change of variables; its log-determinant $\sum \log \sigma_i$ is the entropy term in the ELBO, capturing how "spread out" the posterior proposal is.

### Exercises

#### ★ (Foundation)

**E0.3.6.1.** Compute $\int_0^R \int_0^{2\pi} r^3 e^{-r^2}\, d\theta\, dr$ using polar coordinates; verify by direct computation.

**E0.3.6.2.** Use change of variables to show the area of the ellipse $x^2/a^2 + y^2/b^2 \leq 1$ is $\pi a b$. (Hint: $\varphi(u, v) = (au, bv)$.)

**E0.3.6.3.** Show that if $X$ is absolutely continuous with density $f_X$, then $aX + b$ (for $a \neq 0$) has density $|a|^{-1} f_X((\cdot - b)/a)$. Verify for standard Gaussian shift and scale.

**E0.3.6.4.** Compute $\int_0^\infty e^{-x^2/2}\, dx$ by the polar squaring trick; compare to the half of the Gaussian integral.

#### ★★ (Intermediate)

**E0.3.6.5 (Volume of $n$-sphere).** Deduce $\mathrm{vol}(S^{n-1}) = 2\pi^{n/2}/\Gamma(n/2)$ from the Gaussian integral and $(n-1)$-sphere surface measure decomposition.

**E0.3.6.6 (Jacobian chain rule).** Show directly from the change-of-variables theorem that if $\varphi = \psi \circ \eta$ (composition of diffeomorphisms), then $|\det D\varphi| = |\det D\psi||\det D\eta|$; this matches $\det(AB) = \det A \det B$.

**E0.3.6.7 (Cylindrical coordinates).** In $\mathbb{R}^3$, $\varphi(r, \theta, z) = (r\cos\theta, r\sin\theta, z)$. Compute the Jacobian; use it to find the volume of a cone $\{r \leq z, 0 \leq z \leq h\}$.

**E0.3.6.8 (Gaussian tail in $n$ dimensions).** For $X \sim \mathcal{N}(0, I_n)$, compute $\mathbb{P}(\|X\| > R)$ as a function of $R$ and $n$. (The answer involves the regularized incomplete gamma function.)

**E0.3.6.9 (Density of norm).** For $X \sim \mathcal{N}(0, I_n)$, derive the density of $R = \|X\|$; show $R^2 \sim \chi^2_n$.

**E0.3.6.10 (Change of measure in SDE).** Informally explain Girsanov's theorem: a drift change $dX_t = \mu\, dt + dB_t \to dX_t = dB_t$ is accomplished by a density $\exp(-\mu B_T - \mu^2 T/2)$ on path space. Relate this to the finite-dimensional case of shifting a Gaussian: density ratio is $\exp(-\boldsymbol\mu^\top \mathbf{x} - \|\boldsymbol\mu\|^2/2)$.

#### ★★★ (Challenge)

**E0.3.6.11 (Co-area formula).** State and prove the **co-area formula**: for a Lipschitz function $u: \mathbb{R}^n \to \mathbb{R}$ and integrable $g$,
$$\int_{\mathbb{R}^n} g(\mathbf{x}) |\nabla u(\mathbf{x})|\, d\mathbf{x} = \int_{\mathbb{R}} \left(\int_{u^{-1}(t)} g\, d\mathcal{H}^{n-1}\right) dt,$$
where $\mathcal{H}^{n-1}$ is $(n-1)$-Hausdorff measure. This is a "generalized Fubini" for level sets and underlies geometric measure theory.

**E0.3.6.12 (Non-square Jacobians / Area formula).** Let $\varphi: U \to \mathbb{R}^N$ be a $C^1$-injective map with $U \subseteq \mathbb{R}^n$, $N > n$. Prove the **area formula**: $\mathcal{H}^n(\varphi(A)) = \int_A \sqrt{\det(D\varphi^\top D\varphi)}\, d\mathbf{x}$. This generalizes change of variables to embedded submanifolds (surfaces in $\mathbb{R}^3$, say).

**E0.3.6.13 (Jacobian free energies).** In statistical physics, the partition function $Z = \int e^{-\beta H(\mathbf{x})} d\mathbf{x}$ can be re-expressed in canonically conjugate variables via change of variable. Interpret the Jacobian as a contribution to the "entropy" in the free energy. (This is why physicists love canonical transformations.)

**E0.3.6.14 (Radon transform).** Define $\mathcal{R} f(\boldsymbol\theta, t) = \int_{\boldsymbol\theta \cdot \mathbf{x} = t} f(\mathbf{x})\, d\sigma$ (integral over hyperplanes). Prove the inversion formula involves a Jacobian factor; this is the basis of CT scanning.

**E0.3.6.15 (Brenier's theorem / optimal transport).** Read and state Brenier's theorem: the optimal transport map pushing a measure $\mu$ to $\nu$ under quadratic cost is the gradient of a convex function $T = \nabla \varphi$. The Jacobian of $T$ is the Hessian of $\varphi$; the Monge-Ampère equation $\det \nabla^2 \varphi = \mu/\nu(T)$ is the change-of-variables formula in this context.

---

## Module 0.3 Summary and Forward Pointers

We built the infrastructure of **multivariable calculus formalized**, the framework on which every later module depends.

### What we covered

- **Fréchet vs Gâteaux derivatives** (0.3.1). The Fréchet derivative is the right generalization of the single-variable derivative: a linear map with a *uniform* linear approximation. We saw that continuous partials $\Rightarrow$ Fréchet differentiable, and constructed the pathological $x^3 y/(x^4 + y^2)$ example showing directional derivatives alone are too weak.
- **Chain rule and Taylor's theorem** (0.3.2). Chain rule as matrix product; mean value inequality via the Hahn-Banach lemma; Taylor's theorem with Lagrange, integral, and Peano remainders; first- and second-order optimality conditions using the Hessian.
- **Inverse Function Theorem** (0.3.3). Banach fixed-point theorem proved in full; the IFT gives smooth local invertibility from invertible derivative; diffeomorphisms; local-vs-global subtleties illustrated by polar coordinates.
- **Implicit Function Theorem** (0.3.4). Derived from IFT via the auxiliary map trick; ImFT as the engine of differential geometry (manifolds as zero sets of regular values), of equilibrium theory (comparative statics), and of calibration (market-consistent Greeks).
- **Lagrange multipliers and KKT conditions** (0.3.5). Lagrange via reduction to an unconstrained problem on the constraint manifold; KKT via Farkas' lemma; second-order conditions with bordered Hessian; the multipliers as shadow prices / marginal sensitivities; convexity turns KKT sufficient.
- **Change of variables** (0.3.6). The Jacobian determinant as the *unique* continuous multiplicative volume-scaling factor; the polar-spherical-cylindrical toolkit; the change-of-measure machinery that powers Girsanov, normalizing flows, and every density transformation in statistics.

### Synthesis

The six topics form a tight web. IFT and ImFT are *equivalent*; both rest on Banach fixed-point, which is itself a specialization of the contraction principle. Lagrange multipliers are proved *via* ImFT. The change of variables theorem generalizes differential-geometric intuition from IFT. Every result you will need in real analysis, probability, optimization, or geometry lives on top of this apparatus.

### Quant applications we touched

- Market calibration as an implicit-function problem; market-consistent Greeks.
- Portfolio optimization (Markowitz, risk parity, SVM) as KKT programs.
- Density transformations: multivariate Gaussians, Box-Muller, copulas.
- Normalizing flows (RealNVP, MAF, FFJORD) as learnable diffeomorphisms trained by log-Jacobian.
- Girsanov's theorem previewed as an infinite-dimensional change-of-measure.
- Envelope theorem and shadow prices in constrained optimization.

### What we did *not* cover (and where it's picked up)

- **Lebesgue integration and measure theory** are assumed at a working level. The full construction, Fubini's theorem, and the dominated/monotone convergence theorems are Subject 1 (Measure Theory).
- **Differential forms and Stokes' theorem** — the coordinate-free version of the gradient/divergence/curl/fundamental theorem of calculus — are deferred to Subject 4 (Differential Geometry).
- **Banach-space calculus with unbounded operators** (for SDE and PDE applications) is touched on only briefly; Module 0.4 (Real Analysis / Metric Spaces) continues the story.
- **Morse theory and critical-point analysis** (which classify smooth functions by their critical structure) will appear in Subject 4.

### Forward pointers

- **Module 0.4 (Real Analysis / Metric Spaces)**: completes the topology behind the theorems we used. Sequential compactness, completeness, Heine-Borel, continuity vs. uniform continuity — the bedrock beneath everything in this module.
- **Module 0.5 (Complex Analysis)**: adds the holomorphic dimension; Cauchy-Riemann, Cauchy's integral formula, residues, contour integration — essential for characteristic functions, Fourier transforms, and pricing.
- **Subject 1 (Measure and Integration)**: rebuilds $\int$ on a measure-theoretic foundation. Product measures, change-of-measure theorems, Radon-Nikodym, $L^p$-spaces.
- **Subject 2 (Probability)**: once we have measure theory, we will revisit Gaussian densities and the Girsanov change-of-measure with full rigor.
- **Subject 3 (Stochastic Calculus)**: Itô calculus is a nonlinear change-of-variable for random paths; the chain rule there is **Itô's formula**, which differs from the classical chain rule by a Hessian correction $\tfrac{1}{2} \mathrm{tr}(\sigma\sigma^\top \nabla^2 f)$. Understanding this "quadratic variation" correction is the reason we spent so much time on Hessians and Taylor's theorem in this module.
- **Subject 5 (Optimization)**: KKT reappears with full convex-analytic, subdifferential, and interior-point-method treatment.

*Next up: Module 0.4 — Real Analysis and Metric Spaces.*

---


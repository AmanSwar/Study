# Module 0.5 — Complex Analysis

*Part of "Mathematical Foundations for Quantitative Research: From JEE to Jane Street" — Subject 0 (Quant Foundations), Module 5.*

---

## Table of Contents

- **Topic 0.5.1** — Complex Numbers and Cauchy-Riemann Equations
- **Topic 0.5.2** — Contour Integration and Cauchy's Theorem
- **Topic 0.5.3** — Cauchy's Integral Formula and Consequences
- **Topic 0.5.4** — Laurent Series and Residues
- **Topic 0.5.5** — Analytic Continuation and Special Functions
- **Topic 0.5.6** — Fourier Transforms and Characteristic Functions

---

## Module Overview

Complex analysis is the most elegant subject in undergraduate mathematics, and one of the most *useful* in quantitative finance. Adding a single axis to the real line — the imaginary dimension — and demanding that functions be differentiable in the complex sense (the Cauchy-Riemann equations) produces a theory of astonishing rigidity: a complex-differentiable function has derivatives of all orders, is locally a power series, is determined by its values on any convergent sequence in its domain, and integrates to zero around any loop in a simply connected region.

For a quant, complex analysis is not a museum piece. It is the computational engine behind:
- **Fourier methods in option pricing** — the Carr-Madan formula, Heston model, Lévy processes all use characteristic functions (complex-valued) inverted by contour deformation.
- **Characteristic functions** in probability — defined as $\varphi(t) = \mathbb{E}[e^{itX}]$, always well-defined, and they *determine* a distribution. Lévy's inversion formula is a complex-analytic identity.
- **The Riemann zeta function** — shows up in interest-rate models and in the analysis of stochastic integrals via zeta-regularization.
- **Hilbert transforms and Kramers-Kronig relations** — in implied-volatility surface arbitrage-free calibration.
- **Residue calculus** — for evaluating integrals arising in European option prices under non-Gaussian models.
- **Analytic continuation** — model-free arbitrage bounds on option prices are often discovered by analytic continuation from known regions.

This module builds the theory from scratch, emphasizing both the abstract (Cauchy's theorem, residues) and the concrete (contour-integration techniques, inversion formulas for characteristic functions).

*Prerequisites: Modules 0.1 (Logic), 0.2 (Linear Algebra), 0.3 (Multivariable Calculus), 0.4 (Real Analysis).*

---


## Topic 0.5.1 — Complex Numbers and Cauchy-Riemann Equations

### Motivation

Complex differentiability is a much stronger condition than real differentiability. A real function $f: \mathbb{R}^2 \to \mathbb{R}^2$ is differentiable at a point if it has a linear approximation (a Jacobian). A *complex* function $f: \mathbb{C} \to \mathbb{C}$, by contrast, must be differentiable as a complex function: its derivative $f'(z_0) \in \mathbb{C}$ is a single complex number giving a linear approximation $f(z) \approx f(z_0) + f'(z_0)(z - z_0)$ with *complex* multiplication. Because complex multiplication by $\alpha = a + ib$ is the real-linear map $(x, y) \mapsto (ax - by, bx + ay)$ — a rotation-scaling — complex differentiability is the same as real differentiability *plus* the requirement that the Jacobian be of rotation-scaling form. This extra constraint is the **Cauchy-Riemann equations** and is the source of every miracle in complex analysis.

This topic lays the groundwork: review of $\mathbb{C}$, definition of holomorphy, Cauchy-Riemann equations in both real and complex form, harmonic conjugates, elementary functions (exponential, trigonometric, logarithm, powers).

### Prerequisites

- Real multivariable calculus (Topic 0.3.1–0.3.2).
- Basic linear algebra: change of basis, rotations (Topic 0.2.2).

### The Complex Plane

**Definition 0.5.1.1 (Complex numbers).** $\mathbb{C} = \{a + bi : a, b \in \mathbb{R}\}$ with $i^2 = -1$. Addition is coordinate-wise: $(a + bi) + (c + di) = (a + c) + (b + d)i$. Multiplication: $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$.

- **Real part**: $\mathrm{Re}(a + bi) = a$.
- **Imaginary part**: $\mathrm{Im}(a + bi) = b$.
- **Complex conjugate**: $\overline{a + bi} = a - bi$.
- **Modulus**: $|a + bi| = \sqrt{a^2 + b^2}$. Note $|z|^2 = z \bar z$.
- **Argument**: $\arg(z) = \theta$ where $z = |z|(\cos\theta + i\sin\theta) = |z|e^{i\theta}$. Multi-valued modulo $2\pi$; principal value $\mathrm{Arg}(z) \in (-\pi, \pi]$.

**Polar form.** $z = re^{i\theta}$ with $r = |z|$, $\theta = \arg z$. Multiplication becomes $(r_1 e^{i\theta_1})(r_2 e^{i\theta_2}) = r_1 r_2 e^{i(\theta_1 + \theta_2)}$ — magnitudes multiply, arguments add.

$\mathbb{C}$ is a field: $\mathbb{C}^* = \mathbb{C} \setminus \{0\}$ is a multiplicative group. As a real vector space, $\mathbb{C} \cong \mathbb{R}^2$ with basis $\{1, i\}$. As a metric space, $\mathbb{C}$ is isometric to $\mathbb{R}^2$ via $a + bi \leftrightarrow (a, b)$ with Euclidean distance; in particular, $\mathbb{C}$ is complete.

**Topology.** Open disks $D(z_0, r) = \{z : |z - z_0| < r\}$ are the basic open sets, making $\mathbb{C}$ homeomorphic to $\mathbb{R}^2$. A **domain** in $\mathbb{C}$ is an open, connected subset.

### Complex Differentiability

**Definition 0.5.1.2 (Complex differentiability).** Let $\Omega \subseteq \mathbb{C}$ open, $f: \Omega \to \mathbb{C}$, $z_0 \in \Omega$. Then $f$ is **(complex) differentiable** at $z_0$ iff the limit
$$f'(z_0) = \lim_{h \to 0} \frac{f(z_0 + h) - f(z_0)}{h}$$
exists, where $h \in \mathbb{C}$ can approach $0$ from any direction.

**Definition 0.5.1.3 (Holomorphic / analytic function).** $f$ is **holomorphic** (or **analytic**) on an open set $\Omega$ iff $f$ is complex-differentiable at every point of $\Omega$. *We will use "holomorphic" and "analytic" interchangeably — they are equivalent (Theorem 0.5.3.x below).*

**Definition 0.5.1.4 (Entire function).** A function holomorphic on all of $\mathbb{C}$ is called **entire**.

The key subtlety: for the limit in Definition 0.5.1.2 to exist, the "direction-dependent" limits must all agree. Approaching $h \to 0$ along the real axis ($h = t \in \mathbb{R}$) or along the imaginary axis ($h = it$, $t \in \mathbb{R}$) must give the same result.

### Cauchy-Riemann Equations

Write $f(z) = f(x + iy) = u(x, y) + iv(x, y)$ where $u, v: \Omega \to \mathbb{R}$.

**Theorem 0.5.1.5 (Cauchy-Riemann equations — necessary form).** *If $f = u + iv$ is complex-differentiable at $z_0 = x_0 + iy_0$, then the partial derivatives $u_x, u_y, v_x, v_y$ exist at $(x_0, y_0)$ and satisfy the **Cauchy-Riemann equations***
$$u_x = v_y, \qquad u_y = -v_x. \qquad \square$$
*Moreover, $f'(z_0) = u_x + iv_x = v_y - iu_y$.*

**Proof.** Take $h = t \in \mathbb{R}$, $t \to 0$:
$$f'(z_0) = \lim_{t \to 0} \frac{u(x_0 + t, y_0) - u(x_0, y_0)}{t} + i \cdot \lim_{t \to 0} \frac{v(x_0 + t, y_0) - v(x_0, y_0)}{t} = u_x + iv_x.$$

Take $h = it$, $t \in \mathbb{R}$, $t \to 0$ (so $h \to 0$ along the imaginary axis; note $1/h = 1/(it) = -i/t$):
$$f'(z_0) = \lim_{t \to 0} \frac{u(x_0, y_0 + t) - u(x_0, y_0)}{it} + i \cdot \lim_{t \to 0} \frac{v(x_0, y_0 + t) - v(x_0, y_0)}{it} = \frac{u_y}{i} + \frac{iv_y}{i} = -i u_y + v_y.$$

Setting the two expressions for $f'(z_0)$ equal: $u_x + iv_x = v_y - iu_y$. Comparing real and imaginary parts: $u_x = v_y$ and $v_x = -u_y$, i.e., the C-R equations. $\blacksquare$

**Theorem 0.5.1.6 (Cauchy-Riemann — sufficient form).** *If $u, v: \Omega \to \mathbb{R}$ are $C^1$ (continuously differentiable) on $\Omega$ and satisfy C-R at a point $z_0$, then $f = u + iv$ is complex-differentiable at $z_0$, with $f'(z_0) = u_x(z_0) + iv_x(z_0)$.*

**Proof.** By the multivariate chain rule (Topic 0.3.1), for real $h = (h_1, h_2)$:
$$u(x_0 + h_1, y_0 + h_2) - u(x_0, y_0) = u_x h_1 + u_y h_2 + r_1(h), \quad r_1(h)/|h| \to 0.$$
$$v(x_0 + h_1, y_0 + h_2) - v(x_0, y_0) = v_x h_1 + v_y h_2 + r_2(h), \quad r_2(h)/|h| \to 0.$$

Write $h = h_1 + i h_2 \in \mathbb{C}$. Then
$$f(z_0 + h) - f(z_0) = (u_x h_1 + u_y h_2) + i(v_x h_1 + v_y h_2) + r_1(h) + i r_2(h).$$

Using C-R ($u_y = -v_x$, $v_y = u_x$):
$$= u_x h_1 - v_x h_2 + i(v_x h_1 + u_x h_2) + R(h) = (u_x + iv_x)(h_1 + ih_2) + R(h) = (u_x + iv_x) h + R(h),$$
with $R(h)/|h| \to 0$. Dividing by $h$:
$$\frac{f(z_0 + h) - f(z_0)}{h} = (u_x + iv_x) + \frac{R(h)}{h}.$$
$|R(h)/h| = |R(h)|/|h| \to 0$ as $h \to 0$, so the limit exists and equals $u_x + iv_x$. $\blacksquare$

**Remark.** The $C^1$ hypothesis is essential. A function with partials satisfying C-R but not continuous differentiability may fail to be complex-differentiable. (Looman-Menchoff theorem: if $u, v$ are merely continuous and have partials satisfying C-R everywhere, then $f = u + iv$ is holomorphic — but this requires serious work.)

### Cauchy-Riemann in Complex Form

The C-R equations have a very clean complex restatement. Define the **Wirtinger derivatives**:
$$\frac{\partial}{\partial z} = \frac{1}{2}\left(\frac{\partial}{\partial x} - i \frac{\partial}{\partial y}\right), \qquad \frac{\partial}{\partial \bar z} = \frac{1}{2}\left(\frac{\partial}{\partial x} + i \frac{\partial}{\partial y}\right).$$
These are linear operators on smooth functions; the reason for their form is the symbolic identity $z = x + iy$, $\bar z = x - iy$, so $x = (z + \bar z)/2$, $y = (z - \bar z)/(2i)$, and formal chain rule gives the above.

**Proposition 0.5.1.7.** *$f = u + iv$ ($C^1$) satisfies Cauchy-Riemann iff $\partial f/\partial \bar z = 0$.*

**Proof.** $\partial f/\partial \bar z = \tfrac{1}{2}(f_x + i f_y) = \tfrac{1}{2}((u_x + iv_x) + i(u_y + iv_y)) = \tfrac{1}{2}((u_x - v_y) + i(v_x + u_y))$. Both components vanish iff C-R holds. $\blacksquare$

So: $f$ is holomorphic iff it is "independent of $\bar z$" — intuitively, its Taylor expansion (if it has one) in $(z, \bar z)$ has no $\bar z$ terms.

### Harmonic Conjugates

**Definition 0.5.1.8 (Harmonic function).** $u: \Omega \to \mathbb{R}$ is **harmonic** iff $u \in C^2$ and $\Delta u = u_{xx} + u_{yy} = 0$.

**Proposition 0.5.1.9.** *If $f = u + iv$ is holomorphic and $C^2$, then $u$ and $v$ are harmonic.*

**Proof.** Differentiating C-R: $u_{xx} = (v_y)_x = v_{xy}$, and $u_{yy} = (-v_x)_y = -v_{xy}$. Sum: $u_{xx} + u_{yy} = 0$ (assuming $v \in C^2$, so $v_{xy} = v_{yx}$ by Schwarz). Similarly for $v$. $\blacksquare$

**Definition 0.5.1.10 (Harmonic conjugate).** Given harmonic $u: \Omega \to \mathbb{R}$ (with $\Omega$ simply connected), a **harmonic conjugate** is a harmonic $v$ such that $f = u + iv$ is holomorphic — equivalently, C-R holds.

**Proposition 0.5.1.11.** *On a simply connected domain, every harmonic function has a harmonic conjugate, unique up to an additive real constant.*

**Proof sketch.** C-R specifies $v_x = -u_y, v_y = u_x$. So we seek $v$ with $\nabla v = (-u_y, u_x)$. This vector field is closed: $\partial(-u_y)/\partial y = -u_{yy}$ and $\partial u_x/\partial x = u_{xx}$; by harmonicity $u_{xx} = -u_{yy}$, so the mixed partials match. On a simply connected domain, closed implies exact (a fact from vector calculus / deRham cohomology), so $v$ exists with $\nabla v = (-u_y, u_x)$, unique up to constant. $\blacksquare$

### Elementary Functions

**Exponential.** Defined as the extension of the real exponential to complex variables preserving the Taylor series:
$$e^z := \sum_{n=0}^\infty \frac{z^n}{n!}.$$
This series converges for all $z \in \mathbb{C}$. Key properties:

- $e^{z + w} = e^z e^w$. (Proved by multiplying series and using the binomial theorem.)
- For $z = x + iy$: $e^z = e^x(\cos y + i \sin y)$ (**Euler's formula**). Proved by grouping real and imaginary parts of the series.
- $(e^z)' = e^z$. By termwise differentiation of the power series (justified below in the radius-of-convergence theorem).
- $e^z \neq 0$ for all $z$. (Else $0 = e^z e^{-z} = e^0 = 1$, contradiction.)
- $e^z$ is periodic with period $2\pi i$: $e^{z + 2\pi i} = e^z$.
- $|e^z| = e^{\mathrm{Re}(z)}$, $\arg e^z = \mathrm{Im}(z)$ mod $2\pi$.

**Trigonometric and hyperbolic.**
$$\sin z = \frac{e^{iz} - e^{-iz}}{2i}, \quad \cos z = \frac{e^{iz} + e^{-iz}}{2}, \quad \sinh z = \frac{e^z - e^{-z}}{2}, \quad \cosh z = \frac{e^z + e^{-z}}{2}.$$

Standard identities from the real case extend (Pythagorean, sum-of-angles). But $\sin z$ and $\cos z$ are *unbounded* on $\mathbb{C}$: $\cos(it) = \cosh t \to \infty$ as $t \to \infty$. This contrasts with the real bounded behavior.

**Logarithm.** Define $\log z = \ln |z| + i \arg z$, but $\arg z$ is multi-valued modulo $2\pi$. So $\log z$ is multi-valued. A **branch** is a continuous single-valued choice on some domain; the **principal branch** takes $\arg z \in (-\pi, \pi]$ and is holomorphic on $\mathbb{C} \setminus (-\infty, 0]$ (the "slit plane").

**Complex powers.** $z^w := e^{w \log z}$ — again multi-valued via the choice of $\log$. Principal branch on the slit plane. Note $z^{1/2}$ has the two square-root values differing by sign; a branch cut along the negative real axis is the usual choice.

### Worked Examples

#### Example 0.5.1.12 — $f(z) = z^2$ is holomorphic; $f(z) = \bar z$ is not

$f(z) = z^2$: $u = x^2 - y^2$, $v = 2xy$. C-R: $u_x = 2x = v_y$; $u_y = -2y = -v_x$ ($v_x = 2y$). Holds everywhere; $f$ is entire, $f'(z) = u_x + iv_x = 2x + 2iy = 2z$.

$f(z) = \bar z$: $u = x$, $v = -y$. C-R: $u_x = 1$ but $v_y = -1$. Fails. So $\bar z$ is nowhere complex-differentiable, despite being $C^\infty$ as a real map.

#### Example 0.5.1.13 — C-R from the Wirtinger viewpoint

$f(z) = \bar z$: $\partial f/\partial \bar z = 1 \neq 0$, confirms non-holomorphy.

$f(z, \bar z) = z \bar z = |z|^2$: $\partial f / \partial \bar z = z$, which vanishes only at $z = 0$. So $|z|^2$ is complex-differentiable only at $0$, with derivative $0$ there. It is not holomorphic anywhere (not even at $0$, because holomorphy requires an open neighborhood).

#### Example 0.5.1.14 — Harmonic conjugate construction

$u(x, y) = x^2 - y^2$ (harmonic: $\Delta u = 2 - 2 = 0$). Find $v$ with $v_x = -u_y = 2y$ and $v_y = u_x = 2x$. Integrate: $v = 2xy + C(x)$; differentiate in $x$: $v_x = 2y + C'(x) = 2y$, so $C'(x) = 0$, i.e., $C$ is a constant. Hence $v = 2xy + c$. The holomorphic function is $f(z) = (x^2 - y^2) + 2xy i + ic = z^2 + ic$, as expected.

#### Example 0.5.1.15 — Geometric meaning of multiplication

Multiplication by $\alpha = re^{i\theta}$ is a rotation by $\theta$ composed with a scaling by $r$. Locally, a holomorphic $f$ with $f'(z_0) \neq 0$ acts as a "rotation-scaling" on infinitesimals: $f(z_0 + h) - f(z_0) \approx f'(z_0) h$. This is the geometric content of being **conformal**: holomorphic maps with non-vanishing derivative preserve angles (and orientation), a classical foundation of potential theory and map-making.

### Computational Implementation

```python
import numpy as np

# ---------- Verifying C-R numerically ----------
def check_CR(f, z0, eps=1e-6):
    """Numerical check of Cauchy-Riemann: f_x - f_y/i should ≈ 0."""
    fx = (f(z0 + eps) - f(z0 - eps)) / (2 * eps)
    fy = (f(z0 + 1j*eps) - f(z0 - 1j*eps)) / (2 * eps)
    # C-R: f_x = f_y / i  ->  f_x - f_y/i = 0
    residual = fx - fy / 1j
    return np.abs(residual)

tests = {
    'z^2': lambda z: z**2,
    'z^3': lambda z: z**3,
    'exp(z)': lambda z: np.exp(z),
    'conjugate(z)': lambda z: np.conj(z),
    '|z|^2': lambda z: np.abs(z)**2,
    'Re(z)': lambda z: z.real + 0j,
}
z0 = 1 + 2j
print("C-R residuals at z0 = 1 + 2i:")
for name, f in tests.items():
    print(f"  {name}: |f_x - f_y/i| = {check_CR(f, z0):.2e}")

# ---------- Polar form visualization ----------
def to_polar(z):
    return abs(z), np.angle(z)

z = 3 + 4j
r, theta = to_polar(z)
print(f"\n{z} = {r} * exp({theta:.4f}i)")
print(f"  Verify: {r * np.exp(1j * theta)}")

# ---------- Multiplication as rotation-scaling ----------
# Multiplying by e^(i*pi/4) rotates 45 degrees
import numpy as np
v = 1 + 0j
rot = np.exp(1j * np.pi / 4)
print(f"\n{v} rotated by pi/4: {v * rot} = e^(i*pi/4)")

# ---------- Harmonic conjugate construction ----------
import sympy as sp
x, y = sp.symbols('x y', real=True)
u = x**2 - y**2
# Check harmonicity
laplacian = sp.diff(u, x, 2) + sp.diff(u, y, 2)
print(f"\nu = {u}, Laplacian = {laplacian}")
# Construct v: v_x = -u_y, v_y = u_x
v_candidate = sp.integrate(-sp.diff(u, y), x)
# Add a function of y to match v_y
residual = sp.diff(u, x) - sp.diff(v_candidate, y)
v = v_candidate + sp.integrate(residual, y)
print(f"Harmonic conjugate: v = {v}")
# f = u + iv should be a holomorphic function of z = x + iy
z = sp.Symbol('z')
print(f"As a function of z: f = {sp.simplify((u + sp.I * v).subs({x: (z + sp.conjugate(z))/2, y: (z - sp.conjugate(z))/(2*sp.I)}))}")

# ---------- Exponential in the complex plane ----------
z_points = np.linspace(-2, 2, 5) + 1j * np.linspace(-np.pi, np.pi, 5)[:, None]
e_z = np.exp(z_points)
print(f"\ne^z on a grid, |e^z| pattern:")
print(np.abs(e_z).round(2))
print(f"arg(e^z) pattern:")
print(np.angle(e_z).round(2))

# ---------- Multi-valuedness of log ----------
z_test = 1 + 1j
# Principal: ln|z| + i Arg(z) with Arg in (-pi, pi]
log_principal = np.log(z_test)
print(f"\nPrincipal log(1+i) = {log_principal}")
# Other branches: log_principal + 2*pi*i*k for integer k
for k in [-1, 0, 1, 2]:
    alt = log_principal + 2j * np.pi * k
    # Check: e^alt = z
    print(f"  Branch k={k}: log = {alt}, e^log = {np.exp(alt)}")
```

### [QUANT APPLICATION] Holomorphy in Quant Work

**(A) Characteristic functions are entire on strips.** For a random variable $X$ with density $f$, the characteristic function $\varphi(t) = \mathbb{E}[e^{itX}] = \int f(x) e^{itx}\, dx$ extends to complex $t$ when $\mathbb{E}[e^{\lambda X}] < \infty$ for some $\lambda$. The holomorphic extension to a complex strip is the starting point of Fourier-inversion option pricing.

**(B) Moment generating functions are holomorphic on strips.** $M(t) = \mathbb{E}[e^{tX}]$ (when it exists in a neighborhood of $0$) is holomorphic. Its Taylor coefficients are the moments of $X$; analytic continuation via complex $t$ yields deep probabilistic identities.

**(C) Stock-price dynamics and conformal maps.** Log-transformations $Y = \log S$ in the Black-Scholes framework map the positive reals to all of $\mathbb{R}$, and further analytic maps (e.g., Joukowski transform) yield conformal mappings exploited in boundary-value problems for pricing PDEs.

**(D) Electric potentials and option prices.** Both satisfy Laplace's equation in suitable coordinates (the Black-Scholes PDE is a heat equation, but after transformation it becomes Laplace-like). The harmonic-conjugate construction gives the associated *stream function* — in finance, the hedging portfolio.

**(E) Wirtinger calculus in machine learning.** Neural networks that operate on complex-valued inputs (e.g., for radar, optics, quantum) use the Wirtinger derivative to define gradients without requiring holomorphy — the relevant "gradient" in non-holomorphic optimization is $\nabla = (\partial/\partial \bar z)$, a standard object since 1930s.

### Exercises

#### ★ (Foundation)

**E0.5.1.1.** Prove that for $f$ holomorphic on a domain, if $f'(z) = 0$ everywhere, then $f$ is constant. (Hint: $u_x = u_y = v_x = v_y = 0$ and connect via paths.)

**E0.5.1.2.** Check directly that $\sin z, \cos z, \sinh z, \cosh z$ are entire, and derive $\sin' z = \cos z$, $\cos' z = -\sin z$.

**E0.5.1.3.** Verify the identity $\sin^2 z + \cos^2 z = 1$ for complex $z$.

**E0.5.1.4.** Find all harmonic conjugates of $u(x, y) = e^x \cos y$. What is the resulting holomorphic function?

**E0.5.1.5.** Show that $f(z) = 1/z$ is holomorphic on $\mathbb{C} \setminus \{0\}$, with $f'(z) = -1/z^2$.

**E0.5.1.6.** If $f$ is holomorphic and purely real-valued on a domain, show $f$ is constant.

#### ★★ (Intermediate)

**E0.5.1.7 (C-R in polar coordinates).** If $f(re^{i\theta}) = u(r, \theta) + iv(r, \theta)$ is holomorphic, show the polar-form C-R equations: $u_r = v_\theta/r$ and $v_r = -u_\theta/r$.

**E0.5.1.8 (Laplacian in polar).** Show $\Delta u = u_{rr} + u_r/r + u_{\theta\theta}/r^2$.

**E0.5.1.9 (Harmonic polynomials).** Classify all *real* polynomials of degree $\leq 3$ in $(x, y)$ that are harmonic. What complex polynomials do they come from?

**E0.5.1.10 (Logarithm identities).** Find all $z$ for which $\log(z^2) = 2 \log z$ under the principal branch. Show the identity fails in general and explain why (multi-valuedness).

**E0.5.1.11 (Power series disks).** Show $\sum_{n=0}^\infty a_n z^n$ has a disk of convergence $|z| < R$ where $R = 1/\limsup |a_n|^{1/n}$ (Cauchy-Hadamard), and the series is holomorphic inside the disk.

**E0.5.1.12 (Möbius transformations).** A **Möbius transformation** is $T(z) = (az + b)/(cz + d)$ with $ad - bc \neq 0$. Prove $T$ is holomorphic on $\mathbb{C} \setminus \{-d/c\}$. Show compositions of Möbius transformations are Möbius, and that they form a group.

#### ★★★ (Challenge)

**E0.5.1.13 (Looman-Menchoff).** Read the statement of Looman-Menchoff: continuity + C-R everywhere implies holomorphy. (Remarkable because it drops the usual $C^1$ hypothesis.)

**E0.5.1.14 (Dixon's theorem / differentiability criterion).** Prove: if $f$ is continuous on $\Omega$ and has a continuous derivative $f'$ on $\Omega \setminus \{p\}$, and $\lim_{z \to p} (z - p) f'(z) = 0$, then $f$ is holomorphic on all of $\Omega$ including $p$. (Removable singularities preview.)

**E0.5.1.15 (Conformal equivalence).** Prove: a holomorphic $f: \Omega \to \Omega'$ with $f'(z) \neq 0$ everywhere preserves angles of intersecting curves. Conversely, a $C^1$-map $\mathbb{R}^2 \to \mathbb{R}^2$ that is angle-preserving (and orientation-preserving) is holomorphic.

**E0.5.1.16 (Harmonic is harmonic ⇔ real part of holomorphic).** On a simply connected domain, prove: a $C^2$-function $u$ is harmonic iff it is the real part of a holomorphic function.

---

## Topic 0.5.2 — Contour Integration and Cauchy's Theorem

### 0.5.2.1 Why Contour Integrals?

The real line admits one way to integrate from $a$ to $b$: along the segment $[a, b]$. The complex plane has two dimensions, so integrating from $z_0$ to $z_1$ requires choosing a **path**. Remarkably, for holomorphic integrands on a simply connected domain, *the path doesn't matter* — this is the content of Cauchy's theorem, and it is the gateway to residue calculus, the Cauchy integral formula, the argument principle, and essentially every deep theorem of complex analysis.

Before we can state Cauchy's theorem we must define what we mean by integrating a complex function along a curve.

### 0.5.2.2 Curves and Contours

**Definition 0.5.2.1 (Parametrized curve).** A **parametrized curve** in $\mathbb{C}$ is a continuous map $\gamma: [a, b] \to \mathbb{C}$. Write $\gamma(t) = x(t) + i y(t)$.

- $\gamma$ is **$C^1$** if $x, y$ are continuously differentiable; then $\gamma'(t) = x'(t) + i y'(t)$.
- $\gamma$ is **piecewise $C^1$** if $[a, b]$ admits a partition $a = t_0 < t_1 < \cdots < t_n = b$ with $\gamma|_{[t_{k-1}, t_k]}$ being $C^1$ for each $k$. This is the standard regularity for contour integration.
- $\gamma$ is **closed** if $\gamma(a) = \gamma(b)$.
- $\gamma$ is **simple** if $\gamma$ is injective on $[a, b)$ (so closed simple = Jordan curve, no self-intersections except possibly at endpoints).
- The **trace** $\gamma^* = \gamma([a, b]) \subset \mathbb{C}$ is the set-theoretic image.
- The **reverse** is $(-\gamma)(t) := \gamma(a + b - t)$, tracing $\gamma$ backwards.
- The **concatenation** $\gamma_1 + \gamma_2$ (when $\gamma_1(b_1) = \gamma_2(a_2)$) reparametrizes $\gamma_1$ on $[0, 1/2]$ and $\gamma_2$ on $[1/2, 1]$.

**Length.** If $\gamma$ is piecewise $C^1$, its **length** is
$$
L(\gamma) := \int_a^b |\gamma'(t)| \, dt.
$$
This is invariant under orientation-preserving reparametrization (change of variable in the integral).

**Reparametrization.** Two curves $\gamma_1: [a_1, b_1] \to \mathbb{C}$ and $\gamma_2: [a_2, b_2] \to \mathbb{C}$ are **equivalent** if there is an increasing $C^1$-bijection $\phi: [a_1, b_1] \to [a_2, b_2]$ with $\gamma_1 = \gamma_2 \circ \phi$. Complex integrals are reparametrization-invariant, so we often speak of "the curve" without specifying parametrization.

### 0.5.2.3 The Complex Line Integral

**Definition 0.5.2.2 (Contour integral).** Let $f: \Omega \to \mathbb{C}$ be continuous and $\gamma: [a, b] \to \Omega$ be piecewise $C^1$. The **contour integral** of $f$ along $\gamma$ is
$$
\boxed{\int_\gamma f(z) \, dz := \int_a^b f(\gamma(t)) \, \gamma'(t) \, dt.}
$$
The right side is a complex-valued Riemann integral; splitting into real and imaginary parts:
$$
\int_\gamma f \, dz = \int_a^b \operatorname{Re}[f(\gamma(t)) \gamma'(t)] \, dt + i \int_a^b \operatorname{Im}[f(\gamma(t)) \gamma'(t)] \, dt.
$$

**Alternative form.** Writing $f = u + iv$ and $dz = dx + i \, dy$ (purely formally),
$$
\int_\gamma f \, dz = \int_\gamma (u \, dx - v \, dy) + i \int_\gamma (v \, dx + u \, dy).
$$
This connects complex integration to real line integrals from vector calculus. The C-R equations will later translate into the 2D curl-free condition making the 1-forms $u \, dx - v \, dy$ and $v \, dx + u \, dy$ closed.

**Proposition 0.5.2.3 (Basic properties).** Assume $f, g$ continuous on $\Omega$ and $\gamma, \gamma_1, \gamma_2$ piecewise $C^1$ in $\Omega$.

1. **Linearity.** $\int_\gamma (\alpha f + \beta g) \, dz = \alpha \int_\gamma f \, dz + \beta \int_\gamma g \, dz$ for $\alpha, \beta \in \mathbb{C}$.

2. **Orientation reversal.** $\int_{-\gamma} f \, dz = -\int_\gamma f \, dz$.

3. **Concatenation.** $\int_{\gamma_1 + \gamma_2} f \, dz = \int_{\gamma_1} f \, dz + \int_{\gamma_2} f \, dz$.

4. **Reparametrization invariance.** If $\gamma_1 \sim \gamma_2$ (same orientation), then $\int_{\gamma_1} f \, dz = \int_{\gamma_2} f \, dz$.

*Proof.* (1), (3), (4) follow from the corresponding properties of Riemann integrals; for (4) use $\phi$'s chain rule: $(\gamma_2 \circ \phi)'(s) = \gamma_2'(\phi(s)) \phi'(s)$, and change of variable $t = \phi(s)$. For (2), the map $s := a + b - t$ reverses the parametrization and introduces a factor $-1$ from $\gamma'$, combined with the sign flip in $ds = -dt$. $\blacksquare$

### 0.5.2.4 The ML Inequality

The most frequently invoked estimate in contour integration.

**Theorem 0.5.2.4 (ML inequality).** If $f$ is continuous on the trace of $\gamma$ (piecewise $C^1$), $|f(z)| \leq M$ for all $z \in \gamma^*$, and $L = L(\gamma)$, then
$$
\left| \int_\gamma f \, dz \right| \leq M \cdot L.
$$

*Proof.* Write $I := \int_\gamma f \, dz \in \mathbb{C}$. If $I = 0$, done. Otherwise write $I = |I| e^{i\theta}$ so $|I| = e^{-i\theta} I$. Then
$$
|I| = \operatorname{Re}(e^{-i\theta} I) = \operatorname{Re} \int_a^b e^{-i\theta} f(\gamma(t)) \gamma'(t) \, dt = \int_a^b \operatorname{Re}\bigl[ e^{-i\theta} f(\gamma(t)) \gamma'(t) \bigr] dt.
$$
The integrand satisfies $|\operatorname{Re}(e^{-i\theta} f(\gamma) \gamma')| \leq |f(\gamma(t))| |\gamma'(t)| \leq M |\gamma'(t)|$. Hence
$$
|I| \leq \int_a^b M |\gamma'(t)| \, dt = M \cdot L. \quad \blacksquare
$$

**Remark.** The "factor-out-phase" trick used here (multiply by $e^{-i\theta}$ to rotate $I$ onto the positive real axis) is the same trick used to prove the triangle inequality for complex integrals $|\int f| \leq \int |f|$ in general. Memorize it — it appears in nearly every inequality-based argument in complex analysis.

### 0.5.2.5 Fundamental Integrals

**Example 1: $\int_{|z|=r} z^n \, dz$ for $n \in \mathbb{Z}$.**

Parametrize $\gamma(t) = re^{it}$, $t \in [0, 2\pi]$; then $\gamma'(t) = ire^{it}$, and $\gamma(t)^n = r^n e^{int}$. So
$$
\int_\gamma z^n \, dz = \int_0^{2\pi} r^n e^{int} \cdot ire^{it} \, dt = ir^{n+1} \int_0^{2\pi} e^{i(n+1)t} \, dt.
$$

Case $n \neq -1$: the integral is $\left[ \frac{e^{i(n+1)t}}{i(n+1)} \right]_0^{2\pi} = 0$ since $e^{i(n+1)2\pi} = 1$. So $\int_\gamma z^n \, dz = 0$.

Case $n = -1$: the integral is $ir^0 \int_0^{2\pi} 1 \, dt = 2\pi i$.

**Conclusion.** The master identity:
$$
\boxed{\int_{|z| = r} z^n \, dz = \begin{cases} 0, & n \neq -1, \\ 2\pi i, & n = -1. \end{cases}}
$$

This tiny computation is the single most important integral in complex analysis. The residue theorem is essentially this fact plus linearity. The Cauchy integral formula follows by a change of center.

**Example 2: $\int_{[0, 1+i]} \bar z \, dz$ along two paths.**

Path $\gamma_1$: segment from $0$ to $1+i$. Parametrize $\gamma_1(t) = t(1+i)$ for $t \in [0, 1]$. Then $\overline{\gamma_1(t)} = t(1-i)$ and $\gamma_1'(t) = 1+i$, so
$$
\int_{\gamma_1} \bar z \, dz = \int_0^1 t(1-i)(1+i) \, dt = \int_0^1 t \cdot 2 \, dt = 1.
$$

Path $\gamma_2$: segment $0 \to 1$ then $1 \to 1+i$. On $[0, 1] \ni s$: $\gamma(s) = s$, $\bar z = s$, $dz = ds$; integral $= \int_0^1 s \, ds = 1/2$. On second leg: $\gamma(s) = 1 + is$, $\bar z = 1 - is$, $dz = i \, ds$; integral $= \int_0^1 (1 - is) i \, ds = i + 1/2$. Total: $1 + i$.

So the two paths give different values: $1$ vs $1 + i$. **Conclusion:** $\bar z$ is not path-independent — consistent with the fact that $\bar z$ is *not holomorphic*.

### 0.5.2.6 Primitives and Path Independence

**Definition 0.5.2.5 (Primitive).** A **primitive** of a continuous $f: \Omega \to \mathbb{C}$ is a holomorphic function $F: \Omega \to \mathbb{C}$ with $F' = f$.

**Theorem 0.5.2.6 (Fundamental theorem of calculus for line integrals).** If $F$ is a primitive of $f$ on $\Omega$, and $\gamma: [a, b] \to \Omega$ is piecewise $C^1$, then
$$
\int_\gamma f \, dz = F(\gamma(b)) - F(\gamma(a)).
$$
In particular, if $\gamma$ is closed, $\int_\gamma f \, dz = 0$.

*Proof.* For $C^1$-parts, $(F \circ \gamma)'(t) = F'(\gamma(t)) \gamma'(t) = f(\gamma(t)) \gamma'(t)$, using the complex chain rule. Thus
$$
\int_{t_{k-1}}^{t_k} f(\gamma(t)) \gamma'(t) \, dt = \int_{t_{k-1}}^{t_k} (F \circ \gamma)'(t) \, dt = F(\gamma(t_k)) - F(\gamma(t_{k-1}))
$$
by the fundamental theorem of calculus applied separately to real and imaginary parts. Summing over pieces telescopes: $\sum_k [F(\gamma(t_k)) - F(\gamma(t_{k-1}))] = F(\gamma(b)) - F(\gamma(a))$. $\blacksquare$

**Corollary 0.5.2.7 (Obstruction to primitives).** $1/z$ has no primitive on any domain containing a closed curve winding around $0$.

*Proof.* If $F$ were such a primitive on $\Omega$, then for $\gamma = \{|z| = 1\} \subset \Omega$ we'd have $\int_\gamma z^{-1} \, dz = F(\gamma(2\pi)) - F(\gamma(0)) = 0$. But we computed this integral to equal $2\pi i \neq 0$. Contradiction. $\blacksquare$

This is the deep reason we need **branches** for $\log$ — the natural candidate for a primitive of $1/z$ can only be defined on a simply connected subdomain of $\mathbb{C} \setminus \{0\}$, not on the whole punctured plane.

**Theorem 0.5.2.8 (Existence of primitives).** Let $\Omega \subseteq \mathbb{C}$ be a connected open set, $f: \Omega \to \mathbb{C}$ continuous. The following are equivalent:

(a) $f$ has a primitive on $\Omega$.

(b) $\int_\gamma f \, dz = 0$ for every closed piecewise-$C^1$ curve $\gamma$ in $\Omega$.

(c) $\int_\gamma f \, dz$ depends only on the endpoints of $\gamma$ (path independence).

*Proof.*
**(a) $\Rightarrow$ (b):** Immediate from Theorem 0.5.2.6.

**(b) $\Leftrightarrow$ (c):** If (b) holds and $\gamma_1, \gamma_2$ share endpoints, then $\gamma_1 - \gamma_2$ (concatenation of $\gamma_1$ with the reverse of $\gamma_2$) is closed, so $0 = \int_{\gamma_1 - \gamma_2} f \, dz = \int_{\gamma_1} f \, dz - \int_{\gamma_2} f \, dz$. Conversely, if (c) holds and $\gamma$ is closed, then $\gamma$ and the constant curve at $\gamma(a)$ share endpoints, so $\int_\gamma f \, dz = \int_{\text{const}} f \, dz = 0$.

**(c) $\Rightarrow$ (a):** Fix $z_0 \in \Omega$ (using connectedness to ensure every point is reachable). Define
$$
F(z) := \int_{z_0}^z f(w) \, dw,
$$
where the integral is along *any* piecewise-$C^1$ path from $z_0$ to $z$ in $\Omega$; by (c), this is well-defined.

*Claim: $F$ is holomorphic with $F' = f$.* Fix $z \in \Omega$ and pick a disk $D(z, r) \subset \Omega$. For $h$ small enough that $z + h \in D(z, r)$, route the path from $z_0$ to $z+h$ as (path from $z_0$ to $z$) concatenated with (segment $\sigma$ from $z$ to $z+h$ parametrized as $s \mapsto z + sh$, $s \in [0, 1]$). Then
$$
F(z + h) - F(z) = \int_\sigma f \, dw = \int_0^1 f(z + sh) \cdot h \, ds = h \int_0^1 f(z + sh) \, ds.
$$
Dividing by $h$:
$$
\frac{F(z+h) - F(z)}{h} = \int_0^1 f(z + sh) \, ds.
$$
As $h \to 0$, by continuity of $f$ the integrand converges uniformly in $s \in [0,1]$ to $f(z)$, so the integral converges to $f(z)$. Hence $F'(z) = f(z)$. $\blacksquare$

The key geometric input in **(c) ⇒ (a)** was that a small *segment* from $z$ to $z+h$ lies in $\Omega$ — hence we didn't need any global hypothesis on $\Omega$ other than openness and connectedness.

### 0.5.2.7 Cauchy-Goursat: the Triangle Case

We now state the first major theorem of complex analysis. Despite appearances, it will be proven using only the definition of complex differentiability — no $C^1$-hypothesis needed, no Green's theorem. This is the content of **Goursat's improvement** over Cauchy's original proof.

**Theorem 0.5.2.9 (Goursat's theorem).** If $f$ is holomorphic on an open set $\Omega \subseteq \mathbb{C}$ and $T \subset \Omega$ is a closed triangle (including interior) contained in $\Omega$, then
$$
\int_{\partial T} f(z) \, dz = 0,
$$
where $\partial T$ is the triangular boundary, oriented counter-clockwise.

*Proof.* Let $I := \int_{\partial T} f \, dz$. We will show $|I| = 0$ via a subdivision-and-shrinking argument.

**Step 1: Subdivision.** Connect the midpoints of the three sides of $T$ to form four smaller congruent triangles $T^{(1)}_1, T^{(1)}_2, T^{(1)}_3, T^{(1)}_4$, each similar to $T$ with side lengths half those of $T$. Orient each subtriangle boundary counter-clockwise. Interior edges are traversed once in each direction across the two adjacent subtriangles, hence cancel; thus
$$
\int_{\partial T} f \, dz = \sum_{k=1}^4 \int_{\partial T^{(1)}_k} f \, dz.
$$

**Step 2: Select the dominant subtriangle.** By the triangle inequality, at least one of the four sub-integrals satisfies
$$
\left| \int_{\partial T^{(1)}_k} f \, dz \right| \geq \frac{|I|}{4}.
$$
Call this subtriangle $T_1$.

**Step 3: Iterate.** Apply the same subdivision to $T_1$, obtaining $T_2 \subset T_1$ with
$$
\left| \int_{\partial T_2} f \, dz \right| \geq \frac{1}{4} \left| \int_{\partial T_1} f \, dz \right| \geq \frac{|I|}{16}.
$$
Continue to produce a nested sequence of closed triangles
$$
T \supset T_1 \supset T_2 \supset \cdots
$$
with
$$
\left| \int_{\partial T_n} f \, dz \right| \geq \frac{|I|}{4^n}, \qquad \operatorname{diam}(T_n) = \frac{\operatorname{diam}(T)}{2^n}, \qquad \operatorname{perim}(T_n) = \frac{\operatorname{perim}(T)}{2^n}.
$$

**Step 4: Extract the limit point.** The triangles are a decreasing nested sequence of nonempty compact sets with diameters tending to $0$, so by the nested-compact-set property (Cantor intersection) $\bigcap_n T_n = \{z_0\}$ for some $z_0 \in T \subset \Omega$.

**Step 5: Use holomorphy at $z_0$.** Since $f$ is holomorphic at $z_0$, write (as in the definition of the complex derivative)
$$
f(z) = f(z_0) + f'(z_0)(z - z_0) + \eta(z) (z - z_0),
$$
where $\eta(z) \to 0$ as $z \to z_0$. (This just re-expresses the difference-quotient definition: $\eta(z) := [f(z) - f(z_0)]/(z - z_0) - f'(z_0)$ for $z \neq z_0$, and $\eta(z_0) := 0$.)

**Step 6: Integrate over $\partial T_n$.** The linear-and-constant part $f(z_0) + f'(z_0)(z - z_0)$ has an obvious primitive $G(z) = f(z_0) z + f'(z_0) (z - z_0)^2 / 2$, so by Theorem 0.5.2.6 its integral over the closed curve $\partial T_n$ is $0$. Hence
$$
\int_{\partial T_n} f \, dz = \int_{\partial T_n} \eta(z) (z - z_0) \, dz.
$$

**Step 7: Apply ML.** For $z \in T_n$, $|z - z_0| \leq \operatorname{diam}(T_n)$, and $|\eta(z)| \leq \sup_{w \in T_n} |\eta(w)| =: \epsilon_n$ with $\epsilon_n \to 0$. Thus on $\partial T_n$,
$$
|\eta(z)(z - z_0)| \leq \epsilon_n \cdot \operatorname{diam}(T_n).
$$
By ML,
$$
\left| \int_{\partial T_n} \eta(z)(z - z_0) \, dz \right| \leq \epsilon_n \cdot \operatorname{diam}(T_n) \cdot \operatorname{perim}(T_n) = \frac{\epsilon_n \cdot \operatorname{diam}(T) \cdot \operatorname{perim}(T)}{4^n}.
$$

**Step 8: Conclude.** Combining Steps 3 and 7,
$$
\frac{|I|}{4^n} \leq \left| \int_{\partial T_n} f \, dz \right| \leq \frac{\epsilon_n \cdot \operatorname{diam}(T) \cdot \operatorname{perim}(T)}{4^n},
$$
so $|I| \leq \epsilon_n \cdot \operatorname{diam}(T) \cdot \operatorname{perim}(T)$. Since $\epsilon_n \to 0$, $|I| = 0$. $\blacksquare$

**Commentary.** The triangle argument isolates the difficulty: holomorphy is used only *at the single point* $z_0$ — the limit of the nested triangles. Everywhere else we only needed continuity. This is why Goursat's proof works without assuming $f'$ is continuous (a standard assumption in the classical Cauchy proof via Green's theorem). In fact, we will later prove (via Cauchy integral formula) that holomorphic functions are automatically $C^\infty$, so the $C^1$-vs-merely-complex-differentiable distinction vanishes — but *logically* we need Goursat first.

### 0.5.2.8 Goursat for Rectangles

The triangle version extends to rectangles: split a rectangle along a diagonal into two triangles; the diagonal is traversed once in each direction and cancels. So:

**Corollary 0.5.2.10.** If $f$ is holomorphic on $\Omega$ and $R \subset \Omega$ is a closed rectangle with sides parallel to the axes (in fact, any shape), then $\int_{\partial R} f \, dz = 0$.

More generally, any closed polygon with interior in $\Omega$ works, via triangulation.

### 0.5.2.9 Cauchy's Theorem for Star-Shaped Domains

**Definition 0.5.2.11 (Star-shaped domain).** An open $\Omega \subseteq \mathbb{C}$ is **star-shaped with respect to $z_0 \in \Omega$** if for every $z \in \Omega$, the segment $\{(1-t) z_0 + t z : t \in [0, 1]\}$ lies in $\Omega$.

Examples: any convex set (star-shaped w.r.t. *every* interior point); $\mathbb{C} \setminus \{\text{ray from } 0\}$ (star-shaped w.r.t. any point not on the extended ray).

**Theorem 0.5.2.12 (Cauchy's theorem, star-shaped version).** Let $\Omega$ be star-shaped w.r.t. $z_0$, and $f: \Omega \to \mathbb{C}$ holomorphic. Then $f$ has a primitive on $\Omega$, and $\int_\gamma f \, dz = 0$ for every closed piecewise-$C^1$ curve $\gamma$ in $\Omega$.

*Proof.* Define
$$
F(z) := \int_{[z_0, z]} f(w) \, dw,
$$
the integral along the straight segment from $z_0$ to $z$ (well-defined since the segment is in $\Omega$).

*Claim: $F'(z) = f(z)$.* Fix $z \in \Omega$ and pick a disk $D(z, r) \subset \Omega$. For $h$ small enough that $z + h \in D(z, r)$, consider the triangle with vertices $z_0, z, z+h$. By star-shapedness, all three sides lie in $\Omega$ (the segment $[z, z+h]$ lies in the disk which is in $\Omega$; segments from $z_0$ are in $\Omega$ by definition of star-shaped). By Goursat, $\int_{\partial T} f = 0$, so
$$
F(z+h) - F(z) = \int_{[z_0, z+h]} f \, dw - \int_{[z_0, z]} f \, dw = \int_{[z, z+h]} f \, dw,
$$
where the sign accounting comes from $\int_{[z_0, z+h]} + \int_{[z+h, z]} + \int_{[z, z_0]} = 0$, i.e., $\int_{[z_0, z+h]} = \int_{[z_0, z]} + \int_{[z, z+h]}$. Now parametrize $[z, z+h]$ as $s \mapsto z + sh$, $s \in [0, 1]$:
$$
\frac{F(z+h) - F(z)}{h} = \int_0^1 f(z + sh) \, ds \xrightarrow{h \to 0} f(z).
$$
So $F' = f$. The conclusion $\int_\gamma f = 0$ for closed $\gamma$ now follows from Theorem 0.5.2.6. $\blacksquare$

### 0.5.2.10 Simply Connected Domains and the General Theorem

The star-shaped version suffices for most computations because we can always split a closed contour into pieces each lying in a star-shaped (in fact, convex) subregion. But the cleanest general statement uses:

**Definition 0.5.2.13 (Simply connected).** A connected open set $\Omega \subseteq \mathbb{C}$ is **simply connected** if every continuous closed curve $\gamma$ in $\Omega$ is **contractible** (homotopic to a constant curve in $\Omega$). Equivalently: $\mathbb{C} \setminus \Omega$ has no bounded connected component ("$\Omega$ has no holes").

Examples: $\mathbb{C}$ itself, any open disk, any open half-plane, slit plane $\mathbb{C} \setminus (-\infty, 0]$, any star-shaped region, any convex region. **Not** simply connected: annuli, punctured plane $\mathbb{C} \setminus \{0\}$.

**Theorem 0.5.2.14 (Cauchy's theorem, simply connected version).** If $\Omega$ is a simply connected open subset of $\mathbb{C}$ and $f: \Omega \to \mathbb{C}$ is holomorphic, then $f$ has a primitive on $\Omega$ and $\int_\gamma f \, dz = 0$ for every closed piecewise-$C^1$ curve $\gamma$ in $\Omega$.

*Proof sketch (homotopy version).* Let $\gamma_0, \gamma_1: [0,1] \to \Omega$ be two curves with the same endpoints (or both closed), homotopic via $H: [0,1]^2 \to \Omega$ continuous with $H(0, t) = \gamma_0(t)$, $H(1, t) = \gamma_1(t)$.

The image $H([0,1]^2)$ is compact in $\Omega$; by a Lebesgue number / uniform continuity argument, there is a finite subdivision of $[0,1]^2$ into small squares each of whose image lies in a disk contained in $\Omega$ (disks are star-shaped!). Apply Goursat to each small square and telescope the contributions across the grid; all interior edges cancel, leaving $\int_{\gamma_0} f \, dz = \int_{\gamma_1} f \, dz$.

In particular, if $\gamma$ is closed in simply connected $\Omega$, $\gamma$ is homotopic to a constant curve, over which the integral is $0$.

For the primitive, fix $z_0 \in \Omega$ and define $F(z) = \int_{z_0}^z f \, dw$ along any path; by path-independence (just proven), this is well-defined. The differentiation argument of Theorem 0.5.2.8 applies verbatim. $\blacksquare$

### 0.5.2.11 Deformation and the Winding Number

**Theorem 0.5.2.15 (Cauchy's deformation principle).** Let $f$ be holomorphic on an open set $\Omega$, and let $\gamma_0, \gamma_1$ be closed curves in $\Omega$ that are **homotopic in $\Omega$** (i.e., there is a continuous homotopy $H: [0,1]^2 \to \Omega$ between them). Then
$$
\int_{\gamma_0} f \, dz = \int_{\gamma_1} f \, dz.
$$

This is the continuous deformation of contours — moving a closed contour without crossing singularities doesn't change the integral. It is the workhorse identity for actually evaluating contour integrals (close a line integral to a large semicircle, etc.).

**Winding number.** For a closed curve $\gamma$ not passing through $z_0$, define
$$
n(\gamma, z_0) := \frac{1}{2\pi i} \int_\gamma \frac{dz}{z - z_0} \in \mathbb{Z}.
$$
It counts (with sign) how many times $\gamma$ wraps around $z_0$. That it's an integer follows from the existence of a local logarithm: parametrize $\gamma(t)$ and compute $\log(\gamma(t) - z_0)$ using analytic continuation along $\gamma$; the imaginary part measures angle change, and closedness forces this to be an integer multiple of $2\pi$.

### 0.5.2.12 Worked Examples

**Example 1: $\int_{|z|=1} e^z / z \, dz$ via a star-shaped fact.**

We cannot directly apply Cauchy in a region containing $0$. But note $e^z / z$ is holomorphic on $\mathbb{C} \setminus \{0\}$, not simply connected. Cauchy's theorem doesn't apply. We'll compute this later via the Cauchy integral formula; for now, observe that it is *not* zero (it'll turn out to equal $2\pi i \cdot e^0 = 2\pi i$).

**Example 2: $\int_\gamma \sin z \, dz$ for any closed $\gamma$ in $\mathbb{C}$.**

$\sin z$ is entire. $\mathbb{C}$ is simply connected. By Cauchy, integral $= 0$. Alternatively: $\sin z = (-\cos z)'$, so $-\cos z$ is a global primitive; closed-curve integrals of exact forms are $0$.

**Example 3: $\int_\gamma dz / z$ for $\gamma$ a curve in the slit plane $\mathbb{C} \setminus (-\infty, 0]$.**

The slit plane is simply connected (it's star-shaped w.r.t. $z_0 = 1$: segment from $1$ to any $z$ with $\operatorname{Re} z > 0$ stays in the right half-plane; more carefully, convex combinations of $1$ and $z \in \Omega$ stay in $\Omega$ because the ray $(-\infty, 0]$ is on the left). So $1/z$ has a primitive on the slit plane, namely $\log z$ (principal branch). For $\gamma$ going from $z_1$ to $z_2$: $\int_\gamma dz/z = \log z_2 - \log z_1$.

**Example 4: Gaussian Fresnel $\int_0^\infty \cos(x^2) \, dx$.**

A classical application of Cauchy's theorem. Integrate $e^{-z^2}$ over a pie slice: segment on positive real axis from $0$ to $R$, arc from $Re^{i0}$ to $Re^{i\pi/4}$, and segment back from $Re^{i\pi/4}$ to $0$. Since $e^{-z^2}$ is entire, the total integral is $0$. The arc contribution vanishes as $R \to \infty$ (estimate via ML and $|e^{-z^2}|$ decay on $\{0 \leq \arg z \leq \pi/4\}$ — specifically, use that $\operatorname{Re}(z^2) = r^2 \cos 2\theta$ is positive on this sector). So the real-axis segment and the $\pi/4$-segment contributions are equal and opposite. Computing the real-axis integral gives $\sqrt\pi/2$; equating with the rotated integral gives the Fresnel formula $\int_0^\infty \cos(x^2) \, dx = \int_0^\infty \sin(x^2) \, dx = \sqrt{\pi/8}$.

This is a preview of the standard "move the contour, pick up a convenient integral" technique that drives characteristic-function computation, Laplace-to-Fourier conversion, and Carr-Madan-style option pricing.

**Example 5: Independence of path for $1/(z^2 + 1)$.**

On the upper half-plane $\mathbb{H}$ (simply connected), $1/(z^2 + 1) = 1/[(z-i)(z+i)]$ has singularities at $\pm i$; only $i$ lies in $\mathbb{H}$, so $1/(z^2+1)$ is not holomorphic on all of $\mathbb{H}$. But on $\mathbb{H} \setminus \{i\}$ — not simply connected — closed contours winding around $i$ do *not* have zero integral. (We'll evaluate them via residues.)

### 0.5.2.13 Python: Numerical Contour Integration

```python
import numpy as np

def contour_integral(f, gamma, gamma_prime, t_range=(0, 1), n=10000):
    """
    Numerical contour integral of f along parametrized curve gamma.
    Returns ∫_gamma f(z) dz.

    f           : callable, complex function z -> f(z)
    gamma       : callable, t -> z (parametrization)
    gamma_prime : callable, t -> z' (derivative)
    t_range     : parameter interval (a, b)
    n           : number of sample points (midpoint rule)
    """
    a, b = t_range
    ts = np.linspace(a, b, n, endpoint=False) + 0.5 * (b - a) / n  # midpoints
    dt = (b - a) / n
    integrand = np.array([f(gamma(t)) * gamma_prime(t) for t in ts])
    return np.sum(integrand) * dt

# Test 1: ∫_{|z|=1} z^n dz should equal 0 for n≠-1, 2πi for n=-1.
gamma = lambda t: np.exp(2j * np.pi * t)
gamma_p = lambda t: 2j * np.pi * np.exp(2j * np.pi * t)

for n in [-2, -1, 0, 1, 2, 3]:
    val = contour_integral(lambda z, n=n: z**n, gamma, gamma_p)
    expected = 2j * np.pi if n == -1 else 0
    print(f"n = {n}:  computed = {val:.6f},  expected = {expected}")

# Test 2: ∫_{|z|=1} e^z/z dz should equal 2πi (Cauchy integral formula preview).
val = contour_integral(lambda z: np.exp(z)/z, gamma, gamma_p)
print(f"\ne^z/z:  {val:.6f},  expected = 2πi ≈ {2j*np.pi:.6f}")

# Test 3: ∫_gamma sin(z) dz over any closed curve should be 0.
# Take an ellipse: gamma(t) = 2*cos(2πt) + 3j*sin(2πt)
g_ell = lambda t: 2*np.cos(2*np.pi*t) + 3j*np.sin(2*np.pi*t)
gp_ell = lambda t: -4*np.pi*np.sin(2*np.pi*t) + 6j*np.pi*np.cos(2*np.pi*t)
val = contour_integral(np.sin, g_ell, gp_ell)
print(f"\n∮ sin(z) dz on ellipse: {val:.2e}  (should be 0)")

# Test 4: Goursat — integrate around a triangle for an entire function.
def triangle_integral(f, v1, v2, v3, n=5000):
    """∫ over triangular contour v1 -> v2 -> v3 -> v1."""
    total = 0
    for start, end in [(v1, v2), (v2, v3), (v3, v1)]:
        g = lambda t, s=start, e=end: s + t*(e - s)
        gp = lambda t, s=start, e=end: e - s
        total += contour_integral(f, g, gp, n=n)
    return total

val = triangle_integral(lambda z: z**3 + 2*z + 1, 0, 1+0j, 0+1j)
print(f"\nTriangle integral of polynomial (entire): {val:.2e}  (should be 0)")

val = triangle_integral(lambda z: np.conj(z), 0, 1+0j, 0+1j)
print(f"Triangle integral of conj(z) (NOT holomorphic): {val:.4f}  (nonzero!)")
```

Expected output:
```
n = -2:  computed ≈ 0,       expected = 0
n = -1:  computed ≈ 6.283i,  expected = 2πi
n =  0:  computed ≈ 0,       expected = 0
n =  1:  computed ≈ 0,       expected = 0
...
e^z/z:  ≈ 6.283i,  expected = 2πi ≈ 6.283i
∮ sin(z) dz on ellipse: ≈ 1e-14  (numerical zero)
Triangle integral of polynomial (entire): ≈ 1e-15  (numerical zero)
Triangle integral of conj(z): ≈ -i  (nonzero — confirms conj z not holomorphic)
```

The last test is a numerical Goursat check: polynomials are entire so their triangle integral is *exactly* zero (up to roundoff); but $\bar z$ is not holomorphic and gives a nonzero triangle integral, providing a concrete numerical witness that C-R fails for $\bar z$.

### 0.5.2.14 [QUANT APPLICATION]

**1. Carr-Madan option pricing via Fourier transform.** The Carr-Madan formula represents a European call price as an integral involving the characteristic function of log-price:
$$
C(K) = \frac{e^{-\alpha \log K}}{\pi} \int_0^\infty \operatorname{Re}\left[ e^{-i v \log K} \psi(v; \alpha) \right] dv,
$$
where $\psi$ involves $\phi(v - (\alpha+1) i)$, the characteristic function evaluated at a *complex* argument. The justification for evaluating at complex arguments — and for shifting the integration contour to ensure the integrand decays — is exactly Cauchy's theorem: we deform the real-axis contour into the upper or lower half plane depending on convexity, picking up contributions only where the integrand is singular. This is the basis of numerical option pricing for Heston, variance gamma, CGMY, NIG, and virtually every Lévy-process-driven model.

**2. Laplace transforms of affine processes.** For an affine model with log-price $X_t$, the MGF $\mathbb{E}[e^{u X_T}]$ is known in closed form as $e^{A(T, u) + B(T, u) X_0}$ for complex $u$, with $A, B$ solving Riccati ODEs. Whether the transform remains finite for complex $u$ depends on the analyticity strip — a complex-analytic consideration involving the radius of convergence of the characteristic function. Contour deformation within this strip enables numerical pricing.

**3. Closing contours in option pricing (Black-Scholes).** The Black-Scholes call formula derivation via Fourier transform requires evaluating
$$
\int_{-\infty}^\infty \frac{e^{ivx}}{(v - i a)(v - i b)} dv.
$$
Close with a semicircle in the upper or lower half plane depending on the sign of $x$, picking up residues at $ia$ or $ib$. The arc contribution vanishes by Jordan's lemma (ML-type estimate). Cauchy's theorem is the licensing principle: "deform the real axis into a simply connected region minus a few poles."

**4. Kramers-Kronig relations.** In finance-adjacent signal processing and in any physical system with causal impulse response $h(t)$ supported on $t \geq 0$, the Fourier transform $\hat h(\omega)$ extends holomorphically to the upper half plane. The absence of poles in the upper half plane (Cauchy's theorem applied to $\hat h / (\omega - \omega_0)$ closed in the upper half plane) yields the Kramers-Kronig dispersion relations: $\operatorname{Re} \hat h$ and $\operatorname{Im} \hat h$ are Hilbert transforms of each other. In finance: any causal filter / impulse response on price processes has its real and imaginary parts of the transfer function linked by this relation.

**5. Random variable simulation via inverse characteristic function.** Given $\phi(v) = \mathbb{E}[e^{ivX}]$ and desiring CDF / density of $X$, Gil-Pelaez inversion expresses the CDF as
$$
F(x) = \frac{1}{2} - \frac{1}{\pi} \int_0^\infty \operatorname{Im}\left[ \frac{e^{-ivx} \phi(v)}{v} \right] dv.
$$
The $1/v$ singularity at $v = 0$ is handled by contour deformation around a small semicircle, picking up half a residue (principal value). The rigorous justification is Cauchy's theorem on the strip of analyticity of $\phi$.

### 0.5.2.15 Exercises

#### ★ (Warm-up)

**E0.5.2.1.** Compute $\int_\gamma z \, dz$ where $\gamma$ is the segment from $0$ to $1 + i$. Verify your answer using the primitive $z^2/2$.

**E0.5.2.2.** Compute $\int_{|z|=2} \frac{dz}{z - 1}$. (Parametrize directly; should give $2\pi i$.)

**E0.5.2.3.** Compute $\int_{[0,1]} e^{z} \, dz$ (segment from $0$ to $1$). Verify using the primitive $e^z$.

**E0.5.2.4.** Show that $\int_\gamma (az + b) \, dz = a[\gamma(b) - \gamma(a)]^2/... $ – actually compute $\int_\gamma (az + b) \, dz$ from $z_1$ to $z_2$ and verify independence of path.

**E0.5.2.5.** Verify ML on a specific example: $|\int_{[0, 1+i]} z^2 \, dz| \leq |1+i|^2 \cdot \sqrt 2$? Compute exactly and compare.

**E0.5.2.6.** Let $\gamma$ be the unit square (counter-clockwise). Compute $\int_\gamma \bar z \, dz$ and confirm it is nonzero.

#### ★★ (Standard)

**E0.5.2.7.** Prove that if $f$ is continuous on $\bar D$ (closed unit disk) and holomorphic on $D$, then $\int_{\partial D} f \, dz = 0$. (This is the limiting case of Cauchy's theorem — requires a small boundary-approximation argument: consider $f(rz)$ for $r < 1$, let $r \to 1^-$.)

**E0.5.2.8.** Show that $\int_{|z|=R} z^n \, dz$ as a function of $R > 0$ is constant (for $n \in \mathbb{Z}$, $R \neq 0$). Why does this follow from the deformation principle?

**E0.5.2.9.** Let $\gamma_R$ be the semicircle in the upper half plane of radius $R$, parametrized from $-R$ to $R$. Show that for any polynomial $p(z)$ of degree $\leq n$,
$$
\left| \int_{\gamma_R} p(z) e^{iz} \, dz \right| = O(R^n).
$$
Then show that if $\deg p \leq n - 2$ (with the $e^{iz}$ decay), $\int_{\gamma_R} p(z) e^{iz} \, dz \to 0$ as $R \to \infty$ (a Jordan's-lemma-type conclusion).

**E0.5.2.10.** Compute $\int_{-\infty}^\infty \frac{dx}{1 + x^2} = \pi$ by closing the contour in the upper half plane. (Take a limit $R \to \infty$ in the semicircle + real axis; use residues or direct computation at $z = i$.)

**E0.5.2.11.** Let $f$ be entire and satisfy $|f(z)| \leq M$ for all $z$. Apply Cauchy's theorem to $f(z) / (z - z_0)^2$ over a large circle to prove Liouville's theorem: $f$ is constant. (We will redo this later using Cauchy's estimates.)

**E0.5.2.12.** Let $\Omega = \mathbb{C} \setminus \{0\}$ (an annular-type region). Give an example of a holomorphic $f$ on $\Omega$ with no primitive (hint: what's the obstruction?). Give an example *with* a primitive.

**E0.5.2.13.** Define the **winding number** of a closed curve $\gamma$ around $z_0 \notin \gamma^*$ as $n(\gamma, z_0) = (2\pi i)^{-1} \int_\gamma dz/(z - z_0)$. Compute winding numbers for: (a) $\gamma(t) = e^{2\pi i k t}$ over $[0, 1]$; (b) the boundary of a "figure eight" that passes around $z_0$ once clockwise then once counterclockwise; (c) any simple closed curve in $\mathbb{C} \setminus \{z_0\}$.

**E0.5.2.14 (Fresnel).** Complete the Fresnel integral calculation (Example 4): show rigorously, via ML with a sharper estimate ($e^{-R^2 \sin 2\theta}$ on the arc), that the arc contribution vanishes, then equate to get $\int_0^\infty \cos(x^2) \, dx = \int_0^\infty \sin(x^2) \, dx = \sqrt{\pi/8}$.

#### ★★★ (Challenge)

**E0.5.2.15 (Morera's theorem).** Prove the converse of Cauchy-Goursat: if $f: \Omega \to \mathbb{C}$ is continuous and $\int_{\partial T} f \, dz = 0$ for every closed triangle $T \subset \Omega$, then $f$ is holomorphic on $\Omega$. (Hint: construct a local primitive $F$ in each disk of $\Omega$ — use a fixed base point and the hypothesis to verify path-independence on triangles — and deduce $f = F'$ is holomorphic.)

**E0.5.2.16 (Cauchy's theorem via Green's theorem, classical route).** Assume $f = u + iv \in C^1(\Omega)$ is holomorphic on $\Omega$ and $\gamma = \partial R$ for a nice region $R \subset \Omega$. Write
$$
\int_\gamma f \, dz = \int_\gamma (u + iv)(dx + i \, dy) = \int_\gamma u \, dx - v \, dy + i \int_\gamma v \, dx + u \, dy.
$$
Apply Green's theorem and use C-R to conclude both real integrals vanish. (This was Cauchy's original proof, superseded by Goursat.)

**E0.5.2.17 (Homotopy invariance, rigorous version).** Let $H: [0,1]^2 \to \Omega$ be a continuous homotopy between closed curves $\gamma_0, \gamma_1$ in $\Omega$ (both piecewise $C^1$). Prove $\int_{\gamma_0} f \, dz = \int_{\gamma_1} f \, dz$ for any $f$ holomorphic on $\Omega$. (Subdivide $[0,1]^2$ into a fine grid using a Lebesgue-number argument on the cover of $H([0,1]^2)$ by disks contained in $\Omega$; apply Goursat to each small square.)

**E0.5.2.18 (Cauchy for domains with holes).** Let $\Omega = \{a < |z| < b\}$ be an annulus and $f$ holomorphic on $\Omega$. Show that $\int_{|z|=r_1} f \, dz = \int_{|z|=r_2} f \, dz$ for any $a < r_1 < r_2 < b$. (Apply the homotopy version or connect the two circles by two radial segments, traversed in opposite directions, to form a simply connected region.)

**E0.5.2.19 (Primitives on non-simply-connected domains).** Show that on the annulus $\Omega = \{1 < |z| < 2\}$, the function $1/z$ has no primitive, but $1/z^2$ does. What's the general principle? (Compute integrals over the circle $|z| = 1.5$; use this to define a "period map" $H^1(\Omega, \mathbb{C})$.)

**E0.5.2.20 (Cauchy's theorem for the Riemann sphere).** Treat $\mathbb{C} \cup \{\infty\}$ as a sphere. Show that for any rational function $f$ on the sphere, with singularities $z_1, \ldots, z_n$ (all finite) and residues $r_1, \ldots, r_n$,
$$
\sum_{k=1}^n r_k = -\operatorname{Res}_{\infty} f,
$$
where the residue at infinity is computed via the change of variable $w = 1/z$. Conclude that $\sum_{k} r_k = 0$ for a rational function with no pole at infinity plus the contribution at $\infty$. (Sum-of-residues theorem.)

---

## Topic 0.5.3 — Cauchy's Integral Formula and Consequences

### 0.5.3.1 The Formula

Cauchy's theorem says closed contour integrals of holomorphic functions vanish. The **Cauchy integral formula** is the sharper statement that they *recover the function values* when a small pole is inserted. It is the single most consequential theorem in complex analysis: from it flow analyticity, Cauchy's estimates, Liouville's theorem, maximum principle, and the fundamental theorem of algebra.

**Theorem 0.5.3.1 (Cauchy integral formula).** Let $f$ be holomorphic on an open set containing the closed disk $\bar D(z_0, r)$. Then for every $z \in D(z_0, r)$,
$$
\boxed{f(z) = \frac{1}{2\pi i} \oint_{|w - z_0| = r} \frac{f(w)}{w - z} \, dw,}
$$
where the circle is traversed counter-clockwise.

*Proof.* Fix $z$ in the open disk. Consider the function
$$
g(w) := \begin{cases} \frac{f(w) - f(z)}{w - z}, & w \neq z, \\ f'(z), & w = z. \end{cases}
$$
$g$ is continuous on the closed disk (continuity at $w = z$ is exactly the definition of complex differentiability) and holomorphic on the closed disk minus $\{z\}$.

**Claim:** $\int_{|w - z_0| = r} g(w) \, dw = 0$.

Since $g$ is continuous on the closed disk $\bar D = \bar D(z_0, r)$ (compact) and holomorphic on $\bar D \setminus \{z\}$, we can apply a slight extension of Cauchy-Goursat. Specifically, for each closed triangle $T \subset \bar D$ we have $\int_{\partial T} g \, dw = 0$:

- If $z \notin T$: $g$ is holomorphic in a neighborhood of $T$, so Goursat applies.
- If $z \in T$: split $T$ by joining $z$ to the three vertices (or the nearest vertices if $z$ is on an edge), producing smaller triangles each with $z$ as a vertex. On each such sub-triangle, let side lengths be $L$ and notice $g$ is continuous hence bounded by $M$ on $T$; by ML, $|\int_{\partial T'} g \, dw| \leq 3ML$. Shrinking the sides at $z$ to $0$ (keep the opposite vertex fixed) makes the contribution $\to 0$. This shows $\int_{\partial T} g \, dw = 0$ for any $T \subset \bar D$.

By Morera's theorem (Exercise 0.5.2.15), $g$ has a primitive in a star-shaped neighborhood of any point in $\bar D$; in particular, by the proof of Theorem 0.5.2.12, $g$ has a primitive on the disk and $\int_{|w-z_0|=r} g \, dw = 0$.

**Unpack the claim:**
$$
0 = \int_{|w-z_0|=r} \frac{f(w) - f(z)}{w - z} \, dw = \int_{|w-z_0|=r} \frac{f(w)}{w - z} \, dw - f(z) \int_{|w-z_0|=r} \frac{dw}{w - z}.
$$

Now we need $\int_{|w-z_0|=r} dw/(w-z) = 2\pi i$ for $z \in D(z_0, r)$. This is the winding-number fact: the circle has winding number $1$ around any interior point. Quick proof: for $z = z_0$ we computed this integral directly to equal $2\pi i$. For general interior $z$, apply deformation (Theorem 0.5.2.15) — the circle and any small circle around $z$ are homotopic in $\bar D(z_0, r) \setminus \{z\}$, and $\int_{|w-z|=\epsilon} dw/(w-z) = 2\pi i$ by direct parametrization.

Therefore:
$$
\int_{|w-z_0|=r} \frac{f(w)}{w - z} \, dw = 2\pi i f(z). \quad \blacksquare
$$

**Geometric interpretation.** The value of $f$ at an interior point is a *weighted average* of boundary values, with weights $\frac{1}{2\pi i (w - z)}$ (complex kernel). Taking $z = z_0$ and parametrizing $w = z_0 + re^{i\theta}$, this becomes the **mean value property**:
$$
f(z_0) = \frac{1}{2\pi} \int_0^{2\pi} f(z_0 + re^{i\theta}) \, d\theta.
$$
Average of boundary values equals center value — a profoundly restrictive property.

### 0.5.3.2 Cauchy's Formula for Derivatives

The Cauchy formula can be differentiated under the integral sign because $1/(w - z)$ is a nice function of $z$. We get:

**Theorem 0.5.3.2 (Cauchy integral formula for derivatives).** Under the hypotheses of Theorem 0.5.3.1, $f$ is *infinitely* differentiable on $D(z_0, r)$, and for every integer $n \geq 0$ and every $z \in D(z_0, r)$,
$$
\boxed{f^{(n)}(z) = \frac{n!}{2\pi i} \oint_{|w - z_0| = r} \frac{f(w)}{(w - z)^{n+1}} \, dw.}
$$

*Proof.* Induction on $n$. Base $n = 0$ is Theorem 0.5.3.1. Inductive step: assume the formula for $n$. We show $f^{(n)}$ is differentiable with derivative given by the $n+1$ version.

Fix $z \in D(z_0, r)$ and pick $\delta > 0$ with $\bar D(z, 2\delta) \subset D(z_0, r)$. For $|h| < \delta$,
$$
\frac{f^{(n)}(z+h) - f^{(n)}(z)}{h} = \frac{n!}{2\pi i} \int \frac{f(w)}{h}\left[ \frac{1}{(w - z - h)^{n+1}} - \frac{1}{(w - z)^{n+1}} \right] dw.
$$
The quantity in brackets, divided by $h$, converges pointwise to the derivative of $(w - z)^{-(n+1)}$ with respect to $z$, which is $(n+1)(w - z)^{-(n+2)}$. We need uniform convergence in $w$ on the circle to pass the limit through the integral.

For $w \in \{|w - z_0| = r\}$ we have $|w - z| \geq \delta$ (distance from $z$ to the circle) and $|w - z - h| \geq \delta/2$ (since $|h| < \delta$, the shifted point $z+h$ is still at distance $\geq \delta/2$ from the circle). So $1/(w - z)^{n+1}$ and $1/(w - z - h)^{n+1}$ are uniformly bounded on the circle, and the difference quotient
$$
\frac{1}{h}\left[ \frac{1}{(w - z - h)^{n+1}} - \frac{1}{(w - z)^{n+1}} \right]
$$
is uniformly Cauchy as $h \to 0$ (standard estimate: difference of $n+1$-th powers has $h$ to the first in the numerator, and denominators are bounded). By dominated convergence (or uniform convergence),
$$
(f^{(n)})'(z) = \frac{n!}{2\pi i} \int f(w) \cdot (n+1) (w - z)^{-(n+2)} \, dw = \frac{(n+1)!}{2\pi i} \int \frac{f(w)}{(w - z)^{n+2}} \, dw.
$$
Thus the $(n+1)$-version holds. $\blacksquare$

**Remark.** The existence of *all* higher derivatives from mere complex differentiability at a single point is a deep phenomenon. There is no real-variable analogue. It is a consequence of the *global* nature of Cauchy's theorem — differentiability once implies an integral representation, which in turn implies all higher derivatives.

### 0.5.3.3 Analyticity: Holomorphic = Analytic

**Theorem 0.5.3.3 (Holomorphic implies analytic).** Let $f$ be holomorphic on an open set $\Omega$. Then for every $z_0 \in \Omega$, $f$ has a power series expansion
$$
f(z) = \sum_{n=0}^\infty a_n (z - z_0)^n, \qquad a_n = \frac{f^{(n)}(z_0)}{n!} = \frac{1}{2\pi i} \oint_{|w - z_0| = r} \frac{f(w)}{(w - z_0)^{n+1}} \, dw,
$$
converging in the largest open disk $D(z_0, R) \subset \Omega$.

*Proof.* Fix $z_0 \in \Omega$ and $r$ small enough that $\bar D(z_0, r) \subset \Omega$. For $z \in D(z_0, r)$ and $w \in \{|w - z_0| = r\}$,
$$
\frac{1}{w - z} = \frac{1}{(w - z_0) - (z - z_0)} = \frac{1}{w - z_0} \cdot \frac{1}{1 - \frac{z - z_0}{w - z_0}} = \sum_{n=0}^\infty \frac{(z - z_0)^n}{(w - z_0)^{n+1}},
$$
a geometric series converging because $|z - z_0| / |w - z_0| = |z - z_0|/r < 1$, and the convergence is *uniform* for $w$ on the circle and $|z - z_0| \leq r' < r$.

Multiply by $f(w)/(2\pi i)$ and integrate: uniform convergence lets us swap sum and integral:
$$
f(z) = \frac{1}{2\pi i} \int \frac{f(w)}{w - z} dw = \sum_{n=0}^\infty \left[ \frac{1}{2\pi i} \int \frac{f(w)}{(w - z_0)^{n+1}} dw \right] (z - z_0)^n = \sum_n a_n (z - z_0)^n.
$$
This holds for $|z - z_0| < r$. Since $r$ was arbitrary subject to $\bar D(z_0, r) \subset \Omega$, the series converges in the largest such disk, i.e., $|z - z_0| < R := \operatorname{dist}(z_0, \partial \Omega)$. $\blacksquare$

**Corollary 0.5.3.4 (Radius of convergence = distance to nearest singularity).** The radius of convergence of the Taylor series of $f$ at $z_0$ equals the distance from $z_0$ to the nearest singularity of $f$.

*Proof sketch.* The series converges in $D(z_0, R)$ where $R = \operatorname{dist}(z_0, \partial \Omega)$ (Theorem 0.5.3.3). Conversely, if the series converged on a larger disk, $f$ would extend holomorphically there, contradicting that the boundary of $\Omega$ is the set where $f$ fails to be holomorphic. (For this to be tight, we need $\Omega$ to be the *maximal* domain of holomorphy of $f$; the statement is about this maximal domain.) $\blacksquare$

**Example.** $f(z) = 1/(1 + z^2)$ on $\mathbb{R}$. As a real function it is $C^\infty$ everywhere; its Maclaurin series has radius of convergence $1$, and this is mysterious from the real viewpoint. Complex analysis explains it: the singularities are at $z = \pm i$, each at distance $1$ from $0$. Hence $R = 1$. This is the archetypal example for teaching why power series can have finite radius of convergence despite the real function being smooth.

### 0.5.3.4 Cauchy's Estimates

**Theorem 0.5.3.5 (Cauchy's inequalities).** If $f$ is holomorphic on an open set containing $\bar D(z_0, r)$ and $|f| \leq M$ on $\{|w - z_0| = r\}$, then
$$
|f^{(n)}(z_0)| \leq \frac{n! M}{r^n} \quad \text{for all } n \geq 0.
$$

*Proof.* From Theorem 0.5.3.2 with $z = z_0$:
$$
|f^{(n)}(z_0)| = \left| \frac{n!}{2\pi i} \int_{|w-z_0|=r} \frac{f(w)}{(w - z_0)^{n+1}} dw \right| \leq \frac{n!}{2\pi} \cdot \frac{M}{r^{n+1}} \cdot 2\pi r = \frac{n! M}{r^n},
$$
using ML with bound $M/r^{n+1}$ and length $2\pi r$. $\blacksquare$

This gives *a priori* bounds on derivatives in terms of sup-norm bounds on the function — completely impossible in real analysis (where a bounded function can have arbitrarily wild derivatives).

### 0.5.3.5 Liouville's Theorem and the Fundamental Theorem of Algebra

**Theorem 0.5.3.6 (Liouville).** A bounded entire function is constant.

*Proof.* Let $f$ be entire with $|f| \leq M$ on $\mathbb{C}$. For any $z_0 \in \mathbb{C}$ and any $r > 0$, Cauchy's estimate gives $|f'(z_0)| \leq M/r$. Letting $r \to \infty$: $|f'(z_0)| = 0$. So $f' \equiv 0$, hence $f$ is constant (on connected $\mathbb{C}$). $\blacksquare$

**Theorem 0.5.3.7 (Fundamental theorem of algebra).** Every nonconstant polynomial $p \in \mathbb{C}[z]$ has a root in $\mathbb{C}$.

*Proof.* Suppose not: $p(z) \neq 0$ for all $z \in \mathbb{C}$. Then $1/p$ is entire. As $|z| \to \infty$, $|p(z)| \to \infty$ (for degree $\geq 1$), so $|1/p(z)| \to 0$. In particular, $1/p$ is bounded: outside some large disk $|z| \leq R$, $|1/p| < 1$; on the compact disk $|z| \leq R$, $1/p$ is continuous, hence bounded. By Liouville, $1/p$ is constant, so $p$ is constant — contradicting that $p$ is nonconstant. $\blacksquare$

Beautifully short. A proof of FTA without complex analysis (e.g., via the intermediate value theorem or degree theory) typically takes a page; Cauchy's integral formula cuts it to a paragraph.

**Corollary.** Every polynomial of degree $n$ has exactly $n$ roots in $\mathbb{C}$, counted with multiplicity (by iteratively dividing out factors).

### 0.5.3.6 Morera's Theorem (converse to Cauchy)

**Theorem 0.5.3.8 (Morera).** Let $\Omega \subset \mathbb{C}$ be open, $f: \Omega \to \mathbb{C}$ continuous. If $\int_{\partial T} f \, dz = 0$ for every closed triangle $T \subset \Omega$, then $f$ is holomorphic on $\Omega$.

*Proof.* Fix $z_0 \in \Omega$ and a disk $D(z_0, r) \subset \Omega$. In the disk, define
$$
F(z) := \int_{[z_0, z]} f(w) \, dw.
$$
For $z, z + h \in D(z_0, r)$, the triangle with vertices $z_0, z, z + h$ lies in the disk and has $\int_{\partial T} f = 0$, so $F(z + h) - F(z) = \int_{[z, z+h]} f \, dw$. The usual argument shows $F'(z) = f(z)$, so $F$ is holomorphic with derivative $f$. By Cauchy's integral formula applied to $F$, $F$ has a holomorphic derivative, i.e., $f = F'$ is holomorphic on $D(z_0, r)$. Since $z_0$ was arbitrary, $f$ is holomorphic on $\Omega$. $\blacksquare$

**Application: uniform limits of holomorphic functions are holomorphic.**

**Corollary 0.5.3.9 (Weierstrass convergence theorem).** Let $f_n: \Omega \to \mathbb{C}$ be holomorphic and $f_n \to f$ uniformly on compact subsets of $\Omega$. Then $f$ is holomorphic on $\Omega$, and $f_n^{(k)} \to f^{(k)}$ uniformly on compact subsets for each $k$.

*Proof.* Holomorphy: for each closed triangle $T \subset \Omega$, $\int_{\partial T} f_n \, dz = 0$ (Goursat for each $f_n$). Uniform convergence on $T$ (compact) allows swapping limit and integral: $\int_{\partial T} f \, dz = \lim_n \int_{\partial T} f_n \, dz = 0$. $f$ is continuous (uniform limit of continuous), so Morera applies: $f$ is holomorphic.

Derivatives: for $z_0 \in \Omega$ and closed disk $\bar D(z_0, r) \subset \Omega$, Cauchy's formula gives
$$
f_n^{(k)}(z_0) - f^{(k)}(z_0) = \frac{k!}{2\pi i} \int_{|w-z_0|=r} \frac{f_n(w) - f(w)}{(w - z_0)^{k+1}} dw,
$$
and ML with $\sup_{|w - z_0| = r} |f_n(w) - f(w)| \to 0$ gives $|f_n^{(k)}(z_0) - f^{(k)}(z_0)| \to 0$; uniform on compact subsets follows by a standard compactness argument (cover compact $K$ by finitely many small disks, apply the pointwise estimate uniformly). $\blacksquare$

This is *remarkable*: uniform convergence of holomorphic functions automatically gives uniform convergence of *all* derivatives. In real analysis, uniform convergence of $f_n$ does not even imply $f'_n$ converges (let alone uniformly).

### 0.5.3.7 The Maximum Modulus Principle

**Theorem 0.5.3.10 (Maximum modulus principle).** Let $f$ be holomorphic on a connected open $\Omega$. If $|f|$ attains its maximum at some $z_0 \in \Omega$, then $f$ is constant on $\Omega$.

*Proof.* Suppose $|f(z_0)| = M := \sup_\Omega |f|$. By the mean value property,
$$
f(z_0) = \frac{1}{2\pi} \int_0^{2\pi} f(z_0 + re^{i\theta}) \, d\theta
$$
for small $r$. Taking absolute values and using the integral triangle inequality:
$$
M = |f(z_0)| \leq \frac{1}{2\pi} \int_0^{2\pi} |f(z_0 + re^{i\theta})| \, d\theta \leq \frac{1}{2\pi} \int_0^{2\pi} M \, d\theta = M.
$$
So equality holds everywhere, which forces $|f(z_0 + re^{i\theta})| = M$ for almost every $\theta$. By continuity, $|f| \equiv M$ on the whole circle $|w - z_0| = r$, for every small $r$; hence on a small disk around $z_0$.

**From "$|f|$ constant" to "$f$ constant":** If $|f|^2 = u^2 + v^2 = M^2$ on the disk, take $\partial/\partial x$: $2uu_x + 2vv_x = 0$, similarly $uu_y + vv_y = 0$. Using C-R ($u_x = v_y, u_y = -v_x$): substitute to get $u v_y - v v_x \cdot (-1) = ... $ — cleaner: $|f|^2$ constant means $f \bar f$ constant. If $f \neq 0$ anywhere, compute $\partial(f \bar f)/\partial \bar z = f \partial \bar f / \partial \bar z = f \overline{f'}$ (Wirtinger); so $f \overline{f'} = 0$, hence $f' = 0$ where $f \neq 0$. If $f \equiv 0$ nearby, trivially constant.

So $f$ is constant on a neighborhood of $z_0$. By the identity principle (next theorem), $f$ is constant on all of $\Omega$. $\blacksquare$

**Corollary 0.5.3.11 (Maximum on boundary).** Let $\Omega$ be bounded open with closure $\bar \Omega$, $f$ continuous on $\bar \Omega$ and holomorphic on $\Omega$. Then $\max_{\bar\Omega} |f|$ is attained on $\partial \Omega$.

*Proof.* $|f|$ is continuous on compact $\bar\Omega$, so attains a maximum somewhere. If the max is attained only in the interior, the theorem forces $f$ constant, in which case the max is also attained on the boundary. $\blacksquare$

**Minimum modulus.** If $f$ is nonzero on $\Omega$, then $1/f$ is holomorphic and the same argument gives: $|f|$ attains its *minimum* on $\partial \Omega$.

### 0.5.3.8 The Identity Principle

**Theorem 0.5.3.12 (Identity principle).** Let $f, g$ be holomorphic on a connected open $\Omega$. The following are equivalent:

(a) $f = g$ on $\Omega$.

(b) The set $\{z \in \Omega : f(z) = g(z)\}$ has an accumulation point in $\Omega$.

(c) There exists $z_0 \in \Omega$ with $f^{(n)}(z_0) = g^{(n)}(z_0)$ for all $n \geq 0$.

*Proof.* It suffices to show (b) ⇔ (c) ⇒ (a) after replacing $f - g$ by $h$: we need to show $h$ holomorphic on $\Omega$ with a zero of infinite order at some $z_0$, OR with a zero accumulating in $\Omega$, implies $h \equiv 0$.

**(c) ⇒ (a):** Suppose all derivatives of $h$ vanish at $z_0$. Then the Taylor series of $h$ at $z_0$ is identically zero, so $h$ vanishes on the disk of convergence $D(z_0, R)$. Let
$$
U := \{z \in \Omega : h \equiv 0 \text{ on a neighborhood of } z\}.
$$
$U$ is open by definition. $U$ is also *closed in $\Omega$*: if $z_n \to z \in \Omega$ with $z_n \in U$, then $h$ has all derivatives zero on a sequence converging to $z$ — specifically, all derivatives $h^{(k)}$ are continuous and zero on each $U$-neighborhood, hence zero at $z$. By the argument at $z_0$, $h \equiv 0$ on a neighborhood of $z$, so $z \in U$. By connectedness, $U = \Omega$ (it's nonempty, open, and closed).

**(b) ⇒ (c):** Suppose $h$ has zeros with an accumulation point $z_0 \in \Omega$. Expand $h(z) = \sum a_n (z - z_0)^n$ in a disk $D(z_0, r) \subset \Omega$. If not all $a_n = 0$, let $k$ be the smallest with $a_k \neq 0$; then $h(z) = (z - z_0)^k [a_k + a_{k+1}(z - z_0) + \cdots]$. The factor in brackets is $a_k + O(z - z_0)$, nonzero near $z_0$. So $h(z) = 0$ has solutions $z$ with $z = z_0$ (where $(z - z_0)^k = 0$) *only*, on a punctured neighborhood of $z_0$. But $z_n \to z_0$ are zeros of $h$ with $z_n \neq z_0$, contradiction. So all $a_n = 0$, giving (c). $\blacksquare$

**Consequence.** Two holomorphic functions that agree on a sequence accumulating inside the domain (e.g., on a small arc, or on the real line if the domain contains an interval) must agree everywhere on a connected domain. This is why the complex extension $\sin(z)$ of $\sin(x)$ is *unique* — it is pinned down by its values on the real line.

### 0.5.3.9 Zeros and Order

**Definition 0.5.3.13 (Order of a zero).** If $f$ is holomorphic at $z_0$ with $f(z_0) = 0$ but $f \not\equiv 0$ near $z_0$, then there is a unique integer $k \geq 1$ with $f(z) = (z - z_0)^k g(z)$ where $g$ holomorphic at $z_0$ and $g(z_0) \neq 0$. We call $k$ the **order** (or multiplicity) of the zero. Equivalently, $k = \min\{n : f^{(n)}(z_0) \neq 0\}$.

**Proposition 0.5.3.14 (Zeros are isolated).** The zeros of a nonzero holomorphic function on a connected open set are isolated points.

Already proven in the identity principle. This is why "generically meet transversally / at isolated points" is the right mental model for zeros in complex analysis.

### 0.5.3.10 Schwarz's Lemma

An elegant consequence of the maximum principle applied cleverly.

**Theorem 0.5.3.15 (Schwarz's lemma).** Let $f: D \to D$ be holomorphic (where $D = \{|z| < 1\}$ is the open unit disk), with $f(0) = 0$. Then:

(a) $|f(z)| \leq |z|$ for all $z \in D$.

(b) $|f'(0)| \leq 1$.

(c) Equality in (a) for some $z \neq 0$, or in (b), implies $f(z) = \lambda z$ for some $|\lambda| = 1$ (a rotation).

*Proof.* Define $g(z) := f(z)/z$ for $z \neq 0$, and $g(0) := f'(0)$. Since $f(0) = 0$, $g$ is holomorphic at $0$ (singularity removable because $f(z) = zf'(0) + O(z^2)$, so $g(z) \to f'(0)$).

On any circle $|z| = r < 1$, $|f(z)| < 1$ hence $|g(z)| \leq 1/r$. By the maximum modulus principle applied to $\bar D(0, r)$, $|g| \leq 1/r$ on all of $\bar D(0, r)$. Letting $r \to 1^-$: $|g(z)| \leq 1$ for all $z \in D$.

So $|f(z)/z| \leq 1$, giving (a); and $|g(0)| = |f'(0)| \leq 1$, giving (b).

For (c): equality means $|g|$ attains its max $1$ at some interior point, forcing $g$ constant (maximum modulus principle). Say $g \equiv \lambda$ with $|\lambda| = 1$; then $f(z) = \lambda z$. $\blacksquare$

**Consequence.** Automorphisms of $D$ (holomorphic bijections $D \to D$) that fix $0$ are precisely rotations $z \mapsto \lambda z$. All automorphisms of $D$ are **Möbius transformations** of the form $\lambda (z - a)/(1 - \bar a z)$ — a cornerstone result for hyperbolic geometry and the Riemann mapping theorem.

### 0.5.3.11 Python: Numerical Verification

```python
import numpy as np

def cauchy_integral_value(f, z0, center, r, n_points=5000):
    """Numerically compute (1/2πi) ∮_{|w-center|=r} f(w)/(w - z0) dw.
    Should equal f(z0) by Cauchy's integral formula."""
    ts = np.linspace(0, 2*np.pi, n_points, endpoint=False) + np.pi/n_points
    ws = center + r * np.exp(1j * ts)
    dw = 1j * r * np.exp(1j * ts) * (2*np.pi / n_points)
    integrand = f(ws) / (ws - z0)
    return np.sum(integrand * dw) / (2j * np.pi)

# Test: f(z) = e^z, center=0, r=2, evaluate at z0=1+i
f = np.exp
z0 = 1 + 1j
val = cauchy_integral_value(f, z0, 0, 2)
print(f"Cauchy formula for e^z at z0={z0}: {val:.6f}")
print(f"Direct:                            {f(z0):.6f}")

# Test derivatives: f^(n)(z0) = (n!/2πi) ∮ f(w)/(w-z0)^(n+1) dw
def cauchy_derivative(f, z0, center, r, n, n_points=5000):
    ts = np.linspace(0, 2*np.pi, n_points, endpoint=False) + np.pi/n_points
    ws = center + r * np.exp(1j * ts)
    dw = 1j * r * np.exp(1j * ts) * (2*np.pi / n_points)
    integrand = f(ws) / (ws - z0)**(n+1)
    from math import factorial
    return factorial(n) * np.sum(integrand * dw) / (2j * np.pi)

# f(z) = z^5, z0=0: f^(n)(0) should be 5! for n=5, 0 otherwise.
f = lambda z: z**5
for n in range(7):
    val = cauchy_derivative(f, 0, 0, 1, n)
    from math import factorial
    expected = factorial(5) if n == 5 else 0
    print(f"n={n}: computed={val:.6f}, expected={expected}")

# Maximum modulus check: |sin(z)| on closed disk of radius 2 centered at 0.
# Max should be attained on boundary.
N = 50
zs = np.linspace(-2, 2, N)
grid_max = 0
for x in zs:
    for y in zs:
        z = x + 1j*y
        if abs(z) < 2 - 0.01:  # interior
            grid_max = max(grid_max, abs(np.sin(z)))

ts = np.linspace(0, 2*np.pi, 1000)
boundary_max = max(abs(np.sin(2*np.exp(1j*t))) for t in ts)
print(f"\nInterior max of |sin(z)|:  {grid_max:.4f}")
print(f"Boundary max of |sin(z)|:  {boundary_max:.4f}")
print(f"Maximum modulus principle: boundary max ≥ interior max? {boundary_max >= grid_max}")

# Liouville illustration: any bounded entire function must be constant.
# Compute Cauchy's estimate |f'(0)| ≤ M/r for f = sin, various r.
# sin is NOT bounded — |sin(iy)| = sinh(y) -> ∞ — so Liouville doesn't apply.
for r in [1, 5, 10, 50]:
    M = max(abs(np.sin(r*np.exp(1j*t))) for t in np.linspace(0, 2*np.pi, 1000))
    bound = M / r
    print(f"r={r}: M=sup|sin(re^iθ)|≈{M:.2e}, M/r bound on |f'(0)|={bound:.4f}")
print("|f'(0)| = |cos(0)| = 1 (actual value)")
# The bound M/r grows like e^r/r, so doesn't prove Liouville for sin (as it shouldn't).
```

Expected output shows (a) Cauchy's formula recovers function values and derivatives to high numerical precision; (b) maximum modulus principle is respected; (c) Cauchy's estimate gives *trivial* bounds when $f$ is unbounded, consistent with the fact that $\sin$ is unbounded on $\mathbb{C}$ and hence doesn't violate Liouville.

### 0.5.3.12 [QUANT APPLICATION]

**1. Greeks in option pricing.** The $n$-th derivative Greek $\partial^n V / \partial S^n$ of an option price $V(S)$ can be expressed as a contour integral:
$$
\frac{\partial^n V}{\partial S^n}(S_0) = \frac{n!}{2\pi i} \int_{|w - S_0| = r} \frac{V(w)}{(w - S_0)^{n+1}} dw.
$$
This is useful numerically because it avoids finite-difference instability: instead of computing small-$h$ difference quotients (which are ill-conditioned), compute $V$ at $O(n)$ evenly spaced points on a circle and do a DFT. This is the **Cauchy-Lyness method** for numerical differentiation.

**2. Radius of convergence for series representations.** If an option price has a series expansion in terms of a parameter (e.g., stochastic volatility of volatility $\nu$), the radius of convergence equals the distance to the nearest singularity. For the Heston model, the characteristic function has a branch cut at $\nu = \nu^*$ (critical vol-of-vol), and the series expansion breaks down there. Knowing this complex-analytic structure is essential for extending Taylor methods.

**3. Maximum principle in risk management.** Consider a bond price $B(t, r)$ as a function of interest rate $r$. Under certain affine-model hypotheses, $B$ is holomorphic in $r$ on a strip of the complex plane. The maximum principle then bounds the bond price in terms of its values at extreme scenarios ("stress points"). This is the complex-analytic skeleton of some stress-testing frameworks.

**4. Identity principle and model calibration.** Two different models producing the same prices at a countable set of strikes $\{K_n\}$ with an accumulation point (e.g., $K_n \to K^*$) must produce the same prices *everywhere* — provided the prices are holomorphic in $K$. This restricts what "a few calibration points" can pin down and provides a complex-analytic sanity check on model identifiability.

**5. Schwarz lemma and contraction mappings.** In the study of Markov chains on the disk (e.g., in reflection-principle arguments for the heat equation), the Schwarz lemma plays the role of a universal contraction bound. Also, any holomorphic self-map of the disk is distance-contracting in the Poincaré metric — a fact used in ergodic analyses of complex dynamical systems.

### 0.5.3.13 Exercises

#### ★ (Warm-up)

**E0.5.3.1.** Using Cauchy's integral formula, compute $\int_{|z|=2} \frac{e^z}{z - 1} dz$.

**E0.5.3.2.** Compute $\int_{|z|=1} \frac{\cos z}{z} dz$ and $\int_{|z|=1} \frac{\cos z}{z^2} dz$ (the second uses the derivative form).

**E0.5.3.3.** Find the Taylor series of $f(z) = 1/(1-z)$ at $z_0 = 0$ and at $z_0 = 2$. What is the radius of convergence in each case? Why are they different?

**E0.5.3.4.** Find the Taylor series of $f(z) = \frac{1}{(z - 2)(z - 3)}$ at $z_0 = 0$. What is the radius of convergence? (Use partial fractions.)

**E0.5.3.5.** Prove the mean value property: $f(z_0) = \frac{1}{2\pi} \int_0^{2\pi} f(z_0 + re^{i\theta}) d\theta$ directly from Cauchy's integral formula.

**E0.5.3.6.** Using Cauchy's estimates, show that if $f$ is entire with $|f(z)| \leq A + B|z|^k$ for some $k \in \mathbb{N}$ and constants $A, B > 0$, then $f$ is a polynomial of degree $\leq k$.

#### ★★ (Standard)

**E0.5.3.7 (Ferrari evaluation).** Evaluate $\int_{|z| = 1} \frac{e^z}{(z - 1/2)^3} dz$ using Cauchy's formula for derivatives.

**E0.5.3.8 (Real integrals via contour).** Evaluate $\int_0^{2\pi} \frac{d\theta}{a + b\cos\theta}$ for $a > |b| > 0$. (Hint: substitute $z = e^{i\theta}$, get a contour integral over $|z|=1$, apply Cauchy's formula.)

**E0.5.3.9 (Gutzmer's formula).** Prove: if $f(z) = \sum a_n z^n$ on $\bar D(0, r)$, then
$$
\frac{1}{2\pi} \int_0^{2\pi} |f(re^{i\theta})|^2 d\theta = \sum |a_n|^2 r^{2n}.
$$
(This is Parseval's identity for complex Fourier series, derived from Cauchy's formula applied to $|f|^2 = f \bar f$ — but we have to be careful since $\bar f$ isn't holomorphic. Use direct expansion: $|f|^2 = (\sum a_n z^n)(\sum \bar{a_m} \bar z^m)$ and integrate term by term.)

**E0.5.3.10.** Suppose $f$ is entire with $\operatorname{Re} f(z) \leq M$ for all $z$. Show $f$ is constant. (Hint: apply Liouville to $e^f$.)

**E0.5.3.11.** Suppose $f, g$ are entire with $f(z) g(z) = 0$ for all $z$. Show $f \equiv 0$ or $g \equiv 0$. (Identity principle on zero sets.)

**E0.5.3.12 (Weierstrass approximation via Taylor).** Let $f: [0, 1] \to \mathbb{R}$ be continuous. Show that there exist complex polynomials $p_n(z)$ on a domain containing $[0, 1]$ such that $p_n \to f$ uniformly on $[0, 1]$. (Not the same as Weierstrass via Bernstein; here use Runge's theorem or extend $f$ somehow — read Stein-Shakarchi Ch 2.4.)

**E0.5.3.13 (Schwarz lemma for half-plane).** State and prove a Schwarz lemma for the upper half plane $\mathbb{H}$: if $f: \mathbb{H} \to \mathbb{H}$ is holomorphic with $f(i) = i$, what can you say about $f$? (Use a conformal map $\mathbb{H} \to D$ to reduce to the disk case.)

**E0.5.3.14 (Open mapping theorem).** Prove: a nonconstant holomorphic function on a connected open set maps open sets to open sets. (Hint: use Rouché's theorem or argue directly that if $f$ is nonconstant at $z_0$, then near $f(z_0)$, $f$ takes all values close to $f(z_0)$.)

#### ★★★ (Challenge)

**E0.5.3.15 (Schwarz reflection principle).** Let $f$ be holomorphic on a domain $\Omega \subset \mathbb{H}$ (upper half plane) that extends continuously to the real axis segment $(a, b)$, and assume $f$ is real on $(a, b)$. Show $f$ extends holomorphically to $\Omega \cup (a, b) \cup \Omega^*$, where $\Omega^* = \{\bar z : z \in \Omega\}$, via $f(\bar z) := \overline{f(z)}$. (Use Morera's theorem: check the extended function is continuous and triangle integrals vanish.)

**E0.5.3.16 (Analytic in strip).** Let $f$ be holomorphic on the strip $\{0 < \operatorname{Im} z < 1\}$ and continuous on its closure, with $|f| \leq 1$ on the boundary lines. Show $|f| \leq 1$ on the whole strip, assuming $f$ is bounded. (Phragmén-Lindelöf principle; requires an auxiliary function $e^{\epsilon e^{-iz}}$ or similar to control growth.)

**E0.5.3.17 (Hadamard three-lines theorem).** Under the setup of E0.5.3.16, if $\sup_{\operatorname{Re} z = x} |f(z)| = M(x)$, show $\log M(x)$ is a convex function of $x \in [0, 1]$. (Cornerstone of interpolation theory.)

**E0.5.3.18 (Jensen's formula).** Let $f$ be holomorphic on $\bar D(0, 1)$ with $f(0) \neq 0$, and let $a_1, \ldots, a_n$ be its zeros in $D(0, 1)$ (counted with multiplicity). Prove Jensen's formula:
$$
\log |f(0)| = -\sum_{k=1}^n \log \frac{1}{|a_k|} + \frac{1}{2\pi} \int_0^{2\pi} \log |f(e^{i\theta})| \, d\theta.
$$
(Relates zero distribution to boundary behavior; crucial in value-distribution theory and in the theory of Hardy spaces $H^p$.)

**E0.5.3.19 (Riemann mapping theorem, preview).** State the Riemann mapping theorem: any simply connected proper open subset of $\mathbb{C}$ is conformally equivalent to the unit disk $D$. Using Schwarz's lemma, show that the conformal map is unique up to post-composition with an automorphism of $D$. (The existence proof requires Montel's theorem and a normal-families argument; defer to a complex analysis course.)

**E0.5.3.20 (Blaschke products).** A **Blaschke factor** is $B_a(z) := (z - a)/(1 - \bar a z)$ for $|a| < 1$. Show $B_a$ is an automorphism of $D$. Given a sequence $\{a_n\} \subset D$, the **Blaschke product** $\prod B_{a_n}(z)$ converges uniformly on compact subsets iff $\sum (1 - |a_n|) < \infty$. Prove the convergence condition. (Foundation of $H^\infty$ theory; underlies bounded analytic function factorization.)

---

## Topic 0.5.4 — Laurent Series and Residues

### 0.5.4.1 Motivation

Taylor series represent holomorphic functions inside a disk. But many quant-relevant functions have singularities: $1/z$, $1/(z^2 + 1)$, $e^{1/z}$, $\sin z / z$ near $z = 0$. The key idea of Laurent series is that such functions can be expanded in an **annulus** around a singularity, in a series involving both positive and negative powers of $(z - z_0)$. The coefficient of $(z - z_0)^{-1}$ is called the **residue** and encodes the integral of $f$ around a small loop — by extension, it drives the residue theorem, the most powerful computational tool in complex analysis.

### 0.5.4.2 Laurent Series: Definition and Existence

**Theorem 0.5.4.1 (Laurent expansion).** Let $f$ be holomorphic on the annulus
$$
A := \{z \in \mathbb{C} : r < |z - z_0| < R\}, \qquad 0 \leq r < R \leq \infty.
$$
Then $f$ has a unique **Laurent series** representation
$$
\boxed{f(z) = \sum_{n = -\infty}^\infty a_n (z - z_0)^n, \qquad a_n = \frac{1}{2\pi i} \oint_{|w - z_0| = \rho} \frac{f(w)}{(w - z_0)^{n+1}} dw,}
$$
valid on $A$ and converging uniformly on compact subsets, where $\rho$ is any radius with $r < \rho < R$ (value is independent of $\rho$).

*Proof.* Fix $z \in A$ with $r < |z - z_0| < R$ and pick radii $r < r_1 < |z - z_0| < r_2 < R$. Let $\gamma_1$ and $\gamma_2$ be circles of radii $r_1, r_2$ around $z_0$ (both counter-clockwise).

The region between $\gamma_1$ and $\gamma_2$ minus $\{z\}$ is essentially an annulus. Apply Cauchy's integral formula in the "deformation of contour" spirit:

Draw a radial segment $\sigma$ from $\gamma_1$ to $\gamma_2$ that doesn't pass through $z$. The contour $\gamma_2 - \sigma - \gamma_1 + \sigma$ (go around the outer circle, radially inward, around the inner circle clockwise, radially outward) bounds a simply connected region containing $z$. Applying Cauchy's integral formula:
$$
f(z) = \frac{1}{2\pi i} \oint_{\gamma_2 - \gamma_1} \frac{f(w)}{w - z} dw = \frac{1}{2\pi i} \int_{\gamma_2} \frac{f(w)}{w - z} dw - \frac{1}{2\pi i} \int_{\gamma_1} \frac{f(w)}{w - z} dw.
$$
(The $\sigma$-contributions cancel since $\sigma$ is traversed in both directions.)

**Expand the outer integral.** For $w \in \gamma_2$: $|w - z_0| = r_2 > |z - z_0|$, so
$$
\frac{1}{w - z} = \frac{1}{(w - z_0)(1 - \frac{z - z_0}{w - z_0})} = \sum_{n=0}^\infty \frac{(z - z_0)^n}{(w - z_0)^{n+1}},
$$
uniformly convergent in $w$ on $\gamma_2$. Swap sum and integral:
$$
\frac{1}{2\pi i} \int_{\gamma_2} \frac{f(w)}{w - z} dw = \sum_{n=0}^\infty \left[\frac{1}{2\pi i} \int_{\gamma_2} \frac{f(w)}{(w - z_0)^{n+1}} dw \right] (z - z_0)^n = \sum_{n=0}^\infty a_n (z - z_0)^n.
$$

**Expand the inner integral.** For $w \in \gamma_1$: $|w - z_0| = r_1 < |z - z_0|$, so
$$
\frac{1}{w - z} = \frac{-1}{(z - z_0)(1 - \frac{w - z_0}{z - z_0})} = -\sum_{n=0}^\infty \frac{(w - z_0)^n}{(z - z_0)^{n+1}},
$$
uniformly convergent on $\gamma_1$. Swap sum and integral:
$$
-\frac{1}{2\pi i} \int_{\gamma_1} \frac{f(w)}{w - z} dw = \sum_{n=0}^\infty \left[\frac{1}{2\pi i} \int_{\gamma_1} f(w) (w - z_0)^n dw \right] \cdot \frac{1}{(z - z_0)^{n+1}}.
$$
Re-index: let $m = -n - 1$, so $n = -m - 1 \geq 0 \Leftrightarrow m \leq -1$:
$$
= \sum_{m = -\infty}^{-1} \left[\frac{1}{2\pi i} \int_{\gamma_1} \frac{f(w)}{(w - z_0)^{m+1}} dw \right] (z - z_0)^m = \sum_{m=-\infty}^{-1} a_m (z - z_0)^m.
$$
Notice that $\int_{\gamma_1} f(w) (w - z_0)^{-m-1} dw = \int_{\gamma_\rho} f(w) (w - z_0)^{-m-1} dw$ for any $\rho \in (r, R)$ by homotopy (Theorem 0.5.2.15); so we can write all $a_n$ (positive and negative index) as integrals over the *same* circle $|w - z_0| = \rho$.

Combining:
$$
f(z) = \sum_{n = -\infty}^\infty a_n (z - z_0)^n. \quad \blacksquare
$$

**Uniqueness.** If $f = \sum b_n (z - z_0)^n$ on the annulus, then multiplying by $(z - z_0)^{-m-1}$ and integrating over $|w - z_0| = \rho$:
$$
\int_{|w-z_0|=\rho} \frac{f(w)}{(w - z_0)^{m+1}} dw = \sum_n b_n \int_{|w-z_0|=\rho} (w - z_0)^{n - m - 1} dw.
$$
The integral on the right is $2\pi i$ if $n - m - 1 = -1$, i.e., $n = m$, and $0$ otherwise (Example 1 of Topic 0.5.2). So $2\pi i b_m = 2\pi i a_m$, i.e., $b_m = a_m$. $\blacksquare$

**Terminology.** The **principal part** of $f$ at $z_0$ is $\sum_{n < 0} a_n (z - z_0)^n$ (the singular part). The **regular part** is $\sum_{n \geq 0} a_n (z - z_0)^n$ (a power series, holomorphic in a disk).

### 0.5.4.3 Classification of Isolated Singularities

Let $f$ be holomorphic on a punctured disk $D(z_0, R) \setminus \{z_0\}$. By Theorem 0.5.4.1 with inner radius $r = 0$, $f$ has a Laurent expansion there. We classify based on the principal part:

**Definition 0.5.4.2.**

- $z_0$ is a **removable singularity** if $a_n = 0$ for all $n < 0$. Then $f$ extends holomorphically to $z_0$ by defining $f(z_0) := a_0$.
- $z_0$ is a **pole of order $k$** ($k \geq 1$) if $a_{-k} \neq 0$ and $a_n = 0$ for $n < -k$. The principal part has finitely many nonzero terms.
- $z_0$ is an **essential singularity** if infinitely many $a_n$ with $n < 0$ are nonzero. The principal part is an infinite series.

**Example.**
- $\sin(z)/z$ has removable singularity at $0$: $\sin z / z = 1 - z^2/6 + z^4/120 - \cdots$, no negative powers.
- $1/z^3$ has pole of order $3$ at $0$.
- $e^{1/z} = \sum_{n=0}^\infty (1/z)^n/n! = 1 + 1/z + 1/(2z^2) + \cdots$: infinitely many negative powers, essential singularity at $0$.

### 0.5.4.4 Characterizations of Singularity Type

Laurent analysis gives clean characterizations:

**Theorem 0.5.4.3 (Riemann's removable singularity theorem).** Let $f$ be holomorphic on the punctured disk $D(z_0, R) \setminus \{z_0\}$. The following are equivalent:

(a) $z_0$ is a removable singularity.

(b) $f$ is bounded near $z_0$.

(c) $\lim_{z \to z_0} (z - z_0) f(z) = 0$.

*Proof.*
**(a) ⇒ (b):** If $f$ extends holomorphically, it is continuous hence bounded near $z_0$.

**(b) ⇒ (c):** If $|f| \leq M$ near $z_0$, then $|(z - z_0) f(z)| \leq M|z - z_0| \to 0$.

**(c) ⇒ (a):** Consider $g(z) := (z - z_0)^2 f(z)$ with $g(z_0) := 0$. Near $z_0$, $g(z)/((z-z_0)) = (z - z_0) f(z) \to 0$, so $g$ is differentiable at $z_0$ with $g'(z_0) = 0$. $g$ is also holomorphic on $D \setminus \{z_0\}$ (product of holomorphic functions), and continuous at $z_0$ (since $g(z_0) = 0$ matches the limit). By checking directly that $g$ is complex differentiable at $z_0$ (difference quotient has limit $0$), and using Morera (continuous + vanishing triangle integrals), we see $g$ is holomorphic on the whole disk.

$g$ has a zero of order $\geq 2$ at $z_0$, so $g(z) = (z - z_0)^2 h(z)$ with $h$ holomorphic. Thus $f(z) = h(z)$ on the punctured disk, and defining $f(z_0) := h(z_0)$ extends $f$ holomorphically. $\blacksquare$

**Theorem 0.5.4.4 (Characterization of poles).** For isolated singularity $z_0$ of $f$, the following are equivalent:

(a) $z_0$ is a pole of order $k$.

(b) $(z - z_0)^k f(z)$ has a removable singularity at $z_0$ with nonzero limit $a_{-k}$, and $(z - z_0)^{k-1} f(z)$ is unbounded near $z_0$ (so $k$ is the *least* such power).

(c) $|f(z)| \to \infty$ as $z \to z_0$, and the rate of blow-up is $|z - z_0|^{-k}$: $\lim_{z \to z_0} (z - z_0)^k f(z) = a_{-k} \neq 0$.

*Proof sketch.* (a) ⇔ (b) is immediate from Laurent expansion: if $f = \sum_{n \geq -k} a_n (z - z_0)^n$ with $a_{-k} \neq 0$, then $(z - z_0)^k f(z) = \sum_{n \geq -k} a_n (z - z_0)^{n+k} = a_{-k} + a_{-k+1}(z - z_0) + \cdots$, hence holomorphic with nonzero constant term. (c) follows from (a) since the leading term near $z_0$ is $a_{-k}/(z - z_0)^k \to \infty$. $\blacksquare$

**Theorem 0.5.4.5 (Casorati-Weierstrass).** If $z_0$ is an essential singularity of $f$, then $f$ comes arbitrarily close to every complex value in every punctured neighborhood of $z_0$:
$$
\overline{f(D(z_0, \epsilon) \setminus \{z_0\})} = \mathbb{C} \quad \text{for every } \epsilon > 0.
$$

*Proof.* Suppose for contradiction that $f$ avoids some disk $D(w_0, \delta)$ on $D(z_0, \epsilon) \setminus \{z_0\}$. Then $g(z) := 1/(f(z) - w_0)$ is holomorphic on the punctured disk with $|g| \leq 1/\delta$, hence bounded. By Riemann's theorem, $g$ extends holomorphically to $z_0$ — with some value $g(z_0) =: c$.

Case $c \neq 0$: then $f(z) - w_0 = 1/g(z) \to 1/c$ as $z \to z_0$, so $f$ has a removable singularity, not essential. Contradiction.

Case $c = 0$: then $g$ has a zero of order $m \geq 1$ at $z_0$, and $f - w_0 = 1/g$ has a pole of order $m$, so $f$ has a pole — not essential. Contradiction.

In either case, we contradict the assumption that $z_0$ is essential. $\blacksquare$

**Remark.** Picard's theorem (much deeper) strengthens Casorati-Weierstrass: in every punctured neighborhood of an essential singularity, $f$ takes *every* complex value (with at most one exception) infinitely often. Example: $e^{1/z}$ near $z = 0$ — for every $w \neq 0$, the equation $e^{1/z} = w$ has infinitely many solutions near $0$.

### 0.5.4.5 Residues

**Definition 0.5.4.6 (Residue).** Let $z_0$ be an isolated singularity of $f$ with Laurent series $f(z) = \sum a_n (z - z_0)^n$. The **residue** of $f$ at $z_0$ is
$$
\operatorname{Res}_{z_0} f := a_{-1}.
$$

From the formula $a_{-1} = (2\pi i)^{-1} \oint f \, dw$, equivalently:
$$
\oint_{|w - z_0| = \rho} f(w) \, dw = 2\pi i \operatorname{Res}_{z_0} f
$$
for $\rho$ small enough that the disk $\bar D(z_0, \rho)$ is contained in the domain of $f$ and contains no other singularity.

### 0.5.4.6 Computing Residues

**Case 1: Simple pole.** If $f$ has a simple pole at $z_0$, $f(z) = a_{-1}/(z - z_0) + [\text{holomorphic part}]$, so
$$
\boxed{\operatorname{Res}_{z_0} f = \lim_{z \to z_0} (z - z_0) f(z).}
$$

**Case 2: Pole of order $k$.** $f(z) (z - z_0)^k = a_{-k} + a_{-k+1}(z - z_0) + \cdots$, so differentiating $k-1$ times and taking limits:
$$
\boxed{\operatorname{Res}_{z_0} f = \frac{1}{(k-1)!} \lim_{z \to z_0} \frac{d^{k-1}}{dz^{k-1}} \left[ (z - z_0)^k f(z) \right].}
$$

**Case 3: Quotient $f = g/h$ with $g(z_0) \neq 0$ and $h$ having a simple zero at $z_0$.**
$$
h(z) = h'(z_0)(z - z_0) + O((z - z_0)^2),
$$
so $(z - z_0) f(z) \to g(z_0)/h'(z_0)$:
$$
\boxed{\operatorname{Res}_{z_0} \frac{g}{h} = \frac{g(z_0)}{h'(z_0)} \quad (\text{simple zero of } h).}
$$

**Case 4: Essential singularity.** No shortcut; must read off the $(z - z_0)^{-1}$ coefficient from the full Laurent expansion.

### 0.5.4.7 The Residue Theorem

**Theorem 0.5.4.7 (Residue theorem).** Let $\Omega \subset \mathbb{C}$ be a simply connected open set, $f$ holomorphic on $\Omega$ except at isolated singularities $z_1, \ldots, z_n \in \Omega$. Let $\gamma$ be a closed piecewise-$C^1$ curve in $\Omega \setminus \{z_1, \ldots, z_n\}$ that is homologous to zero in $\Omega$ (in particular, any simple closed curve). Then
$$
\boxed{\oint_\gamma f(z) \, dz = 2\pi i \sum_{k=1}^n n(\gamma, z_k) \operatorname{Res}_{z_k} f,}
$$
where $n(\gamma, z_k)$ is the winding number.

For a **simple closed curve** $\gamma$ traversing counter-clockwise around a region that contains exactly $z_1, \ldots, z_m$ (a subset of the singularities), the formula simplifies to
$$
\oint_\gamma f(z) \, dz = 2\pi i \sum_{k=1}^m \operatorname{Res}_{z_k} f.
$$

*Proof.* Around each $z_k$, pick a small circle $C_k$ of radius $\epsilon_k$ contained in $\gamma$'s interior (if $\gamma$ is simple closed) and not containing any other singularity. The region $R$ bounded by $\gamma$ (outer) and the $C_k$'s (inner, oriented clockwise from outside) is a region on which $f$ is holomorphic — a finitely connected region. By a multi-circle version of Cauchy's theorem (connect the inner circles to the outer via slit cuts, forming a simply connected region; the slits cancel by opposite traversals),
$$
\oint_\gamma f \, dz + \sum_k \oint_{-C_k} f \, dz = 0, \quad \text{i.e.,} \quad \oint_\gamma f \, dz = \sum_k \oint_{C_k} f \, dz = \sum_k 2\pi i \operatorname{Res}_{z_k} f.
$$
The general version with winding numbers is similar but tracks orientation. $\blacksquare$

### 0.5.4.8 Applications: Computing Real Integrals

**Example 1: $\int_{-\infty}^\infty \frac{dx}{x^2 + 1} = \pi$.**

Consider $f(z) = 1/(z^2 + 1) = 1/[(z - i)(z + i)]$. Singularities at $\pm i$. Close the real axis contour with a semicircle $\Gamma_R$ in the upper half plane (radius $R$). The closed contour encloses $z = i$ but not $z = -i$.

**Residue at $i$:** simple pole with $g(z) = 1$, $h(z) = z^2 + 1$, $h'(i) = 2i$, so $\operatorname{Res}_i = 1/(2i)$.

**Arc vanishing:** on $\Gamma_R$, $|z^2 + 1| \geq R^2 - 1$, so $|f| \leq 1/(R^2 - 1)$, and arc length $= \pi R$, giving arc integral bounded by $\pi R/(R^2 - 1) \to 0$.

Therefore:
$$
\int_{-\infty}^\infty \frac{dx}{x^2 + 1} = \lim_{R \to \infty} \int_{-R}^R = \lim_{R\to\infty} \left[ \oint_{\text{semi}} - \int_{\Gamma_R} \right] = 2\pi i \cdot \frac{1}{2i} - 0 = \pi.
$$

**Example 2: $\int_{-\infty}^\infty \frac{\cos x}{x^2 + 1} dx = \pi/e$.**

Consider $f(z) = e^{iz}/(z^2 + 1)$, close in upper half plane. On $\Gamma_R$, $|e^{iz}| = e^{-\operatorname{Im} z} \leq 1$; combined with $|z^2 + 1| \geq R^2 - 1$, the ML estimate gives arc integral $\leq \pi R/(R^2 - 1) \to 0$.

Actually we need the tighter **Jordan's lemma** because $e^{iz}$ doesn't decay to zero on the whole arc — but the integrand decays in absolute value away from the real axis. Jordan's lemma states: $\int_{\Gamma_R} |e^{iz}| \, |dz| \leq \pi$ for any $R > 0$. Combined with $|f| \leq 1/(R^2 - 1)$ on $\Gamma_R$, we get arc integral $\leq \pi/(R^2 - 1) \to 0$.

**Residue at $z = i$:** $\operatorname{Res}_i = e^{i \cdot i}/(2i) = e^{-1}/(2i)$. So
$$
\int_{-\infty}^\infty \frac{e^{ix}}{x^2 + 1} dx = 2\pi i \cdot \frac{e^{-1}}{2i} = \pi/e.
$$
Taking real parts: $\int_{-\infty}^\infty \frac{\cos x}{x^2 + 1} dx = \pi/e$.

**Example 3: $\int_0^{2\pi} \frac{d\theta}{a + b \cos\theta} = \frac{2\pi}{\sqrt{a^2 - b^2}}$ for $a > |b| > 0$.**

Substitute $z = e^{i\theta}$, $d\theta = dz/(iz)$, $\cos\theta = (z + z^{-1})/2$:
$$
\int_0^{2\pi} \frac{d\theta}{a + b\cos\theta} = \oint_{|z|=1} \frac{1}{a + b(z + 1/z)/2} \cdot \frac{dz}{iz} = \oint_{|z|=1} \frac{2}{i(bz^2 + 2az + b)} dz.
$$
Roots of $bz^2 + 2az + b$: $z = (-a \pm \sqrt{a^2 - b^2})/b$. For $a > |b| > 0$, product of roots is $1$ (Vieta), so one root is inside unit disk, one outside. Inside root: $z_+ = (-a + \sqrt{a^2 - b^2})/b$. Residue at $z_+$ of $2/[i \cdot b(z - z_+)(z - z_-)]$ is $2/[i \cdot b(z_+ - z_-)] = 2/[i \cdot 2\sqrt{a^2 - b^2}]$. Integral $= 2\pi i \cdot 2/[2 i \sqrt{a^2 - b^2}] = 2\pi/\sqrt{a^2 - b^2}$.

### 0.5.4.9 The Argument Principle

**Theorem 0.5.4.8 (Argument principle).** Let $f$ be meromorphic on a simply connected $\Omega$, and $\gamma$ a simple closed curve in $\Omega$ not passing through any zero or pole of $f$. Let $Z$ = number of zeros inside $\gamma$ (with multiplicity) and $P$ = number of poles inside $\gamma$ (with multiplicity). Then
$$
\frac{1}{2\pi i} \oint_\gamma \frac{f'(z)}{f(z)} dz = Z - P.
$$

*Proof.* Near a zero of order $m$ at $z_0$: $f(z) = (z - z_0)^m g(z)$ with $g(z_0) \neq 0$. Then $f'/f = m/(z - z_0) + g'/g$; the second term is holomorphic near $z_0$, so the residue at $z_0$ is $m$. Near a pole of order $m$: $f(z) = (z - z_0)^{-m} h(z)$ with $h(z_0) \neq 0$. Then $f'/f = -m/(z - z_0) + h'/h$; residue is $-m$. Sum over all interior zeros and poles using the residue theorem. $\blacksquare$

**Equivalent form:** $f'/f = (d/dz) \log f$ (locally), so the integral measures the change in $\arg f$ as we traverse $\gamma$: $(\text{change in } \arg f)/(2\pi) = Z - P$. This is why the theorem is called the "argument principle" — it counts net zeros and poles via the winding number of $f \circ \gamma$ around $0$.

**Theorem 0.5.4.9 (Rouché's theorem).** Let $f, g$ be holomorphic on a simply connected $\Omega$, $\gamma$ a simple closed curve in $\Omega$, and suppose $|g(z)| < |f(z)|$ on $\gamma$. Then $f$ and $f + g$ have the same number of zeros (with multiplicity) inside $\gamma$.

*Proof.* Consider $h_t(z) := f(z) + tg(z)$ for $t \in [0, 1]$; $h_0 = f$, $h_1 = f + g$. On $\gamma$, $|h_t| \geq |f| - t|g| > 0$, so $h_t$ is never zero on $\gamma$. By the argument principle, $Z_t := (2\pi i)^{-1} \oint_\gamma h_t'/h_t \, dz$ is a continuous integer-valued function of $t$, hence constant: $Z_0 = Z_1$. $\blacksquare$

**Application.** Rouché's theorem is a powerful tool for locating zeros. E.g., for a polynomial $p(z) = z^n + a_{n-1} z^{n-1} + \cdots + a_0$, on a large circle $|z| = R$ with $R > 1 + \sum_k |a_k|$, $|z^n| = R^n$ dominates $|a_{n-1} z^{n-1} + \cdots + a_0| \leq (|a_{n-1}| + \cdots + |a_0|)R^{n-1}$ — if $R$ is large enough. Rouché's theorem then implies $p$ has as many zeros inside $|z| = R$ as $z^n$, namely $n$. This is another proof of the fundamental theorem of algebra.

### 0.5.4.10 Worked Examples

**Example 1: Residue at a double pole.** $f(z) = e^z/z^2$ at $z = 0$.

$e^z = 1 + z + z^2/2! + z^3/3! + \cdots$, so $e^z/z^2 = 1/z^2 + 1/z + 1/2 + z/6 + \cdots$. Residue = coefficient of $1/z$ = $1$.

Alternatively by formula: $\operatorname{Res}_0 = \lim_{z \to 0} \frac{d}{dz}[z^2 \cdot e^z/z^2] = \lim \frac{d}{dz} e^z = \lim e^z = 1$. ✓

**Example 2: Residue at essential singularity.** $f(z) = z^2 e^{1/z}$ at $z = 0$.

$e^{1/z} = 1 + 1/z + 1/(2z^2) + 1/(6 z^3) + \cdots$, so
$$
z^2 e^{1/z} = z^2 + z + 1/2 + 1/(6z) + 1/(24 z^2) + \cdots
$$
Residue = coefficient of $1/z$ = $1/6$.

**Example 3: Sum of squares of reciprocals.** Evaluate $\sum_{n=1}^\infty 1/n^2$ using complex analysis.

Consider $f(z) = \pi \cot(\pi z)/z^2$. This has poles at $z = 0$ and $z = n$ for all nonzero integers $n$. The residue at $z = n$ ($n \neq 0$) is $\lim_{z \to n} (z - n) \pi \cot(\pi z)/z^2 = 1/n^2$ (since $\cot(\pi z) \sim 1/[\pi(z - n)]$ near $n$).

The residue at $z = 0$ requires Laurent expansion: $\cot(\pi z) = 1/(\pi z) - \pi z/3 - \pi^3 z^3/45 - \cdots$, so $\pi \cot(\pi z) = 1/z - \pi^2 z/3 - \cdots$, and $\pi \cot(\pi z)/z^2 = 1/z^3 - \pi^2/(3z) - \cdots$. Residue at $0$ = coefficient of $1/z$ = $-\pi^2/3$.

Integrating $f$ over $|z| = R$ with $R$ chosen between integers and applying ML (the integrand decays away from real axis), we conclude $\oint \to 0$ as $R \to \infty$, so sum of all residues = $0$:
$$
0 = -\frac{\pi^2}{3} + 2\sum_{n=1}^\infty \frac{1}{n^2} \implies \sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}.
$$
Euler's famous Basel problem result. The complex-analytic derivation is short and illuminating once residues are available.

### 0.5.4.11 Python: Residue Computation

```python
import numpy as np
import sympy as sp

# Symbolic residue computation with SymPy.
z = sp.symbols('z')

# Example 1: Residue of e^z / z^2 at z=0
f = sp.exp(z) / z**2
r = sp.residue(f, z, 0)
print(f"Res_0 [e^z/z^2] = {r}")  # should be 1

# Example 2: Residue of z^2 * exp(1/z) at z=0 (essential singularity)
f = z**2 * sp.exp(1/z)
r = sp.residue(f, z, 0)
print(f"Res_0 [z^2 e^(1/z)] = {r}")  # should be 1/6

# Example 3: Residue of 1/(z^2+1) at z=i
f = 1/(z**2 + 1)
r = sp.residue(f, z, sp.I)
print(f"Res_i [1/(z^2+1)] = {r}")  # should be 1/(2i) = -i/2

# Example 4: Residue of e^(iz) / (z^2+1) at z=i (for real integral)
f = sp.exp(sp.I*z) / (z**2 + 1)
r = sp.residue(f, z, sp.I)
print(f"Res_i [e^(iz)/(z^2+1)] = {r}")  # should be e^(-1)/(2i)
print(f"Real integral = {2*sp.pi*sp.I*r}")  # π/e

# Numerical verification: compute ∫_{-∞}^∞ 1/(x^2+1) dx via contour integration.
def numerical_real_integral(residue_val_at_upper_half):
    """∫_{-∞}^∞ f(x) dx = 2πi * (sum of residues in UHP), for nice decay."""
    return 2j * np.pi * residue_val_at_upper_half

# Direct numerical integration for comparison:
from scipy import integrate
val_scipy, _ = integrate.quad(lambda x: 1/(x**2 + 1), -np.inf, np.inf)
print(f"\nscipy.quad result: {val_scipy}")
print(f"Residue theorem:    {np.real(numerical_real_integral(1/(2j)))}")  # π

# Rouché check: does z^5 + 3z^2 + 1 have 5 roots inside |z|=2?
# On |z|=2, |z^5|=32, and |3z^2+1| ≤ 3*4+1 = 13 < 32. So by Rouché,
# z^5 + 3z^2 + 1 has same zeros as z^5 inside |z|=2: namely 5 zeros.
roots = np.roots([1, 0, 0, 3, 0, 1])
inside = [abs(r) < 2 for r in roots]
print(f"\nRoots of z^5 + 3z^2 + 1: {roots}")
print(f"All inside |z|=2: {all(inside)}  (count = {sum(inside)})  (Rouché predicts 5)")

# Numerical residue via contour integral: (1/2πi) ∮ f(z) dz over small circle.
def numerical_residue(f, z0, r=0.01, n=5000):
    ts = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/n
    ws = z0 + r * np.exp(1j*ts)
    dw = 1j * r * np.exp(1j*ts) * (2*np.pi/n)
    return np.sum(f(ws) * dw) / (2j * np.pi)

f = lambda z: np.exp(z) / z**2
print(f"\nNumerical Res_0 [e^z/z^2]: {numerical_residue(f, 0):.6f}  (expected 1)")

f = lambda z: z**2 * np.exp(1/z)
print(f"Numerical Res_0 [z^2 e^(1/z)]: {numerical_residue(f, 0, r=0.5):.6f}  (expected 1/6 ≈ 0.1667)")
```

### 0.5.4.12 [QUANT APPLICATION]

**1. Option pricing via contour integrals (detailed).** The Carr-Madan formula for a European call is
$$
C(K) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-ivk} \psi_T(v) \, dv, \quad \psi_T(v) = \frac{e^{-rT} \phi_T(v - (\alpha+1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v},
$$
where $k = \log K$, $\phi_T$ is the characteristic function of log-price, and $\alpha$ is a damping parameter. The justification that $\phi_T(v - (\alpha+1)i)$ is defined for the complex argument $v - (\alpha+1)i$ requires that the strip $\{\operatorname{Im} \zeta \in [-\alpha - 1, \dots]\}$ lies in the domain of analyticity of the characteristic function. Residues occurring at singularities of $\psi_T$ in this strip determine when we need to deform the contour. For the Heston model in particular, the "little trap" vs "big trap" formulation of the characteristic function corresponds to branch choices in a complex logarithm — a Laurent/branch-cut issue.

**2. Evaluating transform inversions via residues.** The inverse Laplace transform of the bond price in an affine model is
$$
P(0, T) = \frac{1}{2\pi i} \int_{c - i\infty}^{c + i\infty} \tilde P(s) e^{sT} ds,
$$
evaluated by closing the contour in the left half plane and summing residues at the poles of $\tilde P$ (provided they are simple). Residue calculus gives closed-form expressions for option prices in many models.

**3. Local times and local limit theorems via residue theorem.** For a random walk $S_n = X_1 + \cdots + X_n$ with $\mathbb{E}[X] = 0$ and bounded densities, the local central limit theorem is proven by inverting the characteristic function $\phi(t)^n / (2\pi)$ and identifying the leading asymptotic via contour deformation. Residues at dominant poles govern the Edgeworth corrections.

**4. Spectral analysis of Markov chains.** A Markov chain's generator $L$ has spectrum in $\mathbb{C}$; the projectors onto eigenspaces are residues of the resolvent $(L - z)^{-1}$ at eigenvalues. Functional calculus (e.g., $e^{tL}$) is computed by contour integrals over paths encircling the spectrum. This technique is standard in operator-theoretic approaches to option pricing under Markov models.

**5. Density estimation from characteristic functions (Gil-Pelaez).** The inverse formula
$$
F(x) = \frac{1}{2} - \frac{1}{\pi} \int_0^\infty \operatorname{Im}\left[\frac{e^{-ivx}\phi(v)}{v}\right] dv
$$
has an apparent singularity at $v = 0$, handled by contour deformation (take a small semicircle above $0$; pick up half the residue of $\phi(v) e^{-ivx}/v$ at $0$, which is $\phi(0)/2 = 1/2$). This is why the "$1/2$" appears in the Gil-Pelaez formula. Directly: residue calculus is hiding in the inversion formulas for all transforms in finance.

**6. Root-counting in calibration.** Rouché's theorem has practical use in verifying that certain equations (e.g., implied-volatility equations, yield-curve bootstrapping equations) have the correct number of solutions in a given region. Before running Newton's method, one can apply Rouché to verify there is exactly one solution and bracket it — useful for guaranteeing convergence to the intended root.

### 0.5.4.13 Exercises

#### ★ (Warm-up)

**E0.5.4.1.** Find the Laurent series of $1/(z^2 - 1)$ in the annulus $\{1 < |z| < \infty\}$.

**E0.5.4.2.** Find the Laurent series of $e^z/z^3$ centered at $z = 0$. What is the residue at $0$?

**E0.5.4.3.** Classify the singularity at $z = 0$ of: (a) $\sin z / z$; (b) $\sin z / z^2$; (c) $\sin z / z^3$; (d) $\sin(1/z)$; (e) $\cos(z)/z$.

**E0.5.4.4.** Compute the residues at all finite singularities of $f(z) = \frac{1}{z^2(z - 1)}$.

**E0.5.4.5.** Use the residue theorem to evaluate $\oint_{|z|=2} \frac{z}{(z-1)(z+1)} dz$.

**E0.5.4.6.** Verify by direct Laurent expansion that $\operatorname{Res}_0 [1/\sin z] = 1$.

#### ★★ (Standard)

**E0.5.4.7.** Evaluate $\int_{-\infty}^\infty \frac{x^2}{(x^2 + 1)(x^2 + 4)} dx$ by closing contour in the upper half plane.

**E0.5.4.8 (Fourier integrals).** Evaluate $\int_{-\infty}^\infty \frac{\cos(ax)}{x^2 + b^2} dx$ for $a > 0, b > 0$. Check the extreme cases $a = 0$ and $a \to \infty$.

**E0.5.4.9 (Semicircle with indentation).** Evaluate $\int_0^\infty \frac{\sin x}{x} dx = \pi/2$ by closing the contour for $e^{iz}/z$ and indenting around $z = 0$. Explain why the indentation contributes half the residue.

**E0.5.4.10 (Rectangular contour).** Evaluate $\int_{-\infty}^\infty \frac{dx}{\cosh x}$ by integrating $1/\cosh z$ over a rectangle with vertices $\pm R, \pm R + i\pi$. The top segment relates to the bottom via $\cosh(z + i\pi) = -\cosh z$.

**E0.5.4.11 (Keyhole).** Evaluate $\int_0^\infty \frac{x^{s-1}}{1 + x} dx = \pi/\sin(\pi s)$ for $0 < s < 1$ by using a keyhole contour around the positive real axis. The branch cut of $z^{s-1}$ contributes the key.

**E0.5.4.12 (Essential singularity gymnastics).** Compute $\operatorname{Res}_0 [e^{1/z} \sin z]$ by expanding both factors and extracting the $1/z$ coefficient.

**E0.5.4.13 (Rouché).** Show that $z^4 + 8z + 12$ has all four roots in the annulus $\{1 < |z| < 3\}$. (Use Rouché twice: dominant term $z^4$ on $|z| = 3$, dominant term $12$ on $|z| = 1$.)

**E0.5.4.14 (Number of zeros via argument principle).** Use the argument principle to count zeros of $p(z) = z^6 + 5z^3 + 1$ in the right half plane.

#### ★★★ (Challenge)

**E0.5.4.15 (Infinite product for sine).** Prove Euler's product: $\sin(\pi z) = \pi z \prod_{n=1}^\infty \left(1 - z^2/n^2\right)$. (Define $f(z) = \sin(\pi z)/[\pi z \prod (1 - z^2/n^2)]$; show it is entire, nonzero, bounded, hence constant by Liouville + growth; evaluate at $z = 0$.)

**E0.5.4.16 (Basel $1/n^2 = \pi^2/6$, rigorous).** Fill in the details of the residue-theorem calculation in Example 3: carefully bound $|\pi \cot(\pi z)|$ on the square contour $|x| = N + 1/2, |y| \leq N + 1/2$ uniformly in $N$, show the integrals $\oint$ go to zero, and conclude $\sum 1/n^2 = \pi^2/6$. Extend to $\sum 1/n^4, \sum 1/n^{2k}$.

**E0.5.4.17 (Gauss sums).** Use contour integration and Jacobi theta functions to evaluate the quadratic Gauss sum $\sum_{k=0}^{n-1} e^{2\pi i k^2/n}$. (Advanced number theory application.)

**E0.5.4.18 (Picard's little theorem).** State and prove (or read the proof of) Picard's little theorem: a nonconstant entire function omits at most one value of $\mathbb{C}$. (Proof uses the modular function; well beyond this course, but do the statement.)

**E0.5.4.19 (Prime number theorem via Riemann zeta).** Sketch why $\zeta(s) \neq 0$ on $\operatorname{Re} s = 1$ implies the prime number theorem $\pi(x) \sim x/\ln x$ via a contour integral (Perron's formula) involving $-\zeta'(s)/\zeta(s)$. (Famous — read Edwards, *Riemann's Zeta Function*, for the full treatment.)

**E0.5.4.20 (Weyl equidistribution).** Use contour integration to prove Weyl's theorem: if $\alpha$ is irrational, the sequence $\{n\alpha \mod 1\}_{n \geq 1}$ is equidistributed in $[0, 1]$. (Reduce to showing $\sum_{n=1}^N e^{2\pi i k n \alpha} = o(N)$ for each nonzero integer $k$, via geometric-series summation and bounds on $|1 - e^{2\pi i k \alpha}|$; interpret the bounds complex-analytically.)

---

## Topic 0.5.5 — Analytic Continuation and Special Functions

### 0.5.5.1 The Phenomenon of Analytic Continuation

Analytic continuation is the complex-analytic analogue of saying "there is essentially one way to extend a function." The identity principle (Theorem 0.5.3.12) says a holomorphic function on a connected domain is determined by its germ at any single point. Analytic continuation asks the converse question: given a function defined on a smaller domain, can we extend it to a larger one, and in how many ways?

The answer is remarkable: if an extension exists, it is essentially unique (up to the topology of the domain); but **whether** an extension exists depends sensitively on the global geometry — and the resulting "extended function" may be **multi-valued**, forcing the introduction of **Riemann surfaces**.

### 0.5.5.2 Direct Analytic Continuation

**Definition 0.5.5.1 (Direct continuation).** Let $f: \Omega \to \mathbb{C}$ be holomorphic. A **direct analytic continuation** of $f$ is a pair $(g, \Omega')$ where $\Omega' \supseteq \Omega$ is open connected and $g: \Omega' \to \mathbb{C}$ is holomorphic with $g|_\Omega = f$.

**Theorem 0.5.5.2 (Uniqueness of continuation).** If $(g_1, \Omega')$ and $(g_2, \Omega')$ are two direct analytic continuations of $f: \Omega \to \mathbb{C}$ to the same connected domain $\Omega'$, then $g_1 = g_2$.

*Proof.* $g_1 - g_2$ is holomorphic on $\Omega'$ and vanishes on $\Omega$ (nonempty open subset). By the identity principle, $g_1 - g_2 \equiv 0$ on $\Omega'$. $\blacksquare$

### 0.5.5.3 Continuation Along a Path

The interesting case arises when there is no single domain $\Omega'$ to which $f$ extends; instead, $f$ can be continued **along different paths**, potentially giving different answers.

**Definition 0.5.5.3 (Analytic continuation along a path).** Let $\gamma: [0, 1] \to \mathbb{C}$ be a continuous path with $\gamma(0) \in \Omega$. An **analytic continuation of $f$ along $\gamma$** is a family of pairs $\{(f_t, D_t)\}_{t \in [0, 1]}$ where:

- Each $D_t$ is an open disk containing $\gamma(t)$.
- $f_t: D_t \to \mathbb{C}$ is holomorphic.
- $f_0 = f|_{D_0}$.
- For each $t_0 \in [0, 1]$, there exists a neighborhood $(t_0 - \epsilon, t_0 + \epsilon)$ of $t_0$ such that for $t$ in this neighborhood, $D_t \cap D_{t_0}$ is nonempty and $f_t = f_{t_0}$ on the intersection.

**Theorem 0.5.5.4 (Uniqueness along a path).** If continuations $\{f_t\}$ and $\{\tilde f_t\}$ of $f$ along the same path $\gamma$ both exist, then $f_1 = \tilde f_1$ on a neighborhood of $\gamma(1)$.

*Proof.* The set $T := \{t \in [0, 1] : f_t = \tilde f_t \text{ on } D_t \cap \tilde D_t\}$ is nonempty ($0 \in T$ by hypothesis), open (local consistency), and closed (identity principle on disks). By connectedness of $[0, 1]$, $T = [0, 1]$. $\blacksquare$

**Theorem 0.5.5.5 (Monodromy theorem).** Let $\Omega$ be simply connected, $f_0: D_0 \to \mathbb{C}$ holomorphic on an open disk $D_0 \subset \Omega$, and suppose $f_0$ can be analytically continued along every path in $\Omega$ starting at $\gamma(0) \in D_0$. Then the continuations are path-independent (same value at the endpoint regardless of path), and therefore yield a global holomorphic extension $\tilde f: \Omega \to \mathbb{C}$.

*Proof sketch.* Suppose $\gamma_0, \gamma_1: [0,1] \to \Omega$ are two paths from $\gamma(0)$ to a common endpoint $z$, homotopic via $H: [0,1]^2 \to \Omega$ (available because $\Omega$ is simply connected). By a Lebesgue-number argument, subdivide $[0,1]^2$ into fine squares each of whose image under $H$ lies in a disk contained in $\Omega$. On each small square, apply the uniqueness theorem 0.5.5.4 iteratively to conclude that the value at the endpoint of $\gamma_0$ equals the value at the endpoint of $\gamma_1$. The extension $\tilde f(z) := f_t(z)$ for any path ending at $z$ is then well-defined, and holomorphy on each $D_t$ stitches together into global holomorphy. $\blacksquare$

### 0.5.5.4 Multi-valuedness and Branch Points

The monodromy theorem fails without simple connectivity. The prototypical example is $\log z$ on $\mathbb{C} \setminus \{0\}$:

**Example.** Start with $\log z$ defined near $z = 1$ with $\log 1 = 0$. Continue along the unit circle counter-clockwise: at each step, $\log z$ increases by the arc length. After one full revolution, $\log 1$ has become $2\pi i$, not $0$. The continuation depends on the path.

On the *double cover* of $\mathbb{C} \setminus \{0\}$ (equivalently, the universal cover, which is a Riemann surface), $\log z$ becomes single-valued. The two sheets correspond to the two determinations $\log z$ and $\log z + 2\pi i$; all branches differ by integer multiples of $2\pi i$.

**Definition 0.5.5.6 (Branch point).** A point $z_0$ is a **branch point** of a multi-valued analytic function $f$ if continuing $f$ along a small loop around $z_0$ changes its value. Examples: $z = 0$ is a branch point of $\log z$ and of $z^{1/n}$ for $n \geq 2$.

**Principal branch.** For $\log z$, the principal branch is $\operatorname{Log} z := \log|z| + i \arg z$ with $\arg z \in (-\pi, \pi]$. Domain: $\mathbb{C} \setminus (-\infty, 0]$, i.e., the slit plane (avoiding the branch cut along the negative real axis).

### 0.5.5.5 The Schwarz Reflection Principle

A concrete tool for extending analytic functions across lines or arcs, used extensively in constructive continuation:

**Theorem 0.5.5.7 (Schwarz reflection).** Let $\Omega^+ \subset \mathbb{H}$ (upper half plane) be an open set symmetric under reflection such that $\Omega^+ \cap \mathbb{R}$ is a nonempty open interval $I$. Let $\Omega^-$ be the reflection of $\Omega^+$, and $\Omega := \Omega^+ \cup I \cup \Omega^-$. If $f: \Omega^+ \to \mathbb{C}$ is holomorphic, extends continuously to $\Omega^+ \cup I$, and takes *real* values on $I$, then $f$ extends holomorphically to all of $\Omega$ via
$$
\tilde f(z) := \begin{cases} f(z), & z \in \Omega^+ \cup I, \\ \overline{f(\bar z)}, & z \in \Omega^-. \end{cases}
$$

*Proof.* $\tilde f$ is continuous on $\Omega$ (continuity on $\Omega^+$ and $\Omega^-$ is clear; on $I$, $\overline{f(\bar x)} = \overline{f(x)} = f(x)$ since $f$ is real on $I$). $\tilde f$ is holomorphic on $\Omega^+$ by hypothesis and on $\Omega^-$ because $z \mapsto \bar z$ is anti-holomorphic, $f$ is holomorphic, and $z \mapsto \bar z$ (the conjugation) inverts anti-holomorphy — so $\overline{f(\bar z)}$ is holomorphic in $z$.

To verify holomorphy across $I$: use Morera's theorem. Triangle integrals over $T \subset \Omega$ with $T \cap I = \emptyset$ vanish by Cauchy. For triangles $T$ crossing $I$, subdivide into two parts by the intersection with $I$; each part has its base on $I$ and integrals can be taken by limiting from above/below, giving zero total. Hence $\int_{\partial T} \tilde f = 0$ for all triangles, and Morera implies $\tilde f$ holomorphic on $\Omega$. $\blacksquare$

**Variant (arc reflection).** Replace the real axis with the unit circle: if $f$ is holomorphic inside the disk, continuous on the closure, and $|f| \equiv 1$ on the boundary, $f$ extends to $\mathbb{C} \setminus \{0\}$ via $\tilde f(z) = 1/\overline{f(1/\bar z)}$. This is the engine behind Blaschke products.

### 0.5.5.6 The Gamma Function

Perhaps the most celebrated analytic continuation in classical mathematics:

**Definition 0.5.5.8 (Gamma function).** For $\operatorname{Re} s > 0$, define
$$
\Gamma(s) := \int_0^\infty t^{s-1} e^{-t} dt.
$$
The integral converges absolutely and uniformly on compact subsets of $\{\operatorname{Re} s > 0\}$, so $\Gamma$ is holomorphic there.

**Basic properties.** Integration by parts gives $\Gamma(s+1) = s \Gamma(s)$, hence $\Gamma(n+1) = n!$ for $n \geq 0$.

**Analytic continuation.** The functional equation $\Gamma(s) = \Gamma(s+1)/s$ extends the definition to $\{\operatorname{Re} s > -1\}$ (with a simple pole at $s = 0$ of residue $1$). Iterating:
$$
\Gamma(s) = \frac{\Gamma(s + k)}{s(s+1)(s+2)\cdots(s+k-1)}, \qquad \operatorname{Re} s > -k,
$$
extends $\Gamma$ to $\{\operatorname{Re} s > -k\}$ for any $k$, and thus to all of $\mathbb{C} \setminus \{0, -1, -2, \ldots\}$. The continuation has simple poles at non-positive integers: at $s = -n$, residue
$$
\operatorname{Res}_{s = -n} \Gamma(s) = \lim_{s \to -n} (s + n) \Gamma(s) = \lim_{s \to -n} \frac{\Gamma(s + n + 1)}{s(s+1)\cdots(s+n-1)} = \frac{\Gamma(1)}{(-n)(-n+1)\cdots(-1)} = \frac{(-1)^n}{n!}.
$$

**Theorem 0.5.5.9 (Properties of $\Gamma$).**

(a) $\Gamma$ is meromorphic on $\mathbb{C}$ with simple poles at $0, -1, -2, \ldots$ and no zeros.

(b) **Reflection formula:** $\Gamma(s) \Gamma(1 - s) = \pi/\sin(\pi s)$, valid on $\mathbb{C} \setminus \mathbb{Z}$.

(c) **Duplication formula:** $\Gamma(s) \Gamma(s + 1/2) = 2^{1 - 2s} \sqrt{\pi} \, \Gamma(2s)$.

(d) **Weierstrass product:** $1/\Gamma(s) = s \, e^{\gamma s} \prod_{n=1}^\infty (1 + s/n) e^{-s/n}$, where $\gamma$ is the Euler-Mascheroni constant.

(e) **Stirling's formula:** $\log \Gamma(s) = (s - 1/2) \log s - s + (1/2) \log(2\pi) + O(1/|s|)$ uniformly in any sector $|\arg s| \leq \pi - \delta$.

*Proof of (b) (sketch).* Consider $F(s) := \Gamma(s) \Gamma(1 - s) \sin(\pi s)/\pi$. Using the product formula and manipulating the series of $\Gamma(1 - s)$, one shows $F(s) = 1$ for $\operatorname{Re} s \in (0, 1)$. By analytic continuation, $F \equiv 1$ on $\mathbb{C} \setminus \mathbb{Z}$.

Alternative proof via contour integration: integrate $x^{s-1}/(1 + x)$ over a keyhole contour around $[0, \infty)$. The two sides of the branch cut give $\int_0^\infty x^{s-1}/(1+x) \, dx - e^{2\pi i (s-1)} \int_0^\infty x^{s-1}/(1+x) \, dx = (1 - e^{2\pi i s}) \int_0^\infty x^{s-1}/(1+x) dx = -2i \sin(\pi s) e^{i\pi s} \cdot [\text{integral}]$. Set this equal to $2\pi i \operatorname{Res}_{x = -1} x^{s-1}/(1+x) = 2\pi i \cdot (-1)^{s-1} = 2\pi i e^{i\pi(s-1)} = -2\pi i e^{i\pi s}$. Matching: $\int_0^\infty x^{s-1}/(1+x) dx = \pi/\sin(\pi s)$. Then using $\int x^{s-1}/(1 + x) dx = \Gamma(s) \Gamma(1 - s)$ (beta function identity), we conclude. $\blacksquare$

### 0.5.5.7 The Riemann Zeta Function

**Definition 0.5.5.10 (Zeta function).** For $\operatorname{Re} s > 1$, define
$$
\zeta(s) := \sum_{n=1}^\infty \frac{1}{n^s}.
$$

The series converges absolutely and uniformly on $\{\operatorname{Re} s \geq 1 + \epsilon\}$, so $\zeta$ is holomorphic on $\{\operatorname{Re} s > 1\}$.

**Euler product formula:**
$$
\zeta(s) = \prod_p (1 - p^{-s})^{-1}, \qquad \operatorname{Re} s > 1,
$$
where the product is over primes. This connects $\zeta$ to number theory and is the starting point of analytic number theory.

**Analytic continuation.** Riemann showed that $\zeta$ extends meromorphically to all of $\mathbb{C}$, with a single simple pole at $s = 1$ of residue $1$. The continuation uses the integral representation
$$
\zeta(s) = \frac{1}{\Gamma(s)} \int_0^\infty \frac{x^{s-1}}{e^x - 1} dx, \qquad \operatorname{Re} s > 1,
$$
combined with a contour deformation (the "Hankel contour") to extend to the left half plane.

**Functional equation.** Riemann's famous identity:
$$
\zeta(s) = 2^s \pi^{s-1} \sin(\pi s/2) \Gamma(1 - s) \zeta(1 - s).
$$
This relates values in $\{\operatorname{Re} s > 1\}$ to values in $\{\operatorname{Re} s < 0\}$. The **completed zeta** $\xi(s) := \pi^{-s/2} \Gamma(s/2) \zeta(s)$ satisfies $\xi(s) = \xi(1 - s)$ — a remarkable symmetry around the "critical line" $\operatorname{Re} s = 1/2$.

**Trivial zeros:** $\zeta(-2k) = 0$ for $k = 1, 2, 3, \ldots$ (from the $\sin(\pi s/2)$ factor).

**Nontrivial zeros:** the Riemann hypothesis conjectures they all lie on the critical line $\operatorname{Re} s = 1/2$. This is one of the most famous open problems in mathematics.

**Prime number theorem.** Hadamard and de la Vallée Poussin proved in 1896 that $\zeta(s) \neq 0$ on $\operatorname{Re} s = 1$, from which the prime number theorem $\pi(x) \sim x/\ln x$ follows via Perron's formula and residue calculus.

### 0.5.5.8 Special Values of Zeta

**Euler's values at even integers.** The basel problem generalization:
$$
\zeta(2n) = \frac{(-1)^{n+1} B_{2n} (2\pi)^{2n}}{2 (2n)!},
$$
where $B_{2n}$ are Bernoulli numbers. So $\zeta(2) = \pi^2/6$, $\zeta(4) = \pi^4/90$, etc. All are rational multiples of $\pi^{2n}$, in particular *irrational*.

**Odd integer values.** Remarkably, no closed form is known for $\zeta(3), \zeta(5), \zeta(7), \ldots$. Apéry (1979) proved $\zeta(3)$ is irrational — a celebrated result. Irrationality of $\zeta(5), \zeta(7), \ldots$ is open (though it is known that infinitely many of them are irrational).

**Values at negative integers.** From the functional equation: $\zeta(-n) = -B_{n+1}/(n+1)$ for $n \geq 0$. Hence $\zeta(0) = -1/2$, $\zeta(-1) = -1/12$, $\zeta(-3) = 1/120$, etc.

**The "sum $1 + 2 + 3 + \cdots = -1/12$" meme.** This is $\zeta(-1)$ via analytic continuation — the sum of all positive integers *does not* converge, but the analytic continuation of $\zeta$ to $s = -1$ gives $-1/12$. This is rigorously the "regularized sum" via zeta regularization, used in string theory and quantum field theory.

### 0.5.5.9 Natural Boundaries

Not every function continues far. Some power series have a **natural boundary**: a circle beyond which no continuation exists.

**Example (Hadamard gap).** Consider $f(z) := \sum_{n=0}^\infty z^{2^n} = z + z^2 + z^4 + z^8 + \cdots$. The radius of convergence is $1$. One can show $|z| = 1$ is a natural boundary: $f$ does not extend past any point on the unit circle. Such series are called **lacunary**.

**Proof idea.** If $f$ extended past some point $z_0$ on the unit circle, we'd have $f$ holomorphic in a neighborhood $U$ of $z_0$. Using the self-similarity $f(z) = z + z^2 + f(z^2) - (z^2 + z^4)$ (which is a rearrangement leading to $f(z) - z - z^2 = f(z^2)$), we could boostrap continuations to all points on the circle at distance equal to dyadic rationals from $z_0$, and thence to a dense set — but this contradicts the divergence of the power series at any such point.

### 0.5.5.10 Python: Gamma and Zeta

```python
import numpy as np
from scipy import special

# Gamma function — the SciPy implementation uses analytic continuation.
for s in [0.5, 1, 2, -0.5, -1.5, 3.14, 1 + 2j]:
    print(f"Γ({s}) = {special.gamma(s)}")

# Reflection formula: Γ(s) Γ(1-s) = π/sin(πs)
s = 0.3
lhs = special.gamma(s) * special.gamma(1 - s)
rhs = np.pi / np.sin(np.pi * s)
print(f"\nReflection: Γ({s})Γ({1-s}) = {lhs:.6f}, π/sin(π·{s}) = {rhs:.6f}")

# Riemann zeta via scipy (uses functional equation for Re s < 1/2)
from scipy.special import zeta
print(f"\nζ(2) = {zeta(2)}  (should be π²/6 = {np.pi**2/6:.6f})")
print(f"ζ(4) = {zeta(4)}  (should be π⁴/90 = {np.pi**4/90:.6f})")
print(f"ζ(-1) = {zeta(-1, 1)}  (should be -1/12 = {-1/12:.6f})")

# Verify ζ(s) computed by Dirichlet series for Re s > 1
def zeta_dirichlet(s, n_terms=10000):
    return sum(1/(k**s) for k in range(1, n_terms + 1))

for s in [2, 3, 5, 10]:
    print(f"Dirichlet ζ({s}) ≈ {zeta_dirichlet(s):.6f} vs scipy = {zeta(s):.6f}")

# Plot the Riemann zeta function along the critical line
import matplotlib.pyplot as plt
ts = np.linspace(0, 40, 1000)
z_vals = [complex(special.zeta(0.5 + 1j*t)) for t in ts]
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ts, [v.real for v in z_vals], label='Re ζ(1/2 + it)')
ax.plot(ts, [v.imag for v in z_vals], label='Im ζ(1/2 + it)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('t'); ax.set_ylabel('ζ(1/2 + it)')
ax.set_title('Riemann zeta on the critical line (zeros = nontrivial zeros)')
ax.legend()
# Nontrivial zeros occur where both real and imaginary parts are zero.
# First few: t ≈ 14.13, 21.02, 25.01, 30.42, 32.93, 37.59

# Analytic continuation of log via path integration:
def log_continued(z, z0=1, z0_log=0, n=1000):
    """Continue log analytically from z0 (where log z0 = z0_log) to z
    along a straight line."""
    ts = np.linspace(0, 1, n)
    path = z0 + ts * (z - z0)
    # dLog = dz/z along the path
    dz = np.diff(path)
    z_mid = (path[:-1] + path[1:]) / 2
    return z0_log + np.sum(dz / z_mid)

# Continue log along the upper semicircle from 1 to -1.
# Principal log(-1) = iπ. Continue along upper half should give iπ.
z_target = -1 + 0.001j  # slightly above real axis
print(f"\nlog(-1 + 0i+) via upper semicircle ≈ {log_continued(z_target)}")
# Continuing below would give -iπ instead.
z_target = -1 - 0.001j
print(f"log(-1 + 0i-) via lower semicircle ≈ {log_continued(z_target)}")
```

### 0.5.5.11 [QUANT APPLICATION]

**1. Mellin transforms in option pricing.** The Mellin transform
$$
\hat V(s) := \int_0^\infty V(K) K^{s-1} dK
$$
is the complex-analytic version of the Laplace transform in log-variables. In option pricing, Mellin transforms are used for American option valuation (via integral equations) and for characteristic-function-based methods. The inverse Mellin transform requires a Bromwich contour in the complex plane, requiring understanding of the strip of analyticity — a classical analytic continuation setup.

**2. Gamma function in option pricing.** The Greeks formulas for option sensitivities often involve factors of $\Gamma$; explicit expressions for Asian option prices, for example, involve ratios of $\Gamma$ functions in the complex plane. The reflection formula is used to simplify such expressions.

**3. Zeta regularization in statistical physics and quantitative finance.** The regularized sum $\zeta(0) = -1/2$ is used to compute the partition function of certain random-walk and Brownian-motion problems; for example, the "Euler characteristic" of the Brownian path, and certain large-deviation corrections in queueing theory, involve $\zeta(0), \zeta(-1), \zeta(-2), \ldots$ via zeta regularization.

**4. Analytic continuation for Levy processes.** A Lévy process has a characteristic function $\phi_t(u) = e^{t \psi(u)}$ where $\psi$ is the Lévy symbol. The strip of analyticity of $\psi$ in the complex plane determines where expectations $\mathbb{E}[e^{uX_t}]$ exist, which is crucial for pricing options on exponential Lévy processes. Carr-Madan-style methods rely on contour deformations inside this strip.

**5. Branch cuts and tail probabilities.** The Laplace transform of a distribution that is heavy-tailed can have a branch cut on the negative real axis (e.g., for stable distributions). The discontinuity across the branch cut equals $2i \operatorname{Im}(\text{transform}) \cdot \sin(\ldots)$, and is directly related to the tail density by an "inverse Titchmarsh formula." Understanding where these branch cuts lie is essential for computing accurate tail probabilities.

**6. Multi-asset options and product representations.** For basket options, the joint characteristic function may have a product representation
$$
\phi_T(u_1, \ldots, u_d) = \prod_k \phi^{(k)}_T(u_k)
$$
in some models, extending to complex arguments via analytic continuation. The joint strip of analyticity constrains which Fourier methods are valid, and in what domain.

### 0.5.5.12 Exercises

#### ★ (Warm-up)

**E0.5.5.1.** Starting with $f(z) = \sum z^n$ on $|z| < 1$, show $f = 1/(1-z)$ extends meromorphically to $\mathbb{C}$ with a single simple pole at $z = 1$.

**E0.5.5.2.** Let $f(z) = \sum z^{n!}/n!$. Show $f$ is entire.

**E0.5.5.3.** Compute $\Gamma(1/2) = \sqrt\pi$ directly from the integral definition $\int_0^\infty t^{-1/2} e^{-t} dt$ (substitute $t = u^2$).

**E0.5.5.4.** Compute $\Gamma(-1/2)$ using the functional equation. Check: $\Gamma(-1/2) = -2\sqrt\pi$.

**E0.5.5.5.** Verify numerically or via Dirichlet series that $\zeta(4) = \pi^4/90$.

#### ★★ (Standard)

**E0.5.5.6 (Continuation of $\log z$).** Define $\log$ as $\int_1^z dw/w$ along a path from $1$ to $z$ avoiding $0$. Show that two paths differing by one loop around $0$ (counter-clockwise) produce values differing by $2\pi i$, rigorously via Cauchy's theorem applied in $\mathbb{C} \setminus \{0\}$.

**E0.5.5.7 (Bohr-Mollerup theorem).** Prove that if $F: (0, \infty) \to (0, \infty)$ satisfies $F(1) = 1$, $F(x + 1) = x F(x)$, and $\log F$ is convex, then $F = \Gamma$. (Hint: show $\log F(x) - \log \Gamma(x)$ is periodic and convex, hence constant.)

**E0.5.5.8 (Stirling).** Prove Stirling's formula $\Gamma(s) \sim \sqrt{2\pi/s} (s/e)^s$ via saddle-point analysis of the integral $\Gamma(s) = \int_0^\infty t^{s-1} e^{-t} dt$ (substitute $t = su$, find the maximum of $s \log u - u + \text{const}$ — it's at $u = 1$).

**E0.5.5.9.** Using the reflection formula, evaluate $\Gamma(1/3) \Gamma(2/3) = 2\pi/\sqrt 3$.

**E0.5.5.10.** Show that $\zeta$ has no zeros in $\operatorname{Re} s > 1$ using the Euler product.

**E0.5.5.11.** Derive $\zeta(0) = -1/2$ using the functional equation and $\Gamma(1) = 1$. Be careful with the $\sin(\pi s/2)$ factor; use l'Hôpital or a series expansion.

**E0.5.5.12 (Beta function).** Define $B(p, q) := \int_0^1 t^{p-1}(1-t)^{q-1} dt$. Prove $B(p, q) = \Gamma(p) \Gamma(q)/\Gamma(p + q)$ by evaluating $\Gamma(p) \Gamma(q) = \int_0^\infty \int_0^\infty t^{p-1} s^{q-1} e^{-(t+s)} dt \, ds$ and substituting $t = rx, s = r(1 - x)$.

**E0.5.5.13 (Natural boundary).** Prove rigorously that $f(z) = \sum_{n=0}^\infty z^{2^n}$ has $|z| = 1$ as a natural boundary. (Show that for each $z_0$ on the unit circle, $\limsup_{z \to z_0, |z| < 1} |f(z)| = \infty$.)

#### ★★★ (Challenge)

**E0.5.5.14 (Prime number theorem — outline).** Sketch the proof that $\pi(x) \sim x/\ln x$ via the following steps: (i) express $\psi(x) = \sum_{p^k \leq x} \log p$ as a contour integral $\psi(x) = (2\pi i)^{-1} \int_{c - i\infty}^{c + i\infty} -\zeta'(s)/\zeta(s) \cdot x^s/s \, ds$; (ii) shift the contour to $\operatorname{Re} s = 1$, picking up the residue from the pole of $-\zeta'/\zeta$ at $s = 1$, which gives the main term $x$; (iii) the hypothesis $\zeta \neq 0$ on $\operatorname{Re} s = 1$ shows no other contribution arises near this line, and so $\psi(x) \sim x$, implying PNT.

**E0.5.5.15 (Riemann hypothesis implications).** State the Riemann hypothesis and explain why it is equivalent to the sharper bound $|\pi(x) - \operatorname{li}(x)| = O(\sqrt x \log x)$ on the prime counting function. Why does the location of zeros of $\zeta$ matter for the distribution of primes?

**E0.5.5.16 (Theta function).** Define $\theta(z) := \sum_{n \in \mathbb{Z}} e^{i\pi n^2 z}$ for $\operatorname{Im} z > 0$. Using the Poisson summation formula, prove the modular transformation $\theta(-1/z) = \sqrt{z/i} \, \theta(z)$. This is the key ingredient in Riemann's derivation of the functional equation for $\zeta$.

**E0.5.5.17 (Analytic continuation along a fundamental group).** Let $\Omega = \mathbb{C} \setminus \{0, 1\}$, whose fundamental group is the free group on two generators. Describe the monodromy representation of $\log(z(z-1))$ viewed as a multi-valued function on $\Omega$: show it extends to a homomorphism $\pi_1(\Omega) \to \mathbb{Z}^2$ via winding numbers around $0$ and $1$.

**E0.5.5.18 (Hadamard three-circle theorem).** Let $f$ be holomorphic on the annulus $\{r_1 < |z| < r_2\}$ and continuous on the closure. Let $M(r) := \max_{|z| = r} |f(z)|$. Prove that $\log M(r)$ is a convex function of $\log r$. (Apply the maximum modulus principle to $z^\lambda f(z)$ for appropriate $\lambda$.)

**E0.5.5.19 (Jensen's inequality on the boundary).** Let $f$ be holomorphic on the closed disk with $f(0) \neq 0$ and having zeros $a_1, \ldots, a_n$ in the open disk. Prove:
$$
\log|f(0)| \leq \frac{1}{2\pi} \int_0^{2\pi} \log|f(e^{i\theta})| d\theta,
$$
with equality iff $f$ has no zeros in the disk. (Combine Jensen's formula from E0.5.3.18 with the observation that $\sum \log(1/|a_k|) \geq 0$.)

**E0.5.5.20 (Zeta at critical strip).** Investigate numerically the location of the first few zeros of $\zeta(s)$ on the critical line $\operatorname{Re} s = 1/2$. (The first zero is at $t \approx 14.134725$. Use scipy or mpmath.)

---

## Topic 0.5.6 — Fourier Transforms and Characteristic Functions

### 0.5.6.1 Why Complex Analysis for Fourier?

The Fourier transform is real-valued if you stay on the real line — but virtually every interesting property of the Fourier transform is a complex-analytic statement about its analytic extension. The characteristic function $\phi_X(t) = \mathbb{E}[e^{itX}]$ of a random variable extends (when moments exist) to a holomorphic function on a strip, and its analytic-continuation structure encodes tail behavior, moment existence, and density-smoothness in a unified framework. This topic synthesizes the machinery from the preceding topics for quant applications: contour deformation (0.5.2), residue calculus (0.5.4), and analytic continuation (0.5.5), applied to random variables.

### 0.5.6.2 The Fourier Transform

**Definition 0.5.6.1.** For $f \in L^1(\mathbb{R})$ (i.e., $\int |f(x)| dx < \infty$), the **Fourier transform** is
$$
\hat f(\xi) := \int_{-\infty}^\infty e^{-2\pi i \xi x} f(x) \, dx, \qquad \xi \in \mathbb{R}.
$$
(Many conventions exist; in probability, the characteristic function uses $e^{it x}$ — this is another convention.)

**Basic properties (real line).**

1. Linearity.
2. $\widehat{f(\cdot - a)}(\xi) = e^{-2\pi i a \xi} \hat f(\xi)$ (translation in $x$ ↔ modulation in $\xi$).
3. $\widehat{e^{2\pi i a x} f(x)}(\xi) = \hat f(\xi - a)$ (modulation in $x$ ↔ translation in $\xi$).
4. $\widehat{f'}(\xi) = 2\pi i \xi \hat f(\xi)$ (derivative in $x$ ↔ multiplication in $\xi$).
5. $\widehat{x f(x)}(\xi) = \frac{1}{-2\pi i} \hat f'(\xi)$ (multiplication in $x$ ↔ derivative in $\xi$).
6. $\widehat{f \star g}(\xi) = \hat f(\xi) \hat g(\xi)$ (convolution theorem).
7. **Parseval/Plancherel:** $\int |f(x)|^2 dx = \int |\hat f(\xi)|^2 d\xi$.

**Fourier inversion.** For sufficiently nice $f$ (e.g., $f \in \mathcal S$, the Schwartz space; or $\hat f \in L^1$),
$$
f(x) = \int_{-\infty}^\infty e^{2\pi i \xi x} \hat f(\xi) \, d\xi.
$$
The inversion formula is the Fourier-analytic counterpart of the Cauchy integral formula: both express "function values from all of its data (appropriately weighted)."

### 0.5.6.3 Complex Extension and Paley-Wiener

**Theorem 0.5.6.2 (Paley-Wiener).** Let $f \in L^2(\mathbb{R})$. The following are equivalent:

(a) $f$ vanishes on $(-\infty, 0)$ (i.e., $f$ is supported on $[0, \infty)$).

(b) $\hat f$ extends to a holomorphic function on the lower half plane $\{\operatorname{Im} \xi < 0\}$ with $\sup_{\eta < 0} \int |\hat f(\xi + i\eta)|^2 d\xi < \infty$.

*Proof (sketch).* **(a) ⇒ (b):** For $f$ supported on $[0, \infty)$,
$$
\hat f(\xi + i\eta) = \int_0^\infty e^{-2\pi i (\xi + i\eta) x} f(x) dx = \int_0^\infty e^{-2\pi i \xi x} e^{2\pi \eta x} f(x) dx.
$$
For $\eta < 0$, $e^{2\pi \eta x}$ is decaying as $x \to \infty$, so the integral converges and is holomorphic in $\xi + i\eta$ (by differentiating under the integral). The $L^2$ bound uses Plancherel.

**(b) ⇒ (a):** If $\hat f$ extends holomorphically to the lower half plane with bounded $L^2$-norm on horizontal lines, then its inverse Fourier transform $f$ can be computed by shifting the inversion contour into the upper half plane for $x < 0$, showing $f(x) = 0$ there. $\blacksquare$

**Physical meaning.** Causal signals (zero for $t < 0$) have Fourier transforms holomorphic in the lower half plane. This is the Paley-Wiener–Kramers-Kronig–Hilbert transform connection: real and imaginary parts of a causal transfer function are Hilbert-transform pairs.

### 0.5.6.4 Characteristic Functions

**Definition 0.5.6.3 (Characteristic function).** For a random variable $X$ with distribution $\mu$,
$$
\phi_X(t) := \mathbb{E}[e^{itX}] = \int_\mathbb{R} e^{itx} d\mu(x), \qquad t \in \mathbb{R}.
$$
This always exists for real $t$ and $|\phi_X(t)| \leq 1$.

**Theorem 0.5.6.4 (Properties).**

(a) $\phi_X(0) = 1$.

(b) $\phi_X$ is uniformly continuous on $\mathbb{R}$.

(c) $\phi_X$ is *positive-definite*: for any $t_1, \ldots, t_n \in \mathbb{R}$ and $c_1, \ldots, c_n \in \mathbb{C}$,
$$
\sum_{j, k} c_j \bar c_k \phi_X(t_j - t_k) \geq 0.
$$

(d) $\phi_{-X}(t) = \overline{\phi_X(t)}$.

(e) (Independence) If $X, Y$ are independent, $\phi_{X+Y}(t) = \phi_X(t) \phi_Y(t)$.

(f) (Scaling) $\phi_{aX + b}(t) = e^{ibt} \phi_X(at)$.

*Proofs sketch.* (a) trivial. (b) $|\phi(t) - \phi(s)| \leq \mathbb{E}|e^{itX} - e^{isX}| \leq \mathbb{E}[|t-s| |X|] \wedge 2$ — bounded convergence shows uniform continuity. (c) $\sum c_j \bar c_k \phi(t_j - t_k) = \mathbb{E}|\sum c_j e^{it_j X}|^2 \geq 0$. (d)-(f) direct computation. $\blacksquare$

**Theorem 0.5.6.5 (Bochner).** A function $\phi: \mathbb{R} \to \mathbb{C}$ is the characteristic function of some probability measure iff $\phi$ is continuous, $\phi(0) = 1$, and $\phi$ is positive-definite.

The $\Leftarrow$ direction — positive-definiteness implies existence of a representing measure — is the content of Bochner's theorem; the measure is constructed via Fourier inversion on $\phi$.

### 0.5.6.5 Moments and Analytic Extension

**Theorem 0.5.6.6 (Moments from characteristic function).** If $\mathbb{E}|X|^n < \infty$ for some integer $n$, then $\phi_X$ is $n$-times continuously differentiable with
$$
\phi_X^{(k)}(0) = i^k \mathbb{E}[X^k], \qquad k = 0, 1, \ldots, n.
$$

*Proof.* Differentiating under the integral: $\phi^{(k)}(t) = \int (ix)^k e^{itx} d\mu$. At $t = 0$: $\phi^{(k)}(0) = i^k \int x^k d\mu = i^k \mathbb{E}[X^k]$. Differentiation under the integral is justified by $|(ix)^k e^{itx}| \leq |x|^k$, which is in $L^1(\mu)$ by hypothesis. $\blacksquare$

**Theorem 0.5.6.7 (Analytic extension).** If $\mathbb{E}[e^{a|X|}] < \infty$ for some $a > 0$, then $\phi_X$ extends to a holomorphic function on the strip $\{|\operatorname{Im} t| < a\}$, and
$$
\phi_X(t) = \int e^{itx} d\mu(x), \qquad |\operatorname{Im} t| < a,
$$
where the integral continues to converge absolutely.

*Proof.* For $t = u + iv$ with $|v| < a$, $|e^{itx}| = e^{-vx}$, so $|e^{itx}| \leq e^{|v||x|} \leq e^{a|X|}$ pointwise (for $|v| \leq a$) in the support. By dominated convergence, the integral converges and is holomorphic in $t$ (holomorphy under the integral sign: check $\partial/\partial \bar t$ of the integrand is zero, pass through the integral using DCT). $\blacksquare$

**Corollary.** If $X$ has moments of all orders satisfying $\sum |\mathbb{E}[X^n]| r^n/n! < \infty$ for some $r > 0$, then $\phi_X$ is **entire of exponential type** — it extends to all of $\mathbb{C}$ with growth $|\phi_X(t)| \leq C e^{a|t|}$ for some $a \geq 0$.

### 0.5.6.6 Gaussian Characteristic Function

**Theorem 0.5.6.8.** For $X \sim \mathcal N(\mu, \sigma^2)$,
$$
\phi_X(t) = e^{i\mu t - \sigma^2 t^2/2}.
$$

*Proof.* Wlog $\mu = 0$, $\sigma = 1$. Compute
$$
\phi(t) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^\infty e^{itx} e^{-x^2/2} dx = \frac{1}{\sqrt{2\pi}} \int e^{-(x - it)^2/2} e^{-t^2/2} dx.
$$
We need $\int e^{-(x - it)^2/2} dx = \sqrt{2\pi}$. For complex $a := it$, define $I(a) := \int_{-\infty}^\infty e^{-(x - a)^2/2} dx$. The integrand is entire in $a$, and $I(0) = \sqrt{2\pi}$ (standard Gaussian). Deform the contour: $x \in \mathbb{R}$ translated by $a$ gives $y = x - a$, $dy = dx$. Since $e^{-z^2/2}$ decays rapidly as $|\operatorname{Re} z| \to \infty$, contour deformation from $\mathbb{R}$ to $\mathbb{R} + a$ is justified (apply Cauchy on a large rectangle; side contributions $\to 0$). So $I(a) = \sqrt{2\pi}$ for all $a \in \mathbb{C}$. Hence $\phi(t) = e^{-t^2/2}$. $\blacksquare$

**Analytic continuation.** The expression $e^{-t^2/2}$ is entire, agreeing with the integral formula on $\mathbb{R}$; it extends the characteristic function to all of $\mathbb{C}$. For $t = iu$ (pure imaginary), $\phi(iu) = e^{u^2/2}$, which is the moment generating function of $X$ evaluated at $u$.

### 0.5.6.7 Characteristic Functions for Other Distributions

**Cauchy.** $X$ has density $1/[\pi(1 + x^2)]$. Contour integral gives
$$
\phi_X(t) = e^{-|t|}.
$$
Proof via residues: $\phi(t) = \pi^{-1} \int e^{itx}/(1 + x^2) dx$. For $t > 0$, close in upper half plane around pole at $z = i$: $\operatorname{Res}_i [e^{itz}/(1 + z^2)] = e^{-t}/(2i)$. So $\phi(t) = 2\pi i \cdot e^{-t}/(2i\pi) = e^{-t}$. By evenness, $\phi(-t) = e^t$... wait, but $|\phi| \leq 1$ — let me recompute. We need $\phi(-t) = \phi(t)$ if symmetric. For $t > 0$, close in UHP, $\phi(t) = e^{-t}$. For $t < 0$, close in LHP, $\phi(t) = e^{t} = e^{-|t|}$. So $\phi(t) = e^{-|t|}$ for all $t \in \mathbb{R}$. ✓

**Note:** The Cauchy characteristic function is *not differentiable* at $t = 0$. This reflects the fact that the Cauchy distribution has no moments (undefined mean and infinite variance).

**Exponential.** $X \sim \operatorname{Exp}(\lambda)$, density $\lambda e^{-\lambda x}$ on $[0, \infty)$.
$$
\phi_X(t) = \lambda \int_0^\infty e^{itx - \lambda x} dx = \frac{\lambda}{\lambda - it}.
$$
Analytic in $\{\operatorname{Im} t > -\lambda\}$ (so MGF exists for $u < \lambda$); pole at $t = -i\lambda$.

**Gamma.** $X \sim \operatorname{Gamma}(k, \theta)$, density $x^{k-1} e^{-x/\theta}/[\Gamma(k) \theta^k]$.
$$
\phi_X(t) = (1 - i\theta t)^{-k}.
$$
(Branch cut chosen such that at $t = 0$: $\phi = 1$.)

**Poisson.** $X \sim \operatorname{Pois}(\lambda)$.
$$
\phi_X(t) = e^{\lambda(e^{it} - 1)}.
$$
Entire in $t$ (everywhere holomorphic since $e^{it}$ is entire and bounded uniformly on horizontal strips).

### 0.5.6.8 Inversion: Gil-Pelaez

**Theorem 0.5.6.9 (Gil-Pelaez formula).** If $X$ has continuous distribution function $F$, then
$$
F(x) = \frac{1}{2} - \frac{1}{\pi} \int_0^\infty \frac{\operatorname{Im}[e^{-itx} \phi_X(t)]}{t} dt.
$$

*Proof.* Consider $\phi_X(t) = \int e^{ity} dF(y)$, and the auxiliary integral
$$
\int_{-T}^T \frac{e^{-itx} \phi_X(t)}{it} dt = \int_{-T}^T \int \frac{e^{it(y - x)}}{it} dF(y) dt = \int \left[ \int_{-T}^T \frac{e^{it(y-x)}}{it} dt \right] dF(y).
$$
The inner integral $\int_{-T}^T e^{it(y-x)}/(it) dt = 2 \int_0^T \sin(t(y-x))/t \, dt$ converges as $T \to \infty$ to $\pi \operatorname{sgn}(y - x)$ (Dirichlet integral). By dominated convergence, the outer integral converges to
$$
\int \pi \operatorname{sgn}(y - x) dF(y) = \pi \bigl(\mathbb{P}(X > x) - \mathbb{P}(X < x)\bigr) = \pi (1 - 2F(x))
$$
(assuming no atom at $x$). Equating:
$$
\int_{-\infty}^\infty \frac{e^{-itx}\phi_X(t)}{it} dt = \pi - 2\pi F(x).
$$
Rewriting as $1/(it) = -i/t$ and taking the imaginary part (the integral is real-valued by symmetry $\phi(-t) = \overline{\phi(t)}$):
$$
F(x) = \frac{1}{2} - \frac{1}{2\pi i} \int_{-\infty}^\infty \frac{e^{-itx}\phi_X(t)}{t} dt = \frac{1}{2} - \frac{1}{\pi} \int_0^\infty \frac{\operatorname{Im}[e^{-itx} \phi_X(t)]}{t} dt,
$$
using $\operatorname{Im}[e^{-itx}\phi_X(t)/t]$ integration. The $t = 0$ issue: the integrand is bounded there since $\phi_X(0) = 1$ and $e^{-itx} = 1$, so by L'Hopital $\operatorname{Im}[\ldots]/t \to \operatorname{Im}[i(\phi'(0) - x)] = \operatorname{Re}[\phi'(0) - x] = \mathbb{E}[X] - x$ if $\mathbb{E}|X| < \infty$; in general bounded via a principal-value/contour-deformation argument around $t = 0$. $\blacksquare$

### 0.5.6.9 Fourier Inversion and Density Recovery

**Theorem 0.5.6.10.** If $\phi_X \in L^1(\mathbb{R})$, then $X$ has a bounded continuous density $p_X$ given by
$$
p_X(x) = \frac{1}{2\pi} \int_{-\infty}^\infty e^{-itx} \phi_X(t) dt.
$$

*Proof.* The integral on the right defines a bounded continuous function $g(x)$ (continuity by DCT). We want $g = p_X$. Compute $\int e^{i \alpha x} g(x) dx$ (trying to show this equals $\phi_X(-\alpha) = \overline{\phi_X(\alpha)}$, confirming $g$ is the density):
$$
\int e^{i\alpha x} g(x) dx = \frac{1}{2\pi} \int e^{i\alpha x} \int e^{-itx} \phi_X(t) dt \, dx.
$$
By Fubini (assuming enough integrability; a standard mollification argument fills in the details),
$$
= \frac{1}{2\pi} \int \phi_X(t) \int e^{i(\alpha - t) x} dx \, dt = \int \phi_X(t) \delta(\alpha - t) dt = \phi_X(\alpha).
$$
So $g$ has characteristic function $\phi_X$, hence $g = p_X$. $\blacksquare$

### 0.5.6.10 Lévy's Continuity Theorem

**Theorem 0.5.6.11 (Lévy continuity).** Let $\{X_n\}$ be random variables with characteristic functions $\phi_n$. The following are equivalent:

(a) $X_n \xrightarrow{d} X$ (convergence in distribution to some $X$).

(b) $\phi_n(t) \to \phi_X(t)$ pointwise for all $t \in \mathbb{R}$, where $\phi_X$ is continuous at $0$.

*Proof (sketch).* **(a) ⇒ (b):** weak convergence and boundedness of $e^{itx}$ give pointwise convergence of characteristic functions by the definition of weak convergence.

**(b) ⇒ (a):** Hardest direction. The limit $\phi$ satisfies $\phi(0) = 1$ and is continuous at $0$ by hypothesis, and is positive-definite (since each $\phi_n$ is and positive-definiteness passes to pointwise limits). By Bochner, $\phi = \phi_X$ for some $X$. Tightness of $\{X_n\}$: use $\mathbb{P}(|X_n| > K) \leq 2 \int_{-1/K}^{1/K} (1 - \operatorname{Re} \phi_n(t)) dt / ... $ (a standard tightness estimate); pointwise convergence of $\phi_n$ to a function continuous at $0$ makes $\{X_n\}$ tight. Combining tightness with identification of limits gives weak convergence. $\blacksquare$

**Application: CLT.** If $X_i$ are i.i.d. with mean $0$, variance $1$, and $S_n = X_1 + \cdots + X_n$, then $S_n/\sqrt n \xrightarrow{d} \mathcal N(0, 1)$.

*Proof via Lévy continuity.* $\phi_{X_i}(t) = 1 - t^2/2 + o(t^2)$ as $t \to 0$ (using the second-moment assumption and Taylor expansion). By independence,
$$
\phi_{S_n/\sqrt n}(t) = [\phi_{X_1}(t/\sqrt n)]^n = \left[1 - \frac{t^2}{2n} + o(1/n) \right]^n \to e^{-t^2/2},
$$
which is the characteristic function of $\mathcal N(0, 1)$. By Lévy continuity, $S_n/\sqrt n \xrightarrow{d} \mathcal N(0, 1)$. $\blacksquare$

This two-line proof is one of the slickest in all of probability theory and illustrates why characteristic functions are fundamental.

### 0.5.6.11 Fourier Option Pricing in Detail

**Black-Scholes via Fourier transform.** In the Black-Scholes model, the log-price $X_T = \log S_T$ under the risk-neutral measure is
$$
X_T = X_0 + (r - \sigma^2/2)T + \sigma W_T,
$$
so $X_T$ is normal with mean $X_0 + (r - \sigma^2/2)T$ and variance $\sigma^2 T$. Characteristic function:
$$
\phi_T(v) = e^{i v[X_0 + (r - \sigma^2/2)T] - v^2 \sigma^2 T/2}.
$$

**Call price via Fourier.** Price of a European call at time $0$:
$$
C(K) = e^{-rT} \mathbb{E}[(S_T - K)^+] = e^{-rT} \mathbb{E}[(e^{X_T} - e^k)^+], \quad k := \log K.
$$
Writing the payoff $(e^x - e^k)^+$ as a Fourier transform requires a "damping" because the payoff is not $L^1$. The **Carr-Madan trick**: multiply by $e^{\alpha k}$ for $\alpha > 0$ to ensure integrability of the modified price.

Define $c(k) := e^{\alpha k} C(e^k)$. Then
$$
\hat c(v) := \int e^{i v k} c(k) dk = \frac{e^{-rT} \phi_T(v - (\alpha + 1) i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1) v}.
$$
The argument $v - (\alpha + 1)i$ of $\phi_T$ is *complex* — requiring the characteristic function to be analytically extendable to the strip $\{\operatorname{Im} v \in [-(\alpha + 1), 0]\}$. For Black-Scholes this is no issue ($\phi$ is entire). For other models (e.g., Heston), the strip of analyticity has finite width.

Inverse Fourier:
$$
C(K) = e^{-\alpha k} c(k) = \frac{e^{-\alpha k}}{2\pi} \int_{-\infty}^\infty e^{-ivk} \hat c(v) dv = \frac{e^{-\alpha k}}{\pi} \int_0^\infty \operatorname{Re}[e^{-ivk} \hat c(v)] dv.
$$

**Numerically.** Discretize the $v$-integral over $[0, A]$ with step $\Delta v$, evaluate at $N = A/\Delta v$ points $v_j = (j - 1/2) \Delta v$, and compute via FFT. Choose $\alpha \in [0.5, 2]$ (tuning: too small = slow decay of $\hat c$; too large = numerical issues from $e^{\alpha k}$ blowup).

### 0.5.6.12 Lévy Processes and Lévy-Khintchine

**Definition 0.5.6.12 (Lévy process).** A stochastic process $\{X_t\}_{t \geq 0}$ with $X_0 = 0$, independent stationary increments, and càdlàg paths. Fundamental examples: Brownian motion, Poisson process, compound Poisson, stable processes.

**Theorem 0.5.6.13 (Lévy-Khintchine).** Every Lévy process has characteristic function
$$
\phi_{X_t}(u) = e^{t \psi(u)}, \quad \psi(u) = iu b - \frac{\sigma^2 u^2}{2} + \int_{\mathbb R \setminus \{0\}} (e^{iu y} - 1 - iuy \mathbf 1_{|y| \leq 1}) \nu(dy),
$$
where $b \in \mathbb{R}$ (drift), $\sigma \geq 0$ (Brownian variance), and $\nu$ is a **Lévy measure** with $\int \min(y^2, 1) \, d\nu < \infty$.

This is the "characteristic triple" $(b, \sigma^2, \nu)$ classification. Key examples:

- **Brownian motion with drift:** $(b, \sigma^2, 0)$. $\psi(u) = iub - \sigma^2 u^2/2$.
- **Compound Poisson** with jump distribution $\mu$ and rate $\lambda$: $(0, 0, \lambda \mu)$, $\psi(u) = \lambda(\phi_\mu(u) - 1)$.
- **Variance Gamma process:** jumps with two-sided exponential Lévy density; $\psi$ involves $\log(1 - iu\theta\nu + \sigma^2 \nu u^2/2)/\nu$.
- **CGMY/KoBoL:** stable-like jumps; $\nu$ has explicit tails.
- **NIG (Normal Inverse Gaussian):** $\psi$ involves hyperbolic tangent.

Each of these is a celebrated finance model; the Lévy symbol $\psi$ is the analytic object encoding everything.

### 0.5.6.13 Stable Distributions and Heavy Tails

**Definition 0.5.6.14.** A distribution is **$\alpha$-stable** (for $\alpha \in (0, 2]$) if it's closed under iid sums up to scaling and shifts: $X_1 + \cdots + X_n \stackrel{d}{=} n^{1/\alpha} X_1 + c_n$.

**Characteristic function.** Symmetric $\alpha$-stable: $\phi(u) = e^{-|cu|^\alpha}$. For $\alpha = 2$: Gaussian ($e^{-c^2 u^2}$). For $\alpha = 1$: Cauchy ($e^{-c|u|}$). For $\alpha \in (0, 1) \cup (1, 2)$: stable-Paretian; heavy tails like $|x|^{-1-\alpha}$.

**Tail asymptotics.** $\alpha$-stable distributions (for $\alpha < 2$) have $\mathbb{P}(|X| > x) \sim C/x^\alpha$ as $x \to \infty$. No finite variance when $\alpha < 2$; no finite mean when $\alpha \leq 1$.

**Application.** Mandelbrot's 1963 paper proposed stable distributions for cotton prices, initiating the heavy-tail finance tradition. Modern quantitative finance uses stable-tempered distributions (CGMY, GH) to fit equity returns with heavy tails that are thin enough to admit moments.

### 0.5.6.14 Python: Characteristic Function Toolkit

```python
import numpy as np
from scipy.stats import norm, cauchy, gamma, expon
from scipy.special import gamma as gamma_fn

# Gaussian CF
def phi_gaussian(t, mu, sigma):
    return np.exp(1j * mu * t - 0.5 * sigma**2 * t**2)

# Cauchy CF
def phi_cauchy(t, loc=0, scale=1):
    return np.exp(1j * loc * t - scale * np.abs(t))

# Exponential CF
def phi_exp(t, rate):
    return rate / (rate - 1j * t)

# Gamma CF (shape k, scale theta)
def phi_gamma(t, k, theta):
    return (1 - 1j * theta * t)**(-k)

# Poisson CF
def phi_poisson(t, lam):
    return np.exp(lam * (np.exp(1j * t) - 1))

# Verify by direct integration: ∫ e^(itx) p(x) dx = φ(t)
import scipy.integrate as si
def cf_via_integration(density, t, support=(-np.inf, np.inf)):
    real, _ = si.quad(lambda x: np.cos(t*x) * density(x), *support)
    imag, _ = si.quad(lambda x: np.sin(t*x) * density(x), *support)
    return real + 1j * imag

t = 1.5
# Gaussian check
print("Gaussian CF check:")
print(f"  Formula:     {phi_gaussian(t, 0, 1):.6f}")
print(f"  Integration: {cf_via_integration(norm.pdf, t):.6f}")

# Cauchy check
print("Cauchy CF check:")
print(f"  Formula:     {phi_cauchy(t):.6f}")
print(f"  Integration: {cf_via_integration(cauchy.pdf, t):.6f}")

# Fourier inversion for density recovery
def density_from_cf(cf, x, T=100, N=10000):
    """p(x) = (1/2π) ∫ e^(-itx) φ(t) dt"""
    t = np.linspace(-T, T, N)
    dt = t[1] - t[0]
    integrand = np.exp(-1j * t * x) * cf(t)
    return np.real(np.sum(integrand) * dt) / (2 * np.pi)

# Check: recover normal density from its CF
for x in [-1, 0, 1, 2]:
    p_fourier = density_from_cf(lambda t: phi_gaussian(t, 0, 1), x)
    p_direct = norm.pdf(x)
    print(f"N(0,1) pdf at x={x}: Fourier={p_fourier:.6f}, direct={p_direct:.6f}")

# Gil-Pelaez: F(x) = 1/2 - (1/π) ∫_0^∞ Im[e^(-itx) φ(t)] / t dt
def cdf_via_gil_pelaez(cf, x, T=100, N=10000):
    t = np.linspace(1e-6, T, N)
    dt = t[1] - t[0]
    integrand = np.imag(np.exp(-1j * t * x) * cf(t)) / t
    return 0.5 - np.sum(integrand) * dt / np.pi

# Check: recover normal CDF
for x in [-2, -1, 0, 1, 2]:
    F_gp = cdf_via_gil_pelaez(lambda t: phi_gaussian(t, 0, 1), x)
    F_direct = norm.cdf(x)
    print(f"N(0,1) CDF at x={x}: Gil-Pelaez={F_gp:.4f}, direct={F_direct:.4f}")

# Carr-Madan call pricing (Black-Scholes model)
def carr_madan_call(S0, K, r, sigma, T, alpha=1.5, N=2**12, eta=0.25):
    """Carr-Madan FFT pricer for a European call under Black-Scholes."""
    # Black-Scholes characteristic function of log(S_T)
    def phi_BS(v):
        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        return np.exp(1j * v * mu - 0.5 * sigma**2 * T * v**2)

    # damped Fourier transform of call price
    def psi(v):
        num = np.exp(-r * T) * phi_BS(v - (alpha + 1) * 1j)
        den = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
        return num / den

    # FFT setup
    lam = 2 * np.pi / (N * eta)
    beta = -lam * N / 2
    v = np.arange(N) * eta
    ks = beta + lam * np.arange(N)  # log-strike grid

    # apply Simpson's rule
    simpson_weights = (3 + (-1)**(np.arange(N) + 1) - np.concatenate([[1], np.zeros(N-1)])) / 3

    # compute Fourier kernel
    x = np.exp(-1j * beta * v) * psi(v) * eta * simpson_weights
    fft_val = np.fft.fft(x)

    call_prices = np.exp(-alpha * ks) * np.real(fft_val) / np.pi
    strikes = np.exp(ks)

    # find the price at the target K
    from scipy.interpolate import interp1d
    interp = interp1d(strikes, call_prices, kind='cubic', bounds_error=False)
    return float(interp(K))

# Compare with Black-Scholes closed form
def BS_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
price_carr = carr_madan_call(S0, K, r, sigma, T)
price_bs = BS_call(S0, K, r, sigma, T)
print(f"\nCarr-Madan call price: {price_carr:.4f}")
print(f"Black-Scholes closed form: {price_bs:.4f}")

# CLT verification via characteristic functions
n_values = [1, 5, 50, 500]
t_values = np.linspace(-3, 3, 100)
for n in n_values:
    # sum of n iid Exp(1) variables, centered and scaled
    # X_i ~ Exp(1), mean 1, var 1. (X_1+...+X_n - n)/sqrt(n) → N(0,1)
    # CF: φ_X(t) = 1/(1 - it), so CF of (X - 1) is e^(-it)/(1 - it).
    # CF of (∑X_i - n)/sqrt(n) is [e^(-it/sqrt(n))/(1 - it/sqrt(n))]^n
    cf_scaled = (np.exp(-1j * t_values / np.sqrt(n)) / (1 - 1j * t_values / np.sqrt(n)))**n
    cf_limit = np.exp(-t_values**2 / 2)
    max_err = np.max(np.abs(cf_scaled - cf_limit))
    print(f"n={n}: max |CF_n(t) - e^(-t²/2)| = {max_err:.5f}")
```

### 0.5.6.15 [QUANT APPLICATION]

**1. Option pricing panorama.** The following models all use Fourier methods for pricing:
- **Black-Scholes:** CF is Gaussian, entire.
- **Merton jump-diffusion:** CF is Gaussian × Poisson-jumps CF, entire.
- **Heston stochastic volatility:** CF involves hyperbolic functions and requires careful branch selection ("little trap" vs "big trap" formulation).
- **Variance Gamma, CGMY, NIG:** CF has explicit form involving Bessel functions or Gamma functions.
- **Stochastic volatility + jumps** (Bates, SVJ): CF is product of Heston and Merton CFs.
- **Rough Heston:** Fractional calculus in the CF; Mittag-Leffler functions.

All of these yield call prices via Carr-Madan or similar Fourier transform methods.

**2. Dirichlet problem and conformal pricing.** For some exotic options (e.g., barrier options, American options), the pricing PDE can be transformed to a Dirichlet problem on a domain, and solved via conformal mapping. The Laplacian in the log-price coordinate is the key connection; Fourier methods provide the eigenfunction expansion.

**3. Characteristic function of portfolio losses.** For a portfolio $L = \sum w_i X_i$ of losses, $\phi_L(t) = \prod \phi_{X_i}(w_i t)$ (if independent). This is the foundation of **CreditMetrics** and **risk aggregation**: compute VaR and expected shortfall by Fourier inversion of $\phi_L$.

**4. Lévy process simulation via CF.** Some Lévy processes are hard to simulate directly, but their characteristic function has an easy form. Sample paths can be generated by the **CMS algorithm** or via random walk + rejection sampling, using the CF to compute increment densities.

**5. Martingale tests via CF.** To check whether a measure change makes a process a martingale, one verifies $\phi_{X_t}(t; \mathbb{Q}) = \phi_{X_t}(t; \mathbb{P}) \cdot e^{t \psi_\text{adjust}(-i)}$ or similar. This analytic shift is the engine of equivalent martingale measure constructions.

**6. Spectral method for PDEs.** Parabolic PDEs like the heat equation or Black-Scholes PDE transform under Fourier to ODEs in the spectral variable. This is the mathematical basis of "FFT methods for PDE" used in finance: solve in spectral space, transform back.

### 0.5.6.16 Exercises

#### ★ (Warm-up)

**E0.5.6.1.** Compute the Fourier transform of $f(x) = e^{-x^2/2}$ using a contour integral.

**E0.5.6.2.** Compute the Fourier transform of $f(x) = 1/(1 + x^2)$ via residues (closing in upper or lower half plane depending on sign of $\xi$).

**E0.5.6.3.** Use the convolution theorem to compute the density of the sum of two iid $\operatorname{Exp}(\lambda)$ random variables. Check: Gamma(2, 1/λ).

**E0.5.6.4.** Show that the characteristic function of a shifted-by-$\mu$, scaled-by-$\sigma$ version of $X$ is $\phi_X(\sigma t) e^{i\mu t}$.

**E0.5.6.5.** Compute the characteristic function of a uniform $[0, 1]$ random variable: $\phi(t) = (e^{it} - 1)/(it)$.

**E0.5.6.6.** Verify the Gaussian CF $e^{-t^2/2}$ directly by expanding $\mathbb{E}[e^{itX}]$ as a power series in $t$ and using $\mathbb{E}[X^{2k}] = (2k)!/(2^k k!)$.

#### ★★ (Standard)

**E0.5.6.7.** Prove the **CLT for iid $X_i$ with $\mathbb{E}|X|^3 < \infty$:** using a third-order Taylor expansion of $\phi_X$ and Lévy continuity. (This recovers the rate of convergence in the Berry-Esseen theorem, modulo constants.)

**E0.5.6.8 (Characteristic function of sum).** Compute the characteristic function of $S_n = X_1 + \cdots + X_n$ where $X_i$ are iid Bernoulli($p$). Show $\phi_{S_n}(t) = (1 - p + p e^{it})^n$ (binomial CF).

**E0.5.6.9 (Inversion).** Show that for a random variable $X$ with $\phi_X \in L^1$, the density is $p(x) = (2\pi)^{-1} \int e^{-itx} \phi_X(t) dt$. Apply this to recover the Cauchy density from $\phi(t) = e^{-|t|}$.

**E0.5.6.10 (Continuous version of Gil-Pelaez).** Derive the density version: $p(x) = \pi^{-1} \int_0^\infty \operatorname{Re}[e^{-itx} \phi_X(t)] dt$ — this is real-valued by the symmetry $\phi(-t) = \overline{\phi(t)}$.

**E0.5.6.11 (Carr-Madan derivation).** Derive the Carr-Madan formula from scratch: starting with $C(K) = e^{-rT} \mathbb{E}[(S_T - K)^+]$, multiply by $e^{\alpha \log K}$, take Fourier transform in $\log K$, simplify using the joint density/characteristic function, and derive the explicit form of $\hat c(v)$.

**E0.5.6.12 (Heston characteristic function).** Look up the Heston model and write down its characteristic function. Verify that for $\sigma_v = 0$ it reduces to Black-Scholes. Discuss the "little trap" vs "big trap" formulations.

**E0.5.6.13 (Merton jump-diffusion).** The log-price under Merton is
$$
X_T = X_0 + (r - \sigma^2/2 - \lambda \kappa) T + \sigma W_T + \sum_{n=1}^{N_T} Y_n,
$$
where $N_T \sim \operatorname{Pois}(\lambda T)$ and $Y_n \sim \mathcal N(\mu_J, \sigma_J^2)$. Derive the CF of $X_T$ and use Carr-Madan to price a call.

**E0.5.6.14 (Tempered stable distributions).** A **tempered stable** distribution has Lévy density $\nu(dx) = c_\pm e^{-\lambda_\pm |x|} |x|^{-1-\alpha} dx$ (split on positive/negative $x$). Derive its characteristic function by integrating the Lévy-Khintchine formula.

#### ★★★ (Challenge)

**E0.5.6.15 (Multivariate Lévy processes).** Extend the Lévy-Khintchine formula to $\mathbb{R}^d$-valued Lévy processes. Discuss the role of the covariance matrix $\Sigma$ and the $d$-dimensional Lévy measure $\nu$ on $\mathbb{R}^d \setminus \{0\}$.

**E0.5.6.16 (Normal inverse Gaussian).** The NIG distribution has density proportional to $K_1(\alpha \sqrt{1 + ((x - \mu)/\delta)^2}) e^{\beta(x - \mu)}$ where $K_1$ is a modified Bessel function. Derive its characteristic function (it's $\exp[i\mu t + \delta(\sqrt{\alpha^2 - \beta^2} - \sqrt{\alpha^2 - (\beta + it)^2})]$). Show how $\alpha \to \infty$ with appropriate scaling gives Gaussian.

**E0.5.6.17 (Berry-Esseen theorem).** Prove the Berry-Esseen bound: for iid $X_i$ with $\mathbb{E}[X] = 0$, $\mathbb{E}[X^2] = \sigma^2$, $\mathbb{E}|X|^3 = \rho < \infty$,
$$
\sup_x |F_{S_n/(\sigma\sqrt n)}(x) - \Phi(x)| \leq C \rho / (\sigma^3 \sqrt n).
$$
(Hint: Esseen's smoothing lemma, bounding $|\phi_n(t) - e^{-t^2/2}|$ for small $t$ and controlling the tail.)

**E0.5.6.18 (FFT-based pricing).** Implement Carr-Madan for Heston and compare to the direct Monte Carlo pricing. Discuss the choice of damping parameter $\alpha$ (too small ↔ slow decay; too large ↔ instability near $K = 1$).

**E0.5.6.19 (Variance reduction via control variates).** In Monte Carlo pricing for a model with a known Fourier pricer (Black-Scholes), use the Black-Scholes price as a control variate. Express the efficient estimator in terms of the simulated path and the BS price, and verify variance reduction.

**E0.5.6.20 (Affine jump-diffusion).** An affine jump-diffusion has log-price characteristic function of the form $\exp[A(T, u) + B(T, u) X_0]$ where $A, B$ solve Riccati ODEs. Derive these ODEs for the Heston model (squared volatility Feller process + stock price). Discuss when the solutions remain in the upper half plane and avoid branch issues.

---

## Module 0.5 Summary

### The Big Theorems

Module 0.5 assembled the classical edifice of complex analysis. Here's the dependency graph:

```
Complex differentiability (§0.5.1)
     ↓
Cauchy-Riemann equations (§0.5.1)
     ↓
Contour integrals + Goursat's theorem (§0.5.2)
     ↓
Cauchy's integral formula + derivatives (§0.5.3)
     ↓
     ├─→ Holomorphic ⇔ Analytic (§0.5.3)
     ├─→ Liouville + Fundamental theorem of algebra (§0.5.3)
     ├─→ Maximum modulus + Schwarz's lemma (§0.5.3)
     ├─→ Morera ⇒ uniform limits preserve holomorphy (§0.5.3)
     └─→ Laurent expansion + residue theorem (§0.5.4)
                ↓
                ├─→ Real integrals via contour deformation (§0.5.4)
                ├─→ Argument principle + Rouché (§0.5.4)
                └─→ Characteristic-function methods (§0.5.6)
     ↓
Analytic continuation + Γ, ζ (§0.5.5)
```

The five-step chain (Goursat → Cauchy formula → analyticity → Laurent → residue theorem) is the spine of complex analysis. Every subsequent result is either a consequence or an extension into broader settings (Riemann surfaces, several complex variables, hyperfunctions, etc.).

### What This Module Unlocks

After mastering Module 0.5, you can:

1. **Derive characteristic functions** of common distributions (Gaussian, Cauchy, exponential, gamma, Poisson) and verify them via contour integration.
2. **Evaluate improper real integrals** via residues, including Fourier-type $\int e^{iax}/(x^2 + b^2) dx$ integrals and Dirichlet-type $\int \sin(x)/x \, dx$ integrals.
3. **Price options via Fourier transforms** using Carr-Madan and related methods, with analytic control over the convergence strip.
4. **Count zeros of polynomials** and rational functions using the argument principle and Rouché's theorem.
5. **Prove central limit theorems** via Lévy's continuity theorem and second-order Taylor expansion of characteristic functions.
6. **Understand branch-cut subtleties** in pricing formulas (Heston "little trap", logarithmic branches in calibration).
7. **Exploit the analyticity of transforms** to extract tail behavior (heavy tails from branch-cut singularities) and density smoothness (rapid-decay transforms).

### Forward Pointers

**Subject 1 — Measure Theory.** The Fourier inversion theorem and Gil-Pelaez formula were stated with hand-waved dominated convergence arguments. In Subject 1 we will develop Lebesgue integration, dominated/monotone convergence theorems, Fubini's theorem, and $L^p$ spaces — the proper framework in which Fourier analysis on $L^1, L^2$, distributions, and weak convergence of measures all live. The machinery required to make Lévy continuity, the Carr-Madan inversion, and the stochastic integrals of Subject 5 rigorous comes from there.

**Subject 2 — Probability Theory.** Characteristic functions, Lévy processes, Lévy-Khintchine, CLT, and Berry-Esseen all reappear in Subject 2 with full probabilistic rigor (not just "via integrals"). Martingales, conditional expectations, and the functional CLT (Donsker's theorem) require both measure theory and the complex-analytic tools developed here.

**Subject 3 — Stochastic Calculus.** Itô's formula is a stochastic version of the chain rule, but its Fourier-analytic consequences — how the law of $f(X_t)$ evolves when $X$ is a semimartingale — rely on characteristic functions. The Lévy-Khintchine decomposition in Subject 3 generalizes the symbol $\psi$ introduced here.

**Subject 7 — Derivatives Pricing.** Chapter 0.5.6's Carr-Madan material is the technical foundation for pricing under non-Gaussian models. Heston, Bates, VG, NIG, CGMY: each of these is a specific choice of Lévy symbol $\psi$, and the pricer is built on the same contour-integral + FFT scaffolding. The "transform methods chapter" of Derivatives Pricing is essentially Module 0.5.6 scaled up with finance-specific models.

**Subject 9 — Machine Learning for Finance.** Fourier methods show up in Gaussian processes (spectral mixture kernels), neural ODEs with Fourier features, spectral graph convolutions, and regularization via Fourier norms. The dual understanding of "holomorphic/meromorphic function ↔ probability law" developed here transfers to GANs (adversarial divergences interpretable as Fourier distances), normalizing flows (pushforward densities via holomorphic maps in some formulations), and rough volatility ML (where the fractional kernels are governed by complex Mellin transforms).

### Module 0.5 at a Glance

| Topic | Main theorems | Key tools |
|-------|---------------|-----------|
| 0.5.1 Cauchy-Riemann | C-R necessary + sufficient | Holomorphic ⇔ complex diff |
| 0.5.2 Contour integration | Goursat, Cauchy star-shaped | ML inequality, primitives |
| 0.5.3 Cauchy formula | Cauchy integral formula, Liouville, FTA, Morera | Maximum modulus, identity principle |
| 0.5.4 Laurent + residues | Laurent expansion, residue theorem, argument principle, Rouché | Singularity classification |
| 0.5.5 Analytic continuation | Monodromy, Schwarz reflection, Γ, ζ | Riemann surfaces |
| 0.5.6 Fourier + CF | Lévy continuity, Carr-Madan, Lévy-Khintchine | Characteristic triple |

### Recommended Follow-Up Reading

- **Stein-Shakarchi, *Complex Analysis*.** Pedagogical textbook with excellent exercises; the "follow-along" for everything in this module.
- **Ahlfors, *Complex Analysis*.** The classical reference; formal and terse, but beautiful.
- **Conway, *Functions of One Complex Variable, Vol 1 & 2*.** Comprehensive with applications.
- **Priestley, *Introduction to Complex Analysis*.** Friendly and geometric; good for building intuition.
- **Hörmander, *Complex Analysis in Several Variables*.** The next level — multi-variable complex analysis and PDEs.
- **Titchmarsh, *The Theory of the Riemann Zeta-Function*.** Deep dive into the zeta function, its functional equation, its zeros.
- **Carr-Madan (1999), *"Option valuation using the fast Fourier transform."*** The seminal paper for FFT option pricing, readable after Module 0.5.6.
- **Cont-Tankov, *Financial Modelling with Jump Processes*.** The definitive reference for Lévy processes in finance; the applications sections of Module 0.5 draw on this.

---

**End of Module 0.5.** Next up: **Subject 1 — Measure Theory** (probability spaces, σ-algebras, Lebesgue integration, $L^p$-spaces, Fubini, Radon-Nikodym, weak convergence of measures) — the rigorous foundation for probability theory and stochastic calculus.


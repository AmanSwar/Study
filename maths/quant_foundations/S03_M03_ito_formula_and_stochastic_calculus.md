# Module 3.3 — Itô's Formula and Stochastic Calculus

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 3 (Stochastic Processes), Module 3 of 7

---

## Prerequisites

- **Module 2.6** (Martingales, optional stopping, $L^2$-martingale convergence).
- **Module 3.1** (Brownian motion, quadratic variation $\langle B\rangle_t = t$, Lévy's martingale characterization).
- **Module 3.2** (Itô integral, isometry, continuous local martingales, stochastic Fubini).
- Multivariable calculus (gradient, Hessian, chain rule) and real analysis (Taylor expansion, Lipschitz continuity).

This module is the **computational heart** of stochastic calculus. Every closed-form option price, every Greeks computation, every SDE transformation, every measure change — all are applications of a single identity below. Read it, internalize it, and you unlock continuous-time finance.

---

## 1. The heuristic: why Itô's formula has a second-order term

### 1.1 Classical Taylor expansion

For a smooth function $f$ and a smooth path $x(t)$, the fundamental theorem of calculus gives
$$
f(x(t)) = f(x(0)) + \int_0^t f'(x(s))\, x'(s)\, ds = f(x(0)) + \int_0^t f'(x(s))\, dx(s).
$$

The proof is Taylor: $f(x + h) = f(x) + f'(x) h + \tfrac{1}{2} f''(x) h^2 + O(h^3)$. Partition $[0, t]$ into $n$ pieces, sum:
$$
f(x(t)) - f(x(0)) = \sum_k f'(x_k)\,\Delta x_k + \tfrac{1}{2}\sum_k f''(x_k)(\Delta x_k)^2 + O(\|\pi\|^2).
$$

If $x$ is smooth, then $(\Delta x_k)^2 \le (\text{Lip}\,x)^2 \|\pi\|^2 \to 0$, and the second-order sum vanishes. We recover the classical chain rule.

### 1.2 For Brownian paths, $(\Delta B)^2$ does not vanish

From Module 3.1, for Brownian motion,
$$
\sum_k (\Delta B_k)^2 \xrightarrow{L^2} t \qquad \text{as } \|\pi\| \to 0.
$$

Therefore, when we expand $f(B_t)$, the second-order Taylor term does not disappear:
$$
f(B_t) - f(B_0) = \sum_k f'(B_{t_k})\Delta B_k + \tfrac{1}{2}\sum_k f''(B_{t_k})(\Delta B_k)^2 + o(1).
$$

The first sum converges to $\int_0^t f'(B_s) dB_s$ (Itô integral). The second, via the **quadratic variation theorem**, converges to $\tfrac{1}{2}\int_0^t f''(B_s)\, ds$. Putting it together:

$$
\boxed{ \;f(B_t) = f(B_0) + \int_0^t f'(B_s)\, dB_s + \frac{1}{2}\int_0^t f''(B_s)\, ds.\; }
$$

This is **Itô's formula for Brownian motion**. The extra $\tfrac{1}{2}\int f''(B_s) ds$ is the **Itô correction**, directly traceable to the fact that Brownian paths have nonzero quadratic variation.

---

## 2. Itô's formula (full statement and proof)

### 2.1 One-dimensional Itô process

An **Itô process** is $X_t = X_0 + \int_0^t \mu_s\, ds + \int_0^t \sigma_s\, dB_s$ with $\mu$ progressively measurable and $\int_0^T |\mu_s|\,ds < \infty$ a.s., and $\sigma$ predictable with $\int_0^T \sigma_s^2 ds < \infty$ a.s.

In differential notation: $dX_t = \mu_t\, dt + \sigma_t\, dB_t$.

Its quadratic variation: $d\langle X\rangle_t = \sigma_t^2\, dt$.

**Theorem 2.1 (Itô's formula, one dimension).** If $f \in C^{1,2}([0, T] \times \mathbb{R})$ (once continuously differentiable in $t$, twice in $x$), then

$$
\boxed{\;
f(t, X_t) = f(0, X_0) + \int_0^t \partial_t f(s, X_s)\, ds + \int_0^t \partial_x f(s, X_s)\, dX_s + \frac{1}{2}\int_0^t \partial_{xx} f(s, X_s)\, d\langle X\rangle_s.
\;}
$$

Unpacking $dX_s$ and $d\langle X\rangle_s$:
$$
df(t, X_t) = \Bigl[\partial_t f + \mu_t \partial_x f + \tfrac{1}{2}\sigma_t^2 \partial_{xx} f\Bigr]\, dt + \sigma_t \partial_x f\, dB_t.
$$

### 2.2 Proof sketch (localization to bounded processes)

We show the case $f = f(x)$, $X = B$; the general case follows by the same pattern plus managing time/drift dependence.

**Step 1 (bounded, compactly supported $f''$).** Suppose $f \in C^2$ with $f, f', f''$ bounded. Fix $t$ and a partition $\pi_n = \{0 = t_0 < \dots < t_n = t\}$ with $\|\pi_n\| \to 0$. By Taylor with remainder,
$$
f(B_{t_{k+1}}) - f(B_{t_k}) = f'(B_{t_k})\Delta B_k + \tfrac{1}{2}f''(B_{t_k})(\Delta B_k)^2 + R_k,
$$
where $|R_k| \le \tfrac{1}{6}\|f'''\|_\infty |\Delta B_k|^3$. Summing:
$$
f(B_t) - f(B_0) = \underbrace{\sum_k f'(B_{t_k})\Delta B_k}_{A_n} + \tfrac{1}{2}\underbrace{\sum_k f''(B_{t_k})(\Delta B_k)^2}_{B_n} + \underbrace{\sum_k R_k}_{C_n}.
$$

**Step 2 ($A_n \to \int_0^t f'(B_s) dB_s$):** The process $s \mapsto f'(B_s)$ is continuous, so left-continuous adapted, so the left-endpoint Riemann sum $A_n$ converges in $L^2$ to the Itô integral (Module 3.2, Theorem 4.2 / density of $\mathcal{S}$).

**Step 3 ($B_n \to \int_0^t f''(B_s) ds$):** Because $f''$ is continuous bounded and $(\Delta B_k)^2 \approx t_{k+1} - t_k$ in $L^2$, write
$$
B_n = \sum_k f''(B_{t_k})[(\Delta B_k)^2 - (t_{k+1}-t_k)] + \sum_k f''(B_{t_k})(t_{k+1} - t_k).
$$

The second sum is a Riemann sum of $f''(B_\cdot)$; it converges to $\int_0^t f''(B_s) ds$ in $L^2$. For the first sum, the key computation is
$$
E\Bigl[\bigl(\sum_k f''(B_{t_k})[(\Delta B_k)^2 - (t_{k+1}-t_k)]\bigr)^2\Bigr] = \sum_k E[f''(B_{t_k})^2] \cdot 2(t_{k+1}-t_k)^2,
$$
using that $(\Delta B_k)^2 - (t_{k+1}-t_k)$ has mean zero, variance $2(t_{k+1}-t_k)^2$, and is independent across $k$. The RHS $\le 2\|f''\|_\infty^2 t\,\|\pi_n\| \to 0$.

**Step 4 ($C_n \to 0$):** $|C_n| \le \tfrac{1}{6}\|f'''\|_\infty \sum_k |\Delta B_k|^3$. Use $E[|\Delta B_k|^3] = c\, (t_{k+1}-t_k)^{3/2}$ with $c = 2\sqrt{2/\pi}$, so $E[|C_n|] \le c\|f'''\|_\infty\,\|\pi_n\|^{1/2}\, t \to 0$.

**Step 5 (extend to arbitrary $C^2$).** For general $C^2$, use a truncation / smoothing argument: approximate $f$ by $f_\varepsilon \in C^\infty_b$ uniformly on compacts, use Step 1–4 for $f_\varepsilon$, then pass $\varepsilon \to 0$. Localization via stopping times $\tau_M = \inf\{t : |B_t| \ge M\}$ handles the unboundedness of $B$.

**Step 6 (Itô process case).** For $X_t = X_0 + \int \mu + \int \sigma dB$, the proof is analogous with terms $(\Delta X_k)^2 \approx \sigma_{t_k}^2(t_{k+1}-t_k)$ (cross terms $\Delta t \cdot \Delta B$ contribute negligibly because of mesh size).

**Step 7 (time dependence).** Adding $\partial_t f$ is straightforward: the mixed Taylor expansion produces $\partial_t f \cdot \Delta t$ terms, which sum to $\int_0^t \partial_t f ds$.

This completes the proof. $\square$

---

## 3. Multidimensional Itô formula

For $X = (X^1, \ldots, X^d)$ with $dX^i_t = \mu^i_t\, dt + \sum_{j=1}^m \sigma^{ij}_t\, dB^j_t$, where $B = (B^1, \ldots, B^m)$ is a $d$-dimensional Brownian motion, and $f \in C^{1,2}([0,T] \times \mathbb{R}^d)$:

$$
df(t, X_t) = \partial_t f\, dt + \sum_i \partial_i f\, dX^i_t + \frac{1}{2}\sum_{i, j}\partial_{ij} f\, d\langle X^i, X^j\rangle_t,
$$

where the **covariation** is
$$
d\langle X^i, X^j\rangle_t = \sum_{k=1}^m \sigma^{ik}_t \sigma^{jk}_t\, dt = (\sigma_t \sigma_t^T)_{ij}\, dt.
$$

This formula packages up the geometry: the Hessian $\partial_{ij} f$ contracts against the diffusion matrix $\sigma\sigma^T$.

**Example (two-dimensional BM).** For $B = (B^1, B^2)$ independent Brownian motions and $f(x, y)$,
$$
df(B^1_t, B^2_t) = \partial_x f\, dB^1_t + \partial_y f\, dB^2_t + \tfrac{1}{2}(\partial_{xx} f + \partial_{yy} f)\, dt.
$$

The drift is **one half of the Laplacian** — the famous connection between Brownian motion and the heat equation that we exploit in Section 6.

---

## 4. Worked examples of Itô's formula

### 4.1 $f(x) = x^2$ applied to $B_t$

$f'(x) = 2x$, $f''(x) = 2$. Itô gives
$$
B_t^2 = B_0^2 + \int_0^t 2B_s\, dB_s + \tfrac{1}{2}\int_0^t 2\, ds = 2\int_0^t B_s\, dB_s + t.
$$

Rearranging: $\int_0^t B_s dB_s = \tfrac{1}{2}(B_t^2 - t)$. Confirming our Module 3.2 Example 11.2.

### 4.2 $f(t, x) = e^{\lambda x - \tfrac{1}{2}\lambda^2 t}$ — the stochastic exponential

$\partial_t f = -\tfrac{1}{2}\lambda^2 f$, $\partial_x f = \lambda f$, $\partial_{xx} f = \lambda^2 f$. Itô:
$$
df = (-\tfrac{1}{2}\lambda^2 f + \tfrac{1}{2}\lambda^2 f)\, dt + \lambda f\, dB_t = \lambda f\, dB_t.
$$

So $Z_t := f(t, B_t)$ satisfies $dZ_t = \lambda Z_t\, dB_t$, $Z_0 = 1$ — a drift-free Itô process and hence a local martingale. Because $E[Z_t] = 1$ for all $t$ (direct MGF), it is a true martingale. This is the **stochastic exponential** of $\lambda B$, denoted $\mathcal{E}(\lambda B)_t$.

### 4.3 Geometric Brownian motion

Let $S_t = S_0 \exp(\mu t + \sigma B_t)$. Applying Itô to $f(t, x) = S_0 e^{\mu t + \sigma x}$:
$$
dS_t = (\mu + \tfrac{1}{2}\sigma^2) S_t\, dt + \sigma S_t\, dB_t.
$$

Equivalently, $dS_t / S_t = (\mu + \tfrac{1}{2}\sigma^2)\, dt + \sigma\, dB_t$. **This is the famous GBM SDE** — the stock-price model for Black–Scholes. Note the drift shift: $\mu$ in the exponent becomes $\mu + \tfrac{1}{2}\sigma^2$ in the SDE.

### 4.4 Integration by parts

For $X, Y$ two Itô processes: applying multi-dimensional Itô to $f(x, y) = xy$ gives
$$
d(X_t Y_t) = X_t\, dY_t + Y_t\, dX_t + d\langle X, Y\rangle_t.
$$

The bonus term $d\langle X, Y\rangle$ is the **Itô correction to integration by parts**. For independent Brownians, it vanishes; for correlated, it is the quadratic covariation.

### 4.5 Ornstein–Uhlenbeck solution

For $dX_t = -\alpha X_t dt + \sigma dB_t$, apply Itô to $f(t, x) = e^{\alpha t} x$:
$$
d(e^{\alpha t} X_t) = \alpha e^{\alpha t} X_t dt + e^{\alpha t} dX_t = \alpha e^{\alpha t} X_t dt + e^{\alpha t}(-\alpha X_t dt + \sigma dB_t) = \sigma e^{\alpha t} dB_t.
$$

Integrating: $X_t = e^{-\alpha t} X_0 + \sigma \int_0^t e^{-\alpha(t-s)} dB_s$. This is the **Vasicek / OU short-rate solution**. Using the Itô isometry:
$$
\text{Var}(X_t) = \sigma^2 \int_0^t e^{-2\alpha(t-s)} ds = \frac{\sigma^2}{2\alpha}(1 - e^{-2\alpha t}) \xrightarrow{t\to\infty} \frac{\sigma^2}{2\alpha}.
$$

### 4.6 Tanaka's formula

The function $f(x) = |x|$ is not $C^2$ — $f''$ is a delta function at $0$. Still, a version of Itô's formula survives:
$$
|B_t| = \int_0^t \text{sgn}(B_s)\, dB_s + L_t,
$$
where $L_t$ is the **local time at $0$**, a nondecreasing continuous process that grows only when $B = 0$. This extends Itô to convex (and more generally, semi-convex) functions via the Meyer–Tanaka formula.

---

## 5. Lévy's characterization and Girsanov preview

**Recap from Module 3.1.** $M$ continuous local martingale with $M_0 = 0$ and $\langle M\rangle_t = t$ $\Rightarrow$ $M$ is a Brownian motion. Itô's lemma gives the MGF computation: $E[e^{i\theta M_t}] = e^{-\theta^2 t/2}$.

**Girsanov's theorem setup.** Suppose we want to change from $P$ to an equivalent measure $Q$ with $dQ/dP\bigr|_{\mathcal{F}_t} = Z_t$. We need $(Z_t)$ to be a positive $P$-martingale with $Z_0 = 1$. The natural candidate:
$$
Z_t = \exp\Bigl(\int_0^t \theta_s\, dB_s - \tfrac{1}{2}\int_0^t \theta_s^2\, ds\Bigr),
$$
where $\theta$ is predictable. By Itô's lemma,
$$
dZ_t = Z_t \theta_t\, dB_t,
$$

so $Z$ is a local martingale — and a true martingale under **Novikov's condition** $E[\exp(\tfrac{1}{2}\int_0^T \theta_s^2 ds)] < \infty$ (Section 5.3 below).

### 5.1 Girsanov's theorem

**Theorem 5.1 (Girsanov).** Let $\theta$ be predictable with $\int_0^T \theta_s^2 ds < \infty$ a.s. and let $Z_t$ as above. If $(Z_t)_{0 \le t \le T}$ is a true $P$-martingale with $E[Z_T] = 1$, define $Q$ by $dQ/dP = Z_T$. Then under $Q$, the process
$$
\tilde B_t := B_t - \int_0^t \theta_s\, ds
$$
is a standard Brownian motion on $[0, T]$.

**Sketch of proof.** By Lévy's characterization (Module 3.1, Theorem 6.1), it suffices to show that $\tilde B$ is a continuous $Q$-local martingale with $\langle \tilde B\rangle_t = t$.

- **Quadratic variation is preserved** under equivalent measure changes (change of measure does not change null sets, and QV is defined via $P$-a.s. limits which remain $Q$-a.s. limits). So $\langle \tilde B\rangle_t = \langle B - \int \theta ds\rangle_t = \langle B\rangle_t = t$.
- **$\tilde B$ is a $Q$-local martingale.** A process $M$ is a $Q$-local martingale iff $M Z$ is a $P$-local martingale (Bayes rule for conditional expectation under measure change). Apply Itô's product rule:
$$
d(\tilde B_t Z_t) = \tilde B_t\, dZ_t + Z_t\, d\tilde B_t + d\langle \tilde B, Z\rangle_t.
$$

Now $d\tilde B_t = dB_t - \theta_t dt$, $dZ_t = Z_t \theta_t dB_t$, so $d\langle \tilde B, Z\rangle_t = Z_t \theta_t\, dt$. Substituting:
$$
d(\tilde B_t Z_t) = \tilde B_t Z_t \theta_t dB_t + Z_t(dB_t - \theta_t dt) + Z_t \theta_t dt = (Z_t + \tilde B_t Z_t \theta_t)\, dB_t.
$$

This is a $P$-local martingale (drift-free), so $\tilde B$ is a $Q$-local martingale. Combined with quadratic variation, Lévy gives $\tilde B$ a $Q$-Brownian motion. $\square$

### 5.2 Risk-neutral pricing: Girsanov in action

Consider $dS_t = \mu S_t dt + \sigma S_t dB_t$ under $P$. Define $\theta = -\frac{\mu - r}{\sigma}$ (the **market price of risk**). Then
$$
d\tilde B_t = dB_t - \theta dt = dB_t + \tfrac{\mu - r}{\sigma} dt,
$$
and
$$
dS_t = \mu S_t dt + \sigma S_t\bigl(d\tilde B_t - \tfrac{\mu-r}{\sigma}dt\bigr) = r S_t dt + \sigma S_t\, d\tilde B_t.
$$

Under $Q$, the stock has drift $r$ (the risk-free rate), and discounted stock $\tilde S_t = e^{-rt}S_t$ is a $Q$-martingale. **This is the First Fundamental Theorem of Asset Pricing in its simplest Brownian form.**

### 5.3 Novikov's condition

**Theorem 5.2 (Novikov).** If $E[\exp(\tfrac{1}{2}\int_0^T \theta_s^2 ds)] < \infty$, then $Z_t = \exp(\int_0^t \theta dB - \tfrac{1}{2}\int_0^t \theta^2 ds)$ is a true $P$-martingale on $[0, T]$.

*Proof sketch.* $Z$ is a nonnegative local martingale, hence a supermartingale (Fatou), and $E[Z_T] \le 1$. Equality is equivalent to the martingale property. Using the identity
$$
Z_t = \exp(\int_0^t \theta dB - \int_0^t \theta^2 ds) \cdot \exp(\tfrac{1}{2}\int_0^t \theta^2 ds),
$$
apply Cauchy–Schwarz to bound $E[Z_\tau]$ at stopping times, then use localization and dominated convergence. Full details in Karatzas–Shreve, Proposition 3.5.12. $\square$

---

## 6. Connection to PDEs: Feynman–Kac

Itô's lemma bridges probability and PDEs. For an SDE $dX_t = \mu(X_t)dt + \sigma(X_t) dB_t$, define the **generator**
$$
\mathcal{L} f(x) := \mu(x) f'(x) + \tfrac{1}{2}\sigma(x)^2 f''(x).
$$

**Theorem 6.1 (Feynman–Kac).** Let $u(t, x)$ solve the terminal-value PDE
$$
\partial_t u + \mathcal{L} u - r(x) u = 0, \qquad u(T, x) = g(x),
$$
with regularity and growth conditions. Then
$$
u(t, x) = E\Bigl[ \exp\Bigl(-\int_t^T r(X_s) ds\Bigr) g(X_T)\Big|\, X_t = x\Bigr].
$$

*Proof.* Apply Itô's formula to $M_s := e^{-\int_t^s r(X_u) du} u(s, X_s)$:
$$
dM_s = e^{-\int_t^s r du}\Bigl[\bigl(\partial_t u + \mathcal{L} u - r u\bigr) ds + \sigma u_x dB_s\Bigr] = e^{-\int_t^s r du} \sigma u_x\, dB_s,
$$

using the PDE to cancel the drift. So $M$ is a local martingale; under growth conditions a true martingale, giving $E[M_T | \mathcal{F}_t] = M_t$, i.e., $u(t, x) = E[e^{-\int_t^T r ds} g(X_T) | X_t = x]$. $\square$

**Quant application.** The Black–Scholes PDE
$$
\partial_t C + rS \partial_S C + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} C - r C = 0, \qquad C(T, S) = (S - K)^+,
$$

corresponds via Feynman–Kac to
$$
C(t, S) = e^{-r(T-t)} E^Q\bigl[(S_T - K)^+ | S_t = S\bigr],
$$

where $S$ is GBM with drift $r$ under $Q$ (risk-neutral measure). Evaluating the Gaussian expectation yields the **Black–Scholes formula**.

---

## 7. Martingale representation theorem

The **martingale representation theorem** is the converse of stochastic integration: every $\mathcal{F}_t$-martingale (in the Brownian filtration) is an Itô integral.

**Theorem 7.1 (Martingale representation, Itô).** Let $\mathbb{F}^B$ be the augmented natural filtration of Brownian motion on $[0, T]$. Every $L^2$ $\mathbb{F}^B$-martingale $(M_t)$ has a unique predictable representation
$$
M_t = M_0 + \int_0^t H_s\, dB_s, \qquad H \in \mathcal{L}^2.
$$

*Sketch.* Use Wiener chaos decomposition: $L^2(\mathcal{F}_T)$ decomposes into orthogonal chaos spaces $\mathcal{H}_n$, each spanned by iterated Itô integrals $\int \int \cdots \int dB \cdots dB$. Every martingale in this space has an Itô representation. See Nualart, *The Malliavin Calculus*, for details.

### 7.1 Clark–Ocone formula

For sufficiently smooth random variables $F \in L^2(\mathcal{F}_T)$, the predictable integrand is explicitly
$$
H_s = E[D_s F | \mathcal{F}_s],
$$
where $D_s F$ is the **Malliavin derivative**. This is the **Clark–Ocone formula**, a key tool for:

- Computing hedging strategies for general payoffs.
- Computing Greeks via integration by parts on Wiener space.
- Sensitivity analysis in finance.

For example, the BS delta hedge $\Delta_t = \partial_S C(t, S_t)$ is a Malliavin-derivative expression. Clark–Ocone provides the machinery to compute such hedges even for path-dependent payoffs (Asian, lookback).

---

## 8. Python: verifying Itô's formula

```python
import numpy as np

rng = np.random.default_rng(0)
T, n, M = 1.0, 20_000, 3_000
dt = T/n
times = np.linspace(0, T, n+1)

# ---- Simulate many Brownian paths ----
dW = rng.standard_normal(size=(M, n)) * np.sqrt(dt)
W  = np.concatenate([np.zeros((M,1)), np.cumsum(dW, axis=1)], axis=1)

# ---- (a) Verify B_t^2 = 2∫B dB + t ----
lhs = W[:, -1]**2
ito_int = np.sum(W[:, :-1] * dW, axis=1)
rhs = 2 * ito_int + T
print("Check B_T^2 = 2∫B dB + T (should be zero residual):")
print("  mean residual:", np.mean(lhs - rhs), "  std:", np.std(lhs - rhs))

# ---- (b) Verify stochastic exponential E[Z_T] = 1 ----
lam = 0.5
Z_T = np.exp(lam * W[:, -1] - 0.5 * lam**2 * T)
print("\nE[Z_T] stochastic exp (should be 1):", Z_T.mean())

# ---- (c) Verify OU variance closed form ----
alpha, sigma = 2.0, 1.0
X = np.zeros((M, n+1))
for k in range(n):
    X[:, k+1] = X[:, k] + (-alpha * X[:, k]) * dt + sigma * dW[:, k]
print("\nOU var(X_T):", X[:, -1].var())
print("Theory σ²/(2α)(1-e^{-2αT}):", sigma**2/(2*alpha) * (1 - np.exp(-2*alpha*T)))

# ---- (d) Verify Black-Scholes price via Girsanov / risk-neutral expectation ----
S0, K, r, sigma_bs = 100.0, 100.0, 0.05, 0.2
# risk-neutral stock: dS = r S dt + σ S dB, so S_T = S_0 exp((r-σ²/2)T + σB_T)
S_T = S0 * np.exp((r - 0.5 * sigma_bs**2) * T + sigma_bs * W[:, -1])
mc_price = np.exp(-r*T) * np.maximum(S_T - K, 0.0).mean()
mc_se    = np.exp(-r*T) * np.maximum(S_T - K, 0.0).std() / np.sqrt(M)
# Analytic BS
from math import log, sqrt, exp
from scipy.stats import norm
d1 = (log(S0/K) + (r + 0.5*sigma_bs**2)*T) / (sigma_bs*sqrt(T))
d2 = d1 - sigma_bs*sqrt(T)
bs_price = S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
print("\nBS MC price:", mc_price, "±", 2*mc_se)
print("BS analytic: ", bs_price)

# ---- (e) Verify Feynman-Kac: heat equation solution via BM ----
# u(t, x) = E[g(x + B_{T-t})], with u_t + 1/2 u_xx = 0, u(T, x) = g(x)
def g(x): return (x - 0.0)**2   # e.g. squared payoff; exact: u(0,x) = x^2 + T
x0 = 0.5
u_mc = (g(x0 + np.sqrt(T) * rng.standard_normal(M))).mean()
u_analytic = x0**2 + T
print("\nFeynman-Kac MC:", u_mc, "  analytic:", u_analytic)
```

### Expected output commentary

- **(a)** residual `~1e-4` (bounded by discretization error $O(\sqrt{dt})$ of the Itô sum). The identity is provably exact; the numerical residual reflects partition error.
- **(b)** `E[Z_T]` within $\pm 0.005$ of $1$; variance of stochastic exponential is bounded but nonzero.
- **(c)** OU variance matches $\sigma^2/(2\alpha)(1-e^{-2\alpha T})$ to three decimals.
- **(d)** MC BS price matches analytic to two decimals at $M = 3000$.
- **(e)** Feynman–Kac for heat equation: $u(0, x_0) = x_0^2 + T = 0.25 + 1 = 1.25$, MC should match.

The Python confirms everything we've proven: Itô's identity $B_T^2 = 2\int B dB + T$, stochastic exp is a martingale, OU closed form, BS via MC = BS analytic, heat equation via BM.

---

## 9. [QUANT APPLICATION] — Itô's formula throughout finance

### 9.1 Deriving the Black–Scholes PDE

Consider a portfolio $\Pi_t = C(t, S_t) - \Delta_t S_t$, short one call and long $\Delta_t$ shares of stock (self-financing). Itô applied to $C(t, S_t)$:
$$
dC = \partial_t C\, dt + \partial_S C\, dS + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} C\, dt.
$$

For $\Pi$ to be deterministic (riskless), set $\Delta_t = \partial_S C$:
$$
d\Pi = \partial_t C\, dt + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} C\, dt.
$$

No arbitrage: $d\Pi = r \Pi dt = r(C - S \partial_S C) dt$. Equating gives the **Black–Scholes PDE**:
$$
\partial_t C + rS \partial_S C + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} C = r C.
$$

Itô's formula, and specifically the $\tfrac{1}{2}\sigma^2 S^2 \partial_{SS}$ term, is what **creates** the gamma term in this PDE. Without the Itô correction, you get the classical PDE $\partial_t C + \mu S \partial_S C = 0$, which has no volatility and no option price.

### 9.2 Implied volatility surface and volatility risk

The gamma $\Gamma = \partial_{SS} C$ is the PnL of a delta-hedged book times $\tfrac{1}{2}\sigma^2 S^2\, dt$:
$$
d(\text{hedged PnL}) = \tfrac{1}{2}\Gamma (\sigma_{\text{real}}^2 - \sigma_{\text{impl}}^2) S^2\, dt.
$$

This is how dispersion traders, volatility arbitrageurs, and vol market makers extract PnL. The **key insight** is purely Itô: the second-order term in Itô's lemma is exactly the channel through which realized vs implied volatility PnL flows.

### 9.3 Log-SDE transformation (Lamperti transform)

For $dS_t = \mu(S_t) dt + \sigma(S_t) dB_t$ with state-dependent vol, define $Y_t = \int_{S_0}^{S_t} \frac{du}{\sigma(u)}$. Applying Itô:
$$
dY_t = \frac{1}{\sigma(S_t)} dS_t - \tfrac{1}{2}\frac{\sigma'(S_t)}{\sigma(S_t)^2} \sigma(S_t)^2 dt = \Bigl[\frac{\mu(S_t)}{\sigma(S_t)} - \tfrac{1}{2}\sigma'(S_t)\Bigr] dt + dB_t.
$$

Now the new process $Y$ has **unit diffusion**, simplifying Euler schemes, CIR calibration, Heston, and (importantly) giving Feller / transition-density conditions on the original SDE. For GBM, this is the familiar $d(\log S_t) = (r - \sigma^2/2)dt + \sigma dB_t$.

### 9.4 Volatility as realized variance: the "volatility swap"

A variance swap pays $\sigma_{\text{realized}}^2 - K_{\text{var}}$ where
$$
\sigma_{\text{realized}}^2 = \frac{1}{T}\int_0^T \sigma_s^2\, ds.
$$

By Itô on $\log S_t$,
$$
d\log S_t = \bigl(r - \tfrac{1}{2}\sigma_t^2\bigr)dt + \sigma_t dB_t.
$$

Rearranging and integrating gives the famous **model-free replication**:
$$
\frac{1}{T}\int_0^T \sigma_t^2 dt = \frac{2}{T}\bigl[\log(S_0/S_T) + \int_0^T dS_t/S_t\bigr],
$$

which decomposes a variance swap into a strip of options at all strikes (the Neuberger / Demeterfi–Derman–Kamal replication). Itô's formula is the hinge of this entire result.

### 9.5 Greeks via martingale / Malliavin representation

From martingale representation: for a European-style payoff $\phi(S_T)$,
$$
\phi(S_T) = E[\phi(S_T)] + \int_0^T H_s\, dB_s,
$$

and the hedging strategy in stock is $\Delta_t = H_t / (\sigma S_t)$. Applying Itô to a price process $C(t, S_t)$ and matching with this representation gives $\Delta_t = \partial_S C(t, S_t)$ — **reconciling the PDE-based delta with the martingale-representation-based hedge**.

For path-dependent payoffs (Asian, lookback, barrier), the martingale representation expresses the hedging as a conditional expectation of Malliavin derivatives, not a simple state-space derivative.

---

## 10. Summary

- **Itô's formula** $df(t, X) = \partial_t f\, dt + \partial_x f\, dX + \tfrac{1}{2}\partial_{xx} f\, d\langle X\rangle$ is the chain rule for stochastic processes.
- The extra $\tfrac{1}{2}\partial_{xx} f\, d\langle X\rangle$ term is the **Itô correction**, arising because quadratic variation of the driver is nonzero.
- Multi-dimensional extension uses the Hessian contracted against $\sigma\sigma^T$.
- **Girsanov's theorem** changes the drift of Brownian motion via an equivalent measure change with Radon–Nikodym density $Z_T = \exp(\int \theta dB - \tfrac{1}{2}\int \theta^2 ds)$.
- **Feynman–Kac** converts backward PDEs to expectations of SDE terminal functionals.
- **Martingale representation theorem (Itô)**: every $L^2$ Brownian-adapted martingale is an Itô integral; **Clark–Ocone** gives the integrand via Malliavin calculus.

Between Itô, Girsanov, and Feynman–Kac, essentially every continuous-time finance identity is a two-line application.

---

## 11. Exercises

**★ (warm-ups)**

**11.1** Verify Itô's formula for $f(x) = e^{\lambda x}$ applied to $B_t$:
$$
e^{\lambda B_t} = 1 + \lambda\int_0^t e^{\lambda B_s}dB_s + \tfrac{1}{2}\lambda^2\int_0^t e^{\lambda B_s} ds.
$$

**11.2** Derive the SDE for $Y_t = B_t^3$ using Itô and verify the martingale correction term.

**11.3** For GBM $dS/S = \mu dt + \sigma dB$, apply Itô to $\log S$ and derive $d\log S = (\mu - \sigma^2/2) dt + \sigma dB$.

**11.4** Apply Itô's product rule to $X_t Y_t$ where $X = B^1, Y = B^2$ are independent Brownians. Show $d(B^1_t B^2_t) = B^1 dB^2 + B^2 dB^1$.

**11.5** Apply Itô to $f(B^1, B^2) = (B^1)^2 + (B^2)^2$ and show $(B^1_t)^2 + (B^2_t)^2 - 2t$ is a martingale.

**★★ (core techniques)**

**11.6** **(CIR process.)** Let $dX_t = \kappa(\theta - X_t)dt + \sigma\sqrt{X_t}\, dB_t$. Show that $Y_t = X_t e^{\kappa t}$ satisfies
$$
dY_t = \kappa\theta e^{\kappa t} dt + \sigma\sqrt{X_t} e^{\kappa t} dB_t.
$$
Hence derive an integral form for $X_t$.

**11.7** **(Ornstein–Uhlenbeck bridge.)** Define $Y_t = (1 - t/T) X + (t/T) Z + \sigma W^0_t$, where $W^0$ is a Brownian bridge and $X, Z$ are constants. Show $Y_0 = X$ a.s. and $Y_T = Z$ a.s., and compute $\text{Cov}(Y_s, Y_t)$.

**11.8** **(Gaussian integral via Itô.)** Using $d(e^{-t/2}\cos B_t) = \ldots$, prove that $e^{-t/2}\cos B_t$ is a martingale. Compute its expectation to rederive $E[\cos B_t] = e^{-t/2}$.

**11.9** **(Itô product with drift.)** Let $X_t, Y_t$ be Itô processes. Prove
$$
d(X_t Y_t) = X_t dY_t + Y_t dX_t + d\langle X, Y\rangle_t.
$$
Show this recovers classical product rule when either $X$ or $Y$ has bounded variation (so covariation vanishes).

**11.10** **(Novikov explicit.)** Let $\theta_s = \lambda$ constant. Verify $E[\exp(\tfrac{1}{2}\lambda^2 T)] < \infty$ and derive that $Z_t = \exp(\lambda B_t - \tfrac{1}{2}\lambda^2 t)$ is a martingale, confirming Section 5.3 in this case.

**11.11** **(Girsanov for constant drift.)** Let $\theta$ be a constant. Under $Q = Z_T \cdot P$, show that $\tilde B_t = B_t - \theta t$ is a $Q$-Brownian motion.

**11.12** **(Itô vs Stratonovich conversion.)** For $f \in C^2$, prove
$$
\int_0^T f(B_s) \circ dB_s = \int_0^T f(B_s) dB_s + \tfrac{1}{2}\int_0^T f'(B_s) ds.
$$

**11.13** **(Tanaka's formula.)** Show that $|B_t| - \int_0^t \text{sgn}(B_s)\, dB_s = L_t^0$ is nondecreasing and nonzero only on $\{s : B_s = 0\}$. (Full local-time theory in Karatzas–Shreve §3.6.)

**11.14** **(Feynman–Kac for OU.)** For $dX_t = -\alpha X_t dt + \sigma dB_t$, find the stationary density $p^*(x)$. Check it solves the Fokker–Planck PDE $\partial_t p = \alpha \partial_x(xp) + \tfrac{1}{2}\sigma^2 \partial_{xx}p$.

**11.15** **(Heat equation.)** Apply Itô to $u(t, x) = E[g(x + B_{T-t})]$ and verify $\partial_t u + \tfrac{1}{2}\partial_{xx}u = 0$. This reproduces the heat equation.

**★★★ (research-level / quant)**

**11.16** **(Heston SV model.)** Consider
$$
dS_t = \mu S_t dt + \sqrt{V_t}\, S_t dB_t^{(1)}, \qquad dV_t = \kappa(\theta - V_t) dt + \xi\sqrt{V_t}\, dB_t^{(2)},
$$
with $d\langle B^{(1)}, B^{(2)}\rangle_t = \rho\, dt$. Using Itô's formula, derive the SDE for $\log S_t$ and identify the Heston PDE via Feynman–Kac.

**11.17** **(Black–Scholes PDE from Itô.)** Following Section 9.1, complete the derivation of the BS PDE from Itô's lemma on $C(t, S)$ with $S$ = GBM. Write out the cancellation of the $dB$-term (delta hedge) and the resulting ODE in $t$.

**11.18** **(Bachelier formula.)** In the Bachelier model $dS_t = \sigma dB_t$ ($S$ is normal, not lognormal), apply Itô to price a call. Derive the closed form
$$
C(t, S) = (S - K) \Phi(d) + \sigma\sqrt{T-t}\, \phi(d), \quad d = (S-K)/(\sigma\sqrt{T-t}).
$$

**11.19** **(Quadratic hedging error.)** Consider discrete rebalancing of a delta hedge with $n$ equally-spaced times. Using Itô's formula and a Taylor expansion, derive the **Leland** / **Hayne Leland** formula for the leading-order PnL variance $\sim \sigma^4 S^2 \Gamma^2 (T-t)^2/(2n)$. Sketch why this is the right order.

**11.20** **(Malliavin derivative, explicit case.)** For $F = \int_0^T f(s)\, dB_s$ deterministic $f$, compute $D_s F = f(s) \mathbf{1}_{[0, T]}(s)$ using chaos expansion. Verify Clark–Ocone: $H_s = E[D_s F | \mathcal{F}_s] = f(s)$, matching the Itô representation.

**11.21** **(Exponential Itô formula.)** Let $X$ be an Itô process. Define $\mathcal{E}(X)_t = \exp(X_t - \tfrac{1}{2}\langle X\rangle_t)$. Prove that if $X$ is a continuous local martingale starting at $0$, then $\mathcal{E}(X)_t$ is a continuous local martingale with $d\mathcal{E}(X)_t = \mathcal{E}(X)_t dX_t$.

**11.22** **(Quadratic covariation and two-factor models.)** In a two-factor short-rate model
$$
dr_t = \alpha(b - r_t) dt + \sigma dB_t^{(1)}, \qquad db_t = \gamma db dt + \eta dB_t^{(2)},
$$
with $d\langle B^{(1)}, B^{(2)}\rangle = \rho dt$, derive the bond price $P(t, T) = E[e^{-\int_t^T r_s ds} | \mathcal{F}_t]$ using Itô and Feynman–Kac. Show this is affine: $P(t, T) = \exp(A(t, T) - B(t, T) r_t - C(t, T) b_t)$ with ODEs for $A, B, C$.

---

## 12. Forward pointers

- **Module 3.4 (SDEs).** Existence & uniqueness, Picard iteration, comparison theorems, Feller property, Kolmogorov equations, Fokker–Planck.
- **Module 3.5 (Lévy processes).** Add jumps: Lévy–Itô decomposition, jump-Itô formula with additional $[f(x + J) - f(x) - f'(x)J]$ integrator terms.
- **Module 3.6 (Continuous-time Markov processes).** Semigroups, generators, Hille–Yosida characterization.
- **Module 3.7 (Continuous-time martingales).** Doob–Meyer, BDG, martingale representation in full generality.

The Itô formula we have proven today is the single most important technical identity in continuous-time finance. Every subsequent module either applies it, generalizes it, or uses it as a reference point.

**Next module:** SDEs — existence and uniqueness of solutions, connection to PDEs, and Monte Carlo simulation.

---

*End of Module 3.3.*

# Module 3.5 — Lévy Processes and Jump Diffusions

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 3 (Stochastic Processes), Module 5 of 7

---

## Prerequisites

- **Module 2.5** (Characteristic functions, Lévy–Khintchine formula, infinite divisibility, stable distributions).
- **Module 3.1–3.4** (Brownian motion, Itô integral, Itô's formula, SDE existence).
- **Module 2.7** (Poisson processes in the continuous-time Markov chain setting).

Real markets don't move continuously. Earnings surprises, central-bank announcements, bankruptcy events, flash crashes, and lower-liquidity regimes all create jumps. Lévy processes and jump-diffusions are the framework for modeling these phenomena.

---

## 1. Lévy processes: definition

**Definition 1.1.** A **Lévy process** $X = (X_t)_{t \ge 0}$ is a càdlàg stochastic process (right-continuous with left limits) on $\mathbb{R}^d$ with:

- **(L1) $X_0 = 0$ a.s.**
- **(L2) Independent increments:** For $0 \le t_1 < t_2 < \dots < t_n$, the increments $X_{t_2} - X_{t_1}, \ldots, X_{t_n} - X_{t_{n-1}}$ are independent.
- **(L3) Stationary increments:** $X_{t + h} - X_t \stackrel{d}{=} X_h$ for all $h \ge 0$.
- **(L4) Stochastic continuity:** $P(|X_{t + h} - X_t| > \varepsilon) \to 0$ as $h \to 0$, for every $\varepsilon > 0$.

**Examples.**

- **Brownian motion** (continuous paths, Gaussian increments).
- **Poisson process** with rate $\lambda$ (unit jumps at Poisson times).
- **Compound Poisson** $X_t = \sum_{k=1}^{N_t} \xi_k$, where $N$ is Poisson($\lambda$) and $\xi_k$ iid.
- **$\alpha$-stable process** with stable characteristic exponent.
- **Subordinators** (nondecreasing Lévy processes, e.g., Gamma, inverse Gaussian).
- **Variance Gamma (VG) process.**
- **Normal Inverse Gaussian (NIG) process.**

The defining property is that the distribution of $X_t$ is **infinitely divisible** (Module 2.5). This leads to the fundamental structure theorem.

---

## 2. The Lévy–Khintchine formula and Lévy–Itô decomposition

**Theorem 2.1 (Lévy–Khintchine).** Every Lévy process $X$ in $\mathbb{R}^d$ has characteristic function
$$
E[e^{i\langle u, X_t\rangle}] = \exp(t\, \psi(u)),
$$
with characteristic exponent
$$
\psi(u) = i\langle b, u\rangle - \tfrac{1}{2}\langle u, A u\rangle + \int_{\mathbb{R}^d \setminus \{0\}}\bigl(e^{i\langle u, y\rangle} - 1 - i\langle u, y\rangle \mathbf{1}_{|y| \le 1}\bigr)\, \nu(dy),
$$
where:

- $b \in \mathbb{R}^d$ is the **drift**,
- $A$ is a positive semi-definite $d \times d$ matrix (**diffusion covariance**),
- $\nu$ is a $\sigma$-finite **Lévy measure** on $\mathbb{R}^d \setminus \{0\}$ with $\int \min(1, |y|^2) \nu(dy) < \infty$.

The triple $(b, A, \nu)$ is the **Lévy triplet** and uniquely determines the law of the process.

### 2.1 The three building blocks

- **Drift $b$**: deterministic linear trend.
- **Gaussian part (governed by $A$)**: scaled Brownian motion.
- **Jump part (governed by $\nu$)**: jumps with intensity $\nu(dy)$ — expected number of jumps in $[0, t] \times A$ is $t \nu(A)$ for $A \subset \mathbb{R}^d \setminus \{0\}$.

The condition $\int \min(1, |y|^2) \nu(dy) < \infty$ permits **infinitely many small jumps** (accumulation at $0$) while forbidding infinite-rate large jumps.

### 2.2 Lévy–Itô decomposition (structure theorem)

**Theorem 2.2 (Lévy–Itô).** Every Lévy process $X$ with triplet $(b, A, \nu)$ can be written as
$$
X_t = b t + \Sigma W_t + \int_{|y| \ge 1} y\, N_t(dy) + \int_{|y| < 1} y\, \tilde N_t(dy),
$$
where:

- $W$ is a standard Brownian motion,
- $\Sigma\Sigma^T = A$,
- $N(dt, dy)$ is a Poisson random measure on $[0, \infty) \times (\mathbb{R}^d \setminus \{0\})$ with intensity $dt \otimes \nu(dy)$,
- $\tilde N(dt, dy) = N(dt, dy) - dt\, \nu(dy)$ is the **compensated** Poisson random measure (a martingale measure).

The four summands are mutually independent. The large-jump integral $\int_{|y| \ge 1} y\, N_t(dy)$ is a compound Poisson process; the small-jump compensated integral is a square-integrable martingale. Together: every Lévy process = drift + Brownian + compound Poisson + compensated small jumps.

### 2.3 Proof of Lévy–Itô (sketch)

Fix a Lévy process $X$. The jumps of $X$ are **countable** (a càdlàg process has countably many discontinuities) and their "size distribution" defines the Poisson random measure $N$. Specifically,
$$
N_t(A) = \#\{s \in (0, t] : \Delta X_s \in A\}, \qquad A \subset \mathbb{R}^d \setminus \{0\},
$$

where $\Delta X_s = X_s - X_{s^-}$. Stationary and independent increments of $X$ imply $N_t(A)$ is Poisson with mean $t \nu(A)$, and different sets of jump sizes are independent.

Separate large jumps ($|y| \ge 1$) and small jumps ($|y| < 1$). Large jumps are finitely many (Poisson); small jumps need compensation. Subtracting the Poisson compensators and what remains after removing these jump contributions is a continuous Lévy process with the same independent-increment structure — hence a Brownian motion with drift. $\square$

---

## 3. Important examples

### 3.1 Poisson process

$N_t \sim \text{Poisson}(\lambda t)$. Triplet: $b = \lambda \int_{|y| \le 1} y\, \delta_1(dy) = \lambda$ (if we take $b$ including the compensator for unit jumps… depending on cut-off choice), $A = 0$, $\nu = \lambda \delta_1$ (a point mass at $1$).

### 3.2 Compound Poisson

$X_t = \sum_{k=1}^{N_t} \xi_k$, $\xi_k \sim F$ iid. Lévy measure $\nu(dy) = \lambda F(dy)$. If $E[|\xi|] < \infty$, $X$ is of finite variation; if additionally $E[\xi] = 0$, the process is a martingale.

### 3.3 Merton's jump-diffusion model

$$
S_t = S_0 \exp\Bigl(\Bigl(\mu - \tfrac{1}{2}\sigma^2 - \lambda\kappa\Bigr) t + \sigma B_t + \sum_{k=1}^{N_t} Y_k\Bigr),
$$

where $Y_k \sim N(\mu_J, \sigma_J^2)$ iid, $N$ is Poisson($\lambda$), and $\kappa = E[e^Y - 1]$ compensates the drift so that $e^{-rt} S_t$ is a martingale under risk-neutral measure.

### 3.4 Kou's double-exponential model

Jump sizes $Y$ have density
$$
f_Y(y) = p\, \eta_+ e^{-\eta_+ y} \mathbf{1}_{y > 0} + (1-p) \eta_- e^{\eta_- y} \mathbf{1}_{y < 0},
$$

asymmetric (larger downward jumps than upward, matching equity empirics). Allows closed-form option prices via Laplace transforms.

### 3.5 Variance Gamma (VG)

$X_t = \mu G_t + \sigma W_{G_t}$, with $G_t$ a Gamma subordinator and $W$ independent Brownian motion. **Pure jump process** (no Brownian part) with activity controlled by $G$. Has closed-form characteristic function:
$$
E[e^{iu X_t}] = \bigl(1 - i u \mu \nu + \tfrac{1}{2}\sigma^2 u^2 \nu\bigr)^{-t/\nu}.
$$

Used in Madan–Seneta and Carr–Madan option pricing.

### 3.6 Normal Inverse Gaussian (NIG)

$X_t = \mu t + \sigma B_{T_t} + \delta T_t$, with $T_t$ an inverse Gaussian subordinator. NIG has a closed-form Lévy density involving modified Bessel functions.

### 3.7 $\alpha$-stable processes

Lévy with triplet having $\nu(dy) = c_\pm |y|^{-1-\alpha} dy$ on positive / negative axes, $\alpha \in (0, 2)$. Heavy-tailed: $E[|X_t|^\beta] < \infty$ iff $\beta < \alpha$.

---

## 4. Itô's formula for jump processes

Let $X$ be a Lévy (or more generally, a semi-martingale with jumps). The **jump-Itô formula** for $f \in C^{1, 2}$ is

$$
f(t, X_t) = f(0, X_0) + \int_0^t \partial_t f(s, X_{s^-}) ds + \int_0^t \partial_x f(s, X_{s^-}) dX_s^c + \tfrac{1}{2}\int_0^t \partial_{xx} f(s, X_{s^-}) d\langle X^c\rangle_s
$$
$$
+ \sum_{0 < s \le t}\bigl[f(s, X_s) - f(s, X_{s^-}) - \partial_x f(s, X_{s^-}) \Delta X_s\bigr].
$$

where $X^c$ is the continuous part. The jump sum accounts for **all** the jumps — the linearization of $f$ plus the quadratic Itô term is no longer correct when jumps are present; the error is exactly $[f(\text{after}) - f(\text{before}) - f'(\text{before}) \cdot \Delta X]$.

**Compensated form.** For a Lévy process with Lévy measure $\nu$,
$$
f(t, X_t) = f(0, X_0) + \int_0^t [\partial_t f + b\partial_x f + \tfrac{1}{2}A \partial_{xx} f] ds + \sigma \int_0^t \partial_x f\, dW_s
$$
$$
+ \int_0^t\int_\mathbb{R} [f(s, X_{s^-} + y) - f(s, X_{s^-})]\, \tilde N(ds, dy) + \int_0^t\int_\mathbb{R} [f(s, X_{s^-} + y) - f(s, X_{s^-}) - y\partial_x f(s, X_{s^-})]\, \nu(dy) ds.
$$

The first integro-differential term is a martingale (compensated); the second is the deterministic drift correction from the jumps.

---

## 5. Generators and PIDEs

The infinitesimal generator of a Lévy process:
$$
\mathcal{L} f(x) = b\cdot\nabla f(x) + \tfrac{1}{2}\text{tr}(A \nabla^2 f(x)) + \int[f(x + y) - f(x) - \nabla f(x) \cdot y\mathbf{1}_{|y|\le 1}] \nu(dy).
$$

This is an **integro-differential operator** combining local (differential) and nonlocal (integral) parts.

### 5.1 Feynman–Kac for jump processes

For $u(t, x) = E[g(X_T)|X_t = x]$, the PDE becomes a **partial integro-differential equation (PIDE)**:
$$
\partial_t u + \mathcal{L} u = 0, \qquad u(T, x) = g(x).
$$

Solved numerically by finite-difference + quadrature or by Fourier inversion (Carr–Madan). Option prices under jump-diffusion models satisfy such PIDEs.

### 5.2 Merton's formula (jump-diffusion option pricing)

Under Merton's model with jumps $Y \sim N(\mu_J, \sigma_J^2)$, conditioning on the number of jumps $N_T = n$:
$$
C_{\text{Merton}}(S_0, K, T) = \sum_{n=0}^\infty \frac{e^{-\lambda' T}(\lambda' T)^n}{n!}\, C_{BS}(S_0, K, T, r_n, \sigma_n),
$$

where $\lambda' = \lambda(1 + \kappa)$ and $\sigma_n^2 = \sigma^2 + n\sigma_J^2/T$, $r_n = r - \lambda\kappa + n\mu_J/T$.

---

## 6. Girsanov's theorem for jump processes

Changing measure changes both the drift of the Brownian and the jump intensity (and size distribution).

**Theorem 6.1 (Girsanov for Lévy).** Let $X$ be a Lévy process with triplet $(b, A, \nu)$. Choose a predictable process $\theta$ for the Brownian drift and a predictable $Y(\cdot, \cdot)$ satisfying $Y(t, y) > -1$ and $\int \int |Y(t, y)|^2 \nu(dy) ds < \infty$.

Define
$$
Z_t = \exp\Bigl(\int_0^t \theta dB - \tfrac{1}{2}\int_0^t \theta^2 ds + \int_0^t\int Y(s, y)\tilde N(ds, dy) - \int_0^t\int (e^{Y(s, y)} - 1 - Y(s, y))\nu(dy) ds\Bigr).
$$

Under Novikov-like conditions, $Z$ is a true martingale with $E[Z_T] = 1$, and under $Q = Z_T \cdot P$:

- $\tilde B_t = B_t - \int_0^t \theta ds$ is a $Q$-Brownian motion.
- The jumps of $X$ under $Q$ have intensity $e^{Y(s, y)} \nu(dy)$.

Hence changes of measure in Lévy / jump-diffusion worlds affect jump intensity, not just drift.

### 6.1 Market incompleteness

With jumps, the market is **incomplete**: there are multiple equivalent martingale measures $Q$, hence multiple arbitrage-free option prices. This is quite different from Brownian models (BS) which are complete. Choosing the "right" $Q$ requires external information: **calibration** to market prices, or **utility-indifference** principles, or equivalence-preserving principles like the **Esscher transform** (minimize entropy relative to $P$).

---

## 7. Subordinators and stochastic time change

A **subordinator** is a nondecreasing Lévy process. Examples: Gamma, Inverse Gaussian, Poisson. If $T_t$ is a subordinator, then for Brownian motion $W$ independent of $T$, the process $X_t := W_{T_t}$ is a **subordinated Brownian motion**.

Subordinators capture the idea that **market time flows non-uniformly**: when many trades or events occur in wall-clock time $t$, market time $T_t$ advances rapidly; in quiet periods, slowly.

**VG example.** $T$ a Gamma subordinator, $X_t = \mu T_t + \sigma W_{T_t}$. The result is a pure-jump process with finite variation; widely used for options pricing.

**NIG example.** $T$ is Inverse Gaussian, $X_t = \mu T_t + \sigma W_{T_t}$.

---

## 8. Jump-diffusion SDEs

Consider
$$
dX_t = \mu(X_{t^-}) dt + \sigma(X_{t^-}) dB_t + \int_\mathbb{R} \gamma(X_{t^-}, y) \tilde N(dt, dy),
$$

with Lipschitz coefficients. Existence and uniqueness proceed as in Module 3.4, but with an additional jump term. The generator is
$$
\mathcal{L} f = \mu f' + \tfrac{1}{2}\sigma^2 f'' + \int [f(x + \gamma(x, y)) - f(x) - \gamma(x, y) f'(x)]\nu(dy).
$$

### 8.1 Merton vs Kou

- **Merton**: jump $\gamma(x, y) = x(e^y - 1)$, so multiplicative with Gaussian $y$. Smoother Lévy density.
- **Kou**: same multiplicative structure but double-exponential jumps. Fatter tails, asymmetric.

Both are mathematically identical (jump-diffusion SDE framework), just different jump-size distributions.

### 8.2 Affine jump-diffusions

A very powerful and tractable class: $X$ is affine if $\mu$ and $\sigma^2$ are affine in $X$ and the jump intensity is affine. Characteristic function via Riccati ODEs. Covers Heston, Bates, Hull–White, CIR, affine-HJM, etc. — the **workhorse** of modern rates and credit.

---

## 9. Pricing under jumps: Fourier methods

Carr–Madan (Module 2.5, Section 6) is the standard tool: given the CF of $\log S_T$ under $Q$, the call price is
$$
C(K) = \frac{e^{-\alpha \log K}}{\pi}\int_0^\infty e^{-i v \log K} \psi(v) dv,
$$
with
$$
\psi(v) = \frac{e^{-rT}\, \varphi_{\log S_T}(v - (\alpha + 1)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v}.
$$

This is computed via FFT at $O(N\log N)$ for thousands of strikes. The key is that $\varphi_{\log S_T}$ is **closed form** for every Lévy model (Merton, Kou, VG, NIG) and every affine model (Heston, Bates) — even though the density is not.

---

## 10. Python: simulating Lévy processes

```python
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
T, n, M = 1.0, 2_000, 10_000
dt = T/n
times = np.linspace(0, T, n+1)

# ---- (a) Compound Poisson ----
lam = 5.0
# Number of jumps in [0, T] ~ Poisson(lam*T)
N_T = rng.poisson(lam * T, size=M)
CP = np.zeros(M)
for i in range(M):
    jumps = rng.normal(0.0, 1.0, size=N_T[i])
    CP[i] = jumps.sum()
print(f"Compound Poisson: mean={CP.mean():.4f}, theory=0")
print(f"Compound Poisson:  var={CP.var():.4f}, theory={lam*T:.4f}")

# ---- (b) Merton jump-diffusion stock simulation ----
S0 = 100.0
r_rate = 0.05
sigma = 0.2
lam_J = 0.5
mu_J = -0.05
sigma_J = 0.1
# Compensator so that e^{-rT} S_T is Q-martingale
kappa = np.exp(mu_J + 0.5*sigma_J**2) - 1
drift_rn = r_rate - 0.5*sigma**2 - lam_J*kappa
# Simulate
dW = rng.standard_normal(size=(M, n)) * np.sqrt(dt)
# For each step and path: count jumps
N_inc = rng.poisson(lam_J*dt, size=(M, n))
# Sum of jumps: for each path, total jump = sum of iid N(mu_J, sigma_J^2)'s
# Fast: sum of N(mu, sig^2) over N events = N(N*mu, N*sig^2); generate once
jumps_total = rng.normal(N_inc*mu_J, np.sqrt(N_inc)*sigma_J + 1e-16)
log_S = np.log(S0) + np.cumsum(drift_rn*dt + sigma*dW + jumps_total, axis=1)
log_S = np.concatenate([np.full((M, 1), np.log(S0)), log_S], axis=1)
S = np.exp(log_S)
K = 100.0
call_price_mc = np.exp(-r_rate*T) * np.maximum(S[:, -1] - K, 0).mean()
print(f"\nMerton call MC: {call_price_mc:.4f}")

# ---- (c) Merton closed-form series ----
def bs_call(S0, K, r, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

lam_prime = lam_J*(1 + kappa)
mc_sum = 0.0
for n_jumps in range(30):
    p = np.exp(-lam_prime*T)*(lam_prime*T)**n_jumps/np.math.factorial(n_jumps)
    sigma_n = np.sqrt(sigma**2 + n_jumps*sigma_J**2/T)
    r_n = r_rate - lam_J*kappa + n_jumps*(mu_J + 0.5*sigma_J**2)/T
    mc_sum += p*bs_call(S0, K, r_n, T, sigma_n)
print(f"Merton call analytic: {mc_sum:.4f}")

# ---- (d) Variance Gamma simulation via subordinated BM ----
nu_vg = 0.3
# Gamma subordinator with mean 1 and variance nu per unit time
G = rng.gamma(shape=T/nu_vg, scale=nu_vg, size=M)
# VG: X = mu*G + sigma*sqrt(G)*Z
mu_vg = -0.1
sigma_vg = 0.2
Z = rng.standard_normal(M)
X_vg = mu_vg*G + sigma_vg*np.sqrt(G)*Z
print(f"\nVG: mean={X_vg.mean():.4f}")
print(f"VG:  var={X_vg.var():.4f}")
# Theory: mean = mu*T, var = sigma^2*T + mu^2*T*nu
print(f"VG theory mean={mu_vg*T:.4f}, var={sigma_vg**2*T + mu_vg**2*T*nu_vg:.4f}")
```

### Expected output

- Compound Poisson mean $\approx 0$, variance $\approx \lambda T = 5$.
- Merton MC call price ~10-11 (for these params).
- Merton analytic price matches MC to 1-2 cents.
- VG mean and variance match theory.

---

## 11. [QUANT APPLICATION] — Lévy models in practice

### 11.1 Equity vol smile and skew

In the BS world, implied vol is flat across strikes. In reality, OTM puts are expensive (fat left tail) and OTM calls cheap — a **skew**. Lévy models (Merton, Kou, VG, NIG) produce this via jump asymmetry. Calibrating a jump-diffusion model to the full smile requires fitting jump parameters to option prices.

### 11.2 Credit risk: structural + reduced-form

- **Structural (Merton–style)**: firm asset follows GBM; default at first passage. Jump diffusions (CreditGrades, Hillegeist) add jumps to capture sudden defaults.
- **Reduced-form (intensity-based)**: default time has intensity $\lambda_t$, a predictable process. Pricing via survival probability $e^{-\int_0^T \lambda_s ds}$.

Both frameworks naturally extend to Lévy models.

### 11.3 Variance swaps under jumps

Under BS, $\int_0^T \sigma^2 ds$ replication works exactly. Under jumps, additional jump-variance term appears:
$$
\text{Realized var} = \int_0^T \sigma_s^2 ds + \sum_{s \le T} (\Delta X_s)^2.
$$

Variance swap payoff differs from log-contract replication by a **"variance-swap convexity"** term $\propto \sum (\Delta X)^2$ — market makers must hedge this.

### 11.4 Kou's model in rates

The Kou double-exponential model has found use in **interest rate derivative pricing**: the analytical tractability (via Wiener–Hopf factorization) allows closed-form barrier option prices, important for range-accrual notes.

### 11.5 Exotic options under jumps

- **Barrier options**: jumps can "overshoot" the barrier, complicating first-passage analysis; needs integration over jump-overshoot distribution.
- **Lookback options**: running max/min under Lévy — can be computed via Wiener–Hopf factorization.
- **American options with jumps**: variational PIDE, solved via projected SOR or Howard's policy iteration.

### 11.6 High-frequency / market microstructure

Tick-by-tick price changes are better modeled as pure-jump Lévy processes (or Hawkes processes with self-exciting jumps). The **signature plot** of realized variance vs sampling frequency reveals jump activity — a key input for high-frequency trading strategies.

---

## 12. Summary

- A **Lévy process** is a càdlàg process with independent, stationary increments and no fixed jump times.
- **Lévy–Khintchine**: characteristic exponent $\psi(u)$ has drift + Gaussian + jump integral form.
- **Lévy–Itô decomposition**: $X = $ drift + Brownian + compound Poisson (large jumps) + compensated small jumps.
- **Jump Itô formula**: classical Itô + sum over jumps ($f(X) - f(X^-) - f'(X^-)\Delta X$).
- **Generator**: integro-differential $\mathcal{L} = $ local + nonlocal (integral over jumps).
- **Feynman–Kac for jumps**: PIDE instead of PDE.
- **Girsanov for jumps**: changes drift of Brownian + jump intensity/distribution.
- **Workhorse models**: Merton, Kou, VG, NIG, affine jump-diffusions (Heston, Bates).

Lévy processes are the next natural modeling step beyond GBM: they capture the skew, kurtosis, and tail behavior that GBM misses.

---

## 13. Exercises

**★ (warm-ups)**

**13.1** Verify that Brownian motion is a Lévy process with $b = 0$, $A = 1$, $\nu = 0$.

**13.2** Verify that Poisson process with rate $\lambda$ is Lévy with $b = \lambda\mathbf{1}_{\text{in compensator window}}$, $A = 0$, $\nu = \lambda \delta_1$.

**13.3** For a compound Poisson with finite-mean jumps $Y$, compute $E[X_t]$ and $\text{Var}(X_t)$.

**13.4** Compute the characteristic function of a compound Poisson process with $Y \sim N(\mu, \sigma^2)$.

**13.5** Show the Poisson process has $\psi(u) = \lambda(e^{iu} - 1 - iu\mathbf{1}_{|u|\le 1})$ if we don't compensate, or $\psi(u) = \lambda(e^{iu} - 1)$ with drift adjustment.

**★★ (core techniques)**

**13.6** **(Merton characteristic function.)** Derive the CF of $\log S_T$ under Merton's jump-diffusion and verify it is a Lévy–Khintchine exponent.

**13.7** **(Kou closed-form.)** Show that Kou's double-exponential jump model admits a closed-form CF. Derive the formula.

**13.8** **(VG as Lévy.)** Show that the VG process is pure-jump (no Gaussian component), and compute its Lévy measure $\nu_{VG}(dy) = \frac{C}{|y|}e^{-\lambda_\pm |y|} dy$ for appropriate $C, \lambda_\pm$.

**13.9** **(Jump Itô, baby version.)** Apply the jump-Itô formula to $f(X) = X^2$ where $X = \sum_{k=1}^{N_t} \xi_k$ is compound Poisson. Show
$$
X_t^2 = 2\int_0^t X_{s^-} dX_s + \sum_{s \le t}(\Delta X_s)^2.
$$

**13.10** **(Lévy martingale.)** Show that for a Lévy process $X$ with $E[X_t] = bt$, the process $X_t - bt$ is a martingale.

**13.11** **(Compensated Poisson martingale.)** Show that if $\tilde N_t = N_t - \lambda t$, then $\tilde N$ is a martingale. Compute $\langle \tilde N\rangle_t$.

**13.12** **(Exponential martingale, jump case.)** Show $\mathcal{E}(X)_t = \exp(X_t - t\psi(-i))$ is a martingale for a Lévy process $X$ with integrable exponential moment.

**13.13** **(Merton option pricing via conditioning.)** Derive Merton's series formula by conditioning on $N_T = n$ and applying the Black–Scholes formula with adjusted parameters.

**13.14** **(Variance swap jump correction.)** Show that for a Merton jump-diffusion, the replication error of a variance swap vs a log-contract is $\sum_{s\le T}(\Delta X_s)^2$.

**13.15** **(Fokker–Planck with jumps.)** Derive the forward equation for a Lévy process — a PIDE with an integral term $\int[p(t, x - y) - p(t, x)]\nu(dy)$.

**★★★ (research / quant)**

**13.16** **(Carr–Madan FFT for Merton.)** Implement Carr–Madan FFT pricing for Merton model; compare with the exact Merton series. Measure the speed advantage.

**13.17** **(Calibration to SPX smile.)** Using the VG model, calibrate parameters $(\sigma, \nu, \mu)$ to the SPX 1-month option smile. Report the fit quality.

**13.18** **(Bates model.)** The Bates model is Heston + Merton jumps: $dS/S = r dt + \sqrt{V} dB^{(1)} + dJ$, $dV = \kappa(\theta - V) dt + \xi\sqrt{V} dB^{(2)}$, $d\langle B^1, B^2\rangle = \rho dt$, $J = \sum (e^{Y_k} - 1)$. Derive the CF via affine Riccati ODEs.

**13.19** **(Tempered stable processes.)** Read Cont–Tankov (2003) Chapter 4 on tempered stable processes. Understand their Lévy measure and the tempering of heavy tails.

**13.20** **(First passage for jump-diffusion.)** Let $X$ be a Merton jump-diffusion. Compute the first passage time density to a level $L$ using Laplace transforms. Explain why closed forms exist for Kou but not Merton.

**13.21** **(Wiener–Hopf factorization for Kou.)** Use the structure of Kou's two-sided exponential jumps to derive the Wiener–Hopf factorization. Apply it to barrier options.

**13.22** **(Hawkes process.)** Read Hawkes (1971) on self-exciting point processes. Contrast with Lévy / Poisson: Hawkes jumps have intensity $\lambda_t = \mu + \sum_{t_k < t} g(t - t_k)$. Explain why Hawkes is **not Lévy** but still admits a generator and Itô-type formula.

---

## 14. Forward pointers

- **Module 3.6 (Continuous-time Markov processes).** Semigroups, generators, Feller processes — Lévy processes are the prototype Feller process.
- **Module 3.7 (Continuous-time martingales).** Compensators, Doob–Meyer for jump processes.
- **Subjects 4+ (Optimization, PDEs, etc.)**: Merton problem, Hamilton–Jacobi–Bellman, control of jump-diffusions.

The key idea: **jumps are a first-class citizen** in modern financial modeling. Every Fourier-pricing method, every extension of Itô, every calibration to implied smile, depends on the Lévy / jump-diffusion framework.

**Next module:** Continuous-time Markov processes — semigroups, generators, the master theorem that unifies SDEs, Lévy, and jump-diffusions under a single framework.

---

*End of Module 3.5.*

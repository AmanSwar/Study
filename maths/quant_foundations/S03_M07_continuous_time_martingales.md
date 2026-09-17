# Module 3.7 — Continuous-Time Martingales: Doob–Meyer, BDG, and Semi-Martingales

**Mathematical Foundations for Quantitative Research: From JEE to Jane Street**
Subject 3 (Stochastic Processes), Module 7 of 7 — **Subject 3 Capstone**

---

## Prerequisites

- **Module 2.6** (Discrete-time martingales, optional stopping, Doob's inequalities, martingale convergence).
- **Module 3.1** (Brownian motion).
- **Module 3.2** (Itô integral, continuous local martingales).
- **Module 3.3** (Itô's formula).
- **Module 3.5** (Lévy processes, jump processes).
- **Module 3.6** (Semigroups, generators).

This is the **capstone of Subject 3**: the fine structure of continuous-time martingales, Doob–Meyer decomposition, Burkholder–Davis–Gundy (BDG) inequalities, and the general semi-martingale framework. Together, these are the tools by which the entire theory of stochastic integration, SDEs, and option pricing is made rigorous.

---

## 1. Continuous-time martingales: recap

**Definition 1.1.** A càdlàg process $(M_t)_{t \ge 0}$ adapted to $\mathbb{F}$ is a **martingale** if $E[|M_t|] < \infty$ and $E[M_t|\mathcal{F}_s] = M_s$ for $s \le t$. Sub/super analogously with $\le, \ge$.

The discrete-time theory (Module 2.6) transfers: optional stopping, Doob's inequality, martingale convergence. The continuous-time subtleties are:

- **Path regularity** (càdlàg vs continuous).
- **Quadratic variation** (exists for all continuous martingales; a process in its own right).
- **Local martingales** (true martingale up to a localizing sequence of stopping times).
- **Semi-martingales** (local martingale + bounded variation).

---

## 2. Stopping times and optional stopping

**Definition 2.1.** $\tau : \Omega \to [0, \infty]$ is a stopping time if $\{\tau \le t\} \in \mathcal{F}_t$ for all $t$.

**$\mathcal{F}_\tau$**: the collection of events $A \in \mathcal{F}$ with $A \cap \{\tau \le t\} \in \mathcal{F}_t$ for all $t$.

**Theorem 2.2 (Optional stopping, continuous time).** For a càdlàg martingale $M$ and bounded stopping times $\sigma \le \tau \le C$ a.s.,
$$
E[M_\tau | \mathcal{F}_\sigma] = M_\sigma.
$$

Similar to Module 2.6; same three sufficient conditions for unbounded $\tau$:

1. $\tau$ bounded.
2. $M$ uniformly integrable.
3. $E[\tau] < \infty$ and $M$ has bounded increments.

---

## 3. Doob's inequalities (continuous time)

**Theorem 3.1 (Doob's maximal inequality).** For a nonnegative submartingale $X$ and $\lambda > 0$,
$$
P\Bigl(\sup_{t \le T} X_t \ge \lambda\Bigr) \le \frac{E[X_T]}{\lambda}.
$$

**Theorem 3.2 (Doob's $L^p$ inequality, $p > 1$).**
$$
E\Bigl[\sup_{t \le T} X_t^p\Bigr] \le \Bigl(\frac{p}{p-1}\Bigr)^p E[X_T^p].
$$

*Proof.* Approximate by dyadic times $t_n = kT/2^n$; apply discrete-time Doob; take limits using càdlàg paths and Fatou. $\square$

### 3.1 Applications

- BM $\sup_{t \le T} B_t$ distribution and its tail bound.
- $\sup_{t \le T}|\int_0^t H dB|$ bound for Itô integrals.
- $\sup_{t \le T} M_t$ probabilities for any martingale.

---

## 4. Doob's martingale convergence theorem

**Theorem 4.1.** Let $M$ be a right-continuous submartingale with $\sup_t E[M_t^+] < \infty$. Then $M_\infty := \lim_{t\to\infty} M_t$ exists a.s.

*Proof.* Apply the upcrossing inequality (Module 2.6, Lemma 7.4) to a countable dense subset of times, then use right-continuity to extend. Specifically, for any $a < b$, the number of upcrossings of $[a, b]$ by $M$ on a rational grid is bounded, which forces the path limit to exist. $\square$

**Theorem 4.2 (UI martingales.)** $M$ is a UI martingale iff $M_t \to M_\infty$ in $L^1$ iff $M_t = E[M_\infty | \mathcal{F}_t]$ for some $M_\infty \in L^1$.

---

## 5. Doob–Meyer decomposition

One of the **crown jewels** of martingale theory: every non-negative submartingale can be written uniquely as a martingale plus a predictable increasing process.

**Theorem 5.1 (Doob–Meyer).** Let $X$ be a càdlàg submartingale of class (DL) (i.e., $\{X_\tau : \tau \text{ bounded stopping time}\}$ is uniformly integrable). Then there exist:

- a **martingale** $M$ with $M_0 = 0$,
- a **predictable increasing process** $A$ with $A_0 = 0$,

such that $X_t = X_0 + M_t + A_t$, and this decomposition is unique.

*Proof sketch.*

*Uniqueness.* If $X = M_1 + A_1 = M_2 + A_2$ are two such decompositions, $M_1 - M_2 = A_2 - A_1$ is a martingale equal to a predictable BV process, hence constant by a theorem on predictable martingales.

*Existence.* Use the discrete-time Doob decomposition on dyadic grids and pass to the limit. The compensator $A$ is the "predictable part" of the submartingale's drift. See Protter (2004) Chapter III.

### 5.1 Examples

- For $M$ continuous martingale, $X = M^2$ is submartingale, $A = \langle M\rangle$ (quadratic variation).
- For Poisson process $N$ with rate $\lambda$, $X = N$ is submartingale, $A = \lambda t$.
- For compound Poisson with mean $\mu$ jumps at rate $\lambda$, compensator $A = \lambda\mu t$.

### 5.2 The compensator viewpoint

The compensator $A$ is the "predictable expected rate of increase". Doob–Meyer formalizes intuition: the submartingale is "expected to increase" at the predictable rate $dA_t$; subtracting gives a martingale.

---

## 6. Quadratic variation of continuous semi-martingales

**Theorem 6.1.** Every continuous local martingale $M$ has a unique continuous predictable increasing process $\langle M\rangle$ with $\langle M\rangle_0 = 0$ such that $M^2 - \langle M\rangle$ is a continuous local martingale.

Moreover,
$$
\langle M\rangle_t = \lim_{\|\pi\|\to 0}\sum_{t_k \in \pi}(M_{t_{k+1}} - M_{t_k})^2 \qquad \text{in probability},
$$

the $[0, t]$ restriction of a partition refining limit.

**Covariation**: for $M, N$ continuous local martingales,
$$
\langle M, N\rangle := \tfrac{1}{4}(\langle M + N\rangle - \langle M - N\rangle).
$$

### 6.1 Key formula

For $M = \int_0^\cdot H dB$ with $B$ Brownian and $H \in \mathcal{L}^2$:
$$
\langle M\rangle_t = \int_0^t H_s^2 ds.
$$

For multi-D, $\langle M^i, M^j\rangle_t = \int_0^t \sum_k H^{ik}_s H^{jk}_s ds$ (Module 3.3).

---

## 7. Burkholder–Davis–Gundy (BDG) inequalities

**Theorem 7.1 (BDG).** For every $p > 0$, there exist constants $c_p, C_p$ such that for any continuous local martingale $M$ with $M_0 = 0$,
$$
c_p E[\langle M\rangle_T^{p/2}] \le E[\sup_{t \le T}|M_t|^p] \le C_p E[\langle M\rangle_T^{p/2}].
$$

*Proof for $p = 2$.* Immediate from Doob's $L^2$ inequality and $E[M_T^2] = E[\langle M\rangle_T]$. Lower bound from optional stopping.

*General $p$*: by duality and interpolation, or via the exponential martingale trick — see Revuz–Yor Chapter IV. $\square$

**Constants.** Best-possible $C_p$ depend on $p$:
- $C_1 = 3$ (sharp, due to Davis).
- $C_2 = 4$ (Doob's).
- $c_p = $ explicit but small.

### 7.1 BDG for jumps

For a càdlàg local martingale, the quadratic variation $[M] = [M]^c + \sum(\Delta M)^2$ (continuous part + jump squares). BDG replaces $\langle M\rangle$ by $[M]$ in the jump setting.

---

## 8. Stochastic integration for semi-martingales

### 8.1 Definition

A **semi-martingale** is $X = X_0 + M + A$ with $M$ a local martingale and $A$ a BV process. The class of semi-martingales is preserved under:

- Stopping.
- Transformation by $C^2$ functions (Itô's formula).
- Itô integration ($\int H dX$ is semi-martingale if $X$ is).

**Bichteler–Dellacherie theorem:** Semi-martingales are exactly the "good integrators" for a reasonable Itô-type integral.

### 8.2 Stochastic integral for semi-martingales

For $H$ predictable and locally bounded, the integral $\int_0^t H_s dX_s = \int H dM + \int H dA$ decomposes. The Itô integral for the martingale part + Stieltjes integral for the BV part.

### 8.3 Itô–Föllmer formula

For any $C^2$ function $f$ and semi-martingale $X$,
$$
f(X_t) = f(X_0) + \int_0^t f'(X_{s^-}) dX_s + \tfrac{1}{2}\int_0^t f''(X_{s^-}) d[X]^c_s + \sum_{s \le t}[f(X_s) - f(X_{s^-}) - f'(X_{s^-})\Delta X_s].
$$

This generalizes Module 3.5's jump-Itô formula to the full semi-martingale class.

---

## 9. Local time and the Tanaka formula

### 9.1 Local time of continuous semi-martingale

For a continuous semi-martingale $X$ and level $a$, the **local time** at $a$ is
$$
L_t^a := \lim_{\varepsilon\to 0}\frac{1}{2\varepsilon}\int_0^t \mathbf{1}_{\{|X_s - a| \le \varepsilon\}} d\langle X\rangle_s.
$$

$L^a$ is a nondecreasing continuous process that grows only on $\{s : X_s = a\}$.

### 9.2 Tanaka's formula

$$
(X_t - a)^+ = (X_0 - a)^+ + \int_0^t \mathbf{1}_{\{X_{s^-} > a\}} dX_s + \tfrac{1}{2}L_t^a.
$$

Interpreted as an Itô formula for $f(x) = (x - a)^+$, which has second derivative $\delta_a$ (a delta function). Local time $L^a$ **replaces** the $\tfrac{1}{2}\int f''(X) d\langle X\rangle$ term from smooth Itô, because $f''$ concentrates at $a$.

### 9.3 Applications

- **Brownian local time**: $L_t^0$ for $X = B$ appears in Ray–Knight theorems, reflecting Brownian motion, and occupation-time arcsine laws.
- **Options**: digital barrier options involve $\mathbf{1}_{\{X > L\}}$; their replication introduces local-time terms.
- **Model-free variance replication**: a log-contract representation of variance involves local-time integrals near the strike.

---

## 10. Continuous-time martingale representation

**Theorem 10.1 (Brownian martingale representation).** Every $L^2$ $\mathbb{F}^B$-martingale $M$ has an explicit Itô integral representation
$$
M_t = M_0 + \int_0^t H_s dB_s, \quad H \in \mathcal{L}^2.
$$

*Sketch.* Chaos decomposition of $L^2(\mathcal{F}_T)$ into $\bigoplus_n \mathcal{H}_n$ (Wiener chaos), where $\mathcal{H}_n$ = span of $n$-fold iterated Itô integrals; each has a representation. See Nualart (2006). $\square$

**Clark–Ocone formula.** With Malliavin derivative $D_s$:
$$
H_s = E[D_s M_T | \mathcal{F}_s].
$$

This gives a **constructive** representation, critical for hedging path-dependent payoffs.

---

## 11. Semi-martingale decomposition (Protter)

Every semi-martingale has a canonical decomposition: $X = X_0 + M^c + M^d + A$, where

- $M^c$ is a continuous local martingale,
- $M^d$ is a purely discontinuous local martingale (compensated jumps),
- $A$ is a càdlàg BV process.

This is the **Lévy–Itô** decomposition in disguise, for general semi-martingales.

---

## 12. Python: BDG, local time, martingale convergence

```python
import numpy as np

rng = np.random.default_rng(42)
T, n, M = 1.0, 5_000, 10_000
dt = T/n
times = np.linspace(0, T, n+1)

# ---- (a) BDG inequality verification ----
# For M = B (Brownian), <M>_T = T, so BDG: E[sup B^2] <= 4 T.
dW = rng.standard_normal(size=(M, n)) * np.sqrt(dt)
B = np.concatenate([np.zeros((M,1)), np.cumsum(dW, axis=1)], axis=1)
sup_B2 = (B**2).max(axis=1).mean()
print(f"E[sup B^2]: {sup_B2:.4f}, BDG upper bound 4T = {4*T}")
# Exact via reflection: E[max B_t^2] = T(2*something)... check ~1.27T

# ---- (b) Local time of BM via counting formula ----
# L^0_T ~ 1/(2*eps) * ∫_0^T 1_{|B_s| < eps} ds
eps = 0.05
occupation = (np.abs(B[:, :-1]) < eps).mean(axis=1) * T
L0_approx = occupation / (2*eps)
# Theoretical: E[L^0_T] = sqrt(2T/pi)
print(f"\nEmpirical E[L^0_T]: {L0_approx.mean():.4f}")
print(f"Theory sqrt(2T/pi):  {np.sqrt(2*T/np.pi):.4f}")

# ---- (c) Tanaka formula |B_t| = ∫ sgn(B) dB + L^0_t ----
# For a single path:
path = 0
sgn = np.where(B[path, :-1] >= 0, 1.0, -1.0)
ito_sgn = (sgn * dW[path]).cumsum()
L_path = np.abs(B[path, 1:]) - ito_sgn  # This is L^0_t by Tanaka
print(f"\nPath {path}: |B_T| = {np.abs(B[path, -1]):.4f}")
print(f"∫ sgn(B) dB = {ito_sgn[-1]:.4f}")
print(f"L^0_T (via Tanaka) = {L_path[-1]:.4f}")

# ---- (d) Martingale convergence: exp martingale ----
lam = 0.3
Z = np.exp(lam*B - 0.5*lam**2*times)
# Z_T should have mean 1
print(f"\nE[Z_T] stochastic exp: {Z[:, -1].mean():.4f}, theory: 1")

# ---- (e) Doob L^2 inequality verification ----
# For M = B, E[sup_{t<=T} B_t^2] <= 4 E[B_T^2] = 4T
# exact E[sup B_t] relates to reflection principle
print(f"\nE[B_T^2]: {(B[:, -1]**2).mean():.4f} (theory T={T})")
print(f"E[sup B_t^2]: {sup_B2:.4f}")
print(f"Ratio: {sup_B2 / T:.4f}, Doob bound is 4")
```

Expected:
- `E[sup B^2]` should be about $1.27 T$ (since $\sup B \sim$ half-normal $\sqrt{T}$ and squared).
- Local time theoretical value $\sqrt{2T/\pi} \approx 0.798$.
- Tanaka's identity holds exactly for continuous paths.
- Exp martingale mean 1.

---

## 13. [QUANT APPLICATION] — martingale theory in finance

### 13.1 First Fundamental Theorem of Asset Pricing

**No arbitrage $\iff$ existence of equivalent martingale measure $Q$** under which discounted prices are martingales (local martingales if we allow local martingales, true martingales if we require NFLVR — "No Free Lunch with Vanishing Risk"). Delbaen–Schachermayer 1994.

### 13.2 Second FTAP: completeness

Market is complete $\iff$ $Q$ is unique $\iff$ every contingent claim admits an Itô integral representation.

In Brownian models, this is the **martingale representation theorem** (Section 10): the hedge is the Malliavin derivative conditional on $\mathcal{F}_t$.

### 13.3 BDG and $L^p$-hedging

BDG controls the $L^p$-norm of hedge PnL in terms of $L^{p/2}$-norm of integrated vega-gamma variance. Central in robust hedging: a trader's PnL is bounded in $L^p$ iff the $L^{p/2}$-variance of their position is bounded.

### 13.4 Local time and corridor variance swaps

A **corridor variance swap** pays realized variance of $\log S$ only when $S \in [L, H]$. The payoff involves **local time of $\log S$** at the corridor boundaries. Replicating it requires a strip of options across the corridor.

### 13.5 Doob–Meyer and risk

For a submartingale (e.g., squared portfolio value), Doob–Meyer decomposition separates the **predictable risk accumulation** ($A_t$, the variance integral) from the **unpredictable shock** ($M_t$, the martingale part). This is used in risk management to quantify systematic vs idiosyncratic contributions.

### 13.6 Semi-martingale topology and robust pricing

Robust (model-free) pricing uses minimal semi-martingale assumptions on the stock price and derives Fréchet bounds on option prices. Requires deep results on the topology of the space of semi-martingales (Memin, Föllmer–Schied).

---

## 14. Summary of Subject 3

This module closes Subject 3 (Stochastic Processes). Together, the seven modules established:

1. **Module 3.1 (Brownian motion)**: the atom — Lévy's construction, path properties, Donsker's theorem, reflection principle, Lévy's characterization.
2. **Module 3.2 (Stochastic integration)**: the Itô integral as an $L^2$ isometry from $\mathcal{L}^2_{\text{pred}}$ to $L^2(\Omega)$; continuous martingale property; extension to semi-martingales.
3. **Module 3.3 (Itô's formula)**: the chain rule; Girsanov; Feynman–Kac; martingale representation.
4. **Module 3.4 (SDEs)**: existence & uniqueness by Picard; Kolmogorov equations; numerical schemes; finance workhorses (BS, Vasicek, CIR, Heston, local vol, rough vol).
5. **Module 3.5 (Lévy processes & jumps)**: Lévy–Khintchine + Lévy–Itô; jump-Itô; PIDEs; Girsanov for jumps; Merton, Kou, VG, NIG, affine jump-diffusions.
6. **Module 3.6 (Continuous-time Markov)**: semigroup-generator calculus; Hille–Yosida; Dynkin; martingale problems; invariant measures; ergodic theorems.
7. **Module 3.7 (Continuous-time martingales — this module)**: Doob–Meyer decomposition; BDG inequalities; continuous-semi-martingale theory; local time; martingale representation.

### 14.1 What you now have

- The ability to **write down** SDEs for any financial instrument (stock, rate, credit spread, basket).
- The ability to **price** any European or path-dependent option via Feynman–Kac or Monte Carlo.
- The ability to **hedge** via delta + gamma + vega + martingale representation.
- The ability to **change measure** (Girsanov) for risk-neutral pricing and for importance sampling.
- The ability to **incorporate jumps** via Lévy processes and PIDE.
- The ability to **analyze convergence** via ergodic theorems and spectral gaps.
- The ability to **quantify risk** via Doob–Meyer compensators and BDG.

This is the rigorous foundation underlying every continuous-time option pricer, every rates model, every volatility surface calibrator, and every risk-management engine on every quant desk.

---

## 15. Exercises

**★ (warm-ups)**

**15.1** Show that a continuous local martingale with BV paths is constant.

**15.2** For $M$ continuous martingale with $\langle M\rangle_\infty < \infty$ a.s., show $M_\infty$ exists in $L^2$.

**15.3** Show that $B_t^2 - t$ is a martingale; identify $\langle B\rangle_t = t$.

**15.4** For $M$ a BM and $\tau_a = \inf\{t : B_t = a\}$, compute $\langle B^{\tau_a}\rangle_t$.

**15.5** For Poisson process $N$ with rate $\lambda$: verify $(N_t - \lambda t)^2 - \lambda t$ is a martingale (so $[N_\cdot]_t = \lambda t$ as compensator).

**★★ (core)**

**15.6** **(Doob's $L^p$ inequality.)** Prove that for a nonnegative right-continuous submartingale $X$ and $p > 1$: $E[\sup_{t\le T} X_t^p] \le (p/(p-1))^p E[X_T^p]$.

**15.7** **(BDG for $p = 2$.)** Prove the $L^2$ case of BDG directly from Doob's $L^2$ and the isometry $E[M_T^2] = E[\langle M\rangle_T]$.

**15.8** **(Continuous local martingale $\Rightarrow$ martingale.)** Show: a nonnegative continuous local martingale with $E[M_0] < \infty$ and constant expectation is a true martingale.

**15.9** **(Doob–Meyer for Poisson.)** Write the Doob–Meyer decomposition of $N_t^2$ where $N$ is Poisson rate $\lambda$.

**15.10** **(Tanaka's formula.)** Prove Tanaka's formula $(X - a)^+ = (X_0 - a)^+ + \int\mathbf{1}_{X^- > a} dX + \tfrac{1}{2}L^a$ by approximating the convex function $(x - a)^+$ by smooth functions.

**15.11** **(Local time at zero of BM.)** Show $E[L^0_T] = \sqrt{2T/\pi}$. [Hint: use occupation-time formula and $E[\mathbf{1}_{|B_s|<\varepsilon}] \to $ density at $0$.]

**15.12** **(Ray–Knight, sketch.)** State the first Ray–Knight theorem: for $\tau = $ hitting time of level $1$ by BM, the local-time process $(L^x_\tau)_{x \in [0,1]}$ is a squared 2D Bessel process.

**15.13** **(Reflected BM.)** Let $X_t = B_t - \min_{s\le t} B_s$ (reflecting BM). Show $X$ is a semi-martingale with Doob–Meyer decomposition $X = B + L$ where $L = -\min_s B_s$.

**15.14** **(Optional sampling.)** Prove OST for a UI martingale and any stopping times $\sigma \le \tau$ (both $\le \infty$).

**15.15** **(Exponential martingale.)** Show $Z_t = \exp(\int_0^t \theta dB - \tfrac{1}{2}\int \theta^2 ds)$ has $Z_t$ with $\langle Z\rangle_t = \int_0^t Z^2 \theta^2 ds$.

**★★★ (research / quant)**

**15.16** **(BDG with exponent $p = 1$.)** Davis showed $C_1 = 3$ is sharp. Explain the statement: for any continuous local martingale, $E[\sup|M_T|] \le 3 E[\sqrt{\langle M\rangle_T}]$.

**15.17** **(First fundamental theorem of asset pricing.)** Read Delbaen–Schachermayer (1994). State NFLVR precisely and explain why it equivalences NA + integrability.

**15.18** **(Completeness and martingale representation.)** Show that a complete market's stock price $S$ has the property: every $L^2$ $\mathbb{F}^S$-martingale is an Itô integral against $S$ (not $B$). This characterizes complete markets beyond Brownian.

**15.19** **(Jacod–Shiryaev martingale decomposition.)** Read Jacod–Shiryaev Chapter I–III on the martingale/special semi-martingale structure. Identify the canonical decomposition $X = X_0 + M^c + M^d + A^c + A^d$ for a jump-diffusion.

**15.20** **(Föllmer–Schied robust pricing.)** In a market with only model-free assumptions (no Brownian model specified), Föllmer–Schied derive bounds on contingent claim prices using pathwise martingale properties. Explain the role of "pathwise superhedging".

**15.21** **(Stochastic volatility BDG.)** For Heston model, derive the BDG-type estimate on $\sup_t S_t^p$ using $\langle S\rangle_t = \int_0^t V_s S_s^2 ds$.

**15.22** **(Azéma's martingale.)** Study Azéma's martingale $M_t = \text{sgn}(B_t)\sqrt{\pi(t - g_t)/2}$, where $g_t$ is the last zero of $B$ before $t$. Show it's a martingale with $\langle M\rangle_t = t$ (Dambis–Dubins–Schwarz — so it's a time-changed BM) but clearly not Brownian (has jumps). This is a striking counterexample to martingale characterizations.

---

## 16. Subject 3 Capstone: the path ahead

**Subject 3** (Stochastic Processes) is **done**. Between the seven modules:

- We built Brownian motion from scratch (Lévy's construction).
- We built the Itô integral from scratch (Hilbert-space extension).
- We proved Itô's formula from scratch (Taylor + quadratic variation).
- We established existence-uniqueness of SDEs (Picard + Gronwall).
- We extended to Lévy processes and jump diffusions (Lévy–Khintchine).
- We developed semigroup calculus (Hille–Yosida).
- We finished with martingale fine structure (Doob–Meyer, BDG, local time).

**Looking ahead:**

- **Subject 4 (Optimal Stopping & Stochastic Control)**: American options via Snell envelopes, HJB, Merton's portfolio problem.
- **Subject 5 (PDEs in Finance)**: parabolic PDEs, finite differences, free boundaries.
- **Subject 6 (Functional Analysis)**: Sobolev spaces, weak solutions.
- **Subject 7 (Numerical Analysis)**: Monte Carlo, quadrature, PDE solvers.
- **Subject 8 (Optimization & Convex Analysis)**: portfolio optimization, convex duality.
- **Subject 9 (Statistical Learning)**: ML-based quant methods.

Stochastic processes is the **language** of quantitative finance. We now speak it.

---

*End of Module 3.7. Subject 3 complete.*

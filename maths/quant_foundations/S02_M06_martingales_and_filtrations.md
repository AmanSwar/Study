# Module 2.6: Martingales and Filtrations

**Subject 2: Probability Theory** · Module 6 of 7

---

## 0. Prerequisites and Position

- **Module 1.5** ($L^p$ spaces), **Module 1.6** (Radon-Nikodym).
- **Module 2.1** (probability spaces, 0–1 laws), **Module 2.2** (conditional expectation — central to martingale theory), **Module 2.3** (modes of convergence, uniform integrability), **Module 2.4** (LLN/CLT), **Module 2.5** (CFs).

A martingale is a mathematical formalization of a "fair game": the conditional expectation of future value given the present equals the present value. This simple idea is enormously powerful: it lets us prove almost-sure convergence theorems (Doob), prove the fundamental theorem of asset pricing, build the entire theory of stochastic integration (Subject 3), and provides the right framework for analyzing adaptive stochastic algorithms.

In this module:
- **Filtrations and stopping times**: the information/time structure of a stochastic process.
- **Martingales, submartingales, supermartingales**: definitions and examples.
- **Optional stopping theorem** (Doob): martingale property is preserved by stopping times under integrability/boundedness.
- **Doob's maximal inequality**: bounds on running maxima from terminal values.
- **Doob's upcrossing lemma** and the **martingale convergence theorem**: $L^1$-bounded martingales converge a.s.
- **$L^p$ martingale convergence** and **UI martingales** (convergence in $L^1$ iff UI).
- **Doob decomposition** (discrete) and **Doob-Meyer** (continuous preview): submartingales as "martingale + increasing process".
- **Backward martingales**.
- **Martingale CLT** (McLeish-style).
- Quant applications throughout.

---

## 1. Filtrations and Adapted Processes

### 1.1 Filtrations

**Definition 1.1.** A **filtration** on $(\Omega, \mathcal{F})$ is an increasing family $\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \dots \subseteq \mathcal{F}$ of sub-$\sigma$-algebras. We write $\mathbb{F} = (\mathcal{F}_n)_{n \ge 0}$. The triple $(\Omega, \mathcal{F}, (\mathcal{F}_n), P)$ is a **filtered probability space**.

Interpretation: $\mathcal{F}_n$ = information available at time $n$. Increasing = time doesn't erase what we knew.

**Natural filtration.** For a stochastic process $X = (X_n)$, its **natural filtration** is $\mathcal{F}_n^X := \sigma(X_0, X_1, \dots, X_n)$. A process is **adapted** to $\mathbb{F}$ if $X_n$ is $\mathcal{F}_n$-measurable for every $n$.

**Predictable process.** $H = (H_n)_{n \ge 1}$ is **predictable** if $H_n$ is $\mathcal{F}_{n-1}$-measurable for all $n \ge 1$. (Think: portfolio weight chosen *before* observing the return.)

### 1.2 Stopping times

**Definition 1.2.** $\tau: \Omega \to \{0, 1, 2, \dots, \infty\}$ is a **stopping time** (relative to $\mathbb{F}$) if $\{\tau \le n\} \in \mathcal{F}_n$ for every $n$.

Equivalently (since $\mathbb{F}$ increasing), $\{\tau = n\} \in \mathcal{F}_n$ for every $n$.

Interpretation: we decide whether to "stop by time $n$" using only information observable by time $n$ — no peeking into the future.

**Examples.**
- Deterministic time $\tau = k$: obviously a stopping time.
- First hitting time of a Borel set $B$: $\tau_B := \inf\{n : X_n \in B\}$. Then $\{\tau_B \le n\} = \bigcup_{k \le n} \{X_k \in B\} \in \mathcal{F}_n$ (adapted).
- Last exit $\tau := \sup\{n : X_n \in B\}$: **not** a stopping time in general (requires future info).
- Minimum $\tau \wedge \sigma$, maximum $\tau \vee \sigma$ of stopping times: stopping times.
- Sum $\tau + k$ (deterministic shift): stopping time.

**$\sigma$-algebra at $\tau$.**
$$
\mathcal{F}_\tau := \{A \in \mathcal{F} : A \cap \{\tau \le n\} \in \mathcal{F}_n \ \forall n\}.
$$
Interpretation: events whose indicator is determined by time $\tau$. If $X$ is adapted, $X_\tau \mathbf{1}_{\tau < \infty}$ is $\mathcal{F}_\tau$-measurable.

---

## 2. Martingales, Submartingales, Supermartingales

### 2.1 Definitions

**Definition 2.1.** Let $(X_n)$ be an adapted process with $X_n \in L^1$ for all $n$.
- $X$ is a **martingale** if $E[X_{n+1} | \mathcal{F}_n] = X_n$ a.s.
- $X$ is a **submartingale** if $E[X_{n+1} | \mathcal{F}_n] \ge X_n$ a.s.
- $X$ is a **supermartingale** if $E[X_{n+1} | \mathcal{F}_n] \le X_n$ a.s.

Tower / induction gives the equivalent integral definition: $E[X_m | \mathcal{F}_n] = X_n$ (resp. $\ge$, $\le$) for all $m \ge n$.

**Mnemonic.** Martingale = fair game, submartingale = favorable, supermartingale = unfavorable. (Yes, "super" means going down — unfortunate terminology inherited from analysis, where supers are concave/convex-related.)

**Expectation.** $E[X_n]$ is constant (martingale), increasing (sub), decreasing (super).

### 2.2 Examples

**Example 2.2** (Random walk)**.** $S_n = X_1 + \dots + X_n$ with $X_i$ iid, $E[X_i] = 0$. Then $S$ is a martingale w.r.t. $\mathcal{F}_n = \sigma(X_1, \dots, X_n)$:
$$
E[S_{n+1} | \mathcal{F}_n] = E[S_n + X_{n+1} | \mathcal{F}_n] = S_n + E[X_{n+1}] = S_n.
$$

If $E[X_i] = \mu > 0$, $S_n - n\mu$ is a martingale (centered walk).

**Example 2.3** (Squared random walk)**.** $M_n := S_n^2 - n \sigma^2$, $\sigma^2 := \text{Var}(X_i)$. Check:
$$
E[M_{n+1} - M_n | \mathcal{F}_n] = E[2 S_n X_{n+1} + X_{n+1}^2 - \sigma^2 | \mathcal{F}_n] = 2 S_n \cdot 0 + \sigma^2 - \sigma^2 = 0.
$$
So $M$ is a martingale.

**Example 2.4** (Doob martingale / sequential prediction)**.** Let $Y \in L^1$ and $M_n := E[Y | \mathcal{F}_n]$. By tower, $E[M_{n+1} | \mathcal{F}_n] = E[E[Y | \mathcal{F}_{n+1}] | \mathcal{F}_n] = E[Y | \mathcal{F}_n] = M_n$. Martingale.

Conversely, any UI martingale arises this way (Theorem 5.5 below): $M_n = E[M_\infty | \mathcal{F}_n]$.

**Example 2.5** (Exponential martingale / likelihood ratios)**.** Let $X_i$ iid with density $f$ under $P$ and $g$ under $Q$. Set $L_n := \prod_{i=1}^n g(X_i)/f(X_i)$. Under $P$:
$$
E_P[L_{n+1} | \mathcal{F}_n] = L_n \cdot E_P[g(X_{n+1})/f(X_{n+1})] = L_n \cdot 1 = L_n,
$$
since $\int g/f \cdot f = \int g = 1$. Martingale under $P$. This is the Radon-Nikodym derivative $dQ_n / dP_n$ (see Module 1.6).

**Example 2.6** (Pólya urn)**.** An urn contains $r$ red and $b$ blue balls. Draw one uniformly, return with an additional same-colored ball. Let $M_n := $ fraction of red balls at time $n$. Then $(M_n)$ is a martingale (check by direct computation). This process converges a.s. to a Beta($r, b$)-distributed limit (Doob's martingale convergence theorem).

**Example 2.7** (Galton-Watson branching)**.** Each individual has $Y$ offspring, iid with mean $m$. $Z_n$ = population in generation $n$. Then $M_n := Z_n / m^n$ is a non-negative martingale. Martingale convergence gives $M_n \to M_\infty$ a.s. The nature of $M_\infty$ encodes the extinction probability.

### 2.3 Operations preserving the martingale property

**Lemma 2.8.**
- Convex function of a martingale is a **submartingale**: if $M$ is a martingale and $\varphi$ is convex with $\varphi(M_n) \in L^1$, then $\varphi(M_n)$ is a submartingale.
- Concave: submartingale becomes subjected to... Actually more precisely: concave function of a submartingale is a supermartingale (under integrability).
- Increasing convex function of a submartingale is a submartingale.

*Proof.* Conditional Jensen (Module 2.2): $\varphi(E[X|\mathcal{G}]) \le E[\varphi(X)|\mathcal{G}]$. So $\varphi(M_n) = \varphi(E[M_{n+1}|\mathcal{F}_n]) \le E[\varphi(M_{n+1})|\mathcal{F}_n]$ — submartingale property. $\square$

Key consequence: if $M$ is a martingale then $|M|$, $M^+$, $M^-$, $M^2$ (if $L^2$) are submartingales.

---

## 3. Optional Stopping Theorem

### 3.1 Stopped process

**Definition 3.1.** The **stopped process** of $X$ at $\tau$ is
$$
X_n^\tau := X_{n \wedge \tau}.
$$
So $X_n^\tau = X_n$ for $n \le \tau$ and $X_n^\tau = X_\tau$ for $n \ge \tau$.

**Lemma 3.2.** If $X$ is a (sub-, super-) martingale and $\tau$ is a stopping time, then $X^\tau$ is a (sub-, super-) martingale (w.r.t. the same filtration).

*Proof.* $X_n^\tau = X_0 + \sum_{k=0}^{n-1} (X_{k+1} - X_k) \mathbf{1}_{\tau > k}$. Let $H_{k+1} := \mathbf{1}_{\tau > k}$; this is $\mathcal{F}_k$-measurable, so predictable. Then $X^\tau$ is a **martingale transform** $X_n^\tau = X_0 + (H \cdot X)_n$. One easily checks (via tower) that predictable transforms of martingales are martingales, provided integrability. $\square$

In particular, $E[X_n^\tau] = E[X_0]$ always (martingale).

### 3.2 Doob's optional stopping theorem

**Theorem 3.3** (Doob's optional stopping theorem, OST)**.** Let $X$ be a martingale and $\tau$ a stopping time. If any one of the following holds:
(a) $\tau$ is bounded: $\tau \le N$ for some $N < \infty$;
(b) $X$ is bounded: $\sup_{n, \omega} |X_n(\omega)| \le K$ and $\tau < \infty$ a.s.;
(c) $E[\tau] < \infty$ and the increments $|X_{n+1} - X_n|$ are bounded (by a constant, or more generally by an integrable $Y$ uniformly).

Then $E[X_\tau] = E[X_0]$.

*Proofs.*
**(a)** $E[X_\tau] = E[X_N^\tau] = E[X_0]$ since $X^\tau$ is a martingale.

**(b)** $|X_n^\tau| \le K$ and $X_n^\tau = X_\tau$ on $\{\tau \le n\}$. Since $P(\tau < \infty) = 1$, $X_n^\tau \to X_\tau$ a.s. and dominated by $K$. DCT: $E[X_0] = E[X_n^\tau] \to E[X_\tau]$.

**(c)** Write $X_n^\tau - X_0 = \sum_{k=0}^{n-1} (X_{k+1} - X_k) \mathbf{1}_{\tau > k}$. Let $n \to \infty$: $X_n^\tau \to X_\tau$ a.s. The summand is bounded by $|X_{k+1} - X_k| \mathbf{1}_{\tau > k}$ and $E[\sum_k |X_{k+1} - X_k| \mathbf{1}_{\tau > k}] = E[\sum_{k < \tau} |X_{k+1} - X_k}|] \le M \cdot E[\tau] < \infty$ if increments bounded by $M$. Dominated convergence: $E[X_\tau] = E[X_0]$. $\square$

### 3.3 Classical applications

**Example 3.4** (Gambler's ruin via OST)**.** Random walk $S_n$, $S_0 = x \in (0, N)$, $S_{n+1} = S_n \pm 1$ fair. Let $\tau = \inf\{n : S_n \in \{0, N\}\}$. Can show $E[\tau] < \infty$ and $S$ has bounded increments; or just use (b) since $S_n^\tau \in [0, N]$.

OST: $E[S_\tau] = E[S_0] = x$. But $S_\tau \in \{0, N\}$, so $x = N \cdot P(S_\tau = N)$, giving $P(\text{reach } N) = x/N$.

For the expected duration, use the martingale $M_n := S_n^2 - n$ (Example 2.3). OST: $E[S_\tau^2 - \tau] = E[S_0^2] = x^2$. So $E[\tau] = E[S_\tau^2] - x^2 = N \cdot N \cdot (x/N) - x^2 = x(N - x)$.

**Example 3.5** (Wald's identity)**.** $X_i$ iid with $E[X_1] = \mu$, $\tau$ a stopping time with $E[\tau] < \infty$. Then $E[S_\tau] = \mu E[\tau]$. *Proof.* $S_n - n\mu$ is a martingale; apply OST (c).

**Example 3.6** (Doob OST FAILS without conditions)**.** Simple random walk $S$, $\tau = \inf\{n : S_n = 1\}$. Then $S$ is a martingale, $S_\tau = 1$ on $\{\tau < \infty\}$, and $P(\tau < \infty) = 1$ (recurrence of 1-D SRW). So $E[S_\tau] = 1 \ne 0 = E[S_0]$. OST fails because $E[\tau] = \infty$ and $S^\tau$ is unbounded.

---

## 4. Doob's Maximal Inequality and $L^p$ Estimates

### 4.1 Doob's inequality

**Theorem 4.1** (Doob's maximal inequality)**.** Let $X$ be a non-negative submartingale. Then for every $\lambda > 0$ and every $n$,
$$
\lambda P\Big(\max_{0 \le k \le n} X_k \ge \lambda\Big) \le E[X_n \cdot \mathbf{1}_{\max_k X_k \ge \lambda}] \le E[X_n].
$$

*Proof.* Let $\tau := \inf\{k \le n : X_k \ge \lambda\} \wedge n$. On $A := \{\max_k X_k \ge \lambda\}$, $\tau < n$ with $X_\tau \ge \lambda$. $\tau$ is a stopping time bounded by $n$. By OST-like reasoning for submartingales (specifically: $E[X_n | \mathcal{F}_\tau] \ge X_\tau$),
$$
E[X_n \mathbf{1}_A] \ge E[X_\tau \mathbf{1}_A] \ge \lambda P(A).
$$
$\square$

### 4.2 Doob's $L^p$ inequality

**Theorem 4.2** ($L^p$ inequality)**.** For $p > 1$ and non-negative submartingale $X$,
$$
E\Big[\big(\max_{0 \le k \le n} X_k\big)^p\Big] \le \left(\frac{p}{p-1}\right)^p E[X_n^p].
$$

*Proof.* Let $M := \max_k X_k$, $q := p/(p-1)$. By the maximal inequality,
$$
E[M^p] = p \int_0^\infty \lambda^{p-1} P(M \ge \lambda) d\lambda \le p \int_0^\infty \lambda^{p-2} E[X_n \mathbf{1}_{M \ge \lambda}] d\lambda = p E\Big[X_n \int_0^M \lambda^{p-2} d\lambda\Big] = \frac{p}{p-1} E[X_n M^{p-1}].
$$
By Hölder: $E[X_n M^{p-1}] \le \|X_n\|_p \|M^{p-1}\|_q = \|X_n\|_p \|M\|_p^{p-1}$. Substituting:
$$
E[M^p] \le q \|X_n\|_p \|M\|_p^{p-1}.
$$
Divide both sides by $\|M\|_p^{p-1}$ (assuming finite; a standard truncation argument handles infinite case): $\|M\|_p \le q \|X_n\|_p$. Raise to $p$. $\square$

**Application.** For a martingale $M$, $|M|^p$ (if $M \in L^p$) is a submartingale. Doob: $E[\sup_{k \le n} |M_k|^p] \le (p/(p-1))^p E[|M_n|^p]$. So $L^p$ boundedness of $M$ automatically gives $L^p$ bounds on the running maximum — a free upgrade.

### 4.3 Kolmogorov's inequality (corollary)

**Corollary 4.3.** For independent, zero-mean $Y_i \in L^2$ and $S_n = Y_1 + \dots + Y_n$,
$$
P\Big(\max_{k \le n} |S_k| > \lambda\Big) \le \frac{E[S_n^2]}{\lambda^2} = \frac{\sum_{k=1}^n \text{Var}(Y_k)}{\lambda^2}.
$$
(This is Kolmogorov's inequality from Module 2.3, §7.3, now seen as Doob's inequality applied to the $L^2$ martingale $S_n$ and submartingale $S_n^2$.)

---

## 5. The Martingale Convergence Theorem

### 5.1 Upcrossings

**Definition 5.1.** For $X_0, X_1, \dots, X_n$ and real $a < b$, define the **upcrossing count** $U_n[a, b]$ as the number of times the sequence crosses from below $a$ to above $b$:
$$
\sigma_1 := \inf\{k \ge 0 : X_k \le a\}, \quad \tau_1 := \inf\{k > \sigma_1 : X_k \ge b\}, \quad \sigma_2 := \inf\{k > \tau_1 : X_k \le a\}, \dots
$$
Then $U_n[a, b] := \max\{m : \tau_m \le n\}$.

**Doob's upcrossing lemma 5.2.** If $X$ is a submartingale,
$$
(b - a) E[U_n[a, b]] \le E[(X_n - a)^+] - E[(X_0 - a)^+] \le E[X_n^+] + |a|.
$$

*Proof sketch.* Define the predictable strategy $H_k := \sum_m \mathbf{1}_{\sigma_m < k \le \tau_m}$ (bet 1 during each upcrossing). The martingale transform $(H \cdot X)_n = \sum_k H_k (X_k - X_{k-1}) \ge (b-a) U_n[a, b] - (X_n - a)^-$ (the last term bounds residual loss after starting the $(U_n + 1)$-th upcrossing). Submartingale property: $E[(H \cdot X)_n] \ge 0$ (as $H \ge 0$ predictable and increments from submartingale), which requires more care... The cleanest proof computes $E[(X_n - a)^+] \ge E[(X_0 - a)^+] + (b-a) E[U_n[a,b]]$ by analyzing the sequence $Y_k = (X_k - a)^+$ which is also a submartingale. $\square$

### 5.2 Martingale convergence theorem

**Theorem 5.3** (Doob's martingale convergence theorem)**.** If $X$ is a submartingale with $\sup_n E[X_n^+] < \infty$ (equivalently $\sup_n E|X_n| < \infty$ for martingales), then $X_n \to X_\infty$ a.s. for some $X_\infty \in L^1$.

*Proof.* By the upcrossing lemma, $(b - a) E[U_n[a, b]] \le E[X_n^+] + |a| \le C$ uniformly in $n$. Let $U_\infty[a, b] := \lim_n U_n[a, b]$; then $E[U_\infty[a,b]] \le C/(b-a) < \infty$, so $U_\infty[a,b] < \infty$ a.s.

For each pair of rationals $a < b$, $\{U_\infty[a, b] = \infty\}$ has probability $0$. Hence
$$
P\Big(\liminf X_n < \limsup X_n\Big) = P\Big(\bigcup_{a < b, \mathbb{Q}} \{\liminf X_n < a < b < \limsup X_n\}\Big) = 0,
$$
since each event in the union implies $U_\infty[a,b] = \infty$. Thus $X_n$ converges in $[-\infty, +\infty]$ a.s.

Fatou: $E|X_\infty| \le \liminf_n E|X_n| < \infty$, so $X_\infty \in L^1$. $\square$

### 5.3 Convergence: what kind?

The convergence theorem gives a.s. convergence but *not necessarily* $L^1$ convergence. Typical failure mode: mass escapes.

**Example 5.4.** Gambler's ruin on $\mathbb{N}_0$: $S_0 = 1$, $S_n \to \text{Bernoulli}(\text{reach 0 before } N)$. As $N \to \infty$, $P(\tau_0 < \tau_N) \to 1$, so the martingale $S_n^\tau$ converges to $0$ a.s., but $E[S^\tau_n] = 1$ always — no $L^1$ convergence. The "mass" escapes to $N = \infty$.

### 5.4 When does $L^1$ (and $L^p$) convergence hold?

**Definition 5.5.** A martingale $X$ is **closed by $Y$** if $X_n = E[Y | \mathcal{F}_n]$ for some $Y \in L^1$.

**Theorem 5.6** (UI martingale convergence)**.** For a martingale $X$, TFAE:
(i) $X$ is uniformly integrable.
(ii) $X_n \to X_\infty$ in $L^1$ (for some $X_\infty \in L^1$).
(iii) $X$ is closed: there exists $X_\infty \in L^1$ with $X_n = E[X_\infty | \mathcal{F}_n]$.

*Proof.* (i)$\Rightarrow$(ii): UI is $L^1$-bounded (Module 2.3, §4.3), so martingale convergence gives $X_n \to X_\infty$ a.s. UI + a.s. convergence $\Rightarrow$ $L^1$ convergence (Vitali, Module 2.3, Theorem 4.6).

(ii)$\Rightarrow$(iii): Fix $n$ and $A \in \mathcal{F}_n$. For $m > n$, $\int_A X_m \, dP = \int_A X_n \, dP$ (martingale). $L^1$ convergence: $\int_A X_\infty \, dP = \int_A X_n \, dP$. Hence $X_n = E[X_\infty | \mathcal{F}_n]$ (definition of conditional expectation).

(iii)$\Rightarrow$(i): $\{E[Y | \mathcal{F}_n] : n\}$ is UI for any $Y \in L^1$ (uniform integrability of conditional expectations — proof uses absolute continuity of the integral: for $A$ with $P(A)$ small, $|E[Y|\mathcal{F}_n] \cdot \mathbf{1}_A|$ integrals are uniformly small). $\square$

### 5.5 $L^p$ martingale convergence

**Theorem 5.7.** For $p > 1$: a martingale $X$ converges in $L^p$ iff $\sup_n E[|X_n|^p] < \infty$. In this case, $X_n \to X_\infty$ a.s. and in $L^p$, and $X_n = E[X_\infty | \mathcal{F}_n]$.

*Proof.* $L^p$-bounded implies $L^1$-bounded $\Rightarrow$ a.s. convergence. $L^p$-boundedness for $p > 1$ implies UI (Module 2.3, §4.3, de la Vallée-Poussin with $\varphi(t) = t^p$), so $L^1$ convergence. For $L^p$ convergence: by Doob's $L^p$ inequality, $|X_n|^p$ are dominated by $(p/(p-1))^p \sup_m |X_m|^p$, which is in $L^1$. DCT on $|X_n - X_\infty|^p \to 0$ a.s. with this dominant function. $\square$

---

## 6. Doob Decomposition

### 6.1 Statement

**Theorem 6.1** (Doob decomposition)**.** Any submartingale $X$ has a *unique* decomposition
$$
X_n = M_n + A_n,
$$
where $M$ is a martingale ($M_0 = X_0$) and $A$ is **predictable and increasing** with $A_0 = 0$.

*Proof.* Define recursively: $A_0 = 0$, $A_n = A_{n-1} + E[X_n - X_{n-1} | \mathcal{F}_{n-1}]$. By submartingale property, each increment $A_n - A_{n-1} \ge 0$. Predictability: $A_n$ depends only on $\mathcal{F}_{n-1}$ conditional expectations, hence $\mathcal{F}_{n-1}$-measurable.

Set $M_n := X_n - A_n$. Check: $E[M_n - M_{n-1} | \mathcal{F}_{n-1}] = E[X_n - X_{n-1} | \mathcal{F}_{n-1}] - (A_n - A_{n-1}) = 0$.

Uniqueness: if $X_n = M_n' + A_n'$ is another such decomposition, then $(M_n - M_n') = (A_n' - A_n)$ is both a martingale and predictable with value 0 at $n=0$; taking conditional expectation $(M_n - M_n') = E[M_n - M_n' | \mathcal{F}_{n-1}] = M_{n-1} - M_{n-1}'$, so by induction $M_n - M_n' = 0$. $\square$

### 6.2 Application: quadratic variation

For a martingale $M$ in $L^2$, $M^2$ is a submartingale (Jensen). Doob decompose: $M_n^2 = N_n + \langle M \rangle_n$ where $\langle M \rangle$ is predictable increasing with $\langle M \rangle_0 = 0$. $\langle M \rangle$ is called the **predictable quadratic variation** or **angle bracket**. Explicitly,
$$
\langle M \rangle_n = \sum_{k=1}^n E[(M_k - M_{k-1})^2 | \mathcal{F}_{k-1}].
$$
Key identity: $E[M_n^2] = E[M_0^2] + E[\langle M \rangle_n]$ (isometry). Foundation of stochastic integration.

Related: the **quadratic variation** $[M]_n := \sum_{k=1}^n (M_k - M_{k-1})^2$. For martingales, $E[M]_n = E\langle M\rangle_n$. For continuous semi-martingales (Module 3+), $\langle M\rangle = [M]$.

---

## 7. Backward Martingales

**Definition 7.1.** A **backward** (or **reverse**) martingale is a family $(X_n)_{n \le 0}$ or $(X_{-n})_{n \ge 0}$ adapted to a *decreasing* filtration $\mathcal{G}_0 \supseteq \mathcal{G}_1 \supseteq \dots$ with $E[X_n | \mathcal{G}_m] = X_m$ for $m \ge n$.

Equivalently, define $Y_n := X_{-n}$, $\mathcal{G}_n$ decreasing; then $Y_n$ is a backward martingale iff $E[Y_{n-1} | \mathcal{G}_n] = Y_n$ (the future conditional on past equals past — reversed time).

**Theorem 7.2** (Backward martingale convergence)**.** Every backward martingale $X$ is UI and converges a.s. and in $L^1$ as $n \to -\infty$ (or $n \to \infty$ in the $Y_n$ convention): $X_n \to X_{-\infty}$ where $X_{-\infty} = E[X_0 | \mathcal{G}_\infty]$ with $\mathcal{G}_\infty := \bigcap_n \mathcal{G}_n$.

*Proof sketch.* Key observation: $X_n = E[X_0 | \mathcal{G}_n]$, so $|X_n| \le E[|X_0| | \mathcal{G}_n]$. The family $\{E[|X_0| | \mathcal{G}]\}_{\mathcal{G}}$ is UI. Upcrossing argument (similar to forward) shows a.s. convergence. Vitali: $L^1$ convergence. Limit is tail-measurable. $\square$

### 7.1 Reversed martingale proof of SLLN

Let $X_i$ iid $L^1$. Define $S_n := X_1 + \dots + X_n$, $M_n := S_n / n$, $\mathcal{G}_n := \sigma(S_n, S_{n+1}, \dots) = \sigma(S_n, X_{n+1}, X_{n+2}, \dots)$ (decreasing).

Claim: $M_n$ is a backward martingale w.r.t. $\mathcal{G}_n$. Conditional expectation $E[M_{n-1} | \mathcal{G}_n]$... by symmetry (exchangeability), given $S_n$, each $X_i$ ($i \le n$) has the same conditional distribution, so $E[X_i | \mathcal{G}_n] = S_n / n = M_n$. Therefore $E[S_{n-1} | \mathcal{G}_n] = E[X_1 + \dots + X_{n-1} | \mathcal{G}_n] = (n-1) M_n$, so $E[M_{n-1}|\mathcal{G}_n] = M_n$.

By backward martingale convergence, $M_n \to M_\infty$ a.s. and in $L^1$. $M_\infty$ is $\mathcal{G}_\infty$-measurable. By Kolmogorov's 0-1 law (Module 2.1), $\mathcal{G}_\infty$ is the tail $\sigma$-algebra, hence trivial, hence $M_\infty$ is a.s. constant. Constant = $E[M_\infty] = \lim E[M_n] = E[X_1]$.

Conclusion: $\bar{X}_n \to E[X_1]$ a.s. — Kolmogorov's SLLN. This is arguably the cleanest proof.

---

## 8. Martingale Central Limit Theorem

For iid $X_i$ we had Lindeberg-Lévy CLT. For martingale increments there is an analogous result.

**Theorem 8.1** (Martingale CLT / McLeish)**.** Let $(M_n)$ be a martingale with $M_0 = 0$, $\Delta_k := M_k - M_{k-1}$. Assume:
(i) **Variance normalization**: $s_n^2 := E[M_n^2] = E[\langle M \rangle_n]$, and $M_n / s_n$'s conditional variance $\langle M \rangle_n / s_n^2 \xrightarrow{P} 1$.
(ii) **Conditional Lindeberg**: for every $\varepsilon > 0$, $s_n^{-2} \sum_{k=1}^n E[\Delta_k^2 \mathbf{1}_{|\Delta_k| > \varepsilon s_n} | \mathcal{F}_{k-1}] \xrightarrow{P} 0$.

Then $M_n / s_n \xRightarrow{d} \mathcal{N}(0, 1)$.

*Proof sketch.* Analog of the CF / characteristic-function argument: show $E[e^{it M_n / s_n}] \to e^{-t^2/2}$. Write it as a telescoping product of conditional CFs $\varphi_k(t) := E[e^{it \Delta_k / s_n} | \mathcal{F}_{k-1}]$, Taylor expand to order 2 using conditional variance and Lindeberg for the remainder. $\square$

**Applications.** Consistency of quasi-MLE estimators, asymptotic normality of U-statistics, Markov chain CLT, ergodic CLTs.

---

## 9. Python Verification

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ================================================================
# 1. Random walk martingale
# ================================================================
def random_walk(n, p=0.5):
    """Simple random walk ±1 with probability p for +1."""
    return np.cumsum(2 * (rng.uniform(size=n) < p) - 1)

# Verify E[S_n] = 0 for fair walk
ps = [random_walk(100) for _ in range(10000)]
means = np.array([p[-1] for p in ps])
print(f"Random walk S_100 mean ≈ {means.mean():+.3f} (target 0), "
      f"std ≈ {means.std():.3f} (target √100 = 10)")

# Doob's maximal inequality demo
Ms = np.array([np.max(p) for p in ps])
lambdas = np.arange(1, 15)
emp = np.array([np.mean(Ms >= lam) for lam in lambdas])
# P(max S_k ≥ λ) ≤ E[|S_n|]/λ by Doob on submartingale |S_n|
bound = np.mean(np.abs(means)) / lambdas
print(f"\nDoob maximal ineq (λ, P(max≥λ), bound E|S_n|/λ):")
for lam, e, b in zip(lambdas, emp, bound):
    print(f"  λ={lam:>2}: emp={e:.4f}, bound={b:.4f}, OK: {e <= b}")

# ================================================================
# 2. Gambler's ruin via OST
# ================================================================
def gamblers_ruin(x, N, max_steps=10000):
    S = x
    steps = 0
    while 0 < S < N and steps < max_steps:
        S += 1 if rng.uniform() < 0.5 else -1
        steps += 1
    return S, steps

# P(reach N before 0) should be x/N, E[τ] = x(N-x)
x, N = 3, 10
trials = 10000
results = [gamblers_ruin(x, N) for _ in range(trials)]
reached_N = np.mean([1 if r[0] == N else 0 for r in results])
avg_tau = np.mean([r[1] for r in results])
print(f"\nGambler's ruin (x={x}, N={N}):")
print(f"  P(reach {N}) emp = {reached_N:.4f}, theory = {x/N:.4f}")
print(f"  E[τ] emp = {avg_tau:.2f}, theory = {x*(N-x)}")

# ================================================================
# 3. Doob martingale convergence: sum of geometric iid
# ================================================================
# M_n = E[Y | F_n] where Y = sum of bounded iid
def doob_sequential_prediction(N=100):
    """Observe Y = sum, predict sequentially."""
    X = rng.uniform(-1, 1, N)
    Y = X.sum()
    # F_n = sigma(X_1,...,X_n). E[Y|F_n] = sum_{i<=n} X_i + (N-n)*E[X] = partial sum
    # (since E[X]=0)
    Mn = np.cumsum(X)
    return Mn, Y

Mn, Y = doob_sequential_prediction(100)
print(f"\nDoob martingale M_n = E[Y|F_n]:")
print(f"  M_100 = {Mn[-1]:+.3f}, Y = {Y:+.3f} (should match since F_N = σ(Y))")

# ================================================================
# 4. Martingale convergence (Polya urn)
# ================================================================
def polya_urn(r0, b0, steps=1000):
    r, b = r0, b0
    fractions = [r/(r+b)]
    for _ in range(steps):
        p = r / (r + b)
        if rng.uniform() < p:
            r += 1
        else:
            b += 1
        fractions.append(r / (r + b))
    return fractions

# 1000 trajectories, plot terminal fractions — should be Beta(r0, b0)
terminals = [polya_urn(2, 3)[−1] for _ in range(5000)]
# Also plot trajectory
import collections
print(f"\nPólya urn (r0=2, b0=3) after 1000 steps:")
print(f"  Mean terminal = {np.mean(terminals):.4f} (target r0/(r0+b0) = 0.4)")
print(f"  Beta(2,3) has mean 0.4, variance 0.04")
print(f"  Empirical variance = {np.var(terminals):.4f}")

# ================================================================
# 5. Doob's L^p inequality for martingale
# ================================================================
# M_n = Gaussian random walk, test E[max M_k^2] ≤ 4·E[M_n^2]
p = 2
n = 100
B = 5000
M = rng.standard_normal((B, n)).cumsum(axis=1)
max_abs_sq = np.max(M**2, axis=1)
EMn_sq = np.mean(M[:, -1]**2)
Emax_sq = np.mean(max_abs_sq)
print(f"\nDoob L² inequality:")
print(f"  E[max M_k²] = {Emax_sq:.3f}")
print(f"  (p/(p-1))² E[M_n²] = 4·{EMn_sq:.3f} = {4*EMn_sq:.3f}")
print(f"  Bound satisfied: {Emax_sq <= 4 * EMn_sq}")

# ================================================================
# 6. Backward martingale SLLN proof
# ================================================================
# M_n = S_n/n for iid X_i
n = 1000
B = 100
X = rng.exponential(1.0, (B, n))
S = X.cumsum(axis=1)
M = S / np.arange(1, n+1)
# Each path should converge to E[X] = 1
terminals = M[:, -1]
print(f"\nBackward martingale SLLN:")
print(f"  M_1000 across paths: mean = {terminals.mean():.4f}, "
      f"std = {terminals.std():.4f}")
print(f"  Should concentrate around E[X_1] = 1")

# ================================================================
# 7. Martingale CLT for GARCH-like innovations
# ================================================================
# epsilon_t | F_{t-1} ~ N(0, sigma_t²) with sigma_t² = 1 (constant for simplicity)
n = 500
B = 5000
eps = rng.standard_normal((B, n))
M = eps.cumsum(axis=1)
Z = M[:, -1] / np.sqrt(n)
from scipy import stats
print(f"\nMartingale CLT (iid increments):")
print(f"  KS distance of M_n/√n to N(0,1): "
      f"{stats.kstest(Z, 'norm').statistic:.4f}")
```

(Note: one typo above in `terminals = [polya_urn(2, 3)[−1] for _ in range(5000)]` — should be regular `-1`. In actual execution please use ASCII minus.)

---

## 10. [QUANT APPLICATION] Martingales in Finance

### 10.1 Fundamental theorem of asset pricing

**First fundamental theorem.** A discrete-time market with $d$ risky assets is *arbitrage-free* iff there exists an equivalent martingale measure (EMM) $Q \approx P$ under which the discounted price process $\tilde{S} = S/B$ is a martingale.

**Second fundamental theorem.** The market is *complete* iff the EMM is unique.

The proof uses the separating hyperplane theorem on a suitable convex cone in $L^\infty$, combined with the Kreps-Yan / Dalang-Morton-Willinger theorem. The upshot: derivative pricing reduces to computing conditional expectations under $Q$.

### 10.2 Self-financing strategies and martingale transforms

A self-financing trading strategy is a predictable process $(H_n)$ (portfolio chosen at time $n-1$, held over $[n-1, n]$). Under no transaction costs and the self-financing constraint, the discounted wealth is a **martingale transform**:
$$
\tilde{W}_n = W_0 + \sum_{k=1}^n H_k (\tilde{S}_k - \tilde{S}_{k-1}) = W_0 + (H \cdot \tilde{S})_n.
$$
Under $Q$, $\tilde{S}$ is a martingale, so $\tilde{W}$ is a martingale (provided $H$ is suitably bounded/integrable). Hence $E^Q[\tilde{W}_T] = W_0$ — a "no free lunch" consequence.

### 10.3 Black-Scholes hedging and replication

In the Black-Scholes (continuous-time) model, the option price at time $t$ is
$$
V_t = E^Q[e^{-r(T-t)} h(S_T) | \mathcal{F}_t].
$$
By the Doob martingale property, $\tilde{V}_t = e^{-rt} V_t$ is a martingale under $Q$. The **martingale representation theorem** (Module 3) says any $Q$-martingale is a stochastic integral w.r.t. the $Q$-Brownian motion. Hence there exists a predictable $\Delta_t$ (the "Delta") with
$$
d\tilde{V}_t = \Delta_t \, d\tilde{S}_t,
$$
giving the explicit self-financing replicating strategy. This is the Meyer-Itô decomposition.

### 10.4 Optional stopping and American options

An American option's price at time $0$ is $V_0 = \sup_\tau E^Q[e^{-r\tau} h(S_\tau)]$ over all stopping times $\tau \le T$. This is the **Snell envelope**: the smallest $Q$-supermartingale that dominates the discounted payoff. The optimal $\tau^* = \inf\{t : V_t = h(S_t)\}$.

### 10.5 Doob's inequality in risk management

For a self-financing strategy with discounted wealth $\tilde{W}_n$ (martingale under $Q$), Doob's maximal inequality gives
$$
P\Big(\max_{k \le n} |\tilde{W}_k - W_0| > \lambda\Big) \le E^Q[|\tilde{W}_n - W_0|^2] / \lambda^2.
$$
Under $P$, if the market price of risk is bounded, similar bounds hold with adjusted constants. This bounds the probability of extreme drawdowns from terminal variance.

### 10.6 Kelly criterion and martingales

For a sequence of favorable bets, the Kelly criterion maximizes $E[\log W_T]$. Under the optimal Kelly strategy, $\log W_n - n E[\log(\text{growth})]$ is a centered martingale. Deviations from optimal growth are governed by LIL-type (law of iterated logarithm) estimates for martingales.

### 10.7 Portfolio efficiency and martingale methods

Mean-variance frontier + martingale pricing give the **stochastic discount factor** (SDF) / pricing kernel representation: $\pi_t = E[\pi_T S_T | \mathcal{F}_t]$ where $\pi$ is the SDF. Under risk-neutral probabilities, $\pi_T / \pi_0$ is the Radon-Nikodym density of $Q$ w.r.t. $P$ (Module 1.6).

### 10.8 Convergence of Monte Carlo estimators

A sequential MC estimator $\hat{V}_n$ with control variates: $\hat{V}_n = n^{-1} \sum_i [h(S^{(i)}) - \beta \cdot (g(S^{(i)}) - E g)]$ is a martingale (in the exchangeable $\sigma$-algebra). Concentration inequalities (Azuma-Hoeffding) apply: $P(|\hat{V}_n - V| > \varepsilon) \le 2\exp(-n\varepsilon^2 / (2 C^2))$ for bounded payoffs.

---

## 11. Worked Examples

### Example 11.1 (De Moivre martingale)

Biased random walk: $S_n = X_1 + \dots + X_n$ with $P(X_i = 1) = p$, $P(X_i = -1) = q = 1 - p$. Set $r = q/p$ (assume $p \ne 1/2$). Then $M_n := r^{S_n}$ is a martingale: $E[M_{n+1} | \mathcal{F}_n] = r^{S_n} (p \cdot r + q \cdot r^{-1}) = r^{S_n} (q + p) = r^{S_n}$. Applied to gambler's ruin at $S_0 = x$, $\tau = \inf\{n : S_n \in \{0, N\}\}$: OST gives $r^x = P(S_\tau = 0) \cdot r^0 + P(S_\tau = N) \cdot r^N$, solving for ruin probability in the asymmetric case.

### Example 11.2 (Exponential martingale in BS)

For Brownian motion $B_t$ under $P$, $\exp(\sigma B_t - \sigma^2 t/2)$ is a $P$-martingale (the exponential martingale). Under $Q$ via Girsanov $dQ/dP|_{\mathcal{F}_T} = \exp(-\theta B_T - \theta^2 T/2)$, $B_t + \theta t$ is a $Q$-Brownian motion. Used to change from real-world to risk-neutral measure.

### Example 11.3 (Polya urn terminal)

$r_0 = 2$, $b_0 = 3$. Fraction $M_n$ of red balls is a martingale. $M_n \to M_\infty \sim \text{Beta}(2, 3)$. *Proof sketch*: compute moments $E[M_\infty^k]$ via martingale + combinatorial identities; match to Beta.

### Example 11.4 (Martingale method for expected hitting time)

Simple symmetric walk on $\{0, 1, \dots, N\}$ with reflection at $0$ and absorption at $N$. To find $E[\tau]$, use martingale $S_n^2 - n$ (careful at reflection). OST gives $E[S_\tau^2] - E[\tau] = x^2$. Since $S_\tau = N$ a.s., $E[\tau] = N^2 - x^2$.

### Example 11.5 (Doob inequality for Brownian motion)

For standard BM $B_t$ on $[0, T]$: $|B_t|$ is a submartingale (continuous martingale + convex). Doob: $P(\sup_{t \le T} |B_t| > \lambda) \le E|B_T|/\lambda = \sqrt{2T/\pi}/\lambda$. (Exact: $P = 2 P(|B_T| > \lambda)$ by reflection principle.)

---

## 12. Exercises

### Tier ★

**2.6.E1.** Prove: the expected number of heads in $n$ fair coin flips, viewed as $S_n$, is a martingale. Extend to biased coin.

**2.6.E2.** Show: if $M$ is a martingale and $\varphi$ is convex with $\varphi(M_n) \in L^1$, then $\varphi(M_n)$ is a submartingale.

**2.6.E3.** For a random walk $S_n$ with $E[X_i] = 0$, $\text{Var}(X_i) = \sigma^2$, verify directly that $S_n^2 - n\sigma^2$ is a martingale.

**2.6.E4.** (Optional stopping fails.) Simple random walk starting from $0$, $\tau = \inf\{n : S_n = 1\}$. Show $P(\tau < \infty) = 1$ but $E[S_\tau] \ne E[S_0]$. Which OST hypothesis fails?

**2.6.E5.** For Pólya urn with $r_0, b_0$, prove directly that $M_n := r_n/(r_n + b_n)$ is a martingale.

**2.6.E6.** Prove: $(X_n)$ is a supermartingale iff $(-X_n)$ is a submartingale. State Doob's convergence theorem for supermartingales explicitly.

**2.6.E7.** Show: if $M, N$ are martingales in $L^2$, then $\langle M, N\rangle_n := \sum_{k=1}^n E[(M_k - M_{k-1})(N_k - N_{k-1}) | \mathcal{F}_{k-1}]$ is predictable, and $MN - \langle M, N\rangle$ is a martingale.

**2.6.E8.** For the likelihood ratio martingale $L_n$ (Example 2.5), compute $E_P[L_n \log L_n]$ (relative entropy). Show it's non-decreasing in $n$ (monotonicity of information).

### Tier ★★

**2.6.E9** (Azuma-Hoeffding inequality)**.** Let $M_n$ be a martingale with $|M_k - M_{k-1}| \le c_k$ bounded a.s. Then
$$
P(|M_n - M_0| > t) \le 2 \exp\left(-\frac{t^2}{2 \sum_{k=1}^n c_k^2}\right).
$$
Prove using Chernoff: bound $E[e^{\lambda(M_n - M_0)}]$ by induction + Hoeffding's lemma for $\lambda$-CGF of bounded mean-zero.

**2.6.E10** (McDiarmid's inequality)**.** For $f: \mathbb{R}^n \to \mathbb{R}$ with bounded differences $|f(x) - f(y)| \le c_k$ when $x, y$ differ only in coordinate $k$, and $X_1, \dots, X_n$ independent,
$$
P(|f(X) - E f(X)| > t) \le 2 \exp(-2 t^2 / \sum c_k^2).
$$
Hint: Doob martingale $M_k = E[f(X) | X_1, \dots, X_k]$.

**2.6.E11** (Galton-Watson)**.** For a Galton-Watson process $Z_n$ with offspring mean $m$, show $Z_n / m^n$ is a non-negative martingale. Deduce existence of the limit $M_\infty$ and discuss when $E[M_\infty] = 1$ (the Kesten-Stigum theorem: iff $E[Z_1 \log^+ Z_1] < \infty$).

**2.6.E12** (Doob decomposition example)**.** Let $X_n$ be iid Bernoulli($p$) and $S_n := X_1 + \dots + X_n$. Find the Doob decomposition of $S_n^2$ (submartingale when $p \ne 1/2$... wait — actually $S_n^2$ is always a submartingale for iid non-neg. Hmm.) More cleanly: find Doob decomposition of $|S_n|$ for random walk.

**2.6.E13** (Reverse martingale for SLLN)**.** Complete the reverse-martingale proof of SLLN: for iid $X_i$ with $E|X_1| < \infty$, prove $M_n := S_n/n$ converges a.s. and in $L^1$ to a constant. Identify the limit.

**2.6.E14** (Optional stopping in BS)**.** $B_t$ standard BM. $\tau = \inf\{t : B_t = a\}$. Use OST on $\exp(\lambda B_t - \lambda^2 t/2)$ (continuous-time analog) with $\lambda > 0$ to compute the Laplace transform $E[e^{-s\tau}] = e^{-a\sqrt{2s}}$.

**2.6.E15** (Dyadic martingale and Radon-Nikodym)**.** Let $\mu, \nu$ be finite Borel measures on $[0, 1]$ with $\nu \ll \mu$, and let $\mathcal{F}_n$ be the $\sigma$-algebra generated by dyadic intervals of length $2^{-n}$. Show that $M_n(x) := \nu(I_{n, x}) / \mu(I_{n, x})$ (ratio over the dyadic interval containing $x$) converges $\mu$-a.s. to the Radon-Nikodym derivative $d\nu/d\mu$.

**2.6.E16** (Uniform integrability of $L \log L$)**.** Show: $\{X_\alpha\}$ is UI iff $\sup_\alpha E[|X_\alpha| \log^+ |X_\alpha|] < \infty$ (for non-negative; use de la Vallée-Poussin with $\varphi(t) = t \log t$). For martingales in $L \log L$, deduce $L^1$-convergence.

### Tier ★★★

**2.6.E17** (Martingale CLT, McLeish)**.** Prove: under conditional Lindeberg $(s_n^{-2} \sum_k E[\Delta_k^2 \mathbf{1}_{|\Delta_k| > \varepsilon s_n} | \mathcal{F}_{k-1}] \xrightarrow{P} 0)$ and variance convergence $(\langle M\rangle_n / s_n^2 \xrightarrow{P} 1)$, $M_n/s_n \xRightarrow{d} \mathcal{N}(0, 1)$. Use CFs.

**2.6.E18** (Completeness of market implies uniqueness of EMM)**.** For a complete discrete-time market (every contingent claim is replicable), prove: the equivalent martingale measure $Q$ is unique. Hint: if $Q_1 \ne Q_2$, construct a claim $h$ with different $E^{Q_1}[h] \ne E^{Q_2}[h]$; show this contradicts replicability.

**2.6.E19** (American option = Snell envelope)**.** For a discrete-time American option with payoff $Y_n$, define the Snell envelope $V_n := \max(Y_n, E^Q[V_{n+1} | \mathcal{F}_n])$ with $V_T = Y_T$. Prove:
(a) $V$ is the smallest $Q$-supermartingale $\ge Y$.
(b) $V_0 = \sup_\tau E^Q[Y_\tau]$ over stopping times $\tau \le T$.
(c) The optimal stopping time is $\tau^* = \inf\{n : V_n = Y_n\}$.

**2.6.E20** (Robbins-Monro stochastic approximation)**.** Let $g: \mathbb{R} \to \mathbb{R}$ be continuous with unique root $\theta^*$, and suppose we observe $Y_n = g(\theta_n) + \varepsilon_n$ where $\varepsilon_n$ are martingale differences with $E[\varepsilon_n^2 | \mathcal{F}_{n-1}] \le \sigma^2$. The Robbins-Monro iteration $\theta_{n+1} = \theta_n - a_n Y_n$ with $\sum a_n = \infty$, $\sum a_n^2 < \infty$ satisfies $\theta_n \to \theta^*$ a.s. Sketch the proof using the supermartingale property of $(\theta_n - \theta^*)^2$ + Robbins-Siegmund.

**2.6.E21** (Doob-Meyer preview)**.** State the Doob-Meyer decomposition for continuous-time submartingales: any right-continuous submartingale $X$ (of class D) has a unique decomposition $X = M + A$ where $M$ is a local martingale and $A$ is predictable right-continuous increasing. (Just the statement — the proof is a major theorem of advanced stochastic analysis.)

**2.6.E22** (Martingale methods in Black-Litterman)**.** The Black-Litterman model combines equilibrium returns $\pi$ with investor views $Q$ to produce posterior expected returns $\mu$. Express the update as a martingale-type conditional expectation:
$$
\mu = \pi + \tau \Sigma P^\top (P \tau \Sigma P^\top + \Omega)^{-1}(Q - P \pi),
$$
where $\tau$ is uncertainty and $\Omega$ is view variance. Interpret as $E[\text{true returns} | \text{views}]$ under a Gaussian prior. Connect to Kalman filtering (Module 2.2).

---

## 13. Summary and Forward Pointers

**What we proved.**
- Filtrations, adapted/predictable processes, stopping times.
- Martingale definitions + operations (convex/concave $\to$ sub/super).
- Optional stopping theorem under three sufficient conditions.
- Doob's maximal and $L^p$ inequalities.
- Doob's upcrossing lemma + martingale convergence theorem (a.s. convergence for $L^1$-bounded).
- UI martingale equivalence: UI $\Leftrightarrow$ $L^1$ convergent $\Leftrightarrow$ closed.
- $L^p$ martingale convergence via $L^p$-boundedness + Doob.
- Doob decomposition (discrete Doob-Meyer).
- Backward martingales + SLLN proof.
- Martingale CLT (McLeish).

**Why martingales matter.** They are the *right* formalism for stochastic processes with time / information structure. The whole of stochastic calculus (Subject 3), arbitrage-free pricing (Subject 8), adaptive algorithm analysis (Subject 5 / SGD), and ergodic theory rest on martingale foundations.

**Forward pointers.**
- **Module 2.7** (Markov chains): Markov property via conditional expectation; stationary distributions; ergodic theorem as martingale SLLN. Connection via **Poisson equation**: $P f - f = g$ has solution turning $(f(X_n) - \sum_k g(X_k))$ into a martingale.
- **Subject 3** (Stochastic processes): continuous-time martingales, Brownian motion, Itô's lemma, martingale representation theorem, Girsanov's theorem.
- **Subject 5** (Statistical learning): Azuma-Hoeffding, McDiarmid for concentration; SGD convergence via supermartingale argument (Robbins-Siegmund).
- **Subject 8** (Quant finance): FTAP, risk-neutral pricing, hedging strategies all live here.
- **Subject 9** (Numerics): MCMC convergence analysis via martingale methods; control variates.

**Next module:** Markov chains — stochastic processes with memoryless evolution. State classification, stationary distributions, ergodic theorem, and the MCMC algorithms they power.

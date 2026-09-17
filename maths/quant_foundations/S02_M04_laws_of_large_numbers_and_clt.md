# Module 2.4: Laws of Large Numbers and the Central Limit Theorem

**Subject 2: Probability Theory** · Module 4 of 7

---

## 0. Prerequisites and Position in the Curriculum

- **Module 1.3** (Lebesgue integration), **Module 1.5** ($L^p$ spaces), **Module 1.6** (Radon-Nikodym).
- **Module 2.1** (probability spaces, Borel-Cantelli, 0–1 laws).
- **Module 2.2** (expectation, conditional expectation, moment inequalities).
- **Module 2.3** (modes of convergence, Kolmogorov's three-series, tightness/Prokhorov).

This module proves the **two pillars of classical probability**: the **Law of Large Numbers** (weak and strong forms) and the **Central Limit Theorem**. These are arguably the most important theorems in applied probability — they justify everything from Monte Carlo integration to statistical inference to Black-Scholes hedging error asymptotics.

We prove:

- **WLLN** under $L^1$ (Khinchin) and under $L^2$ (Bernoulli-Chebyshev).
- **SLLN** under $L^4$ (Cantelli's direct method), under $L^2$ with independence (Rademacher-Menchov-like), and Etemadi's $L^1$ proof using only pairwise independence.
- **Classical CLT** (Lindeberg-Lévy) via characteristic functions.
- **Lindeberg-Feller CLT** for triangular arrays.
- **Berry-Esseen theorem** (rate of convergence) — statement plus sketch.
- Applications: Monte Carlo, hypothesis testing, portfolio CLT, discretization error.

---

## 1. Warm-up: WLLN under Second Moments

Start with the simplest setting that captures the intuition: iid $L^2$ random variables.

**Theorem 1.1** (Weak LLN, $L^2$ version / Bernoulli-Chebyshev)**.** Let $X_1, X_2, \dots$ be iid with $E[X_1] = \mu$ and $\text{Var}(X_1) = \sigma^2 < \infty$. Let $\bar{X}_n := n^{-1} \sum_{k=1}^n X_k$. Then
$$
\bar{X}_n \xrightarrow{L^2} \mu \quad \text{and hence} \quad \bar{X}_n \xrightarrow{P} \mu.
$$

*Proof.* $E[\bar{X}_n] = \mu$ and $\text{Var}(\bar{X}_n) = \sigma^2/n$ by independence. So
$$
\|\bar{X}_n - \mu\|_2^2 = E[(\bar{X}_n - \mu)^2] = \sigma^2 / n \to 0.
$$
$L^2 \Rightarrow P$ (Module 2.3, Theorem 2.1). Or directly via Chebyshev:
$$
P(|\bar{X}_n - \mu| > \varepsilon) \le \frac{\sigma^2}{n \varepsilon^2} \to 0. \quad\square
$$

This is the version taught in introductory probability. Astonishingly, it was Bernoulli's discovery (1713) for the binomial case, and is the foundation of frequentist statistics.

**Improvements needed.** We want:
1. To drop the $L^2$ assumption to $L^1$ (Khinchin's WLLN).
2. To upgrade $P$ to a.s. (SLLN).
3. To quantify the rate — what is the order of $\bar{X}_n - \mu$? (CLT answers $n^{-1/2}$.)

---

## 2. WLLN under First Moments (Khinchin)

**Theorem 2.1** (Khinchin's WLLN)**.** Let $X_1, X_2, \dots$ be iid with $E|X_1| < \infty$ and $E[X_1] = \mu$. Then $\bar{X}_n \xrightarrow{P} \mu$.

The standard proof uses truncation. Define $Y_k^n := X_k \mathbf{1}_{|X_k| \le n}$. The idea: $Y_k^n$ are uniformly bounded (so Chebyshev applies), the truncation only changes finitely many $X_k$ with high probability (so the answer matches), and the mean error from truncation is small.

*Proof.* Fix $\varepsilon > 0$. Define
$$
Y_k := X_k \mathbf{1}_{|X_k| \le k}, \quad S_n := \sum_{k=1}^n Y_k, \quad T_n := \sum_{k=1}^n X_k.
$$

**Step 1 (truncation doesn't matter, in probability).** $P(X_k \ne Y_k) = P(|X_k| > k)$. By tail sum:
$$
\sum_{k=1}^\infty P(|X_1| > k) \le E|X_1| < \infty.
$$
So $\sum_k P(X_k \ne Y_k) < \infty$. By Borel-Cantelli I, $P(X_k \ne Y_k \text{ i.o.}) = 0$. Hence $(T_n - S_n)/n \to 0$ a.s.; in particular $T_n/n - S_n/n \xrightarrow{P} 0$, and thus $\bar{X}_n = T_n/n$ has the same limit as $S_n / n$ in probability.

**Step 2 (centered truncated variables have vanishing variance).** We show $\text{Var}(Y_k) \le E[X_1^2 \mathbf{1}_{|X_1| \le k}]$. Write $\sigma_k^2 := \text{Var}(Y_k) \le E[Y_k^2] = E[X_1^2 \mathbf{1}_{|X_1| \le k}]$. Now
$$
\frac{1}{n^2} \sum_{k=1}^n \sigma_k^2 \le \frac{1}{n^2} \sum_{k=1}^n E[X_1^2 \mathbf{1}_{|X_1| \le k}].
$$
By layer-cake, $E[X_1^2 \mathbf{1}_{|X_1| \le k}] = \int_0^{k^2} P(X_1^2 > t, |X_1| \le k) \, dt = \int_0^k 2u P(u < |X_1| \le k) \, du$. Actually it's cleaner to bound differently:
$$
\sum_{k=1}^n E[X_1^2 \mathbf{1}_{|X_1| \le k}] = \sum_{k=1}^n \sum_{j=1}^k E[X_1^2 \mathbf{1}_{j-1 < |X_1| \le j}] = \sum_{j=1}^n (n - j + 1) E[X_1^2 \mathbf{1}_{j-1 < |X_1| \le j}]
$$
$$
\le n \sum_{j=1}^n E[X_1^2 \mathbf{1}_{j-1 < |X_1| \le j}] \le n \sum_{j=1}^n j \cdot E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}]
$$
(using $X_1^2 \le j |X_1|$ on $\{|X_1| \le j\}$). So
$$
\frac{1}{n^2} \sum_{k=1}^n \sigma_k^2 \le \frac{1}{n} \sum_{j=1}^n j \cdot E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}] =: \frac{1}{n} \Sigma_n.
$$
We claim $\Sigma_n / n \to 0$. Indeed
$$
\Sigma_n = \sum_{j=1}^n j \cdot E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}] \le E|X_1| + \sum_{j=2}^n j \cdot E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}].
$$
Now $j \le 2(j-1)$ for $j \ge 2$, so $j E[|X_1| \cdot \mathbf{1}_{j-1 < |X_1| \le j}] \le 2 E[|X_1|(j-1) \mathbf{1}_{j-1 < |X_1| \le j}] \le 2 E[X_1^2 \mathbf{1}_{j-1 < |X_1| \le j}]$, wait this is circular. Let's instead argue directly: $\Sigma_n / n \to 0$ is *Cesàro summation*: $a_j := j E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}]$, and
$$
\sum_{j \ge 1} E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}] = E|X_1| < \infty,
$$
so $a_j \le j \cdot b_j$ where $b_j := E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}]$ is summable. But $a_j \le E[|X_1|^2 \mathbf{1}_{j-1 < |X_1| \le j}]$... let me restart with the cleaner approach.

**Step 2 (cleaner).** By Chebyshev,
$$
P\Big(\Big|\frac{S_n - E[S_n]}{n}\Big| > \varepsilon\Big) \le \frac{\text{Var}(S_n)}{n^2 \varepsilon^2} = \frac{1}{n^2 \varepsilon^2} \sum_{k=1}^n \text{Var}(Y_k).
$$
We use $\text{Var}(Y_k) \le E[Y_k^2] = E[X_1^2; |X_1| \le k]$. Now we estimate $n^{-2} \sum_{k=1}^n E[X_1^2; |X_1| \le k]$. Split $E[X_1^2; |X_1| \le k] = E[X_1^2; |X_1| \le A] + E[X_1^2; A < |X_1| \le k]$ for a parameter $A$:
- First term is a constant $C_A$.
- Second: $E[X_1^2; A < |X_1| \le k] \le k E[|X_1|; |X_1| > A]$.

So $\sum_{k=1}^n E[X_1^2; |X_1| \le k] \le n C_A + \frac{n(n+1)}{2} E[|X_1|; |X_1| > A]$. Dividing by $n^2$:
$$
\frac{\text{Var}(S_n)}{n^2} \le \frac{C_A}{n} + E[|X_1|; |X_1| > A].
$$
Let $n \to \infty$, then $A \to \infty$: first term $\to 0$, second term $\to 0$ by DCT (since $E|X_1| < \infty$). So $\text{Var}(S_n)/n^2 \to 0$.

**Step 3 (mean of truncated vs true).** $E[Y_k] = E[X_1; |X_1| \le k] \to E[X_1] = \mu$ as $k \to \infty$ (DCT). So $E[S_n]/n = n^{-1} \sum_{k=1}^n E[Y_k] \to \mu$ (Cesàro).

**Combining.** $|S_n/n - \mu| \le |S_n/n - E[S_n]/n| + |E[S_n]/n - \mu|$. First term $\xrightarrow{P} 0$ by Step 2; second term is deterministic and $\to 0$ by Step 3. So $S_n/n \xrightarrow{P} \mu$, and by Step 1, $\bar{X}_n \xrightarrow{P} \mu$. $\square$

**Remark 2.2** (Weakening independence)**.** Khinchin's WLLN in fact holds for pairwise uncorrelated $X_k$ in $L^2$: the Chebyshev bound only uses linearity of variance. For general $L^1$ it requires some form of asymptotic independence; iid is the cleanest sufficient condition.

---

## 3. SLLN via Fourth Moments (Cantelli)

A very direct path to SLLN uses $L^4$ and Borel-Cantelli.

**Theorem 3.1** (SLLN, $L^4$ version)**.** Let $X_1, X_2, \dots$ iid with $E[X_1] = 0$ and $E[X_1^4] < \infty$. Then $\bar{X}_n \xrightarrow{\text{a.s.}} 0$.

*Proof.* Let $S_n = X_1 + \dots + X_n$. Compute
$$
E[S_n^4] = \sum_{i,j,k,l} E[X_i X_j X_k X_l].
$$
By independence and $E[X_i] = 0$, the only surviving terms are those where each index appears an *even* number of times:
- All four equal ($i = j = k = l$): $n$ terms, each equals $E[X_1^4]$.
- Two pairs ($\{i=j, k=l, i \ne k\}$ and permutations): $3 n(n-1)$ ordered quadruples, each equals $(E[X_1^2])^2$.

So
$$
E[S_n^4] = n E[X_1^4] + 3 n(n-1) (E[X_1^2])^2 \le C n^2
$$
for some constant $C = E[X_1^4] + 3(E[X_1^2])^2$. By Markov:
$$
P(|\bar{X}_n| > \varepsilon) = P(|S_n| > n\varepsilon) \le \frac{E[S_n^4]}{n^4 \varepsilon^4} \le \frac{C}{n^2 \varepsilon^4}.
$$
Summable! Borel-Cantelli I: $P(|\bar{X}_n| > \varepsilon \text{ i.o.}) = 0$. Taking $\varepsilon = 1/k$ and countable union: $\bar{X}_n \to 0$ a.s. $\square$

This is the cleanest SLLN proof when fourth moments exist. It has the right flavor (Borel-Cantelli on a summable tail) and scales cleanly.

---

## 4. Kolmogorov's SLLN via Three-Series

We can do SLLN under $L^2$ (independence, not iid) and even under $L^1$ (iid) via Kolmogorov's methods from Module 2.3.

### 4.1 Kolmogorov's SLLN for $L^2$

**Theorem 4.1** (Kolmogorov's $L^2$ SLLN)**.** Let $X_1, X_2, \dots$ be independent with $E[X_k] = \mu_k$ and $\text{Var}(X_k) = \sigma_k^2 < \infty$. If
$$
\sum_{k=1}^\infty \frac{\sigma_k^2}{k^2} < \infty,
$$
then $\bar{X}_n - n^{-1}\sum_{k=1}^n \mu_k \xrightarrow{\text{a.s.}} 0$.

*Proof.* WLOG $\mu_k = 0$ (center each $X_k$). Consider the series $\sum_k (X_k - 0)/k = \sum X_k/k$. The summands are independent with $E[X_k/k] = 0$ and $\text{Var}(X_k/k) = \sigma_k^2 / k^2$. By hypothesis, $\sum \text{Var}(X_k/k) < \infty$, so by Kolmogorov's one-series theorem (Module 2.3 §7.5), $\sum_k X_k / k$ converges a.s.

**Kronecker's lemma.** If $a_n \uparrow \infty$ and $\sum b_n / a_n$ converges, then $a_n^{-1} \sum_{k=1}^n b_k \to 0$.

*Proof.* Let $s_n = \sum_{k=1}^n b_k/a_k$ with $s_n \to s$. Abel summation:
$$
\sum_{k=1}^n b_k = \sum_{k=1}^n a_k (s_k - s_{k-1}) = a_n s_n - \sum_{k=1}^{n-1} (a_{k+1} - a_k) s_k.
$$
Divide by $a_n$:
$$
\frac{1}{a_n} \sum_{k=1}^n b_k = s_n - \frac{1}{a_n} \sum_{k=1}^{n-1} (a_{k+1} - a_k) s_k.
$$
As $n \to \infty$, $s_n \to s$, and the second term is a weighted average of $s_k$ with weights summing to $(a_n - a_1)/a_n \to 1$, so the second term also $\to s$. Difference $\to 0$. $\square$

Apply with $a_n = n$, $b_n = X_n$: $\sum X_k / k$ converges a.s., so $\bar{X}_n = n^{-1} \sum X_k \xrightarrow{\text{a.s.}} 0$. $\square$

### 4.2 Kolmogorov's SLLN for iid $L^1$

**Theorem 4.2** (Kolmogorov's SLLN, iid $L^1$)**.** Let $X_1, X_2, \dots$ iid with $E|X_1| < \infty$ and $E[X_1] = \mu$. Then $\bar{X}_n \xrightarrow{\text{a.s.}} \mu$.

*Proof.* WLOG $\mu = 0$. Truncate: $Y_k := X_k \mathbf{1}_{|X_k| \le k}$.

**Step 1** (truncation-equivalent in Borel-Cantelli sense). $\sum_k P(X_k \ne Y_k) = \sum_k P(|X_1| > k) \le E|X_1| < \infty$. By BC-I, $X_k = Y_k$ eventually a.s., so $\bar{X}_n$ and $\bar{Y}_n := n^{-1} \sum_{k=1}^n Y_k$ have the same a.s. limit (if any).

**Step 2** ($E[Y_k] \to 0$). By DCT, $E[Y_k] = E[X_1 \mathbf{1}_{|X_1| \le k}] \to E[X_1] = 0$. Hence $n^{-1}\sum_{k=1}^n E[Y_k] \to 0$ (Cesàro).

**Step 3** (variance summability). Compute
$$
\sum_{k=1}^\infty \frac{\text{Var}(Y_k)}{k^2} \le \sum_{k=1}^\infty \frac{E[Y_k^2]}{k^2} = \sum_{k=1}^\infty \frac{E[X_1^2 \mathbf{1}_{|X_1| \le k}]}{k^2}.
$$
By Fubini (sum exchange) with $X_1^2 \mathbf{1}_{|X_1| \le k} = \sum_{j=1}^k X_1^2 \mathbf{1}_{j - 1 < |X_1| \le j}$:
$$
\sum_{k=1}^\infty \frac{1}{k^2} \sum_{j=1}^k E[X_1^2 \mathbf{1}_{j-1 < |X_1| \le j}] = \sum_{j=1}^\infty E[X_1^2 \mathbf{1}_{j-1 < |X_1| \le j}] \sum_{k=j}^\infty \frac{1}{k^2}.
$$
The inner sum is $\sum_{k \ge j} k^{-2} \le 2/j$ (for $j \ge 1$). On $\{j - 1 < |X_1| \le j\}$, $X_1^2 \le j |X_1|$, so $E[X_1^2 \mathbf{1}_{j-1 < |X_1| \le j}] \le j E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}]$. Combining:
$$
\sum_{k \ge 1} \frac{E[Y_k^2]}{k^2} \le 2 \sum_{j=1}^\infty E[|X_1| \mathbf{1}_{j-1 < |X_1| \le j}] = 2 E|X_1| < \infty.
$$

**Step 4** (apply Kolmogorov's $L^2$ SLLN to $Y_k$). By Theorem 4.1 applied to independent $(Y_k)$ with $\sum \text{Var}(Y_k)/k^2 < \infty$,
$$
\bar{Y}_n - \frac{1}{n} \sum_{k=1}^n E[Y_k] \xrightarrow{\text{a.s.}} 0.
$$
Combined with Step 2 (Cesàro mean $\to 0$), $\bar{Y}_n \xrightarrow{\text{a.s.}} 0$. With Step 1, $\bar{X}_n \xrightarrow{\text{a.s.}} 0$. $\square$

---

## 5. Etemadi's SLLN (Pairwise Independence!)

A striking improvement: SLLN holds under *pairwise* independence (plus iid), which is much weaker than mutual independence.

**Theorem 5.1** (Etemadi's SLLN, 1981)**.** Let $X_1, X_2, \dots$ be *pairwise* independent, identically distributed with $E|X_1| < \infty$ and $E[X_1] = \mu$. Then $\bar{X}_n \xrightarrow{\text{a.s.}} \mu$.

*Proof sketch.* WLOG $\mu = 0$, $X_k \ge 0$ (general case: split into positive and negative parts). Truncate $Y_k = X_k \mathbf{1}_{X_k \le k}$.

**Step 1.** Truncation removes a negligible set (same argument as before).

**Step 2.** Use subsequences $n_j = \lfloor \alpha^j \rfloor$ for $\alpha > 1$. Compute
$$
\sum_{j=1}^\infty \frac{\text{Var}(\sum_{k=1}^{n_j} Y_k)}{n_j^2} = \sum_{j=1}^\infty \frac{1}{n_j^2} \sum_{k=1}^{n_j} \text{Var}(Y_k)
$$
(using pairwise independence for $\text{Var}$ of sums). Interchange:
$$
= \sum_{k=1}^\infty \text{Var}(Y_k) \sum_{j : n_j \ge k} \frac{1}{n_j^2} \le C \sum_{k=1}^\infty \frac{\text{Var}(Y_k)}{k^2}
$$
(geometric tail), which is finite by the variance estimate. Chebyshev + Borel-Cantelli I gives $\bar{Y}_{n_j} \to E[Y_1]$ a.s. along the subsequence (after centering).

**Step 3.** "Fill in" between $n_j$ and $n_{j+1}$: since $X_k \ge 0$, $\bar{X}_n$ is monotonically sandwiched:
$$
\frac{n_j}{n_{j+1}} \bar{X}_{n_j} \le \bar{X}_n \le \frac{n_{j+1}}{n_j} \bar{X}_{n_{j+1}} \quad \text{for } n_j \le n \le n_{j+1}.
$$
Letting $\alpha \downarrow 1$ squeezes $n_{j+1}/n_j \to 1$, so $\bar{X}_n \to \mu$ a.s. $\square$

**Significance.** Only pairwise independence is needed — striking because so many constructions in probability (e.g., some ergodic / random walk models) give pairwise but not mutually independent sequences. Etemadi's theorem is the "right" SLLN.

---

## 6. The Central Limit Theorem

We now turn to the CLT, which states that the properly scaled deviation $\sqrt{n}(\bar{X}_n - \mu)$ is asymptotically normal.

**Theorem 6.1** (Lindeberg-Lévy CLT)**.** Let $X_1, X_2, \dots$ iid with $E[X_1] = \mu$ and $\text{Var}(X_1) = \sigma^2 \in (0, \infty)$. Then
$$
\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} = \frac{S_n - n\mu}{\sigma \sqrt{n}} \xRightarrow{d} \mathcal{N}(0, 1).
$$

### 6.1 Proof via characteristic functions

The **characteristic function** (CF) of a random variable $X$ is $\varphi_X(t) := E[e^{itX}]$ (Module 2.5 develops these in depth). For now we need three facts:

1. CFs characterize distributions (Lévy's uniqueness).
2. **Lévy's continuity theorem**: if $\varphi_{X_n}(t) \to \varphi(t)$ pointwise and $\varphi$ is continuous at $0$, then $\varphi$ is the CF of some random variable $X$ and $X_n \xRightarrow{d} X$.
3. CF of $\mathcal{N}(0, 1)$ is $\varphi(t) = e^{-t^2/2}$.
4. If $E[X^2] < \infty$, then $\varphi_X(t) = 1 + it E[X] - \frac{t^2}{2} E[X^2] + o(t^2)$ as $t \to 0$.

Fact 4 follows from dominating $|e^{itX} - (1 + itX - (tX)^2/2)| \le \min(|tX|^2, |tX|^3/6)$ and DCT.

*Proof of CLT.* WLOG $\mu = 0$, $\sigma^2 = 1$ (standardize). Let $Z_n := S_n/\sqrt{n}$. CF of $Z_n$:
$$
\varphi_{Z_n}(t) = E\big[e^{it S_n / \sqrt{n}}\big] = \prod_{k=1}^n E\big[e^{it X_k/\sqrt{n}}\big] = \varphi_{X_1}(t/\sqrt{n})^n.
$$
As $n \to \infty$, $t/\sqrt{n} \to 0$, and by Fact 4,
$$
\varphi_{X_1}(t/\sqrt{n}) = 1 + \frac{it}{\sqrt{n}} E[X_1] - \frac{t^2}{2n} E[X_1^2] + o(1/n) = 1 - \frac{t^2}{2n} + o(1/n).
$$
Using $\log(1 + z) = z + O(z^2)$ for small $z$:
$$
n \log \varphi_{X_1}(t/\sqrt{n}) = n \cdot \left[-\frac{t^2}{2n} + o(1/n)\right] = -\frac{t^2}{2} + o(1).
$$
Hence $\varphi_{Z_n}(t) \to e^{-t^2/2}$ pointwise. By Lévy's continuity theorem (and since $e^{-t^2/2}$ is continuous at $0$), $Z_n \xRightarrow{d} Z \sim \mathcal{N}(0, 1)$. $\square$

### 6.2 Proof via Lindeberg's swapping method (no CFs)

Historical interest: Lindeberg (1922) proved CLT without CFs. The idea: replace the $X_k$ by Gaussians $g_k \sim \mathcal{N}(0, 1)$ one at a time and control the error.

For any smooth test function $f \in C_b^3$, show
$$
|E[f(S_n/\sqrt{n})] - E[f(Z)]| \to 0 \quad \text{where } Z \sim \mathcal{N}(0, 1).
$$
Define $W_k := (X_1 + \dots + X_k + g_{k+1} + \dots + g_n)/\sqrt{n}$, so $W_0 = (g_1 + \dots + g_n)/\sqrt{n} \sim \mathcal{N}(0, 1)$ and $W_n = S_n/\sqrt{n}$. Telescope:
$$
E[f(W_n)] - E[f(W_0)] = \sum_{k=1}^n \left(E[f(W_k)] - E[f(W_{k-1})]\right).
$$
Each term: $W_k = W_{k-1} - g_k/\sqrt{n} + X_k/\sqrt{n}$. Taylor expand $f$ to order 3:
$$
E[f(W_k)] = E\Big[f(U_k) + f'(U_k) \frac{X_k}{\sqrt{n}} + \frac{f''(U_k)}{2} \frac{X_k^2}{n} + R_k\Big],
$$
where $U_k = (X_1 + \dots + X_{k-1} + g_{k+1} + \dots + g_n)/\sqrt{n}$ and $R_k = O(|X_k|^3 / n^{3/2})$. Similarly for $g_k$ in place of $X_k$. Taking expectations and using that $X_k, g_k$ both have mean $0$ and variance $1$ and are independent of $U_k$, the first two-order terms cancel. Only the third-order remainders survive:
$$
|E[f(W_k)] - E[f(W_{k-1})]| \le C \cdot \frac{E[|X_k|^3] + E[|g_k|^3]}{n^{3/2}} \le C' / n^{3/2}
$$
(assuming finite third moment; Lindeberg's actual condition only needs truncated second moments). Sum over $k$: $|E[f(W_n)] - E[f(W_0)]| \le C'/\sqrt{n} \to 0$. $\square$

This proof extends to the non-iid case with the celebrated *Lindeberg condition*.

---

## 7. Lindeberg-Feller Theorem (Triangular Arrays)

**Setting.** For each $n$, let $X_{n,1}, \dots, X_{n, r_n}$ be independent with $E[X_{n,k}] = 0$ and $\sigma_{n,k}^2 := \text{Var}(X_{n,k}) < \infty$. Let $s_n^2 := \sum_{k=1}^{r_n} \sigma_{n,k}^2$. This includes the iid case (taking $r_n = n$, $X_{n,k} = X_k$, $s_n^2 = n \sigma^2$).

**Lindeberg condition.** For every $\varepsilon > 0$,
$$
L_n(\varepsilon) := \frac{1}{s_n^2} \sum_{k=1}^{r_n} E\left[X_{n,k}^2 \mathbf{1}_{|X_{n,k}| > \varepsilon s_n}\right] \to 0.
$$
Informally: no single term's variance is a large fraction of the total, AND the collective "far tail" contribution vanishes.

**Theorem 7.1** (Lindeberg-Feller CLT)**.** Under the setting above and the Lindeberg condition,
$$
\frac{1}{s_n} \sum_{k=1}^{r_n} X_{n,k} \xRightarrow{d} \mathcal{N}(0, 1).
$$
Moreover, assuming the "negligibility" condition $\max_k \sigma_{n,k}^2 / s_n^2 \to 0$, Lindeberg condition is *necessary* as well.

*Proof sketch (CF version).* Set $T_n := s_n^{-1} \sum_k X_{n,k}$, $\varphi_{n,k}(t) := E[e^{itX_{n,k}/s_n}]$. We want $\prod_k \varphi_{n,k}(t) \to e^{-t^2/2}$.

Use $\varphi_{n,k}(t) = 1 + it E[X_{n,k}/s_n] - (t^2/2) \sigma_{n,k}^2/s_n^2 + r_{n,k}(t)$ with $|r_{n,k}(t)| \le (t^2/2) E[(X_{n,k}/s_n)^2 \mathbf{1}_{|X_{n,k}/s_n| > \varepsilon}] + (t/6)|t|^2 \varepsilon E[(X_{n,k}/s_n)^2]$ (the usual Taylor bound with careful splitting at $\varepsilon$). The first mean term is zero. Sum over $k$:
$$
\sum_k (\varphi_{n,k} - 1) = -\frac{t^2}{2} - \frac{t^2}{2} \cdot o(1) \quad \text{by Lindeberg},
$$
and $\sum_k (\varphi_{n,k} - 1) \to -t^2/2$. A standard lemma converts sum of $(\varphi_{n,k} - 1)$ to log of product (using $|\varphi_{n,k} - 1|$ small via negligibility): $\log \prod_k \varphi_{n,k} \to -t^2/2$. Exponentiate and apply Lévy's continuity theorem. $\square$

**Special case: Lyapunov's condition**. If there exists $\delta > 0$ with
$$
\frac{1}{s_n^{2+\delta}} \sum_{k=1}^{r_n} E|X_{n,k}|^{2+\delta} \to 0,
$$
then Lindeberg's condition holds. *Proof.* $E[X_{n,k}^2 \mathbf{1}_{|X_{n,k}| > \varepsilon s_n}] \le (\varepsilon s_n)^{-\delta} E|X_{n,k}|^{2+\delta}$, so $L_n(\varepsilon) \le \varepsilon^{-\delta} \cdot s_n^{-(2+\delta)} \sum_k E|X_{n,k}|^{2+\delta} \to 0$. $\square$

Lyapunov's condition is the workhorse for applied CLTs — easy to check when $2 + \delta$ moments exist.

### 7.1 Example: non-iid CLT with bounded variances

Let $X_k$ independent, $X_k = \pm k$ with probability $1/(2k^2)$ each, $X_k = 0$ with probability $1 - 1/k^2$. Then $E[X_k] = 0$, $\text{Var}(X_k) = 1$, so $s_n^2 = n$. Check Lyapunov with $\delta = 2$: $E[X_k^4] = 2 k^4 \cdot 1/(2k^2) = k^2$. So $\sum E X_k^4 / s_n^4 = (\sum k^2) / n^2 \sim n/3 \to \infty$ — Lyapunov FAILS with $\delta = 2$.

Try $\delta = 1$: $E|X_k|^3 = k^3 / k^2 = k$. $\sum k / n^{3/2} = n(n+1)/(2 n^{3/2}) \sim n^{1/2}/2 \to \infty$. Also fails.

In fact, the CLT fails here: the largest terms dominate. This illustrates that the condition is essential, not just technical.

### 7.2 Example: CLT for a random number of summands

For $N_n$ a random integer with $N_n/n \xrightarrow{P} \nu > 0$ and $X_i$ iid with $E X_1 = 0$, $\text{Var}(X_1) = 1$: by CLT and Anscombe's theorem (a version for random indices),
$$
\frac{S_{N_n}}{\sqrt{N_n}} \xRightarrow{d} \mathcal{N}(0, 1).
$$
Useful in renewal theory and compound Poisson.

---

## 8. Berry-Esseen Theorem (Rate of Convergence)

The CLT gives convergence; Berry-Esseen gives the *rate*.

**Theorem 8.1** (Berry-Esseen, 1941/1942)**.** Let $X_1, \dots, X_n$ iid with $E[X_1] = 0$, $E[X_1^2] = \sigma^2 < \infty$, $E|X_1|^3 = \rho < \infty$. Let $F_n(x) = P(S_n/(\sigma\sqrt{n}) \le x)$ and $\Phi$ the standard normal CDF. Then
$$
\sup_x |F_n(x) - \Phi(x)| \le \frac{C \rho}{\sigma^3 \sqrt{n}},
$$
where $C$ is a universal constant (best known $C < 0.4748$).

*Proof sketch (Esseen's smoothing inequality).* The key lemma:
$$
\sup_x |F(x) - G(x)| \le \frac{2}{\pi} \int_{-T}^T \left|\frac{\hat{F}(t) - \hat{G}(t)}{t}\right| dt + \frac{24 \|G'\|_\infty}{\pi T},
$$
which bounds the Kolmogorov distance by a Fourier integral plus a smoothing error. Apply with $F = F_n$, $G = \Phi$. Bound the CF difference via
$$
|\varphi_{S_n/\sigma\sqrt{n}}(t) - e^{-t^2/2}| \le \frac{C \rho |t|^3}{\sigma^3 \sqrt{n}} e^{-t^2/4}
$$
for $|t| \le c \sigma\sqrt{n}/\rho$, using careful Taylor with third-moment remainder. Integrate, optimize $T$. $\square$

**Why Berry-Esseen matters in practice.**
- Gives a *uniform* error bound for sample size determination (confidence intervals).
- The $1/\sqrt{n}$ rate is sharp — cannot be improved under the stated moment conditions.
- Under additional moments or lattice structure, Edgeworth expansions give higher-order corrections.

---

## 9. Python Verification

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(17)

# ================================================================
# 1. SLLN: verify strong convergence for various distributions
# ================================================================
def slln_experiment(dist_name, sampler, true_mean, N=100000):
    X = sampler(N)
    cummean = np.cumsum(X) / np.arange(1, N+1)
    final_error = cummean[-1] - true_mean
    print(f"{dist_name:>15}: mean@{N} = {cummean[-1]:+.4f}, "
          f"target = {true_mean:+.4f}, error = {final_error:+.4e}")

print("SLLN convergence:")
slln_experiment("Normal(0,1)", lambda n: rng.standard_normal(n), 0.0)
slln_experiment("Exp(1)", lambda n: rng.exponential(1.0, n), 1.0)
slln_experiment("Uniform(0,1)", lambda n: rng.uniform(0, 1, n), 0.5)
slln_experiment("Poisson(3)", lambda n: rng.poisson(3, n), 3.0)

# Heavy-tailed counterexample: Cauchy has no mean, SLLN fails
print("\nCauchy (no mean) — SLLN fails:")
X = rng.standard_cauchy(100000)
cummean = np.cumsum(X) / np.arange(1, 100001)
print(f"  means at n=100, 1000, 10000, 100000: "
      f"{cummean[99]:+.3f}, {cummean[999]:+.3f}, "
      f"{cummean[9999]:+.3f}, {cummean[-1]:+.3f}")
# These should wander, not settle

# ================================================================
# 2. CLT: verify asymptotic normality via histograms and KS distance
# ================================================================
def clt_experiment(dist_name, sampler, mu, sigma, n=100, B=10000):
    """Simulate B batches of size n, compute standardized means."""
    batches = sampler(B, n)  # shape (B, n)
    means = batches.mean(axis=1)
    Z = np.sqrt(n) * (means - mu) / sigma
    ks = stats.kstest(Z, 'norm').statistic
    print(f"{dist_name:>15}: KS distance to N(0,1) at n={n}: {ks:.4f}")
    return Z

print("\nCLT KS distance (should → 0 as n grows):")
for n in [5, 30, 100, 500]:
    print(f"\n  Sample size n = {n}:")
    clt_experiment("Uniform(0,1)",
                   lambda B, n: rng.uniform(0, 1, (B, n)),
                   0.5, np.sqrt(1/12), n=n)
    clt_experiment("Exp(1)",
                   lambda B, n: rng.exponential(1.0, (B, n)),
                   1.0, 1.0, n=n)
    clt_experiment("Bernoulli(0.3)",
                   lambda B, n: rng.binomial(1, 0.3, (B, n)).astype(float),
                   0.3, np.sqrt(0.3 * 0.7), n=n)

# ================================================================
# 3. Berry-Esseen verification: rate of sup |F_n - Phi|
# ================================================================
def berry_esseen_rate(sampler, mu, sigma, rho, B=50000):
    ns = [10, 30, 100, 300, 1000]
    print(f"\nBerry-Esseen rate (rho = {rho:.4f}):")
    print(f"  {'n':>5} | {'emp KS':>10} | {'C*ρ/(σ³√n)':>12} | ratio")
    for n in ns:
        batches = sampler(B, n)
        Z = np.sqrt(n) * (batches.mean(axis=1) - mu) / sigma
        ks = stats.kstest(Z, 'norm').statistic
        theoretical = 0.5 * rho / (sigma**3 * np.sqrt(n))
        print(f"  {n:>5} | {ks:>10.4f} | {theoretical:>12.4f} | "
              f"{ks/theoretical:>.2f}x")

# Exponential: E|X - 1|^3 / sigma^3 = rho. Exp(1): E|X-1|^3 = (integral)
# For Exp(1): E[(X-1)^3] where X-1 has density e^{-x-1} for x > -1...
# Numerical estimate
X = rng.exponential(1.0, 10**6)
rho_exp = np.mean(np.abs(X - 1)**3)
berry_esseen_rate(
    lambda B, n: rng.exponential(1.0, (B, n)),
    mu=1.0, sigma=1.0, rho=rho_exp
)

# ================================================================
# 4. Lindeberg-Feller: CLT for non-iid triangular array
# ================================================================
def lindeberg_example(n):
    """X_{n,k} iid uniform(-1, 1) scaled by sqrt(k/n)."""
    # sigma_{n,k}^2 = k/(3n), s_n^2 = sum_k k/(3n) = (n+1)/6
    # Lindeberg condition trivially holds (bounded ratio).
    Xnk = np.zeros(n)
    for k in range(1, n+1):
        Xnk[k-1] = rng.uniform(-1, 1) * np.sqrt(k/n)
    s_n_sq = np.sum([k/(3*n) for k in range(1, n+1)])
    return np.sum(Xnk) / np.sqrt(s_n_sq)

print("\nLindeberg-Feller triangular array (B=10000 replicates):")
for n in [30, 100, 500]:
    Z = np.array([lindeberg_example(n) for _ in range(10000)])
    ks = stats.kstest(Z, 'norm').statistic
    print(f"  n={n}: KS = {ks:.4f}")

# ================================================================
# 5. Classic Monte Carlo integration with CI
# ================================================================
# Estimate E[f(X)] for f(x) = sin(x^2), X ~ N(0,1)
def mc_integral(N=100000):
    X = rng.standard_normal(N)
    Y = np.sin(X**2)
    est = Y.mean()
    se = Y.std(ddof=1) / np.sqrt(N)
    ci = (est - 1.96*se, est + 1.96*se)
    return est, ci, se

est, ci, se = mc_integral(100000)
print(f"\nMonte Carlo: E[sin(X^2)] ≈ {est:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"  (SE = {se:.6f} — rate ~1/√N as CLT predicts)")
```

**What you should observe.**
- SLLN converges for every distribution with finite mean; Cauchy wanders (no mean).
- CLT KS distance decays roughly like $1/\sqrt{n}$ (Berry-Esseen rate).
- For heavy-skewed distributions (like exponential), convergence is slower for a given $n$.
- Lindeberg-Feller CLT for triangular arrays gives normal limits for bounded / well-spread schemes.

---

## 10. [QUANT APPLICATION] LLN and CLT in Quantitative Finance

### 10.1 Monte Carlo pricing with confidence intervals

Simulate $N$ iid samples of the discounted payoff $h_i = e^{-rT} h(S_T^{(i)})$. The price estimate $\hat{V}_N = N^{-1} \sum h_i$ satisfies:
- **SLLN**: $\hat{V}_N \xrightarrow{\text{a.s.}} V$.
- **CLT**: $\sqrt{N}(\hat{V}_N - V) \xRightarrow{d} \mathcal{N}(0, \sigma^2)$ with $\sigma^2 = \text{Var}(h)$.
- **Confidence interval**: $V \in \hat{V}_N \pm z_{\alpha/2} \hat{\sigma}_N / \sqrt{N}$ asymptotically, by Slutsky.

For a deep OTM option with $P(h > 0)$ small, $\sigma$ is huge relative to $V$, motivating **importance sampling**: simulate under $Q'$ with density $dQ'/dQ = L$, estimate $E^{Q'}[h/L]$. This changes $\sigma^2$ to $E^Q[h^2 L] - V^2 \cdot \text{...}$ — minimized when $L \propto h$, the zero-variance importance distribution. (See Module 1.6, Radon-Nikodym.)

### 10.2 Parameter estimation consistency

For GARCH / stochastic volatility models estimated via MLE or GMM: under identifiability, stationarity, and finite second moments of score derivatives, the estimator $\hat{\theta}_n$ satisfies
$$
\hat{\theta}_n \xrightarrow{\text{a.s.}} \theta_0, \quad \sqrt{n}(\hat{\theta}_n - \theta_0) \xRightarrow{d} \mathcal{N}(0, V^{-1}\Sigma V^{-1}),
$$
where $V$ is the Fisher/sensitivity matrix and $\Sigma$ is the long-run variance. Both follow from ergodic LLN + CLT for martingale differences (Module 2.6).

### 10.3 Portfolio return distribution under diversification

A portfolio of $n$ iid assets with mean $\mu$, variance $\sigma^2$, and weights $w_i = 1/n$ has
$$
R = \frac{1}{n} \sum R_i, \quad \text{Var}(R) = \sigma^2 / n, \quad \sqrt{n}(R - \mu) \xRightarrow{d} \mathcal{N}(0, \sigma^2).
$$
So daily portfolio return distribution is approximately Gaussian for large $n$, even if individual returns are non-Gaussian. But with fat tails (e.g., $\alpha$-stable), the CLT fails and one must use **stable CLT**: $n^{1/\alpha}(\bar{X}_n - \mu) \xRightarrow{d} S_\alpha$, motivating risk models based on stable distributions (or non-Gaussian copulas).

### 10.4 Discretization error in Euler-Maruyama

For an SDE $dX_t = b(X_t) dt + \sigma(X_t) dW_t$ solved via Euler with step $\Delta t$, the discretization error is
$$
\sqrt{1/\Delta t}\, (X_T^{(\Delta t)} - X_T) \xRightarrow{d} \xi,
$$
where $\xi$ is Gaussian-ish (depends on derivatives of $b, \sigma$) — strong convergence of order $1/2$. Weak convergence is order $1$ (Talay-Tubaro). CLT-style results give exact coefficients for the leading error term, enabling **Richardson extrapolation** (run with $\Delta t$ and $\Delta t/2$, combine to cancel leading error).

### 10.5 Berry-Esseen and sample size determination

Suppose we want to estimate an option price $V$ within $0.1\%$ relative error with 95% confidence. CLT gives a nominal sample size, but Berry-Esseen reveals *how well* the CLT approximates for finite $n$. For deep OTM options with extreme skew (high $\rho / \sigma^3$), Berry-Esseen says we need $n \gtrsim (\rho / \sigma^3)^2 / \text{tol}^2$ — potentially millions of paths. This justifies variance reduction.

### 10.6 Functional CLT and hedging error asymptotics

For a discretely-hedged option with $N$ rebalancing times, the tracking error $\varepsilon_N$ satisfies
$$
\sqrt{N}\, \varepsilon_N \xRightarrow{d} \mathcal{N}(0, V_{\text{tracking}})
$$
with $V_{\text{tracking}}$ depending on the Gamma-weighted realized variance. This is a functional CLT + Ito-calculus argument (Bertsimas-Kogan-Lo 2000). Result: sampling error shrinks like $1/\sqrt{N}$, so doubling rebalancing frequency cuts tracking error by $\sqrt{2}$.

### 10.7 Kolmogorov-Smirnov test for Value-at-Risk backtesting

VaR backtest: observe $T$ out-of-sample violation indicators $V_t := \mathbf{1}_{L_t > \text{VaR}_t}$. Under correct model, $V_t$ are iid Bernoulli($\alpha$). The sample mean $\bar{V}_T$ should be $\approx \alpha$. By CLT (Bernoulli-CLT),
$$
\frac{\sqrt{T}(\bar{V}_T - \alpha)}{\sqrt{\alpha(1-\alpha)}} \xRightarrow{d} \mathcal{N}(0, 1).
$$
Reject the model if the test statistic is large — this is the Kupiec test of unconditional coverage.

### 10.8 Large deviations beyond CLT

CLT gives probabilities of order $O(1)$ around the mean. For rare events $P(\bar{X}_n - \mu > a)$ with $a > 0$ fixed, CLT gives $P \approx e^{-na^2/(2\sigma^2)}$ — tail estimate $O(1/\sqrt{n})$, but the truth is exponentially small. Cramér's theorem (Module 3+): $\log P(\bar{X}_n > a) / n \to -I(a)$ where $I$ is the *rate function* (Legendre-transform of log-MGF). Applications: portfolio ruin probabilities, implied vol extrapolation in the wings.

---

## 11. Worked Examples

### Example 11.1 (Simple WLLN from Markov)

Coin tosses $X_i \in \{0, 1\}$, $P(X_i = 1) = p$. $\bar{X}_n \xrightarrow{P} p$:
$$
P(|\bar{X}_n - p| > \varepsilon) \le \frac{p(1-p)}{n\varepsilon^2} \to 0.
$$
This is Bernoulli's original WLLN (1713).

### Example 11.2 (Cauchy distribution — LLN fails)

$X_i$ iid standard Cauchy. $E|X_1| = \infty$, so no LLN. In fact $\bar{X}_n \sim \text{Cauchy}$ for every $n$ (by stability: sum of Cauchys is Cauchy). $\bar{X}_n$ doesn't even converge in probability.

### Example 11.3 (Monte Carlo for $\pi$)

Simulate $U_i \sim \text{Unif}[0,1]^2$, let $X_i = 4 \cdot \mathbf{1}_{U_i \in D}$ with $D$ the unit disc. $E[X_i] = \pi$. SLLN: $\bar{X}_n \to \pi$ a.s. CLT: $\sqrt{n}(\bar{X}_n - \pi) \to \mathcal{N}(0, \pi(4-\pi))$. After $10^6$ samples, SE $\approx \sqrt{\pi(4-\pi)/10^6} \approx 0.0016$.

### Example 11.4 (Binomial $\to$ Normal)

Let $B_n \sim \text{Bin}(n, p)$. Write $B_n = X_1 + \dots + X_n$, $X_i$ iid Bernoulli($p$). CLT:
$$
\frac{B_n - np}{\sqrt{np(1-p)}} \xRightarrow{d} \mathcal{N}(0, 1).
$$
This is the de Moivre-Laplace theorem (1738) — historically the first instance of CLT.

### Example 11.5 (Lindeberg-Feller with varying variances)

$X_k$ independent, $X_k \sim \mathcal{N}(0, k^2)$. Then $s_n^2 = 1 + 4 + \dots + n^2 = n(n+1)(2n+1)/6 \sim n^3/3$. For CLT on $S_n / s_n$: Lindeberg condition is
$$
\frac{1}{s_n^2} \sum_{k=1}^n E[X_k^2 \mathbf{1}_{|X_k| > \varepsilon s_n}] = \frac{1}{s_n^2} \sum k^2 P(|N(0,1)| > \varepsilon s_n / k),
$$
where the last inequality uses $X_k = k N_k$. For $k \le n$, $s_n/k \ge s_n/n \sim n/\sqrt{3}$, so $\varepsilon s_n/k \to \infty$ and the tail $P(|N| > \varepsilon s_n/k)$ decays very fast — Lindeberg holds.

### Example 11.6 (The Random walk recentered)

$X_k$ iid $\pm 1$ symmetric. $S_n = X_1 + \dots + X_n$. CLT: $S_n/\sqrt{n} \xRightarrow{d} \mathcal{N}(0,1)$. More refined: $|S_n|/\sqrt{n} \xRightarrow{d} |N(0,1)|$ (half-normal). This is the basis for the arcsine law and subsequent Module 2.6 martingale limits.

---

## 12. Exercises

### Tier ★ (Foundational)

**2.4.E1.** Prove WLLN for pairwise uncorrelated $X_k \in L^2$ with bounded variances: if $E[X_k] = \mu$, $\text{Var}(X_k) \le V$, and $(X_k)$ are pairwise uncorrelated, then $\bar{X}_n \xrightarrow{P} \mu$.

**2.4.E2.** Show that the CLT holds for independent non-identically distributed $X_k$ with $E[X_k] = 0$ and uniformly bounded: $|X_k| \le C$ a.s. Use Lindeberg's condition.

**2.4.E3.** A coin has probability $p$ of heads, $p$ unknown. How many tosses guarantee $|\hat{p}_n - p| < 0.01$ with probability $\ge 0.95$? Use Chebyshev vs Hoeffding vs CLT and compare.

**2.4.E4.** Let $X_i$ iid $\text{Exp}(1)$. Find the limiting distribution of $(S_n - n)/\sqrt{n}$ and of $\log(S_n/n)$.

**2.4.E5.** Prove: if $X_n \xRightarrow{d} X$ and $E[X_n^2] \to E[X^2] < \infty$, then $X_n \xrightarrow{L^2} X$ provided they share a common space and convergence in distribution is actually convergence in probability (i.e., the limit is a constant, or you have an a.s. subsequence).

**2.4.E6.** Use the CLT to derive: $P(B_n \le np - \lambda \sqrt{np(1-p)}) \to \Phi(-\lambda)$ for $B_n \sim \text{Bin}(n, p)$.

**2.4.E7.** Prove Khinchin's WLLN using characteristic functions: show $\varphi_{\bar{X}_n}(t) = \varphi_{X_1}(t/n)^n \to e^{it\mu}$ and apply Lévy continuity.

**2.4.E8.** Simulate 100 paths of 10000 iid Exp(1) variables. Plot $\bar{X}_n$ as a function of $n$ for each path. Does it converge to $1$?

### Tier ★★ (Intermediate)

**2.4.E9** (Marcinkiewicz-Zygmund)**.** Let $X_i$ iid with $E|X_1|^p < \infty$ for $0 < p < 2$ (no mean assumption when $p < 1$), and for $p \ge 1$, $E[X_1] = 0$. Prove:
$$
\frac{S_n}{n^{1/p}} \xrightarrow{\text{a.s.}} 0.
$$
Hint: use Borel-Cantelli and fourth-moment-style arguments with truncation.

**2.4.E10** (SLLN via martingales — Doob)**.** Show that if $X_i$ iid with $E|X_1| < \infty$, then $M_n := \bar{X}_n$ is a reversed martingale with respect to the filtration $\mathcal{G}_n = \sigma(S_n, S_{n+1}, \dots)$. Apply the reversed martingale convergence theorem (Module 2.6).

**2.4.E11** (Delta method)**.** If $\sqrt{n}(\hat{\theta}_n - \theta) \xRightarrow{d} \mathcal{N}(0, \sigma^2)$ and $g$ is differentiable at $\theta$ with $g'(\theta) \ne 0$, then
$$
\sqrt{n}(g(\hat{\theta}_n) - g(\theta)) \xRightarrow{d} \mathcal{N}(0, (g'(\theta))^2 \sigma^2).
$$
Prove it using Taylor + Slutsky. Apply to $\log \bar{X}_n$ when $X_i > 0$ iid $L^2$.

**2.4.E12** (Multivariate CLT)**.** For iid $\vec{X}_i \in \mathbb{R}^d$ with mean $\vec{\mu}$ and covariance $\Sigma$, prove:
$$
\sqrt{n}(\bar{\vec{X}}_n - \vec{\mu}) \xRightarrow{d} \mathcal{N}_d(\vec{0}, \Sigma).
$$
Use the Cramér-Wold device: show all linear combinations $\vec{a}^\top \cdot \vec{X}$ have the right 1-D CLT.

**2.4.E13** (Lindeberg vs Lyapunov gap)**.** Construct a triangular array satisfying Lindeberg but not Lyapunov's condition. Hint: make the higher-moment divergent but only via a small-probability event.

**2.4.E14.** Let $X_i$ iid $L^1$ with $E[X_1] = \mu$. Prove WLLN in $L^1$: $\bar{X}_n \xrightarrow{L^1} \mu$, if $\bar{X}_n$ is uniformly integrable. Show this UI follows when $X_i \in L^p$ for some $p > 1$.

**2.4.E15** (CLT from Lindeberg swapping)**.** Fill in the details of the Lindeberg swapping proof (§6.2). Show the third-order remainder can be uniformly bounded using $E|X_1|^3 < \infty$ + Markov / truncation to handle only $|X| \le \varepsilon \sqrt{n}$.

**2.4.E16** (Berry-Esseen application)**.** A trader wants to price an option using $10^4$ Monte Carlo samples. The payoff has $\sigma = 0.5$, $\rho = 2.0$, $V \approx 1.2$. Using Berry-Esseen, estimate the worst-case bias of the empirical quantile $\hat{V} \approx V$ at a given confidence level.

### Tier ★★★ (Advanced / Quant)

**2.4.E17** (Donsker's theorem)**.** Let $\xi_i$ iid with mean $0$ and variance $1$. Define
$$
W^{(n)}(t) := \frac{1}{\sqrt{n}}\Big(\sum_{k=1}^{\lfloor nt \rfloor} \xi_k + (nt - \lfloor nt \rfloor) \xi_{\lfloor nt \rfloor + 1}\Big).
$$
State and prove convergence of finite-dimensional distributions $(W^{(n)}(t_1), \dots, W^{(n)}(t_m)) \xRightarrow{d} (W(t_1), \dots, W(t_m))$. Use multivariate CLT + Cramér-Wold.

**2.4.E18** (Arcsine law)**.** For a simple random walk $S_n = X_1 + \dots + X_n$ with $X_i \in \{\pm 1\}$ iid, let $L_n$ be the number of steps $k \le n$ with $S_k > 0$. Prove $L_n / n \xRightarrow{d} \text{Arcsine}(0, 1)$ with density $1/(\pi \sqrt{x(1-x)})$.

**2.4.E19** (CLT for martingales)**.** Let $(M_n)$ be a zero-mean martingale with $E[(M_n - M_{n-1})^2 | \mathcal{F}_{n-1}] = v_n$ and $\sum_k v_k = s_n^2$. Under a Lindeberg-type condition, $M_n / s_n \xRightarrow{d} \mathcal{N}(0, 1)$. State the condition and sketch the proof via CFs.

**2.4.E20** (Berry-Esseen for binomial)**.** For $B_n \sim \text{Bin}(n, p)$, show $\sup_x |P((B_n - np)/\sqrt{np(1-p)}) - \Phi(x)| \le C / \sqrt{n}$ with an explicit $C$ depending on $p$ (via Berry-Esseen with $\rho = p(1-p)(1-2p+2p^2)^{1/2}$, I think).

**2.4.E21** (Stable CLT)**.** Let $X_i$ iid with $P(X_1 > x) \sim C x^{-\alpha}$ as $x \to \infty$ for some $\alpha \in (0, 2)$, i.e. regularly varying tails with no second moment. Show that $(S_n - n a_n) / b_n \xRightarrow{d} S_\alpha$, an $\alpha$-stable distribution, with appropriate centering $a_n$ and scaling $b_n \sim n^{1/\alpha}$. (State and use the characterization of stable domains of attraction.)

**2.4.E22** (Berry-Esseen for tracking error)**.** In a discretely-hedged Black-Scholes portfolio with $n$ rebalancings, the tracking error $\varepsilon_n$ has expansion $\varepsilon_n = n^{-1/2} Z + n^{-1} R_n$ with $Z \sim \mathcal{N}(0, V)$ and $R_n$ bounded in $L^1$. Berry-Esseen-like: $\sup_x |P(\sqrt{n}\varepsilon_n \le x) - \Phi(x/\sqrt{V})| \le C/\sqrt{n}$. State the conditions (smoothness of the option Greeks, moment bounds on the noise) and sketch the argument using martingale CLT with remainder bounds.

---

## 13. Summary and Forward Pointers

**What we proved.**
- Weak LLN: $L^2$ via Chebyshev; $L^1$ via truncation (Khinchin).
- Strong LLN: $L^4$ via Borel-Cantelli; $L^2$ via Kolmogorov's three-series + Kronecker; iid $L^1$ via Kolmogorov; pairwise independent iid $L^1$ via Etemadi.
- Classical CLT: iid $L^2$ via characteristic functions; CF Taylor → Gaussian CF → Lévy continuity.
- Lindeberg-Feller: CLT for triangular arrays under the Lindeberg condition, with Lyapunov as a practical sufficient condition.
- Berry-Esseen: rate of convergence $O(1/\sqrt{n})$ in Kolmogorov distance.

**The big picture.** LLN and CLT together say: the sample mean $\bar{X}_n$ (a) converges to the population mean, and (b) has Gaussian fluctuations of order $1/\sqrt{n}$. Every statistical inference, every Monte Carlo algorithm, and every CLT-based asymptotic in finance ultimately relies on these two theorems.

**Forward pointers.**
- **Module 2.5** (Characteristic functions): CFs are the analytic tool we used. Lévy inversion, Bochner, stable distributions, Lévy-Khintchine — the full theory of distributions via CFs.
- **Module 2.6** (Martingales): LLN/CLT for dependent sequences via martingale methods. Doob's convergence theorem generalizes SLLN; martingale CLT (McLeish / Hall-Heyde) generalizes Lindeberg-Feller.
- **Module 2.7** (Markov chains): ergodic theorem = SLLN for stationary ergodic sequences. CLT for Markov chains via Poisson equation.
- **Subject 3** (Stochastic processes): Brownian motion as the functional CLT limit; Donsker's theorem; SDE theory.
- **Subject 5** (Statistical learning): SLLN underlies empirical risk minimization consistency; CLT underlies Wald tests and confidence intervals for ML models.

**Next module:** Characteristic functions — the deep analytic machinery behind CLT, stable laws, and infinite divisibility.

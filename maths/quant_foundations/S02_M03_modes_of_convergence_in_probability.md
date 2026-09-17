# Module 2.3: Modes of Convergence in Probability

**Subject 2: Probability Theory** · Module 3 of 7

---

## 0. Prerequisites and Position in the Curriculum

Before reading this module you should be comfortable with:

- **Module 1.3** (Lebesgue integration): MCT, Fatou, DCT, absolute continuity of the integral.
- **Module 1.5** ($L^p$ spaces): $L^p$ as Banach spaces, completeness, Hölder / Minkowski.
- **Module 2.1** (probability spaces): random variables, distributions, Borel-Cantelli I/II, 0–1 laws.
- **Module 2.2** (expectation, conditional expectation): moment inequalities, $L^p$ inclusion, Jensen.

This module introduces the **four principal modes of convergence** for random variables — almost sure, in probability, in $L^p$, and in distribution — and proves the fundamental implication relationships, counterexamples, and summability-to-convergence bridges (Borel-Cantelli, Kolmogorov's three-series). It sets up every downstream module: LLN/CLT (Module 2.4), characteristic functions (Module 2.5), and martingale convergence (Module 2.6).

---

## 1. The Four Modes

Let $(\Omega, \mathcal{F}, P)$ be a probability space and let $X, X_1, X_2, \dots$ be real-valued random variables on it. We write $X_n \to X$ in any of four canonical senses.

### 1.1 Almost sure convergence

**Definition 1.1.** $X_n \to X$ **almost surely** (a.s., or *with probability one*) if
$$
P\Big(\big\{\omega : X_n(\omega) \to X(\omega)\big\}\Big) = 1.
$$
Equivalently, $P(X_n \not\to X) = 0$. We write $X_n \xrightarrow{\text{a.s.}} X$.

This is the strongest pointwise notion on the sample space: excluding a set of probability zero, the sample path converges as an ordinary sequence of real numbers.

**Reformulation.** Writing $\{X_n \to X\} = \bigcap_{k \ge 1} \bigcup_{N \ge 1} \bigcap_{n \ge N} \{|X_n - X| < 1/k\}$, one shows

$$
X_n \xrightarrow{\text{a.s.}} X \iff \forall\, \varepsilon > 0,\ P\Big(\limsup_n \{|X_n - X| > \varepsilon\}\Big) = 0.
$$

This is the form we will use for Borel-Cantelli arguments.

### 1.2 Convergence in probability

**Definition 1.2.** $X_n \to X$ **in probability** if for every $\varepsilon > 0$,
$$
P(|X_n - X| > \varepsilon) \to 0 \quad \text{as } n \to \infty.
$$
We write $X_n \xrightarrow{P} X$.

Informally: the "bad set" $\{|X_n - X| > \varepsilon\}$ has vanishing measure for every tolerance $\varepsilon$.

### 1.3 Convergence in $L^p$

**Definition 1.3.** For $p \in [1, \infty)$, $X_n \to X$ in $L^p$ if $X_n, X \in L^p(\Omega, \mathcal{F}, P)$ and
$$
\|X_n - X\|_p^p = E[|X_n - X|^p] \to 0.
$$
We write $X_n \xrightarrow{L^p} X$. For $p = \infty$ we require $\|X_n - X\|_\infty \to 0$ (essentially uniform).

$L^2$ convergence is *mean-square* convergence; $L^1$ is convergence in mean.

### 1.4 Convergence in distribution

**Definition 1.4.** $X_n \to X$ **in distribution** (or *weakly*) if
$$
F_{X_n}(x) \to F_X(x) \quad \text{at every continuity point } x \text{ of } F_X,
$$
where $F_{X_n}, F_X$ are the CDFs. We write $X_n \xRightarrow{d} X$ or $X_n \Rightarrow X$.

**Remark 1.5.** Convergence in distribution depends only on the *laws* $\mu_{X_n} = X_n \# P$ and $\mu_X = X \# P$, not on the common probability space; in fact one can take $X_n, X$ on different spaces. This is fundamentally different from the first three modes, each of which requires $X_n$ and $X$ to live on a common $(\Omega, \mathcal{F}, P)$.

---

## 2. The Implication Hierarchy

We prove the following chain of implications:

$$
\boxed{\ \underbrace{X_n \xrightarrow{L^p}}_{\text{strongest (for } p \ge 1)} \Longrightarrow\ X_n \xrightarrow{L^q} \ (q \le p)\ \Longrightarrow\ X_n \xrightarrow{P}\ \Longrightarrow\ X_n \xRightarrow{d}\ }
$$
and
$$
X_n \xrightarrow{\text{a.s.}} X\ \Longrightarrow\ X_n \xrightarrow{P} X,
$$
but **no other implications hold** in general.

### 2.1 $L^p \Rightarrow L^q$ for $q \le p$

On a probability space $P(\Omega) = 1$. By Jensen's inequality applied to the convex function $t \mapsto t^{p/q}$,
$$
E[|X|^q]^{p/q} \le E[|X|^p] \quad \Longleftrightarrow \quad \|X\|_q \le \|X\|_p.
$$
Apply to $X_n - X$: $\|X_n - X\|_q \le \|X_n - X\|_p \to 0$.

This is the $L^p$ inclusion from Module 1.5 specialized to finite measure.

### 2.2 $L^p \Rightarrow$ in probability

**Theorem 2.1** (Markov's inequality gives $L^p \Rightarrow P$)**.** If $X_n \xrightarrow{L^p} X$ with $p \ge 1$, then $X_n \xrightarrow{P} X$.

*Proof.* For any $\varepsilon > 0$, Markov's inequality gives
$$
P(|X_n - X| > \varepsilon) = P(|X_n - X|^p > \varepsilon^p) \le \frac{E[|X_n - X|^p]}{\varepsilon^p} \to 0. \quad\square
$$

### 2.3 a.s. $\Rightarrow$ in probability

**Theorem 2.2.** If $X_n \xrightarrow{\text{a.s.}} X$, then $X_n \xrightarrow{P} X$.

*Proof.* Fix $\varepsilon > 0$. Define $A_n := \{|X_n - X| > \varepsilon\}$ and $B_N := \bigcup_{n \ge N} A_n$. The sets $B_N$ decrease to $B_\infty := \limsup_n A_n$. On the a.s. convergence set, only finitely many of the $A_n$ hold, so $B_\infty$ has probability $0$. Hence by continuity of probability,
$$
P(A_n) \le P(B_n) \to P(B_\infty) = 0. \quad \square
$$

### 2.4 Convergence in probability $\Rightarrow$ convergence in distribution

**Theorem 2.3.** If $X_n \xrightarrow{P} X$, then $X_n \xRightarrow{d} X$.

*Proof.* Fix a continuity point $x$ of $F_X$ and $\varepsilon > 0$. We bound $F_{X_n}(x) - F_X(x)$ in both directions. Using that $\{X_n \le x\} \subseteq \{X \le x + \varepsilon\} \cup \{|X_n - X| > \varepsilon\}$,
$$
F_{X_n}(x) \le F_X(x + \varepsilon) + P(|X_n - X| > \varepsilon).
$$
Similarly $\{X \le x - \varepsilon\} \subseteq \{X_n \le x\} \cup \{|X_n - X| > \varepsilon\}$ gives
$$
F_X(x - \varepsilon) - P(|X_n - X| > \varepsilon) \le F_{X_n}(x).
$$
Taking $\limsup_n$ and $\liminf_n$ and using $P(|X_n - X| > \varepsilon) \to 0$:
$$
F_X(x - \varepsilon) \le \liminf_n F_{X_n}(x) \le \limsup_n F_{X_n}(x) \le F_X(x + \varepsilon).
$$
Let $\varepsilon \downarrow 0$; since $x$ is a continuity point, both outer bounds converge to $F_X(x)$. $\square$

### 2.5 Partial converse: convergence to a constant

**Theorem 2.4.** If $X_n \xRightarrow{d} c$ where $c$ is a constant, then $X_n \xrightarrow{P} c$.

*Proof.* $F_c$ has a jump at $c$ and is continuous elsewhere. For $\varepsilon > 0$ both $c - \varepsilon$ and $c + \varepsilon$ are continuity points, so $F_{X_n}(c - \varepsilon) \to 0$ and $F_{X_n}(c + \varepsilon) \to 1$. Hence
$$
P(|X_n - c| \le \varepsilon) \ge F_{X_n}(c + \varepsilon) - F_{X_n}(c - \varepsilon) \to 1. \quad \square
$$

---

## 3. Counterexamples (Why No Other Implications Hold)

Take $\Omega = [0,1]$ with Lebesgue measure.

### 3.1 In probability but not a.s.: the **typewriter sequence**

Partition $[0,1]$ into dyadic intervals. For level $k = 1, 2, \dots$, let the $k$-th level consist of $2^k$ disjoint subintervals $I_{k,1}, \dots, I_{k, 2^k}$ each of length $2^{-k}$. Enumerate all these intervals in order $(k, j)$ linearly to get a sequence $J_1, J_2, \dots$. Let $X_n = \mathbf{1}_{J_n}$.

**In probability?** Yes: for $\varepsilon < 1$, $P(|X_n| > \varepsilon) = \lambda(J_n) \to 0$ since $\lambda(J_n) \le 2^{-k(n)}$ where $k(n) \to \infty$.

**Almost surely?** No. For every $\omega \in [0,1]$ and every level $k$, $\omega$ belongs to exactly one $I_{k,j}$, so $X_n(\omega) = 1$ infinitely often. Hence $X_n(\omega) \not\to 0$ for any $\omega$.

This is the canonical "convergence in probability does not imply a.s." example. The "typewriter" name comes from imagining $X_n$'s support sweeping across $[0,1]$ like a typewriter carriage, then restarting with a finer spacing.

### 3.2 a.s. but not in $L^p$: the **tall-narrow spike**

Let $X_n = n^{1/p} \mathbf{1}_{[0, 1/n]}$. Then $X_n \to 0$ everywhere except $\omega = 0$ (a single point), so $X_n \xrightarrow{\text{a.s.}} 0$. But
$$
E[|X_n|^p] = n \cdot \frac{1}{n} = 1 \not\to 0.
$$
The mass concentrates onto a shrinking set but its amplitude grows to preserve the $p$-th moment.

### 3.3 In $L^p$ but not a.s.

The typewriter sequence in §3.1 also does not converge a.s., but $E[X_n^p] = \lambda(J_n) \to 0$, so $X_n \xrightarrow{L^p} 0$. Hence $L^p \not\Rightarrow$ a.s.

### 3.4 In probability but not in $L^p$

Take the tall-narrow spike $X_n = n^{1/p} \mathbf{1}_{[0, 1/n]}$. Then $X_n \xrightarrow{P} 0$ (even a.s.) but $\|X_n\|_p = 1$.

### 3.5 In distribution but not in probability

Let $X, X_1, X_2, \dots$ be iid $\mathcal{N}(0,1)$. Each $X_n$ has the same distribution as $X$, so trivially $X_n \xRightarrow{d} X$. But $X_n - X \sim \mathcal{N}(0, 2)$, so
$$
P(|X_n - X| > \varepsilon) = 2 \Phi(-\varepsilon/\sqrt{2}) > 0 \text{ constant}.
$$
No convergence in probability. In distribution captures only the law; it is blind to joint behavior.

---

## 4. Partial Converses: When the Arrows Reverse

### 4.1 From convergence in probability to a.s.: pass to a subsequence

**Theorem 4.1.** If $X_n \xrightarrow{P} X$, there exists a subsequence $(X_{n_k})$ with $X_{n_k} \xrightarrow{\text{a.s.}} X$.

*Proof.* Choose $n_1 < n_2 < \cdots$ with $P(|X_{n_k} - X| > 2^{-k}) < 2^{-k}$. Then
$$
\sum_k P(|X_{n_k} - X| > 2^{-k}) \le \sum_k 2^{-k} < \infty,
$$
so Borel-Cantelli I gives $P(|X_{n_k} - X| > 2^{-k} \text{ i.o.}) = 0$, i.e. eventually $|X_{n_k} - X| \le 2^{-k}$, which forces $X_{n_k} \to X$. $\square$

**Corollary 4.2.** $L^p$ convergence implies a.s. convergence along some subsequence. (Same argument applied to $P(|X_n - X| > \varepsilon) \le \|X_n - X\|_p^p / \varepsilon^p$.)

This is how one often upgrades in-probability results to a.s.: work along a Borel-Cantelli-compatible subsequence.

### 4.2 From a.s. to $L^p$: dominated convergence and uniform integrability

A.s. convergence does not in general imply $L^p$ convergence (see §3.2). We need a moment bound.

**Theorem 4.3** (Bounded convergence + more)**.** If $X_n \xrightarrow{\text{a.s.}} X$ *and* $\{X_n\}$ is dominated by $Y \in L^p$ (i.e., $|X_n| \le Y$ a.s. with $E[Y^p] < \infty$), then $X_n \xrightarrow{L^p} X$.

*Proof.* $|X_n - X|^p \le (2Y)^p \in L^1$. Apply DCT to the sequence $|X_n - X|^p \to 0$ a.s. $\square$

A much more powerful tool replaces domination with **uniform integrability**.

### 4.3 Uniform integrability

**Definition 4.4.** A family $\{X_\alpha\}_{\alpha \in A}$ is **uniformly integrable** (UI) if
$$
\lim_{M \to \infty} \sup_{\alpha} E[|X_\alpha| \mathbf{1}_{|X_\alpha| > M}] = 0.
$$

UI families are bounded in $L^1$ (take $M = 1$ for the tail, $E|X_\alpha| \le M + E[|X_\alpha| \mathbf{1}_{|X_\alpha| > M}]$).

**Theorem 4.5** (de la Vallée-Poussin)**.** $\{X_\alpha\}$ is UI iff there exists $\varphi: [0, \infty) \to [0, \infty)$ with $\varphi(t)/t \to \infty$ such that $\sup_\alpha E[\varphi(|X_\alpha|)] < \infty$.

*Proof sketch.* *Sufficiency:* If $E[\varphi(|X_\alpha|)] \le C$ and $\varphi(t)/t \ge K$ for $t > M(K)$, then $E[|X_\alpha| \mathbf{1}_{|X_\alpha| > M(K)}] \le C/K$, letting $K \to \infty$ gives UI. *Necessity:* for UI $\{X_\alpha\}$, choose a sequence $M_k$ so that $\sup_\alpha E[|X_\alpha| \mathbf{1}_{|X_\alpha| > M_k}] \le 2^{-k}$ and set $\varphi(t) = \sum_k (t - M_k)^+$, which is convex with superlinear growth. $\square$

**Concrete corollary.** Boundedness in $L^p$ for any $p > 1$ implies UI: take $\varphi(t) = t^p$. In particular, $L^2$-bounded $\Rightarrow$ UI in $L^1$.

### 4.4 Vitali convergence theorem

**Theorem 4.6** (Vitali). $X_n \xrightarrow{L^1} X$ iff
1. $X_n \xrightarrow{P} X$, and
2. $\{X_n\}$ is uniformly integrable.

*Proof.* $(\Rightarrow)$ $L^1$ convergence implies convergence in probability (Theorem 2.1). For UI: $L^1$-convergent sequences are Cauchy, so $\|X_n\|_1 \to \|X\|_1$ and the family is UI via Vitali-like arguments (details: write $E[|X_n| \mathbf{1}_{|X_n| > M}] \le E[|X_n - X| \mathbf{1}_{|X_n| > M}] + E[|X| \mathbf{1}_{|X_n| > M}]$; the first term $\to 0$ in $n$, the second uses absolute continuity of $E[|X|\,\cdot\,]$ and $P(|X_n| > M) \le E|X_n|/M$ bounded).

$(\Leftarrow)$ Extract a.s. subsequence via Theorem 4.1: $X_{n_k} \to X$ a.s. By Fatou on the UI family, $E|X| \le \liminf E|X_{n_k}| < \infty$, so $X \in L^1$. For the full sequence: fix $\varepsilon > 0$, choose $M$ with $\sup_n E[|X_n|\mathbf{1}_{|X_n| > M}] < \varepsilon$ and $E[|X|\mathbf{1}_{|X| > M}] < \varepsilon$. Split
$$
|X_n - X| \le |X_n - X| \mathbf{1}_{|X_n| \le M, |X| \le M} + \text{tails}.
$$
The main term is bounded by $2M$ and tends to $0$ in probability, hence in $L^1$ by bounded convergence on the set it's supported. Tails contribute $\le 4\varepsilon$. $\square$

**Corollary 4.7** ($L^p$ version)**.** $X_n \xrightarrow{L^p} X$ iff $X_n \xrightarrow{P} X$ and $\{|X_n|^p\}$ is UI.

This is the cleanest characterization: $L^p$ convergence = convergence in probability + no mass escaping to infinity in the $p$-th moment.

### 4.5 Scheffé's lemma

**Theorem 4.8** (Scheffé)**.** If $X_n, X \ge 0$, $X_n \xrightarrow{\text{a.s.}} X$, and $E[X_n] \to E[X] < \infty$, then $X_n \xrightarrow{L^1} X$.

*Proof.* $(X - X_n)^+ \le X$ a.s. By DCT on $(X - X_n)^+ \to 0$ dominated by $X \in L^1$, $E[(X - X_n)^+] \to 0$. But $E[|X - X_n|] = E[(X - X_n)^+] + E[(X - X_n)^-]$ and $E[(X - X_n)^-] - E[(X - X_n)^+] = E[X] - E[X_n] \to 0$, so $E[(X - X_n)^-] \to 0$ and thus $E[|X - X_n|] \to 0$. $\square$

Scheffé is the tool of choice for density convergence: if $f_n, f$ are probability densities with $f_n \to f$ a.e. (pointwise limit automatically integrates to 1 by Fatou/monotone arguments), then $\|f_n - f\|_1 \to 0$ — total variation convergence.

---

## 5. Cauchy Criteria and Completeness

The first three modes are complete.

### 5.1 Cauchy in probability

**Proposition 5.1.** $X_n$ is Cauchy in probability iff there exists $X$ such that $X_n \xrightarrow{P} X$.

*Proof.* $(\Leftarrow)$ $P(|X_n - X_m| > \varepsilon) \le P(|X_n - X| > \varepsilon/2) + P(|X_m - X| > \varepsilon/2) \to 0$.

$(\Rightarrow)$ Extract $n_k$ with $P(|X_{n_k} - X_{n_{k+1}}| > 2^{-k}) < 2^{-k}$. By Borel-Cantelli I, $\sum_k |X_{n_k} - X_{n_{k+1}}| < \infty$ a.s., so $(X_{n_k})$ is a.s. Cauchy in $\mathbb{R}$ and converges to some $X$ a.s. Convergence in probability of the full sequence follows from the Cauchy property and the subsequence limit. $\square$

### 5.2 $L^p$ completeness

$L^p(\Omega, \mathcal{F}, P)$ is complete (Riesz-Fischer, Module 1.5), so $L^p$-Cauchy sequences converge in $L^p$.

### 5.3 Almost sure convergence is *not* metrizable

There is no metric on the set of random variables making a.s. convergence topological. Proof idea: metric topologies have the property that $X_n \to X$ iff every subsequence has a further subsequence converging to $X$. But by Theorem 4.1, convergence in probability has this subsequence property, yet a.s. convergence is strictly stronger. If a.s. convergence were metrizable, both would give the same topology, contradiction.

In contrast, convergence in probability is metrizable via the Ky Fan metric
$$
d(X, Y) = \inf\{\varepsilon > 0 : P(|X - Y| > \varepsilon) \le \varepsilon\},
$$
or equivalently $d'(X, Y) = E[|X - Y| \wedge 1]$.

---

## 6. Borel-Cantelli in Action

Borel-Cantelli I gives a very operational sufficient condition for a.s. convergence.

### 6.1 Summability implies a.s. convergence

**Proposition 6.1.** If for every $\varepsilon > 0$,
$$
\sum_{n=1}^\infty P(|X_n - X| > \varepsilon) < \infty,
$$
then $X_n \xrightarrow{\text{a.s.}} X$.

*Proof.* Borel-Cantelli I: $P(|X_n - X| > \varepsilon \text{ i.o.}) = 0$ for each $\varepsilon$. Take $\varepsilon = 1/k$ and use a countable union to conclude $X_n \to X$ a.s. $\square$

**Corollary 6.2.** If $\sum_n E[|X_n - X|^p] < \infty$ for some $p \ge 1$, then $X_n \xrightarrow{\text{a.s.}} X$.

*Proof.* Markov: $P(|X_n - X| > \varepsilon) \le E|X_n - X|^p / \varepsilon^p$, then apply 6.1. $\square$

### 6.2 Example: Rademacher series

Let $\xi_n$ be iid $\pm 1$ with $P(\xi_n = \pm 1) = 1/2$, and let $a_n \in \mathbb{R}$ with $\sum_n a_n^2 < \infty$. Consider $S_n = \sum_{k=1}^n a_k \xi_k$. By Kolmogorov's maximal inequality (Module 2.4) or by martingale $L^2$ convergence (Module 2.6) one shows $S_n$ converges a.s. For the other direction (necessity of $\sum a_n^2 < \infty$) see Kolmogorov's three-series theorem below.

### 6.3 The strong law via summability

We foreshadow the SLLN: for iid $X_i$ with $E|X_1|^4 < \infty$ (strong moment assumption), Borel-Cantelli gives a quick proof. Let $\bar{X}_n = n^{-1} \sum_{i=1}^n X_i$ with $E[X_i] = 0$. Then (expanding and using independence to kill cross terms of odd multiplicity)
$$
E[\bar{X}_n^4] = \frac{1}{n^4}\left[ n \cdot E[X_1^4] + 3 n(n-1) \cdot (E[X_1^2])^2 \right] = O(n^{-2}).
$$
Summable! Corollary 6.2 with $p = 4$ gives $\bar{X}_n \xrightarrow{\text{a.s.}} 0$. Module 2.4 removes the fourth moment assumption (Etemadi / Kolmogorov).

---

## 7. Kolmogorov's Three-Series Theorem

For independent but possibly non-identically distributed $X_n$, when does $\sum_n X_n$ converge almost surely? The answer is a beautiful three-part criterion.

### 7.1 Statement

**Theorem 7.1** (Kolmogorov's three-series theorem)**.** Let $X_1, X_2, \dots$ be independent. Fix $c > 0$ and let $X_n^c := X_n \mathbf{1}_{|X_n| \le c}$ (the truncation at $c$). Then $\sum_n X_n$ converges a.s. iff all three of the following hold:

1. $\sum_n P(|X_n| > c) < \infty$,
2. $\sum_n E[X_n^c]$ converges,
3. $\sum_n \text{Var}(X_n^c) < \infty$.

Moreover, if the three series converge for some $c > 0$ they converge for every $c > 0$.

### 7.2 Proof of sufficiency

Assume (1), (2), (3). We show $\sum_n X_n$ converges a.s.

**Step 1: Truncation reduces to truncated variables.** By (1) and Borel-Cantelli I, $P(|X_n| > c \text{ i.o.}) = 0$, so a.s. $X_n = X_n^c$ for all $n$ large enough. Hence $\sum X_n$ and $\sum X_n^c$ differ by a finite random variable, and convergence of one is equivalent to convergence of the other.

**Step 2: Centering.** By (2), $\sum_n E[X_n^c]$ converges (to some limit). So $\sum_n X_n^c$ converges iff $\sum_n (X_n^c - E[X_n^c])$ converges.

**Step 3: Kolmogorov's inequality.** For independent, zero-mean $Y_1, \dots, Y_n$ in $L^2$,
$$
P\Big(\max_{k \le n} |S_k| > \lambda\Big) \le \frac{\text{Var}(S_n)}{\lambda^2}, \quad S_k := \sum_{j=1}^k Y_j.
$$
*Proof of Kolmogorov's inequality.* Let $\tau = \inf\{k : |S_k| > \lambda\}$ and $A_k = \{\tau = k\}$; the $A_k$ are disjoint and $A_k \in \sigma(Y_1, \dots, Y_k)$. On $A_k$, $|S_k| > \lambda$. Compute
$$
E[S_n^2 \mathbf{1}_{\tau \le n}] = \sum_{k=1}^n E[S_n^2 \mathbf{1}_{A_k}].
$$
Write $S_n = S_k + (S_n - S_k)$ and expand; the cross term vanishes by independence (since $\mathbf{1}_{A_k} S_k \in \sigma(Y_1, \dots, Y_k)$, $S_n - S_k$ independent of this, $E[S_n - S_k] = 0$). Hence
$$
E[S_n^2 \mathbf{1}_{A_k}] \ge E[S_k^2 \mathbf{1}_{A_k}] \ge \lambda^2 P(A_k).
$$
Sum: $E[S_n^2] \ge \lambda^2 P(\tau \le n)$. $\square$

**Step 4: $L^2$ convergence of the centered partial sums.** Let $Y_n = X_n^c - E[X_n^c]$; these are independent, zero-mean, with $\sum \text{Var}(Y_n) = \sum \text{Var}(X_n^c) < \infty$. The partial sums $S_n = \sum_{j \le n} Y_j$ form a Cauchy sequence in $L^2$: $\|S_n - S_m\|_2^2 = \sum_{j = m+1}^n \text{Var}(Y_j) \to 0$. By completeness, $S_n \to S$ in $L^2$, hence in probability. For a.s. convergence use Kolmogorov's inequality: for $m < n$,
$$
P\Big(\max_{m < k \le n} |S_k - S_m| > \varepsilon\Big) \le \frac{1}{\varepsilon^2}\sum_{j=m+1}^n \text{Var}(Y_j).
$$
Letting $n \to \infty$ (by continuity of probability) and then $m \to \infty$ shows the partial sums are a.s. Cauchy, hence convergent a.s. $\square$

### 7.3 Proof of necessity

Assume $\sum_n X_n$ converges a.s. We show (1), (2), (3). (This is the harder direction and requires Kolmogorov's converse inequality.)

**Step 1.** Convergence a.s. implies $X_n \to 0$ a.s., hence $P(|X_n| > c \text{ i.o.}) = 0$. By Borel-Cantelli II (the $X_n$ are independent!), $\sum_n P(|X_n| > c) < \infty$. This gives (1).

**Step 2.** By (1), $X_n = X_n^c$ for all large $n$, so $\sum X_n^c$ also converges a.s. It suffices to show: if $(Y_n)$ are independent, uniformly bounded $|Y_n| \le c$, and $\sum Y_n$ converges a.s., then $\sum E[Y_n]$ converges and $\sum \text{Var}(Y_n) < \infty$.

**Step 3** (Kolmogorov's converse inequality)**.** For independent, zero-mean, bounded $|Y_k| \le c$ random variables and any $\lambda > 0$,
$$
P\Big(\max_{k \le n} |S_k| \le \lambda\Big) \le \frac{(c + \lambda)^2}{\text{Var}(S_n)}.
$$
*Proof.* Let $\tau = \inf\{k : |S_k| > \lambda\}$, $A = \{\tau > n\} = \{\max |S_k| \le \lambda\}$, $A_k = \{\tau = k\}$. On $A$, $|S_n| \le \lambda$ so $E[S_n^2 \mathbf{1}_A] \le \lambda^2 P(A)$. On $A_k$, $|S_{k-1}| \le \lambda$ and $|Y_k| \le c$, so $|S_k| \le \lambda + c$. Then
$$
E[S_n^2] = E[S_n^2 \mathbf{1}_A] + \sum_{k \le n} E[S_n^2 \mathbf{1}_{A_k}].
$$
For the second, writing $S_n = S_k + (S_n - S_k)$, cross term vanishes, $E[(S_n-S_k)^2\mathbf{1}_{A_k}] = P(A_k)\text{Var}(S_n - S_k) \le P(A_k) \text{Var}(S_n)$, and $|S_k| \le \lambda + c$, so
$$
E[S_n^2 \mathbf{1}_{A_k}] \le (\lambda + c)^2 P(A_k) + P(A_k) \text{Var}(S_n).
$$
Summing and using $\sum P(A_k) = P(A^c) = 1 - P(A)$ and $E[S_n^2] = \text{Var}(S_n)$:
$$
\text{Var}(S_n) \le \lambda^2 P(A) + (\lambda+c)^2 (1 - P(A)) + (1 - P(A)) \text{Var}(S_n).
$$
Rearrange: $P(A) \text{Var}(S_n) \le \lambda^2 P(A) + (\lambda+c)^2(1 - P(A)) \le (\lambda+c)^2$. $\square$

**Step 4.** If $\sum Y_n$ converges a.s. with $|Y_n| \le c$ (so $E[Y_n]$ exists and $|E[Y_n]| \le c$), we claim $\sum \text{Var}(Y_n) < \infty$.

Suppose not. Let $T_n = S_n - E[S_n] = \sum_{k \le n} (Y_k - E[Y_k])$; then $T_n$ is a sum of independent bounded zero-mean variables with $|Y_k - E[Y_k]| \le 2c$, and $\text{Var}(T_n) = \sum_{k \le n} \text{Var}(Y_k) \to \infty$. By the converse inequality applied to the $Y_k - E[Y_k]$ (bound $2c$):
$$
P\Big(\max_{k \le n} |T_k| \le \lambda\Big) \le \frac{(\lambda + 2c)^2}{\text{Var}(T_n)} \to 0.
$$
Hence for every $\lambda$, eventually $\max_{k \le n} |T_k| > \lambda$ with probability $> 1/2$, which contradicts the assertion that $(T_n)$ converges a.s. (which would force $\sup |T_n - T| < \infty$ a.s.). Wait, we need $\sum Y_n$ to converge a.s., which implies $S_n$ converges a.s., but $T_n = S_n - E[S_n]$ and $E[S_n] = \sum_{k \le n} E[Y_k]$ is just a deterministic sequence. If $E[S_n]$ converges, $T_n$ converges a.s., and then $\sup |T_n| < \infty$ a.s., contradiction. If $E[S_n]$ does not converge, $S_n = T_n + E[S_n]$ can only converge a.s. if $T_n - (\text{const})$ converges, which again forces $T_n$ convergent along the same subsequence, leading to the same contradiction by taking $n \to \infty$ along $E[S_n]$-stabilizing subsequences. Conclusion: $\sum \text{Var}(Y_n) < \infty$ (this gives (3)).

**Step 5.** With $\sum \text{Var}(Y_n) < \infty$, by sufficiency applied to the zero-mean variables $Y_n - E[Y_n]$, $T_n = \sum_{k \le n} (Y_k - E[Y_k])$ converges a.s. Since $S_n$ also converges a.s., $E[S_n] = S_n - T_n$ converges, giving (2). $\square$

### 7.4 Example: $\sum_n \xi_n / n^\alpha$

Let $\xi_n$ be iid symmetric (e.g. standard normal or $\pm 1$). $X_n = \xi_n / n^\alpha$. With $c = 1$: $P(|X_n| > 1) = P(|\xi_1| > n^\alpha)$ is summable iff $E|\xi_1|^{1/\alpha} < \infty$ (standard). For Gaussians this is summable for every $\alpha > 0$. Truncated mean is $0$ by symmetry. $\text{Var}(X_n^c) \le E[X_n^2] = \text{Var}(\xi_1) / n^{2\alpha}$, summable iff $\alpha > 1/2$. So $\sum \xi_n / n^\alpha$ converges a.s. iff $\alpha > 1/2$ (for Gaussians). For $\pm 1$: $\text{Var}(X_n^c) = 1/n^{2\alpha}$ but also $X_n^c = X_n$ for $n \ge 2$ so the summability of the variance is needed; same threshold $\alpha > 1/2$.

### 7.5 Kolmogorov's one-series theorem

**Corollary 7.2.** If $X_n$ are independent, zero-mean, with $\sum_n \text{Var}(X_n) < \infty$, then $\sum_n X_n$ converges a.s. and in $L^2$.

This is the special case where conditions (1) and (2) are vacuous/automatic. It is the foundation of the $L^2$ / martingale theory of random series.

---

## 8. Convergence in Distribution: Deeper Structure

Convergence in distribution has a rich theory because it is the notion most aligned with central limit phenomena.

### 8.1 Portmanteau theorem

**Theorem 8.1** (Portmanteau)**.** The following are equivalent:

(a) $X_n \xRightarrow{d} X$ (i.e., $F_{X_n}(x) \to F_X(x)$ at continuity points of $F_X$).

(b) $E[f(X_n)] \to E[f(X)]$ for every bounded continuous $f: \mathbb{R} \to \mathbb{R}$.

(c) $E[f(X_n)] \to E[f(X)]$ for every bounded Lipschitz $f$.

(d) $\liminf_n P(X_n \in U) \ge P(X \in U)$ for every open set $U$.

(e) $\limsup_n P(X_n \in C) \le P(X \in C)$ for every closed set $C$.

(f) $P(X_n \in B) \to P(X \in B)$ for every Borel set $B$ with $P(X \in \partial B) = 0$.

*Proof sketch (key parts).*

(a)$\Rightarrow$(b): Approximate bounded continuous $f$ by step functions on a grid of continuity points of $F_X$; use dominated convergence on differences. More cleanly, note (a) is equivalent to convergence of the *distributions* as Borel measures, and (b) is just integrating against a test function in $C_b$.

(b)$\Rightarrow$(c): Lipschitz $\subset$ $C_b$.

(c)$\Rightarrow$(d): For open $U$, approximate $\mathbf{1}_U$ from below by Lipschitz functions $f_k(x) = \min(k \cdot d(x, U^c), 1)$ which increase to $\mathbf{1}_U$. Then $E[f_k(X)] \le \liminf_n E[f_k(X_n)] \le \liminf_n P(X_n \in U)$; MCT gives $P(X \in U) \le \liminf_n P(X_n \in U)$.

(d)$\Leftrightarrow$(e): take complements.

(d)&(e) $\Rightarrow$ (f): For $B$ with $P(X \in \partial B) = 0$, $P(X \in B) = P(X \in B^\circ) = P(X \in \bar{B})$. Sandwich: $P(X \in B^\circ) \le \liminf P(X_n \in B) \le \limsup P(X_n \in B) \le P(X \in \bar{B})$.

(f)$\Rightarrow$(a): For $x$ continuity point of $F_X$, take $B = (-\infty, x]$, then $\partial B = \{x\}$ has $P(X = x) = 0$. $\square$

Part (b) is often taken as the *definition* of weak convergence for probability measures on a general topological space.

### 8.2 Slutsky's theorem

**Theorem 8.2** (Slutsky)**.** If $X_n \xRightarrow{d} X$ and $Y_n \xrightarrow{P} c$ (constant), then
$$
X_n + Y_n \xRightarrow{d} X + c, \qquad X_n Y_n \xRightarrow{d} cX, \qquad X_n / Y_n \xRightarrow{d} X/c \ (c \ne 0).
$$

*Proof of addition.* Note $Y_n \xrightarrow{P} c$ iff $Y_n - c \xrightarrow{P} 0$. For a bounded Lipschitz $f$ with constant $L$,
$$
|E[f(X_n + Y_n)] - E[f(X_n + c)]| \le L \cdot E[|Y_n - c| \wedge (2\|f\|_\infty / L)] \to 0
$$
by bounded convergence on $\{|Y_n - c| > \varepsilon\}$ vanishing. And $E[f(X_n + c)] \to E[f(X + c)]$ by Portmanteau (a)$\Rightarrow$(b). $\square$

The multiplicative version is similar with localization. Key insight: you can plug in-probability-convergent objects into weakly-convergent expressions provided the limit is constant.

**Warning.** Slutsky requires the limit of $Y_n$ to be a constant, not a random variable. If $Y_n \xRightarrow{d} Y$ for non-degenerate $Y$, then $X_n + Y_n$ need not have any limit in distribution — it depends on the joint distribution. Example: $X_n = \xi$, $Y_n = -\xi$ iid $\mathcal{N}(0,1)$ gives $X_n + Y_n = 0$, not $X + Y = \mathcal{N}(0,2)$.

### 8.3 Continuous mapping theorem

**Theorem 8.3** (Continuous mapping)**.** If $X_n \xRightarrow{d} X$ and $g: \mathbb{R} \to \mathbb{R}$ is Borel with $P(X \in D_g) = 0$ where $D_g$ is the set of discontinuities, then $g(X_n) \xRightarrow{d} g(X)$.

*Proof.* Use Portmanteau (f) with $B = g^{-1}(A)$ for $A$ Borel. One checks $\partial B \subseteq D_g \cup g^{-1}(\partial A)$, so for $A$ with $P(g(X) \in \partial A) = 0$, $P(X \in \partial B) = 0$. Then $P(g(X_n) \in A) = P(X_n \in B) \to P(X \in B) = P(g(X) \in A)$. Finally, this along with convergence at continuity points of $F_{g(X)}$ gives $g(X_n) \xRightarrow{d} g(X)$ by choosing $A = (-\infty, t]$ with $t$ continuity point of $F_{g(X)}$. $\square$

Same result holds if $X_n, X$ are $\mathbb{R}^d$-valued and $g: \mathbb{R}^d \to \mathbb{R}^m$; this is the standard form used in asymptotic statistics (e.g., $\sqrt{n}(T_n - \theta) \Rightarrow \mathcal{N}(0, \Sigma)$ implies $g(T_n)$ normal with delta-method variance).

### 8.4 Tightness and Prokhorov's theorem

**Definition 8.4.** A family of probability measures $\{\mu_\alpha\}$ on $\mathbb{R}$ is **tight** if for every $\varepsilon > 0$ there exists $M$ with $\mu_\alpha([-M, M]) > 1 - \varepsilon$ for all $\alpha$.

**Theorem 8.5** (Helly / Prokhorov)**.** A sequence of probability measures $(\mu_n)$ on $\mathbb{R}$ has a weakly convergent subsequence iff it is tight. (On general Polish spaces: tight iff sequentially precompact in the weak topology.)

*Proof sketch on $\mathbb{R}$.* Helly's selection theorem: any sequence of CDFs has a subsequence converging pointwise at all rational points. The limit defines a non-decreasing, right-continuous function $F$, but it need not be a CDF (mass can escape to $\pm \infty$). Tightness exactly prevents this escape: it forces $F(-\infty) = 0$, $F(\infty) = 1$, making $F$ a genuine CDF. Then the pointwise subsequence limit is a weak limit at every continuity point.

Tightness is the probabilistic analog of precompactness. It is the key hypothesis in proving CLT-type results: show tightness + identification of finite-dimensional limits.

---

## 9. Python Computational Demonstrations

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# =====================================================================
# Demo 1: The typewriter sequence — in probability but not a.s.
# =====================================================================
def typewriter_sequence(omega, n):
    """X_n(omega) = 1_{J_n}(omega), where J_n is the n-th dyadic interval
    in the typewriter enumeration."""
    k = 0
    # find level k such that sum_{j <= k} 2^j >= n
    cum = 0
    while cum + 2**(k+1) < n:
        k += 1
        cum += 2**k
    j = n - cum - 1
    k += 1
    width = 2.0**(-k)
    left = j * width
    right = (j + 1) * width
    return (omega >= left) & (omega < right)

# Verify convergence in probability
omega = rng.uniform(0, 1, 10000)
probs = [np.mean(typewriter_sequence(omega, n)) for n in range(1, 500)]
print(f"Typewriter: P(|X_100| > 0.5) ≈ {probs[99]:.4f}")
print(f"Typewriter: P(|X_400| > 0.5) ≈ {probs[399]:.4f}")
# These should decrease, demonstrating convergence in probability.

# =====================================================================
# Demo 2: Tall-narrow spike — a.s. but not in L^p
# =====================================================================
def spike(omega, n, p=2):
    return (omega < 1/n).astype(float) * n**(1/p)

omega = rng.uniform(0, 1, 100000)
for n in [10, 100, 1000, 10000]:
    Xn = spike(omega, n, p=2)
    print(f"n={n:>5}: E[X_n^2] = {np.mean(Xn**2):.4f} "
          f"(should stay at 1), P(X_n > 0.01) = {np.mean(Xn > 0.01):.4f}")

# =====================================================================
# Demo 3: Subsequence extraction — in probability ⇒ a.s. along subseq
# =====================================================================
# Construct X_n converging in probability but not a.s. (typewriter)
# and show a subsequence converging a.s.

def subsequence_convergence(num_samples=1000, N=1000):
    """For each omega, track the typewriter sequence and find
    the subsequence X_{n_k} = X_{2^k} which (for small enough index)
    converges to 0."""
    # Use shallow approximation: X_{2^k} for k=1..10
    omegas = rng.uniform(0, 1, num_samples)
    for k in range(1, 11):
        Xnk = typewriter_sequence(omegas, 2**k).mean()
        print(f"  k={k}: mean of X_{{2^{k}}} = {Xnk:.4f}")

print("\nSubsequence along n_k = 2^k:")
subsequence_convergence()

# =====================================================================
# Demo 4: Uniform integrability and Vitali's theorem
# =====================================================================
# Construct a UI family {|X_n|^2 bounded} and verify L^1 convergence.

def ui_demo():
    """X_n = Z + 1/n where Z ~ N(0,1): UI since L^2-bounded."""
    ns = [10, 100, 1000]
    for n in ns:
        Z = rng.standard_normal(10000)
        Xn = Z + 1/n
        # Uniform integrability: E[|X_n| 1_{|X_n|>M}]
        M = 5
        ui_tail = np.mean(np.abs(Xn) * (np.abs(Xn) > M))
        print(f"n={n}: E[|X_n| 1_{{|X_n|>{M}}}] = {ui_tail:.4f}")
        print(f"      L^1 dist from Z: {np.mean(np.abs(Xn - Z)):.4f}")
print("\nUniform integrability demo:")
ui_demo()

# =====================================================================
# Demo 5: Portmanteau theorem — test via bounded continuous functions
# =====================================================================
# X_n = (Z_1 + ... + Z_n)/sqrt(n) with Z_i iid uniform(-sqrt(3), sqrt(3))
# Should converge in distribution to N(0,1) by CLT.

def portmanteau_test():
    print("\nPortmanteau test (CLT for scaled uniforms):")
    for n in [5, 50, 500]:
        Z = rng.uniform(-np.sqrt(3), np.sqrt(3), (10000, n))
        Sn = Z.sum(axis=1) / np.sqrt(n)
        # Test bounded continuous f(x) = sin(x)
        E_sin_Sn = np.mean(np.sin(Sn))
        # E[sin(N(0,1))] = 0 by symmetry
        # Test f(x) = exp(-x^2/2)
        E_exp_Sn = np.mean(np.exp(-Sn**2 / 2))
        # E[exp(-Z^2/2) for Z~N(0,1)] = 1/sqrt(2) ≈ 0.7071
        print(f"  n={n:>4}: E[sin(S_n)] = {E_sin_Sn:+.4f} (target 0), "
              f"E[exp(-S_n^2/2)] = {E_exp_Sn:.4f} (target 0.7071)")

portmanteau_test()

# =====================================================================
# Demo 6: Slutsky's theorem
# =====================================================================
# X_n ~ N(0,1) (fixed distribution), Y_n = 1 + 1/n (constant limit)
# Then X_n * Y_n → X in distribution.
def slutsky_demo():
    print("\nSlutsky's theorem demo:")
    for n in [1, 10, 100, 1000]:
        X = rng.standard_normal(100000)
        Y = 1 + 1/n  # deterministic
        Z = X * Y
        # Compare CDFs at x=1
        emp = np.mean(Z <= 1)
        true = stats.norm.cdf(1)
        print(f"  n={n:>4}: F_Z(1) emp = {emp:.4f}, target = {true:.4f}")

slutsky_demo()

# =====================================================================
# Demo 7: Kolmogorov's three-series — threshold behavior
# =====================================================================
# S_n = sum xi_k / k^alpha with xi_k iid standard normal.
# Converges a.s. iff alpha > 1/2.
def three_series_demo(alpha, N=10000, trials=5):
    maxes = []
    for _ in range(trials):
        xi = rng.standard_normal(N)
        partial = np.cumsum(xi / np.arange(1, N+1)**alpha)
        maxes.append(np.max(np.abs(partial[-100:])) - np.abs(partial[-101]))
    return np.mean(maxes)

print("\nKolmogorov's three-series threshold (sum ξ_k/k^α):")
for alpha in [0.3, 0.5, 0.6, 0.8, 1.0]:
    oscillation = three_series_demo(alpha)
    print(f"  α = {alpha}: tail oscillation ≈ {oscillation:.4f} "
          f"({'converges' if alpha > 0.5 else 'diverges'})")
```

**Output interpretation.**
- Typewriter: probabilities decrease slowly (typewriter levels are $1/k$-rate), but every $\omega$ is hit infinitely often — exactly the tension between the two modes.
- Tall spike: $E[X_n^2]$ stays at $1$ while $P(X_n > 0.01)$ decays to $0$ — mass escapes in amplitude.
- Subsequence: along $n_k = 2^k$ the typewriter skips through levels and eventually stays out of any fixed region.
- UI demo: tails shrink uniformly, and $L^1$ distance goes to $0$.
- Portmanteau: CLT convergence against smooth test functions.
- Slutsky: multiplicative convergence with constant limit.
- Three-series: sharp threshold at $\alpha = 1/2$.

---

## 10. [QUANT APPLICATION] Convergence in Quantitative Finance

### 10.1 Consistency of Monte Carlo estimators

To price an option with payoff $h(S_T)$ under the risk-neutral measure $Q$, we simulate $S_T^{(1)}, \dots, S_T^{(N)}$ iid and form
$$
\hat{V}_N := \frac{1}{N} \sum_{i=1}^N e^{-rT} h(S_T^{(i)}).
$$
**SLLN** (which we prove in Module 2.4) gives $\hat{V}_N \xrightarrow{\text{a.s.}} V := E^Q[e^{-rT} h(S_T)]$. This is almost-sure consistency — running the simulation long enough gives the right price with probability one.

### 10.2 Asymptotic distribution of estimators (CLT $\to$ confidence intervals)

By **CLT** (Module 2.4),
$$
\sqrt{N}\, (\hat{V}_N - V) \xRightarrow{d} \mathcal{N}(0, \sigma^2), \quad \sigma^2 = \text{Var}_Q(e^{-rT} h(S_T)).
$$
This gives asymptotic confidence intervals via Slutsky: using the plug-in variance estimator $\hat{\sigma}_N^2 \xrightarrow{P} \sigma^2$,
$$
\frac{\sqrt{N}(\hat{V}_N - V)}{\hat{\sigma}_N} \xRightarrow{d} \mathcal{N}(0,1),
$$
so $\hat{V}_N \pm z_{\alpha/2} \hat{\sigma}_N / \sqrt{N}$ has asymptotic coverage $1 - \alpha$.

### 10.3 Uniform integrability and delta-hedging

In local vol / stochastic vol models, one shows that discounted wealth processes $\hat{W}_t$ under a self-financing strategy form a local martingale. Whether they are genuine martingales (needed for no-arbitrage pricing formulas) is precisely the question of **uniform integrability** of $(\hat{W}_t)$. Novikov's condition $E^P[\exp(\frac{1}{2}\int_0^T \theta_s^2 \, ds)] < \infty$ is a sufficient condition for UI of the Girsanov exponential $(\mathcal{E}(\int \theta \, dW))_t$; without it, the change of measure may fail.

### 10.4 Weak convergence and pricing by approximation

When pricing American options or path-dependent claims, we often discretize time: $\pi_n = \{0, T/n, 2T/n, \dots, T\}$. Letting $V_n$ be the price computed on $\pi_n$ and $V$ the true continuous-time price, one establishes $V_n \to V$ via
1. Show discretized processes $S^{(n)}$ satisfy $S^{(n)} \xRightarrow{d} S$ (functional CLT / Donsker).
2. Continuity of the pricing functional in an appropriate topology (e.g., Skorokhod).
3. Apply continuous mapping theorem.

This is the rigorous foundation for binomial tree pricing (CRR model $\to$ Black-Scholes as $n \to \infty$).

### 10.5 Consistency of GARCH / vol estimators

For GARCH(1,1) $\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$, the QMLE $\hat{\theta}_n$ satisfies $\hat{\theta}_n \xrightarrow{\text{a.s.}} \theta_0$ under stationarity and finite-moment conditions. This is a.s. convergence from ergodic theorems — a strengthening of the LLN for stationary sequences (Module 2.7, Markov chains / ergodic theory).

### 10.6 $L^2$ convergence in portfolio optimization

For a continuous-time mean-variance optimal portfolio $\pi_t^*$, one often constructs approximations $\pi_t^{(n)}$ converging in $L^2$:
$$
E\Big[\int_0^T |\pi_t^{(n)} - \pi_t^*|^2 \, dt\Big] \to 0.
$$
$L^2$ convergence is the right notion because the optimization criterion (variance of terminal wealth) is quadratic, and $L^2$ is the natural Hilbert-space setting for projections.

### 10.7 Convergence of empirical risk measures

Historical VaR: $\widehat{\text{VaR}}_\alpha^n = $ empirical $\alpha$-quantile of $n$ past returns. One shows $\widehat{\text{VaR}}_\alpha^n \xrightarrow{\text{a.s.}} \text{VaR}_\alpha$ under mild conditions (Glivenko-Cantelli, or directly by inverting $F_n \to F$ in distribution at continuity points and using a.s. extraction via Borel-Cantelli). Expected Shortfall $\widehat{\text{ES}}_\alpha^n$ converges via Vitali's theorem ($L^1$ for the tail — uniform integrability from moment assumptions on returns).

### 10.8 Kolmogorov's three-series in factor models

For a factor-return decomposition $r_t = \alpha + \sum_k \beta_k f_{t,k} + \varepsilon_t$ with infinitely many factors, the condition $\sum_k \beta_k^2 \text{Var}(f_k) < \infty$ (convergence of the factor sum in $L^2$) comes from Kolmogorov's one-series theorem. This underlies the Ross APT assumption that idiosyncratic risk is diversifiable.

---

## 11. Worked Examples

### Example 11.1 (Convergence in distribution but not in probability)

$X, X_1, X_2, \dots$ iid Bernoulli($1/2$). Each $X_n \sim X$ so $X_n \xRightarrow{d} X$. But $P(|X_n - X| > 1/2) = P(X_n \ne X) = 1/2 \not\to 0$, so no convergence in probability.

### Example 11.2 (Convergence in $L^p$ requires UI)

Let $X_n = n \cdot \mathbf{1}_{U \le 1/n}$ with $U \sim \text{Unif}(0, 1)$. Then $X_n \to 0$ a.s. and $\xrightarrow{P} 0$. But $E[X_n] = 1$ for all $n$, so $X_n \not\to 0$ in $L^1$. The family is not UI: $E[|X_n| \mathbf{1}_{|X_n| > M}] = 1$ once $n > M$.

### Example 11.3 (Scheffé vs general Lebesgue)

Let $f_n(x) = n \cdot \mathbf{1}_{[0, 1/n]}(x)$ on $[0, 1]$. These are densities on $(0, 1)$? Actually $\int f_n = 1$, and $f_n \to 0$ a.e. (Lebesgue). The limit "$0$" is not a density, so Scheffé's hypothesis ($f_n \to f$ with $f$ a density) fails, and indeed $\int f_n = 1 \not\to 0 = \int 0$.

Contrast: $f_n(x) = (n/(n-1)) \cdot \mathbf{1}_{[1/n, 1]}(x)$. Then $f_n \to 1$ a.e., $\int f_n = 1 = \int 1$, and by Scheffé $\|f_n - 1\|_1 \to 0$ (verify: $\int_{[0,1/n]} 1 + \int_{[1/n,1]} |n/(n-1) - 1| = 1/n + (1 - 1/n)/(n-1) = 2/n + o(1/n)$ — indeed $\to 0$).

### Example 11.4 (Three-series with truncated variables)

$X_n$ independent, $X_n = n$ with probability $1/n^2$, $X_n = 0$ otherwise. Truncate at $c = 1$: $X_n^c = 0$ always for $n \ge 2$. All three series trivially converge (they're eventually zero). Original series: $\sum X_n$ — only finitely many non-zero terms a.s. (Borel-Cantelli I on $\sum 1/n^2 < \infty$), so converges a.s. to a finite random variable. Consistent with the theorem.

### Example 11.5 (Three-series fails)

$X_n$ independent, $X_n = \pm n$ each with probability $1/(2n)$, $X_n = 0$ with probability $1 - 1/n$. $P(|X_n| > 1) = 1/n$, *not* summable. Borel-Cantelli II (independence): $P(|X_n| > 1 \text{ i.o.}) = 1$. So $X_n \not\to 0$ a.s., hence $\sum X_n$ cannot converge a.s. And indeed condition (1) fails.

---

## 12. Exercises

### Tier ★ (Foundational)

**2.3.E1.** Prove that if $X_n \xrightarrow{P} X$ and $X_n \xrightarrow{P} Y$, then $X = Y$ a.s.

**2.3.E2.** Show that if $X_n \xrightarrow{P} X$ and $Y_n \xrightarrow{P} Y$, then $X_n + Y_n \xrightarrow{P} X + Y$ and $X_n Y_n \xrightarrow{P} XY$. (For products, use a truncation argument.)

**2.3.E3.** Let $X_n \xrightarrow{L^2} X$. Show $E[X_n] \to E[X]$ and $\text{Var}(X_n) \to \text{Var}(X)$.

**2.3.E4.** Prove that $X_n \xrightarrow{\text{a.s.}} X$ implies $\max_{k \le n} X_k \xrightarrow{\text{a.s.}} \sup_k X_k$ (when the right side is finite).

**2.3.E5.** Give an example of $X_n \xrightarrow{\text{a.s.}} X$ and $Y_n \xrightarrow{\text{a.s.}} Y$ with $X_n, Y_n$ defined on possibly distinct probability spaces, such that no conclusion about joint distributions of $(X_n, Y_n)$ can be drawn.

**2.3.E6.** Show that convergence in probability is preserved under continuous functions: if $X_n \xrightarrow{P} X$ and $g$ is continuous, then $g(X_n) \xrightarrow{P} g(X)$. (Hint: the subsequence characterization.)

**2.3.E7.** Prove that a uniformly bounded sequence $|X_n| \le M$ which converges in probability also converges in $L^p$ for every $p \in [1, \infty)$.

**2.3.E8.** Suppose $E[|X_n|^{1+\delta}] \le C$ for some $\delta > 0$ and $X_n \xrightarrow{P} X$. Show $X_n \xrightarrow{L^1} X$.

### Tier ★★ (Intermediate)

**2.3.E9** (Skorokhod representation)**.** Let $F_n, F$ be CDFs with $F_n \to F$ at continuity points. Using the quantile transform $F^{-1}$, construct $X_n, X$ on $([0,1], \mathcal{B}, \lambda)$ with $X_n \sim F_n$, $X \sim F$, and $X_n \to X$ a.e. (This is the 1-D Skorokhod representation theorem.)

**2.3.E10.** Prove the converse of Theorem 4.1: if *every* subsequence of $X_n$ has a further subsequence converging a.s. to $X$, then $X_n \xrightarrow{P} X$.

**2.3.E11** (Weak $L^p$ and UI)**.** Show: if $\sup_n E[|X_n|^p] < \infty$ for some $p > 1$, then $\{X_n\}$ is UI in $L^1$. Conclude: $L^p$-boundedness + convergence in probability $\Rightarrow$ $L^1$ convergence.

**2.3.E12** (Lyapunov's inequality)**.** For $0 < q \le p \le \infty$, show $\|X\|_q \le \|X\|_p$ on a probability space. Use this to show: $L^p$ convergence implies $L^q$ convergence for $q \le p$.

**2.3.E13** (Ky Fan metric)**.** Show $d(X, Y) := \inf\{\varepsilon > 0 : P(|X - Y| > \varepsilon) \le \varepsilon\}$ is a metric, and $d(X_n, X) \to 0$ iff $X_n \xrightarrow{P} X$.

**2.3.E14.** Prove Scheffé's lemma for general (possibly negative) sequences in $L^1$: if $X_n \xrightarrow{\text{a.s.}} X$ and $\|X_n\|_1 \to \|X\|_1 < \infty$, then $X_n \xrightarrow{L^1} X$.

**2.3.E15** (Convergence of random measures)**.** Let $\mu_n, \mu$ be probability measures on $\mathbb{R}$ with $\mu_n \Rightarrow \mu$. Show: $\int f \, d\mu_n \to \int f \, d\mu$ for all continuous $f$ vanishing at infinity.

**2.3.E16** (Portmanteau for characteristic functions — preview)**.** Using that $\mathcal{F}\mu_n(\xi) = \int e^{i\xi x} \, d\mu_n(x)$, show that weak convergence $\mu_n \Rightarrow \mu$ implies pointwise convergence $\mathcal{F}\mu_n \to \mathcal{F}\mu$. (The converse — Lévy's continuity theorem — is Module 2.5.)

### Tier ★★★ (Advanced / Quant)

**2.3.E17** (Hoeffding's inequality, exponential concentration)**.** Let $X_1, \dots, X_n$ be independent with $X_i \in [a_i, b_i]$ a.s. Prove:
$$
P\Big(\Big|\bar{X}_n - E[\bar{X}_n]\Big| > t\Big) \le 2 \exp\left(-\frac{2n^2 t^2}{\sum (b_i - a_i)^2}\right).
$$
Hint: Use Chernoff's method. Bound $E[e^{\lambda(X_i - EX_i)}] \le \exp(\lambda^2 (b_i - a_i)^2 / 8)$ via convexity.

**2.3.E18** (Robbins-Siegmund)**.** Let $Z_n, \alpha_n, \beta_n, \gamma_n \ge 0$ be adapted and $E[Z_{n+1} | \mathcal{F}_n] \le (1 + \alpha_n) Z_n + \beta_n - \gamma_n$. If $\sum \alpha_n, \sum \beta_n < \infty$ a.s., then $Z_n \to Z_\infty$ a.s. and $\sum \gamma_n < \infty$ a.s. (Supermartingale convergence theorem — used to prove SGD convergence.)

**2.3.E19** (Donsker's theorem in weak form)**.** Let $\xi_i$ iid with $E[\xi_1] = 0$, $\text{Var}(\xi_1) = 1$. Set $S_n = \xi_1 + \dots + \xi_n$. Define the linearly interpolated process
$$
W^{(n)}(t) := \frac{1}{\sqrt{n}}\left[ S_{\lfloor nt \rfloor} + (nt - \lfloor nt \rfloor)(S_{\lfloor nt \rfloor + 1} - S_{\lfloor nt \rfloor}) \right].
$$
Show that $W^{(n)}(t) \xRightarrow{d} W(t)$ for each fixed $t$ as $n \to \infty$. (The functional version — $W^{(n)} \Rightarrow W$ in $C[0, 1]$ — is Donsker's theorem; it requires proving tightness on $C[0,1]$.)

**2.3.E20** (Binomial option pricing convergence — CRR $\to$ Black-Scholes)**.** In the CRR binomial model, $S_t^{(n)}$ up/down factors $u = e^{\sigma \sqrt{T/n}}$, $d = 1/u$, risk-neutral prob $p = (e^{rT/n} - d)/(u - d)$. Show $S_T^{(n)} \xRightarrow{d} S_T := S_0 e^{(r - \sigma^2/2)T + \sigma W_T}$ as $n \to \infty$, and hence the CRR price of a European call converges to the Black-Scholes price. (Hint: show $\log(S_T^{(n)}/S_0) \xRightarrow{d} \mathcal{N}((r - \sigma^2/2)T, \sigma^2 T)$ by computing the mean and variance of the log-returns and applying CLT.)

**2.3.E21** (Glivenko-Cantelli and VaR)**.** Let $X_1, X_2, \dots$ iid with CDF $F$. Let $F_n(x) = n^{-1} \sum_{i=1}^n \mathbf{1}_{X_i \le x}$. Prove Glivenko-Cantelli: $\sup_x |F_n(x) - F(x)| \xrightarrow{\text{a.s.}} 0$. Deduce: empirical quantile $\hat{q}_\alpha^n \xrightarrow{\text{a.s.}} q_\alpha$ if $F$ is continuous at $q_\alpha$. This is the consistency of historical VaR.

**2.3.E22** (Convergence of risk measures)**.** Let $X_n \xrightarrow{L^1} X$. Show:
(a) $\text{ES}_\alpha(X_n) := -\frac{1}{\alpha} \int_0^\alpha F_{X_n}^{-1}(u) \, du \to \text{ES}_\alpha(X)$ for continuity points.
(b) A law-invariant convex risk measure (e.g., entropic, shortfall) is $L^1$-continuous.
(c) Deduce that if the historical return distribution converges to the true distribution in $L^1$ (which holds under UI), then the empirical ES estimator is consistent.

---

## 13. Summary and Forward Pointers

**What we proved.**
- Four modes: a.s., $P$, $L^p$, $d$.
- Implications: $L^p \Rightarrow L^q \ (q \le p) \Rightarrow P \Rightarrow d$, a.s. $\Rightarrow P$.
- Counterexamples for every non-implied arrow.
- Subsequence upgrade: $P \Rightarrow$ a.s. along a subsequence.
- UI + $P$ $\Leftrightarrow$ $L^p$ (Vitali); Scheffé for density convergence.
- Borel-Cantelli summability $\Rightarrow$ a.s.
- Kolmogorov's three-series theorem (necessary and sufficient for a.s. convergence of independent series).
- Portmanteau, Slutsky, continuous mapping, tightness/Prokhorov for weak convergence.

**The big picture.** In Module 2.2 we studied $E[X]$ for one random variable; in Module 2.3 we studied *limits* of random variables. Convergence in distribution is the weakest and most useful for central limit phenomena; a.s. convergence is the strongest and most useful for sample-path reasoning; $L^p$ convergence sits in between and is the right notion for Hilbert-space arguments (projections, regression, Kalman).

**Forward pointers.**
- **Module 2.4** (LLN and CLT): The two flagship theorems of probability. WLLN/SLLN in several flavors (Khinchin, Etemadi, Kolmogorov), CLT for iid (Lindeberg-Lévy) and for triangular arrays (Lindeberg-Feller), Berry-Esseen rates. Uses: Borel-Cantelli (here), Kolmogorov's three-series (here).
- **Module 2.5** (Characteristic functions): CFs are the analytic tool for proving CLT — and more. Lévy's continuity theorem closes the loop: pointwise CF convergence + continuity at $0$ $\Leftrightarrow$ weak convergence. Uses: Portmanteau (here), tightness (here).
- **Module 2.6** (Martingales): Doob's maximal inequality, martingale convergence theorems — unify and extend Kolmogorov's three-series. Uses: UI (here), $L^p$ theory, conditional expectation (Module 2.2).
- **Module 2.7** (Markov chains): Ergodic theorem as SLLN for stationary sequences. Uses: a.s. convergence (here).

**Next module:** Laws of large numbers (WLLN, SLLN) and the central limit theorem — the culmination of everything we've built.

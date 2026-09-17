# Subject 5, Module 1: No-Arbitrage and the Fundamental Theorems of Asset Pricing

> *"The first fundamental theorem says 'no arbitrage is equivalent to the existence of a risk-neutral measure.' Everything in mathematical finance — Black-Scholes, HJM, LIBOR models, exotic pricing, credit derivatives — is a corollary."*

Welcome to Subject 5: **Asset Pricing and Derivatives**. With the machinery of stochastic processes (Subject 3) and stochastic control (Subject 4) in hand, we now turn to the central theoretical edifice of modern quantitative finance: the **no-arbitrage theory** that undergirds the pricing of derivatives.

## Roadmap of Subject 5

- **Module 5.1 (this module):** No-arbitrage foundations and the Fundamental Theorems of Asset Pricing.
- **Module 5.2:** Risk-neutral pricing and the Black-Scholes-Merton formula.
- **Module 5.3:** Exotic options and path-dependent derivatives.
- **Module 5.4:** Interest-rate models (short-rate, HJM, LIBOR market model).
- **Module 5.5:** Credit risk models (structural and reduced-form).
- **Module 5.6:** Stochastic volatility models (Heston, SABR, rough vol).
- **Module 5.7:** Market microstructure and execution.

## Prerequisites

- **Module 2.6** (Martingales and filtrations).
- **Module 3.2-3.4** (Stochastic integration, Itô's formula, SDEs).
- **Module 3.7** (Continuous-time martingales, BDG, semi-martingales).
- **Module 1.6** (Radon-Nikodym, signed measures) — for equivalent measure change.
- Helpful: elementary knowledge of options and bonds.

---

## 5.1.0 Motivation: What is "No Arbitrage"?

### Informal definition

An **arbitrage** is a trading strategy that:
1. Requires no initial capital.
2. Has no possibility of loss.
3. Has a positive probability of profit.

In symbols: strategy $(\pi_t)$ with $V_0(\pi) = 0$, $V_T(\pi) \ge 0$ a.s., and $\mathbb{P}(V_T(\pi) > 0) > 0$.

The **no-arbitrage principle** — really the single economic postulate underlying quantitative finance — says: *in a well-functioning market, such strategies should not exist, or if they do, their profits are quickly competed away*.

This single economic idea has astonishing mathematical consequences. It forces prices to satisfy a precise martingale property under a suitable probability measure, and this in turn allows closed-form pricing of derivatives.

### Why this is deep

On first exposure the no-arbitrage assumption feels circular ("no free lunch" is tautologically true if we define arbitrage as a free lunch). The depth comes from:

1. The space of strategies is enormous — dynamic trading with continuous rebalancing over uncountably many paths. Ruling out arbitrage in this huge space is a genuine restriction.
2. The restriction is precisely equivalent to the existence of a probability measure under which discounted prices are martingales — a purely mathematical statement with zero explicit reference to economics.
3. This equivalent-measure structure then gives you **unique** derivative prices (if markets are complete) or **arbitrage price bounds** (if incomplete).

### Historical perspective

The abstract connection between no-arbitrage and martingales was formulated in the 1970s (Ross 1976, Cox-Ross 1976). The rigorous general version (for semi-martingale price processes in continuous time) is due to Delbaen-Schachermayer (1994, 1998). Their monograph *The Mathematics of Arbitrage* (2006) is the standard reference.

The theorem is stated in progressively cleaner forms:
- Discrete, finite horizon, finite $\Omega$: easy (Harrison-Pliska 1981).
- Discrete, infinite $\Omega$: harder (Dalang-Morton-Willinger 1990).
- Continuous time, semi-martingale prices: Delbaen-Schachermayer (1994, 1998).

---

## 5.1.1 The Discrete Model (Harrison-Pliska)

Start with the cleanest setup where proofs are transparent.

### Setup

- Finite probability space $(\Omega, \mathcal{F}, \mathbb{P})$ with $|\Omega| < \infty$ and $\mathbb{P}(\{\omega\}) > 0$ for all $\omega$.
- Finite set of dates $0 = t_0 < t_1 < \cdots < t_N = T$.
- Filtration $\mathbb{F} = (\mathcal{F}_n)_{n=0}^N$ with $\mathcal{F}_0$ trivial and $\mathcal{F}_N = \mathcal{F}$.
- $d+1$ assets: riskless $S^0_n$ with $S^0_0 = 1$ and positive; $d$ risky assets $S^1_n, \ldots, S^d_n$ adapted.
- **Discounted prices:** $\tilde S^i_n := S^i_n / S^0_n$ (taking $S^0$ as numéraire).

### Trading strategies

A **strategy** $\phi = (\phi^0_n, \vec \phi_n)$ is predictable (decisions made at time $n-1$ based on $\mathcal{F}_{n-1}$, implemented at time $n$). 

Portfolio value at time $n$:
$$V_n(\phi) = \phi^0_n S^0_n + \vec \phi_n \cdot \vec S_n.$$

**Self-financing condition**: no money injected after time 0.
$$V_{n+1}(\phi) - V_n(\phi) = \phi^0_{n+1}(S^0_{n+1} - S^0_n) + \vec \phi_{n+1} \cdot (\vec S_{n+1} - \vec S_n).$$

Equivalently, in terms of discounted portfolio $\tilde V_n = V_n / S^0_n$:
$$\tilde V_n = \tilde V_0 + \sum_{k=1}^n \vec \phi_k \cdot (\tilde{\vec S}_k - \tilde{\vec S}_{k-1}) = \tilde V_0 + (\vec \phi \bullet \Delta \tilde{\vec S})_n.$$

In words: discounted self-financing wealth = initial wealth + stochastic integral of $\vec \phi$ against $\tilde{\vec S}$. This is a **discrete stochastic integral** (martingale transform).

### Arbitrage

**Definition.** Arbitrage is a self-financing strategy $\phi$ with $V_0(\phi) = 0$, $V_N(\phi) \ge 0$ a.s., and $\mathbb{P}(V_N(\phi) > 0) > 0$.

Equivalently, since $V_0 = 0$ and prices are positive, $\tilde V_N(\phi) = (\vec \phi \bullet \Delta \tilde{\vec S})_N \ge 0$ with positive probability of being strictly positive.

**No-arbitrage (NA) condition:** no such $\phi$ exists.

### Equivalent martingale measure (EMM)

**Definition.** A probability measure $\mathbb{Q}$ on $(\Omega, \mathcal{F})$ is an **equivalent martingale measure** (EMM) for $\tilde{\vec S}$ if:
1. $\mathbb{Q} \sim \mathbb{P}$ (equivalent: same null sets).
2. $\tilde{\vec S}$ is a $\mathbb{Q}$-martingale w.r.t. $\mathbb{F}$.

The set of EMMs is denoted $\mathcal{M}(\tilde S)$.

### First Fundamental Theorem (Harrison-Pliska 1981)

> **Theorem.** In the discrete, finite-$\Omega$ setup above, the following are equivalent:
> 1. No-arbitrage (NA) holds.
> 2. There exists an EMM $\mathbb{Q}$ for $\tilde{\vec S}$.

**Proof of (2) ⇒ (1).** Suppose $\mathbb{Q}$ is an EMM and $\phi$ is self-financing with $V_0 = 0$. Then $\tilde V_N = (\vec \phi \bullet \Delta \tilde{\vec S})_N$ is a $\mathbb{Q}$-martingale transform, hence a $\mathbb{Q}$-martingale. So $\mathbb{E}_\mathbb{Q}[\tilde V_N] = \tilde V_0 = 0$. If $\tilde V_N \ge 0$ $\mathbb{Q}$-a.s. (equivalent to $\mathbb{P}$-a.s. since $\mathbb{Q} \sim \mathbb{P}$) and $\mathbb{E}_\mathbb{Q}[\tilde V_N] = 0$, then $\tilde V_N = 0$ $\mathbb{Q}$-a.s., contradicting the arbitrage hypothesis.

**Proof of (1) ⇒ (2).** Define the set of attainable discounted gains:
$$\mathcal{K} := \{(\vec \phi \bullet \Delta \tilde{\vec S})_N : \phi \text{ predictable}\} \subset \mathbb{R}^\Omega.$$

This is a linear subspace of $\mathbb{R}^{|\Omega|}$.

The NA hypothesis says $\mathcal{K} \cap L^0_+ = \{0\}$, where $L^0_+ = \{X : X \ge 0 \text{ a.s.}\}$.

Define the simplex $\mathcal{C} := \{X \in L^0_+ : \mathbb{E}_\mathbb{P}[X] = 1\}$ (a convex compact set in $\mathbb{R}^\Omega$). NA says $\mathcal{K} \cap \mathcal{C} = \emptyset$.

By the **separation theorem for convex sets** (here $\mathbb{R}^{|\Omega|}$ finite-dim, so Hahn-Banach is trivial), there exists a nonzero linear functional $\ell \in (\mathbb{R}^\Omega)^*$ with
$$\ell(X) \le 0 < \ell(Y) \quad \forall X \in \mathcal{K}, Y \in \mathcal{C}.$$

Since $\mathcal{K}$ is a linear subspace, $\ell \equiv 0$ on $\mathcal{K}$. Since $\ell > 0$ on $\mathcal{C}$, we can write $\ell(X) = \mathbb{E}_\mathbb{P}[Z X]$ for some random variable $Z > 0$ a.s. (finite $\Omega$!). Normalize $\mathbb{E}_\mathbb{P}[Z] = 1$ and define $\mathbb{Q}$ by $d\mathbb{Q}/d\mathbb{P} = Z$.

$\ell \equiv 0$ on $\mathcal{K}$ means $\mathbb{E}_\mathbb{Q}[(\vec \phi \bullet \Delta \tilde{\vec S})_N] = 0$ for all predictable $\vec \phi$. Taking $\vec \phi_k = \mathbf{1}_A \cdot e_i$ for $A \in \mathcal{F}_{k-1}$ and standard basis vector $e_i$:
$$\mathbb{E}_\mathbb{Q}[\mathbf{1}_A (\tilde S^i_k - \tilde S^i_{k-1})] = 0.$$
This says $\mathbb{E}_\mathbb{Q}[\tilde S^i_k | \mathcal{F}_{k-1}] = \tilde S^i_{k-1}$, i.e., $\tilde S^i$ is a $\mathbb{Q}$-martingale. So $\mathbb{Q}$ is an EMM. ∎

### Observations

1. **Equivalence of $\mathbb{Q}$ and $\mathbb{P}$** is essential. They must agree on null sets. This captures the intuition that "physical impossibility" under $\mathbb{P}$ remains so under $\mathbb{Q}$; pricing doesn't change what's possible, only relative likelihoods.

2. **Numéraire choice.** $\mathbb{Q}$ depends on the choice of numéraire $S^0$. Different numéraires → different martingale measures (see "change of numéraire" in Module 5.2).

3. **Non-uniqueness.** If $|\Omega| > 2$ (non-trivially random) and there's only one risky asset, there can be many EMMs. Uniqueness characterizes *market completeness*.

### Example: one-period binomial model

$\Omega = \{u, d\}$, $\mathbb{P}(\{u\}) = p, \mathbb{P}(\{d\}) = 1-p$. Risky asset $S_1 = S_0 u$ or $S_0 d$ (with $d < 1+r < u$ where $S^0_1 = 1+r$).

Discounted: $\tilde S_1 = S_0 u / (1+r)$ or $S_0 d / (1+r)$.

Martingale condition $\mathbb{E}_\mathbb{Q}[\tilde S_1] = S_0$:
$$q \cdot \dfrac{S_0 u}{1+r} + (1-q) \cdot \dfrac{S_0 d}{1+r} = S_0 \ \Longrightarrow \ q = \dfrac{(1+r) - d}{u - d}.$$

Requirements: $d < 1+r < u$ (this is *exactly* the no-arbitrage condition for the binomial) ensures $q \in (0, 1)$. Since $\mathbb{Q}$ must be equivalent to $\mathbb{P}$ (both assign positive probability to $u, d$) this is the unique EMM.

---

## 5.1.2 Completeness and the Second Fundamental Theorem

### Attainability and replication

A contingent claim $X$ (time-$T$ payoff, $\mathcal{F}_T$-measurable) is **attainable** (or replicable) if there exists a self-financing strategy $\phi$ with $V_T(\phi) = X$.

The **arbitrage-free price** of an attainable claim is $V_0(\phi)$, independent of the choice of replicating $\phi$ (else arbitrage).

### Completeness

**Definition.** The market is **complete** if every $\mathcal{F}_T$-measurable random variable $X$ (bounded or integrable) is attainable.

### Second Fundamental Theorem

> **Theorem (Harrison-Pliska 1981).** In the discrete, finite-$\Omega$ setup with NA holding:
> $$\text{market is complete} \iff \text{EMM is unique}.$$

**Proof of ⇐.** Suppose EMM is unique, call it $\mathbb{Q}$. Let $X$ be any $\mathcal{F}_T$-measurable random variable (treat as element of $\mathbb{R}^\Omega$). 

Claim: $X$ is attainable from some initial wealth. Suppose not. Then the pair $(x + \mathcal{K})$ (affine space of attainable-from-$x$ wealths) misses $X$ for every $x \in \mathbb{R}$. But $\mathcal{K} + \mathbb{R}$ (cash plus gains) is a linear subspace, and $X \notin \mathcal{K} + \mathbb{R}$.

By separation, there's a linear functional vanishing on $\mathcal{K} + \mathbb{R}$ but nonzero on $X$. This functional corresponds to a signed measure. Decompose into positive parts, normalize, and one constructs a second EMM different from $\mathbb{Q}$, contradicting uniqueness.

**Proof of ⇒.** Suppose completeness holds. Given two EMMs $\mathbb{Q}_1, \mathbb{Q}_2$, for any $X \in L^\infty(\mathcal{F})$ replicate via $\phi$. Then $V_T(\phi)/S^0_T = X/S^0_T$ and
$$\mathbb{E}_{\mathbb{Q}_i}[X/S^0_T] = V_0(\phi)/S^0_0 = V_0(\phi),$$
so $\mathbb{E}_{\mathbb{Q}_1}[X/S^0_T] = \mathbb{E}_{\mathbb{Q}_2}[X/S^0_T]$ for all $X$, giving $\mathbb{Q}_1 = \mathbb{Q}_2$. ∎

### Interpretation

**Complete market ↔ unique price**: in a complete market, every derivative has a unique arbitrage-free price, given by the expected discounted payoff under the unique EMM.

**Incomplete market ↔ price interval**: in incomplete markets, there's a range of EMMs, giving a range of arbitrage-free prices. The **upper hedge** and **lower hedge** prices are:
$$\overline{V}_0(X) = \sup_{\mathbb{Q} \in \mathcal{M}} \mathbb{E}_\mathbb{Q}[X/S^0_T], \quad \underline{V}_0(X) = \inf_{\mathbb{Q} \in \mathcal{M}} \mathbb{E}_\mathbb{Q}[X/S^0_T].$$

### Binomial is complete

In the one-period binomial, claim $X$ on $\{u, d\}$ specified by $(X_u, X_d)$. Replicating portfolio: solve
$$\phi^0 (1+r) + \phi S_0 u = X_u, \quad \phi^0(1+r) + \phi S_0 d = X_d.$$
Two equations, two unknowns: $\phi = (X_u - X_d)/(S_0 (u-d))$ (the delta!), $\phi^0 = (X_u - \phi S_0 u)/(1+r)$. Complete.

### Trinomial is incomplete

$\Omega = \{u, m, d\}$ with three outcomes. Single stock. Only 2 instruments (stock + bond) but 3 states → not enough to span $\mathbb{R}^3$. Incomplete.

### Black-Scholes (continuous time) is complete

With continuous trading and a single Brownian driver, the market $(B_t, S_t)$ with $dS = \mu S dt + \sigma S dB$ has a unique EMM (Girsanov change with $\theta = (\mu - r)/\sigma$) and every claim is replicable by the martingale representation theorem. This completeness is what gives Black-Scholes its unique pricing formula.

---

## 5.1.3 Continuous-Time Fundamental Theorems

### The need for more care

In continuous time, arbitrage becomes more subtle. **Doubling strategies** (betting bigger after each loss) can sometimes create apparent free lunches. The theorem needs to be stated for a broader class of "admissible" strategies.

### Setup

- Filtered probability space $(\Omega, \mathcal{F}, \mathbb{F}, \mathbb{P})$ satisfying the usual conditions.
- Price process $\vec S = (S^1, \ldots, S^d)$ an $\mathbb{R}^d$-valued semi-martingale.
- Numéraire $S^0$ positive continuous; set $\tilde{\vec S} = \vec S / S^0$.

### Admissible strategies

A predictable process $\vec \phi$ is **admissible** if:
1. The stochastic integral $\vec \phi \bullet \tilde{\vec S}$ is defined.
2. There exists $a \ge 0$ such that $(\vec \phi \bullet \tilde{\vec S})_t \ge -a$ for all $t \ge 0$.

The lower bound (bounded-below wealth) rules out doubling strategies.

### No Free Lunch with Vanishing Risk (NFLVR)

Let $\mathcal{K}_{\text{adm}} := \{(\vec \phi \bullet \tilde{\vec S})_T : \vec \phi \text{ admissible}\}$. Define:
- $\mathcal{C} := (\mathcal{K}_{\text{adm}} - L^0_+) \cap L^\infty$ (cone of dominated bounded claims).

**NFLVR:** $\overline{\mathcal{C}}^{L^\infty} \cap L^0_+ = \{0\}$, where the closure is in the norm topology of $L^\infty$.

In words: you cannot approximate a nonnegative nonzero claim by a sequence of (admissible claim) - (nonnegative waste) in $L^\infty$ norm.

NFLVR is NA plus an additional "no approximate arbitrage" condition, needed in continuous time because of the pathological sequences.

### First Fundamental Theorem (Delbaen-Schachermayer 1994)

> **Theorem.** Let $\vec S$ be a locally bounded $\mathbb{R}^d$-valued semi-martingale. Then:
> $$\text{NFLVR holds} \iff \exists \mathbb{Q} \sim \mathbb{P} \text{ such that } \tilde{\vec S} \text{ is a local martingale under } \mathbb{Q}.$$

**Proof (sketch):** The hard direction is (⇒). The argument uses the Kreps-Yan separation theorem: closed convex cones in $L^\infty$ can be separated from compact convex sets not contained in them. The cone $\overline{\mathcal{C}}$ is weak-$*$ closed by NFLVR, and separation yields a positive continuous linear functional (i.e., a positive measure), which normalizes to the EMM.

### Second Fundamental Theorem (continuous time)

> **Theorem.** Under NFLVR, the market is complete iff the EMM is unique. In the complete case, every $\mathcal{F}_T$-measurable bounded random variable $X$ is representable as $X = \mathbb{E}_\mathbb{Q}[X/S^0_T] + (\vec \phi \bullet \tilde{\vec S})_T \cdot S^0_T$ for some admissible $\vec \phi$.

**Equivalent characterization:** Market is complete iff the filtration $\mathbb{F}$ is generated by the price processes (plus the riskless one), equivalently iff there's "just enough" randomness. In Brownian models, this means the number of Brownian drivers equals the number of risky assets.

### Local vs true martingale

A subtlety: in continuous time, $\tilde{\vec S}$ under $\mathbb{Q}$ is generally only a *local* martingale, not a true martingale. Admissibility bounds the integral $\vec \phi \bullet \tilde{\vec S}$ from below but doesn't guarantee integrability.

If $\tilde{\vec S}$ is uniformly integrable under $\mathbb{Q}$ (which it is in many practical cases, including Black-Scholes), it's a true martingale.

**Strict local martingale** example: inverse of Bessel process in dimension $\ge 3$. These give rise to "bubbles" (Cox-Hobson, Jarrow-Protter-Shimbo).

---

## 5.1.4 The Stochastic Discount Factor / State-Price Density

### Definition

Given EMM $\mathbb{Q}$ with Radon-Nikodym derivative $Z_T = d\mathbb{Q}/d\mathbb{P}$, define the **stochastic discount factor (SDF)** (also called state-price density, pricing kernel):
$$M_t := \dfrac{Z_t}{S^0_t}, \quad \text{where } Z_t = \mathbb{E}_\mathbb{P}[Z_T | \mathcal{F}_t].$$

Then any asset with terminal payoff $X$ (adapted cashflow at $T$) has time-0 price
$$V_0(X) = \mathbb{E}_\mathbb{P}[M_T X] = \mathbb{E}_\mathbb{Q}\left[\dfrac{X}{S^0_T}\right].$$

Equivalently, **all traded asset prices multiplied by $M_t$ are $\mathbb{P}$-martingales**:
$$V_t \cdot M_t = \mathbb{E}_\mathbb{P}[M_T V_T | \mathcal{F}_t].$$

### Why this reformulation matters

The SDF formulation is *canonical* — it works regardless of numéraire choice and makes the martingale property explicit under the physical measure $\mathbb{P}$.

**Economic interpretation.** $M_t(\omega)$ is the "price today of $\$1$ delivered at time $t$ in state $\omega$," weighted by the probability of $\omega$. Equivalently, $M_t$ is the marginal rate of substitution for a representative investor.

In equilibrium models, $M_t = e^{-\rho t} u'(C_t) / u'(C_0)$ where $C_t$ is aggregate consumption and $u$ is the representative investor's utility. This connects to Module 4.5 (Merton): the Merton investor's optimal consumption defines the equilibrium SDF.

### SDF in Black-Scholes

In the Black-Scholes model, $S^0_t = e^{rt}$ and $Z_t = \exp(-\theta B_t - \tfrac{1}{2}\theta^2 t)$ where $\theta = (\mu - r)/\sigma$ (market price of risk).

So
$$M_t = e^{-rt} \cdot \exp\left(-\theta B_t - \tfrac{1}{2}\theta^2 t\right) = \exp\left(-rt - \theta B_t - \tfrac{1}{2}\theta^2 t\right).$$

Via Itô:
$$\dfrac{dM_t}{M_t} = -r \, dt - \theta \, dB_t.$$

The SDF has drift $-r$ (the risk-free rate of time decay) and diffusion $-\theta$ (compensation for Brownian risk).

### Hansen-Jagannathan bound

The SDF gives a **bound on Sharpe ratios** in the economy. For any asset with return $R$:
$$\dfrac{|\mathbb{E}[R] - R_f|}{\text{std}(R)} \le \dfrac{\text{std}(M)}{\mathbb{E}[M]}.$$

The RHS is the **Hansen-Jagannathan bound**: the maximum Sharpe ratio is the ratio of SDF standard deviation to its mean. High Sharpe opportunities require a volatile SDF, which (in equilibrium models) requires volatile consumption growth or high risk aversion — the "equity premium puzzle" (Mehra-Prescott 1985).

### Pricing kernel and the equity premium puzzle

Empirically, US equity risk premium is ~6% with volatility ~16%, giving Sharpe ~0.4. The HJ bound then requires SDF volatility $\ge 40\%$.

In a rep-agent CRRA model, $M_t = e^{-\rho t}(C_t/C_0)^{-\gamma}$, so $\text{std}(M_t)/\mathbb{E}[M_t] \approx \gamma \cdot \text{std}(\Delta c)$ where $\Delta c$ is consumption growth. With $\text{std}(\Delta c) \approx 3\%$ (empirical), you need $\gamma \approx 13$ — implausibly high. This is the Mehra-Prescott equity premium puzzle.

**Resolutions:**
- Long-run risks (Bansal-Yaron): consumption has a small persistent component.
- Habit formation (Campbell-Cochrane): relative consumption matters.
- Disaster risk (Rietz, Barro): tail events dominate.
- Bounded rationality / ambiguity aversion.

---

## 5.1.5 Risk-Neutral Probability and the Numéraire

### The risk-neutral probability measure

The EMM $\mathbb{Q}$ is also called the **risk-neutral measure** because under $\mathbb{Q}$ all discounted asset prices have zero drift (martingale); equivalently, expected return equals the riskless rate for every asset.

Under $\mathbb{Q}$:
$$\dfrac{dS_t}{S_t} = r \, dt + \sigma \, dB^\mathbb{Q}_t$$
regardless of the $\mathbb{P}$-drift $\mu$.

Pricing rule for European claim $X$:
$$V_0(X) = e^{-rT} \mathbb{E}_\mathbb{Q}[X].$$

### Change of numéraire

Suppose we take a different positive asset $N_t$ as numéraire. Then the corresponding EMM $\mathbb{Q}^N$ satisfies:
$$\dfrac{d\mathbb{Q}^N}{d\mathbb{Q}} = \dfrac{N_T / N_0}{S^0_T / S^0_0} \cdot \text{normalization}.$$

More precisely, if $\mathbb{Q}^0$ is EMM with numéraire $S^0$, and $N$ is any positive traded asset, then under $\mathbb{Q}^N$ defined by $d\mathbb{Q}^N/d\mathbb{Q}^0 = N_T S^0_0 / (N_0 S^0_T)$, the processes $V_t / N_t$ are $\mathbb{Q}^N$-martingales for every traded $V$.

### Example: forward measure

Take $N = P(t, T)$ (the $T$-maturity zero-coupon bond). Then under the $T$-forward measure $\mathbb{Q}^T$:
$$V_0(X) = P(0, T) \cdot \mathbb{E}_{\mathbb{Q}^T}[X].$$

This is convenient because the pricing factor is outside the expectation. Used heavily in interest-rate derivatives (Module 5.4).

### Example: stock-as-numéraire

For a stock $S$ paying continuous dividend $q$, define the "share measure" $\mathbb{Q}^S$ where $S$ is the numéraire. Under $\mathbb{Q}^S$, the process $e^{qt} \cdot (\text{bond}/S_t)$ is a martingale, equivalently $S_t/N_t$ for any traded $N$ is.

Used in Black-Scholes for computing $\mathbb{P}(S_T > K)$ under different measures — leads to the "BS formula split": $C = S_0 \mathbb{Q}^S(S_T > K) - K e^{-rT} \mathbb{Q}^T(S_T > K) = S_0 N(d_1) - K e^{-rT} N(d_2)$.

---

## 5.1.6 Incomplete Markets and Price Bounds

### Example: trinomial model

$\Omega = \{u, m, d\}$, single stock. EMM set: $\{q_u, q_m, q_d\}$ with $q_u + q_m + q_d = 1$, $q_u u + q_m m + q_d d = 1 + r$, all $> 0$. This is a 1-parameter family.

Claim $X = (X_u, X_m, X_d)$. Range of prices:
$$q_u X_u + q_m X_m + q_d X_d \quad \text{over all feasible } (q_u, q_m, q_d).$$

### Superhedging and subhedging

**Superhedging price:** $\overline V_0(X) = \inf\{x : \exists \phi \text{ admissible with } V_0 = x, V_T \ge X \text{ a.s.}\}$.

**Subhedging price:** $\underline V_0(X) = \sup\{x : \exists \phi \text{ admissible with } V_0 = x, V_T \le X \text{ a.s.}\}$.

**Duality theorem (El Karoui-Quenez 1995):**
$$\overline V_0(X) = \sup_{\mathbb{Q} \in \mathcal{M}} \mathbb{E}_\mathbb{Q}[X/S^0_T], \quad \underline V_0(X) = \inf_{\mathbb{Q} \in \mathcal{M}} \mathbb{E}_\mathbb{Q}[X/S^0_T].$$

Between these are all no-arbitrage prices. In complete markets $\mathcal{M}$ is a singleton and the interval collapses.

### Utility indifference pricing

In incomplete markets, an alternative single-price approach: find the price $x$ such that the investor's expected utility is unchanged by adding the contract at price $x$ and its hedge.

Formally:
$$V(w) = V(w - x ; \text{no contract}) \ \Longleftrightarrow \ x = U^{-1}[V(w + x)] - U^{-1}[V(w)].$$

Linked to exponential utility: for CARA with risk aversion $\alpha$,
$$x = -\dfrac{1}{\alpha} \ln \mathbb{E}_{\mathbb{Q}^*}[e^{-\alpha X}],$$
where $\mathbb{Q}^*$ is the **minimal entropy martingale measure**. Used in insurance, energy markets, illiquid derivatives.

### Good-deal bounds

A narrower range than superhedging: exclude EMMs that generate implausibly high Sharpe ratios (Cochrane-Saa-Requejo 2000). If $\text{std}(M)/\mathbb{E}[M] \le h$ for a "good-deal" bound $h$, the price interval is
$$\left[\inf_{\mathbb{Q}: \text{HJ} \le h} \mathbb{E}_\mathbb{Q}[X/S^0_T], \sup_{\mathbb{Q}: \text{HJ} \le h} \mathbb{E}_\mathbb{Q}[X/S^0_T]\right].$$

Used to price illiquid or exotic instruments with theoretical discipline.

---

## 5.1.7 Python: Illustrations

```python
import numpy as np

# -----------------------------
# Example 1: One-period binomial
# -----------------------------
S0 = 100.0; u = 1.1; d = 0.9; r = 0.01
q = ((1+r) - d) / (u - d)
print(f"Risk-neutral q = {q:.4f}")
assert 0 < q < 1, "No-arbitrage requires d < 1+r < u"

# Price European call with strike 100
K = 100.0
X_u = max(S0*u - K, 0)
X_d = max(S0*d - K, 0)
C = (q * X_u + (1-q) * X_d) / (1+r)
print(f"European call price (binomial) = {C:.4f}")

# Replicating portfolio
delta = (X_u - X_d) / (S0 * (u - d))
B     = (X_u - delta * S0 * u) / (1+r)
print(f"Delta = {delta:.4f}, Bond = {B:.4f}")
print(f"Portfolio value = {delta*S0 + B:.4f} (matches {C:.4f}?)")

# -----------------------------
# Example 2: Multi-period binomial (CRR)
# -----------------------------
def binomial_european(S0, K, r, sigma, T, n, option='call'):
    dt = T/n
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1/u
    q  = (np.exp(r*dt) - d) / (u - d)
    # Terminal payoffs
    S_T = S0 * u**np.arange(n, -1, -1) * d**np.arange(0, n+1)
    if option == 'call':
        V = np.maximum(S_T - K, 0)
    else:
        V = np.maximum(K - S_T, 0)
    # Backward induction
    for step in range(n-1, -1, -1):
        V = np.exp(-r*dt) * (q * V[:-1] + (1-q) * V[1:])
    return V[0]

# Converges to Black-Scholes
from math import log, sqrt, exp
from scipy.stats import norm
def bs_call(S, K, r, sigma, T):
    d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

for n in [10, 100, 1000, 10000]:
    C_bin = binomial_european(100, 100, 0.05, 0.2, 1.0, n)
    C_bs  = bs_call(100, 100, 0.05, 0.2, 1.0)
    print(f"n={n:5d}: binomial = {C_bin:.6f}, BS = {C_bs:.6f}, diff = {C_bin-C_bs:.6f}")

# -----------------------------
# Example 3: Trinomial (incomplete)
# -----------------------------
# States: u=1.2, m=1.0, d=0.85; rate=0; show range of prices
u, m, d, rf = 1.2, 1.0, 0.85, 0.0
# Claim: e.g., digital paying 1 if S_T >= 1
X = np.array([1.0, 1.0, 0.0])  # (u, m, d)
# EMM set: q_u + q_m + q_d = 1, q_u*u + q_m*m + q_d*d = 1+rf, q>0
# Parametrize: q_u from feasible range
import scipy.optimize as opt
def price_range(q_u):
    # From constraints: q_d = (1 - q_u - q_m), and
    # q_u*u + q_m*m + q_d*d = 1+rf
    # Substitute q_d: q_u*u + q_m*m + (1 - q_u - q_m)*d = 1+rf
    # q_u*(u-d) + q_m*(m-d) + d = 1+rf
    # q_m = (1+rf - d - q_u*(u-d))/(m-d)
    q_m = (1 + rf - d - q_u*(u-d)) / (m-d)
    q_d = 1 - q_u - q_m
    if q_u>0 and q_m>0 and q_d>0:
        return q_u*X[0] + q_m*X[1] + q_d*X[2]
    return None

prices = [price_range(q) for q in np.linspace(0.01, 0.99, 101)]
prices = [p for p in prices if p is not None]
print(f"\nTrinomial digital: price range = [{min(prices):.4f}, {max(prices):.4f}]")
print(f"(Incomplete market -- no unique price)")

# -----------------------------
# Example 4: Girsanov SDF simulation
# -----------------------------
N = 10000
T = 1.0; dt = 0.01; n_steps = int(T/dt)
mu, r, sigma = 0.10, 0.03, 0.20
theta = (mu - r) / sigma

S = np.full(N, 100.0); logZ = np.zeros(N)
for _ in range(n_steps):
    dB = np.sqrt(dt) * np.random.randn(N)
    S *= np.exp((mu - sigma**2/2)*dt + sigma*dB)
    logZ += -theta * dB - 0.5*theta**2*dt

Z = np.exp(logZ)
M = np.exp(-r*T) * Z

# Check: E[M * S_T] should equal S_0
price_under_P = np.mean(M * S)
print(f"\nSDF check: E_P[M*S_T] = {price_under_P:.4f} (should be {100:.4f})")
# Equivalently E_Q[e^{-rT} S_T] = S_0
```

---

## 5.1.8 [QUANT APPLICATION]

1. **Derivative pricing.** Every pricing formula in quantitative finance is an instance of $V_0 = \mathbb{E}_\mathbb{Q}[\text{discounted payoff}]$. Black-Scholes, Asians, barriers, CDS, variance swaps — all are discounted $\mathbb{Q}$-expectations.

2. **Model validation.** If a model doesn't admit an EMM, it allows arbitrage and is useless. For instance, modeling stock prices with positive drift and no Brownian noise (pure trend) admits no EMM.

3. **Arbitrage detection.** Practitioners scan for violations: put-call parity, forward-bond spread, triangular currency arbitrage. Any detection means the model is wrong or the opportunity is real (and quickly trade-able).

4. **Calibration.** Pricing models are calibrated to market prices of liquid instruments, which fixes the EMM. Exotic pricing then uses the same EMM.

5. **Risk management / real-measure projections.** For capital and VaR, we use $\mathbb{P}$ (physical), not $\mathbb{Q}$. Converting between the two requires knowledge of the market price of risk.

6. **Complete vs incomplete models.** Black-Scholes and HJM (one-factor) are complete. Stochastic volatility models (Heston) are incomplete (two sources of randomness, one traded asset + one vol index). Markets for VIX futures / options are attempts to complete the vol market.

7. **Benchmark-pricing framework.** Platen's (2006) benchmark approach uses the **growth-optimal portfolio** as numéraire; under this "real-world" pricing there's no EMM change needed, and prices are physical-measure expectations.

---

## 5.1.9 Summary

- **Arbitrage** = positive-probability free lunch. **NFLVR** = its continuous-time refinement.
- **First FTAP**: NFLVR ⇔ existence of equivalent martingale measure. Rigorously proven by Delbaen-Schachermayer (1994, 1998).
- **Second FTAP**: Completeness ⇔ uniqueness of EMM. In complete markets every contingent claim has a unique price.
- **Stochastic Discount Factor**: $M_t = Z_t / S^0_t$; pricing rule $V_0 = \mathbb{E}_\mathbb{P}[M_T X]$.
- **Change of numéraire**: different numéraires give different EMMs; useful for forward-measure pricing, stock-as-numéraire decomposition of BS formula.
- **Incomplete markets**: price interval = superhedging bounds = supremum/infimum over all EMMs.

### Forward pointers

- **Module 5.2**: Concrete application — Black-Scholes formula from five distinct perspectives, Greeks, implied volatility, Dupire local volatility.
- **Module 5.3**: Exotic options; path-dependent payoffs; barrier/Asian/lookback; static replication of variance.
- **Module 5.4**: Interest-rate models; forward measure; HJM framework; LIBOR market model.
- **Module 5.5**: Credit risk; structural (Merton) and reduced-form (Duffie) models; CDS; copulas for CDOs.

---

## Exercises

### Tier 1 (★)

1. Prove that in the one-period binomial model, no-arbitrage is equivalent to $d < 1+r < u$.
2. Compute the arbitrage-free price of a European put on a stock with $S_0 = 100, u = 1.2, d = 0.8, r = 5\%, K = 100$.
3. In the trinomial model with $u = 1.2, m = 1.05, d = 0.9, r = 0$, find the range of EMMs.
4. Show that put-call parity $C - P = S_0 - K e^{-rT}$ follows from risk-neutral pricing.
5. Verify that forward price $F = S_0 e^{rT}$ is the unique arbitrage-free price for a forward contract.
6. Derive the stochastic discount factor for a two-period binomial model.
7. For Black-Scholes, verify $M_t / M_s = \exp(-r(t-s) - \theta (B_t - B_s) - \tfrac{1}{2}\theta^2(t-s))$.
8. Show that $\mathbb{E}_\mathbb{P}[M_T] = e^{-rT}$.

### Tier 2 (★★)

9. Prove Harrison-Pliska first FTAP for the $n$-period binomial with finite $\Omega$.
10. Prove the second FTAP in the discrete finite-$\Omega$ setting via Hahn-Banach.
11. Show that in a complete market, superhedging price = subhedging price = unique arbitrage-free price.
12. Derive the Hansen-Jagannathan bound.
13. Show that in a CRRA representative-agent model, the SDF volatility equals $\gamma \cdot \text{vol}(\Delta c)$ (to first order).
14. Prove the change-of-numéraire formula: if $\mathbb{Q}^0$ is EMM with numéraire $S^0$ and $N$ is any positive traded asset, then $d\mathbb{Q}^N/d\mathbb{Q}^0 \propto N_T/S^0_T$.
15. Derive the Black-Scholes formula using $S$-as-numéraire and $P(t,T)$-as-numéraire separately, and verify they give the same answer.
16. Show that the discounted wealth process under any admissible strategy is a $\mathbb{Q}$-local-martingale.
17. In an incomplete market with two risky assets driven by two Brownians, show that if only one is traded then the market is incomplete.
18. Construct an example in continuous time where NA holds but NFLVR fails (e.g., doubling strategies in BS setting).

### Tier 3 (★★★)

19. Prove the Delbaen-Schachermayer (1994) FTAP for locally bounded semi-martingales. (Requires weak-$*$ closure arguments and Kreps-Yan separation.)
20. Prove the second FTAP for continuous-time complete markets using martingale representation theorem.
21. Investigate the Föllmer-Schweizer decomposition for incomplete markets: minimal martingale measure, local risk minimization.
22. Derive utility indifference pricing for exponential utility, showing it equals $(-1/\alpha) \ln \mathbb{E}_{\mathbb{Q}^*}[e^{-\alpha X}]$ with $\mathbb{Q}^*$ the minimal entropy measure.
23. Study the "bubble" phenomenon: construct a price process that is a strict local martingale under $\mathbb{Q}$ but not a true martingale; show this leads to failure of put-call parity.
24. Derive good-deal bounds (Cochrane-Saa-Requejo) for an incomplete stochastic-volatility model; compute numerically.
25. Prove Platen's benchmark approach: if $S^*$ is the growth-optimal portfolio, then $V_t/S^*_t$ is a $\mathbb{P}$-supermartingale for every wealth process $V$.
26. Investigate model uncertainty pricing: given a set of plausible models $\mathcal{P}$, the robust price is $\sup_{\mathbb{P} \in \mathcal{P}} \sup_{\mathbb{Q} \in \mathcal{M}(\mathbb{P})} \mathbb{E}_\mathbb{Q}[X/S^0_T]$. Compute for a simple parametric uncertainty.

---

*Next module:* Black-Scholes from five angles — replication PDE, risk-neutral expectation, CAPM/SDF, Merton capital-structure, Girsanov change-of-measure. Greeks, implied vol, Dupire, and the smile.

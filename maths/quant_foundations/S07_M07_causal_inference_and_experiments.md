# Subject 7, Module 7: Causal Inference and Experiments

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"Correlation is not causation — but correlation plus structure, design, or both, often is."* — paraphrasing Joshua Angrist

---

## 7.7.0 Where We Are — Subject 7 Capstone

This is the final module of Subject 7 and one of the most pragmatically consequential in the curriculum: it concerns how to *draw causal conclusions from data*, which is what every quant researcher ultimately wants. In earlier modules we built estimators (OLS, ridge, random forests, GARCH, factor models). None of them, by themselves, tells us what would happen if we intervened. That requires additional structure.

This module organizes that structure.

### Prerequisites

- **Module 7.1** (Linear Regression): OLS, HAC errors, high-dimensional regression.
- **Module 7.2** (Classification): logistic regression for propensity scores.
- **Module 7.3** (Trees and Boosting): causal forests.
- **Module 2.7** (Conditional Expectation): foundational for the potential-outcomes framework.
- **Module 0.5** (Optimization): for constrained estimation (e.g., synthetic control weights).

### Plan

1. Potential outcomes and the fundamental problem (§7.7.1).
2. Randomized experiments and inference (§7.7.2).
3. Selection on observables: matching, IPW, doubly robust (§7.7.3).
4. Instrumental variables and LATE (§7.7.4).
5. Difference-in-differences and two-way fixed effects (§7.7.5).
6. Regression discontinuity (§7.7.6).
7. Synthetic control (§7.7.7).
8. Causal ML: double/debiased ML, causal forests, meta-learners (§7.7.8).
9. Causal discovery and graphical models (§7.7.9).
10. Practical quant applications: trade-execution A/B tests, event studies, signal attribution (§7.7.10).
11. Python implementations (§7.7.11).
12. Exercises (§7.7.12).
13. Closing Subject 7.

---

## 7.7.1 Potential Outcomes: The Rubin Framework

For a unit $i$ with binary treatment $D_i \in \{0,1\}$, define potential outcomes $Y_i(1)$ and $Y_i(0)$: the outcomes that *would be realized* under treatment and control respectively. The **observed outcome** is
$$Y_i = D_i Y_i(1) + (1-D_i) Y_i(0).$$

### Individual and average treatment effects

- **Individual treatment effect**: $\tau_i = Y_i(1) - Y_i(0)$ — unobservable.
- **Average treatment effect (ATE)**: $\tau = \mathbb{E}[Y_i(1) - Y_i(0)]$.
- **Average treatment effect on the treated (ATT)**: $\tau_{\text{ATT}} = \mathbb{E}[Y_i(1) - Y_i(0) \mid D_i=1]$.
- **Conditional average treatment effect (CATE)**: $\tau(x) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = x]$.

### The fundamental problem

We only ever see $Y_i(D_i)$, never both potential outcomes for the same unit. Causal inference is the study of how to impute the missing counterfactual.

### Identifying assumptions

The common identifying assumptions are:
- **SUTVA** (Stable Unit Treatment Value): no interference across units; a single version of each treatment.
- **Unconfoundedness / Ignorability**: $(Y(1), Y(0)) \perp D \mid X$.
- **Overlap**: $0 < \mathbb{P}(D=1 \mid X) < 1$.

Under unconfoundedness and overlap, the ATE is identified:
$$\tau = \mathbb{E}[\mathbb{E}[Y \mid D=1, X] - \mathbb{E}[Y \mid D=0, X]].$$
This is the fundamental moment condition behind all observational causal estimators in this module.

---

## 7.7.2 Randomized Experiments

### Random assignment

When $D_i$ is assigned by a random mechanism, $D \perp (Y(1), Y(0))$ automatically. The difference in sample means
$$\hat\tau = \bar Y_{D=1} - \bar Y_{D=0}$$
is an unbiased estimator of the ATE, with variance
$$\mathrm{Var}(\hat\tau) = \frac{\sigma^2_1}{n_1} + \frac{\sigma^2_0}{n_0}.$$

### Power and sample size

For detecting $\tau$ with power $1-\beta$ and significance $\alpha$, the required sample size per arm is approximately
$$n \approx \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{\tau^2}.$$
Quant experiments (e.g., execution algorithms, slippage tests) often run on *orders* or *child orders*, and power is limited by variance of trade-level outcomes (which is often large relative to mean effect). Stratification, covariate adjustment, and sequential testing are essential practical tools.

### Covariate adjustment

Lin (2013) studied post-experiment OLS with treatment, covariates, and interactions:
$$Y_i = \alpha + \tau D_i + X_i^\top \beta + D_i X_i^\top \gamma + u_i.$$
This is unbiased and usually reduces variance versus simple differencing. Variance reduction scales with the $R^2$ of $X$ on $Y$.

### CUPED (Deng–Xu–Kohavi–Walker, 2013)

Popular in tech and quant A/B testing. With a pre-treatment covariate $X$ (correlated with $Y$), use
$$\tilde Y_i = Y_i - \theta X_i, \qquad \theta = \mathrm{Cov}(Y, X)/\mathrm{Var}(X).$$
Then $\mathbb{E}[\tilde Y(1) - \tilde Y(0)] = \tau$ (as $X$ is pre-treatment and unaffected by $D$), and $\mathrm{Var}(\tilde Y) = (1 - \rho^2_{X,Y})\mathrm{Var}(Y)$. Reduces experiment run-time substantially for correlated pre-treatment features.

### Sequential testing and anytime-valid inference

Always-valid $p$-values (Johari–Koomen–Pekelis–Walsh 2015, Howard–Ramdas 2019) let an experimenter continuously monitor the test without $p$-hacking. Mixture martingales and e-processes replace fixed-$n$ $p$-values. Ramdas et al. (2023) provides a comprehensive modern reference.

---

## 7.7.3 Selection on Observables

### Regression adjustment

Under ignorability with $X$ high-dimensional, the naive approach $Y = \tau D + X^\top \beta + u$ requires a correctly specified conditional mean. Lasso or ridge can handle high-$p$, but regularization bias contaminates the estimate of $\tau$ unless we use methods like orthogonalization (§7.7.8).

### Matching

Match each treated unit to one (or more) control units with similar $X$:
- Nearest neighbor (Mahalanobis or Euclidean on $X$).
- Optimal matching (minimize total distance via linear programming).
- Coarsened exact matching (Iacus–King–Porro 2012).

Abadie–Imbens (2006) established large-sample properties and showed that simple matching estimators are $\sqrt n$-inconsistent in high dimensions — their bias term is $O(n^{-1/p})$.

### Inverse propensity weighting (IPW)

Define the propensity score $e(x) = \mathbb{P}(D=1 \mid X=x)$. The Horvitz–Thompson estimator:
$$\hat\tau_{\text{IPW}} = \frac{1}{n}\sum_i \left[\frac{D_i Y_i}{\hat e(X_i)} - \frac{(1-D_i) Y_i}{1 - \hat e(X_i)}\right].$$
Under correctly specified $\hat e$, unbiased. Can be unstable when propensities approach 0 or 1 — "trimming" (dropping units with extreme $\hat e$) is standard.

### Doubly robust (augmented IPW, AIPW)

$$\hat\tau_{\text{AIPW}} = \frac{1}{n}\sum_i\left[\frac{D_i(Y_i - \hat\mu_1(X_i))}{\hat e(X_i)} + \hat\mu_1(X_i) - \frac{(1-D_i)(Y_i - \hat\mu_0(X_i))}{1 - \hat e(X_i)} - \hat\mu_0(X_i)\right].$$
Consistent if *either* the outcome model $\hat\mu_d(x)$ *or* the propensity $\hat e(x)$ is correct. This double-robustness is the gateway to modern machine-learning-based causal estimation.

---

## 7.7.4 Instrumental Variables

### Setup

Suppose $D$ is endogenous: $\mathrm{Cov}(D, u) \neq 0$ in the outcome model $Y = \tau D + X^\top\beta + u$. An **instrument** $Z$ satisfies:
1. **Relevance**: $\mathrm{Cov}(Z, D \mid X) \neq 0$.
2. **Exclusion**: $Z$ affects $Y$ only through $D$ (after conditioning on $X$).
3. **Exogeneity**: $\mathrm{Cov}(Z, u \mid X) = 0$.

### Two-stage least squares (2SLS)

**Stage 1**: $D_i = \pi_0 + Z_i^\top \pi_1 + X_i^\top \pi_2 + v_i$; get $\hat D_i$.
**Stage 2**: $Y_i = \alpha + \tau \hat D_i + X_i^\top \beta + w_i$; OLS estimate of $\tau$.

### LATE (Imbens–Angrist 1994)

With heterogeneous effects and a binary instrument, 2SLS estimates the **Local Average Treatment Effect**:
$$\mathrm{LATE} = \mathbb{E}[Y(1) - Y(0) \mid \text{compliers}],$$
where a complier is a unit whose treatment status is moved by the instrument ($D(1) > D(0)$). LATE is generally not the ATE unless effects are homogeneous or the instrument is "monotonic-everywhere."

### Weak instruments

When $\pi_1$ is close to zero, 2SLS suffers severe finite-sample bias. Stock–Yogo (2005) recommend the F-statistic on the excluded instruments exceeding 10 as a rule of thumb. Anderson–Rubin confidence sets and conditional likelihood ratio (Moreira 2003) give valid inference under weak instruments.

### Many weak instruments

Hausman–Newey–Woutersen (2012), Hansen–Hausman–Newey: limited-information maximum likelihood (LIML) is less biased than 2SLS with many instruments. Jackknife IV (JIVE) and FULLER are also robust.

---

## 7.7.5 Difference-in-Differences

### Classical 2×2 DiD

Two groups (treated / control), two periods (pre / post):
$$\hat\tau_{\text{DiD}} = (\bar Y^{\text{treat}}_{\text{post}} - \bar Y^{\text{treat}}_{\text{pre}}) - (\bar Y^{\text{control}}_{\text{post}} - \bar Y^{\text{control}}_{\text{pre}}).$$

Identification: **parallel trends** — absent treatment, the two groups would have moved identically over time.

### Two-way fixed effects (TWFE)

$$Y_{it} = \alpha_i + \lambda_t + \tau D_{it} + u_{it}.$$
Under parallel trends, OLS on this specification estimates $\tau$ consistently *if treatment timing is uniform*. Goodman-Bacon (2021) decomposed the TWFE estimator when treatment timing varies across units:
$$\hat\tau_{\text{TWFE}} = \sum_{\text{comparisons}} w_k \hat\tau_k,$$
and showed some weights can be negative, producing nonsensical estimates when treatment effects are heterogeneous.

### Modern DiD estimators

- **Callaway–Sant'Anna (2021)**: group-time average treatment effects $\mathrm{ATT}(g,t)$, with aggregation weights.
- **Sun–Abraham (2021)**: interaction-weighted estimator robust to heterogeneous dynamics.
- **de Chaisemartin–D'Haultfoeuille (2020)**: estimator based on units switching treatment status.
- **Borusyak–Jaravel–Spiess (2023)**: imputation approach.

### Event study

Extend TWFE with leads and lags of treatment:
$$Y_{it} = \alpha_i + \lambda_t + \sum_{k=-K}^{K} \delta_k \mathbb{1}\{t - g_i = k\} + u_{it}.$$
Pre-treatment $\hat\delta_k$ ($k<0$) test parallel trends; post-treatment $\hat\delta_k$ ($k\ge 0$) trace dynamic treatment effects.

---

## 7.7.6 Regression Discontinuity

### Sharp RD

Treatment is a deterministic function of a running variable $X$:
$$D_i = \mathbb{1}\{X_i \ge c\}.$$
Under continuity of $\mathbb{E}[Y(0) \mid X]$ and $\mathbb{E}[Y(1) \mid X]$ at $c$, the ATE at the cutoff is
$$\tau(c) = \lim_{x\downarrow c}\mathbb{E}[Y \mid X=x] - \lim_{x\uparrow c}\mathbb{E}[Y \mid X=x].$$

Estimation: local polynomial regression on each side of $c$, with Imbens–Kalyanaraman (2012) or Calonico–Cattaneo–Titiunik (2014) bandwidth selection.

### Fuzzy RD

$\mathbb{P}(D=1 \mid X)$ jumps at $c$ but doesn't go from 0 to 1. Then $\tau(c)$ is the ratio of outcome jump to propensity jump:
$$\tau(c) = \frac{\lim_{x\downarrow c}\mathbb{E}[Y \mid X=x] - \lim_{x\uparrow c}\mathbb{E}[Y \mid X=x]}{\lim_{x\downarrow c}\mathbb{E}[D \mid X=x] - \lim_{x\uparrow c}\mathbb{E}[D \mid X=x]}.$$
Effectively an IV estimator with $Z = \mathbb{1}\{X \ge c\}$.

### Bias-corrected inference

Calonico–Cattaneo–Titiunik (2014) robust confidence intervals correct for the bias of local polynomial estimators. RDHonest (Kolesár–Rothe 2018) provides honest inference under smoothness constraints on the CEF.

### McCrary test

Check for manipulation of the running variable: a density discontinuity at $c$ suggests units sorted across the cutoff (e.g., test scores being rounded up by teachers), invalidating RD.

---

## 7.7.7 Synthetic Control

Abadie–Gardeazabal (2003), Abadie–Diamond–Hainmueller (2010).

### Setup

One treated unit, many control units, panel data. The synthetic control is a weighted average of controls chosen to match the treated unit's *pre-treatment* outcomes (and possibly covariates). Estimate treatment effects by the gap between treated and synthetic post-treatment.

### Weights

$$w^* = \arg\min_{w \ge 0, \mathbf 1^\top w=1} \|X_1 - X_0 w\|_V^2,$$
where $X_1$ is the treated unit's pre-treatment features, $X_0$ the control units' features, and $V$ a diagonal weight chosen to minimize MSPE or informed by prior knowledge.

### Inference via placebos

With only one treated unit, classical asymptotics don't apply. Instead, run the synthetic-control procedure placing each *control* unit in the treatment role, and compare the treated unit's gap to the distribution of placebo gaps.

### Extensions

- **Generalized synthetic control** (Xu 2017): uses a factor model on pre-treatment outcomes to construct the counterfactual; accommodates multiple treated units.
- **Synthetic DiD** (Arkhangelsky et al. 2021): combines synthetic control weights with DiD's double-differencing, delivering efficiency gains.
- **Matrix completion** (Athey et al. 2021): low-rank + noise model of the panel; impute missing counterfactuals via nuclear-norm regularization.

---

## 7.7.8 Causal Machine Learning

### Double/Debiased ML (Chernozhukov et al. 2018)

Goal: estimate the treatment effect in a partially linear model
$$Y = \tau D + g(X) + u, \qquad D = m(X) + v, \qquad \mathbb{E}[u \mid X, D] = 0.$$
With high-dimensional $X$ and ML estimates of $g$ and $m$:

**Step 1.** Split sample into $K$ folds.
**Step 2.** On each fold $k$, fit $\hat g_{-k}$ and $\hat m_{-k}$ on the complement.
**Step 3.** Compute orthogonalized residuals on fold $k$:
$$\tilde Y_i = Y_i - \hat g_{-k}(X_i), \qquad \tilde D_i = D_i - \hat m_{-k}(X_i).$$
**Step 4.** OLS of $\tilde Y$ on $\tilde D$ across all folds:
$$\hat\tau = \frac{\sum_i \tilde D_i \tilde Y_i}{\sum_i \tilde D_i^2}.$$

### Theorem 7.7.1 (Neyman orthogonality).

*The moment condition $\psi(W, \tau; g, m) = (Y - g(X) - \tau(D - m(X)))(D - m(X))$ is Neyman-orthogonal at the true $(g_0, m_0)$: its Gâteaux derivatives with respect to $g$ and $m$ vanish. Hence, plugging in $\hat g, \hat m$ that converge at rate $o_p(n^{-1/4})$ yields $\sqrt n$-consistent and asymptotically normal $\hat\tau$.*

This is profound: it means we can use *any* ML method — random forests, neural nets, lasso — for the nuisance estimates, and get classical inference on $\tau$, provided nuisance convergence beats $n^{-1/4}$.

### Causal forests (Wager–Athey 2018)

Fit a forest where the splitting criterion maximizes *heterogeneity of the treatment effect* rather than the outcome. Estimated CATE $\hat\tau(x)$ is an adaptive nearest-neighbor average, with honest splitting (separating the subsample used to pick splits from the one used to estimate means) guaranteeing pointwise confidence intervals.

Consistency rate: $\sqrt n$ at any fixed $x$, under smoothness of $\tau(\cdot)$ and honest tuning.

### Meta-learners (Künzel et al. 2019)

- **S-learner**: fit $\hat\mu(X, D)$ on pooled data; $\hat\tau(x) = \hat\mu(x,1)-\hat\mu(x,0)$.
- **T-learner**: fit $\hat\mu_1, \hat\mu_0$ separately by treatment; $\hat\tau(x) = \hat\mu_1(x)-\hat\mu_0(x)$.
- **X-learner**: impute missing counterfactuals, then fit $\hat\tau$ as a regression.
- **R-learner**: uses residualization à la double ML.
- **DR-learner**: plug doubly-robust scores into a second-stage learner.

### BART and probabilistic alternatives

Hahn–Murray–Carvalho (2020) **BCF** (Bayesian Causal Forests) use a separate prior on treatment-effect heterogeneity vs. confounding, addressing regularization-induced confounding (RIC) that plagues naive BART-on-pooled-data.

---

## 7.7.9 Causal Discovery and Graphical Models

### DAGs and $d$-separation

A directed acyclic graph (DAG) $G = (V, E)$ encodes structural relations. A path between two nodes is **blocked** by a set $Z$ if it contains either a chain or fork with a middle node in $Z$, or a collider with no descendant in $Z$. Nodes $A, B$ are $d$-separated by $Z$ iff *every* path between them is blocked.

Pearl (1988): $X \perp Y \mid Z$ under all distributions consistent with the DAG iff $X$ and $Y$ are $d$-separated by $Z$.

### Backdoor criterion

To identify the causal effect of $T$ on $Y$, find a set $Z$ such that:
1. No node in $Z$ is a descendant of $T$.
2. $Z$ blocks every backdoor path from $T$ to $Y$ (path starting with $\leftarrow$ into $T$).

Then $P(Y \mid \mathrm{do}(T=t)) = \sum_z P(Y \mid T=t, Z=z) P(Z=z)$.

### PC and FCI algorithms

- **PC** (Peter Spirtes–Clark Glymour): constraint-based discovery assuming no latent confounders; tests conditional independences to orient edges.
- **FCI** (Fast Causal Inference): handles latent confounders and selection bias.
- **GES** (greedy equivalence search): score-based, uses BIC.
- **LiNGAM** (Shimizu et al. 2006): linear non-Gaussian acyclic models identified via ICA.

### Do-calculus

A formal system (Pearl 1995) for manipulating expressions involving the $\mathrm{do}$ operator, resolving which causal effects are identifiable from observational data plus DAG.

---

## 7.7.10 Quant-Specific Causal Analyses

### Trade-execution A/B tests

Splitting orders into $A$ (new algorithm) and $B$ (baseline) is the gold standard for execution-quality evaluation. Key considerations:
- **Stratify by order size and volatility** to reduce variance.
- **Address leakage**: if the two algorithms trade the same symbol simultaneously, their activity interferes — consider time-of-day split or cross-venue isolation.
- **Long-horizon measurement**: arrival slippage and impact decay must be measured with consistent windows.
- Use CUPED with pre-trade signals (arrival price, expected spread, ADV%).

### Event studies for news/earnings

Classical event-study methodology: compute abnormal returns (AR) around an event, cumulate (CAR), and test $\bar{\mathrm{CAR}} = 0$. Issues: event clustering (positive cross-sectional correlation in event time), volatility clustering.

Kolari–Pynnonen (2010) and Boehmer–Masumeci–Poulsen (1991) adjust standard errors for cross-correlation and time-series properties.

### Signal attribution

When a new signal is added to a strategy, the incremental contribution to PnL can be attributed via residualization against the existing signal. Frames this as a partial $R^2$ or, better, a double-ML estimate with the existing model as the nuisance.

### Market-impact causal identification

Firms' own trades move prices, creating simultaneity. Kyle's $\lambda$ estimated by simple regression of price change on signed volume over-states impact because of reverse causality. IV strategies use *exogenous* flow proxies (e.g., index-inclusion additions, mechanical ETF rebalances) to identify genuine impact.

### Policy evaluation (SEC rule changes, tick-size pilots)

DiD on venues before/after a rule change with unaffected instruments as controls. SEC Tick Size Pilot (2016–2018) has been studied with DiD and RD designs.

### Post-trade causal monitoring

For algorithmic trading desks, maintain a causal monitoring stack that continuously estimates treatment effects of parameter changes against a stable baseline. Sequential estimators with anytime validity (mSPRT, always-valid CIs) allow continuous deployment without multiplicity concerns.

---

## 7.7.11 Python Implementations

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
import statsmodels.api as sm

rng = np.random.default_rng(42)

# ----- Simulate confounded data -----
def simulate(n, p, tau=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # confounder effect on both D and Y
    e = 1.0 / (1.0 + np.exp(-X[:, 0] - 0.5*X[:, 1]))
    D = rng.binomial(1, e)
    Y = tau*D + X[:, 0] + 0.5*X[:, 1]**2 + rng.standard_normal(n)
    return X, D, Y

# ----- Naive OLS (biased) -----
def naive_ols(X, D, Y):
    Z = np.column_stack([D, X])
    return sm.OLS(Y, sm.add_constant(Z)).fit()

# ----- IPW -----
def ipw(X, D, Y):
    clf = LogisticRegression(max_iter=1000).fit(X, D)
    e = clf.predict_proba(X)[:, 1].clip(0.05, 0.95)
    return np.mean(D*Y/e - (1-D)*Y/(1-e))

# ----- Doubly robust (AIPW) -----
def aipw(X, D, Y):
    treated = (D == 1)
    mu1 = LinearRegression().fit(X[treated], Y[treated]).predict(X)
    mu0 = LinearRegression().fit(X[~treated], Y[~treated]).predict(X)
    clf = LogisticRegression(max_iter=1000).fit(X, D)
    e = clf.predict_proba(X)[:, 1].clip(0.05, 0.95)
    return np.mean(mu1 - mu0 + D*(Y-mu1)/e - (1-D)*(Y-mu0)/(1-e))

# ----- Double/debiased ML -----
def double_ml(X, D, Y, K=5, rf_params=None):
    rf_params = rf_params or dict(n_estimators=200, min_samples_leaf=5, n_jobs=-1)
    n = len(Y)
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    res_Y = np.zeros(n)
    res_D = np.zeros(n)
    for train, test in kf.split(X):
        g_hat = RandomForestRegressor(**rf_params).fit(X[train], Y[train])
        m_hat = RandomForestClassifier(**rf_params).fit(X[train], D[train])
        res_Y[test] = Y[test] - g_hat.predict(X[test])
        res_D[test] = D[test] - m_hat.predict_proba(X[test])[:, 1]
    tau_hat = np.sum(res_D * res_Y) / np.sum(res_D**2)
    # Standard error via influence function
    psi = (res_Y - tau_hat*res_D) * res_D / np.mean(res_D**2)
    se = np.std(psi, ddof=1) / np.sqrt(n)
    return tau_hat, se

# ----- IV via 2SLS -----
def two_sls(Y, D, Z, X):
    # Stage 1: D on Z, X
    X1 = np.column_stack([Z, X])
    m1 = LinearRegression().fit(X1, D)
    D_hat = m1.predict(X1)
    # Stage 2: Y on D_hat, X
    X2 = np.column_stack([D_hat, X])
    m2 = sm.OLS(Y, sm.add_constant(X2)).fit()
    return m2  # first coefficient after const is tau

# ----- Regression discontinuity (sharp, local linear) -----
def sharp_rd(X, Y, c, h=None):
    from scipy.stats import gaussian_kde
    if h is None:
        h = 1.0 * np.std(X) * len(X)**(-0.2)  # Silverman-ish
    mask = np.abs(X - c) <= h
    Xh, Yh = X[mask], Y[mask]
    Dh = (Xh >= c).astype(int)
    # Local linear regression on each side
    design = np.column_stack([Dh, Xh - c, Dh*(Xh - c)])
    m = sm.OLS(Yh, sm.add_constant(design)).fit()
    return m  # coefficient on Dh is the RD estimate at c

# ----- DiD (2x2) -----
def did_2x2(df, unit_col, time_col, treat_col, outcome_col, t0):
    g_treat = df[df[treat_col]==1].groupby(time_col)[outcome_col].mean()
    g_ctrl = df[df[treat_col]==0].groupby(time_col)[outcome_col].mean()
    tau = (g_treat.loc[g_treat.index >= t0].mean() - g_treat.loc[g_treat.index < t0].mean()) \
        - (g_ctrl.loc[g_ctrl.index >= t0].mean() - g_ctrl.loc[g_ctrl.index < t0].mean())
    return tau

# ----- Synthetic control -----
def synthetic_control(y_treated_pre, X_controls_pre):
    """
    y_treated_pre: (T_pre,) pre-treatment outcome for treated
    X_controls_pre: (T_pre, J) pre-treatment outcomes for controls
    Returns optimal weights w, minimizing ||y - X w||^2 s.t. w>=0, sum(w)=1.
    """
    from scipy.optimize import minimize
    J = X_controls_pre.shape[1]
    obj = lambda w: np.sum((y_treated_pre - X_controls_pre @ w)**2)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bnds = [(0, 1)]*J
    w0 = np.ones(J)/J
    res = minimize(obj, w0, bounds=bnds, constraints=cons)
    return res.x

# ---- Example run ----
if __name__ == "__main__":
    X, D, Y = simulate(n=2000, p=10, tau=1.0, seed=1)
    print("Naive OLS tau:", naive_ols(X, D, Y).params[1])
    print("IPW tau:", ipw(X, D, Y))
    print("AIPW tau:", aipw(X, D, Y))
    tau_dml, se_dml = double_ml(X, D, Y)
    print(f"Double ML tau: {tau_dml:.3f} (SE {se_dml:.3f})")
```

On confounded simulations like this, naive OLS is materially biased while IPW, AIPW, and double ML cluster around the truth. Double ML typically has the smallest MSE once $p$ is non-trivial.

---

## 7.7.12 Exercises

### ★ Exercises

1. In a randomized experiment with $n_1 = n_0 = 500$ and $\sigma_1 = \sigma_0 = 1$, what is the minimum detectable $\tau$ at 80% power, $\alpha=0.05$?
2. Show that CUPED's adjusted outcome has the same mean as $Y$ (so the estimated $\tau$ is unchanged) but lower variance when $\rho^2_{X,Y}>0$.
3. Show that under unconfoundedness, $\mathbb{E}[D Y / e(X)] = \mathbb{E}[Y(1)]$ when $e(X)$ is the true propensity.
4. Derive 2SLS as minimizing squared prediction errors of $Y$ projected on $\hat D$.
5. For a sharp RD with running variable $X$ and threshold $c$, show that the ATE at $c$ equals $\lim_{x\downarrow c}\mathbb{E}[Y|X=x] - \lim_{x\uparrow c}\mathbb{E}[Y|X=x]$ under continuity.
6. Write the synthetic-control objective as a quadratic program and verify convexity.

### ★★ Exercises

7. Implement AIPW for a treatment with simulated data; verify double-robustness by deliberately misspecifying one of $\hat e$ or $\hat\mu_d$ and showing unbiasedness is retained.
8. Derive the Goodman-Bacon decomposition of the TWFE estimator for a 3-period, 3-group panel.
9. Verify via simulation that causal-forest honest estimates of CATE are approximately normal with valid pointwise coverage, while non-honest forests overcover or undercover systematically.
10. Show that 2SLS with a single binary instrument estimates LATE (the effect on compliers) under monotonicity.
11. Derive the R-learner loss $\ell(\tau) = \mathbb{E}[(Y - m(X) - \tau(X)(D - e(X)))^2]$ and show that $\tau^*$ minimizes it when $m$ and $e$ are correctly specified.
12. For a synthetic control application, implement placebo tests and compare the treated-unit gap distribution to placebos.

### ★★★ Exercises

13. Prove Theorem 7.7.1 (Neyman orthogonality of the DML moment condition) and the resulting $\sqrt n$ asymptotic normality, stating the precise nuisance rate conditions.
14. Derive the formula for the influence function of AIPW and show it achieves the semiparametric efficiency bound under correctly specified nuisances.
15. For event studies with heterogeneous treatment effects, prove the Callaway–Sant'Anna decomposition into ATT$(g,t)$ and show their proposed weights yield convex combinations (unlike TWFE).
16. Prove identification of $P(Y \mid \mathrm{do}(T))$ via the backdoor criterion in a DAG, using do-calculus rules.
17. Establish consistency of the PC algorithm under the faithfulness assumption: the conditional-independence relations implied by the DAG exactly match those in the true distribution.
18. Derive bias and variance of the sharp RD local-linear estimator, and the Imbens–Kalyanaraman optimal bandwidth $h^* \propto n^{-1/5}$.

---

## Subject 7 Capstone Summary

Across seven modules, Subject 7 has built a rigorous, modern toolkit:

- **7.1 Linear regression and regularization**: from Gauss–Markov to double descent, the workhorse of every quant regression.
- **7.2 Classification and kernel methods**: Bayes risk, SVMs, RKHS — the geometry of discrimination.
- **7.3 Trees, random forests, gradient boosting**: the non-parametric engines of modern production ML.
- **7.4 Time series foundations**: stationarity, ARMA, cointegration, VAR — the temporal backbone.
- **7.5 Volatility modeling**: ARCH/GARCH/realized — the quantitative substrate of risk management and option pricing.
- **7.6 Factor models and cross-sectional regressions**: the equity-research lingua franca, from CAPM through IPCA.
- **7.7 Causal inference and experiments**: the discipline of extracting structural effects from observational and experimental data.

The common thread: *every estimator requires a model of what we can't see*. Regression adds smoothness; trees add local adaptivity; factor models add structure on the cross-section; causal methods add structure on interventions. The quant researcher's craft is choosing the right combination for the problem — and knowing exactly what assumptions are being purchased in the process.

### Forward pointers

Subject 8 (Advanced Applications) will carry this toolkit into:
- **Portfolio optimization under estimation error**: Michaud resampling, robust optimization, Black–Litterman, hierarchical risk parity.
- **Risk management**: VaR, ES, coherent risk measures, backtesting, stress testing.
- **Algorithmic trading design**: combining microstructure, execution, and signal to deploy strategies end-to-end.
- **Alternative data and NLP**: text as a signal, embeddings in production, causal identification with alt data.
- **Large-scale ML systems for quant**: feature stores, reproducibility, MLOps in trading.

*— End of Module 7.7 and Subject 7. Next: Subject 8.*

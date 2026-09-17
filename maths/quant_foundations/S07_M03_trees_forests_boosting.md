# Module 7.3 — Trees, Random Forests, Gradient Boosting

*Subject 7, Module 3. The tree ensembles — the workhorse of applied machine learning in finance.*

---

## Prerequisites

- **Module 7.1 (Regression, regularization)**.
- **Module 7.2 (Classification loss functions)**.
- **Module 2.2 (Conditional expectation)** — trees estimate $\mathbb{E}[Y|X]$ via piecewise constants.
- **Module 6.1 (Monte Carlo)** — bagging uses bootstrap sampling.

---

## 7.3.1 Decision trees (CART)

Classification And Regression Trees (Breiman-Friedman-Olshen-Stone 1984). A decision tree partitions $\mathcal{X}$ into rectangles $R_1, \ldots, R_M$ and predicts a constant in each:
$$
\hat f(x) = \sum_{m=1}^M c_m \mathbb{1}\{x \in R_m\}.
$$

**Greedy growing.** Starting from $R = \mathcal{X}$, at each step pick feature $j$ and threshold $s$ to minimize impurity:

**For regression** (MSE loss):
$$
\min_{j, s} \left[\sum_{x_i \in R_L} (y_i - \bar y_L)^2 + \sum_{x_i \in R_R} (y_i - \bar y_R)^2\right].
$$

**For classification** — use Gini impurity or entropy:
$$
\text{Gini}(R) = \sum_k p_k(1-p_k), \qquad \text{Entropy}(R) = -\sum_k p_k \log p_k.
$$

**Stopping criteria.** Max depth, min samples per leaf, min impurity decrease. Without stopping, the tree overfits to training noise.

**Pruning.** Grow large tree, then collapse subtrees that don't reduce **cost-complexity**:
$$
R_\alpha(T) = \sum_m \sum_{x_i \in R_m} (y_i - c_m)^2 + \alpha |T|.
$$
Pick $\alpha$ via CV.

**Advantages.**
- Handle mixed continuous/categorical features natively.
- No scaling required.
- Interpretable (human-readable rules).
- Capture nonlinearities and interactions automatically.

**Disadvantages.**
- High variance — small data changes alter splits dramatically.
- Axis-aligned splits struggle with rotated decision boundaries.
- Unstable.

Instability motivates ensembles.

---

## 7.3.2 Bagging (Bootstrap Aggregating)

**Breiman (1996).** For $b = 1, \ldots, B$:
1. Sample $n$ points with replacement from training data.
2. Fit tree $\hat f_b$ on bootstrap sample.

**Prediction**: $\hat f_{\text{bag}}(x) = \frac{1}{B}\sum_b \hat f_b(x)$ (regression) or majority vote (classification).

**Variance reduction.** If $\text{Var}(\hat f_b) = \sigma^2$ and $\text{Cov}(\hat f_{b_1}, \hat f_{b_2}) = \rho\sigma^2$:
$$
\text{Var}(\hat f_{\text{bag}}) = \rho\sigma^2 + (1-\rho)\sigma^2/B.
$$
As $B \to \infty$, variance $\to \rho\sigma^2$. Reducing $\rho$ matters: hence random forests.

**Out-of-bag (OOB) error.** Each bootstrap sample omits ~37% of training points. Predict OOB points with trees that didn't see them — gives a CV-like estimate for free.

---

## 7.3.3 Random forests (Breiman 2001)

**Innovation.** At each split, consider only a random subset $m$ of features (out of $p$). Default: $m = \sqrt{p}$ (classification) or $m = p/3$ (regression).

Decorrelates trees → reduces $\rho$ → larger variance reduction.

**Properties.**
- OOB error converges to true generalization error.
- Feature importance via permutation: $\Delta MSE$ when column $j$ is randomly permuted.
- Out-of-the-box SOTA on many tabular tasks. Little hyperparameter tuning needed.

**Feature importance: permutation vs impurity.** Impurity-based importance is biased toward features with many categories; **permutation importance** is unbiased. For high-cardinality features, use permutation importance.

**Proximities.** Trees provide a notion of distance: fraction of trees where two points end in the same leaf. Used for outlier detection.

---

## 7.3.4 AdaBoost (Freund-Schapire 1997)

**First boosting algorithm**. Sequentially train weak learners, reweighting samples that previous learners misclassified.

**AdaBoost.M1 algorithm** (binary classification with $y \in \{-1, +1\}$):

1. Initialize weights $w_i = 1/n$.
2. For $b = 1, \ldots, B$:
   - Train classifier $G_b$ minimizing weighted error $\text{err}_b = \sum w_i \mathbb{1}\{y_i \ne G_b(x_i)\}$.
   - Compute $\alpha_b = \tfrac{1}{2}\log((1-\text{err}_b)/\text{err}_b)$.
   - Update weights: $w_i \leftarrow w_i \exp(-\alpha_b y_i G_b(x_i))$, then normalize.
3. Output $G(x) = \text{sign}(\sum_b \alpha_b G_b(x))$.

**AdaBoost is exponential-loss boosting**. Friedman-Hastie-Tibshirani (2000) showed AdaBoost minimizes the empirical exponential loss $\sum e^{-y_i f(x_i)}$.

**PAC-Bayesian margin analysis** (Schapire-Freund-Bartlett-Lee 1998): AdaBoost continues to generalize after training error reaches 0 because it increases the *margin distribution*.

---

## 7.3.5 Gradient boosting (Friedman 1999)

**Generalize AdaBoost to any differentiable loss.** Gradient boosting minimizes $\sum L(y_i, f(x_i))$ by sequential stagewise fitting.

**Algorithm (for MSE loss):**

1. $f_0 = \arg\min_c \sum L(y_i, c) = \bar y$.
2. For $b = 1, \ldots, B$:
   - Compute pseudo-residuals $r_i = -\partial L(y_i, f)/\partial f |_{f = f_{b-1}(x_i)}$.
   - Fit tree $h_b$ to $\{(x_i, r_i)\}$.
   - Optimal step $\gamma_b = \arg\min \sum L(y_i, f_{b-1}(x_i) + \gamma h_b(x_i))$.
   - Update $f_b = f_{b-1} + \nu \gamma_b h_b$, where $\nu \in (0,1]$ is the **learning rate** (shrinkage).

**Regularization.**
- Small $\nu$ (shrinkage) — typical 0.01 to 0.1.
- Small tree depth — typical 3 to 8.
- Subsample (stochastic gradient boosting, Friedman 2002).
- Early stopping on validation loss.

**Loss functions.**
- **MSE** (regression).
- **Logistic** (binary classification).
- **Multinomial cross-entropy** (multiclass).
- **Huber** (robust regression, robust to outliers).
- **Quantile** loss (for quantile regression).

---

## 7.3.6 XGBoost, LightGBM, CatBoost

**XGBoost (Chen-Guestrin 2016).** 
- Second-order (Newton) boosting: uses both gradient and Hessian.
- Explicit regularization: $\Omega(h) = \gamma T + \tfrac{1}{2}\lambda \|w\|^2$ for tree with $T$ leaves.
- Split finding via approximate algorithm with weighted quantile sketch.
- Column and row sampling, missing-value handling.
- Blazingly fast, well-engineered.

**Second-order gain** for a split:
$$
\mathcal{G} = \tfrac{1}{2}\left[\frac{(\sum_L g_i)^2}{\sum_L h_i + \lambda} + \frac{(\sum_R g_i)^2}{\sum_R h_i + \lambda} - \frac{(\sum g_i)^2}{\sum h_i + \lambda}\right] - \gamma.
$$
Leaf weight: $w^* = -\sum g_i/(\sum h_i + \lambda)$.

**LightGBM (Ke et al. 2017).** 
- Gradient-based one-side sampling (GOSS): focus on samples with large gradients.
- Exclusive feature bundling (EFB): bundle sparse features.
- Histogram-based binning for splits.
- Leaf-wise growth (vs XGBoost's level-wise) — faster convergence but can overfit.

**CatBoost (Prokhorenkova et al. 2018).**
- Native handling of categorical features via ordered target encoding.
- Ordered boosting to avoid target leakage from prior trees.
- Symmetric trees for speed.

All three are used ubiquitously in production finance ML.

---

## 7.3.7 Interpreting tree ensembles: SHAP values

**Shapley value** (game theory). For feature $j$ and prediction $f(x)$:
$$
\phi_j(x) = \sum_{S \subseteq \{1,\ldots,p\} \setminus j} \frac{|S|!(p-|S|-1)!}{p!}\left[f_S(x_S \cup \{x_j\}) - f_S(x_S)\right].
$$
Unique attribution satisfying efficiency, symmetry, dummy, additivity axioms.

**TreeSHAP (Lundberg-Lee 2017)** computes exact SHAP values for tree ensembles in polynomial time, vs. $2^p$ for exact Shapley.

**Usage**: per-sample attribution ("why did the model predict default probability 0.85 for this borrower?"), feature importance ("which features matter globally?"), interaction effects.

**SHAP has become the standard interpretability tool** for production ML models in finance.

---

## 7.3.8 Strengths and limits of tree ensembles in finance

**Strengths.**
- Best out-of-the-box method for tabular data.
- Handle missing data, categorical features, high dimensionality.
- Robust to outliers and feature scaling.
- Fast training and inference.
- Interpretable via feature importance and SHAP.

**Limits.**
- Struggle with smooth dependencies (e.g., option deltas) where NNs or kernels are better.
- No extrapolation: predict constant values outside training support.
- Don't capture temporal/sequential structure as well as RNNs or transformers.
- Can leak target information if categorical features are encoded with target statistics without out-of-fold care.

**For financial time series**: GBMs with careful feature engineering (rolling stats, expanding windows, regime indicators) often beat deep learning on daily returns prediction.

---

## 7.3.9 Python: tree ensembles

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split, cross_val_score

# ============================================================
# 1. Decision tree regression
# ============================================================
X, y = make_regression(n_samples=500, n_features=10, n_informative=5, noise=5.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
print(f"Single tree R²: {dt.score(X_test, y_test):.4f}")

# ============================================================
# 2. Random forest
# ============================================================
rf = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)
print(f"Random forest R²: {rf.score(X_test, y_test):.4f}")
print(f"Feature importances: {np.round(rf.feature_importances_, 3)}")

# ============================================================
# 3. Gradient boosting with early stopping
# ============================================================
gbm = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                 max_depth=4, subsample=0.8, random_state=42)
gbm.fit(X_train, y_train)
print(f"\nGradient boosting R²: {gbm.score(X_test, y_test):.4f}")

# Validation curve
train_scores, test_scores = [], []
for n in [1, 10, 50, 100, 200, 300, 500]:
    gbm = GradientBoostingRegressor(n_estimators=n, learning_rate=0.05,
                                     max_depth=4, random_state=42)
    gbm.fit(X_train, y_train)
    train_scores.append(gbm.score(X_train, y_train))
    test_scores.append(gbm.score(X_test, y_test))
    print(f"  n={n:3d}: train R²={train_scores[-1]:.4f}  test R²={test_scores[-1]:.4f}")

# ============================================================
# 4. Gradient boosting via xgboost (if installed)
# ============================================================
try:
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    params = {'objective': 'reg:squarederror', 'max_depth': 4,
              'learning_rate': 0.05, 'subsample': 0.8, 'reg_lambda': 1.0}
    booster = xgb.train(params, dtrain, num_boost_round=500,
                        evals=[(dtest, 'test')], early_stopping_rounds=20,
                        verbose_eval=False)
    y_pred = booster.predict(dtest)
    from sklearn.metrics import r2_score
    print(f"\nXGBoost R²: {r2_score(y_test, y_pred):.4f}")
    # Feature importance
    imp = booster.get_score(importance_type='gain')
    print(f"XGBoost feature gains: {imp}")
except ImportError:
    print("\nxgboost not installed; skipping")

# ============================================================
# 5. SHAP values with TreeSHAP
# ============================================================
try:
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test[:50])
    print(f"\nSHAP values computed for 50 test samples, shape: {shap_values.shape}")
    # shap.summary_plot(shap_values, X_test[:50])
except ImportError:
    print("\nshap not installed; skipping")

# ============================================================
# 6. AdaBoost from scratch for classification
# ============================================================
def adaboost_fit(X, y, T=50):
    """AdaBoost with decision stumps as weak learners."""
    n = len(y)
    w = np.ones(n)/n
    classifiers, alphas = [], []
    for t in range(T):
        # Weak learner: decision stump
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X, y, sample_weight=w)
        pred = stump.predict(X)
        err = np.sum(w * (pred != y)) / np.sum(w)
        err = max(err, 1e-10)
        alpha = 0.5*np.log((1-err)/err)
        w = w * np.exp(-alpha * y * pred)
        w = w/w.sum()
        classifiers.append(stump); alphas.append(alpha)
    return classifiers, alphas

def adaboost_predict(classifiers, alphas, X):
    F = sum(a*c.predict(X) for c, a in zip(classifiers, alphas))
    return np.sign(F)

# Demo
Xc, yc = make_classification(n_samples=500, n_features=10, random_state=42)
yc = 2*yc - 1
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.3, random_state=42)
clfs, alphas = adaboost_fit(Xc_train, yc_train, T=100)
yhat = adaboost_predict(clfs, alphas, Xc_test)
print(f"\nAdaBoost test accuracy: {np.mean(yhat == yc_test):.4f}")
```

---

## 7.3.10 [QUANT APPLICATIONS]

1. **Credit default prediction.** GBMs (XGBoost, LightGBM) are industry standard for consumer lending, corporate default, CDS spreads.
2. **Alpha generation.** GBMs on cross-section of stock features for alpha signals — used widely by quant hedge funds.
3. **Churn / prepayment modeling.** Mortgage prepayment, credit card attrition — GBMs routinely outperform logistic regression.
4. **Execution slippage prediction.** Tree ensembles on trade characteristics to predict market-impact.
5. **Order flow classification.** Tree-based classifiers for informed vs noise trader signals (VPIN variants).
6. **Credit card fraud detection.** Production models at major card networks use boosted trees.
7. **Anti-money laundering.** SHAP values to explain model decisions for regulatory reporting.
8. **Fundamental investing.** Tree models on financial ratios to select long-short portfolios.
9. **Macro forecasting.** Boosted trees on macro indicators (GDP, inflation, unemployment nowcasts).
10. **Insurance pricing.** GLM + GBM "model boosting" for non-life insurance premium calculation.

---

## 7.3.11 Exercises

**★ (concept drills).**
1. Derive the optimal constant leaf value for MSE loss (= mean of $y$) and for log-loss (= logit of $\bar p$).
2. Show Gini and entropy impurities agree on ordering of best splits but differ on magnitude.
3. Why do random forests decorrelate trees vs. bagging alone?
4. Explain the out-of-bag error estimate. Why is it similar to LOO-CV?
5. State Friedman's gradient boosting algorithm in terms of functional gradient descent.
6. Why is second-order boosting (XGBoost) often better than first-order (GBM)?
7. What is a SHAP value? State the four axioms it satisfies.

**★★ (calculation).**
8. Implement a decision tree regressor from scratch. Use recursive best-split finding. Test on a 2D example.
9. Implement bagging and random forest manually using your tree regressor.
10. Derive the XGBoost split gain formula with regularization $\gamma T + \tfrac{1}{2}\lambda\|w\|^2$.
11. Implement gradient boosting for quantile regression. Predict 95th percentile of returns given features.
12. Apply XGBoost + SHAP to a public credit default dataset. Interpret top 5 features.
13. Compare Random Forest, XGBoost, LightGBM on a regression benchmark. Tune via CV and report test MSE.

**★★★ (open / research).**
14. **Feature engineering for equity returns.** Build a 50-feature model (momentum, value, quality, volatility) and train GBM on cross-section. Out-of-sample IC?
15. **Causal trees** (Athey-Imbens 2016). Implement the causal forest for treatment effect estimation.
16. **Gradient boosting with monotonicity constraints.** Enforce monotone response in credit scoring. Compare to unconstrained GBM.
17. **Deep neural nets vs. GBM on tabular finance data.** Test fairness of the comparison (tuning, data size) and report.
18. **Ensemble stacking.** Build a stacked ensemble of RF, XGB, LGBM, NN on a financial prediction task. Does stacking help?

---

*— End of Module 7.3. Next: Module 7.4, Time Series Foundations.*

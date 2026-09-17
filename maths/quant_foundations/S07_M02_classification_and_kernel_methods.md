# Module 7.2 — Classification and Kernel Methods

*Subject 7, Module 2. From logistic regression and discriminant analysis to support vector machines and the kernel trick.*

---

## Prerequisites

- **Module 7.1** — regression, regularization, the RKHS perspective.
- **Module 0.3 (multivariable calculus)** — gradients and second derivatives.
- **Module 0.2.5 (spectral theorem)** — Gram matrix spectra.
- **Module 2.2 (conditional expectation)** — Bayes classifier is $\mathbb{P}(Y|X)$.

---

## 7.2.1 Classification as statistical decision

Given $(X, Y)$ with $Y \in \{-1, +1\}$ (or $\{0, 1, \ldots, K-1\}$), a classifier is a function $f: \mathcal{X} \to \mathcal{Y}$. We want to minimize the misclassification risk:
$$
R(f) = \mathbb{P}(f(X) \ne Y).
$$

**Bayes optimal classifier.** $f^*(x) = \arg\max_y \mathbb{P}(Y = y \mid X = x)$, with Bayes risk
$$
R^* = 1 - \mathbb{E}[\max_y \mathbb{P}(Y = y \mid X)].
$$
Every classifier has $R(f) \ge R^*$. All algorithms below are attempts to approximate $f^*$ from data.

**Loss-based approach.** Minimize $\mathbb{E}[\ell(Y, f(X))]$ for surrogate losses $\ell$:
- 0-1 loss: $\ell(y, \hat y) = \mathbb{1}\{y \ne \hat y\}$. Hard to optimize (non-convex).
- Logistic loss: $\ell(y, f) = \log(1 + e^{-yf})$. Convex, smooth.
- Hinge loss: $\ell(y, f) = (1 - yf)_+$. Convex, non-smooth; leads to SVM.
- Exponential loss: $\ell(y, f) = e^{-yf}$. Convex; leads to AdaBoost.
- Squared loss: $\ell(y, f) = (y - f)^2$. Convex, smooth but not Fisher-consistent.

All convex losses above are Fisher consistent: minimizing their population risk recovers the Bayes decision when $\mathcal{F}$ is rich enough.

---

## 7.2.2 Logistic regression

Model $\mathbb{P}(Y=1 \mid X = x) = \sigma(\beta^\top x)$ with $\sigma(u) = 1/(1+e^{-u})$.

**Negative log-likelihood**:
$$
\mathcal{L}(\beta) = -\sum_i [y_i \log \sigma(\beta^\top x_i) + (1 - y_i) \log(1 - \sigma(\beta^\top x_i))].
$$
Equivalent to logistic loss with $y \in \{-1, +1\}$ encoding.

**Gradient**: $\nabla_\beta \mathcal{L} = -\sum_i (y_i - \sigma(\beta^\top x_i)) x_i = -X^\top(y - p)$.

**Hessian**: $\nabla^2 \mathcal{L} = X^\top W X$ with $W = \text{diag}(p_i(1-p_i))$.

**Newton-Raphson / IRLS**. Iteratively Reweighted Least Squares:
$$
\beta^{(t+1)} = \beta^{(t)} + (X^\top W X)^{-1} X^\top(y - p) = (X^\top W X)^{-1} X^\top W z
$$
where $z = X\beta + W^{-1}(y - p)$ is the adjusted response. Converges quadratically near optimum.

**Regularized logistic** (L2): add $\lambda \|\beta\|^2$. Preserves strict convexity even when $X$ has rank deficiency.

**Multiclass: softmax.** For $K$ classes,
$$
\mathbb{P}(Y = k \mid X) = \frac{e^{\beta_k^\top x}}{\sum_j e^{\beta_j^\top x}}.
$$

**Calibration.** Logistic regression outputs calibrated probabilities under the linear model; for arbitrary classifiers, post-hoc calibration (Platt scaling, isotonic) is needed.

---

## 7.2.3 Linear and quadratic discriminant analysis (LDA / QDA)

Assume $X \mid Y = k \sim \mathcal{N}(\mu_k, \Sigma_k)$.

**LDA** (equal covariances $\Sigma_k = \Sigma$). Bayes rule becomes linear: decide class $k$ maximizing
$$
\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log\pi_k.
$$

**QDA** (different covariances). Bayes rule becomes quadratic:
$$
\delta_k(x) = -\tfrac{1}{2}\log|\Sigma_k| - \tfrac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) + \log\pi_k.
$$

**Regularized LDA** shrinks $\hat\Sigma$ toward diagonal or toward a common covariance (Friedman 1989).

**LDA vs logistic**. Both have linear decision boundaries, but LDA is more efficient when the Gaussian assumption holds; logistic is more robust.

**Fisher's interpretation** (1936). Maximize between-class variance / within-class variance via generalized eigenvalue problem. For 2-class case, the Fisher direction is $w^* = \Sigma^{-1}(\mu_1 - \mu_0)$ — parallel to LDA direction.

---

## 7.2.4 Support Vector Machines (Vapnik 1995)

**Separable case.** Find hyperplane $\{x: w^\top x + b = 0\}$ that separates classes with maximum margin:
$$
\max_{w, b} \frac{2}{\|w\|} \text{ s.t. } y_i(w^\top x_i + b) \ge 1, \forall i.
$$
Equivalent to:
$$
\min_{w, b} \tfrac{1}{2}\|w\|^2 \text{ s.t. } y_i(w^\top x_i + b) \ge 1.
$$

**Lagrangian dual**. Form
$$
L = \tfrac{1}{2}\|w\|^2 - \sum_i \alpha_i [y_i(w^\top x_i + b) - 1].
$$
KKT conditions → $w = \sum_i \alpha_i y_i x_i$, $\sum_i \alpha_i y_i = 0$. Substitute:
$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j x_i^\top x_j, \quad \alpha_i \ge 0, \sum \alpha_i y_i = 0.
$$

**Support vectors**: points with $\alpha_i > 0$, exactly on the margin. Typically $O(\sqrt n)$ of them.

**Non-separable (soft margin).** Add slack $\xi_i \ge 0$:
$$
\min \tfrac{1}{2}\|w\|^2 + C\sum \xi_i, \text{ s.t. } y_i(w^\top x_i + b) \ge 1 - \xi_i.
$$
Dual has constraint $0 \le \alpha_i \le C$ ("box constraint").

**Hinge-loss formulation**. The primal is equivalent to
$$
\min_{w,b} \sum_i (1 - y_i(w^\top x_i + b))_+ + \frac{1}{2C}\|w\|^2,
$$
showing SVM as penalized hinge-loss minimization.

---

## 7.2.5 The kernel trick

All computations in SVM's dual involve only inner products $x_i^\top x_j$. Replace with $k(x_i, x_j) = \langle\phi(x_i), \phi(x_j)\rangle$ for some (implicit) feature map $\phi$:
$$
\max_\alpha \sum \alpha_i - \tfrac{1}{2}\sum \alpha_i \alpha_j y_i y_j k(x_i, x_j).
$$

**Classifier**: $f(x) = \sum_{i \in SV} \alpha_i y_i k(x, x_i) + b$.

**Mercer's theorem**. A symmetric function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a valid kernel iff for every finite $\{x_1, \ldots, x_n\}$, the Gram matrix $K_{ij} = k(x_i, x_j)$ is positive semi-definite. Equivalently, $k$ admits a spectral decomposition $k(x,y) = \sum_\lambda \lambda \phi_\lambda(x)\phi_\lambda(y)$.

**Popular kernels.**
- **Linear**: $k(x, y) = x^\top y$.
- **Polynomial**: $k(x, y) = (x^\top y + c)^d$. Feature map = polynomials of degree $\le d$.
- **RBF / Gaussian**: $k(x, y) = \exp(-\gamma\|x-y\|^2)$. Infinite-dim feature space.
- **Sigmoid**: $k(x, y) = \tanh(\alpha x^\top y + c)$. Not always PSD.
- **String kernels** for sequences, **graph kernels** for graphs, etc.

**RKHS.** Functions of the form $f(x) = \sum_i \alpha_i k(x, x_i)$ form a reproducing kernel Hilbert space $\mathcal{H}_k$ with inner product $\langle f, g\rangle = \sum_{i,j} \alpha_i \beta_j k(x_i, x_j)$.

**Representer theorem (Kimeldorf-Wahba 1971, Schölkopf-Herbrich-Smola 2001).** For any loss $L$ and strictly increasing regularizer $\Omega$, the minimizer of
$$
\sum L(y_i, f(x_i)) + \lambda \Omega(\|f\|_{\mathcal{H}})
$$
over $f \in \mathcal{H}_k$ admits $f^*(x) = \sum_i \alpha_i k(x, x_i)$. Finite-parameter optimization in infinite-dim space.

---

## 7.2.6 SMO and training SVMs at scale

**Sequential Minimal Optimization** (Platt 1998). Optimize over pairs of $\alpha_i, \alpha_j$ at a time, respecting equality $\sum \alpha_i y_i = 0$. Closed-form update per pair.

**LibSVM** implementation: $O(n^2)$ to $O(n^3)$ for full training. Kernel matrix doesn't need to be stored; lazy computation with caching.

**Linear SVM** at scale: stochastic subgradient (Pegasos), dual coordinate descent (LIBLINEAR), trust-region Newton. Handles $n = 10^6, p = 10^5$.

**Nonlinear SVM at scale.** Random Fourier features approximate $k(x, y) \approx \phi(x)^\top \phi(y)$ for finite random $\phi: \mathcal{X} \to \mathbb{R}^m$. Then train linear SVM.

---

## 7.2.7 Gaussian processes for classification

**GP prior**: $f \sim GP(0, k)$, likelihood $\mathbb{P}(y = 1 \mid f) = \sigma(f)$.

**Posterior** is non-Gaussian but Laplace or EP approximation gives tractable Gaussian form. Similar cost to SVM ($O(n^3)$) but gives calibrated probabilities.

**Hyperparameter learning**: marginal likelihood $\log p(y)$ optimized over kernel parameters.

---

## 7.2.8 Calibration and probabilistic outputs

SVM outputs a margin, not a probability. **Platt scaling** fits a logistic regression on the margins:
$$
\hat{\mathbb{P}}(y = 1 \mid x) = \sigma(A \cdot f_{\text{SVM}}(x) + B).
$$

**Isotonic regression**: non-parametric monotone calibration. Useful when Platt's logistic assumption is violated.

**Proper scoring rules** to evaluate calibration: Brier score, log-loss.

---

## 7.2.9 Evaluation: ROC, AUC, precision-recall

**Confusion matrix**: TP, FP, TN, FN.
- **Precision** = TP/(TP + FP).
- **Recall** (sensitivity) = TP/(TP + FN).
- **Specificity** = TN/(TN + FP).
- **F1** = harmonic mean of precision and recall.

**ROC curve**: TPR vs FPR as threshold varies.
**AUC** = area under ROC. Probabilistic interpretation: $\mathbb{P}(f(X^+) > f(X^-))$ for independent samples from positive and negative classes.

**For imbalanced data** (fraud detection, default prediction): precision-recall curve is more informative than ROC.

---

## 7.2.10 Python: classification toolkit

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# ============================================================
# 1. Logistic regression via Newton / IRLS
# ============================================================
def logistic_irls(X, y, max_iter=30, tol=1e-8):
    n, p = X.shape
    beta = np.zeros(p)
    for it in range(max_iter):
        eta = X @ beta
        mu = 1/(1 + np.exp(-eta))
        W = mu*(1 - mu)
        XtWX = (X.T * W) @ X
        grad = X.T @ (y - mu)
        dbeta = np.linalg.solve(XtWX + 1e-8*np.eye(p), grad)
        beta = beta + dbeta
        if np.max(np.abs(dbeta)) < tol: break
    return beta

np.random.seed(0)
n, p = 500, 3
X = np.column_stack([np.ones(n), np.random.randn(n, p)])
beta_true = np.array([0.5, 1, -1, 0.5])
y = (np.random.rand(n) < 1/(1 + np.exp(-X @ beta_true))).astype(int)
beta_hat = logistic_irls(X, y)
print(f"Logistic regression true: {beta_true}")
print(f"                    est:  {np.round(beta_hat, 3)}")

# ============================================================
# 2. LDA / QDA
# ============================================================
def lda_fit_predict(X_train, y_train, X_test):
    classes = np.unique(y_train)
    mu = {c: X_train[y_train == c].mean(axis=0) for c in classes}
    pi = {c: np.mean(y_train == c) for c in classes}
    # pooled covariance
    X_centered = np.vstack([X_train[y_train==c] - mu[c] for c in classes])
    Sigma = (X_centered.T @ X_centered) / (len(y_train) - len(classes))
    S_inv = np.linalg.inv(Sigma)
    scores = np.zeros((len(X_test), len(classes)))
    for i, c in enumerate(classes):
        scores[:, i] = X_test @ S_inv @ mu[c] - 0.5*mu[c] @ S_inv @ mu[c] + np.log(pi[c])
    return classes[np.argmax(scores, axis=1)]

def qda_fit_predict(X_train, y_train, X_test):
    classes = np.unique(y_train)
    mu = {c: X_train[y_train == c].mean(axis=0) for c in classes}
    pi = {c: np.mean(y_train == c) for c in classes}
    Sigma = {c: np.cov(X_train[y_train == c].T) for c in classes}
    scores = np.zeros((len(X_test), len(classes)))
    for i, c in enumerate(classes):
        rv = multivariate_normal(mu[c], Sigma[c])
        scores[:, i] = rv.logpdf(X_test) + np.log(pi[c])
    return classes[np.argmax(scores, axis=1)]

# Synthetic 2D
np.random.seed(1)
X_train = np.vstack([np.random.multivariate_normal([0,0], np.eye(2), 100),
                     np.random.multivariate_normal([3,3], [[2,1],[1,2]], 100)])
y_train = np.array([0]*100 + [1]*100)
X_test = np.random.randn(50, 2) * 2 + 1.5
yhat_lda = lda_fit_predict(X_train, y_train, X_test)
yhat_qda = qda_fit_predict(X_train, y_train, X_test)
print(f"\nLDA vs QDA: agree on {np.mean(yhat_lda == yhat_qda):.0%} of test points")

# ============================================================
# 3. Kernel SVM via SMO (minimal implementation)
# ============================================================
def simple_smo(K, y, C=1.0, tol=1e-3, max_passes=10):
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0
    passes = 0
    while passes < max_passes:
        num_changed = 0
        for i in range(n):
            Ei = (alpha * y) @ K[:, i] + b - y[i]
            if (y[i]*Ei < -tol and alpha[i] < C) or (y[i]*Ei > tol and alpha[i] > 0):
                j = np.random.choice(np.delete(np.arange(n), i))
                Ej = (alpha * y) @ K[:, j] + b - y[j]
                a_i_old, a_j_old = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L = max(0, a_j_old - a_i_old); H = min(C, C + a_j_old - a_i_old)
                else:
                    L = max(0, a_i_old + a_j_old - C); H = min(C, a_i_old + a_j_old)
                if L == H: continue
                eta = 2*K[i,j] - K[i,i] - K[j,j]
                if eta >= 0: continue
                alpha[j] = a_j_old - y[j]*(Ei - Ej)/eta
                alpha[j] = np.clip(alpha[j], L, H)
                if abs(alpha[j] - a_j_old) < 1e-5: continue
                alpha[i] = a_i_old + y[i]*y[j]*(a_j_old - alpha[j])
                b1 = b - Ei - y[i]*(alpha[i] - a_i_old)*K[i,i] - y[j]*(alpha[j] - a_j_old)*K[i,j]
                b2 = b - Ej - y[i]*(alpha[i] - a_i_old)*K[i,j] - y[j]*(alpha[j] - a_j_old)*K[j,j]
                if 0 < alpha[i] < C: b = b1
                elif 0 < alpha[j] < C: b = b2
                else: b = (b1 + b2)/2
                num_changed += 1
        passes = passes + 1 if num_changed == 0 else 0
    return alpha, b

def rbf_kernel(X1, X2, gamma=1.0):
    d2 = np.sum(X1**2, axis=1)[:,None] + np.sum(X2**2, axis=1)[None,:] - 2*X1 @ X2.T
    return np.exp(-gamma*d2)

# Nonlinear classification problem: two spirals
np.random.seed(42)
n = 50
t = np.linspace(0, 2*np.pi, n)
X_pos = np.column_stack([t*np.cos(t), t*np.sin(t)]) + 0.1*np.random.randn(n, 2)
X_neg = np.column_stack([t*np.cos(t+np.pi), t*np.sin(t+np.pi)]) + 0.1*np.random.randn(n, 2)
X_svm = np.vstack([X_pos, X_neg])
y_svm = np.array([1]*n + [-1]*n).astype(float)
K = rbf_kernel(X_svm, X_svm, gamma=0.5)
alpha, b = simple_smo(K, y_svm, C=1.0, max_passes=20)
sv = alpha > 1e-4
# Predict on a grid
xx, yy = np.meshgrid(np.linspace(-8, 8, 50), np.linspace(-8, 8, 50))
grid = np.column_stack([xx.ravel(), yy.ravel()])
K_grid = rbf_kernel(grid, X_svm, gamma=0.5)
f_grid = (K_grid * (alpha*y_svm)).sum(axis=1) + b
# plt.contour(xx, yy, f_grid.reshape(xx.shape), [0]); plt.scatter(...)
print(f"\nSVM trained: {sv.sum()} support vectors out of {len(y_svm)}")
print(f"Train accuracy: {np.mean(np.sign((K*y_svm*alpha).sum(axis=1) + b) == y_svm):.1%}")

# ============================================================
# 4. ROC and AUC
# ============================================================
def roc_auc(y_true, scores):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tpr = np.cumsum(y_sorted) / y_sorted.sum()
    fpr = np.cumsum(1 - y_sorted) / (len(y_sorted) - y_sorted.sum())
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc

scores = 1/(1 + np.exp(-(X @ beta_hat)))  # logistic probabilities
fpr, tpr, auc = roc_auc(y, scores)
print(f"\nLogistic AUC: {auc:.4f}")
```

---

## 7.2.11 [QUANT APPLICATIONS]

1. **Default prediction / credit scoring.** Logistic or gradient-boosted logistic on borrower features; used by banks and fintech for origination.
2. **Stock selection.** Classify "top quintile next month" from fundamental + technical features; used by systematic funds.
3. **Event-driven trading.** Classify whether an earnings announcement will produce a large move from pre-announcement signals.
4. **Fraud detection.** Unbalanced classification with cost-sensitive loss; SVM + RBF kernel has been state-of-the-art for structured feature sets.
5. **Regime classification.** Hidden Markov or clustering classifiers on macro factors to identify regimes (high-vol, trending, etc.).
6. **KYC / anti-money laundering.** Classification of suspicious transactions; GP classifiers with calibrated probabilities.
7. **Trade classification.** Lee-Ready and ML-augmented classifiers for buyer- vs. seller-initiated trades.
8. **Bond default / rating migration.** Classification into rating buckets from financial ratios.
9. **Mortgage prepayment modeling.** Classify whether borrower prepays in next period.
10. **Market regime state detection.** Multi-class classification (bull, bear, sideways) on macro features.

---

## 7.2.12 Exercises

**★ (concept drills).**
1. Show that the Bayes classifier minimizes 0-1 risk.
2. Derive IRLS update $\beta^{(t+1)} = (X^\top WX)^{-1} X^\top Wz$ from the Newton-Raphson iteration on logistic loss.
3. Prove that LDA's decision boundary is linear when $\Sigma_0 = \Sigma_1$.
4. Show that QDA's boundary is quadratic.
5. Derive the SVM dual starting from the primal.
6. State Mercer's theorem. Give three examples of PSD kernels.
7. Representer theorem: state it and sketch its proof.

**★★ (calculation).**
8. Implement IRLS for multinomial logistic regression. Test on the Iris dataset.
9. Implement LDA and QDA; compare on a synthetic dataset with known class conditional densities.
10. Implement SMO for kernel SVM. Train on XOR-like 2D data with RBF kernel.
11. Derive the updates for Gaussian-process classification with Laplace approximation.
12. AUC decomposition. Show AUC = $\mathbb{P}(f(X^+) > f(X^-))$.
13. Calibration analysis. Platt-scale an SVM on synthetic data; compare calibration curve before/after.

**★★★ (open / research).**
14. **Structured SVM.** Implement multi-class SVM via Crammer-Singer formulation.
15. **Kernel selection.** Multiple kernel learning (MKL): automatically combine several kernels. Test on a financial dataset.
16. **Deep learning classifiers.** Compare MLP and gradient-boosted trees on a credit default dataset; which wins? Why?
17. **Anomaly detection via one-class SVM.** Implement ν-SVM for one-class classification; apply to trade surveillance.
18. **Gaussian process classification at scale.** Implement inducing-points approximation for GPC; scale to $n = 10^5$.

---

*— End of Module 7.2. Next: Module 7.3, Trees, Random Forests, and Gradient Boosting.*

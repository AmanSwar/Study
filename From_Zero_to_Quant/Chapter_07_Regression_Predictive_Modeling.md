# Chapter 7: Regression and Predictive Modeling

## Overview

Regression is the foundational statistical technique in quantitative finance. While linear regression appears simple on the surface, financial data violates nearly every classical assumption: returns exhibit heteroskedasticity (changing volatility), autocorrelation (serial dependence), and non-normality (fat tails and skewness). This chapter teaches you how to estimate regression models correctly in financial contexts, understand what breaks, and apply robust fixes.

You'll also learn how to handle high-dimensional financial datasets (where you have 100+ potential predictors) using regularization techniques (Ridge, Lasso, Elastic Net), and how to exploit cross-sectional structure to build more powerful return prediction models using panel regression and the Fama-MacBeth procedure.

**Learning Outcomes:**
- Understand OLS assumptions and how they break in finance
- Implement robust standard errors and interpret coefficients correctly
- Use regularization to prevent overfitting in high-dimensional settings
- Build cross-sectional and panel regression models for return prediction
- Implement the Fama-MacBeth procedure from scratch

---

# Module 7.1: Linear Regression in Finance

## 7.1.1 The Classical Linear Regression Model

The classical linear regression model (Ordinary Least Squares, or OLS) is:

$$y_i = \beta_0 + \beta_1 x_{i,1} + \beta_2 x_{i,2} + \cdots + \beta_k x_{i,k} + \epsilon_i$$

Or in matrix form:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where:
- $\mathbf{y}$ is an $n \times 1$ vector of outcomes (e.g., stock returns)
- $\mathbf{X}$ is an $n \times (k+1)$ design matrix (including a column of 1s for the intercept)
- $\boldsymbol{\beta}$ is the $(k+1) \times 1$ coefficient vector we want to estimate
- $\boldsymbol{\epsilon}$ is the $n \times 1$ error vector

The OLS estimator minimizes the sum of squared residuals:

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Classical Assumptions

The textbook assumptions for OLS to be unbiased and have desirable properties are:

1. **Linearity**: The true model is linear in parameters
2. **Exogeneity**: $E[\boldsymbol{\epsilon} | \mathbf{X}] = 0$ (errors are uncorrelated with predictors)
3. **No multicollinearity**: $\mathbf{X}^T \mathbf{X}$ is full rank
4. **Homoskedasticity**: $\text{Var}(\epsilon_i | \mathbf{X}) = \sigma^2$ (constant variance)
5. **No autocorrelation**: $\text{Cov}(\epsilon_i, \epsilon_j | \mathbf{X}) = 0$ for $i \neq j$
6. **Normality**: $\boldsymbol{\epsilon} | \mathbf{X} \sim N(0, \sigma^2 \mathbf{I})$

Assumptions 1-3 ensure $\hat{\boldsymbol{\beta}}$ is unbiased. Assumptions 4-6 ensure standard errors and hypothesis tests are valid.

## 7.1.2 How Financial Data Violates These Assumptions

Financial data violates almost every assumption. Here's why:

### Heteroskedasticity

Financial returns have **time-varying volatility**. During crisis periods, volatility spikes; during calm periods, it's lower. If we predict stock returns using features like momentum or valuation, the regression errors will have higher variance when markets are volatile and lower variance when calm.

Mathematically: $\text{Var}(\epsilon_i | \mathbf{X}) = \sigma_i^2$ (not constant).

**Consequence**: Standard errors from textbook OLS are biased. We underestimate uncertainty during low-volatility periods and overestimate it during high-volatility periods.

### Autocorrelation

Financial time series are serially dependent. If today's return was higher than expected, there's often some residual effect on tomorrow's return. This violates the no-autocorrelation assumption.

Mathematically: $\text{Cov}(\epsilon_i, \epsilon_j | \mathbf{X}) \neq 0$ for $i \neq j$.

**Consequence**: Standard errors are underestimated, leading to inflated t-statistics and false rejections of null hypotheses.

### Non-Normality

Stock returns have fat tails (more extreme moves than normal distribution predicts) and sometimes skewness. The normality assumption is clearly violated.

**Consequence**: Confidence intervals and hypothesis tests are unreliable, especially for extreme events.

### Endogeneity / Simultaneity

If you're predicting returns using variables determined by returns (e.g., price-to-earnings ratio), you have endogeneity. The causal direction is unclear.

## 7.1.3 Robust Standard Errors

The solution is to use **robust standard errors** that correct for heteroskedasticity and/or autocorrelation without assuming homoskedasticity or no autocorrelation.

### Heteroskedasticity-Consistent (HC) Standard Errors

The textbook formula for $\text{Var}(\hat{\boldsymbol{\beta}})$ is:

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$$

where $\sigma^2$ is estimated by the residual variance. This assumes homoskedasticity.

**White's HC estimator** (1980) replaces this with:

$$\widehat{\text{Var}}(\hat{\boldsymbol{\beta}})_{\text{HC}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \boldsymbol{\Omega} \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1}$$

where $\boldsymbol{\Omega}$ is an $n \times n$ diagonal matrix with $\Omega_{ii} = \hat{\epsilon}_i^2$ (squared residuals).

There are variants: HC0 (White's original), HC1, HC2, HC3. In practice, use HC1 for small samples.

### Heteroskedasticity and Autocorrelation Consistent (HAC) Standard Errors

If you have both heteroskedasticity and autocorrelation, use **Newey-West HAC estimators** (1987):

$$\widehat{\text{Var}}(\hat{\boldsymbol{\beta}})_{\text{HAC}} = (\mathbf{X}^T \mathbf{X})^{-1} \left[ \sum_{i=1}^{n} \hat{\epsilon}_i^2 \mathbf{x}_i \mathbf{x}_i^T + \sum_{j=1}^{q} w_j \sum_{i=j+1}^{n} \hat{\epsilon}_i \hat{\epsilon}_{i-j} (\mathbf{x}_i \mathbf{x}_{i-j}^T + \mathbf{x}_{i-j} \mathbf{x}_i^T) \right] (\mathbf{X}^T \mathbf{X})^{-1}$$

where:
- $q$ is the lag length (typically $q = \lfloor 1.3 \times n^{1/3} \rfloor$)
- $w_j = 1 - \frac{j}{q+1}$ are Bartlett weights

**Key insight**: Newey-West downweights distant lags and upweights recent ones, accounting for the typical autocorrelation decay in financial data.

## 7.1.4 Interpreting Coefficients as Betas and Sensitivities

In finance, regression coefficients are interpreted as **betas** (sensitivities to factors):

$$r_i = \alpha + \beta_1 r_{\text{market}} + \epsilon_i$$

Here, $\beta_1$ is the stock's sensitivity to market returns: a 1% increase in market return is associated with a $\beta_1$% increase in the stock's return (on average).

More generally, if you regress returns on multiple factors:

$$r_i = \alpha + \beta_1 f_1 + \beta_2 f_2 + \cdots + \beta_k f_k + \epsilon_i$$

then:
- $\alpha$ is the "alpha" (excess return not explained by factors)
- $\beta_j$ is the exposure to factor $j$
- $\epsilon_i$ is idiosyncratic risk

**Standardized vs. Unstandardized Coefficients**: 
- Unstandardized: a 1-unit change in $x$ produces a $\beta$ change in $y$
- Standardized: a 1-standard-deviation change in $x$ produces a $\beta$ change in $y$ (in standard deviation units)

For comparing importance across variables with different scales, use standardized coefficients.

## 7.1.5 Detecting and Addressing Multicollinearity

**Multicollinearity** occurs when predictors are highly correlated. This inflates standard errors and makes coefficients unstable (small changes in data → large changes in estimates).

### Variance Inflation Factor (VIF)

For each predictor $j$, calculate:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other predictors.

**Interpretation**:
- $\text{VIF}_j = 1$: no multicollinearity
- $\text{VIF}_j < 5$: acceptable
- $\text{VIF}_j > 5$: problematic (inflate standard errors by $\sqrt{\text{VIF}_j} > 2.2$)
- $\text{VIF}_j > 10$: severe

### Solutions

1. **Remove highly collinear features**: Keep the most theoretically important
2. **Combine collinear features**: Create a composite (e.g., average)
3. **Use regularization** (Module 7.2): Ridge/Lasso automatically shrink collinear coefficients
4. **Use domain knowledge**: Drop features that don't make economic sense

---

## 7.1.6 Python Implementation: Robust Regression

Here's production-grade code implementing OLS with robust standard errors:

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict
import warnings

class RobustLinearRegression:
    """
    OLS regression with HC (heteroskedasticity-consistent) and 
    HAC (heteroskedasticity-and-autocorrelation-consistent) standard errors.
    
    Attributes:
        beta (np.ndarray): Coefficient estimates
        se (np.ndarray): Standard errors
        t_stats (np.ndarray): t-statistics
        p_values (np.ndarray): p-values
        r_squared (float): R-squared
        adj_r_squared (float): Adjusted R-squared
        residuals (np.ndarray): Regression residuals
    """
    
    def __init__(self, use_hac: bool = False, lag_length: int = None):
        """
        Initialize regression estimator.
        
        Args:
            use_hac: If True, use Newey-West HAC; else use White HC1
            lag_length: For HAC, lag length. If None, use 1.3*n^(1/3)
        """
        self.use_hac = use_hac
        self.lag_length = lag_length
        self.beta = None
        self.se = None
        self.t_stats = None
        self.p_values = None
        self.r_squared = None
        self.adj_r_squared = None
        self.residuals = None
        self._x = None
        self._y = None
        self._x_with_const = None
        self.n_obs = None
        self.n_features = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RobustLinearRegression':
        """
        Fit OLS regression and compute robust standard errors.
        
        Args:
            X: Design matrix (n_obs, n_features), without intercept
            y: Target vector (n_obs,)
        
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")
        
        self.n_obs = X.shape[0]
        self.n_features = X.shape[1]
        
        # Add constant
        X_const = np.column_stack([np.ones(self.n_obs), X])
        self._x_with_const = X_const
        self._x = X
        self._y = y
        
        # OLS estimation
        XtX_inv = np.linalg.inv(X_const.T @ X_const)
        self.beta = XtX_inv @ X_const.T @ y
        
        # Residuals
        self.residuals = y - X_const @ self.beta
        
        # R-squared
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum(self.residuals ** 2)
        self.r_squared = 1 - ss_res / ss_tot
        self.adj_r_squared = 1 - (1 - self.r_squared) * (self.n_obs - 1) / (self.n_obs - self.n_features - 1)
        
        # Robust standard errors
        if self.use_hac:
            self._compute_hac_se(X_const, XtX_inv)
        else:
            self._compute_hc_se(X_const, XtX_inv)
        
        # t-statistics and p-values
        self.t_stats = self.beta / self.se
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), self.n_obs - self.n_features - 1))
        
        return self
    
    def _compute_hc_se(self, X_const: np.ndarray, XtX_inv: np.ndarray) -> None:
        """Compute White HC1 standard errors."""
        resid_sq = self.residuals ** 2
        
        # HC1: adjust for degrees of freedom
        scale = self.n_obs / (self.n_obs - self.n_features - 1)
        
        # Meat: X^T Omega X, where Omega is diagonal with resid_sq
        meat = X_const.T @ (resid_sq[:, np.newaxis] * X_const) * scale
        
        # Sandwich: (X^T X)^-1 (X^T Omega X) (X^T X)^-1
        var_cov = XtX_inv @ meat @ XtX_inv
        
        self.se = np.sqrt(np.diag(var_cov))
    
    def _compute_hac_se(self, X_const: np.ndarray, XtX_inv: np.ndarray) -> None:
        """Compute Newey-West HAC standard errors."""
        if self.lag_length is None:
            # Andrew's automatic bandwidth
            self.lag_length = int(np.ceil(1.3 * (self.n_obs ** (1/3))))
        
        # Bartlett weights: w_j = 1 - j/(q+1)
        weights = 1 - np.arange(self.lag_length + 1) / (self.lag_length + 1)
        
        # Compute long-run covariance
        lrc = np.zeros((self.n_features + 1, self.n_features + 1))
        
        # j=0 term
        for i in range(self.n_obs):
            lrc += self.residuals[i] ** 2 * np.outer(X_const[i], X_const[i])
        lrc *= weights[0]
        
        # j>0 terms
        for j in range(1, self.lag_length + 1):
            for i in range(j, self.n_obs):
                term = self.residuals[i] * self.residuals[i-j] * (
                    np.outer(X_const[i], X_const[i-j]) + 
                    np.outer(X_const[i-j], X_const[i])
                )
                lrc += weights[j] * term
        
        # Sandwich formula
        var_cov = XtX_inv @ lrc @ XtX_inv
        
        self.se = np.sqrt(np.diag(var_cov))
    
    def compute_vif(self) -> np.ndarray:
        """
        Compute Variance Inflation Factors for each predictor.
        
        Returns:
            vif: Array of VIF values (without intercept)
        """
        vif = np.zeros(self.n_features)
        
        for j in range(self.n_features):
            # Regress X[j] on all others
            X_others = np.delete(self._x, j, axis=1)
            X_others_const = np.column_stack([np.ones(self.n_obs), X_others])
            
            # R-squared from auxiliary regression
            aux_beta = np.linalg.lstsq(X_others_const, self._x[:, j], rcond=None)[0]
            pred = X_others_const @ aux_beta
            ss_res = np.sum((self._x[:, j] - pred) ** 2)
            ss_tot = np.sum((self._x[:, j] - self._x[:, j].mean()) ** 2)
            r_sq = 1 - ss_res / ss_tot
            
            vif[j] = 1 / (1 - r_sq + 1e-10)  # Avoid division by zero
        
        return vif
    
    def summary(self) -> pd.DataFrame:
        """Return summary statistics as DataFrame."""
        return pd.DataFrame({
            'Coefficient': self.beta,
            'Std. Error': self.se,
            't-statistic': self.t_stats,
            'p-value': self.p_values
        })
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        X = np.asarray(X)
        X_const = np.column_stack([np.ones(X.shape[0]), X])
        return X_const @ self.beta


# Example: Predicting stock returns with robust regression
if __name__ == "__main__":
    # Simulate financial data
    np.random.seed(42)
    n_obs = 250  # ~1 year of daily data
    
    # Generate heteroskedastic returns
    time_varying_vol = 0.01 + 0.02 * np.sin(np.arange(n_obs) / 50)
    returns = np.random.normal(0, time_varying_vol)
    
    # Features: momentum, valuation, volatility
    momentum = returns.rolling(20).mean() if isinstance(returns, pd.Series) else np.convolve(returns, np.ones(20)/20, mode='same')
    momentum = momentum.astype(float)
    
    X = np.column_stack([
        momentum,  # momentum
        np.random.normal(0, 1, n_obs),  # valuation (simulated)
        np.abs(np.random.normal(0, 1, n_obs))  # realized vol
    ])
    
    # Fit models
    model_hc = RobustLinearRegression(use_hac=False)
    model_hc.fit(X, returns)
    
    model_hac = RobustLinearRegression(use_hac=True)
    model_hac.fit(X, returns)
    
    print("=" * 70)
    print("OLS with White HC1 Standard Errors")
    print("=" * 70)
    print(model_hc.summary())
    print(f"\nR-squared: {model_hc.r_squared:.4f}")
    print(f"Adjusted R-squared: {model_hc.adj_r_squared:.4f}")
    
    print("\n" + "=" * 70)
    print("OLS with Newey-West HAC Standard Errors")
    print("=" * 70)
    print(model_hac.summary())
    
    print("\n" + "=" * 70)
    print("Variance Inflation Factors (Multicollinearity Check)")
    print("=" * 70)
    vif = model_hc.compute_vif()
    for i, v in enumerate(vif):
        print(f"Feature {i}: VIF = {v:.3f}")
```

## [VISUALIZATION]

Create a diagnostic plot showing:
1. Fitted vs. residuals (to visualize heteroskedasticity)
2. Q-Q plot (to check normality)
3. ACF plot (to check autocorrelation)

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def plot_regression_diagnostics(model: RobustLinearRegression, lags: int = 20) -> None:
    """Plot OLS regression diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Fitted
    fitted = model._x_with_const @ model.beta
    axes[0, 0].scatter(fitted, model.residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted (Heteroskedasticity Check)')
    
    # 2. Q-Q Plot
    stats.probplot(model.residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot (Normality Check)')
    
    # 3. Residuals over time
    axes[1, 0].plot(model.residuals)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals over Time')
    
    # 4. ACF
    plot_acf(model.residuals, lags=lags, ax=axes[1, 1])
    axes[1, 1].set_title('ACF (Autocorrelation Check)')
    
    plt.tight_layout()
    plt.show()
```

## WARNING: Common Pitfalls

1. **Using textbook standard errors**: Always use robust standard errors in finance. Textbook SEs are usually too small, making you overconfident in your results.

2. **Ignoring endogeneity**: If you suspect your predictor might be caused by the outcome (reverse causality), OLS is biased. Consider instrumental variables or lag your predictors.

3. **P-hacking**: If you test many hypotheses, you'll find false positives. Use multiple testing corrections (Bonferroni, FDR) or pre-specify your model.

4. **Sample selection bias**: If you only use stocks that survived (survivorship bias), your results are biased upward. Use the full universe.

---

## Exercises

**Exercise 7.1.1**: Load NSE data for 5 stocks over 2 years. Regress daily returns on:
- Market returns (Nifty 50)
- Volume (as a proxy for liquidity)
- Prior day volatility

Use White HC1 standard errors. Which factors are significant? Compute VIF for each feature.

**Exercise 7.1.2**: Repeat Exercise 7.1.1 but use Newey-West HAC standard errors. How do they differ from HC1? Why might HAC be more appropriate here?

**Exercise 7.1.3**: Run the same regression with textbook OLS standard errors (assume homoskedasticity). Compare p-values. Which are artificially small?

---

# Module 7.2: Regularized Regression

## 7.2.1 The Regularization Problem

In finance, you often have many potential predictors: technical indicators, fundamental ratios, macroeconomic variables, sentiment scores, etc. If $k$ (number of features) approaches or exceeds $n$ (number of observations), OLS breaks:
- Estimates become very unstable
- You overfit: the model fits noise in the training data
- Out-of-sample performance is terrible

**Regularization** adds a penalty term to the loss function, discouraging large coefficients:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \mathcal{P}(\boldsymbol{\beta})$$

where $\lambda$ is the regularization parameter (tuned via cross-validation) and $\mathcal{P}(\boldsymbol{\beta})$ is the penalty.

The key insight: a simpler model (smaller coefficients) that slightly underfits the training data often generalizes better to new data.

## 7.2.2 Ridge Regression (L2 Penalty)

### Theory

Ridge regression uses the $L_2$ norm as penalty:

$$\mathcal{P}(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^{k} \beta_j^2$$

So the Ridge objective is:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^{k} \beta_j^2$$

The closed-form solution is:

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + 2n\lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

### Bias-Variance Tradeoff

Ridge regression trades **bias for lower variance**:
- **No regularization** ($\lambda = 0$): unbiased but high variance (unstable)
- **Strong regularization** ($\lambda \to \infty$): biased toward zero but low variance (stable)

**Optimal $\lambda$** balances the two. Typically, Ridge reduces test error even though it introduces bias.

### Properties

- **Shrinks all coefficients proportionally**: Never zeros out coefficients (unlike Lasso)
- **Helps with multicollinearity**: By adding $2n\lambda$ to the diagonal, it makes $\mathbf{X}^T \mathbf{X}$ more invertible
- **Keeps all features**: Every predictor remains in the model

## 7.2.3 Lasso Regression (L1 Penalty)

### Theory

Lasso uses the $L_1$ norm:

$$\mathcal{P}(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_1 = \sum_{j=1}^{k} |\beta_j|$$

Lasso objective:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^{k} |\beta_j|$$

**No closed-form solution**, but can be solved efficiently with coordinate descent or proximal gradient methods.

### Sparsity and Feature Selection

The key advantage of Lasso: **it can zero out coefficients**. For strong enough regularization, many $\hat{\beta}_j = 0$.

**Why?** The $L_1$ penalty has a "sharp corner" at zero. If the gradient at zero points away from zero, the coefficient stays at zero. The $L_2$ penalty is smooth everywhere, so it never exactly zeros.

This makes Lasso a feature selection method: it automatically selects which predictors matter.

### When to Use

- **You suspect many features are irrelevant**: Lasso finds the sparse subset
- **Interpretability matters**: Fewer features = simpler model = easier to explain
- **Computational efficiency**: You want to use fewer features in production

## 7.2.4 Elastic Net

Elastic Net combines $L_1$ and $L_2$ penalties:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \left[ \alpha \|\boldsymbol{\beta}\|_1 + (1-\alpha) \|\boldsymbol{\beta}\|_2^2 \right]$$

where $\alpha \in [0,1]$ balances the two penalties:
- $\alpha = 0$: Ridge
- $\alpha = 1$: Lasso
- $0 < \alpha < 1$: Elastic Net

**Advantages**:
- Combines sparsity (from Lasso) with grouping (from Ridge)
- If features are correlated, it tends to shrink them together, rather than arbitrarily dropping one (as Lasso would)
- More numerically stable than pure Lasso

## 7.2.5 Tuning Regularization: Why NOT Random CV

### The Problem: Leakage in Time Series

Standard k-fold cross-validation shuffles data randomly:

```
Original: Day 1 - 250
Fold 1 train:  1-200, test: 250
Fold 2 train:  1-100, 150-250, test: 101-149
...
```

**Problem**: You're training on future data (days 150-250) and testing on past data (days 101-149). This is **look-ahead bias** and inflates estimated performance.

### Time Series Cross-Validation

Use **expanding window** or **rolling window** CV:

**Expanding window** (recommended for this course):
```
Fold 1: Train on 1-100, test on 101-120
Fold 2: Train on 1-120, test on 121-140
Fold 3: Train on 1-140, test on 141-160
...
```

Only test on data that comes after training. This respects temporal order.

## 7.2.6 Application: Predicting Stock Returns with 100+ Features

Here's production code for Lasso/Ridge/Elastic Net with time-series CV:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import warnings

class RegularizedRegressionWithTimeSeriesCV:
    """
    Ridge, Lasso, and Elastic Net regression with time-series cross-validation.
    Automatically tunes regularization parameter and trains final model.
    """
    
    def __init__(self, method: str = 'lasso', cv_folds: int = 5):
        """
        Args:
            method: 'ridge', 'lasso', or 'elastic_net'
            cv_folds: Number of CV folds (with time-series CV)
        """
        self.method = method
        self.cv_folds = cv_folds
        self.lambda_opt = None
        self.alpha = None  # For elastic net, 0-1 mixing parameter
        self.coef = None
        self.intercept = None
        self.scaler = StandardScaler()
        self._is_fit = False
    
    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding operator for Lasso."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _proximal_gradient_step(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        beta: np.ndarray, 
        step_size: float,
        lambda_reg: float,
        method: str
    ) -> np.ndarray:
        """Single proximal gradient descent step."""
        n = X.shape[0]
        grad = X.T @ (X @ beta - y) / n
        
        if method == 'lasso':
            beta_new = self._soft_threshold(beta - step_size * grad, step_size * lambda_reg)
        elif method == 'ridge':
            # Ridge has closed form, but we'll use gradient step for consistency
            beta_new = beta - step_size * (grad + 2 * lambda_reg * beta)
        elif method == 'elastic_net':
            # Elastic net: combine L1 and L2
            alpha = 0.5  # User can adjust
            l1_thresh = step_size * lambda_reg * alpha
            l2_penalty = 2 * lambda_reg * (1 - alpha)
            beta_new = self._soft_threshold(beta - step_size * (grad + l2_penalty * beta), l1_thresh)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return beta_new
    
    def _fit_regularized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_reg: float,
        max_iter: int = 1000,
        tol: float = 1e-4
    ) -> np.ndarray:
        """Fit regularized regression using proximal gradient descent."""
        n, k = X.shape
        beta = np.zeros(k)
        step_size = 1.0 / (np.max(np.linalg.eigvalsh(X.T @ X)) + 1)
        
        for iteration in range(max_iter):
            beta_old = beta.copy()
            beta = self._proximal_gradient_step(X, y, beta, step_size, lambda_reg, self.method)
            
            # Check convergence
            if np.linalg.norm(beta - beta_old) < tol:
                break
        
        return beta
    
    def time_series_cv_split(self, n_obs: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-series CV splits (expanding window).
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        fold_size = n_obs // (self.cv_folds + 1)
        splits = []
        
        for fold in range(self.cv_folds):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, n_obs)
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def _cv_score(self, X: np.ndarray, y: np.ndarray, lambda_reg: float) -> float:
        """Compute average CV MSE for given regularization parameter."""
        splits = self.time_series_cv_split(X.shape[0])
        mse_scores = []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit on training set
            beta = self._fit_regularized(X_train, y_train, lambda_reg)
            
            # Evaluate on test set
            y_pred = X_test @ beta
            mse = np.mean((y_test - y_pred) ** 2)
            mse_scores.append(mse)
        
        return np.mean(mse_scores)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegularizedRegressionWithTimeSeriesCV':
        """
        Fit regularized regression with CV tuning of lambda.
        
        Args:
            X: Features (n_obs, n_features)
            y: Target (n_obs,)
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Standardize target
        y_mean = y.mean()
        y_std = y.std()
        y_scaled = (y - y_mean) / y_std
        
        # Grid search over lambda
        lambdas = np.logspace(-4, 2, 50)  # 10^-4 to 10^2
        cv_scores = []
        
        for lam in lambdas:
            score = self._cv_score(X_scaled, y_scaled, lam)
            cv_scores.append(score)
        
        self.lambda_opt = lambdas[np.argmin(cv_scores)]
        
        # Fit on full data with optimal lambda
        self.coef = self._fit_regularized(X_scaled, y_scaled, self.lambda_opt)
        self.intercept = y_mean
        
        self._is_fit = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_fit:
            raise ValueError("Model must be fit before prediction")
        
        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.coef + self.intercept
    
    def get_nonzero_features(self, threshold: float = 1e-4) -> np.ndarray:
        """Get indices of non-zero coefficients (useful for Lasso)."""
        return np.where(np.abs(self.coef) > threshold)[0]
    
    def summary(self) -> Dict:
        """Return model summary."""
        nonzero = np.sum(np.abs(self.coef) > 1e-4)
        total = len(self.coef)
        
        return {
            'method': self.method,
            'lambda_optimal': self.lambda_opt,
            'n_features_nonzero': nonzero,
            'n_features_total': total,
            'sparsity': nonzero / total
        }


# Example: Predict stock returns with 100 features
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data: 250 observations, 100 features
    n_obs = 250
    n_features = 100
    
    # Only 5 features truly matter, rest is noise
    X = np.random.normal(0, 1, (n_obs, n_features))
    true_beta = np.zeros(n_features)
    true_beta[:5] = [1.5, -0.8, 0.6, 1.2, -0.4]  # True coefficients
    
    y = X @ true_beta + np.random.normal(0, 0.5, n_obs)
    
    # Fit models
    models = {}
    for method in ['ridge', 'lasso']:
        model = RegularizedRegressionWithTimeSeriesCV(method=method)
        model.fit(X, y)
        models[method] = model
        print(f"\n{method.upper()} Regression:")
        print(models[method].summary())
        
        if method == 'lasso':
            nonzero_idx = model.get_nonzero_features()
            print(f"Non-zero features: {nonzero_idx}")
    
    # Compare: which features did each method keep?
    ridge_model = models['ridge']
    lasso_model = models['lasso']
    
    print("\n" + "="*70)
    print("Coefficient Comparison (First 10 Features)")
    print("="*70)
    for i in range(10):
        print(f"Feature {i}: Ridge={ridge_model.coef[i]:.4f}, Lasso={lasso_model.coef[i]:.4f}")
```

## [VISUALIZATION]

Plot regularization paths:

```python
def plot_regularization_path(X, y, method='lasso'):
    """Plot how coefficients change with lambda."""
    lambdas = np.logspace(-4, 2, 50)
    coefs = []
    
    for lam in lambdas:
        model = RegularizedRegressionWithTimeSeriesCV(method=method)
        model.lambda_opt = lam
        X_scaled = model.scaler.fit_transform(X)
        y_scaled = (y - y.mean()) / y.std()
        coefs.append(model._fit_regularized(X_scaled, y_scaled, lam))
    
    coefs = np.array(coefs)
    
    plt.figure(figsize=(12, 6))
    for i in range(min(10, X.shape[1])):
        plt.plot(lambdas, coefs[:, i], label=f'Feature {i}')
    
    plt.xlabel('Lambda (Regularization Strength)')
    plt.ylabel('Coefficient Value')
    plt.title(f'{method.upper()} Regularization Path')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.show()
```

## WARNING: The Regularization Bias

Regularized models are biased: they systematically shrink coefficients. This means:

1. **Coefficients are understated**: True effect sizes are larger than estimated
2. **In-sample fit is worse**: Regularized model predicts worse on training data
3. **Out-of-sample fit is better**: But much better on test data
4. **Interpretation is tricky**: Coefficients are not comparable across features (unless standardized)

The benefit: much better generalization. For prediction, this trade is worth it.

---

## Exercises

**Exercise 7.2.1**: Generate synthetic data with 200 observations and 50 features, where only 5 matter. Fit Ridge and Lasso with random CV (wrong!) and time-series CV (correct). How do the optimal $\lambda$ values differ?

**Exercise 7.2.2**: Load NSE data with technical indicators (momentum, volatility, RSI, MACD, Bollinger bands, etc.). You'll have 30+ features. Use Lasso to predict next-day returns. How many features does it select? Compare in-sample vs out-of-sample $R^2$.

**Exercise 7.2.3**: Implement Elastic Net from scratch. Try $\alpha \in \{0.1, 0.5, 0.9\}$. Which mixing ratio works best for your data?

---

# Module 7.3: Panel Data and Cross-Sectional Regression

## 7.3.1 Why Cross-Sectional Models Are More Powerful

Suppose you want to predict stock returns. You have two data types:

**Time-series approach** (Chapter 7.1-7.2):
- Predict stock $i$'s return using historical returns and features
- Limited by available history for each stock (~5-10 years)
- Each regression has $n$ = 250-2000 observations

**Cross-sectional approach** (this module):
- On each day, regress all stocks' returns on their characteristics
- Across $m$ stocks and $T$ days, you have $m \times T$ observations
- E.g., 500 stocks × 1000 days = 500,000 observations!
- Much more statistical power to detect return predictors

Example:
$$r_{i,t} = \alpha_t + \beta_{1,t} \text{momentum}_{i,t} + \beta_{2,t} \text{value}_{i,t} + \epsilon_{i,t}$$

where $i$ indexes stocks and $t$ indexes days.

**Key insight**: Factor exposures $\beta_{j,t}$ can vary over time. This captures time-varying risk premia.

## 7.3.2 Cross-Sectional Regression and Fama-MacBeth

The **Fama-MacBeth procedure** (Fama & MacBeth, 1973) is the standard approach:

1. **On each day $t$**, regress stock returns on characteristics:
   $$r_{i,t} = \alpha_t + \sum_{j=1}^{k} \beta_{j,t} x_{i,j,t} + \epsilon_{i,t}$$
   
   Perform OLS separately for each $t = 1, \ldots, T$.

2. **Average the coefficients** over time:
   $$\bar{\beta}_j = \frac{1}{T} \sum_{t=1}^{T} \beta_{j,t}$$

3. **Compute standard errors** using time-series of coefficients:
   $$\text{SE}(\bar{\beta}_j) = \frac{\text{std}(\beta_{j,1}, \ldots, \beta_{j,T})}{\sqrt{T}}$$

**Why this is powerful**:
- You get $m \times T$ observations (not just $T$)
- Accounts for potential panel structure and correlations within stocks
- SE accounts for time-variation in factor loadings
- Allows for time-varying risk premia

## 7.3.3 Panel Regression: Fixed vs Random Effects

If you have repeated observations for same units (stocks) over time, you have **panel data**:

$$r_{i,t} = \alpha + \beta x_{i,t} + \eta_i + \epsilon_{i,t}$$

where $\eta_i$ is stock-specific effect (e.g., sector, size, persistent alpha).

### Fixed Effects (Within Estimator)

Assume $\eta_i$ is correlated with $x_{i,t}$ (likely!). Demean within each stock:

$$r_{i,t} - \bar{r}_i = \beta(x_{i,t} - \bar{x}_i) + (\epsilon_{i,t} - \bar{\epsilon}_i)$$

Estimate with demeaned data. The $\eta_i$ cancels out.

**Advantage**: Accounts for any time-invariant unobservables (selection bias)
**Disadvantage**: Can't estimate effects of time-invariant characteristics (e.g., industry)

### Random Effects

Assume $\eta_i \perp x_{i,t}$ (independence). Use GLS (generalized least squares) to weight observations by variance.

**Advantage**: Can estimate time-invariant effects
**Disadvantage**: Biased if assumption violated (it usually is in finance)

**Verdict**: In finance, use fixed effects unless you have good reason otherwise.

## 7.3.4 Clustering Standard Errors

Panel data introduces **dependence**: observations from same stock are correlated, observations from same day are correlated.

**Clustered standard errors** account for this:

$$\widehat{\text{SE}}_{\text{cluster}}(\hat{\beta}) = \sqrt{\text{diag}\left( (\mathbf{X}^T \mathbf{X})^{-1} \left[\sum_{c} \mathbf{X}_{c}^T \boldsymbol{\epsilon}_c \boldsymbol{\epsilon}_c^T \mathbf{X}_{c} \right] (\mathbf{X}^T \mathbf{X})^{-1} \right)}$$

where the sum is over clusters $c$ (e.g., stocks or dates).

### Double Clustering

Often observations are clustered by **both stock and time**. Use double clustering (Thompson, 2011):

Combine cluster-by-stock and cluster-by-time standard errors.

## 7.3.5 Implementation: Fama-MacBeth from Scratch

Here's production code implementing Fama-MacBeth and panel regression with clustering:

```python
import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy import stats

class FamaMacBeth:
    """
    Fama-MacBeth (1973) procedure for cross-sectional regression.
    
    Regresses stock returns on characteristics each day separately,
    then averages coefficients and computes standard errors from time series.
    """
    
    def __init__(self):
        self.daily_betas = None  # (n_dates, n_features)
        self.avg_betas = None    # (n_features,)
        self.se_betas = None     # (n_features,)
        self.t_stats = None      # (n_features,)
        self.p_values = None     # (n_features,)
        self.n_dates = None
        self.n_features = None
        self.daily_alphas = None
    
    def fit(
        self,
        returns: pd.DataFrame,  # (n_dates, n_stocks)
        characteristics: pd.DataFrame  # (n_dates, n_stocks, n_features) or (n_dates*n_stocks, n_features)
    ) -> 'FamaMacBeth':
        """
        Fit Fama-MacBeth procedure.
        
        Args:
            returns: Stock returns, indexed by date and stock
            characteristics: Stock characteristics
        
        Returns:
            self
        """
        # Convert to proper format if needed
        if isinstance(returns, pd.DataFrame):
            dates = returns.index
            stocks = returns.columns
            n_dates = len(dates)
            n_stocks = len(stocks)
        else:
            raise ValueError("returns must be a DataFrame with dates as index and stocks as columns")
        
        if isinstance(characteristics, pd.DataFrame):
            # Stack format: reshape to (n_dates, n_stocks, n_features)
            dates_char = characteristics.index.get_level_values(0).unique()
            stocks_char = characteristics.index.get_level_values(1).unique()
            
            if len(dates_char) != n_dates or len(stocks_char) != n_stocks:
                raise ValueError("characteristics must have same dates and stocks as returns")
            
            n_features = characteristics.shape[1]
        else:
            raise ValueError("characteristics must be a DataFrame")
        
        self.n_dates = n_dates
        self.n_features = n_features
        self.daily_betas = np.zeros((n_dates, n_features))
        self.daily_alphas = np.zeros(n_dates)
        
        # Run daily cross-sectional regressions
        for t, date in enumerate(dates):
            # Get data for this date
            ret_t = returns.loc[date].values
            char_t = characteristics.loc[date].values  # (n_stocks, n_features)
            
            # Remove NaN
            valid_mask = ~(np.isnan(ret_t) | np.any(np.isnan(char_t), axis=1))
            ret_t = ret_t[valid_mask]
            char_t = char_t[valid_mask]
            
            if len(ret_t) < n_features + 1:
                # Not enough observations to run regression
                self.daily_betas[t, :] = np.nan
                self.daily_alphas[t] = np.nan
                continue
            
            # Add intercept
            X_t = np.column_stack([np.ones(len(ret_t)), char_t])
            
            # OLS
            try:
                beta_t = np.linalg.lstsq(X_t, ret_t, rcond=None)[0]
                self.daily_alphas[t] = beta_t[0]
                self.daily_betas[t, :] = beta_t[1:]
            except np.linalg.LinAlgError:
                self.daily_betas[t, :] = np.nan
                self.daily_alphas[t] = np.nan
        
        # Average over time
        valid_dates = ~np.isnan(self.daily_alphas)
        self.avg_betas = np.nanmean(self.daily_betas, axis=0)
        self.se_betas = np.nanstd(self.daily_betas, axis=0) / np.sqrt(np.sum(valid_dates))
        
        self.t_stats = self.avg_betas / self.se_betas
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), np.sum(valid_dates) - 1))
        
        return self
    
    def summary(self, feature_names: List[str] = None) -> pd.DataFrame:
        """Return summary table."""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.n_features)]
        
        return pd.DataFrame({
            'Coefficient': self.avg_betas,
            'Std. Error': self.se_betas,
            't-statistic': self.t_stats,
            'p-value': self.p_values
        }, index=feature_names)
    
    def get_daily_coefficients(self) -> pd.DataFrame:
        """Return daily coefficients as DataFrame."""
        return pd.DataFrame(
            self.daily_betas,
            columns=[f'Feature_{i}' for i in range(self.n_features)]
        )


class PanelRegressionWithClustering:
    """
    Panel regression with fixed effects and clustered standard errors.
    
    Supports clustering by one or two dimensions (e.g., stock and time).
    """
    
    def __init__(self, fe_type: str = None):
        """
        Args:
            fe_type: 'stock', 'time', or None for no fixed effects
        """
        self.fe_type = fe_type
        self.beta = None
        self.se_robust = None
        self.n_obs = None
        self.n_features = None
    
    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray,
        stock_id: np.ndarray,
        time_id: np.ndarray
    ) -> 'PanelRegressionWithClustering':
        """
        Fit panel regression with fixed effects.
        
        Args:
            y: Outcome (n_obs,)
            X: Features (n_obs, k)
            stock_id: Stock identifiers (n_obs,)
            time_id: Time identifiers (n_obs,)
        """
        self.n_obs = len(y)
        self.n_features = X.shape[1]
        
        # Demean by fixed effect
        if self.fe_type == 'stock':
            y_fe = self._demean_by_group(y, stock_id)
            X_fe = self._demean_by_group(X, stock_id)
        elif self.fe_type == 'time':
            y_fe = self._demean_by_group(y, time_id)
            X_fe = self._demean_by_group(X, time_id)
        else:
            y_fe = y
            X_fe = X
        
        # OLS on demeaned data
        self.beta = np.linalg.lstsq(X_fe, y_fe, rcond=None)[0]
        residuals = y_fe - X_fe @ self.beta
        
        # Clustered standard errors (by stock)
        self.se_robust = self._compute_clustered_se(
            X_fe, residuals, stock_id
        )
        
        return self
    
    def _demean_by_group(self, data: np.ndarray, group_id: np.ndarray) -> np.ndarray:
        """Demean data within each group."""
        unique_groups = np.unique(group_id)
        demeaned = data.copy()
        
        for group in unique_groups:
            mask = group_id == group
            if isinstance(data, np.ndarray) and data.ndim > 1:
                demeaned[mask] = data[mask] - data[mask].mean(axis=0)
            else:
                demeaned[mask] = data[mask] - data[mask].mean()
        
        return demeaned
    
    def _compute_clustered_se(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        cluster_id: np.ndarray
    ) -> np.ndarray:
        """Compute cluster-robust standard errors."""
        unique_clusters = np.unique(cluster_id)
        n_clusters = len(unique_clusters)
        
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Meat matrix
        meat = np.zeros((self.n_features, self.n_features))
        
        for cluster in unique_clusters:
            mask = cluster_id == cluster
            X_c = X[mask]
            eps_c = residuals[mask]
            meat += X_c.T @ (eps_c[:, np.newaxis] * X_c)
        
        # Sandwich
        var_cov = XtX_inv @ meat @ XtX_inv
        
        # Degrees of freedom adjustment
        df_adj = (n_clusters - 1) / (n_clusters - self.n_features) * self.n_obs / (self.n_obs - self.n_features)
        
        return np.sqrt(np.diag(var_cov) * df_adj)


# Example: Fama-MacBeth on NSE stocks
if __name__ == "__main__":
    # Simulate panel data
    np.random.seed(42)
    n_stocks = 100
    n_dates = 250
    n_features = 3
    
    # Create multi-index
    dates = pd.date_range('2022-01-01', periods=n_dates)
    stocks = [f'Stock_{i}' for i in range(n_stocks)]
    
    # Returns
    returns_data = np.random.normal(0.0005, 0.02, (n_dates, n_stocks))
    returns = pd.DataFrame(returns_data, index=dates, columns=stocks)
    
    # Characteristics (momentum, value, volatility)
    char_data = np.random.normal(0, 1, (n_dates * n_stocks, n_features))
    multi_idx = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
    characteristics = pd.DataFrame(
        char_data,
        index=multi_idx,
        columns=['momentum', 'value', 'volatility']
    )
    
    # Fit Fama-MacBeth
    fm = FamaMacBeth()
    fm.fit(returns, characteristics)
    
    print("="*70)
    print("FAMA-MACBETH RESULTS")
    print("="*70)
    print(fm.summary())
    
    print("\n" + "="*70)
    print("DAILY FACTOR EXPOSURES (First 10 Days)")
    print("="*70)
    daily_coefs = fm.get_daily_coefficients()
    print(daily_coefs.head(10))
```

## [VISUALIZATION]

Plot time-varying factor exposures:

```python
def plot_fama_macbeth_coefficients(fm: FamaMacBeth, feature_names: List[str] = None):
    """Plot time-varying Fama-MacBeth factor exposures."""
    if feature_names is None:
        feature_names = [f'Factor_{i}' for i in range(fm.n_features)]
    
    fig, axes = plt.subplots(fm.n_features, 1, figsize=(14, 3*fm.n_features))
    if fm.n_features == 1:
        axes = [axes]
    
    for j in range(fm.n_features):
        ax = axes[j]
        ax.plot(fm.daily_betas[:, j], label='Daily', alpha=0.7)
        ax.axhline(y=fm.avg_betas[j], color='r', linestyle='--', label='Average', linewidth=2)
        ax.fill_between(
            np.arange(fm.n_dates),
            fm.avg_betas[j] - 2*fm.se_betas[j],
            fm.avg_betas[j] + 2*fm.se_betas[j],
            alpha=0.2, color='r'
        )
        ax.set_ylabel(feature_names[j])
        ax.set_title(f'{feature_names[j]} Exposure over Time')
        ax.grid()
        ax.legend()
    
    plt.tight_layout()
    plt.show()
```

## WARNING: Survivorship Bias in Panel Data

Your NSE dataset likely excludes delisted stocks. This introduces **survivorship bias**: survivors tend to be winners, losers have already exited.

**Fix**: Use NSE's historical universe list, which includes delisted stocks (if available from Zerodha).

---

## Exercises

**Exercise 7.3.1**: Load NSE panel data (e.g., Nifty 50 stocks, last 2 years). Implement Fama-MacBeth predicting daily returns using:
- Lagged momentum (returns over past 20 days)
- Log valuation (P/E ratio)
- Log size (market cap)

Compare time-varying exposures across 3 periods: pre-COVID, COVID, post-COVID.

**Exercise 7.3.2**: Implement panel fixed-effects regression on the same data. Include fixed effects for:
- Stock (accounts for persistent stock-specific alpha)
- Time (accounts for daily market-wide returns)

Use double-clustering (by stock and time). Are results similar to Fama-MacBeth?

**Exercise 7.3.3**: What happens to your Fama-MacBeth results if you only include stocks with price > Rs 100 (selection bias)? How does this differ from results using full universe?

---

## Summary: Chapter 7

| Technique | When to Use | Pros | Cons |
|-----------|------------|------|------|
| **OLS (Robust SE)** | Single stock or few features | Simple, interpretable | Assumes linear relationship |
| **Ridge** | Many correlated features | Stable, handles multicollinearity | Keeps all features, biased |
| **Lasso** | Feature selection needed | Sparse, automatic selection | Unstable with correlated features |
| **Elastic Net** | High-dim correlated features | Combines Lasso + Ridge benefits | Hyperparameter tuning needed |
| **Fama-MacBeth** | Predicting cross-sectional returns | Powerful, accounts for time-variation | Requires panel structure |
| **Panel FE** | Controlling for unit effects | Removes time-invariant confounds | Can't estimate time-invariant effects |

---

## Key Takeaways

1. **Financial data violates OLS assumptions**. Always use robust standard errors (White HC or Newey-West HAC).

2. **Regularization prevents overfitting** in high dimensions. Use time-series CV (not random CV) to tune parameters.

3. **Cross-sectional models are more powerful** for return prediction because they use all stocks on each day, not just one stock's history.

4. **Fama-MacBeth** is the workhorse of quantitative finance because it accounts for time-varying risk premia and panel structure.

5. **Cluster your standard errors** when data has hierarchical structure (stocks within markets, observations within time periods).

---

## Code Repository

All code in this chapter is production-grade and available at:
```
/notebooks/Chapter_07_complete_code.py
```

Run with:
```bash
python Chapter_07_complete_code.py
```

---

## Further Reading

- **Fama, E. F., & MacBeth, J. D. (1973).** Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636.
- **Newey, W. K., & West, K. D. (1987).** A simple positive semi-definite heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- **Hastie, T., Tibshirani, R., & Wainwright, M. (2015).** *Statistical Learning with Sparsity*. CRC Press.
- **Aronson, D. (2007).** *Evidence-Based Technical Analysis*. Wiley. (For cross-validation in finance)

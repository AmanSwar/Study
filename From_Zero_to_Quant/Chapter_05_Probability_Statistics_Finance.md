# Chapter 5: Probability and Statistics for Finance

## Chapter Overview

This chapter bridges the gap between classical statistics and financial applications. As a system engineer or ML practitioner moving into quantitative finance, you already understand probability at a foundational level, but financial data has peculiar characteristics that violate standard statistical assumptions: extreme returns occur far more frequently than theory predicts, correlations between assets evaporate during crises, and ordinary regression assumptions fail spectacularly in practice.

This chapter equips you with:

1. **Distribution theory** specific to financial returns (not just the normal distribution)
2. **Estimation and inference** methods that work with real, messy market data
3. **Dependence structures** beyond simple Pearson correlation
4. **Bayesian frameworks** that leverage domain knowledge and adapt as markets change

By the end, you'll understand why your standard ML toolkit needs recalibration for finance and how to apply robust statistical methods to build reliable trading systems.

**Learning Outcomes:**
- Identify and fit appropriate distributions to financial data
- Perform rigorous hypothesis tests with corrections for multiple testing
- Estimate correlations that don't collapse during market stress
- Build Bayesian models that learn from market regimes
- Implement bootstrap and resampling methods for confidence intervals on trading metrics

---

## Prerequisites

**Required Knowledge:**
- Basic probability: random variables, PDFs, CDFs, expected value, variance
- Linear algebra: matrices, eigenvalues, matrix norms
- Python: NumPy, SciPy, Pandas (intermediate level)
- Chapter 3: Basic time series concepts
- Chapter 4: Risk metrics fundamentals

**Assumed Tools:**
- NumPy, SciPy, Pandas, Matplotlib, Seaborn
- `scipy.stats` for distributions
- `statsmodels` for regression and statistical tests

---

# Module 5.1: Distributions in Finance

## 5.1.1 The Problem with the Normal Distribution

The normal (Gaussian) distribution is ubiquitous in finance: Black-Scholes assumes it, value-at-risk calculations use it, portfolio theory builds on it. Yet real financial returns are **not normally distributed**.

### Mathematical Foundation

The normal distribution is defined by its PDF:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Key Properties:**
- Fully characterized by two parameters: mean $\mu$ and standard deviation $\sigma$
- Symmetric around the mean (zero skewness)
- Thin tails: probability of observing values beyond $\pm 3\sigma$ is only 0.27%
- Closed under linear combinations (essential for portfolio theory)

### Why Finance Breaks the Normal Assumption

Real stock returns violate normality in systematic ways:

1. **Fat Tails (Leptokurtosis)**: Extreme moves occur 10-100x more frequently than normal distribution predicts
   - S&P 500 October 19, 1987: -22% in one day (36 standard deviations under normal!)
   - COVID-19 crash March 2020: -12% daily moves, repeated over days

2. **Skewness**: Many return distributions are negatively skewed
   - Puts are more expensive than calls (skew smile in options markets)
   - Large losses are more likely than large gains of equal magnitude

3. **Heteroskedasticity**: Volatility is not constant
   - Volatility clustering: calm periods followed by turbulent periods
   - Leverage effect: negative returns increase future volatility

### Empirical Evidence: Testing Normality

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

def load_financial_returns(ticker: str, period: str = "10y") -> pd.Series:
    """
    Load daily returns for a given ticker.
    
    Args:
        ticker: Stock ticker (e.g., 'AAPL', '^GSPC')
        period: Time period ('1y', '5y', '10y', 'max')
    
    Returns:
        Series of daily log returns
    """
    data = yf.download(ticker, period=period, progress=False)
    returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
    return returns


def test_normality(returns: pd.Series) -> dict:
    """
    Test whether returns are normally distributed using multiple tests.
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with test statistics and p-values
    """
    results = {}
    
    # Shapiro-Wilk test (best for small samples)
    if len(returns) <= 5000:
        sw_stat, sw_pval = stats.shapiro(returns.dropna())
        results['shapiro_wilk'] = {'statistic': sw_stat, 'p_value': sw_pval}
    
    # Jarque-Bera test (specifically for skewness + kurtosis)
    jb_stat, jb_pval = stats.jarque_bera(returns.dropna())
    results['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pval}
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(returns.dropna(), 'norm', 
                                     args=(returns.mean(), returns.std()))
    results['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_pval}
    
    # Anderson-Darling test
    ad_result = stats.anderson(returns.dropna(), dist='norm')
    results['anderson_darling'] = {'statistic': ad_result.statistic, 
                                   'critical_values': ad_result.critical_values}
    
    return results


# Example usage
returns = load_financial_returns('^GSPC')  # S&P 500
normality_tests = test_normality(returns)

print(f"Testing normality for {len(returns)} observations:")
print(f"Jarque-Bera p-value: {normality_tests['jarque_bera']['p_value']:.2e}")
print(f"Kolmogorov-Smirnov p-value: {normality_tests['kolmogorov_smirnov']['p_value']:.2e}")
print(f"\nAll tests strongly reject normality hypothesis.")
```

**Typical Findings:**
- P-values << 0.001 (reject normality with 99.99% confidence)
- Excess kurtosis 3-10 (normal has kurtosis of 3)
- Skewness -0.5 to -1.0 for equities (left-tailed)

---

## 5.1.2 Fat-Tailed Distributions

### Student's t-Distribution

The Student's t-distribution with $\nu$ (nu) degrees of freedom provides fatter tails than normal:

$$f(x|\nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

**Key Properties:**
- As $\nu \to \infty$, becomes normal distribution
- Smaller $\nu$ = fatter tails
- For $\nu < 4$, kurtosis is infinite (undefined tail risk!)
- $\nu \in [3, 10]$ works well for equity returns

**Why t-distribution for finance:**
- Parsimonious: just one extra parameter ($\nu$)
- Better tail fit than normal without requiring complex mixture models
- Allows t-tests without normality assumption

### Fitting a t-Distribution to Returns

```python
from scipy.stats import t as t_dist

def fit_t_distribution(returns: pd.Series) -> dict:
    """
    Fit a Student's t-distribution to returns using MLE.
    
    Args:
        returns: Series of daily returns
    
    Returns:
        Dictionary with fitted parameters and diagnostics
    """
    returns_clean = returns.dropna().values
    
    # MLE fit (scipy optimizes internally)
    df, loc, scale = t_dist.fit(returns_clean)
    
    # Calculate diagnostics
    fitted_params = {
        'degrees_of_freedom': df,
        'location': loc,
        'scale': scale,
    }
    
    # Compare with normal distribution
    normal_mean = returns_clean.mean()
    normal_std = returns_clean.std()
    
    # Compute tail probabilities
    # Probability of return <= -2% (1 in N years?)
    threshold = -0.02
    
    # t-distribution probability
    t_prob = t_dist.cdf(threshold, df, loc, scale)
    
    # Normal probability
    normal_prob = stats.norm.cdf(threshold, normal_mean, normal_std)
    
    diagnostics = {
        't_tail_prob': t_prob,
        'normal_tail_prob': normal_prob,
        'tail_ratio': t_prob / normal_prob,  # How many times worse?
    }
    
    return {**fitted_params, **diagnostics}


# Fit to S&P 500 returns
returns = load_financial_returns('^GSPC')
params = fit_t_distribution(returns)

print(f"Fitted t-distribution parameters:")
print(f"  Degrees of freedom: {params['degrees_of_freedom']:.1f}")
print(f"  Location: {params['location']:.6f}")
print(f"  Scale: {params['scale']:.6f}")
print(f"\nTail probability at -2%:")
print(f"  t-distribution: {params['t_tail_prob']:.4f}")
print(f"  Normal dist:    {params['normal_tail_prob']:.4f}")
print(f"  Ratio (t/normal): {params['tail_ratio']:.1f}x")
```

### Stable Distributions (Lévy Processes)

Stable distributions are closed under addition: if $X_1, X_2, \ldots, X_n$ are i.i.d. stable, then $\sum X_i$ is also stable (with different parameters). This property is mathematically elegant but requires parameterization by characteristic function:

$$\phi(t) = \exp(i\mu t - \sigma^\alpha |t|^\alpha (1 + i\beta \text{sign}(t) \omega(\alpha, t)))$$

**Parameters:**
- $\alpha \in (0, 2]$: tail thickness (2 = normal, smaller = fatter)
- $\beta \in [-1, 1]$: skewness
- $\sigma > 0$: scale
- $\mu$: location

**Why stable distributions for finance:**
- Match power-law behavior of returns (tail probabilities decay as $P(X > x) \sim x^{-\alpha}$)
- Natural model for "cascade" failure in markets

**Limitation:** Fitting is numerically challenging; use t-distribution for practical purposes.

---

## 5.1.3 Skewness and Kurtosis

### Definition and Interpretation

**Skewness** (3rd standardized moment):

$$\text{Skew} = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$$

- Positive skew: right tail longer than left (upside bias)
- Negative skew: left tail longer (crash risk) - typical for equities
- Equity skew $\approx -0.5$ to $-1.0$; options markets price in more negative skew

**Kurtosis** (4th standardized moment):

$$\text{Kurt} = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^4\right]$$

**Excess Kurtosis** (kurtosis - 3):
- Normal distribution: excess kurtosis = 0
- Positive excess kurtosis: fat tails (market crashes)
- Equity returns typically show excess kurtosis 3-8

### Financial Interpretation

Skewness and kurtosis directly impact risk metrics:

1. **Skewness impact on VaR**: Negative skew means 99% VaR underestimates tail losses
2. **Kurtosis impact on crisis probability**: High kurtosis means catastrophic losses are more probable than normal model predicts
3. **Options pricing**: Options markets charge skew premium (puts more expensive than calls)

### Computing Skewness and Kurtosis

```python
def analyze_distribution_moments(returns: pd.Series) -> dict:
    """
    Analyze the distribution of returns via moments.
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with moments and their interpretation
    """
    ret_clean = returns.dropna().values
    
    results = {
        'mean': np.mean(ret_clean),
        'std': np.std(ret_clean, ddof=1),
        'skewness': stats.skew(ret_clean),
        'excess_kurtosis': stats.kurtosis(ret_clean),
    }
    
    # Standard errors (for testing significance)
    n = len(ret_clean)
    results['skew_se'] = np.sqrt(6 / n)  # Standard error of skewness
    results['kurt_se'] = np.sqrt(24 / n)  # Standard error of kurtosis
    
    # Test if skewness is significantly different from 0
    results['skew_t_stat'] = results['skewness'] / results['skew_se']
    results['skew_p_value'] = 2 * (1 - stats.t.cdf(abs(results['skew_t_stat']), n))
    
    # Test if kurtosis is significantly different from 0
    results['kurt_t_stat'] = results['excess_kurtosis'] / results['kurt_se']
    results['kurt_p_value'] = 2 * (1 - stats.t.cdf(abs(results['kurt_t_stat']), n))
    
    return results


returns = load_financial_returns('^GSPC')
moments = analyze_distribution_moments(returns)

print(f"Return Distribution Analysis (S&P 500):")
print(f"  Mean: {moments['mean']:.4%}")
print(f"  Std Dev: {moments['std']:.4%}")
print(f"  Skewness: {moments['skewness']:.3f} (p={moments['skew_p_value']:.2e})")
print(f"  Excess Kurtosis: {moments['excess_kurtosis']:.3f} (p={moments['kurt_p_value']:.2e})")
```

---

## 5.1.4 QQ Plots: Visualizing Distributional Mismatch

A Q-Q (quantile-quantile) plot compares quantiles of observed data against a theoretical distribution. If the data follows the theoretical distribution, points lie on the diagonal.

### Interpreting QQ Plots

```python
def create_qq_plot(returns: pd.Series, title: str = "Q-Q Plot") -> None:
    """
    Create a Q-Q plot comparing returns to a normal distribution.
    Shows how real data deviates from normality.
    
    Args:
        returns: Series of returns
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ret_clean = returns.dropna().values
    
    # Standard Q-Q plot (data vs normal)
    stats.probplot(ret_clean, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot: S&P 500 Returns vs Normal Distribution")
    axes[0].set_xlabel("Theoretical Quantiles")
    axes[0].set_ylabel("Sample Quantiles")
    axes[0].grid(True, alpha=0.3)
    
    # Highlight tail mismatch with color
    ax2 = axes[1]
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(ret_clean)))
    sample_quantiles = np.percentile(ret_clean, np.linspace(1, 99, len(ret_clean)))
    
    # Color code: blue for central region, red for tails
    colors = np.where(np.abs(theoretical_quantiles) > 2, 'red', 'blue')
    ax2.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, c=colors, s=20)
    
    # Add diagonal
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),
        np.max([ax2.get_xlim(), ax2.get_ylim()]),
    ]
    ax2.plot(lims, lims, 'k-', alpha=0.3, label='Perfect fit')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_title("Q-Q Plot (Tail Deviation Highlighted)")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nQ-Q Plot Interpretation:")
    print("- Points on diagonal = data matches normal distribution")
    print("- Points above diagonal in right tail = fatter right tail than normal")
    print("- Points below diagonal in right tail = thinner right tail than normal")
    print("- RED points show deviations beyond ±2σ (highly significant)")


# Create Q-Q plot for S&P 500
returns = load_financial_returns('^GSPC')
create_qq_plot(returns)
```

**Interpretation Guide:**
- **Center (±1σ)**: Usually follows normal well
- **Right tail (>2σ)**: Points below line = market crashes more often than expected
- **Left tail (<-2σ)**: Points above line = extreme gains rarer than expected

[VISUALIZATION: QQ plot of real stock returns vs normal showing tail deviation]

---

## 5.1.5 Mixture of Normals: Regime-Dependent Returns

Financial markets operate in different regimes (bull, bear, high volatility, low volatility). A mixture of normals model captures this:

$$f(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)$$

Where $\pi_k$ are mixture weights ($\sum \pi_k = 1$).

### Fitting a Mixture Model

```python
from scipy.optimize import minimize
from scipy.stats import norm

def fit_gaussian_mixture(returns: pd.Series, n_components: int = 2) -> dict:
    """
    Fit a Gaussian mixture model (GMM) to returns using EM algorithm.
    
    Args:
        returns: Series of returns
        n_components: Number of mixture components (regimes)
    
    Returns:
        Dictionary with fitted parameters
    """
    from sklearn.mixture import GaussianMixture
    
    ret_clean = returns.dropna().values.reshape(-1, 1)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(ret_clean)
    
    # Extract parameters
    params = {
        'weights': gmm.weights_,
        'means': gmm.means_.flatten(),
        'stds': np.sqrt(gmm.covariances_.flatten()),
        'bic': gmm.bic(ret_clean),
        'aic': gmm.aic(ret_clean),
    }
    
    return params, gmm


def plot_mixture_model(returns: pd.Series, gmm, params: dict) -> None:
    """
    Plot histogram of returns with fitted mixture model overlay.
    
    Args:
        returns: Series of returns
        gmm: Fitted GaussianMixture object
        params: Parameters dictionary from fit_gaussian_mixture
    """
    ret_clean = returns.dropna().values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    ax.hist(ret_clean, bins=100, density=True, alpha=0.5, label='Empirical')
    
    # Fitted mixture
    x = np.linspace(ret_clean.min(), ret_clean.max(), 1000)
    
    # Overall mixture density
    mixture_density = np.zeros_like(x)
    for k in range(len(params['weights'])):
        component_density = params['weights'][k] * norm.pdf(
            x, params['means'][k], params['stds'][k]
        )
        mixture_density += component_density
        ax.plot(x, component_density, '--', alpha=0.7, 
                label=f"Regime {k+1} ({params['weights'][k]:.1%})")
    
    ax.plot(x, mixture_density, 'r-', linewidth=2, label='Mixture')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Density')
    ax.set_title('Gaussian Mixture Model: Dual-Regime Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nMixture Model Results:")
    for k in range(len(params['weights'])):
        print(f"  Regime {k+1}:")
        print(f"    Weight: {params['weights'][k]:.1%}")
        print(f"    Mean: {params['means'][k]:.4%}")
        print(f"    Std: {params['stds'][k]:.4%}")


# Fit 2-component mixture to S&P 500
returns = load_financial_returns('^GSPC')
params, gmm = fit_gaussian_mixture(returns, n_components=2)
plot_mixture_model(returns, gmm, params)
```

**Financial Interpretation:**
- Component 1: "Normal times" (low volatility, small returns)
- Component 2: "Crisis" (high volatility, large negative returns)
- Switching between components explains clustering and non-normality
- More sophisticated: use Hidden Markov Model to model regime transitions

---

## 5.1.6 Practical Guidance: Choosing a Distribution

| Distribution | Best For | Pros | Cons |
|---|---|---|---|
| **Normal** | Teaching only | Simple, closed-form | Massive underestimates tail risk |
| **t-distribution** | Equity returns | Parsimonious, handles tails | Still misses skew |
| **Mixture of normals** | Intraday/short-term | Captures regimes | More parameters to estimate |
| **Stable (α-stable)** | Theoretical purity | Mathematically elegant | Fitting is hard, no mean/variance |

**Recommendation for practitioners:** Use t-distribution for standard modeling, mixture of normals for regime detection, empirical/historical for VaR.

---

## Module 5.1 Summary

- Real financial returns have **fatter tails** and **negative skew** than normal distribution
- **Student's t-distribution** provides practical alternative with one extra parameter
- **Skewness and kurtosis** capture higher-order distributional features critical for risk
- **Q-Q plots** visualize distributional mismatch, especially in tails
- **Mixture models** capture regime switching behavior
- Choose distributions **empirically** not theoretically

---

# Module 5.2: Statistical Estimation and Inference

## 5.2.1 Maximum Likelihood Estimation in Finance

Maximum Likelihood Estimation (MLE) is the workhorse for fitting distributions to financial data. Given observations $X_1, \ldots, X_n$, the MLE finds parameters $\theta$ that maximize the likelihood:

$$\hat{\theta} = \arg\max_\theta L(\theta; X_1, \ldots, X_n) = \arg\max_\theta \prod_{i=1}^n f(X_i | \theta)$$

Working in log space (more stable numerically):

$$\hat{\theta} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \sum_{i=1}^n \log f(X_i | \theta)$$

### Properties of MLEs

1. **Consistency**: $\hat{\theta} \to \theta_0$ as $n \to \infty$ (under regularity conditions)
2. **Asymptotic normality**: $\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} \mathcal{N}(0, I^{-1})$ where $I$ is the Fisher information matrix
3. **Efficiency**: Achieves Cramér-Rao lower bound (best possible variance)

### Implementing MLE for Financial Distributions

```python
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class FinancialDistributionFitter:
    """
    Fit various financial distributions using MLE.
    """
    
    def __init__(self, returns: pd.Series):
        """
        Initialize with return series.
        
        Args:
            returns: Series of returns
        """
        self.returns = returns.dropna().values
        self.n = len(self.returns)
    
    def log_likelihood_normal(self, params: tuple) -> float:
        """
        Log-likelihood for normal distribution.
        
        Args:
            params: (mean, std_dev)
        
        Returns:
            Negative log-likelihood (for minimization)
        """
        mu, sigma = params
        if sigma <= 0:
            return 1e10  # Penalty for invalid std dev
        
        ll = -0.5 * np.sum(((self.returns - mu) / sigma) ** 2)
        ll -= self.n * np.log(sigma)
        ll -= 0.5 * self.n * np.log(2 * np.pi)
        
        return -ll  # Negative for minimization
    
    def log_likelihood_t(self, params: tuple) -> float:
        """
        Log-likelihood for Student's t-distribution.
        
        Args:
            params: (df, mean, scale)
        
        Returns:
            Negative log-likelihood
        """
        df, mu, sigma = params
        if df <= 0 or sigma <= 0:
            return 1e10
        
        x = (self.returns - mu) / sigma
        ll = np.sum(np.log(stats.t.pdf(x, df)))
        
        return -ll
    
    def fit_normal(self) -> dict:
        """Fit normal distribution."""
        x0 = [self.returns.mean(), self.returns.std()]
        
        result = minimize(
            self.log_likelihood_normal,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        mu_hat, sigma_hat = result.x
        
        return {
            'distribution': 'Normal',
            'mu': mu_hat,
            'sigma': sigma_hat,
            'negative_ll': result.fun,
            'aic': 2 * result.fun + 2 * 2,  # 2 parameters
            'bic': 2 * result.fun + 2 * np.log(self.n),
            'success': result.success,
        }
    
    def fit_t(self) -> dict:
        """Fit Student's t-distribution."""
        # Start with normal estimates
        x0 = [5.0, self.returns.mean(), self.returns.std()]
        
        # Bounds: df > 0, sigma > 0
        bounds = [(0.1, 100), (None, None), (1e-6, None)]
        
        result = minimize(
            self.log_likelihood_t,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        df_hat, mu_hat, sigma_hat = result.x
        
        return {
            'distribution': 't-distribution',
            'df': df_hat,
            'mu': mu_hat,
            'sigma': sigma_hat,
            'negative_ll': result.fun,
            'aic': 2 * result.fun + 2 * 3,  # 3 parameters
            'bic': 2 * result.fun + 3 * np.log(self.n),
            'success': result.success,
        }
    
    def compare_fits(self) -> pd.DataFrame:
        """
        Fit both distributions and compare via AIC/BIC.
        
        Returns:
            DataFrame with comparison metrics
        """
        normal_fit = self.fit_normal()
        t_fit = self.fit_t()
        
        # Calculate AIC differences (lower is better)
        aic_diff = normal_fit['aic'] - t_fit['aic']
        bic_diff = normal_fit['bic'] - t_fit['bic']
        
        comparison = pd.DataFrame([normal_fit, t_fit])
        
        print("\n" + "="*60)
        print("DISTRIBUTION COMPARISON")
        print("="*60)
        print(f"\nSample size: {self.n}")
        print(f"\nNormal Distribution:")
        print(f"  μ = {normal_fit['mu']:.6f}")
        print(f"  σ = {normal_fit['sigma']:.6f}")
        print(f"  -LL = {normal_fit['negative_ll']:.2f}")
        print(f"  AIC = {normal_fit['aic']:.2f}")
        print(f"  BIC = {normal_fit['bic']:.2f}")
        
        print(f"\nStudent's t-Distribution:")
        print(f"  df = {t_fit['df']:.2f}")
        print(f"  μ = {t_fit['mu']:.6f}")
        print(f"  σ = {t_fit['sigma']:.6f}")
        print(f"  -LL = {t_fit['negative_ll']:.2f}")
        print(f"  AIC = {t_fit['aic']:.2f}")
        print(f"  BIC = {t_fit['bic']:.2f}")
        
        print(f"\nModel Selection:")
        print(f"  AIC difference (Normal - t): {aic_diff:.2f}")
        print(f"  BIC difference (Normal - t): {bic_diff:.2f}")
        if aic_diff > 10:
            print(f"  → t-distribution is MUCH better (AIC delta > 10)")
        elif aic_diff > 0:
            print(f"  → t-distribution is better")
        else:
            print(f"  → Normal distribution is better (unlikely)")
        
        return comparison


# Fit distributions to real data
returns = load_financial_returns('^GSPC')
fitter = FinancialDistributionFitter(returns)
comparison = fitter.compare_fits()
```

---

## 5.2.2 Hypothesis Testing: t, F, Chi-Squared

### The t-Test: Testing Mean Returns

Is the average daily return significantly different from zero? (Null hypothesis: $\mu = 0$)

$$H_0: \mu = 0 \quad \text{vs} \quad H_1: \mu \neq 0$$

Test statistic:

$$t = \frac{\bar{X}}{\hat{\sigma}/\sqrt{n}} \sim t_{n-1}$$

```python
def test_mean_return(returns: pd.Series, null_mean: float = 0) -> dict:
    """
    Test whether mean return is significantly different from null_mean.
    
    Args:
        returns: Series of returns
        null_mean: Hypothesized mean (default 0)
    
    Returns:
        Dictionary with test results
    """
    ret_clean = returns.dropna().values
    n = len(ret_clean)
    
    # Sample mean and std
    sample_mean = ret_clean.mean()
    sample_std = ret_clean.std(ddof=1)
    
    # Standard error of the mean
    sem = sample_std / np.sqrt(n)
    
    # t-statistic
    t_stat = (sample_mean - null_mean) / sem
    
    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    
    # 95% confidence interval
    ci_lower = sample_mean - stats.t.ppf(0.975, n - 1) * sem
    ci_upper = sample_mean + stats.t.ppf(0.975, n - 1) * sem
    
    results = {
        'n': n,
        'sample_mean': sample_mean,
        'sample_std': sample_std,
        'sem': sem,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant_at_5pct': p_value < 0.05,
        'significant_at_1pct': p_value < 0.01,
    }
    
    return results


# Test S&P 500 mean return
returns = load_financial_returns('^GSPC')
test_result = test_mean_return(returns)

print(f"Mean Return Test (S&P 500)")
print(f"H0: mean return = 0")
print(f"Sample mean: {test_result['sample_mean']:.4%}")
print(f"Standard error: {test_result['sem']:.6%}")
print(f"t-statistic: {test_result['t_statistic']:.3f}")
print(f"p-value: {test_result['p_value']:.4f}")
print(f"95% CI: [{test_result['ci_lower']:.4%}, {test_result['ci_upper']:.4%}]")
print(f"Significant at 5%? {test_result['significant_at_5pct']}")
```

### The F-Test: Comparing Volatilities

Are two assets equally volatile? Null: $\sigma_1^2 = \sigma_2^2$

$$F = \frac{s_1^2}{s_2^2} \sim F_{n_1-1, n_2-1}$$

```python
def test_equal_variance(returns1: pd.Series, returns2: pd.Series) -> dict:
    """
    Test whether two return series have equal variance (Levene's test).
    
    Args:
        returns1: First return series
        returns2: Second return series
    
    Returns:
        Dictionary with test results
    """
    r1_clean = returns1.dropna().values
    r2_clean = returns2.dropna().values
    
    # Levene's test (robust to non-normality)
    stat, p_value = stats.levene(r1_clean, r2_clean)
    
    # Also report classic F-test
    var1 = r1_clean.var(ddof=1)
    var2 = r2_clean.var(ddof=1)
    f_stat = var1 / var2
    f_p_value = 2 * min(stats.f.cdf(f_stat, len(r1_clean)-1, len(r2_clean)-1),
                        1 - stats.f.cdf(f_stat, len(r1_clean)-1, len(r2_clean)-1))
    
    return {
        'levene_statistic': stat,
        'levene_p_value': p_value,
        'f_statistic': f_stat,
        'f_p_value': f_p_value,
        'var1': var1,
        'var2': var2,
        'equal_variance_5pct': p_value > 0.05,
    }


# Compare S&P 500 vs Nasdaq volatility
sp500 = load_financial_returns('^GSPC')
nasdaq = load_financial_returns('^IXIC')
var_test = test_equal_variance(sp500, nasdaq)

print(f"Equal Variance Test")
print(f"Levene test p-value: {var_test['levene_p_value']:.4f}")
print(f"S&P 500 variance: {var_test['var1']:.6%}")
print(f"Nasdaq variance: {var_test['var2']:.6%}")
```

---

## 5.2.3 Multiple Testing Corrections

When conducting many statistical tests, false positive rate explodes:

**Family-Wise Error Rate (FWER):** Probability of at least one false positive among $m$ tests

$$\text{FWER} = 1 - (1 - \alpha)^m$$

For $m=100$ tests and $\alpha = 0.05$: FWER = 99.4% (virtually certain to have at least one false discovery!)

### Bonferroni Correction

Simplest approach: adjust significance level

$$\alpha_{\text{adj}} = \frac{\alpha}{m}$$

**Problem:** Very conservative; loses power.

### Benjamini-Hochberg (FDR) Control

More powerful: control False Discovery Rate (expected proportion of false discoveries among rejections)

$$\text{FDR} = \mathbb{E}\left[\frac{\# \text{ False Positives}}{\# \text{ Rejections}}\right]$$

Algorithm:
1. Compute $m$ p-values, sort: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find largest $i$ such that $p_{(i)} \leq \frac{i}{m} \alpha$
3. Reject all hypotheses $1, \ldots, i$

```python
def multiple_testing_correction(p_values: np.ndarray, method: str = 'BH') -> dict:
    """
    Correct p-values for multiple testing.
    
    Args:
        p_values: Array of p-values
        method: 'bonferroni' or 'BH' (Benjamini-Hochberg)
    
    Returns:
        Dictionary with adjusted p-values and decisions
    """
    m = len(p_values)
    alpha = 0.05
    
    if method == 'bonferroni':
        # Bonferroni: divide by number of tests
        adjusted_p = np.minimum(p_values * m, 1.0)
        threshold = alpha / m
        reject = adjusted_p < threshold
        
    elif method == 'BH':
        # Benjamini-Hochberg: more powerful
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Find largest i such that p_(i) <= i/m * alpha
        i_max = 0
        for i in range(m - 1, -1, -1):
            if sorted_p[i] <= (i + 1) / m * alpha:
                i_max = i
                break
        
        # Reject first i_max+1 hypotheses
        reject = np.zeros(m, dtype=bool)
        reject[sorted_idx[:i_max + 1]] = True
        adjusted_p = np.ones(m) * np.nan
        
        threshold = (i_max + 1) / m * alpha if i_max >= 0 else 0
    
    return {
        'method': method,
        'n_tests': m,
        'n_rejected': np.sum(reject),
        'threshold': threshold,
        'adjusted_p_values': adjusted_p if method == 'bonferroni' else None,
        'reject': reject,
        'rejection_rate': np.sum(reject) / m,
    }


# Example: Testing whether each of 100 assets has positive alpha
np.random.seed(42)
n_assets = 100
# Generate p-values: 95% are null (random), 5% are significant
p_values = np.concatenate([
    stats.uniform.rvs(size=95),  # Null: uniform p-values
    stats.beta.rvs(0.5, 2, size=5)  # Signal: biased toward 0
])

bonferroni_result = multiple_testing_correction(p_values, method='bonferroni')
bh_result = multiple_testing_correction(p_values, method='BH')

print(f"Multiple Testing Correction (100 tests)")
print(f"\nBonferroni:")
print(f"  Rejections: {bonferroni_result['n_rejected']}")
print(f"  Threshold: {bonferroni_result['threshold']:.2e}")

print(f"\nBenjamini-Hochberg (FDR):")
print(f"  Rejections: {bh_result['n_rejected']}")
print(f"  Threshold: {bh_result['threshold']:.4f}")
print(f"\nInterpretation: BH finds more true positives at controlled false discovery rate")
```

---

## 5.2.4 Bootstrap Methods: Confidence Intervals without Normality

The bootstrap is a resampling method that doesn't assume normality. Core idea:

1. Sample $n$ returns **with replacement** from observed data
2. Compute statistic of interest (e.g., Sharpe ratio)
3. Repeat steps 1-2 many times (1000-10000)
4. Use empirical distribution of bootstrap replicates to construct confidence intervals

### Non-parametric Bootstrap for Sharpe Ratios

```python
def bootstrap_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.0,
                          n_bootstrap: int = 10000) -> dict:
    """
    Compute Sharpe ratio and 95% confidence interval via bootstrap.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Daily risk-free rate
        n_bootstrap: Number of bootstrap resamples
    
    Returns:
        Dictionary with Sharpe ratio estimate and CI
    """
    ret_clean = returns.dropna().values
    n = len(ret_clean)
    
    # Observed Sharpe ratio
    mean_ret = ret_clean.mean()
    std_ret = ret_clean.std(ddof=1)
    sharpe_observed = (mean_ret - risk_free_rate) / std_ret * np.sqrt(252)  # Annualize
    
    # Bootstrap replicates
    sharpe_bootstrap = np.zeros(n_bootstrap)
    np.random.seed(42)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        boot_returns = ret_clean[idx]
        
        # Compute Sharpe on bootstrap sample
        boot_mean = boot_returns.mean()
        boot_std = boot_returns.std(ddof=1)
        if boot_std > 0:
            sharpe_bootstrap[i] = (boot_mean - risk_free_rate) / boot_std * np.sqrt(252)
        else:
            sharpe_bootstrap[i] = np.nan
    
    # Remove NaNs
    sharpe_bootstrap = sharpe_bootstrap[~np.isnan(sharpe_bootstrap)]
    
    # Confidence intervals: percentile method
    ci_lower = np.percentile(sharpe_bootstrap, 2.5)
    ci_upper = np.percentile(sharpe_bootstrap, 97.5)
    
    # Standard error
    se = sharpe_bootstrap.std()
    
    results = {
        'sharpe_observed': sharpe_observed,
        'sharpe_mean_bootstrap': sharpe_bootstrap.mean(),
        'sharpe_std_bootstrap': sharpe_bootstrap.std(),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'bootstrap_replicates': sharpe_bootstrap,
    }
    
    return results


# Bootstrap Sharpe ratio for S&P 500
returns = load_financial_returns('^GSPC')
boot_result = bootstrap_sharpe_ratio(returns)

print(f"Bootstrap Sharpe Ratio Analysis")
print(f"Observed Sharpe: {boot_result['sharpe_observed']:.3f}")
print(f"Bootstrap mean: {boot_result['sharpe_mean_bootstrap']:.3f}")
print(f"Bootstrap std: {boot_result['sharpe_std_bootstrap']:.3f}")
print(f"95% CI: [{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]")
print(f"CI width: {boot_result['ci_width']:.3f}")

# Plot bootstrap distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(boot_result['bootstrap_replicates'], bins=50, density=True, alpha=0.7)
ax.axvline(boot_result['sharpe_observed'], color='r', linestyle='--', 
           linewidth=2, label='Observed')
ax.axvline(boot_result['ci_lower'], color='g', linestyle='--', label='95% CI')
ax.axvline(boot_result['ci_upper'], color='g', linestyle='--')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Density')
ax.set_title('Bootstrap Distribution of Sharpe Ratio')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5.2.5 Newey-West Estimator: Handling Autocorrelation

Standard error formulas assume i.i.d. observations. Financial returns violate this:
- Volatility clusters (heteroskedasticity)
- Returns autocorrelated at short frequencies
- These violate OLS assumptions

The Newey-West estimator corrects for both autocorrelation and heteroskedasticity:

$$\hat{\text{Var}}(b) = (X'X)^{-1} \left[ S_0 + \sum_{j=1}^{m} w_j (S_j + S_j') \right] (X'X)^{-1}$$

Where:
- $S_j = \sum_{t=j+1}^n \hat{u}_t \hat{u}_{t-j} x_t x_{t-j}'$ (autocorrelation terms)
- $w_j = 1 - \frac{j}{m+1}$ (Bartlett weights)
- $m$ is lag cutoff (usually $\lfloor 4(n/100)^{2/9} \rfloor$)

### Implementation

```python
def newey_west_regression(y: np.ndarray, X: np.ndarray, 
                         lags: int = None) -> dict:
    """
    Perform OLS regression with Newey-West standard errors.
    
    Args:
        y: Dependent variable (n,)
        X: Regressors (n, k) - should include constant
        lags: Number of lags for HAC correction (auto if None)
    
    Returns:
        Dictionary with coefficients, std errors, t-stats, p-values
    """
    n, k = X.shape
    
    # Auto lag selection if not provided
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    
    # OLS estimation
    b_ols = np.linalg.solve(X.T @ X, X.T @ y)
    residuals = y - X @ b_ols
    
    # Long-run variance (with HAC correction)
    # Compute S_0 (contemporaneous variance)
    S_0 = (X.T * residuals) @ X / n
    
    # Compute lagged covariances
    S_lag = np.zeros((k, k))
    for j in range(1, lags + 1):
        X_lag = X[:-j, :]
        X_lead = X[j:, :]
        u_lag = residuals[:-j]
        u_lead = residuals[j:]
        
        # Bartlett weights
        weight = 1 - j / (lags + 1)
        
        S_j = (X_lag.T * u_lag) @ X_lead / n
        S_lag += weight * (S_j + S_j.T)
    
    # HAC variance
    XTX_inv = np.linalg.inv(X.T @ X)
    var_hac = XTX_inv @ (S_0 + S_lag) @ XTX_inv
    
    # Standard errors
    se = np.sqrt(np.diag(var_hac))
    
    # t-statistics and p-values
    t_stats = b_ols / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
    
    return {
        'coefficients': b_ols,
        'std_errors': se,
        't_statistics': t_stats,
        'p_values': p_values,
        'var_matrix': var_hac,
        'lags_used': lags,
        'n_obs': n,
    }


# Example: Regress stock return on market return with NW correction
sp500 = load_financial_returns('^GSPC')
apple = load_financial_returns('AAPL')

# Align dates
common_dates = sp500.index.intersection(apple.index)
y = apple[common_dates].values
X = np.column_stack([np.ones(len(common_dates)), sp500[common_dates].values])

# OLS with Newey-West
nw_result = newey_west_regression(y, X)

print(f"Alpha (AAPL) = {nw_result['coefficients'][0]:.6f}")
print(f"  NW Std Err = {nw_result['std_errors'][0]:.6f}")
print(f"  t-stat = {nw_result['t_statistics'][0]:.3f}")
print(f"  p-value = {nw_result['p_values'][0]:.4f}")
print(f"\nBeta (AAPL) = {nw_result['coefficients'][1]:.4f}")
print(f"  NW Std Err = {nw_result['std_errors'][1]:.4f}")
print(f"  t-stat = {nw_result['t_statistics'][1]:.3f}")
print(f"  p-value = {nw_result['p_values'][1]:.4f}")
```

---

## Module 5.2 Summary

- **MLE** is standard for fitting distributions; compare via AIC/BIC
- **t-tests, F-tests, chi-squared** for hypothesis testing on returns
- **Multiple testing corrections** (Bonferroni, FDR) essential when testing many strategies
- **Bootstrap** provides non-parametric confidence intervals without normality assumption
- **Newey-West** standard errors correct for autocorrelation and heteroskedasticity

---

# Module 5.3: Correlation and Dependence

## 5.3.1 Correlation Measures: Pearson, Spearman, Kendall

### Pearson Correlation

Most common; measures linear dependence:

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]}{\sigma_X \sigma_Y}$$

**Advantages:** Easy to interpret, efficient (uses full information)
**Disadvantages:** Sensitive to outliers, only captures linear dependence

### Spearman Correlation

Rank-based; measures monotonic dependence:

$$\rho_S = \text{Corr}(\text{rank}(X), \text{rank}(Y))$$

**Advantages:** Robust to outliers and nonlinear relationships
**Disadvantages:** Loses information by using ranks

### Kendall's Tau

Measures concordance (probability that pairs are in same order):

$$\tau = \frac{\# \text{ concordant pairs} - \# \text{ discordant pairs}}{\binom{n}{2}}$$

**Advantages:** More robust interpretation of dependence, especially with ties
**Disadvantages:** Computationally more expensive

### Implementation and Comparison

```python
def compare_correlations(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson, Spearman, and Kendall correlations.
    
    Args:
        returns_df: DataFrame of returns
    
    Returns:
        List of correlation matrices
    """
    results = {}
    
    # Pearson
    pearson = returns_df.corr(method='pearson')
    results['Pearson'] = pearson
    
    # Spearman
    spearman = returns_df.corr(method='spearman')
    results['Spearman'] = spearman
    
    # Kendall
    kendall = returns_df.corr(method='kendall')
    results['Kendall'] = kendall
    
    return results


# Load multiple assets
tickers = ['^GSPC', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
returns_dict = {t: load_financial_returns(t, period='5y') for t in tickers}
returns_df = pd.DataFrame(returns_dict)

# Compare correlations
corr_methods = compare_correlations(returns_df)

print("Correlation Comparison (5-year S&P 500 components):")
for method, corr_matrix in corr_methods.items():
    print(f"\n{method} Correlation (AAPL-MSFT):")
    print(f"  {corr_matrix.loc['AAPL', 'MSFT']:.4f}")

# Visualize differences
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (method, corr) in enumerate(corr_methods.items()):
    im = axes[idx].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[idx].set_xticks(range(len(corr)))
    axes[idx].set_yticks(range(len(corr)))
    axes[idx].set_xticklabels(corr.index, rotation=45, ha='right', fontsize=8)
    axes[idx].set_yticklabels(corr.columns, fontsize=8)
    axes[idx].set_title(f'{method} Correlation')
    
    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            axes[idx].text(j, i, f'{corr.iloc[i, j]:.2f}', 
                          ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.show()
```

---

## 5.3.2 Why Pearson Misleads with Fat Tails

Pearson correlation is driven by extreme observations. With fat-tailed returns, correlation can be misleading:

**Case 1: Positive correlation driven by extreme events**
- When market crashes, all stocks crash (correlation $\approx 1$)
- When market rallies, correlations vary
- Reported average correlation hides crisis behavior

**Case 2: Simpson's Paradox in Finance**
- Correlation positive overall but negative within regimes
- Example: Tech stocks correlated with energy stocks during dollar crash, but negatively correlated during normal times

```python
def correlation_stability(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling correlation to test stability.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window (days)
    
    Returns:
        DataFrame of rolling correlations over time
    """
    # Assuming 2 assets for simplicity
    if returns.shape[1] != 2:
        raise ValueError("Provide exactly 2 return series")
    
    corr_rolling = returns.rolling(window).corr()
    
    # Extract correlation between the two assets
    corr_ts = []
    dates = []
    
    for date in returns.index[window:]:
        try:
            corr_value = corr_rolling.loc[date].iloc[0, 1]
            corr_ts.append(corr_value)
            dates.append(date)
        except:
            pass
    
    return pd.Series(corr_ts, index=dates)


# Analyze correlation stability
asset1 = load_financial_returns('AAPL', period='5y')
asset2 = load_financial_returns('MSFT', period='5y')

correlation_data = pd.DataFrame({
    'AAPL': asset1,
    'MSFT': asset2
}).dropna()

rolling_corr = correlation_stability(correlation_data, window=60)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(rolling_corr.index, rolling_corr.values, alpha=0.7, linewidth=1)
ax.axhline(rolling_corr.mean(), color='r', linestyle='--', label=f'Mean = {rolling_corr.mean():.3f}')
ax.fill_between(rolling_corr.index, 
                rolling_corr.mean() - rolling_corr.std(), 
                rolling_corr.mean() + rolling_corr.std(),
                alpha=0.2, label='±1 std')
ax.set_xlabel('Date')
ax.set_ylabel('60-Day Rolling Correlation')
ax.set_title('Correlation Instability: AAPL-MSFT (2019-2024)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"AAPL-MSFT Correlation Statistics (60-day rolling):")
print(f"  Mean: {rolling_corr.mean():.3f}")
print(f"  Std: {rolling_corr.std():.3f}")
print(f"  Min: {rolling_corr.min():.3f} (Date: {rolling_corr.idxmin()})")
print(f"  Max: {rolling_corr.max():.3f} (Date: {rolling_corr.idxmax()})")
print(f"\nInterpretation: Correlation varies from {rolling_corr.min():.1%} to {rolling_corr.max():.1%}")
print(f"Using static correlation ignores this {rolling_corr.std():.1%} volatility in dependence!")
```

[VISUALIZATION: Rolling 60-day correlation showing instability]

---

## 5.3.3 Correlation ≠ Causation ≠ Dependence

These three concepts are often confused in finance:

### Correlation

Just comovement: $X$ and $Y$ move together. No causal mechanism.

$$\text{Cor}(X, Y) \neq 0 \not\Rightarrow \text{Causal relationship}$$

### Causation

$X$ causes $Y$: changing $X$ changes $Y$ (holding all else constant). Requires:
- Temporal precedence (X before Y)
- Covariation
- No alternative explanations

**Finance example:** Does Fed rate (X) cause equity returns (Y)?
- Correlation exists: Fed tightening → lower equity valuations
- Causation: Through multiple channels (discount rate, growth expectations)
- Requires controlled experiment or natural experiment to establish

### Dependence

Broader than correlation; any association structure.

$$\text{If } \exists \text{ joint distribution} \not= F_X \times F_Y \Rightarrow \text{ Dependence}$$

Two random variables can be:
- Uncorrelated but dependent (nonlinear relationships)
- Correlated and independent in tail events (tail independence)

```python
def test_tail_independence(returns: pd.DataFrame, quantile: float = 0.05) -> dict:
    """
    Test for tail independence: are extreme events in X and Y independent?
    
    Args:
        returns: DataFrame with two return series
        quantile: Tail quantile (0.05 = bottom 5%)
    
    Returns:
        Dictionary with tail dependence measures
    """
    r1 = returns.iloc[:, 0].dropna()
    r2 = returns.iloc[:, 1].dropna()
    
    # Threshold for tail events
    thresh1 = r1.quantile(quantile)
    thresh2 = r2.quantile(quantile)
    
    # Identify tail events
    tail_r1 = r1 <= thresh1
    tail_r2 = r2 <= thresh2
    
    # Joint probability of both in tail
    joint_tail = (tail_r1 & tail_r2).sum() / len(r1)
    
    # Marginal probabilities
    p1 = tail_r1.sum() / len(r1)
    p2 = tail_r2.sum() / len(r2)
    
    # Expected joint probability if independent
    expected_independent = p1 * p2
    
    # Tail dependence coefficient
    tail_dependence = joint_tail / min(p1, p2) if min(p1, p2) > 0 else np.nan
    
    return {
        'joint_tail_prob': joint_tail,
        'marginal_p1': p1,
        'marginal_p2': p2,
        'expected_if_independent': expected_independent,
        'tail_dependence_coeff': tail_dependence,
        'realized_vs_expected': joint_tail / expected_independent if expected_independent > 0 else np.nan,
    }


# Test tail dependence
aapl = load_financial_returns('AAPL', period='5y')
msft = load_financial_returns('MSFT', period='5y')

returns_aligned = pd.DataFrame({
    'AAPL': aapl,
    'MSFT': msft
}).dropna()

tail_analysis = test_tail_independence(returns_aligned, quantile=0.05)

print("Tail Dependence Analysis (5% tail):")
print(f"  Joint tail probability: {tail_analysis['joint_tail_prob']:.4f}")
print(f"  Expected if independent: {tail_analysis['expected_if_independent']:.4f}")
print(f"  Realized / Expected: {tail_analysis['realized_vs_expected']:.2f}x")
print(f"\nInterpretation: Crashes are {tail_analysis['realized_vs_expected']:.1f}x more likely")
print(f"when both AAPL and MSFT are down (tail dependence > 1 = positive tail dependence)")
```

---

## 5.3.4 Copulas: Modeling Dependence Separately from Margins

A copula is a function $C: [0,1]^d \to [0,1]$ that couples marginal distributions to form a joint distribution:

$$F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d))$$

**Key insight:** Dependence structure is separate from marginal distributions!

### Gaussian Copula

Most common in finance; allows different marginals with Gaussian copula structure:

$$C(u_1, u_2; \rho) = \Phi_\rho(\Phi^{-1}(u_1), \Phi^{-1}(u_2))$$

where $\Phi_\rho$ is bivariate normal CDF with correlation $\rho$.

**Advantage:** Easy to implement, correlation interpretation
**Disadvantage:** Underestimates tail dependence (the main source of portfolio risk!)

### t-Copula

Better for fat tails; uses t-distribution instead of normal:

$$C(u_1, u_2; \nu, \rho) = t_{\nu, \rho}(t_\nu^{-1}(u_1), t_\nu^{-1}(u_2))$$

where $t_\nu$ is univariate t-CDF and $t_{\nu, \rho}$ is bivariate t-CDF.

**Advantage:** Captures tail dependence
**Disadvantage:** More parameters to estimate

### Implementation

```python
from scipy.optimize import minimize

class CopulaFitter:
    """
    Fit copulas to return data.
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize with return data.
        
        Args:
            returns_df: DataFrame with exactly 2 return series
        """
        if returns_df.shape[1] != 2:
            raise ValueError("Provide exactly 2 return series")
        
        self.returns = returns_df.dropna()
        self.r1 = self.returns.iloc[:, 0].values
        self.r2 = self.returns.iloc[:, 1].values
        self.n = len(self.returns)
        
        # Convert to uniform marginals via empirical CDF
        self.u1 = stats.rankdata(self.r1) / (self.n + 1)
        self.u2 = stats.rankdata(self.r2) / (self.n + 1)
    
    def fit_gaussian_copula(self) -> dict:
        """
        Fit Gaussian copula (parametric approach).
        
        Returns:
            Dictionary with correlation parameter
        """
        # Transform to standard normal
        z1 = stats.norm.ppf(self.u1)
        z2 = stats.norm.ppf(self.u2)
        
        # Correlation
        rho = np.corrcoef(z1, z2)[0, 1]
        
        # Log-likelihood
        ll = 0
        for i in range(self.n):
            # Copula PDF for Gaussian copula
            term = -0.5 * np.log(1 - rho**2)
            term += 0.5 * (z1[i]**2 + z2[i]**2)
            term -= (rho**2 * z1[i]**2 - 2 * rho * z1[i] * z2[i] + rho**2 * z2[i]**2) / (2 * (1 - rho**2))
            ll += term
        
        return {
            'copula_type': 'Gaussian',
            'correlation': rho,
            'log_likelihood': ll,
            'u1': self.u1,
            'u2': self.u2,
        }
    
    def fit_t_copula(self, initial_rho: float = 0.5, 
                    initial_df: float = 10) -> dict:
        """
        Fit t-copula using MLE.
        
        Args:
            initial_rho: Initial correlation guess
            initial_df: Initial degrees of freedom guess
        
        Returns:
            Dictionary with fitted parameters
        """
        def negative_ll(params):
            rho, nu = params
            if rho <= -1 or rho >= 1 or nu <= 1:
                return 1e10
            
            # Convert to t-distribution
            t_inv_u1 = stats.t.ppf(self.u1, nu)
            t_inv_u2 = stats.t.ppf(self.u2, nu)
            
            # Compute copula log-likelihood
            ll = 0
            for i in range(self.n):
                # Correlation matrix
                sigma = np.array([[1, rho], [rho, 1]])
                sigma_inv = np.linalg.inv(sigma)
                
                # t-copula density
                term = -0.5 * (nu + 2) * np.log(1 + (t_inv_u1[i]**2 + t_inv_u2[i]**2 - 
                                                       2 * rho * t_inv_u1[i] * t_inv_u2[i]) / 
                                                  (nu * (1 - rho**2)))
                term += 0.5 * (nu + 2) * (np.log(1 + t_inv_u1[i]**2 / nu) + 
                                         np.log(1 + t_inv_u2[i]**2 / nu))
                term += np.log(np.linalg.det(sigma_inv)) / 2
                
                ll += term
            
            return -ll
        
        result = minimize(
            negative_ll,
            [initial_rho, initial_df],
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        rho_fit, nu_fit = result.x
        
        return {
            'copula_type': 't-copula',
            'correlation': rho_fit,
            'degrees_of_freedom': nu_fit,
            'log_likelihood': -result.fun,
            'u1': self.u1,
            'u2': self.u2,
        }


# Fit copulas to asset pair
aapl = load_financial_returns('AAPL', period='5y')
spy = load_financial_returns('^GSPC', period='5y')

returns_paired = pd.DataFrame({
    'AAPL': aapl,
    'SPY': spy
}).dropna()

fitter = CopulaFitter(returns_paired)
gaussian = fitter.fit_gaussian_copula()
t_copula = fitter.fit_t_copula()

print("Copula Fit Comparison (AAPL vs SPY):")
print(f"\nGaussian Copula:")
print(f"  Correlation: {gaussian['correlation']:.4f}")
print(f"  Log-Likelihood: {gaussian['log_likelihood']:.2f}")

print(f"\nt-Copula:")
print(f"  Correlation: {t_copula['correlation']:.4f}")
print(f"  Degrees of Freedom: {t_copula['degrees_of_freedom']:.1f}")
print(f"  Log-Likelihood: {t_copula['log_likelihood']:.2f}")

# Visualize uniform marginals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

u1, u2 = gaussian['u1'], gaussian['u2']

# Scatter of copula data
axes[0].scatter(u1, u2, alpha=0.3, s=10)
axes[0].set_xlabel('AAPL (Uniform)')
axes[0].set_ylabel('SPY (Uniform)')
axes[0].set_title('Empirical Copula Data')
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3)

# Heatmap of copula density
axes[1].hexbin(u1, u2, gridsize=30, cmap='YlOrRd', mincnt=1)
axes[1].set_xlabel('AAPL (Uniform)')
axes[1].set_ylabel('SPY (Uniform)')
axes[1].set_title('Copula Density (Concentration)')

plt.tight_layout()
plt.show()
```

---

## 5.3.5 Correlation Breakdown During Crises

A key stylized fact: **correlations spike during market stress**.

Why?
1. Flight to safety: all assets sold simultaneously
2. Systematic risk dominates idiosyncratic risk
3. Leverage forced unwinds (everyone sells everything)

This has major portfolio implications: diversification collapses when you need it most!

```python
def identify_crisis_periods(returns: pd.Series, threshold: float = -0.02) -> pd.DatetimeIndex:
    """
    Identify crisis periods as days with return below threshold.
    
    Args:
        returns: Series of returns
        threshold: Daily return threshold (default -2%)
    
    Returns:
        DatetimeIndex of crisis dates
    """
    return returns[returns < threshold].index


def correlation_by_regime(returns_df: pd.DataFrame, 
                         crisis_dates: pd.DatetimeIndex) -> dict:
    """
    Compare correlation in normal vs. crisis periods.
    
    Args:
        returns_df: DataFrame of returns
        crisis_dates: Dates identified as crises
    
    Returns:
        Dictionary with correlations by regime
    """
    # Identify crisis period dates
    in_crisis = returns_df.index.isin(crisis_dates)
    
    # Normal period correlation
    normal_returns = returns_df[~in_crisis]
    normal_corr = normal_returns.corr().iloc[0, 1]
    
    # Crisis period correlation
    crisis_returns = returns_df[in_crisis]
    if len(crisis_returns) > 1:
        crisis_corr = crisis_returns.corr().iloc[0, 1]
    else:
        crisis_corr = np.nan
    
    return {
        'normal_corr': normal_corr,
        'crisis_corr': crisis_corr,
        'corr_increase': crisis_corr - normal_corr,
        'n_normal': len(normal_returns),
        'n_crisis': len(crisis_returns),
    }


# Analyze correlation breakdown for stock pair
aapl = load_financial_returns('AAPL', period='10y')
msft = load_financial_returns('MSFT', period='10y')

returns_paired = pd.DataFrame({
    'AAPL': aapl,
    'MSFT': msft
}).dropna()

# Identify crisis periods
spy = load_financial_returns('^GSPC', period='10y')
crisis_dates = identify_crisis_periods(spy[returns_paired.index], threshold=-0.02)

regime_corr = correlation_by_regime(returns_paired, crisis_dates)

print("Correlation Breakdown During Crises:")
print(f"Normal period correlation: {regime_corr['normal_corr']:.4f}")
print(f"Crisis period correlation: {regime_corr['crisis_corr']:.4f}")
print(f"Increase: {regime_corr['corr_increase']:.4f}")
print(f"\nNormal days: {regime_corr['n_normal']}")
print(f"Crisis days: {regime_corr['n_crisis']}")

if regime_corr['corr_increase'] > 0:
    print(f"\nWARNING: Correlation increased by {regime_corr['corr_increase']:.1%} in crises!")
    print("Diversification benefits disappear when most needed.")
```

---

## Module 5.3 Summary

- **Pearson correlation** is most common but sensitive to outliers; compare with Spearman/Kendall
- **Correlation ≠ causation ≠ dependence**: understand the distinction
- **Copulas** model dependence separately from marginals
- **Rolling correlations** reveal instability; static correlations are dangerous
- **Tail dependence** and **correlation breakdown** are critical for portfolio risk management

---

# Module 5.4: Bayesian Statistics for Finance

## 5.4.1 Bayes' Theorem and Financial Inference

Bayes' theorem connects prior beliefs, observed data, and posterior beliefs:

$$P(\theta | \text{Data}) = \frac{P(\text{Data} | \theta) P(\theta)}{P(\text{Data})}$$

Or, in terms of proportionality:

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

**Components:**

- **Prior** $P(\theta)$: What you believe about parameter before seeing data
- **Likelihood** $P(\text{Data} | \theta)$: Probability of observing data given parameter
- **Posterior** $P(\theta | \text{Data})$: Updated belief after observing data
- **Marginal likelihood** $P(\text{Data})$: Evidence (normalizing constant)

### Why Bayesian Statistics for Finance?

1. **Incorporate domain knowledge**: Priors encode market regime beliefs
2. **Sequential learning**: Update beliefs as new data arrives (crucial for live trading)
3. **Principled uncertainty**: Full posterior distribution, not just point estimates
4. **Handle parameter uncertainty**: Especially important with short time series
5. **Small sample efficiency**: Priors help when you don't have 30 years of data

---

## 5.4.2 Bayesian Updating with Conjugate Priors

A prior is **conjugate** to a likelihood if the posterior has the same form as the prior. This enables analytical solutions.

### Example: Estimating Return Mean with Normal-Normal Conjugacy

**Model:**
$$R_t | \mu \sim N(\mu, \sigma^2) \quad \text{(likelihood)}$$
$$\mu | \mu_0, \tau_0^2 \sim N(\mu_0, \tau_0^2) \quad \text{(prior)}$$

**Posterior:** (analytically tractable!)
$$\mu | \text{Data} \sim N(\mu_n, \tau_n^2)$$

where:
$$\mu_n = \frac{\frac{\mu_0}{\tau_0^2} + \frac{\sum R_i}{\sigma^2}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}$$

$$\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}$$

**Interpretation:**
- Posterior mean is weighted average of prior mean and data mean
- Posterior precision (inverse variance) is sum of prior and data precision
- With more data ($n \to \infty$), prior fades and posterior converges to data

### Implementation

```python
def bayesian_mean_return_estimation(returns: pd.Series,
                                   prior_mean: float = 0.0,
                                   prior_std: float = 0.02) -> dict:
    """
    Estimate return mean using Bayesian inference with conjugate prior.
    
    Args:
        returns: Series of returns
        prior_mean: Prior belief about mean return
        prior_std: Prior uncertainty (std dev)
    
    Returns:
        Dictionary with prior, posterior, and comparison
    """
    ret_data = returns.dropna().values
    n = len(ret_data)
    
    # Data statistics
    data_mean = ret_data.mean()
    data_std = ret_data.std(ddof=1)
    data_sem = data_std / np.sqrt(n)
    
    # Prior parameters
    prior_precision = 1 / (prior_std ** 2)
    data_precision = n / (data_std ** 2)
    
    # Posterior parameters
    posterior_precision = prior_precision + data_precision
    posterior_std = np.sqrt(1 / posterior_precision)
    
    posterior_mean = (
        (prior_mean * prior_precision + data_mean * data_precision) / posterior_precision
    )
    
    # Posterior credible interval (Bayesian CI)
    ci_lower = posterior_mean - 1.96 * posterior_std
    ci_upper = posterior_mean + 1.96 * posterior_std
    
    # Frequentist CI for comparison
    freq_ci_lower = data_mean - 1.96 * data_sem
    freq_ci_upper = data_mean + 1.96 * data_sem
    
    return {
        'data': {
            'n': n,
            'mean': data_mean,
            'std': data_std,
            'sem': data_sem,
        },
        'prior': {
            'mean': prior_mean,
            'std': prior_std,
            'precision': prior_precision,
        },
        'posterior': {
            'mean': posterior_mean,
            'std': posterior_std,
            'precision': posterior_precision,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        },
        'frequentist_ci': {
            'lower': freq_ci_lower,
            'upper': freq_ci_upper,
        },
    }


# Estimate return mean with Bayesian approach
returns = load_financial_returns('^GSPC')

# Scenario 1: Weak prior (uncertain about mean)
bayes_weak = bayesian_mean_return_estimation(
    returns, prior_mean=0.0, prior_std=0.05
)

# Scenario 2: Strong prior (confident in positive drift)
bayes_strong = bayesian_mean_return_estimation(
    returns, prior_mean=0.001, prior_std=0.001
)

print("Bayesian Mean Estimation Comparison:")
print(f"\nData (S&P 500):")
print(f"  Mean: {bayes_weak['data']['mean']:.6f}")
print(f"  Std: {bayes_weak['data']['std']:.6f}")

print(f"\n--- Weak Prior (μ₀=0, σ=5%) ---")
print(f"Posterior Mean: {bayes_weak['posterior']['mean']:.6f}")
print(f"Posterior Std: {bayes_weak['posterior']['std']:.6f}")
print(f"95% CI: [{bayes_weak['posterior']['ci_lower']:.6f}, {bayes_weak['posterior']['ci_upper']:.6f}]")

print(f"\n--- Strong Prior (μ₀=0.1%, σ=0.1%) ---")
print(f"Posterior Mean: {bayes_strong['posterior']['mean']:.6f}")
print(f"Posterior Std: {bayes_strong['posterior']['std']:.6f}")
print(f"95% CI: [{bayes_strong['posterior']['ci_lower']:.6f}, {bayes_strong['posterior']['ci_upper']:.6f}]")

print(f"\n--- Frequentist (No Prior) ---")
print(f"Mean: {bayes_weak['data']['mean']:.6f}")
print(f"95% CI: [{bayes_weak['frequentist_ci']['lower']:.6f}, {bayes_weak['frequentist_ci']['upper']:.6f}]")

# Visualize posterior distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Weak prior
x = np.linspace(-0.002, 0.004, 1000)

ax = axes[0]
ax.plot(x, stats.norm.pdf(x, bayes_weak['prior']['mean'], 
        bayes_weak['prior']['std']), 'b--', label='Prior', linewidth=2)
ax.plot(x, stats.norm.pdf(x, bayes_weak['data']['mean'], 
        bayes_weak['data']['sem']), 'g--', label='Likelihood', linewidth=2)
ax.plot(x, stats.norm.pdf(x, bayes_weak['posterior']['mean'], 
        bayes_weak['posterior']['std']), 'r-', label='Posterior', linewidth=2)
ax.set_xlabel('Mean Return')
ax.set_ylabel('Density')
ax.set_title('Bayesian Updating: Weak Prior')
ax.legend()
ax.grid(True, alpha=0.3)

# Strong prior
ax = axes[1]
ax.plot(x, stats.norm.pdf(x, bayes_strong['prior']['mean'], 
        bayes_strong['prior']['std']), 'b--', label='Prior', linewidth=2)
ax.plot(x, stats.norm.pdf(x, bayes_strong['data']['mean'], 
        bayes_strong['data']['sem']), 'g--', label='Likelihood', linewidth=2)
ax.plot(x, stats.norm.pdf(x, bayes_strong['posterior']['mean'], 
        bayes_strong['posterior']['std']), 'r-', label='Posterior', linewidth=2)
ax.set_xlabel('Mean Return')
ax.set_ylabel('Density')
ax.set_title('Bayesian Updating: Strong Prior')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 5.4.3 Bayesian Linear Regression with Shrinkage

Standard OLS can overfit with many predictors. Bayesian regression adds regularization through priors.

**Model:**
$$Y | X, \beta, \sigma^2 \sim N(X\beta, \sigma^2 I)$$
$$\beta | \lambda \sim N(0, \lambda^{-1} I) \quad \text{(L2 regularization / Ridge)}$$

This induces shrinkage: $\hat{\beta}_{\text{Bayes}}$ is pulled toward zero compared to OLS.

### Implementation

```python
def bayesian_regression_shrinkage(y: np.ndarray, X: np.ndarray,
                                 prior_precision: float = 1.0) -> dict:
    """
    Bayesian linear regression with L2 prior (Ridge regression).
    
    Args:
        y: Dependent variable
        X: Regressors (including constant)
        prior_precision: Lambda (higher = more shrinkage)
    
    Returns:
        Dictionary with coefficients, uncertainties, and comparison with OLS
    """
    n, k = X.shape
    
    # OLS estimate
    XTX = X.T @ X
    XTy = X.T @ y
    beta_ols = np.linalg.solve(XTX, XTy)
    residuals_ols = y - X @ beta_ols
    sigma2_ols = (residuals_ols ** 2).mean()
    
    # Bayesian estimate (with normal-inverse-gamma prior)
    # Posterior mean
    prior_cov = np.eye(k) / prior_precision
    posterior_cov = np.linalg.inv(XTX + prior_precision * np.eye(k))
    beta_bayes = posterior_cov @ XTy
    
    # Posterior variance
    residuals_bayes = y - X @ beta_bayes
    sigma2_bayes = (residuals_bayes ** 2).sum() / (n + k)
    
    # Standard errors
    se_ols = np.sqrt(np.diag(posterior_cov) * sigma2_ols)
    se_bayes = np.sqrt(np.diag(posterior_cov) * sigma2_bayes)
    
    return {
        'beta_ols': beta_ols,
        'beta_bayes': beta_bayes,
        'se_ols': se_ols,
        'se_bayes': se_bayes,
        'shrinkage_amount': np.linalg.norm(beta_bayes) / np.linalg.norm(beta_ols),
        'sigma2_ols': sigma2_ols,
        'sigma2_bayes': sigma2_bayes,
        'posterior_cov': posterior_cov,
    }


# Example: Predict asset return from multiple factors
np.random.seed(42)
n_obs = 252  # 1 year of daily data
n_factors = 10

# Generate synthetic factor data
X = np.random.randn(n_obs, n_factors)
X = np.column_stack([np.ones(n_obs), X])  # Add constant

# Generate return with sparse true model (only 3 factors matter)
true_beta = np.array([0.0001, 0.5, -0.3, 0.2, 0, 0, 0, 0, 0, 0, 0])
y = X @ true_beta + 0.02 * np.random.randn(n_obs)

# Compare OLS and Bayesian
bayes_result = bayesian_regression_shrinkage(y, X, prior_precision=10.0)

fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(bayes_result['beta_ols']))
width = 0.35

ax.bar(x_pos - width/2, bayes_result['beta_ols'], width, label='OLS', alpha=0.7)
ax.bar(x_pos + width/2, bayes_result['beta_bayes'], width, label='Bayesian', alpha=0.7)

ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Factor')
ax.set_ylabel('Coefficient')
ax.set_title('OLS vs Bayesian Regression (Shrinkage)')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'F{i}' if i > 0 else 'Const' for i in range(len(bayes_result['beta_ols']))])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("Bayesian Shrinkage in Regression:")
print(f"L2 norm of OLS coefficients: {np.linalg.norm(bayes_result['beta_ols']):.6f}")
print(f"L2 norm of Bayesian coefficients: {np.linalg.norm(bayes_result['beta_bayes']):.6f}")
print(f"Shrinkage amount: {bayes_result['shrinkage_amount']:.2%}")
print(f"\nOut-of-sample R²:")
print(f"  OLS: {1 - bayes_result['sigma2_ols'] / np.var(y):.3f}")
print(f"  Bayesian: {1 - bayes_result['sigma2_bayes'] / np.var(y):.3f}")
```

---

## 5.4.4 Sequential Bayesian Learning: Updating as Data Arrives

One of Bayesian statistics' great strengths: natural sequential updating.

**Scenario:** You're live-trading and getting new daily data. How should you update your beliefs about strategy profitability?

```python
def sequential_bayesian_learning(returns_stream: pd.Series,
                                prior_mean: float = 0.0,
                                prior_std: float = 0.01,
                                window_size: int = 20) -> pd.DataFrame:
    """
    Update Bayesian estimate of mean return as new data arrives.
    
    Args:
        returns_stream: Time series of returns
        prior_mean: Initial prior mean
        prior_std: Initial prior std
        window_size: Update every window_size observations
    
    Returns:
        DataFrame tracking posterior evolution
    """
    returns = returns_stream.dropna().values
    
    posterior_means = []
    posterior_stds = []
    posterior_dates = []
    n_obs_list = []
    
    current_prior_mean = prior_mean
    current_prior_std = prior_std
    
    for t in range(window_size, len(returns), window_size):
        # Data so far
        data_so_far = returns[:t]
        n = len(data_so_far)
        
        # Data statistics
        data_mean = data_so_far.mean()
        data_std = data_so_far.std(ddof=1)
        data_sem = data_std / np.sqrt(n)
        
        # Bayesian update
        prior_precision = 1 / (current_prior_std ** 2)
        data_precision = n / (data_std ** 2)
        
        posterior_precision = prior_precision + data_precision
        posterior_std = np.sqrt(1 / posterior_precision)
        posterior_mean = (
            (current_prior_mean * prior_precision + data_mean * data_precision) / 
            posterior_precision
        )
        
        posterior_means.append(posterior_mean)
        posterior_stds.append(posterior_std)
        posterior_dates.append(t)
        n_obs_list.append(n)
        
        # Update prior for next iteration
        current_prior_mean = posterior_mean
        current_prior_std = posterior_std
    
    return pd.DataFrame({
        'n_obs': n_obs_list,
        'posterior_mean': posterior_means,
        'posterior_std': posterior_stds,
        'update_idx': posterior_dates,
    })


# Simulate strategy learning
np.random.seed(42)
true_mean = 0.0005
strategy_returns = np.random.normal(true_mean, 0.02, 500)  # 500 days

evolution = sequential_bayesian_learning(
    pd.Series(strategy_returns),
    prior_mean=0.0,
    prior_std=0.01,
    window_size=10
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top: Posterior mean evolution
ax = axes[0]
ax.plot(evolution['n_obs'], evolution['posterior_mean'], 'b-', linewidth=2, label='Posterior Mean')
ax.fill_between(
    evolution['n_obs'],
    evolution['posterior_mean'] - 1.96 * evolution['posterior_std'],
    evolution['posterior_mean'] + 1.96 * evolution['posterior_std'],
    alpha=0.3,
    label='95% CI'
)
ax.axhline(true_mean, color='r', linestyle='--', label='True Mean')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_ylabel('Expected Return')
ax.set_title('Sequential Bayesian Learning: Strategy Mean Return')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom: Posterior uncertainty over time
ax = axes[1]
ax.plot(evolution['n_obs'], evolution['posterior_std'], 'g-', linewidth=2)
ax.set_xlabel('Sample Size')
ax.set_ylabel('Posterior Std Dev')
ax.set_title('Uncertainty Decreases with Data')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Sequential Learning Summary:")
print(f"True mean return: {true_mean:.6f}")
print(f"\nInitial belief: N(0, 0.01)")
print(f"After {evolution['n_obs'].iloc[0]} observations: "
      f"N({evolution['posterior_mean'].iloc[0]:.6f}, {evolution['posterior_std'].iloc[0]:.6f})")
print(f"After {evolution['n_obs'].iloc[-1]} observations: "
      f"N({evolution['posterior_mean'].iloc[-1]:.6f}, {evolution['posterior_std'].iloc[-1]:.6f})")
```

---

## 5.4.5 When Bayesian Beats Frequentist in Finance

**Frequentist approach:** 
- Maximize likelihood (ignore prior information)
- Report point estimate + standard error
- "Parameter is either in CI or not" (no probability statement)

**Bayesian approach:**
- Combine likelihood with prior (domain knowledge)
- Report full posterior distribution
- Probability statements about parameter (true belief update)

### Key Advantages in Finance:

1. **Short time series problem:** Equities have been trading for 100+ years but most investors only have 10-20 year histories. Priors help!

2. **Regime switching:** Prior beliefs about volatility regimes help with small-sample estimation.

3. **Multiple hypothesis testing:** Prior probabilities reduce false discovery rate naturally.

4. **Real-time adaptation:** Sequential updating as new data arrives.

```python
def compare_frequentist_bayesian_sharpe(returns: pd.Series,
                                       prior_mean_sharpe: float = 0.5,
                                       prior_std_sharpe: float = 0.5) -> dict:
    """
    Compare frequentist vs Bayesian Sharpe ratio estimation.
    
    Args:
        returns: Daily returns
        prior_mean_sharpe: Prior belief about annualized Sharpe ratio
        prior_std_sharpe: Prior uncertainty
    
    Returns:
        Dictionary with both estimates and their properties
    """
    ret_clean = returns.dropna().values
    n = len(ret_clean)
    
    # Frequentist Sharpe ratio
    mean_ret = ret_clean.mean()
    std_ret = ret_clean.std(ddof=1)
    sharpe_freq = mean_ret / std_ret * np.sqrt(252)
    
    # Standard error (Jobson-Korkie, 1981)
    se_sharpe = np.sqrt((1 + 0.5 * sharpe_freq**2) / n)
    ci_freq_lower = sharpe_freq - 1.96 * se_sharpe
    ci_freq_upper = sharpe_freq + 1.96 * se_sharpe
    
    # Bayesian Sharpe ratio with normal prior
    # Assume mean and std estimates are exact (simplification)
    prior_precision = 1 / (prior_std_sharpe ** 2)
    likelihood_precision = n / (0.1 ** 2)  # Approximate likelihood precision
    
    posterior_precision = prior_precision + likelihood_precision
    posterior_std = np.sqrt(1 / posterior_precision)
    posterior_mean = (
        (prior_mean_sharpe * prior_precision + sharpe_freq * likelihood_precision) / 
        posterior_precision
    )
    
    ci_bayes_lower = posterior_mean - 1.96 * posterior_std
    ci_bayes_upper = posterior_mean + 1.96 * posterior_std
    
    return {
        'frequentist': {
            'estimate': sharpe_freq,
            'se': se_sharpe,
            'ci_lower': ci_freq_lower,
            'ci_upper': ci_freq_upper,
            'ci_width': ci_freq_upper - ci_freq_lower,
        },
        'bayesian': {
            'estimate': posterior_mean,
            'std': posterior_std,
            'ci_lower': ci_bayes_lower,
            'ci_upper': ci_bayes_upper,
            'ci_width': ci_bayes_upper - ci_bayes_lower,
            'prior_mean': prior_mean_sharpe,
            'prior_std': prior_std_sharpe,
        },
        'data': {
            'n_obs': n,
            'mean_ret': mean_ret,
            'std_ret': std_ret,
        },
    }


# Compare approaches on a short sample
returns_short = load_financial_returns('^GSPC', period='2y')
comparison = compare_frequentist_bayesian_sharpe(
    returns_short,
    prior_mean_sharpe=0.5,
    prior_std_sharpe=1.0
)

print("Frequentist vs Bayesian Sharpe Ratio Estimation")
print(f"Sample size: {comparison['data']['n_obs']} days (~{comparison['data']['n_obs']/252:.1f} years)")
print(f"Annual mean: {comparison['data']['mean_ret'] * 252:.4%}")
print(f"Annual std: {comparison['data']['std_ret'] * np.sqrt(252):.4%}")

print(f"\n--- Frequentist ---")
print(f"Sharpe estimate: {comparison['frequentist']['estimate']:.3f}")
print(f"Standard error: {comparison['frequentist']['se']:.3f}")
print(f"95% CI: [{comparison['frequentist']['ci_lower']:.3f}, {comparison['frequentist']['ci_upper']:.3f}]")
print(f"CI width: {comparison['frequentist']['ci_width']:.3f}")

print(f"\n--- Bayesian (with weak prior) ---")
print(f"Posterior mean: {comparison['bayesian']['estimate']:.3f}")
print(f"Posterior std: {comparison['bayesian']['std']:.3f}")
print(f"95% CI: [{comparison['bayesian']['ci_lower']:.3f}, {comparison['bayesian']['ci_upper']:.3f}]")
print(f"CI width: {comparison['bayesian']['ci_width']:.3f}")

print(f"\nInterpretation:")
print(f"- Bayesian incorporates prior belief (E[S] = 0.5)")
print(f"- With small sample, Bayesian estimate more stable")
print(f"- Frequentist estimate can be extreme with volatile short samples")
```

---

## Module 5.4 Summary

- **Bayes' theorem** provides principled framework for updating beliefs with data
- **Conjugate priors** enable analytical solutions (normal-normal, normal-gamma)
- **Bayesian regression** with priors provides shrinkage and prevents overfitting
- **Sequential learning** naturally updates as new data arrives
- **Bayesian beats frequentist** in small-sample finance applications and multiple testing

---

# Chapter Summary and Key Takeaways

## Core Concepts Mastered

1. **Distributions Beyond Normal**
   - Real returns have fat tails and negative skew
   - t-distribution captures this with one extra parameter
   - Mixture models capture regime switching
   - Always check distributional assumptions with QQ plots

2. **Statistical Inference for Markets**
   - MLE for distribution fitting
   - Hypothesis testing with multiple-testing corrections
   - Bootstrap for non-parametric confidence intervals
   - Newey-West for time series regression

3. **Understanding Dependence**
   - Correlation is not causation, not even dependence
   - Rolling correlations reveal instability
   - Tail dependence matters more than average correlation
   - Copulas model dependence structure separately

4. **Bayesian Framework**
   - Use priors to incorporate domain knowledge
   - Sequential updating as data arrives
   - Shrinkage priors prevent overfitting
   - More efficient than frequentist with small samples

## When to Use Which Method

| Problem | Solution |
|---------|----------|
| Estimating VaR | Use t-distribution, not normal |
| Testing strategy alpha | Bootstrap Sharpe ratio, not standard error |
| Portfolio correlation | Use rolling window + copulas for crisis periods |
| Factor model with few observations | Bayesian shrinkage instead of OLS |
| Multiple asset strategies | FDR correction instead of Bonferroni |
| Live trading systems | Sequential Bayesian learning |

## Critical Warnings

**WARNING 1: Normal distribution assumption fails for VaR, options pricing, and risk limits**
- A -3σ event happens once every 370 years by normal theory
- In real data, happens every 3-5 years
- Use t-distribution or historical simulation instead

**WARNING 2: Correlation breaks down in crises**
- Diversification provides no benefit when you need it most
- Static correlation matrices are dangerous for portfolio construction
- Use regime-switching models or copulas

**WARNING 3: Multiple testing inflation**
- With 100 candidate strategies, expect 5 spurious winners at 5% level
- Always correct for multiple testing
- Bayesian approach naturally penalizes over-parameterization

**WARNING 4: Small-sample bias in performance metrics**
- Sharpe ratio biased upward with few observations
- Bootstrap to get proper confidence intervals
- Bayesian shrinkage prevents overconfidence

---

# Chapter Project: Complete Statistical Analysis of Multi-Asset Trading System

## Objective

Build a complete statistical analysis of a multi-asset trading strategy using all tools from this chapter.

## Data Requirements

- Daily returns for 3-5 assets (stocks, commodities, or crypto)
- At least 3 years of data
- Aligned date indices

## Project Steps

### Step 1: Distributional Analysis

```python
def project_distributional_analysis(returns_df: pd.DataFrame) -> dict:
    """
    Complete distributional analysis of multi-asset returns.
    """
    results = {}
    
    for asset in returns_df.columns:
        returns = returns_df[asset]
        
        # 1. Test normality
        normality = test_normality(returns)
        
        # 2. Fit distributions
        fitter = FinancialDistributionFitter(returns)
        normal_fit = fitter.fit_normal()
        t_fit = fitter.fit_t()
        
        # 3. Compute moments
        moments = analyze_distribution_moments(returns)
        
        # 4. Create QQ plot
        create_qq_plot(returns, title=f"Q-Q Plot: {asset}")
        
        results[asset] = {
            'normality_tests': normality,
            'fits': {'normal': normal_fit, 't': t_fit},
            'moments': moments,
        }
    
    return results
```

### Step 2: Correlation and Dependence Analysis

```python
def project_correlation_analysis(returns_df: pd.DataFrame) -> dict:
    """
    Analyze correlation structure and regime dependence.
    """
    # Static correlations
    corr_pearson = returns_df.corr(method='pearson')
    corr_spearman = returns_df.corr(method='spearman')
    
    # Rolling correlations
    rolling_correlations = {}
    for i in range(returns_df.shape[1]):
        for j in range(i+1, returns_df.shape[1]):
            pair = f"{returns_df.columns[i]}-{returns_df.columns[j]}"
            pair_data = returns_df.iloc[:, [i, j]]
            rolling_correlations[pair] = correlation_stability(pair_data, window=60)
    
    # Tail dependence
    tail_dependence = {}
    for i in range(returns_df.shape[1]):
        for j in range(i+1, returns_df.shape[1]):
            pair = f"{returns_df.columns[i]}-{returns_df.columns[j]}"
            pair_data = returns_df.iloc[:, [i, j]]
            tail_dependence[pair] = test_tail_independence(pair_data)
    
    return {
        'pearson': corr_pearson,
        'spearman': corr_spearman,
        'rolling': rolling_correlations,
        'tail_dependence': tail_dependence,
    }
```

### Step 3: Bayesian Factor Model

```python
def project_bayesian_factor_model(returns_df: pd.DataFrame,
                                 factor_df: pd.DataFrame) -> dict:
    """
    Estimate asset alphas with Bayesian shrinkage.
    """
    results = {}
    
    for asset in returns_df.columns:
        y = returns_df[asset].values
        X = np.column_stack([np.ones(len(y)), factor_df.values])
        
        bayes_fit = bayesian_regression_shrinkage(y, X, prior_precision=5.0)
        
        results[asset] = {
            'alpha_ols': bayes_fit['beta_ols'][0],
            'alpha_bayes': bayes_fit['beta_bayes'][0],
            'betas_ols': bayes_fit['beta_ols'][1:],
            'betas_bayes': bayes_fit['beta_bayes'][1:],
            'shrinkage': bayes_fit['shrinkage_amount'],
        }
    
    return results
```

### Step 4: Bootstrap Performance Metrics

```python
def project_bootstrap_analysis(returns_df: pd.DataFrame) -> dict:
    """
    Bootstrap confidence intervals for Sharpe ratios and other metrics.
    """
    results = {}
    
    for asset in returns_df.columns:
        returns = returns_df[asset]
        
        # Bootstrap Sharpe
        sharpe_boot = bootstrap_sharpe_ratio(returns, n_bootstrap=10000)
        
        # Other metrics
        results[asset] = sharpe_boot
    
    return results
```

### Step 5: Multiple Testing Correction

```python
def project_test_strategy_significance(returns_df: pd.DataFrame) -> dict:
    """
    Test each asset for positive alpha with multiple-testing correction.
    """
    p_values = []
    
    for asset in returns_df.columns:
        returns = returns_df[asset]
        test = test_mean_return(returns)
        p_values.append(test['p_value'])
    
    p_values = np.array(p_values)
    
    # Apply corrections
    bonferroni = multiple_testing_correction(p_values, method='bonferroni')
    bh = multiple_testing_correction(p_values, method='BH')
    
    return {
        'p_values': p_values,
        'bonferroni': bonferroni,
        'benjamini_hochberg': bh,
    }
```

## Deliverables

1. **Comprehensive PDF report** with:
   - Distributional analysis for each asset
   - Correlation matrices and rolling correlations
   - Tail dependence analysis
   - Bayesian shrinkage results
   - Multiple testing corrections
   - Investment implications

2. **Python module** with:
   - All functions above
   - Clear documentation
   - Type hints
   - Ready for production use

3. **Key findings:**
   - Which distribution best fits each asset?
   - How unstable are correlations?
   - What is true alpha after correcting for multiple testing?
   - How much does Bayesian shrinkage improve alpha estimates?

---

## Further Reading and Resources

### Textbooks
- **Statistical Rethinking** by Richard McElreath (Bayesian foundations)
- **An Introduction to Statistical Learning** by James, Witten, Hastie, Tibshirani (ML + stats)
- **Advances in Financial Machine Learning** by Marcos López de Prado (finance-specific)

### Papers
- Mandelbrot & Hudson (2004): "The Misbehavior of Markets" (fat tails)
- Cont (2001): "Empirical Properties of Asset Returns" (stylized facts)
- Embrechts et al. (2002): "Extreme Value Theory for Insurance and Finance" (extreme events)

### Online Resources
- StatQuest with Josh Starmer (YouTube: distributions, Bayesian methods)
- Quantstart.com (quantitative finance tutorials)
- ArXiv.org (recent finance papers)

---

**End of Chapter 5**

---

## Appendix: Production Code Template

```python
"""
Chapter 5 Production Code: Statistical Methods for Finance

This module provides production-ready implementations of all Chapter 5 concepts.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class FinancialStatsToolkit:
    """
    Comprehensive toolkit for financial statistics.
    """
    
    def __init__(self, returns_df: pd.DataFrame):
        """
        Initialize with multi-asset return data.
        
        Args:
            returns_df: DataFrame with asset returns as columns
        """
        self.returns = returns_df.dropna()
        self.assets = returns_df.columns.tolist()
    
    def complete_analysis(self) -> Dict:
        """
        Run full statistical analysis pipeline.
        
        Returns:
            Dictionary with all results
        """
        results = {
            'distributions': self._analyze_distributions(),
            'correlations': self._analyze_correlations(),
            'bootstrap': self._bootstrap_metrics(),
            'bayesian': self._bayesian_analysis(),
        }
        
        return results
    
    def _analyze_distributions(self) -> Dict:
        """Fit and test distributions."""
        dist_results = {}
        
        for asset in self.assets:
            returns = self.returns[asset]
            fitter = FinancialDistributionFitter(returns)
            dist_results[asset] = {
                'normal': fitter.fit_normal(),
                't': fitter.fit_t(),
                'moments': analyze_distribution_moments(returns),
            }
        
        return dist_results
    
    def _analyze_correlations(self) -> Dict:
        """Analyze correlation structure."""
        return {
            'pearson': self.returns.corr(method='pearson'),
            'spearman': self.returns.corr(method='spearman'),
        }
    
    def _bootstrap_metrics(self) -> Dict:
        """Bootstrap confidence intervals."""
        boot_results = {}
        
        for asset in self.assets:
            returns = self.returns[asset]
            boot_results[asset] = bootstrap_sharpe_ratio(returns)
        
        return boot_results
    
    def _bayesian_analysis(self) -> Dict:
        """Bayesian estimation."""
        bayes_results = {}
        
        for asset in self.assets:
            returns = self.returns[asset]
            bayes_results[asset] = bayesian_mean_return_estimation(returns)
        
        return bayes_results
    
    def summary_report(self) -> str:
        """Generate text summary of analysis."""
        results = self.complete_analysis()
        
        report = "="*60 + "\n"
        report += "FINANCIAL STATISTICS ANALYSIS REPORT\n"
        report += "="*60 + "\n\n"
        
        for asset in self.assets:
            report += f"\n{asset}\n" + "-"*40 + "\n"
            
            dist = results['distributions'][asset]
            report += f"Distribution: t(df={dist['t']['df']:.1f})\n"
            
            moments = dist['moments']
            report += f"Skewness: {moments['skewness']:.3f}\n"
            report += f"Excess Kurtosis: {moments['excess_kurtosis']:.3f}\n"
        
        return report


# Usage
if __name__ == "__main__":
    # Load data
    tickers = ['^GSPC', 'AAPL', 'MSFT']
    returns_dict = {t: load_financial_returns(t, period='5y') for t in tickers}
    returns_df = pd.DataFrame(returns_dict)
    
    # Analyze
    toolkit = FinancialStatsToolkit(returns_df)
    results = toolkit.complete_analysis()
    
    print(toolkit.summary_report())
```

This chapter provides you with robust statistical tools for financial analysis. Master these concepts, and you'll build systems that work with real market data—not theoretical assumptions.


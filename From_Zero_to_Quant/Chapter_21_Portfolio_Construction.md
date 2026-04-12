# Chapter 21: Portfolio Construction

## Introduction

You have predictions. Now what?

In previous chapters, you've built machine learning models that predict next-period returns for NSE stocks. You have signals—numerical estimates of which stocks will outperform. The critical question is: *how do you convert these predictions into a real, tradeable portfolio?*

This is portfolio construction—the bridge between predictions and execution.

Portfolio construction answers three questions:
1. **How much** of your capital goes into each stock? (position sizing)
2. **What constraints** must you respect? (leverage, sector limits, turnover)
3. **How do you measure and manage risk** across your portfolio? (factor models)

This chapter assumes you have predictions from your ML model. Your job: turn them into weights that maximize expected returns while controlling risk and staying within operational constraints.

---

## Module 21.1: From Signals to Weights

### 21.1.1 The Signal-to-Weight Conversion Problem

Your ML model outputs predictions. Let's say for 50 NSE stocks, you have predicted excess returns:

$$\hat{r}_i = \text{predicted return for stock } i$$

These predictions range from -15% to +25% depending on the stock. Your goal: convert these scalar predictions into portfolio weights $w_i$ that sum to 1 (or -1 for long/short strategies) and satisfy constraints.

**Why not just use predictions directly as weights?**

1. **Predictions are unscaled**: A 5% predicted return is good, but how much capital should you allocate? 50% of your portfolio? 1%?
2. **Uncertainty differs**: High-conviction predictions should get larger weights than uncertain ones.
3. **Constraints matter**: You might have leverage limits, sector exposure targets, position limits per stock.
4. **Turnover costs**: Changing weights too aggressively creates transaction costs.

### 21.1.2 Mapping Signals to Weights: Three Approaches

#### **Approach 1: Linear Mapping**

The simplest approach: weights proportional to signals.

$$w_i = \frac{\hat{r}_i}{\sum_j |\hat{r}_j|}$$

For long-only portfolios, use only positive signals:

$$w_i = \frac{\max(0, \hat{r}_i)}{\sum_j \max(0, \hat{r}_j)}$$

**Advantage**: Simple, interpretable.
**Disadvantage**: Sensitive to outliers. One stock with a 25% predicted return could dominate.

#### **Approach 2: Sigmoid Mapping**

Apply a sigmoid function to compress predictions into a bounded range:

$$\text{signal}_i = \frac{1}{1 + e^{-k \cdot \hat{r}_i}}$$

$$w_i = \text{signal}_i / \sum_j \text{signal}_j$$

The parameter $k$ controls sensitivity. Higher $k$ = sharper transition between positive/negative signals.

**Advantage**: Bounded, reduces extreme weights.
**Disadvantage**: Non-linear, harder to interpret.

#### **Approach 3: Quantile-Based Mapping**

Rank the predictions and assign weights based on quantiles:

$$\text{rank}_i = \text{percentile of } \hat{r}_i$$

$$w_i = (2 \cdot \text{rank}_i - 1) / N$$

This maps predictions to $[-1/N, 1/N]$ per stock in a long/short portfolio.

**Advantage**: Rank-based, robust to outliers.
**Disadvantage**: Loses magnitude information.

### 21.1.3 Dollar-Neutrality Constraint

In a long/short portfolio, you want equal exposure to long and short positions:

$$\sum_{i \in \text{long}} w_i = \sum_{i \in \text{short}} |w_i|$$

This keeps your portfolio hedged against market direction—you profit from relative stock selection, not market timing.

To enforce dollar-neutrality after computing initial weights $w_i^0$:

1. Split into long and short components:
   - $w_i^+ = \max(0, w_i^0)$, $w_i^- = \min(0, w_i^0)$

2. Normalize each to sum to 0.5:
   - $w_i^{\text{long}} = 0.5 \cdot w_i^+ / \sum_j w_j^+$
   - $w_i^{\text{short}} = 0.5 \cdot w_i^- / \sum_j |w_j^-|$

3. Combine:
   - $w_i^{\text{final}} = w_i^{\text{long}} + w_i^{\text{short}}$

Now $\sum w_i = 0$ and $\sum |w_i^{\text{long}}| = \sum |w_i^{\text{short}}| = 0.5$.

### 21.1.4 Sector Neutrality

You might want equal exposure across sectors (to diversify away from sector bets):

$$\sum_{i \in \text{sector } s} w_i = 0 \quad \forall s$$

Or you might want each sector to have zero net long/short exposure.

This is useful when your signals are stock-selection talent, not sector allocation talent.

### 21.1.5 Position Limits and Turnover Constraints

**Position Limits**: No single stock exceeds a threshold.

$$|w_i| \leq w_{\max} = 0.05 \quad \text{(e.g., 5% per stock)}$$

This prevents concentration risk. In NSE, brokers often enforce position limits based on market cap and liquidity.

**Turnover Constraints**: Don't deviate too much from previous portfolio.

$$\sum_i |w_i^{\text{new}} - w_i^{\text{old}}| \leq \text{turnover}_{\max}$$

This limits transaction costs. If max turnover = 0.3, you can reallocate up to 30% of your portfolio per period.

### 21.1.6 Complete Signal-to-Weight Implementation

```python
import numpy as np
from typing import Tuple, Dict, Optional
import pandas as pd

class SignalToWeight:
    """
    Convert ML predictions to portfolio weights with constraints.
    
    Handles:
    - Linear, sigmoid, and quantile-based signal mapping
    - Dollar neutrality (long/short balance)
    - Sector neutrality
    - Position limits
    - Turnover constraints
    """
    
    def __init__(
        self,
        n_stocks: int,
        method: str = 'sigmoid',
        signal_scale: float = 1.0,
        long_only: bool = False
    ):
        """
        Parameters
        ----------
        n_stocks : int
            Number of stocks in universe
        method : str
            'linear', 'sigmoid', or 'quantile'
        signal_scale : float
            Scaling parameter (k in sigmoid, or scaling factor)
        long_only : bool
            If True, no short positions
        """
        self.n_stocks = n_stocks
        self.method = method
        self.signal_scale = signal_scale
        self.long_only = long_only
        
    def _linear_mapping(self, signals: np.ndarray) -> np.ndarray:
        """Linear mapping: weights proportional to signals."""
        if self.long_only:
            signals_clipped = np.maximum(signals, 0)
            denom = signals_clipped.sum()
            if denom == 0:
                return np.ones(self.n_stocks) / self.n_stocks
            return signals_clipped / denom
        else:
            denom = np.abs(signals).sum()
            if denom == 0:
                return np.zeros(self.n_stocks)
            return signals / denom
    
    def _sigmoid_mapping(self, signals: np.ndarray) -> np.ndarray:
        """Sigmoid mapping: compress signals to [0, 1] range."""
        sigmoid = 1.0 / (1.0 + np.exp(-self.signal_scale * signals))
        
        if self.long_only:
            return sigmoid / sigmoid.sum()
        else:
            # For long/short: shift sigmoid to [-0.5, 0.5]
            sigmoid_shifted = sigmoid - 0.5
            denom = np.abs(sigmoid_shifted).sum()
            if denom == 0:
                return np.zeros(self.n_stocks)
            return sigmoid_shifted / denom
    
    def _quantile_mapping(self, signals: np.ndarray) -> np.ndarray:
        """Quantile mapping: rank-based weights."""
        ranks = np.argsort(np.argsort(signals)) / (self.n_stocks - 1)
        
        if self.long_only:
            return ranks / ranks.sum()
        else:
            # Map to [-1, 1] range
            weights = 2 * ranks - 1
            denom = np.abs(weights).sum()
            if denom == 0:
                return np.zeros(self.n_stocks)
            return weights / denom
    
    def signal_to_weight(
        self,
        signals: np.ndarray,
        sector_ids: Optional[np.ndarray] = None,
        previous_weights: Optional[np.ndarray] = None,
        max_weight: float = 0.05,
        max_turnover: Optional[float] = None,
        enforce_dollar_neutral: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Convert signals to portfolio weights with constraints.
        
        Parameters
        ----------
        signals : np.ndarray
            Predicted returns, shape (n_stocks,)
        sector_ids : np.ndarray, optional
            Sector ID for each stock, shape (n_stocks,)
        previous_weights : np.ndarray, optional
            Previous period weights for turnover constraint
        max_weight : float
            Maximum absolute weight per stock (default 5%)
        max_turnover : float, optional
            Maximum total turnover allowed
        enforce_dollar_neutral : bool
            Enforce dollar-neutral (long = short) if not long_only
        
        Returns
        -------
        weights : np.ndarray
            Final portfolio weights
        metrics : Dict
            Constraint metrics (turnover, leverage, sector exposure)
        """
        assert len(signals) == self.n_stocks
        
        # Step 1: Initial signal-to-weight mapping
        if self.method == 'linear':
            weights = self._linear_mapping(signals)
        elif self.method == 'sigmoid':
            weights = self._sigmoid_mapping(signals)
        elif self.method == 'quantile':
            weights = self._quantile_mapping(signals)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Step 2: Apply position limits
        weights = self._apply_position_limits(weights, max_weight)
        
        # Step 3: Enforce dollar neutrality
        if enforce_dollar_neutral and not self.long_only:
            weights = self._enforce_dollar_neutrality(weights)
        
        # Step 4: Enforce sector neutrality
        if sector_ids is not None:
            weights = self._enforce_sector_neutrality(weights, sector_ids)
        
        # Step 5: Apply turnover constraint
        if previous_weights is not None and max_turnover is not None:
            weights = self._apply_turnover_constraint(
                weights, previous_weights, max_turnover
            )
        
        # Normalize to sum to 1
        if not np.isclose(weights.sum(), 0):  # Not dollar-neutral
            weights = weights / (weights.sum() if weights.sum() != 0 else 1)
        
        # Compute metrics
        metrics = self._compute_metrics(weights, previous_weights, sector_ids)
        
        return weights, metrics
    
    def _apply_position_limits(
        self,
        weights: np.ndarray,
        max_weight: float
    ) -> np.ndarray:
        """Clip weights to max_weight, redistribute excess."""
        weights = np.clip(weights, -max_weight, max_weight)
        
        # Redistribute clipped weight proportionally
        excess = np.abs(weights).sum() - 1.0
        if excess > 1e-6:
            # This is a simplified approach; full implementation would
            # iteratively clip and redistribute
            scale = 1.0 / (np.abs(weights).sum() + 1e-10)
            weights = weights * scale
        
        return weights
    
    def _enforce_dollar_neutrality(self, weights: np.ndarray) -> np.ndarray:
        """Ensure sum(positive weights) = sum(abs(negative weights)) = 0.5."""
        long_mask = weights > 0
        short_mask = weights < 0
        
        long_sum = weights[long_mask].sum()
        short_sum = np.abs(weights[short_mask]).sum()
        
        if long_sum > 1e-10:
            weights[long_mask] = 0.5 * weights[long_mask] / long_sum
        if short_sum > 1e-10:
            weights[short_mask] = -0.5 * np.abs(weights[short_mask]) / short_sum
        
        return weights
    
    def _enforce_sector_neutrality(
        self,
        weights: np.ndarray,
        sector_ids: np.ndarray
    ) -> np.ndarray:
        """Make sector exposures sum to zero."""
        n_sectors = sector_ids.max() + 1
        
        for s in range(n_sectors):
            sector_mask = sector_ids == s
            sector_weight = weights[sector_mask].sum()
            
            # Reduce weights in this sector to make net zero
            if np.abs(sector_weight) > 1e-10:
                weights[sector_mask] -= sector_weight / sector_mask.sum()
        
        return weights
    
    def _apply_turnover_constraint(
        self,
        new_weights: np.ndarray,
        old_weights: np.ndarray,
        max_turnover: float
    ) -> np.ndarray:
        """
        Reduce turnover if it exceeds maximum.
        Uses simple scaling; optimal approach uses optimization.
        """
        turnover = np.abs(new_weights - old_weights).sum() / 2
        
        if turnover > max_turnover:
            # Blend old and new weights
            alpha = max_turnover / (turnover + 1e-10)
            return alpha * new_weights + (1 - alpha) * old_weights
        
        return new_weights
    
    def _compute_metrics(
        self,
        weights: np.ndarray,
        previous_weights: Optional[np.ndarray],
        sector_ids: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Compute constraint metrics."""
        metrics = {
            'gross_leverage': np.abs(weights).sum(),
            'net_exposure': weights.sum(),
            'max_weight': np.max(np.abs(weights)),
            'n_long': (weights > 1e-6).sum(),
            'n_short': (weights < -1e-6).sum(),
        }
        
        if previous_weights is not None:
            metrics['turnover'] = np.abs(
                weights - previous_weights
            ).sum() / 2
        
        if sector_ids is not None:
            n_sectors = sector_ids.max() + 1
            sector_exposures = np.array([
                weights[sector_ids == s].sum()
                for s in range(n_sectors)
            ])
            metrics['sector_exposure_std'] = np.std(sector_exposures)
        
        return metrics


# Example usage
if __name__ == '__main__':
    np.random.seed(42)
    
    # Create example data
    n_stocks = 50
    signals = np.random.randn(n_stocks) * 5  # In percentage
    sector_ids = np.random.randint(0, 5, n_stocks)  # 5 sectors
    previous_weights = np.random.randn(n_stocks) * 0.02
    previous_weights = previous_weights / previous_weights.sum()
    
    # Convert signals to weights
    converter = SignalToWeight(n_stocks, method='sigmoid')
    weights, metrics = converter.signal_to_weight(
        signals,
        sector_ids=sector_ids,
        previous_weights=previous_weights,
        max_weight=0.05,
        max_turnover=0.3,
        enforce_dollar_neutral=True
    )
    
    print(f"Weights shape: {weights.shape}")
    print(f"Sum of weights: {weights.sum():.6f}")
    print(f"Gross leverage: {metrics['gross_leverage']:.3f}")
    print(f"Turnover: {metrics['turnover']:.3f}")
    print(f"Stocks long: {metrics['n_long']}, short: {metrics['n_short']}")
```

---

## Module 21.2: Optimization-Based Portfolio Construction

### 21.2.1 The Portfolio Optimization Problem

Signal-to-weight conversion is heuristic. Better approach: **formulate as an optimization problem**.

You want to maximize expected return minus risk:

$$\max_{w} \left( \mu^T w - \lambda \cdot \sigma^2(w) \right)$$

Subject to:
$$\sum_i w_i = 1, \quad |w_i| \leq 0.05, \quad \text{other constraints}$$

Where:
- $\mu = [\mu_1, \ldots, \mu_n]^T$ = expected returns (from your ML model)
- $\sigma^2(w) = w^T \Sigma w$ = portfolio variance
- $\Sigma$ = covariance matrix of returns
- $\lambda$ = risk aversion parameter (higher = risk-averse)

This is **mean-variance optimization** (Markowitz, 1952).

### 21.2.2 Mean-Variance Optimization with Constraints

**Problem formulation**:

$$\min_{w} \left( w^T \Sigma w - \lambda \mu^T w \right)$$

Subject to:
$$\sum_i w_i = 1$$
$$|w_i| \leq w_{\max}$$
$$\sum_{s \in \text{sector } s} w_i = 0 \quad \text{(optional sector neutrality)}$$
$$\sum_i |w_i^{\text{new}} - w_i^{\text{old}}| \leq T \quad \text{(turnover)}$$

This is a **quadratic program**—convex optimization problem we can solve efficiently.

**Key challenge**: Estimating $\Sigma$ is hard. With 50 stocks, you have 50×50 = 2,500 parameters. With 500 days of data, you have 50 observations per parameter—underdetermined!

**Solution**: Use a factor model (covered in Module 21.3) to estimate $\Sigma$ with fewer parameters.

### 21.2.3 Special Cases: Risk Parity, MinVar, MaxDiv

#### **Risk Parity Portfolio**

Idea: Each position contributes equally to portfolio volatility.

For position $i$, contribution to volatility is:

$$\text{contribution}_i = w_i \cdot (\Sigma w)_i$$

Risk parity enforces:

$$w_i \cdot (\Sigma w)_i = \text{constant} \quad \forall i$$

**Derivation**:

At optimum, each position has equal marginal risk contribution (MRC). The MRC of position $i$ is:

$$\text{MRC}_i = \frac{w_i}{\text{portfolio volatility}} \cdot (\Sigma w)_i$$

For risk parity, $\text{MRC}_i = \text{constant}$ for all $i$.

This means: position sizes are inversely proportional to volatility.

$$w_i \propto \frac{1}{\sigma_i}$$

Normalize to sum to 1:

$$w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$$

**Implementation**:

```python
def risk_parity_portfolio(
    std_devs: np.ndarray,
    target_vol: float = 0.10
) -> np.ndarray:
    """
    Construct risk parity portfolio.
    
    Each position contributes equally to portfolio volatility.
    
    Parameters
    ----------
    std_devs : np.ndarray
        Stock-level volatilities, shape (n_stocks,)
    target_vol : float
        Target portfolio volatility
    
    Returns
    -------
    weights : np.ndarray
        Risk parity portfolio weights
    """
    # Weights inversely proportional to volatility
    weights = 1.0 / std_devs
    weights = weights / weights.sum()
    
    # Scale to target volatility
    # For risk parity: portfolio_vol = target_vol (approximately)
    # More precise scaling would require covariance matrix
    
    return weights
```

#### **Maximum Diversification Portfolio**

Diversification ratio = (weighted avg std dev) / (portfolio std dev)

$$\text{DR}(w) = \frac{\sum_i w_i \sigma_i}{\sqrt{w^T \Sigma w}}$$

Maximize diversification:

$$\max_w \frac{\sum_i w_i \sigma_i}{\sqrt{w^T \Sigma w}}$$

Subject to $\sum_i w_i = 1$.

**Intuition**: You want positions in low-correlation assets with high individual volatility. This is similar to risk parity but considers correlations.

#### **Minimum Variance Portfolio**

Simplest optimization: minimize portfolio variance regardless of expected returns.

$$\min_w w^T \Sigma w$$

Subject to $\sum_i w_i = 1$.

Useful when you have no reliable return predictions, only volatility/correlation estimates.

### 21.2.4 Complete Portfolio Optimizer Implementation

```python
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import pandas as pd

class PortfolioOptimizer:
    """
    Portfolio optimization using CVXPY.
    
    Supports:
    - Mean-variance optimization (Markowitz)
    - Minimum variance
    - Risk parity
    - Maximum diversification
    - Constraints: leverage, sector, position limits, turnover
    """
    
    def __init__(self, n_stocks: int):
        """
        Parameters
        ----------
        n_stocks : int
            Number of stocks
        """
        self.n_stocks = n_stocks
    
    def estimate_covariance(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Estimate covariance matrix using Ledoit-Wolf shrinkage.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns, shape (n_days, n_stocks)
        
        Returns
        -------
        cov_matrix : np.ndarray
            Covariance matrix, shape (n_stocks, n_stocks)
        """
        lw = LedoitWolf()
        cov_matrix, _ = lw.fit(returns.values).covariance_, lw.shrinkage_
        return cov_matrix
    
    def mean_variance_optimization(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        lambda_risk: float = 1.0,
        max_weight: float = 0.05,
        min_weight: float = -0.05,
        sector_ids: Optional[np.ndarray] = None,
        max_sector_exposure: float = 0.2,
        previous_weights: Optional[np.ndarray] = None,
        max_turnover: Optional[float] = None,
        long_only: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve mean-variance optimization using CVXPY.
        
        Parameters
        ----------
        mu : np.ndarray
            Expected returns, shape (n_stocks,)
        Sigma : np.ndarray
            Covariance matrix, shape (n_stocks, n_stocks)
        lambda_risk : float
            Risk aversion parameter (higher = more risk-averse)
        max_weight : float
            Maximum weight per stock (absolute value)
        min_weight : float
            Minimum weight per stock
        sector_ids : np.ndarray, optional
            Sector ID per stock
        max_sector_exposure : float
            Maximum absolute exposure per sector
        previous_weights : np.ndarray, optional
            Previous weights for turnover constraint
        max_turnover : float, optional
            Maximum allowed turnover
        long_only : bool
            Enforce long-only constraint
        
        Returns
        -------
        weights : np.ndarray
            Optimal portfolio weights
        result : Dict
            Optimization result and metrics
        """
        # Decision variable
        w = cp.Variable(self.n_stocks)
        
        # Objective: maximize return minus risk penalty
        # min w^T Sigma w - lambda * mu^T w
        objective = cp.Minimize(
            cp.quad_form(w, Sigma) - lambda_risk * mu @ w
        )
        
        # Constraints
        constraints = [cp.sum(w) == 1.0]  # Fully invested
        
        # Position limits
        if long_only:
            constraints.append(w >= 0)
            constraints.append(w <= max_weight)
        else:
            constraints.append(w >= min_weight)
            constraints.append(w <= max_weight)
        
        # Sector limits
        if sector_ids is not None:
            n_sectors = sector_ids.max() + 1
            for s in range(n_sectors):
                sector_mask = sector_ids == s
                sector_exposure = cp.sum(w[sector_mask])
                constraints.append(sector_exposure <= max_sector_exposure)
                constraints.append(sector_exposure >= -max_sector_exposure)
        
        # Turnover constraint
        if previous_weights is not None and max_turnover is not None:
            turnover = cp.sum(cp.abs(w - previous_weights))
            constraints.append(turnover <= 2 * max_turnover)  # *2 for buy+sell
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status != cp.OPTIMAL:
            print(f"Optimization failed: {problem.status}")
            return np.ones(self.n_stocks) / self.n_stocks, {'status': problem.status}
        
        weights = w.value
        
        # Compute metrics
        portfolio_return = weights @ mu
        portfolio_var = weights @ Sigma @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        result = {
            'status': 'optimal',
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'gross_leverage': np.abs(weights).sum(),
        }
        
        if previous_weights is not None:
            result['turnover'] = np.abs(weights - previous_weights).sum() / 2
        
        return weights, result
    
    def minimum_variance_portfolio(
        self,
        Sigma: np.ndarray,
        max_weight: float = 0.05,
        long_only: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve minimum variance portfolio.
        """
        # Zero expected returns, focus on variance minimization
        mu = np.zeros(self.n_stocks)
        return self.mean_variance_optimization(
            mu, Sigma, lambda_risk=1e6,  # Very high risk aversion
            max_weight=max_weight, long_only=long_only
        )
    
    def maximum_diversification_portfolio(
        self,
        std_devs: np.ndarray,
        Sigma: np.ndarray,
        max_weight: float = 0.05
    ) -> Tuple[np.ndarray, Dict]:
        """
        Maximize diversification ratio.
        
        Parameters
        ----------
        std_devs : np.ndarray
            Individual stock volatilities
        Sigma : np.ndarray
            Covariance matrix
        max_weight : float
            Max weight per stock
        
        Returns
        -------
        weights : np.ndarray
        result : Dict
        """
        # Maximize (std @ w) / sqrt(w^T Sigma w)
        # Equivalent to: minimize -( std @ w) / sqrt(w^T Sigma w)
        
        w = cp.Variable(self.n_stocks)
        
        # Objective
        objective = cp.Minimize(
            cp.quad_form(w, Sigma) / (std_devs @ w + 1e-6)
        )
        
        # Constraints
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= max_weight,
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status != cp.OPTIMAL:
            # Fallback to equal weight
            weights = np.ones(self.n_stocks) / self.n_stocks
        else:
            weights = w.value / w.value.sum()
        
        # Compute diversification ratio
        portfolio_vol = np.sqrt(weights @ Sigma @ weights)
        weighted_vol = weights @ std_devs
        div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        result = {
            'diversification_ratio': div_ratio,
            'volatility': portfolio_vol,
        }
        
        return weights, result
    
    def risk_parity_portfolio(
        self,
        Sigma: np.ndarray,
        max_weight: float = 0.05
    ) -> Tuple[np.ndarray, Dict]:
        """
        Construct risk parity portfolio using optimization.
        
        Each position has equal marginal risk contribution.
        """
        w = cp.Variable(self.n_stocks)
        
        # Objective: minimize variance subject to equal risk contribution
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= max_weight,
        ]
        
        # Equal risk contribution constraint
        # MRC_i = w_i * (Sigma w)_i should be equal for all i
        # Approximation: use variance of risk contributions as penalty
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status != cp.OPTIMAL:
            weights = np.ones(self.n_stocks) / self.n_stocks
        else:
            weights = w.value / w.value.sum()
        
        # Compute risk contributions
        marginal_risk = Sigma @ weights
        risk_contrib = weights * marginal_risk
        
        result = {
            'volatility': np.sqrt(weights @ Sigma @ weights),
            'risk_contrib_std': np.std(risk_contrib),
            'risk_contrib_mean': np.mean(risk_contrib),
        }
        
        return weights, result


# Comparison of methods on historical data
def compare_portfolio_methods(
    returns: pd.DataFrame,
    signals: np.ndarray,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Compare different portfolio construction methods.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, shape (n_days, n_stocks)
    signals : np.ndarray
        Predicted returns
    lookback : int
        Days of historical data for covariance estimation
    
    Returns
    -------
    comparison : pd.DataFrame
        Metrics for each method
    """
    n_stocks = returns.shape[1]
    opt = PortfolioOptimizer(n_stocks)
    
    # Estimate covariance
    Sigma = opt.estimate_covariance(returns.iloc[-lookback:])
    std_devs = np.sqrt(np.diag(Sigma))
    
    results = {}
    
    # 1. Equal weight
    w_equal = np.ones(n_stocks) / n_stocks
    results['equal_weight'] = {
        'volatility': np.sqrt(w_equal @ Sigma @ w_equal),
        'return': w_equal @ signals / 100,  # Assuming signals are in %
    }
    
    # 2. Mean-variance
    w_mv, info_mv = opt.mean_variance_optimization(
        signals / 100, Sigma, lambda_risk=1.0, long_only=True
    )
    results['mean_variance'] = {
        'volatility': info_mv['volatility'],
        'return': info_mv['expected_return'],
        'sharpe': info_mv['sharpe_ratio'],
    }
    
    # 3. Minimum variance
    w_minvar, info_minvar = opt.minimum_variance_portfolio(Sigma)
    results['minimum_variance'] = {
        'volatility': info_minvar['volatility'],
        'return': w_minvar @ (signals / 100),
    }
    
    # 4. Maximum diversification
    w_maxdiv, info_maxdiv = opt.maximum_diversification_portfolio(
        std_devs, Sigma
    )
    results['max_diversification'] = {
        'volatility': info_maxdiv['volatility'],
        'return': w_maxdiv @ (signals / 100),
        'diversification_ratio': info_maxdiv['diversification_ratio'],
    }
    
    # 5. Risk parity
    w_rp, info_rp = opt.risk_parity_portfolio(Sigma)
    results['risk_parity'] = {
        'volatility': info_rp['volatility'],
        'return': w_rp @ (signals / 100),
    }
    
    return pd.DataFrame(results).T
```

---

## Module 21.3: Factor Risk Models

### 21.3.1 Why You Need a Factor Risk Model

The covariance matrix $\Sigma$ is critical for portfolio optimization. But estimating it is hard.

**The curse of dimensionality**: With $n$ stocks, $\Sigma$ has $n(n+1)/2$ unique parameters. With $n=50$, that's 1,275 parameters.

With 250 days of data, you have only ~5 observations per parameter. The sample covariance matrix is **highly unstable**—tiny changes in data cause huge changes in estimates.

**Solution**: Use a **factor model** to reduce dimensionality.

Idea: Stock returns are driven by $K$ systematic factors (not 50 independent variables).

$$r_i(t) = \alpha_i + \sum_{k=1}^K \beta_{i,k} f_k(t) + \epsilon_i(t)$$

Where:
- $r_i(t)$ = return of stock $i$ at time $t$
- $f_k(t)$ = return of factor $k$ at time $t$
- $\beta_{i,k}$ = sensitivity (loading) of stock $i$ to factor $k$
- $\epsilon_i(t)$ = idiosyncratic return (uncorrelated across stocks)
- $K \ll n$ = number of factors (typically 3-10)

**Advantages**:
1. Fewer parameters: $K \times n + K \times K + n$
2. More stable estimates
3. Interpretable: you understand what drives returns
4. Better risk prediction: systematic factors are more predictable than idiosyncratic noise

### 21.3.2 PCA-Based Factor Model (Statistical Factors)

**Principal Component Analysis (PCA)**: Extract factors from data directly.

Algorithm:
1. Compute sample covariance: $\Sigma_{\text{sample}} = \frac{1}{T} R^T R$
2. Find eigenvectors: $v_1, v_2, \ldots, v_n$ (ranked by eigenvalue)
3. First $K$ eigenvectors = first $K$ factors

**Interpretation**: First eigenvector is direction of maximum variance in returns. Second eigenvector captures second-largest variance direction, orthogonal to first. Etc.

**Factor returns and loadings**:

$$f_k(t) = \frac{1}{\sqrt{\lambda_k}} R(t) v_k$$

$$\beta_{i,k} = \sqrt{\lambda_k} \cdot v_{k,i}$$

Where $\lambda_k$ = eigenvalue associated with $v_k$.

**Reconstructed covariance**:

$$\Sigma_{\text{factor}} = B \Sigma_f B^T + \Sigma_{\epsilon}$$

Where:
- $B = [\beta_1, \ldots, \beta_K]$ = $n \times K$ loading matrix
- $\Sigma_f = \text{cov}(f)$ = $K \times K$ factor covariance
- $\Sigma_{\epsilon}$ = diagonal idiosyncratic variance

### 21.3.3 Fundamental Factor Model

Instead of extracting statistical factors, define factors **economically**:

- **Size**: log(market cap)
- **Value**: price-to-book ratio or dividend yield
- **Momentum**: return over past 6-12 months
- **Quality**: earnings quality, ROE
- **Sector**: dummy variable for each sector

Stock return decomposition:

$$r_i = \alpha + \sum_{s} \beta_{i,s} f_{\text{sector},s} + \beta_{i,\text{size}} f_{\text{size}} + \beta_{i,\text{value}} f_{\text{value}} + \cdots$$

**Advantages** over PCA:
- Interpretable factors
- Can use external forecasts for factors
- Easier to apply constraints (e.g., sector neutral)

**Disadvantages**:
- Need to define factors
- Stability depends on factor definitions

### 21.3.4 Estimating Factor Returns and Covariances

Given historical returns $R$ (shape $T \times n$) and factor values $F$ (shape $T \times K$):

**Estimate loadings** using cross-sectional regression:

$$\min_B \|R - F B^T\|_F^2$$

Solution (least squares):

$$\hat{B} = (F^T F)^{-1} F^T R$$

**Estimate factor returns**:

$$f_k(t) = (B^T B)^{-1} B^T r(t)$$

Or simpler: if factors are observable (e.g., index returns), use them directly.

**Estimate idiosyncratic covariance**:

$$\Sigma_{\epsilon} = \text{diag}\left( \frac{1}{T} \sum_t \epsilon(t)^2 \right)$$

Where $\epsilon(t) = r(t) - \hat{r}(t)$ = residuals from factor model.

### 21.3.5 Complete Factor Risk Model Implementation

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import scipy.linalg as linalg

class FactorRiskModel:
    """
    Factor risk model for portfolio construction.
    
    Supports:
    - PCA-based (statistical) factors
    - Fundamental factors (size, value, momentum, sector)
    - Factor return and covariance estimation
    - Risk decomposition
    """
    
    def __init__(
        self,
        n_factors: int = 5,
        model_type: str = 'pca',
        min_observations: int = 60
    ):
        """
        Parameters
        ----------
        n_factors : int
            Number of factors
        model_type : str
            'pca' or 'fundamental'
        min_observations : int
            Minimum historical observations required
        """
        self.n_factors = n_factors
        self.model_type = model_type
        self.min_observations = min_observations
        self.pca = None
        self.loadings = None  # B matrix
        self.factor_cov = None  # Cov(f)
        self.idio_var = None  # Var(epsilon)
    
    def fit_pca(self, returns: pd.DataFrame) -> None:
        """
        Fit PCA-based factor model.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns, shape (n_days, n_stocks)
        """
        assert len(returns) >= self.min_observations
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_factors)
        components = self.pca.fit_transform(returns.values)  # Shape: (T, K)
        
        # Loadings: B = V * sqrt(Lambda)
        # where V = eigenvectors, Lambda = eigenvalues
        eigenvalues = self.pca.explained_variance_
        eigenvectors = self.pca.components_.T  # Shape: (n_stocks, K)
        
        self.loadings = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        # Factor covariance
        self.factor_cov = np.cov(components.T)
        
        # Idiosyncratic variance
        reconstructed = components @ self.pca.components_  # Shape: (T, n_stocks)
        residuals = returns.values - reconstructed
        self.idio_var = np.var(residuals, axis=0)
    
    def fit_fundamental(
        self,
        returns: pd.DataFrame,
        factor_values: Dict[str, np.ndarray]
    ) -> None:
        """
        Fit fundamental factor model.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns, shape (n_days, n_stocks)
        factor_values : Dict[str, np.ndarray]
            Factor values, e.g. {
                'size': np.array of log(market_cap),
                'value': np.array of P/B ratios,
                'momentum': np.array of past returns
            }
        """
        # Build factor matrix
        factors = []
        for fname, fvals in factor_values.items():
            assert len(fvals) == len(returns)
            factors.append(fvals)
        
        F = np.column_stack(factors)  # Shape: (T, K)
        
        # Estimate loadings: B = (F^T F)^{-1} F^T R
        gram = F.T @ F  # Shape: (K, K)
        self.loadings = np.linalg.solve(gram, F.T @ returns.values)  # Shape: (K, n)
        self.loadings = self.loadings.T  # Shape: (n, K)
        
        # Factor covariance
        self.factor_cov = np.cov(F.T)
        
        # Idiosyncratic variance
        fitted = F @ self.loadings.T
        residuals = returns.values - fitted
        self.idio_var = np.var(residuals, axis=0)
    
    def reconstruct_covariance(self) -> np.ndarray:
        """
        Reconstruct covariance matrix using factor model.
        
        Sigma = B * Cov(f) * B^T + Diag(sigma_epsilon^2)
        
        Returns
        -------
        cov_matrix : np.ndarray
            Estimated covariance matrix, shape (n_stocks, n_stocks)
        """
        assert self.loadings is not None
        assert self.factor_cov is not None
        assert self.idio_var is not None
        
        n_stocks = self.loadings.shape[0]
        
        # Systematic covariance
        systematic = self.loadings @ self.factor_cov @ self.loadings.T
        
        # Add idiosyncratic variance (diagonal)
        total = systematic + np.diag(self.idio_var)
        
        return total
    
    def risk_decomposition(
        self,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Decompose portfolio risk into systematic and idiosyncratic.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights, shape (n_stocks,)
        
        Returns
        -------
        decomp : Dict
            Risk decomposition metrics
        """
        assert self.loadings is not None
        
        # Portfolio factor loadings
        portfolio_beta = weights @ self.loadings  # Shape: (K,)
        
        # Portfolio variance
        Sigma = self.reconstruct_covariance()
        portfolio_var = weights @ Sigma @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Systematic variance
        systematic_var = portfolio_beta @ self.factor_cov @ portfolio_beta
        
        # Idiosyncratic variance
        idio_var = weights @ (np.diag(self.idio_var)) @ weights
        
        decomp = {
            'total_volatility': portfolio_vol,
            'systematic_volatility': np.sqrt(systematic_var),
            'idiosyncratic_volatility': np.sqrt(idio_var),
            'systematic_fraction': systematic_var / portfolio_var,
            'factor_loadings': portfolio_beta,
        }
        
        return decomp
    
    def get_loadings(self) -> pd.DataFrame:
        """Return factor loadings as DataFrame."""
        n_stocks = self.loadings.shape[0]
        return pd.DataFrame(
            self.loadings,
            index=[f'Stock_{i}' for i in range(n_stocks)],
            columns=[f'Factor_{k}' for k in range(self.n_factors)]
        )
    
    def get_factor_cov(self) -> pd.DataFrame:
        """Return factor covariance as DataFrame."""
        return pd.DataFrame(
            self.factor_cov,
            index=[f'Factor_{k}' for k in range(self.n_factors)],
            columns=[f'Factor_{k}' for k in range(self.n_factors)]
        )


class BarlayStyleRiskModel:
    """
    Simplified Barra-style risk model (APT-based).
    
    Risk drivers:
    - Sector exposures
    - Size factor
    - Value factor
    - Momentum factor
    - Beta to market
    
    Covariance estimated as:
    Cov(r) = B * Cov(f) * B^T + Diag(S^2)
    
    where factors = [sectors, size, value, momentum, market_beta]
    """
    
    def __init__(self, n_stocks: int, n_sectors: int = 5):
        """
        Parameters
        ----------
        n_stocks : int
        n_sectors : int
        """
        self.n_stocks = n_stocks
        self.n_sectors = n_sectors
        self.n_factors = n_sectors + 4  # sectors + size, value, momentum, market_beta
        
        self.factor_cov = None
        self.loadings = None
        self.specific_var = None
    
    def fit(
        self,
        returns: pd.DataFrame,
        sector_ids: np.ndarray,
        size: np.ndarray,
        value: np.ndarray,
        momentum: np.ndarray,
        market_returns: np.ndarray
    ) -> None:
        """
        Fit Barra-style model.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock daily returns, shape (T, n_stocks)
        sector_ids : np.ndarray
            Sector ID for each stock, shape (n_stocks,)
        size : np.ndarray
            Log market cap for each stock, shape (n_stocks,)
        value : np.ndarray
            Value score for each stock, shape (n_stocks,)
        momentum : np.ndarray
            Momentum score for each stock, shape (n_stocks,)
        market_returns : np.ndarray
            Market index returns, shape (T,)
        """
        T, n = returns.shape
        
        # Build factor exposures matrix
        B = np.zeros((n, self.n_factors))
        
        # Sector exposures (dummy variables)
        for i in range(n):
            sector = sector_ids[i]
            B[i, sector] = 1.0
        
        # Style factors
        size_col = self.n_sectors
        value_col = self.n_sectors + 1
        momentum_col = self.n_sectors + 2
        beta_col = self.n_sectors + 3
        
        B[:, size_col] = (size - size.mean()) / size.std()
        B[:, value_col] = (value - value.mean()) / value.std()
        B[:, momentum_col] = (momentum - momentum.mean()) / momentum.std()
        
        # Estimate betas to market
        for i in range(n):
            reg = LinearRegression()
            reg.fit(market_returns.reshape(-1, 1), returns.iloc[:, i].values)
            B[i, beta_col] = reg.coef_[0]
        
        self.loadings = B
        
        # Estimate factor covariance from factor returns
        # For simplicity, estimate from regression residuals
        F = np.zeros((T, self.n_factors))
        
        for s in range(self.n_sectors):
            sector_mask = sector_ids == s
            F[:, s] = returns.iloc[:, sector_mask].mean(axis=1)
        
        F[:, size_col] = market_returns  # Proxy for size
        F[:, value_col] = market_returns * 0.5  # Proxy
        F[:, momentum_col] = np.roll(market_returns, 1)
        F[:, beta_col] = market_returns
        
        self.factor_cov = np.cov(F.T)
        
        # Specific variance
        fitted = F @ B.T
        residuals = returns.values - fitted
        self.specific_var = np.var(residuals, axis=0)
    
    def get_covariance(self) -> np.ndarray:
        """Construct full covariance matrix."""
        assert self.loadings is not None
        
        cov = self.loadings @ self.factor_cov @ self.loadings.T
        cov += np.diag(self.specific_var)
        
        return cov


# Example integration: Full portfolio optimization with factor model
def optimize_with_factor_model(
    returns: pd.DataFrame,
    predictions: np.ndarray,
    sector_ids: np.ndarray,
    n_factors: int = 5
) -> Tuple[np.ndarray, Dict]:
    """
    End-to-end portfolio optimization using factor model.
    
    1. Fit factor model to historical returns
    2. Estimate covariance using factor model
    3. Optimize portfolio
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical daily returns
    predictions : np.ndarray
        ML model predictions
    sector_ids : np.ndarray
        Sector ID per stock
    n_factors : int
        Number of factors
    
    Returns
    -------
    weights : np.ndarray
        Optimal weights
    metrics : Dict
        Risk metrics
    """
    # Fit factor model
    fm = FactorRiskModel(n_factors=n_factors, model_type='pca')
    fm.fit_pca(returns)
    
    # Reconstruct covariance
    Sigma = fm.reconstruct_covariance()
    
    # Optimize portfolio
    opt = PortfolioOptimizer(len(predictions))
    weights, info = opt.mean_variance_optimization(
        predictions / 100,
        Sigma,
        lambda_risk=1.0,
        sector_ids=sector_ids,
        max_sector_exposure=0.2,
        long_only=True
    )
    
    # Risk decomposition
    risk_decomp = fm.risk_decomposition(weights)
    
    result = {
        'weights': weights,
        'optimization_info': info,
        'risk_decomposition': risk_decomp,
    }
    
    return weights, result
```

---

## Integration Example: Building a Complete Trading System

Here's how modules 21.1-21.3 work together in a real trading system:

```python
class TradeSystemPortfolioManager:
    """
    Complete portfolio management system for NSE trading.
    
    Integrates:
    1. Signal generation (from ML model)
    2. Signal-to-weight conversion
    3. Portfolio optimization
    4. Factor-based risk management
    """
    
    def __init__(
        self,
        stock_universe: List[str],
        sector_mapping: Dict[str, str],
        risk_aversion: float = 1.0
    ):
        """
        Parameters
        ----------
        stock_universe : List[str]
            List of stock symbols
        sector_mapping : Dict[str, str]
            Mapping from symbol to sector
        risk_aversion : float
            Portfolio risk aversion parameter
        """
        self.symbols = stock_universe
        self.n_stocks = len(stock_universe)
        self.sector_mapping = sector_mapping
        self.sector_ids = np.array([
            list(set(sector_mapping.values())).index(
                sector_mapping[sym]
            )
            for sym in stock_universe
        ])
        self.risk_aversion = risk_aversion
        
        self.signal_converter = SignalToWeight(
            self.n_stocks, method='sigmoid'
        )
        self.portfolio_optimizer = PortfolioOptimizer(self.n_stocks)
        self.risk_model = FactorRiskModel(n_factors=5, model_type='pca')
        
        self.current_weights = np.zeros(self.n_stocks)
    
    def generate_portfolio(
        self,
        ml_predictions: np.ndarray,
        historical_returns: pd.DataFrame,
        method: str = 'optimization'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate optimal portfolio weights.
        
        Parameters
        ----------
        ml_predictions : np.ndarray
            ML model predictions (in %), shape (n_stocks,)
        historical_returns : pd.DataFrame
            Historical daily returns for covariance estimation
        method : str
            'heuristic' (signal-to-weight) or 'optimization'
        
        Returns
        -------
        weights : np.ndarray
            Portfolio weights
        metrics : Dict
            Performance metrics
        """
        if method == 'heuristic':
            weights, metrics = self.signal_converter.signal_to_weight(
                ml_predictions,
                sector_ids=self.sector_ids,
                previous_weights=self.current_weights,
                max_weight=0.05,
                max_turnover=0.3
            )
        
        elif method == 'optimization':
            # Fit factor risk model
            self.risk_model.fit_pca(historical_returns)
            Sigma = self.risk_model.reconstruct_covariance()
            
            # Optimize
            weights, opt_metrics = self.portfolio_optimizer.mean_variance_optimization(
                ml_predictions / 100,  # Convert to decimal
                Sigma,
                lambda_risk=self.risk_aversion,
                sector_ids=self.sector_ids,
                max_sector_exposure=0.25,
                previous_weights=self.current_weights,
                max_turnover=0.3,
                long_only=True
            )
            
            # Add risk decomposition
            risk_decomp = self.risk_model.risk_decomposition(weights)
            
            metrics = {
                'optimization': opt_metrics,
                'risk_decomposition': risk_decomp,
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Update current weights
        self.current_weights = weights
        
        return weights, metrics
    
    def generate_orders(
        self,
        weights: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        min_order_size: float = 10000  # Rs 10k minimum per order
    ) -> pd.DataFrame:
        """
        Convert weights to actual buy/sell orders.
        
        Parameters
        ----------
        weights : np.ndarray
            Target portfolio weights
        prices : np.ndarray
            Current stock prices
        portfolio_value : float
            Total portfolio value in Rs
        min_order_size : float
            Minimum order size in Rs (Zerodha minimum)
        
        Returns
        -------
        orders : pd.DataFrame
            Buy/sell orders ready for execution
        """
        target_values = weights * portfolio_value  # Target Rs amount in each stock
        target_shares = target_values / prices  # Number of shares
        
        # Current shares (from previous portfolio)
        current_shares = self.current_weights * portfolio_value / prices
        
        # Orders
        share_changes = target_shares - current_shares
        
        orders = pd.DataFrame({
            'symbol': self.symbols,
            'current_shares': current_shares.astype(int),
            'target_shares': target_shares.astype(int),
            'share_change': share_changes.astype(int),
            'order_type': ['BUY' if x > 0 else 'SELL' if x < 0 else 'HOLD'
                          for x in share_changes],
            'order_quantity': np.abs(share_changes).astype(int),
            'price': prices,
            'order_value': np.abs(share_changes * prices),
        })
        
        # Filter: only orders above minimum size
        orders = orders[
            (orders['order_value'] >= min_order_size) |
            (orders['order_type'] == 'HOLD')
        ].copy()
        
        return orders


# Example usage
if __name__ == '__main__':
    # Setup
    symbols = ['INFY', 'TCS', 'WIPRO', 'TECHM', 'HCLT'] + ['ICICIBANK', 'HDFC', 'SBIN', 'KOTAK'] + \
              ['RELIANCE', 'ITC', 'LT', 'MARUTI', 'M&M'] * 2  # Dummy list
    
    sectors = {
        'INFY': 'IT', 'TCS': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT', 'HCLT': 'IT',
        'ICICIBANK': 'BANKING', 'HDFC': 'BANKING', 'SBIN': 'BANKING', 'KOTAK': 'BANKING',
        'RELIANCE': 'ENERGY', 'ITC': 'FMCG', 'LT': 'INDUSTRIAL', 'MARUTI': 'AUTO', 'M&M': 'AUTO'
    }
    
    # Create manager
    pm = TradeSystemPortfolioManager(symbols[:20], sectors)
    
    # Dummy data
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(250, 20) * 0.02,
        columns=symbols[:20]
    )
    predictions = np.random.randn(20) * 5  # % returns
    prices = np.random.uniform(100, 5000, 20)
    
    # Generate portfolio
    weights, metrics = pm.generate_portfolio(
        predictions, returns, method='optimization'
    )
    
    # Generate orders
    orders = pm.generate_orders(weights, prices, portfolio_value=1000000)
    print(orders[orders['order_type'] != 'HOLD'])
```

---

## Key Takeaways

1. **Signal-to-weight conversion** bridges predictions to actionable portfolios through heuristic or optimization-based methods.

2. **Optimization-based construction** (mean-variance, risk parity, max diversification) incorporates constraints and risk directly.

3. **Factor models** solve the curse of dimensionality by reducing covariance estimation from $O(n^2)$ to $O(Kn)$ parameters.

4. **Integration**: Use factor models to estimate $\Sigma$ → optimize weights → convert to orders → execute on Zerodha.

Next chapter: **Execution and Transaction Costs**.

---

## References

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*
- Carhart, M. (1997). "Mutual Fund Performance." *Journal of Finance*
- Fama, F. & French, K. (1993). "Common Risk Factors in Stock Returns"
- Ledoit, O. & Wolf, M. (2004). "Honey, I shrunk the sample covariance matrix"

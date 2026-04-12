# Chapter 8: Optimization for Portfolio Management

## Introduction

You've built a predictive machine learning model that forecasts stock returns. You have historical price data. You know which stocks you want to trade. But here's the critical question: **How do you allocate capital across these stocks to maximize returns while controlling risk?**

This is the portfolio optimization problem, and it sits at the heart of quantitative trading. Unlike ad-hoc allocation methods, modern portfolio optimization uses rigorous mathematical techniques rooted in convex optimization theory. This chapter teaches you:

1. **Convex optimization foundations** — the mathematical framework underlying all portfolio problems
2. **Mean-variance optimization** — the Markowitz model that won Harry Markowitz a Nobel Prize
3. **Covariance matrix estimation** — why your sample covariance matrix is garbage and how to fix it
4. **Advanced construction methods** — hierarchical clustering, Black-Litterman, risk parity, and transaction costs

By the end of this chapter, you'll implement production-ready portfolio optimization pipelines on NSE data using Zerodha APIs and Python's CVXPY library. You'll understand why naive approaches fail and why professionals use sophisticated estimation techniques.

---

# Module 8.1: Convex Optimization Foundations

## What is Convex Optimization?

Convex optimization is the study of minimizing (or maximizing) a convex function over a convex set. This matters because:

- **Convex problems have a unique global optimum** (no local minima to trap you)
- **They're computationally efficient** (solvable in polynomial time)
- **They're numerically stable** (small perturbations don't destroy solutions)

Portfolio optimization is naturally a convex problem, which is why we can solve it reliably at scale.

### Convex Sets

A set $\mathcal{C}$ is **convex** if for any two points $x, y \in \mathcal{C}$ and any $\lambda \in [0,1]$:

$$\lambda x + (1-\lambda) y \in \mathcal{C}$$

**Intuition**: A set is convex if you can draw a straight line between any two points and stay inside the set.

**Examples**:
- **Convex**: Balls, ellipsoids, polyhedra, simplexes
- **Non-convex**: Donuts, crescent moons, discrete sets

For portfolios, the constraint that weights sum to 1 and are non-negative ($w_i \geq 0$, $\sum w_i = 1$) defines a convex set called a **simplex**.

### Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if:

$$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y) \quad \forall \lambda \in [0,1], x, y$$

**Intuition**: The function lies below any chord connecting two points on its graph.

**Characterization via Hessian**: For twice-differentiable $f$, convexity is equivalent to the Hessian matrix $H$ being positive semidefinite:

$$H = \nabla^2 f \succeq 0$$

This means all eigenvalues are $\geq 0$.

**Key examples for portfolios**:
- Linear functions: $f(w) = c^T w$ (trivially convex)
- Quadratic functions: $f(w) = w^T Q w$ (convex if $Q \succeq 0$)
- Variance: $w^T \Sigma w$ (convex, since covariance matrix is positive semidefinite)

### Convex Optimization Problems

A convex optimization problem has the standard form:

$$\begin{align}
\text{minimize} \quad & f_0(w) \\
\text{subject to} \quad & f_i(w) \leq 0, \quad i = 1, \ldots, m \\
& h_j(w) = 0, \quad j = 1, \ldots, p
\end{align}$$

where:
- $f_0$ is a convex function (objective)
- $f_i$ are convex functions (inequality constraints)
- $h_j$ are affine functions (equality constraints)

**Why this matters**: Any local minimum is a global minimum. We can use efficient algorithms.

### Common Portfolio Problem Classes

#### Linear Programming (LP)

Minimize a linear objective over linear constraints:

$$\begin{align}
\text{minimize} \quad & c^T w \\
\text{subject to} \quad & Aw \leq b \\
& Ew = f \\
& w \geq 0
\end{align}$$

**Portfolio example**: Maximize expected return subject to budget constraints.

$$\text{minimize} \quad -\mu^T w \quad \text{(negative because we want to maximize)}$$

subject to $w \geq 0$, $\mathbf{1}^T w = 1$.

#### Quadratic Programming (QP)

Minimize a quadratic objective over linear constraints:

$$\begin{align}
\text{minimize} \quad & \frac{1}{2} w^T Q w + c^T w \\
\text{subject to} \quad & Aw \leq b \\
& Ew = f
\end{align}$$

**Portfolio example**: Minimize portfolio variance (a quadratic function) subject to return and constraint inequalities.

#### Second-Order Cone Programming (SOCP)

A generalization accommodating norm constraints:

$$\begin{align}
\text{minimize} \quad & c^T w \\
\text{subject to} \quad & \|A_i w + b_i\|_2 \leq c_i^T w + d_i, \quad i = 1, \ldots, m \\
& Ew = f
\end{align}$$

**Portfolio example**: Minimize worst-case risk under return uncertainty (robust optimization).

### The Lagrangian and KKT Conditions

For a general convex problem, the **Lagrangian** is:

$$L(w, \lambda, \nu) = f_0(w) + \sum_{i=1}^m \lambda_i f_i(w) + \sum_{j=1}^p \nu_j h_j(w)$$

where $\lambda_i \geq 0$ are dual variables for inequality constraints, and $\nu_j$ are dual variables for equality constraints.

The **Karush-Kuhn-Tucker (KKT) conditions** are necessary and sufficient for optimality:

1. **Stationarity**: $\nabla f_0(w) + \sum_{i=1}^m \lambda_i \nabla f_i(w) + \sum_{j=1}^p \nu_j \nabla h_j(w) = 0$
2. **Dual feasibility**: $\lambda_i \geq 0$ for all $i$
3. **Complementary slackness**: $\lambda_i f_i(w) = 0$ for all $i$ (if constraint is inactive, dual variable is zero)
4. **Primal feasibility**: All constraints satisfied

**Interpretation**: At optimality, the gradient of the objective is a weighted sum of constraint gradients. The weights tell you how "tight" each constraint is.

### Duality: Primal-Dual Formulation

The **dual problem** associated with the primal is:

$$\begin{align}
\text{maximize} \quad & g(\lambda, \nu) \\
\text{subject to} \quad & \lambda \geq 0
\end{align}$$

where the **dual function** is:

$$g(\lambda, \nu) = \inf_w L(w, \lambda, \nu)$$

**Key properties**:
- Dual optimal value $\leq$ Primal optimal value (**weak duality**)
- For convex problems, they're equal (**strong duality**)
- Dual variables give shadow prices: how much the objective improves if you relax a constraint

**Portfolio interpretation**: If you could add $1 more dollar to invest, the dual variable tells you the expected utility improvement.

### Example: Primal-Dual for Mean-Variance Portfolio

**Primal problem** (minimize variance for given expected return):

$$\begin{align}
\text{minimize} \quad & w^T \Sigma w \\
\text{subject to} \quad & \mu^T w = R \quad \text{(target return)} \\
& \mathbf{1}^T w = 1 \quad \text{(weights sum to 1)} \\
& w \geq 0 \quad \text{(no short selling)}
\end{align}$$

The Lagrangian is:

$$L(w, \lambda, \nu_1, \nu_2, \nu_3) = w^T \Sigma w + \lambda (\mu^T w - R) + \nu_1 (\mathbf{1}^T w - 1) + \nu_3^T (-w)$$

Setting $\frac{\partial L}{\partial w} = 0$:

$$2\Sigma w + \lambda \mu + \nu_1 \mathbf{1} - \nu_3 = 0$$

The dual function gives you insights into sensitivities and provides bounds if you can't solve the primal exactly.

## CVXPY Implementation

[CVXPY](https://www.cvxpy.org/) is Python's gold standard for convex optimization. It automatically transforms your problem into standard form and calls industrial-grade solvers.

```python
import numpy as np
import cvxpy as cp
from typing import Tuple, Dict
import pandas as pd
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    weights: np.ndarray
    objective_value: float
    status: str
    solver_time: float
    constraints_satisfied: Dict[str, bool]

def solve_lp_portfolio(
    expected_returns: np.ndarray,
    n_assets: int,
    constraints: Dict = None
) -> OptimizationResult:
    """
    Solve linear programming portfolio problem: maximize return subject to constraints.
    
    Args:
        expected_returns: (n_assets,) array of expected returns
        n_assets: Number of assets
        constraints: Dict with keys like 'max_position' (float in [0,1])
        
    Returns:
        OptimizationResult with optimal weights
    """
    # Decision variable: portfolio weights
    w = cp.Variable(n_assets)
    
    # Objective: maximize return = minimize negative return
    objective = cp.Minimize(-expected_returns @ w)
    
    # Constraints
    constraint_list = [
        w >= 0,              # No short selling
        cp.sum(w) == 1       # Fully invested
    ]
    
    # Add optional constraints
    if constraints:
        if 'max_position' in constraints:
            max_pos = constraints['max_position']
            constraint_list.append(w <= max_pos)
        if 'min_position' in constraints:
            min_pos = constraints['min_position']
            constraint_list.append(w >= min_pos)
    
    # Solve
    problem = cp.Problem(objective, constraint_list)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    return OptimizationResult(
        weights=w.value,
        objective_value=problem.value,
        status=problem.status,
        solver_time=problem.solver_stats.solve_time,
        constraints_satisfied={
            'sum_to_one': np.isclose(w.value.sum(), 1.0),
            'non_negative': np.all(w.value >= -1e-6)
        }
    )

def solve_qp_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float,
    constraints: Dict = None
) -> OptimizationResult:
    """
    Solve quadratic programming portfolio problem: minimize variance for target return.
    
    Args:
        expected_returns: (n_assets,) array of expected returns
        cov_matrix: (n_assets, n_assets) covariance matrix
        target_return: Desired portfolio expected return
        constraints: Dict with position limits
        
    Returns:
        OptimizationResult with optimal weights
    """
    n_assets = len(expected_returns)
    w = cp.Variable(n_assets)
    
    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    
    # Constraints
    constraint_list = [
        expected_returns @ w == target_return,  # Target return
        cp.sum(w) == 1,                         # Fully invested
        w >= 0                                  # No short selling
    ]
    
    # Add optional constraints
    if constraints:
        if 'max_position' in constraints:
            constraint_list.append(w <= constraints['max_position'])
    
    # Solve
    problem = cp.Problem(objective, constraint_list)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    return OptimizationResult(
        weights=w.value,
        objective_value=problem.value,
        status=problem.status,
        solver_time=problem.solver_stats.solve_time,
        constraints_satisfied={
            'target_return': np.isclose(expected_returns @ w.value, target_return),
            'sum_to_one': np.isclose(w.value.sum(), 1.0)
        }
    )

# Example usage
if __name__ == "__main__":
    # Toy example with 5 assets
    expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.11])
    cov_matrix = np.array([
        [0.04, 0.01, 0.005, 0.002, 0.008],
        [0.01, 0.05, 0.010, 0.003, 0.009],
        [0.005, 0.01, 0.03, 0.001, 0.005],
        [0.002, 0.003, 0.001, 0.06, 0.010],
        [0.008, 0.009, 0.005, 0.010, 0.04]
    ])
    
    # Solve LP: maximize return
    result_lp = solve_lp_portfolio(expected_returns, n_assets=5)
    print("LP Solution (max return):")
    print(f"  Weights: {result_lp.weights}")
    print(f"  Expected return: {expected_returns @ result_lp.weights:.4f}")
    print(f"  Status: {result_lp.status}\n")
    
    # Solve QP: minimize variance for 12% target return
    result_qp = solve_qp_portfolio(expected_returns, cov_matrix, target_return=0.12)
    print("QP Solution (min variance for 12% return):")
    print(f"  Weights: {result_qp.weights}")
    print(f"  Portfolio variance: {result_qp.objective_value:.6f}")
    print(f"  Status: {result_qp.status}")
```

## Key Takeaways: Module 8.1

- Convex problems have unique global optima and are computationally tractable
- Portfolio problems are naturally convex (variance is a convex function of weights)
- CVXPY abstracts away solver details; you focus on formulating the problem correctly
- KKT conditions provide insights into constraint tightness and sensitivity

---

# Module 8.2: Mean-Variance Optimization (Markowitz)

## The Efficient Frontier

**Mean-Variance Optimization** (MVO), pioneered by Harry Markowitz in 1952, asks: *What is the set of portfolios that minimize risk for each level of expected return?* This set is the **Efficient Frontier**.

### Mathematical Formulation

Given:
- $\mu \in \mathbb{R}^n$: Vector of expected asset returns
- $\Sigma \in \mathbb{R}^{n \times n}$: Covariance matrix (positive semidefinite)
- $w \in \mathbb{R}^n$: Portfolio weights

For a given target return $R$, the **Markowitz problem** is:

$$\begin{align}
\text{minimize} \quad & w^T \Sigma w \quad \text{(portfolio variance)} \\
\text{subject to} \quad & \mu^T w = R \quad \text{(target expected return)} \\
& \mathbf{1}^T w = 1 \quad \text{(weights sum to 1)} \\
& w \geq 0 \quad \text{(no short selling, optional)}
\end{align}$$

As you vary $R$ across a range, you trace out the **Efficient Frontier** in (return, risk) space.

### Efficient Frontier Without Risk-Free Asset

Without a risk-free asset, the efficient frontier is a **hyperbola** in (return, standard deviation) space.

**Key insight from the two-fund theorem**: Any portfolio on the efficient frontier can be written as a convex combination of two "fundamental" portfolios.

For the unrestricted problem (allowing short selling, $w \in \mathbb{R}^n$), the two fundamental portfolios are:

1. **Global Minimum Variance Portfolio** (GMVP): Minimizes risk regardless of return.

$$w_{\text{GMVP}} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}$$

Expected return: $\mu_{\text{GMVP}} = \mathbf{1}^T \Sigma^{-1} \mathbf{1})^{-1}$

2. **Variance-Free Portfolio**: The orthogonal portfolio in return space.

**Parametric form** of the efficient frontier:

$$w(R) = w_{\text{GMVP}} + \alpha (w_{\text{var-free}} - w_{\text{GMVP}})$$

for a parameter $\alpha$ that's uniquely determined by $R$.

### Efficient Frontier With Risk-Free Asset

With a risk-free asset at rate $r_f$, the picture simplifies dramatically.

**Key insight**: The efficient frontier becomes a **straight line** (the **Capital Allocation Line**, or CAL).

The optimal risky portfolio is the **Tangency Portfolio** (or **Market Portfolio** in equilibrium):

$$w^* = \frac{\Sigma^{-1} (\mu - r_f \mathbf{1})}{(\mu - r_f \mathbf{1})^T \Sigma^{-1} (\mu - r_f \mathbf{1})}$$

This portfolio maximizes the **Sharpe ratio**:

$$\text{Sharpe Ratio} = \frac{\mu^T w - r_f}{\sqrt{w^T \Sigma w}}$$

Once you've found the tangency portfolio, any efficient portfolio is a linear combination:
- Invest fraction $y$ in the tangency portfolio
- Invest fraction $1-y$ in the risk-free asset (or borrow if $y > 1$)

This gives expected return:

$$R = y \mu^T w^* + (1-y) r_f$$

and risk (standard deviation):

$$\sigma = y \sqrt{w^*^T \Sigma w^*}$$

The efficient frontier is the line:

$$R = r_f + \frac{\mu^T w^* - r_f}{\sqrt{w^*^T \Sigma w^*}} \sigma$$

The slope is the Sharpe ratio of the tangency portfolio.

### Practical Constraints: Long-Only, Position Limits, Sector Limits

In practice, you must impose realistic constraints:

1. **Long-only constraint** ($w_i \geq 0$): No short selling
2. **Position limits** ($w_i \leq w_{\max}$): No single position dominates (e.g., $w_{\max} = 0.10$ = 10%)
3. **Sector constraints**: Limit total weight in each sector

These constraints typically push the efficient frontier **inward** (higher risk for given return), but they're essential for:
- Regulatory compliance (many pension funds can't short)
- Practical market impact (shorting is expensive and risky)
- Concentration risk management

## Why MVO Fails in Practice: Estimation Error Amplification

Here's the dirty secret: Markowitz optimization **amplifies estimation error**.

### The Problem

Your inputs ($\mu$ and $\Sigma$) are estimated from historical data, which introduces error. Markowitz optimization tends to:
- **Overweight assets with slightly positive estimation error** in returns
- **Underweight assets with slightly negative estimation error**
- This creates portfolios with **extreme weights** (some very large, some zero)
- Out-of-sample performance is often **terrible**

**Example**: With 100 assets estimated from 5 years of data, each return estimate has ~20% sampling noise. The optimizer happily shorts the "worst" assets and goes long the "best," not realizing it's largely overfitting to noise.

### Michaud Resampled Efficient Frontier

Richard Michaud's **Resampled Frontier** is a practical fix. The algorithm:

1. **Estimate** mean $\hat{\mu}$ and covariance $\hat{\Sigma}$ from historical data
2. **Repeat** $K$ times (e.g., $K=100$):
   - Generate synthetic returns by resampling from a distribution centered at $\hat{\mu}$ and $\hat{\Sigma}$
   - Estimate mean/covariance from the synthetic data
   - Solve Markowitz problems at multiple return levels
   - Store the optimal weights
3. **Average** the weights across all resamples at each return level

The result is a "smoothed" frontier that's more robust to estimation error. Empirically, resampled portfolios have much better out-of-sample Sharpe ratios than standard MVO.

```python
def resampled_efficient_frontier(
    returns: np.ndarray,
    target_returns: np.ndarray,
    n_resamples: int = 100,
    confidence: float = 0.90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Michaud resampled efficient frontier.
    
    Args:
        returns: (T, n_assets) matrix of historical returns
        target_returns: (n_points,) array of target portfolio returns
        n_resamples: Number of resampling iterations
        confidence: Confidence level for resampling distribution
        
    Returns:
        mean_weights: (n_points, n_assets) resampled portfolio weights
        std_weights: (n_points, n_assets) std dev of weights across resamples
    """
    T, n_assets = returns.shape
    n_points = len(target_returns)
    
    # Estimate parameters
    mu_hat = returns.mean(axis=0)
    sigma_hat = np.cov(returns.T)
    
    # Storage for resamples
    all_weights = np.zeros((n_resamples, n_points, n_assets))
    
    for resample_idx in range(n_resamples):
        # Generate synthetic returns via correlated resampling
        synthetic_returns = np.random.multivariate_normal(
            mean=mu_hat,
            cov=sigma_hat,
            size=T
        )
        
        # Estimate from synthetic data
        mu_synth = synthetic_returns.mean(axis=0)
        sigma_synth = np.cov(synthetic_returns.T)
        
        # Solve Markowitz for each target return
        for point_idx, target_r in enumerate(target_returns):
            w = cp.Variable(n_assets)
            
            objective = cp.Minimize(cp.quad_form(w, sigma_synth))
            constraints = [
                mu_synth @ w == target_r,
                cp.sum(w) == 1,
                w >= 0
            ]
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                all_weights[resample_idx, point_idx, :] = w.value
    
    # Average across resamples
    mean_weights = all_weights.mean(axis=0)
    std_weights = all_weights.std(axis=0)
    
    return mean_weights, std_weights
```

## Implementation: Efficient Frontier with Realistic Constraints

```python
class EfficientFrontier:
    """
    Compute the efficient frontier with constraints.
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.03
    ):
        """
        Args:
            expected_returns: (n_assets,) expected returns
            cov_matrix: (n_assets, n_assets) covariance matrix
            risk_free_rate: Risk-free rate (e.g., 0.03 for 3%)
        """
        self.mu = expected_returns
        self.Sigma = cov_matrix
        self.rf = risk_free_rate
        self.n_assets = len(expected_returns)
    
    def compute_frontier(
        self,
        n_points: int = 50,
        max_position: float = 0.10,
        allow_short: bool = False
    ) -> pd.DataFrame:
        """
        Compute efficient frontier by solving QP at multiple return levels.
        
        Args:
            n_points: Number of return levels to sample
            max_position: Maximum position size (e.g., 0.10 for 10%)
            allow_short: Allow short selling
            
        Returns:
            DataFrame with columns: return, volatility, sharpe_ratio, weights...
        """
        
        # Range of target returns: from GMVP return to max return
        mu_min = self.compute_gmvp()[0]  # Return of GMVP
        mu_max = self.mu.max() * 1.05  # Slightly above highest return
        target_returns = np.linspace(mu_min, mu_max, n_points)
        
        results = []
        
        for target_r in target_returns:
            w = cp.Variable(self.n_assets)
            
            # Objective: minimize variance
            objective = cp.Minimize(cp.quad_form(w, self.Sigma))
            
            # Constraints
            constraints = [
                self.mu @ w == target_r,
                cp.sum(w) == 1
            ]
            
            if allow_short:
                constraints.append(w >= -1)  # Can short up to 100%
            else:
                constraints.append(w >= 0)   # Long-only
            
            constraints.append(w <= max_position)  # Position limits
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                portfolio_var = problem.value
                portfolio_std = np.sqrt(portfolio_var)
                sharpe = (target_r - self.rf) / (portfolio_std + 1e-10)
                
                row = {
                    'target_return': target_r,
                    'return': self.mu @ w.value,
                    'volatility': portfolio_std,
                    'variance': portfolio_var,
                    'sharpe_ratio': sharpe
                }
                
                # Add individual asset weights
                for i, w_i in enumerate(w.value):
                    row[f'w_{i}'] = w_i
                
                results.append(row)
        
        return pd.DataFrame(results)
    
    def compute_gmvp(self) -> Tuple[float, np.ndarray]:
        """
        Compute Global Minimum Variance Portfolio.
        
        Returns:
            (expected_return, weights)
        """
        w = cp.Variable(self.n_assets)
        
        objective = cp.Minimize(cp.quad_form(w, self.Sigma))
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        return self.mu @ w.value, w.value
    
    def compute_tangency_portfolio(self) -> Tuple[float, float, np.ndarray]:
        """
        Compute maximum Sharpe ratio (tangency) portfolio.
        
        Returns:
            (expected_return, sharpe_ratio, weights)
        """
        w = cp.Variable(self.n_assets)
        
        # Maximize Sharpe ratio = minimize negative Sharpe ratio
        # Using the trick: w = t * w', where t = 1 / (mu^T w' - rf)
        # This avoids division and makes the problem convex
        
        t = cp.Variable(positive=True)
        w_tilde = cp.Variable(self.n_assets)
        
        objective = cp.Minimize(cp.quad_form(w_tilde, self.Sigma))
        constraints = [
            (self.mu - self.rf * np.ones(self.n_assets)) @ w_tilde == 1,
            cp.sum(w_tilde) == t,
            w_tilde >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        w_opt = w_tilde.value / t.value
        ret = self.mu @ w_opt
        std = np.sqrt(np.quad_form(w_opt, self.Sigma).value)
        sharpe = (ret - self.rf) / std
        
        return ret, sharpe, w_opt

# Example: NSE Portfolio
if __name__ == "__main__":
    # Example: 5 large-cap NSE stocks
    stocks = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'BAJAJFINSV']
    expected_returns = np.array([0.12, 0.15, 0.10, 0.14, 0.16])  # Annual
    
    # Sample covariance matrix
    cov_matrix = np.array([
        [0.04, 0.015, 0.010, 0.012, 0.008],
        [0.015, 0.05, 0.012, 0.015, 0.010],
        [0.010, 0.012, 0.035, 0.010, 0.008],
        [0.012, 0.015, 0.010, 0.045, 0.012],
        [0.008, 0.010, 0.008, 0.012, 0.040]
    ])
    
    # Create efficient frontier
    ef = EfficientFrontier(expected_returns, cov_matrix, risk_free_rate=0.05)
    
    # Compute GMVP
    gmvp_ret, gmvp_w = ef.compute_gmvp()
    print(f"Global Minimum Variance Portfolio:")
    print(f"  Return: {gmvp_ret:.4f}, Weights: {gmvp_w}")
    
    # Compute tangency portfolio
    tan_ret, tan_sharpe, tan_w = ef.compute_tangency_portfolio()
    print(f"\nTangency Portfolio (Max Sharpe):")
    print(f"  Return: {tan_ret:.4f}, Sharpe: {tan_sharpe:.4f}")
    print(f"  Weights: {tan_w}")
    
    # Compute full frontier
    frontier_df = ef.compute_frontier(n_points=30, max_position=0.20)
    print(f"\nEfficient Frontier (30 points):")
    print(frontier_df[['return', 'volatility', 'sharpe_ratio']].head(10))
```

[VISUALIZATION: Efficient Frontier]

The visualization should show:
- **X-axis**: Portfolio volatility (standard deviation)
- **Y-axis**: Portfolio expected return
- **Curve**: The efficient frontier (hyperbola without risk-free asset; line with risk-free asset)
- **Points**: GMVP, tangency portfolio, individual assets
- **Constraints**: How long-only and position limits shift the frontier inward

## Key Takeaways: Module 8.2

- Efficient frontier minimizes variance for each return level (or vice versa)
- With a risk-free asset, the frontier is a straight line, and you invest in the tangency portfolio + risk-free asset
- MVO amplifies estimation error; use Michaud resampling for robustness
- Realistic constraints (long-only, position limits) are essential in practice

---

# Module 8.3: Covariance Matrix Estimation

## Why Sample Covariance is Terrible

Your first instinct might be to estimate covariance from historical returns:

$$\hat{\Sigma}_{\text{sample}} = \frac{1}{T} \sum_{t=1}^T (r_t - \bar{r})(r_t - \bar{r})^T$$

**Don't.** Here's why:

### The Curse of Dimensionality

With $n$ assets and $T$ observations, you're estimating $\frac{n(n+1)}{2}$ covariance parameters. Each estimate has sampling error.

**Example**: 100 stocks, 5 years of daily data ($T \approx 1250$ days)
- Parameters to estimate: $\frac{100 \cdot 101}{2} = 5050$
- Observations: $1250$
- **Ratio**: 4 observations per parameter!

The sample covariance is **biased and has huge variance**. When you feed it to Markowitz optimization, the optimizer overfits to noise.

### Eigenvalue Spreading

The sample covariance matrix has eigenvalues scattered across a large range, with many small eigenvalues corresponding to noise. This causes:
- **Unstable portfolio weights** that fluctuate wildly with tiny data changes
- **Extreme concentration**: Some assets get huge weights, others zero

### Condition Number Blowup

The condition number $\kappa(\Sigma) = \frac{\lambda_{\max}}{\lambda_{\min}}$ of the sample covariance is huge, making numerical optimization unstable.

## Ledoit-Wolf Shrinkage

The **Ledoit-Wolf estimator** is a linear combination of sample and target covariance:

$$\hat{\Sigma}_{\text{LW}} = (1 - \alpha) \hat{\Sigma}_{\text{sample}} + \alpha \Sigma_{\text{target}}$$

where $\alpha \in [0,1]$ is the shrinkage intensity.

### Choosing the Shrinkage Target

Common targets:

1. **Single-factor model** (diagonal with common variance):

$$\Sigma_{\text{target}} = \text{diag}(\hat{\sigma}_1^2, \ldots, \hat{\sigma}_n^2)$$

The diagonal elements are sample variances; off-diagonal elements are zero (assumes independence).

2. **Constant correlation model**:

$$[\Sigma_{\text{target}}]_{ij} = \begin{cases}
\hat{\sigma}_i^2 & \text{if } i = j \\
\bar{\rho} \hat{\sigma}_i \hat{\sigma}_j & \text{if } i \neq j
\end{cases}$$

where $\bar{\rho}$ is the average sample correlation.

3. **Single-factor model with systematic risk**:

Uses PCA or factor models to identify the dominant source of variation.

### Optimal Shrinkage Intensity

Ledoit and Wolf (2004) derived the optimal $\alpha$ in closed form:

$$\alpha^* = \frac{\text{var}(\text{sample} - \text{target})}{\text{var}(\text{sample} - \text{truth})}$$

This minimizes **expected squared Frobenius norm** between the estimator and true covariance.

The formula is:

$$\alpha^* = \frac{(1-2/n) \text{tr}(\hat{\Sigma}_{\text{sample}}^2) - \text{tr}((\hat{\Sigma}_{\text{sample}})^2)}{(n+1-2/n) \text{tr}(\hat{\Sigma}_{\text{sample}}^2) - \text{tr}((\hat{\Sigma}_{\text{sample}})^2)}$$

where $n$ is the number of assets.

```python
def ledoit_wolf_shrinkage(
    returns: np.ndarray,
    target: str = 'diagonal'
) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf shrinkage estimator.
    
    Args:
        returns: (T, n_assets) matrix of historical returns
        target: 'diagonal' or 'constant_correlation'
        
    Returns:
        Shrunk covariance matrix, shrinkage intensity alpha
    """
    T, n = returns.shape
    
    # Sample covariance
    Sigma_sample = np.cov(returns.T)
    
    # Choose target
    if target == 'diagonal':
        Sigma_target = np.diag(np.diag(Sigma_sample))
    
    elif target == 'constant_correlation':
        # Correlation matrix with average off-diagonal correlation
        diag_std = np.sqrt(np.diag(Sigma_sample))
        corr_sample = np.corrcoef(returns.T)
        avg_corr = (corr_sample.sum() - n) / (n * (n - 1))  # Exclude diagonal
        
        Sigma_target = np.diag(diag_std) @ (avg_corr * (np.ones((n, n)) - np.eye(n)) + np.eye(n)) @ np.diag(diag_std)
    
    # Optimal shrinkage intensity
    term1 = (1 - 2/n) * np.trace(Sigma_sample @ Sigma_sample)
    term2 = np.trace(Sigma_sample @ Sigma_sample)
    numerator = term1
    denominator = (n + 1 - 2/n) * np.trace(Sigma_sample @ Sigma_sample) - np.trace(Sigma_sample @ Sigma_sample)
    
    alpha = numerator / denominator
    alpha = np.clip(alpha, 0, 1)  # Ensure in [0, 1]
    
    # Shrunk covariance
    Sigma_shrunk = (1 - alpha) * Sigma_sample + alpha * Sigma_target
    
    return Sigma_shrunk, alpha
```

## PCA-Based Covariance Estimation

**Principal Component Analysis** decomposes covariance into principal directions:

$$\Sigma = V \Lambda V^T$$

where:
- $V$ is the matrix of eigenvectors (principal directions)
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ are eigenvalues (variances along principal directions)

### Denoising via Eigenvalue Threshold

Noise eigenvalues follow the **Marchenko-Pastur distribution** (for random data). López de Prado's denoising approach:

1. Compute eigenvalues of the sample covariance
2. Estimate the noise floor using Marchenko-Pastur theory
3. Set eigenvalues below the threshold to a fixed value $\lambda_{\text{noise}}$
4. Reconstruct the covariance matrix

```python
def marchenko_pastur_threshold(
    returns: np.ndarray,
    q: float = None
) -> float:
    """
    Estimate noise floor using Marchenko-Pastur distribution.
    
    Args:
        returns: (T, n_assets) matrix
        q: Ratio T/n (if None, computed from data)
        
    Returns:
        Upper bound of eigenvalue support (threshold for noise)
    """
    T, n = returns.shape
    if q is None:
        q = T / n
    
    # Marchenko-Pastur theoretical maximum eigenvalue
    # lambda_plus = (1 + sqrt(1/q))^2
    lambda_plus = (1 + np.sqrt(1/q)) ** 2
    
    return lambda_plus

def denoise_covariance(
    returns: np.ndarray,
    n_components: int = None
) -> np.ndarray:
    """
    Denoise covariance matrix using PCA.
    
    Args:
        returns: (T, n_assets) matrix
        n_components: Number of principal components to keep
                      (if None, auto-detect using Marchenko-Pastur)
        
    Returns:
        Denoised covariance matrix
    """
    T, n = returns.shape
    
    # Compute sample covariance and eigendecomposition
    Sigma_sample = np.cov(returns.T)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_sample)
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Determine number of components to keep
    if n_components is None:
        lambda_plus = marchenko_pastur_threshold(returns)
        n_components = np.sum(eigenvalues > lambda_plus)
    
    # Reconstruct with denoised eigenvalues
    eigenvalues_denoised = np.copy(eigenvalues)
    eigenvalues_denoised[:n - n_components] = eigenvalues_denoised[n - n_components]  # Set noise to fixed value
    
    Sigma_denoised = eigenvectors @ np.diag(eigenvalues_denoised) @ eigenvectors.T
    
    return Sigma_denoised
```

## Comparing Estimators: Simulation Study

```python
def compare_estimators(
    true_cov: np.ndarray,
    returns: np.ndarray,
    n_trials: int = 100
) -> pd.DataFrame:
    """
    Compare covariance estimators on a simulated dataset.
    
    Args:
        true_cov: True covariance matrix (for benchmarking)
        returns: (T, n_assets) historical returns
        n_trials: Number of bootstrap resamples
        
    Returns:
        DataFrame with Frobenius norm errors for each estimator
    """
    
    errors = {
        'sample': [],
        'ledoit_wolf_diag': [],
        'ledoit_wolf_corr': [],
        'denoised': [],
    }
    
    for trial in range(n_trials):
        # Bootstrap resample
        idx = np.random.choice(len(returns), size=len(returns), replace=True)
        returns_boot = returns[idx]
        
        Sigma_sample = np.cov(returns_boot.T)
        
        # Ledoit-Wolf
        Sigma_lw_diag, _ = ledoit_wolf_shrinkage(returns_boot, target='diagonal')
        Sigma_lw_corr, _ = ledoit_wolf_shrinkage(returns_boot, target='constant_correlation')
        
        # Denoised
        Sigma_denoised = denoise_covariance(returns_boot)
        
        # Compute Frobenius norms
        errors['sample'].append(np.linalg.norm(Sigma_sample - true_cov, 'fro'))
        errors['ledoit_wolf_diag'].append(np.linalg.norm(Sigma_lw_diag - true_cov, 'fro'))
        errors['ledoit_wolf_corr'].append(np.linalg.norm(Sigma_lw_corr - true_cov, 'fro'))
        errors['denoised'].append(np.linalg.norm(Sigma_denoised - true_cov, 'fro'))
    
    return pd.DataFrame(errors)
```

[VISUALIZATION: Covariance Estimation Methods]

Show:
- Eigenvalue spectrum of sample vs. true covariance
- Marchenko-Pastur distribution overlay
- Frobenius norm error for each estimator across trials
- How shrinkage intensity $\alpha$ varies with $T/n$ ratio

## Key Takeaways: Module 8.3

- Sample covariance is biased and has huge sampling error
- Ledoit-Wolf shrinkage (toward diagonal or constant correlation) reduces estimation error dramatically
- PCA denoising removes noise eigenvectors identified via Marchenko-Pastur theory
- Estimator choice matters: shrunk covariance leads to better out-of-sample portfolio performance

---

# Module 8.4: Advanced Portfolio Construction

## Risk Parity: Equal Risk Contribution

**Risk Parity** is a portfolio construction method that weights assets inversely to their volatility, so each contributes equally to total portfolio risk.

### Mathematical Formulation

Let $\sigma_i^2$ be the marginal variance of asset $i$. The **contribution to risk** from asset $i$ is:

$$\text{RC}_i = w_i [\Sigma w]_i = w_i (\Sigma w)_i$$

where $(\Sigma w)_i$ is the $i$-th component of the vector $\Sigma w$.

In a **Risk Parity portfolio**, all contributions are equal:

$$\text{RC}_i = \text{RC}_j \quad \forall i, j$$

or equivalently:

$$w_i [\Sigma w]_i = \frac{w^T \Sigma w}{n} \quad \forall i$$

### Solving for Risk Parity Weights

One approach uses convex optimization:

$$\begin{align}
\text{minimize} \quad & \sum_{i=1}^n (w_i [\Sigma w]_i - \bar{\text{RC}})^2 \\
\text{subject to} \quad & \sum_i w_i = 1 \\
& w_i \geq 0
\end{align}$$

A simpler heuristic that works well:

$$w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$$

Normalize the inverse volatilities to sum to 1.

### Advantages

- **Diversification**: No single asset dominates the risk budget
- **Robustness**: Less sensitive to covariance estimation errors (relies mainly on marginal variances)
- **Intuitive**: Simple to implement and explain to stakeholders

### Disadvantages

- **Ignores expected returns**: Pure risk-based, doesn't use return forecasts
- **Concentration in low-volatility assets**: If one asset is very stable, it gets overweighted
- **Mean reversion risk**: If volatilities change, the portfolio becomes unbalanced

```python
def risk_parity_portfolio(
    cov_matrix: np.ndarray,
    max_position: float = 0.30
) -> np.ndarray:
    """
    Construct risk parity portfolio.
    
    Args:
        cov_matrix: (n_assets, n_assets) covariance matrix
        max_position: Maximum position size
        
    Returns:
        Risk parity weights
    """
    n = cov_matrix.shape[0]
    
    # Marginal standard deviations
    marginal_std = np.sqrt(np.diag(cov_matrix))
    
    # Inverse volatility weights (heuristic)
    inv_vol = 1.0 / marginal_std
    w_heuristic = inv_vol / inv_vol.sum()
    
    # Refine with convex optimization for exact risk parity
    w = cp.Variable(n)
    
    sigma_w = cov_matrix @ w
    portfolio_var = cp.sum_squares(sigma_w)
    
    # Risk contributions
    rc = cp.multiply(w, sigma_w)
    target_rc = portfolio_var / n
    
    # Objective: minimize deviation from target risk contribution
    objective = cp.Minimize(cp.sum_squares(rc - target_rc))
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_position
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    return w.value
```

## Hierarchical Risk Parity (HRP)

**Hierarchical Risk Parity**, developed by Marcos López de Prado, uses **clustering** to build a portfolio that's robust to covariance estimation errors.

### Algorithm Overview

1. **Compute the distance matrix**: Use $d_{ij} = \sqrt{1 - \rho_{ij}}$ where $\rho_{ij}$ is correlation between assets $i$ and $j$

2. **Hierarchical clustering**: Apply linkage (e.g., Ward's method) to group similar assets

3. **Traverse the tree**: Starting from the root, recursively partition assets into two groups

4. **Allocate within groups**: Use inverse volatility weighting within each leaf

5. **Refine**: Adjust weights to enforce risk parity across branches

### Why HRP Works

- **Cluster-first**: Groups similar assets, reducing the impact of misestimated correlations between distant assets
- **Recursive**: Builds hierarchy from bottom up, avoiding the need to invert a large matrix
- **Robust**: Less sensitive to covariance misspecification

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

def hrp_portfolio(
    returns: np.ndarray
) -> np.ndarray:
    """
    Hierarchical Risk Parity portfolio construction.
    
    Args:
        returns: (T, n_assets) historical returns
        
    Returns:
        HRP weights
    """
    n = returns.shape[1]
    
    # Compute correlation matrix
    corr = np.corrcoef(returns.T)
    
    # Distance matrix: d_ij = sqrt(1 - rho_ij)
    # Ensure correlation is in [-1, 1] and cap at 1 to avoid numerical issues
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt((1 - corr) / 2)
    
    # Convert to condensed distance matrix for linkage
    dist_condensed = squareform(dist)
    
    # Hierarchical clustering
    linkage_matrix = linkage(dist_condensed, method='ward')
    
    # Initialize weights
    w = np.ones(n) / n
    
    # Recursively allocate weights
    def _get_clusters(node_idx, k=2):
        """Get cluster assignment for a node"""
        if node_idx < n:  # Leaf node
            return [node_idx]
        else:  # Internal node
            left = int(linkage_matrix[node_idx - n, 0])
            right = int(linkage_matrix[node_idx - n, 1])
            return _get_clusters(left) + _get_clusters(right)
    
    def _allocate_weights(cluster_idx, current_weight):
        """Recursively allocate weights down the tree"""
        if len(cluster_idx) == 1:
            w[cluster_idx[0]] = current_weight
        else:
            # Split into two clusters
            mid = len(cluster_idx) // 2
            left_cluster = cluster_idx[:mid]
            right_cluster = cluster_idx[mid:]
            
            # Compute volatility of each sub-cluster
            left_vol = np.std(returns[:, left_cluster].mean(axis=1))
            right_vol = np.std(returns[:, right_cluster].mean(axis=1))
            
            # Allocate inversely to volatility
            total_vol = left_vol + right_vol
            left_weight = current_weight * right_vol / total_vol
            right_weight = current_weight * left_vol / total_vol
            
            _allocate_weights(left_cluster, left_weight)
            _allocate_weights(right_cluster, right_weight)
    
    # Start with all assets
    all_assets = list(range(n))
    _allocate_weights(all_assets, 1.0)
    
    return w
```

## Black-Litterman Model

The **Black-Litterman model** combines:
1. **Market equilibrium views** (from implied returns of market portfolio)
2. **Investor views** (active bets on return deviations)

to produce a robust return estimate that respects market prices while incorporating your insights.

### Market Equilibrium Step

Assume the market portfolio $w_{\text{mkt}}$ (e.g., cap-weighted index) is optimal. Reverse-engineer the expected returns it implies:

$$\mu_{\text{implied}} = \delta \Sigma w_{\text{mkt}}$$

where $\delta$ is the risk-aversion coefficient.

$$\delta = \frac{\mu_{\text{mkt}} - r_f}{\sigma_{\text{mkt}}^2}$$

This gives the **equilibrium return vector** $\mu_{\text{eq}} = \mu_{\text{implied}}$.

### Incorporating Views

You have a view that asset $i$'s excess return over asset $j$ is $V_{ij}$. Express this as:

$$P \mu = v + \epsilon$$

where:
- $P$ is a matrix of view specifications (e.g., $P = [1, -1, 0, \ldots]$ for "asset 1 outperforms asset 2")
- $v$ is the vector of view returns
- $\epsilon \sim N(0, \Omega)$ is view uncertainty

### Posterior Expected Returns

Bayesian update:

$$\mu_{\text{BL}} = \mu_{\text{eq}} + \Sigma P^T (\Omega + P \Sigma P^T)^{-1} (v - P \mu_{\text{eq}})$$

The updated covariance matrix is:

$$\Sigma_{\text{BL}} = \Sigma - \Sigma P^T (\Omega + P \Sigma P^T)^{-1} P \Sigma$$

### Advantages

- **Market-aware**: Respects market-implied returns, avoids extreme deviations
- **Incorporates views**: Allows you to incorporate active insights
- **Stable weights**: Less extreme than pure Markowitz
- **Intuitive**: Easy to communicate views to stakeholders

```python
class BlackLittermanModel:
    """Black-Litterman portfolio optimization"""
    
    def __init__(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        risk_free_rate: float = 0.03,
        risk_aversion: float = None
    ):
        """
        Args:
            cov_matrix: (n_assets, n_assets) covariance matrix
            market_weights: (n_assets,) market portfolio weights (e.g., cap-weighted)
            risk_free_rate: Risk-free rate
            risk_aversion: Risk aversion coefficient (if None, computed from market portfolio)
        """
        self.Sigma = cov_matrix
        self.w_mkt = market_weights
        self.rf = risk_free_rate
        self.n = len(market_weights)
        
        # Compute equilibrium return
        self.mu_mkt = (self.w_mkt @ cov_matrix @ self.w_mkt) ** 0.5  # Market portfolio return
        
        # Risk aversion coefficient
        if risk_aversion is None:
            self.delta = (self.mu_mkt - self.rf) / (self.w_mkt @ cov_matrix @ self.w_mkt)
        else:
            self.delta = risk_aversion
        
        # Implied equilibrium returns
        self.mu_eq = self.delta * cov_matrix @ self.w_mkt
    
    def add_relative_view(
        self,
        assets_long: list,
        assets_short: list,
        expected_return_diff: float,
        confidence: float = 0.5
    ):
        """
        Add a relative view: assets_long outperform assets_short by expected_return_diff.
        
        Args:
            assets_long: Indices of assets to go long
            assets_short: Indices of assets to go short
            expected_return_diff: Expected return differential
            confidence: Confidence in the view (higher = lower omega)
        """
        # Build P matrix row
        p_row = np.zeros(self.n)
        p_row[assets_long] = 1.0 / len(assets_long)
        p_row[assets_short] = -1.0 / len(assets_short)
        
        if not hasattr(self, 'P'):
            self.P = p_row.reshape(1, -1)
            self.v = np.array([expected_return_diff])
            self.confidence = np.array([confidence])
        else:
            self.P = np.vstack([self.P, p_row])
            self.v = np.append(self.v, expected_return_diff)
            self.confidence = np.append(self.confidence, confidence)
    
    def get_posterior_returns(self) -> np.ndarray:
        """Compute posterior expected returns after incorporating views"""
        
        if not hasattr(self, 'P'):
            return self.mu_eq
        
        # Uncertainty in views: Omega = diag(P * Sigma * P^T * tau / confidence)
        # tau is a scaling parameter (often set to 1)
        tau = 1.0
        omega_diag = np.diag(self.P @ self.Sigma @ self.P.T) * tau / self.confidence
        Omega = np.diag(omega_diag)
        
        # Posterior returns
        term = self.Sigma @ self.P.T @ np.linalg.inv(Omega + self.P @ self.Sigma @ self.P.T)
        mu_bl = self.mu_eq + term @ (self.v - self.P @ self.mu_eq)
        
        return mu_bl
    
    def get_posterior_covariance(self) -> np.ndarray:
        """Compute posterior covariance after incorporating views"""
        
        if not hasattr(self, 'P'):
            return self.Sigma
        
        tau = 1.0
        omega_diag = np.diag(self.P @ self.Sigma @ self.P.T) * tau / self.confidence
        Omega = np.diag(omega_diag)
        
        term = self.Sigma @ self.P.T @ np.linalg.inv(Omega + self.P @ self.Sigma @ self.P.T) @ self.P @ self.Sigma
        Sigma_bl = self.Sigma - term
        
        return Sigma_bl
```

## Transaction Costs and Multi-Period Optimization

In reality, rebalancing has costs. A more realistic objective:

$$\text{maximize} \quad \mu^T w - \lambda \sigma(w) - \gamma \|w - w_{\text{prev}}\|_1$$

where:
- First term: Expected return
- Second term: Risk penalty
- Third term: Transaction cost (linear in position changes, $\gamma$ = transaction cost coefficient)

The $L_1$ norm $\|w - w_{\text{prev}}\|_1$ captures the total turnover.

### Multi-Period Stochastic Optimization

For longer horizons, use **stochastic programming**:

$$\begin{align}
\text{maximize} \quad & \mathbb{E}[U(W_T)] \\
\text{subject to} \quad & \text{Budget constraints at each time} \\
& \text{Position limits} \\
& \text{Transaction costs}
\end{align}$$

where $W_T$ is terminal wealth.

This requires discretizing the return distribution and solving a large tree of sub-problems.

```python
def optimize_with_transaction_costs(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    prev_weights: np.ndarray,
    target_return: float,
    transaction_cost_coeff: float = 0.001
) -> np.ndarray:
    """
    Optimize portfolio with linear transaction costs.
    
    Args:
        expected_returns: Expected returns
        cov_matrix: Covariance matrix
        prev_weights: Previous portfolio weights
        target_return: Target expected return
        transaction_cost_coeff: Cost per unit of turnover (e.g., 0.001 for 0.1% per unit)
        
    Returns:
        Optimal weights with transaction costs
    """
    n = len(expected_returns)
    w = cp.Variable(n)
    
    # Objective: maximize return, minimize risk, minimize transaction costs
    objective = cp.Maximize(
        expected_returns @ w 
        - 0.5 * cp.quad_form(w, cov_matrix) 
        - transaction_cost_coeff * cp.norm(w - prev_weights, 1)
    )
    
    constraints = [
        expected_returns @ w == target_return,
        cp.sum(w) == 1,
        w >= 0
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    return w.value
```

## Robust Optimization

What if your return estimates are uncertain? **Robust optimization** minimizes worst-case risk under return uncertainty:

$$\begin{align}
\text{minimize} \quad & \max_{\mu \in U} \quad w^T \Sigma w \\
\text{subject to} \quad & \mu^T w \geq R \\
& \mathbf{1}^T w = 1 \\
& w \geq 0
\end{align}$$

where $U$ is an **uncertainty set** for returns (e.g., ellipsoid or box constraints).

This is an SOCP problem that can be solved with CVXPY.

---

# Comprehensive Example: Building a Quantitative Portfolio on NSE with Zerodha

Here's a full pipeline combining all concepts:

```python
import yfinance as yf  # Can replace with Zerodha API calls
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class NSEPortfolioOptimizer:
    """Complete portfolio optimization pipeline for NSE stocks"""
    
    def __init__(
        self,
        tickers: list,
        risk_free_rate: float = 0.05,
        lookback_days: int = 252
    ):
        """
        Args:
            tickers: List of NSE tickers (e.g., ['RELIANCE.NS', 'TCS.NS'])
            risk_free_rate: Annual risk-free rate
            lookback_days: Historical data for covariance estimation
        """
        self.tickers = tickers
        self.rf = risk_free_rate
        self.lookback_days = lookback_days
        self.returns = None
        self.Sigma = None
        self.mu = None
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data and compute returns"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        data = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']
        
        # Compute daily returns
        self.returns = data.pct_change().dropna()
        
        return self.returns
    
    def estimate_covariance(self, method: str = 'ledoit_wolf') -> np.ndarray:
        """Estimate covariance matrix with shrinkage"""
        
        if method == 'ledoit_wolf':
            self.Sigma, alpha = ledoit_wolf_shrinkage(self.returns.values)
            print(f"Ledoit-Wolf shrinkage: alpha={alpha:.4f}")
        
        elif method == 'sample':
            self.Sigma = np.cov(self.returns.values.T)
        
        elif method == 'denoised':
            self.Sigma = denoise_covariance(self.returns.values)
        
        return self.Sigma
    
    def estimate_returns(self) -> np.ndarray:
        """Estimate expected returns (annualized)"""
        self.mu = self.returns.mean() * 252  # Annualize
        return self.mu
    
    def optimize_mvo(self, target_return: float = None) -> dict:
        """Optimize using Markowitz mean-variance"""
        
        if target_return is None:
            target_return = self.mu.mean()
        
        ef = EfficientFrontier(self.mu.values, self.Sigma, self.rf)
        weights = self._solve_markowitz(target_return)
        
        port_return = self.mu @ weights
        port_vol = np.sqrt(weights @ self.Sigma @ weights)
        port_sharpe = (port_return - self.rf) / port_vol
        
        return {
            'method': 'MVO',
            'weights': weights,
            'return': port_return,
            'volatility': port_vol,
            'sharpe': port_sharpe
        }
    
    def optimize_rp(self) -> dict:
        """Optimize using Risk Parity"""
        weights = risk_parity_portfolio(self.Sigma)
        
        port_return = self.mu @ weights
        port_vol = np.sqrt(weights @ self.Sigma @ weights)
        port_sharpe = (port_return - self.rf) / port_vol
        
        return {
            'method': 'Risk Parity',
            'weights': weights,
            'return': port_return,
            'volatility': port_vol,
            'sharpe': port_sharpe
        }
    
    def optimize_hrp(self) -> dict:
        """Optimize using Hierarchical Risk Parity"""
        weights = hrp_portfolio(self.returns.values)
        
        port_return = self.mu @ weights
        port_vol = np.sqrt(weights @ self.Sigma @ weights)
        port_sharpe = (port_return - self.rf) / port_vol
        
        return {
            'method': 'HRP',
            'weights': weights,
            'return': port_return,
            'volatility': port_vol,
            'sharpe': port_sharpe
        }
    
    def compare_portfolios(self) -> pd.DataFrame:
        """Compare all optimization methods"""
        
        results = []
        results.append(self.optimize_mvo())
        results.append(self.optimize_rp())
        results.append(self.optimize_hrp())
        
        df = pd.DataFrame(results)
        return df[['method', 'return', 'volatility', 'sharpe']]
    
    def _solve_markowitz(self, target_return: float) -> np.ndarray:
        """Helper: solve Markowitz QP"""
        n = len(self.mu)
        w = cp.Variable(n)
        
        objective = cp.Minimize(cp.quad_form(w, self.Sigma))
        constraints = [
            self.mu @ w == target_return,
            cp.sum(w) == 1,
            w >= 0,
            w <= 0.20  # Position limit: max 20%
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        return w.value

# Usage
if __name__ == "__main__":
    # NSE tickers
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFC.NS', 'INFY.NS', 'BAJAJFINSV.NS']
    
    # Create optimizer
    optimizer = NSEPortfolioOptimizer(tickers, risk_free_rate=0.05)
    
    # Fetch data
    optimizer.fetch_data()
    
    # Estimate inputs
    optimizer.estimate_covariance(method='ledoit_wolf')
    optimizer.estimate_returns()
    
    # Compare optimizations
    comparison = optimizer.compare_portfolios()
    print(comparison)
```

---

# Exercises

## Exercise 8.1: Convex Optimization Fundamentals

**Problem**: Given the quadratic function $f(w) = w^T A w + b^T w + c$ where $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, $b = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$, $c = 0$:

a) Show that $f$ is convex.
b) Find the minimum using CVXPY.
c) Verify the KKT conditions.

**Solution**:
- Compute the Hessian: $H = 2A$. Eigenvalues are 1 and 3 (both positive), so $A \succ 0$ and $f$ is convex.
- CVXPY code: Define $w = cp.Variable(2)$, minimize $cp.quad_form(w, A)$.
- At the optimum $w^* = 0$, verify $\nabla f(w^*) = 0$.

---

## Exercise 8.2: Efficient Frontier with Constraints

**Problem**: Using the data from the NSE example:
- Compute the efficient frontier WITH and WITHOUT long-only constraints
- Plot both frontiers
- Explain why the constrained frontier is "inside" the unconstrained one

**Expected Output**:
- Figure showing two hyperbolas (or lines if you include risk-free asset)
- Table of (return, volatility, sharpe_ratio) for key points

---

## Exercise 8.3: Covariance Estimation Robustness

**Problem**: 
a) Generate synthetic returns from a known covariance matrix.
b) Estimate covariance using three methods: sample, Ledoit-Wolf, denoised.
c) Compute the Frobenius norm error for each method.
d) Repeat with different sample sizes and plot convergence.

**Expected Output**:
- Plot showing error vs. sample size for each estimator
- Ledoit-Wolf and denoised should outperform sample for small $T/n$

---

## Exercise 8.4: Black-Litterman with Active Views

**Problem**:
a) Start with cap-weighted market portfolio (equal weights for simplicity).
b) Add a view: "TCS will outperform INFY by 2% over the next year with 70% confidence."
c) Compute the posterior Black-Litterman return estimates.
d) Optimize a portfolio using posterior returns.
e) Compare with pure Markowitz.

**Expected Output**:
- Table showing prior vs. posterior expected returns
- Weights comparison: Markowitz vs. Black-Litterman

---

## Exercise 8.5: Transaction Cost Aware Rebalancing

**Problem**:
a) Start with an initial portfolio.
b) Optimize a new portfolio WITHOUT transaction cost penalty.
c) Optimize WITH a 0.1% per-unit transaction cost.
d) Compare the turnover and out-of-sample performance metrics.

**Expected Output**:
- Summary table: old weights, new weights (no cost), new weights (with cost)
- Turnover comparison
- Expected transaction cost savings

---

[VISUALIZATION: Portfolio Optimization Flowchart]

Flowchart showing:
- Data ingestion → Covariance estimation → Return estimation
- Three parallel optimization paths: MVO, Risk Parity, HRP
- Performance evaluation and selection

---

# WARNING: Common Pitfalls

WARNING: **Estimation Error Amplification**
- Markowitz optimization is extremely sensitive to return estimate errors. Use robust techniques (Michaud resampling, Black-Litterman) in production.

WARNING: **Unrealistic Constraints**
- Overly tight constraints (very small position limits) can make the problem infeasible. Always verify feasibility before solving.

WARNING: **Singular Covariance Matrix**
- If covariance matrix is singular (more assets than observations), add regularization or shrinkage. Never invert a poorly-conditioned matrix.

WARNING: **Time-Varying Risk**
- Covariance matrices are NOT stationary. Use rolling windows or exponential weighted moving average (EWMA) for recent data.

WARNING: **Look-Ahead Bias**
- When backtesting, ensure you use only data available at decision time. Reestimate covariance and returns at each rebalancing date.

---

# Summary

This chapter equipped you with the complete portfolio optimization toolkit:

1. **Convex optimization** provides the mathematical foundation—problems are solvable, solutions are globally optimal.
2. **Markowitz mean-variance** optimization is elegant but fragile to estimation error.
3. **Covariance matrix estimation** via shrinkage and denoising is critical for stable portfolios.
4. **Advanced methods** (HRP, Black-Litterman, risk parity) address real-world challenges.
5. **CVXPY** is your production-ready solver.

In the next chapter, we'll integrate these portfolios into a complete backtesting and execution framework, accounting for realistic market frictions, liquidity constraints, and multi-period dynamics.

---

**End of Chapter 8**

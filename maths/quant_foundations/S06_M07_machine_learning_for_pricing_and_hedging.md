# Module 6.7 — Machine Learning for Pricing and Hedging

*Subject 6 capstone. Deep hedging, deep BSDE, signature methods, and neural operator learning for finance.*

---

## Prerequisites

- **Module 4.4 (HJB equations)** — deep BSDE solves HJB via backward SDE representation.
- **Module 4.7 (Reinforcement learning)** — deep hedging is policy-gradient-style learning.
- **Module 5.2 (Black-Scholes)** — classical hedging benchmark.
- **Module 6.1–6.6** — numerical methods this module extends.
- **Module 3.4 (SDEs)** and **Module 3.5 (Lévy processes)** — generative SDEs for training data.

---

## 6.7.1 Why neural networks now?

Classical quant numerics (trees, PDEs, MC, Fourier) achieve 3–4 decimal places at best for realistic portfolios. Three pressures are driving ML methods into production:

1. **Speed.** Real-time risk aggregation across 10⁵+ positions demands sub-millisecond pricing. A trained neural surrogate evaluates in 10 µs while an ADI solver takes 10 ms.
2. **Dimensionality.** 50-asset baskets, path-dependent Asian Bermudans, or whole-book MC VaR have nominal $d \ge 100$. NNs + automatic differentiation + SGD break the curse of dimensionality in ways traditional methods cannot.
3. **Market reality.** Transaction costs, discrete hedging, illiquidity, and real-world risk measures (ES, utility) break the Black-Scholes framework. NN-based optimization handles these constraints directly.

**The basic approach.** Replace classical rules (BS delta hedge, HJB optimal control) with a neural network trained to minimize an economic objective (P&L variance, expected shortfall, utility) on simulated or historical data.

---

## 6.7.2 Deep hedging (Buehler-Gonon-Teichmann-Wood 2019)

**Setup.** Discrete-time hedging of a claim $Z$ with maturity $T$ over dates $0 = t_0 < t_1 < \ldots < t_M = T$. Tradable assets: $S^j$, $j = 1, \ldots, d$. Hedging strategy: $\delta = (\delta_{t_k})_k$ where $\delta_{t_k}$ is a $\mathbb{R}^d$-valued function of observable information at $t_k$.

**P&L at maturity:**
$$
L_T(\delta) = -Z + \sum_{k=0}^{M-1} \delta_{t_k} \cdot (S_{t_{k+1}} - S_{t_k}) - \sum_k c(\delta_{t_k} - \delta_{t_{k-1}}, S_{t_k}),
$$
where $c$ is transaction cost. The seller of $Z$ wants this to be nonnegative with the lowest upfront premium $V_0$.

**Convex risk measure.** For a convex risk measure $\rho$ (e.g., entropic risk, expected shortfall, or CVaR):
$$
V_0 = \inf_\delta \rho(-L_T(\delta)).
$$

**Neural network parameterization.** $\delta_{t_k}(x) = \text{NN}_{t_k}^\theta(x)$ where $x$ is a feature vector (current $S$, signals, inventory, etc.). One NN per time step, or a shared NN across time.

**Loss function.** For entropic risk $\rho_\eta(X) = \tfrac{1}{\eta}\log\mathbb{E}[e^{-\eta X}]$:
$$
\mathcal{L}(\theta) = \frac{1}{\eta}\log\mathbb{E}[\exp(-\eta L_T(\delta^\theta))].
$$
Or for CVaR at level $\alpha$:
$$
\mathcal{L}(\theta) = V^\alpha + \frac{1}{\alpha}\mathbb{E}[(V^\alpha - L_T)_+],
$$
where $V^\alpha$ is the VaR at level $\alpha$ (a separate parameter minimized jointly).

**Training.** Sample $N$ paths from the market model (possibly calibrated to real data), compute $L_T$ for each path, backpropagate the loss to the NN parameters.

**Results.** In BS with zero transaction costs, deep hedging recovers the BS delta exactly. With non-zero transaction costs, it produces a non-BS hedge that substantially outperforms the classical delta. With jumps, rough vol, or exotic features, deep hedging often outperforms all classical alternatives.

---

## 6.7.3 Key advantages of deep hedging

1. **Transaction costs.** Handled automatically by the loss, no Clark-Ocone-style expansion needed.
2. **Multiple risk factors.** Volatility, interest rates, signals — all become NN inputs.
3. **Discrete hedging.** Once per hour or once per day? Just change $M$.
4. **Non-smooth payoffs.** Barriers, digitals, worst-of — NN smooths the decision surface via SGD regularization.
5. **Model-free backtesting.** Train on synthetic model paths, deploy on real data; compare realized P&L.

**Caveats.**
- Still needs a market-generation model (or a bootstrap procedure on historical data).
- Training requires care: reward engineering, learning-rate scheduling, network architecture.
- Interpretability is a challenge — deltas are human-readable; NN outputs are not.

---

## 6.7.4 Deep BSDE (E-Han-Jentzen 2017)

**Problem.** Solve the nonlinear backward stochastic differential equation (BSDE):
$$
Y_t = \xi + \int_t^T f(s, X_s, Y_s, Z_s) ds - \int_t^T Z_s dB_s,
$$
where $X$ is a forward SDE, $\xi = g(X_T)$, and $(Y, Z)$ are solution processes. By Feynman-Kac, $Y_t = u(t, X_t)$ for $u$ solving a semilinear PDE.

**Deep BSDE algorithm.**
- **Forward simulation.** Simulate $X$ and $B$ from $0$ to $T$.
- **NN for $Z_t$.** $Z_t = \text{NN}^\theta_t(X_t)$ — gradients of value function approximated by NNs.
- **NN for $Y_0$.** A learnable parameter $\theta_0$.
- **Euler scheme for $Y$**: $Y_{t_{k+1}} \approx Y_{t_k} - f(t_k, X_{t_k}, Y_{t_k}, Z_{t_k}) \Delta t + Z_{t_k} \Delta B_{t_k}$.
- **Loss**: $\mathcal{L}(\theta) = \mathbb{E}[(Y_T - g(X_T))^2]$.
- **Train** with SGD, one path per batch step.

**Convergence (Han-Jentzen-E 2018).** As the number of time steps $M$ and network width increase, $Y_0^\theta \to Y_0$ in probability.

**Breakthrough**: solves 100-dim PDEs like Allen-Cahn, HJB, and Black-Scholes-Barenblatt without curse of dimensionality. Training time $\sim$ minutes on a GPU.

**In finance**: used for CVA/FVA/XVA in high-dimensional derivative books, and for pricing exotic Bermudans in multi-factor models.

---

## 6.7.5 Signature methods (Lyons, Levin-Lyons-Ni, Chevyrev-Kormilitzin)

**Signatures** are the algebra-free, coordinate-free way to encode a path. For a path $X: [0, T] \to \mathbb{R}^d$, the signature $\mathbb{S}(X)$ is
$$
\mathbb{S}(X) = (1, \int dX, \int\int dX dX, \int\int\int dX dX dX, \ldots) \in T((\mathbb{R}^d)),
$$
the tensor algebra of iterated integrals.

**Universality** (Bonnier-Kormilitzin-Oberhauser 2019). Any continuous functional $f: C([0,T], \mathbb{R}^d) \to \mathbb{R}$ can be approximated by a linear functional of $\mathbb{S}(X)$ — the **signature kernel** yields a universal feature representation of paths.

**Application to pricing.** 
- Path-dependent options on $X$ become linear functions of the truncated signature.
- Regression on signatures gives a principled nonparametric alternative to Longstaff-Schwartz polynomial basis.
- Signature kernel MMD gives a powerful test for path distributions.

**Log-signatures** are compressed (free Lie algebra). Fewer parameters for the same approximation power.

**Production use**: Bloomberg, G-Research, and academic groups apply signatures to calibration, hedging, and forecasting. Rough path theory (Lyons 1998) is the theoretical underpinning.

---

## 6.7.6 Neural SDEs and generative models for market data

**Neural SDE**: $dX_t = f_\theta(t, X_t) dt + g_\theta(t, X_t) dB_t$ where $f, g$ are NNs. Fit to historical data by **pathwise likelihood** (Wang-Blanchet 2021) or **sig-Wasserstein** distance.

**Applications.**
- **Market simulation**: generate realistic paths for stress testing, backtesting strategies, or training deep hedgers.
- **Calibration**: fit to vanilla options by minimizing IV-surface distance.
- **Counterfactual analysis**: what would the P&L have been under an alternative regime?

**Critic-based training.** For distributional matching, use GAN-style critic (Wasserstein GAN) on path signatures.

**Neural rough paths (Kidger-Morrill-Foster-Lyons 2020).** Combines NSDEs with rough path theory; effective for irregularly-sampled financial data.

---

## 6.7.7 Operator learning: pricing as a map

**Goal.** Learn the operator $\Pi: \theta \mapsto V(\theta)$, mapping model parameters (Heston $\kappa, \theta, \xi, \rho, V_0$; or local vol surface $\sigma(K, T)$) to option prices/volatility surfaces.

**Architectures.**
- **Plain MLP**: concatenate parameters + strikes/maturities → price.
- **Fourier Neural Operator (Kovachki et al.)**: represent map as spectral convolution, faster than MLP for structured problems.
- **DeepONet (Lu et al.)**: branch net for parameters, trunk net for coordinates.

**Training**. Generate millions of (parameters, prices) pairs via classical pricer (Fourier for Heston, LSM for Bermudan). Train NN offline.

**Deployment**. Inference in 10 µs. Calibration loop: 100 evaluations per iteration × 20 iterations = 2 ms total vs. 2 minutes classical.

**Industrial use.** All major banks have operator-learning pipelines for their Heston / local-vol / LMM calibration.

---

## 6.7.8 Practical deep learning for quant

**Data generation.**
- Simulate under the pricing model (Heston, rough Bergomi, LMM).
- Augment with empirical noise (residuals from regression, historical shocks).
- Use importance sampling to oversample the tails of interest.

**Architecture choices.**
- 3–5 hidden layers, width 64–256 per layer.
- Smooth activations (tanh, softplus) preferred over ReLU for gradient flow through hedging horizons.
- Batch normalization rarely helps for MC-style losses.
- Residual / skip connections for very deep networks.

**Optimization.**
- Adam optimizer with learning rate 1e-3, cosine decay.
- Batch size 512–4096; larger batches reduce variance but slow optimization.
- Warm-start with BS hedge: initialize NN output to match BS delta as pre-training.

**Regularization.**
- L2 weight decay.
- Dropout in input or intermediate layers.
- Early stopping on validation P&L.

**Evaluation.**
- Out-of-sample P&L distribution (test set of simulated paths).
- Tail statistics: ES at 1%, 5%.
- Robustness: how does performance degrade under parameter perturbation?

---

## 6.7.9 Python: deep hedging skeleton

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1. Deep hedging of a European call in BS with transaction cost
# ============================================================
class HedgingNet(nn.Module):
    """One NN per time step, input = (t, S, current_holding)."""
    def __init__(self, n_steps, hidden=64):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1)
            )
            for _ in range(n_steps)
        ])

    def forward(self, t_idx, S_t):
        return self.nets[t_idx](torch.stack([S_t, torch.log(S_t)], dim=-1)).squeeze(-1)

def simulate_deep_hedging(params, net, N=4096, M=20, tc_rate=0.001):
    S0, K, r, sigma, T = params
    dt = T/M
    device = next(net.parameters()).device
    S = torch.full((N,), S0, device=device)
    cash = torch.zeros(N, device=device)
    holding = torch.zeros(N, device=device)

    for m in range(M):
        delta_new = net(m, S)  # NN output = new holding
        # Transaction cost
        tc = tc_rate * S * torch.abs(delta_new - holding)
        cash = cash - (delta_new - holding)*S - tc
        holding = delta_new
        # Step price
        Z = torch.randn(N, device=device)
        S = S * torch.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        cash = cash * np.exp(r*dt)

    # Close position at T: sell holding
    tc_final = tc_rate * S * torch.abs(holding)
    cash = cash + holding * S - tc_final
    # Subtract call payoff
    payoff = torch.clamp(S - K, min=0)
    pnl = cash - payoff
    return pnl

def train_deep_hedger(params, n_epochs=500, lr=1e-3):
    M = 20
    net = HedgingNet(n_steps=M, hidden=64)
    opt = optim.Adam(net.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        pnl = simulate_deep_hedging(params, net, N=4096, M=M)
        # Entropic risk / mean-variance loss: mean + c*var
        loss = -pnl.mean() + 0.5 * pnl.var()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"mean P&L={pnl.mean().item():.4f}, std={pnl.std().item():.4f}")
        losses.append(loss.item())
    return net, losses

# Run:
torch.manual_seed(2026)
params = (100, 100, 0.05, 0.2, 1.0)
# Uncomment to train (takes ~30 seconds):
# net, losses = train_deep_hedger(params, n_epochs=300)

# ============================================================
# 2. Deep BSDE for 100-dim Black-Scholes-Barenblatt (E-Han-Jentzen)
# ============================================================
class DeepBSDE(nn.Module):
    def __init__(self, d=100, M=20, hidden=64):
        super().__init__()
        self.d = d
        self.M = M
        # NN for Z (gradient) at each time step
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, d)
            )
            for _ in range(M)
        ])
        # Y_0 is a learnable parameter
        self.Y0 = nn.Parameter(torch.zeros(1))

    def forward(self, X_0, dt, sigma, r, driver, terminal):
        """Simulate forward and solve Y backward via Euler."""
        N = X_0.shape[0]
        X = X_0.clone()
        Y = self.Y0.expand(N)
        for m in range(self.M):
            Z = self.nets[m](X) / self.d  # normalize
            dB = torch.randn_like(X) * np.sqrt(dt)
            # Forward X: dX = rX dt + σX dB (geometric)
            X = X + r*X*dt + sigma*X*dB
            # Backward Y via forward Euler
            f = driver(X, Y, Z)
            Y = Y - f*dt + (Z * dB).sum(dim=-1)
        return Y, terminal(X), X

def train_deep_bsde():
    d, M = 10, 20  # small example for demo
    net = DeepBSDE(d=d, M=M, hidden=64)
    opt = optim.Adam(net.parameters(), lr=1e-2)
    T = 1.0; dt = T/M
    r, sigma = 0.05, 0.2
    N = 512
    def driver(X, Y, Z): return -r*Y  # linear BS driver
    def terminal(X): return (X.max(dim=-1).values - 100).clamp(min=0)
    for ep in range(500):
        X_0 = 100*torch.ones(N, d)
        Y_T, g_XT, _ = net(X_0, dt, sigma, r, driver, terminal)
        loss = ((Y_T - g_XT)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 100 == 0:
            print(f"DeepBSDE ep {ep+1}: loss={loss.item():.4e}, Y0={net.Y0.item():.4f}")
    return net

# Uncomment to run:
# net_bsde = train_deep_bsde()

# ============================================================
# 3. Neural network for Heston option price surrogate
# ============================================================
class HestonPriceNet(nn.Module):
    """Input: (kappa, theta, xi, rho, V0, K, T) → price."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

# Training data generation (sketch):
# 1. Sample 10^5 parameter vectors from realistic ranges
# 2. For each, price via Carr-Madan FFT or COS
# 3. Train NN to predict price; use (K, T) as conditional inputs
# 4. Deploy: 10 μs inference vs 10 ms FFT → 1000x speedup

print("ML skeletons defined. Uncomment train_deep_hedger() etc. to run.")
```

**What to check.**
- Deep hedger should converge to P&L std close to BS delta hedge residual for zero transaction costs.
- With nonzero $tc$, deep hedger achieves lower std than BS delta.
- Deep BSDE Y0 should converge to the true option price (compare to closed form).
- Heston NN surrogate trained on 10⁵ samples should match Carr-Madan to $\sim$10⁻⁴ relative error.

---

## 6.7.10 [QUANT APPLICATIONS]

1. **Deep hedging in production.** Major banks (JPM, Credit Suisse before, Goldman) deployed deep-hedging for equity derivatives, especially structured products with transaction costs.
2. **XVA calculation.** Deep BSDE for nested XVA portfolios; 100-dim diffusions priced in minutes.
3. **Exotic calibration.** NN surrogate for Bermudan-swaption repricing enables real-time Greeks and scenario analysis.
4. **High-frequency market making.** Deep-RL for quoting with inventory constraints, transaction costs, adverse selection modeling.
5. **Optimal execution with signals.** NN replaces Almgren-Chriss with flexible, signal-aware policies.
6. **Portfolio construction.** Deep-RL for multi-asset portfolios under real-world frictions; beats classical Merton / Markowitz in backtests.
7. **Risk factor discovery.** Autoencoders on factor returns; principal-component decomposition with nonlinearity.
8. **Generative market scenarios.** NSDEs / signature GANs for synthetic data augmentation in stress tests and regulatory capital.
9. **Volatility-surface interpolation.** SVI + NN residual correction gives arbitrage-free surfaces calibrated to market in real time.
10. **Anomaly detection.** Variational autoencoders detect regime changes, failed calibrations, data-quality issues.

---

## 6.7.11 Exercises

**★ (concept drills).**
1. Why does deep hedging recover BS delta when there are no transaction costs?
2. What is the role of the $Z$ process in a BSDE? Why do we approximate it with a neural network?
3. Give an example of a problem where signatures outperform polynomial bases for path regression.
4. In operator learning (e.g., DeepONet), why does splitting inputs into parameters + coordinates help?
5. What are the main differences between deep hedging and deep reinforcement learning?
6. Why is training deep BSDE stable even in 100 dimensions?

**★★ (calculation).**
7. Implement deep hedger for a European call in BS with $tc = 0.1\%$ per-trade. Compare terminal P&L to delta hedger. Quantify the variance reduction.
8. Extend the deep hedger to price a down-and-out barrier. Does it handle the barrier monitoring cleanly?
9. Deep BSDE for Black-Scholes-Barenblatt. Implement and verify $Y_0 \approx$ analytic solution for $d = 50$.
10. Train a Heston price surrogate. Use 10⁵ random Heston parameter sets; target accuracy $10^{-4}$ relative error. Compare calibration speed using NN vs Carr-Madan.
11. Learn a signature-based regressor for a 10-dim Bermudan swaption. Compare to polynomial LSM; measure convergence.

**★★★ (open / research).**
12. **Deep hedging beyond BS.** Hedge a cliquet in rough Bergomi. Compare deep hedger P&L to naive Black-Scholes and implied-vol rolling delta.
13. **Deep BSDE for HJB.** Solve a 20-dim Merton portfolio problem with consumption via deep BSDE. Compare to the closed-form CRRA solution.
14. **Neural SDE calibration.** Fit a neural SDE to historical S&P 500 returns. Test on stress-regime detection.
15. **Signature kernel calibration.** Calibrate a rough vol model using signature-MMD loss. Compare to IV-surface distance.
16. **Deep optimal stopping in $d = 100$.** Reproduce Becker-Cheridito-Jentzen for max-call on 100 assets. How many training samples are needed?
17. **Meta-learning across models.** Train one NN that prices options under both Heston and rough Bergomi by conditioning on a learned model embedding. How transferable are the representations?

---

## Subject 6 — Capstone Summary

Subject 6 traversed the algorithmic landscape of quantitative finance:

**Module 6.1 (Monte Carlo)** — the universal solver. $N^{-1/2}$ error, dimension-independent. Variance reduction (antithetic, control, importance, multilevel) makes it tractable.

**Module 6.2 (Quasi-Monte Carlo)** — deterministic points with $(\log N)^d/N$ discrepancy. Combined with Brownian bridge or PCA, QMC converges orders of magnitude faster than MC on financial payoffs.

**Module 6.3 (Finite Differences)** — explicit, implicit, Crank-Nicolson schemes. ADI for multi-dimensional problems. PSOR for American LCPs. The classical gold standard in 1D and 2D.

**Module 6.4 (Trees and Lattices)** — CRR binomial, Leisen-Reimer, Hull-White short-rate trees. Pedagogically clear and practically useful in single-factor settings.

**Module 6.5 (Fourier Methods)** — when the characteristic function is known (Heston, Bates, CGMY), Carr-Madan FFT and Fang-Oosterlee COS give spectral convergence. Dominant for vanilla calibration.

**Module 6.6 (American Options)** — Longstaff-Schwartz regression-based MC, dual martingale methods, stochastic mesh, neural-network optimal stopping. The only way to price high-dimensional Bermudans.

**Module 6.7 (Machine Learning)** — deep hedging, deep BSDE, signatures, operator learning. The frontier that unifies and extends all prior techniques.

**Bridge to Subject 7.** We have built pricers that compute a number. The next question: what should we believe about the dynamics of markets, and how do we estimate parameters from data? Subject 7 covers statistical learning and econometrics for quant: ML regression, classification, time-series models, factor structures, and causal inference — the tools that turn data into actionable models.

---

*— End of Module 6.7 and Subject 6. Next: Subject 7 — Statistical Learning & Econometrics for Quant.*

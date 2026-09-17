# Subject 4, Module 7: Reinforcement Learning and Approximate Dynamic Programming

> *"RL is optimal control without a model. Its theorems are the Bellman equations in disguise; its algorithms are the way we solve those equations when the state space is too big to enumerate and the transition dynamics are unknown."*

## Prerequisites

- **Module 4.3**: Bellman's principle, contraction mappings, policy iteration.
- **Module 4.4**: HJB (the continuous-time analog).
- **Module 4.5**: Merton (concrete example of a solvable DP).
- **Module 4.6**: LQG (closed-form solution used as a benchmark and as the kernel of iLQR/DDP).
- Helpful: probability and statistical learning basics, stochastic approximation.

## 4.7.0 Why RL for Quants?

**Optimal control** (Modules 4.3-4.6) assumes the model is known: transition probabilities $P(x'|x, a)$, reward $R(x, a)$. When you have a Merton problem with specified $\mu, \sigma$, you solve the HJB and you're done.

**Reinforcement learning (RL)** tackles the case when the model is *unknown* or *too complex to use directly*, and you must learn the optimal policy from data / simulation / interaction with the environment.

In practice many quant problems are a spectrum between the two:
- **Model-based**: known dynamics, solved via HJB → Merton, Almgren-Chriss.
- **Model-free**: unknown or non-parametric dynamics, solved via RL → market making with adversarial order flow, portfolio choice under regime-switching of unknown form.
- **Hybrid**: approximate model, warm start with model-based solution, then fine-tune with RL data → most real trading systems.

Quant applications of RL:
1. Execution algorithms (LOB microstructure too complex for clean models).
2. Market making under partial information (Guéant-Lehalle with data-driven extensions).
3. Portfolio construction with transaction costs and realistic frictions.
4. Option hedging under realistic microstructure (deep hedging, Buehler et al.).
5. Regime-adaptive asset allocation.
6. Pairs trading / statistical arbitrage.

Roadmap:
- **4.7.1** MDP review and value-based methods (policy evaluation, TD).
- **4.7.2** Q-learning (off-policy TD for optimal control).
- **4.7.3** Function approximation and semi-gradient methods.
- **4.7.4** Policy-gradient methods (REINFORCE, actor-critic).
- **4.7.5** Deep RL: DQN, DDPG, PPO.
- **4.7.6** Continuous-time and policy-space extensions.
- **4.7.7** Convergence theory and stochastic approximation.
- **4.7.8** Applications to quant problems.

---

## 4.7.1 MDP Review and Policy Evaluation

### MDP setup

Tuple $(\mathcal{X}, \mathcal{A}, P, r, \gamma)$:
- $\mathcal{X}$: state space (finite for now; continuous handled later).
- $\mathcal{A}$: action space.
- $P(x' | x, a)$: transition probability.
- $r(x, a)$: reward.
- $\gamma \in (0, 1)$: discount.

Policy $\pi: \mathcal{X} \to \Delta(\mathcal{A})$ (possibly stochastic).

Value functions:
$$V^\pi(x) = \mathbb{E}_\pi\left[ \sum_{t \ge 0} \gamma^t r(X_t, A_t) \mid X_0 = x\right].$$
$$Q^\pi(x, a) = r(x, a) + \gamma \sum_{x'} P(x' | x, a) V^\pi(x').$$

Optimal: $V^*(x) = \sup_\pi V^\pi(x)$, $Q^*(x, a) = \sup_\pi Q^\pi(x, a)$.

**Bellman optimality equations:**
$$V^*(x) = \sup_{a \in \mathcal{A}} Q^*(x, a),$$
$$Q^*(x, a) = r(x, a) + \gamma \sum_{x'} P(x' | x, a) V^*(x') = r(x, a) + \gamma \sum_{x'} P(x' | x, a) \sup_{a'} Q^*(x', a').$$

### Policy evaluation: Monte Carlo

Given $\pi$, estimate $V^\pi(x)$:
- Run trajectory $x_0, a_0, r_0, x_1, a_1, \ldots$ under $\pi$.
- Return $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$.
- MC estimator: $V^\pi(x) \approx$ empirical mean of $G_t$ when $X_t = x$.

Unbiased but high variance, requires full trajectories.

### Temporal Difference (TD) learning

Key insight (Sutton 1988): update $V(x)$ using a **bootstrapped** target $r_t + \gamma V(x_{t+1})$ instead of waiting for full return.

**TD(0) update:**
$$V(x_t) \leftarrow V(x_t) + \alpha \underbrace{[r_t + \gamma V(x_{t+1}) - V(x_t)]}_{\text{TD error } \delta_t},$$
with step size $\alpha_t$.

TD(0) is **online**, **incremental**, and converges to $V^\pi$ under standard stochastic-approximation conditions: $\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$ (Robbins-Monro).

**TD($\lambda$):** interpolation between TD(0) and MC using eligibility traces, trading bias-variance.

### Convergence theorem (tabular case)

Under ergodicity of the Markov chain under $\pi$ and Robbins-Monro step sizes, $V_n(x) \to V^\pi(x)$ almost surely.

Proof: TD(0) is a stochastic approximation algorithm for solving the fixed-point equation $V = T^\pi V$, where $T^\pi$ is the Bellman evaluation operator (contraction in sup norm with factor $\gamma$).

---

## 4.7.2 Q-Learning

### The algorithm

**Watkins (1989)**: update Q-values from samples:

$$Q(x_t, a_t) \leftarrow Q(x_t, a_t) + \alpha_t [r_t + \gamma \max_{a'} Q(x_{t+1}, a') - Q(x_t, a_t)].$$

This is TD applied to the **Bellman optimality equation**, using the max over actions in the target. It is **off-policy**: samples can come from any exploratory behavior policy, while we learn the optimal Q.

### Convergence theorem (Watkins & Dayan 1992)

If every state-action pair is visited infinitely often, step sizes satisfy Robbins-Monro, and the state-action space is finite, then $Q_n(x, a) \to Q^*(x, a)$ a.s.

Proof sketch: Q-learning is a stochastic approximation scheme for the fixed-point of the Bellman optimality operator $T^*$ which is a $\gamma$-contraction in sup norm. The Kushner-Clark ODE method or martingale convergence establishes convergence.

### Exploration

To ensure all state-action pairs are visited, the behavior policy needs exploration. Common choices:
- **$\epsilon$-greedy**: with probability $\epsilon$ pick random action, else argmax Q.
- **Softmax (Boltzmann)**: $P(a | x) \propto \exp(Q(x, a) / \tau)$, temperature $\tau$.
- **Optimism in the face of uncertainty (UCB)**: $a = \text{argmax}_a [Q(x, a) + c / \sqrt{N(x, a)}]$.

---

## 4.7.3 Function Approximation

### The curse of dimensionality

Tabular methods store one value per state (or state-action pair). With $|\mathcal{X}| = 10^{10}$ (large portfolio state space) or continuous state, tabular is infeasible.

**Solution: function approximation.** Parameterize $V$ or $Q$ with parameters $\theta \in \mathbb{R}^d$:
$$Q(x, a) \approx Q_\theta(x, a).$$

Common choices:
1. **Linear in features:** $Q_\theta(x, a) = \theta^\top \phi(x, a)$ for feature map $\phi$.
2. **Neural network:** $Q_\theta(x, a) = \text{NN}(\phi; \theta)$.
3. **Kernel methods / GP:** $Q_\theta(x, a) = \sum_i \alpha_i k((x, a), (x_i, a_i))$.

### Semi-gradient TD

Update parameters toward the TD target:
$$\theta \leftarrow \theta + \alpha \delta_t \nabla_\theta Q_\theta(x_t, a_t),$$
where $\delta_t = r_t + \gamma Q_\theta(x_{t+1}, \text{argmax} Q) - Q_\theta(x_t, a_t)$.

**"Semi-gradient"** because the target $r + \gamma Q_{\theta}(x', a')$ itself depends on $\theta$ but we treat it as a constant for gradient purposes. This is *not* a true gradient step (not minimizing a loss), but works in practice for well-conditioned problems.

### Deadly triad (Sutton & Barto)

Combining (1) bootstrapping, (2) off-policy learning, and (3) function approximation can cause divergence. Example: Baird's counterexample shows a linear Q-learning algorithm can diverge with innocent-looking features.

Mitigation:
- **Target network:** use a separate $\theta^-$ for target computation, updated slowly (DQN innovation).
- **Gradient TD:** true-gradient methods (GTD, GTD2) that minimize a proper loss.
- **Importance sampling ratios** to correct off-policy bias.

---

## 4.7.4 Policy-Gradient Methods

### The idea

Parameterize the policy directly: $\pi_\theta(a | x)$. Optimize
$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t \ge 0} \gamma^t r(X_t, A_t) \right].$$

Compute gradient w.r.t. $\theta$ and ascend.

### Policy Gradient Theorem (Sutton et al. 1999)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_t \gamma^t \nabla_\theta \log \pi_\theta(A_t | X_t) \cdot Q^{\pi_\theta}(X_t, A_t) \right].$$

**Proof idea.** Differentiate the expected return through the action distribution using the log-derivative trick $\nabla \log \pi = (\nabla \pi) / \pi$. Magic: the environment dynamics drop out because they don't depend on $\theta$.

### REINFORCE

Vanilla policy-gradient Monte Carlo estimator:
$$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t | x_t) G_t,$$
with $G_t$ the realized return from time $t$.

**Variance reduction:** subtract a state-dependent baseline $b(x_t)$:
$$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t | x_t) (G_t - b(x_t)).$$

The baseline doesn't change the expectation but can dramatically reduce variance. Optimal baseline: $b(x) = V^\pi(x)$.

### Actor-Critic

Combine policy-gradient (actor) with a learned value function (critic):
- **Actor:** $\pi_\theta(a | x)$.
- **Critic:** $V^{\pi_\theta}_\phi(x)$ or $Q^{\pi_\theta}_\phi(x, a)$.

Advantage function $A^\pi(x, a) = Q^\pi(x, a) - V^\pi(x)$ is the "how much better than average".

**A2C update:** $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a | x) A(x, a)$, where $A(x, a) = r + \gamma V_\phi(x') - V_\phi(x)$ is the TD-error advantage estimate.

Critic update: standard TD on $V_\phi$.

Actor-critic has lower variance than REINFORCE (critic smooths out noise) and is more sample efficient.

### Natural policy gradient and TRPO/PPO

Natural gradient: rescale the gradient by the inverse Fisher information matrix of $\pi_\theta$, giving the "steepest ascent in KL divergence."

**TRPO (Schulman 2015):** approximate natural gradient with trust region.
**PPO (Schulman 2017):** simpler variant with clipped objective. The workhorse of modern policy gradient algorithms.

---

## 4.7.5 Deep Reinforcement Learning

### DQN (Mnih et al. 2013, Nature 2015)

Q-learning + deep neural network + three tricks:
1. **Experience replay**: store $(x_t, a_t, r_t, x_{t+1})$ in buffer; sample minibatches. Breaks temporal correlation.
2. **Target network** $\theta^-$: fixed for $k$ steps, then copied. Stabilizes bootstrapping target.
3. **Reward/state preprocessing**: clipping, normalization.

Loss:
$$L(\theta) = \mathbb{E}_{(x, a, r, x') \sim \mathcal{D}} \left[ (r + \gamma \max_{a'} Q_{\theta^-}(x', a') - Q_\theta(x, a))^2 \right].$$

Variants: Double DQN (mitigates max-bias), Dueling DQN (separate $V$ and advantage streams), Prioritized Experience Replay.

### Continuous actions: DDPG / TD3

For continuous action spaces, the $\max_a Q(x, a)$ becomes nontrivial. DDPG (Lillicrap 2015):
- Actor $\mu_\theta: \mathcal{X} \to \mathcal{A}$ (deterministic).
- Critic $Q_\phi(x, a)$.
- Use $\mu_\theta(x)$ as argmax approximation.

Updates:
$$\phi \leftarrow \phi + \alpha \nabla_\phi (r + \gamma Q_{\phi^-}(x', \mu_{\theta^-}(x')) - Q_\phi(x, a))^2,$$
$$\theta \leftarrow \theta + \alpha \nabla_\theta Q_\phi(x, \mu_\theta(x)).$$

TD3 (Fujimoto 2018): adds twin critics (take min to reduce overestimation), target smoothing, delayed policy updates.

### Soft Actor-Critic (SAC)

Maximum-entropy RL: add entropy bonus to the objective.
$$J(\pi) = \mathbb{E}_\pi \left[ \sum_t \gamma^t (r(x_t, a_t) + \alpha H(\pi(\cdot | x_t))) \right].$$

Natural exploration, robust, and often SOTA on continuous control benchmarks.

### PPO

**Proximal Policy Optimization**. Policy gradient with a clipped surrogate:
$$L(\theta) = \mathbb{E}\left[ \min\left( \dfrac{\pi_\theta(a | x)}{\pi_{\theta_{old}}(a | x)} A, \text{clip}(\cdot, 1 - \epsilon, 1 + \epsilon) A \right) \right].$$

Most popular deep-RL algorithm in practice as of 2024.

---

## 4.7.6 Continuous-Time RL

### From discrete to continuous

Continuous-time RL bridges HJB and RL. Instead of iteration on Bellman equation, use time discretization + RL.

**Deep BSDE method (Han-Jentzen-E 2018):** turn the HJB-PDE into a *forward* BSDE with neural-network parameterization of the control. Training minimizes a terminal boundary violation.

**Reinforcement learning via PDE:** Jia and Zhou (2022) derive policy improvement and TD learning directly in continuous time using Itô calculus. Key insight: the martingale property of the value function (under the optimal policy) gives a loss function for policy evaluation.

### Exploration in continuous time

Wang-Zhou (2020): solve entropy-regularized continuous-time RL with Gaussian policies, deriving a modified HJB with entropy term. For LQG the optimal exploratory policy is Gaussian with mean = LQR controller and variance inversely proportional to value function second derivative.

---

## 4.7.7 Convergence Theory: Stochastic Approximation

### Robbins-Monro

Find $\theta^*$ with $f(\theta^*) = 0$ using noisy observations $f(\theta) + \eta$. Iteration:
$$\theta_{n+1} = \theta_n + \alpha_n [f(\theta_n) + \eta_n].$$

**Theorem (Robbins-Monro 1951):** If $\sum \alpha_n = \infty$, $\sum \alpha_n^2 < \infty$, $\eta_n$ is martingale-difference with bounded variance, and $f$ is sufficiently regular (e.g., $f(\theta) = -\nabla F$ for convex $F$), then $\theta_n \to \theta^*$ a.s.

### Kushner-Clark ODE method

A classical technique to analyze stochastic approximation. The iterates track an underlying ODE:
$$\dot \theta(t) = f(\theta(t)).$$

If the ODE has globally asymptotically stable equilibrium $\theta^*$ and certain noise/regularity hold, the stochastic iterates converge to $\theta^*$.

### Application to TD and Q-learning

- **TD(0) with linear FA:** tracks ODE $\dot \theta = A\theta + b$; converges iff $A$ is Hurwitz, which holds for on-policy learning.
- **Q-learning (tabular):** tracks a nonlinear but monotone ODE; converges by contraction arguments.
- **Q-learning (FA):** may diverge; two-timescale stochastic approximation (Borkar) needed for analysis.

---

## 4.7.8 Quant Finance Applications of RL

### 1. Optimal execution under complex frictions (Nevmyvaka, Feng, Kearns 2006)

Apply Q-learning to learn an execution policy on real limit-order-book data. Features: price, remaining shares, time. Reward: negative implementation shortfall. Shows RL outperforms Almgren-Chriss when there are predictable patterns and nonlinear impact.

### 2. Market making (Guéant-Lehalle-Tapia 2013 + RL extensions)

Model-based: HJB gives optimal bid/ask given inventory and volatility. RL extension: learn quote policy from LOB simulator, relaxing Markovian and Gaussian assumptions. Agents like TensorTrade, Microsoft MARL paper.

### 3. Hedging derivatives (Deep Hedging, Buehler-Gonon-Teichmann-Wood 2019)

Given payoff $g(S_T)$, learn hedging strategy as neural network $(\delta_t = \text{NN}(S_t, t; \theta))$ minimizing variance of PnL. Generalizes Black-Scholes hedging to transaction costs, market frictions, partial information.

### 4. Portfolio allocation (Jiang, Xu, Liang 2017)

Learn allocation across crypto assets using historical price features; daily rebalancing. Shows deep RL can exploit short-term momentum and mean-reversion.

### 5. Order flow prediction (HFT)

Model order-book imbalance evolution as MDP; learn actions (aggressive vs patient placement) to minimize adverse selection. Critical for electronic market making.

### 6. Pairs trading (Lu-Shen-Wang 2023)

Use actor-critic to learn dynamic entry/exit for cointegrated pairs, competitive with model-based strategies while adapting to changing cointegration coefficients.

### 7. Option pricing under model uncertainty

Train RL agents to price exotics under an ensemble of models, producing robust price bands that hedge against model risk.

### 8. Optimal mortgage prepayment (Fan 2014)

Model mortgage holder as RL agent maximizing utility; predicts prepayment behavior that isn't captured by pure rationality.

### 9. Currency management / carry trades

Policy gradient for dynamic currency allocation accounting for carry, momentum, value signals.

### 10. Reinforcement learning for market making on decentralized exchanges

AMM (Uniswap) liquidity provision: choose tick ranges dynamically. Deep RL used by professional LP optimizers.

---

## 4.7.9 Python: Tabular and Deep RL Examples

```python
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Example 1: Tabular Q-learning on Gridworld
# -----------------------------
# 4x4 grid, reward +1 at goal, 0 elsewhere
N = 4
goal = (3, 3)
actions = [(0,1), (1,0), (0,-1), (-1,0)]  # R,D,L,U

def step(state, a):
    x, y = state
    dx, dy = actions[a]
    x2 = max(0, min(N-1, x+dx))
    y2 = max(0, min(N-1, y+dy))
    new_state = (x2, y2)
    reward = 1.0 if new_state == goal else 0.0
    done = new_state == goal
    return new_state, reward, done

Q = np.zeros((N, N, 4))
gamma = 0.9
alpha = 0.5
epsilon = 0.1

for episode in range(5000):
    s = (0, 0)
    done = False
    while not done:
        if np.random.random() < epsilon:
            a = np.random.randint(4)
        else:
            a = int(np.argmax(Q[s[0], s[1]]))
        s2, r, done = step(s, a)
        target = r + gamma * np.max(Q[s2[0], s2[1]]) * (not done)
        Q[s[0], s[1], a] += alpha * (target - Q[s[0], s[1], a])
        s = s2

V = Q.max(axis=2)
print("Learned V(x):")
print(np.round(V, 3))
# Should be gamma^(manhattan_distance_to_goal)

# -----------------------------
# Example 2: Policy gradient (REINFORCE) on CartPole-like
# -----------------------------
# Simple: learn sign-based policy for 1D random walk aim at target
np.random.seed(0)

def run_episode(theta):
    x = 0.0
    traj_actions = []
    traj_states = []
    traj_rewards = []
    for t in range(50):
        prob_right = 1.0 / (1.0 + np.exp(-theta * x))
        a = 1 if np.random.random() < prob_right else -1
        x_new = x + a + 0.3 * np.random.randn()
        r = -abs(x_new - 5.0)  # target is x=5
        traj_states.append(x)
        traj_actions.append(a)
        traj_rewards.append(r)
        x = x_new
    return traj_states, traj_actions, traj_rewards

def grad_log_pi(theta, x, a):
    p = 1.0 / (1.0 + np.exp(-theta * x))
    if a == 1:
        return x * (1 - p)
    else:
        return -x * p

theta = 0.0
lr = 1e-3
rewards_over_eps = []
for ep in range(2000):
    states, acts, rs = run_episode(theta)
    G = np.cumsum(rs[::-1])[::-1]  # returns
    baseline = np.mean(G)
    g = sum(grad_log_pi(theta, s, a) * (g - baseline) for s, a, g in zip(states, acts, G))
    theta += lr * g
    rewards_over_eps.append(sum(rs))

plt.figure(figsize=(9, 4))
plt.plot(np.convolve(rewards_over_eps, np.ones(50)/50, 'valid'))
plt.xlabel('episode'); plt.ylabel('smoothed episode reward')
plt.title(f'REINFORCE (final theta = {theta:.3f})')
plt.grid(alpha=0.3); plt.show()

# -----------------------------
# Example 3: DQN skeleton (simplified)
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def train_dqn(env_step, n_states, n_actions, n_episodes=500):
    Q = QNetwork(n_states, n_actions)
    Q_target = QNetwork(n_states, n_actions)
    Q_target.load_state_dict(Q.state_dict())
    opt = optim.Adam(Q.parameters(), lr=1e-3)
    buffer = deque(maxlen=10000)
    gamma = 0.99
    eps = 1.0
    for ep in range(n_episodes):
        # (... environment loop with replay and target update ...)
        pass  # skeleton
    return Q

# -----------------------------
# Example 4: Mini deep hedging
# -----------------------------
# Hedge a European call under simulated GBM with transaction costs.
# Learn delta(t, S) as neural network minimizing variance of P&L.
# (pseudo code; full implementation would need more care)
```

---

## 4.7.10 Summary

- **RL** solves optimal-control problems when the model is unknown.
- **Value-based (Q-learning)**, **policy-based (REINFORCE)**, **actor-critic** are the three families.
- **Function approximation** (linear, neural, kernel) is necessary for large/continuous state spaces.
- **Deep RL** (DQN, PPO, SAC, TD3) combines gradient-based learning with deep networks.
- **Convergence** relies on stochastic-approximation theory, contraction mappings, and two-timescale arguments.
- **Quant applications** span execution, market making, hedging, portfolio management, and more — especially where real data exhibits patterns no parametric model captures.
- **Continuous-time RL** bridges HJB and RL via martingale characterizations and deep BSDE.

### Forward pointers

**Subject 4 Capstone.** This module completes Subject 4 (Optimal Stopping and Control). We've covered:
- Module 4.1: Optimal stopping and the Snell envelope.
- Module 4.2: American options and free-boundary problems.
- Module 4.3: Dynamic programming and Bellman.
- Module 4.4: HJB equations and viscosity solutions.
- Module 4.5: Merton's portfolio problem.
- Module 4.6: LQG control and the Kalman filter.
- Module 4.7: Reinforcement learning and ADP.

Together these form a complete treatment of stochastic control theory for finance: from the clean theorem-driven DP framework to the messy-data-driven RL world.

**Subject 5 (Asset Pricing)** will build on these: the stochastic discount factor, the no-arbitrage price as a solution to a (stochastic) HJB, the fundamental theorems of asset pricing (FTAP) in full generality.

---

## Exercises

### Tier 1 (★)

1. Implement Q-learning on the gambler's ruin problem (Module 4.3) and verify convergence to the closed-form solution.
2. Compare TD(0) vs. MC policy evaluation on a random walk: which has lower MSE at $N = 100$ episodes?
3. Derive the policy-gradient theorem for a simple bandit problem (one-state MDP).
4. Show that REINFORCE is unbiased but has high variance; verify empirically on a 2-arm bandit.
5. Compute the optimal baseline for REINFORCE in closed form.

### Tier 2 (★★)

6. Prove that the Bellman optimality operator $T^*$ is a $\gamma$-contraction in sup norm.
7. Derive the natural policy gradient using the Fisher information matrix.
8. Prove Watkins-Dayan's Q-learning convergence using the ODE method.
9. Derive the advantage-decomposition: show $\text{Adv}^\pi(x, a) = \mathbb{E}[\delta_t | x_t = x, a_t = a]$.
10. Show Baird's counterexample diverges under linear Q-learning; explain why.

### Tier 3 (★★★)

11. Implement a **deep hedging** algorithm for a European call under GBM with 0.1% transaction costs. Compare P&L variance against Black-Scholes delta.
12. Implement **DDPG** on a simulated Almgren-Chriss optimal execution problem with non-linear impact; compare with model-based Almgren-Chriss.
13. Derive and implement **continuous-time Q-learning** following Jia-Zhou (2022) for a simple LQG problem; verify it recovers the analytical LQG solution.
14. Implement **SAC** for a simulated market-maker environment with inventory risk. Compare with Avellaneda-Stoikov closed-form.
15. Prove convergence of TD(0) with linear function approximation for on-policy learning using two-timescale stochastic approximation.
16. Implement and analyze **Double Q-learning** (Hasselt 2010), showing it reduces overestimation bias on a 2-state stochastic MDP.
17. Design an RL-based dynamic portfolio strategy that adapts to regime changes, using a hidden Markov model as environment. Compare with a model-based Bayesian regime filter.
18. **Inverse RL:** given observed trader actions and rewards, recover the trader's utility function via maximum-entropy IRL.
19. Implement **PPO** on a simulated rates-market-making environment with correlated assets. Measure sample efficiency vs. actor-critic.
20. Derive the **entropy-regularized HJB** for continuous-time max-entropy RL and show it generalizes SAC.
21. Prove **safety guarantees** for a CMDP (constrained MDP) using Lagrangian RL; implement on a portfolio problem with drawdown constraint.
22. **Multi-agent market making:** model two market makers as MARL agents; study emergent behavior in their equilibrium strategies.

---

## Subject 4 Capstone Summary

You now have in your toolkit:

- **Optimal stopping** (Snell envelope, American options): decide when to exercise or terminate.
- **HJB / dynamic programming**: solve full control problems when you can write down an SDE and a reward.
- **Verification theorems**: prove your candidate value function and control are indeed optimal.
- **Viscosity solutions**: extend to non-smooth value functions (essential in finance).
- **Numerical PDE methods**: binomial trees, PSOR, Kushner-Dupuis Markov chain approx, deep BSDE.
- **Closed-form Merton ratio**: the touchstone of portfolio theory.
- **LQR / LQG**: the linear-quadratic-Gaussian special case that solves itself.
- **Kalman-Bucy filter**: optimal linear state estimation, the "other Riccati."
- **Separation principle**: decouple estimation and control.
- **Reinforcement learning**: model-free and sample-based solutions for everything above.

Together with **Subject 3** (stochastic processes, martingales, PDE-SDE duality), this gives the complete toolkit for modeling, controlling, and estimating dynamic financial systems.

**Next: Subject 5 on Asset Pricing** will use these tools to derive the fundamental theorems, stochastic discount factors, and the modern pricing theory of derivatives.

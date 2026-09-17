# Subject 8, Module 5: Bayesian Methods and MCMC for Finance

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street*

> *"A Bayesian is one who, vaguely expecting a horse, and catching a glimpse of a donkey, strongly concludes he has seen a mule."* — Stephen Senn

---

## 8.5.0 Where We Are

Bayesian methods offer a coherent framework for combining prior knowledge with data, quantifying uncertainty, and performing decision-theoretic optimization. Finance is awash in problems where data is limited (expected returns!) or non-stationary (stochastic volatility!) — exactly where Bayesian reasoning pays off.

### Prerequisites

- **Module 2.1, 2.7** (Probability, conditional expectation).
- **Module 3.1** (Brownian motion, for stochastic volatility particle filters).
- **Module 7.1** (linear regression baseline for Bayesian regression).
- **Module 7.5** (Volatility modeling, target for Bayesian SV).
- **Module 8.1** (Portfolio optimization, for Bayesian portfolio choice).

### Plan

1. Bayesian inference: priors, likelihoods, posteriors (§8.5.1).
2. Conjugate priors and closed-form updates (§8.5.2).
3. Metropolis-Hastings (§8.5.3).
4. Gibbs sampling (§8.5.4).
5. Hamiltonian Monte Carlo (§8.5.5).
6. Variational inference (§8.5.6).
7. Particle filters for state-space models (§8.5.7).
8. Bayesian portfolio optimization and model averaging (§8.5.8).
9. Python: core samplers and Bayesian regression (§8.5.9).
10. Applications (§8.5.10).
11. Exercises (§8.5.11).

---

## 8.5.1 Bayesian Inference

### Bayes' theorem

Given data $y$ and parameters $\theta$ with prior $p(\theta)$ and likelihood $p(y \mid \theta)$:
$$p(\theta \mid y) = \frac{p(y \mid \theta) p(\theta)}{p(y)}, \quad p(y) = \int p(y \mid \theta) p(\theta) d\theta.$$

### Posterior predictive

$$p(\tilde y \mid y) = \int p(\tilde y \mid \theta) p(\theta \mid y) d\theta,$$
integrating over parameter uncertainty. Fundamental in finance for return prediction, where point estimates understate true uncertainty.

### Decision theory

Under a loss $L(\theta, a)$, the Bayes-optimal action minimizes expected posterior loss:
$$a^* = \arg\min_a \int L(\theta, a) p(\theta \mid y) d\theta.$$
For squared loss this is the posterior mean; for 0/1 loss the posterior mode; for absolute the posterior median.

### Model evidence and Bayes factors

$$p(y) = \int p(y \mid \theta) p(\theta) d\theta$$
is the model evidence. To compare models $M_1, M_2$:
$$\text{BF}_{12} = \frac{p(y \mid M_1)}{p(y \mid M_2)}.$$
Jeffreys' scale: BF > 3 is substantial, > 10 strong, > 100 decisive.

---

## 8.5.2 Conjugate Priors

### Normal with known variance

Prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$; data $y_i \sim \mathcal{N}(\theta, \sigma^2)$, $n$ observations. Posterior:
$$\theta \mid y \sim \mathcal{N}\left(\mu_n, \sigma_n^2\right), \quad \mu_n = \frac{\mu_0/\sigma_0^2 + n\bar y/\sigma^2}{1/\sigma_0^2 + n/\sigma^2}, \quad \frac{1}{\sigma_n^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}.$$
Posterior precision = prior precision + data precision.

### Normal-inverse-gamma for $(\mu, \sigma^2)$

$$\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0), \quad \mu \mid \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2/\kappa_0).$$
Posterior:
$$\alpha_n = \alpha_0 + n/2, \quad \beta_n = \beta_0 + \tfrac{1}{2}\sum_i (y_i - \bar y)^2 + \frac{\kappa_0 n}{\kappa_0 + n}\cdot \frac{(\bar y - \mu_0)^2}{2}, \quad \kappa_n = \kappa_0 + n, \quad \mu_n = \frac{\kappa_0 \mu_0 + n \bar y}{\kappa_0 + n}.$$

### Dirichlet-multinomial

For categorical data with $K$ classes and prior $\theta \sim \text{Dirichlet}(\alpha)$, posterior is $\text{Dirichlet}(\alpha + c)$ where $c$ is the count vector.

### Bayesian linear regression

$y = X\beta + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, prior $\beta \sim \mathcal{N}(\beta_0, V_0)$ with known $\sigma^2$:
$$\beta \mid y \sim \mathcal{N}(\beta_n, V_n), \quad V_n^{-1} = V_0^{-1} + X^\top X/\sigma^2, \quad \beta_n = V_n(V_0^{-1}\beta_0 + X^\top y/\sigma^2).$$

The ridge estimator is the posterior mean under a Gaussian prior centered at zero.

---

## 8.5.3 Metropolis-Hastings

For posteriors without closed form, MCMC provides samples. MH constructs a Markov chain with stationary distribution $\pi(\theta) \propto p(\theta \mid y)$:

**Algorithm.** Given current $\theta_t$:
1. Propose $\theta^* \sim q(\cdot \mid \theta_t)$.
2. Compute acceptance ratio
$$\alpha = \min\left(1, \frac{\pi(\theta^*) q(\theta_t \mid \theta^*)}{\pi(\theta_t) q(\theta^* \mid \theta_t)}\right).$$
3. Accept $\theta_{t+1} = \theta^*$ with probability $\alpha$; else $\theta_{t+1} = \theta_t$.

### Theorem 8.5.1 (MH detailed balance).

*The transition kernel of MH satisfies $\pi(\theta) P(\theta, \theta') = \pi(\theta') P(\theta', \theta)$, hence $\pi$ is the stationary distribution of the chain.*

**Proof.** For $\theta \neq \theta'$:
$$\pi(\theta) P(\theta, \theta') = \pi(\theta) q(\theta' \mid \theta) \min\left(1, \frac{\pi(\theta') q(\theta \mid \theta')}{\pi(\theta) q(\theta' \mid \theta)}\right) = \min(\pi(\theta) q(\theta' \mid \theta), \pi(\theta') q(\theta \mid \theta')).$$
By symmetry this equals $\pi(\theta')P(\theta', \theta)$. $\square$

### Tuning proposals

Acceptance rate heuristics:
- 1-D: aim for ~44% (Gelman-Roberts-Gilks 1997).
- Multi-D with random-walk proposals: optimal is ~23%.
- Adaptive MH: tune proposal covariance to sample covariance (Haario-Saksman-Tamminen 2001), with care to preserve ergodicity.

### Convergence diagnostics

- **Trace plots**: visual mixing.
- **Effective sample size (ESS)**: $n_{\text{eff}} = n / (1 + 2\sum_{k=1}^\infty \rho_k)$ with autocorrelations $\rho_k$.
- **$\hat R$ (Gelman-Rubin)**: variance across chains vs. within chains; target < 1.01.

---

## 8.5.4 Gibbs Sampling

When the joint posterior factorizes into tractable full conditionals, Gibbs sampling cycles through:
$$\theta^{(t+1)}_k \sim p(\theta_k \mid \theta_{-k}^{(t)}, y), \qquad k = 1, \dots, K.$$
Each step samples one coordinate given the rest; the accept rate is 1 (special case of MH).

### Bayesian linear regression via Gibbs

Joint posterior of $(\beta, \sigma^2)$:
- $\beta \mid \sigma^2, y \sim \mathcal{N}(\beta_n, \sigma^2 V_n)$,
- $\sigma^2 \mid \beta, y \sim \text{Inv-Gamma}(\alpha_n, \beta_n(\beta))$.
Alternate samples; burn-in and collect.

### Stochastic volatility (Jacquier-Polson-Rossi 1994)

Latent log-volatilities $h_t$ with $r_t = \exp(h_t/2)\varepsilon_t$ and $h_{t+1} = \mu + \phi(h_t - \mu) + \eta_{t+1}$. Gibbs updates:
1. $\mu, \phi, \sigma_\eta$ given $\{h_t\}$: conjugate for normal-inverse-gamma.
2. $\{h_t\}$ given $\mu, \phi, \sigma_\eta, \{r_t\}$: mixture-of-normals augmentation (Kim-Shephard-Chib 1998) provides a Gaussian conditional.

---

## 8.5.5 Hamiltonian Monte Carlo

HMC augments $\theta \in \mathbb{R}^D$ with a momentum $p \in \mathbb{R}^D$ and simulates Hamiltonian dynamics on
$$H(\theta, p) = -\log\pi(\theta) + \tfrac{1}{2} p^\top M^{-1} p.$$
Leapfrog discretization with step size $\epsilon$ for $L$ steps yields a proposal $(\theta^*, p^*)$; accept with probability
$$\alpha = \min(1, \exp(H(\theta_t, p_t) - H(\theta^*, p^*))).$$
Discretization error would break detailed balance without the accept/reject step; with it, HMC produces exact samples from the augmented distribution, and marginalizing over $p$ gives $\pi$.

### Why HMC?

HMC exploits gradient information to propose large, high-acceptance moves, achieving effective sample sizes orders of magnitude larger than random-walk MH in high dimensions. **NUTS** (No-U-Turn Sampler, Hoffman-Gelman 2014) auto-tunes $L$; it is the core sampler in Stan, PyMC, and NumPyro.

### Typical applications

- Hierarchical Bayesian models (multi-level regressions).
- State-space and structural models with many latent states.
- Non-conjugate priors (e.g., student-$t$ errors).

---

## 8.5.6 Variational Inference

### The variational lower bound (ELBO)

Approximate the posterior $p(\theta \mid y)$ by a tractable family $q_\phi(\theta)$. Maximize
$$\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}[\log p(y, \theta) - \log q_\phi(\theta)] = \log p(y) - \mathrm{KL}(q_\phi \| p(\cdot \mid y)).$$
Maximizing ELBO minimizes KL-divergence from $q$ to the posterior.

### Mean-field VI

Factorize $q(\theta) = \prod_i q_i(\theta_i)$ and solve coordinate-wise:
$$q_i^*(\theta_i) \propto \exp\{\mathbb{E}_{-i}[\log p(y, \theta)]\}.$$

### Stochastic VI with reparameterization

When $q_\phi$ is Gaussian ($\mu_\phi, \Sigma_\phi$), write $\theta = \mu_\phi + \Sigma_\phi^{1/2} \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. Gradients of the ELBO are
$$\nabla_\phi \mathrm{ELBO} = \mathbb{E}_\epsilon[\nabla_\phi \log p(y, \theta(\epsilon, \phi)) - \nabla_\phi \log q_\phi(\theta)].$$
Monte Carlo approximation + stochastic gradient ascent scales VI to massive datasets (Kingma–Welling 2014 in the VAE; Hoffman et al. 2013 for probabilistic models).

### Trade-offs vs. MCMC

VI is faster and scales better, but tends to *underestimate* posterior variance due to mode-seeking. For decision-theoretic quant applications where uncertainty matters, MCMC is often preferred; VI is used for initial exploration and massive-scale learning.

---

## 8.5.7 Particle Filters for State-Space Models

### Setup

$$X_{t+1} = f(X_t, \epsilon_t), \quad Y_t = g(X_t) + \eta_t,$$
with $X_t$ latent state. The filtering distribution $p(X_t \mid Y_{1:t})$ is generally intractable beyond linear-Gaussian cases.

### Bootstrap particle filter

1. **Initialize** particles $\{x_0^{(i)}\}_{i=1}^N$ from prior.
2. **Propagate**: $\tilde x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)})$.
3. **Weight**: $w_t^{(i)} \propto p(y_t \mid \tilde x_t^{(i)})$.
4. **Resample**: draw $x_t^{(i)}$ with probabilities $w_t^{(i)}$.
5. Continue.

### Theorem 8.5.2 (Particle-filter consistency).

*As $N \to \infty$, the empirical measure $\hat p_N(x_t \mid y_{1:t}) = N^{-1}\sum_i \delta_{x_t^{(i)}}$ converges weakly to $p(x_t \mid y_{1:t})$.*

Del Moral-Doucet-Jasra (2012) developed SMC samplers extending PF ideas to generic sequential inference.

### Stochastic volatility particle filter

$y_t = \exp(h_t/2)\varepsilon_t$, $h_{t+1} = \mu + \phi(h_t - \mu) + \sigma_\eta \eta_{t+1}$. Bootstrap PF with $N = 1000$ particles provides real-time volatility filtering; APF (Pitt-Shephard 1999) uses look-ahead proposals for better performance.

### Financial applications

- Real-time volatility estimation (stochastic volatility models).
- Jump detection (latent jump intensity).
- Regime filtering in Markov-switching models.
- Credit-default intensity tracking.

---

## 8.5.8 Bayesian Portfolio Optimization

### Parameter uncertainty in MVO

Pástor (2000): incorporate parameter uncertainty by integrating the utility over the posterior:
$$w^* = \arg\max_w \int U(w; \mu, \Sigma) p(\mu, \Sigma \mid y) \, d\mu d\Sigma.$$
For CRRA utility and Gaussian returns with a normal-inverse-Wishart posterior, this has a semi-analytic solution that shrinks toward the prior mean in proportion to parameter uncertainty.

### Black-Litterman (Module 8.1) is Bayesian

Equilibrium $\pi$ as prior mean, subjective views $(P, q, \Omega)$ as additional data, posterior as the BL formula — literally Bayes' rule for the vector $\mu$.

### Bayesian model averaging

For multiple factor models $M_k$ with posterior weights $p(M_k \mid y)$:
$$\bar \alpha_i = \sum_k p(M_k \mid y) \cdot \hat\alpha_i^{(k)}.$$
Marginal likelihoods $p(y \mid M_k)$ are computed via bridge sampling, thermodynamic integration, or Chib's method.

### Decision-theoretic sizing

For a position sizing decision $w_i$ with a convex loss $L(w, \mu)$, the Bayes-optimal policy solves
$$\min_w \int L(w, \mu) p(\mu \mid y) d\mu.$$
Under mean-variance + parameter uncertainty, the solution is shrunk relative to the plug-in optimum, precisely because uncertainty penalizes concentrated bets.

---

## 8.5.9 Python Implementations

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ----- Random-walk Metropolis-Hastings -----
def rw_mh(log_target, theta0, n_iter=10000, step=0.1, burn=1000):
    d = len(theta0)
    theta = theta0.copy()
    lp = log_target(theta)
    samples = np.zeros((n_iter, d))
    n_accept = 0
    for i in range(n_iter):
        proposal = theta + step * np.random.standard_normal(d)
        lp_new = log_target(proposal)
        if np.log(np.random.rand()) < lp_new - lp:
            theta = proposal
            lp = lp_new
            n_accept += 1
        samples[i] = theta
    return samples[burn:], n_accept/n_iter

# Example: Bayesian mean of Gaussian data
def example_mh():
    y = rng.standard_normal(100) + 2.0
    def log_post(theta):
        mu = theta[0]; sigma = np.exp(theta[1])
        # Prior: mu ~ N(0,10), log sigma ~ N(0,1)
        lp = -0.5*mu**2/10 - 0.5*theta[1]**2
        ll = stats.norm.logpdf(y, mu, sigma).sum()
        return lp + ll
    samples, ar = rw_mh(log_post, np.array([0.0, 0.0]), n_iter=20000, step=0.1)
    print(f"Accept rate: {ar:.2%}")
    print(f"Post mean mu: {samples[:,0].mean():.3f}, sigma: {np.exp(samples[:,1].mean()):.3f}")

# ----- Gibbs for Bayesian regression -----
def bayes_reg_gibbs(X, y, n_iter=5000, burn=1000, a0=0.01, b0=0.01, V0_scale=100.0):
    n, p = X.shape
    V0 = V0_scale * np.eye(p)
    V0_inv = np.linalg.inv(V0)
    samples = np.zeros((n_iter, p+1))
    sigma2 = 1.0
    beta = np.zeros(p)
    for i in range(n_iter):
        # Sample beta | sigma2, y
        Vn_inv = V0_inv + X.T @ X / sigma2
        Vn = np.linalg.inv(Vn_inv)
        betan = Vn @ (X.T @ y / sigma2)
        beta = np.random.multivariate_normal(betan, Vn)
        # Sample sigma2 | beta, y
        resid = y - X @ beta
        an = a0 + n/2
        bn = b0 + 0.5 * resid @ resid
        sigma2 = 1.0 / np.random.gamma(an, 1/bn)
        samples[i] = np.concatenate([beta, [sigma2]])
    return samples[burn:]

# ----- Bootstrap particle filter for SV -----
def sv_bootstrap_pf(y, mu, phi, sigma_eta, N=1000):
    T = len(y)
    particles = np.random.normal(mu, sigma_eta/np.sqrt(1-phi**2), N)
    h_filt = np.zeros(T)
    for t in range(T):
        particles = mu + phi*(particles - mu) + sigma_eta*np.random.standard_normal(N)
        logw = stats.norm.logpdf(y[t], 0, np.exp(particles/2))
        logw -= logw.max()
        w = np.exp(logw); w /= w.sum()
        h_filt[t] = (w * particles).sum()
        # Resample
        idx = np.random.choice(N, size=N, p=w)
        particles = particles[idx]
    return h_filt

# ----- HMC (single leapfrog step for demo; NUTS requires more) -----
def hmc_step(theta, log_target, grad_log_target, eps=0.1, L=10):
    d = len(theta)
    p = np.random.standard_normal(d)
    p_new = p - 0.5*eps*(-grad_log_target(theta))
    theta_new = theta + eps*p_new
    for _ in range(L-1):
        p_new = p_new - eps*(-grad_log_target(theta_new))
        theta_new = theta_new + eps*p_new
    p_new = p_new - 0.5*eps*(-grad_log_target(theta_new))
    H_old = -log_target(theta) + 0.5*p @ p
    H_new = -log_target(theta_new) + 0.5*p_new @ p_new
    if np.log(np.random.rand()) < H_old - H_new:
        return theta_new, True
    return theta, False

# ----- Mean-field VI for Gaussian likelihood, Gaussian prior -----
def mean_field_vi(y, prior_mu=0, prior_sigma=10, n_iter=100):
    # q(mu) = N(mq, sq)
    # q(sigma^2) = InvGamma(aq, bq)
    n = len(y)
    a0, b0 = 0.01, 0.01
    mq, aq, bq = y.mean(), a0 + n/2, b0 + 0.5*y.var()*n
    sq = 1.0 / (1.0/prior_sigma**2 + n*aq/bq)
    for _ in range(n_iter):
        mq = sq*(prior_mu/prior_sigma**2 + (y.sum())*aq/bq)
        bq = b0 + 0.5*((y - mq)**2 + sq).sum()
        sq = 1.0 / (1.0/prior_sigma**2 + n*aq/bq)
    return mq, sq, aq, bq

if __name__ == "__main__":
    example_mh()
```

### Running diagnostics

For MCMC in practice, compute:
- **Trace plot** of each parameter.
- **ESS** per parameter; aim for > 1000 for reliable summary stats.
- **$\hat R$** across 4+ chains; < 1.01.
- **Autocorrelation plots**.
- **Posterior predictive checks**: simulate from the posterior and compare to data.

---

## 8.5.10 Applications

1. **Bayesian MVO**: shrinkage portfolios with explicit posterior uncertainty on $\mu, \Sigma$.
2. **Black-Litterman** across asset classes with analyst-derived views.
3. **Bayesian stochastic volatility**: filtered volatility path for option pricing and risk.
4. **Bayesian ARCH/GARCH**: priors for parameter identification on small samples.
5. **Credit risk**: Bayesian hierarchical default models at issuer-sector-rating levels.
6. **Regime detection**: Markov-switching GARCH with MCMC for transition probabilities.
7. **Bayesian hierarchical factor models**: shared priors across subgroups of assets.
8. **Bayesian state-space yield curve**: Nelson-Siegel factors with Kalman + MCMC.
9. **Term-structure Bayesian filtering**: Hull-White calibrated with MCMC.
10. **Bayesian model averaging** for forecasts (inflation, GDP, vol).
11. **Bayesian A/B testing**: continuous monitoring with anytime-valid posterior CIs.
12. **Bayesian decision theory** for optimal execution sizing under cost uncertainty.

---

## 8.5.11 Exercises

### ★

1. Derive the posterior $\theta \mid y$ for a Gaussian with known $\sigma^2$ and Gaussian prior.
2. Show that the Dirichlet is conjugate to the multinomial.
3. Compute the MH acceptance probability for a symmetric random-walk proposal.
4. Explain why Gibbs sampling is a special case of MH with acceptance probability 1.
5. For a Bayesian linear regression with prior $\beta \sim \mathcal{N}(0, (\lambda)^{-1}I)$, derive the ridge interpretation.
6. State Bayes' factor and compute BF for a simple normal-mean-shift model with two candidate means.

### ★★

7. Implement MH for a bivariate banana-shaped density and diagnose mixing via autocorrelation.
8. Implement Gibbs for Bayesian linear regression with normal-inverse-gamma prior; verify posterior predictive coverage.
9. Derive the Kim-Shephard-Chib (1998) mixture-of-normals approximation for log-chi-squared and implement an SV Gibbs sampler.
10. Implement the leapfrog integrator for HMC and show energy conservation for small step size; show how step size affects accept rate.
11. Derive the ELBO for a Gaussian variational family and implement stochastic VI with the reparameterization trick.
12. Implement a bootstrap particle filter for an Ornstein-Uhlenbeck state with Gaussian observation noise and compare filtered mean to the Kalman filter.

### ★★★

13. Prove the ergodic theorem for MH: time-averaged samples converge to posterior expectations a.s. under standard regularity.
14. Derive the optimal acceptance rate of 0.234 for random-walk MH in high dimensions (Roberts-Gelman-Gilks 1997).
15. Prove convergence of Gibbs sampling for a jointly continuous posterior under Harris recurrence.
16. Derive the consistency of the bootstrap particle filter as the number of particles goes to infinity (Chopin 2004).
17. Derive the reparameterized gradient of the ELBO and prove it is an unbiased estimator.
18. Formulate and prove the Bayesian optimization rate for sequential decision problems under the full posterior, relating it to the Bayesian regret.

---

*— End of Module 8.5. Next: Module 8.6, High-Frequency Trading and Market Making.*

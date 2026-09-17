# Module 6.5 — Fourier and Transform Methods

*Subject 6, Module 5. When you know the characteristic function but not the density: Carr-Madan, COS, Lewis, and beyond.*

---

## Prerequisites

- **Module 2.5 (Characteristic functions)** — $\phi(u) = \mathbb{E}[e^{iuX}]$ and its inversion.
- **Module 0.5.6 (Fourier transforms)** — Fourier analysis in $L^2$ and $L^1$.
- **Module 5.6 (Stochastic volatility)** — Heston's char function is the primary application.
- **Module 5.2 (Black-Scholes)** — analytic benchmark.

---

## 6.5.1 Why Fourier?

Affine stochastic volatility models (Heston, Bates, CGMY, NIG, ...) have **closed-form characteristic functions** $\phi(u; T)$ but not closed-form densities. Monte Carlo is slow; PDEs are 2D and expensive. **Fourier methods exploit the char function directly** to price options in $O(N \log N)$ or $O(N)$ operations.

**Heston** (Module 5.6, equation 5.6.7):
$$
\phi(u; T) = \exp\left(A(u, T) + B(u, T) V_0 + iu\ln S_0\right),
$$
with $A, B$ given explicit Riccati solutions. Given $\phi$, how do we recover option prices?

Three main techniques:
1. **Carr-Madan FFT** (1999): damped call price transform.
2. **Lewis-Lipton** (2001): contour-integral representation.
3. **Fang-Oosterlee COS** (2008): Fourier cosine series expansion of density.

---

## 6.5.2 The dampened call transform (Carr-Madan 1999)

**Problem.** The European call price $C(k)$ as a function of log-strike $k = \ln K$ is not integrable: $C(k) \to S_0$ as $k \to -\infty$ (deep ITM, unbounded). Standard FFT requires $L^1$.

**Fix: damp by $e^{\alpha k}$ with $\alpha > 0$.** Define
$$
c_T(k) = e^{\alpha k} C(k).
$$
For $\alpha > 0$ large enough, $c_T \in L^1$.

**Fourier transform**: $\psi_T(v) = \int_{-\infty}^{\infty} e^{ivk} c_T(k) dk$.

**Derivation of $\psi_T$ in terms of $\phi_T$.** Using $C(k) = e^{-rT} \mathbb{E}^\mathbb{Q}[(S_T - e^k)_+]$ and Fubini,
$$
\psi_T(v) = e^{-rT}\int_{-\infty}^{\infty} e^{(\alpha+iv)k} \mathbb{E}[(e^{s} - e^k)_+ \mid S_0] dk
$$
where $s = \ln S_T$. After swapping integration order and computing the inner integral:
$$
\boxed{\psi_T(v) = \frac{e^{-rT} \phi_T(v - i(\alpha+1))}{\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}.}
$$

**Inversion.** By Fourier inversion theorem:
$$
C(k) = \frac{e^{-\alpha k}}{\pi}\int_0^\infty e^{-ivk} \psi_T(v) dv.
$$

**Discretization → FFT.** Truncate at $v_{max}$, sample at $v_j = j\eta$ for $j = 0, \ldots, N-1$, and note that if we choose $k_m = -N\eta/2 + m\Delta k$ with $\Delta k = 2\pi/(N\eta)$, then:
$$
C(k_m) \approx \frac{e^{-\alpha k_m}}{\pi} \cdot \eta \cdot \sum_{j=0}^{N-1} e^{-2\pi i j m / N} e^{i v_j N\eta/2} \psi_T(v_j).
$$
The sum is a DFT — computable via FFT in $O(N \log N)$.

**Typical parameters.** $N = 2^{12}$, $\eta = 0.25$, $\alpha = 1.5$.

**Choice of $\alpha$.** Need $\mathbb{E}[S_T^{\alpha+1}] < \infty$ for $\psi_T$ to exist. For Heston, this imposes a condition on $\alpha$ related to the moments of the Heston density. Practical rule: try $\alpha = 0.75$ or $1.5$ and verify by comparison to MC.

---

## 6.5.3 Lewis-Lipton integral representation (2001)

**Idea.** Use the Plancherel/Parseval identity and contour integration to avoid the damping parameter.

**Theorem (Lewis 2001).** For a European call with strike $K$:
$$
C(S_0, K, T) = S_0 - \frac{\sqrt{S_0 K} e^{-rT/2}}{\pi}\int_0^\infty \Re\left[e^{ivk}\frac{\phi_T(v - i/2)}{v^2 + 1/4}\right] dv,
$$
where $k = \ln(K/S_0) - rT$.

**Advantages.**
- No damping parameter $\alpha$.
- Integral is smooth, well-behaved. Adaptive quadrature (scipy `quad`) works fine.
- Handles all moneyness.

**Disadvantage**: $O(N)$ for each strike — slower than FFT when many strikes needed. Use for calibration with sparse strikes.

**Connection to Carr-Madan.** Lewis-Lipton is essentially the special case $\alpha = 1/2$ with the contour shifted. The integrand is symmetric around the ATM, making convergence uniform.

---

## 6.5.4 The COS method (Fang-Oosterlee 2008)

**Idea.** Expand the density as a **Fourier cosine series** on a truncation interval $[a, b]$:
$$
f(x) \approx \frac{2}{b-a} \sum_{n=0}^{N-1}{}' A_n \cos\left(n\pi\frac{x-a}{b-a}\right), \quad A_n = \int_a^b f(x)\cos\left(n\pi\frac{x-a}{b-a}\right) dx,
$$
where $\sum'$ means the $n=0$ term gets weight 1/2.

**Key identity.** The cosine-series coefficient is (up to truncation error)
$$
A_n \approx \Re\left[\phi_T\left(\frac{n\pi}{b-a}\right) e^{-i n\pi a/(b-a)}\right].
$$
This uses the Fourier transform on $[a, b]$ and identifies it with the characteristic function when $[a, b]$ is chosen to cover the support of $f$ essentially.

**Option pricing formula.**
$$
C(S_0, K, T) = e^{-rT} \sum_{n=0}^{N-1}{}' A_n \cdot V_n,
$$
where $V_n = \int_a^b g(x)\cos(n\pi (x-a)/(b-a)) dx$ are **payoff coefficients** that for a European call have explicit closed form:
$$
V_n = \frac{2}{b-a}K[\chi_n(0, b) - \psi_n(0, b)],
$$
with
$$
\chi_n(c, d) = \frac{1}{1 + (n\pi/(b-a))^2}\left[\cos(n\pi(d-a)/(b-a))e^d - \cos(n\pi(c-a)/(b-a))e^c\right.
$$
$$
\left. + (n\pi/(b-a))(\sin(n\pi(d-a)/(b-a))e^d - \sin(n\pi(c-a)/(b-a))e^c)\right],
$$
$$
\psi_n(c, d) = \frac{b-a}{n\pi}\left[\sin(n\pi(d-a)/(b-a)) - \sin(n\pi(c-a)/(b-a))\right] \text{ (for } n \ge 1\text{)}.
$$

**Advantages.**
- Exponential convergence in $N$ for smooth densities: error $\sim e^{-cN}$.
- Deals naturally with all payoffs; closed-form $V_n$ for many exotics.
- No damping parameter.

**Truncation interval.** A good heuristic:
$$
a = c_1 - L\sqrt{c_2}, \qquad b = c_1 + L\sqrt{c_2}
$$
where $c_1, c_2$ are first two cumulants of $X_T = \ln S_T$ and $L = 10$ (or higher for heavier tails).

**COS is dominant for calibration** — best convergence rate in the standard setting.

---

## 6.5.5 Alternatives: CONV, BENCHOP, saddle-point

**CONV method (Lord-Fang-Bervoets-Oosterlee 2008).** Combines FFT with Heston-style convolution for high-order quadrature. Produces spectral convergence but is more complex than COS.

**SWIFT (Ortiz-Gracia-Oosterlee 2013)**: wavelet-based alternative to COS, with exponential convergence and better handling of discontinuous payoffs.

**Saddle-point methods.** For deep OTM or deep ITM options, replace the integral by a Laplace approximation around the saddle point of the integrand. Useful when the characteristic function is complex to invert at extreme strikes.

**Fractional FFT.** When strikes are logarithmically spaced (as in ATM-centered strikes), use fractional FFT (FrFFT) to evaluate at arbitrary strike grids without re-sampling.

---

## 6.5.6 Multi-asset / multi-dim Fourier

For basket options on $d$ assets with joint characteristic function $\Phi(\mathbf{u}; T)$:
$$
C(\mathbf{K}, T) = e^{-rT} \mathbb{E}[\max(\mathbf{w}^\top \mathbf{S}_T - K, 0)].
$$

**Hurd-Zhou (2010)**: multi-dim FFT with damping $\boldsymbol\alpha$. Cost $O(N^d \log N)$ — feasible for $d \le 3$.

**Leentvaar-Oosterlee (2008)**: COS extension to multi-dim with exponential convergence but tensor-structured computation.

**Truncation via spectral decomposition.** For $d = 20$ baskets, reduce to effective $d_T = 2$–$3$ using PCA on covariance, then apply 2D or 3D FFT / COS.

---

## 6.5.7 Implied volatility from Fourier prices

Once we have call prices $C(K, T)$ at a strip of strikes, implied volatility is computed via Brent root-finding or Jäckel's rational-function approximation (2015). Standard part of a calibration pipeline.

**Greeks via Fourier differentiation.** Differentiate $\phi$ w.r.t. parameters under the integral sign. Delta is free (the pricing integral already contains $S_0$). Vega requires differentiating $\phi$ w.r.t. $\sigma$ (or $\xi, V_0$, etc. in Heston).

---

## 6.5.8 Python: Fourier pricers

```python
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from numpy.fft import fft, ifft

# ============================================================
# 1. Black-Scholes characteristic function (sanity test)
# ============================================================
def bs_cf(u, S0, r, sigma, T):
    return np.exp(1j*u*(np.log(S0) + (r-0.5*sigma**2)*T) - 0.5*sigma**2*u**2*T)

# ============================================================
# 2. Carr-Madan FFT pricer
# ============================================================
def carr_madan_call(cf, S0, r, T, N=2**12, eta=0.25, alpha=1.5):
    """cf: characteristic function of log(S_T/S_0), returning complex values."""
    lam = 2*np.pi/(N*eta)
    b = N*lam/2
    # ku grid
    ku = -b + lam*np.arange(N)
    # vj grid
    vj = eta*np.arange(N)

    # Integrand: ψ_T(v) as per Carr-Madan
    # Using the full char function of ln S_T (not ln S_T/S_0)
    def psi(v):
        return np.exp(-r*T)*cf(v - 1j*(alpha+1), S0, r, T) \
               /(alpha**2 + alpha - v**2 + 1j*(2*alpha+1)*v)
    x_j = np.exp(1j*b*vj)*psi(vj)*eta
    # Apply Simpson's 1/3 weights for accuracy
    w = np.ones(N); w[1:-1:2] = 4; w[2:-1:2] = 2; w = w/3
    x_j_s = x_j*w

    y = fft(x_j_s)
    call = np.exp(-alpha*ku)/np.pi * np.real(y)
    K = np.exp(ku)
    return K, call

# BS test
K_grid, C_grid = carr_madan_call(bs_cf, 100, 0.05, 1.0)
# Compare at K=100
idx = np.argmin(np.abs(K_grid - 100))
def bs_call_closed(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
print(f"Carr-Madan BS call at K={K_grid[idx]:.1f}: {C_grid[idx]:.4f}")
print(f"Closed-form:                              {bs_call_closed(100,100,0.05,0.2,1.0):.4f}")

# ============================================================
# 3. Lewis-Lipton integral pricer
# ============================================================
def lewis_lipton_call(cf, S0, K, r, T, v_max=100):
    k = np.log(K/S0) - r*T
    def integrand(v):
        return np.real(np.exp(1j*v*k)*cf(v - 1j/2, S0, r, T) / (v**2 + 0.25))
    integral, _ = quad(integrand, 0, v_max)
    # Adjusting for S0 normalization
    return S0 - np.sqrt(S0*K)*np.exp(-r*T/2)/np.pi * integral

# NB: for BS need to use ψ(v) = E[e^(iv*ln(S_T/S_0))] not absolute ln(S_T)
def bs_cf_relative(u, S0, r, sigma, T):
    return np.exp(1j*u*(r-0.5*sigma**2)*T - 0.5*sigma**2*u**2*T)

# Simpler Lewis-Lipton using relative log returns (cleaner convention)
def lewis_lipton_call_v2(cf_rel, S0, K, r, T, v_max=100):
    """cf_rel: char fn of ln(S_T/S_0) (no forward shift)."""
    k = np.log(K/S0)
    def integrand(v):
        return np.real(np.exp(-1j*v*k)*cf_rel(v - 1j/2, S0, r, 0.2, T) / (v**2 + 0.25))
    integral, _ = quad(integrand, 0, v_max, limit=200)
    return S0 - np.sqrt(S0*K)*np.exp(-r*T/2)/np.pi * integral

ll_price = lewis_lipton_call_v2(bs_cf_relative, 100, 100, 0.05, 1.0)
print(f"Lewis-Lipton BS call:      {ll_price:.4f}")

# ============================================================
# 4. COS method for European call in BS
# ============================================================
def cos_method_call(cf_rel, S0, K, r, T, N=128, L=10):
    # Cumulants for truncation interval (BS: c1 = (r-σ²/2)T, c2 = σ²T)
    # For general model, compute numerically or use model-specific formulas.
    # Here use an illustrative fixed range.
    x0 = np.log(S0/K)
    # Use BS cumulants as approximation
    c1 = (r - 0.5*0.2**2)*T + x0
    c2 = 0.2**2 * T
    a = c1 - L*np.sqrt(c2)
    b = c1 + L*np.sqrt(c2)

    def V_call(n):
        # Payoff coefficients for call: K*(e^y - 1)+, y = x - ln(K/S0) mapped
        # Using original formulation: x = ln(S_T)
        # V_n = 2/(b-a) ∫_0^b (e^x - K) cos(nπ(x-a)/(b-a)) dx, with K shifted
        # Simplified (strike-adjusted call coefficient):
        if n == 0: 
            return (np.exp(b) - np.exp(0))  # Rough placeholder
        k_ = n*np.pi/(b-a)
        def X_(c, d):
            return (1/(1+k_**2))*(np.cos(k_*(d-a))*np.exp(d) - np.cos(k_*(c-a))*np.exp(c)
                 + k_*np.sin(k_*(d-a))*np.exp(d) - k_*np.sin(k_*(c-a))*np.exp(c))
        def Psi_(c, d):
            return (np.sin(k_*(d-a)) - np.sin(k_*(c-a)))/k_
        return 2/(b-a)*(X_(0, b) - Psi_(0, b))  # roughly the call payoff coefs

    # For brevity, full COS implementation omitted — see Fang-Oosterlee paper
    # Compute coefficients
    u = np.arange(N)*np.pi/(b-a)
    phi = cf_rel(u, S0, r, 0.2, T)
    # Integration weights include discount and strike factor
    A = np.zeros(N)
    A[0] = 0.5*np.real(phi[0]*np.exp(-1j*u[0]*a))
    for n in range(1, N):
        A[n] = np.real(phi[n]*np.exp(-1j*u[n]*a))
    # Payoff coefficients (simplified, for illustration)
    Vn = np.zeros(N)
    # Use numerical integration for payoff coefficients
    from scipy.integrate import quad as Q
    def g(x): return K*(np.exp(x + x0) - 1.0) * (np.exp(x + x0) > 1)
    for n in range(N):
        f_int = lambda x, n=n: g(x)*np.cos(n*np.pi*(x-a)/(b-a))
        val, _ = Q(f_int, max(a, -x0), b, limit=100)
        Vn[n] = 2/(b-a)*val
    price = np.exp(-r*T)*np.sum(A*Vn*(np.concatenate([[0.5], np.ones(N-1)])))
    return price

# The implementation above is simplified; for a clean COS reference see the Fang-Oosterlee paper.
print(f"COS method implementation is simplified; skipping verification.")
print(f"For a production COS pricer see the official Fang-Oosterlee reference code.")

# ============================================================
# 5. Heston pricing via Carr-Madan
# ============================================================
def heston_cf(u, S0, V0, r, kappa, theta, xi, rho, T):
    d = np.sqrt((rho*xi*1j*u - kappa)**2 + xi**2*(1j*u + u**2))
    g = (kappa - rho*xi*1j*u - d)/(kappa - rho*xi*1j*u + d)
    C = r*1j*u*T + kappa*theta/xi**2 * ((kappa - rho*xi*1j*u - d)*T
        - 2*np.log((1 - g*np.exp(-d*T))/(1-g)))
    D = (kappa - rho*xi*1j*u - d)/xi**2 * (1 - np.exp(-d*T))/(1 - g*np.exp(-d*T))
    return np.exp(C + D*V0 + 1j*u*np.log(S0))

def carr_madan_heston(S0, V0, K, r, kappa, theta, xi, rho, T, alpha=1.5):
    def cf(u, S0_, r_, T_):
        return heston_cf(u, S0_, V0, r_, kappa, theta, xi, rho, T_)
    K_grid, C_grid = carr_madan_call(cf, S0, r, T, alpha=alpha)
    return np.interp(K, K_grid, C_grid)

# Heston parameters: typical calibration-grade
S0, V0, r = 100, 0.04, 0.05
kappa, theta, xi, rho, T = 2.0, 0.04, 0.3, -0.7, 1.0

for K in [80, 90, 100, 110, 120]:
    price = carr_madan_heston(S0, V0, K, r, kappa, theta, xi, rho, T)
    print(f"Heston call K={K}: {price:.4f}")
```

**What to check.** 
- Carr-Madan BS call at $K=100$ should match closed-form to $10^{-4}$ or better.
- Lewis-Lipton should give identical price (both are valid inversions of the same char function).
- Heston prices should show the characteristic smile (IVs extracted from prices).

---

## 6.5.9 [QUANT APPLICATIONS]

1. **Calibration of Heston to vanilla surfaces.** Each iteration requires 100–300 repriced options; COS or Carr-Madan makes this fast enough to calibrate in seconds.
2. **VIX pricing.** VIX formulation is Fourier-friendly; evaluated via characteristic function of $\int V_s ds$.
3. **Exotic pricing in Heston / Bates / CGMY.** Options with early exercise use Fourier-assisted LSM or use the tree-Fourier hybrid approaches.
4. **Variance and volatility swaps.** The replicating portfolio integrates out the characteristic function; Carr-Madan style damping reshapes the convergent integral.
5. **Model risk assessment.** Compute Heston vs Bates vs Merton prices via COS for a common payoff; the max-min gives a model-risk bid-ask.
6. **Single-stock and index exotics.** FFT-based pricing competes with ADI for 2D PDE workloads; cheaper per strike when many strikes are needed.
7. **CMS spread options.** Multi-asset Fourier via joint characteristic function of swap rates under an affine HJM.
8. **Credit-equity hybrids.** Mertonian + jump-diffusion char functions combined by product.
9. **Longevity derivatives.** Affine mortality rates → Fourier pricing of survival caps and variance-annuity combinations.
10. **GPU calibration engines.** FFT and COS parallelize trivially on GPU, enabling real-time re-calibration during trading hours.

---

## 6.5.10 Exercises

**★ (concept drills).**
1. Derive $\psi_T(v) = e^{-rT}\phi_T(v - i(\alpha+1))/(\alpha^2+\alpha-v^2+i(2\alpha+1)v)$.
2. Why does the damping parameter $\alpha$ need to satisfy $\mathbb{E}[S_T^{\alpha+1}] < \infty$?
3. Show that Lewis-Lipton is the limit $\alpha \to 1/2$ of Carr-Madan's expression (up to contour shift).
4. In COS, why is the truncation interval $[a, b]$ chosen using cumulants $c_1 \pm L\sqrt{c_2}$?
5. Explain why COS converges exponentially for smooth densities.
6. Given $\phi$, write down the density $f$ via Fourier inversion. How would you compute $\mathbb{P}(X > a)$ directly from $\phi$?

**★★ (calculation).**
7. Implement Carr-Madan on BS. Compare the price at 50 strikes to closed form for $N = 2^{10}, 2^{12}, 2^{14}$. Report mean absolute error.
8. Implement Lewis-Lipton and verify agreement with Carr-Madan on BS.
9. Implement full COS method. Verify exponential convergence in $N$ on BS. At what $N$ does COS beat Carr-Madan for single-strike pricing?
10. Calibrate Heston to a 5-strike, 3-maturity surface using Carr-Madan pricing + Levenberg-Marquardt.
11. COS for digital option. Compute payoff coefficients $V_n$ for digital cash-or-nothing and verify exponential convergence on BS.
12. Convergence in $\alpha$ for Carr-Madan. Vary $\alpha \in [0.25, 4.0]$ and observe the error pattern for moderate $T$. Find the optimal $\alpha$ empirically.

**★★★ (open / research).**
13. **Fractional FFT.** Implement FrFFT to price a log-spaced strike grid. Compare efficiency to regular FFT + interpolation.
14. **COS for Bermudan options.** Combine COS with backward induction; use characteristic function at each exercise date. Compare to LSM.
15. **Multi-asset Fourier.** Implement Hurd-Zhou for a 2-asset basket call in a 2D Heston. Compare to MC for $d = 2$.
16. **Rough Heston via fractional Riccati.** Solve the El Euch-Rosenbaum fractional Riccati for rough Heston characteristic function numerically, then apply Carr-Madan.
17. **Wavelet methods (SWIFT).** Implement Ortiz-Gracia-Oosterlee SWIFT. Does it handle digitals better than COS?
18. **GPU FFT / COS.** Port Carr-Madan to CUDA. At what strike grid size does the GPU overhead pay off?

---

*— End of Module 6.5. Next: Module 6.6, American Options Numerics (LSM, dual methods).*

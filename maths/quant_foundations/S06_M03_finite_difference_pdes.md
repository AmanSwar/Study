# Module 6.3 — Finite Difference Methods for PDEs

*Subject 6, Module 3. Grid-based pricing for 1D and 2D PDEs — explicit, implicit, Crank-Nicolson, ADI, and early-exercise handling.*

---

## Prerequisites

- **Module 5.2 (Black-Scholes PDE)** — the prototypical parabolic PDE.
- **Module 4.2 (American options free boundary)** — the LCP formulation.
- **Module 0.3 (Fréchet / finite differences)** — the approximation theory.
- **Module 0.4 (Continuous functions, uniform convergence)** — for stability.

---

## 6.3.1 The BS PDE as our testbed

Recall the Black-Scholes PDE for option value $V(t, S)$:
$$
\partial_t V + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} V + r S \partial_S V - rV = 0, \qquad V(T, S) = f(S).
$$

This is a **backward parabolic PDE** with spatially-variable coefficients. Substituting $x = \log S$, $\tau = T - t$:
$$
\partial_\tau V = \tfrac{1}{2}\sigma^2 \partial_{xx} V + (r - \tfrac{1}{2}\sigma^2)\partial_x V - rV, \qquad V(0, x) = f(e^x).
$$
Constant-coefficient version — easier to discretize.

---

## 6.3.2 Finite difference discretization

**Grid.** $x_j = x_{min} + j\Delta x$ for $j = 0, \ldots, J$; $\tau_n = n\Delta\tau$ for $n = 0, \ldots, N$. Denote $V_j^n \approx V(\tau_n, x_j)$.

**Derivative approximations.**
- **Forward difference**: $\partial_x V \approx (V_{j+1} - V_j)/\Delta x$, error $O(\Delta x)$.
- **Backward difference**: $\partial_x V \approx (V_j - V_{j-1})/\Delta x$, error $O(\Delta x)$.
- **Centered difference**: $\partial_x V \approx (V_{j+1} - V_{j-1})/(2\Delta x)$, error $O(\Delta x^2)$.
- **Second derivative**: $\partial_{xx} V \approx (V_{j+1} - 2V_j + V_{j-1})/\Delta x^2$, error $O(\Delta x^2)$.
- **Time**: forward $(V^{n+1} - V^n)/\Delta\tau$, error $O(\Delta\tau)$.

**Combine these into a time-stepping rule.**

---

## 6.3.3 Three schemes: explicit, implicit, Crank-Nicolson

For the model problem $\partial_\tau V = \mathcal{L} V$ where $\mathcal{L}$ is a linear spatial operator, the three canonical schemes are:

**Explicit (forward Euler in time):**
$$
V_j^{n+1} = V_j^n + \Delta\tau \mathcal{L}_h V_j^n.
$$
Easy to implement (matrix-vector multiplication each step). Conditionally stable — $\Delta\tau$ must be small.

**Implicit (backward Euler in time):**
$$
V_j^{n+1} = V_j^n + \Delta\tau \mathcal{L}_h V_j^{n+1}.
$$
Requires solving a linear system $(I - \Delta\tau \mathcal{L}_h) V^{n+1} = V^n$ each step. Unconditionally stable.

**Crank-Nicolson** (average of the two):
$$
V_j^{n+1} - \tfrac{\Delta\tau}{2}\mathcal{L}_h V_j^{n+1} = V_j^n + \tfrac{\Delta\tau}{2}\mathcal{L}_h V_j^n.
$$
Unconditionally stable, $O(\Delta\tau^2 + \Delta x^2)$ accuracy — the practitioner's workhorse.

**For BS** with $\mathcal{L}_h V_j = \tfrac{\sigma^2}{2\Delta x^2}(V_{j+1}-2V_j+V_{j-1}) + \tfrac{r-\sigma^2/2}{2\Delta x}(V_{j+1}-V_{j-1}) - rV_j$, the tridiagonal structure means each implicit step is $O(J)$ via Thomas algorithm.

---

## 6.3.4 Stability and the CFL condition

**Von Neumann analysis.** Expand $V_j^n = \sum_k \hat V_k^n e^{ikj\Delta x}$. Each Fourier mode satisfies $\hat V_k^{n+1} = g(k\Delta x) \hat V_k^n$ where $g$ is the **amplification factor**. Stability requires $|g| \le 1$.

**Explicit, pure diffusion**: $g = 1 - 4\nu \sin^2(k\Delta x/2)$ where $\nu = \sigma^2 \Delta\tau/(2\Delta x^2)$. Stability: $\nu \le 1/2$, i.e.
$$
\boxed{\Delta\tau \le \Delta x^2 / \sigma^2}
$$
— the **CFL (Courant-Friedrichs-Lewy) condition**. Halving $\Delta x$ forces $\Delta\tau$ to shrink by $4\times$, total work up by $8\times$.

**Implicit and Crank-Nicolson**: $|g| < 1$ for **all** $\Delta\tau$. Unconditionally stable.

**But Crank-Nicolson has oscillations.** For $\Delta\tau \gg \Delta x^2$, $g \to -1$ for high modes — **decaying oscillations**, not the monotone decay of the continuous problem. Bad for non-smooth payoffs (digitals, barriers).

**Fix: Rannacher smoothing.** First two time steps use fully-implicit (strongly damping), then Crank-Nicolson. Or use TR-BDF2 (trapezoidal rule + backward differentiation).

---

## 6.3.5 Boundary conditions

**Deep OTM (left boundary)**: $V(t, x_{min}) \to 0$ for calls, $V(t, x_{min}) \to K e^{-r(T-t)} - e^{x_{min}}$ for puts.

**Deep ITM (right boundary)**: $V(t, x_{max}) \approx e^{x_{max}} - K e^{-r(T-t)}$ for calls.

**Better: second-derivative-zero** $\partial_{SS}V = 0$. The rationale: far from strike, the optionality is linear in $S$, so curvature vanishes. Implemented as $V_J = 2V_{J-1} - V_{J-2}$.

**For barriers**: Dirichlet $V = $ rebate at the barrier. Or extend the grid to enforce conditions naturally.

**Grid sizing**. Natural choice: $x_{max} = \log S_0 + 4\sigma\sqrt T$, $x_{min} = \log S_0 - 4\sigma\sqrt T$. Cover 4σ ensures boundary effects are exponentially small.

**Non-uniform grids** concentrate points near the strike/barrier where the function is sharpest. Transformed grid $y = \text{sinh}^{-1}((x-K_*)/\alpha)$ is a common choice.

---

## 6.3.6 American options: PSOR, the LCP

Recall that an American option solves the **Linear Complementarity Problem** (LCP):
$$
\min(-\partial_t V - \mathcal{L} V + rV, \; V - g(t, S)) = 0,
$$
where $g$ is the intrinsic value. Discretized:
$$
(A V^{n+1})_j \ge b_j, \quad V^{n+1}_j \ge g_j, \quad (V^{n+1}_j - g_j)((A V^{n+1})_j - b_j) = 0.
$$

**Projected SOR (PSOR)**. At each iteration $k$:
$$
V_j^{(k+1)} = \max\left\{g_j,\; V_j^{(k)} + \omega\left[\frac{b_j - \sum_{i \ne j} A_{ji} V_i^{(k+1 \text{ or } k)}}{A_{jj}} - V_j^{(k)}\right]\right\}.
$$
Convergent for $\omega \in (0, 2)$, optimal $\omega \approx 1.8$ for typical parameters.

**Brennan-Schwartz** (1977) is an explicit relaxation: take the linear implicit solution, then project pointwise onto the constraint $V \ge g$. Not as accurate as full PSOR but much faster.

**Penalty method** (Forsyth-Vetzal 2002). Replace the LCP with a PDE
$$
\partial_t V + \mathcal{L} V - rV + \rho \max(g - V, 0) = 0
$$
for large $\rho$. Smooths the free boundary and enables standard implicit solvers.

**Modern state-of-the-art**: front-fixing transformations (Wu-Kwok) or policy iteration (Forsyth-Labahn) give fast, accurate American pricers.

---

## 6.3.7 2D PDEs: stochastic vol, quanto, two-asset

For bivariate state $(S, V)$ or $(S_1, S_2)$, the PDE has cross terms. Example Heston:
$$
\partial_t V + \tfrac12 v S^2 V_{SS} + \rho\xi v S V_{SV} + \tfrac12 \xi^2 v V_{VV} + rS V_S + \kappa(\theta - v) V_V - rV = 0.
$$

**Naive approach**: $N_S \times N_V$ unknowns, $N_\tau$ time steps. Implicit step requires inverting an $N_S N_V \times N_S N_V$ matrix — expensive.

**Alternating-Direction Implicit (ADI)** splits the operator $\mathcal{L} = \mathcal{L}_S + \mathcal{L}_V + \mathcal{L}_{SV}$ and marches with implicit treatment in one direction at a time.

**Douglas (2D, without cross term)**:
$$
V^* = V^n + \tfrac{\Delta\tau}{2}\mathcal{L}_S (V^* + V^n) + \Delta\tau \mathcal{L}_V V^n,
$$
$$
V^{n+1} = V^* + \tfrac{\Delta\tau}{2}\mathcal{L}_V (V^{n+1} - V^n).
$$
Each substep is tridiagonal — $O(N_S N_V)$ per time step.

**Craig-Sneyd** (1988) and **Hundsdorfer-Verwer** (2007) handle cross terms via predictor-corrector with $\alpha$, $\theta$ weights.

**Douglas-Rachford** for the cross term. A typical choice (Heston):
$$
Y_0 = V^n + \Delta\tau \mathcal{L} V^n
$$
$$
Y_1 = Y_0 + \tfrac{\Delta\tau}{2}\mathcal{L}_S(Y_1 - V^n)
$$
$$
V^{n+1} = Y_1 + \tfrac{\Delta\tau}{2}\mathcal{L}_V(V^{n+1} - V^n).
$$

**Operator-splitting error**: Douglas scheme is $O(\Delta\tau)$ in the presence of cross terms. Hundsdorfer-Verwer is $O(\Delta\tau^2)$ for well-separated cross terms.

---

## 6.3.8 Convergence theorem (Lax equivalence)

**Definitions.**
- **Consistency**: local truncation error $\tau_h \to 0$ as $h \to 0$.
- **Stability**: $\|V^n\|$ stays bounded for fixed $T$ as $h \to 0$.
- **Convergence**: $\|V^n - V(\tau_n, \cdot)\| \to 0$.

**Lax equivalence theorem (1956)**: for a well-posed linear IBVP, consistency + stability $\iff$ convergence.

**Von Neumann's theorem**: for constant-coefficient linear schemes, stability $\iff |g(k)| \le 1 + C\Delta\tau$ for all $k$ (von Neumann condition).

**Practical order.**
- Crank-Nicolson: $O(\Delta\tau^2 + \Delta x^2)$.
- Implicit Euler: $O(\Delta\tau + \Delta x^2)$.
- Explicit Euler: $O(\Delta\tau + \Delta x^2)$ under CFL.

**Richardson extrapolation.** Compute on grid $(h, h/2)$ and extrapolate:
$$
V^{\text{extrap}} = 2V_{h/2} - V_h
$$
boosts order by one if the leading error is known.

---

## 6.3.9 Python implementation

```python
import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm
import matplotlib.pyplot as plt

# ============================================================
# 1. Crank-Nicolson for BS European call
# ============================================================
def bs_cn_european_call(S0, K, r, sigma, T, J=400, N=200):
    x_min = np.log(S0) - 4*sigma*np.sqrt(T)
    x_max = np.log(S0) + 4*sigma*np.sqrt(T)
    dx = (x_max - x_min)/J
    dt = T/N
    x = np.linspace(x_min, x_max, J+1)

    # Payoff
    V = np.maximum(np.exp(x) - K, 0.0)

    # Operator coefficients (constant, thanks to log-transform)
    a = 0.5*sigma**2/dx**2 - (r-0.5*sigma**2)/(2*dx)
    b = -sigma**2/dx**2 - r
    c = 0.5*sigma**2/dx**2 + (r-0.5*sigma**2)/(2*dx)

    # Matrices for (I - dt/2 L) V^{n+1} = (I + dt/2 L) V^n
    interior = slice(1, J)
    # Build tridiagonal (banded) representations:
    A_upper = -0.5*dt*c*np.ones(J-1)
    A_diag  = 1 - 0.5*dt*b*np.ones(J-1)
    A_lower = -0.5*dt*a*np.ones(J-1)
    B_upper =  0.5*dt*c*np.ones(J-1)
    B_diag  = 1 + 0.5*dt*b*np.ones(J-1)
    B_lower =  0.5*dt*a*np.ones(J-1)

    ab = np.zeros((3, J-1))
    ab[0, 1:] = A_upper[:-1]
    ab[1, :]  = A_diag
    ab[2, :-1]= A_lower[1:]

    for n in range(N):
        # Boundary: V(t, x_min) = 0, V(t, x_max) = S_max - K*exp(-r*(T-t+dt))
        t_next = (n+1)*dt
        bdy_low  = 0.0
        bdy_high = np.exp(x_max) - K*np.exp(-r*(T-t_next))
        # RHS = B*V_interior + boundary contributions
        v_in = V[interior]
        rhs = B_lower*np.concatenate(([V[0]], v_in[:-1])) \
            + B_diag*v_in \
            + B_upper*np.concatenate((v_in[1:], [V[-1]]))
        # Add boundary terms to first/last entries of rhs
        rhs[0]  += 0.5*dt*a*(V[0] + bdy_low) - A_lower[0]*bdy_low
        rhs[-1] += 0.5*dt*c*(V[-1] + bdy_high) - A_upper[-1]*bdy_high
        # Solve tridiagonal
        v_new = solve_banded((1,1), ab, rhs)
        V[0] = bdy_low; V[-1] = bdy_high
        V[interior] = v_new

    # Interpolate at S0
    return np.interp(np.log(S0), x, V)

def bs_analytic_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

params = (100, 100, 0.05, 0.2, 1.0)
v_cn = bs_cn_european_call(*params)
v_an = bs_analytic_call(*params)
print(f"Crank-Nicolson: {v_cn:.6f}")
print(f"Analytic:       {v_an:.6f}")
print(f"Error:          {abs(v_cn-v_an):.2e}")

# ============================================================
# 2. PSOR for American put
# ============================================================
def am_put_psor(S0, K, r, sigma, T, J=400, N=200, omega=1.7, tol=1e-6):
    x_min = np.log(S0) - 4*sigma*np.sqrt(T)
    x_max = np.log(S0) + 4*sigma*np.sqrt(T)
    dx = (x_max - x_min)/J
    dt = T/N
    x = np.linspace(x_min, x_max, J+1)
    S = np.exp(x)
    V = np.maximum(K - S, 0.0)
    payoff = V.copy()

    a = 0.5*sigma**2/dx**2 - (r-0.5*sigma**2)/(2*dx)
    b = -sigma**2/dx**2 - r
    c = 0.5*sigma**2/dx**2 + (r-0.5*sigma**2)/(2*dx)

    for n in range(N):
        # RHS of Crank-Nicolson implicit half
        rhs = np.zeros(J-1)
        for j in range(1, J):
            rhs[j-1] = 0.5*dt*a*V[j-1] + (1 + 0.5*dt*b)*V[j] + 0.5*dt*c*V[j+1]
        # Solve (I - dt/2 L) V^new = rhs with projection
        V_new = V[1:J].copy()
        for iter in range(500):
            err = 0.0
            for j in range(J-1):
                left  = V_new[j-1] if j > 0    else V[0]
                right = V_new[j+1] if j < J-2 else V[-1]
                resid = rhs[j] + 0.5*dt*a*left + 0.5*dt*c*right \
                       - (1 - 0.5*dt*b)*V_new[j]
                # Gauss-Seidel update with projection
                update = V_new[j] + omega*resid/(1 - 0.5*dt*b)
                update = max(update, payoff[j+1])
                err = max(err, abs(update - V_new[j]))
                V_new[j] = update
            if err < tol: break
        V[1:J] = V_new
        # Boundaries
        V[0]  = K - S[0]   # deep ITM put
        V[-1] = 0.0        # deep OTM put
    return np.interp(np.log(S0), x, V)

am_price = am_put_psor(100, 100, 0.05, 0.2, 1.0)
print(f"\nAmerican put via PSOR: {am_price:.4f}")
print(f"European put analytic: {100*np.exp(-0.05*1.0)*norm.cdf(norm.ppf(0.5)) - bs_analytic_call(100,100,0.05,0.2,1.0)+100-100*np.exp(-0.05):.4f} (approximate check)")

# ============================================================
# 3. ADI for 2D Heston (illustration — simplified)
# ============================================================
def heston_adi_call(S0, V0, K, r, kappa, theta, xi, rho, T,
                    J_S=100, J_V=50, N=100):
    S_max = 4*K; V_max = 0.5
    dS = S_max/J_S; dV = V_max/J_V
    dt = T/N
    S = np.linspace(0, S_max, J_S+1)
    v = np.linspace(0, V_max, J_V+1)

    # Initial condition: call payoff
    u = np.maximum(S[:, None] - K, 0.0) * np.ones((J_S+1, J_V+1))

    # Simple explicit scheme for illustration — not production quality
    # Production ADI requires careful operator splitting
    for n in range(N):
        u_new = u.copy()
        for j in range(1, J_S):
            for i in range(1, J_V):
                Su = S[j]; Vu = v[i]
                dSS = (u[j+1,i] - 2*u[j,i] + u[j-1,i])/dS**2
                dVV = (u[j,i+1] - 2*u[j,i] + u[j,i-1])/dV**2
                dSV = (u[j+1,i+1] - u[j+1,i-1] - u[j-1,i+1] + u[j-1,i-1])/(4*dS*dV)
                dS_ = (u[j+1,i] - u[j-1,i])/(2*dS)
                dV_ = (u[j,i+1] - u[j,i-1])/(2*dV)
                L = (0.5*Vu*Su**2*dSS + rho*xi*Vu*Su*dSV + 0.5*xi**2*Vu*dVV
                     + r*Su*dS_ + kappa*(theta-Vu)*dV_ - r*u[j,i])
                u_new[j,i] = u[j,i] + dt*L
        # Boundaries (simplified)
        u_new[0, :] = 0
        u_new[-1, :] = S_max - K*np.exp(-r*(T-(n+1)*dt))
        u_new[:, 0] = np.maximum(S - K*np.exp(-r*(T-(n+1)*dt)), 0)
        u_new[:, -1] = u_new[:, -2]  # outflow
        u = u_new

    # Interpolate at (S0, V0)
    idx_S = np.searchsorted(S, S0) - 1
    idx_V = np.searchsorted(v, V0) - 1
    return u[idx_S, idx_V]

# demonstration (low resolution — result only approximate)
# v_heston = heston_adi_call(100,0.04,100,0.05,2.0,0.04,0.3,-0.7,1.0,J_S=50,J_V=25,N=50)
# print(f"Heston call via ADI: {v_heston:.4f}")
print("(Skipping Heston ADI runtime — see Module 5.6 Python for FFT pricing)")
```

**What to check.**
- CN on BS European call should match the analytic formula to $10^{-3}$ or better with $J=400, N=200$.
- Halving $J$ doubles the error (suggests a CN-based order higher than 2 is hard to hit without Rannacher smoothing due to the non-smooth payoff).
- PSOR converges in 30–100 iterations per time step; optimal $\omega$ near 1.7.
- ADI simple explicit version illustrates the structure; production use requires Douglas-Rachford with operator splitting.

---

## 6.3.10 [QUANT APPLICATIONS]

1. **Equity exotics pricing.** Barriers, touches, digitals, American options in local volatility — 1D or 2D PDEs (with variance as second dim) are the default.
2. **Interest-rate Greeks.** Tree or PDE solvers for Hull-White or BK models used to compute deltas and vegas of large swaption books.
3. **Cash-flow CVA with exposure profiles.** Bermudan-style path construction requires PSOR-like backward induction.
4. **FX pricing with stochastic rates.** Coupled 3D PDE for FX-IR models; ADI with three-directional splitting.
5. **Credit derivative pricing.** Reduced-form models lead to PDEs for bond and CDS prices.
6. **Energy / commodity options.** Storage options, swing options modeled as 2D PDE (spot price + inventory).
7. **Insurance products.** Variable annuities with guaranteed income benefits require 3D PDE (fund value + account value + ratchet).
8. **Calibration.** PDE-based pricers faster than MC for repeated calls during minimization.
9. **Greeks via PDE.** Deltas and gammas are free (finite-difference operator itself); vegas by differentiating w.r.t. vol at grid construction time.
10. **Benchmarking.** PDE pricers are often the gold standard against which MC and tree prices are validated.

---

## 6.3.11 Exercises

**★ (concept drills).**
1. Derive the CFL condition $\Delta\tau \le \Delta x^2/\sigma^2$ for the explicit scheme on the heat equation. Why does going to 2D tighten the constraint?
2. Verify that centered differences are $O(\Delta x^2)$. What is the leading error term for $\partial_{SS} V$?
3. Why does Crank-Nicolson have oscillations for non-smooth payoffs? Describe the Rannacher smoothing fix.
4. State the Lax equivalence theorem. Why is it essential for convergence analysis?
5. Explain why the log-transform turns the BS PDE into constant coefficients. Why is that desirable?
6. For an American option, write the LCP formulation. Why can't we just solve the PDE and then project?

**★★ (calculation).**
7. Implement Crank-Nicolson for a European call and verify quadratic convergence as $(\Delta\tau, \Delta x) \to 0$. Plot log-error vs log-h.
8. Add Rannacher smoothing (two fully-implicit steps, then CN) to the European call. Measure the improvement in Greeks accuracy near maturity.
9. Implement PSOR for an American put. Verify that optimal $\omega$ is ~1.7 for typical parameters. How do iterations scale with grid size?
10. Derive the Douglas ADI scheme for a 2D convection-diffusion with no cross term. Show it is $O(\Delta\tau^2 + \Delta x^2 + \Delta y^2)$.
11. For Heston, derive the boundary conditions at $v = 0$ and $v = v_{max}$. How does the Feller condition affect the lower boundary treatment?

**★★★ (open / research).**
12. **Adaptive mesh refinement.** Implement a 1D solver that refines the grid near the strike dynamically. Measure efficiency gain on OTM puts.
13. **Policy iteration for American options** (Forsyth-Labahn 2007). Compare to PSOR on convergence and robustness.
14. **Semi-Lagrangian schemes** for advection-dominated problems (e.g., high-interest-rate FX). Implement and compare to upwinding.
15. **PDE vs MC for 3D.** Implement a 3D PDE solver for stochastic-vol + stochastic-rate + stochastic-dividend model. At what accuracy/dimension crossover does MC win?
16. **Deep PDE solver** (Han-Jentzen-E 2018). Implement a neural-network-based solver for a 100-dim BSDE and compare to tensor-train methods.

---

*— End of Module 6.3. Next: Module 6.4, Trees and Lattices.*

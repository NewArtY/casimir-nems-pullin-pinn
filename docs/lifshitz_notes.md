# Thermal (Lifshitz) shift of the pull-in fold boundary

Companion notes for `src/lifshitz_boundary.py` (PRApplied, Fig. 3 data).
Rest gap `d = 100 nm` throughout unless stated.

## 1. Modified (finite-T) net force

The dimensionless zero-T net force is

    g0(xi) = alpha/(1-xi)^2 + beta/(1-xi)^4 - xi ,     xi in [0,1) ,

with `alpha` the electrostatic drive (~1/gap^2), `beta` the Casimir drive
(~1/gap^4) and the physical gap `(d - x) = d(1 - xi)`.

The leading finite-temperature Lifshitz correction multiplies **only** the
Casimir pressure by `[1 + kappa T (1 - xi)]`, evaluated at the *instantaneous*
gap:

    kappa(d) = (60 zeta(3) / pi^3) * k_B d / (hbar c)   [1/K],
    60 zeta(3)/pi^3 = 2.3261 ,   zeta(3) = 1.2020569 .

The coefficient is the ratio of the classical (high-T) Casimir pressure
P_cl = zeta(3) k_B T/(4 pi a^3) to the zero-temperature pressure
P_0 = pi^2 hbar c/(240 a^4); the derivation is Appendix A of the manuscript
(`article/sections/appendix_lifshitz.tex`). An earlier version of these notes
used 720 zeta(3)/pi^3 = 27.913, which is too large by exactly 12: it divided
the classical term by the FREE-ENERGY coefficient 720 instead of the pressure
coefficient 240, and dropped the factor 4 in P_cl. All numbers below were
regenerated after the fix.

Hence the thermal net force

    g(xi) = alpha/(1-xi)^2 + beta[1 + kappa T (1-xi)]/(1-xi)^4 - xi
          = alpha/u^2 + beta/u^4 + beta*gamma/u^3 - xi ,   u = 1-xi ,

where `gamma = kappa(d) T` is the dimensionless thermal strength. The
correction introduces a **new (1-xi)^{-3} term sitting between** the
electrostatic `^{-2}` and the Casimir `^{-4}` terms:

    beta[1+kappa T(1-xi)]/(1-xi)^4  =  beta/(1-xi)^4  +  beta kappa T/(1-xi)^3 .

At `d = 100 nm`: `k_B/(hbar c) = 436.70 m^-1 K^-1`, `kappa = 1.0158e-4 /K`,
so `kappa T = 0.010158` (100 K) and `0.030474` (300 K).

The correction is exactly `physics.beta_lifshitz(beta, xi, T, d)/(1-xi)^4`;
`net_force_T` and the RK4 integrator therefore use the identical Casimir
prefactor, so the static-fold and dynamic-pull-in results are mutually
consistent by construction.

Derivative w.r.t. xi (used for the fold `g'=0` condition; `d/dxi = -d/du`, and
each `u^{-n}` contributes `+n u^{-n-1}`):

    g'(xi) = 2 alpha/u^3 + 4 beta/u^5 + 3 beta*gamma/u^4 - 1 .

## 2. Fold locus as a 2x2 LINEAR system

Pull-in is the saddle-node (fold) where the stable node and the saddle merge:
`g = 0` and `g' = 0`. The extra `u^{-3}` term destroys the clean closed form,
but **both conditions remain linear in (alpha, beta)**. Writing them at a fixed
pull-in gap `u = 1 - xi_PI` (so `xi = 1 - u`, `gamma = kappa(d) T`):

    g  = 0 :  alpha u^{-2}   + beta ( u^{-4}   + gamma u^{-3} ) = 1 - u ,
    g' = 0 :  alpha 2 u^{-3} + beta ( 4 u^{-5} + 3 gamma u^{-4}) = 1 .

In matrix form `M [alpha, beta]^T = rhs`:

    M   = [[ u^{-2},       u^{-4} + gamma u^{-3}    ],
           [ 2 u^{-3},   4 u^{-5} + 3 gamma u^{-4}  ]] ,
    rhs = [ 1 - u ,  1 ]^T .

`fold_boundary_T(T, d)` sweeps `u`, solves this 2x2 system exactly at each `u`
(`numpy.linalg.solve`), and keeps the physical branch `alpha_c >= 0` and
`beta_c >= 0`.

**T = 0 check (analytic).** With `gamma = 0`, `det M = 2 u^{-7}` and Cramer's
rule gives

    alpha_c(u) = u^2 (4 - 5u)/2 ,   beta_c(u) = u^4 (3u - 2)/2 ,

the known closed form (physical branch `u in [2/3, 4/5]`,
`xi_PI in [1/5, 1/3]`; endpoints `(4/27, 0)` and `(0, 256/3125)`).
`fold_boundary_T(0, d)` reproduces it to **max|dalpha| = 1.2e-16,
max|dbeta| = 6.9e-17**, and the residuals `max|g| = 8e-17`, `max|g'| = 2e-16`
along the numeric locus (validated in `_validate_T0`).

**Effect of gamma > 0.** The thermal term is purely attractive, so the fold
surface moves *inward* (smaller drive needed to pull in): at fixed `beta` the
critical `alpha_c` drops, and the Casimir-axis intercept `beta*` drops. The
`beta = 0` endpoint is *unchanged* (`u = 2/3`, `xi_PI = 1/3`): with no Casimir
term there is nothing for the Lifshitz factor to multiply.

Closed forms for the two axes at general `gamma`:
* `beta = 0` endpoint: `u = 2/3` (T-independent), `alpha_c = 4/27`.
* `alpha = 0` intercept `beta*`: `u` solves `4 gamma u^2 + (5 - 3 gamma) u - 4 = 0`,
  then `beta* = u^5/(4 + 3 gamma u)`. (Reduces to `u=4/5`, `beta*=256/3125` at
  `gamma=0`.) The code obtains the same numbers by interpolation.

## 3. Numeric shifts (d = 100 nm)

Boundary intercepts:

| T (K) | kappa*T | beta* (alpha=0) | alpha_c at beta=0.03 |
|------:|--------:|----------------:|---------------------:|
|     0 | 0.000000 | 0.081907       | 0.087627 |
|   100 | 0.010158 | 0.081257       | 0.087213 |
|   300 | 0.030474 | 0.079914       | 0.086384 |

Shift at fixed **beta = 0.03** (relevant Fig. 3 operating line), and the
Casimir-axis intercept `beta*` shift. Voltage mapping uses `alpha ~ V^2`, i.e.
`V_PI ~ sqrt(alpha)` so `DV_PI/V_PI = sqrt(alpha_c(T)/alpha_c(0)) - 1`:

| T (K) | Dalpha_c/alpha_c | DV_PI/V_PI | Dbeta*/beta* |
|------:|-----------------:|-----------:|-------------:|
|   100 |   -5.67 %        |  -2.87 %   |  -8.89 %     |
|   300 |  -16.95 %        |  -8.87 %   | -22.58 %     |

So heating from 0 to 300 K lowers the critical electrostatic drive at
beta=0.03 by ~17 % (an ~8.9 % drop in pull-in voltage), and pulls the
pure-Casimir stability limit `beta*` down by ~23 %. The effect is one-sided
(always destabilising) and largest along the Casimir axis, where the corrected
term dominates.

Data files (regenerated by the script):
* `results/lifshitz_boundaries.npz` — per-T arrays `alpha_c, beta_c, xi_pi`,
  plus `beta_star` and `alpha_at_beta003`.
* `results/lifshitz_boundaries.csv` — columns `T_K, xi_pi, u, alpha_c, beta_c`.
* `results/lifshitz_shift_summary.csv` — the shift table above.

## 4. Dynamic (RK4) confirmation

Starting from rest (`xi = v = 0`, `zeta = 0.5`) with the Lifshitz-corrected
Casimir term, at the probe point `alpha = 0.99 * alpha_c(T=0, beta=0.03) =
0.08675`, `beta = 0.03`:

* `T = 0 K`  -> `pull_in = False` (stable; point lies just inside the T=0 fold),
* `T = 300 K` -> `pull_in = True`  (the fold has moved inward past it).

This confirms the static-fold shift has the correct sign and magnitude: warming
the device pulls in an operating point that was stable cold.

## 5. Honest discussion of magnitude - two different length scales

The per-cent shift at 100 nm is set by `kappa T` (0.0102 at 100 K, 0.0305 at
300 K), and the thermal wavelength is

    lambda_T = hbar c / (k_B T) = 7.633 um at 300 K.

With the corrected prefactor the two scales are consistent with each other.
The linear form doubles the Casimir pressure at

    a = lambda_T / 2.3261 = 3.28 um  (at 300 K),

i.e. at 0.43 lambda_T, so the quantum-to-classical crossover sits at the
thermal wavelength, as it must. The old prefactor 27.913 put that crossover at
273 nm = 0.036 lambda_T, which contradicted the premise that lambda_T is the
crossover scale; that inconsistency is what exposed the error.

At `d = 100 nm` the linear form gives `kappa T (1 - xi) = 0.76 %` (100 K) and
`2.29 %` (300 K) at the pull-in gap, hence a `-0.24 %` / `-0.71 %` shift in
`V_PI` at beta = 0.03 and a `-0.8 %` / `-2.4 %` shift in `beta*`. These remain
UPPER BOUNDS and not predictions: at `d << lambda_T` the ideal-mirror thermal
correction to the pressure is

    P(a,T) = P_0(a) [1 + (16/3) (a/lambda_T)^4] ,

with no linear term at all (Appendix A of the manuscript). That is `1.6e-7` at
100 nm and 300 K, seven orders of magnitude below the linear estimate. The
linear form becomes the physical answer only as `a` approaches `lambda_T`.

"""Thermal (Lifshitz) shift of the pull-in fold boundary.

The zero-temperature Casimir-electrostatic NEMS pull-in model has net force

    g0(xi) = alpha/(1-xi)^2 + beta/(1-xi)^4 - xi ,

whose fold (saddle-node) locus g0 = g0' = 0 admits the closed form

    alpha_c(u) = u^2 (4 - 5 u) / 2 ,   beta_c(u) = u^4 (3 u - 2) / 2 ,

with u = 1 - xi_PI the dimensionless gap at pull-in.

Finite temperature multiplies the Casimir term by the leading Lifshitz factor
[1 + kappa T (1 - xi)] with the *instantaneous* gap (d - x) = d (1 - xi):

    kappa(d) = (60 zeta(3) / pi^3) * k_B d / (hbar c)      [1/K] ,

so the thermal net force is

    g(xi) = alpha/(1-xi)^2 + beta[1 + kappa T (1-xi)]/(1-xi)^4 - xi
          = alpha/(1-xi)^2 + beta/(1-xi)^4 + beta kappa T/(1-xi)^3 - xi .

The new (1-xi)^{-3} term (between the electrostatic ^{-2} and Casimir ^{-4}
terms) breaks the clean closed form, but the two fold conditions g = g' = 0 are
STILL LINEAR in (alpha, beta).  For each pull-in position xi (gap u = 1 - xi)
we solve the 2x2 linear system for (alpha_c, beta_c); at T = 0 this reproduces
the closed form above to machine precision.

Run ``python src/lifshitz_boundary.py`` to regenerate the boundaries, the
T = 0 validation, and results/lifshitz_boundaries.{npz,csv}.
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Robust imports: work both as a package module (from . import ...) and when
# invoked directly as ``python src/lifshitz_boundary.py``.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_HERE))  # add .../PRA/code
    from src import config
    from src.physics import beta_lifshitz, net_force  # reused (validation)
    from src.rk4 import rk4_integrate
    from src.bifurcation import fold_boundary
else:
    from . import config
    from .physics import beta_lifshitz, net_force
    from .rk4 import rk4_integrate
    from .bifurcation import fold_boundary


# ---------------------------------------------------------------------------
# Thermal prefactor kappa(d)
# ---------------------------------------------------------------------------
PREFAC = 60.0 * config.ZETA3 / np.pi ** 3           # = 2.3261...  (see Appendix A)


def kappa(d):
    """Thermal Casimir coefficient kappa(d) = (60 zeta3/pi^3) k_B d/(hbar c) [1/K].

    kappa T is the dimensionless strength of the leading Lifshitz correction at
    the rest gap; the correction to the Casimir term is [1 + kappa T (1-xi)].
    """
    return PREFAC * config.K_B * d / (config.HBAR * config.C_LIGHT)


# ---------------------------------------------------------------------------
# 1. Thermal net force and its xi-derivative
# ---------------------------------------------------------------------------
def net_force_T(xi, alpha, beta, T, d):
    """Finite-T net force.

        g(xi) = alpha/(1-xi)^2 + beta[1 + kappa T (1-xi)]/(1-xi)^4 - xi
              = alpha/u^2 + beta/u^4 + beta*gamma/u^3 - xi ,   u = 1-xi ,

    with gamma = kappa(d) T.  For T = 0 this is exactly ``physics.net_force``.
    """
    xi = np.asarray(xi, dtype=float)
    u = 1.0 - xi
    g = kappa(d) * T
    return alpha / u ** 2 + beta / u ** 4 + beta * g / u ** 3 - xi


def net_force_T_prime(xi, alpha, beta, T, d):
    """d/dxi of net_force_T.

        g'(xi) = 2 alpha/u^3 + 4 beta/u^5 + 3 beta*gamma/u^4 - 1 ,   u = 1-xi .

    (Since d/dxi = -d/du and each u^{-n} contributes +n u^{-n-1}.)
    """
    xi = np.asarray(xi, dtype=float)
    u = 1.0 - xi
    g = kappa(d) * T
    return 2.0 * alpha / u ** 3 + 4.0 * beta / u ** 5 + 3.0 * beta * g / u ** 4 - 1.0


# ---------------------------------------------------------------------------
# 2. Numerical fold boundary at temperature T
# ---------------------------------------------------------------------------
#
# Fold conditions at pull-in position xi (gap u = 1 - xi), gamma = kappa(d) T:
#
#   g  = 0 :  alpha * u^{-2}    + beta * ( u^{-4}    + gamma u^{-3} ) = xi = 1 - u
#   g' = 0 :  alpha * 2 u^{-3}  + beta * ( 4 u^{-5}  + 3 gamma u^{-4}) = 1
#
# This is a 2x2 LINEAR system  M [alpha, beta]^T = rhs  with
#
#   M   = [[ u^{-2},        u^{-4} + gamma u^{-3}   ],
#          [ 2 u^{-3},   4 u^{-5} + 3 gamma u^{-4}  ]] ,
#   rhs = [ 1 - u , 1 ]^T .
#
# Solving per u gives the fold locus (alpha_c(u), beta_c(u)).  At gamma = 0,
#   det M = 2 u^{-7},   alpha_c = u^2(4-5u)/2,   beta_c = u^4(3u-2)/2  (exact).
# ---------------------------------------------------------------------------
def _fold_ab_at_u(u, gamma):
    """Solve the 2x2 fold linear system at a single gap u.  Returns (alpha,beta)."""
    u2, u3, u4, u5 = u ** -2, u ** -3, u ** -4, u ** -5
    M = np.array([[u2, u4 + gamma * u3],
                  [2.0 * u3, 4.0 * u5 + 3.0 * gamma * u4]], dtype=float)
    rhs = np.array([1.0 - u, 1.0], dtype=float)
    return np.linalg.solve(M, rhs)


def fold_boundary_T(T, d, n=400, u_lo=0.60, u_hi=0.85):
    """Numerical fold (pull-in) locus at temperature T for rest gap d.

    Parametrised by the gap u = 1 - xi_PI.  For each u the two fold conditions
    g = g' = 0 form a 2x2 linear system in (alpha, beta), solved exactly.

    Only the physically admissible branch (alpha_c >= 0 and beta_c >= 0) is
    returned, sorted by increasing beta_c (so beta_c runs 0 -> beta*).

    Parameters
    ----------
    T : float
        Temperature [K].
    d : float
        Rest gap [m] (sets kappa).
    n : int
        Number of gap samples across [u_lo, u_hi] before masking.
    u_lo, u_hi : float
        Scan window for u; the physical branch (T-dependent) lies inside it.

    Returns
    -------
    alpha_c, beta_c, xi_pi : ndarray
        Fold-locus arrays on the physical branch, sorted by beta_c.
    """
    gamma = kappa(d) * T
    u = np.linspace(u_lo, u_hi, n)
    alpha_c = np.empty_like(u)
    beta_c = np.empty_like(u)
    for i, ui in enumerate(u):
        alpha_c[i], beta_c[i] = _fold_ab_at_u(ui, gamma)
    xi_pi = 1.0 - u

    phys = (alpha_c >= 0.0) & (beta_c >= 0.0)
    alpha_c, beta_c, xi_pi = alpha_c[phys], beta_c[phys], xi_pi[phys]

    order = np.argsort(beta_c)
    return alpha_c[order], beta_c[order], xi_pi[order]


# ---------------------------------------------------------------------------
# Helpers: read the boundary at fixed beta, and the alpha=0 intercept beta*.
# ---------------------------------------------------------------------------
def alpha_c_at_beta(T, d, beta_fixed, n=4000):
    """Critical alpha on the fold boundary at a fixed beta (linear interpolation).

    beta_c increases monotonically along the physical branch (0 -> beta*), so a
    1-D interpolation of alpha_c against beta_c is single-valued.
    """
    alpha_c, beta_c, _ = fold_boundary_T(T, d, n=n)
    if not (beta_c.min() <= beta_fixed <= beta_c.max()):
        raise ValueError(f"beta={beta_fixed} outside branch [{beta_c.min():.4g},{beta_c.max():.4g}]")
    # beta_c already sorted ascending
    return float(np.interp(beta_fixed, beta_c, alpha_c))


def beta_star(T, d, n=4000):
    """Casimir-axis intercept beta* = beta_c at alpha = 0 (interp on alpha_c->0)."""
    alpha_c, beta_c, _ = fold_boundary_T(T, d, n=n)
    # near the alpha=0 end alpha_c decreases to 0 as beta_c increases; interpolate
    # beta_c as a function of alpha_c on the descending tail.
    order = np.argsort(alpha_c)
    a_sorted = alpha_c[order]
    b_sorted = beta_c[order]
    return float(np.interp(0.0, a_sorted, b_sorted))


# ---------------------------------------------------------------------------
# 4. Dynamic (RK4) cross-check that the boundary moves the right way.
# ---------------------------------------------------------------------------
def pulls_in_from_rest(alpha, beta, T, d, zeta=0.5, t_max=200.0, dt=2e-3):
    """RK4 from rest (xi=v=0) with the Lifshitz-corrected Casimir term.

    Returns True if the device pulls in.  Uses rk4_integrate, whose beta is
    Lifshitz-corrected per displacement exactly as net_force_T prescribes.
    """
    _, _, _, info = rk4_integrate(alpha, beta, zeta, 0.0, 0.0, t_max, dt, T=T, d=d)
    return bool(info["pull_in"])


# ---------------------------------------------------------------------------
# Result generation / self-test
# ---------------------------------------------------------------------------
def _validate_T0(n=400, tol=1e-9):
    """Confirm fold_boundary_T(T=0) reproduces the closed-form fold locus."""
    alpha_c, beta_c, xi_pi = fold_boundary_T(0.0, config.DEFAULTS["d"], n=n)
    u = 1.0 - xi_pi
    a_closed = u ** 2 * (4.0 - 5.0 * u) / 2.0
    b_closed = u ** 4 * (3.0 * u - 2.0) / 2.0
    da = float(np.max(np.abs(alpha_c - a_closed)))
    db = float(np.max(np.abs(beta_c - b_closed)))
    # also residual of g=g'=0 along the numeric locus
    res_g = float(np.max(np.abs(net_force_T(xi_pi, alpha_c, beta_c, 0.0, config.DEFAULTS["d"]))))
    res_gp = float(np.max(np.abs(net_force_T_prime(xi_pi, alpha_c, beta_c, 0.0, config.DEFAULTS["d"]))))
    ok = da < tol and db < tol and res_g < tol and res_gp < tol
    return ok, da, db, res_g, res_gp


def main():
    d = config.DEFAULTS["d"]  # 100 nm
    temps = [0.0, 100.0, 300.0]
    beta_fixed = 0.03

    print("=" * 72)
    print("Thermal (Lifshitz) shift of the Casimir-electrostatic pull-in fold")
    print("=" * 72)
    print(f"d = {d*1e9:.1f} nm,  kappa(d) = {kappa(d):.6e} /K,  prefac = {PREFAC:.5f}")
    print(f"kappa*T: 100 K -> {kappa(d)*100:.5f}   300 K -> {kappa(d)*300:.5f}")
    lam_T300 = config.HBAR * config.C_LIGHT / (config.K_B * 300.0)
    print(f"lambda_T(300K) = hbar c/kB T = {lam_T300*1e6:.3f} um ; "
          f"per-cent onset d = lambda_T/prefac = {lam_T300/PREFAC*1e9:.1f} nm")

    # --- T=0 validation against the closed form ---------------------------
    ok, da, db, rg, rgp = _validate_T0()
    print("\n[validate] fold_boundary_T(T=0) vs closed form u^2(4-5u)/2, u^4(3u-2)/2")
    print(f"    max|dalpha|={da:.2e}  max|dbeta|={db:.2e}  "
          f"max|g|={rg:.2e}  max|g'|={rgp:.2e}  -> {'OK' if ok else 'FAIL'}")
    assert ok, "T=0 numeric fold does not reproduce the closed form to 1e-9"

    # --- boundaries at each T --------------------------------------------
    per_T = {}
    print("\n[boundaries]  physical branch endpoints and key intercepts")
    for T in temps:
        a_c, b_c, xi_pi = fold_boundary_T(T, d)
        bstar = beta_star(T, d)
        a_at = alpha_c_at_beta(T, d, beta_fixed)
        per_T[T] = dict(alpha_c=a_c, beta_c=b_c, xi_pi=xi_pi,
                        beta_star=bstar, alpha_at_beta003=a_at)
        print(f"  T={T:5.0f} K: beta*(alpha=0)={bstar:.6f}  "
              f"alpha_c(beta={beta_fixed})={a_at:.6f}  "
              f"alpha_endpoint(beta=0)={a_c.max():.6f}  "
              f"xi_PI range=[{xi_pi.min():.4f},{xi_pi.max():.4f}]")

    # --- quantified shifts ------------------------------------------------
    a0 = per_T[0.0]["alpha_at_beta003"]
    bstar0 = per_T[0.0]["beta_star"]
    print(f"\n[shift @ beta={beta_fixed}]  alpha_c(T=0) = {a0:.6f}")
    shift_rows = []
    for T in temps[1:]:
        aT = per_T[T]["alpha_at_beta003"]
        dalpha_rel = (aT - a0) / a0 * 100.0
        # alpha ~ V^2  =>  V_PI ~ sqrt(alpha)  =>  dV/V = sqrt(aT/a0) - 1
        dV_rel = (np.sqrt(aT / a0) - 1.0) * 100.0
        bstarT = per_T[T]["beta_star"]
        dbstar_rel = (bstarT - bstar0) / bstar0 * 100.0
        shift_rows.append((T, aT, dalpha_rel, dV_rel, bstarT, dbstar_rel))
        print(f"  T={T:5.0f} K: alpha_c={aT:.6f}  "
              f"Dalpha_c/alpha_c={dalpha_rel:+.3f}%  "
              f"DV_PI/V_PI={dV_rel:+.3f}%  |  "
              f"beta*={bstarT:.6f}  Dbeta*/beta*={dbstar_rel:+.3f}%")

    # --- dynamic RK4 cross-check -----------------------------------------
    # Operating point just INSIDE the T=0 boundary at beta_fixed (stable at T=0);
    # raising T lowers the boundary so the same point should pull in.
    print("\n[dynamic RK4 cross-check]  from rest, zeta=0.5")
    a_probe = 0.99 * a0
    for T in (0.0, 300.0):
        pi = pulls_in_from_rest(a_probe, beta_fixed, T, d)
        print(f"  (alpha={a_probe:.5f}, beta={beta_fixed}, T={T:5.0f} K): "
              f"pull_in={pi}")

    # --- save results -----------------------------------------------------
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.normpath(os.path.join(here, "..", "results"))
    os.makedirs(res_dir, exist_ok=True)

    npz_path = os.path.join(res_dir, "lifshitz_boundaries.npz")
    save_kw = {"temps": np.array(temps), "d": d, "kappa": kappa(d)}
    for T in temps:
        tag = f"T{int(T)}"
        save_kw[f"alpha_c_{tag}"] = per_T[T]["alpha_c"]
        save_kw[f"beta_c_{tag}"] = per_T[T]["beta_c"]
        save_kw[f"xi_pi_{tag}"] = per_T[T]["xi_pi"]
        save_kw[f"beta_star_{tag}"] = per_T[T]["beta_star"]
        save_kw[f"alpha_at_beta003_{tag}"] = per_T[T]["alpha_at_beta003"]
    np.savez(npz_path, **save_kw)
    print(f"\n[save] {npz_path}")

    # CSV: one row per (T, sample) of the boundary
    csv_path = os.path.join(res_dir, "lifshitz_boundaries.csv")
    rows = []
    for T in temps:
        a_c, b_c, xi_pi = per_T[T]["alpha_c"], per_T[T]["beta_c"], per_T[T]["xi_pi"]
        for aa, bb, xx in zip(a_c, b_c, xi_pi):
            rows.append((T, xx, 1.0 - xx, aa, bb))
    data = np.array(rows, dtype=float)
    np.savetxt(csv_path, data, delimiter=",",
               header="T_K,xi_pi,u,alpha_c,beta_c", comments="", fmt="%.10e")
    print(f"[save] {csv_path}")

    # summary CSV of shifts
    shift_csv = os.path.join(res_dir, "lifshitz_shift_summary.csv")
    with open(shift_csv, "w", encoding="utf-8") as f:
        f.write("T_K,alpha_c_at_beta003,dAlpha_rel_pct,dV_PI_rel_pct,"
                "beta_star,dBetaStar_rel_pct\n")
        f.write(f"0,{a0:.10e},0.0,0.0,{bstar0:.10e},0.0\n")
        for (T, aT, da_r, dV_r, bsT, db_r) in shift_rows:
            f.write(f"{int(T)},{aT:.10e},{da_r:.6f},{dV_r:.6f},"
                    f"{bsT:.10e},{db_r:.6f}\n")
    print(f"[save] {shift_csv}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

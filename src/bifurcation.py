"""
Bifurcation structure of the dimensionless Casimir-electrostatic NEMS oscillator
================================================================================

    xi'' + 2 zeta xi' + xi = alpha/(1-xi)**2 + beta/(1-xi)**4 ,   xi in [0,1) ,

with alpha, beta, zeta >= 0.  The right-hand side is the (attractive) electrode
force: alpha scales the electrostatic (parallel-plate) term ~1/gap**2 and beta
scales the Casimir term ~1/gap**4, where the dimensionless gap is (1-xi).

Static equilibria are the roots of the net force

        g(xi)  = alpha/(1-xi)**2 + beta/(1-xi)**4 - xi = 0 ,
        g'(xi) = 2 alpha/(1-xi)**3 + 4 beta/(1-xi)**5 - 1 .

A stable equilibrium has g'(xi) < 0  (it is a minimum of the effective
potential V(xi) with V'(xi) = -g(xi)); an unstable one (saddle) has g'(xi) > 0.
Pull-in is the saddle-node (fold) bifurcation where the stable node and the
saddle collide:  g(xi) = 0 AND g'(xi) = 0.

Closed-form fold locus (parametrised by the gap at pull-in u = 1 - xi_PI):

        alpha_c(u) = u**2 (4 - 5 u) / 2 ,
        beta_c(u)  = u**4 (3 u - 2) / 2 .

Physical positivity of both forces restricts u in [2/3, 4/5], i.e.
xi_PI = 1 - u in [1/5, 1/3], with endpoints
    (alpha_c, beta_c) = (4/27, 0)      at u = 2/3 (xi_PI = 1/3, pure electrostatic)
    (alpha_c, beta_c) = (0, 256/3125)  at u = 4/5 (xi_PI = 1/5, pure Casimir).

Pure numpy/scipy, deterministic, importable, no plotting.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.optimize import brentq

__all__ = [
    "net_force",
    "net_force_prime",
    "fold_boundary",
    "equilibria",
    "is_stable_from_rest",
    "stability_boundary_grid",
]

# Reference constants (exact rationals) for the two physical fold endpoints.
ALPHA_ENDPOINT = 4.0 / 27.0          # alpha_c at beta = 0  (u = 2/3, xi_PI = 1/3)
BETA_STAR = 256.0 / 3125.0           # beta_c  at alpha = 0 (u = 4/5, xi_PI = 1/5)

# Physical parameter window for the fold gap u = 1 - xi_PI.
U_MIN = 2.0 / 3.0                    # beta_c >= 0  <=>  u >= 2/3
U_MAX = 4.0 / 5.0                    # alpha_c >= 0 <=>  u <= 4/5

_EPS = 1.0e-12                       # keep 1-xi strictly positive


# ---------------------------------------------------------------------------
# Net force and its derivative
# ---------------------------------------------------------------------------
def net_force(xi, alpha, beta):
    """g(xi) = alpha/(1-xi)^2 + beta/(1-xi)^4 - xi (attractive force minus spring)."""
    u = 1.0 - np.asarray(xi, dtype=float)
    return alpha / u**2 + beta / u**4 - np.asarray(xi, dtype=float)


def net_force_prime(xi, alpha, beta):
    """g'(xi) = 2 alpha/(1-xi)^3 + 4 beta/(1-xi)^5 - 1."""
    u = 1.0 - np.asarray(xi, dtype=float)
    return 2.0 * alpha / u**3 + 4.0 * beta / u**5 - 1.0


# ---------------------------------------------------------------------------
# 1. Closed-form fold (pull-in) boundary
# ---------------------------------------------------------------------------
def fold_boundary(n=400, full=False):
    """Saddle-node (pull-in) locus in closed form.

    Parameters
    ----------
    n : int
        Number of samples of the gap parameter u = 1 - xi_PI.
    full : bool
        If False (default) sweep only the physical, both-forces-positive branch
        u in [2/3, 4/5].  If True sweep the full u in (0, 1] for completeness.

    Returns
    -------
    alpha_c, beta_c, xi_pi, physical : ndarray
        alpha_c(u), beta_c(u), the pull-in displacement xi_PI = 1 - u, and a
        boolean mask that is True where BOTH alpha_c >= 0 and beta_c >= 0
        (i.e. where the fold point corresponds to a physically admissible
        parameter pair).  On the default branch `physical` is all True.
    """
    if full:
        # avoid the essential singularity at u -> 0 (alpha_c, beta_c -> 0 there)
        u = np.linspace(_EPS, 1.0, n)
    else:
        u = np.linspace(U_MIN, U_MAX, n)

    alpha_c = u**2 * (4.0 - 5.0 * u) / 2.0
    beta_c = u**4 * (3.0 * u - 2.0) / 2.0
    xi_pi = 1.0 - u
    physical = (alpha_c >= 0.0) & (beta_c >= 0.0)
    return alpha_c, beta_c, xi_pi, physical


# ---------------------------------------------------------------------------
# 2. Equilibria: all real roots of g(xi)=0, split by stability
# ---------------------------------------------------------------------------
def equilibria(alpha, beta, n_scan=4000, xi_max=1.0 - 1.0e-9, tol=1.0e-9):
    """Find all equilibria xi in [0, xi_max] of g(xi)=0 and classify stability.

    Strategy: dense scan of g on a grid, bracket every sign change (plus exact
    grid zeros), refine with Brent's method, deduplicate, then split by the sign
    of g'(xi): g'<0 -> stable node, g'>=0 -> unstable saddle.

    Returns
    -------
    stable, unstable : ndarray
        Sorted arrays of stable and unstable equilibrium positions.
    """
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha, beta must be non-negative")

    xs = np.linspace(0.0, xi_max, n_scan)
    gs = net_force(xs, alpha, beta)

    roots = []
    for i in range(len(xs) - 1):
        ga, gb = gs[i], gs[i + 1]
        if ga == 0.0:
            roots.append(xs[i])
        elif ga * gb < 0.0:
            try:
                r = brentq(net_force, xs[i], xs[i + 1], args=(alpha, beta),
                           xtol=1.0e-14, rtol=1.0e-14)
                roots.append(r)
            except ValueError:
                pass
    # include right endpoint exact zero (rare)
    if gs[-1] == 0.0:
        roots.append(xs[-1])

    # deduplicate
    roots = np.array(sorted(roots))
    if roots.size:
        keep = np.concatenate(([True], np.diff(roots) > tol))
        roots = roots[keep]

    if roots.size == 0:
        return np.array([]), np.array([])

    gp = net_force_prime(roots, alpha, beta)
    stable = roots[gp < 0.0]
    unstable = roots[gp >= 0.0]
    return stable, unstable


# ---------------------------------------------------------------------------
# 3. Reachability from rest (static pull-in criterion)
# ---------------------------------------------------------------------------
def is_stable_from_rest(alpha, beta, **kw):
    """True if the device settles to a stable equilibrium starting from xi=0.

    Because g(0) = alpha + beta > 0 and g(xi) -> +infinity as xi -> 1, the net
    force pushes the plate outward from rest; the first zero it reaches (the
    smallest root) is always the stable node whenever any equilibrium exists.
    Hence, in the *static* fold sense, the device is stable-from-rest iff a
    stable equilibrium exists, i.e. iff the operating point lies below the fold
    surface.  Above the fold no equilibrium remains and the plate pulls in.
    """
    stable, _ = equilibria(alpha, beta, **kw)
    return bool(stable.size > 0)


# ---------------------------------------------------------------------------
# 4. Boolean pull-in / stable grid (for RK4 phase-diagram cross-check)
# ---------------------------------------------------------------------------
def stability_boundary_grid(alpha_arr, beta_arr, **kw):
    """Boolean grid[i, j] = is_stable_from_rest(alpha_arr[j], beta_arr[i]).

    Rows index beta (y-axis), columns index alpha (x-axis).  True = a stable
    rest state exists; False = the device pulls in from rest.
    """
    A = np.asarray(alpha_arr, dtype=float)
    B = np.asarray(beta_arr, dtype=float)
    grid = np.zeros((B.size, A.size), dtype=bool)
    for i, b in enumerate(B):
        for j, a in enumerate(A):
            grid[i, j] = is_stable_from_rest(a, b, **kw)
    return grid


# ---------------------------------------------------------------------------
# Self-test / result generation
# ---------------------------------------------------------------------------
def _save_fold_csv(path, n=400):
    alpha_c, beta_c, xi_pi, _ = fold_boundary(n=n, full=False)
    u = 1.0 - xi_pi
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.column_stack([u, xi_pi, alpha_c, beta_c])
    np.savetxt(path, data, delimiter=",", header="u,xi_pi,alpha_c,beta_c",
               comments="", fmt="%.12e")
    return path


if __name__ == "__main__":
    print("=" * 70)
    print("Casimir-electrostatic NEMS pull-in: fold bifurcation structure")
    print("=" * 70)

    # (a) fold endpoints and the beta=0 check --------------------------------
    a_c, b_c, xi_pi, phys = fold_boundary(n=400, full=False)
    print("\n[a] Physical fold branch  u in [2/3, 4/5]  (xi_PI in [1/5, 1/3])")
    print(f"    endpoint u=2/3 : alpha_c={a_c[0]:.10f}  beta_c={b_c[0]:.10f}  xi_PI={xi_pi[0]:.6f}")
    print(f"    endpoint u=4/5 : alpha_c={a_c[-1]:.10f}  beta_c={b_c[-1]:.10f}  xi_PI={xi_pi[-1]:.6f}")
    print(f"    reference      : 4/27={ALPHA_ENDPOINT:.10f}  256/3125={BETA_STAR:.10f}")
    assert abs(a_c[0] - ALPHA_ENDPOINT) < 1e-9 and abs(b_c[0]) < 1e-12, "beta=0 endpoint wrong"
    assert abs(b_c[-1] - BETA_STAR) < 1e-9 and abs(a_c[-1]) < 1e-12, "alpha=0 endpoint wrong"
    assert np.all(phys), "physical branch should be all-positive"
    print("    OK: alpha_c(beta=0)=4/27, beta_c(alpha=0)=256/3125 confirmed.")

    # cross-check: closed-form fold point is a true double root of g ----------
    print("\n    residual check g=g'=0 along the closed-form locus:")
    for idx in (0, 200, 399):
        a, b, x = a_c[idx], b_c[idx], xi_pi[idx]
        print(f"      xi_PI={x:.4f}: g={net_force(x,a,b):+.2e}  g'={net_force_prime(x,a,b):+.2e}")
        assert abs(net_force(x, a, b)) < 1e-9 and abs(net_force_prime(x, a, b)) < 1e-9

    # (b) is_stable_from_rest agrees with the fold locus ---------------------
    print("\n[b] is_stable_from_rest vs fold locus (radial scaling by 0.98 / 1.02):")
    ok = True
    for u in (2.0 / 3.0, 0.70, 0.72, 0.75, 0.78, 4.0 / 5.0):
        a = u**2 * (4.0 - 5.0 * u) / 2.0
        b = u**4 * (3.0 * u - 2.0) / 2.0
        # nudge slightly off the exact singular endpoints so both stay >= 0
        a = max(a, 0.0)
        b = max(b, 0.0)
        inside = is_stable_from_rest(0.98 * a, 0.98 * b)
        outside = is_stable_from_rest(1.02 * a, 1.02 * b)
        good = inside and (not outside)
        ok = ok and good
        print(f"    xi_PI={1-u:.4f} (a={a:.5f},b={b:.5f}): "
              f"inside->{inside}  outside->{outside}  {'OK' if good else 'FAIL'}")
    assert ok, "is_stable_from_rest disagrees with the fold locus"
    print("    OK: points just inside are stable, just outside pull in.")

    # sanity: pure-electrostatic node position at low drive
    st, un = equilibria(0.05, 0.0)
    print(f"\n    example (alpha=0.05, beta=0): stable={np.round(st,5)} unstable={np.round(un,5)}")

    # (c) save results/fold_boundary.csv ------------------------------------
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(here, "..", "results", "fold_boundary.csv"))
    _save_fold_csv(csv_path, n=400)
    print(f"\n[c] wrote {csv_path}")
    print("\nAll checks passed.")

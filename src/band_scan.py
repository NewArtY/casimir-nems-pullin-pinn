"""Width of the kinetic-overshoot band as a function of the damping ratio.

The appendix reports the width at four damping ratios and fits the quadratic
closure.  This module resolves the same curve on a finer grid, so that the
approach to ``zeta_c`` can be read rather than inferred from four points.

Method.  For ``alpha`` below the fold the equilibria are the roots of
``xi (1-xi)^2 = alpha``: a node below ``1/3`` and a saddle above.  Crossing the
saddle is a finite-time, decidable event, and past it collapse is certain;
failing to cross, the orbit settles onto the node, which is hyperbolic and
therefore reached on a damping time.  Bisecting on that event gives
``alpha_dyn(zeta)``.

The horizon is the one weakness of the criterion: near ``zeta_c`` the width
shrinks below what a finite horizon resolves, and the scan is therefore cut at
``zeta = 0.35``, where the width is still ``3.6e-4``.  The two values the
appendix obtains by a different construction, at ``zeta = 0.1`` and ``0.3``,
are recomputed here as a check and must agree to three digits.

Run:  python -m src.band_scan
"""

from __future__ import annotations

import csv
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

try:
    from src import config
except ImportError:  # pragma: no cover - direct execution
    from . import config

ALPHA_C = 4.0 / 27.0
T_MAX = 500.0
OUT = os.path.join(str(config.RESULTS_DIR), "band_scan.csv")

# the appendix values, by backward continuation of the strong stable manifold
REFERENCE = {0.10: 1.3482e-2, 0.30: 1.5435e-3}


def saddle(alpha):
    """Upper equilibrium, the larger root of xi (1-xi)^2 = alpha."""
    return brentq(lambda x: x * (1 - x) ** 2 - alpha, 1.0 / 3.0, 1 - 1e-13,
                  xtol=1e-16, rtol=8.9e-16)


def collapses(alpha, zeta):
    """True if the from-rest orbit crosses the saddle."""
    xs = saddle(alpha)

    def rhs(t, y):
        return [y[1], -2 * zeta * y[1] - y[0] + alpha / (1 - y[0]) ** 2]

    def hit(t, y):
        return y[0] - xs
    hit.terminal, hit.direction = True, 1.0

    s = solve_ivp(rhs, (0.0, T_MAX), [0.0, 0.0], method="DOP853",
                  rtol=1e-12, atol=1e-13, events=hit)
    return len(s.t_events[0]) > 0


def width(zeta, iters=48):
    lo, hi = 0.05, ALPHA_C * (1 - 1e-13)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if collapses(mid, zeta):
            hi = mid
        else:
            lo = mid
    return ALPHA_C - 0.5 * (lo + hi)


def main():
    config.set_all_seeds()
    zetas = [0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.24, 0.28, 0.30, 0.32, 0.35]
    rows = []
    print(f"{'zeta':>6} {'alpha_c - alpha_dyn':>21}")
    for z in zetas:
        w = width(z)
        rows.append((z, w))
        print(f"{z:6.2f} {w:21.5e}")

    print("\ncheck against the appendix, which uses a different construction")
    ok = True
    for z, ref in REFERENCE.items():
        got = dict(rows)[z]
        rel = abs(got - ref) / ref
        flag = "ok" if rel < 2e-3 else "MISMATCH"
        ok &= rel < 2e-3
        print(f"  zeta = {z:4.2f}: {got:.5e} against {ref:.5e}, "
              f"relative {rel:.1e}  {flag}")
    if not ok:
        raise SystemExit("the scan disagrees with the appendix; do not use it")

    # quadratic closure: w = C (zeta_c - zeta)^2, fitted on the upper half
    zc = 0.395919
    z = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows])
    sel = z >= 0.20
    C = float(np.mean(w[sel] / (zc - z[sel]) ** 2))
    print(f"\nquadratic closure fitted on zeta >= 0.20:  "
          f"w = {C:.4f} (zeta_c - zeta)^2   with zeta_c = {zc}")

    with open(OUT, "w", newline="", encoding="ascii") as fh:
        wr = csv.writer(fh)
        wr.writerow(["zeta", "band_width", "quadratic_fit"])
        for zz, ww in rows:
            wr.writerow([f"{zz:.4f}", f"{ww:.6e}", f"{C * (zc - zz) ** 2:.6e}"])
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()

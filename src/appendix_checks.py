"""Numerical checks behind the bracket of Appendix B and its closure.

The rest of the archive regenerates the figures and the numbers of the main
text.  The statements of Appendix B that bear on the disagreement with the
literature are checked here instead, because they need no trained model and no
sweep: each one is a small deterministic computation.

Three checks, in the order the appendix uses them.

1. The constant of the barrier theorem.  The hypothesis is
   ``zeta^2 >= 1 - 2 alpha_c(beta) - 4 beta``; the right-hand side is largest
   where ``h(u) = 2 alpha_c + 4 beta`` is smallest along the fold.  That
   minimum sits at ``u = 1/sqrt(2)``, giving the uniform sufficient constant
   ``zeta >= 2^(-1/4)``.  Taking instead the value at ``beta = 0``,
   ``sqrt(19/27)``, would make the theorem false on ``beta`` in [0, 0.033].

2. The damping ``zeta_c`` at which the band closes, by bisection on a
   crossing criterion evaluated in the phase plane, where time does not enter
   and a diverging transit time cannot be mistaken for convergence.

3. The orbit leaves the degenerate equilibrium tangent to the center
   direction, not along the strong direction: ``xi_dot/(xi_f - xi)^2`` tends to
   ``g''(xi_f)/(4 zeta)`` while ``xi_dot/(xi_f - xi)`` falls to zero.  This is
   the step that the two published proofs assign the strong-direction slope.

Run:  python -m src.appendix_checks
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

ALPHA_C0 = 4.0 / 27.0            # fold at beta = 0
XI_F0 = 1.0 / 3.0
EPS = 1e-12


# ---------------------------------------------------------------- fold
def fold(u):
    """Closed-form fold, parametrised by the pull-in gap u in [2/3, 4/5]."""
    return 0.5 * u ** 2 * (4 - 5 * u), 0.5 * u ** 4 * (3 * u - 2)


def h_of_u(u):
    """2 alpha_c + 4 beta along the fold."""
    a, b = fold(u)
    return 2 * a + 4 * b


# ------------------------------------------------- 1. barrier constant
def check_constant():
    u = np.linspace(2 / 3, 4 / 5, 400001)
    i = int(np.argmin(h_of_u(u)))
    zmin = np.sqrt(1 - h_of_u(u)[i])
    print("1. barrier constant")
    print(f"   argmin of h on the fold   u = {u[i]:.9f}   "
          f"(1/sqrt2 = {2 ** -0.5:.9f})")
    print(f"   uniform sufficient zeta      {zmin:.9f}   "
          f"(2^(-1/4) = {2 ** -0.25:.9f})")
    print(f"   value at beta = 0            {np.sqrt(1 - 2 * ALPHA_C0):.9f}   "
          f"(sqrt(19/27) = {np.sqrt(19 / 27):.9f})")
    bad = u[(1 - h_of_u(u)) > 19 / 27]
    b_lo, b_hi = fold(bad.min())[1], fold(bad.max())[1]
    print(f"   the beta = 0 value fails for beta in "
          f"[{min(b_lo, b_hi):.5f}, {max(b_lo, b_hi):.5f}]")


# ------------------------------------- phase-plane crossing criterion
def _g(xi, alpha, beta):
    u = 1.0 - xi
    return alpha / u ** 2 + beta / u ** 4 - xi


def turns_back(alpha, zeta, beta=0.0, xi_f=XI_F0):
    """True if the speed dies before xi_f, i.e. no collapse.

    Along the ascent, with q = xi_dot as a function of xi,
    dq/dxi = g/q - 2 zeta, started from q = sqrt(2 g(0) eps).  No time
    horizon enters, so a diverging transit time cannot be mistaken for
    convergence, which is what a finite horizon in tau would do.

    The reduction needs g > 0 on the whole interval, so it is used here only
    at the fold, where the two equilibria have merged at xi_f and g vanishes
    only there.  Below the fold g has an interior zero at the node, the
    right-hand side is 0/0 there, and this form does not apply; the band
    widths quoted in the appendix come from the backward construction of the
    strong stable manifold instead.
    """
    q0 = np.sqrt(2.0 * _g(EPS, alpha, beta) * EPS)

    def rhs(xi, y):
        return [_g(xi, alpha, beta) / max(y[0], 1e-300) - 2.0 * zeta]

    def stop(xi, y):
        return y[0] - 1e-14
    stop.terminal, stop.direction = True, -1.0

    s = solve_ivp(rhs, (EPS, xi_f - 1e-12), [q0], method="Radau",
                  rtol=1e-13, atol=1e-16, events=stop)
    return len(s.t_events[0]) > 0


# ------------------------------------------------------------ 2. zeta_c
def check_zeta_c():
    f = lambda z: 1.0 if turns_back(ALPHA_C0 * (1 - 1e-12), z) else -1.0
    lo, hi = 0.30, 0.50
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    print(f"\n2. zeta_c at beta = 0        {0.5 * (lo + hi):.6f}")


# ---------------------------------------------------------- 3. tangency
def check_tangency():
    print("\n3. departure from the degenerate equilibrium")
    print(f"   {'zeta':>6} {'xi./(xi_f-xi)^2':>18} {'g\"/(4 zeta)':>14}"
          f" {'xi./(xi_f-xi)':>16}")
    for z in (1.0, 0.8389, 0.45):
        def rhs(t, y):
            return [y[1], -2 * z * y[1] - y[0]
                    + ALPHA_C0 / (1 - y[0]) ** 2]
        s = solve_ivp(rhs, (0.0, 4000.0), [0.0, 0.0], method="DOP853",
                      rtol=1e-13, atol=1e-14, dense_output=True)
        xi, v = s.y[0, -1], s.y[1, -1]
        d = XI_F0 - xi
        print(f"   {z:6.4f} {v / d ** 2:18.4f} {9.0 / (8.0 * z):14.4f}"
              f" {v / d:16.3e}")


def main():
    print("Appendix B: numerical checks of the bracket and its closure\n")
    check_constant()
    check_zeta_c()
    check_tangency()


if __name__ == "__main__":
    main()

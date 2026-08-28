"""Dimensionless Casimir-electrostatic NEMS pull-in model.

Equation of motion (over-damped/under-damped mass-spring plate):

    xi'' = -2 zeta xi' - xi + alpha / (1 - xi)^2 + beta / (1 - xi)^4

with

    * xi     : normalised displacement, xi = x / d, xi in [0, 1)
    * zeta   : damping ratio
    * alpha  : electrostatic control parameter (attractive, ~1/(1-xi)^2)
    * beta   : Casimir control parameter        (attractive, ~1/(1-xi)^4)

The conservative part derives from a potential U(xi):

    net_force(xi) = alpha/(1-xi)^2 + beta/(1-xi)^4 - xi = -dU/dxi

so that the equation of motion reads  xi'' = -2 zeta xi' - dU/dxi.

Integrating -net_force gives, up to an additive constant,

    U(xi) = xi^2 / 2 - alpha/(1 - xi) - beta / (3 (1 - xi)^3).

Note the MINUS signs on the attractive terms: they make U a *decreasing*
function toward xi -> 1 (the pull-in cliff), consistent with an attractive
force that grows without bound at contact.  (A naive +alpha/(1-xi) would give
the wrong force sign; the signs below are the ones for which the
finite-difference identity -U'(xi) == net_force(xi) holds -- see
``verify_potential``.)
"""

from __future__ import annotations

import numpy as np

from . import config


def force_rhs(xi, xidot, alpha, beta, zeta):
    """Right-hand side acceleration xi'' of the equation of motion.

    xi'' = -2 zeta xi' - xi + alpha/(1-xi)^2 + beta/(1-xi)^4
    """
    one_minus = 1.0 - xi
    return (
        -2.0 * zeta * xidot
        - xi
        + alpha / one_minus ** 2
        + beta / one_minus ** 4
    )


def net_force(xi, alpha, beta):
    """Conservative net force = -dU/dxi.

    net_force(xi) = alpha/(1-xi)^2 + beta/(1-xi)^4 - xi

    Its zeros are the mechanical equilibria; the outer (smaller-gap) zero is
    the unstable one that sets the pull-in threshold.
    """
    one_minus = 1.0 - xi
    return alpha / one_minus ** 2 + beta / one_minus ** 4 - xi


def potential(xi, alpha, beta):
    """Analytic potential energy U(xi) with -dU/dxi == net_force(xi).

    U(xi) = xi^2 / 2 - alpha/(1 - xi) - beta / (3 (1 - xi)^3)

    (constant of integration dropped).
    """
    one_minus = 1.0 - xi
    return 0.5 * xi ** 2 - alpha / one_minus - beta / (3.0 * one_minus ** 3)


def verify_potential(alpha=0.07, beta=0.02, xi_grid=None, h=1e-6, rtol=1e-5, atol=1e-7):
    """Finite-difference check that -U'(xi) == net_force(xi).

    Returns
    -------
    (ok, max_abs_err) : tuple[bool, float]
        ``ok`` is True when the central-difference derivative of ``potential``
        matches ``net_force`` to within (rtol, atol) on the whole grid.
    """
    if xi_grid is None:
        xi_grid = np.linspace(0.0, 0.85, 200)
    xi_grid = np.asarray(xi_grid, dtype=float)

    dU = (potential(xi_grid + h, alpha, beta) - potential(xi_grid - h, alpha, beta)) / (2.0 * h)
    minus_dU = -dU
    nf = net_force(xi_grid, alpha, beta)

    abs_err = np.abs(minus_dU - nf)
    ok = bool(np.all(abs_err <= atol + rtol * np.abs(nf)))
    return ok, float(np.max(abs_err))


def beta_lifshitz(beta, xi, T, d):
    """Finite-temperature (Lifshitz) correction to the Casimir parameter beta.

    beta -> beta * [ 1 + (60 zeta(3) / pi^3) * k_B T d (1 - xi) / (hbar c) ]

    The physical gap is (d - x) = d (1 - xi); the correction is the leading
    thermal term of the Lifshitz formula for the Casimir pressure between
    ideal plates.  For T = 0 it returns beta unchanged.

    Parameters
    ----------
    beta : float or array
        Zero-temperature Casimir control parameter.
    xi : float or array
        Normalised displacement in [0, 1).
    T : float
        Temperature [K].
    d : float
        Rest gap [m].
    """
    if T == 0.0 or d is None:
        return beta
    prefac = 60.0 * config.ZETA3 / np.pi ** 3
    gap = d * (1.0 - xi)
    correction = 1.0 + prefac * config.K_B * T * gap / (config.HBAR * config.C_LIGHT)
    return beta * correction


# ---------------------------------------------------------------------------
# Static fold (saddle-node) curve, parametrised by the equilibrium position u.
# ---------------------------------------------------------------------------
def fold_alpha_beta(u):
    """Fold curve (alpha_c, beta_c) parametrised by equilibrium position u.

        alpha_c(u) = u^2 (4 - 5 u) / 2
        beta_c(u)  = u^4 (3 u - 2) / 2

    Points (alpha, beta) *above* this curve have no mechanical equilibrium
    (the electrostatic+Casimir attraction overwhelms the restoring force) and
    the device pulls in; points below retain a stable equilibrium.
    Physical (beta_c > 0) requires u > 2/3.
    """
    u = np.asarray(u, dtype=float)
    alpha_c = u ** 2 * (4.0 - 5.0 * u) / 2.0
    beta_c = u ** 4 * (3.0 * u - 2.0) / 2.0
    return alpha_c, beta_c


# ---------------------------------------------------------------------------
# Undamped from-rest pull-in curve (McLellan, Medina, Xu & Yang, ZAMM 96, 1406
# (2016), Eq. (4.14) with a = 1, i.e. no dielectric layer).
# ---------------------------------------------------------------------------
def fold_undamped(x):
    """Closed-form undamped from-rest pull-in curve, parametrised by x.

    Released from rest, an undamped actuator collapses when the energy at the
    saddle no longer exceeds the energy at the origin.  That double-root
    condition is linear in (alpha, beta) and was solved in closed form by
    McLellan et al.; ``x`` is the pull-in displacement and runs from the root
    of alpha = 0 up to x = 1/2, where beta = 0 and alpha = 1/8.

    Parameters
    ----------
    x : float or array
        Pull-in displacement, in (0, 1/2].

    Returns
    -------
    (alpha, beta) : tuple of arrays
        The undamped boundary. It bounds the from-rest threshold from below at
        every damping ratio, the static fold bounding it from above.
    """
    x = np.asarray(x, dtype=float)
    den = 2.0 * (3.0 - x)
    alpha = (-3.0 + 15.0 * x - 20.0 * x ** 2 + 10.0 * x ** 3
             - 2.0 * x ** 4) / den
    beta = 3.0 * (1.0 - x) ** 3 * (1.0 - 2.0 * x) / den
    return alpha, beta


def fold_undamped_x_range(n=400):
    """Sample x over the physical branch of :func:`fold_undamped`.

    The branch runs from the root of alpha(x) = 0 to x = 1/2 (beta = 0).
    """
    from scipy.optimize import brentq

    def a_of_x(x):
        return fold_undamped(x)[0]

    x_lo = brentq(a_of_x, 1e-9, 0.5 - 1e-12, xtol=1e-14)
    return np.linspace(x_lo, 0.5, n)

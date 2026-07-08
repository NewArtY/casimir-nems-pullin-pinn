"""Fixed-step classical RK4 integrator for the pull-in ODE with event detection.

The 2nd-order equation

    xi'' = -2 zeta xi' - xi + alpha/(1-xi)^2 + beta/(1-xi)^4

is written as the first-order system for the state y = (xi, v = xi'):

    xi' = v
    v'  = force_rhs(xi, v, alpha, beta, zeta)

with an optional finite-temperature Lifshitz correction to beta that depends
on the instantaneous displacement (see ``physics.beta_lifshitz``).

Pull-in event detection stops the integration when the plate reaches the
contact barrier xi >= 1 - eps, or escapes the physical domain [0, 1).  The
trajectory is classified into one of three regimes:

    * 'pullin'  : trajectory reaches the contact barrier,
    * 'stable'  : bounded oscillation that decays / settles,
    * 'growing' : amplitude grows toward pull-in but has not reached the
                  barrier within t_max.
"""

from __future__ import annotations

import numpy as np

from .physics import force_rhs, beta_lifshitz


def _deriv(xi, v, alpha, beta, zeta):
    return v, force_rhs(xi, v, alpha, beta, zeta)


def rk4_integrate(
    alpha,
    beta,
    zeta,
    xi0,
    v0,
    t_max,
    dt,
    T=0.0,
    d=None,
    eps_barrier=1e-6,
    escape_low=-1e-9,
):
    """Integrate the pull-in ODE with fixed-step RK4 and pull-in detection.

    Parameters
    ----------
    alpha, beta, zeta : float
        Electrostatic, Casimir and damping parameters.
    xi0, v0 : float
        Initial displacement and velocity.
    t_max, dt : float
        Integration horizon and fixed step.
    T : float, optional
        Temperature [K]; if > 0 the Casimir parameter is Lifshitz-corrected
        at every displacement via ``beta_lifshitz``.
    d : float or None, optional
        Rest gap [m] required for the Lifshitz correction (ignored if T == 0).
    eps_barrier : float, optional
        Pull-in fires when xi >= 1 - eps_barrier.
    escape_low : float, optional
        Lower guard: integration stops if xi < escape_low (escapes [0,1)).

    Returns
    -------
    t, xi, v : np.ndarray
        Time, displacement and velocity samples (up to and including the
        stopping event).
    info : dict
        {'pull_in': bool, 'tau_pullin': float | None,
         'regime': 'stable'|'growing'|'pullin', 'xi_max': float,
         'escaped': bool, 'n_steps': int}
    """
    n = int(np.ceil(t_max / dt))
    t = np.empty(n + 1, dtype=float)
    xi = np.empty(n + 1, dtype=float)
    v = np.empty(n + 1, dtype=float)

    t[0], xi[0], v[0] = 0.0, float(xi0), float(v0)

    barrier = 1.0 - eps_barrier
    pull_in = False
    escaped = False
    tau_pullin = None
    last = 0

    def beta_eff(x):
        return beta_lifshitz(beta, x, T, d) if (T and d is not None) else beta

    for i in range(n):
        x0, u0 = xi[i], v[i]

        # Guard: if we are already numerically at the cliff, stop cleanly.
        if x0 >= barrier:
            pull_in = True
            tau_pullin = t[i]
            last = i
            break

        b0 = beta_eff(x0)
        k1x, k1v = _deriv(x0, u0, alpha, b0, zeta)

        xa, ua = x0 + 0.5 * dt * k1x, u0 + 0.5 * dt * k1v
        k2x, k2v = _deriv(xa, ua, alpha, beta_eff(xa), zeta)

        xb, ub = x0 + 0.5 * dt * k2x, u0 + 0.5 * dt * k2v
        k3x, k3v = _deriv(xb, ub, alpha, beta_eff(xb), zeta)

        xc, uc = x0 + dt * k3x, u0 + dt * k3v
        k4x, k4v = _deriv(xc, uc, alpha, beta_eff(xc), zeta)

        x_new = x0 + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        u_new = u0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        # Blow-up guard: NaN/inf or overshoot past the cliff -> treat as pull-in.
        if not np.isfinite(x_new) or not np.isfinite(u_new) or x_new >= barrier:
            # Linearly interpolate the crossing time to xi = barrier.
            denom = (x_new - x0)
            if np.isfinite(x_new) and denom > 0:
                frac = (barrier - x0) / denom
                frac = min(max(frac, 0.0), 1.0)
            else:
                frac = 1.0
            tau_pullin = t[i] + frac * dt
            xi[i + 1] = barrier
            v[i + 1] = u_new if np.isfinite(u_new) else u0
            t[i + 1] = t[i] + dt
            pull_in = True
            last = i + 1
            break

        # Escape below the physical domain.
        if x_new < escape_low:
            escaped = True
            xi[i + 1] = x_new
            v[i + 1] = u_new
            t[i + 1] = t[i] + dt
            last = i + 1
            break

        xi[i + 1] = x_new
        v[i + 1] = u_new
        t[i + 1] = t[i] + dt
        last = i + 1
    else:
        last = n

    t = t[: last + 1]
    xi = xi[: last + 1]
    v = v[: last + 1]

    xi_max = float(np.max(xi))

    if pull_in:
        regime = "pullin"
    else:
        regime = _classify_regime(xi, v, dt, escaped)

    info = {
        "pull_in": bool(pull_in),
        "tau_pullin": (float(tau_pullin) if tau_pullin is not None else None),
        "regime": regime,
        "xi_max": xi_max,
        "escaped": bool(escaped),
        "n_steps": int(last),
    }
    return t, xi, v, info


def _local_maxima(xi):
    """Indices of interior local maxima of the xi trajectory."""
    if xi.size < 3:
        return np.empty(0, dtype=int)
    left = xi[1:-1] > xi[:-2]
    right = xi[1:-1] >= xi[2:]
    return np.where(left & right)[0] + 1


def _classify_regime(xi, v, dt, escaped, settle_tol=5e-3, margin=0.05, v_tol=1e-4):
    """Classify a non-pull-in trajectory as 'stable' or 'growing'.

    Heuristic
    ---------
    * If the last ~15% of the trajectory has settled (peak-to-peak below
      ``settle_tol``) -> 'stable'.
    * Else, using successive oscillation peaks: increasing envelope ->
      'growing'; decaying or roughly constant (conservative) -> 'stable'.
    * If the motion is essentially monotonic (a saddle-node "ghost" creep
      toward the barrier), a trajectory still climbing at the end
      (xi near its running max and v > v_tol) -> 'growing'; otherwise
      'stable'.
    """
    n = xi.size
    if n < 3:
        return "stable"

    # Settled?  Look at peak-to-peak over the tail window.
    tail = max(3, int(0.15 * n))
    ptp_tail = float(np.max(xi[-tail:]) - np.min(xi[-tail:]))
    if ptp_tail < settle_tol and abs(v[-1]) < 10 * v_tol:
        return "stable"

    peaks = _local_maxima(xi)
    if peaks.size >= 2:
        first_peak = xi[peaks[0]]
        last_peak = xi[peaks[-1]]
        # Compare against a small positive reference to avoid divide issues.
        ref = max(abs(first_peak), 1e-3)
        if last_peak > first_peak + margin * ref:
            return "growing"
        return "stable"

    # (Near-)monotonic trajectory: is it still climbing toward the barrier?
    running_max = float(np.max(xi))
    still_climbing = (xi[-1] >= running_max - settle_tol) and (v[-1] > v_tol)
    rose = (running_max - xi[0]) > settle_tol
    if still_climbing and rose:
        return "growing"
    return "stable"

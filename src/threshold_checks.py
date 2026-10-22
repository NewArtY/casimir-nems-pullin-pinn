"""Convergence and independence checks for the from-rest threshold.

The from-rest threshold alpha_dyn(beta; zeta) is located elsewhere in the
archive by bisection on a classification of integrated orbits.  Near the
threshold the pull-in time diverges, so the tolerance of a bisection is not by
itself the accuracy of the threshold.  This module separates the two and
checks the threshold three independent ways.

Methods
-------
A. ``rk4``: the fixed-step RK4 bisection used for Fig. 3 and the phase
   diagram (``src.rk4.rk4_integrate``), at several steps h and horizons T.
   An orbit counts as collapsing if it reaches xi = 1 - 1e-6 before T.

B. ``cert``: adaptive DOP853 with a classification by two finite-time
   certificates, so that no horizon enters.  Crossing the saddle xi_s with
   positive speed is a certificate of collapse (beyond the saddle g > 0 and
   there is no equilibrium).  The energy E = xi_dot^2/2 + V(xi) falling below
   V(xi_s) while xi < xi_s is a certificate of no collapse, because E is
   nonincreasing.  Every orbit off the stable manifold of the saddle meets one
   of the two in finite time.

C. ``manifold``: the stable manifold of the saddle, continued backward in
   time from the saddle to the release point.  The release point lies below
   it (no collapse) when the manifold reaches xi = xi_0 with positive speed,
   and above it (collapse) when the manifold turns at xi_t > xi_0.  The signed
   quantity is zero exactly at the threshold and is found by Brent's method.
   No forward orbit is integrated at all.

Further checks
--------------
* band widths alpha_c - alpha_dyn at beta = 0 on the zeta grid of Table S8,
  extended toward zeta_c, by B and C;
* the slope d alpha_dyn / d beta at zeta = 0.1 by C, with Richardson
  extrapolation, against the surrogate of Fig. 3(b);
* zeta_c(beta) for release from xi = 0 and from the zero-bias rest position
  xi_r, by the phase-plane criterion at the fold (no horizon);
* zeta_0(beta) for beta > beta_d, the damping below which the from-rest orbit
  collapses at zero bias, so that alpha_dyn = alpha_MMXY = 0 there;
* the counterexample to the limit of Flores (2017): at beta = 0 and finite
  damping above zeta_c, biases within 1e-8 of 4/27 do not collapse;
* the delayed pull-in of Fig. 1(b): a point above the fold that passes the
  saddle-node bottleneck slowly before collapsing.

Writes ``results/threshold_checks.json``.  Run:  python -m src.threshold_checks
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

try:
    from src import config
    from src.rk4 import rk4_integrate
    from src.damping_regimes import (g, potential, fold_of_beta, saddle,
                                     alpha_mmxy, zero_bias_equilibrium,
                                     dies_at_fold)
except ImportError:  # pragma: no cover - package-relative execution
    from . import config
    from .rk4 import rk4_integrate
    from .damping_regimes import (g, potential, fold_of_beta, saddle,
                                  alpha_mmxy, zero_bias_equilibrium,
                                  dies_at_fold)

OUT = os.path.join(str(config.RESULTS_DIR), "threshold_checks.json")
OUT_ZC = os.path.join(str(config.RESULTS_DIR), "zeta_c_scan.json")
print = functools.partial(print, flush=True)  # noqa: A001 - unbuffered log
BETA_D = 0.0705227          # end of the closed-form undamped threshold
BETA_STAR = 256.0 / 3125.0


def gprime(xi, alpha, beta):
    u = 1.0 - xi
    return 2.0 * alpha / u ** 3 + 4.0 * beta / u ** 5 - 1.0


def _lower(beta):
    """A bias known not to collapse for zeta > 0: the undamped threshold.

    Above beta_d that threshold is zero and the zero-bias orbit may collapse;
    the callers test that case explicitly.
    """
    return alpha_mmxy(beta) if beta < BETA_D else 1e-13


# ---------------------------------------------------------------- method A
def collapses_rk4(alpha, beta, zeta, h, T):
    _, _, _, info = rk4_integrate(alpha=float(alpha), beta=float(beta),
                                  zeta=float(zeta), xi0=0.0, v0=0.0,
                                  t_max=float(T), dt=float(h),
                                  eps_barrier=1e-6)
    return bool(info["pull_in"])


def alpha_dyn_rk4(beta, zeta, h, T, tol=1e-6):
    lo, hi = 0.0, 0.20
    if collapses_rk4(lo, beta, zeta, h, T):
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if collapses_rk4(mid, beta, zeta, h, T):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- method B
def collapses_cert(alpha, beta, zeta, xi0=0.0, rtol=1e-12):
    """Horizon-free classification by the two certificates (see module doc)."""
    a_c, _ = fold_of_beta(beta)
    if alpha >= a_c:
        return True                      # no equilibrium: Theorem 1, upper edge
    xs = saddle(alpha, beta)
    Vs = potential(xs, alpha, beta)
    if potential(xi0, alpha, beta) < Vs:
        return False                     # below the undamped threshold

    def rhs(t, y):
        return [y[1], -2.0 * zeta * y[1] + g(y[0], alpha, beta)]

    def cross(t, y):
        return y[0] - xs
    cross.terminal, cross.direction = True, 1.0

    def safe(t, y):
        return 0.5 * y[1] ** 2 + potential(y[0], alpha, beta) - Vs
    safe.terminal, safe.direction = True, -1.0

    s = solve_ivp(rhs, (0.0, 1e6), [xi0, 0.0], method="DOP853",
                  rtol=rtol, atol=rtol * 1e-2, events=(cross, safe))
    if len(s.t_events[0]):
        return True
    if len(s.t_events[1]):
        return False
    raise RuntimeError("no certificate reached")


def alpha_dyn_cert(beta, zeta, xi0=0.0, rtol=1e-12, iters=44):
    a_c, _ = fold_of_beta(beta)
    lo = _lower(beta)
    if collapses_cert(lo, beta, zeta, xi0, rtol):
        return 0.0
    hi = a_c * (1.0 - 1e-13)
    if not collapses_cert(hi, beta, zeta, xi0, rtol):
        return a_c
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if collapses_cert(mid, beta, zeta, xi0, rtol):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- method C
def manifold_signal(alpha, beta, zeta, xi0=0.0, eps=1e-9):
    """Signed position of the release point relative to the stable manifold.

    Positive: the manifold reaches xi0 with that positive speed, and the
    release point (xi0, 0) lies below it, so the orbit does not collapse.
    Negative: the manifold turns at xi_t > xi0; the value is -(xi_t - xi0),
    and the release point lies above it, so the orbit collapses.
    """
    xs = saddle(alpha, beta)
    gp = gprime(xs, alpha, beta)
    lam_s = -zeta - np.sqrt(zeta ** 2 + gp)           # stable eigenvalue
    y0 = [xs - eps, -eps * lam_s]                     # left branch, v > 0

    def rhs_back(t, y):
        return [-y[1], 2.0 * zeta * y[1] - g(y[0], alpha, beta)]

    def reach(t, y):
        return y[0] - xi0
    reach.terminal, reach.direction = True, -1.0

    def turn(t, y):
        return y[1]
    turn.terminal, turn.direction = True, -1.0

    s = solve_ivp(rhs_back, (0.0, 1e5), y0, method="DOP853",
                  rtol=1e-13, atol=1e-15, events=(reach, turn))
    if len(s.t_events[0]):
        return float(s.y_events[0][0][1])
    if len(s.t_events[1]):
        return -float(s.y_events[1][0][0] - xi0)
    raise RuntimeError("backward manifold reached neither event")


def alpha_dyn_manifold(beta, zeta, xi0=0.0):
    a_c, _ = fold_of_beta(beta)
    hi = a_c * (1.0 - 1e-12)
    if manifold_signal(hi, beta, zeta, xi0) > 0:
        return a_c                       # no overshoot even at the fold
    lo = _lower(beta) if xi0 == 0.0 else 1e-13
    if manifold_signal(lo, beta, zeta, xi0) < 0:
        return 0.0                       # collapses at zero bias
    return brentq(lambda a: manifold_signal(a, beta, zeta, xi0), lo, hi,
                  xtol=1e-15, rtol=4e-15, maxiter=200)


# ------------------------------------------------------------ zeta_c(beta)
def zeta_c_scan(beta, xi_start, lo=0.05, hi=0.70, iters=24):
    def dies(z):
        return dies_at_fold(beta, xi_start, lambda x: z)
    if dies(lo) or not dies(hi):
        raise RuntimeError(f"zeta_c(beta={beta}) outside [{lo}, {hi}]")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if dies(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- zeta_0(beta)
def zeta_0(beta, iters=40):
    """Largest damping at which the zero-bias from-rest orbit still collapses."""
    lo, hi = 0.0, 1.0
    if not collapses_cert(1e-14, beta, 1e-12):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if collapses_cert(1e-14, beta, mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------- the report
def section(title):
    print("\n" + title)
    print("-" * len(title))


def fast_checks():
    """Parts 1-5 and 7: a few minutes."""
    config.set_all_seeds()
    out = {}
    t_start = time.time()

    # ---- 1. the threshold at zeta = 0.1 by three methods --------------------
    section("1. alpha_dyn(beta) at zeta = 0.1, three methods")
    zeta = 0.1
    betas = [0.0, 0.01, 0.03, 0.05, 0.07]
    rk4_cfg = [(1e-2, 200.0), (5e-3, 200.0), (2.5e-3, 200.0), (5e-3, 400.0),
               (5e-3, 100.0)]
    rows = []
    for b in betas:
        am = alpha_dyn_manifold(b, zeta)
        ac10 = alpha_dyn_cert(b, zeta, rtol=1e-10)
        ac12 = alpha_dyn_cert(b, zeta, rtol=1e-12)
        rk = {f"h={h:g},T={T:g}": alpha_dyn_rk4(b, zeta, h, T)
              for h, T in rk4_cfg}
        row = dict(beta=b, manifold=am, cert_rtol1e10=ac10,
                   cert_rtol1e12=ac12, rk4=rk,
                   alpha_c=fold_of_beta(b)[0], alpha_mmxy=alpha_mmxy(b))
        rows.append(row)
        dev = max(abs(v - am) for v in rk.values())
        print(f"  beta={b:5.3f}  manifold {am:.10f}  cert(1e-12) "
              f"{ac12 - am:+.1e}  cert(1e-10) {ac10 - am:+.1e}  "
              f"max|rk4 - manifold| {dev:.1e}")
        for k, v in rk.items():
            print(f"        rk4 {k:>15}: {v:.7f}  ({v - am:+.1e})")
    out["zeta0.1"] = rows

    # ---- 2. band widths at beta = 0 -----------------------------------------
    section("2. band width alpha_c - alpha_dyn at beta = 0")
    zetas = [0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.24, 0.28, 0.30, 0.32,
             0.35, 0.37, 0.38, 0.39]
    a_c0 = 4.0 / 27.0
    band = []
    for z in zetas:
        wm = a_c0 - alpha_dyn_manifold(0.0, z)
        wc = a_c0 - alpha_dyn_cert(0.0, z)
        band.append(dict(zeta=z, width_manifold=wm, width_cert=wc))
        print(f"  zeta={z:4.2f}  manifold {wm:.6e}  cert {wc:.6e}  "
              f"rel diff {abs(wm - wc) / wm:.1e}")
    out["band_beta0"] = band

    # ---- 3. slope of the threshold ------------------------------------------
    section("3. d alpha_dyn / d beta at zeta = 0.1 (method C)")
    slopes = []
    for b in (0.01, 0.03, 0.05):
        d1 = (alpha_dyn_manifold(b + 2.5e-3, 0.1)
              - alpha_dyn_manifold(b - 2.5e-3, 0.1)) / 5e-3
        d2 = (alpha_dyn_manifold(b + 1.25e-3, 0.1)
              - alpha_dyn_manifold(b - 1.25e-3, 0.1)) / 2.5e-3
        rich = (4.0 * d2 - d1) / 3.0
        slopes.append(dict(beta=b, fd_h2p5e3=d1, fd_h1p25e3=d2,
                           richardson=rich))
        print(f"  beta={b:4.2f}  h=2.5e-3 {d1:.6f}  h=1.25e-3 {d2:.6f}  "
              f"Richardson {rich:.6f}")
    out["slopes"] = slopes

    # ---- 4. zeta_0(beta) above beta_d ---------------------------------------
    section("4. zeta_0(beta): zero-bias collapse below this damping")
    z0 = []
    for b in (0.072, 0.075, 0.078, 0.080):
        v = zeta_0(b)
        z0.append(dict(beta=b, zeta_0=v))
        print(f"  beta={b:5.3f}  zeta_0 = {v:.6f}")
    # the example quoted in the text
    s = solve_ivp(lambda t, y: [y[1], -2e-3 * y[1] + g(y[0], 0.0, 0.08)],
                  (0, 100), [0.0, 0.0], method="DOP853", rtol=1e-12,
                  atol=1e-14,
                  events=lambda t, y: y[0] - 0.98)
    out["zeta_0"] = z0
    out["example_beta0.08_zeta1e-3_tau_0.98"] = float(s.t_events[0][0])
    print(f"  beta=0.08, zeta=1e-3, alpha=0 reaches xi=0.98 at tau="
          f"{s.t_events[0][0]:.4f}")

    # ---- 5. counterexample to the limit of Flores (2017) --------------------
    section("5. beta = 0: biases within 1e-8 of 4/27 at finite damping")
    ce = []
    for z in (0.45, 0.50, 0.70):
        for k in (4, 6, 8):
            a = a_c0 * (1.0 - 10.0 ** (-k))
            c = collapses_cert(a, 0.0, z)
            ce.append(dict(zeta=z, alpha_F=2 * z, rel_below=10.0 ** (-k),
                           collapses=c))
            print(f"  zeta={z:4.2f} (alpha_F={2 * z:4.2f})  "
                  f"alpha=4/27(1-1e-{k})  collapses: {c}")
    out["counterexample"] = ce

    # ---- 7. delayed pull-in of Fig. 1(b) ------------------------------------
    section("7. Fig. 1(b): delayed pull-in above the fold")
    a, b, z = 0.072421875, 0.0407373046875, 1.0
    a_c, xi_f = fold_of_beta(b)
    u_f = 1.0 - xi_f
    K = 2 * np.pi * z * u_f ** 2 * np.sqrt(2.0 / (15 * u_f - 8))

    def rhs(t, y):
        return [y[1], -2 * z * y[1] + g(y[0], a, b)]
    # 1 - 1e-4: at tau ~ 39 the float spacing in time stops an
    # error-controlled step near 1 - 1e-5; the remaining time is ~1e-8
    ev = lambda t, y: y[0] - (1 - 1e-4)
    ev.terminal = True
    s = solve_ivp(rhs, (0, 500), [0.0, 0.0], method="DOP853", rtol=1e-12,
                  atol=1e-14, events=ev, dense_output=True)
    tau_star = float(s.t_events[0][0])
    tt = np.linspace(0, tau_star, 400001)
    xx = s.sol(tt)[0]
    vv = s.sol(tt)[1]
    i_min = int(np.argmin(vv[tt > 1.0]) + np.searchsorted(tt, 1.0))
    near = np.abs(xx - xi_f) < 0.05
    t_near = float(np.sum(near) * (tt[1] - tt[0]))
    rec = dict(alpha=a, beta=b, zeta=z, alpha_c=a_c, xi_f=xi_f,
               excess=a - a_c, tau_star=tau_star, K=K,
               tau_bottleneck_leading=K / np.sqrt(a - a_c),
               xi_at_30=float(s.sol(30.0)[0]),
               v_min=float(vv[i_min]), xi_at_vmin=float(xx[i_min]),
               time_within_005_of_xi_f=t_near)
    out["delayed_pullin"] = rec
    for k, v in rec.items():
        print(f"  {k:>26}: {v}")

    out["wall_time_s"] = time.time() - t_start
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[save] {OUT}  ({out['wall_time_s']:.0f} s)")


ZC_BETAS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)


def zeta_c_checks(betas=ZC_BETAS, out_path=OUT_ZC):
    """Part 6: zeta_c(beta) by the phase-plane criterion; slow (Radau)."""
    config.set_all_seeds()
    t_start = time.time()
    section("6. zeta_c(beta), release from 0 and from xi_r")
    zc = []
    for b in betas:
        xr = zero_bias_equilibrium(b)
        z_from0 = zeta_c_scan(b, 0.0)
        z_fromr = zeta_c_scan(b, xr) if b > 0 else z_from0
        zc.append(dict(beta=b, xi_r=xr, zeta_c_from0=z_from0,
                       zeta_c_from_xir=z_fromr))
        print(f"  beta={b:4.2f}  xi_r={xr:.4f}  zeta_c(0)={z_from0:.6f}  "
              f"zeta_c(xi_r)={z_fromr:.6f}")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(dict(zeta_c=zc, bisection_steps=24,
                           bracket=[0.05, 0.70],
                           wall_time_s=time.time() - t_start), fh, indent=2)
    print(f"[save] {out_path}")


def main(argv=()):
    """Run all parts; from the command line, ``fast`` or ``zetac [betas]``.

    ``zetac 0.0,0.03,0.06`` runs a subset of beta and writes
    ``results/zeta_c_scan_<betas>.json``, so the slow part can be split
    over processes; Table S12 merges every ``zeta_c_scan*.json``.
    """
    part = argv[0] if argv else "all"
    if part in ("fast", "all"):
        fast_checks()
    if part in ("zetac", "all"):
        if len(argv) > 1:              # a subset, for parallel runs
            betas = tuple(float(x) for x in argv[1].split(","))
            tag = argv[1].replace(",", "_")
            zeta_c_checks(betas, OUT_ZC.replace(".json", f"_{tag}.json"))
        else:
            zeta_c_checks()


if __name__ == "__main__":
    main(sys.argv[1:])

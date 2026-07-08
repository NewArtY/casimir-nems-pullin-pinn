"""Pull-in PHASE DIAGRAM (Fig. 2) for the Casimir-electrostatic NEMS model.

Dimensionless equation of motion, released from rest:

    xi'' + 2 zeta xi' + xi = alpha/(1-xi)^2 + beta/(1-xi)^4 ,
    xi(0) = 0 , xi'(0) = 0 , xi in [0, 1) .

For every point of an (alpha, beta) grid we integrate the ODE with a
deterministic fixed-step RK4 and detect the pull-in event (xi -> 1).  This
yields

    * the DYNAMIC pull-in region (RK4-from-rest),
    * the time-to-pull-in map tau_PI(alpha, beta),
    * the maximum displacement max_xi(alpha, beta),

which we overlay on the ANALYTIC static fold (saddle-node) locus from
``src.bifurcation`` (``fold_boundary`` + ``is_stable_from_rest``).  Because a
plate released from rest overshoots its quasi-static equilibrium, the dynamic
pull-in boundary sits slightly INSIDE the static fold (lower alpha, beta); we
quantify that gap.

Design / efficiency notes
-------------------------
The reference integrator ``src.rk4.rk4_integrate`` is a per-trajectory Python
loop.  A 200x200 grid x ~O(1e4) steps is ~1.6e9 scalar RK4 evaluations, which
is intractable in pure Python.  We therefore VECTORISE: the whole grid is
advanced simultaneously as flat numpy state vectors, reproducing the *exact*
deterministic event-detection semantics of ``rk4_integrate`` (same barrier,
same linear crossing-time interpolation, same escape guard).  Points that have
pulled in / escaped are compacted out of the active set so cost falls as the
front advances.  We cross-validate the vectorised result against the reference
``rk4_integrate`` on a handful of sample points (must match to machine noise).

This file only READS the existing modules; it writes only NEW output files
(results/phase_diagram.npz, results/phase_boundary_static.csv, and a small
logs/phase_diagram.json provenance record).

Run from the ``code/`` directory:

    python src/phase_diagram.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# Make ``code/`` importable so ``from src.* import ...`` works from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np  # noqa: E402

from src.config import set_all_seeds, PROJECT_ROOT  # noqa: E402
from src.bifurcation import (  # noqa: E402
    fold_boundary,
    is_stable_from_rest,
    ALPHA_ENDPOINT,
    BETA_STAR,
)
from src.rk4 import rk4_integrate  # noqa: E402  (reference cross-check)


# ---------------------------------------------------------------------------
# Grid / integration configuration
# ---------------------------------------------------------------------------
# Grid spans the physical fold endpoints:
#   alpha_c = 4/27 ~ 0.1481 at beta = 0   (pure electrostatic)
#   beta*   = 256/3125 ~ 0.0819 at alpha = 0 (pure Casimir)
ALPHA_MIN, ALPHA_MAX = 0.0, 0.20
BETA_MIN, BETA_MAX = 0.0, 0.09
N_ALPHA = 200
N_BETA = 200

DT = 5.0e-3
T_MAX = 200.0

# Representative damping.  A real underdamped NEMS has zeta ~ 1e-4, but that
# demands t_max ~ 1e4 to resolve the slow ring-down and is not needed to locate
# the pull-in *front*.  zeta = 0.1 keeps the plate clearly underdamped (so the
# kinetic overshoot that separates the dynamic from the static boundary is
# visible) while keeping settling times ~O(10) and the grid tractable.  We also
# run a more strongly damped zeta = 0.7 for comparison: heavier damping kills
# overshoot and pushes the dynamic boundary back toward the static fold.
ZETA_PRIMARY = 0.1
ZETA_COMPARE = 0.7

# Event-detection constants -- identical to src.rk4.rk4_integrate defaults.
EPS_BARRIER = 1.0e-6
ESCAPE_LOW = -1.0e-9
BARRIER = 1.0 - EPS_BARRIER


# ---------------------------------------------------------------------------
# Vectorised deterministic RK4 over the whole (alpha, beta) grid
# ---------------------------------------------------------------------------
def _force(xi, v, alpha, beta, zeta):
    """xi'' = -2 zeta v - xi + alpha/(1-xi)^2 + beta/(1-xi)^4 (elementwise)."""
    om = 1.0 - xi
    om2 = om * om
    return -2.0 * zeta * v - xi + alpha / om2 + beta / (om2 * om2)


def integrate_grid(alpha_flat, beta_flat, zeta, dt=DT, t_max=T_MAX,
                   eps_barrier=EPS_BARRIER, escape_low=ESCAPE_LOW):
    """Advance every grid point from rest with vectorised RK4 + pull-in events.

    Reproduces ``src.rk4.rk4_integrate`` semantics exactly, in parallel over all
    points.  Deterministic: the result depends only on (alpha, beta, zeta, dt,
    t_max) -- no randomness anywhere.

    Returns
    -------
    pulled : bool  ndarray (N,)   -- pulled in within t_max
    tau    : float ndarray (N,)   -- time to pull-in (NaN if not pulled in)
    ximax  : float ndarray (N,)   -- max displacement reached
    escaped: bool  ndarray (N,)   -- left the physical domain below 0
    """
    a = np.asarray(alpha_flat, dtype=float)
    b = np.asarray(beta_flat, dtype=float)
    N = a.size
    barrier = 1.0 - eps_barrier

    xi = np.zeros(N, dtype=float)      # released from rest
    v = np.zeros(N, dtype=float)
    ximax = np.zeros(N, dtype=float)
    tau = np.full(N, np.nan, dtype=float)
    pulled = np.zeros(N, dtype=bool)
    escaped = np.zeros(N, dtype=bool)
    active = np.ones(N, dtype=bool)

    n_steps = int(np.ceil(t_max / dt))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i in range(n_steps):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            cur_t = i * dt

            x = xi[idx]
            u = v[idx]
            aa = a[idx]
            bb = b[idx]

            # Classical RK4 stage evaluations (intermediates unguarded, exactly
            # as in the scalar reference integrator).
            k1x = u
            k1v = _force(x, u, aa, bb, zeta)
            xa = x + 0.5 * dt * k1x
            ua = u + 0.5 * dt * k1v
            k2x = ua
            k2v = _force(xa, ua, aa, bb, zeta)
            xb = x + 0.5 * dt * k2x
            ub = u + 0.5 * dt * k2v
            k3x = ub
            k3v = _force(xb, ub, aa, bb, zeta)
            xc = x + dt * k3x
            uc = u + dt * k3v
            k4x = uc
            k4v = _force(xc, uc, aa, bb, zeta)

            x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            u_new = u + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

            finite = np.isfinite(x_new) & np.isfinite(u_new)

            # --- pull-in: non-finite OR crossed the contact barrier ---------
            bad = (~finite) | (x_new >= barrier)
            denom = x_new - x
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(
                    finite & (denom > 0.0),
                    np.clip((barrier - x) / denom, 0.0, 1.0),
                    1.0,
                )
            gi = idx[bad]
            tau[gi] = cur_t + frac[bad] * dt
            ximax[gi] = np.maximum(ximax[gi], barrier)
            pulled[gi] = True
            active[gi] = False

            # --- escape below the physical domain ---------------------------
            esc = (~bad) & (x_new < escape_low)
            ge = idx[esc]
            xi[ge] = x_new[esc]
            escaped[ge] = True
            active[ge] = False

            # --- continue integrating ---------------------------------------
            cont = (~bad) & (~esc)
            gc = idx[cont]
            xi[gc] = x_new[cont]
            v[gc] = u_new[cont]
            ximax[gc] = np.maximum(ximax[gc], x_new[cont])

    return pulled, tau, ximax, escaped


# ---------------------------------------------------------------------------
# Static fold: vectorised "no equilibrium -> pull-in" test (matches bifurcation)
# ---------------------------------------------------------------------------
def static_pullin_grid(alpha_flat, beta_flat, n_xi=4000):
    """True where NO static equilibrium exists (device pulls in quasi-statically).

    g(xi) = alpha/(1-xi)^2 + beta/(1-xi)^4 - xi.  Since g(0)=alpha+beta>0 and
    g -> +inf as xi->1, an equilibrium exists iff g dips to <= 0 somewhere, i.e.
    iff min_xi g(xi) <= 0.  Static pull-in <=> min_xi g(xi) > 0.  This is exactly
    the fold criterion encoded by ``bifurcation.is_stable_from_rest`` and is
    vectorised over the whole grid.
    """
    a = np.asarray(alpha_flat, dtype=float)
    b = np.asarray(beta_flat, dtype=float)
    xs = np.linspace(0.0, 1.0 - 1.0e-6, n_xi)
    min_g = np.full(a.size, np.inf)
    for xv in xs:
        om = 1.0 - xv
        g = a / (om * om) + b / (om ** 4) - xv
        min_g = np.minimum(min_g, g)
    return min_g > 0.0


# ---------------------------------------------------------------------------
# Boundary extraction helpers
# ---------------------------------------------------------------------------
def _first_true_alpha(mask_row, alpha_axis):
    """Smallest alpha at which mask_row is True (pull-in front), else NaN."""
    where = np.flatnonzero(mask_row)
    if where.size == 0:
        return np.nan
    return float(alpha_axis[where[0]])


def alpha_gap_stats(pullin_mask, static_mask, alpha_axis, beta_axis):
    """Mean alpha-gap between the dynamic and static pull-in fronts.

    For each beta row (restricted to beta <= beta* where the fold exists), find
    the smallest alpha that pulls in dynamically and statically; the gap
    static_alpha - dynamic_alpha is >= 0 when kinetic overshoot moves the
    dynamic front inward.
    """
    gaps = []
    dyn_front = []
    stat_front = []
    betas = []
    for i, bval in enumerate(beta_axis):
        if bval > BETA_STAR:  # fold does not extend past the pure-Casimir end
            continue
        a_dyn = _first_true_alpha(pullin_mask[i], alpha_axis)
        a_stat = _first_true_alpha(static_mask[i], alpha_axis)
        if np.isnan(a_dyn) or np.isnan(a_stat):
            continue
        gaps.append(a_stat - a_dyn)
        dyn_front.append(a_dyn)
        stat_front.append(a_stat)
        betas.append(bval)
    gaps = np.asarray(gaps)
    return {
        "beta": np.asarray(betas),
        "alpha_dynamic": np.asarray(dyn_front),
        "alpha_static": np.asarray(stat_front),
        "gap": gaps,
        "mean_gap": float(np.mean(gaps)) if gaps.size else float("nan"),
        "median_gap": float(np.median(gaps)) if gaps.size else float("nan"),
        "max_gap": float(np.max(gaps)) if gaps.size else float("nan"),
        "n_rows": int(gaps.size),
    }


# ---------------------------------------------------------------------------
# Cross-check the vectorised integrator against the scalar reference
# ---------------------------------------------------------------------------
def _crosscheck_reference(zeta, dt, t_max):
    """Compare vectorised RK4 to src.rk4.rk4_integrate on a few points."""
    samples = [
        (0.05, 1.0e-4),   # deep stable
        (0.20, 0.09),     # deep pull-in
        (0.14, 0.02),     # near the electrostatic fold
        (0.02, 0.075),    # near the Casimir fold
        (0.10, 0.05),     # interior
    ]
    a_s = np.array([s[0] for s in samples])
    b_s = np.array([s[1] for s in samples])
    pulled, tau, ximax, _ = integrate_grid(a_s, b_s, zeta, dt=dt, t_max=t_max)

    max_tau_err = 0.0
    max_xi_err = 0.0
    all_pull_match = True
    for k, (av, bv) in enumerate(samples):
        _, _, _, info = rk4_integrate(
            alpha=av, beta=bv, zeta=zeta, xi0=0.0, v0=0.0,
            t_max=t_max, dt=dt,
        )
        ref_pull = info["pull_in"]
        all_pull_match = all_pull_match and (ref_pull == bool(pulled[k]))
        if ref_pull and info["tau_pullin"] is not None:
            max_tau_err = max(max_tau_err, abs(info["tau_pullin"] - tau[k]))
        max_xi_err = max(max_xi_err, abs(info["xi_max"] - ximax[k]))
    return all_pull_match, max_tau_err, max_xi_err


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run():
    set_all_seeds(42)  # deterministic provenance (no RNG is actually used)
    t_wall0 = time.perf_counter()

    results_dir = PROJECT_ROOT / "results"
    logs_dir = PROJECT_ROOT / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    alpha_axis = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)
    beta_axis = np.linspace(BETA_MIN, BETA_MAX, N_BETA)
    # Rows index beta (y-axis), columns index alpha (x-axis) -- matches
    # bifurcation.stability_boundary_grid convention.
    AA, BB = np.meshgrid(alpha_axis, beta_axis)  # shape (N_BETA, N_ALPHA)
    a_flat = AA.ravel()
    b_flat = BB.ravel()

    print("=" * 74)
    print("Casimir-electrostatic NEMS pull-in PHASE DIAGRAM (Fig. 2)")
    print("=" * 74)
    print(f"grid           : {N_ALPHA} (alpha) x {N_BETA} (beta) = {a_flat.size} points")
    print(f"alpha range    : [{ALPHA_MIN}, {ALPHA_MAX}]   beta range: [{BETA_MIN}, {BETA_MAX}]")
    print(f"integrator     : vectorised RK4, dt={DT}, t_max={T_MAX} "
          f"({int(np.ceil(T_MAX/DT))} steps max)")
    print(f"fold endpoints : alpha_c(beta=0)=4/27={ALPHA_ENDPOINT:.6f}  "
          f"beta*(alpha=0)=256/3125={BETA_STAR:.6f}")

    # --- reference cross-check (determinism / correctness) ------------------
    ok_pull, tau_err, xi_err = _crosscheck_reference(ZETA_PRIMARY, DT, T_MAX)
    print("-" * 74)
    print(f"cross-check vs src.rk4.rk4_integrate (5 pts): "
          f"pull_in match={ok_pull}  max|dtau|={tau_err:.2e}  max|dxi|={xi_err:.2e}")
    assert ok_pull, "vectorised integrator disagrees with reference on pull-in classification"

    # --- static fold mask (vectorised) + spot-check vs is_stable_from_rest --
    static_flat = static_pullin_grid(a_flat, b_flat)
    n_spot_ok = 0
    spot_pts = [(0.05, 1e-4), (0.20, 0.09), (0.14, 0.02), (0.02, 0.075),
                (0.10, 0.05), (0.16, 0.0), (0.0, 0.085)]
    for av, bv in spot_pts:
        static_pi = not is_stable_from_rest(av, bv)
        vec_pi = bool(static_pullin_grid(np.array([av]), np.array([bv]))[0])
        n_spot_ok += int(static_pi == vec_pi)
    print(f"static mask spot-check vs is_stable_from_rest: {n_spot_ok}/{len(spot_pts)} agree")
    assert n_spot_ok == len(spot_pts), "vectorised static mask disagrees with bifurcation module"
    static_mask = static_flat.reshape(N_BETA, N_ALPHA)

    # --- primary dynamic integration ---------------------------------------
    print("-" * 74)
    print(f"[zeta = {ZETA_PRIMARY}]  integrating grid ...")
    t0 = time.perf_counter()
    pulled, tau, ximax, escaped = integrate_grid(a_flat, b_flat, ZETA_PRIMARY)
    t_int = time.perf_counter() - t0
    print(f"    done in {t_int:.1f} s")

    pullin_mask = pulled.reshape(N_BETA, N_ALPHA)
    tau_pullin = tau.reshape(N_BETA, N_ALPHA)
    max_xi = ximax.reshape(N_BETA, N_ALPHA)

    frac_pull = float(pulled.mean())
    tau_valid = tau[np.isfinite(tau)]
    tau_lo = float(tau_valid.min()) if tau_valid.size else float("nan")
    tau_hi = float(tau_valid.max()) if tau_valid.size else float("nan")

    gap = alpha_gap_stats(pullin_mask, static_mask, alpha_axis, beta_axis)

    # cells where kinetic overshoot pulls in but statics predict a stable rest
    overshoot_cells = pulled.reshape(N_BETA, N_ALPHA) & (~static_mask)
    frac_overshoot = float(overshoot_cells.mean())

    print(f"    fraction pulling in (dynamic) : {frac_pull:.4f}")
    print(f"    fraction pulling in (static)  : {static_mask.mean():.4f}")
    print(f"    tau_PI range                  : [{tau_lo:.3f}, {tau_hi:.3f}]")
    print(f"    escaped points                : {int(escaped.sum())}")
    print(f"    mean  alpha-gap (static-dyn)  : {gap['mean_gap']:.5f}  over {gap['n_rows']} beta-rows")
    print(f"    median/max alpha-gap          : {gap['median_gap']:.5f} / {gap['max_gap']:.5f}")
    print(f"    overshoot cells (dyn&~static) : {frac_overshoot*100:.2f}% of grid")

    # --- comparison damping -------------------------------------------------
    print("-" * 74)
    print(f"[zeta = {ZETA_COMPARE}]  integrating grid (comparison) ...")
    t0 = time.perf_counter()
    pulled2, tau2, ximax2, _ = integrate_grid(a_flat, b_flat, ZETA_COMPARE)
    t_int2 = time.perf_counter() - t0
    pullin_mask2 = pulled2.reshape(N_BETA, N_ALPHA)
    gap2 = alpha_gap_stats(pullin_mask2, static_mask, alpha_axis, beta_axis)
    print(f"    done in {t_int2:.1f} s")
    print(f"    fraction pulling in (dynamic) : {pulled2.mean():.4f}")
    print(f"    mean alpha-gap (static-dyn)   : {gap2['mean_gap']:.5f}  "
          f"(vs {gap['mean_gap']:.5f} at zeta={ZETA_PRIMARY})")

    # --- static fold boundary curve for overlay + CSV -----------------------
    alpha_c, beta_c, xi_pi, physical = fold_boundary(n=400, full=False)

    npz_path = results_dir / "phase_diagram.npz"
    np.savez_compressed(
        npz_path,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        alpha_grid=AA,
        beta_grid=BB,
        pullin_mask=pullin_mask,
        tau_pullin=tau_pullin,
        max_xi=max_xi,
        static_mask=static_mask,
        pullin_mask_compare=pullin_mask2,
        tau_pullin_compare=tau2.reshape(N_BETA, N_ALPHA),
        zeta=np.float64(ZETA_PRIMARY),
        zeta_compare=np.float64(ZETA_COMPARE),
        dt=np.float64(DT),
        t_max=np.float64(T_MAX),
        fold_alpha_c=alpha_c,
        fold_beta_c=beta_c,
        fold_xi_pi=xi_pi,
    )
    print("-" * 74)
    print(f"wrote {npz_path}")

    csv_path = results_dir / "phase_boundary_static.csv"
    u = 1.0 - xi_pi
    data = np.column_stack([u, xi_pi, alpha_c, beta_c, physical.astype(float)])
    np.savetxt(
        csv_path, data, delimiter=",",
        header="u,xi_pi,alpha_c,beta_c,physical", comments="", fmt="%.12e",
    )
    print(f"wrote {csv_path}")

    t_wall = time.perf_counter() - t_wall0
    print(f"total wall-clock: {t_wall:.1f} s")

    # --- provenance log -----------------------------------------------------
    log = {
        "grid": {"n_alpha": N_ALPHA, "n_beta": N_BETA,
                 "alpha_range": [ALPHA_MIN, ALPHA_MAX],
                 "beta_range": [BETA_MIN, BETA_MAX]},
        "integration": {"dt": DT, "t_max": T_MAX,
                        "eps_barrier": EPS_BARRIER, "escape_low": ESCAPE_LOW},
        "zeta_primary": ZETA_PRIMARY,
        "zeta_compare": ZETA_COMPARE,
        "crosscheck": {"pull_in_match": bool(ok_pull),
                       "max_dtau": tau_err, "max_dxi": xi_err},
        "fold_endpoints": {"alpha_c_beta0": ALPHA_ENDPOINT, "beta_star": BETA_STAR},
        "summary_primary": {
            "fraction_pullin_dynamic": frac_pull,
            "fraction_pullin_static": float(static_mask.mean()),
            "tau_pullin_min": tau_lo, "tau_pullin_max": tau_hi,
            "mean_alpha_gap": gap["mean_gap"],
            "median_alpha_gap": gap["median_gap"],
            "max_alpha_gap": gap["max_gap"],
            "n_boundary_rows": gap["n_rows"],
            "overshoot_cell_fraction": frac_overshoot,
            "n_escaped": int(escaped.sum()),
        },
        "summary_compare": {
            "fraction_pullin_dynamic": float(pulled2.mean()),
            "mean_alpha_gap": gap2["mean_gap"],
        },
        "wall_clock_s": t_wall,
        "integration_time_primary_s": t_int,
        "integration_time_compare_s": t_int2,
    }
    log_path = logs_dir / "phase_diagram.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print(f"wrote {log_path}")
    print("=" * 74)

    return log


if __name__ == "__main__":
    run()

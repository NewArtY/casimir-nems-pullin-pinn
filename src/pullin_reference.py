"""Adaptive reference for the pull-in trajectory, integrated all the way in.

``traj_pullin.npz`` stops at the PINN training horizon ``T = 0.98 tau_*``,
where ``xi = 0.742``.  That is the right window for training, but it leaves the
pull-in panel of Fig. 1 without the collapse it is named after.  This module
re-integrates the same case with the adaptive DOP853 solver up to the barrier
event and stores the result separately, so the figure can draw the reference
through to ``tau_*`` while the network is drawn only where it was trained.

Run:  python -m src.pullin_reference
"""

from __future__ import annotations

import os

import numpy as np

from . import config
from .verify import adaptive_reference

RESULTS = str(config.RESULTS_DIR)
OUT = os.path.join(RESULTS, "traj_pullin_ref_full.npz")

N_GRID = 4000


def main() -> str:
    config.set_all_seeds()

    src = np.load(os.path.join(RESULTS, "traj_pullin.npz"), allow_pickle=True)
    alpha, beta, zeta = (float(src["alpha"]), float(src["beta"]),
                         float(src["zeta"]))
    xi0, v0 = float(src["t"][0]) * 0.0, 0.0

    sol = adaptive_reference(alpha, beta, zeta, xi0, v0, (0.0, 20.0))
    tau_star = float(sol.t[-1])

    # A uniform grid resolves the early motion but not the last decade of the
    # gap, where xi moves from 0.74 to 1. Concatenate a uniform grid with one
    # that is geometric in the remaining time to the pole.
    t_lin = np.linspace(0.0, 0.98 * tau_star, N_GRID // 2)
    frac = np.geomspace(0.02, 1e-6, N_GRID // 2)
    t_tail = tau_star * (1.0 - frac)
    t = np.unique(np.concatenate([t_lin, t_tail]))
    xi = sol.sol(t)[0]

    # Clip the last few points that the dense output may push past the barrier.
    keep = xi < 1.0
    t, xi = t[keep], xi[keep]

    np.savez(OUT, t=t, xi=xi, tau_star=tau_star,
             alpha=alpha, beta=beta, zeta=zeta,
             xi_at_halt=float(xi[-1]))
    print(f"[save] {OUT}")
    print(f"  tau_star = {tau_star:.10f}   xi_final = {xi[-1]:.8f}   "
          f"points = {t.size}")
    return OUT


if __name__ == "__main__":
    main()

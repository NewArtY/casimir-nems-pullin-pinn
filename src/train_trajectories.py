"""Train the single-trajectory PINN for the three Fig. 1 regimes.

For each of the STABLE / GROWING / PULL-IN regimes (the same (alpha, beta,
zeta, IC) used by the RK4 smoke test) this script:

    (a) trains a PINN (deterministic; wall-clock logged),
    (b) saves the weights to   models/pinn_<regime>.pth,
    (c) saves the loss history to logs/history_<regime>.json,
    (d) computes a high-accuracy reference with scipy solve_ivp (DOP853,
        rtol=1e-10) on the SAME time span,
    (e) evaluates the PINN on a dense tau grid and computes L2 / Linf errors of
        xi (and xi') against the reference,
    (f) saves the comparison arrays to results/traj_<regime>.npz.

Finally it prints a table of final loss and max error per regime.

Parametrization per regime
--------------------------
* stable  : raw displacement + hard IC, Fourier features at the damped natural
            frequency (long horizon, damped oscillation).
* growing : raw displacement + hard IC (overdamped monotone creep).
* pull-in : rapidity xi = 1 - exp(-theta) + hard IC on [0, 0.98 tau_pullin]
            (pole-safe: xi < 1 guaranteed by construction).

Run from the code/ directory:   python src/train_trajectories.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# Make code/ importable regardless of launch directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from src.config import set_all_seeds, RESULTS_DIR, SMOKE_RESULTS_JSON  # noqa: E402
from src.physics import force_rhs  # noqa: E402
from src.pinn import (  # noqa: E402
    train_pinn, evaluate_xi_xidot, save_model, save_history,
)


# ---------------------------------------------------------------------------
# Reference solver (high accuracy)
# ---------------------------------------------------------------------------
def reference_solution(alpha, beta, zeta, xi0, v0, T, n_eval=6000,
                       method="DOP853", rtol=1e-10, atol=1e-12):
    """Dense high-accuracy reference (t, xi, xi') on [0, T]."""
    def rhs(t, y):
        xi, v = y
        return [v, force_rhs(xi, v, alpha, beta, zeta)]

    t_eval = np.linspace(0.0, T, n_eval)
    sol = solve_ivp(rhs, (0.0, T), [xi0, v0], method=method,
                    rtol=rtol, atol=atol, dense_output=True, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"reference solve_ivp failed: {sol.message}")
    xi = sol.y[0]
    v = sol.y[1]
    return t_eval, xi, v


# ---------------------------------------------------------------------------
# tau_pullin from the committed smoke results
# ---------------------------------------------------------------------------
def read_tau_pullin(default=2.6222369392167972):
    try:
        with open(SMOKE_RESULTS_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for c in data.get("cases", []):
            if c.get("name") == "pullin" and c.get("tau_pullin"):
                return float(c["tau_pullin"])
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# Regime configurations (SAME physics as the RK4 smoke test).
# ---------------------------------------------------------------------------
def build_regimes():
    tau_pi = read_tau_pullin()

    stable = dict(
        name="stable",
        alpha=0.05, beta=1.0e-4, zeta=0.6, xi0=0.0, v0=0.0, T=60.0,
        train=dict(parametrization="raw", hard_ic=True,
                   n_fourier=2, fourier_omega=0.8,
                   width=64, n_hidden=4, n_collocation=3000,
                   adam_iters=1500, adam_lr=2e-3, lbfgs_iters=1800,
                   lambda_ic=1.0, lambda_pull=1.0, log_every=500),
    )

    growing = dict(
        name="growing",
        alpha=0.072421875, beta=0.0407373046875, zeta=1.0,
        xi0=0.0, v0=0.0, T=30.0,
        train=dict(parametrization="raw", hard_ic=True,
                   n_fourier=0, fourier_omega=1.0,
                   width=64, n_hidden=4, n_collocation=2000,
                   adam_iters=2000, adam_lr=1e-3, lbfgs_iters=500,
                   lambda_ic=1.0, lambda_pull=1.0, log_every=500),
    )

    pullin = dict(
        name="pullin",
        alpha=0.20, beta=0.10, zeta=0.5, xi0=0.0, v0=0.0,
        T=0.98 * tau_pi,
        tau_pullin=tau_pi,
        train=dict(parametrization="rapidity", hard_ic=True,
                   n_fourier=0, fourier_omega=1.0,
                   width=64, n_hidden=4, n_collocation=1600,
                   adam_iters=3000, adam_lr=1e-3, lbfgs_iters=500,
                   lambda_ic=1.0, lambda_pull=1.0, log_every=500),
    )

    return [stable, growing, pullin]


# ---------------------------------------------------------------------------
# Per-regime driver
# ---------------------------------------------------------------------------
def run_regime(reg, seed=42):
    name = reg["name"]
    print("\n" + "=" * 72)
    print(f"REGIME: {name}   (alpha={reg['alpha']:.6g}, beta={reg['beta']:.6g}, "
          f"zeta={reg['zeta']:.3g}, T={reg['T']:.6g})")
    print(f"  parametrization = {reg['train']['parametrization']}"
          f"  hard_ic = {reg['train']['hard_ic']}"
          f"  n_fourier = {reg['train']['n_fourier']}")
    print("=" * 72)

    set_all_seeds(seed)

    t_wall = time.time()
    model, history = train_pinn(
        alpha=reg["alpha"], beta=reg["beta"], zeta=reg["zeta"],
        xi0=reg["xi0"], v0=reg["v0"], T=reg["T"], seed=seed,
        **reg["train"],
    )
    wall = time.time() - t_wall
    history["regime"] = name
    history["wall_time_total"] = wall

    # (b) save weights, (c) save history
    mpath = save_model(model, f"pinn_{name}")
    hpath = save_history(history, name)

    # (d) high-accuracy reference on the same span
    t_ref, xi_ref, v_ref = reference_solution(
        reg["alpha"], reg["beta"], reg["zeta"], reg["xi0"], reg["v0"], reg["T"],
    )

    # (e) evaluate PINN on the dense grid; errors vs reference
    xi_pinn, v_pinn = evaluate_xi_xidot(model, t_ref)
    err_xi = np.abs(xi_pinn - xi_ref)
    err_v = np.abs(v_pinn - v_ref)
    l2_xi = float(np.sqrt(np.mean(err_xi ** 2)))
    linf_xi = float(np.max(err_xi))
    l2_v = float(np.sqrt(np.mean(err_v ** 2)))
    linf_v = float(np.max(err_v))

    # (f) save comparison arrays
    npz_path = RESULTS_DIR / f"traj_{name}.npz"
    np.savez(
        npz_path,
        t=t_ref, xi_pinn=xi_pinn, xi_ref=xi_ref,
        xidot_pinn=v_pinn, xidot_ref=v_ref,
        alpha=reg["alpha"], beta=reg["beta"], zeta=reg["zeta"],
        xi0=reg["xi0"], v0=reg["v0"], T=reg["T"],
        parametrization=reg["train"]["parametrization"],
    )

    final = {
        "L_ode": history["L_ode"][-1],
        "L_ic": history["L_ic"][-1],
        "L_pull": history["L_pull"][-1],
        "L_total": history["L_total"][-1],
    }

    print(f"  wall-clock       : {wall:6.1f} s")
    print(f"  final loss       : total={final['L_total']:.3e} "
          f"(ode={final['L_ode']:.3e}, ic={final['L_ic']:.3e}, "
          f"pull={final['L_pull']:.3e})")
    print(f"  xi  error        : L2={l2_xi:.3e}   Linf={linf_xi:.3e}")
    print(f"  xi' error        : L2={l2_v:.3e}   Linf={linf_v:.3e}")
    print(f"  ref xi range     : [{xi_ref.min():.5f}, {xi_ref.max():.5f}]  "
          f"(final {xi_ref[-1]:.5f})")
    print(f"  saved            : {mpath.name}, {hpath.name}, {npz_path.name}")

    return {
        "name": name,
        "parametrization": reg["train"]["parametrization"],
        "wall_time": wall,
        "final_loss": final,
        "l2_xi": l2_xi, "linf_xi": linf_xi,
        "l2_v": l2_v, "linf_v": linf_v,
        "xi_ref_max": float(xi_ref.max()),
        "xi_ref_final": float(xi_ref[-1]),
        "T": reg["T"],
        "tau_pullin": reg.get("tau_pullin"),
    }


def main():
    seed = set_all_seeds(42)
    print("#" * 72)
    print(f"# PINN trajectory training  (seed={seed}, torch CPU float64)")
    print("#" * 72)

    regimes = build_regimes()
    summary = [run_regime(reg, seed=seed) for reg in regimes]

    # ---- summary table ---------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    header = (f"{'regime':>8} | {'param':>9} | {'wall[s]':>7} | "
              f"{'L_total':>10} | {'Linf(xi)':>10} | {'L2(xi)':>10} | "
              f"{'Linf(xi.)':>10}")
    print(header)
    print("-" * len(header))
    for s in summary:
        print(f"{s['name']:>8} | {s['parametrization']:>9} | "
              f"{s['wall_time']:7.1f} | {s['final_loss']['L_total']:10.3e} | "
              f"{s['linf_xi']:10.3e} | {s['l2_xi']:10.3e} | {s['linf_v']:10.3e}")
    print("=" * 72)
    total_wall = sum(s["wall_time"] for s in summary)
    print(f"total training wall-clock: {total_wall:.1f} s")

    # persist a machine-readable summary too
    out = RESULTS_DIR / "pinn_traj_summary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "regimes": summary}, fh, indent=2)
    print(f"wrote {out}")

    return summary


if __name__ == "__main__":
    main()

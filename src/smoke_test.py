"""Smoke test: exercise the three dynamical regimes and verify the potential.

Run from the ``code/`` directory:

    python src/smoke_test.py

It (a) seeds all RNGs, (b) integrates three representative cases chosen to
display the STABLE, GROWING and PULL-IN regimes, (c) prints regime,
tau_pullin, max xi for each, (d) verifies the finite-difference potential
identity -U'(xi) == net_force(xi), and (e) writes results/smoke_results.json.
"""

from __future__ import annotations

import json
import os
import sys

# Make ``code/`` importable so that ``from src.physics import ...`` works
# regardless of the directory python is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np  # noqa: E402

from src.config import set_all_seeds, SMOKE_RESULTS_JSON  # noqa: E402
from src.physics import verify_potential, fold_alpha_beta  # noqa: E402
from src.rk4 import rk4_integrate  # noqa: E402


# ---------------------------------------------------------------------------
# Representative cases (tuned so each falls cleanly into one regime).
# ---------------------------------------------------------------------------
def build_cases():
    cases = []

    # (1) STABLE: small alpha, beta well below the fold, released from rest.
    #     Damped approach to the stable equilibrium near xi ~ 0.055.
    cases.append(dict(
        name="stable",
        alpha=0.05, beta=1.0e-4, zeta=0.6,
        xi0=0.0, v0=0.0, t_max=60.0, dt=1.0e-3,
    ))

    # (2) GROWING: parameters just ABOVE the fold at u = 0.75 (3% over), with
    #     overdamped dynamics (zeta = 1), released from rest.  There is no
    #     equilibrium, but the saddle-node "ghost" left behind near xi ~ 0.75
    #     throttles the quasi-static approach: the plate creeps monotonically
    #     upward (velocity small but strictly positive and growing) and has NOT
    #     reached the barrier by t_max = 30.  It *would* pull in at t ~ 38.8,
    #     so this is a genuine run-up captured mid-climb -> 'growing'.
    a_c, b_c = fold_alpha_beta(0.75)          # (~0.0703, ~0.0396)
    cases.append(dict(
        name="growing",
        alpha=float(a_c) * 1.03, beta=float(b_c) * 1.03, zeta=1.0,
        xi0=0.0, v0=0.0, t_max=30.0, dt=1.0e-3,
    ))

    # (3) PULL-IN: alpha, beta well above the fold so net_force > 0 for all
    #     xi in [0,1): the plate accelerates straight into contact.
    cases.append(dict(
        name="pullin",
        alpha=0.20, beta=0.10, zeta=0.5,
        xi0=0.0, v0=0.0, t_max=60.0, dt=1.0e-3,
    ))

    return cases


def run():
    seed = set_all_seeds(42)

    # (d) Potential identity check.
    ok, max_err = verify_potential(alpha=0.07, beta=0.02)
    print("=" * 68)
    print(f"Seed = {seed}")
    print(f"Potential identity  -U'(xi) == net_force(xi):  "
          f"{'PASS' if ok else 'FAIL'}  (max |err| = {max_err:.3e})")
    print("=" * 68)

    cases = build_cases()
    results = {
        "seed": seed,
        "potential_identity": {"passed": bool(ok), "max_abs_err": max_err},
        "cases": [],
    }

    for c in cases:
        t, xi, v, info = rk4_integrate(
            alpha=c["alpha"], beta=c["beta"], zeta=c["zeta"],
            xi0=c["xi0"], v0=c["v0"], t_max=c["t_max"], dt=c["dt"],
        )
        tau = info["tau_pullin"]
        tau_str = f"{tau:.4f}" if tau is not None else "None"
        print(
            f"[{c['name']:>7}] "
            f"alpha={c['alpha']:.5f}  beta={c['beta']:.5f}  zeta={c['zeta']:.3f}  "
            f"xi0={c['xi0']:.2f} v0={c['v0']:.2f}  ||  "
            f"regime={info['regime']:>7}  "
            f"tau_pullin={tau_str:>8}  "
            f"max_xi={info['xi_max']:.5f}"
        )
        results["cases"].append({
            "name": c["name"],
            "params": {
                "alpha": c["alpha"], "beta": c["beta"], "zeta": c["zeta"],
                "xi0": c["xi0"], "v0": c["v0"],
                "t_max": c["t_max"], "dt": c["dt"],
            },
            "regime": info["regime"],
            "pull_in": info["pull_in"],
            "tau_pullin": tau,
            "max_xi": info["xi_max"],
            "escaped": info["escaped"],
            "n_steps": info["n_steps"],
        })

    print("=" * 68)
    regimes = {c["regime"] for c in results["cases"]}
    all_three = regimes >= {"stable", "growing", "pullin"}
    print(f"Regimes observed: {sorted(regimes)}  ->  "
          f"{'ALL THREE PRESENT' if all_three else 'MISSING SOME'}")

    with open(SMOKE_RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {SMOKE_RESULTS_JSON}")

    return results, (ok and all_three)


if __name__ == "__main__":
    _, success = run()
    sys.exit(0 if success else 1)

"""Dense validation of the slope of the damped-threshold surrogate (Fig. 3b).

The surrogate of ``dynamic_boundary_design.py`` is refitted with the archived
recipe and seed, which reproduces the archived network, and its weights are
saved to ``models/dyn_boundary_surrogate.pth``.  Its slope
d alpha_dyn / d beta at zeta = 0.1 is compared on a dense grid of beta with
the slope of the separatrix construction (``threshold_checks``), obtained by
second-order differences at two steps and Richardson extrapolation
(one-sided near beta = 0).

Writes ``results/surrogate_validation.json``.
Run:  python -m src.surrogate_validation
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import torch

try:
    from src import config
    from src import dynamic_boundary_design as D
    from src.threshold_checks import alpha_dyn_manifold
except ImportError:  # pragma: no cover
    from . import config
    from . import dynamic_boundary_design as D
    from .threshold_checks import alpha_dyn_manifold

print = functools.partial(print, flush=True)  # noqa: A001
OUT = os.path.join(str(config.RESULTS_DIR), "surrogate_validation.json")
ZETA = 0.1


def exact_slope(b0):
    am = lambda b: alpha_dyn_manifold(b, ZETA)
    if b0 < 0.003:
        def d(h):
            f = [am(b0 + k * h) for k in range(3)]
            return (-3 * f[0] + 4 * f[1] - f[2]) / (2 * h)
    else:
        def d(h):
            return (am(b0 + h) - am(b0 - h)) / (2 * h)
    h = 1e-3
    return (4 * d(h / 2) - d(h)) / 3


def main():
    config.set_all_seeds()
    with open(os.path.join(str(config.RESULTS_DIR),
                           "dynamic_boundary_design.json"), encoding="utf-8") as fh:
        db = json.load(fh)
    b = np.array(db["boundary"]["beta_grid"])
    a = np.array(db["boundary"]["alpha_dyn"])
    res = D.fit_surrogate(b, a, verbose=False)
    net = res[0] if isinstance(res, tuple) else res
    torch.save(net.state_dict(),
               config.PROJECT_ROOT / "models" / "dyn_boundary_surrogate.pth")
    archived = {round(r["beta"], 4): r["dalpha_dyn_dbeta_surrogate"]
                for r in db["sensitivities"]}
    from scipy.interpolate import CubicSpline
    spline = CubicSpline(b, a)          # not-a-knot, same 33 points
    rows = []
    for b0 in np.round(np.arange(0.0, 0.0721, 0.0025), 4):
        s = D.surrogate_sensitivity(net, float(b0))[1]
        e = exact_slope(float(b0))
        sp = float(spline(b0, 1))
        rows.append(dict(beta=float(b0), surrogate=s, separatrix=e,
                         spline=sp, rel_err=(s - e) / e,
                         rel_err_spline=(sp - e) / e))
        print(f"beta={b0:.4f}  surrogate {s:+.5f}  spline {sp:+.5f}  "
              f"separatrix {e:+.5f}  rel {100 * (s - e) / e:+.2f}% / "
              f"{100 * (sp - e) / e:+.3f}%")
    for b0, s in archived.items():
        mine = D.surrogate_sensitivity(net, b0)[1]
        print(f"refit reproduces archived slope at {b0}: {mine:.6f} vs {s:.6f}")
    inner = [abs(r["rel_err"]) for r in rows if 0.01 <= r["beta"] <= 0.05]
    out = dict(zeta=ZETA, rows=rows,
               max_abs_rel_err_0p01_0p05=float(max(inner)),
               max_abs_rel_err_all=float(max(abs(r["rel_err"]) for r in rows)),
               max_abs_rel_err_spline=float(max(abs(r["rel_err_spline"])
                                                for r in rows)))
    # redraw Fig. 3 with the exact slope and the spline baseline in panel (b)
    dense = dict(beta=np.array([r["beta"] for r in rows]),
                 separatrix=np.array([r["separatrix"] for r in rows]),
                 spline=np.array([r["spline"] for r in rows]))
    D._make_figure(b, a, np.array(db["boundary"]["alpha_static"]), net,
                   db["sensitivities"], dense=dense)
    print(f"max |rel err| spline: {out['max_abs_rel_err_spline']:.2e}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"max |rel err| on [0.01,0.05]: {out['max_abs_rel_err_0p01_0p05']:.4f}")
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()

"""Finite-conductivity estimate for the representative devices.

The device numbers of the main text use the ideal-conductor Casimir force.
This module estimates how much of the release-point effect survives for gold.
It computes the T = 0 Lifshitz pressure between two gold half-spaces in the
plasma model, eps(i xi) = 1 + omega_p^2 / xi^2 with hbar omega_p = 9.0 eV, and
its ratio eta(a) to the ideal-conductor pressure.  As a crude estimate it then
rescales each device's beta by eta(d) at the rest gap and recomputes the
undamped step threshold for release from xi = 0 and from the zero-bias rest
position.  The rescaling ignores the gap dependence of eta along the stroke
(eta falls as the gap closes), so it is an order-of-magnitude estimate.

Writes ``results/conductivity_estimate.json``.
Run:  python -m src.conductivity_estimate
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.integrate import quad

try:
    from src import config
    from src.damping_regimes import (fold_of_beta, zero_bias_equilibrium,
                                     alpha_undamped)
    from src.device_table import DEVICES, beta_param
except ImportError:  # pragma: no cover
    from . import config
    from .damping_regimes import (fold_of_beta, zero_bias_equilibrium,
                                  alpha_undamped)
    from .device_table import DEVICES, beta_param

HBAR_OMEGA_P_EV = 9.0
OUT = os.path.join(str(config.RESULTS_DIR), "conductivity_estimate.json")


def eta_plasma(a):
    """P_plasma(a) / P_ideal(a) at T = 0.

    In units with x = 2 a q and y = 2 a xi / c, the ideal pressure integral
    is int_0^inf dx x^3/(e^x - 1) = pi^4/15 after the frequency integral;
    the plasma one weights each (x, y) by the reflection coefficients.
    """
    ev = 1.602176634e-19
    wp = HBAR_OMEGA_P_EV * ev / config.HBAR
    s = 2 * a * wp / config.C_LIGHT        # dimensionless plasma frequency

    def inner(x):
        # integrate over y in [0, x] (q >= xi/c), Jacobian gives x^2 dy
        def f(y):
            eps = 1.0 + s ** 2 / max(y, 1e-300) ** 2
            kperp2 = x ** 2 - y ** 2
            ke = np.sqrt(kperp2 + eps * y ** 2)
            rtm = (eps * x - ke) / (eps * x + ke)
            rte = (x - ke) / (x + ke)
            e = np.exp(-x)
            return sum(r * r * e / (1 - r * r * e) for r in (rtm, rte))
        return quad(f, 0.0, x, limit=200)[0] * x ** 2 / 2.0
    num = quad(inner, 0.0, 60.0, limit=400)[0]
    # ideal: rtm = rte = 1 -> 2 * e/(1-e) * x^2 * x / 2 = x^3/(e^x - 1)
    return num / (np.pi ** 4 / 15.0)


def step_ratio(beta, xi_start):
    a_c, _ = fold_of_beta(beta)
    return float(np.sqrt(alpha_undamped(beta, xi_start) / a_c))


def main():
    rows = []
    for name, desc, A_um2, d_nm, k, Q, f0 in DEVICES:
        beta = float(beta_param(A_um2 * 1e-12, d_nm * 1e-9, k))
        eta = eta_plasma(d_nm * 1e-9)
        out = dict(name=name, d_nm=d_nm, eta=eta, beta_ideal=beta,
                   beta_gold=eta * beta)
        for tag, b in (("ideal", beta), ("gold", eta * beta)):
            r0 = step_ratio(b, 0.0)
            rr = step_ratio(b, zero_bias_equilibrium(b))
            out[f"step_from0_{tag}"] = r0
            out[f"step_fromxr_{tag}"] = rr
            out[f"underestimate_pct_{tag}"] = 100.0 * (1 - r0 / rr)
        rows.append(out)
        print(f"{name}: d={d_nm:5.0f} nm  eta={eta:.3f}  "
              f"ideal {out['step_from0_ideal']:.4f}/{out['step_fromxr_ideal']:.4f}"
              f" ({out['underestimate_pct_ideal']:.1f}%)   gold "
              f"{out['step_from0_gold']:.4f}/{out['step_fromxr_gold']:.4f}"
              f" ({out['underestimate_pct_gold']:.1f}%)")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(hbar_omega_p_eV=HBAR_OMEGA_P_EV, devices=rows), fh,
                  indent=2)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()

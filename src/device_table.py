"""Device-parameter table: real gold-coated NEMS actuators -> pull-in behavior.

Applied-significance artifact for *Physical Review Applied*.  For each of a set
of representative gold-coated MEMS/NEMS capacitive actuators (parallel-plate,
lumped single-degree-of-freedom, quoted rest gap ``d``) this script computes:

  * the dimensionless control parameters
        beta = pi^2 hbar c A / (240 k d^5)         (Casimir strength, fixed)
        zeta = 1 / (2 Q)                            (damping ratio)
    and the electrostatic prefactor  alpha / V0^2 = eps0 A / (2 k d^3);
  * the CRITICAL pull-in voltage at T = 0, from the static fold
        alpha_c(u) = u^2 (4 - 5u)/2 ,  beta_c(u) = u^4 (3u - 2)/2
    evaluated at the device's own beta, then
        V_PI = sqrt( alpha_c / (alpha/V0^2) ) = sqrt( 2 alpha_c k d^3 /(eps0 A) );
  * the thermally-derated pull-in voltage at T = 300 K, from the leading
    Lifshitz-shifted fold (Casimir term multiplied by [1 + kappa(d) T (1-xi)],
    kappa(d) = (60 zeta(3)/pi^3) k_B d /(hbar c)), solved with the exact 2x2
    linear fold system of ``lifshitz_boundary.fold_boundary_T``.

Devices whose beta exceeds the (temperature-dependent) Casimir ceiling beta*
have no static equilibrium at any voltage and are flagged (spontaneous
stiction); no finite V_PI is defined for them.

Prints a clean table and writes ``results/device_table.json`` and
``results/device_table.csv``.  Run:  ``python src/device_table.py``.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Robust imports: work as a package module and when run as a script.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_HERE))  # add .../PRA/code
    from src import config
    from src.physics import fold_alpha_beta
    from src.lifshitz_boundary import fold_boundary_T, kappa
else:
    from . import config
    from .physics import fold_alpha_beta
    from .lifshitz_boundary import fold_boundary_T, kappa


# ---------------------------------------------------------------------------
# Device definitions (representative, order-of-magnitude gold-NEMS values;
# see docs/physics_notes.md Sec. 2).  Geometry in SI-friendly engineering units.
# ---------------------------------------------------------------------------
DEVICES = [
    # name, description,               A [um^2], d [nm], k [N/m], Q,     f0 [Hz]
    ("A", "MEMS gold plate",            100.0,   100.0,   0.5,    1.0e4, 50.0e3),
    ("B", "sub-100 nm NEMS",             25.0,    50.0,   2.0,    1.0e3, 1.0e6),
    ("C", "stiff NEMS",                 100.0,   200.0,   5.0,    5.0e3, 200.0e3),
    ("D", "intermediate-gap NEMS",       50.0,    75.0,   1.0,    2.0e3, 300.0e3),
]

BETA_STAR_T0 = 256.0 / 3125.0  # 0.081920 Casimir ceiling at T = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def beta_param(A, d, k):
    """Casimir control parameter beta = pi^2 hbar c A / (240 k d^5)."""
    return (np.pi ** 2 * config.HBAR * config.C_LIGHT * A) / (240.0 * k * d ** 5)


def alpha_prefactor(A, d, k):
    """alpha / V0^2 = eps0 A / (2 k d^3)   [1/V^2] (so alpha = prefac * V0^2)."""
    return config.EPS0 * A / (2.0 * k * d ** 3)


def alpha_c_of_beta(T, d, beta):
    """Critical alpha_c on the (T-shifted) fold at fixed device beta.

    Returns (alpha_c, xi_pi, beta_star, on_branch).

    * At T = 0 the branch is the closed form alpha_c(u)=u^2(4-5u)/2,
      beta_c(u)=u^4(3u-2)/2; at T > 0 it is the exact 2x2-linear Lifshitz fold.
    * ``beta_star`` is the alpha=0 intercept (Casimir ceiling at this T, d).
    * ``on_branch`` is False when beta > beta_star: no static equilibrium
      exists at any voltage (spontaneous stiction); alpha_c is returned NaN.
    """
    a_c, b_c, xi_pi = fold_boundary_T(T, d, n=8000)
    # b_c is sorted ascending, 0 -> beta_star; a_c descends 4/27 -> 0.
    beta_star = float(b_c.max())
    if beta > beta_star:
        return float("nan"), float("nan"), beta_star, False
    # beta below the smallest sampled positive beta_c -> essentially the
    # beta -> 0 endpoint (alpha_c -> 4/27, xi_PI -> 1/3); np.interp clamps.
    alpha_c = float(np.interp(beta, b_c, a_c))
    xi = float(np.interp(beta, b_c, xi_pi))
    return alpha_c, xi, beta_star, True


def compute_device(name, desc, A_um2, d_nm, k, Q, f0):
    """Compute all reported quantities for one device."""
    A = A_um2 * 1e-12   # um^2 -> m^2
    d = d_nm * 1e-9     # nm   -> m

    beta = beta_param(A, d, k)
    zeta = 1.0 / (2.0 * Q)
    pref = alpha_prefactor(A, d, k)             # alpha/V0^2 [1/V^2]

    # effective mass from the quoted resonance (for provenance only)
    omega0 = 2.0 * np.pi * f0
    m_eff = k / omega0 ** 2

    # --- T = 0 pull-in ----------------------------------------------------
    ac0, xi0, bstar0, ok0 = alpha_c_of_beta(0.0, d, beta)
    # --- T = 300 K pull-in (Lifshitz-derated) -----------------------------
    ac3, xi3, bstar3, ok3 = alpha_c_of_beta(300.0, d, beta)

    def v_pi(alpha_c, ok):
        if not ok or not np.isfinite(alpha_c):
            return float("nan")
        return float(np.sqrt(alpha_c / pref))

    V0 = v_pi(ac0, ok0)
    V300 = v_pi(ac3, ok3)
    if np.isfinite(V0) and np.isfinite(V300) and V0 > 0:
        dV_pct = (V300 / V0 - 1.0) * 100.0
    else:
        dV_pct = float("nan")

    above_ceiling = (not ok0) or (not ok3)

    return {
        "name": name,
        "desc": desc,
        "A_um2": A_um2,
        "d_nm": d_nm,
        "k_N_per_m": k,
        "Q": Q,
        "f0_Hz": f0,
        "m_eff_kg": m_eff,
        "beta": beta,
        "zeta": zeta,
        "alpha_over_V2": pref,
        "kappaT_300K": float(kappa(d) * 300.0),
        "beta_star_T0": bstar0,
        "beta_star_300K": bstar3,
        "alpha_c_T0": ac0,
        "alpha_c_300K": ac3,
        "xi_pi_T0": xi0,
        "xi_pi_300K": xi3,
        "V_PI_0K": V0,
        "V_PI_300K": V300,
        "dV_PI_pct": dV_pct,
        "above_ceiling": above_ceiling,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(x, spec):
    return "n/a" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else format(x, spec)


def print_table(rows):
    print("=" * 96)
    print("Gold-coated NEMS actuators: predicted Casimir-electrostatic pull-in "
          "(parallel-plate, SDOF)")
    print("=" * 96)
    hdr = (f"{'Dev':>3} {'A/um2':>7} {'d/nm':>6} {'k(N/m)':>7} {'Q':>7} "
           f"{'beta':>10} {'zeta':>9} {'xiPI(0)':>8} "
           f"{'V_PI(0K)':>9} {'V_PI(300K)':>11} {'dV/%':>7}  flag")
    print(hdr)
    print("-" * 96)
    for r in rows:
        flag = "STICTION(beta>beta*)" if r["above_ceiling"] else ""
        print(f"{r['name']:>3} {r['A_um2']:>7.0f} {r['d_nm']:>6.0f} "
              f"{r['k_N_per_m']:>7.2f} {r['Q']:>7.0f} "
              f"{r['beta']:>10.4e} {r['zeta']:>9.2e} "
              f"{_fmt(r['xi_pi_T0'], '.4f'):>8} "
              f"{_fmt(r['V_PI_0K'], '.3f'):>9} "
              f"{_fmt(r['V_PI_300K'], '.3f'):>11} "
              f"{_fmt(r['dV_PI_pct'], '+.2f'):>7}  {flag}")
    print("-" * 96)
    print(f"Casimir ceiling at T=0:  beta* = 256/3125 = {BETA_STAR_T0:.6f} "
          f"(no static equilibrium for beta > beta*).")
    print("V_PI(300K) includes the first-order thermal Lifshitz correction "
          "at each device's own gap d.")


def save_outputs(rows):
    res_dir = str(config.RESULTS_DIR)
    os.makedirs(res_dir, exist_ok=True)

    json_path = os.path.join(res_dir, "device_table.json")
    payload = {
        "model": {
            "description": "parallel-plate electrostatic + ideal-conductor "
                           "Casimir, lumped SDOF, gold electrodes",
            "alpha": "eps0 A V0^2 / (2 k d^3)",
            "beta": "pi^2 hbar c A / (240 k d^5)",
            "zeta": "1/(2Q)",
            "fold_T0": "alpha_c(u)=u^2(4-5u)/2, beta_c(u)=u^4(3u-2)/2, u=1-xi",
            "V_PI": "sqrt(2 alpha_c k d^3 / (eps0 A))",
            "lifshitz": "beta -> beta[1 + (60 zeta(3)/pi^3) k_B T (d-x)/(hbar c)]",
            "beta_star_T0": BETA_STAR_T0,
            "T_derated_K": 300.0,
        },
        "constants_SI": {
            "eps0": config.EPS0, "hbar": config.HBAR,
            "c": config.C_LIGHT, "k_B": config.K_B, "zeta3": config.ZETA3,
        },
        "devices": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(res_dir, "device_table.csv")
    cols = ["name", "desc", "A_um2", "d_nm", "k_N_per_m", "Q", "f0_Hz",
            "m_eff_kg", "beta", "zeta", "alpha_over_V2", "kappaT_300K",
            "beta_star_T0", "beta_star_300K", "alpha_c_T0", "alpha_c_300K",
            "xi_pi_T0", "xi_pi_300K", "V_PI_0K", "V_PI_300K", "dV_PI_pct",
            "above_ceiling"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})

    return json_path, csv_path


def main():
    rows = [compute_device(*d) for d in DEVICES]
    print_table(rows)
    json_path, csv_path = save_outputs(rows)
    print(f"\n[save] {json_path}")
    print(f"[save] {csv_path}")

    # Provenance echo for the paper text.
    print("\n[detail] per-device dimensionless / thermal numbers")
    for r in rows:
        print(f"  {r['name']}: m_eff={r['m_eff_kg']*1e15:8.3f} fg  "
              f"alpha/V^2={r['alpha_over_V2']:.4f} 1/V^2  "
              f"kappa*T(300K)={r['kappaT_300K']:.4f}  "
              f"beta*(300K)={r['beta_star_300K']:.5f}  "
              f"alpha_c: {_fmt(r['alpha_c_T0'],'.5f')}->{_fmt(r['alpha_c_300K'],'.5f')}")


if __name__ == "__main__":
    main()

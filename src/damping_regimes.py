"""Which from-rest threshold applies to the tabulated devices, and when.

The bracket of Appendix B places the from-rest threshold between the undamped
curve and the static fold, and the barrier theorem puts it on the fold once
the damping ratio exceeds ``2^(-1/4)``.  The devices of Table I carry quality
factors of 1e3 to 1e4, so in vacuum they sit at the lower edge of the
bracket.  In a gas the squeeze film between the plates adds a damping that,
for sub-micrometre gaps, can exceed the theorem's constant.  This module
estimates that damping and reports, per device,

  * the zero-bias equilibrium ``xi_0``, from ``xi_0 (1 - xi_0)^4 = beta``.  A
    step bias applied to a device at rest starts the orbit there, not at
    ``xi = 0``;
  * the undamped step-actuation threshold, from ``V(xi_s) = V(xi_start)``, and
    the corresponding fraction of the static pull-in voltage, for release
    from ``xi = 0`` (the problem of the appendix) and from ``xi_0``;
  * the squeeze-film damping ratio in air at 300 K and 1 atm, in the
    incompressible Reynolds limit and with the gas compressibility of the
    linearized Blech solution evaluated at the natural frequency, together
    with the squeeze number;
  * the damping ``zeta_c(beta)`` at which the band closes for that device,
    by the phase-plane criterion at the fold, and the pressures at which the
    lower damping estimate reaches ``zeta_c`` and ``2^(-1/4)``;
  * for a device whose air damping lies below ``zeta_c``, the band width at
    1 atm, with a constant damping ratio and with the damping evaluated at the
    instantaneous gap.

Gas model.  Squeeze-film force on a square plate of side ``a`` at gap ``h``,
isothermal linearized Reynolds equation (Blech 1983), with the rarefied-gas
effective viscosity ``mu / (1 + 9.638 Kn^1.159)`` of Veijola et al. (1995),
``Kn = lambda / h`` and ``lambda = (mu / p) sqrt(pi R T / (2 M))``.  Edge
effects and perforations are neglected.  The incompressible coefficient
``0.4217 mu_eff a^4 / h^3`` is an upper estimate of the damping, and the
Blech value at ``omega_0`` a lower one, because the overdamped approach to
pull-in is slower than the natural period.

Run:  python -m src.damping_regimes
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

try:
    from src import config
    from src.device_table import DEVICES, beta_param
except ImportError:  # pragma: no cover - direct execution
    from . import config
    from .device_table import DEVICES, beta_param

OUT = os.path.join(str(config.RESULTS_DIR), "damping_regimes.json")

# air at 300 K
T_GAS = 300.0
MU_AIR = 1.846e-5          # Pa s, dry air at 300 K
M_AIR = 0.0289647          # kg/mol
R_GAS = 8.314462618
P_ATM = 101325.0
ZETA_UNIFORM = 2.0 ** -0.25
ZETA_C0_APPENDIX = 0.395919  # appendix value at beta = 0, release from xi = 0


# ------------------------------------------------------------ gas damping
def mean_free_path(p):
    return MU_AIR / p * np.sqrt(np.pi * R_GAS * T_GAS / (2.0 * M_AIR))


def mu_eff(p, h):
    kn = mean_free_path(p) / h
    return MU_AIR / (1.0 + 9.638 * kn ** 1.159)


_ODD = np.arange(1, 400, 2, dtype=float)
_M, _N = np.meshgrid(_ODD, _ODD, indexing="ij")
_S0 = float(np.sum(1.0 / (_M ** 2 * _N ** 2 * (_M ** 2 + _N ** 2))))
C_SQUARE = 768.0 * _S0 / np.pi ** 6          # 0.4217 for a square plate


def c_incompressible(p, h, a):
    return C_SQUARE * mu_eff(p, h) * a ** 4 / h ** 3


def squeeze_number(p, h, a, omega):
    return 12.0 * mu_eff(p, h) * omega * a ** 2 / (p * h ** 2)


def c_blech(p, h, a, omega):
    """Damping coefficient of the linearized compressible film at omega."""
    s = squeeze_number(p, h, a, omega)
    k2 = _M ** 2 + _N ** 2
    terms = k2 / (_M ** 2 * _N ** 2 * (k2 ** 2 + s ** 2 / np.pi ** 4))
    return 64.0 * s * p * a ** 2 / (np.pi ** 6 * h * omega) * float(np.sum(terms))


# ------------------------------------------------------- dimensionless model
def g(xi, alpha, beta):
    u = 1.0 - xi
    return alpha / u ** 2 + beta / u ** 4 - xi


def potential(xi, alpha, beta):
    u = 1.0 - xi
    return 0.5 * xi ** 2 - alpha / u - beta / (3.0 * u ** 3)


def fold_of_beta(beta):
    """(alpha_c, xi_f) on the fold at the given beta, from the closed form."""
    if beta <= 0.0:
        return 4.0 / 27.0, 1.0 / 3.0
    u = brentq(lambda u: 0.5 * u ** 4 * (3 * u - 2) - beta, 2.0 / 3.0, 0.8,
               xtol=1e-15)
    return 0.5 * u ** 2 * (4 - 5 * u), 1.0 - u


def zero_bias_equilibrium(beta):
    """Stable root of xi (1 - xi)^4 = beta, the rest position at V = 0."""
    if beta <= 0.0:
        return 0.0
    return brentq(lambda x: x * (1 - x) ** 4 - beta, 0.0, 0.2, xtol=1e-16)


def saddle(alpha, beta):
    """Upper equilibrium (smaller gap) for alpha below the fold."""
    m = minimize_scalar(lambda x: g(x, alpha, beta), bounds=(0.0, 0.95),
                        method="bounded", options={"xatol": 1e-13})
    if g(m.x, alpha, beta) >= 0.0:
        return None
    return brentq(lambda x: g(x, alpha, beta), m.x, 1 - 1e-12, xtol=1e-15)


def alpha_undamped(beta, xi_start):
    """Undamped step threshold: V(xi_s) = V(xi_start) at the saddle."""
    a_c, _ = fold_of_beta(beta)

    def deficit(alpha):
        xs = saddle(alpha, beta)
        return potential(xs, alpha, beta) - potential(xi_start, alpha, beta)
    return brentq(deficit, 1e-9 if beta > 0 else 1e-6, a_c * (1 - 1e-9),
                  xtol=1e-14)


def alpha_mmxy(beta):
    """Closed form of McLellan et al. in the gap variable (release from 0)."""
    if beta <= 0.0:
        return 0.125
    u = brentq(lambda u: 3 * u ** 3 * (2 * u - 1) / (2 * (u + 2)) - beta,
               0.5, 0.6914140, xtol=1e-15)
    return u * (3 - 2 * u - 2 * u ** 2 - 2 * u ** 3) / (2 * (u + 2))


def dies_at_fold(beta, xi_start, zeta_of_xi):
    """True if, at alpha = alpha_c, the from-rest speed dies before xi_f.

    Phase-plane criterion: with q = xi_dot as a function of xi,
    dq/dxi = g/q - 2 zeta(xi).  The speed either dies before xi_f (no
    overshoot) or reaches xi_f positive.  No time enters, so the bottleneck
    at the fold cannot be mistaken for convergence.  If the speed dies, the
    orbit does not collapse at alpha_c, and by the monotonicity in alpha of
    Lemma 2 at no smaller alpha either: the threshold is the fold itself.
    """
    a_c, xi_f = fold_of_beta(beta)
    alpha = a_c * (1 - 1e-12)
    eps = 1e-12
    x0 = xi_start + eps
    q0 = np.sqrt(2.0 * g(x0, alpha, beta) * eps)

    def rhs(x, y):
        return [g(x, alpha, beta) / max(y[0], 1e-300) - 2.0 * zeta_of_xi(x)]

    def stop(x, y):
        return y[0] - 1e-14
    stop.terminal, stop.direction = True, -1.0
    s = solve_ivp(rhs, (x0, xi_f - 1e-12), [q0], method="Radau",
                  rtol=1e-13, atol=1e-16, events=stop)
    return len(s.t_events[0]) > 0


def zeta_c(beta, xi_start):
    """Constant damping at which the from-rest orbit stops crossing xi_f."""
    def dies(zeta):
        return dies_at_fold(beta, xi_start, lambda x: zeta)

    lo, hi = 0.30, 0.45
    if dies(lo) or not dies(hi):
        raise RuntimeError(f"zeta_c(beta={beta}) outside [{lo}, {hi}]")
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if dies(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def collapses(alpha, beta, xi_start, zeta_of_xi):
    """True if the orbit released at rest from xi_start crosses the saddle.

    Decided by events, not by a horizon: crossing the saddle means collapse,
    and the speed vanishing first means the orbit turns back, after which the
    energy, nonincreasing for any nonnegative damping, keeps it below the
    saddle (Lemma 2).
    """
    xs = saddle(alpha, beta)

    def rhs(t, y):
        return [y[1], -2.0 * zeta_of_xi(y[0]) * y[1] + g(y[0], alpha, beta)]

    def hit(t, y):
        return y[0] - xs
    hit.terminal, hit.direction = True, 1.0

    def turn(t, y):
        return y[1]
    turn.terminal, turn.direction = True, -1.0

    s = solve_ivp(rhs, (0.0, 1e4), [xi_start, 0.0], method="DOP853",
                  rtol=1e-11, atol=1e-13, events=(hit, turn))
    if len(s.t_events[0]):
        return True
    if len(s.t_events[1]):
        return False
    raise RuntimeError("neither crossing nor turning point within horizon")


def alpha_dyn(beta, xi_start, zeta_of_xi):
    a_c, _ = fold_of_beta(beta)
    if dies_at_fold(beta, xi_start, zeta_of_xi):
        return a_c
    lo = alpha_undamped(beta, xi_start)
    hi = a_c * (1 - 1e-10)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if collapses(mid, beta, xi_start, zeta_of_xi):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ driver
def device_report(name, desc, A_um2, d_nm, k, Q, f0):
    A = A_um2 * 1e-12
    d = d_nm * 1e-9
    a = np.sqrt(A)
    omega0 = 2.0 * np.pi * f0
    m = k / omega0 ** 2
    crit = 2.0 * np.sqrt(k * m)
    beta = float(beta_param(A, d, k))
    a_c, xi_f = fold_of_beta(beta)
    xi0 = zero_bias_equilibrium(beta)
    z_int = 1.0 / (2.0 * Q)

    def zeta_gas(p, which):
        if which == "incompressible":
            return c_incompressible(p, d, a) / crit
        return c_blech(p, d, a, omega0) / crit

    z_hi = zeta_gas(P_ATM, "incompressible")
    z_lo = zeta_gas(P_ATM, "blech")
    sigma = squeeze_number(P_ATM, d, a, omega0)

    a_low0 = alpha_mmxy(beta)
    a_lowr = alpha_undamped(beta, xi0)
    zc = zeta_c(beta, xi0)

    def pressure_for(target):
        f = lambda lp: z_int + zeta_gas(10.0 ** lp, "blech") - target
        if f(np.log10(P_ATM)) < 0:
            return None
        return float(10.0 ** brentq(f, -3.0, np.log10(P_ATM), xtol=1e-6))

    rep = {
        "name": name, "desc": desc, "A_um2": A_um2, "d_nm": d_nm,
        "k_N_per_m": k, "Q": Q, "f0_Hz": f0, "m_eff_kg": m,
        "beta": beta, "alpha_c": a_c, "xi_f": xi_f,
        "xi0_zero_bias": xi0,
        "alpha_undamped_from_0": a_low0,
        "alpha_undamped_from_xi0": a_lowr,
        "Vdyn_over_VPI_vacuum_from_0": float(np.sqrt(a_low0 / a_c)),
        "Vdyn_over_VPI_vacuum_from_xi0": float(np.sqrt(a_lowr / a_c)),
        "zeta_intrinsic": z_int,
        "mean_free_path_1atm_nm": mean_free_path(P_ATM) * 1e9,
        "knudsen_1atm": mean_free_path(P_ATM) / d,
        "squeeze_number_1atm": sigma,
        "zeta_air_incompressible": z_int + z_hi,
        "zeta_air_blech": z_int + z_lo,
        "zeta_c_from_xi0": zc,
        "p_for_zeta_c_Pa": pressure_for(zc),
        "p_for_uniform_constant_Pa": pressure_for(ZETA_UNIFORM),
    }

    if z_int + z_lo < zc:
        const = lambda x, z=z_int + z_lo: z
        c_atm = c_incompressible(P_ATM, d, a)

        def local(x):
            h = d * (1.0 - x)
            return z_int + (c_incompressible(P_ATM, h, a) / c_atm) * z_lo
        ad_c = alpha_dyn(beta, xi0, const)
        ad_l = alpha_dyn(beta, xi0, local)
        rep["band_1atm_constant"] = {
            "alpha_dyn": ad_c, "width": a_c - ad_c,
            "Vdyn_over_VPI": float(np.sqrt(ad_c / a_c))}
        rep["band_1atm_gap_dependent"] = {
            "alpha_dyn": ad_l, "width": a_c - ad_l,
            "Vdyn_over_VPI": float(np.sqrt(ad_l / a_c))}
    return rep


def self_checks():
    """Reproduce two values the appendix obtains independently."""
    z = zeta_c(0.0, 0.0)
    if abs(z - ZETA_C0_APPENDIX) > 2e-5:
        raise SystemExit(f"zeta_c(0) = {z:.6f}, appendix {ZETA_C0_APPENDIX}")
    for beta in (0.0, 0.02, 0.05):
        x, y = alpha_undamped(beta, 0.0), alpha_mmxy(beta)
        if abs(x - y) > 1e-9:
            raise SystemExit(f"undamped threshold {x} vs closed form {y}")
    ad = alpha_dyn(0.0, 0.0, lambda x: 0.1)
    if abs((4.0 / 27.0 - ad) - 1.3482e-2) > 5e-6:
        raise SystemExit(f"band width at zeta = 0.1: {4/27 - ad:.6e}")
    print(f"self-checks: zeta_c(0) = {z:.6f}, closed form reproduced, "
          f"band(0.1) = {4/27 - ad:.4e}")
    return z


def main():
    print(f"square-plate Reynolds coefficient {C_SQUARE:.5f}")
    z0 = self_checks()
    rows = [device_report(*dev) for dev in DEVICES]
    for r in rows:
        print(f"\n{r['name']}: beta={r['beta']:.4e}  xi0={r['xi0_zero_bias']:.4f}"
              f"  Kn={r['knudsen_1atm']:.2f}  sigma={r['squeeze_number_1atm']:.2f}")
        print(f"   vacuum: Vdyn/VPI = {r['Vdyn_over_VPI_vacuum_from_0']:.4f} "
              f"(from 0), {r['Vdyn_over_VPI_vacuum_from_xi0']:.4f} (from xi0)")
        print(f"   air: zeta in [{r['zeta_air_blech']:.3f}, "
              f"{r['zeta_air_incompressible']:.3f}]   zeta_c = "
              f"{r['zeta_c_from_xi0']:.4f}")
        print(f"   p(zeta_c) = {r['p_for_zeta_c_Pa']}   "
              f"p(2^-1/4) = {r['p_for_uniform_constant_Pa']}")
        for key in ("band_1atm_constant", "band_1atm_gap_dependent"):
            if key in r:
                b = r[key]
                print(f"   {key}: width {b['width']:.3e}  Vdyn/VPI "
                      f"{b['Vdyn_over_VPI']:.5f}")
    payload = {
        "gas": {"T_K": T_GAS, "mu_Pa_s": MU_AIR, "M_kg_per_mol": M_AIR,
                "p_atm_Pa": P_ATM,
                "effective_viscosity": "mu/(1+9.638 Kn^1.159), Veijola 1995",
                "film": "isothermal linearized Reynolds, square plate, Blech 1983",
                "square_coefficient": C_SQUARE},
        "zeta_uniform": ZETA_UNIFORM,
        "zeta_c_beta0_from_0": z0,
        "devices": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {OUT}")


if __name__ == "__main__":
    main()

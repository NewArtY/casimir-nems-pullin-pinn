"""Numbers quoted in Sec. III and Appendix B that no other stage produces.

Each block is a small deterministic computation; the text quotes its output.

1. Equilibrium count of Theorem "static" on random points of the quadrant.
2. Collapse exponent: fits of ln u against ln(tau_* - tau) on the archived
   adaptive reference (Fig. 1c), and an independent run at beta = 0.
3. Speed q_f at the fold point at alpha = alpha_c, beta = 0, by the
   phase-plane equation dq/dxi = g/q - 2 zeta (no time horizon).
4. Saddle-node passage above zeta_c: tau_PI sqrt(alpha - alpha_c) / K.
5. Hyperbolic-saddle escape below zeta_c: d tau_PI / d ln(alpha - alpha_dyn)
   against -1/lambda_+ at beta = 0, zeta = 0.1.
6. Logarithmic against power-law fits of the archived phase diagram near
   the dynamic boundary (Fig. 2), on eight rows.
7. The constants Theta, Omega, L of the a posteriori bound (Theorem 5).

Writes ``results/appendix_numbers.json``.  Run:  python -m src.appendix_numbers
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

try:
    from src import config
    from src.damping_regimes import g, fold_of_beta, saddle
    from src.threshold_checks import alpha_dyn_manifold, gprime
except ImportError:  # pragma: no cover
    from . import config
    from .damping_regimes import g, fold_of_beta, saddle
    from .threshold_checks import alpha_dyn_manifold, gprime

print = functools.partial(print, flush=True)  # noqa: A001
RES = config.RESULTS_DIR
OUT = os.path.join(str(RES), "appendix_numbers.json")


def tau_pullin(alpha, beta, zeta, gap=1e-4, t_max=1e6):
    """Time to reach 1 - xi = gap from rest at xi = 0 (DOP853)."""
    def rhs(t, y):
        return [y[1], -2 * zeta * y[1] + g(y[0], alpha, beta)]

    def ev(t, y):
        return y[0] - (1 - gap)
    ev.terminal = True
    s = solve_ivp(rhs, (0, t_max), [0.0, 0.0], method="DOP853", rtol=1e-12,
                  atol=1e-14, events=ev)
    return float(s.t_events[0][0])


# ------------------------------------------------------------------ 1
def equilibrium_count(n=3000, band=2e-4, n_u=400000, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 0.22, n)
    b = rng.uniform(0, 0.10, n)
    u = np.linspace(1e-6, 1 - 1e-9, n_u)
    kept = agree = 0
    for ai, bi in zip(a, b):
        if bi <= 256 / 3125:
            ac = fold_of_beta(bi)[0]
            if abs(ai - ac) < band:
                continue
            expect = 2 if ai < ac else 0
        else:
            expect = 0
        kept += 1
        phi = (1 - u) * u ** 4 - ai * u ** 2 - bi
        n_roots = int(np.sum(np.sign(phi[1:]) != np.sign(phi[:-1])))
        agree += int(n_roots == expect)
    return dict(kept=kept, agree=agree)


# ------------------------------------------------------------------ 2
def exponent_fits():
    d = np.load(os.path.join(RES, "traj_pullin_ref_full.npz"))
    tau_s = float(d["tau_star"])
    t, xi = d["t"], d["xi"]
    s = tau_s - t
    u = 1 - xi
    out = []
    for lo, hi in ((1e-3, 1e-2), (1e-4, 1e-3), (2e-5, 1e-4)):
        m = (s > lo) & (s < hi)
        p, c = np.polyfit(np.log(s[m]), np.log(u[m]), 1)
        out.append(dict(window=[lo, hi], exponent=float(p),
                        amplitude=float(np.exp(c)), n=int(m.sum())))
    beta = float(d["beta"])
    C = (25 * beta / 6) ** 0.2
    # independent run at beta = 0, alpha = 0.3, zeta = 0.5
    a0, z0 = 0.3, 0.5

    def rhs(t, y):
        return [y[1], -2 * z0 * y[1] + g(y[0], a0, 0.0)]
    ev = lambda t, y: y[0] - (1 - 1e-6)
    ev.terminal = True
    sol = solve_ivp(rhs, (0, 50), [0, 0], method="DOP853", rtol=1e-13,
                    atol=1e-15, events=ev, dense_output=True)
    t_end = float(sol.t[-1])
    u_end = 1 - float(sol.y[0, -1])
    C2 = (9 * a0 / 2) ** (1 / 3)
    ts2 = t_end + (u_end / C2) ** 1.5          # remaining time to contact
    tt = np.linspace(0, t_end, 400001)
    ss = ts2 - tt
    uu = 1 - sol.sol(tt)[0]
    m = (ss > 2e-5) & (ss < 1e-4)
    p2 = float(np.polyfit(np.log(ss[m]), np.log(uu[m]), 1)[0])
    return dict(tau_star=tau_s, C=C, windows=out, beta0_exponent=p2)


# ------------------------------------------------------------------ 3
def q_fold(zeta, beta=0.0):
    a_c, xi_f = fold_of_beta(beta)
    alpha = a_c
    eps = 1e-12
    q0 = np.sqrt(2.0 * g(eps, alpha, beta) * eps)

    def rhs(x, y):
        return [g(x, alpha, beta) / max(y[0], 1e-300) - 2.0 * zeta]

    def stop(x, y):
        return y[0] - 1e-14
    stop.terminal, stop.direction = True, -1.0
    s = solve_ivp(rhs, (eps, xi_f), [q0], method="Radau", rtol=1e-13,
                  atol=1e-16, events=stop)
    if len(s.t_events[0]):
        return 0.0, float(xi_f - s.t_events[0][0])
    return float(s.y[0, -1]), None


# ------------------------------------------------------------------ 4
def bottleneck(zeta, beta, mus):
    a_c, xi_f = fold_of_beta(beta)
    u = 1 - xi_f
    K = 2 * np.pi * zeta * u ** 2 * np.sqrt(2 / (15 * u - 8))
    taus = [tau_pullin(a_c + m, beta, zeta) for m in mus]
    ratio = [t * np.sqrt(m) / K for t, m in zip(taus, mus)]
    slope = float(np.log(taus[-1] / taus[-2]) / np.log(mus[-1] / mus[-2]))
    return dict(zeta=zeta, beta=beta, K=K, mu=list(mus), tau=taus,
                ratio=ratio, local_slope=slope)


# ------------------------------------------------------------------ 5
def saddle_escape(zeta=0.1, beta=0.0):
    ad = alpha_dyn_manifold(beta, zeta)
    xs = saddle(ad, beta)
    gp = gprime(xs, ad, beta)
    lam = -zeta + np.sqrt(zeta ** 2 + gp)
    ds = 10.0 ** np.arange(-6, -11, -1.0)
    taus = np.array([tau_pullin(ad + d, beta, zeta) for d in ds])
    slope = float(np.polyfit(np.log(ds), taus, 1)[0])
    return dict(alpha_dyn=ad, xi_s=xs, gprime=gp, lambda_plus=lam,
                minus_inv_lambda=-1 / lam, measured_slope=slope,
                deltas=ds.tolist(), taus=taus.tolist())


# ------------------------------------------------------------------ 6
def phase_rows(n_rows=8, n_cells=14):
    p = np.load(os.path.join(RES, "phase_diagram.npz"))
    A, B = p["alpha_axis"], p["beta_axis"]
    mask, tau = p["pullin_mask"], p["tau_pullin"]
    rows_idx = np.round(np.linspace(0, 155, n_rows)).astype(int)
    out = []
    for i in rows_idx:
        pull = mask[i]
        j0 = int(np.argmax(pull))              # first pull-in cell
        js = np.arange(j0, j0 + n_cells)
        a, t = A[js], tau[i, js]
        a_hi = A[j0]                            # boundary lies in (A[j0-1], A[j0])
        a_lo = A[j0 - 1]

        def r_log(x):
            return x[0] * np.log(a - x[2]) + x[1] - t

        def r_pow(x):
            return x[0] * np.log(a - x[2]) + x[1] - np.log(t)
        x0 = [-1.3, 0.0, 0.5 * (a_lo + a_hi)]
        bnds = ([-10, -100, a_lo - 1e-3], [10, 100, a_hi - 1e-12])
        fl = least_squares(r_log, x0, bounds=bnds, xtol=1e-15, ftol=1e-15)
        fp = least_squares(r_pow, [-0.3, 0, x0[2]], bounds=bnds, xtol=1e-15,
                           ftol=1e-15)
        # compare residuals in the same variable, tau
        res_log = float(np.sqrt(np.mean(fl.fun ** 2)))
        res_pow = float(np.sqrt(np.mean((np.exp(fp.fun + np.log(t)) - t) ** 2)))
        out.append(dict(beta=float(B[i]), A=float(fl.x[0]),
                        alpha_dyn=float(fl.x[2]), p=float(fp.x[0]),
                        rms_log=res_log, rms_pow=res_pow))
    # the largest tau_PI of the sweep, and its distance from the boundary
    # given by the logarithmic fit of its own row
    tt = np.where(mask, tau, np.nan)
    i, j = np.unravel_index(np.nanargmax(tt), tt.shape)
    j0 = int(np.argmax(mask[i]))
    js = np.arange(j0, j0 + n_cells)
    a, t = A[js], tau[i, js]
    fl = least_squares(lambda x: x[0] * np.log(a - x[2]) + x[1] - t,
                       [-1.3, 0.0, 0.5 * (A[j0 - 1] + A[j0])],
                       bounds=([-10, -100, A[j0 - 1] - 1e-3],
                               [10, 100, A[j0] - 1e-12]),
                       xtol=1e-15, ftol=1e-15)
    return dict(rows=out, tau_max=float(tt[i, j]), beta_at_max=float(B[i]),
                distance_at_max=float(A[j] - fl.x[2]),
                d_alpha=float(A[1] - A[0]))


# ------------------------------------------------------------------ 7
def gronwall_constants():
    d = np.load(os.path.join(RES, "traj_pullin.npz"))
    a, b, z, T = (float(d[k]) for k in ("alpha", "beta", "zeta", "T"))
    xr, vr = d["xi_ref"], d["xidot_ref"]
    xp, vp = d["xi_pinn"], d["xidot_pinn"]
    th = lambda x: -np.log(1 - x)
    Theta = float(max(th(xr).max(), th(xp).max()))
    Omega = float(max(np.abs(vr / (1 - xr)).max(), np.abs(vp / (1 - xp)).max()))
    terms = dict(exp=np.exp(Theta), elec=3 * a * np.exp(3 * Theta),
                 cas=5 * b * np.exp(5 * Theta), damp=2 * (Omega + z))
    L = max(1.0, sum(terms.values()))
    return dict(T=T, Theta=Theta, Omega=Omega, L=float(L),
                casimir_term=float(terms["cas"]), LT=float(L * T))


# ------------------------------------------------------------------ 8
def beta_star_exact(d=100e-9, temps=(100.0, 300.0)):
    """Casimir-only intercept under the zero-frequency bound, by maximizing
    (1-u) u^4 / (1 + gamma u) with gamma = kappa(d) T (Proposition thermal)."""
    from scipy.optimize import minimize_scalar
    kappa = (60 * config.ZETA3 / np.pi ** 3) * config.K_B * d / (
        config.HBAR * config.C_LIGHT)
    out = {}
    for T in temps:
        gam = kappa * T
        r = minimize_scalar(lambda u: -(1 - u) * u ** 4 / (1 + gam * u),
                            bounds=(0.5, 0.95), method="bounded",
                            options={"xatol": 1e-13})
        out[str(T)] = dict(gamma=gam, beta_star=float(-r.fun),
                           first_order=float(256 / 3125 - 1024 / 15625 * gam))
    return out


# ------------------------------------------------------------------ 9
def zero_bias_collapse(beta=0.08, zeta=0.1):
    """Bounded pull-in time when the zero-bias orbit itself collapses."""
    return {str(a): tau_pullin(a, beta, zeta, gap=1e-3)
            for a in (0.0, 1e-6, 1e-4)}


def main():
    config.set_all_seeds()
    out = {}
    out["beta_star_exact"] = beta_star_exact()
    print("8.", out["beta_star_exact"])
    out["zero_bias_collapse_beta0.08_zeta0.1_tau_to_0.999"] = zero_bias_collapse()
    print("9.", out["zero_bias_collapse_beta0.08_zeta0.1_tau_to_0.999"])
    out["equilibrium_count"] = equilibrium_count()
    print("1.", out["equilibrium_count"])
    out["exponent"] = exponent_fits()
    print("2.", json.dumps(out["exponent"], indent=1))
    out["q_f"] = {}
    for z in (0.1, 0.3, 0.39, 0.396, 0.40):
        q, at = q_fold(z)
        out["q_f"][str(z)] = dict(q_f=q, dies_at_distance=at)
        print(f"3. zeta={z}: q_f={q:.4e}  dies at xi_f - xi = {at}")
    out["bottleneck"] = [bottleneck(1.0, 0.0, [1e-7, 1e-8]),
                         bottleneck(0.7, 0.0, [1e-7, 1e-8]),
                         bottleneck(0.7, 0.03, [1e-6, 1e-7])]
    for r in out["bottleneck"]:
        print(f"4. zeta={r['zeta']} beta={r['beta']}: ratio={r['ratio']}  "
              f"local slope={r['local_slope']:.5f}")
    out["saddle_escape"] = saddle_escape()
    print("5.", {k: v for k, v in out["saddle_escape"].items()
                 if k not in ("deltas", "taus")})
    out["phase_rows"] = phase_rows()
    for r in out["phase_rows"]["rows"]:
        print(f"6. beta={r['beta']:.4f} A={r['A']:.3f} p={r['p']:.3f} "
              f"alpha_dyn={r['alpha_dyn']:.6f} rms log/pow="
              f"{r['rms_log']:.3e}/{r['rms_pow']:.3e}")
    out["gronwall"] = gronwall_constants()
    print("7.", out["gronwall"])
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()

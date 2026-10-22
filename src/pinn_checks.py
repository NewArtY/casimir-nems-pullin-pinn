"""Residual audit and parametrization ablation for the pull-in network.

Three checks on the pull-in trajectory network of Fig. 1(c).

1. Residuals.  The network is parametrized in the rapidity theta, with
   xi = 1 - exp(-theta), but it is trained on the residual of the original
   equation, r_xi = xi'' + 2 zeta xi' + xi - alpha/(1-xi)^2 - beta/(1-xi)^4,
   evaluated on that xi.  The residual of the theta equation is
   r_theta = exp(theta) r_xi.  Both are evaluated here on the collocation
   points (where the mean square of r_xi must reproduce the logged loss) and
   on a grid 125 times denser, which estimates the sup norm better than the
   root mean square does.  A grid value is still a sample, not a bound.

2. The a posteriori estimate of Appendix B (Theorem 5), evaluated with the
   dense-grid sup of r_theta in place of ||r_theta||_inf: the horizon up to
   which the local form keeps |theta_NN - theta| below R.  Because the input
   is a sample of the residual, the result is an estimate, not a certificate.

3. Ablation.  The same architecture, collocation grid, optimizer schedule,
   seed, and window, trained with the raw parametrization (hard initial
   conditions in xi, soft barrier at xi >= 1 - 1e-3), against the archived
   rapidity network.

Writes ``results/pinn_checks.json``.  Run:  python -m src.pinn_checks
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

try:
    from src import config
    from src.pinn import load_model, train_pinn, evaluate_xi_xidot, save_model
except ImportError:  # pragma: no cover
    from . import config
    from .pinn import load_model, train_pinn, evaluate_xi_xidot, save_model

OUT = os.path.join(str(config.RESULTS_DIR), "pinn_checks.json")
DT = torch.float64


def residuals(model, tau_np):
    """(r_xi, r_theta, theta, theta_dot) on the grid tau_np."""
    a, b, z = float(model.alpha), float(model.beta), float(model.zeta)
    tau = torch.as_tensor(tau_np, dtype=DT).reshape(-1, 1).requires_grad_(True)
    xi = model(tau)
    one = torch.ones_like(xi)
    xd = torch.autograd.grad(xi, tau, one, create_graph=True)[0]
    xdd = torch.autograd.grad(xd, tau, torch.ones_like(xd), create_graph=True)[0]
    r_xi = xdd + 2 * z * xd + xi - a / (1 - xi) ** 2 - b / (1 - xi) ** 4
    th = -torch.log(1 - xi)
    thd = torch.autograd.grad(th, tau, torch.ones_like(th), create_graph=True)[0]
    thdd = torch.autograd.grad(thd, tau, torch.ones_like(thd))[0]
    r_th = (thdd + 2 * z * thd + (torch.exp(th) - 1) - thd ** 2
            - a * torch.exp(3 * th) - b * torch.exp(5 * th))
    f = lambda t: t.detach().reshape(-1).numpy()
    return f(r_xi), f(r_th), f(th), f(thd)


def reference(a, b, z, T, n):
    def rhs(t, y):
        return [y[1], -2 * z * y[1] - y[0] + a / (1 - y[0]) ** 2
                + b / (1 - y[0]) ** 4]
    t = np.linspace(0.0, T, n)
    s = solve_ivp(rhs, (0, T), [0.0, 0.0], method="DOP853", rtol=1e-12,
                  atol=1e-14, t_eval=t)
    xi, v = s.y
    return t, xi, v


def local_horizon(t, theta, thetad, r_inf, R, a, b, z):
    """Largest tau with B_R <= R, Eq. (B-local) of Appendix B."""
    LR = np.maximum(1.0, np.exp(theta + R) + 3 * a * np.exp(3 * (theta + R))
                    + 5 * b * np.exp(5 * (theta + R))
                    + 2 * (np.abs(thetad) + R + z))
    # B(tau) = r_inf * int_0^tau exp(int_s^tau L) ds, with I(t) = int_0^t L:
    # B = r_inf * exp(I(tau)) * int_0^tau exp(-I(s)) ds
    dt = t[1] - t[0]
    I = np.concatenate([[0.0], np.cumsum(0.5 * (LR[1:] + LR[:-1]) * dt)])
    e = np.exp(-I)
    J = np.concatenate([[0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * dt)])
    B = r_inf * np.exp(I) * J
    ok = np.nonzero(B > R)[0]
    return float(t[ok[0] - 1]) if len(ok) else float(t[-1])


def main():
    config.set_all_seeds()
    out = {}
    model = load_model("pinn_pullin")
    a, b, z = float(model.alpha), float(model.beta), float(model.zeta)
    T = float(model.T)
    with open(os.path.join(str(config.PROJECT_ROOT), "logs",
                           "history_pullin.json"), encoding="utf-8") as fh:
        hist = json.load(fh)
    n_col = hist["config"]["n_collocation"]
    L_logged = hist["L_ode"][-1]

    # ---- 1. residuals ---------------------------------------------------
    tc = np.linspace(0.0, T, n_col)
    rx_c, rt_c, th_c, _ = residuals(model, tc)
    td = np.linspace(0.0, T, 125 * (n_col - 1) + 1)
    rx_d, rt_d, th_d, thd_d = residuals(model, td)
    res = dict(
        loss_logged=L_logged,
        loss_recomputed=float(np.mean(rx_c ** 2)),
        rms_rxi_colloc=float(np.sqrt(np.mean(rx_c ** 2))),
        sup_rxi_colloc=float(np.max(np.abs(rx_c))),
        sup_rxi_dense=float(np.max(np.abs(rx_d))),
        rms_rtheta_colloc=float(np.sqrt(np.mean(rt_c ** 2))),
        sup_rtheta_colloc=float(np.max(np.abs(rt_c))),
        sup_rtheta_dense=float(np.max(np.abs(rt_d))),
        tau_of_sup_rtheta=float(td[np.argmax(np.abs(rt_d))]),
        theta_max=float(th_d.max()), xi_max=float(1 - np.exp(-th_d.max())),
        n_dense=int(len(td)),
    )
    out["residuals"] = res
    print("1. residuals of the archived pull-in network")
    for k, v in res.items():
        print(f"   {k:>20}: {v}")

    # ---- 2. local Gronwall horizons ------------------------------------
    t, xi, v = reference(a, b, z, T, 200001)
    theta = -np.log(1 - xi)
    thetad = v / (1 - xi)
    hz = []
    for label, rinf in (("rms_rxi_as_in_v1.2", np.sqrt(L_logged)),
                        ("dense_sup_rtheta", res["sup_rtheta_dense"])):
        for R in (2.83e-3, 1e-2, 1e-1):
            tau_R = local_horizon(t, theta, thetad, rinf, R, a, b, z)
            xi_R = float(np.interp(tau_R, t, xi))
            hz.append(dict(input=label, r_inf=float(rinf), R=R,
                           tau=tau_R, xi=xi_R))
            print(f"2. {label:>20}  r={rinf:.3e}  R={R:g}  "
                  f"tau={tau_R:.3f}  xi={xi_R:.3f}")
    out["gronwall_local"] = hz

    # ---- 3. ablation: raw parametrization, identical budget -------------
    cfg = dict(hist["config"])
    cfg["parametrization"] = "raw"
    for k in ("seed",):
        cfg.pop(k, None)
    t0 = time.time()
    raw, h_raw = train_pinn(alpha=a, beta=b, zeta=z, xi0=0.0, v0=0.0, T=T,
                            seed=42, log_every=500, **cfg)
    wall = time.time() - t0
    save_model(raw, "pinn_pullin_raw")
    tr, xr, vr = reference(a, b, z, T, 6000)
    ab = {}
    for name, m in (("rapidity", model), ("raw", raw)):
        xp, vp = evaluate_xi_xidot(m, tr)
        e = np.abs(xp - xr)
        reg = xr < 0.5
        ab[name] = dict(E_inf=float(e.max()),
                        E_2=float(np.sqrt(np.mean(e ** 2))),
                        E_inf_regular=float(e[reg].max()),
                        E_inf_late=float(e[~reg].max()),
                        E_inf_v=float(np.abs(vp - vr).max()),
                        xi_max_net=float(xp.max()))
    ab["raw"]["L_ode_final"] = h_raw["L_ode"][-1]
    ab["raw"]["L_pull_final"] = h_raw["L_pull"][-1]
    ab["raw"]["wall_s"] = wall
    ab["rapidity"]["L_ode_final"] = L_logged
    ab["config"] = cfg
    out["ablation"] = ab
    print("3. ablation (same architecture, grid, schedule, seed, window)")
    for name in ("rapidity", "raw"):
        print(f"   {name:>9}: " + "  ".join(
            f"{k}={v:.3e}" for k, v in ab[name].items()
            if isinstance(v, float)))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()

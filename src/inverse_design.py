"""Inverse design with the differentiable physics-informed fold surrogate.

This script answers the referee's key demand: it shows the trained parametric
boundary surrogate ``models/pinn_param_boundary.pth`` is a *design tool*, not a
slower ODE solver.  The point is that the surrogate is DIFFERENTIABLE end to
end, so a designer gets exact gradients "for free" by ``torch`` autograd and
can invert a design spec by gradient descent -- amortised over the one-time
training cost.

Two demonstrations, both differentiating THROUGH the surrogate:

(1) Autodiff design sensitivities.
    * dalpha_c/dbeta  -- how the critical electrostatic bias on the pull-in
      fold moves with the Casimir loading, from the implicit-function theorem
      applied to the surrogate fold indicator M_net(alpha,beta)=0.
      Validated against the closed-form fold, for which dalpha_c/dbeta = -1/u^2
      exactly (u = 1 - xi_PI is the fold gap).
    * pull-in-voltage geometry sensitivities dV_PI/dd, dV_PI/dA, dV_PI/dk for
      device A, through the full differentiable pipeline

          (d, A, k) -> beta(d,A,k) -> alpha_c = surrogate_fold(beta) -> V_PI

      with V_PI = sqrt(2 alpha_c k d^3 / (eps0 A)).  autograd propagates the
      chain rule through the surrogate; validated against finite differences of
      the fully-analytic pipeline (Brent fold solve).

(2) Gradient-based inverse design.
    A device-engineer spec: "device A has fixed plate area A and spring k; what
    is the SMALLEST gap d (largest Casimir loading beta) the device tolerates
    while its pull-in voltage stays at a target V_PI*?"  As d shrinks, beta ~
    1/d^5 grows and V_PI drops, so V_PI(d) is monotone -- a well-posed inverse.
    Solved by Adam on d through the differentiable surrogate; the recovered
    (d*, beta*) is validated against a Brent root-find on the analytic V_PI(d).

Everything is CPU-only, float64, deterministic (seed 42), no training, < 5 s.

Run:  python src/inverse_design.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Robust imports: work as a package module and when run as a script.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_HERE))  # add .../PRA/code
    from src import config
    from src.pinn_boundary import (
        ParamMLP, ALPHA_MAX, BETA_MAX, ALPHA_EP, BETA_STAR,
    )
    from src import bifurcation as bif
else:
    from . import config
    from .pinn_boundary import (
        ParamMLP, ALPHA_MAX, BETA_MAX, ALPHA_EP, BETA_STAR,
    )
    from . import bifurcation as bif

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

PROJECT_ROOT = config.PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Physical constants (SI).
EPS0 = config.EPS0
HBAR = config.HBAR
C_LIGHT = config.C_LIGHT

# Device A (the paper's reference gold MEMS plate).
DEV_A = dict(A_um2=100.0, d_nm=100.0, k=0.5)   # A = 100 um^2, d = 100 nm, k = 0.5 N/m
DEV_A_SI = dict(A=100.0e-12, d=100.0e-9, k=0.5)

# Device B (the sub-100 nm NEMS of Table I): twice the Casimir loading.
DEV_B = dict(A_um2=25.0, d_nm=50.0, k=2.0)     # A = 25 um^2, d = 50 nm, k = 2 N/m
DEV_B_SI = dict(A=25.0e-12, d=50.0e-9, k=2.0)


# ===========================================================================
# Load the trained surrogate
# ===========================================================================
def load_surrogate(path: Path = MODELS_DIR / "pinn_param_boundary.pth") -> ParamMLP:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {"width": 64, "depth": 3})
    net = ParamMLP(width=meta["width"], depth=meta["depth"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)   # freeze; we differentiate w.r.t. INPUTS only
    return net


# ===========================================================================
# Differentiable surrogate primitives
# ===========================================================================
def u_net(net, alpha, beta):
    """Surrogate critical gap u*(alpha, beta); alpha, beta are scalar tensors."""
    ab = torch.stack([alpha.reshape(()), beta.reshape(())]).reshape(1, 2)
    return net(ab).reshape(())


def Mtil_net(net, alpha, beta):
    """Surrogate fold indicator M = alpha u^2 + beta - (1-u) u^4 ; =0 on the fold.

    Differentiable in alpha and beta through the network u*(alpha, beta).
    M > 0 : pull-in (no equilibrium) ; M < 0 : stable.
    """
    u = u_net(net, alpha, beta)
    a = alpha.reshape(())
    b = beta.reshape(())
    return a * u ** 2 + b - (1.0 - u) * u ** 4


def _bisect_alpha_c(net, beta_val, lo=1e-9, hi=None, iters=60):
    """Root alpha_c of M_net(alpha, beta_val)=0 on [lo, hi] (no grad, bisection).

    M is monotone increasing in alpha at fixed beta (more electrostatic drive ->
    pull-in), so a sign change is bracketed and bisection is robust.
    """
    if hi is None:
        hi = ALPHA_MAX
    with torch.no_grad():
        b = torch.as_tensor(float(beta_val), dtype=DTYPE)
        lo_t = torch.as_tensor(float(lo), dtype=DTYPE)
        hi_t = torch.as_tensor(float(hi), dtype=DTYPE)
        f_lo = Mtil_net(net, lo_t, b)
        f_hi = Mtil_net(net, hi_t, b)
        if f_lo * f_hi > 0:
            # no bracketed root in the window -> return the endpoint nearest zero
            return hi_t if abs(float(f_hi)) < abs(float(f_lo)) else lo_t
        for _ in range(iters):
            mid = 0.5 * (lo_t + hi_t)
            f_mid = Mtil_net(net, mid, b)
            if f_lo * f_mid <= 0:
                hi_t = mid
                f_hi = f_mid
            else:
                lo_t = mid
                f_lo = f_mid
        return 0.5 * (lo_t + hi_t)


def alpha_c_surrogate(net, beta):
    """Differentiable critical bias alpha_c(beta) from the surrogate fold.

    Root-solve alpha_c s.t. M_net(alpha_c, beta)=0 (detached), then reattach the
    gradient by the implicit-function theorem so that

        d alpha_c / d beta = -(dM/dbeta) / (dM/dalpha)   (exact),

    evaluated at the root through the differentiable network.  ``beta`` may carry
    an autograd graph (e.g. beta(d,A,k)); the chain rule then propagates through.
    """
    beta_det = beta.detach()
    a_star = _bisect_alpha_c(net, float(beta_det)).detach()

    # dM/dalpha at the root (constant coefficient in the IFT reattach).
    a_req = a_star.clone().requires_grad_(True)
    M_a = Mtil_net(net, a_req, beta_det)
    dM_da = torch.autograd.grad(M_a, a_req)[0].detach()

    # Reattach with the LIVE beta: value ~ a_star (M~0), gradient = IFT slope.
    M_live = Mtil_net(net, a_star, beta)          # depends on beta's graph
    return a_star - M_live / dM_da


def beta_of_geometry(A, d, k):
    """Casimir control parameter beta = pi^2 hbar c A / (240 k d^5) (differentiable)."""
    return (np.pi ** 2 * HBAR * C_LIGHT * A) / (240.0 * k * d ** 5)


def V_PI_surrogate(net, d, A, k):
    """Pull-in voltage through the differentiable surrogate pipeline.

    (d, A, k) -> beta -> alpha_c = surrogate_fold(beta) -> V_PI
              = sqrt(2 alpha_c k d^3 / (eps0 A)).
    """
    beta = beta_of_geometry(A, d, k)
    alpha_c = alpha_c_surrogate(net, beta)
    return torch.sqrt(2.0 * alpha_c * k * d ** 3 / (EPS0 * A))


# ===========================================================================
# Analytic ground truth (closed-form fold, Brent solve) for validation
# ===========================================================================
def alpha_c_analytic(beta):
    """Closed-form critical bias alpha_c on the fold at Casimir loading beta.

    Solve beta_c(u) = u^4 (3u-2)/2 = beta for u in [2/3, 4/5], then
    alpha_c = u^2 (4 - 5u)/2.  Returns (alpha_c, u).
    """
    if beta <= 0.0:
        return ALPHA_EP, 2.0 / 3.0
    if beta >= BETA_STAR:
        return 0.0, 4.0 / 5.0
    u = brentq(lambda uu: uu ** 4 * (3.0 * uu - 2.0) / 2.0 - beta,
               2.0 / 3.0, 4.0 / 5.0, xtol=1e-14, rtol=1e-14)
    a_c = u ** 2 * (4.0 - 5.0 * u) / 2.0
    return a_c, u


def dalpha_c_dbeta_analytic(beta):
    """Exact fold sensitivity dalpha_c/dbeta = -1/u^2 (see module docstring)."""
    _, u = alpha_c_analytic(beta)
    return -1.0 / u ** 2, u


def V_PI_analytic(d, A, k):
    """Fully-analytic pull-in voltage (Brent fold solve). NaN above the ceiling."""
    beta = (np.pi ** 2 * HBAR * C_LIGHT * A) / (240.0 * k * d ** 5)
    if beta >= BETA_STAR:
        return float("nan")
    a_c, _ = alpha_c_analytic(beta)
    return float(np.sqrt(2.0 * a_c * k * d ** 3 / (EPS0 * A)))


# ===========================================================================
# DEMO 1 -- autodiff design sensitivities
# ===========================================================================
def demo_sensitivities(net):
    print("-" * 72)
    print("DEMO 1: autodiff design sensitivities THROUGH the surrogate")
    print("-" * 72)

    # --- (1a) dalpha_c/dbeta at three operating points ---------------------
    beta_points = [0.010, DEV_A_beta(), 0.050]
    dadb_rows = []
    print("  (1a) fold sensitivity  dalpha_c/dbeta  (surrogate autograd vs -1/u^2)")
    print(f"       {'beta':>8} {'alpha_c':>10} {'surrogate':>12} "
          f"{'analytic':>12} {'|rel err|':>10}")
    for b0 in beta_points:
        beta_t = torch.tensor(float(b0), dtype=DTYPE, requires_grad=True)
        a_c = alpha_c_surrogate(net, beta_t)
        (dadb_sur,) = torch.autograd.grad(a_c, beta_t)
        dadb_sur = float(dadb_sur)
        dadb_an, u = dalpha_c_dbeta_analytic(b0)
        rel = abs(dadb_sur - dadb_an) / abs(dadb_an)
        a_c_an, _ = alpha_c_analytic(b0)
        a_c = float(a_c.detach())
        print(f"       {b0:>8.4f} {a_c:>10.5f} {dadb_sur:>12.5f} "
              f"{dadb_an:>12.5f} {rel:>10.2e}")
        dadb_rows.append({
            "beta": float(b0),
            "u_fold": float(u),
            "alpha_c_surrogate": float(a_c),
            "alpha_c_analytic": float(a_c_an),
            "dalpha_c_dbeta_surrogate": dadb_sur,
            "dalpha_c_dbeta_analytic": float(dadb_an),
            "rel_err": float(rel),
        })

    # --- (1b) V_PI geometry sensitivities, at two devices ------------------
    return {
        "dalpha_c_dbeta": dadb_rows,
        "V_PI_sensitivities": _vpi_block(net, DEV_A_SI, DEV_A, "A"),
        "V_PI_sensitivities_deviceB": _vpi_block(net, DEV_B_SI, DEV_B, "B"),
    }


def _vpi_block(net, dev_si, dev, label):
    """Autograd elasticities at one operating point, against the analytic
    pipeline.

    Device A sits at beta = 0.026 and device B at beta = 0.052, which is 63%
    of the Casimir ceiling, so the pair brackets the loading of the tabulated
    devices and exercises the surrogate where the fold is steepest.
    """
    print(f"\n  (1b) pull-in-voltage geometry sensitivities (device {label})")
    d = torch.tensor(dev_si["d"], dtype=DTYPE, requires_grad=True)
    A = torch.tensor(dev_si["A"], dtype=DTYPE, requires_grad=True)
    k = torch.tensor(dev_si["k"], dtype=DTYPE, requires_grad=True)
    V = V_PI_surrogate(net, d, A, k)
    gd, gA, gk = torch.autograd.grad(V, [d, A, k])
    sur = {"V_PI": float(V.detach()), "dV_dd": float(gd),
           "dV_dA": float(gA), "dV_dk": float(gk)}

    # finite-difference ground truth through the analytic pipeline
    def fd(fn, x, rel=1e-6):
        h = rel * abs(x)
        return (fn(x + h) - fn(x - h)) / (2.0 * h)

    d0, A0, k0 = dev_si["d"], dev_si["A"], dev_si["k"]
    an = {
        "V_PI": V_PI_analytic(d0, A0, k0),
        "dV_dd": fd(lambda x: V_PI_analytic(x, A0, k0), d0),
        "dV_dA": fd(lambda x: V_PI_analytic(d0, x, k0), A0),
        "dV_dk": fd(lambda x: V_PI_analytic(d0, A0, x), k0),
    }

    # report as dimensionless elasticities so d, A, k are comparable
    x0 = {"dV_dd": d0, "dV_dA": A0, "dV_dk": k0}
    print(f"       V_PI(surrogate) = {sur['V_PI']:.6f} V   "
          f"V_PI(analytic) = {an['V_PI']:.6f} V")
    print(f"       {'sensitivity':>10} {'surrogate':>14} {'analytic(FD)':>14} "
          f"{'|rel err|':>10}   {'elasticity':>10}")
    vpi_rows = {}
    for key, unit in (("dV_dd", "V/m"), ("dV_dA", "V/m^2"), ("dV_dk", "V/(N/m)")):
        s, a = sur[key], an[key]
        rel = abs(s - a) / abs(a)
        elas = s * x0[key] / sur["V_PI"]
        elas_an = a * x0[key] / an["V_PI"]
        print(f"       {key:>10} {s:>14.5e} {a:>14.5e} {rel:>10.2e}   {elas:>10.4f}")
        vpi_rows[key] = {
            "surrogate": float(s), "analytic_fd": float(a),
            "rel_err": float(rel), "unit": unit,
            "elasticity_surrogate": float(elas),
            "elasticity_analytic": float(elas_an),
        }

    return {
        "V_PI_surrogate_V": sur["V_PI"],
        "V_PI_analytic_V": an["V_PI"],
        "operating_point": {
            "A_um2": dev["A_um2"], "d_nm": dev["d_nm"], "k_N_per_m": dev["k"],
            "beta": float(beta_of_geometry(dev_si["A"], dev_si["d"],
                                           dev_si["k"])),
        },
        "components": vpi_rows,
    }


def DEV_A_beta():
    """Device A Casimir loading beta (~0.026)."""
    return float(beta_of_geometry(DEV_A_SI["A"], DEV_A_SI["d"], DEV_A_SI["k"]))


# ===========================================================================
# DEMO 2 -- gradient-based inverse design
# ===========================================================================
def demo_inverse_design(net, V_target=0.30, seed=42):
    print("\n" + "-" * 72)
    print("DEMO 2: gradient-based inverse design THROUGH the surrogate")
    print("-" * 72)
    print(f"  Spec: device A (A=100 um^2, k=0.5 N/m). Find the SMALLEST gap d")
    print(f"        (largest Casimir loading beta ~ 1/d^5) whose pull-in voltage")
    print(f"        equals V_PI* = {V_target:.3f} V.")

    config.set_all_seeds(seed)
    A0, k0 = DEV_A_SI["A"], DEV_A_SI["k"]

    # Optimise in log-gap so d stays positive; start at device A's 100 nm.
    d_init = DEV_A_SI["d"]
    logd = torch.tensor(np.log(d_init), dtype=DTYPE, requires_grad=True)
    A = torch.tensor(A0, dtype=DTYPE)
    k = torch.tensor(k0, dtype=DTYPE)
    V_tar = torch.tensor(float(V_target), dtype=DTYPE)

    opt = torch.optim.Adam([logd], lr=3e-2)
    history = {"iter": [], "d_nm": [], "V_PI": [], "loss": []}
    t0 = time.time()
    max_iter = 600
    tol = 1e-16
    # Track the best-so-far iterate: Adam momentum can overshoot past the
    # near-zero-loss point, so the returned design is the lowest-loss one seen.
    best = {"loss": float("inf"), "logd": float(logd.detach()),
            "V": float("nan"), "iter": 0, "t": 0.0}
    n_iter = 0
    for it in range(max_iter):
        opt.zero_grad()
        d = torch.exp(logd)
        V = V_PI_surrogate(net, d, A, k)
        loss = (V - V_tar) ** 2
        loss.backward()
        cur_loss = float(loss.detach())
        if cur_loss < best["loss"]:
            best = {"loss": cur_loss, "logd": float(logd.detach()),
                    "V": float(V.detach()), "iter": it, "t": time.time() - t0}
        if it % 10 == 0 or it == max_iter - 1:
            history["iter"].append(it)
            history["d_nm"].append(float(torch.exp(logd.detach())) * 1e9)
            history["V_PI"].append(float(V.detach()))
            history["loss"].append(cur_loss)
        opt.step()
        iters_run = it + 1
        if cur_loss < tol:
            break
    wall = time.time() - t0

    # Report the best-so-far design.
    n_iter = best["iter"] + 1
    if history["iter"][-1] != best["iter"]:
        history["iter"].append(best["iter"])
        history["d_nm"].append(np.exp(best["logd"]) * 1e9)
        history["V_PI"].append(best["V"])
        history["loss"].append(best["loss"])
    d_rec = float(np.exp(best["logd"]))
    beta_rec = float(beta_of_geometry(A0, d_rec, k0))
    V_rec = float(V_PI_surrogate(net, torch.tensor(d_rec, dtype=DTYPE),
                                 A, k))

    # --- analytic ground truth: Brent root-find on analytic V_PI(d)=V* -----
    def gV(dd):
        return V_PI_analytic(dd, A0, k0) - V_target
    # bracket: V_PI increases monotonically with d.  Below the stiction gap
    # d_ceil (beta = beta*) V_PI is undefined (NaN); bracket just above it.
    d_ceil = (np.pi ** 2 * HBAR * C_LIGHT * A0 / (240.0 * k0 * BETA_STAR)) ** 0.2
    d_lo, d_hi = d_ceil * 1.0001, 400e-9
    d_analytic = brentq(gV, d_lo, d_hi, xtol=1e-16, rtol=1e-14)
    beta_analytic = (np.pi ** 2 * HBAR * C_LIGHT * A0) / (240.0 * k0 * d_analytic ** 5)

    err_d = abs(d_rec - d_analytic)
    err_beta = abs(beta_rec - beta_analytic)
    t_best = best["t"]
    print(f"\n  converged in {n_iter} Adam steps ({t_best*1e3:.1f} ms to best; "
          f"loss<{tol:g} reached at {iters_run} steps, {wall*1e3:.1f} ms total)")
    print(f"       {'quantity':>14} {'recovered':>16} {'analytic':>16} {'|abs err|':>12}")
    print(f"       {'gap d [nm]':>14} {d_rec*1e9:>16.6f} {d_analytic*1e9:>16.6f} "
          f"{err_d*1e9:>12.3e}")
    print(f"       {'beta':>14} {beta_rec:>16.6f} {beta_analytic:>16.6f} "
          f"{err_beta:>12.3e}")
    print(f"       {'V_PI [V]':>14} {V_rec:>16.6f} {V_target:>16.6f} "
          f"{abs(V_rec-V_target):>12.3e}")
    print(f"  => at fixed area/stiffness, device A tolerates a Casimir loading")
    print(f"     up to beta ~ {beta_rec:.4f} (gap >= {d_rec*1e9:.2f} nm) at "
          f"V_PI = {V_target:.2f} V.")

    return {
        "spec": {"description": "smallest gap d (max Casimir loading) at target V_PI",
                 "A_um2": DEV_A["A_um2"], "k_N_per_m": DEV_A["k"],
                 "V_PI_target_V": float(V_target), "d_init_nm": d_init * 1e9},
        "recovered": {"d_nm": d_rec * 1e9, "beta": beta_rec, "V_PI_V": V_rec},
        "analytic": {"d_nm": d_analytic * 1e9, "beta": beta_analytic},
        "error": {"abs_err_d_nm": err_d * 1e9, "abs_err_beta": err_beta,
                  "abs_err_V_PI_V": abs(V_rec - V_target)},
        "convergence": {"adam_iters_to_best": n_iter,
                        "wall_time_to_best_ms": t_best * 1e3,
                        "iters_to_tol": iters_run,
                        "wall_time_total_ms": wall * 1e3,
                        "max_iter": max_iter,
                        "final_loss": best["loss"], "tol": tol},
        "history": history,
    }


# ===========================================================================
# Optional figure
# ===========================================================================
def make_figure(sens, inv):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if __package__ in (None, ""):
            from src import figstyle
        else:
            from . import figstyle
    except Exception as exc:  # pragma: no cover
        print(f"  [figure skipped: {exc}]")
        return None

    figstyle.apply_style()
    P = figstyle.PALETTE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.COL_DOUBLE, 2.7))

    # -- (a) inverse-design convergence -----------------------------------
    it = np.asarray(inv["history"]["iter"])
    loss = np.asarray(inv["history"]["loss"])
    ax1.semilogy(it, np.sqrt(loss) * 1e3, "o-", color=P["blue"], ms=3.5)
    ax1.set_xlabel("Adam iteration")
    ax1.set_ylabel(r"$|V_{\mathrm{PI}}-V_{\mathrm{PI}}^{*}|$  [mV]")
    ax1.set_title("(a) inverse design: gradient descent")
    ax1.grid(True, which="both")
    ax1.text(0.95, 0.92,
             rf"$d^\ast={inv['recovered']['d_nm']:.2f}$ nm"
             + "\n"
             + rf"err $={inv['error']['abs_err_d_nm']:.1e}$ nm",
             transform=ax1.transAxes, ha="right", va="top")

    # -- (b) sensitivity validation: elasticities surrogate vs analytic ----
    comp = sens["V_PI_sensitivities"]["components"]
    labels = [r"$d$", r"$A$", r"$k$"]
    keys = ["dV_dd", "dV_dA", "dV_dk"]
    e_sur = [comp[k]["elasticity_surrogate"] for k in keys]
    e_an = [comp[k]["elasticity_analytic"] for k in keys]
    x = np.arange(len(labels))
    w = 0.36
    ax2.bar(x - w / 2, e_sur, w, label="surrogate autograd", color=P["blue"])
    ax2.bar(x + w / 2, e_an, w, label="analytic (FD)", color=P["orange"])
    ax2.axhline(0.0, color="k", lw=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel(r"elasticity  $\partial\ln V_{\mathrm{PI}}/\partial\ln x$")
    ax2.set_title("(b) $V_{\\mathrm{PI}}$ sensitivities (device A)")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    paths = figstyle.savefig(fig, "fig5_inverse_design")
    plt.close(fig)
    print(f"  saved figure: {paths}")
    return paths


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 72)
    print("Inverse design with the differentiable physics-informed fold surrogate")
    print("CPU float64, deterministic (seed 42), no training.")
    print("=" * 72)

    config.set_all_seeds(42)
    net = load_surrogate()
    print(f"  loaded surrogate: models/pinn_param_boundary.pth")
    print(f"  device A Casimir loading beta = {DEV_A_beta():.6f}\n")

    t0 = time.time()
    sens = demo_sensitivities(net)
    inv = demo_inverse_design(net, V_target=0.30)
    fig_paths = make_figure(sens, inv)
    wall = time.time() - t0

    # --- honest validation verdict ----------------------------------------
    max_rel_dadb = max(r["rel_err"] for r in sens["dalpha_c_dbeta"])
    max_rel_vpi = max(c["rel_err"] for c in
                      sens["V_PI_sensitivities"]["components"].values())
    inv_ok = inv["error"]["abs_err_d_nm"] < 1e-2  # < 0.01 nm
    sens_ok = max_rel_dadb < 5e-2 and max_rel_vpi < 5e-2

    payload = {
        "meta": {
            "description": "Inverse design / autodiff sensitivities through the "
                           "differentiable PINN fold surrogate.",
            "surrogate": "models/pinn_param_boundary.pth",
            "dtype": "float64", "seed": 42, "torch": torch.__version__,
            "wall_time_s": wall,
            "analytic_fold": "alpha_c(u)=u^2(4-5u)/2, beta_c(u)=u^4(3u-2)/2, u=1-xi_PI",
            "dalpha_c_dbeta_closed_form": "-1/u^2",
        },
        "demo1_sensitivities": sens,
        "demo2_inverse_design": inv,
        "validation": {
            "max_rel_err_dalpha_c_dbeta": max_rel_dadb,
            "max_rel_err_V_PI_sensitivity": max_rel_vpi,
            "inverse_abs_err_d_nm": inv["error"]["abs_err_d_nm"],
            "inverse_abs_err_beta": inv["error"]["abs_err_beta"],
            "sensitivities_validate": bool(sens_ok),
            "inverse_validates": bool(inv_ok),
            "all_validate": bool(sens_ok and inv_ok),
        },
    }
    if fig_paths:
        payload["meta"]["figure"] = [os.path.basename(p) for p in fig_paths]

    out = RESULTS_DIR / "inverse_design.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  dalpha_c/dbeta   : max rel err vs -1/u^2      = {max_rel_dadb:.2e}")
    print(f"  V_PI geom. sens. : max rel err vs analytic FD = {max_rel_vpi:.2e}")
    print(f"  inverse design   : |d_rec - d_analytic|       = "
          f"{inv['error']['abs_err_d_nm']:.2e} nm  "
          f"({inv['convergence']['adam_iters_to_best']} steps, "
          f"{inv['convergence']['wall_time_to_best_ms']:.1f} ms to best)")
    print(f"  all validations pass: {payload['validation']['all_validate']}")
    print(f"\n  saved: {out}")
    print(f"  total wall-clock: {wall:.2f} s")
    print("=" * 72)
    return payload


if __name__ == "__main__":
    main()

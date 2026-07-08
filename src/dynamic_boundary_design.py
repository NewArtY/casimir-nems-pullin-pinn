"""No-closed-form design sensitivity of the DYNAMIC pull-in boundary.

Motivation
----------
The paper's differentiable fold surrogate is validated on the STATIC
saddle-node fold alpha_c(u)=u^2(4-5u)/2, beta_c(u)=u^4(3u-2)/2, whose design
sensitivity is the closed form dalpha_c/dbeta = -1/u^2.  To substantiate the
manuscript's claim that the surrogate methodology "extends to settings where no
closed-form fold exists", this script targets the paper's OWN featured physics
result: the DYNAMIC, from-rest (kinetic-overshoot) pull-in boundary at damping
zeta = 0.1.  A plate released from rest overshoots its quasi-static equilibrium,
so it collapses at a lower electrostatic bias than the static fold; that dynamic
threshold alpha_dyn(beta) has NO closed form -- it is defined only implicitly by
the RK4-integrated equation of motion.

Pipeline (all REAL, deterministic seed 42, CPU float64):

  1. Compute alpha_dyn(beta) at a grid of beta by BISECTION in alpha on the
     event "pulls in from rest", using the reference integrator
     ``src.rk4.rk4_integrate`` with the SAME dt / t_max / zeta as the saved
     phase diagram (results/phase_diagram.npz: dt=5e-3, t_max=200, zeta=0.1).
  2. Fit a small smooth differentiable torch MLP surrogate alpha_dyn_hat(beta).
  3. Read off the design sensitivity dalpha_dyn/dbeta by torch autograd.
  4. VALIDATE that sensitivity against an INDEPENDENT central finite difference
     of the RK4-recomputed boundary alpha_dyn(beta +/- h) (fresh bisections).
  5. Quantify the kinetic-overshoot margin delta(beta)=alpha_c^static(beta) -
     alpha_dyn(beta) -- itself a no-closed-form quantity.

Run:  python src/dynamic_boundary_design.py
"""

from __future__ import annotations

import json
import os
import time

# ---------------------------------------------------------------------------
# Robust imports: work as a package module and when run as a script.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_HERE))  # add .../PRA/code
    from src import config
    from src.rk4 import rk4_integrate
    from src.bifurcation import ALPHA_ENDPOINT, BETA_STAR
else:
    from . import config
    from .rk4 import rk4_integrate
    from .bifurcation import ALPHA_ENDPOINT, BETA_STAR

import numpy as np
import torch
from scipy.optimize import brentq

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

PROJECT_ROOT = config.PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Integration settings -- IDENTICAL to results/phase_diagram.npz (zeta=0.1).
# ---------------------------------------------------------------------------
ZETA = 0.1
DT = 5.0e-3
T_MAX = 200.0
EPS_BARRIER = 1.0e-6

# beta grid: span [0, 0.072], safely below BETA_STAR=0.08192 and below the
# beta (~0.076) at which the dynamic boundary reaches alpha=0, so alpha_dyn>0
# is well defined and the bisection bracket [0, ALPHA_HI] stays valid.
BETA_LO = 0.0
BETA_HI = 0.072
N_BETA = 33

ALPHA_HI = 0.20            # bisection upper bracket (deep pull-in for all beta)
BIS_TOL = 1.0e-6           # alpha bisection tolerance (< requested 1e-5)

# Validation operating points and finite-difference step.
VALID_BETAS = (0.01, 0.03, 0.05)
FD_H = 2.5e-3


# ---------------------------------------------------------------------------
# Reference dynamic boundary by RK4 bisection
# ---------------------------------------------------------------------------
def pulls_in(alpha: float, beta: float) -> bool:
    """True if the plate, released from rest, pulls in within t_max (RK4)."""
    _, _, _, info = rk4_integrate(
        alpha=float(alpha), beta=float(beta), zeta=ZETA,
        xi0=0.0, v0=0.0, t_max=T_MAX, dt=DT, eps_barrier=EPS_BARRIER,
    )
    return bool(info["pull_in"])


def alpha_dyn_bisect(beta: float, lo: float = 0.0, hi: float = ALPHA_HI,
                     tol: float = BIS_TOL) -> float:
    """Dynamic pull-in threshold alpha_dyn(beta) by bisection in alpha.

    Pull-in from rest is monotone in the electrostatic drive alpha (more drive
    -> collapse), so the event flips exactly once: stable below alpha_dyn,
    pull-in above.  Requires pulls_in(lo)=False and pulls_in(hi)=True.
    """
    if pulls_in(lo, beta):
        return float("nan")   # boundary at/below lo (no positive threshold)
    if not pulls_in(hi, beta):
        return float("nan")   # no pull-in even at hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if pulls_in(mid, beta):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Static fold (closed form) for the overshoot margin
# ---------------------------------------------------------------------------
def alpha_c_static(beta: float) -> float:
    """Closed-form static fold bias alpha_c(beta): invert beta_c(u), then alpha_c."""
    if beta <= 0.0:
        return float(ALPHA_ENDPOINT)
    if beta >= BETA_STAR:
        return 0.0
    u = brentq(lambda uu: uu ** 4 * (3.0 * uu - 2.0) / 2.0 - beta,
               2.0 / 3.0, 4.0 / 5.0, xtol=1e-14, rtol=1e-14)
    return float(u ** 2 * (4.0 - 5.0 * u) / 2.0)


# ---------------------------------------------------------------------------
# Differentiable surrogate alpha_dyn_hat(beta)
# ---------------------------------------------------------------------------
A_SCALE = 0.15   # output scale so the raw network head is order 1


class BoundaryMLP(torch.nn.Module):
    """Small smooth tanh MLP: beta -> alpha_dyn_hat.  Modest capacity keeps the
    fitted derivative stable."""

    def __init__(self, width: int = 32, depth: int = 2):
        super().__init__()
        layers = [torch.nn.Linear(1, width)]
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.Linear(width, 1))
        self.layers = torch.nn.ModuleList(layers)
        self.act = torch.tanh
        for lin in self.layers:
            torch.nn.init.xavier_normal_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        # normalise beta to [-1, 1] over the fit window
        x = (beta.reshape(-1, 1) / BETA_HI) * 2.0 - 1.0
        for lin in self.layers[:-1]:
            x = self.act(lin(x))
        return A_SCALE * self.layers[-1](x).reshape(-1)


def fit_surrogate(beta_np, alpha_np, seed=42, width=32, depth=2,
                  adam_iters=6000, adam_lr=5e-3, lbfgs_iters=400, verbose=True):
    config.set_all_seeds(seed)
    torch.set_default_dtype(DTYPE)

    beta_t = torch.as_tensor(beta_np, dtype=DTYPE)
    alpha_t = torch.as_tensor(alpha_np, dtype=DTYPE)

    net = BoundaryMLP(width=width, depth=depth)

    def loss_fn():
        pred = net(beta_t)
        return torch.mean((pred - alpha_t) ** 2)

    opt = torch.optim.Adam(net.parameters(), lr=adam_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_iters)
    for it in range(adam_iters):
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        opt.step()
        sched.step()
        if verbose and (it % 1500 == 0 or it == adam_iters - 1):
            print(f"    [adam {it:5d}] mse={float(L.detach()):.3e}")

    if lbfgs_iters and lbfgs_iters > 0:
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_iters,
                               history_size=60, tolerance_grad=1e-16,
                               tolerance_change=1e-18,
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            L = loss_fn()
            L.backward()
            return L

        lb.step(closure)
        if verbose:
            print(f"    [lbfgs]      mse={float(loss_fn()):.3e}")

    net.eval()
    with torch.no_grad():
        rmse = float(torch.sqrt(torch.mean((net(beta_t) - alpha_t) ** 2)))
    return net, rmse


def surrogate_sensitivity(net, beta_val):
    """(alpha_dyn_hat, dalpha_dyn_hat/dbeta) at beta_val via autograd."""
    b = torch.tensor(float(beta_val), dtype=DTYPE, requires_grad=True)
    a = net(b).reshape(())
    (grad,) = torch.autograd.grad(a, b)
    return float(a.detach()), float(grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("=" * 74)
    print("No-closed-form design sensitivity of the DYNAMIC pull-in boundary")
    print(f"zeta={ZETA}, dt={DT}, t_max={T_MAX} (matches results/phase_diagram.npz)")
    print("=" * 74)
    config.set_all_seeds(42)
    t_wall0 = time.perf_counter()

    # -- (1) RK4 dynamic boundary alpha_dyn(beta) by bisection ---------------
    beta_grid = np.linspace(BETA_LO, BETA_HI, N_BETA)
    print(f"[1] RK4 bisection of alpha_dyn(beta) at {N_BETA} beta in "
          f"[{BETA_LO}, {BETA_HI}] (tol={BIS_TOL:g}) ...")
    t0 = time.perf_counter()
    alpha_dyn = np.array([alpha_dyn_bisect(b) for b in beta_grid])
    print(f"    done in {time.perf_counter()-t0:.1f} s")
    if np.any(~np.isfinite(alpha_dyn)):
        bad = beta_grid[~np.isfinite(alpha_dyn)]
        print(f"    WARNING: no positive boundary at beta={bad}")

    alpha_static = np.array([alpha_c_static(b) for b in beta_grid])
    margin = alpha_static - alpha_dyn   # kinetic-overshoot margin delta(beta)

    a_dyn_b0 = float(alpha_dyn[0])
    print(f"    beta=0 : alpha_dyn={a_dyn_b0:.6f}  (static 4/27={ALPHA_ENDPOINT:.6f}, "
          f"margin={ALPHA_ENDPOINT-a_dyn_b0:.6f})")
    print(f"    mean overshoot margin delta = {np.mean(margin):.6f} "
          f"(min {np.min(margin):.6f}, max {np.max(margin):.6f})")

    # -- (2) fit differentiable surrogate ------------------------------------
    print("[2] fitting differentiable surrogate alpha_dyn_hat(beta) ...")
    net, rmse = fit_surrogate(beta_grid, alpha_dyn)
    print(f"    surrogate fit RMSE vs RK4 boundary = {rmse:.3e} (alpha units)")

    # -- (3+4) autograd sensitivity vs independent RK4 finite difference -----
    print("[3+4] sensitivity dalpha_dyn/dbeta: surrogate autograd vs RK4-FD ...")
    print(f"      {'beta':>7} {'alpha_dyn':>11} {'surrogate':>12} "
          f"{'RK4-FD':>12} {'|rel err|':>11}")
    sens_rows = []
    for b0 in VALID_BETAS:
        a_hat, dadb_sur = surrogate_sensitivity(net, b0)
        # INDEPENDENT central FD of the freshly RK4-bisected boundary
        a_plus = alpha_dyn_bisect(b0 + FD_H)
        a_minus = alpha_dyn_bisect(b0 - FD_H)
        dadb_fd = (a_plus - a_minus) / (2.0 * FD_H)
        rel = abs(dadb_sur - dadb_fd) / abs(dadb_fd)
        print(f"      {b0:>7.3f} {a_hat:>11.6f} {dadb_sur:>12.5f} "
              f"{dadb_fd:>12.5f} {rel:>11.2e}")
        sens_rows.append({
            "beta": float(b0),
            "alpha_dyn_surrogate": float(a_hat),
            "dalpha_dyn_dbeta_surrogate": float(dadb_sur),
            "dalpha_dyn_dbeta_rk4_fd": float(dadb_fd),
            "fd_h": FD_H,
            "fd_alpha_plus": float(a_plus),
            "fd_alpha_minus": float(a_minus),
            "rel_err": float(rel),
        })

    max_rel = max(r["rel_err"] for r in sens_rows)
    fit_ok = rmse < 1.0e-3
    sens_ok = max_rel < 0.10
    robust = fit_ok and sens_ok
    print("-" * 74)
    print(f"    surrogate fit RMSE < 1e-3     : {fit_ok}  ({rmse:.3e})")
    print(f"    max sensitivity rel err < 10% : {sens_ok}  ({max_rel*100:.2f}%)")
    print(f"    VERDICT: {'ROBUST' if robust else 'NOT ROBUST'}")

    # -- (5) figure (optional) -----------------------------------------------
    fig_paths = _make_figure(beta_grid, alpha_dyn, alpha_static, net)

    # -- (6) save JSON -------------------------------------------------------
    wall = time.perf_counter() - t_wall0
    payload = {
        "meta": {
            "description": "No-closed-form design sensitivity dalpha_dyn/dbeta of "
                           "the from-rest dynamic (kinetic-overshoot) pull-in "
                           "boundary, via a differentiable surrogate, validated "
                           "against RK4 finite differences.",
            "seed": 42, "dtype": "float64", "torch": torch.__version__,
            "zeta": ZETA, "dt": DT, "t_max": T_MAX, "eps_barrier": EPS_BARRIER,
            "alpha_bisection_tol": BIS_TOL, "fd_h": FD_H,
            "wall_time_s": wall,
            "static_fold_closed_form": "alpha_c(u)=u^2(4-5u)/2, beta_c(u)=u^4(3u-2)/2",
            "dalpha_c_dbeta_static_closed_form": "-1/u^2",
            "dynamic_boundary_has_closed_form": False,
        },
        "boundary": {
            "beta_grid": beta_grid.tolist(),
            "alpha_dyn": alpha_dyn.tolist(),
            "alpha_static": alpha_static.tolist(),
            "overshoot_margin": margin.tolist(),
            "alpha_dyn_beta0": a_dyn_b0,
            "alpha_static_beta0": float(ALPHA_ENDPOINT),
            "margin_beta0": float(ALPHA_ENDPOINT - a_dyn_b0),
            "mean_overshoot_margin": float(np.mean(margin)),
            "median_overshoot_margin": float(np.median(margin)),
            "min_overshoot_margin": float(np.min(margin)),
            "max_overshoot_margin": float(np.max(margin)),
        },
        "surrogate": {
            "architecture": "tanh MLP, 1->32->32->1",
            "width": 32, "depth": 2, "output_scale": A_SCALE,
            "fit_rmse_alpha": float(rmse),
        },
        "sensitivities": sens_rows,
        "validation": {
            "fit_rmse_alpha": float(rmse),
            "fit_rmse_below_1e-3": bool(fit_ok),
            "max_sensitivity_rel_err": float(max_rel),
            "sensitivity_rel_err_below_10pct": bool(sens_ok),
            "ROBUST": bool(robust),
        },
    }
    if fig_paths:
        payload["meta"]["figure"] = [os.path.basename(p) for p in fig_paths]

    out = RESULTS_DIR / "dynamic_boundary_design.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("-" * 74)
    print(f"saved: {out}")
    if fig_paths:
        for p in fig_paths:
            print(f"saved: {p}")
    print(f"total wall-clock: {wall:.1f} s")
    print("=" * 74)
    return payload


def _make_figure(beta_grid, alpha_dyn, alpha_static, net):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if __package__ in (None, ""):
            from src import figstyle
        else:
            from . import figstyle
    except Exception as exc:  # pragma: no cover
        print(f"    [figure skipped: {exc}]")
        return None

    figstyle.apply_style()
    P = figstyle.PALETTE
    bb = np.linspace(beta_grid.min(), beta_grid.max(), 400)
    with torch.no_grad():
        a_fit = net(torch.as_tensor(bb, dtype=DTYPE)).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(figstyle.COL_SINGLE, 2.7))
    ax.plot(beta_grid, alpha_static, "-", color=P["black"], lw=1.4,
            label=r"static fold $\alpha_c(\beta)$ (closed form)")
    ax.plot(bb, a_fit, "-", color=P["blue"], lw=1.4,
            label=r"surrogate $\hat\alpha_{\mathrm{dyn}}(\beta)$")
    ax.plot(beta_grid, alpha_dyn, "o", color=P["vermillion"], ms=3.2,
            label=r"RK4 dynamic $\alpha_{\mathrm{dyn}}(\beta)$")
    ax.fill_between(beta_grid, alpha_dyn, alpha_static, color=P["orange"],
                    alpha=0.18, lw=0)
    ax.set_xlabel(r"Casimir loading $\beta$")
    ax.set_ylabel(r"electrostatic bias $\alpha$")
    ax.set_title(r"Dynamic vs static pull-in boundary ($\zeta=0.1$)")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    paths = figstyle.savefig(fig, "fig6_dynamic_boundary")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    run()

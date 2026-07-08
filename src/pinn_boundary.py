"""Parametric physics-informed surrogate for the pull-in (fold) boundary.

Option 1 (recommended): a physics-informed equilibrium surrogate with the
fold-tangency condition.  This is an HONEST demonstration that the PINN
methodology can *locate the pull-in stability boundary* in the (alpha, beta)
control plane and reproduce the closed-form fold of Fig. 2.

Physics
-------
The static equilibria of the dimensionless NEMS oscillator

    xi'' + 2 zeta xi' + xi = alpha/(1-xi)^2 + beta/(1-xi)^4

are the roots of the net force

    g(xi; a, b) = a/(1-xi)^2 + b/(1-xi)^4 - xi .

Because g(0) = a + b > 0 and g -> +inf as xi -> 1, g has a *single interior
minimum*: writing u = 1 - xi, the stationarity condition g'(xi) = 0 is

    g'(xi) = 2a/u^3 + 4b/u^5 - 1 = 0    <=>    R(u; a, b) := 2 a u^2 + 4 b - u^5 = 0

(the second form is g' multiplied by u^5, a de-stiffened *polynomial* residual
with the same, unique root u* in (0,1) over the whole parameter rectangle).
The value of g at that minimiser controls how many equilibria exist:

    m(a, b) = min_xi g = g(1 - u*; a, b) ,
    Mtil(a, b) := u*^4 * m = a u*^2 + b - (1 - u*) u*^4 .

    Mtil < 0  ->  the minimum dips below zero  ->  TWO equilibria  -> STABLE,
    Mtil > 0  ->  the minimum stays above zero ->  NO equilibrium  -> PULL-IN,
    Mtil = 0  ->  the stable node and saddle collide -> the FOLD (pull-in).

The zero level set {Mtil = 0} is exactly the closed-form fold
alpha_c(u) = u^2 (4 - 5u)/2, beta_c(u) = u^4 (3u - 2)/2 (verified analytically:
both R and Mtil vanish identically along it).

PINN methodology
----------------
A small tanh MLP  u*_theta(alpha, beta)  is trained *purely* from the physics
residual  mean[ R(u*_theta; a, b)^2 ]  over collocation points in the
(alpha, beta) rectangle -- NO supervised root targets are used.  The network
must discover the tangency locus of the force field on its own.  The predicted
pull-in boundary is then the emergent zero level set of Mtil evaluated with the
*learned* u*_theta, and we compare it -- honestly -- with the analytic fold and
with the RK4 dynamic phase diagram.

Everything is CPU-only, float64, deterministic, and runs in a few seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq

from .config import set_all_seeds, PROJECT_ROOT
from . import bifurcation as bif

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
for _d in (MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Control-plane rectangle (matches results/phase_diagram.npz axes).
ALPHA_MAX = 0.20
BETA_MAX = 0.09

# Physical fold endpoints (exact rationals).
ALPHA_EP = 4.0 / 27.0        # alpha_c at beta = 0  (u = 2/3)
BETA_STAR = 256.0 / 3125.0   # beta_c  at alpha = 0 (u = 4/5)


# ---------------------------------------------------------------------------
# Parametric network  (alpha, beta) -> u* = 1 - xi*  in (0, 1)
# ---------------------------------------------------------------------------
class ParamMLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(2, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.ModuleList(layers)
        self.act = torch.tanh
        for lin in self.layers:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, ab: torch.Tensor) -> torch.Tensor:
        # ab[..., 0]=alpha, ab[..., 1]=beta ; normalise to [-1, 1]
        a = ab[..., 0:1] / ALPHA_MAX * 2.0 - 1.0
        b = ab[..., 1:2] / BETA_MAX * 2.0 - 1.0
        x = torch.cat([a, b], dim=-1)
        for lin in self.layers[:-1]:
            x = self.act(lin(x))
        # sigmoid -> u* strictly in (0, 1); clamp interior for numerical safety
        u = torch.sigmoid(self.layers[-1](x))
        return 1e-6 + (1.0 - 2e-6) * u


# ---------------------------------------------------------------------------
# Physics residual (de-stiffened, polynomial) and boundary indicator
# ---------------------------------------------------------------------------
def residual_R(u, a, b):
    """R = u^5 * g'(1-u) = 2 a u^2 + 4 b - u^5 ; zero at the force minimiser."""
    return 2.0 * a * u ** 2 + 4.0 * b - u ** 5


def indicator_Mtil(u, a, b):
    """Mtil = u^4 * min_xi g ; >0 pull-in, <0 stable, =0 fold."""
    return a * u ** 2 + b - (1.0 - u) * u ** 4


# ---------------------------------------------------------------------------
# Training  (pure physics residual, no supervised targets)
# ---------------------------------------------------------------------------
def _collocation(n_side: int) -> torch.Tensor:
    ga = torch.linspace(0.0, ALPHA_MAX, n_side, dtype=DTYPE)
    gb = torch.linspace(0.0, BETA_MAX, n_side, dtype=DTYPE)
    A, B = torch.meshgrid(ga, gb, indexing="ij")
    return torch.stack([A.reshape(-1), B.reshape(-1)], dim=-1)


def train(width=64, depth=3, n_side=90,
          adam_iters=4000, adam_lr=3e-3, lbfgs_iters=300,
          seed=42, verbose=True):
    set_all_seeds(seed)
    torch.set_default_dtype(DTYPE)

    net = ParamMLP(width=width, depth=depth)
    ab = _collocation(n_side)                # (N, 2) fixed collocation set
    a = ab[:, 0:1]
    b = ab[:, 1:2]

    history = {"step": [], "phase": [], "L_res": [], "elapsed": []}
    t0 = time.time()

    def loss_fn():
        u = net(ab)
        R = residual_R(u, a, b)
        return torch.mean(R ** 2)

    opt = torch.optim.Adam(net.parameters(), lr=adam_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_iters)
    for it in range(adam_iters):
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        opt.step()
        sched.step()
        if verbose and (it % 500 == 0 or it == adam_iters - 1):
            history["step"].append(it)
            history["phase"].append("adam")
            history["L_res"].append(float(L.detach()))
            history["elapsed"].append(time.time() - t0)
            print(f"  [adam {it:5d}] L_res={float(L):.3e}")

    if lbfgs_iters and lbfgs_iters > 0:
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=lbfgs_iters,
                               history_size=50, tolerance_grad=1e-14,
                               tolerance_change=1e-16,
                               line_search_fn="strong_wolfe")
        st = {"n": 0, "L": None}

        def closure():
            lb.zero_grad()
            L = loss_fn()
            L.backward()
            st["n"] += 1
            st["L"] = float(L)
            return L

        lb.step(closure)
        history["step"].append(adam_iters + st["n"])
        history["phase"].append("lbfgs")
        history["L_res"].append(st["L"])
        history["elapsed"].append(time.time() - t0)
        if verbose:
            print(f"  [lbfgs {st['n']:5d}] L_res={st['L']:.3e}")

    history["wall_time"] = time.time() - t0
    net.eval()
    return net, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def ustar_field(net, alpha_np, beta_np):
    ab = torch.stack([torch.as_tensor(np.asarray(alpha_np, float).reshape(-1), dtype=DTYPE),
                      torch.as_tensor(np.asarray(beta_np, float).reshape(-1), dtype=DTYPE)],
                     dim=-1)
    return net(ab).reshape(-1).cpu().numpy()


def Mtil_field(net, alpha_np, beta_np):
    u = ustar_field(net, alpha_np, beta_np)
    a = np.asarray(alpha_np, float).reshape(-1)
    b = np.asarray(beta_np, float).reshape(-1)
    return a * u ** 2 + b - (1.0 - u) * u ** 4


def _ustar_true(a, b):
    """Ground-truth minimiser via Brent on the polynomial residual."""
    if a <= 0 and b <= 0:
        return 1e-6
    lo, hi = 1e-9, 1.0 - 1e-12
    if residual_R(lo, a, b) * residual_R(hi, a, b) > 0:
        return hi
    return brentq(residual_R, lo, hi, args=(a, b))


def predicted_alpha_at_beta(net, beta, n_alpha=800):
    """Root in alpha of Mtil_net(alpha, beta) = 0 on [0, ALPHA_MAX] (or nan)."""
    ag = np.linspace(0.0, ALPHA_MAX, n_alpha)
    M = Mtil_field(net, ag, np.full_like(ag, beta))
    sgn = np.sign(M)
    idx = np.where(np.diff(sgn) != 0)[0]
    if idx.size == 0:
        return np.nan
    i = idx[0]
    m0, m1 = M[i], M[i + 1]
    if m1 == m0:
        return ag[i]
    return ag[i] - m0 * (ag[i + 1] - ag[i]) / (m1 - m0)


# ---------------------------------------------------------------------------
# Main: train, quantify honestly, save
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Parametric physics-informed pull-in boundary (Option 1: equilibrium")
    print("surrogate with fold tangency).  Pure-residual PINN, CPU float64.")
    print("=" * 72)

    net, history = train()
    print(f"  training wall-clock: {history['wall_time']:.2f} s")

    # -- (0) how well did the pure-residual PINN learn the minimiser field? --
    rng = np.random.default_rng(0)
    a_s = rng.uniform(0, ALPHA_MAX, 4000)
    b_s = rng.uniform(0, BETA_MAX, 4000)
    u_net = ustar_field(net, a_s, b_s)
    u_true = np.array([_ustar_true(a, b) for a, b in zip(a_s, b_s)])
    u_rmse = float(np.sqrt(np.mean((u_net - u_true) ** 2)))
    u_max = float(np.max(np.abs(u_net - u_true)))
    print(f"  learned minimiser field u*:  RMSE={u_rmse:.3e}  max|err|={u_max:.3e}")

    # -- (1) boundary agreement:  |Delta alpha| along the analytic fold -------
    u_fold = np.linspace(bif.U_MIN, bif.U_MAX, 200)
    a_fold = u_fold ** 2 * (4.0 - 5.0 * u_fold) / 2.0
    b_fold = u_fold ** 4 * (3.0 * u_fold - 2.0) / 2.0
    a_pred = np.array([predicted_alpha_at_beta(net, b) for b in b_fold])
    good = np.isfinite(a_pred)
    dalpha = a_pred[good] - a_fold[good]
    mean_da = float(np.mean(np.abs(dalpha)))
    max_da = float(np.max(np.abs(dalpha)))
    # relative to the alpha span of the fold (4/27)
    rel_da = mean_da / ALPHA_EP
    print(f"  boundary |Delta alpha| vs analytic fold: "
          f"mean={mean_da:.3e}  max={max_da:.3e}  (mean/ (4/27) = {rel_da*100:.2f}%)")

    # -- (2) beta=0 electrostatic pull-in:  alpha_c = 4/27 --------------------
    a_pred_b0 = predicted_alpha_at_beta(net, 0.0)
    err_b0 = abs(a_pred_b0 - ALPHA_EP)
    print(f"  beta=0 slice: alpha_c(PINN)={a_pred_b0:.5f}  vs 4/27={ALPHA_EP:.5f}"
          f"  (|err|={err_b0:.3e})")
    # alpha=0 Casimir pull-in: beta_c = 256/3125
    #   (find beta root of Mtil_net(0, beta))
    bg = np.linspace(0.0, BETA_MAX, 800)
    Mb = Mtil_field(net, np.zeros_like(bg), bg)
    sgn = np.sign(Mb)
    ii = np.where(np.diff(sgn) != 0)[0]
    if ii.size:
        i0 = ii[0]
        b_pred_a0 = bg[i0] - Mb[i0] * (bg[i0 + 1] - bg[i0]) / (Mb[i0 + 1] - Mb[i0])
    else:
        b_pred_a0 = np.nan
    err_a0 = abs(b_pred_a0 - BETA_STAR)
    print(f"  alpha=0 slice: beta_c(PINN)={b_pred_a0:.5f} vs 256/3125={BETA_STAR:.5f}"
          f"  (|err|={err_a0:.3e})")

    # -- (3) classification accuracy on the phase-diagram grid ----------------
    pd = np.load(RESULTS_DIR / "phase_diagram.npz", allow_pickle=True)
    Ag, Bg = pd["alpha_grid"], pd["beta_grid"]          # (200, 200)
    static_mask = pd["static_mask"]                     # True = PULL-IN (no static eq.)
    pullin_mask = pd["pullin_mask"]                     # True = pull-in (RK4, zeta=0.1)
    Mgrid = Mtil_field(net, Ag.reshape(-1), Bg.reshape(-1)).reshape(Ag.shape)
    pred_pullin = Mgrid > 0.0
    truth_static_pullin = static_mask                   # analytic pull-in (True=pull-in)
    acc_static = float(np.mean(pred_pullin == truth_static_pullin))
    acc_rk4 = float(np.mean(pred_pullin == pullin_mask))
    # agreement of the two ground truths themselves (context for the RK4 number)
    acc_static_vs_rk4 = float(np.mean(truth_static_pullin == pullin_mask))
    print(f"  grid classification (PINN pull-in = Mtil>0):")
    print(f"     vs analytic static fold : {acc_static*100:.2f}%")
    print(f"     vs RK4 dynamic (zeta=0.1): {acc_rk4*100:.2f}%   "
          f"[static-vs-RK4 baseline {acc_static_vs_rk4*100:.2f}%]")

    # -- save artefacts -------------------------------------------------------
    model_path = MODELS_DIR / "pinn_param_boundary.pth"
    torch.save({"state_dict": net.state_dict(),
                "meta": {"width": 64, "depth": 3,
                         "alpha_max": ALPHA_MAX, "beta_max": BETA_MAX}},
               model_path)

    # predicted boundary curve (dense in beta) for Fig. 2 overlay
    beta_curve = np.linspace(0.0, BETA_STAR, 200)
    alpha_curve = np.array([predicted_alpha_at_beta(net, b) for b in beta_curve])
    a_fold_full, b_fold_full, xi_pi_full, _ = bif.fold_boundary(n=400, full=False)

    np.savez(
        RESULTS_DIR / "pinn_boundary.npz",
        # PINN-predicted pull-in boundary (parametric by beta)
        alpha_pred=alpha_curve, beta_pred=beta_curve,
        # analytic fold for direct overlay/comparison
        fold_alpha_c=a_fold_full, fold_beta_c=b_fold_full, fold_xi_pi=xi_pi_full,
        # per-fold-sample agreement
        fold_beta_samples=b_fold, fold_alpha_analytic=a_fold,
        fold_alpha_pinn=a_pred,
        # learned scalar field on the phase grid (for shading if desired)
        alpha_grid=Ag, beta_grid=Bg, Mtil_grid=Mgrid, pred_pullin=pred_pullin,
        # metrics
        mean_dalpha=mean_da, max_dalpha=max_da, rel_dalpha=rel_da,
        u_field_rmse=u_rmse, u_field_max=u_max,
        alpha_c_beta0_pinn=a_pred_b0, alpha_c_beta0_true=ALPHA_EP,
        beta_c_alpha0_pinn=b_pred_a0, beta_c_alpha0_true=BETA_STAR,
        acc_vs_static=acc_static, acc_vs_rk4=acc_rk4,
        acc_static_vs_rk4=acc_static_vs_rk4,
    )

    log = {
        "option": "1 (equilibrium surrogate with fold tangency, pure-residual PINN)",
        "wall_time_train_s": history["wall_time"],
        "network": {"width": 64, "depth": 3, "params_in": 2,
                    "collocation": 90 * 90, "adam_iters": 4000,
                    "lbfgs_iters": 300},
        "final_residual_loss": history["L_res"][-1],
        "metrics": {
            "u_field_rmse": u_rmse, "u_field_max_abs": u_max,
            "boundary_mean_abs_dalpha": mean_da,
            "boundary_max_abs_dalpha": max_da,
            "boundary_mean_dalpha_rel_pct": rel_da * 100.0,
            "alpha_c_beta0_pinn": a_pred_b0, "alpha_c_beta0_ref": ALPHA_EP,
            "alpha_c_beta0_abs_err": err_b0,
            "beta_c_alpha0_pinn": float(b_pred_a0), "beta_c_alpha0_ref": BETA_STAR,
            "beta_c_alpha0_abs_err": float(err_a0),
            "grid_acc_vs_static_pct": acc_static * 100.0,
            "grid_acc_vs_rk4_pct": acc_rk4 * 100.0,
            "grid_static_vs_rk4_baseline_pct": acc_static_vs_rk4 * 100.0,
        },
        "history": history,
    }
    with open(LOGS_DIR / "pinn_boundary.json", "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    print(f"\n  saved: {model_path}")
    print(f"  saved: {RESULTS_DIR / 'pinn_boundary.npz'}")
    print(f"  saved: {LOGS_DIR / 'pinn_boundary.json'}")
    print("=" * 72)
    return log


if __name__ == "__main__":
    main()

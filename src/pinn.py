"""Physics-informed neural network (PINN) library for the pull-in ODE.

Governing (dimensionless) equation of motion:

    xi'' + 2 zeta xi' + xi = alpha/(1-xi)**2 + beta/(1-xi)**4 ,   xi in [0, 1) ,

with initial conditions xi(0) = xi0, xi'(0) = v0.

This module provides a *single-trajectory* PINN: the network maps the scalar
time tau -> xi(tau).  Time derivatives xi', xi'' are obtained by
``torch.autograd.grad`` and the ODE residual is enforced at collocation
points.  Everything is CPU-only, float64 and fully deterministic (seed via
``config.set_all_seeds``).

Output parametrizations (selectable)
------------------------------------
* ``"raw"``      : the network output N(tau) is the displacement directly.
* ``"rapidity"`` : the network defines a "rapidity" theta(tau) >= 0 and
                   xi = 1 - exp(-theta).  For any finite theta this guarantees
                   xi < 1, so the movable pole xi -> 1 (pull-in) is mapped to
                   theta -> +infinity and never crossed.  This is the
                   pull-in-safe parametrization used for the collapse regime.

Each parametrization can be combined with a **hard** initial-condition
encoding (``hard_ic=True``) that makes xi(0)=xi0 and xi'(0)=v0 hold exactly by
construction, or a **soft** one (``hard_ic=False``) enforced through the loss.

Composite loss:  L = L_ODE + lambda_ic * L_IC + lambda_pull * L_pullin .
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config
from .config import set_all_seeds, PROJECT_ROOT
from .physics import force_rhs

# ---------------------------------------------------------------------------
# Paths / global dtype
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# float64 everywhere: the accuracy target (~1e-3) is easier to hit in double
# precision, and the nets are tiny so CPU cost is negligible.
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


# ---------------------------------------------------------------------------
# Fully-connected network
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Fully-connected tanh network, scalar in -> scalar out.

    Parameters
    ----------
    n_hidden : int
        Number of hidden layers (default 4).
    width : int
        Neurons per hidden layer (default 64).
    T : float
        Time horizon used to normalise the input to [-1, 1] when
        ``normalize`` is True: x = 2*tau/T - 1.
    normalize : bool
        Normalise the scalar time input before the first layer.
    """

    def __init__(self, n_hidden: int = 4, width: int = 64, T: float = 1.0,
                 normalize: bool = True, n_fourier: int = 0,
                 fourier_omega: float = 1.0):
        super().__init__()
        self.normalize = bool(normalize)
        self.n_fourier = int(n_fourier)
        self.register_buffer("T", torch.tensor(float(T), dtype=DTYPE))
        self.register_buffer("fourier_omega",
                             torch.tensor(float(fourier_omega), dtype=DTYPE))

        # Input features: normalised time + (optional) sin/cos harmonics of
        # the *physical* time, which let a tanh MLP resolve the oscillatory
        # transient of a long-horizon trajectory without wasting depth on it.
        in_dim = 1 + 2 * self.n_fourier
        layers = [nn.Linear(in_dim, width)]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.ModuleList(layers)
        self.activation = torch.tanh
        self._init_weights()

    def _init_weights(self) -> None:
        for lin in self.layers:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _features(self, tau: torch.Tensor) -> torch.Tensor:
        x = 2.0 * tau / self.T - 1.0 if self.normalize else tau
        if self.n_fourier <= 0:
            return x
        feats = [x]
        for k in range(1, self.n_fourier + 1):
            wk = k * self.fourier_omega * tau
            feats.append(torch.sin(wk))
            feats.append(torch.cos(wk))
        return torch.cat(feats, dim=-1)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        x = self._features(tau)
        for lin in self.layers[:-1]:
            x = self.activation(lin(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Trajectory model = network + output parametrization + IC encoding
# ---------------------------------------------------------------------------
class PINNTrajectory(nn.Module):
    """xi(tau) trajectory model with selectable parametrization / IC encoding.

    Parameters
    ----------
    alpha, beta, zeta : float
        ODE parameters.
    xi0, v0 : float
        Initial displacement and velocity.
    T : float
        Time horizon (collocation domain is [0, T]); also the input
        normalisation scale.
    parametrization : {"raw", "rapidity"}
        Output map (see module docstring).
    hard_ic : bool
        Encode xi(0)=xi0, xi'(0)=v0 exactly by construction.
    delta : float
        Contact clearance used by the pull-in barrier (penalise xi >= 1-delta).
    """

    def __init__(self, alpha, beta, zeta, xi0, v0, T,
                 parametrization="raw", hard_ic=True,
                 n_hidden=4, width=64, normalize=True, delta=1e-3,
                 n_fourier=0, fourier_omega=1.0):
        super().__init__()
        if parametrization not in ("raw", "rapidity"):
            raise ValueError(f"unknown parametrization {parametrization!r}")
        self.parametrization = parametrization
        self.hard_ic = bool(hard_ic)

        self.net = MLP(n_hidden=n_hidden, width=width, T=T, normalize=normalize,
                       n_fourier=n_fourier, fourier_omega=fourier_omega)

        # Store scalars as buffers so they travel with state_dict / device.
        for name, val in dict(alpha=alpha, beta=beta, zeta=zeta,
                              xi0=xi0, v0=v0, T=T, delta=delta).items():
            self.register_buffer(name, torch.tensor(float(val), dtype=DTYPE))

        # Rapidity IC anchors:  theta0 = -ln(1-xi0),  thetadot0 = v0/(1-xi0)
        one_minus0 = 1.0 - float(xi0)
        self.register_buffer("theta0",
                             torch.tensor(-np.log(one_minus0), dtype=DTYPE))
        self.register_buffer("thetadot0",
                             torch.tensor(float(v0) / one_minus0, dtype=DTYPE))

    # -- output map ---------------------------------------------------------
    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        N = self.net(tau)

        if self.parametrization == "raw":
            if self.hard_ic:
                g1 = 1.0 - torch.exp(-tau)          # g1(0)=0, g1'(0)=1
                g2 = g1 * g1                          # g2(0)=0, g2'(0)=0
                xi = self.xi0 + self.v0 * g1 + g2 * N
            else:
                xi = N
            return xi

        # rapidity:  xi = 1 - exp(-theta),  theta >= 0
        if self.hard_ic:
            g1 = 1.0 - torch.exp(-tau)
            g2 = g1 * g1
            theta = self.theta0 + self.thetadot0 * g1 + g2 * N
        else:
            theta = torch.nn.functional.softplus(N)  # theta >= 0 by construction
        xi = 1.0 - torch.exp(-theta)
        return xi


# ---------------------------------------------------------------------------
# Autograd derivatives and loss
# ---------------------------------------------------------------------------
def trajectory_derivatives(model: PINNTrajectory, tau: torch.Tensor):
    """Return (xi, xi', xi'') at the (leaf, requires_grad) points ``tau``."""
    xi = model(tau)
    ones = torch.ones_like(xi)
    xid = torch.autograd.grad(xi, tau, grad_outputs=ones, create_graph=True)[0]
    xidd = torch.autograd.grad(xid, tau, grad_outputs=torch.ones_like(xid),
                               create_graph=True)[0]
    return xi, xid, xidd


def compute_losses(model: PINNTrajectory, tau_col: torch.Tensor):
    """Composite-loss components (L_ODE, L_IC, L_pullin) as scalar tensors."""
    alpha = model.alpha; beta = model.beta; zeta = model.zeta
    xi0 = model.xi0; v0 = model.v0; delta = model.delta

    # --- ODE residual at collocation points -------------------------------
    tau = tau_col.detach().clone().requires_grad_(True)
    xi, xid, xidd = trajectory_derivatives(model, tau)
    # residual = xi'' - [ -2 zeta xi' - xi + alpha/(1-xi)^2 + beta/(1-xi)^4 ]
    residual = xidd - force_rhs(xi, xid, alpha, beta, zeta)
    L_ode = torch.mean(residual ** 2)

    # --- initial condition ------------------------------------------------
    tau0 = torch.zeros(1, 1, dtype=DTYPE, requires_grad=True)
    xi_0, xid_0, _ = trajectory_derivatives(model, tau0)
    L_ic = (xi_0 - xi0) ** 2 + (xid_0 - v0) ** 2
    L_ic = L_ic.squeeze()

    # --- pull-in / domain barrier -----------------------------------------
    over = torch.relu(xi - (1.0 - delta))
    under = torch.relu(-xi)
    L_pull = torch.mean(over ** 2) + torch.mean(under ** 2)

    return L_ode, L_ic, L_pull


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_collocation(T: float, n: int) -> torch.Tensor:
    """Deterministic collocation grid on [0, T] (endpoints included)."""
    tau = torch.linspace(0.0, float(T), int(n), dtype=DTYPE).reshape(-1, 1)
    return tau


def train_pinn(alpha, beta, zeta, xi0, v0, T,
               parametrization="raw", hard_ic=True,
               n_hidden=4, width=64, normalize=True, delta=1e-3,
               n_fourier=0, fourier_omega=1.0,
               n_collocation=2000,
               adam_iters=8000, adam_lr=1e-3,
               lbfgs_iters=400,
               lambda_ic=1.0, lambda_pull=1.0,
               log_every=250, seed=config.DEFAULT_SEED,
               verbose=True):
    """Train a single-trajectory PINN.  Returns (model, history dict).

    Adam is run first (``adam_iters`` steps) then an optional L-BFGS refine
    (``lbfgs_iters`` closure evaluations).  Loss components are logged every
    ``log_every`` Adam steps and once per L-BFGS phase boundary.
    """
    set_all_seeds(seed)
    torch.set_default_dtype(DTYPE)

    model = PINNTrajectory(alpha, beta, zeta, xi0, v0, T,
                           parametrization=parametrization, hard_ic=hard_ic,
                           n_hidden=n_hidden, width=width, normalize=normalize,
                           delta=delta, n_fourier=n_fourier,
                           fourier_omega=fourier_omega)
    model.train()

    tau_col = make_collocation(T, n_collocation)

    history = {
        "params": dict(alpha=alpha, beta=beta, zeta=zeta, xi0=xi0, v0=v0, T=T),
        "config": dict(parametrization=parametrization, hard_ic=hard_ic,
                       n_hidden=n_hidden, width=width, normalize=normalize,
                       delta=delta, n_fourier=n_fourier,
                       fourier_omega=fourier_omega, n_collocation=n_collocation,
                       adam_iters=adam_iters, adam_lr=adam_lr,
                       lbfgs_iters=lbfgs_iters, lambda_ic=lambda_ic,
                       lambda_pull=lambda_pull, seed=seed),
        "step": [], "phase": [],
        "L_ode": [], "L_ic": [], "L_pull": [], "L_total": [], "elapsed": [],
    }

    def _f(x):
        return float(x.detach()) if torch.is_tensor(x) else float(x)

    def _log(step, phase, L_ode, L_ic, L_pull, L_total, t0):
        history["step"].append(int(step))
        history["phase"].append(phase)
        history["L_ode"].append(_f(L_ode))
        history["L_ic"].append(_f(L_ic))
        history["L_pull"].append(_f(L_pull))
        history["L_total"].append(_f(L_total))
        history["elapsed"].append(float(time.time() - t0))

    t0 = time.time()

    # --- Adam -------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
    for it in range(adam_iters):
        opt.zero_grad()
        L_ode, L_ic, L_pull = compute_losses(model, tau_col)
        L = L_ode + lambda_ic * L_ic + lambda_pull * L_pull
        L.backward()
        opt.step()
        if verbose and (it % log_every == 0 or it == adam_iters - 1):
            _log(it, "adam", L_ode, L_ic, L_pull, L, t0)
            print(f"  [adam {it:6d}] L={float(L):.3e} "
                  f"(ode={float(L_ode):.3e} ic={float(L_ic):.3e} "
                  f"pull={float(L_pull):.3e})")

    # --- L-BFGS refine ----------------------------------------------------
    if lbfgs_iters and lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=lbfgs_iters,
            history_size=50, tolerance_grad=1e-12, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        state = {"n": 0, "last": None}

        def closure():
            lbfgs.zero_grad()
            L_ode, L_ic, L_pull = compute_losses(model, tau_col)
            L = L_ode + lambda_ic * L_ic + lambda_pull * L_pull
            L.backward()
            state["n"] += 1
            state["last"] = (L_ode, L_ic, L_pull, L)
            return L

        lbfgs.step(closure)
        L_ode, L_ic, L_pull, L = state["last"]
        _log(adam_iters + state["n"], "lbfgs", L_ode, L_ic, L_pull, L, t0)
        if verbose:
            print(f"  [lbfgs {state['n']:6d}] L={float(L):.3e} "
                  f"(ode={float(L_ode):.3e} ic={float(L_ic):.3e} "
                  f"pull={float(L_pull):.3e})")

    history["wall_time"] = float(time.time() - t0)
    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_xi(model: PINNTrajectory, tau_np: np.ndarray) -> np.ndarray:
    tau = torch.as_tensor(np.asarray(tau_np, dtype=float),
                          dtype=DTYPE).reshape(-1, 1)
    return model(tau).reshape(-1).cpu().numpy()


def evaluate_xi_xidot(model: PINNTrajectory, tau_np: np.ndarray):
    """Return (xi, xi') as numpy arrays on the given tau grid (autograd v)."""
    tau = torch.as_tensor(np.asarray(tau_np, dtype=float),
                          dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
    xi = model(tau)
    xid = torch.autograd.grad(xi, tau, grad_outputs=torch.ones_like(xi))[0]
    return (xi.detach().reshape(-1).cpu().numpy(),
            xid.detach().reshape(-1).cpu().numpy())


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_model(model: PINNTrajectory, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {
            "parametrization": model.parametrization,
            "hard_ic": model.hard_ic,
            "alpha": float(model.alpha), "beta": float(model.beta),
            "zeta": float(model.zeta), "xi0": float(model.xi0),
            "v0": float(model.v0), "T": float(model.T),
            "delta": float(model.delta),
            "n_fourier": int(model.net.n_fourier),
            "fourier_omega": float(model.net.fourier_omega),
        },
    }, path)
    return path


def load_model(name: str, n_hidden=4, width=64, normalize=True) -> PINNTrajectory:
    path = MODELS_DIR / f"{name}.pth"
    blob = torch.load(path, weights_only=False)
    m = blob["meta"]
    model = PINNTrajectory(
        m["alpha"], m["beta"], m["zeta"], m["xi0"], m["v0"], m["T"],
        parametrization=m["parametrization"], hard_ic=m["hard_ic"],
        n_hidden=n_hidden, width=width, normalize=normalize, delta=m["delta"],
        n_fourier=m.get("n_fourier", 0),
        fourier_omega=m.get("fourier_omega", 1.0),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def save_history(history: dict, name: str) -> Path:
    path = LOGS_DIR / f"history_{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    return path

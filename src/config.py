"""Global configuration for the Casimir-electrostatic NEMS pull-in study.

This module centralises:
  * deterministic RNG seeding (numpy + torch),
  * physical constants (SI units),
  * default dimensionless parameters,
  * output paths.

The dynamical model is the dimensionless pull-in equation

    xi'' = -2 zeta xi' - xi + alpha / (1 - xi)^2 + beta / (1 - xi)^4 ,

where xi = x / d in [0, 1) is the normalised plate displacement,
alpha is the electrostatic control parameter, beta the Casimir control
parameter and zeta the (dimensionless) damping ratio.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:  # torch is part of the reproducibility contract but keep import defensive
    import torch

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - torch is expected to be present
    torch = None  # type: ignore
    _HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42


def set_all_seeds(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG used in the project (numpy + torch) deterministically.

    Parameters
    ----------
    seed : int
        Master seed applied to numpy and torch (CPU).

    Returns
    -------
    int
        The seed that was applied (for logging / provenance).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if _HAVE_TORCH:
        torch.manual_seed(seed)
        # CPU-only run, but keep the call for portability / determinism.
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older/newer torch APIs
            pass
    return seed


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
EPS0 = 8.8541878128e-12       # vacuum permittivity           [F/m]
HBAR = 1.054571817e-34        # reduced Planck constant        [J s]
C_LIGHT = 299792458.0         # speed of light                 [m/s]
K_B = 1.380649e-23            # Boltzmann constant             [J/K]
ZETA3 = 1.2020569031595942854  # Apery's constant, Riemann zeta(3)


# ---------------------------------------------------------------------------
# Default dimensionless parameters and integration settings
# ---------------------------------------------------------------------------
DEFAULTS = {
    "alpha": 0.05,      # electrostatic control parameter
    "beta": 1.0e-4,     # Casimir control parameter
    "zeta": 0.5,        # damping ratio
    "xi0": 0.0,         # initial normalised displacement
    "v0": 0.0,          # initial normalised velocity
    "t_max": 60.0,      # integration horizon (dimensionless time)
    "dt": 1.0e-3,       # fixed RK4 step
    "T": 0.0,           # temperature [K] for the Lifshitz correction
    "d": 100e-9,        # gap at rest [m] for the Lifshitz correction
    "eps_barrier": 1e-6,  # pull-in barrier: event fires at xi >= 1 - eps
}


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../PRA/code
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_RESULTS_JSON = RESULTS_DIR / "smoke_results.json"

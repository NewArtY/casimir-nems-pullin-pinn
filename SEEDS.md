# RNG Seeds & Determinism Guarantees

Every artefact in this archive (`results/`, `models/`, `logs/`, `figures/`) is
produced **deterministically** from a single fixed master seed. This document is
the authoritative record of that contract.

## Master seed

```
DEFAULT_SEED = 42        # src/config.py
```

Every executable stage calls **`src.config.set_all_seeds(42)`** before doing any
work. That function (see `src/config.py`) applies the seed to *all* sources of
randomness in the project:

| RNG / knob | Set to | Purpose |
|---|---|---|
| `os.environ["PYTHONHASHSEED"]` | `"42"` | stable hashing across processes |
| `numpy.random.seed(42)` | 42 | numpy legacy global RNG |
| `torch.manual_seed(42)` | 42 | PyTorch CPU RNG (weight init, any sampling) |
| `torch.cuda.manual_seed_all(42)` | 42 | only if CUDA present (it is **not** here) |
| `torch.use_deterministic_algorithms(True, warn_only=True)` | — | forces deterministic kernels where available |

## Where randomness actually enters

The only stochastic element in the whole pipeline is **PINN weight
initialization** (Xavier/normal init in `src/pinn.py` and `src/pinn_boundary.py`),
which draws from the torch RNG. Seeding torch with 42 before each training run
makes those draws — and therefore the trained weights and every downstream
metric — reproducible.

Per-stage seeding:

| Stage (script) | Seeding call | Notes |
|---|---|---|
| `smoke_test.py` | `set_all_seeds(42)` | RK4 is deterministic; seed for provenance. |
| `bifurcation.py` | (none needed) | Pure numpy/scipy, no RNG. |
| `train_trajectories.py` | `set_all_seeds(42)` globally **and** again per regime (`run_regime(..., seed=42)`) | Each of the 3 regimes trains from the same seed state. |
| `verify.py` | (none needed) | Deterministic ODE solves; reads saved arrays. |
| `phase_diagram.py` | `set_all_seeds(42)` | Vectorised RK4 is deterministic; **no RNG is actually used** — seed is provenance only. |
| `lifshitz_boundary.py` | (none needed) | Linear-algebra fold solve, deterministic. |
| `pinn_boundary.py` | `set_all_seeds(42)` inside `train()` | Also uses a **local** `numpy.random.default_rng(0)` to draw 4000 evaluation points for the minimiser-field RMSE diagnostic — an explicit, independently seeded generator (seed 0), so that diagnostic is reproducible too. |
| `make_figures.py` | (none needed) | Pure renderer, no RNG. |

## Numerical-precision contract

* **CPU only.** No GPU/CUDA is used anywhere; `torch.cuda.is_available()` is
  `False` on the reference machine. This removes GPU non-determinism entirely.
* **float64 everywhere.** `torch.set_default_dtype(torch.float64)` is set in the
  PINN modules; numpy/scipy default to float64. Double precision is what makes
  the ~1e-3 accuracy targets and the fold residuals (~1e-16) reproducible.
* **Deterministic optimizers.** Adam and L-BFGS (with `strong_wolfe` line
  search) are deterministic given fixed initial weights and a fixed collocation
  set; collocation grids are fixed `linspace`/`meshgrid` tensors, not sampled.
* **Fixed grids and steps.** The RK4 phase-diagram sweep uses fixed `dt`,
  `t_max` and `linspace` axes; the reference solves use fixed tolerances
  (`rtol/atol = 1e-10` for training scoring, `1e-12` for the verification study).

## Reproducibility expectation

Re-running `python run_all.py` on the pinned software stack (see
`requirements.txt`: Python 3.13.2, torch 2.12.0+cpu, numpy 2.4.4, scipy 1.17.1,
matplotlib 3.10.9) regenerates every artefact to solver tolerance. Small
last-bit differences in trained-weight metrics can occur across *different* CPU
architectures or BLAS builds (that is intrinsic to floating-point reductions),
but the seed contract guarantees bitwise-stable RNG draws and reproducible
qualitative and quantitative results (e.g. PINN `L∞(ξ)` ≈ 1.05/1.95/2.83 × 10⁻³
for stable/growing/pull-in).

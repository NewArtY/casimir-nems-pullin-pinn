# Casimir–Electrostatic NEMS Pull-In: Bifurcation, Phase Diagrams & Physics-Informed Neural Networks

Reproducible computational archive accompanying the article

> **Casimir–Electrostatic Pull-In Bifurcation in Nonlinear Nanomechanical Oscillators: Phase Diagrams and Physics-Informed Neural-Network Solutions with Relativistic Lifshitz Corrections**
> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, and Qing-Hua Qin
> *Physical Review Applied* (in preparation, 2026).

## Authors

| Author | Affiliation | ORCID |
|---|---|---|
| **N. S. Akintsov** | School of Artificial Intelligence and Computer Science, Nantong University, Nantong 226019, China | [0000-0002-1040-1292](https://orcid.org/0000-0002-1040-1292) |
| **A. P. Nevecheria** | Department of Mathematical and Computer Methods, Kuban State University, Krasnodar 350040, Russia | [0000-0001-6736-4691](https://orcid.org/0000-0001-6736-4691) |
| **S. N. Andreev** | Joint-Stock Company "Center for Research and Development", Moscow 101000, Russia | [0000-0003-3588-2894](https://orcid.org/0000-0003-3588-2894) |
| **Qing-Hua Qin** | Institute of Advanced Interdisciplinary Technology, Shenzhen MSU-BIT University, Shenzhen 518172, China | [0000-0003-0948-784X](https://orcid.org/0000-0003-0948-784X) |

## Abstract

This code models the *pull-in instability* of a parallel-plate nano-electromechanical
(NEMS) oscillator subject to both an electrostatic drive and the attractive
Casimir force. It (i) derives the closed-form saddle-node (fold) bifurcation
that separates the stable from the pull-in regime in the (electrostatic,
Casimir) control plane, (ii) maps the dynamic pull-in phase diagram by
deterministic Runge–Kutta integration from rest and quantifies how kinetic
overshoot moves the dynamic boundary inside the static fold, (iii) trains
mesh-free **physics-informed neural networks (PINNs)** that solve the governing
ODE — including a pull-in-safe *rapidity* parametrization that resolves the
movable pole where fixed-step RK4 fails — and a parametric PINN that learns the
entire fold boundary directly from the force-balance residual, and (iv) computes
the finite-temperature **Lifshitz** correction and the resulting thermal shift
of the pull-in voltage. All results, models, logs and the four publication
figures are regenerated deterministically from a single fixed seed.

## Governing equation

The dimensionless equation of motion for the normalized plate displacement
`ξ = x/d ∈ [0, 1)` (with `1 − ξ` the instantaneous gap) is

```
ξ'' + 2 ζ ξ' + ξ = α / (1 − ξ)²  +  β / (1 − ξ)⁴ ,        ξ(0)=ξ0 , ξ'(0)=v0 ,
```

equivalently `ξ'' = −2 ζ ξ' − ξ + α/(1−ξ)² + β/(1−ξ)⁴`, where

* `α` — electrostatic control number (`α ∝ V²`, attractive `~1/gap²`),
* `β` — Casimir control number (attractive `~1/gap⁴`),
* `ζ` — damping ratio.

The conservative part derives from the effective potential
`U(ξ) = ξ²/2 − α/(1−ξ) − β/[3(1−ξ)³]`, so `net_force(ξ) = −U'(ξ)`.
The finite-temperature (Lifshitz) model multiplies the Casimir term by the
leading factor `[1 + κ(d) T (1−ξ)]` with `κ(d) = (720 ζ(3)/π³) k_B d /(ħ c)`.

## Module map (`src/`)

| File | Role |
|---|---|
| `config.py` | Global configuration: **deterministic seeding** (`set_all_seeds`, seed 42), SI physical constants, default dimensionless parameters, output paths. |
| `physics.py` | Core force/potential model: `force_rhs`, `net_force`, effective potential `U`, `verify_potential` (finite-difference identity `−U'=net_force`), `beta_lifshitz` thermal correction, `fold_alpha_beta`. |
| `rk4.py` | Fixed-step classical RK4 integrator with pull-in event detection and regime classification (stable / growing / pull-in); optional Lifshitz-corrected `β`. |
| `bifurcation.py` | Static fold (saddle-node) analysis: closed-form fold locus `α_c(u), β_c(u)`, `equilibria`, `is_stable_from_rest`, stability grid. Endpoints `α_c=4/27`, `β*=256/3125`. |
| `pinn.py` | Single-trajectory PINN library (τ → ξ): fully-connected net, autograd ODE residual, `raw` / `rapidity` parametrizations, hard/soft IC encoding, Adam+L-BFGS training, save/load helpers. CPU float64. |
| `train_trajectories.py` | Trains the PINN for the three Fig. 1 regimes, scores vs a DOP853 (`rtol=1e-10`) reference, writes `models/pinn_*.pth`, `logs/history_*.json`, `results/traj_*.npz`. |
| `verify.py` | Verification study: adaptive DOP853 (`rtol=atol=1e-12`) ground truth, fixed-step RK4 step-study near the movable pole, PINN accuracy table (regular vs near-pull-in). |
| `phase_diagram.py` | Vectorised RK4 sweep over a 200×200 (α, β) grid from rest; dynamic pull-in mask, `τ_PI` map, overlay of the analytic static fold and the static/dynamic boundary gap. |
| `lifshitz_boundary.py` | Finite-`T` fold boundary from the `g = g' = 0` linear system; `T=0` validation vs the closed form; thermal shift of `α_c`, `V_PI`, and `β*`. |
| `pinn_boundary.py` | Parametric physics-informed surrogate `u*_θ(α,β)` trained purely on the force-balance residual; emergent pull-in boundary (zero level set of the fold indicator) vs the analytic fold. Uses **relative imports** — invoke as `python -m src.pinn_boundary`. |
| `figstyle.py` | Publication figure style (Physical Review column widths, colour-blind-safe Okabe–Ito palette, `savefig` → PDF + 600-dpi PNG). |
| `make_figures.py` | Pure renderer: loads saved arrays and writes the four figures to `figures/`. Recomputes no physics. |
| `smoke_test.py` | Quick end-to-end check: exercises the three regimes, verifies the potential identity, writes `results/smoke_results.json`. |

## Installation

Python **3.13** and a **CPU-only** PyTorch build are required. No GPU is used.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate   |   POSIX:  source .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Exact pinned versions: Python 3.13.2, torch 2.12.0+cpu, numpy 2.4.4,
scipy 1.17.1, matplotlib 3.10.9 (see `requirements.txt`).

## Reproduction

### One command

```bash
python run_all.py               # full pipeline, ~6-9 min on a laptop CPU
python run_all.py --skip-training   # reuse committed models/ (seconds)
python run_all.py --list        # show stage names
```

`run_all.py` seeds all RNGs, runs every stage in dependency order with printed
banners and timings, and rebuilds `results/`, `models/`, `logs/`, `figures/`.

### Or step by step (run from the `code/` directory)

```bash
python src/smoke_test.py            # 1. smoke test  -> results/smoke_results.json
python src/bifurcation.py           # 2. static fold -> results/fold_boundary.csv
python src/train_trajectories.py    # 3. PINN training (SLOW ~5-8 min)
                                    #    -> models/pinn_*.pth, logs/history_*.json,
                                    #       results/traj_*.npz, pinn_traj_summary.json
python src/verify.py                # 4. verification -> results/verification_*.{csv,json},
                                    #    results/rk4_stepstudy.csv
python src/phase_diagram.py         # 5. phase diagram -> results/phase_diagram.npz,
                                    #    phase_boundary_static.csv, logs/phase_diagram.json
python src/lifshitz_boundary.py     # 6. Lifshitz shift -> results/lifshitz_*.{npz,csv}
python -m src.pinn_boundary         # 7. parametric PINN boundary (relative imports!)
                                    #    -> models/pinn_param_boundary.pth,
                                    #       results/pinn_boundary.npz, logs/pinn_boundary.json
python src/make_figures.py          # 8. figures -> figures/fig{1..4}.{pdf,png}
```

> **Note.** `pinn_boundary.py` uses package-relative imports, so it must be run
> as a module (`python -m src.pinn_boundary`), not as `python src/pinn_boundary.py`.
> `run_all.py` handles this automatically.

### Determinism

Every stage calls `src.config.set_all_seeds(42)` (numpy + torch,
`PYTHONHASHSEED=42`, `torch.use_deterministic_algorithms(True, warn_only=True)`).
All computation is **CPU-only float64**. Re-running reproduces every artefact to
solver tolerance. See **[SEEDS.md](SEEDS.md)** for the full RNG/determinism
contract.

## Expected key numerical outputs

| Quantity | Value | Source |
|---|---|---|
| Electrostatic fold endpoint `α_c` at `β=0` | `4/27 ≈ 0.148148` | `bifurcation.py` |
| Casimir fold endpoint `β*` at `α=0` | `256/3125 ≈ 0.081920` | `bifurcation.py` |
| PINN trajectory `L∞(ξ)` — stable | `1.05 × 10⁻³` | `results/pinn_traj_summary.json` |
| PINN trajectory `L∞(ξ)` — growing | `1.95 × 10⁻³` | `results/pinn_traj_summary.json` |
| PINN trajectory `L∞(ξ)` — pull-in | `2.83 × 10⁻³` | `results/pinn_traj_summary.json` |
| Pull-in time `τ*` (pull-in regime, DOP853 1e-12) | `≈ 2.6226` | `results/verification_summary.json` |
| Parametric-PINN boundary vs analytic fold | mean `|Δα| ≲ 1%` of `4/27` | `logs/pinn_boundary.json` |
| Lifshitz `ΔV_PI/V_PI` at `β=0.03`, `T=100 K` | `−2.88%` | `results/lifshitz_shift_summary.csv` |
| Lifshitz `ΔV_PI/V_PI` at `β=0.03`, `T=300 K` | `−8.87%` | `results/lifshitz_shift_summary.csv` |
| Lifshitz `β*` shift, `T=300 K` | `−22.6%` (`0.0819 → 0.0634`) | `results/lifshitz_shift_summary.csv` |

Fixed-step RK4 overshoots the movable pole (`ξ > 1`) for every step `h` tested,
whereas the adaptive reference and the rapidity-PINN remain well posed — this is
the central methodological point substantiated by `verify.py` and Fig. 4.

## Directory layout

```
code/
├── run_all.py            # master pipeline (this archive's single entry point)
├── requirements.txt      # exact pinned dependencies (CPU-only torch)
├── LICENSE               # MIT (code); CC-BY-4.0 for data/figures (see note)
├── CITATION.cff          # citation metadata (software + preferred article)
├── .zenodo.json          # Zenodo deposit metadata
├── SEEDS.md              # RNG seeds & determinism guarantees
├── README.md             # this file
├── src/                  # source modules (see module map)
├── docs/                 # theory notes (physics, bifurcation, PINN, Lifshitz)
├── results/              # .npz / .csv / .json numerical outputs
├── models/               # trained PINN weights (*.pth)
├── logs/                 # training / provenance logs (*.json)
└── figures/              # publication figures fig1..fig6 (PDF + PNG)
```

## Hardware & runtime

Reference machine: laptop-class **CPU only** (no CUDA; `torch.cuda.is_available()`
is `False`). Memory footprint is modest (the PINNs are tiny tanh MLPs, float64).

| Stage | Approx. runtime |
|---|---|
| smoke, bifurcation, verify, lifshitz, pinn_boundary, figures | seconds each |
| phase_diagram (200×200 vectorised RK4, two damping values) | ~tens of seconds |
| **train_trajectories** (3 regimes, Adam + L-BFGS) | **~5-8 min** (dominant cost) |
| **Full `run_all.py`** | **~6-9 min** end to end |

## License

Source code: **MIT** (see `LICENSE`).
Data and figures (`results/`, `models/`, `logs/`, `figures/`, `docs/`):
**CC-BY-4.0**. Please credit the authors and the associated article.

## How to cite

If you use this software or its data, please cite **both** the article and this
archive.

**Article (in preparation):**

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, and Q.-H. Qin,
> "Casimir–Electrostatic Pull-In in Nanomechanical Actuators: Phase Diagrams,
> Design Sensitivities, and Physics-Informed Neural-Network Surrogates,"
> *Physical Review Applied* (in preparation, 2026).

**Software archive (this repository):**

> N. S. Akintsov, A. P. Nevecheria, S. N. Andreev, and Q.-H. Qin,
> "Casimir–Electrostatic NEMS Pull-In: Bifurcation, Phase Diagrams &
> Physics-Informed Neural Networks" (Version 1.1.0), Zenodo (2026).
> DOI: [10.5281/zenodo.22142711](https://doi.org/10.5281/zenodo.22142711)
> (version 1.1.0; concept DOI [10.5281/zenodo.21269554](https://doi.org/10.5281/zenodo.21269554) resolves to the latest version).

A machine-readable citation is provided in `CITATION.cff`.

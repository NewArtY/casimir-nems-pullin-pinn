#!/usr/bin/env python3
"""Master pipeline: regenerate the entire archive from scratch, deterministically.

Running

    python run_all.py

executes every stage of the Casimir-electrostatic NEMS pull-in study in
dependency order, rebuilding everything under ``results/``, ``models/``,
``logs/`` and ``figures/`` from the source modules in ``src/``.  Each stage is
run in-process (imported and called), with a printed banner and a wall-clock
timing; a final summary table reports per-stage status and elapsed time.

Determinism
-----------
Every stage seeds all RNGs (numpy + torch) with the fixed master seed 42 via
``src.config.set_all_seeds`` before doing any work.  All computation is CPU-only
float64.  See SEEDS.md for the full determinism contract.

Runtime (reference machine, CPU-only)
-------------------------------------
The three quick analysis/verification stages and the figure renderer finish in
seconds.  The single expensive stage is PINN trajectory training
(``train_trajectories``), which takes a few minutes (Adam + L-BFGS for three
regimes; ~5-8 min total on a typical laptop CPU).  The parametric-boundary PINN
(``pinn_boundary``) trains in a few seconds.  Expect ~6-9 minutes end-to-end.

Options
-------
    python run_all.py                 # full pipeline
    python run_all.py --skip-training # skip the slow PINN trajectory training
                                      # (reuses committed models/ + traj_*.npz)
    python run_all.py --only smoke bifurcation figures
    python run_all.py --list          # list stage names and exit
"""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Make ``code/`` (this directory) importable so ``import src.*`` always works,
# regardless of the directory python is launched from.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Also expose src/ on the path so scripts that do ``import figstyle`` resolve.
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Stage definitions.
#
# Each stage is (name, kind, target, description):
#   kind == "call"   -> import module `target` and call its entry function
#                       (module, func) ; func defaults to the module callable.
#   kind == "script" -> run `target` (a path) as __main__ via runpy.
#
# We run the package modules via import + entry-point call so relative imports
# (e.g. src/pinn_boundary.py uses ``from .config import ...``) resolve cleanly.
# ---------------------------------------------------------------------------
STAGES = [
    ("smoke",       "src.smoke_test",         "run",  "Smoke test: 3 regimes + potential identity"),
    ("bifurcation", "src.bifurcation",        None,   "Static fold bifurcation (closed-form + checks)"),
    ("training",    "src.train_trajectories", "main", "PINN trajectory training (SLOW ~5-8 min)"),
    ("verify",      "src.verify",             "main", "Verification study: PINN vs RK4 near pull-in"),
    ("phase",       "src.phase_diagram",      "run",  "Pull-in phase diagram (vectorised RK4 sweep)"),
    ("lifshitz",    "src.lifshitz_boundary",  "main", "Thermal (Lifshitz) fold-boundary shift"),
    ("pinn_bnd",    "src.pinn_boundary",      "main", "Parametric physics-informed pull-in boundary"),
    ("pullin_ref",  "src.pullin_reference",   "main", "Adaptive reference through to the barrier (Fig. 1c)"),
    ("figures",     "src.make_figures",       "main", "Render publication figures (Fig. 1-4)"),
    ("band_scan",   "src.band_scan",          "main", "Band width vs damping, Table S8 (SLOW ~3 min)"),
    ("app_checks",  "src.appendix_checks",    "main", "Numerical checks of Appendix B (SLOW ~5 min)"),
    ("inverse",     "src.inverse_design",     "main", "Inverse design + autodiff sensitivities (Fig. 5)"),
    ("dyn_bnd",     "src.dynamic_boundary_design", "run", "No-closed-form dynamic-boundary sensitivity (Fig. 6, SLOW ~2-3 min)"),
]

# Stages that are quick (used by --skip-training which drops 'training').
SLOW_STAGES = {"training", "dyn_bnd", "app_checks", "band_scan"}


def _banner(idx, total, name, desc):
    bar = "=" * 74
    print("\n" + bar)
    print(f"[{idx}/{total}]  STAGE: {name}")
    print(f"        {desc}")
    print(bar, flush=True)


def _run_stage(module_name, func_name):
    """Import ``module_name`` and invoke its entry point.

    If ``func_name`` is given, call ``module.func_name()``.  Otherwise the
    module's work is done at import time under an ``if __name__ == '__main__'``
    guard, so we execute it with runpy as ``__main__``.
    """
    if func_name is None:
        # Modules whose driver lives only under the __main__ guard
        # (e.g. bifurcation.py). Run the file as __main__.
        mod = importlib.import_module(module_name)
        runpy.run_path(mod.__file__, run_name="__main__")
        return
    mod = importlib.import_module(module_name)
    entry = getattr(mod, func_name)
    entry()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the full NEMS pull-in pipeline.")
    parser.add_argument("--skip-training", action="store_true",
                        help="skip the slow PINN trajectory training stage "
                             "(reuse committed models/ and results/traj_*.npz)")
    parser.add_argument("--only", nargs="+", metavar="STAGE",
                        help="run only the named stage(s), in the given order")
    parser.add_argument("--list", action="store_true",
                        help="list stage names and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Available stages (in pipeline order):")
        for name, _mod, _fn, desc in STAGES:
            slow = "  [SLOW]" if name in SLOW_STAGES else ""
            print(f"  {name:12s} {desc}{slow}")
        return 0

    stages = list(STAGES)
    if args.only:
        wanted = list(args.only)
        by_name = {s[0]: s for s in STAGES}
        unknown = [w for w in wanted if w not in by_name]
        if unknown:
            parser.error(f"unknown stage(s): {', '.join(unknown)}  "
                         f"(known: {', '.join(by_name)})")
        stages = [by_name[w] for w in wanted]
    elif args.skip_training:
        stages = [s for s in stages if s[0] not in SLOW_STAGES]

    print("#" * 74)
    print("# Casimir-electrostatic NEMS pull-in -- FULL REPRODUCIBILITY PIPELINE")
    print(f"# python {sys.version.split()[0]}  |  CPU-only float64  |  master seed 42")
    print(f"# stages: {', '.join(s[0] for s in stages)}")
    print("#" * 74)

    total = len(stages)
    results = []
    t_all = time.perf_counter()

    for i, (name, module_name, func_name, desc) in enumerate(stages, start=1):
        _banner(i, total, name, desc)
        t0 = time.perf_counter()
        try:
            _run_stage(module_name, func_name)
            dt = time.perf_counter() - t0
            results.append((name, "OK", dt))
            print(f"\n-- stage '{name}' completed in {dt:.1f} s", flush=True)
        except Exception:  # keep going so the summary shows what failed
            dt = time.perf_counter() - t0
            results.append((name, "FAIL", dt))
            print(f"\n!! stage '{name}' FAILED after {dt:.1f} s", flush=True)
            traceback.print_exc()
            break  # dependency order: later stages likely need this one

    t_total = time.perf_counter() - t_all

    print("\n" + "#" * 74)
    print("# PIPELINE SUMMARY")
    print("#" * 74)
    print(f"{'stage':<14}{'status':<8}{'time [s]':>10}")
    print("-" * 32)
    for name, status, dt in results:
        print(f"{name:<14}{status:<8}{dt:>10.1f}")
    print("-" * 32)
    print(f"{'TOTAL':<14}{'':<8}{t_total:>10.1f}")

    n_fail = sum(1 for _, s, _ in results if s != "OK")
    if n_fail:
        print(f"\n{n_fail} stage(s) FAILED.")
        return 1
    print("\nAll stages completed successfully. "
          "Archive regenerated deterministically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

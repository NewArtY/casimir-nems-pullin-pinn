# Changelog

All notable changes to this archive. Versions follow the Zenodo releases of
the record `10.5281/zenodo.21269555` (concept DOI); each release carries its
own version DOI.

## 1.1.0 - 2026-08-26

`.zenodo.json` keeps the concept DOI `10.5281/zenodo.21269555`. Zenodo mints a
separate version DOI when 1.1.0 is published; update `.zenodo.json`,
`CITATION.cff`, `README.md`, and the `Zenodo2026` entry of `article/refs.bib`
with that version DOI after release.

### Fixed

- **Thermal Lifshitz coefficient, corrected by a factor of 12.** The finite-
  temperature correction to the Casimir parameter used
  `c_L = 720 zeta(3)/pi^3 = 27.913`. The correct coefficient is the ratio of
  the classical (high-temperature) Casimir pressure
  `P_cl = zeta(3) k_B T/(4 pi a^3)` to the zero-temperature pressure
  `P_0 = pi^2 hbar c/(240 a^4)`, that is

      c_L = 60 zeta(3)/pi^3 = 2.3261 .

  The old value divided the classical term by the free-energy coefficient 720
  instead of the pressure coefficient 240 (factor 3) and dropped the 4 in
  `P_cl` (factor 4). The derivation is now Appendix A of the manuscript.
  Affects `src/physics.py`, `src/lifshitz_boundary.py`, `src/device_table.py`,
  `docs/lifshitz_notes.md`, and every downstream number:

  | Quantity | 1.0.0 | 1.1.0 |
  |---|---|---|
  | `kappa(100 nm)` | 1.219e-3 /K | 1.0158e-4 /K |
  | `dV_PI/V_PI` at `beta = 0.03`, 100/300 K | -2.88 % / -8.87 % | -0.237 % / -0.712 % |
  | `beta*` at 100/300 K | 0.074623 / 0.063412 | 0.081257 / 0.079914 |
  | Device `dV_PI` A/B/C/D at 300 K | -7.1/-13.6/-0.03/-5.7 % | -0.57/-1.07/-0.003/-0.46 % |

  The qualitative conclusions are unchanged: the shift is one-sided,
  destabilizing, vanishes at `beta = 0`, and grows toward the Casimir ceiling.
  The classical-limit bound is now itself sub-percent at sub-100-nm gaps,
  which makes it consistent with the low-temperature result
  `P = P_0 [1 + (16/3)(a/lambda_T)^4]` derived in the same appendix.

### Added

- `src/make_sm_tables.py` emits the Supplemental Material LaTeX tables
  directly from `results/` and `logs/`, so no number in the Supplemental
  Material is transcribed by hand.

### Changed

- `figures/fig3_lifshitz_shift.*` recomposed: with the corrected coefficient
  the three fold curves are indistinguishable in the `(alpha, beta)` plane, so
  the figure now reports the relative shift as a function of `beta`.
- Regenerated `results/lifshitz_boundaries.{npz,csv}`,
  `results/lifshitz_shift_summary.csv`, `results/device_table.{csv,json}`.

## 1.0.0 - 2026-07-08

Initial archive accompanying the submitted manuscript.

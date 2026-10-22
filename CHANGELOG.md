# Changelog

All notable changes to this archive. Versions follow the Zenodo releases of
the record `10.5281/zenodo.21269554` (concept DOI, resolves to the
latest version); each release carries its own version DOI.

| version | date | version DOI |
| :-- | :-- | :-- |
| 1.2.0 | 2026-10-22 | `10.5281/zenodo.23555718` |
| 1.1.0 | 2026-08-28 | `10.5281/zenodo.22142711` |
| 1.0.0 | 2026-07-09 | `10.5281/zenodo.21269555` |

## 1.2.0 - 2026-10-22 (accompanies the Physical Review E manuscript)

Published 2026-10-22 as `10.5281/zenodo.23555718`; the manuscript cites this
version DOI. The first attempt to release 1.2.0 failed because `.zenodo.json`
fixed `"doi"` to the concept DOI; that field is now omitted, so Zenodo mints
the version DOI itself.

### Added

- `src/damping_regimes.py`: squeeze-film damping of the representative
  devices in air (isothermal Reynolds film on a square plate, Blech 1983,
  with the rarefied-gas viscosity of Veijola et al. 1995), the zero-bias rest
  position, the undamped step threshold from that position, the damping
  `zeta_c` at which the band closes for each device, the pressures at which
  the gas damping reaches `zeta_c` and `2^(-1/4)`, and the band width in air
  with a constant and with a gap-dependent damping profile. Writes
  `results/damping_regimes.json`. Self-checks against the appendix:
  `zeta_c(0) = 0.395919`, the closed form of McLellan et al., and the band
  width `1.3482e-2` at `zeta = 0.1`.
- `make_sm_tables.py` now also emits Table S10 and the device table of the
  main text, `article/sections/tab_devices.tex`.
- `run_all.py` stage `damping`.

### Changed

- Figure numbering follows the restructured manuscript: the files keep their
  names, but `fig6_dynamic_boundary` is now Fig. 3, `fig3_lifshitz_shift`
  Fig. 4, `fig4_diagnostics` Fig. 5, and `fig5_inverse_design` Fig. 6.

## 1.1.0 - 2026-08-26

Published 2026-08-28 as `10.5281/zenodo.22142711`. Note that
`10.5281/zenodo.21269555`, used as the concept DOI before this release, is in
fact the version DOI of 1.0.0; the concept DOI is `10.5281/zenodo.21269554`.
The manuscript now cites the 1.1.0 version DOI, since that is the archive its
numbers come from.

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

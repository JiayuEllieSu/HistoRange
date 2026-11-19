# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- `pyproject.toml` and `MANIFEST.in`: the package installs with `pip install .` (optional `test` and `dev` extras), and the source distribution carries the tests, example and docs.
- Test suite under `tests/` covering every module, plus an end-to-end test of the example.
- `examples/synthetic_walkthrough.py`: the complete secreted-interaction workflow on simulated data.
- Documentation: expanded `README.md`, `docs/api.md`, `docs/settings.md`.
- `CITATION.cff`, this changelog, and GitHub Actions CI (`.github/workflows/tests.yml`).
- `histrange.__version__`.

### Fixed
- `evidence.mesh_robust`, `anisotropy_robust`, `response_robust`: added a 1e-9-bit tolerance.
  Without it, 22 of the 37 exact 0.5-bit (4-step) mesh shifts on the master grid were judged not
  robust because of floating-point roundoff, contrary to Methods §3.17.
- `settings.SHORT_BAND_BITS[0.90]`: 1.312 → 1.3125, the value in the frozen calibration
  (`om_boundary_final.csv`). The library now reproduces the published 90% bands to the reported
  precision (e.g. 21.7–133.7 µm for ℓ̂ = 53.8 µm, previously 21.7–133.6 µm).
- `geometry.morphology_mismatch`: blocks are now permuted only among blocks at least half inside
  tissue, matching the frozen real-specimen control (`shuffled_in_tissue`). Previously background
  blocks were swapped into tissue, which on a section covering part of the frame cut the in-tissue
  anisotropy by roughly half instead of only scrambling it. Output is unchanged when tissue fills
  the image.

- `transport.Operator.validate`: detect indefinite stiffness matrices by computing the
  algebraically smallest generalized eigenvalue instead of searching near the no-flux zero mode.
  Fail validation if the spectral calculation does not complete, and require the previously
  expected-failure regression to pass.
- Restore the public GitHub Actions workflow for tests, linting, and wheel builds.
- Include the changelog in the public repository and source distributions, so package
  metadata checks can run consistently across supported environments.

## [0.1.0] - 2026-09-14

### Added
- Initial public release of the core library: histology preprocessing, H&E-derived transport
  geometry, P1 finite-element mesh and screened-transport operator, source/receiver projection,
  contact exposure, Matérn-3/2 GLS association testing with dense and Vecchia covariance,
  parametric bootstrap with wrong-ligand calibration, Benjamini–Hochberg control, T1 range
  estimation, boundary-aware range reporting, and evidence-layer helpers.

[Unreleased]: https://github.com/JiayuEllieSu/HistoRange/compare/a510971...HEAD
[0.1.0]: https://github.com/JiayuEllieSu/HistoRange/commit/a510971

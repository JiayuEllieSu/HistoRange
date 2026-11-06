# Fixed analysis settings

Every constant in [`histrange/settings.py`](../histrange/settings.py), what it controls, where it is
used, and the Methods section it comes from. These are the values used for every specimen in the
manuscript. They are module-level constants rather than a configuration file on purpose: the analysis
is prespecified, so changing a value is a sensitivity analysis and should be reported as one.

## Histology and tissue support (Methods §3.5)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `H_STAIN` | (0.650, 0.704, 0.286) | Haematoxylin reference stain vector in optical density | `histology.stain_basis` |
| `E_STAIN` | (0.072, 0.990, 0.105) | Eosin reference stain vector | `histology.stain_basis` |
| `MASK_SIGMA_UM` | 2.0 µm | Gaussian smoothing of OD magnitude before Otsu thresholding | `histology.tissue_mask` |
| `OPENING_RADIUS_UM` | 4.0 µm | Disk radius for morphological opening (removes specks) | `histology.tissue_mask` |
| `CLOSING_RADIUS_UM` | 8.0 µm | Disk radius for morphological closing (bridges small gaps; larger holes survive) | `histology.tissue_mask` |
| `MIN_COMPONENT_AREA_UM2` | 500 µm² | Connected tissue components smaller than this are removed | `histology.tissue_mask` |

## Transport geometry (Methods §3.6)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `DERIVATIVE_SCALE_UM` | 2.0 µm | σ_G, pre-smoothing scale of the eosin channel before differentiation | `geometry.build` |
| `TENSOR_SCALE_UM` | 12.0 µm | σ_T, smoothing scale of the structure tensor and crossing moments | `geometry.build` |
| `COHERENCY_FLOOR` | 0.20 | C₀; coherency at or below this gives isotropic transport | `geometry.build` |
| `CROSSING_THRESHOLD` | 0.35 | X_max; crossing statistic above this reverts to isotropic transport | `geometry.build` |
| `BETA_MAX` | ½ ln 3 ≈ 0.549 | Maximum log-anisotropy; e^(2β_max) = 3 is the largest principal conductance ratio | `geometry.build` |

## Mesh (Methods §3.8)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `DEFAULT_MESH_UM` | 4.0 µm | Default P1 lattice spacing h | `mesh.build` |
| `FALLBACK_MESH_UM` | (8.0, 16.0) µm | Mesh ladder used only when the default would exceed the node budget | `mesh.build` |
| `MAX_MESH_NODES` | 2,000,000 | Node budget per mesh | `mesh.build` |

## Range grid (Methods §3.9)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `RANGE_MIN_UM` | 16 µm | Master grid floor | `master_range_grid`, `trusted_range_grid` |
| `RANGE_MAX_UM` | 512 µm | Master grid ceiling | `master_range_grid`, `trusted_range_grid` |
| `RANGE_INTERVALS_PER_OCTAVE` | 8 | log₂-uniform spacing; 41 master values | `master_range_grid` |
| `MIN_RANGE_POINTS` | 9 | Minimum trusted values required for secreted range profiling | `trusted_range_grid` |

The trusted grid keeps master values in
[max(16 µm, 4h, 0.5 d_NN), min(512 µm, 0.25 D_geo)], where h is the mesh spacing, d_NN the median
receiver nearest-neighbour distance and D_geo the tissue geodesic diameter.

## Molecular eligibility (Methods §3.3)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `SOURCE_MIN_POSITIVE` | 20 | Minimum ligand-positive observations | `molecular.eligibility` |
| `SOURCE_MIN_RAW_COUNTS` | 25 | Minimum aggregate raw counts across required ligand subunits | `molecular.eligibility` |
| `RECEPTOR_MIN_COMPETENT` | 30 | Minimum receptor-competent receivers | `molecular.eligibility` |
| `STATE_MIN_FRACTION` | 0.10 | Receptor competence must reach this fraction in at least one state | `molecular.eligibility` |
| `STATE_MIN_SIZE` | 30 | Only states with at least this many observations count | `molecular.eligibility` |

An observation is ligand-positive (receptor-competent) when every required subunit has at least one
raw count and at least 1 CP10K.

## Spatial covariance and detection (Methods §3.11–3.12)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `NUGGET_BOUNDS` | (0.05, 0.95) | Bounds on the nugget fraction τ | `spatial.fit_covariance` |
| `NUGGET_START` | 0.25 | First optimizer start for τ | `spatial.fit_covariance` |
| `DENSE_GP_MAX_RECEIVERS` | 2,000 | Full covariance up to this many receivers; Vecchia above | `spatial.fit_covariance` |
| `VECCHIA_NEIGHBORS` | 30 | Conditioning-set size of the max–min Vecchia approximation | `spatial.fit_covariance` |
| `FALLBACK_PHI_POINTS` | 16 | φ grid size for the grid search used only if every optimizer start fails | `spatial.fit_covariance` |
| `FALLBACK_NUGGET_GRID` | (0.05 … 0.95) | τ grid for the same fallback | `spatial.fit_covariance` |
| `BOOTSTRAP_DRAWS` | 9,999 | Parametric bootstrap draws, with the +1 correction | `spatial.bootstrap_pvalue` |
| `SECRETED_CALIBRATION_C` | 0.6589 | c in p_sec = min(1, c · p_boot^γ) | `spatial.calibrated_secreted_pvalue` |
| `SECRETED_CALIBRATION_GAMMA` | 0.7998 | γ in the same calibration | `spatial.calibrated_secreted_pvalue` |

The Matérn correlation length is bounded by max(10 µm, 2 d_NN) ≤ φ ≤ min(1000 µm, D_geo / 3)
(`spatial.phi_bounds`). Contact-annotated interactions use the raw bootstrap probability.

## Range reporting (Methods §3.14–3.15)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `SHORT_RANGE_SPLIT_UM` | 64 µm | ℓ̂ below this is the boundary (short-range) regime; at or above, the interior regime | `range.report_range` |
| `SHORT_BAND_BITS` | {0.90: 1.3125, 0.95: 1.750} | Half-width B (log₂ units) of the two-sided band ℓ̂ · 2^±B. The 90% value is the frozen calibration constant (`om_boundary_final.csv`); the Methods print it rounded as 1.312 | `range.report_range` |
| `SHORT_UPPER_BITS` | {0.90: 0.875, 0.95: 1.500} | One-sided bound ℓ̂ · 2^u_b in the boundary regime | `range.report_range` |
| `LONG_UPPER_BITS` | {0.90: 1.000, 0.95: 1.250} | One-sided bound ℓ̂ · 2^u_i in the interior regime | `range.report_range` |

The support threshold τ is the 99th percentile of Λ_T1 over the specimen's own near-null secreted
modules (not detected, Λ_A ≤ 3), computed by `range.support_threshold`.

## Evidence layers (Methods §3.16–3.17)

| Setting | Value | Meaning | Used by |
|---|---|---|---|
| `SPECIFICITY_MIN_R2` | 0.01 | Held-out partial R² required in both folds | `evidence.specificity_supported` |
| `ANISOTROPY_MULTIPLIERS` | (0.5, 1.0, 1.5) | β multipliers for the anisotropy-strength arm (pass as `beta_scale` to `transport.assemble`) | caller |
| `SATURATION_K` | (0.25, 0.5, 1, 2, 4) | K values for the response-form arm | caller |
| `MAX_RANGE_SPREAD_BITS` | 1.0 | Maximum span (anisotropy) or shift (response form) of re-estimated ranges | `evidence.anisotropy_robust`, `evidence.response_robust` |
| `MAX_MESH_SHIFT_BITS` | 0.5 | Maximum range shift on the adjacent mesh rung | `evidence.mesh_robust` |

## Values set in code rather than in `settings.py`

These are part of the prespecified analysis but live next to the code that uses them.

| Where | Value | Meaning |
|---|---|---|
| `molecular.qc_mask` | ≥ 500 UMI, ≥ 200 genes, mito ≤ 0.20 | Observation-level QC (Methods §3.3) |
| `molecular.detected_gene_mask` | ≥ max(10, ⌈0.001 n⌉) observations | Gene detection filter (Methods §3.3) |
| `contact.contact_exposure` | 1.5 d_NN | Adjacency radius (Methods §3.10) |
| `spatial.phi_bounds` | 10 µm, 2 d_NN, 1000 µm, D_geo/3 | Matérn φ bounds (Methods §3.11) |
| `range.support_threshold` | 0.99 quantile | Support threshold (Methods §3.14) |
| `geometry.morphology_mismatch` | 64 µm blocks | Block size of the morphology-mismatch control |
| `evidence.checkerboard_folds` | 8 × 8 blocks | Spatial cross-fitting partition |
| `transport.Operator.validate` | 1e-10, 1e-8 | Relative symmetry and eigenvalue tolerances |
| `transport.MAX_CACHED_FACTORIZATIONS` | 4 | LU factorizations kept per operator (performance only) |

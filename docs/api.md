# API reference

Function-level reference for the `histrange` package. Units are microns (µm) throughout. Image
arrays are indexed `[row, col]`; point coordinates are `(x, y)` with `x = col * pixel_size_um` and
`y = row * pixel_size_um`.

> `histrange.range` is a module name, so `from histrange import *` replaces the builtin `range` in
> your namespace. Import it as `from histrange import range as range_estimation` instead.

Top-level shortcuts: `histrange.prepare_histology` (= `histology.prepare`), `histrange.build_geometry`
(= `geometry.build`), `histrange.build_mesh` (= `mesh.build`), `histrange.master_range_grid`,
`histrange.trusted_range_grid`, and `histrange.__version__`.

---

## `histrange.molecular` — scores, QC and eligibility

| Function | Description |
|---|---|
| `cp10k(raw, library_size)` | Counts per 10,000. `raw` may be a vector or an `(n, genes)` matrix; library sizes below 1 are clamped to 1. |
| `complex_count(subunit_counts)` | Abundance of a multi-subunit complex as its least abundant subunit (row-wise minimum). A vector passes through unchanged. |
| `bounded_score(raw_complex, library_size)` | Bounded activation score A = Z / (Z + K) with Z = log1p(CP10K) and K the specimen median of positive Z (Methods Eq. 3.3). Values lie in [0, 1); zeros map to 0. |
| `qc_mask(total_umi, detected_genes, mito_fraction=None)` | Boolean mask of observations with ≥ 500 UMI, ≥ 200 genes and (if given) mitochondrial fraction ≤ 0.20. |
| `detected_gene_mask(counts)` | Genes with a nonzero count in at least max(10, ⌈0.001 n⌉) observations. |
| `eligibility(ligand_subunits, receptor_subunits, library_size, states=None)` | Applies the frozen source and receiver gates (Methods §3.3). Returns a dict with `eligible`, `n_positive_source`, `total_raw_source`, `n_competent_receivers`, `max_state_fraction`, and the boolean vectors `ligand_positive` and `receptor_competent`. Subunit arrays are `(n,)` or `(n, subunits)`. |

## `histrange.histology` — H&E preprocessing

| Function / class | Description |
|---|---|
| `prepare(image, pixel_size_um=1.0, capture_mask=None)` | Full preprocessing: optical density, colour deconvolution and tissue support. Returns a `Histology`. `image` is `(rows, cols, 3)` RGB in 0–255 or 0–1. `capture_mask` restricts the Otsu threshold and the support to the capture area. |
| `Histology` | Attributes `mask`, `eosin`, `hematoxylin`, `od_norm`, `pixel_size_um`; property `shape`; method `tissue_area_um2()`. |
| `optical_density(image)` | OD = −log((I + 1) / 256) per channel. An alpha channel is ignored. |
| `stain_basis()` | Orthonormalised `[h, e, h × e]` basis from the reference stain vectors. |
| `separate_stains(od, basis)` | Per-pixel stain concentrations, truncated at zero. |
| `otsu_threshold(values, bins=256)` | Otsu threshold. When several thresholds tie (an empty gap between modes), the lowest is returned. |
| `tissue_mask(od, pixel_size_um, capture_mask=None)` | Smoothing → Otsu → opening → closing → removal of components below 500 µm². Internal holes are kept. Raises `ValueError` if the mask is empty. |

## `histrange.geometry` — transport tensor from H&E

| Function / class | Description |
|---|---|
| `build(histology)` | Structure-tensor orientation, coherency, fourth-moment crossing statistic and directional strength β (Methods Eqs. 3.5–3.9). Returns a `Geometry`. |
| `Geometry` | Pixel fields `theta` (tissue-line orientation, radians), `coherency`, `crossing`, `beta`, `mask`, and `pixel_size_um`. `tensor_at(points_um, beta_scale=1.0)` returns `(n, 3)` tensor components `[a11, a12, a22]` sampled at the nearest pixel (identity outside tissue; coordinates outside the image are clipped to the edge). `anisotropic_fraction()` is the share of tissue pixels with β > 0. |
| `tensor_components(theta, beta)` | `R(θ) diag(e^β, e^−β) R(θ)ᵀ` as `[a11, a12, a22]`; determinant 1. |
| `structure_tensor(channel, sigma_px)` | Derivatives and smoothed tensor entries `(ex, ey, jxx, jxy, jyy)`. |
| `crossing_statistic(ex, ey, sigma_px)` | X = \|m₄ − m₂²\| from the weighted circular moments of gradient direction. |
| `isotropic(geometry)` | Control with β = 0 everywhere (A = I). |
| `rotated(geometry, degrees=90.0)` | Control with orientation turned and anisotropy magnitude and support preserved. |
| `morphology_mismatch(geometry, block_um=64.0, seed=0)` | Control that permutes orientation/anisotropy blocks among the blocks that are at least half inside tissue, destroying the correspondence with position while keeping the in-tissue (θ, β) distribution. Images smaller than 2 × 2 blocks are returned unchanged (as a copy). |

## `histrange.mesh` — finite-element mesh

| Function / class | Description |
|---|---|
| `build(histology, spacing_um=None, keep_components="all")` | Structured P1 lattice over the tissue support with alternating diagonals. With `spacing_um=None` it tries 4 µm, then 8 and 16 µm if the node budget is exceeded. `keep_components="all"` keeps every island (with a warning that transport cannot cross between them); `"largest"` keeps only the largest (with a warning giving the number of nodes dropped). |
| `Mesh` | `nodes (N, 2)`, `triangles (T, 3)` (counter-clockwise), `spacing_um`, `areas`, `centroids`, `node_areas`, `n_components`, `node_components`; properties `n_nodes`, `n_triangles`, `connected`. |
| `Mesh.locate(points, n_candidates=16, boundary_tolerance_um=0.0)` | Containing triangle, barycentric weights and a report (`n_inside`, `n_outside`, `outside_indices`, `n_snapped`, `snapped_indices`, `max_snap_distance_um`). Points outside the support get triangle `-1`. With a positive tolerance, near misses are snapped to the boundary. |
| `Mesh.contains(points, ...)` / `component_of(points, ...)` | Boolean containment, and the component label (or `-1`). |
| `Mesh.require_inside(points, ..., label="observations")` | Like `locate` but raises `OutsideTissueError` listing offending indices. Observations off tissue must be filtered, never projected. |
| `Mesh.geodesic_diameter()` / `component_diameters()` | Graph-geodesic diameter along mesh edges, estimated by a double Dijkstra sweep (a lower bound on the exact graph diameter, typically tight on compact tissue); cached. |
| `Mesh.median_nearest_neighbour()` | Median mesh-edge length. |
| `OutsideTissueError` | Subclass of `ValueError`. |

Note: snapping clamps and renormalises barycentric weights, which lands on the boundary but not
exactly at the orthogonal foot, so `max_snap_distance_um` can slightly overstate the true distance.
The tolerance test is therefore conservative.

## `histrange.transport` — screened transport

| Function / class | Description |
|---|---|
| `assemble(mesh, geometry, beta_scale=1.0, max_cached=None)` | Assembles the anisotropic P1 stiffness `S` (tensor at triangle centroids), consistent mass `M`, and lumped mass. No-flux boundaries are natural in the weak form. `beta_scale` multiplies β (anisotropy-strength robustness arm). Returns an `Operator`. |
| `Operator.solve(ell, load)` | Solves (S + ℓ⁻² M) u = load. LU factorizations are cached per ℓ (default 4 most recent). |
| `Operator.log_range_derivative(ell, u)` | ∂u/∂ log ℓ = (2/ℓ²)(S + ℓ⁻² M)⁻¹ M u. |
| `Operator.residual(ell, u, load)` | Relative residual ‖(S + ℓ⁻² M)u − load‖ / ‖load‖. |
| `Operator.mass_solve(load)` | Solves M x = load. |
| `Operator.checks(spectrum=False)` | Symmetry, minimum lumped mass, no-flux residual ‖S 1‖, finiteness, component count; optionally the smallest generalized eigenvalue. |
| `Operator.validate(symmetry_tolerance=1e-10, eigenvalue_tolerance=1e-8, spectrum=True, strict=True)` | Runs the checks and raises `OperatorError` on failure (or returns `(report, failures)` with `strict=False`). |
| `exposure_field(operator, ell, load)` | Convenience wrapper for `operator.solve`. |

Known limitation: the eigenvalue part of `validate` searches near zero with shift-invert Lanczos for
one eigenvalue. Because the no-flux operator always has an exact zero eigenvalue (constants), the
search returns it and cannot detect a materially negative eigenvalue elsewhere in the spectrum.
Correctly assembled operators are positive semidefinite by construction (symmetric positive-definite
tensor, P1 elements), so this does not affect results, but the check is not a real safeguard yet.
`tests/test_transport.py::test_validation_rejects_indefinite_stiffness` is a strict expected failure
documenting it.

## `histrange.projection` — sources and receivers

| Function | Description |
|---|---|
| `source_operator(mesh, coordinates, radius_um=None, boundary_tolerance_um=0.0, location=None)` | Sparse `(nodes, n_obs)` deposition matrix. With a radius, each observation's unit mass is spread over mesh nodes inside its capture footprint (weighted by node area, restricted to its own tissue component); with no radius, or if the footprint covers no node, barycentric point deposition is used. Each column sums to 1. |
| `receiver_operator(...)` | Sparse `(n_obs, nodes)` averaging matrix with the same footprint rule; each row sums to 1. |
| `deposit(source_matrix, ligand)` | Load vector `P_L · L`. |
| `interpolate(receiver_matrix, field)` | Receiver exposure `P_R · u`. |
| `nodal_density(load, node_areas)` | Load per unit area. |

`location=(tri, weights)` reuses a previous `mesh.locate` result. Any observation outside the
support raises `OutsideTissueError`.

## `histrange.contact` — contact-annotated interactions

| Function | Description |
|---|---|
| `contact_operator(coordinates, radius, components=None)` | Row-normalised adjacency within `radius`, no self-links, optionally restricted to equal component labels. Isolated rows stay zero. |
| `contact_exposure(coordinates, ligand_score, median_nn, components=None)` | Mean ligand score of neighbours within 1.5 `median_nn` (Methods Eqs. 3.16–3.18). |

The operator is square over the supplied coordinates. To use receptor-competent receivers as rows
and all QC observations as senders, index the rows of the returned matrix.

## `histrange.spatial` — covariance, association test, multiplicity

| Function / class | Description |
|---|---|
| `fit_covariance(response, baseline, coordinates, phi_bounds, method="auto", dense_max_receivers=None)` | REML fit of σ²[τI + (1 − τ)K_φ] under the baseline-only model with Matérn-3/2 K_φ. `"auto"` uses dense Cholesky up to 2,000 receivers and Vecchia above. Five L-BFGS-B starts; if all fail, a φ × τ grid search (`fit_method="grid_fallback"`). Returns a `Covariance`. |
| `Covariance` | `phi`, `nugget`, `sigma2`, `fit_method`, `approximation`; `whiten(x)` and `quadratic(x)` = xᵀΣ⁻¹x. |
| `profile(response, baseline, exposures, covariance)` | Whitened GLS profile over candidate exposures (Methods Eq. 3.22). Returns `Q` (loss per exposure), `Q0` (baseline-only loss), `alpha` (amplitudes), `degenerate` (exposure inside the baseline span), `independence`, and the whitened bases used by the bootstrap. The association statistic is Λ_A = `Q0 − Q.min()`. |
| `bootstrap_pvalue(fit, draws=9999, seed=0)` | Parametric bootstrap repeating the full range search in every draw, with the +1 correction. Smallest attainable value is 1 / (draws + 1). `draws=0` returns `None`. |
| `calibrated_secreted_pvalue(p_boot)` | p_sec = min(1, 0.6589 p_boot^0.7998) (Methods Eq. 3.24). Secreted interactions only; contact uses p_boot directly. |
| `benjamini_hochberg(pvalues)` | BH q-values, monotone and capped at 1. |
| `phi_bounds(median_nn_um, geodesic_diameter_um)` | (max(10, 2 d_NN), min(1000, D_geo / 3)). |
| `matern32(distance, phi)`, `gls_fit(design, response)`, `orthonormal_basis(matrix)` | Building blocks; redundant columns are dropped by an SVD rank cut (rcond 1e-10). |

## `histrange.vecchia` — nearest-neighbour Gaussian-process approximation

| Function / class | Description |
|---|---|
| `maximin_order(coordinates)` | Deterministic max–min ordering starting from the point closest to the centroid. |
| `neighbour_sets(coordinates, order, n_neighbours)` | Nearest previously ordered points for each point. |
| `plan(coordinates, n_neighbours=30, order=None)` | Precomputed ordering and conditioning sets, reusable across (φ, τ). |
| `build_factor(coordinates_or_plan, phi, nugget, kernel, order=None, n_neighbours=30)` | Sparse factor U with UUᵀ ≈ Σ⁻¹; `apply_transpose(x)` whitens and `logdet()` gives log det Σ. Exact when every point conditions on all previous points. |

## `histrange.range` — post-detection range estimation

| Function | Description |
|---|---|
| `fit_range_profile(counts, library_size, covariates, exposures, ranges)` | T1: NB2 GLM on raw receptor counts over all in-support bins with a log library-size offset, covariates, and one standardised exposure term, profiled over ℓ. Dispersion is estimated once under the no-exposure model. Returns `Lambda` (= 2 × max log-likelihood gain), `ell_hat`, `best_index`, `profile`, `dispersion`, `amplitude`, `n`, `loglik_null`. Pass `np.zeros((n, 0))` for no covariates. |
| `support_threshold(near_null_statistics)` | 99th percentile of the finite near-null Λ_T1 values; NaN if none. |
| `report_range(ell_hat, best_index, n_ranges, lambda_t1, support_threshold, grid_min, grid_max, level=0.95)` | Boundary-aware claim (Methods §3.14–3.15). Returns `{"status": "unresolved", "reason": ...}` if the optimum is on a grid edge or Λ_T1 ≤ τ; otherwise `"interval"` (`lower_um`, `upper_um`) when ℓ̂ < 64 µm and the band lies inside the trusted grid, `"upper_bound"` (`upper_um`) when the band's lower edge reaches the trusted floor or ℓ̂ ≥ 64 µm, and `"uninformative"` when the bound exceeds the trusted ceiling. `level` must be 0.90 or 0.95. |

T1 is not a detector: run it only for interactions already detected by the spatial model.

The frozen phase driver (`histra_apply.py`) refuses to form τ from fewer than 10 near-null modules;
`support_threshold` returns a value from any non-empty set, so enforce that minimum in the caller.

## `histrange.evidence` — specificity, geometry and robustness

| Function | Description |
|---|---|
| `checkerboard_folds(coordinates, blocks=8)` | Two interleaved spatial folds (0/1) from an 8 × 8 checkerboard over the bounding box. |
| `partial_r2(loss_without_target, loss_with_target)` | Held-out incremental contribution (Methods Eq. 3.32); NaN if the reference loss is not positive. |
| `specificity_supported(fold_r2, threshold=0.01)` | True only when there are at least two folds and every fold reaches the threshold. |
| `geometry_gain(isotropic_loss, matched_loss, n_receivers)` | (Q_iso − Q_matched) / n_HO (Methods Eq. 3.33); positive favours matched histology. |
| `anisotropy_robust(ranges_um, max_spread_bits=1.0)` | Span log₂(max / min) of the β-multiplier estimates ≤ 1 bit. |
| `response_robust(reference_um, perturbed_um, max_shift_bits=1.0)` | Largest \|log₂(ℓ_K / ℓ_ref)\| over the response-form arm ≤ 1 bit (as computed in the frozen pipeline). |
| `mesh_robust(reference_um, adjacent_um, max_shift_bits=0.5)` | \|log₂(ℓ_adjacent / ℓ_ref)\| ≤ 0.5 bit. |

All three robustness comparisons add `BITS_TOLERANCE = 1e-9` bit so that an exactly grid-aligned
shift (4 grid steps = 0.5 bit, 8 steps = 1 bit) is not flipped by floating-point roundoff (Methods §3.17).

## `histrange.settings`

Constants and the two grid functions; see [`settings.md`](settings.md).

| Function | Description |
|---|---|
| `master_range_grid()` | 41 log₂-uniform values from 16 to 512 µm. |
| `trusted_range_grid(mesh_spacing_um, median_nn_um, geodesic_diameter_um)` | Master values within [max(16, 4h, 0.5 d_NN), min(512, 0.25 D_geo)]. Raises `ValueError` if fewer than nine remain. |

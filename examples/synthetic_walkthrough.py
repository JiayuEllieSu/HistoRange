"""End-to-end HistoRange walkthrough on a synthetic specimen.

This script runs every stage of the secreted-interaction workflow on simulated data
with a known answer, using only the public ``histrange`` API:

    1. H&E preprocessing             histrange.histology.prepare
    2. Directional transport geometry histrange.geometry.build
    3. Finite-element mesh            histrange.mesh.build
    4. Screened-transport operator    histrange.transport.assemble (+ validate)
    5. Source / receiver projection   histrange.projection
    6. Trusted range grid             histrange.settings.trusted_range_grid
    7. Molecular eligibility          histrange.molecular.eligibility
    8. Spatial association test       histrange.spatial (GLS profile, bootstrap,
                                      wrong-ligand calibration, Benjamini-Hochberg)
    9. Post-detection range (T1)      histrange.range.fit_range_profile
   10. Support gate + reporting       histrange.range.support_threshold / report_range
   11. Range robustness               histrange.evidence (anisotropy and mesh arms)

The specimen is a 400 x 400 um tissue section with fibres running at 30 degrees and a
lumen in the middle, measured on a 16 um bin lattice. A ligand is expressed by a band of
source bins; the receptor responds to the transported ligand at a true effective range of
about 41.5 um. A set of null receptor modules with no ligand dependence stands in for the
specimen's near-null secreted modules and fills out the Benjamini-Hochberg family.

Run from the repository root:

    python examples/synthetic_walkthrough.py

The confirmatory analysis in the manuscript uses 9,999 bootstrap draws
(``histrange.settings.BOOTSTRAP_DRAWS``); this demo uses 1,999 so it finishes in about
half a minute. Note the resolution floor this implies: the smallest attainable bootstrap
probability is 1 / (draws + 1), so with too few draws even a strong signal cannot survive
Benjamini-Hochberg across a large family.
"""

import argparse
import time
import warnings

import numpy as np

import histrange
from histrange import evidence, geometry, molecular, projection, spatial, transport
from histrange import range as range_estimation
from histrange.settings import ANISOTROPY_MULTIPLIERS, master_range_grid, trusted_range_grid

TRUE_RANGE_INDEX = 11  # master grid value 16 * 2**(11/8) = 41.5 um


# --------------------------------------------------------------------------- simulation
def synthetic_he(field_um, pixel_size_um, fibre_angle_deg, lumen_radius_um, seed):
    """RGB image of eosin-rich fibres on a white slide, with a central lumen."""
    rng = np.random.default_rng(seed)
    n = int(round(field_um / pixel_size_um))
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) * pixel_size_um
    a = np.deg2rad(fibre_angle_deg)
    across = -np.sin(a) * xx + np.cos(a) * yy
    eosin = 0.55 + 0.45 * np.sin(2.0 * np.pi * across / 14.0)
    hematoxylin = np.full_like(eosin, 0.15)
    od = hematoxylin[..., None] * histrange.settings.H_STAIN + eosin[..., None] * histrange.settings.E_STAIN
    rgb = 256.0 * np.exp(-od) - 1.0
    centre = 0.5 * field_um
    tissue = np.hypot(xx - centre, yy - centre) > lumen_radius_um
    rgb = np.where(tissue[..., None], rgb, 250.0)
    rgb = rgb + rng.normal(0.0, 1.0, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def bin_centres(field_um, bin_um):
    g = (np.arange(int(field_um // bin_um)) + 0.5) * bin_um
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


def smooth_field(coords, rng, scale_um=150.0, n_waves=6):
    """A smooth random nuisance surface (residual spatial structure the GP must absorb)."""
    out = np.zeros(len(coords))
    for _ in range(n_waves):
        k = rng.normal(0.0, 2.0 * np.pi / scale_um, 2)
        out += np.cos(coords @ k + rng.uniform(0, 2 * np.pi))
    return out / np.sqrt(n_waves)


def standardise(x):
    return (x - x.mean()) / max(x.std(), 1e-12)


# --------------------------------------------------------------------------- workflow
def run(field_um=400.0, bin_um=16.0, n_null=20, draws=1999, seed=0, mesh_um=None, verbose=True):
    """Run the walkthrough and return a dictionary of the main results."""
    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # 1. H&E preprocessing -------------------------------------------------------------
    image = synthetic_he(field_um, pixel_size_um=2.0, fibre_angle_deg=30.0, lumen_radius_um=50.0, seed=seed)
    hist = histrange.prepare_histology(image, pixel_size_um=2.0)
    say("1  %r, tissue area %.0f um^2" % (hist, hist.tissue_area_um2()))

    # 2. Transport geometry ------------------------------------------------------------
    geo = histrange.build_geometry(hist)
    inside = geo.mask & (geo.beta > 0)
    angle = np.degrees(0.5 * np.angle(np.mean(np.exp(2j * geo.theta[inside]))))
    say("2  %r, mean fibre orientation %.1f deg (truth 30.0)" % (geo, angle))

    # 3. Mesh --------------------------------------------------------------------------
    mesh = histrange.build_mesh(hist, spacing_um=mesh_um)
    say("3  %r, geodesic diameter %.0f um" % (mesh, mesh.geodesic_diameter()))

    # 4. Operator ----------------------------------------------------------------------
    op = transport.assemble(mesh, geo)
    report, _ = op.validate()  # raises OperatorError if any check fails
    say("4  %r, relative asymmetry %.1e, no-flux residual %.1e"
        % (op, report["relative_symmetry"], report["constant_residual"]))

    # 5. Observations and projection ---------------------------------------------------
    coords = bin_centres(field_um, bin_um)
    keep = mesh.contains(coords)  # bins in the lumen or off-tissue are dropped, never projected
    coords = coords[keep]
    footprint = 0.5 * bin_um
    src = projection.source_operator(mesh, coords, radius_um=footprint)
    rec = projection.receiver_operator(mesh, coords, radius_um=footprint)
    n = len(coords)
    say("5  %d of %d bins inside tissue support" % (n, keep.size))

    # 6. Trusted range grid ------------------------------------------------------------
    median_nn = float(bin_um)  # regular lattice
    ranges = trusted_range_grid(mesh.spacing_um, median_nn, mesh.geodesic_diameter())
    grid = master_range_grid()
    ell_true = grid[TRUE_RANGE_INDEX]
    say("6  trusted grid %.1f-%.1f um (%d values); true range %.1f um" % (ranges[0], ranges[-1], ranges.size, ell_true))

    # 7. Simulated molecular data and eligibility ---------------------------------------
    library = rng.gamma(10.0, 300.0, n)
    source_band = np.abs(coords[:, 0] - 0.30 * field_um) < 40.0
    ligand_counts = rng.poisson(np.where(source_band, 4.0, 0.05))
    ligand = molecular.bounded_score(ligand_counts, library)
    load = projection.deposit(src, ligand)

    def exposures_for(operator, receiver_matrix):
        return [projection.interpolate(receiver_matrix, operator.solve(ell, load)) for ell in ranges]

    exposures = exposures_for(op, rec)
    h_true = projection.interpolate(rec, op.solve(ell_true, load))

    def receptor_counts(signal):
        log_mu = np.log(library * 5e-4) + 0.35 * smooth_field(coords, rng) + signal
        return rng.poisson(np.exp(log_mu))

    target_counts = receptor_counts(0.55 * standardise(h_true))
    null_counts = [receptor_counts(0.0) for _ in range(n_null)]
    elig = molecular.eligibility(ligand_counts, target_counts, library)
    say("7  eligible=%s (%d ligand-positive bins, %d competent receivers)"
        % (elig["eligible"], elig["n_positive_source"], elig["n_competent_receivers"]))

    # 8. Spatial association test --------------------------------------------------------
    diameter = mesh.geodesic_diameter()

    def detect(counts, n_draws, seed_offset):
        competent = molecular.eligibility(ligand_counts, counts, library)["receptor_competent"]
        y = molecular.bounded_score(counts, library)[competent]
        baseline = np.column_stack([np.ones(competent.sum()), np.log(library[competent])])
        bounds = spatial.phi_bounds(median_nn, diameter)
        cov = spatial.fit_covariance(y, baseline, coords[competent], bounds)
        fit = spatial.profile(y, baseline, [h[competent] for h in exposures], cov)
        p_boot = spatial.bootstrap_pvalue(fit, draws=n_draws, seed=seed + seed_offset)
        return {
            "Lambda_A": fit["Q0"] - float(fit["Q"].min()),
            "p_boot": p_boot,
            "p_sec": spatial.calibrated_secreted_pvalue(p_boot),
            "profile_optimum_um": float(ranges[int(np.argmin(fit["Q"]))]),
            "fit": fit,
            "cov": cov,
            "competent": competent,
            "y": y,
            "baseline": baseline,
        }

    target = detect(target_counts, draws, 0)
    nulls = [detect(c, draws, k + 1) for k, c in enumerate(null_counts)]
    q = spatial.benjamini_hochberg([target["p_sec"]] + [m["p_sec"] for m in nulls])
    detected = q <= 0.05
    say("8  target: Lambda_A=%.1f, p_boot=%.4g, p_sec=%.4g, q=%.4g -> %s; %d/%d null modules detected"
        % (target["Lambda_A"], target["p_boot"], target["p_sec"], q[0],
           "DETECTED" if detected[0] else "not detected", int(detected[1:].sum()), n_null))
    say("   %r" % target["cov"])

    # 9. T1 range estimation (all in-support bins, receptor zeros included) -----------------
    no_covariates = np.zeros((n, 0))
    t1 = range_estimation.fit_range_profile(target_counts, library, no_covariates, exposures, ranges)
    say("9  T1: Lambda_T1=%.1f, ell_hat=%.1f um (truth %.1f um)" % (t1["Lambda"], t1["ell_hat"], ell_true))

    # 10. Support threshold from near-null modules, then boundary-aware report ----------------
    near_null = [
        range_estimation.fit_range_profile(c, library, no_covariates, exposures, ranges)["Lambda"]
        for c, m, d in zip(null_counts, nulls, detected[1:])
        if m["Lambda_A"] <= 3.0 and not d
    ]
    tau = range_estimation.support_threshold(near_null)
    reports = {
        level: range_estimation.report_range(
            t1["ell_hat"], t1["best_index"], ranges.size, t1["Lambda"], tau, ranges[0], ranges[-1], level=level
        )
        for level in (0.90, 0.95)
    }
    say("10 tau=%.2f from %d near-null modules" % (tau, len(near_null)))
    for level, rep in reports.items():
        say("   %d%%: %s" % (round(100 * level), _describe(rep)))

    # 11. Range robustness (frozen procedure: profile optimum under the fitted covariance) ----
    def optimum(exps, mask=None):
        mask = target["competent"] if mask is None else mask
        fit = spatial.profile(target["y"], target["baseline"], [h[mask] for h in exps], target["cov"])
        return float(ranges[int(np.argmin(fit["Q"]))])

    aniso = {}
    for multiplier in ANISOTROPY_MULTIPLIERS:
        op_m = transport.assemble(mesh, geo, beta_scale=multiplier)
        aniso[multiplier] = optimum(exposures_for(op_m, rec))
    aniso_ok = evidence.anisotropy_robust(list(aniso.values()))

    coarse = histrange.build_mesh(hist, spacing_um=2.0 * mesh.spacing_um)
    op_c = transport.assemble(coarse, geo)
    rec_c = projection.receiver_operator(coarse, coords, radius_um=footprint, boundary_tolerance_um=coarse.spacing_um)
    src_c = projection.source_operator(coarse, coords, radius_um=footprint, boundary_tolerance_um=coarse.spacing_um)
    load_c = projection.deposit(src_c, ligand)
    exps_c = [projection.interpolate(rec_c, op_c.solve(ell, load_c)) for ell in ranges]
    mesh_ok = evidence.mesh_robust(target["profile_optimum_um"], optimum(exps_c))
    say("11 anisotropy x%s optima %s um -> robust=%s"
        % ("/".join("%g" % m for m in aniso), "/".join("%.1f" % v for v in aniso.values()), aniso_ok))
    say("   %.0f um vs %.0f um mesh optimum %.1f vs %.1f um -> robust=%s"
        % (coarse.spacing_um, mesh.spacing_um, optimum(exps_c), target["profile_optimum_um"], mesh_ok))

    # Geometry controls exist for the held-out geometry-gain analysis (Methods 3.18).
    controls = {
        "isotropic": geometry.isotropic(geo),
        "rotated_90": geometry.rotated(geo, 90.0),
        "morphology_mismatch": geometry.morphology_mismatch(geo, seed=seed),
    }
    say("   geometry controls built: %s" % ", ".join(controls))
    say("done in %.1f s" % (time.time() - t0))

    return {
        "n_bins": n,
        "ranges": ranges,
        "ell_true": ell_true,
        "fibre_angle_deg": angle,
        "operator_report": report,
        "eligibility": elig,
        "target_detection": {k: target[k] for k in ("Lambda_A", "p_boot", "p_sec", "profile_optimum_um")},
        "q_values": q,
        "detected": detected,
        "t1": t1,
        "tau": tau,
        "n_near_null": len(near_null),
        "reports": reports,
        "anisotropy_optima": aniso,
        "anisotropy_robust": aniso_ok,
        "mesh_robust": mesh_ok,
    }


def _describe(rep):
    if rep["status"] == "interval":
        return "effective range %.1f-%.1f um (two-sided)" % (rep["lower_um"], rep["upper_um"])
    if rep["status"] == "upper_bound":
        return "effective range < %.1f um (one-sided)" % rep["upper_um"]
    if rep["status"] == "uninformative":
        return "bound exceeds the trusted grid ceiling: not informative"
    return "unresolved (%s)" % rep["reason"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=1999, help="bootstrap draws (manuscript: 9999)")
    parser.add_argument("--null-modules", type=int, default=20)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run(n_null=args.null_modules, draws=args.draws, seed=args.seed)


if __name__ == "__main__":
    main()

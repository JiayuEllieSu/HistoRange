import numpy as np
import pytest
from scipy.spatial.distance import cdist

from histrange import spatial
from histrange.settings import NUGGET_BOUNDS


def _simulate_gp(rng, n=250, phi=60.0, tau=0.3, sigma2=2.0, extent=400.0):
    coords = rng.uniform(0, extent, (n, 2))
    cov = sigma2 * (tau * np.eye(n) + (1 - tau) * spatial.matern32(cdist(coords, coords), phi))
    noise = np.linalg.cholesky(cov) @ rng.standard_normal(n)
    z = rng.standard_normal(n)
    baseline = np.column_stack([np.ones(n), z])
    return coords, baseline, baseline @ np.array([1.0, 0.5]) + noise


def _identity_covariance(n):
    return spatial.Covariance(phi=1.0, nugget=1.0, sigma2=1.0, chol=np.eye(n), converged=True)


def test_matern32_values():
    assert spatial.matern32(np.array([0.0]), 50.0)[0] == 1.0
    d = np.linspace(0, 300, 50)
    k = spatial.matern32(d, 50.0)
    assert np.all(np.diff(k) < 0)
    s = np.sqrt(3.0)
    assert spatial.matern32(np.array([50.0]), 50.0)[0] == pytest.approx((1 + s) * np.exp(-s))


def test_orthonormal_basis_drops_redundant_columns(rng):
    a = rng.standard_normal((40, 2))
    redundant = np.column_stack([a, a[:, 0] + a[:, 1]])
    q, rank, s = spatial.orthonormal_basis(redundant)
    assert rank == 2 and q.shape == (40, 2)
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)
    q0, r0, _ = spatial.orthonormal_basis(np.zeros((5, 2)))
    assert r0 == 0 and q0.shape == (5, 0)


def test_gls_fit_matches_least_squares(rng):
    a = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    fit = spatial.gls_fit(a, y)
    coef, res, *_ = np.linalg.lstsq(a, y, rcond=None)
    np.testing.assert_allclose(fit["coefficients"], coef)
    assert fit["rss"] == pytest.approx(res[0])
    assert fit["full_rank"] and fit["rank"] == 3


def test_gls_fit_rank_deficient_design(rng):
    a = rng.standard_normal((50, 2))
    y = rng.standard_normal(50)
    fit = spatial.gls_fit(np.column_stack([a, 2 * a[:, 0]]), y)
    assert fit["rank"] == 2 and not fit["full_rank"]
    assert fit["rss"] == pytest.approx(spatial.gls_fit(a, y)["rss"])
    empty = spatial.gls_fit(np.zeros((4, 1)), np.ones(4))
    assert empty["rank"] == 0 and empty["rss"] == pytest.approx(4.0)


def test_phi_bounds_follow_methods_3_11():
    assert spatial.phi_bounds(16.0, 3000.0) == (32.0, 1000.0)
    assert spatial.phi_bounds(2.0, 600.0) == (10.0, 200.0)


def test_fit_covariance_dense_recovers_simulated_parameters(rng):
    coords, baseline, y = _simulate_gp(rng, n=300, phi=60.0, tau=0.3)
    cov = spatial.fit_covariance(y, baseline, coords, phi_bounds=(10.0, 400.0))
    assert cov.approximation == "dense" and cov.fit_method == "optimizer"
    assert 30.0 < cov.phi < 120.0
    assert NUGGET_BOUNDS[0] <= cov.nugget <= NUGGET_BOUNDS[1]
    assert 0.5 < cov.sigma2 < 5.0
    assert "Covariance(phi=" in repr(cov)


def test_whitening_matches_the_fitted_covariance(rng):
    coords, baseline, y = _simulate_gp(rng, n=120)
    cov = spatial.fit_covariance(y, baseline, coords, phi_bounds=(10.0, 400.0), method="dense")
    sigma = cov.sigma2 * (cov.nugget * np.eye(120) + (1 - cov.nugget) * spatial.matern32(cdist(coords, coords), cov.phi))
    x = rng.standard_normal(120)
    assert cov.quadratic(x) == pytest.approx(x @ np.linalg.solve(sigma, x), rel=1e-8)


def test_auto_switches_to_vecchia_and_agrees_with_dense(rng):
    coords, baseline, y = _simulate_gp(rng, n=200, phi=50.0, tau=0.3)
    dense = spatial.fit_covariance(y, baseline, coords, (10.0, 300.0), method="dense")
    approx = spatial.fit_covariance(y, baseline, coords, (10.0, 300.0), dense_max_receivers=100)
    assert approx.approximation == "vecchia" and approx.chol is None
    assert approx.phi == pytest.approx(dense.phi, rel=0.15)
    assert approx.nugget == pytest.approx(dense.nugget, abs=0.08)
    x = rng.standard_normal(200)
    assert approx.quadratic(x) == pytest.approx(dense.quadratic(x), rel=0.1)


def test_fit_covariance_rejects_unknown_method(rng):
    coords, baseline, y = _simulate_gp(rng, n=20)
    with pytest.raises(ValueError, match="method"):
        spatial.fit_covariance(y, baseline, coords, (10.0, 100.0), method="exact")


def test_profile_equals_ordinary_least_squares_under_identity_covariance(rng):
    n = 80
    baseline = np.column_stack([np.ones(n), rng.standard_normal(n)])
    exposures = [rng.standard_normal(n) for _ in range(4)]
    y = baseline @ [0.2, 0.1] + 0.8 * exposures[2] + 0.3 * rng.standard_normal(n)
    fit = spatial.profile(y, baseline, exposures, _identity_covariance(n))
    assert fit["Q0"] == pytest.approx(spatial.gls_fit(baseline, y)["rss"])
    for k, h in enumerate(exposures):
        full = spatial.gls_fit(np.column_stack([baseline, h]), y)
        assert fit["Q"][k] == pytest.approx(full["rss"])
        assert fit["alpha"][k] == pytest.approx(full["coefficients"][-1])
    assert np.argmin(fit["Q"]) == 2
    assert np.all(fit["Q"] <= fit["Q0"] + 1e-12)
    assert fit["baseline_rank"] == 2


def test_profile_flags_exposures_inside_the_baseline_span(rng):
    n = 40
    baseline = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    fit = spatial.profile(y, baseline, [3.0 * baseline[:, 1] - 1.0], _identity_covariance(n))
    assert fit["degenerate"][0]
    assert fit["Q"][0] == pytest.approx(fit["Q0"])
    assert fit["alpha"][0] == 0.0


def test_profile_rejects_rank_zero_baseline():
    with pytest.raises(ValueError, match="rank zero"):
        spatial.profile(np.ones(5), np.zeros((5, 1)), [np.arange(5.0)], _identity_covariance(5))


def test_bootstrap_detects_strong_signal(rng):
    n = 100
    baseline = np.ones((n, 1))
    exposures = [rng.standard_normal(n) for _ in range(6)]
    y = 1.5 * exposures[1] + rng.standard_normal(n)
    fit = spatial.profile(y, baseline, exposures, _identity_covariance(n))
    assert spatial.bootstrap_pvalue(fit, draws=199, seed=0) == pytest.approx(1 / 200)


def test_bootstrap_is_calibrated_under_the_null():
    rng = np.random.default_rng(7)
    n, pvals = 60, []
    baseline = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    exposures = [rng.standard_normal(n) for _ in range(8)]
    for rep in range(60):
        y = baseline @ [0.3, -0.2] + rng.standard_normal(n)
        fit = spatial.profile(y, baseline, exposures, _identity_covariance(n))
        pvals.append(spatial.bootstrap_pvalue(fit, draws=199, seed=rep))
    pvals = np.asarray(pvals)
    assert 0.35 < pvals.mean() < 0.65
    assert np.mean(pvals <= 0.05) < 0.15


def test_bootstrap_is_seeded_and_can_be_disabled(rng):
    n = 30
    fit = spatial.profile(rng.standard_normal(n), np.ones((n, 1)), [rng.standard_normal(n)], _identity_covariance(n))
    assert spatial.bootstrap_pvalue(fit, draws=99, seed=4) == spatial.bootstrap_pvalue(fit, draws=99, seed=4)
    assert spatial.bootstrap_pvalue(fit, draws=0) is None
    p = spatial.bootstrap_pvalue(fit, draws=99, seed=4)
    assert 1 / 100 <= p <= 1.0


def test_secreted_calibration_formula_and_cap():
    assert spatial.calibrated_secreted_pvalue(0.01) == pytest.approx(0.6589 * 0.01 ** 0.7998)
    assert spatial.calibrated_secreted_pvalue(1.0) == pytest.approx(0.6589)
    out = spatial.calibrated_secreted_pvalue(np.array([1e-4, 0.05, 0.6, 1.0]))
    assert isinstance(out, np.ndarray) and np.all(np.diff(out) > 0) and out.max() <= 1.0
    assert isinstance(spatial.calibrated_secreted_pvalue(0.2), float)


def test_benjamini_hochberg_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(spatial.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_benjamini_hochberg_is_monotone_and_capped(rng):
    p = rng.uniform(size=200)
    q = spatial.benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-15)
    assert np.all(q >= p) and q.max() <= 1.0
    np.testing.assert_allclose(spatial.benjamini_hochberg([0.9, 0.95]), [0.95, 0.95])

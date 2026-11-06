import numpy as np
import pytest

from histrange import range as range_module
from histrange.settings import SHORT_BAND_BITS, SHORT_UPPER_BITS, master_range_grid


@pytest.fixture(scope="module")
def range_problem():
    """Counts with an injected range on a 1 mm field, as in the frozen T1 smoke test."""
    rng = np.random.default_rng(0)
    n = 4000
    ranges = master_range_grid()
    xy = rng.random((n, 2)) * 1000.0
    d = np.hypot(*(xy - np.array([250.0, 250.0])).T)
    exposures = [np.exp(-d / ell) for ell in ranges]
    umi = rng.gamma(20, 200, n)
    return rng, ranges, exposures, umi


def _counts(rng, umi, exposure=None, amplitude=0.9):
    log_mu = np.log(umi * 1e-4)
    if exposure is not None:
        h = (exposure - exposure.mean()) / exposure.std()
        log_mu = log_mu + amplitude * h
    return rng.poisson(np.exp(log_mu)).astype(float)


def test_t1_recovers_an_injected_range_within_one_bit(range_problem):
    rng, ranges, exposures, umi = range_problem
    k_true = 16  # 64 um
    y = _counts(rng, umi, exposures[k_true])
    fit = range_module.fit_range_profile(y, umi, np.zeros((y.size, 0)), exposures, ranges)
    assert abs(np.log2(fit["ell_hat"] / ranges[k_true])) <= 1.0
    assert fit["Lambda"] > 100
    assert fit["amplitude"] > 0
    assert fit["profile"].shape == (ranges.size,)
    assert fit["best_index"] == int(np.argmax(fit["profile"]))
    assert fit["n"] == y.size


def test_t1_statistic_is_small_under_the_null(range_problem):
    rng, ranges, exposures, umi = range_problem
    y = _counts(rng, umi)
    fit = range_module.fit_range_profile(y, umi, np.zeros((y.size, 1)), exposures, ranges)
    assert fit["Lambda"] < 30
    assert np.all(fit["profile"] >= -1e-6)  # nested models: the exposure term can only help


def test_t1_dispersion_estimate_tracks_overdispersion():
    rng = np.random.default_rng(3)
    n, alpha = 6000, 0.5
    lib = rng.gamma(20, 200, n)
    mu = lib * 1e-3
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(float)
    exposures = [rng.standard_normal(n) for _ in range(3)]
    fit = range_module.fit_range_profile(y, lib, np.zeros((n, 0)), exposures, [16.0, 32.0, 64.0])
    assert fit["dispersion"] == pytest.approx(alpha, rel=0.2)
    poisson = range_module.fit_range_profile(rng.poisson(mu).astype(float), lib, np.zeros((n, 0)), exposures,
                                             [16.0, 32.0, 64.0])
    assert poisson["dispersion"] < 0.05


GRID_MIN, GRID_MAX = 16.0, 512.0


def _report(ell, index=10, n=41, lam=100.0, tau=50.0, lo=GRID_MIN, hi=GRID_MAX, level=0.95):
    return range_module.report_range(ell, index, n, lam, tau, lo, hi, level=level)


@pytest.mark.parametrize("index", [0, 40])
def test_report_refuses_a_profile_maximum_on_the_grid_edge(index):
    out = _report(40.0, index=index)
    assert out["status"] == "unresolved" and "boundary" in out["reason"]


@pytest.mark.parametrize("lam, tau", [(50.0, 50.0), (10.0, 50.0), (100.0, np.nan)])
def test_report_refuses_unsupported_ranges(lam, tau):
    out = _report(40.0, lam=lam, tau=tau)
    assert out["status"] == "unresolved" and "threshold" in out["reason"]


def test_short_range_interval_when_band_is_inside_the_trusted_grid():
    grid = master_range_grid()
    ell = grid[14]  # 53.8 um
    for level in (0.90, 0.95):
        width = SHORT_BAND_BITS[level]
        out = _report(ell, lo=10.0, level=level)
        assert out["status"] == "interval" and out["level"] == level
        assert out["lower_um"] == pytest.approx(ell / 2 ** width)
        assert out["upper_um"] == pytest.approx(ell * 2 ** width)


def test_short_range_lower_edge_on_the_floor_falls_back_to_upper_bound():
    grid = master_range_grid()
    ell = grid[10]  # 38.05 um: 95% lower edge 11.3 um is below the 16 um floor
    out = _report(ell, level=0.95)
    assert out["status"] == "upper_bound"
    assert out["upper_um"] == pytest.approx(107.6, abs=0.05)  # cf. WNT11-FZD7, Phase 6a
    out90 = _report(ell, level=0.90)
    assert out90["status"] == "upper_bound"
    assert out90["upper_um"] == pytest.approx(69.8, abs=0.05)


def test_short_range_uses_the_specimen_floor_not_the_master_floor():
    grid = master_range_grid()
    ell = grid[14]  # 53.8 um; 90% lower edge 21.7 um
    assert _report(ell, lo=16.0, level=0.90)["status"] == "interval"
    assert _report(ell, lo=32.0, level=0.90)["status"] == "upper_bound"


def test_long_range_is_one_sided():
    out = _report(100.0, index=20)
    assert out["status"] == "upper_bound"
    assert out["upper_um"] == pytest.approx(100.0 * 2 ** 1.25)
    out90 = _report(100.0, index=20, level=0.90)
    assert out90["upper_um"] == pytest.approx(200.0)


def test_regime_split_at_64_um_belongs_to_the_long_regime():
    out = _report(64.0, index=16)
    assert out["status"] == "upper_bound"
    assert out["upper_um"] == pytest.approx(64.0 * 2 ** 1.25)


def test_bound_beyond_the_grid_ceiling_is_vacuous():
    out = _report(430.5, index=38)  # cf. FGF18-FGFR2, Phase 6a: < 1024 um is not a claim
    assert out == {"status": "uninformative", "level": 0.95}


def test_report_rejects_unsupported_levels():
    with pytest.raises(ValueError, match="level"):
        _report(40.0, level=0.99)


def test_support_threshold_is_the_99th_percentile_of_finite_values():
    values = np.arange(1, 101, dtype=float)
    assert range_module.support_threshold(values) == pytest.approx(np.quantile(values, 0.99))
    with_nan = np.concatenate([values, [np.nan, np.inf]])
    assert range_module.support_threshold(with_nan) == pytest.approx(np.quantile(values, 0.99))
    assert np.isnan(range_module.support_threshold([]))


@pytest.mark.parametrize(
    "grid_index, lo90, hi90, up90, up95",
    [
        (14, 21.7, 133.7, 98.7, 152.2),  # CCL17-CCR4 (Phase 6a) and CXCL12-CXCR4 (lung), l_hat 53.8 um
        (13, 19.9, 122.6, 90.5, 139.6),  # NRG1-ERBB3 (Phase 6a), l_hat 49.4 um
    ],
)
def test_bands_reproduce_published_frozen_claims(grid_index, lo90, hi90, up90, up95):
    """The frozen rule file (om_boundary_final.csv) has B = 1.3125 bits at 90%, not 1.312."""
    ell = master_range_grid()[grid_index]
    band = _report(ell, index=grid_index, level=0.90)
    assert band["status"] == "interval"
    assert round(band["lower_um"], 1) == lo90
    assert round(band["upper_um"], 1) == hi90
    # The separately calibrated 90% one-sided bound (Methods Eq. 3.30).
    assert round(ell * 2 ** SHORT_UPPER_BITS[0.90], 1) == up90
    one_sided = _report(ell, index=grid_index, level=0.95)  # 95% lower edge is below 16 um
    assert one_sided["status"] == "upper_bound"
    assert round(one_sided["upper_um"], 1) == up95

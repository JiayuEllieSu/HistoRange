import numpy as np
import pytest

from histrange import molecular
from histrange.settings import (
    RECEPTOR_MIN_COMPETENT,
    SOURCE_MIN_POSITIVE,
    SOURCE_MIN_RAW_COUNTS,
    STATE_MIN_FRACTION,
    STATE_MIN_SIZE,
)


def test_cp10k_scales_to_ten_thousand_per_library():
    raw = np.array([[1.0, 3.0], [5.0, 5.0]])
    lib = raw.sum(axis=1)
    out = molecular.cp10k(raw, lib)
    np.testing.assert_allclose(out.sum(axis=1), 1e4)


def test_cp10k_vector_and_zero_library_clamp():
    out = molecular.cp10k([2.0, 0.0], [0.0, 0.0])
    # A zero library size is clamped to one count rather than dividing by zero.
    np.testing.assert_allclose(out, [2e4, 0.0])


def test_complex_count_is_least_abundant_subunit():
    subunits = np.array([[3, 1], [0, 5], [4, 4]])
    np.testing.assert_array_equal(molecular.complex_count(subunits), [1, 0, 4])


def test_complex_count_passes_single_subunit_through():
    np.testing.assert_array_equal(molecular.complex_count([2, 0, 7]), [2, 0, 7])


@pytest.mark.parametrize("bad", [np.zeros((3, 0)), np.zeros((2, 2, 2))])
def test_complex_count_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        molecular.complex_count(bad)


def test_bounded_score_is_half_at_the_median_positive_level(rng):
    raw = rng.poisson(3.0, 501).astype(float)
    lib = np.full(raw.size, 5000.0)
    score = molecular.bounded_score(raw, lib)
    z = np.log1p(molecular.cp10k(raw, lib))
    k = np.median(z[z > 0])
    assert np.all((score >= 0) & (score < 1))
    np.testing.assert_allclose(score, z / (z + k))
    at_median = np.isclose(z, k)
    if at_median.any():
        np.testing.assert_allclose(score[at_median], 0.5)
    assert np.all(score[raw == 0] == 0)


def test_bounded_score_is_monotone_in_abundance():
    raw = np.arange(0, 50, dtype=float)
    score = molecular.bounded_score(raw, np.full(raw.size, 1e4))
    assert np.all(np.diff(score) > 0)


def test_bounded_score_all_zero_input_stays_zero():
    score = molecular.bounded_score(np.zeros(10), np.full(10, 1000.0))
    np.testing.assert_array_equal(score, 0.0)


def test_qc_mask_thresholds_are_inclusive():
    umi = np.array([499, 500, 500, 2000])
    genes = np.array([300, 200, 199, 300])
    np.testing.assert_array_equal(molecular.qc_mask(umi, genes), [False, True, False, True])
    mito = np.array([0.0, 0.20, 0.0, 0.21])
    np.testing.assert_array_equal(molecular.qc_mask(umi, genes, mito), [False, True, False, False])


def test_detected_gene_mask_uses_max_of_ten_and_point_one_percent():
    n = 20_000  # ceil(0.001 * n) = 20 > 10
    counts = np.zeros((n, 3))
    counts[:19, 0] = 1
    counts[:20, 1] = 1
    counts[:, 2] = 1
    np.testing.assert_array_equal(molecular.detected_gene_mask(counts), [False, True, True])


def test_detected_gene_mask_floor_of_ten_for_small_specimens():
    counts = np.zeros((500, 2))
    counts[:9, 0] = 1
    counts[:10, 1] = 1
    np.testing.assert_array_equal(molecular.detected_gene_mask(counts), [False, True])


def test_detected_gene_mask_requires_matrix():
    with pytest.raises(ValueError):
        molecular.detected_gene_mask(np.zeros(5))


def _eligibility_inputs(n_source=SOURCE_MIN_POSITIVE, n_receptor=RECEPTOR_MIN_COMPETENT, source_count=2):
    n = 200
    lig = np.zeros(n)
    lig[:n_source] = source_count
    rec = np.zeros(n)
    rec[n - n_receptor:] = 1
    lib = np.full(n, 2000.0)  # 1 raw count = 5 CP10K, which passes the CP10K >= 1 rule
    return lig, rec, lib


def test_eligibility_passes_exactly_at_frozen_thresholds():
    lig, rec, lib = _eligibility_inputs()
    out = molecular.eligibility(lig, rec, lib)
    assert out["n_positive_source"] == SOURCE_MIN_POSITIVE
    assert out["total_raw_source"] >= SOURCE_MIN_RAW_COUNTS
    assert out["n_competent_receivers"] == RECEPTOR_MIN_COMPETENT
    assert out["eligible"] is True
    assert out["max_state_fraction"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_source": SOURCE_MIN_POSITIVE - 1},
        {"n_receptor": RECEPTOR_MIN_COMPETENT - 1},
        {"source_count": 1},  # 20 positive observations but only 20 < 25 raw counts
    ],
)
def test_eligibility_fails_one_below_each_threshold(kwargs):
    lig, rec, lib = _eligibility_inputs(**kwargs)
    assert molecular.eligibility(lig, rec, lib)["eligible"] is False


def test_eligibility_requires_cp10k_of_at_least_one():
    lig, rec, lib = _eligibility_inputs()
    deep = np.full(lib.size, 20_000.0)  # 1 raw count = 0.5 CP10K
    out = molecular.eligibility(lig, rec, deep)
    assert out["n_competent_receivers"] == 0
    assert out["eligible"] is False


def test_eligibility_requires_every_subunit():
    n = 100
    lib = np.full(n, 2000.0)
    lig = np.zeros(n)
    lig[:40] = 2
    receptor = np.zeros((n, 2))
    receptor[:50, 0] = 1
    receptor[25:75, 1] = 1  # only rows 25..49 carry both subunits
    out = molecular.eligibility(lig, receptor, lib)
    assert out["n_competent_receivers"] == 25
    np.testing.assert_array_equal(np.flatnonzero(out["receptor_competent"]), np.arange(25, 50))


def test_eligibility_total_raw_counts_aggregate_across_subunits():
    n = 60
    lib = np.full(n, 2000.0)
    ligand = np.zeros((n, 2))
    ligand[:20] = [1, 1]  # 20 positive observations, 40 aggregate raw counts
    rec = np.zeros(n)
    rec[-30:] = 1
    out = molecular.eligibility(ligand, rec, lib)
    assert out["total_raw_source"] == 40
    assert out["eligible"] is True


def test_eligibility_receiver_state_rule():
    lig, rec, lib = _eligibility_inputs(n_receptor=40)
    n = rec.size
    states = np.array(["a"] * (n - 100) + ["b"] * 100)
    # 40 competent receivers all sit in state "b" (size 100): fraction 0.40.
    out = molecular.eligibility(lig, rec, lib, states=states)
    assert out["max_state_fraction"] == pytest.approx(0.40)
    assert out["eligible"] is True

    # Spread competence so no state of >= 30 observations reaches 10%.
    rec2 = np.zeros(n)
    rec2[np.arange(0, n, 11)[:RECEPTOR_MIN_COMPETENT]] = 1
    small = np.array(["s%d" % (i // (STATE_MIN_SIZE - 1)) for i in range(n)])  # every state too small
    out2 = molecular.eligibility(lig, rec2, lib, states=small)
    assert out2["max_state_fraction"] == 0.0
    assert out2["eligible"] is False


def test_eligibility_state_fraction_threshold_is_inclusive():
    n = 300
    lib = np.full(n, 2000.0)
    lig = np.zeros(n)
    lig[:30] = 2
    rec = np.zeros(n)
    rec[:30] = 1  # exactly 10% of a single 300-observation state
    states = np.zeros(n, dtype=int)
    out = molecular.eligibility(lig, rec, lib, states=states)
    assert out["max_state_fraction"] == pytest.approx(STATE_MIN_FRACTION)
    assert out["eligible"] is True


def test_eligibility_shape_checks():
    with pytest.raises(ValueError):
        molecular.eligibility(np.zeros(5), np.zeros(6), np.ones(5))
    with pytest.raises(ValueError):
        molecular.eligibility(np.zeros(5), np.zeros(5), np.ones(5), states=np.zeros(4))

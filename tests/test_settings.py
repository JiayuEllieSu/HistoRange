import numpy as np
import pytest

from histrange import settings
from histrange.settings import master_range_grid, trusted_range_grid


def test_master_grid_is_log2_uniform_16_to_512_with_41_values():
    grid = master_range_grid()
    assert grid.size == 41
    assert grid[0] == pytest.approx(16.0)
    assert grid[-1] == pytest.approx(512.0)
    steps = np.log2(grid[1:] / grid[:-1])
    np.testing.assert_allclose(steps, 1.0 / settings.RANGE_INTERVALS_PER_OCTAVE, rtol=1e-12)


def test_master_grid_hits_every_octave_exactly():
    grid = master_range_grid()
    for octave in (16.0, 32.0, 64.0, 128.0, 256.0, 512.0):
        assert np.any(np.isclose(grid, octave, rtol=0, atol=1e-9))


@pytest.mark.parametrize(
    "mesh_um, nn_um, diameter_um, expected_low, expected_high",
    [
        (4.0, 16.0, 4000.0, 16.0, 512.0),  # every floor term ties at 16, ceiling is 512
        (8.0, 16.0, 4000.0, 32.0, 512.0),  # 4h binds the floor
        (4.0, 100.0, 4000.0, 50.0, 512.0),  # 0.5 d_NN binds the floor
        (4.0, 16.0, 1000.0, 16.0, 250.0),  # 0.25 D_geo binds the ceiling
    ],
)
def test_trusted_grid_bounds_follow_methods_3_9(mesh_um, nn_um, diameter_um, expected_low, expected_high):
    grid = trusted_range_grid(mesh_um, nn_um, diameter_um)
    assert grid.min() >= expected_low - 1e-9
    assert grid.max() <= expected_high + 1e-9
    master = master_range_grid()
    # The first excluded value on either side must be outside the bounds.
    below = master[master < grid.min() - 1e-9]
    above = master[master > grid.max() + 1e-9]
    if below.size:
        assert below.max() < expected_low
    if above.size:
        assert above.min() > expected_high


def test_trusted_grid_is_subset_of_master_grid():
    grid = trusted_range_grid(8.0, 20.0, 1500.0)
    master = master_range_grid()
    assert all(np.any(np.isclose(master, g)) for g in grid)


def test_trusted_grid_requires_nine_values():
    # 16 um mesh -> floor 64 um; 0.25 * 400 = 100 um ceiling -> only 64..100 survive.
    with pytest.raises(ValueError, match="nine"):
        trusted_range_grid(16.0, 10.0, 400.0)


def test_trusted_grid_accepts_exactly_nine_values():
    master = master_range_grid()
    # Floor at 16 um, ceiling chosen so exactly master[0..8] = 16..32 um survive.
    grid = trusted_range_grid(4.0, 10.0, 4.0 * master[8])
    assert grid.size == settings.MIN_RANGE_POINTS


def test_beta_max_gives_three_to_one_conductance_ratio():
    assert np.exp(2.0 * settings.BETA_MAX) == pytest.approx(3.0)


def test_reporting_constants_match_methods_3_15():
    assert settings.SHORT_RANGE_SPLIT_UM == 64.0
    # Manuscript prints 1.312; the frozen calibration (om_boundary_final.csv) is 1.3125 = 21/16.
    assert settings.SHORT_BAND_BITS == {0.90: 1.3125, 0.95: 1.750}
    assert settings.SHORT_UPPER_BITS == {0.90: 0.875, 0.95: 1.500}
    assert settings.LONG_UPPER_BITS == {0.90: 1.000, 0.95: 1.250}


def test_detection_calibration_constants_match_methods_3_12():
    assert settings.SECRETED_CALIBRATION_C == pytest.approx(0.6589)
    assert settings.SECRETED_CALIBRATION_GAMMA == pytest.approx(0.7998)
    assert settings.BOOTSTRAP_DRAWS == 9999


def test_stain_vectors_match_methods_3_5():
    np.testing.assert_allclose(settings.H_STAIN, [0.650, 0.704, 0.286])
    np.testing.assert_allclose(settings.E_STAIN, [0.072, 0.990, 0.105])

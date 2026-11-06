import numpy as np
import pytest

from histrange import evidence


def test_checkerboard_folds_alternate_between_adjacent_blocks():
    g = (np.arange(64) + 0.5) / 64 * 800.0
    xx, yy = np.meshgrid(g, g)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    folds = evidence.checkerboard_folds(coords, blocks=8)
    assert set(np.unique(folds)) == {0, 1}
    assert folds.mean() == pytest.approx(0.5)
    grid = folds.reshape(64, 64)
    # 64 points per axis / 8 blocks = 8 points per block.
    assert grid[0, 0] != grid[0, 8] and grid[0, 0] != grid[8, 0] and grid[0, 0] == grid[8, 8]


def test_checkerboard_folds_handle_degenerate_extent():
    coords = np.column_stack([np.zeros(10), np.linspace(0, 1, 10)])
    folds = evidence.checkerboard_folds(coords)
    assert folds.shape == (10,) and set(np.unique(folds)) <= {0, 1}


def test_partial_r2():
    assert evidence.partial_r2(10.0, 8.0) == pytest.approx(0.2)
    assert evidence.partial_r2(10.0, 12.0) == pytest.approx(-0.2)
    assert np.isnan(evidence.partial_r2(0.0, 1.0))


@pytest.mark.parametrize(
    "folds, expected",
    [([0.05, 0.02], True), ([0.01, 0.01], True), ([0.05, 0.009], False), ([0.5], False), ([], False)],
)
def test_specificity_requires_both_folds(folds, expected):
    assert evidence.specificity_supported(folds) is expected


def test_geometry_gain_sign_and_scaling():
    assert evidence.geometry_gain(110.0, 100.0, 50) == pytest.approx(0.2)
    assert evidence.geometry_gain(100.0, 110.0, 50) < 0
    assert evidence.geometry_gain(1.0, 0.0, 0) == 1.0  # n clamped to 1


def test_anisotropy_robustness_span():
    assert evidence.anisotropy_robust([40.0, 60.0, 80.0]) is True  # exactly 1 bit
    assert evidence.anisotropy_robust([40.0, 81.0]) is False


def test_response_robustness_is_max_shift_from_reference():
    # Matches the frozen pipeline (p8_10_robustness.py): max |log2(ell_K / ell_ref)| <= 1 bit.
    assert evidence.response_robust(64.0, [32.0, 128.0]) is True
    assert evidence.response_robust(64.0, [30.0]) is False


def test_mesh_robustness_half_bit():
    assert evidence.mesh_robust(64.0, 64.0 * 2 ** 0.4) is True
    assert evidence.mesh_robust(64.0, 64.0 * 2 ** 0.6) is False
    assert evidence.mesh_robust(64.0, 64.0 / 2 ** 0.6) is False


def test_grid_aligned_thresholds_are_not_flipped_by_roundoff():
    # Methods 3.17: a tolerance stops machine-precision roundoff at an exact grid-aligned
    # boundary from changing a verdict. On the 1/8-octave grid, 4 steps is exactly 0.5 bit
    # and 8 steps exactly 1 bit.
    from histrange.settings import master_range_grid

    g = master_range_grid()
    for i in range(g.size - 8):
        assert evidence.anisotropy_robust([g[i], g[i + 4], g[i + 8]]), i
        assert evidence.response_robust(g[i], [g[i + 8]]), i
        assert evidence.response_robust(g[i + 8], [g[i]]), i
    for i in range(g.size - 4):
        assert evidence.mesh_robust(g[i], g[i + 4]), i
        assert evidence.mesh_robust(g[i + 4], g[i]), i
    # One grid step beyond each limit still fails.
    for i in range(g.size - 9):
        assert not evidence.anisotropy_robust([g[i], g[i + 9]])
        assert not evidence.response_robust(g[i], [g[i + 9]])
    for i in range(g.size - 5):
        assert not evidence.mesh_robust(g[i], g[i + 5])

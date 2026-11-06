import numpy as np
import pytest

from histrange import contact


def _square_lattice(n=7, spacing=1.0):
    g = np.arange(n) * spacing
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _hex_lattice(n=7, spacing=1.0):
    pts = []
    for row in range(n):
        for col in range(n):
            pts.append([col * spacing + 0.5 * spacing * (row % 2), row * spacing * np.sqrt(3) / 2])
    return np.asarray(pts)


def _degrees(op):
    return np.diff(op.indptr)


def test_square_lattice_includes_axial_and_diagonal_neighbours():
    pts = _square_lattice()
    op = contact.contact_operator(pts, 1.5)
    centre = 3 * 7 + 3
    assert _degrees(op)[centre] == 8  # sqrt(2) d_NN < 1.5 d_NN < 2 d_NN


def test_hex_lattice_includes_first_ring_only():
    pts = _hex_lattice()
    op = contact.contact_operator(pts, 1.5)
    centre = 3 * 7 + 3
    assert _degrees(op)[centre] == 6  # sqrt(3) d_NN > 1.5 d_NN excludes the second ring


def test_operator_is_row_stochastic_without_self_links():
    pts = _square_lattice(5)
    op = contact.contact_operator(pts, 1.5)
    np.testing.assert_allclose(np.asarray(op.sum(axis=1)).ravel(), 1.0)
    assert np.all(op.diagonal() == 0)


def test_isolated_receivers_keep_a_zero_row():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
    op = contact.contact_operator(pts, 1.5)
    np.testing.assert_allclose(np.asarray(op.sum(axis=1)).ravel(), [1.0, 1.0, 0.0])


def test_neighbourhoods_do_not_bridge_tissue_components():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    components = np.array([0, 0, 1, 1])
    op = contact.contact_operator(pts, 1.5, components=components).toarray()
    assert op[1, 2] == 0 and op[2, 1] == 0
    assert op[0, 1] == 1 and op[3, 2] == 1


def test_radius_boundary_is_inclusive():
    pts = np.array([[0.0, 0.0], [1.5, 0.0]])
    op = contact.contact_operator(pts, 1.5)
    assert op[0, 1] == pytest.approx(1.0)


def test_contact_exposure_is_mean_neighbour_ligand():
    pts = _square_lattice(3)
    ligand = np.arange(9, dtype=float)
    exposure = contact.contact_exposure(pts, ligand, median_nn=1.0)
    # The centre (index 4) sees all 8 others; mean of 0..8 without 4 is 4.
    assert exposure[4] == pytest.approx(4.0)
    # Corner 0 sees indices 1, 3 and 4.
    assert exposure[0] == pytest.approx((1 + 3 + 4) / 3)


def test_contact_exposure_scales_with_median_spacing():
    pts = _square_lattice(4, spacing=16.0)
    ligand = np.random.default_rng(1).uniform(size=16)
    a = contact.contact_exposure(pts, ligand, median_nn=16.0)
    b = contact.contact_exposure(pts / 16.0, ligand, median_nn=1.0)
    np.testing.assert_allclose(a, b)

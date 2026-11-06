import warnings

import numpy as np
import pytest
from synthetic import make_histology

from histrange import mesh as mesh_module
from histrange.mesh import OutsideTissueError


def _signed_areas(m):
    p0, p1, p2 = (m.nodes[m.triangles[:, k]] for k in range(3))
    return 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def test_rectangle_mesh_counts_area_and_orientation(square_mesh):
    # 100 x 80 px mask at 1 um/px; lattice nodes at 0, 4, ..., 96 by 0, 4, ..., 76.
    assert square_mesh.n_nodes == 25 * 20
    assert square_mesh.n_triangles == 2 * 24 * 19
    assert square_mesh.areas.sum() == pytest.approx(96.0 * 76.0)
    assert square_mesh.node_areas.sum() == pytest.approx(96.0 * 76.0)
    assert np.all(_signed_areas(square_mesh) > 0)  # every triangle is counter-clockwise
    assert square_mesh.connected
    assert square_mesh.spacing_um == 4.0
    assert "500 nodes" in repr(square_mesh)


def test_alternating_diagonals_balance_directions(square_mesh):
    e = square_mesh.edges()
    d = square_mesh.nodes[e[:, 1]] - square_mesh.nodes[e[:, 0]]
    diagonal = (d[:, 0] != 0) & (d[:, 1] != 0)
    slope_sign = np.sign(d[diagonal, 0] * d[diagonal, 1])
    # Neither diagonal direction dominates, so the lattice carries no directional bias.
    assert abs(np.sum(slope_sign == 1) - np.sum(slope_sign == -1)) <= 1


def test_median_nearest_neighbour_is_mesh_spacing(square_mesh):
    assert square_mesh.median_nearest_neighbour() == pytest.approx(4.0)


def test_geodesic_diameter_is_between_euclidean_and_manhattan(square_mesh):
    diameter = square_mesh.geodesic_diameter()
    assert np.hypot(96.0, 76.0) - 1e-9 <= diameter <= 96.0 + 76.0 + 1e-9
    assert square_mesh.geodesic_diameter() == diameter  # cached


def test_disconnected_tissue_warns_and_keeps_all_components(two_blob_histology):
    with pytest.warns(UserWarning, match="2 disconnected components"):
        m = mesh_module.build(two_blob_histology, spacing_um=4.0)
    assert m.n_components == 2
    assert not m.connected
    areas = np.sort(m.component_areas())
    assert areas[1] > areas[0] > 0
    assert m.component_diameters().size == 2


def test_keep_largest_component_drops_the_smaller_island(two_blob_histology):
    with pytest.warns(UserWarning, match="discards"):
        m = mesh_module.build(two_blob_histology, spacing_um=4.0, keep_components="largest")
    assert m.n_components == 1
    assert m.nodes[:, 0].max() < 70  # only the left island survives
    assert m.triangles.max() == m.n_nodes - 1


def test_keep_components_validation(square_histology):
    with pytest.raises(ValueError, match="keep_components"):
        mesh_module.build(square_histology, keep_components="biggest")


def test_mesh_ladder_falls_back_when_node_budget_is_exceeded(monkeypatch):
    hist = make_histology(np.ones((200, 200), bool))
    monkeypatch.setattr(mesh_module, "MAX_MESH_NODES", 1000)
    m = mesh_module.build(hist)
    # 4 um would need 50 x 50 = 2500 nodes, so the ladder steps to 8 um (25 x 25 = 625).
    assert m.spacing_um == 8.0
    assert m.n_nodes <= 1000


def test_mesh_ladder_raises_when_every_rung_fails(monkeypatch):
    hist = make_histology(np.ones((200, 200), bool))
    monkeypatch.setattr(mesh_module, "MAX_MESH_NODES", 10)
    with pytest.raises(ValueError, match="node budget"):
        mesh_module.build(hist)


def test_explicit_spacing_too_coarse_for_tissue():
    hist = make_histology(np.ones((10, 10), bool))
    with pytest.raises(ValueError, match="spacing 16.0 um"):
        mesh_module.build(hist, spacing_um=16.0)


def test_locate_returns_barycentric_weights_that_reconstruct_points(square_mesh, rng):
    pts = rng.uniform([1, 1], [95, 75], (300, 2))
    tri, w, report = square_mesh.locate(pts)
    assert np.all(tri >= 0)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= 0)
    corners = square_mesh.nodes[square_mesh.triangles[tri]]
    recon = np.einsum("nk,nkd->nd", w, corners)
    np.testing.assert_allclose(recon, pts, atol=1e-9)
    assert report["n_inside"] == 300 and report["n_outside"] == 0


def test_locate_mesh_nodes_and_single_point(square_mesh):
    tri, w, _ = square_mesh.locate(square_mesh.nodes[:5])
    assert np.all(tri >= 0)
    tri1, w1, _ = square_mesh.locate([40.0, 40.0])
    assert tri1.shape == (1,) and w1.shape == (1, 3)
    tri0, w0, report0 = square_mesh.locate(np.zeros((0, 2)))
    assert tri0.size == 0 and report0["n_points"] == 0


def test_points_outside_support_are_rejected_not_projected(square_mesh):
    pts = np.array([[10.0, 10.0], [200.0, 10.0], [50.0, -3.0]])
    tri, _, report = square_mesh.locate(pts)
    np.testing.assert_array_equal(tri >= 0, [True, False, False])
    np.testing.assert_array_equal(report["outside_indices"], [1, 2])
    np.testing.assert_array_equal(square_mesh.contains(pts), [True, False, False])
    with pytest.raises(OutsideTissueError, match="2 of 3 receivers"):
        square_mesh.require_inside(pts, label="receivers")


def test_boundary_tolerance_snaps_near_misses_only(square_mesh):
    pts = np.array([[50.0, -1.0], [50.0, -10.0]])
    tri, w, report = square_mesh.locate(pts, boundary_tolerance_um=2.0)
    np.testing.assert_array_equal(tri >= 0, [True, False])
    assert report["n_snapped"] == 1
    # Clamped-and-renormalised barycentrics land on the boundary but not at the orthogonal
    # foot, so the reported distance is a slight (conservative) overestimate of the true 1 um.
    assert 1.0 <= report["max_snap_distance_um"] <= 1.5
    np.testing.assert_allclose(w[0].sum(), 1.0)
    snapped = w[0] @ square_mesh.nodes[square_mesh.triangles[tri[0]]]
    assert snapped[1] == pytest.approx(0.0, abs=1e-9)


def test_points_inside_a_hole_are_outside_the_mesh():
    mask = np.ones((100, 100), bool)
    mask[30:70, 30:70] = False
    m = mesh_module.build(make_histology(mask), spacing_um=4.0)
    assert not m.contains([[50.0, 50.0]])[0]
    assert m.contains([[10.0, 10.0]])[0]


def test_component_of_labels_points_by_island(two_blob_histology):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mesh_module.build(two_blob_histology, spacing_um=4.0)
    labels = m.component_of([[20.0, 20.0], [100.0, 20.0], [80.0, 50.0]])
    assert labels[0] >= 0 and labels[1] >= 0 and labels[0] != labels[1]
    assert labels[2] == -1


def test_mesh_respects_pixel_size():
    hist = make_histology(np.ones((50, 50), bool), pixel_size_um=2.0)  # 98 x 98 um of tissue
    m = mesh_module.build(hist, spacing_um=8.0)
    assert m.nodes.max() == pytest.approx(96.0)
    assert m.areas.sum() == pytest.approx(96.0 * 96.0)

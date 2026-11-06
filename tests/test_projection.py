import warnings

import numpy as np
import pytest

from histrange import mesh as mesh_module
from histrange import projection
from histrange.mesh import OutsideTissueError


def test_point_operators_are_unit_mass_and_transposes(square_mesh, rng):
    pts = rng.uniform([2, 2], [94, 74], (50, 2))
    src = projection.source_operator(square_mesh, pts)
    rec = projection.receiver_operator(square_mesh, pts)
    assert src.shape == (square_mesh.n_nodes, 50)
    assert rec.shape == (50, square_mesh.n_nodes)
    np.testing.assert_allclose(np.asarray(src.sum(axis=0)).ravel(), 1.0)
    np.testing.assert_allclose(np.asarray(rec.sum(axis=1)).ravel(), 1.0)
    np.testing.assert_allclose(src.toarray(), rec.T.toarray())


def test_receiver_interpolation_is_exact_for_linear_fields(square_mesh, rng):
    pts = rng.uniform([0, 0], [96, 76], (200, 2))
    field = 2.0 * square_mesh.nodes[:, 0] - 3.0 * square_mesh.nodes[:, 1] + 1.0
    rec = projection.receiver_operator(square_mesh, pts)
    np.testing.assert_allclose(projection.interpolate(rec, field), 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 1.0)


def test_deposit_conserves_total_ligand(square_mesh, rng):
    pts = rng.uniform([2, 2], [94, 74], (30, 2))
    ligand = rng.uniform(0, 1, 30)
    for radius in (None, 6.0):
        src = projection.source_operator(square_mesh, pts, radius_um=radius)
        assert projection.deposit(src, ligand).sum() == pytest.approx(ligand.sum())


def test_footprint_operators_cover_the_capture_disk(square_mesh):
    pts = np.array([[48.0, 40.0], [20.0, 20.0]])
    radius = 9.0
    src = projection.source_operator(square_mesh, pts, radius_um=radius).tocsc()
    rec = projection.receiver_operator(square_mesh, pts, radius_um=radius).tocsr()
    for k in range(2):
        nodes = src[:, k].nonzero()[0]
        dist = np.linalg.norm(square_mesh.nodes[nodes] - pts[k], axis=1)
        assert np.all(dist <= radius + 1e-9)
        expected = np.flatnonzero(np.linalg.norm(square_mesh.nodes - pts[k], axis=1) <= radius)
        np.testing.assert_array_equal(np.sort(nodes), expected)
        weights = src[nodes, k].toarray().ravel()
        np.testing.assert_allclose(weights, square_mesh.node_areas[nodes] / square_mesh.node_areas[nodes].sum())
    np.testing.assert_allclose(np.asarray(rec.sum(axis=1)).ravel(), 1.0)


def test_footprint_falls_back_to_point_weights_when_no_node_is_covered(square_mesh):
    tri, _, _ = square_mesh.locate([[49.0, 41.0]])
    centre = square_mesh.centroids[tri[0]]
    src_tiny = projection.source_operator(square_mesh, [centre], radius_um=0.1)
    src_point = projection.source_operator(square_mesh, [centre])
    np.testing.assert_allclose(src_tiny.toarray(), src_point.toarray())


def test_footprint_never_crosses_into_another_tissue_component(two_blob_histology):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mesh_module.build(two_blob_histology, spacing_um=4.0)
    pt = np.array([[60.0, 25.0]])  # right edge of the left island, 30 um from the right island
    src = projection.source_operator(m, pt, radius_um=45.0)
    nodes = src.nonzero()[0]
    own = m.component_of(pt)[0]
    assert np.all(m.node_components[nodes] == own)
    assert np.all(m.nodes[nodes, 0] < 70)


def test_precomputed_location_matches_direct_lookup(square_mesh, rng):
    pts = rng.uniform([2, 2], [94, 74], (20, 2))
    tri, w, _ = square_mesh.locate(pts)
    direct = projection.receiver_operator(square_mesh, pts)
    cached = projection.receiver_operator(square_mesh, pts, location=(tri, w))
    np.testing.assert_allclose(direct.toarray(), cached.toarray())


def test_precomputed_location_shape_and_outside_checks(square_mesh):
    pts = np.array([[10.0, 10.0], [20.0, 20.0]])
    tri, w, _ = square_mesh.locate(pts)
    with pytest.raises(ValueError, match="does not match"):
        projection.source_operator(square_mesh, pts, location=(tri[:1], w[:1]))
    tri_bad = tri.copy()
    tri_bad[1] = -1
    with pytest.raises(OutsideTissueError):
        projection.source_operator(square_mesh, pts, location=(tri_bad, w))


def test_observations_outside_tissue_are_refused(square_mesh):
    with pytest.raises(OutsideTissueError, match="source observations"):
        projection.source_operator(square_mesh, [[10.0, 10.0], [500.0, 10.0]])
    with pytest.raises(OutsideTissueError, match="receiver observations"):
        projection.receiver_operator(square_mesh, [[500.0, 10.0]])


def test_nodal_density_divides_by_node_area():
    areas = np.array([2.0, 4.0, 0.0])
    np.testing.assert_allclose(projection.nodal_density([2.0, 2.0, 5.0], areas), [1.0, 0.5, 5.0])

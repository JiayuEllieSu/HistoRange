import numpy as np
import pytest
from scipy.spatial.distance import cdist

from histrange import vecchia
from histrange.spatial import matern32


def _covariance(points, phi, tau):
    return tau * np.eye(len(points)) + (1.0 - tau) * matern32(cdist(points, points), phi)


@pytest.fixture
def points(rng):
    return rng.uniform(0, 300, (60, 2))


def test_maximin_order_is_a_permutation_starting_near_the_centroid(points):
    order = vecchia.maximin_order(points)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(points)))
    centroid = points.mean(axis=0)
    assert order[0] == np.argmin(np.sum((points - centroid) ** 2, axis=1))


def test_maximin_order_spreads_early_points(points):
    order = vecchia.maximin_order(points)
    # Each new point is the one farthest from everything already chosen.
    for k in range(1, 10):
        chosen = points[order[:k]]
        gaps = cdist(points, chosen).min(axis=1)
        gaps[order[:k]] = -np.inf
        assert order[k] == np.argmax(gaps)


def test_maximin_order_empty():
    assert vecchia.maximin_order(np.zeros((0, 2))).size == 0


def test_neighbour_sets_are_nearest_previous_points(points):
    order = vecchia.maximin_order(points)
    sets = vecchia.neighbour_sets(points, order, 5)
    assert sets[0].size == 0
    for k in range(1, len(points)):
        prefix = order[:k]
        assert set(sets[k]) <= set(prefix)
        assert sets[k].size == min(k, 5)
        d = np.linalg.norm(points[prefix] - points[order[k]], axis=1)
        chosen = np.linalg.norm(points[sets[k]] - points[order[k]], axis=1)
        assert chosen.max() <= np.sort(d)[sets[k].size - 1] + 1e-12


def test_plan_tail_matches_neighbour_sets(points):
    p = vecchia.plan(points, n_neighbours=6)
    sets = vecchia.neighbour_sets(points, p.order, 6)
    assert p.n == len(points)
    for row, k in enumerate(range(7, len(points))):
        assert set(p.tail_index[row]) == set(sets[k])


@pytest.mark.parametrize("tau", [0.05, 0.3, 0.9])
def test_full_conditioning_is_exact(points, tau):
    phi = 80.0
    n = len(points)
    factor = vecchia.build_factor(points, phi, tau, matern32, n_neighbours=n)
    sigma = _covariance(points, phi, tau)
    u = factor.matrix.toarray()
    np.testing.assert_allclose(u @ u.T, np.linalg.inv(sigma), rtol=1e-7, atol=1e-8)
    assert factor.logdet() == pytest.approx(np.linalg.slogdet(sigma)[1], rel=1e-9)
    assert factor.shape == (n, n)


def test_apply_transpose_whitens_vectors_and_matrices(points, rng):
    factor = vecchia.build_factor(points, 50.0, 0.2, matern32, n_neighbours=len(points))
    x = rng.standard_normal(len(points))
    m = rng.standard_normal((len(points), 3))
    u = factor.matrix.toarray()
    np.testing.assert_allclose(factor.apply_transpose(x), u.T @ x)
    np.testing.assert_allclose(factor.apply_transpose(m), u.T @ m)


def test_accuracy_improves_with_more_neighbours(rng):
    pts = rng.uniform(0, 400, (150, 2))
    phi, tau = 120.0, 0.1
    sigma = _covariance(pts, phi, tau)
    exact_logdet = np.linalg.slogdet(sigma)[1]
    errors = []
    for m in (2, 8, 30):
        f = vecchia.build_factor(pts, phi, tau, matern32, n_neighbours=m)
        errors.append(abs(f.logdet() - exact_logdet))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] / abs(exact_logdet) < 0.02


def test_build_factor_accepts_a_reusable_plan(points):
    p = vecchia.plan(points, n_neighbours=10)
    a = vecchia.build_factor(p, 60.0, 0.25, matern32)
    b = vecchia.build_factor(points, 60.0, 0.25, matern32, n_neighbours=10)
    np.testing.assert_allclose(a.matrix.toarray(), b.matrix.toarray())
    assert a.n_neighbours == 10

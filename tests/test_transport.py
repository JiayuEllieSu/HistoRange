import numpy as np
import pytest
from scipy.sparse import csr_matrix
from synthetic import uniform_geometry

from histrange import geometry, projection, transport
from histrange.transport import OperatorError


@pytest.fixture
def iso_operator(square_mesh):
    geo = uniform_geometry(np.ones((80, 100), bool), beta=0.0)
    return transport.assemble(square_mesh, geo)


@pytest.fixture
def aniso_operator(square_mesh):
    geo = uniform_geometry(np.ones((80, 100), bool), theta=0.7, beta=0.5 * np.log(3.0))
    return transport.assemble(square_mesh, geo)


def _point_load(mesh, xy):
    return projection.deposit(projection.source_operator(mesh, [xy]), [1.0])


def _second_moments(mesh, u, centre):
    w = mesh.node_areas * u
    d = mesh.nodes - np.asarray(centre)
    return np.sum(w * d[:, 0] ** 2), np.sum(w * d[:, 1] ** 2)


@pytest.mark.parametrize("fixture", ["iso_operator", "aniso_operator"])
def test_assembled_operator_passes_validation(fixture, request):
    op = request.getfixturevalue(fixture)
    report, failures = op.validate()
    assert failures == []
    assert report["valid"] is True
    assert report["relative_symmetry"] < 1e-12
    assert report["constant_residual"] < 1e-10  # no-flux: constants are in the null space
    assert report["min_lumped_mass"] > 0
    assert report["min_generalized_eigenvalue"] > -1e-8 * report["scale"]


def test_mass_matrices_integrate_the_domain_area(iso_operator, square_mesh):
    ones = np.ones(square_mesh.n_nodes)
    area = square_mesh.areas.sum()
    assert ones @ (iso_operator.mass @ ones) == pytest.approx(area)
    assert iso_operator.lumped.sum() == pytest.approx(area)


def test_constant_source_gives_constant_field(aniso_operator, square_mesh):
    ell = 25.0
    load = aniso_operator.mass @ np.full(square_mesh.n_nodes, 2.0)
    u = aniso_operator.solve(ell, load)
    # With no-flux boundaries the screened equation maps a uniform source c to ell^2 c.
    np.testing.assert_allclose(u, 2.0 * ell * ell, rtol=1e-9)


@pytest.mark.parametrize("ell", [16.0, 40.0, 128.0])
def test_screened_transport_conserves_mass(aniso_operator, square_mesh, ell):
    load = _point_load(square_mesh, (30.0, 41.0))
    u = aniso_operator.solve(ell, load)
    total = np.ones(square_mesh.n_nodes) @ (aniso_operator.mass @ u)
    # 1^T K = 0 exactly, so 1^T M u = ell^2 * (total load).
    assert total == pytest.approx(ell * ell * load.sum(), rel=1e-9)
    assert aniso_operator.residual(ell, u, load) < 1e-10


def test_point_source_field_decays_with_distance(large_square_mesh):
    geo = uniform_geometry(np.ones((200, 200), bool), beta=0.0, pixel_size_um=2.0)
    op = transport.assemble(large_square_mesh, geo)
    centre = (200.0, 200.0)
    u = op.solve(20.0, _point_load(large_square_mesh, centre))
    on_axis = np.abs(large_square_mesh.nodes[:, 1] - centre[1]) < 1e-9
    right = on_axis & (large_square_mesh.nodes[:, 0] >= centre[0])
    profile = u[right][np.argsort(large_square_mesh.nodes[right, 0])]
    assert np.all(np.diff(profile) < 0)
    mx, my = _second_moments(large_square_mesh, u, centre)
    assert mx / my == pytest.approx(1.0, abs=0.03)  # isotropic: no lattice direction is preferred


def test_larger_range_spreads_the_field_further(iso_operator, square_mesh):
    load = _point_load(square_mesh, (48.0, 40.0))
    near = np.argmin(np.linalg.norm(square_mesh.nodes - [48.0, 40.0], axis=1))
    far = np.argmin(np.linalg.norm(square_mesh.nodes - [90.0, 40.0], axis=1))
    ratios = []
    for ell in (8.0, 32.0, 128.0):
        u = iso_operator.solve(ell, load)
        ratios.append(u[far] / u[near])
    assert ratios[0] < ratios[1] < ratios[2]


def test_anisotropy_stretches_the_field_three_to_one(large_square_mesh, anisotropic_x_geometry):
    centre = (200.0, 200.0)
    load = _point_load(large_square_mesh, centre)
    u = transport.assemble(large_square_mesh, anisotropic_x_geometry).solve(20.0, load)
    mx, my = _second_moments(large_square_mesh, u, centre)
    # A = diag(e^b, e^-b) with e^(2 b_max) = 3, so the spread ratio along/across fibres is 3.
    assert mx / my == pytest.approx(3.0, rel=0.06)

    rotated = geometry.rotated(anisotropic_x_geometry, 90.0)
    u_rot = transport.assemble(large_square_mesh, rotated).solve(20.0, load)
    mx_r, my_r = _second_moments(large_square_mesh, u_rot, centre)
    assert my_r / mx_r == pytest.approx(3.0, rel=0.06)


def test_beta_scale_controls_anisotropy_strength(large_square_mesh, anisotropic_x_geometry):
    centre = (200.0, 200.0)
    load = _point_load(large_square_mesh, centre)
    ratios = []
    for scale in (0.5, 1.0, 1.5):
        op = transport.assemble(large_square_mesh, anisotropic_x_geometry, beta_scale=scale)
        assert op.beta_scale == scale
        mx, my = _second_moments(large_square_mesh, op.solve(20.0, load), centre)
        ratios.append(mx / my)
    assert ratios[0] < ratios[1] < ratios[2]


def test_log_range_derivative_matches_finite_difference(aniso_operator, square_mesh):
    load = _point_load(square_mesh, (50.0, 30.0))
    ell, eps = 30.0, 1e-5
    u = aniso_operator.solve(ell, load)
    analytic = aniso_operator.log_range_derivative(ell, u)
    up = aniso_operator.solve(ell * np.exp(eps), load)
    down = aniso_operator.solve(ell * np.exp(-eps), load)
    numeric = (up - down) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-10 * np.abs(numeric).max())


def test_mass_solve_inverts_the_mass_matrix(iso_operator, square_mesh, rng):
    x = rng.standard_normal(square_mesh.n_nodes)
    np.testing.assert_allclose(iso_operator.mass_solve(iso_operator.mass @ x), x, rtol=1e-8, atol=1e-10)


def test_factorization_cache_is_bounded_and_reused(square_mesh):
    geo = uniform_geometry(np.ones((80, 100), bool))
    op = transport.assemble(square_mesh, geo, max_cached=2)
    first = op.factor(16.0)
    assert op.factor(16.0) is first
    op.factor(32.0)
    op.factor(64.0)
    assert len(op._cache) == 2
    assert op.factor(16.0) is not first  # evicted, so refactorised
    op.clear()
    assert op._cache == {} and op._mass_factor is None


def test_exposure_field_is_the_solution(iso_operator, square_mesh):
    load = _point_load(square_mesh, (20.0, 20.0))
    np.testing.assert_allclose(transport.exposure_field(iso_operator, 32.0, load), iso_operator.solve(32.0, load))


def _corrupted(op, **changes):
    fields = dict(mesh=op.mesh, stiffness=op.stiffness, mass=op.mass, lumped=op.lumped)
    fields.update(changes)
    return transport.Operator(**fields)


def test_validation_rejects_asymmetric_stiffness(iso_operator):
    n = iso_operator.stiffness.shape[0]
    bump = csr_matrix(([1.0], ([0], [1])), shape=(n, n))
    bad = _corrupted(iso_operator, stiffness=iso_operator.stiffness + bump)
    with pytest.raises(OperatorError, match="asymmetry"):
        bad.validate(spectrum=False)
    report, failures = bad.validate(spectrum=False, strict=False)
    assert report["valid"] is False
    assert any("no-flux" in f for f in failures)


def test_validation_rejects_nonpositive_lumped_mass(iso_operator):
    lumped = iso_operator.lumped.copy()
    lumped[3] = 0.0
    _, failures = _corrupted(iso_operator, lumped=lumped).validate(spectrum=False, strict=False)
    assert any("lumped mass" in f for f in failures)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect: _smallest_generalized_eigenvalue uses shift-invert Lanczos centred at "
        "~0 with k=1. The no-flux operator always has an exact zero eigenvalue (constants), so "
        "the search returns it and never sees a materially negative eigenvalue (here the true "
        "minimum is about -2.6). Correctly assembled operators are PSD by construction, so "
        "results are unaffected, but the PSD safeguard cannot fire. Remove this marker once fixed."
    ),
)
def test_validation_rejects_indefinite_stiffness(iso_operator):
    n = iso_operator.stiffness.shape[0]
    # A symmetric, row-sum-zero perturbation with a large negative direction.
    i, j = 10, 11
    bump = csr_matrix(([-5.0, 5.0, 5.0, -5.0], ([i, i, j, j], [i, j, i, j])), shape=(n, n))
    bad = _corrupted(iso_operator, stiffness=iso_operator.stiffness + bump)
    _, failures = bad.validate(strict=False)
    assert any("eigenvalue" in f for f in failures)


def test_checks_without_spectrum_and_repr(iso_operator):
    report = iso_operator.checks()
    assert "min_generalized_eigenvalue" not in report
    assert report["n_components"] == 1
    assert repr(iso_operator).startswith("Operator(500 nodes")

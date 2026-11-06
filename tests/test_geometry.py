import numpy as np
import pytest
from synthetic import make_histology, uniform_geometry

from histrange import geometry
from histrange.settings import BETA_MAX, CROSSING_THRESHOLD


def _angle_error(theta, target):
    """Smallest difference between two axial (mod pi) orientations."""
    d = np.mod(theta - target + 0.5 * np.pi, np.pi) - 0.5 * np.pi
    return np.abs(d)


def _striped_histology(angle_deg, period_px=16, size=128, pixel_size_um=1.0):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    a = np.deg2rad(angle_deg)
    perp = -np.sin(a) * xx + np.cos(a) * yy
    eosin = 0.5 + 0.5 * np.sin(2 * np.pi * perp / period_px)
    return make_histology(np.ones((size, size), bool), eosin=eosin, pixel_size_um=pixel_size_um)


def test_tensor_components_have_unit_determinant_and_expected_eigenpairs(rng):
    theta = rng.uniform(-np.pi, np.pi, 200)
    beta = rng.uniform(0, BETA_MAX, 200)
    a11, a12, a22 = geometry.tensor_components(theta, beta).T
    np.testing.assert_allclose(a11 * a22 - a12 * a12, 1.0, rtol=1e-12)
    for t, b, x11, x12, x22 in zip(theta[:10], beta[:10], a11, a12, a22):
        mat = np.array([[x11, x12], [x12, x22]])
        direction = np.array([np.cos(t), np.sin(t)])
        np.testing.assert_allclose(mat @ direction, np.exp(b) * direction, atol=1e-12)
        normal = np.array([-np.sin(t), np.cos(t)])
        np.testing.assert_allclose(mat @ normal, np.exp(-b) * normal, atol=1e-12)


def test_zero_beta_gives_identity_tensor():
    out = geometry.tensor_components(np.array([0.3, 1.2]), np.zeros(2))
    np.testing.assert_allclose(out, [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], atol=1e-15)


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 45.0, 90.0, 135.0])
def test_build_recovers_tissue_line_orientation(angle_deg):
    geo = geometry.build(_striped_histology(angle_deg))
    core = (slice(30, 98), slice(30, 98))
    err = _angle_error(geo.theta[core], np.deg2rad(angle_deg))
    assert np.degrees(np.median(err)) < 2.0
    # Clean stripes are fully coherent and uncrossed, so beta saturates.
    assert np.median(geo.beta[core]) > 0.9 * BETA_MAX
    assert geo.beta.max() <= BETA_MAX + 1e-12


def test_build_is_isotropic_on_featureless_tissue():
    hist = make_histology(np.ones((64, 64), bool), eosin=np.full((64, 64), 0.7))
    geo = geometry.build(hist)
    np.testing.assert_array_equal(geo.beta, 0.0)
    assert geo.anisotropic_fraction() == 0.0


def test_build_zeroes_beta_outside_tissue():
    hist = _striped_histology(0.0)
    hist.mask[:, :40] = False
    geo = geometry.build(hist)
    assert np.all(geo.beta[:, :40] == 0)
    assert geo.anisotropic_fraction() > 0.5


def test_crossing_statistic_separates_single_and_crossing_orientations():
    shape = (64, 64)
    ex = np.ones(shape)
    ey = np.zeros(shape)
    single = geometry.crossing_statistic(ex, ey, sigma_px=4.0)
    assert single.max() < 1e-9

    # Interleave pure-x and pure-y gradients: m2 averages to 0 and m4 to 1.
    checker = (np.add.outer(np.arange(64), np.arange(64)) % 2).astype(bool)
    ex2 = np.where(checker, 1.0, 0.0)
    ey2 = np.where(checker, 0.0, 1.0)
    crossing = geometry.crossing_statistic(ex2, ey2, sigma_px=4.0)
    assert np.median(crossing) > CROSSING_THRESHOLD


def test_isotropic_control_removes_all_anisotropy():
    geo = geometry.build(_striped_histology(30.0))
    iso = geometry.isotropic(geo)
    np.testing.assert_array_equal(iso.beta, 0.0)
    np.testing.assert_array_equal(iso.mask, geo.mask)
    pts = np.array([[60.0, 60.0], [10.0, 90.0]])
    np.testing.assert_allclose(iso.tensor_at(pts), [[1, 0, 1], [1, 0, 1]], atol=1e-15)


def test_rotated_control_turns_orientation_and_keeps_magnitude():
    mask = np.ones((20, 20), bool)
    geo = uniform_geometry(mask, theta=0.0, beta=BETA_MAX)
    rot = geometry.rotated(geo, 90.0)
    np.testing.assert_allclose(rot.theta, 0.5 * np.pi)
    np.testing.assert_array_equal(rot.beta, geo.beta)
    a = geo.tensor_at([[5.0, 5.0]])[0]
    b = rot.tensor_at([[5.0, 5.0]])[0]
    # A 90-degree turn swaps the principal conductances.
    np.testing.assert_allclose([b[0], b[2]], [a[2], a[0]], rtol=1e-12)
    assert rot.beta is not geo.beta  # the control must not alias the matched field


def test_morphology_mismatch_preserves_marginal_anisotropy(rng):
    mask = np.ones((128, 128), bool)
    beta = rng.uniform(0, BETA_MAX, mask.shape)
    theta = rng.uniform(-np.pi, np.pi, mask.shape)
    geo = geometry.Geometry(theta, np.ones_like(beta), np.zeros_like(beta), beta, mask, 1.0)
    shuffled = geometry.morphology_mismatch(geo, block_um=32.0, seed=3)
    np.testing.assert_allclose(np.sort(shuffled.beta.ravel()), np.sort(beta.ravel()))
    np.testing.assert_allclose(np.sort(shuffled.theta.ravel()), np.sort(theta.ravel()))
    assert not np.allclose(shuffled.theta, theta)
    # Blocks move intact: each 32 x 32 output block equals some input block.
    out_block = shuffled.theta[:32, :32]
    matches = [
        np.array_equal(out_block, theta[i:i + 32, j:j + 32]) for i in range(0, 128, 32) for j in range(0, 128, 32)
    ]
    assert sum(matches) == 1


def test_morphology_mismatch_is_seeded_and_keeps_support():
    mask = np.ones((64, 64), bool)
    mask[:8] = False
    geo = geometry.build(_striped_histology(20.0, size=64))
    geo = geometry.Geometry(geo.theta, geo.coherency, geo.crossing, np.where(mask, geo.beta, 0), mask, 1.0)
    a = geometry.morphology_mismatch(geo, block_um=16.0, seed=1)
    b = geometry.morphology_mismatch(geo, block_um=16.0, seed=1)
    c = geometry.morphology_mismatch(geo, block_um=16.0, seed=2)
    np.testing.assert_array_equal(a.theta, b.theta)
    assert not np.array_equal(a.theta, c.theta)
    assert np.all(a.beta[~mask] == 0)


def test_morphology_mismatch_small_image_is_unchanged_copy():
    geo = uniform_geometry(np.ones((10, 10), bool), theta=0.4, beta=0.2)
    out = geometry.morphology_mismatch(geo, block_um=64.0)
    np.testing.assert_array_equal(out.theta, geo.theta)
    assert out.theta is not geo.theta


def test_tensor_at_uses_identity_outside_tissue_and_clips_coordinates():
    mask = np.zeros((10, 10), bool)
    mask[:, :5] = True
    geo = uniform_geometry(mask, theta=0.0, beta=BETA_MAX, pixel_size_um=2.0)
    pts = np.array([[2.0, 2.0], [16.0, 2.0], [-50.0, 3.0], [500.0, 500.0]])
    out = geo.tensor_at(pts)
    np.testing.assert_allclose(out[0], [3 ** 0.5, 0, 3 ** -0.5], rtol=1e-12)
    np.testing.assert_allclose(out[1], [1, 0, 1], atol=1e-15)
    np.testing.assert_allclose(out[2], out[0])  # clipped to column 0, inside tissue
    np.testing.assert_allclose(out[3], [1, 0, 1], atol=1e-15)  # clipped to a non-tissue corner


def test_tensor_at_beta_scale_multiplies_anisotropy():
    geo = uniform_geometry(np.ones((4, 4), bool), theta=0.0, beta=0.4)
    out = geo.tensor_at([[1.0, 1.0]], beta_scale=1.5)[0]
    np.testing.assert_allclose(out[0], np.exp(0.6))


def test_geometry_repr():
    geo = uniform_geometry(np.ones((3, 5), bool), beta=0.1)
    assert repr(geo) == "Geometry(3 x 5 px, anisotropic 100.0%)"


def test_morphology_mismatch_stays_within_tissue_when_tissue_is_partial(rng):
    # Frozen real-specimen control (phase6b_lung/scripts/ablations.py::shuffled_in_tissue):
    # only blocks at least half inside tissue are permuted, so background never enters tissue
    # and the in-tissue (theta, beta) distribution is kept rather than diluted with zeros.
    mask = np.zeros((128, 128), bool)
    mask[:64, :] = True
    mask[64:, :64] = True  # an L made of whole 32 x 32 blocks
    beta = np.where(mask, rng.uniform(0.05, BETA_MAX, mask.shape), 0.0)
    theta = rng.uniform(-np.pi, np.pi, mask.shape)
    geo = geometry.Geometry(theta, np.ones_like(beta), np.zeros_like(beta), beta, mask, 1.0)
    for seed in range(5):
        out = geometry.morphology_mismatch(geo, block_um=32.0, seed=seed)
        np.testing.assert_allclose(np.sort(out.beta[mask]), np.sort(beta[mask]))
        assert np.all(out.beta[mask] > 0)
        np.testing.assert_array_equal(out.theta[~mask], theta[~mask])


def test_morphology_mismatch_keeps_in_tissue_anisotropy_on_a_disk_section():
    from synthetic import disk_mask, he_image

    from histrange import histology

    tissue = disk_mask(400, 400, (200, 200), 150)
    hist = histology.prepare(he_image(400, 400, tissue, stripe_period_px=14, stripe_angle_deg=30), 1.0)
    geo = geometry.build(hist)
    for seed in range(3):
        out = geometry.morphology_mismatch(geo, block_um=64.0, seed=seed)
        assert out.beta[out.mask].mean() == pytest.approx(geo.beta[geo.mask].mean(), rel=0.10)

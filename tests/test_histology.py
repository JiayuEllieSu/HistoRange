import numpy as np
import pytest
from synthetic import disk_mask, he_image

from histrange import histology
from histrange.settings import E_STAIN, H_STAIN


def test_optical_density_of_white_and_black():
    img = np.zeros((1, 2, 3))
    img[0, 0] = 255
    od = histology.optical_density(img)
    np.testing.assert_allclose(od[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(od[0, 1], np.log(256.0))


def test_optical_density_rescales_unit_interval_images():
    img = np.random.default_rng(0).integers(0, 256, (4, 4, 3))
    np.testing.assert_allclose(histology.optical_density(img / 255.0), histology.optical_density(img))


def test_optical_density_drops_alpha_channel():
    rgba = np.full((2, 2, 4), 128.0)
    assert histology.optical_density(rgba).shape == (2, 2, 3)


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 2)])
def test_optical_density_rejects_non_rgb(shape):
    with pytest.raises(ValueError):
        histology.optical_density(np.zeros(shape))


def test_stain_basis_rows_are_unit_and_residual_is_orthogonal():
    basis = histology.stain_basis()
    np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0)
    np.testing.assert_allclose(basis[0], H_STAIN / np.linalg.norm(H_STAIN))
    np.testing.assert_allclose(basis[1], E_STAIN / np.linalg.norm(E_STAIN))
    assert abs(basis[2] @ basis[0]) < 1e-12
    assert abs(basis[2] @ basis[1]) < 1e-12


def test_separate_stains_recovers_known_concentrations(rng):
    basis = histology.stain_basis()
    conc = rng.uniform(0.0, 2.0, (5, 6, 2))
    od = conc[..., 0:1] * basis[0] + conc[..., 1:2] * basis[1]
    out = histology.separate_stains(od, basis)
    np.testing.assert_allclose(out[..., 0], conc[..., 0], atol=1e-10)
    np.testing.assert_allclose(out[..., 1], conc[..., 1], atol=1e-10)
    np.testing.assert_allclose(out[..., 2], 0.0, atol=1e-10)


def test_separate_stains_truncates_negative_concentrations():
    basis = histology.stain_basis()
    od = (-1.0 * basis[0]).reshape(1, 1, 3)
    out = histology.separate_stains(od, basis)
    assert out.min() >= 0.0


def test_otsu_splits_a_bimodal_sample(rng):
    low = rng.normal(0.1, 0.02, 4000)
    high = rng.normal(0.9, 0.02, 1000)
    t = histology.otsu_threshold(np.concatenate([low, high]))
    # Every threshold inside the empty gap ties; the first bin of the gap is returned.
    assert np.all(high > t)
    assert np.mean(low <= t) > 0.999


def test_otsu_constant_input():
    assert histology.otsu_threshold(np.full(10, 3.0)) == 3.0


def test_disk_structuring_element():
    assert histology.disk(0.5) is None
    d = histology.disk(2.0)
    assert d.shape == (5, 5)
    assert d[2, 2] and d[0, 2] and not d[0, 0]
    np.testing.assert_array_equal(d, d.T)


def test_tissue_mask_recovers_disk_and_removes_specks():
    rows = cols = 160
    tissue = disk_mask(rows, cols, (80, 80), 55)
    speck = disk_mask(rows, cols, (12, 12), 4)  # ~50 um^2 at 1 um/px: below the 500 um^2 floor
    img = he_image(rows, cols, tissue | speck)
    mask = histology.tissue_mask(histology.optical_density(img), pixel_size_um=1.0)
    iou = (mask & tissue).sum() / (mask | tissue).sum()
    assert iou > 0.97
    assert not mask[speck].any()


def test_tissue_mask_keeps_internal_holes():
    rows = cols = 160
    tissue = disk_mask(rows, cols, (80, 80), 60) & ~disk_mask(rows, cols, (80, 80), 20)
    img = he_image(rows, cols, tissue)
    mask = histology.tissue_mask(histology.optical_density(img), pixel_size_um=1.0)
    assert not mask[80, 80]  # a 40 um lumen survives the 8 um closing
    assert mask[80, 120]


def test_tissue_mask_respects_capture_area():
    rows = cols = 120
    tissue = disk_mask(rows, cols, (60, 60), 50)
    capture = np.zeros((rows, cols), dtype=bool)
    capture[:, :60] = True
    img = he_image(rows, cols, tissue)
    mask = histology.tissue_mask(histology.optical_density(img), 1.0, capture_mask=capture)
    assert mask.any()
    assert not mask[:, 60:].any()


def test_tissue_mask_raises_on_blank_slide():
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        histology.tissue_mask(histology.optical_density(img), 1.0)


def test_prepare_returns_consistent_histology():
    rows, cols = 90, 110
    tissue = disk_mask(rows, cols, (55, 45), 35)
    hist = histology.prepare(he_image(rows, cols, tissue), pixel_size_um=2.0)
    assert hist.shape == (rows, cols)
    assert hist.eosin.shape == hist.hematoxylin.shape == hist.od_norm.shape == (rows, cols)
    assert hist.pixel_size_um == 2.0
    assert hist.tissue_area_um2() == pytest.approx(hist.mask.sum() * 4.0)
    # Pink tissue carries more eosin than haematoxylin in the synthetic image.
    assert hist.eosin[hist.mask].mean() > hist.hematoxylin[hist.mask].mean()
    assert "Histology(" in repr(hist)

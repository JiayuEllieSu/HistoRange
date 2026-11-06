"""Synthetic inputs shared by the test suite. Deterministic; no real data is read."""

import numpy as np

from histrange.geometry import Geometry
from histrange.histology import Histology


def make_histology(mask, eosin=None, pixel_size_um=1.0):
    """Build a Histology object directly from a mask (bypassing colour deconvolution)."""
    mask = np.asarray(mask, dtype=bool)
    if eosin is None:
        eosin = np.ones(mask.shape)
    eosin = np.asarray(eosin, dtype=float)
    return Histology(
        mask=mask,
        eosin=eosin,
        hematoxylin=np.zeros(mask.shape),
        od_norm=eosin.copy(),
        pixel_size_um=float(pixel_size_um),
    )


def uniform_geometry(mask, theta=0.0, beta=0.0, pixel_size_um=1.0):
    """A Geometry with the same orientation and anisotropy at every in-tissue pixel."""
    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    return Geometry(
        theta=np.full(shape, float(theta)),
        coherency=np.ones(shape),
        crossing=np.zeros(shape),
        beta=np.where(mask, float(beta), 0.0),
        mask=mask,
        pixel_size_um=float(pixel_size_um),
    )


def he_image(rows, cols, tissue_mask, stripe_period_px=None, stripe_angle_deg=0.0, seed=0):
    """Synthetic RGB H&E-like image: white background, pink tissue, optional eosin stripes.

    Stripes run *along* ``stripe_angle_deg`` (measured from the +x axis), so the expected
    tissue-line orientation is that angle.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    if stripe_period_px:
        a = np.deg2rad(stripe_angle_deg)
        # Coordinate perpendicular to the stripe direction.
        perp = -np.sin(a) * xx + np.cos(a) * yy
        eosin = 0.55 + 0.45 * np.sin(2.0 * np.pi * perp / float(stripe_period_px))
    else:
        eosin = np.full((rows, cols), 0.8)
    hema = np.full((rows, cols), 0.15)
    # Beer-Lambert mixing of the reference stains (see histrange.settings).
    h = np.array([0.650, 0.704, 0.286])
    e = np.array([0.072, 0.990, 0.105])
    od = hema[..., None] * h + eosin[..., None] * e
    rgb = 256.0 * np.exp(-od) - 1.0
    rgb = np.where(np.asarray(tissue_mask, bool)[..., None], rgb, 250.0)
    rgb = rgb + rng.normal(0.0, 1.0, rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def disk_mask(rows, cols, center, radius):
    yy, xx = np.mgrid[0:rows, 0:cols]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2

"""Shared fixtures for the HistoRange test suite.

Every fixture is generated from a fixed seed so that the suite is deterministic.
Nothing in the suite reads real data; see ``synthetic.py`` for the generators.
"""

import warnings

import numpy as np
import pytest
from synthetic import make_histology, uniform_geometry

from histrange.mesh import build as build_mesh
from histrange.settings import BETA_MAX


@pytest.fixture
def rng():
    return np.random.default_rng(20261019)


@pytest.fixture
def square_histology():
    """A 100 x 80 um fully tissue-covered rectangle at 1 um/px."""
    return make_histology(np.ones((80, 100), dtype=bool))


@pytest.fixture
def square_mesh(square_histology):
    return build_mesh(square_histology, spacing_um=4.0)


@pytest.fixture
def large_square_mask():
    """400 x 400 um tissue at 2 um/px, large enough that a 20 um field never sees the edge."""
    return np.ones((200, 200), dtype=bool)


@pytest.fixture
def large_square_mesh(large_square_mask):
    return build_mesh(make_histology(large_square_mask, pixel_size_um=2.0), spacing_um=4.0)


@pytest.fixture
def two_blob_histology():
    """Two disconnected tissue islands of unequal size."""
    mask = np.zeros((60, 140), dtype=bool)
    mask[5:55, 5:65] = True
    mask[10:40, 90:130] = True
    return make_histology(mask)


@pytest.fixture
def anisotropic_x_geometry(large_square_mask):
    return uniform_geometry(large_square_mask, theta=0.0, beta=BETA_MAX, pixel_size_um=2.0)


@pytest.fixture
def quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

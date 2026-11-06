"""End-to-end check: the shipped example runs and recovers the planted answer."""

import importlib.util
import pathlib
import re
import warnings

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic_walkthrough.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("synthetic_walkthrough", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def walkthrough():
    module = _load_example()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # A smaller BH family and draw count keep this under ~15 s; the floor is
        # 0.6589 * (1/1000)**0.7998 * 9 = 0.024, so a strong signal can still be detected.
        return module.run(n_null=8, draws=999, seed=0, verbose=False)


@pytest.mark.slow
def test_walkthrough_recovers_geometry_and_operator(walkthrough):
    assert abs(walkthrough["fibre_angle_deg"] - 30.0) < 2.0
    assert walkthrough["operator_report"]["valid"]
    assert walkthrough["ranges"].size >= 9


@pytest.mark.slow
def test_walkthrough_detects_the_planted_interaction_only(walkthrough):
    assert walkthrough["eligibility"]["eligible"]
    assert walkthrough["detected"][0]
    assert walkthrough["detected"][1:].sum() <= 1
    assert walkthrough["target_detection"]["Lambda_A"] > 3.0


@pytest.mark.slow
def test_walkthrough_range_is_supported_and_near_truth(walkthrough):
    t1 = walkthrough["t1"]
    assert t1["Lambda"] > walkthrough["tau"]
    assert abs(np.log2(t1["ell_hat"] / walkthrough["ell_true"])) <= 1.0
    for level, rep in walkthrough["reports"].items():
        assert rep["status"] in {"interval", "upper_bound"}
        assert rep["level"] == level
        assert rep.get("upper_um", np.inf) >= walkthrough["ell_true"]


@pytest.mark.slow
def test_walkthrough_robustness_arms_run(walkthrough):
    assert set(walkthrough["anisotropy_optima"]) == {0.5, 1.0, 1.5}
    assert isinstance(walkthrough["anisotropy_robust"], bool)
    assert isinstance(walkthrough["mesh_robust"], bool)


@pytest.mark.slow
def test_readme_quickstart_runs(capsys):
    """Execute the README quickstart verbatim on synthetic inputs so it cannot drift from the API."""
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(encoding="utf-8"), re.S)
    assert blocks, "README has no python code block"
    ex = _load_example()
    rng = np.random.default_rng(1)
    spot_um, field_um = 16.0, 320.0
    coords = ex.bin_centres(field_um, spot_um)
    library_size = rng.gamma(10.0, 300.0, len(coords))
    ligand_counts = rng.poisson(np.where(np.abs(coords[:, 0] - 100.0) < 40.0, 4.0, 0.05))
    near = np.exp(-np.hypot(coords[:, 0] - 100.0, coords[:, 1] - 160.0) / 60.0)
    receptor_counts = rng.poisson(library_size * 5e-4 * np.exp(0.8 * (near - near.mean()) / near.std()))
    namespace = {
        "image": ex.synthetic_he(field_um, 2.0, 30.0, 40.0, seed=1),
        "pixel_size_um": 2.0,
        "coords": coords,
        "ligand_counts": ligand_counts,
        "receptor_counts": receptor_counts,
        "library_size": library_size,
        "spot_um": spot_um,
        "near_null_lambdas": rng.chisquare(1, 50),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        exec(compile(blocks[0], "README.md", "exec"), namespace)
    assert namespace["claim"]["status"] in {"interval", "upper_bound", "uninformative", "unresolved"}
    assert 0 < namespace["p_sec"] <= 1
    assert "status" in capsys.readouterr().out

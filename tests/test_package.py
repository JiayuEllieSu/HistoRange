import pathlib
import re

import histrange

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_public_api_is_importable():
    for name in histrange.__all__:
        assert hasattr(histrange, name), name
    assert callable(histrange.prepare_histology)
    assert callable(histrange.build_geometry)
    assert callable(histrange.build_mesh)


def test_version_is_single_sourced():
    assert re.fullmatch(r"\d+\.\d+\.\d+", histrange.__version__)
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^version: %s$" % re.escape(histrange.__version__), citation, re.M)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "[%s]" % histrange.__version__ in changelog


def test_metadata_has_no_placeholder_urls():
    for name in ("README.md", "CITATION.cff", "pyproject.toml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "OWNER" not in text and "example.com" not in text, name
        assert "github.com/JiayuEllieSu/HistoRange" in text, name


def test_requirements_match_pyproject():
    reqs = {line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()}
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for req in reqs:
        assert '"%s"' % req in pyproject, req

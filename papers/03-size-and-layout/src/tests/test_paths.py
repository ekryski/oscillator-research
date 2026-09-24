"""Every data and results location anchors on the package, and both roots take an override."""

import importlib

import harness.utils.paths as paths


def test_roots_anchor_on_the_package_not_the_cwd():
    assert paths.SRC_ROOT.name == "src" and (paths.SRC_ROOT / "harness" / "utils" / "paths.py").is_file()
    assert paths.PAPER_ROOT == paths.SRC_ROOT.parent
    assert paths.RESULTS_DIR == paths.PAPER_ROOT / "results"
    assert paths.PAPER02_ROOT.name == "02-untrained-reservoirs"


def test_the_data_override_is_read_at_import_and_the_results_override_per_call(monkeypatch, tmp_path):
    monkeypatch.setenv("OSC_DATA_DIR", str(tmp_path / "corpora"))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.CACHE_DIR == tmp_path / "corpora" / "cache"
    finally:
        monkeypatch.delenv("OSC_DATA_DIR")
        importlib.reload(paths)
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path / "elsewhere"))
    assert paths.results_root() == tmp_path / "elsewhere"


def test_the_data_directory_is_not_committed():
    assert (paths.SRC_ROOT / "data" / ".gitignore").is_file()

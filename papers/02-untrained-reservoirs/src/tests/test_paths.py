"""Contracts for `harness.paths`.

Every data and results location is anchored on the package's own file so that
commands work from any working directory, and both roots take an environment
override so a reproduction can be written into a fresh tree without touching
the committed one. Both properties are easy to break silently, so both are
pinned here.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, ".")
import harness.utils.paths as paths


def test_roots_anchor_on_the_package_not_the_cwd():
    # src/harness/utils/paths.py -> SRC_ROOT is src/, PAPER_ROOT its parent
    assert paths.SRC_ROOT.name == "src"
    assert (paths.SRC_ROOT / "harness" / "utils" / "paths.py").is_file()
    assert paths.PAPER_ROOT == paths.SRC_ROOT.parent
    assert paths.DATA_DIR == paths.SRC_ROOT / "data"
    assert paths.RESULTS_DIR == paths.PAPER_ROOT / "results"


def test_derived_paths_hang_off_their_roots():
    assert paths.CACHE_DIR.parent == paths.DATA_DIR
    assert paths.AUDIOMNIST_DIR.parts[-2:] == ("AudioMNIST", "data")
    assert paths.FIGURES_DIR.parent == paths.RESOURCES_DIR
    assert paths.AUDIO_DIR.parent == paths.RESOURCES_DIR


def test_environment_overrides_redirect_both_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("OSC_DATA_DIR", str(tmp_path / "corpora"))
    monkeypatch.setenv("OSC_RESULTS_DIR", str(tmp_path / "elsewhere"))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DATA_DIR == tmp_path / "corpora"
        assert reloaded.RESULTS_DIR == tmp_path / "elsewhere"
        assert reloaded.CACHE_DIR == tmp_path / "corpora" / "cache"
    finally:
        monkeypatch.delenv("OSC_DATA_DIR")
        monkeypatch.delenv("OSC_RESULTS_DIR")
        importlib.reload(paths)


def test_the_committed_results_tree_is_where_the_paper_points():
    assert paths.RESULTS_DIR.is_dir(), "the committed per-run results should be present"
    assert (paths.RESULTS_DIR / "envelope").is_dir()
    assert (paths.PAPER_ROOT / "resources" / "figures").is_dir()
    drafts = list(paths.PAPER_ROOT.glob("*-DRAFT.md"))
    assert len(drafts) == 1, f"expected exactly one manuscript, found {drafts}"


def test_data_directory_is_not_committed():
    """The corpus and the derived bank are rebuilt, never shipped."""
    keep = {p.name for p in Path(paths.DATA_DIR).glob("*")} if paths.DATA_DIR.is_dir() else set()
    assert not (keep - {"README.md", ".gitignore", "AudioMNIST", "cache"}), (
        "src/data/ should hold only its README, its .gitignore, and untracked data")
    assert (paths.DATA_DIR / ".gitignore").is_file()

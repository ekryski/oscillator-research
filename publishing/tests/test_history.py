"""The build history names its entries after the day and the commit, finds the
reference a diff is drawn against in the documented order, and names the diff
after both ends.

A tiny repository stands in for the real one: one paper folder with a
manuscript and a figure, a publishing/ folder, and a `main` branch. No pandoc
or TeX is needed; the one path that would build a commit is asked not to.
"""

import datetime as dt
import os
import subprocess
from pathlib import Path

import pytest
from history import (
    Entry,
    diff_name,
    entries,
    figure_changes,
    is_diff_output,
    paper_in,
    record,
    resolve,
    source_changes,
    stamp,
)

SLUG = "07-tiny"
FACE = "iclr"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    paper = root / "papers" / SLUG
    (paper / "resources" / "figures").mkdir(parents=True)
    (root / "publishing" / "lib").mkdir(parents=True)
    (paper / "tiny-paper.md").write_text("## 1 Introduction\n\nFirst.\n")
    (paper / "resources" / "figures" / "fig1.pdf").write_bytes(b"%PDF fig1 v1")
    (paper / "tiny-paper-iclr.tex").write_text("\\documentclass{article}\\begin{document}First.\\end{document}\n")
    (root / "publishing" / "lib" / "placeholder.py").write_text("")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    git(root, "config", "user.email", "t@example.org")
    git(root, "config", "user.name", "t")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "the paper")
    return root


def head(root: Path) -> str:
    return git(root, "rev-parse", "--short", "HEAD")


def kept(root: Path, label: str, *, mark: bool = False, when: int | None = None) -> Entry:
    """Put a build in the history, optionally made to look `when` seconds old."""
    paper = paper_in(root, SLUG)
    tex, pdf = root / "a.tex", root / "a.pdf"
    tex.write_text(f"% {label}\n")
    pdf.write_bytes(b"%PDF")
    e = record(root, paper, FACE, tex, pdf, label, git(root, "rev-parse", "HEAD"), mark=mark)
    if when is not None:
        os.utime(e.tex, (when, when))
    return e


# --- the stamp ---------------------------------------------------------------

def test_a_clean_tree_is_stamped_with_the_day_and_the_commit(repo):
    assert stamp(repo, paper_in(repo, SLUG), dt.date(2026, 11, 7)) == f"2026-11-07-{head(repo)}"


def test_an_edited_manuscript_or_figure_makes_the_stamp_dirty(repo):
    paper = paper_in(repo, SLUG)
    (paper.dir / "tiny-paper.md").write_text("## 1 Introduction\n\nSecond.\n")
    assert stamp(repo, paper, dt.date(2026, 11, 7)) == f"2026-11-07-{head(repo)}-dirty"
    git(repo, "checkout", "--", ".")
    (paper.dir / "resources" / "figures" / "fig1.pdf").write_bytes(b"%PDF fig1 v2")
    assert source_changes(repo, paper) == [f"papers/{SLUG}/resources/figures/fig1.pdf"]


def test_a_change_under_publishing_makes_the_stamp_dirty(repo):
    (repo / "publishing" / "lib" / "placeholder.py").write_text("changed")
    assert stamp(repo, paper_in(repo, SLUG)).endswith("-dirty")


def test_the_built_formats_a_build_rewrites_do_not_make_it_dirty(repo):
    # a build writes the .tex, PDF, HTML, EPUB and Word files beside the
    # manuscript; those changing is the build's own doing, not an edit
    paper = paper_in(repo, SLUG)
    (paper.dir / "tiny-paper-iclr.tex").write_text("rebuilt with another pandoc\n")
    (paper.dir / "tiny-paper.html").write_text("<html>")
    (paper.dir / "tiny-paper-title-page.docx").write_bytes(b"PK")
    assert source_changes(repo, paper) == []
    assert not stamp(repo, paper).endswith("-dirty")


# --- the history -------------------------------------------------------------

def test_a_build_is_kept_under_the_paper_the_face_and_its_stamp(repo):
    e = kept(repo, "2026-11-07-abc1234")
    assert e.tex == repo / "publishing" / ".history" / SLUG / FACE / "2026-11-07-abc1234.tex"
    assert e.pdf.read_bytes() == b"%PDF"
    assert e.tex.with_suffix(".commit").read_text().split() == [git(repo, "rev-parse", "HEAD")]


def test_a_mark_is_kept_by_name_beside_the_dated_entries(repo):
    e = kept(repo, "iclr-2027-submitted", mark=True)
    assert e.tex == repo / "publishing" / ".history" / SLUG / FACE / "marks" / "iclr-2027-submitted.tex"
    assert e.kind == "mark"
    with pytest.raises(ValueError):
        kept(repo, "../escape", mark=True)


def test_entries_come_newest_first_marks_included(repo):
    kept(repo, "2026-11-01-aaaaaaa", when=1_000)
    kept(repo, "submitted", mark=True, when=2_000)
    kept(repo, "2026-11-05-bbbbbbb", when=3_000)
    assert [e.label for e in entries(repo, SLUG, FACE)] == ["2026-11-05-bbbbbbb", "submitted", "2026-11-01-aaaaaaa"]


# --- the reference -----------------------------------------------------------

def test_with_no_history_the_reference_is_the_last_commit_on_main_touching_the_paper(repo, tmp_path):
    paper = paper_in(repo, SLUG)
    (repo / "publishing" / "lib" / "placeholder.py").write_text("tooling only")
    git(repo, "commit", "-qam", "tooling")                    # on main, but not the paper's
    e = resolve(repo, paper, FACE, into=tmp_path / "ref", build=False)
    assert e.kind == "main"
    assert e.commit == git(repo, "rev-parse", "HEAD~1")
    assert e.label == git(repo, "rev-parse", "--short", "HEAD~1")
    # the committed .tex of that face is what it compares with, fetched into place
    assert e.tex == tmp_path / "ref" / f"{e.label}.tex"
    assert "First." in e.tex.read_text()


def test_the_previous_entry_beats_main(repo, tmp_path):
    paper = paper_in(repo, SLUG)
    kept(repo, "2026-11-01-aaaaaaa", when=1_000)
    newest = kept(repo, "submitted", mark=True, when=2_000)
    e = resolve(repo, paper, FACE, into=tmp_path / "ref", build=False)
    assert (e.kind, e.label) == ("mark", "submitted")
    # a snapshot, so that keeping the new build cannot change the reference under the diff
    assert e.tex != newest.tex and e.tex.read_text() == newest.tex.read_text()


def test_a_named_reference_is_a_mark_first_then_a_file_then_a_git_ref(repo, tmp_path):
    paper = paper_in(repo, SLUG)
    kept(repo, "2026-11-05-bbbbbbb")                          # the default, which --diff-against overrides
    kept(repo, "v1", mark=True)
    other = tmp_path / "v1.tex"
    other.write_text("a file that happens to share the name\n")
    assert resolve(repo, paper, FACE, "v1", into=tmp_path / "r1", build=False).kind == "mark"
    by_path = resolve(repo, paper, FACE, str(other), into=tmp_path / "r2", build=False)
    assert (by_path.kind, by_path.label, by_path.commit) == ("path", "v1", "")
    assert by_path.tex.read_text() == other.read_text()
    by_ref = resolve(repo, paper, FACE, "main", into=tmp_path / "r3", build=False)
    assert (by_ref.kind, by_ref.commit) == ("ref", git(repo, "rev-parse", "main"))
    assert by_ref.label == head(repo)
    with pytest.raises(ValueError):
        resolve(repo, paper, FACE, "no-such-thing", into=tmp_path / "r4", build=False)


def test_a_commit_without_a_committed_tex_needs_a_build(repo, tmp_path):
    paper = paper_in(repo, SLUG)
    (paper.dir / "tiny-paper-iclr.tex").unlink()
    git(repo, "commit", "-qam", "drop the built tex")
    e = resolve(repo, paper, FACE, "HEAD", into=tmp_path / "ref", build=False)
    assert e.kind == "ref" and e.tex is None             # named, but not built, as asked


def test_without_a_main_branch_or_history_there_is_nothing_to_compare_with(repo, tmp_path):
    git(repo, "branch", "-m", "main", "trunk")
    assert resolve(repo, paper_in(repo, SLUG), FACE, into=tmp_path / "ref", build=False) is None


# --- the diff ----------------------------------------------------------------

def test_the_diff_is_named_after_both_ends_and_the_output_check_skips_it():
    name = diff_name("tiny-paper", "iclr", "iclr-2027-submitted", "2026-11-07-4b91ebd-dirty")
    assert name == "tiny-paper-iclr-diff-iclr-2027-submitted-to-2026-11-07-4b91ebd-dirty.pdf"
    assert is_diff_output(name)
    assert is_diff_output(diff_name("tiny-paper", "preprint", "2026-11-01-aaaaaaa", "2026-11-07-4b91ebd"))
    for shipped in ("tiny-paper-iclr.pdf", "tiny-paper-preprint.tex", "a-diff-model-iclr.pdf", "tiny-paper-iclr-diff.pdf"):
        assert not is_diff_output(shipped)


def test_figures_are_compared_by_checksum_with_the_reference_commit(repo):
    paper = paper_in(repo, SLUG)
    then = git(repo, "rev-parse", "HEAD")
    figures = paper.dir / "resources" / "figures"
    (figures / "fig1.pdf").write_bytes(b"%PDF fig1 v2")
    (figures / "fig2.png").write_bytes(b"\x89PNG new")
    (figures / "fig1.pdf").touch()
    assert figure_changes(repo, then, paper) == {"changed": ["fig1.pdf"], "added": ["fig2.png"], "removed": []}
    (figures / "fig1.pdf").unlink()
    assert figure_changes(repo, then, paper)["removed"] == ["fig1.pdf"]


def test_a_touched_but_unchanged_figure_is_not_reported(repo):
    paper = paper_in(repo, SLUG)
    fig = paper.dir / "resources" / "figures" / "fig1.pdf"
    os.utime(fig, (1, 1))
    assert figure_changes(repo, git(repo, "rev-parse", "HEAD"), paper) == {"changed": [], "added": [], "removed": []}

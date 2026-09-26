"""The venue and face flags resolve the way the README says they do.

`--dry-run` exits before pandoc or TeX is needed, so these run anywhere and
check the one thing worth checking about the flags: which face each command
resolves to, and therefore which file it would write.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = "01"
PDF = "papers/01-evidence-audit/from-synchronization-physics-to-trained-dynamics"


def plan(*flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "publishing/publish.sh", PAPER, *flags, "--dry-run"],
                          cwd=ROOT, capture_output=True, text=True)


def test_no_flags_means_the_named_preprint_face_in_the_house_style():
    out = plan().stdout
    assert "venue: tmlr" in out
    assert "face: preprint" in out
    assert f"would write {PDF}-preprint.pdf" in out


def test_naming_a_venue_means_its_anonymous_submission_face():
    out = plan("--tmlr").stdout
    assert "face: submission" in out
    assert f"would write {PDF}-tmlr.pdf" in out


def test_a_face_flag_overrides_the_venue_default():
    assert f"would write {PDF}-preprint.pdf" in plan("--tmlr", "--preprint").stdout
    assert f"would write {PDF}-tmlr-accepted.pdf" in plan("--tmlr", "--accepted").stdout
    assert f"would write {PDF}-tmlr.pdf" in plan("--anonymous").stdout


def test_flag_order_does_not_matter():
    assert plan("--preprint", "--tmlr").stdout == plan("--tmlr", "--preprint").stdout


def test_an_unknown_venue_fails_and_says_where_the_template_would_go():
    result = plan("--nowhere")
    assert result.returncode != 0
    assert "no venue template" in result.stderr
    assert "publishing/templates/nowhere.latex" in result.stderr


def test_the_arxiv_bundle_is_always_the_preprint_face():
    out = plan("--tmlr").stdout
    assert "arxiv.tar.gz (preprint face, tmlr style)" in out


def test_help_prints_the_usage_from_the_script_header():
    result = subprocess.run(["bash", "publishing/publish.sh", "--help"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--tmlr" in result.stdout and "--preprint" in result.stdout


def test_the_neural_networks_venue_resolves_to_its_own_named_submission_file():
    # Neural Networks is single-anonymized: the submission face keeps the
    # author, so nothing in the plan may claim otherwise
    out = plan("--neunet").stdout
    assert "venue: neunet" in out
    assert "face: submission" in out
    assert f"would write {PDF}-neunet.pdf" in out


def test_a_venue_may_ship_metadata_for_pandoc_beside_its_template():
    # the knobs pandoc's partials read (paragraph indent, natbib options) are
    # the venue's to set, and the build must find them where the README says
    venue_yaml = ROOT / "publishing/templates/neunet.yaml"
    assert venue_yaml.exists()
    text = venue_yaml.read_text()
    assert "indent: true" in text and "natbiboptions:" in text
    script = (ROOT / "publishing/publish.sh").read_text()
    assert 'venue_metadata "$VENUE"' in script


def test_a_venue_preprint_keeps_its_own_file_name():
    # the house style's preprint is `…-preprint.pdf`; another venue's preprint
    # face must not overwrite it, and the house style's must keep its name
    assert f"would write {PDF}-neunet-preprint.pdf" in plan("--neunet", "--preprint").stdout
    assert f"would write {PDF}-preprint.pdf" in plan("--tmlr", "--preprint").stdout


def test_the_iclr_venue_resolves_to_its_own_anonymous_submission_file():
    out = plan("--iclr").stdout
    assert "venue: iclr" in out and "face: submission" in out
    assert f"would write {PDF}-iclr.pdf" in out
    assert f"would write {PDF}-iclr-preprint.pdf" in plan("--iclr", "--preprint").stdout


def test_a_style_name_with_an_underscore_reaches_bibtex_as_written():
    # pandoc escapes metadata for LaTeX, so as metadata iclr2027_conference
    # became iclr2027\_conference and BibTeX found no style file. As a template
    # variable it is inserted verbatim, which is how the build must pass it.
    import shutil
    import pytest
    if shutil.which("pandoc") is None:
        pytest.skip("needs pandoc")
    def bibstyle(flag: str) -> str:
        out = subprocess.run(["pandoc", "--to=latex", "--natbib", "--standalone",
                              "--template=publishing/templates/iclr.latex",
                              f"--{flag}=biblio-style=iclr2027_conference"],
                             cwd=ROOT, input="x", capture_output=True, text=True).stdout
        return next(l for l in out.splitlines() if l.startswith("\\bibliographystyle"))
    assert bibstyle("variable") == "\\bibliographystyle{iclr2027_conference}"
    assert bibstyle("metadata") != "\\bibliographystyle{iclr2027_conference}"
    script = (ROOT / "publishing/publish.sh").read_text()
    assert "--variable=biblio-style=" in script and "--metadata=biblio-style=" not in script

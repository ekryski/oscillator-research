"""Citations point at the preprint's DOI once it has one, and at the repository until then."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cite_this  # noqa: E402

META = {"title": "A Survey of Things", "authors": [{"name": "Eric Kryski", "affiliation": "", "email": ""}],
        "keywords": ["things", "surveys"]}


def test_without_a_doi_every_format_points_at_the_repository():
    bib, apa = cite_this.bibtex(META, "01-slug"), cite_this.styles(META, "01-slug")["apa"]
    assert "@techreport{" in bib and "github.com/ekryski/oscillator-research/blob/main/papers/01-slug/" in bib
    assert "[Preprint]. https://github.com" in apa and "doi" not in bib.lower()


def test_with_a_doi_every_format_points_at_it_and_names_the_server():
    meta = {**META, "doi": "10.2139/ssrn.7445198", "preprint_server": "SSRN"}
    bib = cite_this.bibtex(meta, "01-slug")
    assert "@misc{" in bib and "doi          = {10.2139/ssrn.7445198}" in bib
    assert "howpublished = {SSRN preprint}" in bib and "url          = {https://doi.org/10.2139/ssrn.7445198}" in bib
    styles = cite_this.styles(meta, "01-slug")
    assert styles["apa"].endswith("[Preprint]. SSRN. https://doi.org/10.2139/ssrn.7445198")
    assert all("https://doi.org/10.2139/ssrn.7445198" in s for s in styles.values())
    assert "DO  - 10.2139/ssrn.7445198" in cite_this.ris(meta, "01-slug")


def test_the_metadata_reader_picks_up_the_doi_and_server(tmp_path):
    yaml = tmp_path / "paper.yaml"
    yaml.write_text("author:\n  - name: Eric Kryski\ndoi: 10.2139/ssrn.7445198\npreprint-server: SSRN\nkeywords:\n  - things\n")
    meta = cite_this.load_metadata(yaml)
    assert meta["doi"] == "10.2139/ssrn.7445198" and meta["preprint_server"] == "SSRN"

"""Turning the committed .bib into what tmlr.bst can actually read."""

from bibtex_compat import convert, sort_key_for

ONLINE = """@online{un-0,
  author        = {Unconventional AI},
  title         = {Un-0: an oscillator language model},
  url           = {https://unconv.ai/blog/un-0},
  urldate       = {2026-08-16},
  note          = {not peer reviewed},
}
"""


def test_online_becomes_a_type_bibtex_knows():
    # tmlr.bst drops an entry whose type it does not define — silently, and then
    # natbib rejects the whole bibliography
    out = convert(ONLINE)
    assert out.startswith("@misc{un-0,")


def test_online_keeps_its_url_where_a_style_file_looks_for_it():
    assert r"howpublished  = {\url{https://unconv.ai/blog/un-0}}" in convert(ONLINE)


def test_urldate_is_folded_into_the_note_not_dropped():
    out = convert(ONLINE)
    assert "urldate" not in out
    assert "not peer reviewed. Accessed 2026-08-16" in out


def test_a_comment_inside_an_entry_is_removed():
    # BibTeX treats it as a syntax error and skips the rest of the entry
    text = "@article{k,\n  % cited in the manuscript as: ESN\n  author = {Jaeger},\n}"
    out = convert(text)
    assert "% cited" not in out
    assert "author = {Jaeger}" in out


def test_an_arxiv_eprint_survives_as_a_note():
    text = "@misc{k,\n  author = {A},\n  eprint = {2010.00951},\n  archiveprefix = {arXiv},\n}"
    out = convert(text)
    assert "eprint" not in out and "archiveprefix" not in out
    assert "arXiv:2010.00951" in out


def test_an_entry_with_no_author_gets_a_sort_key():
    # one unlabelled \bibitem makes natbib refuse every citation in the paper
    out = convert("@misc{rodan2011,\n  title = {A},\n}")
    assert "key           = {Rodan}" in out


def test_an_entry_with_an_author_is_left_alone():
    assert "key " not in convert("@misc{rodan2011,\n  author = {Rodan, A},\n}")


def test_sort_key_reads_the_citekey():
    assert sort_key_for("bertschinger2004") == "Bertschinger"
    assert sort_key_for("oscillator-ising-machines") == "Oscillator Ising Machines"


def test_a_multi_line_author_is_seen_and_not_overwritten():
    text = "@inproceedings{k,\n  author = {A One and\n            B Two},\n}"
    assert "key " not in convert(text)


def test_a_citation_label_is_not_an_author_list():
    # "Castaldo, Aristides, Clusella, Garcia-Ojalvo & Ruffini" reads to BibTeX
    # as one name with four comma separators, which aborts the entry and takes
    # the whole author-year bibliography with it
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    from fetch_metadata import is_label
    assert is_label("Castaldo, Aristides, Clusella, Garcia-Ojalvo & Ruffini")
    assert is_label("Dan, Ding \\& Wu")                 # BibTeX-escaped ampersand
    assert is_label("Gray, König, Engel & Singer")
    assert not is_label("Kuramoto, Yoshiki")
    assert not is_label("Rusch, T. Konstantin and Mishra, Siddhartha")
    assert not is_label("")


def test_a_doi_makes_the_publisher_url_redundant():
    # tmlr.bst prints both, so the entry would state its address twice
    out = convert("@article{k,\n  title = {T},\n  doi = {10.1/x},\n"
                  "  url = {https://example.org/10.1/x},\n}")
    assert "url" not in out and "10.1/x" in out


def test_an_arxiv_url_is_dropped_once_the_id_is_in_the_note():
    out = convert("@misc{k,\n  title = {T},\n  eprint = {2401.00001},\n"
                  "  url = {https://arxiv.org/abs/2401.00001},\n}")
    assert out.count("2401.00001") == 1 and "arXiv:2401.00001" in out


def test_a_url_survives_when_it_is_the_only_identifier():
    # the ML venues mint no DOI, so their record page is the address
    out = convert("@inproceedings{k,\n  title = {T},\n"
                  "  url = {https://openreview.net/forum?id=abc},\n}")
    assert "openreview.net" in out


def test_a_preprint_pointer_survives_beside_the_record():
    out = convert("@inproceedings{k,\n  title = {T},\n"
                  "  url = {https://openreview.net/forum?id=abc},\n"
                  "  note = {Preprint: arXiv:2401.00001},\n}")
    assert "openreview.net" in out and "arXiv:2401.00001" in out


def test_a_misc_doi_is_folded_into_the_note_so_it_renders():
    # tmlr.bst has no doi slot for @misc, so a bioRxiv entry would lose it
    out = convert("@misc{k,\n  title = {T},\n  howpublished = {bioRxiv},\n"
                  "  doi = {10.1101/2025.01.01.000000},\n}")
    assert "doi: 10.1101/2025.01.01.000000" in out and "\n  doi " not in out


def test_a_bare_underscore_in_a_title_is_escaped():
    # unescaped, TeX reads it as a subscript in text mode and the entry breaks
    out = convert("@article{k,\n  title = {Hopf normal form with S_N symmetry},\n}")
    assert r"S\_N" in out


def test_an_underscore_inside_math_or_a_url_is_left_alone():
    out = convert("@article{k,\n  title = {With {$S_N$} symmetry},\n"
                  "  howpublished = {\\url{http://x.org/paper_files}},\n}")
    assert "$S_N$" in out and "paper_files" in out

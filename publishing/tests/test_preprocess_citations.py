"""Which work a citation resolves to, when the label alone cannot say.

Every case here is a real one from this repository, reduced to a fixture. Two
works answering to "Nunley 2026", a blog post and the repository beside it both
called "Unconventional AI 2026", an old DOI whose parentheses the manuscript
percent-encodes. The label collapses them; the link does not, so the link is
what resolution reads first.
"""

from check_citemap import shared_links
from preprocess import alias_map, entry_links, rewrite_brackets, rewrite_links, url_index, url_key

#: two works by one author in one year, an old parenthesised DOI, a blog post
#: and its code repository, and one work the bibliography can only name
BIB = """
@misc{kuramoto-attention,
  author        = {Nunley, Kai},
  title         = {Kuramoto Attention},
  year          = {2026},
  eprint        = {2606.11585},
  archiveprefix = {arXiv},
}

@misc{fsn,
  author        = {Nunley, Kai},
  title         = {Frequency Selective Networks},
  year          = {2026},
  eprint        = {2606.18694},
  archiveprefix = {arXiv},
}

@article{goodwin1967,
  author        = {Goodwin, Brian C.},
  title         = {Oscillatory behavior in enzymatic control processes},
  year          = {1967},
  doi           = {10.1016/0022-5193(67)90051-3},
}

@online{unconventional-ai,
  author        = {{Unconventional AI}},
  title         = {Un-0},
  year          = {2026},
  url           = {https://unconventional.ai/un-0},
}

@online{unconventional-ai-code,
  author        = {{Unconventional AI}},
  title         = {Un-0, source},
  year          = {2026},
  url           = {https://github.com/unconventional-ai/un-0},
}

@article{hopfield1982,
  author        = {Hopfield, John J.},
  title         = {Neural networks and physical systems},
  year          = {1982},
  journal       = {PNAS},
}
"""

#: what the manuscript calls each of them. "Nunley 2026" and "Unconventional AI
#: 2026" each name two works, which is the whole problem: only one of the pair
#: can own the label-derived key.
CITEMAP = {
    "kuramoto-attention": ["Kuramoto Attention", "Nunley 2026"],
    "fsn": ["FSN"],
    "goodwin1967": ["Goodwin 1967"],
    "unconventional-ai": ["Un-0", "Unconventional AI 2026"],
    "unconventional-ai-code": ["the Un-0 repository"],
    "hopfield1982": ["Hopfield 1982"],
}

KNOWN = alias_map(CITEMAP)
BY_URL = url_index(BIB)


def rewrite(text: str) -> tuple[str, list[str]]:
    """The rewritten text and the disagreements, dropping the count."""
    out, _, warnings = rewrite_links(text, KNOWN, BY_URL)
    return out, warnings


# --- the collisions that motivated resolving by URL ---------------------------

def test_two_works_sharing_one_label_resolve_to_different_keys():
    text, _ = rewrite(
        "attention ([Nunley 2026](https://arxiv.org/abs/2606.11585)) and "
        "selectivity ([Nunley 2026](https://arxiv.org/abs/2606.18694))")
    assert "[@{kuramoto-attention}]" in text
    assert "[@{fsn}]" in text


def test_a_blog_post_and_its_repository_do_not_collapse():
    text, _ = rewrite(
        "[Unconventional AI 2026](https://unconventional.ai/un-0) released "
        "[Unconventional AI 2026](https://github.com/unconventional-ai/un-0)")
    assert "@{unconventional-ai}" in text
    assert "@{unconventional-ai-code}" in text


def test_a_link_the_citemap_never_labelled_still_resolves():
    # the label derives no key at all; only the URL can place this citation
    text, warnings = rewrite("[the second paper](https://arxiv.org/abs/2606.18694)")
    assert text == "@{fsn}"
    assert warnings == []


# --- the forms one URL is written in ------------------------------------------

def test_a_percent_encoded_doi_resolves():
    # the manuscript escapes the parentheses; the .bib field does not
    text, _ = rewrite("[an enzyme model](https://doi.org/10.1016/0022-5193%2867%2990051-3)")
    assert text == "@{goodwin1967}"


def test_an_arxiv_link_resolves_through_the_eprint_field():
    text, _ = rewrite("[the attention paper](https://arxiv.org/abs/2606.11585)")
    assert text == "@{kuramoto-attention}"


def test_an_arxiv_version_and_a_pdf_link_are_the_same_work():
    assert url_key("https://arxiv.org/abs/2606.11585v3") == "arxiv:2606.11585"
    assert url_key("https://arxiv.org/pdf/2606.11585.pdf") == "arxiv:2606.11585"


def test_scheme_and_trailing_slash_carry_no_meaning():
    text, _ = rewrite("[the Un-0 post](http://unconventional.ai/un-0/)")
    assert text == "@{unconventional-ai}"


def test_a_doi_url_and_a_bare_doi_are_one_key():
    assert url_key("https://doi.org/10.1162/NECO") == url_key("10.1162/neco")


def test_a_link_and_an_identifier_of_different_kinds_never_collide():
    # a bare 10.x DOI and a path that happens to look like one are not the same
    assert url_key("10.1016/0022-5193(67)90051-3").startswith("doi:")
    assert url_key("https://example.org/10.1016/0022-5193").startswith("url:")


# --- the fallback, which is what keeps working manuscripts working ------------

def test_an_unknown_url_falls_back_to_the_label():
    text, warnings = rewrite("[Hopfield 1982](https://example.org/no-such-record)")
    assert text == "@{hopfield1982}"
    assert warnings == []


def test_an_entry_with_no_identifier_still_resolves_by_label():
    # hopfield1982 carries no doi, eprint or url, so nothing indexes it
    assert not [k for k, v in BY_URL.items() if v == "hopfield1982"]
    text, _ = rewrite("[Hopfield 1982](https://doi.org/10.1073/pnas.79.8.2554)")
    assert text == "@{hopfield1982}"


def test_a_bracket_citation_has_no_url_and_resolves_anyway():
    text, hits = rewrite_brackets("as [Hopfield 1982] framed it", KNOWN)
    assert text == "as [@hopfield1982] framed it"
    assert hits == 1


def test_a_link_that_is_neither_a_known_url_nor_a_known_label_is_left_alone():
    source = "[the TMLR author guide](https://example.org/guide)"
    text, warnings = rewrite(source)
    assert text == source
    assert warnings == []


# --- what gets reported --------------------------------------------------------

def test_a_label_that_disagrees_with_its_url_is_reported():
    _, warnings = rewrite("[Nunley 2026](https://arxiv.org/abs/2606.18694)")
    assert len(warnings) == 1
    assert "@kuramoto-attention" in warnings[0] and "@fsn" in warnings[0]


def test_the_link_wins_the_disagreement():
    text, _ = rewrite("[Nunley 2026](https://arxiv.org/abs/2606.18694)")
    assert text == "@{fsn}"


def test_one_disagreement_cited_many_times_is_reported_once():
    _, warnings = rewrite("[Nunley 2026](https://arxiv.org/abs/2606.18694) "
                          "and again [Nunley 2026](https://arxiv.org/abs/2606.18694)")
    assert len(warnings) == 1


def test_a_link_carrying_a_second_doi_for_the_labelled_work_is_reported():
    # the live case: the label names an entry whose own DOI is a different paper,
    # and the linked DOI is in no entry at all, so nothing resolves it
    _, warnings = rewrite("[Goodwin 1967](https://doi.org/10.1038/s41467-023-37190-9)")
    assert len(warnings) == 1
    assert "10.1038/s41467-023-37190-9" in warnings[0] and "@goodwin1967" in warnings[0]


def test_an_arxiv_link_to_a_work_the_entry_records_by_doi_is_not_reported():
    # linking the preprint of a work whose entry records the published DOI is
    # the ordinary case and must stay quiet, or the real warnings drown
    _, warnings = rewrite("[Goodwin 1967](https://arxiv.org/abs/1234.56789)")
    assert warnings == []


# --- the index itself ----------------------------------------------------------

def test_every_identifier_field_indexes_its_entry():
    assert BY_URL["arxiv:2606.11585"] == "kuramoto-attention"
    assert BY_URL["doi:10.1016/0022-5193(67)90051-3"] == "goodwin1967"
    assert BY_URL["url:unconventional.ai/un-0"] == "unconventional-ai"


def test_an_entry_url_written_as_a_doi_link_indexes_as_a_doi():
    # some entries record the DOI in `url`; a manuscript linking it must match
    index = url_index("@misc{x, url = {https://doi.org/10.1162/neco}}")
    assert index == {"doi:10.1162/neco": "x"}


def test_an_entry_answers_to_every_identifier_it_carries():
    links = entry_links({"doi": "10.1162/neco", "eprint": "2606.11585",
                         "url": "https://e.org/z"})
    assert links == ["doi:10.1162/neco", "arxiv:2606.11585", "url:e.org/z"]


def test_two_entries_holding_one_identifier_are_reported():
    twice = """
@misc{first,  eprint = {2606.11585}}
@misc{second, eprint = {arXiv:2606.11585v2}}
"""
    assert shared_links(twice) == [("arxiv:2606.11585", "first", "second")]


def test_a_bibliography_with_no_duplicate_identifiers_is_quiet():
    assert shared_links(BIB) == []

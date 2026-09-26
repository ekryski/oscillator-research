"""Which work a citation resolves to, when the label alone cannot say.

Every case here is a real one from this repository, reduced to a fixture. Two
works answering to "Nunley 2026", a blog post and the repository beside it both
called "Unconventional AI 2026", an old DOI whose parentheses the manuscript
percent-encodes. The label collapses them; the link does not, so the link is
what resolution reads first.
"""

from check_citemap import label_only, shared_links
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


class TestArxivIdInANote:
    """Most DBLP entries put the venue page in `url` and the preprint in `note`.

    A manuscript that cites the preprint then matches the entry on nothing but
    its author-year label, which is the path a mislabelled entry slips through.
    """

    def test_a_preprint_note_is_an_identifier(self):
        fields = {"url": "https://openreview.net/forum?id=F3s69XzWOia",
                  "note": "Preprint: arXiv:2010.00951"}
        assert "arxiv:2010.00951" in entry_links(fields)

    def test_an_old_style_archive_id_is_matched(self):
        assert "arxiv:cond-mat/0210694" in entry_links(
            {"note": "Preprint: arXiv:cond-mat/0210694"})

    def test_a_version_suffix_is_dropped(self):
        assert "arxiv:2511.08094" in entry_links(
            {"note": "Preprint: arXiv:2511.08094v2"})

    def test_a_note_with_no_preprint_adds_nothing(self):
        assert entry_links({"note": "Published as a poster."}) == []

    def test_the_note_id_matches_the_link_a_manuscript_writes(self):
        # the whole point: both sides reduce to one key
        fields = {"url": "https://openreview.net/forum?id=x",
                  "note": "Preprint: arXiv:2010.00951"}
        assert url_key("https://arxiv.org/abs/2010.00951") in \
            entry_links(fields)

    def test_an_entry_still_answers_to_its_other_identifiers(self):
        fields = {"doi": "10.1016/j.neunet.2019.03.005",
                  "note": "Preprint: arXiv:1808.04962"}
        links = entry_links(fields)
        assert "doi:10.1016/j.neunet.2019.03.005" in links
        assert "arxiv:1808.04962" in links


class TestParenthesisedCitations:
    """A parenthesis whose content is citations must not nest its own.

    `([A](u))` was already handled. `([A](u); [B](u))` was not, and rendered
    "(A (2001); B (2002))" — a nested pair that ran through the whole paper,
    seven times, into the submitted PDF.
    """

    BIB = """@misc{a, title={A}, url={https://example.org/a}}
@misc{b, title={B}, url={https://example.org/b}}
"""
    KNOWN = {"a": "a", "b": "b"}

    def rewrite(self, text):
        return rewrite_links(text, self.KNOWN, url_index(self.BIB))[0]

    def test_a_single_citation_in_parentheses_is_bracketed(self):
        assert self.rewrite("text ([A 2001](https://example.org/a)).") == "text [@{a}]."

    def test_two_citations_in_one_parenthesis_are_one_bracketed_group(self):
        out = self.rewrite("text ([A 2001](https://example.org/a); [B 2002](https://example.org/b)).")
        assert out == "text [@{a}; @{b}]."

    def test_three_citations_in_one_parenthesis(self):
        out = self.rewrite("([A 2001](https://example.org/a); [B 2002](https://example.org/b); "
                           "[A 2001](https://example.org/a))")
        assert out == "[@{a}; @{b}; @{a}]"

    def test_a_trailing_qualifier_is_kept_as_a_suffix(self):
        # "(…, preprint)" is a real pattern in the manuscript
        out = self.rewrite("text ([A 2001](https://example.org/a), preprint).")
        assert out == "text [@{a}, preprint]."

    def test_a_citation_outside_parentheses_stays_narrative(self):
        # "[A 2001](u) shows" must read "A (2001) shows", not "(A, 2001) shows"
        assert self.rewrite("[A 2001](https://example.org/a) shows") == "@{a} shows"

    def test_a_parenthesis_mixing_prose_between_citations_is_left_alone(self):
        # the parentheses there are the author's own and carry meaning
        src = "([A 2001](https://example.org/a); as restated by [B 2002](https://example.org/b))"
        assert self.rewrite(src) == "(@{a}; as restated by @{b})"

    def test_an_unresolved_label_leaves_the_parenthesis_untouched(self):
        src = "([Nobody 1999](https://example.org/zz))"
        assert self.rewrite(src) == src


class TestLabelOnlyCitations:
    """Citations nothing can verify: no identifier links them to their entry.

    This is the path the Zhang 2023 mis-citation took. A label fits any work by
    those authors in that year, so if the entry holds a different one, nothing
    in the build notices.
    """

    BIB = """@misc{a, title={A}, doi={10.1000/a}}
"""

    def check(self, text):
        return label_only(text, self.BIB)

    def test_a_link_matching_an_entry_is_not_reported(self):
        assert self.check("text ([A 2001](https://doi.org/10.1000/a)).") == []

    def test_a_link_matching_no_entry_is_reported(self):
        assert self.check("([A 2001](https://doi.org/10.9999/zz))") == [
            ("A 2001", "https://doi.org/10.9999/zz")]

    def test_a_bracket_citation_is_reported(self):
        # it carries no link at all, so nothing checks the entry is the work meant
        assert self.check("as shown [A 2001].") == [("A 2001", "(no link: bracket citation)")]

    def test_each_work_in_a_multi_work_bracket_is_reported(self):
        out = self.check("[A 2001; B 2002]")
        assert out == [("A 2001", "(no link: bracket citation)"),
                       ("B 2002", "(no link: bracket citation)")]

    def test_the_label_half_of_a_link_is_not_a_bracket_citation(self):
        # scanning raw text reports every link citation in the paper, since
        # "[Label](url)" opens with a bracket of its own
        assert self.check("text ([A 2001](https://doi.org/10.1000/a)) more") == []

    def test_notation_that_is_not_author_year_is_left_alone(self):
        # the manuscript writes matrix shapes the same way
        assert self.check("rows [T x 16] at 62.5 fps") == []

    def test_an_already_rewritten_citation_is_left_alone(self):
        assert self.check("text [@a] more") == []

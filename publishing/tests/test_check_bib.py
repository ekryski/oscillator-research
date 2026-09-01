"""What counts as a complete bibliography entry."""

from check_bib import problems

TECHREPORT = {"author": "Jaeger, Herbert", "title": "The echo state approach",
              "year": "2001", "institution": "German National Research Center",
              "url": "https://example.org/x.pdf"}


def test_a_techreport_is_published_by_its_institution():
    # not a journal or a booktitle, but a venue all the same
    assert problems("techreport", TECHREPORT) == []


def test_a_thesis_is_published_by_its_school():
    thesis = {**TECHREPORT, "school": "ETH"}
    del thesis["institution"]
    assert problems("phdthesis", thesis) == []


def test_a_journal_article_with_no_venue_is_still_incomplete():
    bare = {k: v for k, v in TECHREPORT.items() if k != "institution"}
    assert "no venue" in problems("article", bare)


def test_a_preprint_needs_no_venue():
    assert problems("misc", {"author": "A B", "title": "A real title here",
                             "year": "2024", "eprint": "2401.00001"}) == []


def test_an_online_source_needs_no_venue_or_year():
    assert problems("online", {"author": "Someone", "title": "Un-0",
                               "url": "https://example.org"}) == []

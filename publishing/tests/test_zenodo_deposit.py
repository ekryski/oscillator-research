"""The Zenodo record is built from the paper's own front matter, not retyped."""

import zenodo_deposit as zd

PAPER_YAML = """# front matter
author:
  - name: Eric Kryski
    affiliation: Independent researcher
    email: hello@example.org
    orcid: 0009-0005-7751-9059
  - name: Ada Lovelace
    affiliation: Analytical Engine Co.
keywords:
  - coupled oscillators
  - "Kuramoto model"
lang: en-GB
"""


def test_authors_become_zenodo_creators_in_family_given_order():
    creators = zd.authors(PAPER_YAML)
    assert creators[0] == {"name": "Kryski, Eric", "affiliation": "Independent researcher",
                           "orcid": "0009-0005-7751-9059"}
    assert creators[1] == {"name": "Lovelace, Ada", "affiliation": "Analytical Engine Co."}


def test_an_author_without_an_orcid_carries_no_orcid_key():
    assert "orcid" not in zd.authors(PAPER_YAML)[1]


def test_keywords_are_read_and_unquoted():
    assert zd.keywords(PAPER_YAML) == ["coupled oscillators", "Kuramoto model"]


def test_metadata_is_a_preprint_with_a_reserved_doi_and_the_repository_linked():
    meta = zd.build_metadata(PAPER_YAML, "A Survey", "It surveys things.", "2026-09-10",
                             repository="https://example.org/repo")
    assert meta["upload_type"] == "publication"
    assert meta["publication_type"] == "preprint"
    assert meta["title"] == "A Survey"
    assert meta["version"] == "2026-09-10"
    assert meta["license"] == "cc-by-4.0"
    assert meta["prereserve_doi"] is True
    assert meta["description"].startswith("<p>It surveys things.</p>")
    assert "https://example.org/repo" in meta["description"]
    assert meta["related_identifiers"][0]["identifier"] == "https://example.org/repo"


def test_a_front_matter_without_keywords_gives_an_empty_list():
    assert zd.keywords("author:\n  - name: Solo Author\n") == []

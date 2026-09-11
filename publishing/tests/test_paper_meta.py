"""paper_meta reads the flat paper.yaml the way the front matter is written."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paper_meta  # noqa: E402

SAMPLE = """\
# a comment
author:
  - name: "Ada Lovelace"
    affiliation: Analytical Engine Society, London, United Kingdom
    email: ada@example.org
    orcid: 0000-0000-0000-0001
    credit: Conceptualization, Writing – original draft
  - name: Charles Babbage
    affiliation: Analytical Engine Society, London, United Kingdom
shorttitle: Engines
highlights:
  - First point
  - Second point
funding: This research did not receive any specific grant.
keywords:
  - engines
  - "difference engines"
lang: en-GB
"""


def test_authors_carry_their_fields_in_order():
    people = paper_meta.authors(SAMPLE)
    assert [p["name"] for p in people] == ["Ada Lovelace", "Charles Babbage"]
    assert people[0]["email"] == "ada@example.org"
    assert people[0]["credit"].startswith("Conceptualization")
    assert "email" not in people[1]


def test_scalars_and_lists_read_by_key_and_drop_quotes():
    assert paper_meta.scalar(SAMPLE, "shorttitle") == "Engines"
    assert paper_meta.scalar(SAMPLE, "funding").endswith("specific grant.")
    assert paper_meta.scalar(SAMPLE, "missing") == ""
    assert paper_meta.items(SAMPLE, "keywords") == ["engines", "difference engines"]
    assert paper_meta.items(SAMPLE, "highlights") == ["First point", "Second point"]
    assert paper_meta.items(SAMPLE, "missing") == []


def test_a_list_stops_at_the_next_key():
    assert "funding" not in " ".join(paper_meta.items(SAMPLE, "highlights"))

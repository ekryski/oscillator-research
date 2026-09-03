"""front_matter renders the submission pieces a journal wants from paper.yaml."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import front_matter  # noqa: E402

META = """\
author:
  - name: Ada Lovelace
    affiliation: Analytical Engine Society, London, United Kingdom
    email: ada@example.org
    orcid: 0000-0000-0000-0001
    credit: Conceptualization, Writing – original draft
highlights:
  - Engines can be programmed
  - Programs can be stored on cards
  - A third point for the count
competing-interests: The author declares no competing interests.
funding: No grant.
data-availability: No data.
keywords:
  - engines
  - programs
"""


def test_author_lines_are_name_affiliation_and_correspondence():
    assert front_matter.author_lines(META) == [
        "Ada Lovelace",
        "Analytical Engine Society, London, United Kingdom",
        "Corresponding author: Ada Lovelace, ada@example.org (ORCID 0000-0000-0000-0001)",
    ]


def test_declarations_are_named_sections_at_the_manuscript_top_level():
    text = front_matter.declarations(META)
    assert "## CRediT authorship contribution statement {-}" in text
    assert "**Ada Lovelace:** Conceptualization, Writing – original draft." in text
    assert text.index("Declaration of competing interest") < text.index("## Funding") < text.index("## Data availability")


def test_docx_body_puts_keywords_first_and_declarations_last():
    out = front_matter.docx_body(META, "## 1 Introduction\n\nText.\n")
    assert out.startswith("**Keywords:** engines; programs\n\n## 1 Introduction")
    assert out.rstrip().endswith("No data.")


def test_highlights_render_as_a_file_of_their_own():
    text = front_matter.highlights(META)
    assert text.startswith("# Highlights\n")
    assert text.count("\n- ") == 3


def test_highlights_over_the_limits_fail_the_build():
    too_long = META.replace("- Engines can be programmed",
                            "- " + "x" * (front_matter.HIGHLIGHT_MAX_CHARS + 1))
    with pytest.raises(SystemExit, match="over the 85 allowed"):
        front_matter.highlights(too_long)
    too_few = META.replace("  - Programs can be stored on cards\n  - A third point for the count\n", "")
    with pytest.raises(SystemExit, match="wants 3 to 5"):
        front_matter.highlights(too_few)


def test_a_paper_without_highlights_gets_no_file():
    assert front_matter.highlights("author:\n  - name: A\n") == ""


def test_title_page_marks_affiliations_and_defaults_acknowledgements():
    text = front_matter.title_page(META)
    assert "**Authors.** Ada Lovelace^a^" in text
    assert "**Affiliations.** ^a^ Analytical Engine Society" in text
    assert "**Acknowledgements.** None." in text
    assert "**Funding.** No grant." in text

"""Recomputing the section numbers pandoc will assign, and checking references."""

from check_sections import float_counts, issues, numbering

DOC = """## Abstract

One paragraph.

## Introduction

### Scope

### Terms

## Method

### Setup

#### Detail

### Teardown

#### Use of AI assistance {-}

<!-- appendix -->

## First table

## Second table

### A subsection of it
"""


def test_the_abstract_is_not_a_numbered_section():
    # the build lifts it into metadata, so Introduction is section 1
    assert numbering(DOC)["1"] == "Introduction"


def test_subsections_continue_their_parent_count():
    n = numbering(DOC)
    assert (n["1.1"], n["1.2"], n["2.1"], n["2.2"]) == ("Scope", "Terms", "Setup", "Teardown")


def test_a_deeper_heading_nests_rather_than_restarting():
    assert numbering(DOC)["2.1.1"] == "Detail"


def test_an_unnumbered_heading_is_skipped():
    assert "Use of AI assistance {-}" not in numbering(DOC).values()


def test_the_appendix_is_lettered_not_numbered():
    n = numbering(DOC)
    assert (n["A"], n["B"], n["B.1"]) == ("First table", "Second table", "A subsection of it")


def test_a_live_reference_is_not_reported():
    assert issues(DOC + "\nAs Section 2.1 shows, and Appendix B records.\n") == []


def test_a_reference_past_the_end_is_reported():
    found = issues(DOC + "\nSee Section 3.4 and Appendix Z.\n")
    assert [ref for ref, _ in found] == ["Section 3.4", "Appendix Z"]


def test_the_finding_carries_enough_prose_to_locate_it():
    (_, context), = issues(DOC + "\nthe distinctive phrase here, Section 9.\n")
    assert "distinctive phrase here" in context


FLOATS = DOC + """

![A figure caption.](a.png)

| a | b |
|---|---|
| 1 | 2 |

: A table caption.

![Another figure.](b.png)
"""


def test_figures_and_tables_are_counted_in_document_order():
    assert float_counts(FLOATS) == {"Figure": 2, "Table": 1}


def test_a_figure_number_past_the_last_figure_is_reported():
    assert [ref for ref, _ in issues(FLOATS + "\nAs Figure 5 shows.\n")] == ["Figure 5"]


def test_a_figure_number_within_range_is_not_reported():
    assert issues(FLOATS + "\nAs Figure 2 and Table 1 show.\n") == []


def test_an_emphasised_figure_reference_is_still_matched():
    assert [ref for ref, _ in issues(FLOATS + "\nsee (*Figure 9*).\n")] == ["Figure 9"]

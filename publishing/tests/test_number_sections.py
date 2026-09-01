"""Writing the build's section numbers into the headings, repeatably."""

from check_sections import numbering
from number_sections import renumber

DOC = """## Abstract

One paragraph.

## Introduction

### Scope

## Method

### Setup

#### Detail

#### Use of AI assistance {-}

<!-- appendix -->

## Appendix {-}

## First table

## Second table

### A subsection of it
"""


def headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


def test_body_headings_carry_the_number_the_build_would_print():
    assert headings(renumber(DOC)[0])[:5] == [
        "## Abstract",
        "## 1 Introduction",
        "### 1.1 Scope",
        "## 2 Method",
        "### 2.1 Setup",
    ]


def test_the_abstract_is_left_unnumbered():
    assert "## Abstract" in headings(renumber(DOC)[0])


def test_the_appendix_is_lettered():
    out = headings(renumber(DOC)[0])
    assert out[-3:] == ["## A First table", "## B Second table", "### B.1 A subsection of it"]


def test_an_unnumbered_heading_is_skipped_and_consumes_no_number():
    out = headings(renumber(DOC)[0])
    # the `{-}` heading keeps its text, and Detail before it still took 2.1.1
    assert "#### Use of AI assistance {-}" in out
    assert "#### 2.1.1 Detail" in out
    assert "## Appendix {-}" in out


def test_nothing_but_the_headings_moves():
    out, _ = renumber(DOC)
    # blank lines around a heading are structural in Markdown: welding a heading
    # to the paragraph below it would change how the document parses
    assert [line for line in out.splitlines() if not line.startswith("#")] == [
        line for line in DOC.splitlines() if not line.startswith("#")
    ]
    assert out.endswith("\n")


def test_running_twice_does_not_stack_numbers():
    once, first = renumber(DOC)
    twice, second = renumber(once)
    assert twice == once
    assert second == []
    assert first  # the first pass really did change something


def test_the_report_names_every_heading_that_moved():
    _, changed = renumber(DOC)
    assert ("## Introduction", "## 1 Introduction") in changed
    # unchanged headings are not reported
    assert all(before != "## Abstract" for before, _ in changed)


def test_inserting_a_section_renumbers_everything_after_it():
    numbered, _ = renumber(DOC)
    inserted = numbered.replace("## 2 Method", "## Background\n\n## 2 Method")
    out = headings(renumber(inserted)[0])
    assert "## 2 Background" in out
    assert "## 3 Method" in out
    assert "### 3.1 Setup" in out


def test_inserting_an_appendix_reletters_the_ones_after_it():
    numbered, _ = renumber(DOC)
    inserted = numbered.replace("## A First table", "## Preliminaries\n\n## A First table")
    out = headings(renumber(inserted)[0])
    assert out[-4:] == [
        "## A Preliminaries",
        "## B First table",
        "## C Second table",
        "### C.1 A subsection of it",
    ]


def test_a_deeper_heading_is_not_mistaken_for_a_number():
    # "### A subsection of it" opens with a bare letter, but at that depth a
    # lettered number would read "A.1", so the title survives untouched
    assert "### B.1 A subsection of it" in headings(renumber(DOC)[0])


def test_a_lone_appendix_titled_like_a_letter_is_not_stripped():
    doc = DOC.replace("## First table", "## A note on the sample")
    out = headings(renumber(doc)[0])
    assert "## A A note on the sample" in out


def test_a_stale_number_on_a_now_unnumbered_heading_is_removed():
    doc = DOC.replace("#### Use of AI assistance {-}", "#### 2.1.2 Use of AI assistance {-}")
    assert "#### Use of AI assistance {-}" in headings(renumber(doc)[0])


def test_the_written_numbers_are_the_ones_check_sections_resolves_against():
    out, _ = renumber(DOC)
    for number, title in numbering(out).items():
        assert title.startswith(f"{number} ")

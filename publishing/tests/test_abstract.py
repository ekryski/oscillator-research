"""Lifting the abstract out of the manuscript."""

import abstract

DOC = """## Abstract

First line of the abstract.
Second line, same paragraph.

## Introduction

Body text that mentions an abstract idea.
"""


def test_read_collapses_the_wrapping():
    assert abstract.read(DOC) == "First line of the abstract. Second line, same paragraph."


def test_strip_removes_the_section_and_nothing_else():
    out = abstract.strip(DOC)
    assert out.startswith("## Introduction")
    assert "Second line" not in out
    assert "an abstract idea" in out


def test_a_manuscript_without_one_is_left_alone():
    doc = "## Introduction\n\nText.\n"
    assert abstract.read(doc) == ""
    assert abstract.strip(doc) == doc


def test_the_word_abstract_in_prose_is_not_a_heading():
    doc = "## Introduction\n\nWe take an abstract view.\n\n## Method\n\nText.\n"
    assert abstract.read(doc) == ""


def test_yaml_is_a_block_scalar_so_punctuation_needs_no_quoting():
    y = abstract.as_yaml('A: colon, a "quote", and a #hash.')
    assert y.startswith("---\nabstract: |\n  ")
    assert '"' in y and ":" in y


def test_no_abstract_yields_no_metadata_file():
    assert abstract.as_yaml("") == ""


def test_strip_is_idempotent():
    once = abstract.strip(DOC)
    assert abstract.strip(once) == once

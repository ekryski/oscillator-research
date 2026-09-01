"""Lifting the title out of the manuscript so it renders once, from metadata."""

import title as title_mod

DOC = """# From Physics to Dynamics: A Survey

## Abstract

Coupled oscillator networks are drawing attention.

## 1 Introduction
"""


def test_the_title_is_read_from_the_top_heading():
    assert title_mod.read(DOC) == "From Physics to Dynamics: A Survey"


def test_the_title_is_removed_from_the_body():
    out = title_mod.strip(DOC)
    assert out.startswith("## Abstract")
    assert "From Physics to Dynamics" not in out


def test_a_manuscript_with_no_title_is_unchanged():
    body = "## Abstract\n\nText.\n"
    assert title_mod.read(body) == ""
    assert title_mod.strip(body) == body


def test_only_a_leading_heading_counts_as_the_title():
    # a `#` further down is not the title; the manuscripts start their own
    # sections at `##`, so nothing below the top should ever be lifted
    body = "## Abstract\n\nText.\n\n# Not the title\n"
    assert title_mod.read(body) == ""
    assert title_mod.strip(body) == body


def test_a_section_heading_is_not_mistaken_for_the_title():
    assert title_mod.read("## 1 Introduction\n") == ""


def test_leading_blank_lines_are_allowed():
    assert title_mod.read("\n\n# A Title\n\n## Abstract\n") == "A Title"


def test_the_yaml_quotes_a_title_containing_a_colon():
    # every title here has one, and bare `a: b` is a mapping to YAML
    out = title_mod.as_yaml("From A: A Survey")
    assert out == '---\ntitle: "From A: A Survey"\n---\n'


def test_the_yaml_escapes_a_quote_in_the_title():
    assert title_mod.as_yaml('The "Edge" of Chaos') == \
        '---\ntitle: "The \\"Edge\\" of Chaos"\n---\n'


def test_no_title_writes_no_metadata():
    # an empty metadata file is valid and overrides nothing; a `title: ""`
    # would blank the title block instead
    assert title_mod.as_yaml("") == ""

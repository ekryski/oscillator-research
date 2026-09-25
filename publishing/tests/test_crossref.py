from crossref import anchor, link
from preprocess import APPENDIX

DOC = """## 1 Introduction

As Section 2.1 shows, and Appendix A.2, A.1 and B confirm (see Section 9).

## 2 Methods

### 2.1 Design

## Reproducibility statement {-}

<!-- appendix -->

## A Glossary

### A.1 The pipeline

### A.2 Arms {#arms}

## B Design

Back to Section 2 and `Section 2.1`, and [Section 2.1](#elsewhere).
"""


def test_numbered_headings_get_ids_and_an_own_id_is_kept():
    text, ids = anchor(DOC, APPENDIX)
    assert ids == {"1": "sec-1", "2": "sec-2", "2.1": "sec-2-1", "A": "sec-A", "A.1": "sec-A-1",
                   "A.2": "arms", "B": "sec-B"}
    assert "### 2.1 Design {#sec-2-1}" in text and "### A.2 Arms {#arms}" in text
    assert "## Reproducibility statement {-}" in text


def test_references_and_their_lists_become_links_and_a_dangling_one_stays_text():
    text, ids = anchor(DOC, APPENDIX)
    out, n = link(text, ids)
    assert ("As [Section 2.1](#sec-2-1) shows, and [Appendix A.2](#arms), [A.1](#sec-A-1) and [B](#sec-B) "
            "confirm (see Section 9).") in out
    assert n == 5


def test_code_existing_links_and_headings_are_left_alone():
    text, ids = anchor(DOC, APPENDIX)
    out, _ = link(text, ids)
    assert "[Section 2](#sec-2) and `Section 2.1`, and [Section 2.1](#elsewhere)." in out
    assert "### 2.1 Design {#sec-2-1}" in out

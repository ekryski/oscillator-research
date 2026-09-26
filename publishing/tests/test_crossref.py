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


FIGS = """Intro (Figure b2-two) and Figures a1-one and b2-two, and Figure z9-missing; Figure fig03-sec4-3-three.

![First, with a [link](https://x.org).](resources/figures/a1-one.png)

![Second.](resources/figures/b2-two.png){width=90%}

![Third.](resources/figures/fig03-sec4-3-three.png)
"""


def test_figures_are_numbered_in_order_and_named_references_link_to_them():
    from crossref import link_figures, number_figures
    text, numbers = number_figures(FIGS)
    assert numbers == {"a1-one": 1, "b2-two": 2, "fig03-sec4-3-three": 3}
    assert "![First, with a [link](https://x.org).](resources/figures/a1-one.png){#fig-a1-one}" in text
    assert "![Second.](resources/figures/b2-two.png){#fig-b2-two width=90%}" in text
    out, n = link_figures(text, numbers)
    assert ("Intro ([Figure 2](#fig-b2-two)) and [Figures 1](#fig-a1-one) and [2](#fig-b2-two), and Figure z9-missing; "
            "[Figure 3](#fig-fig03-sec4-3-three).") in out
    assert n == 4

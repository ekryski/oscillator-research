"""Numbering figure and table captions in the formats pandoc leaves unnumbered.

These run the real filter through pandoc rather than asserting on its source,
because the thing that breaks is the AST contract: pandoc 3 renamed the Image
walk to Figure once, and a filter that silently matches nothing still exits 0.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

FILTER = Path(__file__).resolve().parents[1] / "filters" / "number-floats.lua"

pytestmark = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")

FIGURE = "![A caption.](resources/figures/a.png)"

TABLE = """| System | Result |
|---|---|
| ESN | 0.9 |

: A table caption."""


def doc(*blocks: str) -> str:
    """Join blocks with the blank line pandoc needs to keep them separate."""
    return "\n\n".join(blocks) + "\n"


def render(src: str, to: str = "html5") -> str:
    """Run pandoc over `src` with only this filter applied."""
    return subprocess.run(
        ["pandoc", "--from=markdown+pipe_tables", f"--to={to}", f"--lua-filter={FILTER}"],
        input=src, capture_output=True, text=True, check=True,
    ).stdout


def test_a_figure_caption_is_prefixed():
    assert "Figure 1. A caption." in render(FIGURE)


def test_a_table_caption_is_prefixed():
    # the manuscript says "every paper in Table 1"; without this the HTML
    # reader has no table labelled 1 to look for
    assert "Table 1. A table caption." in render(TABLE)


def test_figures_and_tables_count_separately():
    out = render(doc(FIGURE, TABLE, FIGURE))
    assert "Figure 1." in out and "Figure 2." in out
    assert "Table 1." in out and "Table 2." not in out


def test_each_float_type_counts_in_document_order():
    out = render(doc(TABLE, FIGURE, TABLE))
    assert out.index("Table 1.") < out.index("Figure 1.") < out.index("Table 2.")


def test_latex_is_left_to_number_its_own_floats():
    # LaTeX writes "Figure 1:" itself; prefixing here gives "Figure 1: Figure 1."
    out = render(doc(FIGURE, TABLE), to="latex")
    assert "Figure 1." not in out and "Table 1." not in out


def test_an_uncaptioned_table_does_not_consume_a_number():
    # LaTeX only advances the counter on \caption, so an uncaptioned float must
    # not shift the numbers of the captioned ones either
    uncaptioned = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    assert "Table 1. A table caption." in render(doc(uncaptioned, TABLE))

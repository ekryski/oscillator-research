"""Authored figures keep their Unicode math; the render copy sets it as positioned runs."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import svg_render as sr  # noqa: E402
from svg_render import Run  # noqa: E402

NS = {"s": sr.SVG_NS}


def fixed_width(text, size, bold=False, italic=False):
    """A stand-in font: every character is half its size wide, so sums are easy to check."""
    return len(text) * size * 0.5


def every_glyph(ch, bold=False, italic=False):
    return True


def svg_with(*texts: str) -> str:
    return f'<svg xmlns="{sr.SVG_NS}" width="400" height="100">' + "".join(texts) + "</svg>"


def test_script_letters_become_plain_letters_at_their_level():
    assert sr.runs("θᵢ") == [Run("θ"), Run("i", -1)]
    assert sr.runs("ΦᵀΦ") == [Run("Φ"), Run("T", +1), Run("Φ")]


def test_neighbouring_scripts_share_one_run():
    # "to the minus one" is a single raised "−1", not two fragments set apart
    assert sr.runs("A⁻¹") == [Run("A"), Run("−1", +1)]
    assert sr.runs("Kᵢⱼ") == [Run("K"), Run("ij", -1)]


def test_a_symbol_the_font_lacks_is_swapped_for_one_it_has():
    assert sr.runs("ℝᵈ") == [Run("R", 0, True), Run("d", +1)]
    assert sr.runs("x ∈ y") == [Run("x in y")]


def test_a_combining_mark_is_refused_with_the_way_out():
    why = sr.unrenderable("θ̇", every_glyph)
    assert len(why) == 1 and "COMBINING DOT ABOVE" in why[0] and "dθ/dt" in why[0]


def test_a_character_with_no_glyph_is_named():
    why = sr.unrenderable("a⟪b", lambda ch: ch != "⟪")
    assert len(why) == 1 and "no glyph" in why[0] and "⟪" in why[0]
    assert sr.unrenderable("plain text", every_glyph) == []


def test_a_centred_line_is_centred_on_its_anchor_by_its_true_width():
    # "ab" at size 10 is 10 wide; the subscript "i" at 0.7 is 3.5 wide: 13.5 in all.
    # It ends the line, so no kern follows it and the line is centred on its ink.
    placed = sr.layout([Run("ab"), Run("i", -1)], x=100, y=50, size=10, anchor="middle",
                       width_of=fixed_width)
    (x0, y0, s0, r0), (x1, y1, s1, r1) = placed
    assert x0 == pytest.approx(100 - 13.5 / 2) and x1 == pytest.approx(x0 + 10)
    assert (s0, s1) == (10, pytest.approx(7.0))
    assert y0 == 50 and y1 == pytest.approx(50 + 10 * sr.SUB_DROP)


def test_a_superscript_rises_and_an_end_anchor_ends_at_x():
    placed = sr.layout([Run("W"), Run("T", +1)], x=200, y=40, size=10, anchor="end",
                       width_of=fixed_width)
    assert placed[0][0] == pytest.approx(200 - (5 + 3.5))
    assert placed[1][1] == pytest.approx(40 - 10 * sr.SUPER_RISE)


def test_leading_spaces_are_stepped_over_not_drawn():
    # a renderer drops whitespace at the start of a <text>, so the run must
    # start where its first visible character belongs
    placed = sr.layout([Run("a"), Run("i", -1), Run("  b")], x=0, y=0, size=10,
                       anchor="start", width_of=fixed_width)
    assert placed[2][3].text == "b"
    assert placed[2][0] == pytest.approx(5 + 3.5 + 10 * sr.SUB_KERN + 2 * 5)


def test_a_thin_space_follows_a_subscript_but_not_a_superscript_or_the_last_run():
    # an italic slash leans back under a subscript and swallows it without the space
    after_sub = sr.layout([Run("a"), Run("i", -1), Run("/")], 0, 0, 10, "start", fixed_width)
    after_sup = sr.layout([Run("a"), Run("T", +1), Run("/")], 0, 0, 10, "start", fixed_width)
    assert after_sub[2][0] - after_sup[2][0] == pytest.approx(10 * sr.SUB_KERN)


def test_only_lines_that_need_it_are_rewritten():
    out = sr.normalize(svg_with('<text x="10" y="20" font-size="12">plain label</text>',
                                '<text x="200" y="60" font-size="10" text-anchor="middle" '
                                'fill="#123" font-style="italic">θᵢ</text>'),
                       fixed_width, every_glyph)
    texts = ET.fromstring(out).findall(".//s:text", NS)
    assert [t.text for t in texts] == ["plain label", "θ", "i"]
    assert texts[0].get("x") == "10"                        # untouched
    assert all(t.get("fill") == "#123" and t.get("font-style") == "italic" for t in texts[1:])
    assert not any(ch in sr.SUBSCRIPT for t in texts for ch in t.text)


def test_the_source_text_is_never_altered_only_the_copy():
    source = svg_with('<text x="1" y="2" font-size="10">Φᵀ</text>')
    sr.normalize(source, fixed_width, every_glyph)
    assert "Φᵀ" in source


def test_every_unrenderable_line_is_reported_at_once():
    with pytest.raises(ValueError) as err:
        sr.normalize(svg_with('<text x="1" y="2" font-size="10">θ̇ᵢ</text>',
                              '<text x="1" y="9" font-size="10">K̂</text>'),
                     fixed_width, every_glyph)
    assert "DOT ABOVE" in str(err.value) and "CIRCUMFLEX" in str(err.value)


def test_nested_markup_is_refused_rather_than_mislaid():
    with pytest.raises(ValueError, match="child elements"):
        sr.normalize(svg_with('<text x="1" y="2" font-size="10">a<tspan>ᵢ</tspan></text>'),
                     fixed_width, every_glyph)


def test_a_script_line_without_its_own_geometry_is_refused():
    with pytest.raises(ValueError, match="numeric x, y and font-size"):
        sr.normalize(svg_with('<text x="1" y="2">θᵢ</text>'), fixed_width, every_glyph)


def test_a_real_render_writes_both_files(tmp_path):
    pytest.importorskip("reportlab"); pytest.importorskip("svglib")
    if not all((sr.FONT_DIR / f).exists() for f in sr.FONT_FILES.values()):
        pytest.skip("the render fonts are not on this machine")
    import shutil
    if shutil.which("pdftoppm", path=sr.POPPLER_PATH) is None:
        pytest.skip("poppler is not installed")
    svg = tmp_path / "fig.svg"
    svg.write_text(svg_with('<rect width="400" height="100" fill="#fff"/>',
                            '<text x="200" y="50" font-size="14" text-anchor="middle">'
                            'W = (ΦᵀΦ + λI)⁻¹ΦᵀY, x ∈ ℝᵈ</text>'), encoding="utf-8")
    pdf, png = sr.render(svg)
    assert pdf.stat().st_size > 0 and png.stat().st_size > 0

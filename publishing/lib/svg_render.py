#!/usr/bin/env python3
"""Render an authored SVG to the vector PDF and the PNG the builds use.

A hand-drawn figure says θᵢ and Φᵀ the natural way, with Unicode subscript and
superscript letters. No ordinary font carries all of them, and a renderer that
lacks one prints a box where the symbol should be and says nothing. The
manuscripts already solve the same problem for prose: they keep their Unicode
math, and only the LaTeX path converts it. This does that for figures.

The SVG on disk is left as the author wrote it. At render time a temporary copy
is normalized: every subscript and superscript becomes an ordinary letter set
small and shifted, each run placed at an x computed from the font's own
metrics, and the few symbols the font lacks are swapped for ones it has. The
copy is then rendered by svglib, the same cairo-free path paper 01's figures
take. Anything with no faithful equivalent is refused with a message, not
guessed at: a combining accent such as the dot in θ̇ needs text shaping that
svglib does not do, so the author writes dθ/dt instead.

    uv run --with svglib --with reportlab python3 publishing/lib/svg_render.py \\
        papers/02-untrained-reservoirs/resources/figures

Runs are positioned one by one, not as <tspan>s inside one <text>, because
svglib mislays spans in a centred line (they overprint) and over-advances after
a small span in a left-aligned one. Absolute placement depends on neither.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
#: Unicode subscript and superscript letters, and the plain letter each stands for
SUBSCRIPT = {"ᵢ": "i", "ⱼ": "j", "ᵣ": "r", "ₜ": "t", "ₙ": "n", "ₖ": "k", "ₘ": "m",
             "₀": "0", "₁": "1", "₂": "2", "₃": "3"}
SUPERSCRIPT = {"ᵀ": "T", "ᵈ": "d", "ⁿ": "n", "ⁱ": "i", "⁻": "−", "⁺": "+",
               "⁰": "0", "¹": "1", "²": "2", "³": "3"}
#: symbols the render font lacks, and what stands in: (replacement, set in bold)
SUBSTITUTE = {"ℝ": ("R", True), "∈": ("in", False), "⟨": ("‹", False), "⟩": ("›", False)}

#: a raised or lowered run is set at this fraction of the line's size
SCRIPT_SCALE = 0.7
#: how far a subscript drops and a superscript rises, as fractions of the line's size
SUB_DROP, SUPER_RISE = 0.28, 0.38
#: the thin space after a subscript, as a fraction of the line's size. An italic
#: slash or parenthesis leans back under the glyph before it, which is exactly
#: where a subscript sits: without this, the i of dθᵢ/dt disappears into the slash.
SUB_KERN = 0.1

#: reportlab's built-in Helvetica is Latin-1 only. The metrically identical
#: Arial files under the same names carry Greek and the math operators.
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONT_FILES = {"Helvetica": "Arial.ttf", "Helvetica-Bold": "Arial Bold.ttf",
              "Helvetica-Oblique": "Arial Italic.ttf",
              "Helvetica-BoldOblique": "Arial Bold Italic.ttf"}
#: presentation attributes a positioned run inherits from the element it replaces
CARRIED = ("fill", "font-family", "font-style", "opacity", "fill-opacity")
PNG_DPI = 220
POPPLER_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"


@dataclass(frozen=True)
class Run:
    """A stretch of text at one level: on the line, lowered, or raised."""
    text: str
    level: int = 0          # -1 subscript, 0 on the line, +1 superscript
    bold: bool = False


def runs(text: str) -> list[Run]:
    """Split a line into runs, turning script letters into plain ones.

    Neighbouring characters at the same level and weight share a run, so "⁻¹"
    is one raised "−1" and not two fragments.
    """
    out: list[Run] = []
    for ch in text:
        if ch in SUBSCRIPT:
            piece = Run(SUBSCRIPT[ch], -1)
        elif ch in SUPERSCRIPT:
            piece = Run(SUPERSCRIPT[ch], +1)
        elif ch in SUBSTITUTE:
            stand_in, bold = SUBSTITUTE[ch]
            piece = Run(stand_in, 0, bold)
        else:
            piece = Run(ch)
        if out and (out[-1].level, out[-1].bold) == (piece.level, piece.bold):
            out[-1] = Run(out[-1].text + piece.text, piece.level, piece.bold)
        else:
            out.append(piece)
    return out


def needs_layout(text: str) -> bool:
    """Whether a line holds anything that has to be set as separate runs."""
    return any(ch in SUBSCRIPT or ch in SUPERSCRIPT or ch in SUBSTITUTE for ch in text)


def unrenderable(text: str, has_glyph: Callable[[str], bool]) -> list[str]:
    """Why this text cannot be rendered faithfully, one reason per character."""
    reasons = []
    for ch in dict.fromkeys(text):
        name = unicodedata.name(ch, f"U+{ord(ch):04X}")
        if unicodedata.combining(ch):
            reasons.append(f"{name}: a combining mark needs text shaping this renderer "
                           f"does not do; write the explicit form, such as dθ/dt for θ̇")
        elif not ch.isspace() and not has_glyph(ch):
            reasons.append(f"{name} ({ch!r}): the render font has no glyph for it")
    return reasons


def layout(pieces: list[Run], x: float, y: float, size: float, anchor: str,
           width_of: Callable[[str, float, bool], float]) -> list[tuple[float, float, float, Run]]:
    """Place each run: (x, y, font size, run), honouring the line's anchor.

    Leading spaces are stepped over and not drawn, because a renderer drops
    whitespace at the start of a <text> and the run would land early.
    """
    sized = [(p, size * SCRIPT_SCALE if p.level else size) for p in pieces]
    # the kern separates a subscript from what follows, so the last run gets none
    kerns = [size * SUB_KERN if p.level < 0 and i < len(pieces) - 1 else 0.0
             for i, p in enumerate(pieces)]
    total = sum(width_of(p.text, s, p.bold) for p, s in sized) + sum(kerns)
    cursor = x - {"start": 0.0, "middle": total / 2, "end": total}[anchor]
    placed = []
    for (piece, run_size), kern in zip(sized, kerns):
        lead = len(piece.text) - len(piece.text.lstrip(" "))
        shift = {0: 0.0, -1: size * SUB_DROP, +1: -size * SUPER_RISE}[piece.level]
        if piece.text.strip():
            placed.append((cursor + width_of(" " * lead, run_size, piece.bold), y + shift,
                           run_size, Run(piece.text.strip(" "), piece.level, piece.bold)))
        cursor += width_of(piece.text, run_size, piece.bold) + kern
    return placed


def _is_bold(el: ET.Element) -> bool:
    return el.get("font-weight", "normal") in ("bold", "600", "700", "800", "900")


def normalize(svg: str, width_of: Callable[[str, float, bool, bool], float],
              has_glyph: Callable[[str, bool, bool], bool]) -> str:
    """The SVG with every script-bearing line replaced by positioned runs.

    `width_of(text, size, bold, italic)` and `has_glyph(ch, bold, italic)` come
    from the render font, so placement and the glyph check use the same face
    the renderer will. Raises ValueError naming every line it cannot render.
    """
    ET.register_namespace("", SVG_NS)
    root = ET.fromstring(svg)
    parent_of = {child: parent for parent in root.iter() for child in parent}
    problems = []
    for el in list(root.iter(f"{{{SVG_NS}}}text")):
        text = el.text or ""
        if len(el):
            problems.append(f"{text!r}: already has child elements; only plain <text> is handled")
            continue
        italic = el.get("font-style") in ("italic", "oblique")
        pieces = runs(text)
        for piece in pieces:
            bold = piece.bold or _is_bold(el)
            problems += [f"{text!r}: {why}" for why in
                         unrenderable(piece.text, lambda ch, b=bold: has_glyph(ch, b, italic))]
        if not needs_layout(text):
            continue
        try:
            size, x, y = float(el.get("font-size")), float(el.get("x")), float(el.get("y"))
        except (TypeError, ValueError):
            problems.append(f"{text!r}: needs its own numeric x, y and font-size to be laid out")
            continue
        group = ET.Element(f"{{{SVG_NS}}}g")
        placed = layout(pieces, x, y, size, el.get("text-anchor", "start"),
                        lambda t, s, b: width_of(t, s, b or _is_bold(el), italic))
        for run_x, run_y, run_size, piece in placed:
            run_el = ET.SubElement(group, f"{{{SVG_NS}}}text", {
                "x": f"{run_x:.2f}", "y": f"{run_y:.2f}", "font-size": f"{run_size:.2f}"})
            for attr in CARRIED:
                if el.get(attr) is not None:
                    run_el.set(attr, el.get(attr))
            if piece.bold or _is_bold(el):
                run_el.set("font-weight", "bold")
            run_el.text = piece.text
        group.tail = el.tail
        holder = parent_of[el]
        holder.insert(list(holder).index(el), group)
        holder.remove(el)
    if problems:
        raise ValueError("cannot render faithfully:\n  " + "\n  ".join(dict.fromkeys(problems)))
    return ET.tostring(root, encoding="unicode")


def _face(bold: bool, italic: bool) -> str:
    return {(False, False): "Helvetica", (True, False): "Helvetica-Bold",
            (False, True): "Helvetica-Oblique", (True, True): "Helvetica-BoldOblique"}[(bold, italic)]


def register_fonts() -> None:
    """Put the Unicode faces under the names svglib resolves Helvetica to."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    missing = [f for f in FONT_FILES.values() if not (FONT_DIR / f).exists()]
    if missing:
        sys.exit(f"render fonts not found in {FONT_DIR}: {', '.join(missing)}")
    for name, filename in FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))


def render(svg_path: Path) -> tuple[Path, Path]:
    """Write <name>.pdf and <name>.png beside the SVG; returns both paths."""
    from reportlab.graphics import renderPDF
    from reportlab.pdfbase import pdfmetrics
    from svglib.svglib import svg2rlg
    register_fonts()

    def width_of(text: str, size: float, bold: bool, italic: bool) -> float:
        return pdfmetrics.stringWidth(text, _face(bold, italic), size)

    def has_glyph(ch: str, bold: bool, italic: bool) -> bool:
        return ord(ch) in pdfmetrics.getFont(_face(bold, italic)).face.charToGlyph

    try:
        normalized = normalize(svg_path.read_text(encoding="utf-8"), width_of, has_glyph)
    except ValueError as err:
        sys.exit(f"{svg_path}: {err}")
    pdf, png = svg_path.with_suffix(".pdf"), svg_path.with_suffix(".png")
    with tempfile.NamedTemporaryFile("w", suffix=".svg", encoding="utf-8", delete=False) as tmp:
        tmp.write(normalized)
    try:
        renderPDF.drawToFile(svg2rlg(tmp.name), str(pdf))
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    subprocess.run(["pdftoppm", "-png", "-r", str(PNG_DPI), "-singlefile", str(pdf),
                    str(png.with_suffix(""))], check=True, env={"PATH": POPPLER_PATH})
    return pdf, png


def main() -> None:
    ap = argparse.ArgumentParser(description="render authored SVG figures to PDF and PNG")
    ap.add_argument("targets", nargs="+", type=Path,
                    help="SVG files, or folders whose .svg files are all rendered")
    a = ap.parse_args()
    svgs = [p for t in a.targets for p in (sorted(t.glob("*.svg")) if t.is_dir() else [t])]
    if not svgs:
        sys.exit("no .svg files found")
    for svg in svgs:
        pdf, png = render(svg)
        print(f"{svg.name} -> {pdf.name}, {png.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rewrite a manuscript's reading-friendly citations into pandoc's `@key` form.

The manuscripts stay the editing surface: they cite the way a person wants to
read them — inline markdown links carrying a DOI, or [Author Year] brackets —
and this rewrite happens on a copy at build time. Nothing here ever writes back
to the source.

Two conversions, chosen so citeproc reproduces the sentence's original grammar:

    ([Kuramoto & Battogtokh 2002](url))  ->  [@kuramoto2002]     "(Kuramoto ... 2002)"
    [Hopfield (1982)](url) framed ...    ->  @hopfield1982 ...   "Hopfield (1982) framed ..."

`--appendix-out` splits the manuscript at an `<!-- appendix -->` marker and
writes the tail to its own file. TMLR puts the appendix after the references,
and its author guide excludes appendices from the length that risks delaying
review, so the split is what keeps the main body inside the two-week window.

`--for latex` additionally maps the manuscripts' bare Unicode Greek and math
characters onto LaTeX math. This is not cosmetic: the default TeX text font has
no glyph for theta, omega, or a subscript i, and the engine drops such
characters SILENTLY — an equation in the prose loses half its symbols and the
PDF still builds. Mapping them is also more correct than finding a font with
the coverage, because these characters really are mathematics. HTML, EPUB, and
DOCX keep the Unicode, which their readers handle natively.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract as abstract_mod
from extract_bib import BRACKET, LINK, NOT_A_CITATION, citekey, split_author_year

#: an image reference, whose source the LaTeX path needs as a vector PDF
#: a local one only: a remote image has no PDF beside it to swap to
#: a caption may itself contain a markdown link, so the alternation has to allow
#: one level of nested brackets. Without it the image is left pointing at the PNG,
#: the TMLR build copies only PDFs, and LaTeX drops the figure to a draft box while
#: the citation pass that follows it fails wholesale.
IMAGE_SRC = re.compile(r"(!\[(?:[^\]\[\\]|\\.|\[[^\]\[]*\])*\]\()(?!\w+:)([^)\s]+)\.png(\))",
                       re.S)


def to_vector_images(text: str) -> tuple[str, int]:
    """Point image references at the PDF beside each PNG.

    The manuscripts reference `resources/figures/x.png`, because that is what
    renders on GitHub and in every HTML-ish format. LaTeX should take the vector
    PDF built from the same SVG: a raster figure in a submission PDF is visibly
    worse, and reviewers zoom.
    """
    return IMAGE_SRC.subn(r"\1\2.pdf\3", text)


#: everything after this marker is appendix material
APPENDIX = re.compile(r"^<!--\s*appendix\s*-->\s*$", re.M)

# Unicode -> LaTeX math, for the characters the default TeX text font lacks.
# Latin letters with accents are NOT here: they render fine, and rewriting them
# would mangle author names in the prose.
SYMBOLS = {
    "θ": r"\theta", "ω": r"\omega", "φ": r"\varphi", "α": r"\alpha",
    "β": r"\beta", "λ": r"\lambda", "σ": r"\sigma", "ρ": r"\rho",
    "μ": r"\mu", "π": r"\pi", "Φ": r"\Phi", "Δ": r"\Delta", "Σ": r"\Sigma",
    "Γ": r"\Gamma", "Ω": r"\Omega", "τ": r"\tau", "ε": r"\epsilon",
    "γ": r"\gamma", "δ": r"\delta",
    "≤": r"\leq", "≥": r"\geq", "≈": r"\approx", "≲": r"\lesssim",
    "∈": r"\in", "∝": r"\propto", "×": r"\times", "·": r"\cdot",
    "±": r"\pm", "→": r"\to", "⇒": r"\Rightarrow", "−": "-",
    "⟨": r"\langle", "⟩": r"\rangle",
    # table marks: no glyph in the text font, and pdflatex errors on them
    "✓": r"\checkmark", "✗": r"\times",
}
# kept apart from SYMBOLS so that runs of them coalesce: "Kᵢⱼ" has to become
# K_{ij} and not K_i_j, which is a LaTeX error rather than a typo
SUBSCRIPTS = {"ᵢ": "i", "ⱼ": "j", "ᵣ": "r"}
SUPERSCRIPTS = {"ᵀ": r"\mathsf{T}", "²": "2", "¹": "1", "⁻": "-"}
MATH_CHARS = {**SYMBOLS, **SUBSCRIPTS, **SUPERSCRIPTS}
# combining marks bind to the character BEFORE them, so they resolve first
COMBINING = {"\u0307": "dot", "\u0302": "hat"}
MATH_RUN = re.compile("(" + "|".join(re.escape(c) for c in MATH_CHARS) + ")+")
# inline code and fenced blocks are verbatim: never rewrite inside them
VERBATIM = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)


def _run_to_latex(run: str) -> str:
    """One run of math characters -> LaTeX, with adjacent scripts merged."""
    out: list[str] = []
    pending: list[str] = []
    mark = ""

    def flush() -> None:
        nonlocal mark
        if pending:
            out.append(f"{mark}{{{''.join(pending)}}}")
            pending.clear()
            mark = ""

    for ch in run:
        if ch in SUBSCRIPTS or ch in SUPERSCRIPTS:
            want = "_" if ch in SUBSCRIPTS else "^"
            if mark and mark != want:
                flush()
            mark = want
            pending.append(SUBSCRIPTS.get(ch) or SUPERSCRIPTS[ch])
        else:
            flush()
            out.append(SYMBOLS[ch])
    flush()
    return "".join(out)


# TeX script syntax written as prose: pandoc escapes it into literal characters
# and any Unicode mapped nearby then lands outside math, producing TeX that does
# not compile. Real formulas belong in $...$; this catches the ones that are not.
PSEUDO_MATH = re.compile(r"(?<![$\\])[\^_]\{")


def warn_pseudo_math(text: str) -> list[str]:
    stripped = VERBATIM.sub("", re.sub(r"\$[^$\n]+\$", "", text))
    return [line.strip() for line in stripped.splitlines() if PSEUDO_MATH.search(line)]


def to_latex_math(text: str) -> tuple[str, int]:
    """Map bare Unicode Greek and math characters onto LaTeX math."""
    hits = 0

    def convert(chunk: str) -> str:
        nonlocal hits

        def combining(m: re.Match) -> str:
            nonlocal hits
            hits += 1
            # a Greek base is already a macro; a plain Latin one is itself
            base = SYMBOLS.get(m.group(1), m.group(1))
            return f"\\ensuremath{{\\{COMBINING[m.group(2)]}{{{base}}}}}"

        # "theta + combining dot above" is one symbol, not two
        chunk = re.sub(f"(.)([{''.join(COMBINING)}])", combining, chunk)

        def run(m: re.Match) -> str:
            nonlocal hits
            hits += len(m.group(0))
            return f"\\ensuremath{{{_run_to_latex(m.group(0))}}}"

        chunk = MATH_RUN.sub(run, chunk)
        # a subscript that followed an accented base ends up in its own group;
        # merging neighbours keeps it attached to the symbol it belongs to
        return chunk.replace("}\\ensuremath{", "")

    parts = VERBATIM.split(text)
    parts[::2] = [convert(chunk) for chunk in parts[::2]]
    return "".join(parts), hits


def alias_map(citemap: dict[str, list[str]]) -> dict[str, str]:
    """label-derived key -> the canonical key it was merged into.

    One work is often cited two ways — by system name in a table, by author and
    year in prose — which derives two keys for one paper. The bibliography
    merges them; this is how both spellings still resolve, instead of the
    citations silently splitting across two entries.
    """
    out = {}
    for canonical, labels in citemap.items():
        out[canonical] = canonical
        for label in labels:
            who, year = split_author_year(label)
            out[citekey(who, year)] = canonical
    return out


def rewrite_links(text: str, known: dict[str, str]) -> tuple[str, int]:
    hits = 0

    def one(m: re.Match) -> str:
        nonlocal hits
        label = re.sub(r"\s+", " ", m.group(1)).strip()
        who, year = split_author_year(label)
        key = known.get(citekey(who, year))
        if key is None:
            return m.group(0)
        hits += 1
        # brace form: a bare @key would swallow a trailing hyphenated word
        # ("[Mamba](url)-class" -> "@mamba-class", a key that does not exist)
        return f"@{{{key}}}"

    # a parenthesised citation becomes a bracketed one so citeproc supplies
    # the parentheses itself rather than nesting them
    text, n = re.subn(
        r"\((\[[^\]\[]{2,120}\]\(https?://[^)\s]+\))\)",
        lambda m: "[" + one(re.match(LINK, m.group(1))).replace("@", "@", 1) + "]",
        text)
    text = LINK.sub(one, text)
    return text, hits


def rewrite_brackets(text: str, known: dict[str, str]) -> tuple[str, int]:
    hits = 0

    def one(m: re.Match) -> str:
        nonlocal hits
        inner = m.group(1)
        if inner.startswith("@"):
            return m.group(0)
        keys = []
        for piece in inner.split(";"):
            if NOT_A_CITATION.match(piece.strip()):
                return m.group(0)
            who, year = split_author_year(piece)
            key = known.get(citekey(who, year))
            if key is None:
                return m.group(0)
            keys.append(f"@{key}")
        hits += len(keys)
        return "[" + "; ".join(keys) + "]"

    return BRACKET.sub(one, text), hits


def main() -> None:
    ap = argparse.ArgumentParser(description="rewrite citations for pandoc")
    ap.add_argument("manuscript", type=Path)
    ap.add_argument("citemap", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--appendix-out", type=Path,
                    help="write everything after the <!-- appendix --> marker here")
    ap.add_argument("--abstract-out", type=Path,
                    help="lift the ## Abstract section out of the body and write it here as "
                         "pandoc metadata, so it renders in the title block and not twice")
    ap.add_argument("--for", dest="target", default="generic", choices=("generic", "latex"),
                    help="latex also maps Unicode Greek and math onto LaTeX math")
    a = ap.parse_args()

    known = alias_map(json.loads(a.citemap.read_text()))
    text = a.manuscript.read_text()
    if a.abstract_out is not None:
        # the abstract is written in the manuscript and rendered from metadata;
        # leaving it in the body too is what produced two of them
        a.abstract_out.write_text(abstract_mod.as_yaml(abstract_mod.read(text)))
        text = abstract_mod.strip(text)
    appendix = ""
    if a.appendix_out is not None and APPENDIX.search(text):
        text, appendix = APPENDIX.split(text, 1)
    text, n_links = rewrite_links(text, known)
    text, n_brackets = rewrite_brackets(text, known)
    n_math = n_img = 0
    if a.target == "latex":
        text, n_img = to_vector_images(text)
        for line in warn_pseudo_math(text):
            print(f"  WARNING: TeX script syntax outside math — wrap it in $...$:\n"
                  f"    {line[:100]}", file=sys.stderr)
        text, n_math = to_latex_math(text)
    # No "# References" heading is appended here. citeproc adds one from
    # `reference-section-title`, and the LaTeX path gets one from
    # \\bibliography — appending a third produced the heading twice at once.
    text = text.rstrip() + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    if a.appendix_out is not None:
        # the same rewriting has to run over the appendix, or its citations
        # never resolve; it is written separately only so the LaTeX builds can
        # place it after the bibliography
        if appendix.strip():
            appendix, _ = rewrite_links(appendix, known)
            appendix, _ = rewrite_brackets(appendix, known)
            if a.target == "latex":
                appendix, _ = to_latex_math(appendix)
                appendix, _ = to_vector_images(appendix)
        a.appendix_out.write_text(appendix.strip() + "\n" if appendix.strip() else "")
    extra = f", {n_math} math characters" if n_math else ""
    extra += f", {n_img} images to vector" if n_img else ""
    print(f"{a.out.name}: {n_links} link citations, "
          f"{n_brackets} bracket citations rewritten{extra}")


if __name__ == "__main__":
    main()

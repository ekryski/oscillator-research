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

A link citation resolves by its URL first and by its label only as a fallback.
The label is not unique and never was: "Nunley 2026" named two different arXiv
papers, "Rusch & Mishra 2021" named coRNN and UnICORNN, "Unconventional AI 2026"
named a blog post and the repository beside it. Resolving by label alone made
two works collapse into one — four citations pointed at the wrong paper and the
other paper vanished from the bibliography, with nothing in the build to say so.
The URL in the link is the thing that is actually unique per work, so that is
what resolution keys off; the label lookup stays for the citations that carry no
matchable URL, and a label that disagrees with its URL is reported rather than
silently obeyed.

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
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract as abstract_mod
import bibfile
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


# One work is spelled several ways across a manuscript and its bibliography: the
# prose percent-encodes the parentheses in an old DOI, the .bib does not; the
# prose writes an arxiv.org/abs URL, the .bib writes a bare eprint id; either
# side may carry a trailing slash, http instead of https, or a version suffix.
# Matching raw strings therefore fails on works that are plainly the same, so
# both sides are reduced to a canonical key first.
DOI_URL = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/(.+)$", re.I)
ARXIV_URL = re.compile(r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf|html)/(.+)$", re.I)
#: a `doi` field, which holds the identifier rather than a link to it
BARE_DOI = re.compile(r"^10\.\d{4,9}/\S+$")
#: an `eprint` field: a modern arXiv id, or an old archive/number one, either
#: optionally prefixed `arXiv:` and optionally carrying a version
BARE_ARXIV = re.compile(
    r"^(?:arxiv:)?(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?$", re.I)
#: which bibliography fields identify the work rather than describe it
IDENTIFIER_FIELDS = ("doi", "eprint", "url")


def url_key(value: str) -> str:
    """One link or identifier reduced to the form both sides of a match share.

    The kind is part of the key — a DOI, an arXiv id and a plain URL live in
    separate namespaces — so a bibliography entry whose `url` field happens to
    be a doi.org link still matches a manuscript link written the same way.
    """
    value = value.strip().rstrip("/")
    if m := DOI_URL.match(value):
        # https://doi.org/10.1016/0022-5193%2867%2990051-3 is the same work as
        # doi 10.1016/0022-5193(67)90051-3; DOIs are also case-insensitive
        return "doi:" + unquote(m.group(1)).lower()
    if BARE_DOI.match(value):
        return "doi:" + unquote(value).lower()
    if m := ARXIV_URL.match(value):
        # /pdf/ links carry the extension the /abs/ ones do not
        value = re.sub(r"\.pdf$", "", m.group(1), flags=re.I)
    if m := BARE_ARXIV.match(value):
        return "arxiv:" + m.group(1).lower()
    # anything else compares as a URL: scheme and host case carry no meaning,
    # the rest of the path does and is left alone
    host, slash, path = re.sub(r"^https?://", "", value, flags=re.I).partition("/")
    return "url:" + host.lower() + slash + path


def entry_links(fields: dict[str, str]) -> list[str]:
    """Every canonical link key one bibliography entry answers to.

    Classified by shape rather than by field name, so an entry that records its
    DOI in `url` — several do, where the DOI is the only page the work has —
    still matches a manuscript link written as a doi.org address.
    """
    return [url_key(fields[f]) for f in IDENTIFIER_FIELDS if fields.get(f)]


def url_index(bib: str) -> dict[str, str]:
    """link key -> citekey, the index that makes resolution unique per work.

    First entry wins when two claim one link, which keeps this deterministic;
    that duplication is a bibliography error in its own right and is what
    `check_citemap.py` reports.
    """
    out: dict[str, str] = {}
    for entry in bibfile.entries(bib):
        for key in entry_links(entry.fields):
            out.setdefault(key, entry.key)
    return out


#: Characters with no visual width. They survive copy-paste, they survive most
#: editors, and a run of them encodes arbitrary text that no reader can see —
#: the tag block exists for exactly that and is the usual way generated prose
#: gets watermarked. None of them has a legitimate use in these manuscripts, so
#: they are removed from every built format rather than reported and left.
#: Variation selectors are included: they matter for emoji presentation, which
#: an academic manuscript does not have, and they are a known carrier too.
HIDDEN = re.compile(
    "[\u00ad\u200b-\u200f\u2060-\u2064\ufeff\u180e"   # zero-width, directional marks
    "\ufe00-\ufe0f"                                        # variation selectors
    "\u202a-\u202e\u2066-\u2069"                          # bidi overrides
    "\U000e0000-\U000e007f]"                               # tag block
)
#: a no-break space is invisible as a *difference* rather than invisible outright;
#: deleting it would run two words together, so it is normalised, not dropped
NBSP = "\u00a0"


def strip_hidden(text: str) -> tuple[str, int]:
    """Remove zero-width and formatting characters; normalise no-break spaces."""
    text, n = HIDDEN.subn("", text)
    n += text.count(NBSP)
    return text.replace(NBSP, " "), n


def rewrite_links(text: str, known: dict[str, str],
                  by_url: dict[str, str] | None = None) -> tuple[str, int, list[str]]:
    """Rewrite `[Label](url)` citations, resolving by URL and falling back to label.

    Returns the rewritten text, how many citations resolved, and the
    disagreements worth reporting: a label whose own lookup lands on a different
    work than its URL does means the citemap and the bibliography no longer
    agree about which work that label names, which is exactly the state that
    used to send citations to the wrong paper without anyone noticing.
    """
    by_url = by_url or {}
    # the same index read backwards: which links each entry answers to. A link
    # the index does not know still says something when the entry its label
    # names carries a DIFFERENT identifier of the same kind.
    owns: dict[str, list[str]] = {}
    for link, key in by_url.items():
        owns.setdefault(key, []).append(link)
    hits = 0
    disagreements: list[str] = []

    def one(m: re.Match) -> str:
        nonlocal hits
        label = re.sub(r"\s+", " ", m.group(1)).strip()
        who, year = split_author_year(label)
        by_label = known.get(citekey(who, year))
        link = url_key(m.group(2))
        key = by_url.get(link)
        if key is not None:
            # the .bib is keyed canonically, but map through the alias table
            # anyway so a URL and a label can never resolve to two spellings of
            # one key and then be reported as disagreeing
            key = known.get(key, key)
            if by_label is not None and by_label != key:
                disagreements.append(
                    f"'{label}' resolves to @{by_label} by label but @{key} by its link "
                    f"{m.group(2)} — the link wins; fix the label or the bibliography")
        else:
            key = by_label
            # Comparing only within one namespace: a manuscript that links the
            # arXiv preprint of a work whose entry records the published DOI is
            # the normal case and says nothing. Two DOIs, or two arXiv ids, for
            # one work is the abnormal one — the label names an entry that is
            # demonstrably about something else.
            kind = link.split(":", 1)[0] + ":"
            rival = [x for x in owns.get(key, ()) if x.startswith(kind)]
            if rival:
                disagreements.append(
                    f"'{label}' links to {link} but @{key} records {', '.join(rival)} "
                    f"— one of the two names a different work")
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
    # one label cited twenty times would otherwise report the same problem
    # twenty times over, which is how a real warning gets scrolled past
    return text, hits, list(dict.fromkeys(disagreements))


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
    ap.add_argument("--bibliography", type=Path,
                    help="the .bib whose doi/eprint/url fields resolve link citations "
                         "(default: bibliography.bib beside the citemap)")
    ap.add_argument("--appendix-out", type=Path,
                    help="write everything after the <!-- appendix --> marker here")
    ap.add_argument("--abstract-out", type=Path,
                    help="lift the ## Abstract section out of the body and write it here as "
                         "pandoc metadata, so it renders in the title block and not twice")
    ap.add_argument("--for", dest="target", default="generic", choices=("generic", "latex"),
                    help="latex also maps Unicode Greek and math onto LaTeX math")
    a = ap.parse_args()

    known = alias_map(json.loads(a.citemap.read_text()))
    # the bibliography is what makes resolution unique per work; without one the
    # label lookup is all there is, which is what every build did before
    bib = a.bibliography or a.citemap.parent / "bibliography.bib"
    by_url = url_index(bib.read_text()) if bib.exists() else {}
    text = a.manuscript.read_text()
    if a.abstract_out is not None:
        # the abstract is written in the manuscript and rendered from metadata;
        # leaving it in the body too is what produced two of them
        a.abstract_out.write_text(abstract_mod.as_yaml(abstract_mod.read(text)))
        text = abstract_mod.strip(text)
    appendix = ""
    if a.appendix_out is not None and APPENDIX.search(text):
        text, appendix = APPENDIX.split(text, 1)
    # before anything else, so no invisible character reaches an output format
    text, n_hidden = strip_hidden(text)
    text, n_links, disagreements = rewrite_links(text, known, by_url)
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
            appendix, n_appx = strip_hidden(appendix)
            n_hidden += n_appx
            appendix, _, appx_disagreements = rewrite_links(appendix, known, by_url)
            disagreements += appx_disagreements
            appendix, _ = rewrite_brackets(appendix, known)
            if a.target == "latex":
                appendix, _ = to_latex_math(appendix)
                appendix, _ = to_vector_images(appendix)
        a.appendix_out.write_text(appendix.strip() + "\n" if appendix.strip() else "")
    # the body and the appendix are rewritten separately and cite the same works,
    # so the same disagreement can arrive from both
    for line in dict.fromkeys(disagreements):
        print(f"  WARNING: a citation's label and link name different works:\n"
              f"    {line}", file=sys.stderr)
    extra = f", {n_math} math characters" if n_math else ""
    extra += f", {n_img} images to vector" if n_img else ""
    # loud rather than silent: a hidden character in the source means the
    # Markdown still carries it even though the built formats no longer do
    extra += f", {n_hidden} HIDDEN CHARACTERS STRIPPED" if n_hidden else ""
    print(f"{a.out.name}: {n_links} link citations, "
          f"{n_brackets} bracket citations rewritten{extra}")


if __name__ == "__main__":
    main()

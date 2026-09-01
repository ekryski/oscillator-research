#!/usr/bin/env python3
"""Rewrite a bibliography into what legacy BibTeX understands.

The committed .bib is written for correctness and for citeproc: web sources are
typed `@online` with a `urldate`, and each entry carries a comment recording how
the manuscript spells that citation. BibTeX — which is what TMLR's tmlr.bst runs
under — predates both. It has no `@online`, silently drops entries whose type
its style file does not define, and treats a `%` inside an entry as a syntax
error. A dropped entry is not a visible failure: the citation just renders as a
bare key and natbib then refuses the whole bibliography.

So the source stays correct and this produces the dialect BibTeX wants, at build
time, on a copy.

    python3 publishing/lib/bibtex_compat.py in.bib --out references.bib
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bibfile import fields_of

#: `@online` is the right type and has been for twenty years, but tmlr.bst
#: inherits plainnat's fixed list. `@misc` with `howpublished` is the
#: pre-BibLaTeX spelling of the same thing and renders identically.
LEGACY_TYPE = {"online": "misc", "electronic": "misc", "software": "misc"}
#: fields BibTeX has no slot for; `note` is where their content ends up
UNKNOWN_FIELDS = ("urldate", "archiveprefix", "primaryclass", "eprint", "keywords")


def field(block: str, name: str) -> str | None:
    return fields_of(block).get(name)


#: a citekey is `surnameYEAR`, which is where the label comes from when the
#: entry itself has no author yet
KEY_FROM_CITEKEY = re.compile(r"^([a-z][a-z-]*?)-?(1[89]\d\d|20\d\d)?$")


def sort_key_for(citekey: str) -> str:
    """A label for an entry that has no author.

    BibTeX builds author-year labels from the author field. An entry still
    missing one yields a bare `\\bibitem{key}`, and natbib then refuses the
    whole bibliography with "not compatible with author-year citations" — one
    incomplete entry takes the paper down. BibTeX's `key` field exists for
    exactly this, so the citekey (which came from how the manuscript cites the
    work) stands in until the real author is filled in.
    """
    m = KEY_FROM_CITEKEY.match(citekey)
    stem = (m.group(1) if m else citekey).replace("-", " ")
    return stem.title()


#: fields typeset as prose. A bare `_` there is a TeX subscript in text mode,
#: which aborts the entry: the reference renders as one run-on italic line and
#: everything after the underscore is swallowed. Registry metadata supplies
#: them (S_N, paper_files), so this cannot be left to hand-editing.
TEXT_FIELDS = ("title", "journal", "booktitle", "series", "publisher",
               "institution", "school", "note", "howpublished")
#: spans inside which an underscore is already legal and must be left alone
VERBATIM_SPAN = re.compile(r"\\(?:url|path|href)\{[^}]*\}|\$[^$]*\$")


def escape_underscores(value: str) -> str:
    """Escape every `_` that is not already escaped, in math, or inside a URL."""
    out, at = [], 0
    for m in VERBATIM_SPAN.finditer(value):
        out.append(re.sub(r"(?<!\\)_", r"\\_", value[at:m.start()]))
        out.append(m.group(0))
        at = m.end()
    out.append(re.sub(r"(?<!\\)_", r"\\_", value[at:]))
    return "".join(out)


def escape_text_fields(block: str) -> str:
    """Escape underscores in every prose field of one entry."""
    for name in TEXT_FIELDS:
        value = field(block, name)
        if not value or "_" not in value:
            continue
        m = re.search(rf"^(\s*{name}\s*=\s*)\{{.*?\}}(,?[ \t]*)$", block, re.M | re.S)
        if m:
            body = "{" + escape_underscores(value) + "}"
            block = block[:m.start()] + m.group(1) + body + m.group(2) + block[m.end():]
    return block


def one_identifier(block: str) -> str:
    """Leave each entry pointing at itself once.

    tmlr.bst prints every identifier field it finds, so an entry carrying both
    a DOI and the publisher page that DOI resolves to states its address twice.
    The order of preference is DOI, then arXiv, then URL. A URL is kept when it
    is the only identifier, which is the case for the ML venues that mint no
    DOI, and a preprint id already folded into `note` is kept beside it because
    it is often the copy a reader can actually open.
    """
    url = field(block, "url")
    if not url:
        return block
    note = field(block, "note") or ""
    resolves_elsewhere = field(block, "doi") or (
        "arxiv.org/abs/" in url.lower() and "arXiv:" in note)
    if resolves_elsewhere:
        block = re.sub(r"^\s*url\s*=\s*\{.*?\},?\s*$\n?", "", block, flags=re.M)
    return block


def convert(block: str) -> str:
    m = re.match(r"@(\w+)\{([^,]+),", block)
    if not m:
        return block
    kind = m.group(1).lower()
    # a comment inside an entry is a BibTeX syntax error, and it takes the rest
    # of the entry down with it
    block = re.sub(r"^\s*%.*\n?", "", block, flags=re.M)

    if kind in LEGACY_TYPE:
        url, seen = field(block, "url"), field(block, "urldate")
        block = re.sub(r"^@\w+\{", f"@{LEGACY_TYPE[kind]}{{", block)
        # howpublished is where a style file looks for a web address; without it
        # a @misc entry renders as a title and nothing else
        if url and not field(block, "howpublished"):
            block = block.replace("\n}", f"\n  howpublished  = {{\\url{{{url}}}}},\n}}", 1)
        elif url:
            # howpublished already names the venue, and tmlr.bst prints both
            # fields, so leaving `url` in makes the entry state its address
            # twice. Fold it into `note` rather than adding a second note
            # field: BibTeX keeps the first of a repeated field and drops the
            # rest, which silently loses the address.
            block = re.sub(r"^\s*url\s*=\s*\{.*?\},?\s*$\n?", "", block, flags=re.M)
            existing = field(block, "note")
            merged = f"{existing}. " if existing else ""
            merged += f"Available at \\url{{{url}}}"
            if existing:
                block = re.sub(r"^(\s*note\s*= ).*$", lambda m: m.group(1) + "{" + merged + "},",
                               block, count=1, flags=re.M)
            else:
                block = block.replace("\n}", f"\n  note          = {{{merged}}},\n}}", 1)
        if seen:
            note = field(block, "note")
            block = re.sub(r"^\s*note\s*=\s*\{.*?\},?\s*$", "", block, flags=re.M)
            block = block.replace(
                "\n}", f"\n  note          = {{{note + '. ' if note else ''}"
                       f"Accessed {seen}}},\n}}", 1)

    # eprint/archivePrefix are natbib-era arXiv fields that plain BibTeX styles
    # ignore; fold the identifier into note so the reader still gets it
    if (eprint := field(block, "eprint")) and not field(block, "doi"):
        if "arXiv:" not in (field(block, "note") or ""):
            block = block.replace("\n}", f"\n  note          = {{arXiv:{eprint}}},\n}}", 1)
    # a @misc has no DOI slot in tmlr.bst, so a bioRxiv or dataset entry would
    # print its venue and drop the identifier entirely
    if kind in (*LEGACY_TYPE, "misc") and (doi := field(block, "doi")):
        if "doi" not in (field(block, "note") or "").lower():
            existing = field(block, "note")
            merged = f"{existing}. doi: {doi}" if existing else f"doi: {doi}"
            block = re.sub(r"^\s*doi\s*=\s*\{.*?\},?\s*$\n?", "", block, flags=re.M)
            if existing:
                block = re.sub(r"^(\s*note\s*= ).*$", lambda m: m.group(1) + "{" + merged + "},",
                               block, count=1, flags=re.M)
            else:
                block = block.replace("\n}", f"\n  note          = {{{merged}}},\n}}", 1)
    block = one_identifier(block)
    block = escape_text_fields(block)
    if not field(block, "author") and not field(block, "editor") and not field(block, "key"):
        block = block.replace("\n}", f"\n  key           = {{{sort_key_for(m.group(2))}}},\n}}", 1)
    for name in UNKNOWN_FIELDS:
        block = re.sub(rf"^\s*{name}\s*=\s*\{{.*?\}},?\s*$\n?", "", block, flags=re.M)
    return block


def main() -> None:
    ap = argparse.ArgumentParser(description="make a bibliography safe for legacy BibTeX")
    ap.add_argument("bib", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    blocks = re.split(r"(?=^@)", a.bib.read_text(), flags=re.M)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(convert(b) for b in blocks))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Make the prose's "Section 4.3" and "Appendix H.2" references into links to their headings.

The manuscripts write cross-references as plain text, the way a reader of the
Markdown wants them. At build time each numbered heading gets an explicit id
(`sec-4-3`, `sec-H-2`) and each reference that resolves becomes a link to it, so
every format can jump there: a `\\hyperref` in LaTeX, an anchor in HTML and
EPUB, a bookmark in Word.

Two steps, because the LaTeX path takes the numbers off the headings
(`number_sections.unnumber`) before anything else reads them:

    anchor(text)     -> the ids go on the headings while their numbers are still there
    link(text, ids)  -> the references become links, after the citation rewrite,
                        which would otherwise read "[H.6]" as a bracket citation

The numbering is the one `check_sections` computes, so a reference that does not
resolve stays plain text here and is reported there. A list continues its
reference: in "Appendix H.2, H.6 and H.7" all three become links.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from string import ascii_uppercase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract
from check_sections import HEADING, TOP_LEVEL, UNNUMBERED

#: one section number or appendix letter, with its subsections
NUMBER = r"(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\b"
REFERENCE = re.compile(rf"\b(?:Sections?|Appendix|Appendices)\s+({NUMBER}(?:(?:,\s*|,?\s+and\s+){NUMBER})*)")
ITEM = re.compile(NUMBER)
#: text a reference inside of is left alone: a heading, a link or image, inline
#: code, an HTML comment, an attribute block
SKIP = re.compile(r"(?m)^#{1,6}\s.*$|!?\[[^\]\n]*\]\([^)\n]*\)|`[^`\n]*`|<!--[\s\S]*?-->|\{#[^}\n]*\}")


def ident(number: str) -> str:
    return "sec-" + number.replace(".", "-")


def anchor(text: str, appendix: re.Pattern[str]) -> tuple[str, dict[str, str]]:
    """The text with an id on every numbered heading, and each number's id.

    Numbered as the build numbers: the abstract is not a section, and past the
    appendix marker the top level is lettered. A heading that already carries an
    id keeps it, and references to it link there.
    """
    found = abstract.SECTION.search(text)
    skip = found.span() if found else (-1, -1)
    marker = appendix.search(text)
    start_appendix = marker.start() if marker else len(text)
    ids: dict[str, str] = {}
    counters: list[int] = []
    lettered = False
    out, last = [], 0
    for m in HEADING.finditer(text):
        if skip[0] <= m.start() < skip[1] or UNNUMBERED.search(m.group(2)):
            continue
        if not lettered and m.start() > start_appendix:
            lettered, counters = True, []
        depth = len(m.group(1)) - TOP_LEVEL
        counters = counters[:depth + 1]
        counters += [0] * (depth + 1 - len(counters))
        counters[depth] += 1
        head = ascii_uppercase[counters[0] - 1] if lettered else str(counters[0])
        number = ".".join([head, *(str(c) for c in counters[1:])])
        own = re.search(r"\{[^}]*#([\w:.-]+)[^}]*\}$", m.group(2))
        ids[number] = own.group(1) if own else ident(number)
        if not own:
            out += [text[last:m.end(2)], f" {{#{ids[number]}}}"]
            last = m.end(2)
    out.append(text[last:])
    return "".join(out), ids


def link(text: str, ids: dict[str, str]) -> tuple[str, int]:
    """The text with every resolving reference made a link to its heading."""
    hits = 0

    def one(m: re.Match[str]) -> str:
        nonlocal hits
        whole, first = m.group(0), m.start(1) - m.start(0)
        pieces, last = [], 0
        for i, item in enumerate(ITEM.finditer(m.group(1))):
            s, e = first + item.start(), first + item.end()
            target = ids.get(item.group(0))
            if target is None:
                continue
            # the first item carries its word: "Section 4.3" is the link, not "4.3"
            s = 0 if i == 0 else s
            pieces += [whole[last:s], f"[{whole[s:e]}](#{target})"]
            last = e
            hits += 1
        return "".join(pieces) + whole[last:]

    out, last = [], 0
    for m in SKIP.finditer(text):
        out += [REFERENCE.sub(one, text[last:m.start()]), m.group(0)]
        last = m.end()
    out.append(REFERENCE.sub(one, text[last:]))
    return "".join(out), hits

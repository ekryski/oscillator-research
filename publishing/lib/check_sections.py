#!/usr/bin/env python3
"""Report prose cross-references pointing at a section that does not exist.

Section numbers live nowhere in the manuscripts: pandoc assigns them from the
heading order at build time. So every "Section 5.1" in the prose is a hand-kept
copy of a number the build computes, and moving one heading silently falsifies
some of those copies while leaving them looking perfectly plausible. This
recomputes the numbering the way the build does and checks each reference
against it.

    python3 publishing/lib/check_sections.py           # report
    python3 publishing/lib/check_sections.py --strict  # and fail

What this catches is references left dangling, which is the large and silent
class. A reference that resolves is only live, not necessarily right: pointing
at a real but wrong section still passes here, so the context is printed beside
each finding for the cases where a human has to judge.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from string import ascii_uppercase

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract
from paths import papers
from preprocess import APPENDIX

#: `--shift-heading-level-by=-1` in the build makes `##` a section and `###` a
#: subsection, so a heading's depth is its `#` count minus two
TOP_LEVEL = 2
HEADING = re.compile(r"^(#{2,6})\s+(.*?)\s*$", re.M)
#: pandoc's ways of spelling "do not number this heading"
UNNUMBERED = re.compile(r"\{[^}]*(?:\.unnumbered|(?<=\{)-)[^}]*\}$")
SECTION_REF = re.compile(r"\bSection\s+(\d+(?:\.\d+)*)")
APPENDIX_REF = re.compile(r"\bAppendix\s+([A-Z])(?:\.(\d+))?\b")
FIGURE_REF = re.compile(r"\bFigure\s*\*?\s*(\d+)")
TABLE_REF = re.compile(r"\bTable\s*\*?\s*(\d+)")
#: a figure is an image block of its own; a table caption is a line opening ":"
FIGURE = re.compile(r"(?m)^!\[")
TABLE_CAPTION = re.compile(r"(?m)^:\s+\S")


def numbering(text: str) -> dict[str, str]:
    """Every number pandoc will assign, mapped to its heading text.

    Derived from the same two rules the build applies: the abstract is lifted
    into metadata rather than left as a section, and everything past the
    appendix marker is lettered rather than numbered.
    """
    parts = APPENDIX.split(abstract.strip(text), 1)
    body, appendix = parts[0], parts[1] if len(parts) > 1 else ""
    out: dict[str, str] = {}
    for source, lettered in ((body, False), (appendix, True)):
        counters: list[int] = []
        for hashes, title in HEADING.findall(source):
            if UNNUMBERED.search(title):
                continue
            depth = len(hashes) - TOP_LEVEL
            # drop any deeper level, open this one if the document skipped a
            # level, then advance it: a new "###" continues its parent's count
            counters = counters[:depth + 1]
            counters += [0] * (depth + 1 - len(counters))
            counters[depth] += 1
            head = ascii_uppercase[counters[0] - 1] if lettered else str(counters[0])
            out[".".join([head, *(str(c) for c in counters[1:])])] = title
    return out


def float_counts(text: str) -> dict[str, int]:
    """How many figures and tables the document has, which is how far a
    "Figure 7" reference can point. LaTeX numbers both in document order and
    does not reset either at the appendix, so counting occurrences is enough."""
    body = abstract.strip(text)
    return {"Figure": len(FIGURE.findall(body)), "Table": len(TABLE_CAPTION.findall(body))}


def issues(text: str) -> list[tuple[str, str]]:
    """(reference, surrounding prose) for every reference that resolves to nothing."""
    known = numbering(text)
    counts = float_counts(text)
    found = []
    for pattern, label in ((SECTION_REF, "Section"), (APPENDIX_REF, "Appendix")):
        for m in pattern.finditer(text):
            ref = ".".join(g for g in m.groups() if g)
            if ref not in known:
                context = re.sub(r"\s+", " ", text[max(0, m.start() - 62):m.end() + 10])
                found.append((f"{label} {ref}", f"…{context.strip()}…"))
    # figure and table numbers are assigned by document order, so the only thing
    # checkable without a cross-reference package is that the number exists
    for label, pattern in (("Figure", FIGURE_REF), ("Table", TABLE_REF)):
        for m in pattern.finditer(text):
            if not 1 <= int(m.group(1)) <= counts[label]:
                context = re.sub(r"\s+", " ", text[max(0, m.start() - 62):m.end() + 10])
                found.append((f"{label} {m.group(1)}", f"…{context.strip()}…"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="check section cross-references resolve")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any dangle")
    a = ap.parse_args()
    total = 0
    for paper in papers():
        if not paper.manuscript:
            continue
        found = issues(paper.manuscript.read_text())
        total += len(found)
        print(f"\n{paper.slug}: {len(found)} cross-reference(s) pointing at nothing")
        for ref, count in Counter(ref for ref, _ in found).most_common():
            context = next(c for r, c in found if r == ref)
            print(f"  {ref:16s} {count}x  {context[:88]}")
    return 1 if (total and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

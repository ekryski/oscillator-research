#!/usr/bin/env python3
"""Write the section numbers pandoc computes back into the manuscript's headings.

The numbers exist only at build time: pandoc derives them from heading order and
prints "2.3.3" in the PDF and the HTML. The Markdown source, which is what most
people actually read, carries headings with no numbers at all — so a sentence
saying "the controls in Section 2.3.3" or "listed in Appendix D" points at
nothing a GitHub reader can find. This puts the number in the heading text, so
the source reads the way the built formats do.

    python3 publishing/lib/number_sections.py          # rewrite, report changes
    python3 publishing/lib/number_sections.py --check  # report only, fail if stale

The numbering itself is not recomputed here. `check_sections.numbering` already
derives it exactly as the build does — abstract lifted into metadata, appendix
lettered, `{-}` headings skipped — and this walks the same headings in the same
order and writes those numbers down, so the two can never disagree.

Rewriting is idempotent: an existing number of the shape a heading's own
position would produce is stripped before the current one is written, which is
what stops a second run from producing "1 1 Introduction".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract
from check_sections import HEADING, TOP_LEVEL, UNNUMBERED, numbering
from paths import papers
from preprocess import APPENDIX

#: shapes a number of a given depth can take: "3", "3.1", "3.1.2" in the body and
#: "C", "C.1", "C.1.2" in the appendix. Matching the shape to the heading's own
#: depth is what keeps stripping conservative — "### A subsection of it" cannot
#: be read as a number, because at that depth a lettered number needs a ".1".
DIGIT_SHAPE = r"\d+(?:\.\d+){%d}"
LETTER_SHAPE = r"[A-Z](?:\.\d+){%d}"
#: a top-level appendix heading that already opens with a bare letter
LETTERED_HEADING = re.compile(r"^[A-Z]\s+\S")


def strip_number(title: str, depth: int, allow_bare_letter: bool) -> str:
    """A heading's text without the number a previous run left on it.

    Only a prefix shaped like a number for *this* heading's depth is removed, so
    prose is left alone. The one genuinely ambiguous case is a top-level
    appendix heading beginning with a one-letter word — "A survey of ..." looks
    exactly like appendix A — which `allow_bare_letter` decides.
    """
    shapes = [DIGIT_SHAPE % depth]
    if depth or allow_bare_letter:
        shapes.append(LETTER_SHAPE % depth)
    for shape in shapes:
        # require something after the number: a heading that is only a number
        # is not one we wrote, and stripping it would leave an empty heading
        m = re.match(rf"(?:{shape})\s+(?=\S)", title)
        if m:
            return title[m.end():]
    return title


def _appendix_is_lettered(text: str) -> bool:
    """Whether the appendix's top-level headings already carry their letters.

    Decided by majority rather than per heading, because one heading opening
    with "A" is a title and most of them opening with consecutive single letters
    is our own numbering. A run that has just had a heading inserted into it is
    still a majority, which is what lets an insertion renumber everything after
    it; a document where one appendix happens to be called "A note on ..." is
    not, and is left intact.
    """
    parts = APPENDIX.split(text, 1)
    if len(parts) < 2:
        return False
    tops = [title for hashes, title in HEADING.findall(parts[1])
            if len(hashes) == TOP_LEVEL and not UNNUMBERED.search(title)]
    return bool(tops) and 2 * sum(bool(LETTERED_HEADING.match(t)) for t in tops) > len(tops)


def renumber(text: str) -> tuple[str, list[tuple[str, str]]]:
    """The manuscript with numbered headings, and every heading line that moved."""
    # the numbers, in document order; `numbering` skips exactly the headings
    # skipped below, so consuming them in order pairs each with its heading
    numbers = iter(numbering(text))
    allow_bare_letter = _appendix_is_lettered(text)
    # the abstract is lifted into metadata by the build and never numbered; its
    # own definition of where it ends is the one to trust
    found = abstract.SECTION.search(text)
    abstract_span = found.span() if found else (-1, -1)
    changed: list[tuple[str, str]] = []

    def rewrite(m: re.Match[str]) -> str:
        hashes, title = m.group(1), m.group(2)
        # HEADING ends `\s*$`, which swallows the heading's own line break; put
        # it back or every heading is welded to the blank line after it
        raw = m.group(0)
        tail = raw[len(raw.rstrip()):]
        if abstract_span[0] <= m.start() < abstract_span[1]:
            return raw
        depth = len(hashes) - TOP_LEVEL
        if UNNUMBERED.search(title):
            # not numbered, but it may still carry a number from before it was
            # marked, and a stale one is worse than none
            body = strip_number(title, depth, allow_bare_letter)
        else:
            body = f"{next(numbers)} {strip_number(title, depth, allow_bare_letter)}"
        line = f"{hashes} {body}"
        if line != raw.rstrip():
            changed.append((raw.rstrip(), line))
        return line + tail

    out = HEADING.sub(rewrite, text)
    # every number the build will assign has to have found a heading to sit on;
    # if it did not, this walk and `numbering`'s disagree and the result is wrong
    if next(numbers, None) is not None:
        raise AssertionError("heading walk disagrees with check_sections.numbering")
    return out, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="write section numbers into manuscript headings")
    ap.add_argument("--check", action="store_true", help="report only, exit non-zero if stale")
    a = ap.parse_args()
    stale = 0
    for paper in papers():
        if not paper.manuscript:
            continue
        text = paper.manuscript.read_text()
        out, changed = renumber(text)
        stale += len(changed)
        verb = "out of date" if a.check else "renumbered"
        print(f"\n{paper.slug}: {len(changed)} heading(s) {verb}")
        for before, after in changed:
            print(f"  {before}\n  -> {after}")
        if changed and not a.check:
            paper.manuscript.write_text(out)
    return 1 if (stale and a.check) else 0


if __name__ == "__main__":
    sys.exit(main())

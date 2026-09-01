#!/usr/bin/env python3
"""Report which bibliography entries are not yet complete enough to publish.

Completeness is checked, not annotated. A note in the .bib saying "VERIFY" goes
stale the moment someone fixes the entry and forgets the note; a check reads the
data as it actually is, every build.

    python3 publishing/lib/check_bib.py           # report
    python3 publishing/lib/check_bib.py --strict  # and fail if anything is missing

What counts as complete: an author, a real title, a year, and either a venue or
a resolvable identifier. Web sources are held to the same bar minus the venue —
they are cited deliberately (both papers lean on results that exist only as
blog posts) and are typed @online with the date they were read.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bibfile import entries
from paths import papers

IDENTIFIERS = ("doi", "eprint", "url")
#: where "where was this published" lives, which depends on the entry type — a
#: technical report's venue is the institution that issued it, and a thesis's is
#: the school. Checking only the journal/booktitle pair reports those as
#: venueless no matter how completely they are filled in.
VENUE_FIELDS = ("journal", "booktitle", "publisher", "institution", "school",
                "organization", "howpublished")


def problems(kind: str, f: dict) -> list[str]:
    out = []
    author = f.get("author", "").strip()
    title = f.get("title", "").strip()
    if not author:
        out.append("no author")
    elif re.fullmatch(r"[\d/.\s-]+", author):
        out.append(f"author looks like a date ({author!r})")
    if not title:
        out.append("no title")
    elif len(title.split()) < 2 and kind != "online":
        # a repository or post is legitimately known by a one-word name; a
        # journal article recorded under one is a citation that has lost its title
        out.append(f"title is a short name ({title!r}), not the published title")
    if not f.get("year") and kind != "online":
        out.append("no year")
    if not any(f.get(i) for i in IDENTIFIERS):
        out.append("no identifier to resolve")
    if kind != "online" and not any(f.get(v) for v in VENUE_FIELDS):
        if not f.get("eprint"):          # a preprint needs no venue
            out.append("no venue")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="check bibliography completeness")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if anything is missing")
    a = ap.parse_args()
    worst = 0
    for paper in papers():
        path = paper.bib
        if not path.exists():
            continue
        found = entries(path.read_text())
        rows = [(e.key, issues) for e in found if (issues := problems(e.kind, e.fields))]
        print(f"\n{paper.slug}: {len(found) - len(rows)}/{len(found)} complete")
        for key, issues in rows:
            print(f"  {key:22s} {'; '.join(issues)}")
        worst = max(worst, len(rows))
    if worst:
        print("\nThese need their metadata confirmed against the source before submission.")
    return 1 if (worst and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Report citation labels that two different works both answer to.

Citations resolve by label, not by URL: `[Nunley 2026](...)` becomes a citekey
derived from the label's author and year. When two entries derive the same key
one silently wins, every citation meant for the other lands on the wrong paper,
and that paper drops out of the bibliography entirely. Nothing else in the build
notices — the reference list is still complete, every claim is still cited, and
the citation simply points at a different work.

This has happened five times in this repository: coRNN against UnICORNN, Kuramoto
Attention against FSN, the Un-0 blog against its code repository, two Huang papers
that collided the moment one was upgraded to its published year, and HiPPO against
Mamba. The fix each time is a letter suffix, which the label parser now understands.

    python3 publishing/lib/check_citemap.py            # report
    python3 publishing/lib/check_citemap.py --strict   # and fail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_bib import citekey, split_author_year
from paths import papers


def collisions(citemap: dict[str, list[str]]) -> list[tuple[str, str, str, str]]:
    """(derived key, label, entry that claims it, entry that loses it)."""
    seen: dict[str, tuple[str, str]] = {}
    out = []
    for key, labels in citemap.items():
        for label in labels:
            derived = citekey(*split_author_year(label))
            if derived in seen and seen[derived][0] != key:
                out.append((derived, label, seen[derived][0], key))
            else:
                seen[derived] = (key, label)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="citation labels two works share")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    total = 0
    for paper in papers():
        if not paper.citemap.exists():
            continue
        found = collisions(json.loads(paper.citemap.read_text()))
        total += len(found)
        print(f"\n{paper.slug}: {len(found)} label(s) claimed by two works")
        for derived, label, first, second in found:
            print(f"  {label:34s} -> {derived:18s} {first} vs {second}")
    if total:
        print("\nDisambiguate with a letter suffix, as in 'Huang et al. 2026a'.")
    return 1 if (total and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

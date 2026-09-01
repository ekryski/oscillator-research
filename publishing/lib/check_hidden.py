#!/usr/bin/env python3
"""Report invisible characters in the sources a build reads.

`preprocess.py` strips these from every built format, so a submission PDF is
clean either way. This checks the files themselves, because the Markdown is what
gets read on GitHub and the `.bib` never passes through preprocess at all.

The characters have no visual width. They survive copy-paste and most editors,
and a run of them encodes text no reader can see — the Unicode tag block exists
for that purpose and is the usual carrier when generated prose is watermarked.
None has a legitimate use in a manuscript or a bibliography.

    python3 publishing/lib/check_hidden.py            # report
    python3 publishing/lib/check_hidden.py --strict   # and fail
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import papers
from preprocess import HIDDEN, NBSP


def offenders(text: str) -> dict[str, int]:
    """Character name -> how many times it occurs."""
    out: dict[str, int] = {}
    for ch in text:
        if HIDDEN.match(ch) or ch == NBSP:
            name = unicodedata.name(ch, f"U+{ord(ch):04X}")
            out[name] = out.get(name, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="invisible characters in the sources")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    total = 0
    for paper in papers():
        found: dict[str, dict[str, int]] = {}
        for f in (paper.manuscript, paper.bib):
            if f and f.exists() and (hits := offenders(f.read_text())):
                found[f.name] = hits
        n = sum(sum(h.values()) for h in found.values())
        total += n
        print(f"\n{paper.slug}: {n} invisible character(s) in the source")
        for name, hits in found.items():
            for ch, count in sorted(hits.items(), key=lambda kv: -kv[1]):
                print(f"  {name[:44]:46s} {count:4d}  {ch}")
    if total:
        print("\nThe built formats have these stripped; the source files still carry them.")
    return 1 if (total and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

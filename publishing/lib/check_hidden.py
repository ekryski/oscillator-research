#!/usr/bin/env python3
"""Report characters in the sources that could carry an unseen mark.

The build removes them from every copy it formats (see sanitize.py), so a
submission is clean either way. This checks the files themselves, because the
Markdown is what gets read on GitHub and a source that carries a mark should be
cleaned at the source: invisible and control characters, unusual spaces, Latin
look-alike letters from other scripts, and trailing whitespace, in each paper's
manuscript, bibliography and front matter.

    python3 publishing/lib/check_hidden.py            # report
    python3 publishing/lib/check_hidden.py --strict   # and fail
    python3 publishing/lib/check_hidden.py --fix      # clean the sources in place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import papers
from sanitize import clean


def main() -> int:
    ap = argparse.ArgumentParser(description="characters in the sources that could carry an unseen mark")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any are found")
    ap.add_argument("--fix", action="store_true", help="write the cleaned text back to each source")
    a = ap.parse_args()
    total = 0
    for paper in papers():
        lines = []
        for f in (paper.manuscript, paper.bib, paper.metadata):
            if not (f and f.exists()):
                continue
            text = f.read_text(encoding="utf-8")
            cleaned, found = clean(text, markdown=f.suffix == ".md")
            if not found:
                continue
            total += sum(found.values())
            lines.append(f"  {f.name}: " + ", ".join(f"{n} {kind}" for kind, n in sorted(found.items())))
            if a.fix:
                f.write_text(cleaned, encoding="utf-8")
        print(f"\n{paper.slug}: " + ("clean" if not lines else "found in the sources" + (", cleaned" if a.fix else "")))
        print("\n".join(lines)) if lines else None
    if total and not a.fix:
        print("\nThe built formats have these removed; `--fix` removes them from the sources too.")
    return 1 if (total and a.strict and not a.fix) else 0


if __name__ == "__main__":
    sys.exit(main())

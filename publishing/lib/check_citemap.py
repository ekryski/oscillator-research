#!/usr/bin/env python3
"""Report the two ways a citation can still land on the wrong work.

A shared LABEL was the original hazard. `[Nunley 2026](...)` derives a citekey
from the label's author and year, and when two works derive the same key one
silently wins: every citation meant for the other lands on the wrong paper, and
that paper drops out of the bibliography entirely. Nothing else in the build
notices — the reference list is still complete and every claim is still cited.
This happened five times here: coRNN against UnICORNN, Kuramoto Attention
against FSN, the Un-0 blog against its code repository, two Huang papers that
collided the moment one was upgraded to its published year, and HiPPO against
Mamba. The fix each time is a letter suffix, which the label parser understands.

`preprocess.py` now resolves link citations by URL first, so a shared label no
longer misdirects a citation that carries a URL the bibliography knows. It still
misdirects the ones that do not — bracket citations, and links to works whose
entry records no matching identifier — so the report stays.

A shared LINK is the hazard that URL resolution introduces: two bibliography
entries carrying one identifier means one work was entered twice, the index that
resolution reads keeps only the first, and citations to the second silently
follow it. That is the label collision again, one level down, so it is reported
in the same place.

    python3 publishing/lib/check_citemap.py            # report
    python3 publishing/lib/check_citemap.py --strict   # and fail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bibfile
import preprocess
from extract_bib import AUTHOR_YEAR, LINK, citekey, split_author_year
from paths import papers
from preprocess import entry_links


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


def shared_links(bib: str) -> list[tuple[str, str, str]]:
    """(link key, entry that claims it, entry that loses it) in one .bib.

    Two entries carrying one DOI, eprint or URL are two records of one work.
    Only the first is reachable through the URL index, so the second's citations
    resolve to the first — and the second's own entry is never cited at all.
    """
    seen: dict[str, str] = {}
    out = []
    for entry in bibfile.entries(bib):
        for link in entry_links(entry.fields):
            if link in seen and seen[link] != entry.key:
                out.append((link, seen[link], entry.key))
            else:
                seen.setdefault(link, entry.key)
    return out


def label_only(manuscript: str, bib: str) -> list[tuple[str, str]]:
    """(label, link) for every citation no bibliography identifier matches.

    These are the citations that can go wrong quietly. A citation whose link
    matches a `doi`, `eprint` or `url` in the bibliography is checked by that
    match: point it at the wrong entry and the identifiers disagree. A citation
    with no such match is resolved on its author-year label alone, and a label
    fits any work by those authors in that year — which is how a "Zhang et al.
    (2023)" in the prose came to print a Zhang 2023 about a different subject.

    Reported per distinct (label, link) pair, since one work is usually cited
    many times and one entry is the thing to fix.
    """
    by_url = preprocess.url_index(bib)
    out = []
    for m in preprocess.LINK.finditer(manuscript):
        label = " ".join(m.group(1).split())
        if preprocess.NOT_A_CITATION.search(label):
            continue
        if not by_url.get(preprocess.url_key(m.group(2))):
            out.append((label, m.group(2)))
    # A bracket citation carries no link at all, so it is label-resolved by
    # construction and nothing checks that its entry holds the work meant. Paper
    # 02 was written entirely this way and passed a check that only looked at
    # links, which is the check reporting a clean bill it had not earned.
    #
    # Links are stripped first, as rewrite_links does before rewrite_brackets
    # runs: the label half of `[Label](url)` is itself a bracket, and scanning
    # the raw text reports every link citation in the paper. A bracket is only
    # a citation if it parses as author-year, which leaves notation like the
    # matrix shape `[T x 16]` alone.
    bare = LINK.sub(" ", manuscript)
    for m in preprocess.BRACKET.finditer(bare):
        if m.group(1).startswith("@"):
            continue
        for part in m.group(1).split(";"):
            label = " ".join(part.split())
            if label and AUTHOR_YEAR.match(label) and not preprocess.NOT_A_CITATION.search(label):
                out.append((label, "(no link: bracket citation)"))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="labels and links two works share")
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
        if not paper.bib.exists():
            continue
        dupes = shared_links(paper.bib.read_text())
        total += len(dupes)
        print(f"{paper.slug}: {len(dupes)} link(s) claimed by two bibliography entries")
        for link, first, second in dupes:
            print(f"  {link:52s} {first} vs {second}")
        if not paper.manuscript:
            continue
        loose = label_only(paper.manuscript.read_text(), paper.bib.read_text())
        total += len(loose)
        print(f"{paper.slug}: {len(loose)} citation(s) matched on the label alone")
        for label, link in loose:
            print(f"  {label:34s} {link}")
    if total:
        print("\nDisambiguate a label with a letter suffix, as in 'Huang et al. 2026a';\n"
              "a shared link means one work has two entries — merge them; a citation\n"
              "matched on its label alone wants the identifier its link uses recorded\n"
              "in the entry, as a doi, an eprint, a url or a 'Preprint: arXiv:...' note.")
    return 1 if (total and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

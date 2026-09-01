#!/usr/bin/env python3
"""Report authors named in prose before they are cited.

A survey names the same dozen systems and models over and over. The failure this
catches is citing each of them once, usually in a table, and then discussing them
by bare name for the rest of the paper: a reader who enters at Section 5 meets
"WONN" with no way to reach the work. The rule enforced here is the usual one,
cite on first mention within each top-level section.

    python3 publishing/lib/check_first_cite.py            # report
    python3 publishing/lib/check_first_cite.py --strict   # and fail

Both citation styles in this repository count: a markdown link carrying the
identifier, and an `[Author Year]` bracket. A citation later in the same sentence
counts too, because "the canonical Adler injection form [Adler 1946]" cites at
first mention even though the surname comes first.

EPONYMS are the deliberate exception. Kuramoto, Winfree, Sakaguchi, Stuart,
Landau and Adler each name a coupling law as well as a person, and the papers use
them that way constantly ("the Kuramoto family", "Adler injection"). They are
cited where the model is introduced; requiring a citation on every later section
that mentions the law would bury the prose in links that point somewhere the
reader has already been.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bibfile import entries
from paths import papers

#: surnames that also name a coupling law or an injection form
EPONYMS = {"Kuramoto", "Winfree", "Sakaguchi", "Stuart", "Landau", "Adler",
           "Hopf", "Daido", "Ermentrout", "Morris", "Lecar", "Hodgkin", "Huxley",
           "Abrams", "Strogatz", "Huygens", "Airy", "Blondel", "Poincaré", "Pol"}
#: a system name carries its own capitalisation: ESN, UnICORNN, D-LinOSS, Un-0.
#: A citemap label without it is a common word the manuscript happened to link,
#: and "survey" appears on nearly every page of a survey.
DISTINCTIVE = re.compile(r"[A-Z].*[A-Z]|[A-Z].*[0-9]")
#: a citation this far into the same sentence still counts as "on first mention"
SAME_SENTENCE = 140
#: a citemap label shaped like an author-year citation rather than a system name
AUTHOR_YEAR = re.compile(r"\(\d{4}\)$|\s(1[89]|20)\d\d[a-z]?$")


def system_names(citemap: Path) -> dict[str, str]:
    """Named systems -> citekey, from the labels the manuscript has used.

    Checking author surnames alone misses these entirely. UnICORNN, D-LinOSS and
    Neural Wave Machines were each named in the prose with no citation attached,
    and every one of their authors was cited elsewhere in the same section, so
    an author-level check reported the section clean.
    """
    if not citemap.exists():
        return {}
    out = {}
    for key, labels in json.loads(citemap.read_text()).items():
        for label in labels:
            if AUTHOR_YEAR.search(label) or len(label) < 3:
                continue
            if not DISTINCTIVE.search(label):
                continue
            words = [w for w in re.split(r"[^\w]+", label) if w]
            if words and all(w in EPONYMS for w in words):
                continue          # a coupling model, checked as an eponym
            out.setdefault(label, key)
    return out


def surnames(bib: Path) -> dict[str, str]:
    """First-author surname -> citekey, for every entry with an author."""
    out: dict[str, str] = {}
    for e in entries(bib.read_text()):
        author = e.fields.get("author", "")
        if not author:
            continue
        first = author.split(" and ")[0]
        fam = (first.split(",")[0] if "," in first else first.split()[-1]).strip("{} ")
        if len(fam) > 2 and fam[0].isupper() and fam not in EPONYMS:
            out.setdefault(fam, e.key)
    return out


def sections(text: str):
    for m in re.finditer(r"(?m)^## .+$", text):
        rest = text[m.end():]
        nxt = re.search(r"(?m)^## ", rest)
        yield m.group(0).lstrip("# ").strip(), rest[:nxt.start() if nxt else len(rest)]


def issues(md: Path, bib: Path, citemap: Path | None = None) -> list[tuple[str, str, str]]:
    text = md.read_text()
    found = []
    targets = dict(surnames(bib))
    if citemap:
        targets.update(system_names(citemap))
    known_systems = system_names(citemap) if citemap else {}
    for head, body in sections(text):
        # a system named in a table cell is an entry in a list, not a sentence
        # making a claim; the section lead is what points at the inventory
        prose = re.sub(r"(?m)^\|.*$", "", body)
        for fam, key in targets.items():
            body = prose if fam in known_systems else body
            mentions = [m.start() for m in
                        re.finditer(rf"(?<![\w\-]){re.escape(fam)}(?![\w\-])", body)]
            if not mentions:
                continue
            cited = [m.start() for m in re.finditer(
                rf"\[[^\]]*{re.escape(fam)}[^\]]*\]\(https?://"
                rf"|\[[^\]]*{re.escape(fam)}[^\]]*(?:1[89]\d\d|20\d\d)[^\]]*\]", body)]
            # a system is cited by the reference that follows its name
            if fam not in surnames(bib):
                cited += [m.end() for m in
                          re.finditer(rf"{re.escape(fam)}\s*\(\[[^\]]+\]\(https?://", body)]
            first = min(mentions)
            if not cited:
                found.append((head, fam, f"{key}: named {len(mentions)}x, never cited here"))
            elif first < min(cited) and not any(abs(first - c) < SAME_SENTENCE for c in cited):
                snippet = re.sub(r"\s+", " ", body[max(0, first - 60):first + 40])
                found.append((head, fam, f"{key}: bare first mention — …{snippet}…"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="authors named before they are cited")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    total = 0
    for paper in papers():
        if not (paper.manuscript and paper.bib.exists()):
            continue
        found = issues(paper.manuscript, paper.bib, paper.citemap)
        total += len(found)
        print(f"\n{paper.slug}: {len(found)} author(s) named before being cited")
        for head, fam, why in sorted(found):
            print(f"  {head[:34]:36s} {fam:18s} {why[:96]}")
    return 1 if (total and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

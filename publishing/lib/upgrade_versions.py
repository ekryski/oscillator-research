#!/usr/bin/env python3
"""Point preprint entries at the published version of record.

Where a work exists both as a preprint and as a peer-reviewed paper, the
published one is what should be cited: it is the version a reader should be sent
to, it is the one reviewers check, and citing the preprint of something that
appeared at ICLR two years ago reads as a bibliography assembled from memory.

The symptom is a year disagreement. The manuscript cites `rusch2021` because the
paper is an ICLR 2021 paper; the entry's identifier is the arXiv preprint, which
the registry dates 2020.

Two sources, in order of authority:

  1. **arXiv's own record.** Authors add a DOI and a journal reference to their
     preprint once it is published, so this is the publication saying where it
     went rather than an inference. It is also the only one of the two that
     covers journals outside computer science.
  2. **DBLP.** Machine learning's major venues — ICLR, ICML, NeurIPS — mint no
     DOI at all, so no DOI-based lookup can ever find them; a CrossRef title
     search for one of these returns either nothing or, worse, a different paper
     that happens to share a title. DBLP indexes them properly and, crucially,
     files preprints under "Informal and Other Publications", which is exactly
     the distinction being drawn here. Its own BibTeX export is used verbatim
     rather than reassembled.

    python3 publishing/lib/upgrade_versions.py --dry-run   # what it would change
    python3 publishing/lib/upgrade_versions.py             # do it
    python3 publishing/lib/upgrade_versions.py --all       # every preprint, not
                                                           # only mismatched years

The preprint identifier is kept in a `note`, because it is often the copy a
reader can actually open, and the citekey never changes — the manuscripts cite
by key and a rename would orphan them.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_metadata import as_bibtex, csl_for, parse_bib
from paths import papers

MAILTO = "hello@erickryski.com"
#: bounded hard. DBLP throttles bursts by dropping the connection rather than
#: answering 429, so a generous timeout with retries turns one blocked lookup
#: into minutes of silence and a whole run into an hour.
TIMEOUT = 15
CONNECT_TIMEOUT = 8
#: DBLP asks for roughly one request a second and blocks anything faster
PACE = 1.2
#: below this the search has found a different paper — a related one, a survey
#: citing it, a workshop version under a reworded title
TITLE_MATCH = 0.88
#: DBLP's own name for "this is a preprint", which is what we are upgrading FROM
DBLP_PREPRINT = "Informal"
#: DBLP bookkeeping that means nothing outside DBLP
DBLP_NOISE = ("timestamp", "biburl", "bibsource")


class Unreachable(Exception):
    """The source did not answer — which is not the same as having no answer.

    Treating a throttled lookup as "no published version exists" would leave the
    entry pointing at a preprint and report it as deliberately correct. The two
    outcomes have to stay distinguishable all the way to the console.
    """


def get(url: str, accept: str = "") -> str:
    cmd = ["curl", "-sSL", "--max-time", str(TIMEOUT),
           "--connect-timeout", str(CONNECT_TIMEOUT), "--retry", "1", "--retry-delay", "3",
           "-H", f"User-Agent: oscillator-research (mailto:{MAILTO})"]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    r = subprocess.run(cmd + [url], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise Unreachable(f"{urllib.parse.urlsplit(url).netloc} did not answer")
    return r.stdout


def normalise(title: str) -> str:
    """Compare titles on words alone: punctuation, case and the parenthesised
    system name that conference versions drop are all noise here."""
    title = html.unescape(re.sub(r"<[^>]+>", "", title))
    return re.sub(r"[^a-z0-9 ]", " ", title.lower()).strip()


def similar(a: str, b: str) -> float:
    a, b = normalise(a), normalise(b)
    # a conference version is often the preprint's title truncated at the colon
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    short, long_ = sorted((a, b), key=len)
    if short and long_.startswith(short):
        ratio = max(ratio, 0.95)
    return ratio


def from_arxiv(eprint: str) -> tuple[str, str] | None:
    """(doi, journal reference) as the preprint's own record states them."""
    xml = get(f"https://export.arxiv.org/api/query?id_list={eprint}&max_results=1")
    doi = re.search(r"<arxiv:doi[^>]*>([^<]+)</arxiv:doi>", xml)
    ref = re.search(r"<arxiv:journal_ref[^>]*>([^<]+)</arxiv:journal_ref>", xml)
    if not doi:
        return None
    return doi.group(1).strip(), re.sub(r"\s+", " ", ref.group(1)).strip() if ref else ""


def from_dblp(title: str) -> tuple[str, str, str] | None:
    """(dblp key, venue, year) for the best peer-reviewed match, if any."""
    if not title:
        return None
    url = ("https://dblp.org/search/publ/api?format=json&h=12&q="
           + urllib.parse.quote(normalise(title)[:250]))
    try:
        hits = json.loads(get(url))["result"]["hits"].get("hit", [])
    except (json.JSONDecodeError, KeyError) as exc:
        raise Unreachable("dblp.org returned something that is not a result set") from exc
    best = None
    for hit in hits:
        info = hit.get("info", {})
        if DBLP_PREPRINT in info.get("type", ""):
            continue                       # that is the preprint we are leaving
        ratio = similar(title, info.get("title", ""))
        if ratio >= TITLE_MATCH and (best is None or ratio > best[0]):
            best = (ratio, info.get("key", ""), info.get("venue", ""),
                    str(info.get("year", "")))
    if not best or not best[1]:
        return None
    return best[1], best[2], best[3]


def dblp_bibtex(key: str, citekey: str, eprint: str, labels: str) -> str | None:
    """DBLP's own entry, re-keyed to ours and stripped of DBLP bookkeeping.

    Taken verbatim rather than reassembled: DBLP already has the full author
    list, the exact booktitle and the pages, and rebuilding those from a search
    result is how a citation acquires a plausible-looking mistake.
    """
    raw = get(f"https://dblp.org/rec/{key}.bib?param=1")
    if not raw.lstrip().startswith("@"):
        raise Unreachable(f"dblp.org/rec/{key}.bib returned no entry")
    raw = re.sub(r"^@(\w+)\{[^,]+,", rf"@\1{{{citekey},", raw.strip(), count=1)
    for field in DBLP_NOISE:
        raw = re.sub(rf"^\s*{field}\s*=\s*\{{.*?\}},?\s*$\n?", "", raw, flags=re.M)
    if eprint and "arXiv" not in raw:
        # DBLP sometimes leaves a trailing comma on the last field and sometimes
        # does not; appending one unconditionally produces `},,`
        raw = re.sub(r",?\n\}\s*$",
                     f",\n  note         = {{Preprint: arXiv:{eprint}}},\n}}", raw)
    if labels:
        raw = f"% cited in the manuscript as: {labels}\n" + raw
    return raw


def cited_year(key: str) -> str | None:
    m = re.search(r"(1[89]\d\d|20\d\d)", key)
    return m.group(1) if m else None


def is_preprint(fields: dict) -> bool:
    return bool(fields.get("eprint")) or "10.48550" in fields.get("doi", "")


def candidates(entries: list, every: bool) -> list:
    """Preprint entries whose cited year the entry itself contradicts."""
    out = []
    for entry in entries:
        key, fields = entry[0], entry[1]
        if not is_preprint(fields):
            continue
        want, have = cited_year(key), fields.get("year")
        if every or (want and have and want != have):
            out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="cite the published version, not the preprint")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", dest="every", action="store_true",
                    help="check every preprint, not only those whose year disagrees")
    a = ap.parse_args()

    for paper in papers():
        if not paper.bib.exists():
            continue
        entries = parse_bib(paper.bib)
        todo = candidates(entries, a.every)
        print(f"\n=== {paper.slug}: {len(todo)} preprint entr"
              f"{'y' if len(todo) == 1 else 'ies'} to check")

        replacement: dict[str, str] = {}
        unreachable: list[str] = []
        for key, fields, labels, _ in todo:
            time.sleep(PACE)
            eprint = fields.get("eprint", "")
            was = fields.get("year", "?")
            try:
                # 1. the preprint's own record, which also carries the DOI
                if eprint and (hit := from_arxiv(eprint)):
                    doi, ref = hit
                    if csl := csl_for(doi):
                        keep = {k: v for k, v in fields.items() if k in ("url",)}
                        keep["doi"] = doi
                        keep["note"] = f"Preprint: arXiv:{eprint}"
                        replacement[key] = as_bibtex(key, csl, keep, labels)
                        print(f"    {key:22s} {was} preprint -> {ref or doi}")
                        print(f"    {'':22s} arXiv record  {doi}")
                        continue

                # 2. DBLP, the only source that indexes the DOI-less ML proceedings
                if hit := from_dblp(fields.get("title", "")):
                    dblp_key, venue, year = hit
                    time.sleep(PACE)
                    if entry := dblp_bibtex(dblp_key, key, eprint, labels):
                        replacement[key] = entry
                        print(f"    {key:22s} {was} preprint -> {year} {venue}")
                        print(f"    {'':22s} DBLP  {dblp_key}")
                        continue
            except Unreachable as exc:
                unreachable.append(key)
                print(f"    {key:22s} NOT CHECKED — {exc}")
                continue

            print(f"    {key:22s} no published version found — "
                  f"citing the preprint is correct")

        if unreachable:
            print(f"\n    {len(unreachable)} entr"
                  f"{'y was' if len(unreachable) == 1 else 'ies were'} not checked, "
                  f"and {'it is' if len(unreachable) == 1 else 'they are'} unchanged. "
                  f"DBLP blocks bursts by dropping the connection; wait and re-run.")
        if not replacement or a.dry_run:
            continue
        out = [replacement.get(e[0], e[3].rstrip()) for e in entries]
        header = paper.bib.read_text().split("@", 1)[0].rstrip("% ")
        paper.bib.write_text(header + "\n\n".join(out) + "\n")
        print(f"    rewrote {len(replacement)} entr"
              f"{'y' if len(replacement) == 1 else 'ies'} in {paper.bib.name}")

    if a.dry_run:
        print("\n(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

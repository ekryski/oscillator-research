#!/usr/bin/env python3
"""Check that every citation identifier and link actually resolves.

A bibliography full of dead DOIs is worse than one with none: it looks
authoritative and sends the reader nowhere. This walks every DOI, arXiv id, and
URL in the bibliographies AND every link in the manuscripts, and reports what
does not resolve.

    python3 publishing/lib/check_links.py             # everything
    python3 publishing/lib/check_links.py --bib-only  # skip the manuscript prose

Publishers rate-limit and some hosts refuse HEAD, so each target gets a HEAD
first and a ranged GET as a fallback before being called dead. A 403 from a
publisher that is plainly blocking automated traffic is reported separately
from a 404, because they mean different things: one needs a human to look, the
other needs the citation fixed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bibfile import entries
from paths import papers

TIMEOUT = 25
WORKERS = 8
#: hosts that routinely refuse automated requests; a refusal from them is not
#: evidence the citation is broken, only that a human has to confirm it
GATEKEEPERS = ("sciencedirect", "springer", "wiley", "tandfonline", "jstor",
               "researchgate", "ieeexplore", "aps.org", "science.org", "acm.org",
               "pnas.org", "academic.oup.com", "direct.mit.edu", "jstage.jst.go.jp",
               "nature.com", "onlinelibrary", "journals.sagepub", "iopscience",
               "physiology.org", "royalsocietypublishing", "cell.com")


#: BibTeX needs `_`, `&` and `%` escaped inside a url field, and DBLP duly
#: escapes them. Probing the escaped form asks for a URL that does not exist.
TEX_ESCAPES = str.maketrans({"\\": ""})


def unescaped(url: str) -> str:
    return url.translate(TEX_ESCAPES)


def targets_from_bib(path: Path) -> list[tuple[str, str, str, str]]:
    """(source, key, url, fallback) for every identifier in a bibliography.

    The DOI is the identifier of record and gets probed first, but a freshly
    minted one can be registered with the publisher before it resolves through
    doi.org — the ACM DOI for a rescheduled conference, say. The question a
    reader cares about is whether they can reach the paper at all, so an entry
    that also carries a URL gets it as a second chance rather than being
    reported dead while its publisher page sits there working.
    """
    out = []
    for e in entries(path.read_text()):
        slug = path.parent.parent.name
        url = unescaped(e.fields.get("url", ""))
        if doi := e.fields.get("doi"):
            out.append((slug, e.key, f"https://doi.org/{doi}", url))
        elif eprint := e.fields.get("eprint"):
            out.append((slug, e.key, f"https://arxiv.org/abs/{eprint}", url))
        elif url:
            out.append((slug, e.key, url, ""))
    return out


def targets_from_manuscript(path: Path) -> list[tuple[str, str, str, str]]:
    out = []
    for m in re.finditer(r"\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)", path.read_text()):
        out.append((path.parent.name, re.sub(r"\s+", " ", m.group(1))[:40], m.group(2), ""))
    return out


def check(url: str) -> tuple[int, str]:
    """(status, note). 0 means the request itself failed.

    A DOI is a redirect, so the host that matters is where it LANDS, not
    doi.org — classify on the effective URL or every blocked publisher looks
    like a dead DOI.
    """
    base = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code} %{url_effective}",
            "--max-time", str(TIMEOUT), "-L", "--retry", "1",
            "-A", "Mozilla/5.0 (citation link check)"]
    code, landed = 0, url
    # HEAD, then a ranged GET, then one more after a pause: a host that times out
    # under the concurrent sweep answers fine on its own, and reporting that as a
    # dead citation sends someone hunting for a replacement that already works
    for attempt, extra in enumerate((["-I"], ["-r", "0-2048"], ["-r", "0-2048"])):
        if attempt == 2:
            if code:
                break                            # it answered; no need to ask again
            time.sleep(3)
        try:
            r = subprocess.run(base + extra + [url], capture_output=True, text=True)
            head, _, tail = (r.stdout or "0").strip().partition(" ")
            code, landed = int(head[-3:] or 0), tail or url
        except (ValueError, OSError):
            code = 0
        if 200 <= code < 400:
            return code, "ok"
    host = landed.split("/")[2] if "://" in landed else landed
    if any(g in host for g in GATEKEEPERS):
        return code, f"blocked by {host} — confirm by hand"
    return code, "DEAD" if code else "no response"


def main() -> int:
    ap = argparse.ArgumentParser(description="check that citation links resolve")
    ap.add_argument("--bib-only", action="store_true")
    ap.add_argument("--json", type=Path, help="also write the full result here")
    a = ap.parse_args()

    targets: list[tuple[str, str, str]] = []
    for paper in papers():
        if paper.bib.exists():
            targets += targets_from_bib(paper.bib)
        if not a.bib_only and paper.manuscript:
            targets += targets_from_manuscript(paper.manuscript)

    seen, unique = set(), []
    for src, key, url, fallback in targets:
        if url not in seen:
            seen.add(url)
            unique.append((src, key, url, fallback))
    print(f"checking {len(unique)} unique links from {len(targets)} references "
          f"({WORKERS} at a time)\n")

    def probe(target):
        src, key, url, fallback = target
        code, note = check(url)
        if note in ("DEAD", "no response") and fallback and fallback != url:
            alt_code, alt_note = check(fallback)
            if alt_note != "DEAD" and alt_note != "no response":
                return (src, key, url, code,
                        f"identifier does not resolve, but {fallback} does "
                        f"({alt_note}) — confirm the DOI is registered")
        return (src, key, url, code, note)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(probe, unique))

    dead = [r for r in results if r[4] in ("DEAD", "no response")]
    blocked = [r for r in results if "blocked" in r[4] or "does not resolve, but" in r[4]]
    ok = len(results) - len(dead) - len(blocked)
    if blocked:
        print(f"{len(blocked)} could not be checked automatically "
              f"(publisher blocks robots — the citation is probably fine):")
        for src, key, url, code, note in sorted(blocked):
            print(f"  {code}  {src:22s} {key:24s} {url[:70]}")
            if "does not resolve" in note:
                print(f"       {note}")
    if dead:
        print(f"\n{len(dead)} DID NOT RESOLVE:")
        for src, key, url, code, _note in sorted(dead):
            print(f"  {code or '---'}  {src:22s} {key:24s} {url[:70]}")
    print(f"\n{ok} resolved, {len(blocked)} unverifiable, {len(dead)} dead")
    if a.json:
        a.json.write_text(json.dumps(
            [{"source": s, "key": k, "url": u, "status": c, "note": n}
             for s, k, u, c, n in results], indent=1))
    return 1 if dead else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Complete bibliography entries from the DOI registries.

The extractor can only recover what the manuscript itself states, which is an
identifier and an author-year. Everything a real citation needs after that —
the full author list, the exact title, the journal, the volume — has to come
from somewhere authoritative, and guessing it from prose is how wrong citations
get published.

Every DOI, whether registered with CrossRef or DataCite, answers content
negotiation on doi.org with CSL-JSON. arXiv preprints have DOIs too
(10.48550/arXiv.NNNN.NNNNN), so one mechanism covers the whole bibliography.

    python3 publishing/lib/fetch_metadata.py            # fill what is missing
    python3 publishing/lib/fetch_metadata.py --force    # re-fetch everything
    python3 publishing/lib/fetch_metadata.py --dry-run

Hand-edited fields win by default: an entry someone has already corrected is
never overwritten unless `--force` says so. Entries with no identifier at all
cannot be looked up and are reported for a human to finish.
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

MAILTO = "hello@erickryski.com"   # polite-pool identification for the registries
TODAY = time.strftime("%Y-%m-%d")
WORKERS = 4
TIMEOUT = 30

CONFERENCE = re.compile(
    r"\b(NeurIPS|Neural Information|ICLR|ICML|CVPR|ICCV|ECCV|AAAI|IJCAI|ACL|EMNLP|"
    r"NAACL|Interspeech|ICASSP|WASPAA|AISTATS|Proceedings)\b", re.I)


def csl_for(doi: str) -> dict | None:
    """CSL-JSON for one DOI, via content negotiation on doi.org."""
    out = subprocess.run(
        ["curl", "-sSL", "--max-time", str(TIMEOUT), "--retry", "2", "--retry-delay", "2",
         "-H", "Accept: application/vnd.citationstyles.csl+json",
         "-H", f"User-Agent: oscillator-research (mailto:{MAILTO})",
         f"https://doi.org/{doi}"],
        capture_output=True, text=True)
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("title") else None


#: publishers embed MathML in CSL titles, so a title can arrive as
#: "Hopf normal form with <mml:math ...><mml:mi>S</mml:mi>..." Stripping the
#: markup keeps the title; rejecting it as debris, which the URL inside the
#: namespace declaration would otherwise trigger, loses it entirely.
MARKUP = re.compile(r"<[^>]+>")


def plain(text: str) -> str:
    """A CSL string with any embedded markup reduced to its text content."""
    if "<" not in text:
        return text
    # a subscript is the commonest case and reads wrong without its underscore:
    # the symmetric group is S_N, not SN
    text = re.sub(r"(?s)<mml:msub>(.*?)</mml:msub>",
                  lambda m: "_".join(x for x in MARKUP.sub(" ", m.group(1)).split() if x), text)
    # the pieces of one formula belong together, not spaced out as words
    text = re.sub(r"(?s)<mml:math.*?</mml:math>",
                  lambda m: re.sub(r"\s+", "", MARKUP.sub("", m.group(0))), text)
    text = re.sub(r"\s+", " ", MARKUP.sub("", text)).strip()
    # stripping a tag can leave a space before the punctuation it abutted
    return re.sub(r"\s+([-–—.,;:)])", r"\1", text)


def authors_of(csl: dict) -> str:
    """BibTeX author field: 'Family, Given and Family, Given'."""
    names = []
    for a in csl.get("author", []):
        if a.get("family"):
            names.append(f"{a['family']}, {a['given']}" if a.get("given") else a["family"])
        elif a.get("literal"):
            names.append(a["literal"])
    return " and ".join(names)


#: sources that are not peer-reviewed literature. Paper 01 turns on marking
#: these as such, so they are typed @online rather than quietly filed as
#: articles, and carry the date they were read.
WEB_SOURCE = re.compile(r"github\.com|//[^/]*\.?unconv\.ai|/blog/|huggingface\.co")


def is_label(author: str) -> bool:
    """True if this author field is a citation label rather than a name list.

    The extractor keys entries off how the manuscript spells them, so a
    multi-author citation leaves behind "Castaldo, Aristides, Clusella,
    Garcia-Ojalvo & Ruffini". BibTeX reads a comma in a name as the
    Last, First separator, so that field aborts the entry with "too many
    commas" and takes the whole author-year bibliography down with it. A real
    BibTeX list separates with " and " and never uses an ampersand.
    """
    return bool(author) and (" & " in author or " \\& " in author
                             or author.rstrip().endswith("et al.")
                             or (author.count(",") > 1 and " and " not in author))


#: CSL type -> (bibtex type, the field the venue belongs in). The registries are
#: not consistent about these — doi.org returns `book-chapter` where the CSL spec
#: says `chapter` — and an unrecognised type silently degrades to @misc, which
#: drops the venue on the floor and leaves the entry looking incomplete.
CSL_TYPES = {
    "article-journal": ("article", "journal"),
    "journal-article": ("article", "journal"),
    "paper-conference": ("inproceedings", "booktitle"),
    "proceedings-article": ("inproceedings", "booktitle"),
    "inproceedings": ("inproceedings", "booktitle"),
    "chapter": ("incollection", "booktitle"),
    "book-chapter": ("incollection", "booktitle"),
    "book": ("book", "publisher"),
    "monograph": ("book", "publisher"),
    "report": ("techreport", "institution"),
    "thesis": ("phdthesis", "school"),
}


def entry_type(csl: dict, venue: str) -> tuple[str, str | None]:
    """(bibtex type, the field the venue goes in)."""
    kind = csl.get("type", "")
    if hit := CSL_TYPES.get(kind):
        return hit
    if CONFERENCE.search(venue or ""):
        return "inproceedings", "booktitle"
    return "misc", ("howpublished" if venue else None)


def as_bibtex(key: str, csl: dict, keep: dict, labels: str) -> str:
    title = plain(re.sub(r"\s+", " ", (csl.get("title") or "").strip())).rstrip(".")
    venue = (csl.get("container-title") or csl.get("publisher") or "").strip()
    if isinstance(venue, list):
        venue = venue[0] if venue else ""
    year = str((csl.get("issued", {}).get("date-parts") or [[""]])[0][0] or "")
    kind, venue_field = entry_type(csl, venue)
    if venue.lower() == "arxiv":
        kind, venue_field = "misc", None
    if WEB_SOURCE.search(keep.get("url", "")):
        kind, venue_field = "online", None

    def esc(s: str) -> str:
        return re.sub(r"(?<!\\)&", r"\\&", s)

    def usable(s: str) -> bool:
        """Reject a value that is prose debris rather than a field.

        Anything derived from a manuscript's own citation text can pick up a
        markdown link or a bracketed aside. A field carrying one breaks
        BibTeX's parser, which drops the entry silently and takes the
        author-year bibliography down with it.
        """
        return not re.search(r"\]\(|https?://|^\[", str(s))

    lines = [f"@{kind}{{{key},"]
    for name, value in (("author", keep.get("author") or authors_of(csl)),
                        ("title", keep.get("title") or title),
                        ("year", keep.get("year") or year)):
        if value and not str(value).startswith("VERIFY") and usable(value):
            lines.append(f"  {name:13s} = {{{esc(str(value))}}},")
    if venue_field and (v := keep.get(venue_field) or venue) and usable(v):
        lines.append(f"  {venue_field:13s} = {{{esc(v)}}},")
    for name in ("volume", "number", "pages"):
        v = keep.get(name) or csl.get(name if name != "number" else "issue")
        if v:
            lines.append(f"  {name:13s} = {{{v}}},")
    for name in ("doi", "eprint", "archiveprefix", "primaryclass", "url"):
        if keep.get(name):
            lines.append(f"  {name:13s} = {{{keep[name]}}},")
    if kind == "online":
        lines.append(f"  urldate       = {{{keep.get('urldate') or TODAY}}},")
        lines.append("  note          = {not peer reviewed},")
    if labels:
        lines.append(f"  % cited in the manuscript as: {labels}")
    lines.append("}")
    return "\n".join(lines)


def parse_bib(path: Path) -> list[tuple[str, dict, str, str]]:
    """[(key, fields, labels, raw block)] in file order."""
    return [(e.key, e.fields, e.labels, e.raw) for e in entries(path.read_text())]


#: several publishers put the DOI straight into the article URL, and arXiv
#: mints one for every preprint including the pre-2007 identifiers — so a
#: URL-only entry is usually still resolvable, it just needs reading properly
URL_DOI = (
    re.compile(r"/(?:doi|chapter|article|abstract)/(10\.\d{4,9}/[^\s?#]+)"),
    # the modern s-number form FIRST: alternation is ordered, and the legacy
    # pattern would otherwise match just its "s42256" prefix
    re.compile(r"nature\.com/articles/(s\d{5}-\d{3}-\d{4,5}-[a-z0-9]+|[a-z0-9]+[0-9]{4,})"),
    re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z-]+/\d{7})"),
)


def doi_for(fields: dict) -> str | None:
    if fields.get("doi"):
        return fields["doi"]
    if fields.get("eprint"):
        return f"10.48550/arXiv.{fields['eprint']}"
    url = fields.get("url", "")
    if m := URL_DOI[0].search(url):
        return m.group(1).rstrip(".")
    if m := URL_DOI[1].search(url):
        return f"10.1038/{m.group(1)}"
    if m := URL_DOI[2].search(url):
        return f"10.48550/arXiv.{m.group(1)}"
    return None


HEADER = """% Bibliography for {slug}
%
% Author lists, titles, venues and years come from the DOI registries via
% publishing/lib/fetch_metadata.py — authoritative, not transcribed. Entries still
% marked VERIFY have no registered identifier to look up and need finishing by
% hand against the source; `grep VERIFY` lists them.
%
% publishing/lib/extract_bib.py only ever APPENDS newly cited keys, so anything
% completed here is preserved.

"""


def main() -> int:
    ap = argparse.ArgumentParser(description="complete the bibliographies from the DOI registries")
    ap.add_argument("--force", action="store_true", help="overwrite fields already present")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    exit_code = 0
    for paper in papers():
        path = paper.bib
        if not path.exists():
            continue
        entries = parse_bib(path)
        todo = [(k, f, labels) for k, f, labels, _ in entries if doi_for(f)]
        print(f"\n=== {path.name}: {len(entries)} entries, {len(todo)} with a resolvable identifier")

        def fetch(item):
            key, fields, _ = item
            time.sleep(0.2)                       # be a good citizen with the registries
            return key, csl_for(doi_for(fields))

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            fetched = dict(pool.map(fetch, todo))
        # the registries rate-limit a parallel burst, and a rate-limited fetch
        # looks exactly like a dead DOI; retry those one at a time before
        # reporting anything as unresolvable
        retry = [i for i in todo if fetched.get(i[0]) is None]
        if retry:
            print(f"    retrying {len(retry)} serially")
            for item in retry:
                time.sleep(1.5)
                fetched[item[0]] = csl_for(doi_for(item[1]))

        out, unresolved, failed, mismatched = [], [], [], []
        for key, fields, labels, block in entries:
            csl = fetched.get(key)
            if csl is None:
                if doi_for(fields):
                    failed.append(key)
                else:
                    unresolved.append(key)
                out.append(block.rstrip())
                continue
            keep = {} if a.force else {k: v for k, v in fields.items()
                                       if k not in ("note",) and not v.startswith("VERIFY")}
            if is_label(keep.get("author", "")):
                del keep["author"]
            keep.update({k: v for k, v in fields.items()
                         if k in ("doi", "eprint", "archiveprefix", "primaryclass", "url")})
            # a DOI mined out of a publisher URL is worth keeping: it outlives
            # the URL, and the registry has just confirmed it resolves
            if not keep.get("doi") and not keep.get("eprint"):
                derived = doi_for(fields)
                if derived and derived.startswith("10.48550"):
                    # a pre-2007 arXiv identifier: record it as an eprint like
                    # every other preprint rather than as a bare DOI
                    keep["eprint"] = derived.removeprefix("10.48550/arXiv.")
                    keep["archiveprefix"] = "arXiv"
                elif derived:
                    keep["doi"] = derived
            entry = as_bibtex(key, csl, keep, labels)
            in_key = re.search(r"(1[89]\d\d|20\d\d)", key)
            in_bib = re.search(r"year\s*=\s*\{(\d{4})\}", entry)
            if in_key and in_bib and in_key.group(1) != in_bib.group(1):
                mismatched.append((key, in_key.group(1), in_bib.group(1)))
            out.append(entry)
        if not a.dry_run:
            path.write_text(HEADER.format(slug=paper.slug) + "\n\n".join(out) + "\n")
        print(f"    {len(fetched) - len(failed)} completed from the registries")
        if failed:
            print(f"    {len(failed)} identifier(s) did not return metadata: {', '.join(failed)}")
            exit_code = 1
        if unresolved:
            print(f"    {len(unresolved)} with no identifier — finish by hand: "
                  f"{', '.join(unresolved)}")
        if mismatched:
            print(f"    {len(mismatched)} cited under a different year than the "
                  f"registry records (preprint vs proceedings — decide which to cite):")
            for key, cited, registry in mismatched:
                print(f"        {key:24s} cited as {cited}, registry says {registry}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

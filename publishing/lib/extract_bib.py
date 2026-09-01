#!/usr/bin/env python3
"""Build a starter BibTeX file and citation map from a manuscript.

The manuscripts are written for reading, not for BibTeX: paper 01 cites with
inline markdown links carrying a DOI or arXiv URL, paper 02 cites with
[Author Year] brackets against a trailing reference list. Both forms carry a
stable identifier (a DOI, an arXiv id, or a URL) and an author-year, which is
enough to generate an entry that resolves — and enough to key it.

What this CANNOT recover is the parts the prose never states: full author
lists, journal names, volumes, pages. Entries missing those are marked
`VERIFY = {...}` so an incomplete bibliography is loud rather than silent.
Run `refresh` after editing a manuscript; existing hand-completed entries are
preserved and only new keys are appended.

    python publishing/lib/extract_bib.py papers/01-evidence-audit/*-DRAFT.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import paper_for

# [text](url) where url is a DOI, arXiv, or any http(s) target
LINK = re.compile(r"\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)")
# [Author Year] / [Author & Author Year] / [Author et al. Year], possibly ; joined
BRACKET = re.compile(r"\[([A-Za-z][^\]\[]{2,110}?)\]")
#: a trailing letter disambiguates two works by the same authors in the same year
#: (Huang et al. 2026a / 2026b). Without it in the year group both labels derive the
#: same citekey, silently collapse, and one of the two works is never cited at all.
AUTHOR_YEAR = re.compile(r"^(?P<who>.+?)[\s,]*\(?(?P<year>(?:1[89]\d\d|20\d\d)[a-z]?)\)?$")
STOPWORDS = {"the", "a", "an", "and", "of", "in", "on", "for", "to"}
# bracket text that looks like a citation but is the prose talking ABOUT citations,
# or a cross-reference to the manuscript's own sections
NOT_A_CITATION = re.compile(
    r"^(Author\s+Year|Section|Fig|Figure|Table|Appendix|Eq|Equation|H\d|R\d+|"
    r"MET|NOT MET|sic|\d+)\b", re.I)


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return re.sub(r"-+", "-", text)


def split_author_year(text: str) -> tuple[str, str | None]:
    m = AUTHOR_YEAR.match(text.strip().rstrip("."))
    if not m:
        return text.strip(), None
    return m.group("who").strip(" ,&"), m.group("year")


def citekey(who: str, year: str | None) -> str:
    lead = re.split(r"[,&]| et al| and ", who)[0].strip()
    lead = " ".join(w for w in lead.split() if w.lower() not in STOPWORDS)
    base = slug(lead) or "ref"
    return f"{base}{year}" if year else base


def identifier(url: str) -> dict[str, str]:
    """DOI / arXiv id / bare URL, whichever the link actually carries."""
    if m := re.search(r"doi\.org/(10\.[^\s)]+)", url):
        return {"doi": m.group(1)}
    if m := re.search(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})", url):
        return {"eprint": m.group(1), "archiveprefix": "arXiv", "primaryclass": "cs.LG"}
    return {"url": url}


def entries_from_links(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for match in LINK.finditer(text):
        label, url = match.group(1), match.group(2)
        label = re.sub(r"\s+", " ", label).strip()
        if label.startswith("!") or url.endswith((".png", ".svg", ".jpg")):
            continue
        who, year = split_author_year(label)
        key = citekey(who, year)
        entry = out.setdefault(key, {"_labels": set()})
        entry["_labels"].add(label)
        entry.update(identifier(url))
        if year:
            entry["year"] = year
        # "Hopfield (1982)" -> author; "coRNN" -> a system/short name, so title
        entry["author" if year else "title"] = who
    return out


def entries_from_brackets(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for raw in BRACKET.finditer(text):
        for piece in re.sub(r"\s+", " ", raw.group(1)).split(";"):
            who, year = split_author_year(piece)
            if NOT_A_CITATION.match(piece.strip()):
                continue
            if not year and not re.match(r"^[A-Za-z][A-Za-z.&' -]+$", who.strip()):
                continue  # prose in brackets, not a citation
            if len(who) > 60 or "(" in who:
                continue
            key = citekey(who, year)
            entry = out.setdefault(key, {"_labels": set()})
            entry["_labels"].add(piece.strip())
            entry["author"] = who
            if year:
                entry["year"] = year
    return out


def bibtex_escape(value: str) -> str:
    """BibTeX reserves & — an unescaped one aborts the LaTeX run."""
    return re.sub(r"(?<!\\)&", r"\\&", value)


def format_entry(key: str, fields: dict) -> str:
    labels = sorted(fields.pop("_labels", []))
    kind = "@article" if "doi" in fields else ("@misc" if "url" in fields else "@misc")
    lines = [f"{kind}{{{key},"]
    if "author" in fields:
        lines.append(f"  author        = {{{bibtex_escape(fields['author'])}}},")
    lines.append(f"  title         = "
                 f"{{{bibtex_escape(fields.get('title', 'VERIFY: title not stated in the manuscript'))}}},")
    if "year" in fields:
        lines.append(f"  year          = {{{fields['year']}}},")
    for f in ("doi", "eprint", "archiveprefix", "primaryclass", "url"):
        if f in fields:
            lines.append(f"  {f:13s} = {{{fields[f]}}},")
    if "doi" not in fields and "eprint" not in fields and "url" not in fields:
        lines.append("  note          = {VERIFY: no identifier in the manuscript},")
    seen = "; ".join(re.sub(r"\s+", " ", lb) for lb in labels)
    lines.append(f"  % cited in the manuscript as: {seen}")
    lines.append("}")
    return "\n".join(lines)


def identifiers_in(path: Path) -> dict[str, dict]:
    """{citekey: identifier fields} from an existing .bib, for cross-filling."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for block in re.split(r"(?=^@)", path.read_text(), flags=re.M):
        m = re.match(r"@\w+\{([^,]+),", block)
        if not m:
            continue
        fields = {f: v for f, v in re.findall(r"^\s*(\w+)\s*=\s*\{([^}]*)\}", block, re.M)
                  if f in ("doi", "eprint", "archiveprefix", "primaryclass", "url")}
        if fields:
            out[m.group(1)] = fields
    return out


def existing_keys(path: Path, citemap: Path) -> set[str]:
    """Keys already in the bibliography, plus every key merged into one of them.

    Without the second half, a work cited two ways would have its merged-away
    spelling re-added on the next run, undoing the merge every time.
    """
    keys = set()
    if path.exists():
        keys |= set(re.findall(r"^@\w+\{([^,]+),", path.read_text(), re.M))
    if citemap.exists():
        for canonical, labels in json.loads(citemap.read_text()).items():
            keys.add(canonical)
            for label in labels:
                keys.add(citekey(*split_author_year(label)))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description="build a starter .bib from a manuscript")
    ap.add_argument("manuscript", type=Path)
    ap.add_argument("--out", type=Path,
                    help="bib path (default: the paper's own references/bibliography.bib)")
    ap.add_argument("--inherit", type=Path, action="append", default=[],
                    help="fill identifiers for shared keys from another .bib "
                         "(the two manuscripts cite many of the same works)")
    a = ap.parse_args()

    text = a.manuscript.read_text()
    found = {**entries_from_brackets(text), **entries_from_links(text)}
    for source in a.inherit:
        for key, ident in identifiers_in(source).items():
            if key in found and not ({"doi", "eprint", "url"} & set(found[key])):
                found[key].update(ident)
    paper = paper_for(a.manuscript)
    out = a.out or (paper.bib if paper else
                    a.manuscript.parent / "references" / "bibliography.bib")
    out.parent.mkdir(parents=True, exist_ok=True)

    map_path = paper.citemap if paper else out.with_suffix(".citemap.json")
    have = existing_keys(out, map_path)
    new = {k: v for k, v in sorted(found.items()) if k not in have}
    body = "\n\n".join(format_entry(k, dict(v)) for k, v in new.items())
    if have:
        with out.open("a") as fh:
            fh.write("\n\n" + body + "\n" if body else "")
        print(f"{out}: {len(have)} kept, {len(new)} appended")
    else:
        header = (f"% Bibliography for {a.manuscript.name}\n"
                  f"% Generated by publishing/lib/extract_bib.py from the manuscript's own\n"
                  f"% citations. Entries marked VERIFY need metadata the prose never states\n"
                  f"% (full author lists, journal, volume, pages) before submission.\n\n")
        out.write_text(header + body + "\n")
        print(f"{out}: {len(new)} entries written")

    # merged entries own their aliases, so only ADD keys the map does not cover
    citemap = json.loads(map_path.read_text()) if map_path.exists() else {}
    covered = {citekey(*split_author_year(lb)) for lbs in citemap.values() for lb in lbs}
    for key, v in sorted(found.items()):
        if key not in citemap and key not in covered:
            citemap[key] = sorted(v["_labels"])
    map_path.write_text(json.dumps(dict(sorted(citemap.items())), indent=1) + "\n")
    unresolved = sum(1 for v in found.values() if not ({"doi", "eprint", "url"} & set(v)))
    print(f"{map_path}: {len(citemap)} keys ({unresolved} without an identifier)")


if __name__ == "__main__":
    main()

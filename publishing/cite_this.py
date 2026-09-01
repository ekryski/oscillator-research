#!/usr/bin/env python3
"""Emit citation metadata for the papers in this repository.

Someone who wants to cite this work should not have to hand-assemble an entry
from a README. This reads the same per-paper front matter the build uses and
writes every format a reader is likely to want:

    CITATION.cff                          the repository's own citation file
                                          (GitHub renders a "Cite this
                                          repository" button from it)
    papers/<paper>/README.md              the citation section, injected between
                                          markers so it cannot drift
    papers/<paper>/metadata/citation.bib  BibTeX
    papers/<paper>/metadata/citation.ris  RIS (EndNote, Zotero, Mendeley)
    papers/<paper>/metadata/citation.txt  APA, MLA, Chicago, IEEE, Harvard
    papers/<paper>/metadata/citation.md   a ready-to-paste block

The author list and keywords come from each paper's metadata/paper.yaml and the
title and abstract from the manuscript itself, so each of them has one source and
updating it there updates every format.

    python3 publishing/cite_this.py                # write all of it
    python3 publishing/cite_this.py --print apa    # just show one style
"""
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import abstract as abstract_mod
import title as title_mod
from paths import ROOT
from paths import papers as paper_folders

REPO_URL = "https://github.com/ekryski/oscillator-research"
#: the citation section is regenerated in place between these markers
START, END = "<!-- citation:start -->", "<!-- citation:end -->"


def load_metadata(path: Path) -> dict:
    """A deliberately small YAML reader: these files are flat, and depending on
    PyYAML for six keys would put a dependency between a reader and a citation."""
    text = path.read_text()
    out: dict = {"authors": []}
    if m := re.search(r'^subtitle:\s*"?(.+?)"?\s*$', text, re.M):
        out["subtitle"] = m.group(1)
    for block in re.findall(r"^\s*-\s*name:\s*(.+?)\s*$"
                            r"(?:\n\s*affiliation:\s*(.+?)\s*$)?"
                            r"(?:\n\s*email:\s*(.+?)\s*$)?", text, re.M):
        name, affiliation, email = block
        out["authors"].append({"name": name, "affiliation": affiliation, "email": email})
    if block := re.search(r"^keywords:\n((?:\s{2}-\s.*\n)+)", text, re.M):
        out["keywords"] = [line.strip()[2:].strip()
                           for line in block.group(1).splitlines() if line.strip()]
    return out


def full_title(meta: dict) -> str:
    return f"{meta['title']}: {meta['subtitle']}" if meta.get("subtitle") else meta["title"]


def surname(name: str) -> str:
    return name.split()[-1]


def initials(name: str) -> str:
    return " ".join(f"{p[0]}." for p in name.split()[:-1])


def year() -> str:
    try:
        stamp = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=format:%Y"],
                               cwd=ROOT, capture_output=True, text=True, check=True)
        return stamp.stdout.strip() or str(datetime.date.today().year)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return str(datetime.date.today().year)


TITLE_STOPWORDS = {"a", "an", "the", "from", "on", "of", "in", "to", "for", "and", "with"}


def citekey(meta: dict, slug: str) -> str:
    """author + year + the title's first distinctive word — the convention most
    reference managers produce, so a hand-typed \\cite is likely to match."""
    lead = surname(meta["authors"][0]["name"]).lower() if meta["authors"] else "anon"
    words = (re.sub(r"[^a-z]", "", w.lower()) for w in meta["title"].split())
    word = next((w for w in words if w and w not in TITLE_STOPWORDS), "paper")
    return f"{lead}{year()}{word}"


def bibtex(meta: dict, slug: str) -> str:
    authors = " and ".join(a["name"] for a in meta["authors"])
    return "\n".join([
        f"@techreport{{{citekey(meta, slug)},",
        f"  author      = {{{authors}}},",
        f"  title       = {{{full_title(meta)}}},",
        f"  year        = {{{year()}}},",
        "  institution = {Independent research},",
        "  type        = {Preprint},",
        f"  url         = {{{REPO_URL}/blob/main/papers/{slug}/}},",
        f"  keywords    = {{{', '.join(meta['keywords'])}}},",
        "}",
    ])


def ris(meta: dict, slug: str) -> str:
    lines = ["TY  - RPRT"]
    lines += [f"AU  - {surname(a['name'])}, {initials(a['name'])}" for a in meta["authors"]]
    lines += [f"TI  - {full_title(meta)}", f"PY  - {year()}",
              f"UR  - {REPO_URL}/blob/main/papers/{slug}/"]
    lines += [f"KW  - {k}" for k in meta["keywords"]]
    if meta.get("abstract"):
        lines.append(f"AB  - {meta['abstract']}")
    lines.append("ER  - ")
    return "\n".join(lines)


def styles(meta: dict, slug: str) -> dict[str, str]:
    y, url = year(), f"{REPO_URL}/blob/main/papers/{slug}/"
    names = [a["name"] for a in meta["authors"]]
    apa_names = ", ".join(f"{surname(n)}, {initials(n)}" for n in names)
    mla_names = " and ".join(f"{surname(n)}, {' '.join(n.split()[:-1])}" for n in names)
    ieee_names = ", ".join(f"{initials(n)} {surname(n)}" for n in names)
    title = full_title(meta)
    return {
        "apa": f"{apa_names} ({y}). {title} [Preprint]. {url}",
        "mla": f'{mla_names}. "{title}." {y}, {url}.',
        "chicago": f'{apa_names.rstrip(".")}. {y}. "{title}." Preprint. {url}.',
        "ieee": f'{ieee_names}, "{title}," preprint, {y}. [Online]. Available: {url}',
        "harvard": f"{apa_names} ({y}) '{title}'. Preprint. Available at: {url}",
    }


def markdown_block(meta: dict, slug: str) -> str:
    """The citation section a paper's README carries.

    Generated rather than hand-written, so a title or author change lands in
    one place and cannot drift out of sync with the built formats.
    """
    styles_ = styles(meta, slug)
    return "\n".join([
        "## Citing this paper",
        "",
        f"{styles_['apa']}",
        "",
        "```bibtex",
        bibtex(meta, slug),
        "```",
        "",
        "Other formats, regenerated by `python3 publishing/cite_this.py`:",
        "",
        "- [BibTeX](metadata/citation.bib)",
        "- [RIS](metadata/citation.ris) (EndNote, Zotero, Mendeley)",
        "- [APA, MLA, Chicago, IEEE, Harvard](metadata/citation.txt)",
        "- [CITATION.cff](../../CITATION.cff) for the repository as a whole",
        "",
        "Please cite the version you actually used: these manuscripts are "
        "drafts, and every number in them is reproducible from the record "
        "shipped beside them.",
        "",
    ])


def citation_cff(papers: list[tuple[str, dict]]) -> str:
    lead = papers[0][1]
    authors = "\n".join(
        f"  - family-names: {surname(a['name'])}\n"
        f"    given-names: {' '.join(a['name'].split()[:-1])}"
        + (f"\n    email: {a['email']}" if a.get("email") else "")
        + (f"\n    affiliation: {a['affiliation']}" if a.get("affiliation") else "")
        for a in lead["authors"])
    refs = "\n".join(
        f"  - type: report\n"
        f"    title: \"{full_title(m)}\"\n"
        f"    year: {year()}\n"
        f"    url: \"{REPO_URL}/blob/main/papers/{slug}/\""
        for slug, m in papers)
    keywords = sorted({k for _, m in papers for k in m["keywords"]})
    return f"""# Citation metadata for this repository.
# Generated by publishing/cite_this.py from papers/*/metadata/paper.yaml and
# the manuscripts' own titles and abstracts — edit those.
cff-version: 1.2.0
message: "If you use this research, please cite the relevant paper below."
type: software
title: "Oscillator Research"
abstract: >-
  Scientific research into coupled oscillators as a computational substrate for
  speech. Every experimental paper ships with the code and the raw per-run data
  that produced its numbers.
authors:
{authors}
repository-code: "{REPO_URL}"
url: "{REPO_URL}"
license: Apache-2.0
date-released: "{datetime.date.today().isoformat()}"
keywords:
{chr(10).join(f'  - {k}' for k in keywords)}
references:
{refs}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="citation metadata for these papers")
    ap.add_argument("--print", dest="style",
                    choices=("apa", "mla", "chicago", "ieee", "harvard", "bibtex", "ris"),
                    help="print one style to stdout instead of writing files")
    a = ap.parse_args()

    papers = []
    for p in paper_folders():
        if not p.metadata.exists():
            continue
        meta = load_metadata(p.metadata)
        # the abstract is written in the manuscript, not the metadata file
        if p.manuscript:
            # both come from the manuscript, so a citation cannot claim a title
            # or an abstract the paper itself does not carry
            meta["title"] = title_mod.read(p.manuscript)
            meta["abstract"] = abstract_mod.read(p.manuscript)
        papers.append((p, meta))
    if not papers:
        raise SystemExit("no papers with a metadata/paper.yaml")

    if a.style:
        for paper, meta in papers:
            body = {"bibtex": bibtex, "ris": ris}.get(a.style)
            print(body(meta, paper.slug) if body else styles(meta, paper.slug)[a.style])
            print()
        return

    for paper, meta in papers:
        out, slug = paper.citations, paper.slug
        out.mkdir(parents=True, exist_ok=True)
        (out / "citation.bib").write_text(bibtex(meta, slug) + "\n")
        (out / "citation.ris").write_text(ris(meta, slug) + "\n")
        text = "\n\n".join(f"{name.upper()}\n{body}" for name, body in styles(meta, slug).items())
        (out / "citation.txt").write_text(f"{full_title(meta)}\n\n{text}\n")
        (out / "citation.md").write_text(markdown_block(meta, slug))
        print(f"wrote {out.relative_to(ROOT)}/citation.{{bib,ris,txt,md}}")

    for paper, meta in papers:
        slug = paper.slug
        readme = paper.dir / "README.md"
        if not readme.exists():
            print(f"  (no README at {readme.relative_to(ROOT)} — skipped)")
            continue
        body = readme.read_text()
        block = markdown_block(meta, slug).rstrip()
        wrapped = f"{START}\n{block}\n{END}"
        if START in body and END in body:
            head, rest = body.split(START, 1)
            body = head + wrapped + rest.split(END, 1)[1]
        else:
            body = body.rstrip() + "\n\n" + wrapped + "\n"
        readme.write_text(body)
        print(f"updated {readme.relative_to(ROOT)}")

    cff = ROOT / "CITATION.cff"
    cff.write_text(citation_cff([(p.slug, m) for p, m in papers]))
    print(f"wrote {cff.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

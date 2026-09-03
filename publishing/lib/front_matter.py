#!/usr/bin/env python3
"""Markdown for the parts of a submission that live in paper.yaml, not the manuscript.

A journal's Word route wants the title-page details, the keywords and the
declarations inside the manuscript file, a title page as a file of its own, and
the highlights as another file with "highlights" in its name. The LaTeX route
gets the same pieces from the venue template. This script is the Word route's
source for them, from the same paper.yaml, so the two never drift.

    front_matter.py <paper.yaml> author-lines         one line per title-block paragraph
    front_matter.py <paper.yaml> keywords             the keywords paragraph
    front_matter.py <paper.yaml> declarations         CRediT, competing interests, funding, data
    front_matter.py <paper.yaml> highlights           the highlights file, checked against the limits
    front_matter.py <paper.yaml> title-page           the separate title-page file
    front_matter.py <paper.yaml> docx-body --body F   F with the keywords first, the declarations last

Headings in `declarations` are written at the manuscript's own top level (`##`)
so they land beside its sections after the build's heading shift; the two
standalone files use `#`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_meta import authors, items, scalar  # noqa: E402

#: Elsevier's limits: three to five highlights, at most 85 characters each, spaces included
HIGHLIGHT_COUNT = (3, 5)
HIGHLIGHT_MAX_CHARS = 85
#: the declarations a journal wants as named sections, in the order they are printed
DECLARATIONS = (("competing-interests", "Declaration of competing interest"),
                ("funding", "Funding"),
                ("data-availability", "Data availability"))
AFFILIATION_MARKS = "abcdefghijklmnopqrstuvwxyz"


def affiliations(people: list[dict[str, str]]) -> list[str]:
    """The distinct affiliations, in order of first appearance."""
    out: list[str] = []
    for p in people:
        if p.get("affiliation") and p["affiliation"] not in out:
            out.append(p["affiliation"])
    return out


def corresponding(people: list[dict[str, str]]) -> dict[str, str]:
    """The corresponding author: the one marked, else the first with an email."""
    marked = [p for p in people if p.get("corresponding", "").lower() == "true"]
    with_email = [p for p in people if p.get("email")]
    return (marked or with_email or [{}])[0]


def author_lines(meta: str) -> list[str]:
    """The paragraphs under the title in a Word manuscript: names, affiliations, correspondence."""
    people = authors(meta)
    lines = [p["name"] for p in people] + affiliations(people)
    lead = corresponding(people)
    if lead:
        line = f"Corresponding author: {lead['name']}, {lead['email']}"
        if lead.get("orcid"):
            line += f" (ORCID {lead['orcid']})"
        lines.append(line)
    return lines


def keywords_paragraph(meta: str) -> str:
    kws = items(meta, "keywords")
    return f"**Keywords:** {'; '.join(kws)}" if kws else ""


def declarations(meta: str) -> str:
    """The end-matter sections a journal wants, from the fields that are present."""
    parts = []
    credits = [f"**{p['name']}:** {p['credit'].rstrip('.')}." for p in authors(meta) if p.get("credit")]
    if credits:
        parts.append("## CRediT authorship contribution statement {-}\n\n" + "\n\n".join(credits))
    for key, heading in DECLARATIONS:
        value = scalar(meta, key)
        if value:
            parts.append(f"## {heading} {{-}}\n\n{value}")
    return "\n\n".join(parts)


def highlights(meta: str) -> str:
    """The highlights file; empty when the paper has none; an error when they break the limits."""
    hs = items(meta, "highlights")
    if not hs:
        return ""
    lo, hi = HIGHLIGHT_COUNT
    problems = []
    if not lo <= len(hs) <= hi:
        problems.append(f"{len(hs)} highlights, and the journal wants {lo} to {hi}")
    problems += [f"{len(h)} characters, over the {HIGHLIGHT_MAX_CHARS} allowed: {h!r}"
                 for h in hs if len(h) > HIGHLIGHT_MAX_CHARS]
    if problems:
        raise SystemExit("highlights: " + "; ".join(problems))
    return "# Highlights\n\n" + "\n".join(f"- {h}" for h in hs) + "\n"


def title_page(meta: str) -> str:
    """The title-page file: authors with affiliation marks, correspondence, ORCID, acknowledgements."""
    people = authors(meta)
    affs = affiliations(people)

    def mark(p: dict[str, str]) -> str:
        return f"^{AFFILIATION_MARKS[affs.index(p['affiliation'])]}^" if p.get("affiliation") in affs else ""

    lines = ["**Authors.** " + ", ".join(p["name"] + mark(p) for p in people)]
    if affs:
        lines.append("**Affiliations.** " + "; ".join(f"^{AFFILIATION_MARKS[i]}^ {a}" for i, a in enumerate(affs)))
    lead = corresponding(people)
    if lead:
        lines.append(f"**Corresponding author.** {lead['name']}, {lead['email']}")
    orcids = [f"{p['name']} {p['orcid']}" for p in people if p.get("orcid")]
    if orcids:
        lines.append("**ORCID.** " + "; ".join(orcids))
    lines.append("**Acknowledgements.** " + (scalar(meta, "acknowledgements") or "None."))
    funding = scalar(meta, "funding")
    if funding:
        lines.append(f"**Funding.** {funding}")
    return "\n\n".join(lines) + "\n"


def docx_body(meta: str, body: str) -> str:
    """The manuscript body as the Word route wants it: keywords under the abstract, declarations at the end."""
    head = keywords_paragraph(meta)
    decl = declarations(meta)
    out = (head + "\n\n" if head else "") + body.rstrip() + "\n"
    if decl:
        out += "\n" + decl + "\n"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown for the submission pieces that live in paper.yaml")
    ap.add_argument("paper_yaml", type=Path)
    ap.add_argument("part", choices=["author-lines", "keywords", "declarations", "highlights",
                                     "title-page", "docx-body"])
    ap.add_argument("--body", type=Path, help="the preprocessed manuscript body (docx-body)")
    a = ap.parse_args()
    meta = a.paper_yaml.read_text()
    if a.part == "author-lines":
        print("\n".join(author_lines(meta)))
    elif a.part == "keywords":
        print(keywords_paragraph(meta))
    elif a.part == "declarations":
        print(declarations(meta))
    elif a.part == "highlights":
        sys.stdout.write(highlights(meta))
    elif a.part == "title-page":
        sys.stdout.write(title_page(meta))
    else:
        if a.body is None:
            ap.error("docx-body needs --body")
        sys.stdout.write(docx_body(meta, a.body.read_text()))


if __name__ == "__main__":
    main()

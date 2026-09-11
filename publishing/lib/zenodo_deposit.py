#!/usr/bin/env python3
"""Create a Zenodo deposit for a paper's preprint PDF, and reserve its DOI.

A journal submission wants a citable preprint and arXiv wants a DOI-bearing
version to cross-reference, and both want the same PDF. This builds the Zenodo
record's metadata from the paper's own front matter (`metadata/paper.yaml`),
title and abstract, so the record never drifts from the manuscript, uploads
the preprint PDF, and stops at a DRAFT with a reserved DOI. Publishing is a
separate, explicit step, because a published Zenodo record cannot be deleted.

    export ZENODO_TOKEN=...            # a personal access token with deposit:write
    python3 publishing/lib/zenodo_deposit.py papers/01-evidence-audit --dry-run
    python3 publishing/lib/zenodo_deposit.py papers/01-evidence-audit
    python3 publishing/lib/zenodo_deposit.py papers/01-evidence-audit --publish

`--sandbox` targets sandbox.zenodo.org (token in ZENODO_SANDBOX_TOKEN) for a
rehearsal; sandbox DOIs are not real. The token is read from the environment
only, never from a flag, so it does not land in shell history.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abstract as abstract_mod  # noqa: E402
import title as title_mod  # noqa: E402

#: the two Zenodo instances; the sandbox mints throwaway DOIs for rehearsal
HOSTS = {"production": "https://zenodo.org", "sandbox": "https://sandbox.zenodo.org"}
TOKEN_VARS = {"production": "ZENODO_TOKEN", "sandbox": "ZENODO_SANDBOX_TOKEN"}
#: every manuscript in this repository is released under the same licence,
#: the one TMLR and arXiv both accept and the one CITATION.cff records
LICENSE = "cc-by-4.0"
#: the record type Zenodo uses for a paper that has not been through review
UPLOAD_TYPE, PUBLICATION_TYPE = "publication", "preprint"
#: the repository the paper's code and record live in, linked from the record
REPOSITORY = "https://github.com/ekryski/oscillator-research"
#: `- name: ...` entries under `author:` in paper.yaml, and the keys each carries
AUTHOR_BLOCK = re.compile(r"^\s*-\s*name:\s*(?P<name>.+?)\s*$(?P<rest>(?:\n\s{4,}\w+:.*$)*)", re.M)
FIELD = re.compile(r"^\s+(\w+):\s*(.+?)\s*$", re.M)
KEYWORD_LINE = re.compile(r"^\s*-\s*(.+?)\s*$", re.M)


def authors(paper_yaml: str) -> list[dict[str, str]]:
    """The `author:` list of paper.yaml as Zenodo creators.

    paper.yaml is parsed with the same regex approach as byline.py rather
    than PyYAML, so this script has no dependency the rest of the pipeline
    lacks. Zenodo wants "Family, Given"; the front matter keeps display order.
    """
    out = []
    for m in AUTHOR_BLOCK.finditer(paper_yaml):
        fields = dict(FIELD.findall(m.group("rest")))
        given, _, family = m.group("name").strip('" ').rpartition(" ")
        creator = {"name": f"{family}, {given}" if given else family}
        if fields.get("affiliation"):
            creator["affiliation"] = fields["affiliation"].strip('" ')
        if fields.get("orcid"):
            creator["orcid"] = fields["orcid"].strip('" ')
        out.append(creator)
    return out


def keywords(paper_yaml: str) -> list[str]:
    """The `keywords:` list of paper.yaml, up to the next top-level key."""
    m = re.search(r"^keywords:\s*\n((?:\s*-\s*.+\n?)+)", paper_yaml, re.M)
    return [k.strip('" ') for k in KEYWORD_LINE.findall(m.group(1))] if m else []


def build_metadata(paper_yaml: str, title: str, abstract: str, version: str,
                   repository: str = REPOSITORY) -> dict:
    """The Zenodo metadata block for one preprint, from the paper's own sources."""
    description = (f"<p>{abstract}</p>"
                   f"<p>Preprint. The manuscript, its bibliography and the code that "
                   f"builds it are at <a href=\"{repository}\">{repository}</a>.</p>")
    return {
        "upload_type": UPLOAD_TYPE,
        "publication_type": PUBLICATION_TYPE,
        "title": title,
        "creators": authors(paper_yaml),
        "description": description,
        "keywords": keywords(paper_yaml),
        "license": LICENSE,
        "access_right": "open",
        "version": version,
        "related_identifiers": [
            {"identifier": repository, "relation": "isSupplementedBy",
             "resource_type": "software"},
        ],
        "prereserve_doi": True,
    }


def manuscript_of(paper_dir: Path) -> Path:
    """The paper's Markdown: the one .md in the folder that is not its README."""
    candidates = [p for p in paper_dir.glob("*.md") if p.name != "README.md"]
    if len(candidates) != 1:
        sys.exit(f"expected one manuscript in {paper_dir}, found {len(candidates)}")
    return candidates[0]


def last_changed(path: Path) -> str:
    """The manuscript's last commit date, the same stamp the builds carry."""
    out = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unversioned"


def request(method: str, url: str, token: str, body: bytes | None = None,
            content_type: str = "application/json") -> dict:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        sys.exit(f"{method} {url} -> {err.code}: {err.read().decode(errors='replace')[:600]}")


def deposit(host: str, token: str, pdf: Path, metadata: dict, publish: bool) -> dict:
    """Create the draft, upload the PDF, set the metadata; publish only if asked."""
    api = f"{host}/api/deposit/depositions"
    draft = request("POST", api, token, b"{}")
    bucket = draft["links"]["bucket"]
    request("PUT", f"{bucket}/{pdf.name}", token, pdf.read_bytes(),
            content_type="application/octet-stream")
    draft = request("PUT", f"{api}/{draft['id']}", token,
                    json.dumps({"metadata": metadata}).encode())
    if publish:
        draft = request("POST", f"{api}/{draft['id']}/actions/publish", token, b"")
    return draft


def main() -> None:
    ap = argparse.ArgumentParser(description="deposit a paper's preprint on Zenodo")
    ap.add_argument("paper_dir", type=Path, help="e.g. papers/01-evidence-audit")
    ap.add_argument("--pdf", type=Path, help="the PDF to deposit (default: the -preprint build)")
    ap.add_argument("--version", help="record version (default: the manuscript's last commit date)")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--dry-run", action="store_true", help="print the metadata and stop")
    ap.add_argument("--publish", action="store_true",
                    help="publish the record; without it the deposit stays a private draft")
    a = ap.parse_args()

    manuscript = manuscript_of(a.paper_dir)
    paper_yaml = (a.paper_dir / "metadata" / "paper.yaml").read_text()
    text = manuscript.read_text()
    metadata = build_metadata(paper_yaml, title_mod.read(text), abstract_mod.read(text),
                              a.version or last_changed(manuscript))
    pdf = a.pdf or manuscript.with_name(manuscript.stem + "-preprint.pdf")
    if a.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print(f"\nwould upload {pdf} ({'exists' if pdf.exists() else 'MISSING'})")
        return
    if not pdf.exists():
        sys.exit(f"no PDF at {pdf}; build it with TMLR_MODE=preprint bash publishing/publish.sh")

    instance = "sandbox" if a.sandbox else "production"
    token = os.environ.get(TOKEN_VARS[instance], "")
    if not token:
        sys.exit(f"set {TOKEN_VARS[instance]} to a Zenodo personal access token (scope: deposit:write)")
    record = deposit(HOSTS[instance], token, pdf, metadata, a.publish)
    doi = record.get("doi") or record.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    state = "published" if a.publish else "draft (not yet public)"
    print(f"{state}: {record['links'].get('html', record['links'].get('latest_draft', ''))}")
    print(f"DOI: {doi}")


if __name__ == "__main__":
    main()

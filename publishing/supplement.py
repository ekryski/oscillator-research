#!/usr/bin/env python3
"""Build a paper's anonymized supplementary material: its code, run record and figures, as one zip.

    python3 publishing/supplement.py 02     # papers/02-*/<manuscript stem>-supplement.zip

A double-blind venue takes the code and data as a zip that must not identify the
author. The zip holds the committed files (git ls-files, so no corpus, cache or
lock file) of the paths a paper lists below, under one neutral top folder, with
the paper's metadata/supplement-README.md as its README. Because the repository
is public, anything that would lead back to it is taken out on the way:

    the run record's git commits      dropped from every run's `env`
    files that name the repository    left out (the pod script clones it)
    passages that need the repository cut from the README that carries them

and a last scan of every file refuses to write the zip if an identifying string
survives. The zip is a build output, not committed.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = "supplement"

#: per paper: what goes in, what stays out, and the passages cut from files that go in
PAPERS = {
    "02": {
        "include": ("src", "results", "resources/figures"),
        "exclude": ("src/scripts/pod_run.sh",),
        "cut": {
            "src/README.md": (
                re.compile(r" `scripts/pod_run\.sh [^\n]*? from the pushed branch\."),
                re.compile(r" Authored SVGs render to PDF and PNG from the repository root:\n\n```bash\n.*?```\n", re.S),
            ),
        },
    },
}

#: strings that would identify the author or the repository; any hit stops the build
IDENTIFYING = re.compile(r"(?i)kryski|\beric\b|/users/|oscillator-research|\bek/|github\.com/(?!soerenab/)|waffuru")


def paper_dir(number: str) -> Path:
    matches = sorted(REPO.glob(f"papers/{number}-*"))
    if len(matches) != 1:
        sys.exit(f"no single paper folder for {number}: {matches}")
    return matches[0]


def manuscript(paper: Path) -> Path:
    drafts = sorted(paper.glob("*-DRAFT.md"))
    if len(drafts) != 1:
        sys.exit(f"expected one *-DRAFT.md in {paper}, found {drafts}")
    return drafts[0]


def tracked(paper: Path, paths: tuple[str, ...]) -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=paper, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def scrub_record(raw: bytes) -> bytes:
    """A run-record file without its git commits, in the harness's own layout (indent 1, trailing newline)."""
    data = json.loads(raw)
    if "runs" not in data:
        return raw
    for run in data["runs"].values():
        run.get("env", {}).pop("commit", None)
    return (json.dumps(data, indent=1) + "\n").encode()


def scannable(rel: str, raw: bytes) -> str:
    """The text an identifying string could hide in: a record file without its per-clip bits, which are
    base64 and so match any short string by chance."""
    if rel.startswith("results/") and rel.endswith(".json"):
        data = json.loads(raw)
        for run in data.get("runs", {}).values():
            for cell in run.get("cells", []):
                cell.pop("correct", None)
        return json.dumps(data)
    return raw.decode(errors="ignore")


def build(number: str, out: Path | None = None) -> Path:
    paper, conf = paper_dir(number), PAPERS[number]
    files = [f for f in tracked(paper, conf["include"]) if f not in conf["exclude"]]
    contents: dict[str, bytes] = {}
    for rel in files:
        raw = (paper / rel).read_bytes()
        if rel.startswith("results/") and rel.endswith(".json"):
            raw = scrub_record(raw)
        for pattern in conf["cut"].get(rel, ()):
            text, n = pattern.subn("", raw.decode())
            if n != 1:
                sys.exit(f"{rel}: expected one passage matching {pattern.pattern[:60]!r}, found {n}")
            raw = text.encode()
        contents[rel] = raw
    contents["README.md"] = (paper / "metadata" / "supplement-README.md").read_bytes()

    kept = [rel for rel, raw in contents.items() if rel.startswith("results/") and b'"commit"' in raw]
    if kept:
        sys.exit(f"git commits would ship in: {', '.join(kept)}")
    hits = [(rel, m.group(0)) for rel, raw in contents.items() if not rel.endswith((".png", ".pdf"))
            for m in IDENTIFYING.finditer(scannable(rel, raw))]
    if hits:
        shown = "\n".join(f"  {rel}: {hit!r}" for rel, hit in hits[:20])
        sys.exit(f"identifying strings would ship ({len(hits)}):\n{shown}")

    out = out or paper / f"{manuscript(paper).stem.removesuffix('-DRAFT')}-supplement.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in sorted(contents):
            info = zipfile.ZipInfo(f"{TOP}/{rel}", date_time=(2026, 1, 1, 0, 0, 0))  # no build time in the zip
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, contents[rel])
    out.write_bytes(buf.getvalue())
    print(f"wrote {out.relative_to(REPO)}: {len(contents)} files, {out.stat().st_size / 1e6:.1f} MB")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paper", choices=sorted(PAPERS), help="the paper's number")
    ap.add_argument("--out", type=Path, help="where to write the zip (default: beside the manuscript)")
    a = ap.parse_args(argv)
    build(a.paper, a.out)


if __name__ == "__main__":
    main()

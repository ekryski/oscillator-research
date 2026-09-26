#!/usr/bin/env python3
"""Build a paper's anonymized supplementary material: its code, run record and audio examples, as one zip.

    python3 publishing/supplement.py 02     # papers/02-*/<manuscript stem>-supplement.zip
    bash publishing/publish.sh 02 --iclr    # the same, as the build's `supplement` format

A double-blind venue takes the code and data as a zip that must not identify the
author. For a paper with a metadata/supplement-README.md, the zip holds the
committed files (git ls-files, so no corpus, cache or lock file) of

    src/                        the experiment code and its tests
    results/                    the run record, its summary and its README
    resources/audio/examples/   as audio/: clips as the arms hear them

under one neutral top folder, with the supplement README as its README. The
paper's own scripts (its schematics, its GPU-pod runner) live outside src/ and
stay out. Every file is cleaned on the way (sanitize.py: no invisible
character, look-alike letter or odd space in the text, no metadata chunk in
the sound), and because the repository is public the run record's git commits
are dropped, and a last scan of every file refuses to write the zip if a
string that names the author, the repository or the paper's folder survives. The
zip is a build output, not committed.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from sanitize import neutral  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TOP = "supplement"
README = Path("metadata/supplement-README.md")
#: (path in the paper's folder, path in the zip)
CONTENTS = (("src", "src"), ("results", "results"), ("resources/audio/examples", "audio"))
#: strings that would identify the author or the repository; any hit stops the build
IDENTIFYING = r"(?i)kryski|\beric\b|/users/|oscillator-research|\bek/|github\.com/(?!soerenab/)"


def paper_dir(paper: str) -> Path:
    number = paper.split("-")[0]
    matches = sorted(REPO.glob(f"papers/{number}-*"))
    if len(matches) != 1:
        sys.exit(f"no single paper folder for {paper}: {matches}")
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


def build(paper_arg: str, out: Path | None = None) -> Path:
    paper = paper_dir(paper_arg)
    if not (paper / README).exists():
        sys.exit(f"{paper.name} has no {README}: nothing says what its supplement is")
    contents: dict[str, bytes] = {}
    for source, target in CONTENTS:
        for rel in tracked(paper, (source,)):
            raw = (paper / rel).read_bytes()
            if rel.startswith("results/") and rel.endswith(".json"):
                raw = scrub_record(raw)
            contents[target + rel[len(source):]] = raw
    contents["README.md"] = (paper / README).read_bytes()
    stripped: Counter = Counter()
    for rel, raw in contents.items():
        contents[rel], found = neutral(rel, raw)
        stripped += found
    if stripped:
        print("    stripped " + ", ".join(f"{n} {kind}" for kind, n in sorted(stripped.items())))

    kept = [rel for rel, raw in contents.items() if rel.startswith("results/") and b'"commit"' in raw]
    if kept:
        sys.exit(f"git commits would ship in: {', '.join(kept)}")
    identifying = re.compile(f"{IDENTIFYING}|{re.escape(paper.name)}")
    hits = [(rel, m.group(0)) for rel, raw in contents.items() if not rel.endswith((".png", ".pdf", ".wav"))
            for m in identifying.finditer(scannable(rel, raw))]
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
    ap.add_argument("paper", help="the paper's number or folder name, e.g. 02")
    ap.add_argument("--out", type=Path, help="where to write the zip (default: beside the manuscript)")
    a = ap.parse_args(argv)
    build(a.paper, a.out)


if __name__ == "__main__":
    main()

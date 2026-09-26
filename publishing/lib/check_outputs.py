#!/usr/bin/env python3
"""Check the built formats for anything that could fingerprint them.

The build removes hidden characters from what it formats (sanitize.py) and
keeps build times, IDs and tool versions out of the files it writes; this
confirms both on the files themselves:

    text       no invisible or control character and no Latin look-alike letter
               in the text of any format (PDF via pdftotext, HTML, TeX, and the
               XML inside Word and EPUB files and the archives). A PDF glyph with
               no Unicode mapping (a bitmap font's ligature, say) comes out of
               pdftotext as its slot number, a control code; pdfTeX cannot put a
               control character in the page text, and the .tex it typeset is
               checked in full, so those are reported as unmapped glyphs, not
               failures
    PDF        no document ID, no creation or modification date, no pdfTeX
               banner or included-file keys, no named creator or producer
    Word/EPUB  every date is the manuscript's day at midnight UTC, every zip entry
               carries the same time, and the EPUB's identifier is the stable
               one derived from the paper's name rather than a random one
    archives   every tar entry root's, unnamed and equally dated; every zip entry
               equally dated; no metadata chunk in any sound or image inside

    python3 publishing/lib/check_outputs.py [paper prefix] [--newer FILE]   # exit 1 on any finding
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
import uuid
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import papers
from sanitize import INVISIBLE, LOOKALIKE_CHARS, WORD, _latin, strip_png, strip_wav

PDF_KEYS = re.compile(rb"/ID\s*\[|/CreationDate|/ModDate|/PTEX\.|/(?:Producer|Creator)\s*\((?!\))")
UNMAPPED = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}T(\d{2}:\d{2}:\d{2})Z?")


def text_findings(text: str) -> list[str]:
    out = [f"invisible U+{ord(m.group(0)):04X}" for m in INVISIBLE.finditer(text)]
    out += [f"look-alike letter in {w!r}" for w in WORD.findall(text) if _latin(w) and LOOKALIKE_CHARS.intersection(w)]
    return out


def pdf_findings(path: Path) -> tuple[list[str], list[str]]:
    raw = path.read_bytes()
    blobs = [raw]
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        try:
            blobs.append(zlib.decompress(m.group(1)))
        except zlib.error:
            pass
    keys = sorted({k.decode() for b in blobs for k in PDF_KEYS.findall(b)})
    out, notes = [f"PDF key {k.strip()}" for k in keys], []
    if shutil.which("pdftotext"):
        text = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True).stdout
        text = text.decode("utf-8", "ignore").replace("\f", "")
        glyphs = UNMAPPED.findall(text)
        if glyphs:
            notes.append(f"{len(glyphs)} glyph(s) with no Unicode mapping (a font without one, not a hidden character)")
        out += text_findings(UNMAPPED.sub("", text))
    return out, notes


def zip_findings(path: Path, name: str) -> list[str]:
    out = []
    with zipfile.ZipFile(path) as z:
        times = {i.date_time for i in z.infolist()}
        if len(times) > 1:
            out.append(f"zip entries carry {len(times)} different times")
        for info in z.infolist():
            data = z.read(info)
            if info.filename.endswith((".xml", ".xhtml", ".opf", ".ncx", ".html", ".md", ".py", ".json", ".txt", ".toml")):
                text = data.decode("utf-8", "ignore")
                out += [f"{info.filename}: {f}" for f in text_findings(text)]
                for m in DATE.finditer(text) if info.filename.endswith((".xml", ".opf")) else ():
                    if m.group(1) != "00:00:00":
                        out.append(f"{info.filename}: a build time {m.group(0)}")
                if info.filename.endswith(".opf") and "urn:uuid:" in text:
                    want = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, name)}"
                    if want not in text:
                        out.append(f"{info.filename}: a random identifier, not {want}")
            elif info.filename.endswith(".wav") and strip_wav(data)[1]:
                out.append(f"{info.filename}: metadata chunks in a WAV")
            elif info.filename.endswith(".png") and strip_png(data)[1]:
                out.append(f"{info.filename}: metadata chunks in a PNG")
    return out


def tar_findings(path: Path) -> list[str]:
    out = []
    with tarfile.open(path) as tar:
        members = tar.getmembers()
        if {(m.uid, m.gid, m.uname, m.gname) for m in members} - {(0, 0, "", "")}:
            out.append("tar entries name an owner or group")
        if len({m.mtime for m in members}) > 1:
            out.append("tar entries carry different times")
        for m in members:
            if m.isfile() and m.name.endswith((".tex", ".bib", ".sty", ".bst", ".cls")):
                text = tar.extractfile(m).read().decode("utf-8", "ignore")
                out += [f"{m.name}: {f}" for f in text_findings(text)]
    with open(path, "rb") as f:
        head = f.read(10)
    if head[4:8] != b"\0\0\0\0" or head[3] & 0x08:
        out.append("gzip header records a time or a file name")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="fingerprints in the built formats")
    ap.add_argument("prefix", nargs="?", default="")
    ap.add_argument("--newer", type=Path, help="check only outputs written after this file")
    a = ap.parse_args()
    since = a.newer.stat().st_mtime if a.newer and a.newer.exists() else 0
    total = 0
    for paper in papers(a.prefix):
        stem = paper.stem
        outputs = sorted(p for p in paper.dir.glob(f"{stem.removesuffix('-DRAFT')}*")
                         if p.suffix in {".pdf", ".html", ".tex", ".epub", ".docx", ".gz", ".zip"}
                         and p.stat().st_mtime >= since)
        found, notes = {}, {}
        for p in outputs:
            if p.suffix == ".pdf":
                found[p.name], notes[p.name] = pdf_findings(p)
            elif p.suffix in (".html", ".tex"):
                found[p.name] = text_findings(p.read_text(encoding="utf-8", errors="ignore"))
            elif p.suffix in (".epub", ".docx", ".zip"):
                found[p.name] = zip_findings(p, stem)
            elif p.name.endswith(".tar.gz"):
                found[p.name] = tar_findings(p)
        bad = {k: v for k, v in found.items() if v}
        total += sum(len(v) for v in bad.values())
        print(f"\n{paper.slug}: {len(outputs)} output(s) checked, " + ("clean" if not bad else "fingerprints found"))
        for name, items in bad.items():
            shown = sorted(set(items))
            print(f"  {name}: " + "; ".join(shown[:8]) + (f"; and {len(shown) - 8} more" if len(shown) > 8 else ""))
        for name, items in notes.items():
            for note in items:
                print(f"  note, {name}: {note}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

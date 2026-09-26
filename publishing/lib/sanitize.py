#!/usr/bin/env python3
"""Characters that can carry a mark no reader sees, and how the build removes them.

Text can be fingerprinted without changing a word a reader would notice: a run
of Unicode tag characters spells out an identifier (the usual carrier when
generated prose is watermarked), zero-width and directional marks hide bits
between letters, a Cyrillic "а" passes for a Latin "a", and a no-break or thin
space passes for an ordinary one. Control characters and trailing whitespace
carry bits the same way. None of them has a use in a manuscript, a
bibliography or the front matter, so the build removes them from every copy it
formats, and `check_hidden.py` reports them in the sources.

    invisible     deleted: zero-width and joiner characters, directional marks and
                  isolates, variation selectors, the tag block, soft hyphens,
                  Hangul and Braille fillers, invisible operators, interlinear
                  annotation, private-use characters, ASCII and C1 control
                  characters other than tab, newline and carriage return
    spaces        replaced by an ordinary space: no-break, narrow, thin, hair,
                  en, em, ideographic and other width-bearing spaces, and the
                  Unicode line and paragraph separators
    look-alikes   a Cyrillic or Greek letter that looks like a Latin one, inside a
                  word written in Latin letters, becomes that Latin letter (a
                  whole Greek word, such as θ or λ, is left alone); fullwidth
                  ASCII becomes ASCII
    trailing      spaces and tabs at the end of a line are dropped; in Markdown a
                  hard line break (two or more trailing spaces) becomes the
                  backslash form, so it survives without the spaces

Statistical watermarks, which shift word choice rather than characters, cannot
be removed this way; nothing short of rewriting the prose removes them.

    python3 publishing/lib/sanitize.py IN --out OUT [--markdown]   # write a cleaned copy
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

INVISIBLE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"                 # ASCII and C1 controls, not tab, LF, CR
    "­͏؜ᅟᅠ឴឵"            # soft hyphen, grapheme joiner, Arabic mark, fillers
    "᠋-᠏​-‏‪-‮"               # Mongolian selectors, zero-width, directional
    "⁠-⁯ㅤ︀-️﻿ﾠ"          # word joiner, invisible operators, isolates, VS, BOM
    "￰-￼-"                            # specials, interlinear annotation, private use
    "\U0001d173-\U0001d17a\U000e0000-\U000e007f"            # musical formatting, tag block
    "\U000e0100-\U000e01ef\U000f0000-\U0010ffff]"           # variation selectors supplement, private use
)
SPACES = re.compile("[   -     ⠀　]")
TRAILING = re.compile(r"[ \t]+(?=\r?\n|\Z)")
HARD_BREAK = re.compile(r"(?<=\S) {2,}(?=\r?\n)")

#: letters that render as a Latin letter, by the Latin letter they pass for
LOOKALIKES = str.maketrans({
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "ԁ": "d", "ԛ": "q", "ԝ": "w", "һ": "h", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S", "Ԛ": "Q", "Ԝ": "W",
    # Greek
    "ο": "o", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})
LOOKALIKE_CHARS = frozenset(chr(c) for c in LOOKALIKES)
FULLWIDTH = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}
WORD = re.compile(r"[^\W\d_]+")


def _latin(word: str) -> bool:
    return any("a" <= ch.lower() <= "z" for ch in word)


def clean(text: str, *, markdown: bool = False) -> tuple[str, Counter]:
    """The text with every carrier above removed, and how many of each kind there were."""
    found: Counter = Counter()
    text, found["invisible"] = INVISIBLE.subn("", text)
    text, found["spaces"] = SPACES.subn(" ", text)
    n = sum(1 for ch in text if ord(ch) in FULLWIDTH)
    if n:
        text, found["look-alikes"] = text.translate(FULLWIDTH), n

    def unmask(m: re.Match) -> str:
        word = m.group(0)
        if not (_latin(word) and LOOKALIKE_CHARS.intersection(word)):
            return word
        found["look-alikes"] += sum(ch in LOOKALIKE_CHARS for ch in word)
        return word.translate(LOOKALIKES)

    text = WORD.sub(unmask, text)
    if markdown:
        text, breaks = HARD_BREAK.subn("\\\\", text)
        found["trailing"] += breaks
    text, n = TRAILING.subn("", text)
    found["trailing"] += n
    return text, +found


#: the chunks a sound or an image needs; everything else (a WAV's PEAK chunk, which dates the file, or
#: LIST/INFO; a PNG's text, time and Exif chunks) is metadata and is dropped
WAV_KEEP = {b"fmt ", b"fact", b"data"}
PNG_DROP = {b"tEXt", b"zTXt", b"iTXt", b"tIME", b"eXIf"}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".json", ".txt", ".lock", ".yaml", ".yml", ".cfg", ".ini", ".sh",
                 ".csv", ".tex", ".bib", ".sty", ".bst", ".cls", ".html", ".xml", ".css"}


def strip_wav(raw: bytes) -> tuple[bytes, int]:
    """A RIFF WAV with only its format, fact and sample chunks, and how many others were dropped."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return raw, 0
    body, dropped, i = b"", 0, 12
    while i + 8 <= len(raw):
        cid, size = raw[i:i + 4], int.from_bytes(raw[i + 4:i + 8], "little")
        chunk = raw[i:i + 8 + size + (size & 1)]
        if cid in WAV_KEEP:
            body += chunk
        else:
            dropped += 1
        i += 8 + size + (size & 1)
    return b"RIFF" + (4 + len(body)).to_bytes(4, "little") + b"WAVE" + body, dropped


def strip_png(raw: bytes) -> tuple[bytes, int]:
    """A PNG without its text, time and Exif chunks, and how many were dropped."""
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return raw, 0
    out, dropped, i = raw[:8], 0, 8
    while i + 8 <= len(raw):
        size = int.from_bytes(raw[i:i + 4], "big")
        chunk = raw[i:i + 12 + size]
        if raw[i + 4:i + 8] in PNG_DROP:
            dropped += 1
        else:
            out += chunk
        i += 12 + size
    return out, dropped


def neutral(name: str, raw: bytes) -> tuple[bytes, Counter]:
    """Any file as it should ship: text cleaned, sound and images without their metadata."""
    suffix = Path(name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text, found = clean(raw.decode("utf-8"), markdown=suffix == ".md")
        return text.encode("utf-8"), found
    if suffix == ".wav":
        raw, n = strip_wav(raw)
        return raw, Counter({"metadata chunks": n}) if n else Counter()
    if suffix == ".png":
        raw, n = strip_png(raw)
        return raw, Counter({"metadata chunks": n}) if n else Counter()
    return raw, Counter()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", action="store_true", help="keep hard line breaks, in the backslash form")
    a = ap.parse_args(argv)
    text, found = clean(a.source.read_text(encoding="utf-8"), markdown=a.markdown)
    a.out.write_text(text, encoding="utf-8")
    if found:
        print(f"    {a.source.name}: stripped " + ", ".join(f"{n} {kind}" for kind, n in sorted(found.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

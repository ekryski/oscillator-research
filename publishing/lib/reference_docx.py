#!/usr/bin/env python3
"""Write the reference .docx the Word builds take their styles from.

pandoc styles a .docx after a reference document, and its stock one colours
links in Word's light theme blue, which reads faint on screen and fainter in
print. This takes pandoc's own reference document and sets its Hyperlink style
to a darker blue, so the reference stays pandoc's but for that one change and
no binary has to live in the repository.

    python3 publishing/lib/reference_docx.py publishing/.work/reference.docx
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

#: the colour of every link in the Word outputs: a dark blue that stays legible in print
LINK_COLOR = "0645AD"
#: the character style Word applies to links, as pandoc's reference document names it
HYPERLINK_STYLE = re.compile(r'<w:style [^>]*w:styleId="Hyperlink".*?</w:style>', re.S)
#: the colour element inside it; a `w:themeColor` there would override `w:val`, so the whole element goes
COLOR_ELEMENT = re.compile(r"<w:color [^>]*/>")


def restyle(docx: bytes, color: str = LINK_COLOR) -> bytes:
    """The reference document with its Hyperlink style coloured `color`."""
    src = zipfile.ZipFile(io.BytesIO(docx))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/styles.xml":
                styles = data.decode()
                block = HYPERLINK_STYLE.search(styles)
                if block is None or COLOR_ELEMENT.search(block.group(0)) is None:
                    raise SystemExit("reference.docx: no coloured Hyperlink style to recolour")
                recoloured = COLOR_ELEMENT.sub(f'<w:color w:val="{color}"/>', block.group(0), count=1)
                data = (styles[:block.start()] + recoloured + styles[block.end():]).encode()
            dst.writestr(item, data)
    return out.getvalue()


def pandoc_default() -> bytes:
    """pandoc's own reference document, as this pandoc ships it."""
    return subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                          check=True, capture_output=True).stdout


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: reference_docx.py <output.docx>")
    Path(sys.argv[1]).write_bytes(restyle(pandoc_default()))


if __name__ == "__main__":
    main()

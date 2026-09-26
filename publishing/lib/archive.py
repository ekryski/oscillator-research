#!/usr/bin/env python3
"""Pack a folder as a .tar.gz that records nothing about the machine or the moment it was made.

`tar czf` stores each file's owner and group by name and number, its
modification time, and gzip adds the archive's own name and time: enough to tie
a bundle to one account and one build. Here every entry is owned by root with no
names, dated SOURCE_DATE_EPOCH (0 if unset), and listed in sorted order, and the
gzip header carries no name or time, so the same files always give the same bytes.

    python3 publishing/lib/archive.py <folder> <out.tar.gz>     # the folder is the archive's top entry
"""

from __future__ import annotations

import gzip
import io
import os
import sys
import tarfile
from pathlib import Path


def pack(folder: Path, out: Path) -> None:
    when = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))

    def neutral(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = when
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        paths = [folder, *sorted(folder.rglob("*"))]
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(folder.parent)), recursive=False, filter=neutral)
    with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        gz.write(buf.getvalue())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pack(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))

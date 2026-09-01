"""Read a manuscript's title out of the manuscript.

The same argument as abstract.py, applied to the line above it. A reader who
opens the Markdown on GitHub should see what the paper is called, and a
manuscript that opens straight into "## Abstract" reads like a fragment. But
the title also has to reach the title block of every built format, where
pandoc sets it from metadata, so leaving it in the body renders it twice: once
as the typeset title and once as a stray heading above the abstract.

So: one source, the `# ` heading at the top of the Markdown. `strip` removes it
from the body copy at build time, and `as_yaml` hands the same text to pandoc
as metadata. Nothing else in the pipeline sees an H1 — the manuscript's own
sections start at `##`, which is what `check_sections.py` and
`number_sections.py` match, so the title is invisible to the numbering.
"""

from __future__ import annotations

import re
from pathlib import Path

#: a single `# ` heading at the very top of the document, blank lines allowed
#: before it. Anchored to the start so a `#` further down, which the manuscripts
#: do not use, could never be mistaken for the title.
TITLE = re.compile(r"\A\s*\#[^\S\n]+(\S.*?)[^\S\n]*$\n?", re.M)


def read(manuscript: Path | str) -> str:
    """The title as written. Empty if the manuscript does not open with one."""
    text = manuscript if isinstance(manuscript, str) else manuscript.read_text()
    m = TITLE.match(text)
    return m.group(1).strip() if m else ""


def strip(text: str) -> str:
    """The manuscript without its title heading."""
    return TITLE.sub("", text, count=1).lstrip()


def as_yaml(title: str) -> str:
    """A metadata file carrying just the title.

    Double-quoted rather than bare, because every title in this repository so
    far contains a colon and a bare `a: b` is a nested mapping to YAML, not a
    string.
    """
    if not title:
        return ""
    return f'---\ntitle: "{_escape(title)}"\n---\n'


def _escape(title: str) -> str:
    return title.replace("\\", "\\\\").replace('"', '\\"')

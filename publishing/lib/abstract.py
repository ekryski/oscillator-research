"""Read a manuscript's abstract out of the manuscript.

The abstract belongs with the prose, where it is edited alongside the paper
rather than in a metadata file someone has to remember exists. But it also has
to reach the title block of every built format, and the `AB` field of the RIS
citation, so it cannot simply sit in the body: pandoc would render it once as
metadata and once as an ordinary section, which is exactly the duplication this
module exists to prevent.

So: one source, the `## Abstract` section of the Markdown. `strip` removes it
from the body copy at build time, and `as_yaml` hands the same text to pandoc as
metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

#: the abstract section, up to the next heading of the same level or higher
SECTION = re.compile(r"(?ims)^\#\#\s+abstract\s*$\n+(.*?)(?=^\#\#\s|\Z)")


def read(manuscript: Path | str) -> str:
    """The abstract as written, whitespace collapsed. Empty if there is none."""
    text = manuscript if isinstance(manuscript, str) else manuscript.read_text()
    m = SECTION.search(text)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def strip(text: str) -> str:
    """The manuscript without its abstract section."""
    return SECTION.sub("", text, count=1).lstrip()


def as_yaml(abstract: str) -> str:
    """A metadata file carrying just the abstract.

    Written as a folded block so the abstract can contain any punctuation
    without quoting, and indented two spaces because that is what makes it a
    block scalar rather than a new key.
    """
    if not abstract:
        return ""
    body = "\n".join("  " + line for line in _wrap(abstract, 96))
    return f"---\nabstract: |\n{body}\n---\n"


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]

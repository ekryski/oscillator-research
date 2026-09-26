"""Where everything lives.

Each paper is self-contained: the manuscript, the front matter that becomes its
title block, the bibliography it cites, and the built formats all sit in one
folder, so a reader who wants the PDF finds it beside the Markdown rather than
in a build directory. The scripts in publishing/ are the only shared part, and
this module is the single place that knows the layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "papers"
#: preprocessed manuscripts and build logs; regenerated, never committed
WORK = ROOT / "publishing" / ".work"
#: the vendored TMLR style file and the pandoc templates that use it
TEMPLATES = ROOT / "publishing" / "templates"


@dataclass(frozen=True)
class Paper:
    """One paper's folder and the files the publishing scripts need from it."""

    slug: str
    dir: Path

    @property
    def manuscript(self) -> Path | None:
        """The Markdown source. Named for the paper, so the built formats that
        take their name from it are self-describing once downloaded."""
        found = sorted(self.dir.glob("*-DRAFT.md")) or sorted(self.dir.glob("*.md"))
        return next((p for p in found if p.name != "README.md"), None)

    @property
    def stem(self) -> str:
        m = self.manuscript
        return m.stem if m else self.slug

    @property
    def metadata(self) -> Path:
        return self.dir / "metadata" / "paper.yaml"

    @property
    def bib(self) -> Path:
        return self.dir / "references" / "bibliography.bib"

    @property
    def citemap(self) -> Path:
        return self.dir / "references" / "citemap.json"

    @property
    def citations(self) -> Path:
        """Formats for citing this paper, as opposed to the works it cites."""
        return self.dir / "metadata"


def papers(prefix: str = "") -> list[Paper]:
    """Every paper folder, or those whose name starts with `prefix`."""
    return [Paper(d.name, d) for d in sorted(PAPERS.glob("*/"))
            if d.is_dir() and d.name.startswith(prefix)]


def paper_for(path: Path) -> Paper | None:
    """The paper a file belongs to, for scripts handed a manuscript path."""
    path = path.resolve()
    for p in papers():
        if p.dir in path.parents:
            return p
    return None

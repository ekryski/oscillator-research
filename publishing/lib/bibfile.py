"""Reading BibTeX, without assuming a field fits on one line.

The obvious regex — `^\\s*(\\w+)\\s*=\\s*\\{(.*?)\\},?$` per line — works right up
until the bibliography acquires an entry from somewhere that wraps its columns.
DBLP wraps every author list and most titles:

    author       = {T. Konstantin Rusch and
                    Siddhartha Mishra},

A line-anchored pattern sees no author there, so a completeness check reports a
perfectly good entry as missing one, and a converter helpfully adds a fallback
label the entry does not need. Both failures point the wrong way — they make a
correct bibliography look broken, which is the kind of noise that gets ignored
right before it matters.

So: find `name = {`, then balance braces to the close. Values keep their internal
line breaks; callers that want a single line collapse the whitespace themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: the start of one entry: `@type{key,`
ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)
#: the start of one field: `name = {`  (quoted and bare values are handled too)
FIELD = re.compile(r"(\w+)\s*=\s*", re.M)


def _value_at(text: str, i: int) -> tuple[str, int]:
    """The field value starting at `i`, and where it ends.

    Braces nest — `title = {The {ESN} Approach}` is one value, not two — so the
    only correct way through is to count them.
    """
    if i < len(text) and text[i] == "{":
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j], j + 1
            j += 1
        return text[i + 1:], len(text)          # unbalanced; take what there is
    if i < len(text) and text[i] == '"':
        j = text.find('"', i + 1)
        j = len(text) if j < 0 else j
        return text[i + 1:j], j + 1
    m = re.compile(r"[^,}\n]*").match(text, i)  # a bare number or macro
    return m.group(0).strip(), m.end()


def fields_of(block: str) -> dict[str, str]:
    """Every field in one entry, whitespace in the values collapsed."""
    out: dict[str, str] = {}
    body = block[block.index(",") + 1:] if "," in block else ""
    i = 0
    while (m := FIELD.search(body, i)) is not None:
        # a `%` comment can carry something that looks like a field; skip it
        line_start = body.rfind("\n", 0, m.start()) + 1
        if body[line_start:m.start()].lstrip().startswith("%"):
            i = body.find("\n", m.start()) + 1 or len(body)
            continue
        value, i = _value_at(body, m.end())
        out.setdefault(m.group(1).lower(), re.sub(r"\s+", " ", value).strip())
    return out


@dataclass(frozen=True)
class Entry:
    kind: str
    key: str
    fields: dict[str, str]
    raw: str

    @property
    def labels(self) -> str:
        """How the manuscript spells this citation, recorded as a comment.

        Two spellings are in the files: the current one, and the shorter form an
        earlier version of the extractor wrote. Matching only the current one
        loses the provenance of every entry written before it.
        """
        m = re.search(r"% cited (?:in the manuscript )?as: (.*)", self.raw)
        return m.group(1).strip() if m else ""


#: the provenance comment the generators write: how a manuscript spells this
#: citation. Two forms are in the files — the current one and a shorter form an
#: earlier extractor wrote — and both mean the same thing.
PROVENANCE = re.compile(r"^[ \t]*% cited (?:in the manuscript )?as: ", re.M)
#: a run of comment lines closing a block, with any blank lines after it. The
#: generators write the comment, a blank line, then the entry, so those blank
#: lines are part of the separator rather than the end of the run.
TRAILING_COMMENT = re.compile(r"(?:^[ \t]*%.*\n)+\s*\Z", re.M)
#: leading comments and blank lines, skipped to find where the entry starts
LEADING_COMMENT = re.compile(r"\A\s*(?:[ \t]*%.*\n\s*)*")


def entries(text: str) -> list[Entry]:
    """Every entry in a .bib, in file order, each with its raw block.

    Splitting on `@` alone puts a comment that INTRODUCES an entry at the tail
    of the previous one — and these files use exactly that placement, because a
    `%` inside an entry is a BibTeX syntax error. Left uncorrected, every
    entry's recorded provenance is its neighbour's, and a rewrite that preserves
    the raw block drags the comment one entry further away on every pass.

    A trailing comment run is therefore carried onto the next block. What marks
    it as belonging there is its content, not its position: a `% cited as:` line
    is provenance and travels, while the file's own header comments sit in the
    same place and do not.
    """
    out: list[Entry] = []
    carried = ""
    for block in re.split(r"(?=^@)", text, flags=re.M):
        block, carried = carried + block, ""
        start = LEADING_COMMENT.match(block).end()
        # searched from 0, not from `start`: a comment-only block is entirely
        # "leading", and skipping past it would leave nothing to carry. A
        # comment that leads an entry cannot match here anyway — the pattern is
        # anchored to the end of the block.
        tail = TRAILING_COMMENT.search(block)
        if tail and PROVENANCE.search(tail.group(0)):
            carried = tail.group(0)
            block = block[:tail.start()]
        if m := ENTRY.match(block, start):
            out.append(Entry(m.group(1).lower(), m.group(2),
                             fields_of(block[start:]), block))
    return out

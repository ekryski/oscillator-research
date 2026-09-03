#!/usr/bin/env python3
"""The parts of metadata/paper.yaml the build scripts read, without PyYAML.

paper.yaml is hand-written and flat: top-level scalars, top-level lists of
strings, and the `author:` list of maps. That is all these readers handle,
which is why the build has no dependency the system python3 lacks. Quoted
values lose their quotes.
"""

import re

#: a `- name:` entry under `author:`, with the indented `key: value` lines under it
AUTHOR_BLOCK = re.compile(r"^\s*-\s*name:\s*(?P<name>.+?)\s*$(?P<rest>(?:\n\s{4,}[\w-]+:.*$)*)", re.M)
FIELD = re.compile(r"^\s+([\w-]+):\s*(.+?)\s*$", re.M)
ITEM = re.compile(r"^\s*-\s*(.+?)\s*$", re.M)


def unquote(value: str) -> str:
    return value.strip().strip('"').strip("'")


def authors(paper_yaml: str) -> list[dict[str, str]]:
    """Each author as a dict of its fields, `name` included, in front-matter order."""
    out = []
    for m in AUTHOR_BLOCK.finditer(paper_yaml):
        fields = {k: unquote(v) for k, v in FIELD.findall(m.group("rest"))}
        fields["name"] = unquote(m.group("name"))
        out.append(fields)
    return out


def scalar(paper_yaml: str, key: str) -> str:
    """A top-level `key: value` line; empty when absent or valueless."""
    m = re.search(rf"^{re.escape(key)}:[ \t]*(.*?)[ \t]*$", paper_yaml, re.M)
    return unquote(m.group(1)) if m else ""


def items(paper_yaml: str, key: str) -> list[str]:
    """A top-level `key:` followed by `- item` lines; empty when absent."""
    m = re.search(rf"^{re.escape(key)}:[ \t]*\n((?:[ \t]*-[ \t]*.+\n?)+)", paper_yaml, re.M)
    return [unquote(i) for i in ITEM.findall(m.group(1))] if m else []

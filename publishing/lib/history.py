#!/usr/bin/env python3
"""Keep every build of a face, and mark what changed since an earlier one.

A revised manuscript is read against the version the reader already has, so
each build of a face is kept under publishing/.history/, named for the day and
the commit it was built from, and the build then writes a marked-up PDF beside
the manuscript (additions underlined in blue, deletions struck through in red,
through latexdiff) against the previous build or a named reference. Nothing
here touches the shipped outputs: they keep their stable names and stay
byte-identical across rebuilds, and the stamp never appears inside one.

    <paper>  the paper's folder name, 02-untrained-reservoirs
    <face>   the suffix the build writes, iclr / preprint / iclr-preprint / tmlr-accepted

    .history/<paper>/<face>/<YYYY-MM-DD>-<hash>[-dirty].tex|.pdf    one build; -dirty when the paper's
                                                                   sources or publishing/ had uncommitted changes
    .history/<paper>/<face>/marks/<name>.tex|.pdf                  a build kept under a name (--mark)
    <entry>.commit                                                 the commit each was built from

The reference a diff is drawn against, in order:

    --diff-against X   a mark of that name; else a .tex file at that path; else a git
                       ref, taking the committed <face>.tex at that commit when there is
                       one and building the commit in a temporary worktree otherwise
    otherwise          the most recent earlier entry of the face, dated or marked
    otherwise          the last commit on main that touched the paper, as for a ref

    python3 publishing/lib/history.py stamp <paper>                          2026-11-07-4b91ebd[-dirty]
    python3 publishing/lib/history.py record <paper> <face> <tex> <pdf> [--mark NAME]
    python3 publishing/lib/history.py mark <paper> <face> <name> --at REF --venue V --form F
    python3 publishing/lib/history.py reference <paper> <face> --venue V --form F --into DIR [--against X] [--no-build]
                                                                             label, .tex, commit, kind (tab separated)
    python3 publishing/lib/history.py diff <reference.tex> <new.tex> <out.tex> exit 2 when latexdiff is not installed
    python3 publishing/lib/history.py figures <commit> <paper>               figure files that differ from the working tree
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, Paper

#: what a build writes beside the manuscript; a change to these is the build's own
#: doing, not an edit, so it does not make a tree dirty
OUTPUT_SUFFIXES = {".pdf", ".tex", ".html", ".epub", ".docx", ".gz", ".zip"}
MARK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: a diff PDF's name: <manuscript>-<face>-diff-<reference label>-to-<stamp>.pdf
DIFF_OUTPUT = re.compile(r"-diff-.+-to-\d{4}-\d{2}-\d{2}-[0-9a-f]{4,40}(-dirty)?\.pdf$")
#: how latexdiff is run: underline-and-colour markup, figures left to the checksum
#: report because latexdiff can only see a figure's file name, not its content
LATEXDIFF_ARGS = ["--encoding=utf8", "--graphics-markup=none"]


@dataclass(frozen=True)
class Entry:
    """One reference version of a face: a kept build, a mark, a file or a commit."""

    label: str
    tex: Path | None
    pdf: Path | None
    commit: str  # the full hash it was built from, "" when unknown
    kind: str  # build | mark | path | ref | main
    dirty: bool = False


def git(*args: str, cwd: Path, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def paper_in(root: Path, slug: str) -> Paper:
    return Paper(slug, root / "papers" / slug)


def face_dir(root: Path, slug: str, face: str) -> Path:
    return root / "publishing" / ".history" / slug / face


# --- the stamp ---------------------------------------------------------------

def source_changes(root: Path, paper: Paper) -> list[str]:
    """Uncommitted changes that would alter a build: anything under the paper's
    folder other than its built formats, and anything under publishing/."""
    rel = paper.dir.relative_to(root)
    # not through git(): a line's first column may be a space, which strip would take
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", str(rel), "publishing"],
                            cwd=root, capture_output=True, text=True, check=True).stdout
    stem = paper.stem.removesuffix("-DRAFT")
    changed = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1].strip('"')
        p = Path(path)
        if p.parent == rel and p.name.startswith(stem) and p.suffix in OUTPUT_SUFFIXES:
            continue
        changed.append(path)
    return changed


def stamp(root: Path, paper: Paper, today: dt.date | None = None) -> str:
    """<YYYY-MM-DD>-<short hash>[-dirty]: the day, HEAD, and whether the tree
    held uncommitted changes to the paper or to publishing/."""
    day = (today or dt.date.today()).isoformat()
    short = git("rev-parse", "--short", "HEAD", cwd=root)
    dirty = "-dirty" if source_changes(root, paper) else ""
    return f"{day}-{short}{dirty}"


def diff_name(stem: str, face: str, label: str, build_stamp: str) -> str:
    return f"{stem}-{face}-diff-{label}-to-{build_stamp}.pdf"


def is_diff_output(name: str) -> bool:
    """The diff PDF is a working aid, never shipped: the output check skips it."""
    return DIFF_OUTPUT.search(name) is not None


# --- the history -------------------------------------------------------------

def record(root: Path, paper: Paper, face: str, tex: Path, pdf: Path, label: str, commit: str,
           *, mark: bool = False, dirty: bool = False) -> Entry:
    """Copy a build into the history, dated or under a mark's name."""
    where = face_dir(root, paper.slug, face)
    if mark:
        if not MARK_NAME.match(label):
            raise ValueError(f"a mark is named with letters, digits, dots, dashes and underscores, not {label!r}")
        where = where / "marks"
    where.mkdir(parents=True, exist_ok=True)
    kept_tex, kept_pdf = where / f"{label}.tex", where / f"{label}.pdf"
    # copyfile, not copy2: the entry's own time is what orders the history
    shutil.copyfile(tex, kept_tex)
    shutil.copyfile(pdf, kept_pdf)
    (where / f"{label}.commit").write_text(f"{commit}{' dirty' if dirty else ''}\n")
    return Entry(label, kept_tex, kept_pdf, commit, "mark" if mark else "build", dirty)


def entries(root: Path, slug: str, face: str) -> list[Entry]:
    """Every kept build of the face, dated and marked, newest first."""
    d = face_dir(root, slug, face)
    found = []
    for tex in [*d.glob("*.tex"), *d.glob("marks/*.tex")]:
        note = tex.with_suffix(".commit")
        words = note.read_text().split() if note.exists() else []
        found.append(Entry(tex.stem, tex, tex.with_suffix(".pdf"), words[0] if words else "",
                           "mark" if tex.parent.name == "marks" else "build", "dirty" in words))
    return sorted(found, key=lambda e: e.tex.stat().st_mtime, reverse=True)


def mark_of(root: Path, slug: str, face: str, name: str) -> Entry | None:
    return next((e for e in entries(root, slug, face) if e.kind == "mark" and e.label == name), None)


# --- references in git -------------------------------------------------------

def main_branch(root: Path) -> str | None:
    for name in ("main", "origin/main"):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", name], cwd=root, capture_output=True).returncode == 0:
            return name
    return None


def committed_tex(root: Path, commit: str, paper: Paper, face: str) -> str | None:
    """The path, inside the repository, of the face's .tex as committed at `commit`."""
    rel = paper.dir.relative_to(root)
    names = git("ls-tree", "--name-only", commit, "--", f"{rel}/", cwd=root).splitlines()
    found = [n for n in names if n.endswith(f"-{face}.tex")]
    return found[0] if len(found) == 1 else None


def build_at(root: Path, commit: str, paper: Paper, face: str, venue: str, form: str, into: Path) -> tuple[Path, Path]:
    """Build the face of `paper` as it was at `commit`, in a temporary worktree,
    and leave the .tex and PDF in `into` named by the commit's short hash.

    The worktree, rather than an export, so the build finds the manuscript's
    last-changed date in git and reproduces that commit's PDF byte for byte.
    HISTORY=0 keeps that build from recording itself or drawing its own diff."""
    label = git("rev-parse", "--short", commit, cwd=root)
    if not git("ls-tree", "--name-only", commit, "--", f"papers/{paper.slug}", cwd=root):
        raise RuntimeError(f"papers/{paper.slug} does not exist at {label}")
    tmp = Path(tempfile.mkdtemp(prefix="publish-at-"))
    tree = tmp / "tree"
    git("worktree", "add", "--detach", str(tree), commit, cwd=root)
    try:
        env = {**os.environ, "FORMATS": "venue", "HISTORY": "0"}
        r = subprocess.run(["bash", "publishing/publish.sh", paper.slug, f"--{venue}", f"--{form}"],
                           cwd=tree, env=env, capture_output=True, text=True)
        then = Paper(paper.slug, tree / "papers" / paper.slug)
        tex, pdf = then.dir / f"{then.stem}-{face}.tex", then.dir / f"{then.stem}-{face}.pdf"
        if r.returncode or not pdf.exists():
            tail = "\n".join((r.stdout + r.stderr).splitlines()[-15:])
            raise RuntimeError(f"building {paper.slug} ({venue}, {form}) at {label} failed:\n{tail}")
        into.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tex, into / f"{label}.tex")
        shutil.copyfile(pdf, into / f"{label}.pdf")
    finally:
        git("worktree", "remove", "--force", str(tree), cwd=root, check=False)
        shutil.rmtree(tmp, ignore_errors=True)
    return into / f"{label}.tex", into / f"{label}.pdf"


def from_commit(root: Path, commit: str, paper: Paper, face: str, venue: str, form: str,
                into: Path | None, kind: str, build: bool) -> Entry:
    """The face at a commit: its committed .tex when there is one, else a build."""
    full = git("rev-parse", "--verify", f"{commit}^{{commit}}", cwd=root)
    label = git("rev-parse", "--short", full, cwd=root)
    tex = pdf = None
    if into is not None:
        into.mkdir(parents=True, exist_ok=True)
        path = committed_tex(root, full, paper, face)
        if path:
            tex = into / f"{label}.tex"
            tex.write_bytes(subprocess.run(["git", "show", f"{full}:{path}"], cwd=root, capture_output=True, check=True).stdout)
        elif build:
            tex, pdf = build_at(root, full, paper, face, venue, form, into)
    return Entry(label, tex, pdf, full, kind)


def resolve(root: Path, paper: Paper, face: str, against: str | None = None, *, venue: str = "", form: str = "",
            into: Path | None = None, build: bool = True) -> Entry | None:
    """The version a build is compared with. With `into`, its .tex is copied
    there (so that a later write to the history cannot change it under the
    diff); with `into` None only the label and commit are found."""
    slug = paper.slug

    def snapshot(e: Entry) -> Entry:
        if into is None:
            return e
        into.mkdir(parents=True, exist_ok=True)
        copy = into / f"{e.label}.tex"
        shutil.copyfile(e.tex, copy)
        return Entry(e.label, copy, e.pdf, e.commit, e.kind, e.dirty)

    if against:
        marked = mark_of(root, slug, face, against)
        if marked:
            return snapshot(marked)
        given = Path(against)
        if given.suffix == ".tex" and given.is_file():
            label = re.sub(r"[^A-Za-z0-9._-]", "-", given.stem)
            return snapshot(Entry(label, given.resolve(), None, "", "path"))
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{against}^{{commit}}"],
                          cwd=root, capture_output=True).returncode == 0:
            return from_commit(root, against, paper, face, venue, form, into, "ref", build)
        raise ValueError(f"{against!r} is not a mark of {slug}/{face}, a .tex file, or a git ref")
    kept = entries(root, slug, face)
    if kept:
        return snapshot(kept[0])
    branch = main_branch(root)
    if branch:
        last = git("log", "-1", "--format=%H", branch, "--", str(paper.dir.relative_to(root)), cwd=root)
        if last:
            return from_commit(root, last, paper, face, venue, form, into, "main", build)
    return None


# --- the diff itself ---------------------------------------------------------

def latexdiff(reference: Path, new: Path, out: Path) -> bool:
    """Write latexdiff's markup of `new` against `reference`; False when latexdiff is not installed."""
    exe = shutil.which("latexdiff")
    if not exe:
        return False
    with open(out, "w") as f:
        r = subprocess.run([exe, *LATEXDIFF_ARGS, str(reference), str(new)], stdout=f, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        raise RuntimeError(f"latexdiff failed:\n{r.stderr.strip()}")
    return True


def figure_changes(root: Path, commit: str, paper: Paper) -> dict[str, list[str]]:
    """Figure files under resources/figures/ whose content differs between
    `commit` and the working tree, by git's own checksum."""
    figures = paper.dir / "resources" / "figures"
    rel = figures.relative_to(root)
    then = {}
    for line in git("ls-tree", "-r", commit, "--", f"{rel}/", cwd=root).splitlines():
        meta, path = line.split("\t", 1)
        then[str(Path(path).relative_to(rel))] = meta.split()[2]
    files = sorted(p for p in figures.rglob("*") if p.is_file()) if figures.is_dir() else []
    now = {}
    if files:
        r = subprocess.run(["git", "hash-object", "--stdin-paths"], cwd=root, capture_output=True, text=True,
                           input="\n".join(str(p) for p in files) + "\n", check=True)
        now = dict(zip((str(p.relative_to(figures)) for p in files), r.stdout.split(), strict=True))
    return {"changed": sorted(n for n in now if n in then and now[n] != then[n]),
            "added": sorted(n for n in now if n not in then),
            "removed": sorted(n for n in then if n not in now)}


# --- command line ------------------------------------------------------------

def shown(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="build history and marked-up diffs")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stamp", help="the entry name a build of this tree gets")
    p.add_argument("paper")
    p = sub.add_parser("record", help="keep a build, dated and under a mark's name when asked")
    for name in ("paper", "face"):
        p.add_argument(name)
    for name in ("tex", "pdf"):
        p.add_argument(name, type=Path)
    p.add_argument("--mark", help="also keep it under this name")
    p = sub.add_parser("mark", help="build a commit's face and keep it under a name")
    for name in ("paper", "face", "name"):
        p.add_argument(name)
    p.add_argument("--at", required=True, help="the git ref to build")
    p.add_argument("--venue", required=True)
    p.add_argument("--form", required=True, help="submission, preprint or accepted")
    p = sub.add_parser("reference", help="the version a build is compared with")
    for name in ("paper", "face"):
        p.add_argument(name)
    p.add_argument("--venue", required=True)
    p.add_argument("--form", required=True)
    p.add_argument("--into", type=Path, help="where the reference .tex is placed")
    p.add_argument("--against", help="a mark, a .tex path or a git ref")
    p.add_argument("--no-build", action="store_true", help="name the reference, but do not build a commit for it")
    p = sub.add_parser("diff", help="latexdiff two .tex files")
    for name in ("reference", "new", "out"):
        p.add_argument(name, type=Path)
    p = sub.add_parser("figures", help="figure files that differ between a commit and the working tree")
    for name in ("commit", "paper"):
        p.add_argument(name)
    a = ap.parse_args()
    root = ROOT

    if a.command == "diff":
        if not latexdiff(a.reference, a.new, a.out):
            print("latexdiff is not installed (brew install latexdiff, or tlmgr install latexdiff)", file=sys.stderr)
            return 2
        return 0

    paper = paper_in(root, a.paper)
    if not paper.dir.is_dir():
        print(f"no paper folder papers/{a.paper}", file=sys.stderr)
        return 1
    if a.command == "stamp":
        print(stamp(root, paper))
    elif a.command == "record":
        head = git("rev-parse", "HEAD", cwd=root)
        label = stamp(root, paper)
        dirty = label.endswith("-dirty")
        e = record(root, paper, a.face, a.tex, a.pdf, label, head, dirty=dirty)
        print(f"{shown(e.tex, root)} and .pdf")
        if a.mark:
            e = record(root, paper, a.face, a.tex, a.pdf, a.mark, head, mark=True, dirty=dirty)
            print(f"{shown(e.tex, root)} and .pdf")
    elif a.command == "mark":
        if not MARK_NAME.match(a.name):
            print(f"a mark is named with letters, digits, dots, dashes and underscores, not {a.name!r}", file=sys.stderr)
            return 1
        full = git("rev-parse", "--verify", f"{a.at}^{{commit}}", cwd=root)
        with tempfile.TemporaryDirectory(prefix="publish-mark-") as tmp:
            tex, pdf = build_at(root, full, paper, a.face, a.venue, a.form, Path(tmp))
            e = record(root, paper, a.face, tex, pdf, a.name, full, mark=True)
        print(f"{shown(e.tex, root)} and .pdf (built from {full[:7]})")
    elif a.command == "reference":
        e = resolve(root, paper, a.face, a.against, venue=a.venue, form=a.form,
                    into=None if a.no_build else a.into, build=not a.no_build)
        # a tab-separated line; the empty fields are written as "-" because
        # bash's read collapses consecutive tabs (tab is IFS whitespace)
        if e:
            print("\t".join([e.label, str(e.tex) if e.tex else "-", e.commit or "-", e.kind]))
    elif a.command == "figures":
        changes = figure_changes(root, a.commit, paper)
        short = a.commit[:7]
        if not any(changes.values()):
            print(f"no figure under resources/figures/ differs from {short}")
        for what, names in changes.items():
            if names:
                print(f"figures {what} since {short}: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

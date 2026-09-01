# `publishing/` — how the manuscripts become submittable

The manuscripts under `papers/` stay plain, readable Markdown; that is the
editing surface. Everything a venue wants — a title block, an abstract, real
citations, a generated bibliography, a conforming style file — is added here, on
a copy, at build time. **Nothing in this directory ever writes back to a
manuscript.**

The built formats land beside the Markdown they came from, in the paper's own
folder, so someone who opens `papers/02-untrained-reservoirs/` finds the PDF
next to the source rather than a build instruction.

## Build

```bash
bash publishing/publish.sh              # every paper, every format
bash publishing/publish.sh 02           # one paper
FORMATS="tmlr pdf" bash publishing/publish.sh   # `pdf` and `tex` are off by default: they duplicate the tmlr build
```

| format | for |
|---|---|
| `-tmlr.pdf` / `-tmlr.tex` | TMLR submission, via the journal's own style file |
| `-arxiv.tar.gz` | arXiv upload: `.tex` + style + `references.bib` + figures |
| `.epub` | e-readers |
| `.html` | a single self-contained file, images embedded and styled |
| `.docx` | venues that ask for Word |
| `.pdf` | reading and desk review — **off by default**, duplicates `-tmlr.pdf` |
| `.tex` | plain LaTeX source — **off by default**, duplicates `-tmlr.tex` |

Requires [pandoc](https://pandoc.org) and a TeX engine — BasicTeX is enough, and
its default packages cover everything the build needs. TeX binaries install to
`/Library/TeX/texbin`, which is not always on a non-interactive shell's PATH:

```bash
export PATH="/Library/TeX/texbin:$PATH"
```

## The TMLR build

[TMLR](https://jmlr.org/tmlr/) requires its own LaTeX style file, and states
that tweaking it may be grounds for rejection. `templates/tmlr/` therefore holds
the official files **unmodified**, vendored from
[JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file); every
adjustment lives in `templates/tmlr.latex`, the pandoc template that uses them.

One manuscript, three faces, selected by `TMLR_MODE`:

```bash
TMLR_MODE=submission bash publishing/publish.sh 01   # anonymous (the default)
TMLR_MODE=preprint   bash publishing/publish.sh 01   # for arXiv or a website
TMLR_MODE=accepted   bash publishing/publish.sh 01   # camera-ready
```

- **submission** — the author block is replaced with "Anonymous authors", and
  the running head reads "Under review as submission to TMLR". TMLR rejects
  non-anonymous submissions without review, so the build also blanks the PDF's
  own `pdfauthor` metadata: double-blind covers the file, not just the page.
  Neither manuscript names its author or links to a personal repository, so the
  anonymous build is genuinely anonymous — confirm with
  `pdftotext … - | grep -i <surname>` before uploading.
- **preprint** — de-anonymized, with every mention of TMLR removed. This is the
  face the arXiv bundle uses.
- **accepted** — camera-ready. Set `tmlr-month`, `tmlr-year` and
  `tmlr-openreview` in the paper's `metadata/paper.yaml` first; without them the
  header renders the template's `MM/YYYY` placeholders.

TMLR reviews papers whose main body runs past 12 pages on a longer timescale, so
the build reports the page count of each PDF it produces.

### Why the TMLR PDF is built with pdflatex

Every other format uses `xelatex`. `tmlr.sty` sets up `lmodern` with `T1`
encoding and Computer Modern math, which is the pdflatex-native combination;
under xelatex the fonts are re-resolved through `fontspec` and the Greek in the
equations drops out of the PDF **without an error**. The build picks `pdflatex`
for this one format and fails if any character goes missing.

## Figures

Manuscripts reference figures by real relative path, ending `.png`:

```markdown
![Self-contained caption.](resources/figures/fig7-model-timeline.png)
```

which is what renders on GitHub and in every HTML-ish format. `lib/preprocess.py`
points the LaTeX copies at the vector `.pdf` built from the same source, because
a raster figure in a submission PDF pixelates the moment a reviewer zooms. Each
figure therefore ships as three files: the `.svg` or script that produced it,
the `.pdf` for LaTeX, and the `.png` for everything else.

Two rules a caption has to follow. It must be **self-contained**, because most
reviewers read figures and captions before prose. And it must contain **no
links**: a `](` inside the alt text is where an image reference ends as far as
most Markdown parsers are concerned, so a citation in a caption silently turns
the figure's source into that citation's URL. Cite in the body instead.

## The title and the abstract

Both are written in the manuscript, the title as its `# ` heading and the
abstract under `## Abstract`, so they are edited beside the prose rather than in
a metadata file someone has to remember exists, and so a reader who opens the
Markdown on GitHub sees what the paper is called and what it claims.

`lib/title.py` and `lib/abstract.py` lift each one out at build time and hand it
to pandoc as title-block metadata, then remove it from the body copy: leaving it
in both places is what renders it twice. The title is stripped from every build,
the abstract only from the builds that pass `--abstract-out`, because pandoc
sets the title in all of them but not every path wants the abstract in the title
block. Neither is ever numbered as a section, by different routes: the title is
invisible to the numbering, because the manuscripts' own sections start at `##`
and that is where `check_sections.py` begins matching, while the abstract's
heading is matched and then explicitly skipped, over the span `abstract.py`
itself defines.

Each is therefore a single source. `metadata/paper.yaml` carries what is left:
the author list, keywords and the LaTeX front matter, the parts of a title block
that are not also part of the paper as it reads. `cite_this.py` reads the title
and the abstract from the manuscript for the BibTeX, RIS and CFF exports, so the
citation metadata cannot drift from the paper.

## Appendices

Everything after an `<!-- appendix -->` marker in a manuscript is appendix
material. TMLR places the appendix **after the references**, and its author
guide excludes appendices from the length that risks a longer review, so the
split is what keeps the main body inside the two-week window.

The TMLR and arXiv builds render the appendix separately and inject it through
pandoc's `include-after`, which the template emits below `\bibliography`. The
reading formats take the whole document unsplit, appendix inline, because
someone scrolling to the end expects to find it there. That difference is also
why there are two LaTeX copies: the reading PDF cites through citeproc and the
TMLR build through natbib, so a single appendix fragment cannot serve both.

## Greek and math in the prose

The manuscripts write equations inline, in Unicode: `θ̇ᵢ = ωᵢ + couplingᵢ(θ; K)`.
The default TeX text font has no glyph for theta, or omega, or a subscript i —
and **it drops such characters silently**. The PDF still builds; the equation
just comes out with holes in it.

So the LaTeX path gets its own preprocessed copy, in which those characters are
mapped onto real LaTeX math (`\ensuremath{\dot{\theta}_{i}}`). That is more
correct than hunting for a font with the coverage, because the characters really
are mathematics, and it needs no package BasicTeX lacks. HTML, EPUB, and DOCX
keep the Unicode, which their readers handle natively.

Two safeguards, because a silent failure is the whole problem here:

- the build **fails** if TeX reports any missing character, naming the ones to
  add to `MATH_CHARS` in `lib/preprocess.py`;
- the preprocessor **warns** when it finds TeX script syntax (`^{`, `_{`)
  written as prose rather than inside `$...$`, since pandoc escapes that into
  literal characters and any mapped symbol nearby then lands outside math.

Genuine formulas should be written as `$...$` math in the manuscript. They
typeset better everywhere, and HTML and EPUB get real MathML out of it.

## How citations work

One convention, in the two grammatical positions every venue distinguishes:

```markdown
textual        [Kuramoto (1975)](https://doi.org/10.1007/BFb0013365) showed that ...
parenthetical  ... self-organize ([Kuramoto 1975](https://doi.org/10.1007/BFb0013365)).
grouped        ... reservoir computing ([Jaeger 2001](url); [Maass et al. 2002](url)).
```

These are `\citet` and `\citep`, and `lib/preprocess.py` emits the right one
for each. A system is **named in the prose and cited beside it**, not
hyperlinked:

```markdown
yes   AKOrN ([Miyato et al. 2024](https://arxiv.org/abs/2410.13821)) uses ...
no    [AKOrN](https://arxiv.org/abs/2410.13821) uses ...
```

The second form looks like a citation and is not one. Because the citekey is
derived from the label, a work cited both by system name and by author-year
acquires **two bibliography entries for one DOI**, and two works that share a
label collapse into **one entry that is wrong for at least one of them**. Both
happened here before the convention was enforced. `lib/check_first_cite.py`
keeps the first-mention discipline, and paper 02 uses `[Author Year]` brackets
without links, which the same machinery understands.


Three steps turn that into real citations, all on a copy:

1. **`lib/extract_bib.py`** reads the manuscript's own citations and maintains
   the paper's `references/bibliography.bib`. Each entry keeps whatever
   identifier the prose carried — a DOI, an arXiv id, a URL — and the
   author-year that keys it. Re-running it appends new keys and leaves
   hand-completed entries alone, so the bibliography improves monotonically.

2. **`lib/preprocess.py`** rewrites the citations into pandoc's `@key` form,
   choosing the form that preserves the sentence's grammar: a parenthesised
   citation becomes `[@key]` so citeproc supplies the parentheses, and an
   in-text one becomes `@{key}` so the author name stays in the sentence.

3. **`lib/bibtex_compat.py`** produces the dialect legacy BibTeX understands,
   for the TMLR and arXiv paths only. `tmlr.bst` predates `@online`, drops
   entries whose type it does not recognise, and treats a `%` inside an entry as
   a syntax error — and a dropped entry is not a visible failure, it just
   renders as a bare key until natbib rejects the whole bibliography. The
   committed `.bib` stays correct; this hands BibTeX what it can read.

`publish.sh` runs all three, then pandoc.

### Completing the metadata

What the prose never states — full author lists, journal names, volumes,
pages — cannot be recovered from it. **`lib/fetch_metadata.py`** gets it from the
source of truth instead:

```bash
python3 publishing/lib/fetch_metadata.py            # fill what is missing
python3 publishing/lib/fetch_metadata.py --force    # re-fetch everything
```

Every DOI answers content negotiation on `doi.org` with CSL-JSON, whether it is
registered with CrossRef or DataCite, and arXiv mints a DOI for every preprint —
so one mechanism covers the whole bibliography. Several publishers also put the
DOI straight into the article URL, which is mined rather than given up on.
Hand-edited fields are never overwritten without `--force`.

Entries with no registered identifier cannot be looked up and are reported for a
human to finish. **`lib/check_bib.py`** is what says which:

```bash
python3 publishing/lib/check_bib.py           # report
python3 publishing/lib/check_bib.py --strict  # and fail if anything is missing
```

Completeness is checked, not annotated. A note in the `.bib` saying "VERIFY"
goes stale the moment someone fixes the entry and forgets the note; a check
reads the data as it actually is, every build — which is why `publish.sh` ends
by running it.

### Preprint or version of record

Where a work exists both as a preprint and as a published paper, cite the
published one — it is the version a reader should be sent to, and it is what
reviewers check. `fetch_metadata.py` reports every entry whose cited year
disagrees with the year its identifier resolves to, which is almost always an
entry pointing at the arXiv preprint of something that later appeared at a
conference. `lib/upgrade_versions.py` resolves those against CrossRef and
rewrites them to the proceedings version:

```bash
python3 publishing/lib/upgrade_versions.py --dry-run   # what it would change
python3 publishing/lib/upgrade_versions.py             # do it
```

The preprint's identifier is kept in a `note`, since that is often the copy a
reader can actually open.

### Do the section pointers still resolve

```bash
python3 publishing/lib/check_sections.py           # report
python3 publishing/lib/check_sections.py --strict  # and fail
```

Section numbers appear nowhere in the manuscripts: pandoc assigns them from the
heading order, under `--number-sections --shift-heading-level-by=-1`. So every
"Section 5.1" in the prose is a hand-kept copy of a number the build computes,
and moving one heading falsifies some of those copies while leaving them looking
perfectly plausible. This recomputes the numbering the same way the build does,
including the two rules that make it non-obvious — the abstract is lifted into
metadata rather than left as a section, and everything past the `<!-- appendix
-->` marker is lettered — then checks every `Section N.M` and `Appendix X`
reference against it. `publish.sh` runs it on every build.

Figure and table numbers are checked the same way and are the more fragile of
the two, since inserting one figure renumbers every reference after it. There
the only checkable property without a cross-reference package is that the number
exists at all, so both are counted in document order and a reference past the
last one is reported.

It catches references left dangling, not references merely wrong: one pointing
at a real but unintended section still resolves. The surrounding prose is
printed beside each finding for the cases a human has to judge.

### One address per reference

`tmlr.bst` prints every identifier field it finds, so an entry carrying both a
DOI and the publisher page that DOI resolves to states its address twice, and
the bibliography ends up in half a dozen visibly different shapes.
`lib/bibtex_compat.py` reduces each entry to one address, preferring **DOI, then
arXiv, then URL**. A URL survives when it is the only identifier, which is the
case for the ML venues that mint none, and a preprint id folded into `note`
survives beside the record it was upgraded from, since that is often the copy a
reader can actually open. `@misc` has no DOI slot in the style, so a bioRxiv or
dataset entry gets its DOI folded into `note` rather than dropped.

The same pass escapes bare underscores in the prose fields. TeX reads `S_N` in
a title as a subscript in text mode, which aborts the entry: the reference
renders as one run-on italic line and swallows everything after it, with no
error that names the entry. Registry metadata supplies such titles, so this
cannot be left to hand-editing. Underscores already inside `$…$` or a `\url{}`
are left alone.

### Do the links still work

```bash
python3 publishing/lib/check_links.py             # everything
python3 publishing/lib/check_links.py --bib-only  # skip the manuscript prose
```

A bibliography full of dead DOIs is worse than one with none: it looks
authoritative and sends the reader nowhere. This walks every DOI, arXiv id and
URL in the bibliographies *and* every link in the manuscripts. A 403 from a
publisher that plainly blocks robots is reported separately from a 404, because
they mean different things — one needs a human to look, the other needs the
citation fixed. Classification is on the *effective* URL, so a blocked publisher
is not reported as a dead DOI.

### A different citation style

The reading formats take their style from `CSL`, which names a file in
`publishing/csl/`. It defaults to `apa`, and `ieee` is vendored alongside it:

```
CSL=ieee bash publishing/publish.sh
```

An unknown name fails the build rather than producing a document with no
citations in it. For any other venue, drop its file in `publishing/csl/` — the
[CSL style repository](https://github.com/citation-style-language/styles) has
essentially every journal. The TMLR path ignores `CSL` entirely: it runs the
journal's own `tmlr.bst` through BibTeX.

**Author-year and numeric are not interchangeable for these manuscripts.** They
are written in author-year prose — "Fries (2015) develops that observation",
"dated as in Rivera-Sierra et al. (2026)" — and a numeric style replaces the
name with a bracketed number, so the sentence loses its subject and reads "[12]
develops that observation". Worse, a table cell whose entire content is a
citation renders as "[122]" and nothing else, which is how Appendix C's model
column came out blank. Switching to `ieee` for a numeric venue therefore means
rewriting the prose to match, in the form those venues expect: "Reference [12]
shows...". The switch is one variable; the rewrite is the work.

### Section numbers

Numbers live in the heading text of the Markdown, written by
`lib/number_sections.py`, so that a reader on GitHub can resolve "Section 2.3.3"
and "Appendix D" against visible headings. The script is idempotent and has a
`--check` mode that the build runs, so a moved heading fails rather than
shipping stale numbers.

Because the numbers are in the text, **both other numbering sources are off**:
pandoc's `--number-sections`, and LaTeX's own. The second is easy to miss —
`tmlr.latex` sets `secnumdepth`, which pandoc's flag does not touch, so dropping
`--number-sections` alone still produces "1 1 Introduction" in the PDF. The
build passes `--variable=secnumdepth=0` for that.

### Citations that can land on the wrong work

`lib/check_citemap.py` reports three ways a citation goes wrong quietly, and the
build fails on any of them.

A **shared label**: `[Nunley 2026](...)` derives its citekey from the label's
author and year, so two works by one author in one year collapse onto a single
key. One wins, the other's citations all follow it, and the other drops out of
the bibliography — with the reference list still complete and every claim still
cited. It happened five times here. A letter suffix, `Huang et al. 2026a`, is
the fix.

A **shared link**: two entries carrying one DOI or arXiv id are two records of
one work, and only the first is reachable, so the second is never cited. Merge
them.

A citation **matched on its label alone**: its link matches no `doi`, `eprint`
or `url` in the bibliography, so nothing checks that the entry is the work the
link points at. This is how a "Zhang et al. (2023)" in the prose came to print a
Zhang 2023 about a different subject. Recording the identifier the link uses is
what fixes it, and `entry_links` reads the arXiv id out of a `Preprint:
arXiv:...` note as well as out of the fields, because most of the DBLP-sourced
entries carry the venue page in `url` and the preprint only in the note.

### Figure and table numbers

LaTeX numbers its own floats, so the PDF captions already read "Figure 3:" and
"Table 1:". HTML, EPUB and DOCX number nothing, which left the manuscript's
"every paper in Table 1" pointing at no table a reader could identify.
`filters/number-floats.lua` prefixes the caption in exactly the formats that
need it, figures and tables on independent counters, in document order, and
returns an empty filter under LaTeX so the numbers are never written twice.

Hardcoding the number into the Markdown is the other option and is worse: the
PDF would then read "Figure 3: Figure 3.", and inserting one figure would
silently renumber every figure after it. `lib/check_sections.py` covers the
other half, failing the build when the prose cites a "Figure 7" or "Table 6"
that no caption in the document provides.

### Invisible characters

`lib/preprocess.py` strips zero-width spaces, directional and bidi marks,
variation selectors and the Unicode tag block from every built format, and says
so loudly when it finds any. The tag block is the usual carrier when prose is
watermarked: a run of it encodes text no reader can see. A no-break space is
normalised to a space rather than deleted, since removing it would run two words
together. `lib/check_hidden.py` reports what the source files still hold — the
Markdown is what gets read on GitHub, and the `.bib` never passes through
preprocess.

## Citing these papers

```bash
python3 publishing/cite_this.py             # write every format
python3 publishing/cite_this.py --print apa # just show one
```

Writes, into each paper's `metadata/` folder, `citation.bib`, `citation.ris`,
and a `citation.txt` carrying APA, MLA, Chicago, IEEE and Harvard. It also
writes `CITATION.cff` at the repository root (GitHub renders a "Cite this
repository" button from it) and injects the citation section into each paper's
README, between `<!-- citation:start -->` and `<!-- citation:end -->`.
Regenerating is idempotent, so a title or author change lands everywhere at once
and no copy can drift.

All of it derives from each paper's `metadata/paper.yaml`, which also supplies
the title block for every built format. Adding a co-author or fixing a title is
a one-place edit.

## Layout

```
publishing/
├── publish.sh              the one entry point
├── cite_this.py            citation metadata for these papers
├── lib/
│   ├── paths.py            where everything lives — the only module that knows
│   ├── extract_bib.py      manuscript -> starter .bib + citekey map
│   ├── preprocess.py       manuscript -> pandoc @key citations (on a copy)
│   ├── bibtex_compat.py    .bib -> the dialect legacy BibTeX reads
│   ├── fetch_metadata.py   complete entries from the DOI registries
│   ├── upgrade_versions.py preprint entries -> the published version of record
│   ├── check_bib.py        which entries are not yet complete enough to publish
│   ├── check_first_cite.py is every system and author cited where first named
│   ├── check_sections.py   does every Section/Appendix pointer resolve
│   ├── check_citemap.py    can any citation land on the wrong work
│   ├── check_hidden.py     invisible characters left in the source files
│   ├── number_sections.py  write the section numbers into the headings
│   └── check_links.py      does every citation still resolve
├── templates/
│   ├── tmlr.latex          pandoc template targeting the TMLR style file
│   └── tmlr/               the official TMLR style files, vendored unmodified
└── .work/                  preprocessed copies and build logs (gitignored)
```

Each paper owns the rest:

```
papers/<paper>/
├── <title>-DRAFT.md        the manuscript — the editing surface
├── <title>-DRAFT-tmlr.pdf  the submission, and -tmlr.tex beside it
├── <title>-DRAFT.epub …    epub, html and docx for reading and sharing
├── <title>-DRAFT-arxiv.tar.gz   the posting bundle
├── metadata/
│   ├── paper.yaml          title, authors, abstract, keywords
│   └── citation.{bib,ris,txt,md}   how to cite this paper
└── references/
    ├── bibliography.bib    the works it cites (committed — hand-edit this)
    └── citemap.json        citekey -> how the manuscript spells it
```

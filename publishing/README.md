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
bash publishing/publish.sh                 # every paper: named, TMLR style, no venue mentioned
bash publishing/publish.sh 02              # one paper, matched on its folder prefix
bash publishing/publish.sh 01 --tmlr       # the anonymous TMLR submission
bash publishing/publish.sh 01 --tmlr --accepted   # TMLR camera-ready
bash publishing/publish.sh 01 --tmlr --dry-run    # say what would be written, build nothing
FORMATS="venue pdf" bash publishing/publish.sh 01 # `pdf` and `tex` are off by default: they duplicate the venue build
```

| format | for |
|---|---|
| `-<venue>.pdf` / `.tex` | the submission to that venue, in its own style file — `-tmlr.pdf` for TMLR |
| `-preprint.pdf` / `.tex` | the named face of the same build: author on the title page and in the PDF metadata, no venue mentioned |
| `-<venue>-accepted.pdf` / `.tex` | camera-ready for that venue |
| `-arxiv.tar.gz` | arXiv upload: `.tex` + style + `references.bib` + figures, always the preprint face in the TMLR style |
| `.epub` | e-readers |
| `.html` | a single self-contained file, images embedded and styled |
| `.docx` | venues that ask for Word |
| `.pdf` | a plain article-class PDF for reading and desk review — **off by default** |
| `.tex` | plain LaTeX source — **off by default** |

Requires [pandoc](https://pandoc.org) and a TeX engine — BasicTeX is enough, and
its default packages cover everything the build needs. TeX binaries install to
`/Library/TeX/texbin`, or under `/usr/local/texlive/<year>/bin/<platform>`,
neither of which is always on a non-interactive shell's PATH; without an engine
the build skips the venue PDF and says so:

```bash
export PATH="/Library/TeX/texbin:$PATH"
```

## Venues and faces

Two flags decide the LaTeX build, and together they read as the decision they
record.

**The venue flag** says which style file to format for. `--tmlr` uses
[TMLR](https://jmlr.org/tmlr/)'s; `--<name>` uses any other venue that has been
added as `templates/<name>.latex` with its style files in `templates/<name>/`
(see below). Naming a venue means formatting *for* it, so the face defaults to
that venue's submission form. With no venue flag the TMLR style is used anyway,
because it is vendored, it builds, and its author-year citation format is the
one these manuscripts are written in.

**The face flag** says who the document is for:

- `--submission` (also `--anonymous`) — the venue's review copy. For TMLR the
  author block is replaced with "Anonymous authors", the running head reads
  "Under review as submission to TMLR", and the build blanks the PDF's own
  `pdfauthor` metadata, because double-blind covers the file and not just the
  page. Neither manuscript names its author or links to a personal repository,
  so the anonymous build is genuinely anonymous — confirm with
  `pdftotext … - | grep -i <surname>` before uploading.
- `--preprint` — the author named on the title page and in the PDF metadata,
  and no venue mentioned anywhere. This is the face for a website, Zenodo or
  arXiv, and **it is the default when no flag is given**: ordinary,
  non-anonymous publishing in the house style.
- `--accepted` — camera-ready. For TMLR, set `tmlr-month`, `tmlr-year` and
  `tmlr-openreview` in the paper's `metadata/paper.yaml` first; without them the
  header renders the template's `MM/YYYY` placeholders.

Each combination writes its own file, so building one face never overwrites
another:

| command | face | writes |
|---|---|---|
| `publish.sh 01` | preprint (named) | `…-preprint.pdf`, `…-preprint.tex` |
| `publish.sh 01 --preprint` | the same, spelled out | `…-preprint.pdf` |
| `publish.sh 01 --tmlr` | TMLR submission (anonymous) | `…-tmlr.pdf`, `…-tmlr.tex` |
| `publish.sh 01 --tmlr --preprint` | named, TMLR style | `…-preprint.pdf` |
| `publish.sh 01 --tmlr --accepted` | TMLR camera-ready | `…-tmlr-accepted.pdf` |
| `publish.sh 01 --neunet` | *Neural Networks* submission (named: the journal is single-blind) | `…-neunet.pdf`, `…-neunet.tex` |
| `publish.sh 01 --neunet --preprint` | named, Elsevier style, running foot "Preprint" and the date instead of the class's "Preprint submitted to Elsevier" | `…-neunet-preprint.pdf`, `…-neunet-preprint.tex` |
| `publish.sh 02 --iclr` | ICLR submission (anonymous, line-numbered, "Under review" running head) | `…-iclr.pdf`, `…-iclr.tex` |
| `publish.sh 02 --iclr --preprint` | named, ICLR style, no venue claimed | `…-iclr-preprint.pdf` |

The venue flag only changes the LaTeX build. The reading formats (`epub`,
`html`, `docx`, `pdf`) always carry the author, and the arXiv bundle is always
the preprint face in the TMLR style, whichever venue the PDF was built for.
`--dry-run` prints the resolved venue, face, formats and output paths and exits
before pandoc is needed, which is also how `tests/test_publish_cli.py` checks
the resolution.

TMLR reviews papers whose main body runs past 12 pages on a longer timescale, so
the build reports the page count of each PDF it produces.

### The TMLR style

TMLR requires its own LaTeX style file, and states that tweaking it may be
grounds for rejection. `templates/tmlr/` therefore holds the official files
**unmodified**, vendored from
[JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file); every
adjustment lives in `templates/tmlr.latex`, the pandoc template that uses them.
The template selects the face from two metadata fields the build sets,
`venue-face` (`submission`, `preprint` or `accepted`) and `venue-submission`
(true only for the anonymous face), which is what any other venue's template is
expected to read as well.

### The Neural Networks style

`--neunet` formats for Elsevier's *Neural Networks*, which takes LaTeX
submissions in Elsevier's CAS single-column class. `templates/neunet/` holds
`cas-sc.cls`, `cas-common.sty` and `cas-model2-names.bst` **unmodified**,
vendored from Elsevier's `els-cas-templates` bundle (v2.4, with its README and
manifest); `templates/neunet.latex` is the pandoc template that drives them.
The journal is single-anonymized, so the submission face carries the author
and the class's own running foot, "Preprint submitted to Elsevier", which is
what the journal expects to see. The preprint face (`--neunet --preprint`,
written as `…-neunet-preprint.pdf`) is the same page with that foot replaced
by "Preprint" and the manuscript's date, so it can go to SSRN, arXiv or a
website without claiming a destination. There is no accepted face: Elsevier
typesets the final article itself.

The template fills Elsevier's front matter from `metadata/paper.yaml`: the
structured affiliation (`organization`, `city`, `state`, `country`), `orcid`,
`credit` (the CRediT roles, which `\printcredits` prints as their own section),
`shorttitle` and `keywords`. `shorttitle` is the running head the class prints
at the top of every page; the running foot, "Preprint submitted to Elsevier",
is the class's own wording for any manuscript not yet accepted and cannot be
changed to the journal's name. The end matter follows the journal's order:
the appendices, then `competing-interests`, `funding` and `data-availability`
as unnumbered declaration sections directly before the references. The
generative-AI declaration the journal requires is part of the manuscript
itself, its last section after the appendices, so every format carries it in
the same place. The highlights are not printed in the PDF: the journal wants
them as a separate file, which the docx build writes (below).

`templates/neunet.yaml` sets two pandoc-level knobs no template can set for
itself: `indent: true`, so pandoc does not load `parskip` over the class's
paragraph shape, and `natbiboptions: authoryear`, the citation form the
journal's APA-style references call for.

The class needs packages BasicTeX does not ship. Once, per machine:

```bash
tlmgr init-usertree; tlmgr --usermode install stix inconsolata footmisc xstring moreverb makecell sttools wrapfig multirow
```

Without `stix` the class silently falls back to Computer Modern and says so in
the TeX log (`publishing/.work/<paper>.neunet.final.log`).

### The Word submission files

Journals that take Word manuscripts want the same things the LaTeX template
prints, inside the `.docx`, plus two files beside it. `lib/front_matter.py`
renders all of it from `metadata/paper.yaml`, so the Word and LaTeX routes
never drift:

- `….docx` is the manuscript in submission form: the title block with the
  affiliation and the corresponding author, the keywords under the abstract,
  the body with its appendices, then CRediT, competing interests, funding and
  data availability directly before the references. A portal that extracts
  metadata from Word files finds title, abstract, keywords and author where it
  expects them.
- `…-title-page.docx` is the separate title page a portal asks for: authors
  with affiliation marks, corresponding author, ORCID, acknowledgements
  (`acknowledgements:` in `paper.yaml`, "None." when absent) and funding.
- `…-highlights.docx` holds the `highlights:` list and nothing else, with
  "highlights" in the file name as Elsevier asks. The build refuses to write it
  when there are fewer than three or more than five, or one runs over 85
  characters; a paper without highlights gets no file.

All three take their styles from one reference document, pandoc's own with a
single change: links in a darker blue (`LINK_COLOR` in
`lib/reference_docx.py`), because the stock theme blue reads faint. The
document is generated into `.work/` at build time, so no binary lives here.

### The ICLR style

`--iclr` formats for ICLR, which is double-blind: naming the venue gives the
anonymous face, with "Anonymous authors", line numbers and the running head
"Under review as a conference paper at ICLR". `templates/iclr/` holds ICLR's
own 2027 style files **unmodified** (see its README for the source), and
`templates/iclr.latex` drives them. `--iclr --preprint` names the author
through the style's camera-ready switch and clears the running head it would
print, because a preprint is not an ICLR paper; `--iclr --accepted` keeps it.

ICLR allows nine pages of main text at submission. References, appendices and
the AI-use, ethics and reproducibility statements do not count, and the
appendix goes after the references. The build reports total pages, so check
which page the reference list starts on.

The anonymous face blanks the PDF's author metadata, but it cannot anonymize
the prose. A third-person citation of your own preprint is the form ICLR
permits; a repository URL or an acknowledgement is not. Check the file:

```bash
pdftotext …-iclr.pdf - | grep -n -i -E "<surname>|github|<your domain>"
```

The style needs font metrics BasicTeX does not ship. Once, per machine:

```bash
tlmgr init-usertree; tlmgr --usermode install helvetic times courier
```

Two traps this venue exposed, both now handled in the build and both worth
knowing for the next one. A template that skips pandoc's fonts partial, to keep
the venue's own typeface, must load `iftex` itself, because the common partial
tests `\ifLuaTeX`. And the bibliography style reaches pandoc as a template
variable, not as metadata: pandoc escapes metadata for LaTeX, so
`iclr2027_conference` became `iclr2027\_conference`, BibTeX found no such
style, and every citation came out undefined with no LaTeX error at all.

### Adding a venue

1. Put the venue's official style files, unmodified, in `templates/<name>/`:
   every `.sty`, `.cls`, `.bst` and helper `.tex` it ships. The build stages
   all of them beside the generated `.tex` and passes the `.bst` it finds there
   to natbib as the bibliography style.
2. Write `templates/<name>.latex`, a pandoc template that loads that style and
   reads `venue-face` / `venue-submission` to choose between the anonymous,
   named and camera-ready title blocks. `templates/tmlr.latex` is the worked
   example for a double-blind venue, including the appendix injection after
   the references and the `pdfauthor` handling for each face;
   `templates/neunet.latex` is the one for a single-blind journal class with
   its own front matter (highlights, keywords, CRediT, declarations).
3. If the template needs metadata that pandoc's own LaTeX partials read
   (`indent`, `natbiboptions`, `colorlinks`, …), put it in
   `templates/<name>.yaml`. The build passes that file after the paper's own
   metadata, so the venue's values win.
4. Build with `bash publishing/publish.sh <paper> --<name>`; the PDF lands as
   `…-<name>.pdf`. Run `--dry-run` first to see the resolution.

If the venue wants numeric citations, see "A different citation style" below;
the venue build cites through natbib and the venue's `.bst`, not through `CSL`.

### Why the venue PDF is built with pdflatex

Every other format uses `xelatex`. `tmlr.sty` sets up `lmodern` with `T1`
encoding and Computer Modern math, which is the pdflatex-native combination;
under xelatex the fonts are re-resolved through `fontspec` and the Greek in the
equations drops out of the PDF **without an error**. Venue style files are
generally written for pdflatex, so the build picks it for this one format and
fails if any character goes missing.

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

A citation **matched on its label alone**: either its link matches no `doi`,
`eprint` or `url` in the bibliography, or it is a bracket citation and carries no
link to match. Either way nothing checks that the entry is the work meant. This is how a "Zhang et al. (2023)" in the prose came to print a
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

### Hidden characters and fingerprints

A file can be marked without changing a word a reader would notice, and a build can leave traces that tie a submitted file to one machine or one moment. The build removes both, so an anonymous submission carries nothing but the paper.

**Characters.** `lib/sanitize.py` defines what could carry an unseen mark, and every format is built from cleaned copies: the manuscript through `lib/preprocess.py`, the bibliography and `metadata/paper.yaml` through `sanitize.py` itself. It deletes invisible characters (zero-width and joiner characters, directional marks, variation selectors, the Unicode tag block that generated prose is usually watermarked with, soft hyphens, fillers, private-use and control characters); turns no-break, thin and other unusual spaces into ordinary ones; turns a Cyrillic or Greek letter that passes for a Latin one, inside a Latin word, into that Latin letter (whole Greek words such as θ are left alone), and fullwidth ASCII into ASCII; and drops trailing whitespace, keeping a Markdown hard break in its backslash form. `lib/check_hidden.py` reports what the sources themselves still hold, since the Markdown is what gets read on GitHub; `--fix` cleans them in place. Watermarks that shift word choice rather than characters cannot be removed this way.

**Files.** Every date and timestamp a format records is the manuscript's last-changed day at midnight UTC (`SOURCE_DATE_EPOCH`), never the build's time or time zone. The LaTeX PDFs carry no document ID, no dates, no creator or producer, and none of pdfTeX's keys for included figures, which otherwise name each figure file and copy its own metadata (`templates/no-fingerprints.latex`); every glyph maps back to its characters, so copied text reads as written. The EPUB's identifier is derived from the paper's name instead of drawn at random, the HTML carries no generator tag, the arXiv tarball records no owner, group or file times (`lib/archive.py`), and the figures are saved with no date or tool version. The supplement zip is cleaned file by file on the way in: its text as above, its sound and images without metadata chunks (libsndfile's PEAK chunk in a float WAV records when the file was written). Two builds of the same sources are byte-identical.

**The check.** `lib/check_outputs.py` runs at the end of every build over the files it just wrote, and fails the build on any hidden character or look-alike letter in their text, any ID, date or tool field in a PDF, a build time or random identifier in a Word or EPUB file, or an owner, varying time or metadata chunk in an archive. A PDF glyph with no Unicode mapping, such as a bitmap font's ligature, is reported as a note rather than a failure: text extraction shows it as a control code, but pdfTeX cannot put one in the page, and the `.tex` it typeset is checked in full.

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
├── <title>.md              the manuscript — the editing surface
├── <title>-preprint.pdf    the named face, and -preprint.tex beside it
├── <title>-tmlr.pdf        the anonymous TMLR submission, and -tmlr.tex beside it
├── <title>.epub …          epub, html and docx for reading and sharing
├── <title>-arxiv.tar.gz    the posting bundle
├── metadata/
│   ├── paper.yaml          title, authors, abstract, keywords
│   └── citation.{bib,ris,txt,md}   how to cite this paper
└── references/
    ├── bibliography.bib    the works it cites (committed — hand-edit this)
    └── citemap.json        citekey -> how the manuscript spells it
```

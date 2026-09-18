# ICLR 2027 style files

Vendored **unmodified** from ICLR's own bundle,
<https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>, fetched
18 September 2026: `iclr2027_conference.sty`, `iclr2027_conference.bst`, and the
copies of `fancyhdr.sty`, `natbib.sty` and `math_commands.tex` that ship with
it. The bundle's sample paper and sample bibliography are left out.

ICLR asks for its official style files, unaltered, so every adjustment lives in
`../iclr.latex`, the pandoc template that loads them. `natbib.sty` and
`fancyhdr.sty` are staged beside the generated `.tex`, so ICLR's copies are the
ones LaTeX finds, as they would be in ICLR's own template.

Next year's files replace these, and the `\usepackage` line in `../iclr.latex`
changes its year with them.

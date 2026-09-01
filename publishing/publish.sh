#!/usr/bin/env bash
# Build the readable and submittable formats of each manuscript.
#
#     bash publishing/publish.sh              # every paper, every format
#     bash publishing/publish.sh 02           # one paper (match on folder prefix)
#     FORMATS="tmlr pdf" bash publishing/publish.sh
#     TMLR_MODE=preprint bash publishing/publish.sh 01
#
# The manuscripts stay plain readable Markdown. Everything a venue wants — a
# title block, an abstract, real citations, a generated bibliography, a
# conforming style file — is added here, on a copy, at build time. Nothing
# under papers/ is modified except by writing the built formats beside the
# Markdown, so a reader lands on the folder and finds the PDF next to the
# source rather than in a build directory.
#
# Pipeline per paper:
#   1. refresh the .bib from the manuscript's own citations (new keys appended,
#      hand-completed entries preserved)
#   2. rewrite [Author Year] / [Label](doi) into pandoc @key citations
#   3. run pandoc once per output format
#   4. report which bibliography entries are still incomplete
#
# Formats:
#   tmlr   TMLR-conforming PDF + LaTeX, via the vendored official style file.
#          TMLR_MODE picks the face: submission (anonymous, the default),
#          preprint (de-anonymized, no mention of TMLR), accepted (camera-ready,
#          needs tmlr-month / tmlr-year / tmlr-openreview in metadata).
#   pdf epub html docx   general reading formats, citeproc-rendered
#   tex arxiv            LaTeX source, and a self-contained arXiv upload bundle
#
# Two companion tools, run when you want them rather than every build:
#   publishing/lib/fetch_metadata.py  completes entries from the DOI registries
#   publishing/lib/check_links.py     confirms every citation link still resolves
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORK="publishing/.work"
TEMPLATES="publishing/templates"
# tmlr is the submission; epub/html/docx are for reading and sharing; arxiv is the
# posting bundle. The plain `pdf` and `tex` formats are dropped from the default
# because they duplicate the tmlr ones — pass them explicitly if you want them.
FORMATS="${FORMATS:-tmlr epub html docx arxiv}"
TMLR_MODE="${TMLR_MODE:-submission}"
FILTER="${1:-}"

command -v pandoc >/dev/null || {
    echo "pandoc is required: brew install pandoc" >&2; exit 1
}

# A LaTeX PDF is what TMLR, arXiv and most journals want. Without a TeX engine
# we still produce a PDF, via HTML, and say so rather than failing quietly.
PDF_ENGINE=""
for e in xelatex lualatex pdflatex tectonic; do
    command -v "$e" >/dev/null && { PDF_ENGINE="$e"; break; }
done
# tmlr.sty sets up lmodern with T1 and Computer Modern math, which is the
# pdflatex-native combination; under xelatex the fonts are re-resolved through
# fontspec and the Greek in the equations drops out of the PDF silently.
TMLR_ENGINE="$(command -v pdflatex >/dev/null && echo pdflatex || echo "$PDF_ENGINE")"
PDF_VIA_HTML=0
if [ -z "$PDF_ENGINE" ] && command -v weasyprint >/dev/null; then
    PDF_ENGINE="weasyprint"
    PDF_VIA_HTML=1
fi

# A dropped glyph does not fail a TeX run on its own: the PDF is produced,
# silently missing symbols. Catch it here or ship an equation with holes in it.
check_glyphs() {
    grep -q "Missing character" "$1" || return 0
    echo "    ERROR: characters dropped from the PDF —" >&2
    grep -o "There is no [^ ]*" "$1" | sort -u | head >&2
    echo "    add them to MATH_CHARS in publishing/lib/preprocess.py" >&2
    return 1
}

built=0
missing=0
for dir in papers/*/; do
    slug="$(basename "$dir")"
    [ -n "$FILTER" ] && [[ "$slug" != "$FILTER"* ]] && continue
    # the manuscript is the paper's only Markdown file that is not its README.
    # `-DRAFT` was once required in the name and is now merely tolerated, so a
    # paper can drop it once the draft is out the door.
    manuscript="$(ls "$dir"*-DRAFT.md 2>/dev/null | head -1)"
    [ -z "$manuscript" ] && manuscript="$(ls "$dir"*.md 2>/dev/null | grep -v '/README\.md$' | head -1)"
    [ -z "$manuscript" ] && { echo "skip $slug: no manuscript Markdown file"; continue; }

    bib="$dir/references/bibliography.bib"
    citemap="$dir/references/citemap.json"
    meta="$dir/metadata/paper.yaml"
    # built formats take the manuscript's own name, so a downloaded PDF still
    # says which paper it is and sorts next to the Markdown it came from
    name="$(basename "$manuscript" .md)"
    mkdir -p "$WORK"

    echo
    echo "=== $slug"
    echo "--- refreshing the bibliography"
    inherit=""
    [ "$slug" != "01-evidence-audit" ] && [ -f papers/01-evidence-audit/references/bibliography.bib ] \
        && inherit="--inherit papers/01-evidence-audit/references/bibliography.bib"
    python3 publishing/lib/extract_bib.py "$manuscript" $inherit

    echo "--- rewriting citations"
    body="$WORK/$slug.md"
    abstract_yaml="$WORK/$slug.abstract.yaml"
    python3 publishing/lib/preprocess.py "$manuscript" "$citemap" --out "$body" \
        --abstract-out "$abstract_yaml"
    # the LaTeX path needs its own copy: the default TeX text font has no glyph
    # for Greek or for a subscript i, and drops such characters SILENTLY, so
    # they are mapped to math rather than left to vanish
    # Two LaTeX copies. The reading formats take the whole document, appendix
    # and all, because a reader scrolling to the end expects to find it there.
    # TMLR and arXiv take a split copy, because TMLR places the appendix after
    # the references and excludes it from the length that risks a longer review.
    tex_full="$WORK/$slug.tex.md"
    tex_body="$WORK/$slug.tex.split.md"
    tex_appx="$WORK/$slug.tex.appendix.md"
    python3 publishing/lib/preprocess.py "$manuscript" "$citemap" --out "$tex_full" \
        --abstract-out /dev/null --for latex
    python3 publishing/lib/preprocess.py "$manuscript" "$citemap" --out "$tex_body" \
        --abstract-out /dev/null --appendix-out "$tex_appx" --for latex

    # the manuscript's own last-changed date, so a rebuild is reproducible
    date="$(git log -1 --format=%ad --date=short -- "$manuscript" 2>/dev/null)"
    [ -z "$date" ] && date="$(date +%F)"

    common=(--from=markdown+tex_math_dollars+pipe_tables+footnotes
            --metadata-file="$meta" --metadata-file="$abstract_yaml" --metadata=date="$date"
            --resource-path="$dir:$dir/resources/figures" --standalone)
    # The manuscripts reference figures by real relative path, ending .png, so
    # they render on GitHub and in every HTML-ish format. preprocess.py points
    # the LaTeX copies at the vector PDF beside each PNG, because a raster
    # figure in a submission PDF is visibly worse and reviewers zoom.
    raster=(--lua-filter=publishing/filters/number-figures.lua
            --css=publishing/css/tables.css)
    # the HTML outputs separate table rows through tables.css; this is the same
    # decision for the LaTeX ones, where booktabs rules only the head and foot
    vector=(--lua-filter=publishing/filters/table-row-rules.lua)
    # section numbers come from pandoc, not from the heading text, so every
    # format agrees; and the reading formats need a plain author string,
    # because pandoc's stock template renders our structured author as "true"
    # the manuscript's top level is `##`, because the title and abstract come
    # from metadata rather than being restated in the prose. Without the shift
    # pandoc maps `##` to a subsection and every number comes out as 0.n
    common+=(--number-sections --shift-heading-level-by=-1
             --variable=author="$(python3 publishing/lib/byline.py "$meta")")
    # TMLR places the appendix after the references, and its author guide
    # excludes appendices from the length that risks a longer review. The LaTeX
    # paths therefore render it separately and inject it through include-after,
    # which the template emits below \bibliography. The reading formats keep it
    # inline, where someone scrolling to the end expects to find it.
    appendix_arg=()
    if [ -s "$tex_appx" ]; then
        appx_tex="$WORK/$slug.appendix.tex"
        { echo "\\appendix"
          pandoc "${common[@]}" "${vector[@]}" --to=latex --natbib \
                 --standalone=false "$tex_appx"
        } > "$appx_tex"
        appendix_arg=(--include-after-body="$appx_tex")
    fi

    # citeproc renders the bibliography itself for the reading formats; the
    # TMLR path instead hands the .bib to BibTeX so the journal's own .bst runs
    # Numeric [1] citations for the reading formats. The TMLR path does NOT use
    # this: its stylefile mandates natbib author-year — "citations within the
    # text should ... include the authors' last names and year" — so the two
    # builds cite differently on purpose.
    cite=(--citeproc --bibliography="$bib" --csl=publishing/csl/ieee.csl)

    for fmt in $FORMATS; do
        case "$fmt" in
            tmlr)
                if [ -z "$PDF_ENGINE" ] || [ "$PDF_VIA_HTML" = 1 ]; then
                    echo "    (tmlr needs a TeX engine: install BasicTeX or MacTeX)"
                    continue
                fi
                out="$WORK/tmlr-$slug"
                rm -rf "$out"; mkdir -p "$out"
                cp "$TEMPLATES"/tmlr/*.sty "$TEMPLATES"/tmlr/math_commands.tex "$out/"
                cp "$TEMPLATES"/tmlr/tmlr.bst "$out/"
                python3 publishing/lib/bibtex_compat.py "$bib" --out "$out/references.bib"
                if [ -d "$dir/resources/figures" ]; then
                    mkdir -p "$out/resources/figures"
                    cp "$dir"/resources/figures/*.pdf "$out/resources/figures/" 2>/dev/null
                fi
                # --natbib leaves the citations as \citep/\citet for BibTeX,
                # which is what tmlr.bst and TMLR's instructions expect
                pandoc "${common[@]}" "${vector[@]}" --to=latex --natbib \
                    --template="$TEMPLATES/tmlr.latex" \
                    --metadata=tmlr-mode="$TMLR_MODE" \
                    --metadata=tmlr-submission="$([ "$TMLR_MODE" = submission ] && echo true)" \
                    --metadata=biblio-style=tmlr ${appendix_arg[@]+"${appendix_arg[@]}"} \
                    --output="$out/$name.tex" "$tex_body" || continue
                log="$WORK/$slug.tmlr.log"
                final="$WORK/$slug.tmlr.final.log"
                # BibTeX needs the full latex/bibtex/latex/latex cycle to
                # resolve \citep keys and settle the cross-references. The
                # first pass has no .bbl yet, so its "undefined citation"
                # warnings are expected — only the final pass is diagnostic,
                # and it gets its own log.
                (cd "$out" \
                    && "$TMLR_ENGINE" -interaction=nonstopmode "$name.tex" \
                    && bibtex "$name" \
                    && "$TMLR_ENGINE" -interaction=nonstopmode "$name.tex") >"$log" 2>&1
                (cd "$out" && "$TMLR_ENGINE" -interaction=nonstopmode "$name.tex") >"$final" 2>&1
                if [ -f "$out/$name.pdf" ]; then
                    cp "$out/$name.pdf" "$dir$name-tmlr.pdf"
                    cp "$out/$name.tex" "$dir$name-tmlr.tex"
                    # the TeX log hard-wraps at 79 columns, and will happily
                    # split "(14 pages" across two lines
                    pages="$(tr -d '\n' <"$final" | grep -oE "Output written[^)]*" \
                             | grep -oE "[0-9]+ pages" | tail -1)"
                    echo "    $dir$name-tmlr.pdf ($TMLR_MODE${pages:+, $pages})"
                    check_glyphs "$final" || missing=1
                    undefined="$(grep -c "Citation .* undefined" "$final")"
                    [ "$undefined" != 0 ] && {
                        echo "    ERROR: $undefined citation(s) did not resolve — see $final" >&2
                        missing=1
                    }
                else
                    echo "    ERROR: the TMLR build failed — see $log" >&2
                    grep -m3 "^!" "$log" >&2
                    missing=1
                fi
                ;;
            pdf)
                if [ -z "$PDF_ENGINE" ]; then
                    echo "    (no PDF engine: install a TeX distribution, or 'brew install weasyprint')"
                elif [ "$PDF_VIA_HTML" = 1 ]; then
                    pandoc "${common[@]}" "${cite[@]}" "${raster[@]}" --to=html5 --embed-resources \
                        --pdf-engine=weasyprint --output="$dir$name.pdf" "$body" \
                        && echo "    $dir$name.pdf (via weasyprint; install TeX for a LaTeX PDF)"
                else
                    log="$WORK/$slug.pdf.log"
                    pandoc "${common[@]}" "${cite[@]}" "${vector[@]}" --pdf-engine="$PDF_ENGINE" \
                        --include-in-header="$TEMPLATES/float-fit.latex" \
                        --output="$dir$name.pdf" "$tex_full" 2>"$log" \
                        && echo "    $dir$name.pdf ($PDF_ENGINE)"
                    check_glyphs "$log" || missing=1
                    grep -v "Missing character" "$log" | grep -i "warning" | head -3
                fi
                ;;
            epub)
                pandoc "${common[@]}" "${cite[@]}" "${raster[@]}" --to=epub3 --toc --toc-depth=2 \
                    --output="$dir$name.epub" "$body" && echo "    $dir$name.epub"
                ;;
            html)
                pandoc "${common[@]}" "${cite[@]}" "${raster[@]}" --to=html5 --toc --toc-depth=3 \
                    --embed-resources --output="$dir$name.html" "$body" \
                    && echo "    $dir$name.html"
                ;;
            docx)
                pandoc "${common[@]}" "${cite[@]}" "${raster[@]}" --to=docx --output="$dir$name.docx" "$body" \
                    && echo "    $dir$name.docx"
                ;;
            tex)
                pandoc "${common[@]}" "${cite[@]}" "${vector[@]}" --to=latex \
                    --output="$dir$name.tex" "$tex_full" \
                    && echo "    $dir$name.tex"
                ;;
            arxiv)
                # arXiv wants LaTeX source plus everything it references, and it
                # runs BibTeX itself, so ship the .bib rather than a baked-in
                # bibliography. The preprint face of the TMLR style is the right
                # one here: de-anonymized, with no mention of the journal.
                bundle="$WORK/arxiv-$slug"
                rm -rf "$bundle"; mkdir -p "$bundle"
                pandoc "${common[@]}" "${vector[@]}" --to=latex --natbib \
                    --template="$TEMPLATES/tmlr.latex" --metadata=tmlr-mode=preprint \
                    --metadata=biblio-style=tmlr ${appendix_arg[@]+"${appendix_arg[@]}"} \
                    --output="$bundle/$name.tex" "$tex_body" || continue
                python3 publishing/lib/bibtex_compat.py "$bib" --out "$bundle/references.bib"
                cp "$TEMPLATES"/tmlr/*.sty "$TEMPLATES"/tmlr/tmlr.bst \
                   "$TEMPLATES"/tmlr/math_commands.tex "$bundle/"
                if [ -d "$dir/resources/figures" ]; then
                    mkdir -p "$bundle/resources/figures"
                    cp "$dir"/resources/figures/*.pdf "$bundle/resources/figures/" 2>/dev/null
                fi
                (cd "$WORK" && tar czf "$ROOT/$dir$name-arxiv.tar.gz" "arxiv-$slug")
                echo "    $dir$name-arxiv.tar.gz (tex + style + references.bib + figures)"
                ;;
            *) echo "    unknown format '$fmt'" ;;
        esac
    done

    built=$((built + 1))
done

echo
echo "=== bibliography completeness"
python3 publishing/lib/check_bib.py

echo
echo "=== citation labels shared by two works"
python3 publishing/lib/check_citemap.py

echo
echo "=== authors named before they are cited"
python3 publishing/lib/check_first_cite.py

echo
echo "=== section cross-references"
python3 publishing/lib/check_sections.py

echo
echo "built $built paper(s), beside their Markdown under papers/"
[ "$missing" = 1 ] && { echo "one or more builds failed or dropped characters — see above" >&2; exit 1; }
echo "citation metadata for these papers: python3 publishing/cite_this.py"

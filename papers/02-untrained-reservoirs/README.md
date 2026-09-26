# Spoken-Digit Recognition Without Training

Geometry, coupling, and drive effects in oscillator networks.

**[Read the paper →](spoken-digit-recognition-without-training.md)**

An untrained coupled oscillator network (1,024 oscillators, 2,048 parameters) is read by a linear readout on spoken-digit recognition (AudioMNIST, speakers held out, with added noise at signal-to-noise ratios of 0 and −5 dB) and compared with a spectrogram-only baseline, an uncoupled oscillator network, two leaky-integrator banks and five trained baselines of about the same size, every arm read the same way. Eight experiments, 6,353 runs in all, vary the coupling function, lattice geometry (a coil and a cochlea among them), natural frequencies, restoring strength, coupling ceiling, input pathway, input gain, readout width and training-set size, on recognition and on a temporal-order task.

Most of the accuracy comes from the input representation and the readout. At a matched width the Kuramoto network reads within a percentage point of the readout fitted to its input alone in noise, and near chance when the audio drives it in quadrature; it carries the order of events by integrating its input, as a non-oscillating leaky-integrator bank does. A free oscillator amplitude (Stuart–Landau) is the one design choice that moved accuracy by more than about a point, lifting the network above its input and into the range of the trained baselines, which otherwise read higher; the lattice geometry, the cochlea included, moved it by less than a point.

[`results/README.md`](results/README.md) lists the experiments and says which file of the record holds each; the terms are defined in the paper's glossary (Appendix A).

## What is here

| path | contents |
|---|---|
| [`spoken-digit-recognition-without-training.md`](spoken-digit-recognition-without-training.md) | the manuscript, and the file to edit |
| [`….pdf`](spoken-digit-recognition-without-training.pdf) · [`.html`](spoken-digit-recognition-without-training.html) · [`.epub`](spoken-digit-recognition-without-training.epub) · [`.docx`](spoken-digit-recognition-without-training.docx) | the same paper to read or download |
| [`…-iclr.pdf`](spoken-digit-recognition-without-training-iclr.pdf) | the ICLR 2027 submission build, anonymous and line-numbered, in ICLR's own style |
| [`…-preprint.pdf`](spoken-digit-recognition-without-training-preprint.pdf) | the preprint build, with the author named and no venue claimed |
| [`…-arxiv.tar.gz`](spoken-digit-recognition-without-training-arxiv.tar.gz) | LaTeX source, style files, bibliography and figures, ready to upload |
| [`references/bibliography.bib`](references/bibliography.bib) | the works it cites |
| [`metadata/`](metadata/) | its front matter, and how to cite it |
| [`src/`](src/) | the experiment harness, its driver, and the tests: [start here](src/README.md) |
| [`scripts/`](scripts/) | the paper's own scripts, outside the experiment code: the appendix schematics, the audio examples, and a runner for a GPU pod ([below](#scripts)) |
| [`results/`](results/) | the record, one file per experiment and task, with `summary.md` and a [README](results/README.md) saying what each file holds; every number in the paper comes from it |
| `resources/figures/` | the paper's figures |
| [`resources/audio/examples/`](resources/audio/examples/) | clips as the arms hear them: a test digit clean, at 0 dB and at −5 dB, and order-task sequences in both orders |
| [`resources/notes/`](resources/notes/) | the working design notes, with their decision log, and the AudioMNIST prior-art survey, kept for reference |
| [`resources/pilot/`](resources/pilot/) | the pilot's noise calibration: the script, its clips and its record |
| `…-supplement.zip` | the anonymized supplementary material for review: `src/`, `results/` and the audio examples; built, not committed ([below](#rebuilding-the-paper)) |

## Checking the numbers

No corpus needed; the committed results are enough:

```bash
cd src && uv sync && uv run python -m harness.experiment.summary
```

Every accuracy and paired difference prints with its spread over seeds and a 95% interval over test clips. [`src/README.md`](src/README.md) covers getting the data, the code layout, the tests, and re-running the experiments from scratch.

## Scripts

The paper's own scripts sit in [`scripts/`](scripts/), outside the experiment code, and run with its environment, from `src/`:

```bash
cd src
uv run python ../scripts/draw_coupling_figure.py      # the appendix schematics; also draw_geometries_ and draw_read_figure.py
uv run python ../scripts/export_audio_examples.py     # the audio examples, from the bank
```

The schematics read nothing from the record. Authored SVGs render to PDF and PNG from the repository root with `uv run --with svglib --with reportlab python3 publishing/lib/svg_render.py papers/02-untrained-reservoirs/resources/figures`. [`scripts/pod_run.sh`](scripts/pod_run.sh) runs any experiment on a RunPod GPU pod, or any Linux machine with a GPU, from the pushed branch: `bash scripts/pod_run.sh <AudioMNIST checkout> [experiment ...]`.

## Rebuilding the paper

```bash
bash publishing/publish.sh 02            # every format
bash publishing/publish.sh 02 --iclr     # the ICLR submission build, and the supplement zip
```

Regenerates every format above, in place, from the Markdown, and writes the anonymized supplement zip beside them (the `supplement` format, `python3 publishing/supplement.py 02` on its own). See [publishing/README.md](../../publishing/README.md).

<!-- citation:start -->
## Citing this paper

Kryski, E. (2026). Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Oscillator Networks [Preprint]. https://github.com/ekryski/oscillator-research/blob/main/papers/02-untrained-reservoirs/

```bibtex
@techreport{kryski2026spokendigit,
  author      = {Eric Kryski},
  title       = {Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Oscillator Networks},
  year        = {2026},
  institution = {Independent research},
  type        = {Preprint},
  url         = {https://github.com/ekryski/oscillator-research/blob/main/papers/02-untrained-reservoirs/},
  keywords    = {coupled oscillators, reservoir computing, speech recognition, Kuramoto model, Stuart-Landau, lattice geometry},
}
```

Other formats, regenerated by `python3 publishing/cite_this.py`:

- [BibTeX](metadata/citation.bib)
- [RIS](metadata/citation.ris) (EndNote, Zotero, Mendeley)
- [APA, MLA, Chicago, IEEE, Harvard](metadata/citation.txt)
- [CITATION.cff](../../CITATION.cff) for the repository as a whole

Please cite the version you actually used: these manuscripts are drafts, and every number in them is reproducible from the record shipped beside them.
<!-- citation:end -->

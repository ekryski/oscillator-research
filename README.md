# oscillator-research

![Sound enters a lattice of coupled oscillators; the field's collective state is read out as a digit.](resources/field.png)

> Scientific research into coupled oscillators as a computational substrate for speech recognition and generation.

## What this is about

A coupled-oscillator network is a population of rhythmic units that pull each
other toward phase agreement. Wire them onto a lattice, drive them with sound,
and the field's collective state becomes a representation of that sound: the
physics does the transducing. The idea is old — synchronization theory, the
Hopf model of the cochlea, reservoir computing, Mead's case for running physics
directly — and it is currently enjoying a revival in machine learning, with
oscillator-based systems posting competitive results in long-sequence modeling,
vision, and image generation, and with physical oscillator hardware classifying
spoken digits at a fraction of the energy a trained network needs.

Speech is a natural target, because speech is oscillation at every scale:
prosody near 1 Hz, syllable rhythm at 4-8 Hz, phone transitions at 10-40 Hz,
pitch and formants from 100 Hz to several kHz. An oscillator field is a
frequency-selective medium with intrinsic timescales, locking behavior, and
spatial wave modes — exactly the representational resources that structure
would seem to want.

Whether that promise survives contact with controlled measurement is the open
question, and it is the question this research addresses. The work here is
deliberately small-scale and control-heavy: single-variable comparisons,
pre-registered decision criteria written before each run, no-dynamics floors
under every accuracy, parameter-matched conventional baselines, and randomized
twins for every designed structure. Negative results are reported with the same
weight as positive ones.

Every paper ships with the code and the raw per-run data that produced its
numbers, so any claim here can be peer reviewed and validated or refuted.

## Papers

| # | Paper | What it does |
|---|---|---|
| 01 | [From Synchronization Physics to Trained Dynamics: A Survey of Oscillator Networks in Machine Learning](papers/01-evidence-audit/from-synchronization-physics-to-trained-dynamics.md) | A prior-art survey of the oscillator-computing revival, auditing 88 published sources — including load-bearing results that exist only as technical blog posts — and organizing them into nine hypotheses the literature motivates but does not settle. For each, it states what supports it, what bounds it, and the controlled experiment that would decide it. |
| 02 | [Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Frozen Oscillator Fields](papers/02-untrained-reservoirs/) | Freezes every physics parameter of a 1,024-oscillator field and measures spoken-digit recognition across 1,940 pre-registered experiment runs — six coupling laws, six lattice geometries, three frequency structures, three drive pathways — against conventional baselines at exact parameter parity. Untrained fields beat every trained baseline at that budget yet clear their own no-dynamics floor by only ~3 points, and no design axis moves the result; on a task that provably requires memory, the same frozen fields read temporal order at 0.97-1.00. |

Paper 01 is a survey without any code or data. Paper 02 also includes the code used in the experiment (src + tests), launch and scoring scripts, per-run raw results across the entire experiment matrix, and audio and
figure assets used in or that support the reader in understanding the paper — see its [README](papers/02-untrained-reservoirs/README.md) for how to reproduce the experiments.

## Layout

```
papers/
├── 01-evidence-audit/
│   ├── README.md
│   ├── from-synchronization-physics-to-trained-dynamics.md   the manuscript
│   ├── ….pdf .html .epub .docx .tex                          built beside it
│   ├── …-tmlr.pdf  …-arxiv.tar.gz                      submission builds
│   ├── metadata/    front matter, and how to cite this paper
│   └── references/  the works it cites
└── 02-untrained-reservoirs/
    ├── … the same, plus
    ├── src/         harness, sweep driver, tests
    ├── results/     the record: 1,940 experiment runs
    └── resources/   figures and audio

publishing/          the scripts that turn a manuscript into those formats
```

Each paper is self-contained: the manuscript, the formats built from it, the
bibliography it cites, and its own citation metadata all live in one folder.

## Publishing

The manuscripts are plain Markdown; `publishing/` turns them into every other
format without touching the source, writing each one beside the Markdown it came
from — PDF, EPUB, HTML, DOCX, LaTeX, a TMLR submission build against the
journal's own style file, and a ready-to-upload arXiv bundle:

```bash
bash publishing/publish.sh
```

Citations are resolved against a real bibliography whose author lists, titles,
venues and years come from the DOI registries and DBLP rather than from prose,
with tooling to check that every link still resolves and that no entry is too
incomplete to publish.

To cite a paper, see the citation section in its README —
[01](papers/01-evidence-audit/README.md#citing-this-paper),
[02](papers/02-untrained-reservoirs/README.md#citing-this-paper) — or
[CITATION.cff](CITATION.cff) for the repository as a whole. Details in
[publishing/README.md](publishing/README.md).

## Data

Speech corpora are not committed. Paper 02 uses
[AudioMNIST](https://github.com/soerenab/AudioMNIST); download it and build the
local bank as described in that paper's README.

## License

[Apache 2.0](LICENSE).

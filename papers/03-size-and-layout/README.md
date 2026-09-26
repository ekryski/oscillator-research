# Size and Channel Layout in Untrained Coupled Oscillator Networks

How the number of oscillators and their arrangement affect spoken-digit recognition and memory, against controls of the same size.

**[Read the draft →](size-and-channel-layout-DRAFT.md)** · **Status: design and harness; no experiment has been run.**

Paper 02 ([Spoken-Digit Recognition Without Training](../02-untrained-reservoirs/)) read one untrained coupled oscillator network, 1,024 oscillators in 4 channels of a 16 × 16 lattice with 2,048 parameters, by a linear readout on speaker-disjoint AudioMNIST, against a spectrogram-only baseline, an uncoupled copy of the network, leaky-integrator banks and trained baselines of the same size. Paper 01 ([From Synchronization Physics to Trained Dynamics](../01-evidence-audit/)), a survey of oscillator networks in machine learning, found that no published system had ablated its oscillator or parameter count or its channel layout.

Paper 03 runs that ablation for paper 02's networks, at a signal-to-noise ratio of 0 dB and input gain 1. The lattice runs from 8 × 8 to 128 × 128 and the channel count from 1 to 16, so the network has 64 to 262,144 oscillators and 128 to 524,288 parameters, and every lattice is driven either by its own number of mel bands, one per row, or by paper 02's 16 bands mapped onto its rows; at 64 and 128 bands the front end's analysis window is varied too. Every channel's coupling kernel is scaled to exactly the coupling ceiling. At every size the network is compared with the same controls as in paper 02, matched in size: its uncoupled copy, a leaky-integrator bank with its states and parameters, the spectrogram-only baseline on the same rows, and five trained baselines sized to its parameter count. The readout stays at paper 02's 192 features, so whatever changes with size is the network's. Besides recognition, the networks run paper 02's order task and a new task, naming each digit of a spoken sequence of two, three or four, and every network records its synchronization and its locking to its drive. The size study is crossed with paper 02's six coupling functions, its eight lattice geometries (the coil and the cochlea among them) and its spectrogram and quadrature pathways: nine experiments, 13,316 runs.

The experiments are listed in the paper's Appendix B, with their cost in Appendix E, and [`results/README.md`](results/README.md) says which file of the record holds each; the terms are defined in the paper's glossary (Appendix A).

## How it builds on papers 01 and 02

- **The gap** it fills is one paper 01 names: parameter and oscillator count, and channel layout, had not been ablated.
- **The tasks, data, protocol, noise, read and readout** are paper 02's, unchanged, and so is every arm at 16 × 16; the digit-sequence task is new.
- **Paper 02's runs are reused**, not rerun, where its record holds them completely: 66 of the 234 runs paper 02 made at 16 × 16 with 4 channels are read from its record and cited to it, and the other 168, which it recorded under its fixed projection only for a read paper 03 reports, are run again on the CPU. Scaling every kernel exactly to the coupling ceiling, which paper 03 introduces because a random 8 × 8 kernel usually falls short of it, leaves every 16 × 16 network bit-identical to paper 02's, and a reuse check runs a sample of paper 02's runs again and compares every cell with its record.

## What is here

| path | contents |
|---|---|
| [`size-and-channel-layout-DRAFT.md`](size-and-channel-layout-DRAFT.md) | the manuscript and the design document: a draft abstract, the introduction, background and methods, results left to write, and appendices (glossary, experimental design, channels and layers, coupling functions, computation and cost, coupling functions and lattice geometries, methods in full) |
| [`references/`](references/) | the bibliography and citation map |
| [`metadata/`](metadata/) | the front matter |
| [`src/`](src/) | the experiment harness, derived from paper 02's, its driver and the tests: [start here](src/README.md) |
| [`scripts/`](scripts/) | the paper's own scripts, outside the experiment code: the schematics and a runner for a GPU pod ([below](#scripts)) |
| [`results/`](results/) | the record, one file per experiment, task and lattice, as the experiments run, with a [README](results/README.md) saying what each file holds |
| `results/benchmark/` | GPU benchmark reports (`plan benchmark`), which price the experiments on the machine that ran them |
| `resources/figures/` | the paper's figures |

## Checking the design

```bash
cd src && uv sync
uv run pytest                                       # the contract tests, no data needed
uv run python -m harness.experiment.plan estimate   # runs, CPU- and GPU-hours and memory for every experiment and lattice
uv run python -m harness.experiment.plan benchmark --device cuda --out ../results/benchmark/cuda.json   # time it on a GPU
```

Running the experiments needs paper 02's digit bank; see [src/data/README.md](src/data/README.md).

## Scripts

The paper's own scripts sit in [`scripts/`](scripts/), outside the experiment code, and run with its environment, from `src/`:

```bash
cd src
uv run python ../scripts/draw_size_figure.py          # the schematics; also draw_channels_, draw_coupling_ and draw_geometries_figure.py
```

The schematics read nothing from the record: the channels, coupling-function and geometry schematics are paper 02's, relabelled where paper 03's lattices differ, and the size figure is paper 03's own. [`scripts/pod_run.sh`](scripts/pod_run.sh) sets up a RunPod GPU pod, or any Linux machine with a GPU, from the pushed branch, runs the tests and the benchmark, and then any experiments it is given: `bash scripts/pod_run.sh [<AudioMNIST checkout> experiment ...]`.

## Rebuilding the paper

```bash
bash publishing/publish.sh 03            # every format
```

Regenerates every format, in place, from the Markdown. See [publishing/README.md](../../publishing/README.md).

# Size and Channel Layout in Untrained Coupled Oscillator Networks

How the number of oscillators and their arrangement affect spoken-digit recognition and memory, against controls of the same size.

**[Read the draft →](size-and-channel-layout-DRAFT.md)** · **Status: design and harness only; no run has been made.**

## What it asks

Paper 02 ([Spoken-Digit Recognition Without Training](../02-untrained-reservoirs/)) read one untrained coupled oscillator network, 1,024 oscillators in 4 channels of a 16 × 16 lattice with 2,048 parameters, by a linear readout on speaker-disjoint AudioMNIST, against a spectrogram-only baseline, an uncoupled copy of the network, leaky-integrator banks and trained baselines of the same size. Paper 01 ([From Synchronization Physics to Trained Dynamics](../01-evidence-audit/)), a survey of oscillator networks in machine learning, found that no published system had ablated its oscillator or parameter count or its channel layout.

Paper 03 runs that ablation for paper 02's networks. The lattice runs from 8 × 8 to 128 × 128 and the channel count from 1 to 16, so the network has 64 to 262,144 oscillators and 128 to 524,288 parameters, and every lattice is driven either by its own number of mel bands, one per row, or by paper 02's 16 bands mapped onto its rows; at 64 and 128 bands the front end's analysis window is varied too. At every size the network is compared with the same controls as in paper 02, matched in size: its uncoupled copy, a leaky-integrator bank with its states and parameters, the spectrogram-only baseline on the same rows, and five trained baselines (GRU, TCN, CNN, transformer, S4D) sized to its parameter count. The readout stays at paper 02's 192 features, so whatever changes with size is the network's, and every projected read is recorded under paper 02's fixed projection and under one seeded by the run's seed. Besides recognition, the size tier runs paper 02's order task (which of two digits came first), and a new tier asks the network to name each digit of a spoken sequence of two, three or four. Every network also records its rotation rates, paper 02's secondary read, and its synchronization and locking to its drive. The size study then crosses paper 02's six coupling functions, six lattice geometries and three input pathways, all at 0 dB and input gain 1. The study asks questions and states no hypotheses. It runs on a CUDA GPU, on Apple Silicon's GPU or on the CPU; only the CPU is bit-identical to paper 02.

## How it builds on papers 01 and 02

- **The gap** it fills is one paper 01 names: parameter and oscillator count, and channel layout, had not been ablated.
- **The tasks, data, protocol, noise, read and readout** are paper 02's, unchanged, and so is every arm at 16 × 16; the digit-sequence task is new.
- **Paper 02's cells are reused**, not rerun: its 16 × 16, 4-channel runs (225 of the 15,489 planned) are read from its record and cited to it. Scaling every coupling kernel exactly to the coupling ceiling, which paper 03 introduces because a random 8 × 8 kernel usually falls short of it, leaves every 16 × 16 network bit-identical to paper 02's; a reuse gate re-runs a sample and requires every cell to match.
- **The size tier** began as paper 02's widened tier 4, which the author moved into this paper.

## What is here

| path | contents |
|---|---|
| [`size-and-channel-layout-DRAFT.md`](size-and-channel-layout-DRAFT.md) | the manuscript, in paper 02's outline: a draft abstract, the introduction, background and methods, results left to write, and appendices (glossary, experimental design, channels and layers, coupling functions) |
| [`DESIGN.md`](DESIGN.md) | the design: questions, data and tasks, arms and size matching, the read and the instruments, the tiers, what is taken from paper 02, reporting, the open questions, and a dated decision log |
| [`TIERS.md`](TIERS.md) | what each tier runs, its run count, its cost in CPU- and GPU-hours and memory, and the stages to run it in |
| [`references/`](references/) | the bibliography and citation map |
| [`metadata/`](metadata/) | the front matter |
| [`src/`](src/) | the harness, derived from paper 02's, with its tests: [start here](src/README.md) |
| `resources/figures/` | the figures, drawn by `src/scripts/draw_*_figure.py` (three adapted from paper 02's) |
| `results/` | the record, as the tiers run |
| `results/benchmark/` | GPU benchmark reports (`plan benchmark`), which price the tiers on the machine that ran them |

## Checking the design

```bash
cd src && uv sync
uv run pytest                                    # the contract tests, no data needed
uv run python -m harness.experiment.plan estimate   # runs, CPU- and GPU-hours and memory for every tier and lattice
uv run python -m harness.experiment.plan benchmark --device cuda --out ../results/benchmark/cuda.json   # time it on a GPU
```

Running the tiers needs paper 02's digit bank; see [src/data/README.md](src/data/README.md).

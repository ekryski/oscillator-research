# Size and Channel Layout in Untrained Coupled Oscillator Networks

How the number of oscillators and their arrangement affect spoken-digit recognition, against controls of the same size.

**[Read the draft →](size-and-channel-layout-DRAFT.md)** · **Status: design and harness only. The registration is a draft, not frozen, and no run has been made.**

## What it asks

Paper 02 ([Spoken-Digit Recognition Without Training](../02-untrained-reservoirs/)) read one untrained coupled oscillator network, 1,024 oscillators in 4 channels of a 16 × 16 lattice with 2,048 parameters, by a linear readout on speaker-disjoint AudioMNIST, against a spectrogram-only baseline, an uncoupled copy of the network, leaky-integrator banks and trained baselines of the same size. Paper 01 ([From Synchronization Physics to Trained Dynamics](../01-evidence-audit/)), a survey of oscillator networks in machine learning, found that no published system had ablated its oscillator or parameter count or its channel layout.

Paper 03 runs that ablation for paper 02's networks. The lattice runs from 8 × 8 to 128 × 128 and the channel count from 1 to 16, so the network has 64 to 262,144 oscillators and 128 to 524,288 parameters, and every lattice is driven either by its own number of mel bands, one per row, or by paper 02's 16 bands mapped onto its rows. At every size the network is compared with the same controls as in paper 02, matched in size: its uncoupled copy, a leaky-integrator bank with its states and parameters, the spectrogram-only baseline on the same rows, and five trained baselines (GRU, TCN, CNN, transformer, S4D) sized to its parameter count. The readout stays at paper 02's 192 features, so whatever changes with size is the network's. The size study then crosses paper 02's six coupling functions, six lattice geometries and three input pathways.

## How it builds on papers 01 and 02

- **The gap** it fills is one paper 01 names: parameter and oscillator count, and channel layout, had not been ablated.
- **The task, data, protocol, noise, read and readout** are paper 02's, unchanged, and so is every arm at 16 × 16.
- **Paper 02's cells are reused**, not rerun: its 16 × 16, 4-channel runs (477 of the 32,223 planned) are read from its record and cited to it. Scaling every coupling kernel exactly to the coupling ceiling, which paper 03 introduces because a random 8 × 8 kernel usually falls short of it, leaves every 16 × 16 network bit-identical to paper 02's; a reuse gate re-runs a sample and requires every cell to match.
- **The size tier** began as paper 02's widened tier 4, which the author moved into this paper.

## What is here

| path | contents |
|---|---|
| [`size-and-channel-layout-DRAFT.md`](size-and-channel-layout-DRAFT.md) | the manuscript: title, a draft abstract, section headings, the experimental design, and the glossary |
| [`REGISTRATION.md`](REGISTRATION.md) | the draft pre-registration: questions, data, arms and size matching, the read, the tiers, what is taken from paper 02, reporting |
| [`TIERS.md`](TIERS.md) | what each tier runs, its run count, its cost in CPU-hours and memory, and the stages to run it in |
| [`references/`](references/) | the bibliography and citation map |
| [`metadata/`](metadata/) | the front matter |
| [`src/`](src/) | the harness, derived from paper 02's, with its tests: [start here](src/README.md) |
| `results/confirmatory/` | the record, as the tiers run |

## Checking the design

```bash
cd src && uv sync
uv run pytest                                    # the contract tests, no data needed
uv run python -m harness.confirm.plan estimate   # runs, CPU-hours and memory for every tier and lattice
```

Running the tiers needs paper 02's digit bank; see [src/data/README.md](src/data/README.md).

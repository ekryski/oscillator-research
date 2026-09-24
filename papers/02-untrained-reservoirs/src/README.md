# Experiment code

Everything needed to rerun the study and check its numbers. Its record is [`../results/`](../results/), whose [README](../results/README.md) says what each experiment ran and which file holds it. Every number in the paper comes from that record, and nothing else is in it.

## Quick start

```bash
cd papers/02-untrained-reservoirs/src
uv sync
uv run pytest                                   # contract tests, no data needed
uv run python -m harness.experiment.summary     # every accuracy and difference, from the committed record
```

Neither needs the corpus: the tests run on a small synthetic bank, and the summary reads the committed record. AudioMNIST is only needed to rerun experiments.

## Getting the data

The task is spoken digits 0 to 9 from [AudioMNIST](https://github.com/soerenab/AudioMNIST) (Becker et al. 2018): 30,000 recordings, 60 speakers × 10 digits × 50 repetitions at 48 kHz.

```bash
git clone --depth 1 https://github.com/soerenab/AudioMNIST data/AudioMNIST
uv run python -m harness.experiment.protocol --build-bank
```

**Neither the corpus nor the derived bank is committed.** `src/data/` is gitignored down to its README: the corpus belongs to its own authors, and the bank is a deterministic function of it. See [data/README.md](data/README.md) for what the build does and for the `OSC_DATA_DIR` override.

## The experiment

Every arm (the spectrogram-only baseline, the coupled and uncoupled oscillator networks, two leaky-integrator banks, five trained baselines) is read by one contract: the same three statistics over the same fixed window for every clip, standardized with the training set's own statistics, projected to a common width, and classified by the same linear readout.

```bash
uv run python -m harness.experiment.protocol --build-bank   # the 50-repetition bank
uv run python -m harness.experiment.plan prepare            # front-end rows, once per pathway and noise level
uv run python -m harness.experiment.plan run leak-check controls --workers 3 --threads 2
uv run python -m harness.experiment.plan run design becker-folds quadrature projection sweep cochlea --workers 6 --threads 1
uv run python -m harness.experiment.summary                 # every accuracy and difference with its spread, and the leak check
uv run python -m harness.experiment.figures                 # the paper's figures and their tables
```

`plan run <experiment> --dry-run` prints what would run. Every run is recorded under an identity derived from its specification, so a sweep stopped at any point restarts where it left off, and `--task order` or `--task recognition` lets two machines split an experiment without sharing a file. `--device mps` or `--device cuda` simulates the untrained arms on a GPU; the readout is closed-form and runs on the CPU, and the trained baselines always train on the CPU. A GPU rounds differently from the CPU, so its cells can differ from the CPU's by a test clip or two. `scripts/pod_run.sh <AudioMNIST checkout> [experiment ...]` runs any experiment on a RunPod pod from the pushed branch. Set `OSC_RESULTS_DIR` to write a reproduction into a fresh folder instead of the committed record.

| module | what it owns |
|---|---|
| `experiment/protocol.py` | the bank, Protocols A and B, the order task, per-clip noise, row caches |
| `experiment/arms.py` | the arms, the reads each one records, and the coherence instruments |
| `experiment/readout.py` | standardize, project, ridge, per-clip correctness |
| `experiment/run.py` | one run, and the record it writes |
| `experiment/plan.py` | the experiments, and the parallel driver |
| `experiment/record.py` | the record read back, one cell at a time |
| `experiment/summary.py` | every accuracy and paired difference, with its spread, and the zero-input leak check: `summary.json` and `summary.md` |
| `experiment/figures.py` | the paper's figures, and the tables printed beside them |
| `experiment/terms.py` | the paper's names for the record's labels, used by every table, figure and report |
| `stimuli/` | the log-mel front end, the hop-rate and quadrature rows, the digit clips |
| `models/field.py`, `phase.py`, `stuart_landau.py` | the oscillator network and its two cores |
| `models/geometries/` | one module per lattice geometry, behind a four-method interface |
| `models/leaky_bank.py` | the leaky-integrator bank |
| `models/baselines/` | the five trained baselines |
| `measurement/` | the read's statistics, the projection, the ridge, the phase instruments |
| `scripts/draw_*_figure.py` | the appendix schematics (channels, geometries, coupling functions) |

### Terms in the code

The paper's glossary (Appendix A) defines the terms. The code and the record use them in short form; `experiment/terms.py` spells them out for everything people read. The model layer (`models/`) keeps its own older parameter names, which `experiment/arms.py` maps onto.

| paper term | in the code and the record | in the model layer |
|---|---|---|
| spectrogram-only baseline | arm kind `baseline`; its whole-clip reads are `windowed@wholeclip` and `pooled@wholeclip` | |
| coupled oscillator network | arm kind `network`, labels `coupled-<coupling>-<geometry>-<frequencies>-restoring<λ>-ceiling<c>` | `OscillatorField` |
| uncoupled oscillator network | `network` with `coupled=False`, labels `uncoupled-...` | kernel zeroed |
| leaky-integrator bank, state-matched / width-matched | `bank-state` / `bank-width` | `LeakyBank`, 4 / 8 channels |
| trained baselines | `trained-gru`, `trained-tcn`, `trained-cnn`, `trained-transformer`, `trained-s4d` | `models/baselines/` |
| reservoir | the untrained arms (`UNTRAINED` in `plan.py`) | |
| coupling function | `coupling`: `kuramoto`, `kuramoto-sakaguchi`, `second-harmonic`, `winfree`, `stuart-landau`, `stuart-landau-fixed` | `core` and `coupling` |
| lattice geometry | `geometry` | `boundary` |
| natural frequencies: random / tonotopic / identical | `frequencies` | `natural_freqs` |
| restoring strength λ | `restoring` | `damping` |
| coupling ceiling | `ceiling` | `spectral_clamp` |
| input gain | `gain`, `g` in run ids | `gain` |
| input pathway: spectrogram / quadrature | `pathway` | |

## Tests

```bash
uv run pytest                          # everything
uv run pytest tests/test_geometries.py # one module
uv run pytest -q -k order
```

One file per module. They are contract tests: each pins a property some claim in the paper depends on, such as that a run's identity comes from its specification alone, that an untrained arm with no input reads exactly chance, that cached rows equal the rows a run would compute, that the instruments change no cell, that uncoupling zeroes the coupling and nothing else, and that the Apple GPU reads an untrained arm exactly as the CPU does on a small synthetic bank (on the real data the two agree within two test clips in all but one of 1,296 cells; see the summary).

### Adding a geometry

Subclass `Geometry` (or `PlanarGeometry`), implement the four methods, and register it in `models/geometries/__init__.py`. The parametrized tests then cover it: FFT against dense equivalence, forward determinism, every coupling function, the ceiling correction, and the tonotopic map.

## Figures

`experiment/figures.py` draws the result figures from the record. The appendix schematics are drawn by `scripts/draw_*_figure.py`, and read nothing from the record. Authored SVGs render to PDF and PNG from the repository root:

```bash
uv run --with svglib --with reportlab python3 publishing/lib/svg_render.py \
    papers/02-untrained-reservoirs/resources/figures
```

## Determinism

**Forward passes are bit-identical**, on every geometry and both coupling implementations, as `tests/test_geometries.py` asserts. That is the replication claim that matters, because every number from a reservoir comes from an untrained network, which never runs a backward pass.

**Backward is reproducible only to float32 rounding** on the dense coupling path: its gradient accumulates through a gather whose reduction order is not pinned. This affects only the trained baselines.

The rest of the protocol is deterministic by construction: each clip's noise is seeded by the clip's identity, the ridge is closed-form, the fixed projection is drawn from one seed, the seeded projection from the run's seed, and the test set is the same clips in the same order for every arm and seed.

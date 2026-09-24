# Experiment code

The harness for paper 03, derived from paper 02's harness (`papers/02-untrained-reservoirs/src/harness/experiment/`) and copied rather than imported, so each paper's record is tied to the code that made it. What each tier runs is in [TIERS.md](../TIERS.md), the design in [DESIGN.md](../DESIGN.md).

## Quick start

```bash
cd papers/03-size-and-layout/src
uv sync
uv run pytest                                    # the contract tests; three minutes, no data needed
uv run python -m harness.experiment.plan estimate   # runs, CPU- and GPU-hours and memory per tier and lattice
```

To run anything, the bank and row caches are needed: see [data/README.md](data/README.md).

## What changed from paper 02

Paper 02's harness ran one lattice (16 × 16) and, in its widened tier 4, lattices of 8 × 8 and 32 × 32. Paper 03 carries over only what it runs (no Becker protocol; paper 02's order task, its rotation-rate read and its coherence instruments are carried over) and changes the things below. Every change leaves paper 02's 16 × 16, 4-channel runs bit for bit as they were, which is what lets paper 03 take them from paper 02's record.

| change | where | why it leaves paper 02's cells alone |
|---|---|---|
| **Exact kernel scaling.** Every channel's coupling kernel is scaled so its spectral peak is exactly the coupling ceiling, up as well as down (`kernel_scaling="exact"`). Paper 02 only scaled down (`"cap"`, still available). | `models/phase.py` (`ceiling_factor`), `models/stuart_landau.py` | A random 16 × 16 kernel's peak is 1.42 or more on every geometry, seed and channel either paper draws, so the ceiling binds and the two rules are one computation. `tests/test_kernel_scaling.py` checks the coupling operator bit for bit for every 16 × 16 arm, both ceilings and both implementations. |
| **Lattices to 128 × 128.** FFT coupling from 32 × 32 up; the cube cut into the most nearly square slab where G is not a square (2 × 4, 4 × 8, 8 × 16); log-spaced bands spanning four octaves at every band count (carrier pathway, tonotopic frequencies). | `models/phase.py`, `models/geometries/cube.py`, `stimuli/filterbank.py` | At 16 × 16 the coupling path, the cube (4 × 4) and the bands are paper 02's. |
| **A front end for 64 and 128 bands.** A zero-padded transform (1,024 and 2,048 points) behind paper 02's 512-sample window and frames, so no mel filter is empty. | `stimuli/frontend.py`, `stimuli/audio.py` | Up to 32 bands the front end is paper 02's exactly (`tests/test_frontend.py` compares it with paper 02's formula). |
| **A longer analysis window.** At 64 and 128 bands the own-band lattices also run under a real window of 1,024 or 2,048 samples (`Arm.window`, `-w<N>` in labels), every frame centred where paper 02's is. | `stimuli/frontend.py`, `experiment/protocol.py` (`front_end`, `rows_path`) | Paper 02's 512-sample window is the default (`window=0`) and unchanged. |
| **The streamed read.** An arm with more than 4,096 states is simulated, standardized and projected one channel at a time, one read at a time, recording its instruments on the first pass. | `experiment/stream.py`, `experiment/run.py` | Arms up to 4,096 states are read in memory by paper 02's code. The streamed read is tested against it, reads and instruments alike (`tests/test_stream.py`). |
| **The digit-sequence task.** Two, three or four different digits joined into one clip, as paper 02 joins its order task's pairs, each position named by a 10-way readout of its own on the whole-span read; exact chance levels in `protocol.sequence_chance`. | `experiment/protocol.py` (`sequence_set`, `sequence_clips`), `experiment/readout.py`, `experiment/run.py` | New; paper 02's order clips are rebuilt bit for bit by the same joining code (`tests/test_memory_tasks.py`). |
| **Two projections.** Every projected read is recorded under paper 02's fixed matrix (generator seed 4242) and under one seeded by the run's seed (4242 + 1 + seed), each cell tagged `projection`: `fixed`, `seeded`, or `none` for an unprojected read. | `measurement/features.py`, `experiment/readout.py`, `experiment/stream.py` | The fixed cells are paper 02's arithmetic; its untagged cells are read as fixed (or `none` where unprojected). |
| **Devices.** `--device auto\|cpu\|mps\|cuda`; the untrained arms, the front ends' batches and the projection run on the device, the ridge on the CPU in float64. The trained baselines train on the CPU, as in paper 02, unless `plan run --trained-device` moves them. | `utils/device.py`, `models/phase.py` (`auto_impl`), `experiment/run.py` | The CPU path is paper 02's; a GPU is close but not bit-identical, so the reuse check and any paper 02 cell run here use the CPU. |
| **Trained baselines sized to each network.** One width knob per architecture, set so the parameter count meets 2 · C · G². | `experiment/arms.py` (`trained_width`) | At 2,048 parameters on 16 rows the widths are paper 02's (18, 12, 13, 16, 16), and so are the parameter counts. |

Before the first commit of this harness, paper 02's harness and this one were run side by side on synthetic input: the trajectories and windowed features of all 26 coupling-function and geometry configurations, the uncoupled network and both banks at 16 × 16 (3 seeds, 2 gains, spectrogram and quadrature pathways; 318 cases) had identical SHA-256 hashes, and the five trained baselines trained for two epochs gave identical features. `uv run python -m harness.experiment.gates reuse` repeats the check on the real record.

## Layout

```
src/
├── harness/
│   ├── experiment/      the tiers, one run, the shared read, the summary, the checks, the benchmark
│   ├── models/          the coupled oscillator network (phase and Stuart-Landau cores),
│   │   ├── geometries/  one module per lattice geometry
│   │   └── baselines/   one module per trained baseline
│   ├── stimuli/         the bank's clip loader, the three front ends, the injection path
│   ├── measurement/     the summary statistics and the fixed projection (paper 02's, unchanged)
│   └── utils/           constants and paths
├── scripts/            pod_run.sh (everything on a GPU pod) and the figure scripts
├── tests/               one file per concern
└── data/                the bank and row caches (gitignored)
```

| module | what it owns |
|---|---|
| `experiment/plan.py` | the tiers, the cells taken from paper 02 (`paper02_group`, `paper02_run`), the row caches (`cache_jobs`), the cost model (`seconds`, `memory_gb`), and the parallel driver |
| `experiment/benchmark.py` | `plan benchmark`: times one batch of every lattice and channel count on a device and extrapolates every run and tier |
| `experiment/run.py` | one run: its identity, its clips, its batches, and the record it writes |
| `experiment/arms.py` | the arms; building an untrained arm from its seed; one channel of it; sizing and training a trained baseline |
| `experiment/readout.py` | standardize, project, ridge: paper 02's readout, split so a streamed read can hand it projected features |
| `experiment/stream.py` | the read of an arm too large to hold, one channel at a time |
| `experiment/protocol.py` | the bank, Protocol A, per-clip noise, the order and digit-sequence sets and clips, the front ends, the band mapping (`to_rows`), row caches |
| `experiment/summary.py` | every accuracy, instrument and paired difference with its spread, paper 02's cells folded in, the order task's pairs pooled and the sequence's positions taken together: `summary.json`, `summary.md` |
| `experiment/gates.py` | the integrity checks: reuse (paper 02's cells re-run and compared) and zero input (every cell at chance) |
| `experiment/terms.py` | the paper's names for the record's labels |

## Running the tiers

```bash
uv run python -m harness.experiment.plan benchmark --device cuda --out ../results/benchmark/cuda.json   # price the tiers on this GPU
uv run python -m harness.experiment.plan prepare                         # row caches: every pathway, band count, window, order and sequence set
uv run python -m harness.experiment.gates reuse                          # paper 02's 16 x 16 cells, re-run and compared with its record
uv run python -m harness.experiment.plan run gate --workers 4 --threads 1
uv run python -m harness.experiment.gates check                          # every zero-input cell reads chance
uv run python -m harness.experiment.plan run size trained sequence --grids 8 16 32 --device mps --workers 2
uv run python -m harness.experiment.plan run size trained sequence --grids 64 128 --device auto --workers 2
uv run python -m harness.experiment.summary                              # accuracies and paired differences
```

`plan run <tier> --dry-run` prints what would run, with each run's estimated time on the chosen device (one CPU thread, or the M1 Max's GPU for `mps`) and memory. `--device auto` (the default) takes CUDA if there is one, else Apple Silicon's GPU, else the CPU; the device and GPU each run used is in its record. `--grids` and `--channels` select a stage of a tier without changing any run's identity, so a tier can be run small lattices first and large ones on another machine. Every run is recorded under an identity derived from its specification, so a sweep stopped at any point restarts where it left off; runs that paper 02 recorded are skipped and read from its record. `scripts/pod_run.sh` sets up a GPU pod from the pushed branch, runs the tests and the benchmark, and then any tiers it is given.

## The read at scale

Every arm is read by paper 02's contract: three statistics per signal over four windows of frames 16 to 61 (on the memory tasks, one window over the whole span), with a network's rotation rates as a second read, standardized with the training set's own statistics, projected by a fixed Gaussian matrix to widths 192, 1,024 and 4,096, and classified by the ridge readout. Up to 4,096 states the read is paper 02's code in memory. Above it (from 32 × 32 at 8 channels, 64 × 64 at 2, and every 128 × 128 arm) the features of a run do not fit (a 128 × 128, 16-channel network has 6,291,456 features per clip, 202 GB for a run's 8,048 clips, and paper 02's projection matrix for it would be 103 GB), so `experiment/stream.py` reads it channel by channel, one read at a time. Channels are independent, and standardization is per feature, so the read splits exactly: each channel is simulated alone, its training features kept (3.2 GB at 128 × 128), standardized, and projected into the run's projected features; its test clips are projected batch by batch. Given paper 02's matrix it reproduces the in-memory read's accuracies, penalties and per-clip correctness exactly in the tests. Its own matrices are drawn per channel, 100 times the in-memory generator seed plus the channel (`424200 + c` fixed, `100 * (4243 + seed) + c` seeded), with the same distribution and scale as paper 02's, a channel's two drawn at once in two threads.

## Terms in the code

The paper's glossary (Appendix A of the manuscript) defines the terms. The code and the record use them in short form, exactly as paper 02's do, so a paper 02 cell and the paper 03 cell it stands for share a label and a run identity; `experiment/terms.py` spells them out in full for everything people read. Paper 03's labels add the size where it is not paper 02's, in this order: `-ch<C>` (a channel count other than 4), `-<G>x<G>` (a lattice other than 16 x 16), `-16bands` (paper 02's 16 bands mapped onto the rows) and `-w<N>` (an analysis window other than 512 samples). The model layer (`harness/models`: `OscillatorField`, `damping=`, `spectral_clamp=`, `boundary=`) keeps its own parameter names, which `experiment/arms.py` maps onto.

| paper term | in the code and the record |
|---|---|
| spectrogram-only baseline | arm kind `baseline`, label `baseline[-<G>x<G>][-16bands][-w<N>]`; its whole-clip read is `windowed@wholeclip` |
| coupled oscillator network | arm kind `network`, labels `coupled-<coupling>-<geometry>-<frequencies>-restoring<λ>-ceiling<c>[-ch<C>][-<G>x<G>][-16bands][-w<N>]` |
| uncoupled oscillator network | `network` with `coupled=False`, labels `uncoupled-...` |
| leaky-integrator bank, state-matched | `bank-state` (4 channels), `bank-width` (8), `bank-ch<C>` otherwise, then the lattice suffixes (`LeakyBank`); paper 02's width-matched bank, `bank-width`, is the 8-channel state-matched bank here |
| trained baselines, sized to a network | `trained-<arch>[-ch<C>][-<G>x<G>][-16bands][-w<N>]`: `gru`, `tcn`, `cnn`, `transformer`, `s4d` |
| reservoir | the untrained arms: `network` and `bank` (`build_untrained`, `untrained_signals`, `untrained_features`) |
| readout | the ridge (`readout.py`) |
| coupling function | `coupling`: `kuramoto`, `kuramoto-sakaguchi`, `second-harmonic`, `winfree`, `stuart-landau`, `stuart-landau-fixed` (`arms.CORES` maps them to the model layer's names) |
| lattice geometry | `geometry` |
| lattice, G × G | `grid` |
| channel | `channels` |
| band mapping: one band per row / 16 bands mapped onto the rows | `bands`: `0` / `16`; `-16bands` in labels |
| analysis window: paper 02's 512 samples / a longer one | `window`: `0` / `1024`, `2048`; `-w<N>` in labels |
| recognition / order task / digit-sequence task | `task`: `recognition` / `order` (`pair`) / `sequence` (`length`; a cell's `position`) |
| the primary read / the whole-span read | `windowed` (four windows) / `pooled` (one window) |
| rotation rates added | a read's `+rate` suffix (`windowed+rate`, `pooled+rate`) |
| order parameter R, locking to the drive, share entrained | a run's `instruments` (`network_instruments`): `R`, `plv`, `entrained`, `amplitude` |
| natural frequencies: random / tonotopic / identical | `frequencies`: `random` / `tonotopic` / `identical` |
| restoring strength λ | `restoring`, `restoring<λ>` in labels |
| coupling ceiling | `ceiling`, `ceiling<c>` in labels; exact scaling is `kernel_scaling="exact"` |
| coupling kernel | `kernel` |
| input gain | `gain`, `g` in run ids |
| input pathway: spectrogram / quadrature / carrier | `pathway`: `spectrogram` / `quadrature` / `carrier` |
| mel spectrogram, band energies | the front-end rows (`hop_rows`), cached as `spectrogram-*.pt` |
| a trained baseline's training and read | `train_baseline`, `trained_blocks`, `trained_features`, `trained_width` |
| matched pair | the cells `summary.compare` pairs |
| parameter budget, 2 · C · G² | `Arm.budget` |
| fixed / seeded projection | a cell's `projection`: `fixed` / `seeded` (`none` if unprojected) |
| a record file | `results/<tier>-<task>-<pathway>-<G>x<G>[-<coupling>].json` |

## Determinism and devices

Forward passes are bit-identical from run to run on every geometry (`tests/test_geometries.py`), and every number comes from a forward pass of an untrained arm or a trained baseline's features. The streamed read's sums run in a different order from the in-memory read's, so the two agree to float32 rounding rather than bit for bit; an arm is read one way or the other by its size alone.

Only the CPU is bit-identical to paper 02. On MPS or CUDA the rows, the arms' parameters and the trained baselines' starting weights are the CPU's (all made on the CPU; the trained baselines also train there unless moved), but the GPU's FFTs, matrix products and reductions sum in other orders, so its trajectories differ from the CPU's at the level of rounding (within 2e-3 after the short scans of `tests/test_device.py`) and whole runs by a few test clips. The dense and FFT couplings are chosen per device by speed (`models/phase.py`, `auto_impl`): on MPS the dense operator measured 2 to 80 times faster than FFT up to 64 x 64. The MPS tests skip where MPS is absent.

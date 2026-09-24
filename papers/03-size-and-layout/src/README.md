# Experiment code

The harness for paper 03, derived from paper 02's confirmatory harness (`papers/02-untrained-reservoirs/src/harness/confirm/`) and copied rather than imported, so each paper's record is tied to the code that made it. What each tier runs is in [TIERS.md](../TIERS.md), the design in [REGISTRATION.md](../REGISTRATION.md) (a draft, not frozen).

## Quick start

```bash
cd papers/03-size-and-layout/src
uv sync
uv run pytest                                    # the contract tests; three minutes, no data needed
uv run python -m harness.confirm.plan estimate   # runs, CPU- and GPU-hours and memory per tier and lattice
```

To run anything, the bank and row caches are needed: see [data/README.md](data/README.md).

## What changed from paper 02

Paper 02's harness ran one lattice (16 × 16) and, in its widened tier 4, lattices of 8 × 8 and 32 × 32. Paper 03 carries over only what it needs (no Becker protocol, no order task, no exploratory harness, none of the exploratory geometries or cores) and changes five things. Every change leaves paper 02's 16 × 16, 4-channel runs bit for bit as they were, which is what lets paper 03 take them from paper 02's record.

| change | where | why it leaves paper 02's cells alone |
|---|---|---|
| **Exact kernel scaling.** Every channel's coupling kernel is scaled so its spectral peak is exactly the coupling ceiling, up as well as down (`kernel_scaling="exact"`). Paper 02 only scaled down (`"cap"`, still available). | `models/phase.py` (`ceiling_factor`), `models/stuart_landau.py` | A random 16 × 16 kernel's peak is 1.42 or more on every geometry, seed and channel either paper draws, so the ceiling binds and the two rules are one computation. `tests/test_kernel_scaling.py` checks the coupling operator bit for bit for every 16 × 16 arm, both ceilings and both implementations. |
| **Lattices to 128 × 128.** FFT coupling from 32 × 32 up; the cube cut into the most nearly square slab where G is not a square (2 × 4, 4 × 8, 8 × 16); log-spaced bands spanning four octaves at every band count (carrier pathway, tonotopic frequencies). | `models/phase.py`, `models/geometries/cube.py`, `stimuli/filterbank.py` | At 16 × 16 the coupling path, the cube (4 × 4) and the bands are paper 02's. |
| **A front end for 64 and 128 bands.** A zero-padded transform (1,024 and 2,048 points) behind paper 02's 512-sample window and frames, so no mel filter is empty. | `stimuli/frontend.py`, `stimuli/audio.py` | Up to 32 bands the front end is paper 02's exactly (`tests/test_frontend.py` compares it with paper 02's formula). |
| **The streamed read.** An arm with more than 4,096 states is simulated, standardized and projected one channel at a time. | `confirm/stream.py`, `confirm/run.py` | Arms up to 4,096 states are read in memory by paper 02's code. The streamed read is tested against it (`tests/test_stream.py`). |
| **Two projections.** Every projected read is recorded under paper 02's fixed matrix (generator seed 4242) and under one seeded by the run's seed (4242 + 1 + seed), each cell tagged `projection`: `fixed`, `seeded`, or `none` for an unprojected read. | `measurement/features.py`, `confirm/readout.py`, `confirm/stream.py` | The fixed cells are paper 02's arithmetic; its untagged cells are read as fixed (or `none` where unprojected). |
| **Devices.** `--device auto\|cpu\|mps\|cuda`; models, front ends, the projection and the trained baselines run on the device, the ridge on the CPU in float64. | `utils/device.py`, `models/phase.py` (`auto_impl`), `confirm/run.py` | The CPU path is paper 02's; a GPU is close but not bit-identical, so the reuse gate and any paper 02 cell run here use the CPU. |
| **Trained baselines sized to each network.** One width knob per architecture, set so the parameter count meets 2 · C · G². | `confirm/arms.py` (`ann_width`) | At 2,048 parameters on 16 rows the widths are paper 02's (18, 12, 13, 16, 16), and so are the parameter counts. |

Before the first commit of this harness, paper 02's harness and this one were run side by side on synthetic input: the trajectories and windowed features of all 26 coupling-function and geometry configurations, the uncoupled network and both banks at 16 × 16 (3 seeds, 2 gains, band-energy and quadrature pathways; 318 cases) had identical SHA-256 hashes, and the five trained baselines trained for two epochs gave identical features. `uv run python -m harness.confirm.gates reuse` repeats the check on the real record.

## Layout

```
src/
├── harness/
│   ├── confirm/         the tiers, one run, the shared read, the summary, the gates
│   ├── models/          the coupled oscillator network (phase and Stuart-Landau cores),
│   │   ├── geometries/  one module per lattice geometry
│   │   └── baselines/   one module per trained baseline
│   ├── stimuli/         the bank's clip loader, the three front ends, the injection path
│   ├── measurement/     the summary statistics and the fixed projection (paper 02's, unchanged)
│   └── utils/           constants and paths
├── scripts/pod_run.sh   everything on a GPU pod
├── tests/               one file per concern
└── data/                the bank and row caches (gitignored)
```

| module | what it owns |
|---|---|
| `confirm/plan.py` | the tiers, the cells taken from paper 02 (`paper02_group`), the cost model (`seconds`, `memory_gb`), and the parallel driver |
| `confirm/run.py` | one run: its identity, its clips, its batches, and the record it writes |
| `confirm/arms.py` | the arms; building an untrained arm from its seed; one channel of it; sizing and training a trained baseline |
| `confirm/readout.py` | standardize, project, ridge: paper 02's readout, split so a streamed read can hand it projected features |
| `confirm/stream.py` | the read of an arm too large to hold, one channel at a time |
| `confirm/protocol.py` | the bank, Protocol A, per-clip noise, the front ends, the band mapping (`to_rows`), row caches |
| `confirm/summary.py` | every accuracy and paired difference with its spread, paper 02's cells folded in: `summary.json`, `summary.md` |
| `confirm/gates.py` | the reuse gate and the zero-input gate |
| `confirm/terms.py` | the paper's names for the record's labels |

## Running the tiers

```bash
uv run python -m harness.confirm.plan prepare                         # row caches: every pathway, noise level and band count
uv run python -m harness.confirm.gates reuse                          # paper 02's 16 x 16 cells reproduce under this harness
uv run python -m harness.confirm.plan run gate --workers 4 --threads 1
uv run python -m harness.confirm.gates check                          # every zero-input cell reads chance
uv run python -m harness.confirm.plan run size trained --grids 8 16 32 --device mps --workers 2
uv run python -m harness.confirm.plan run size trained --grids 64 128 --device auto --workers 2
uv run python -m harness.confirm.summary                              # accuracies and paired differences
```

`plan run <tier> --dry-run` prints what would run, with each run's estimated time on the chosen device (one CPU thread, or the M1 Max's GPU for `mps`) and memory. `--device auto` (the default) takes CUDA if there is one, else Apple Silicon's GPU, else the CPU; the device and GPU each run used is in its record. `--grids` and `--channels` select a stage of a tier without changing any run's identity, so a tier can be run small lattices first and large ones on another machine. Every run is recorded under an identity derived from its specification, so a sweep stopped at any point restarts where it left off; runs that paper 02 recorded are skipped and read from its record. `scripts/pod_run.sh` runs any tier on a GPU pod from the pushed branch.

## The read at scale

Every arm is read by paper 02's contract: three statistics per signal over four windows of frames 16 to 61, standardized with the training set's own statistics, projected by a fixed Gaussian matrix to widths 192, 1,024 and 4,096, and classified by the ridge readout. Up to 4,096 states the read is paper 02's code in memory. Above it (from 32 × 32 at 8 channels, 64 × 64 at 2, and every 128 × 128 arm) the features of a run do not fit (a 128 × 128, 16-channel network has 6,291,456 features per clip, 202 GB for a run's 8,048 clips, and paper 02's projection matrix for it would be 103 GB), so `confirm/stream.py` reads it channel by channel. Channels are independent, and standardization is per feature, so the read splits exactly: each channel is simulated alone, its training features kept (3.2 GB at 128 × 128), standardized, and projected into the run's projected features; its test clips are projected batch by batch. Given paper 02's matrix it reproduces the in-memory read's accuracies, penalties and per-clip correctness exactly in the tests. Its own matrices are drawn per channel, 100 times the in-memory generator seed plus the channel (`424200 + c` fixed, `100 * (4243 + seed) + c` seeded), with the same distribution and scale as paper 02's, a channel's two drawn at once in two threads.

## Terms in the code

The paper's glossary (Appendix A of the manuscript) defines the terms. The code and the record keep paper 02's labels, because they are the runs' identities and paper 03 takes cells from paper 02's record under them; `confirm/terms.py` turns them into the paper's terms for everything people read.

| paper term | in the code and the record |
|---|---|
| spectrogram-only baseline | arm kind `floor`; its whole-clip read is `windowed@wholeclip` |
| coupled oscillator network | arm kind `field`, labels `field-<coupling>-<geometry>-<ω>-lam<λ>-clamp<ceiling>[-c<C>][-<G>x<G>][-16bands]` |
| uncoupled oscillator network | `field` with `severed=True`, labels `severed-...` |
| leaky-integrator bank, state-matched | `bank-c<C>[-<G>x<G>][-16bands]` (`LeakyBank`); paper 02's `bank-c8`, its width-matched bank, is the 8-channel state-matched bank here |
| trained baselines, sized to a network | `ann-<arch>[-c<C>][-<G>x<G>][-16bands]`: `gru`, `tcn`, `cnn`, `transformer`, `s4d` |
| reservoir | the untrained arms: `field` and `bank` |
| untrained | "frozen" |
| readout | the ridge (`readout.py`) |
| coupling function | `physics`: `kuramoto`, `sakaguchi`, `harmonic2`, `winfree`, `sl`, `sl-fixedamp` |
| lattice geometry | `boundary` |
| lattice, G × G | `grid` |
| channel | `channels` |
| band mapping: one band per row / 16 bands mapped onto the rows | `bands`: `0` / `16`; `-16bands` in labels |
| natural frequencies: random / tonotopic / identical | `omega`: `random` / `designed` / `uniform` |
| restoring strength λ | `damping`, `lam` in labels |
| coupling ceiling | `clamp`, `spectral_clamp`; exact scaling is `kernel_scaling="exact"` |
| coupling kernel | `kernel` |
| input gain | `gain`, `g` in run ids |
| input pathway: band-energy / quadrature / carrier | `drive`: `envelope` / `quadrature` / `carrier` |
| mel spectrogram, band energies | the front-end rows (`hop_rows`) |
| matched pair | the cells `summary.compare` pairs |
| parameter budget, 2 · C · G² | `Arm.budget` |
| fixed / seeded projection | a cell's `projection`: `fixed` / `seeded` (`none` if unprojected) |

## Determinism and devices

Forward passes are bit-identical from run to run on every geometry (`tests/test_geometries.py`), and every number comes from a forward pass of an untrained arm or a trained baseline's features. The streamed read's sums run in a different order from the in-memory read's, so the two agree to float32 rounding rather than bit for bit; an arm is read one way or the other by its size alone.

Only the CPU is bit-identical to paper 02. On MPS or CUDA the rows, the arms' parameters and the trained baselines' starting weights are the CPU's (all made on the CPU), but the GPU's FFTs, matrix products and reductions sum in other orders, so its trajectories differ from the CPU's at the level of rounding (within 2e-3 after the short scans of `tests/test_device.py`) and whole runs by a few test clips. The dense and FFT couplings are chosen per device by speed (`models/phase.py`, `auto_impl`): on MPS the dense operator measured 2 to 80 times faster than FFT up to 64 x 64. The MPS tests skip where MPS is absent.

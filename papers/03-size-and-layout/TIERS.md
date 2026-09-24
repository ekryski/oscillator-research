# The tiers

What each tier of paper 03's run tests, what it costs, and what it takes from paper 02. The design and the reason for each choice are in [DESIGN.md](DESIGN.md); the tiers as code are in [`src/harness/confirm/plan.py`](src/harness/confirm/plan.py), which is authoritative where the two could differ. Every run is one arm at one lattice, channel count, band mapping, analysis window, input pathway, task and seed, at 0 dB and input gain 1 (32 on the carrier pathway), fitted on 2,048 training clips and read at widths 192, 1,024 and 4,096 under two projections. The terms are defined in the paper's glossary (Appendix A); the code and the record keep paper 02's labels, mapped in [the harness README](src/README.md#terms-in-the-code).

| tier | question | arms | varied | fixed | runs |
|---|---|---|---|---|---|
| gate | Does the pipeline leak at every size, in memory and streamed? | the coupled and uncoupled oscillator networks and the state-matched leaky-integrator bank, with no input | lattice 8 × 8 to 128 × 128; 1 and 16 channels | input gain 0; 0 dB; seed 0 | 30 |
| size | How do the number of oscillators and their layout move accuracy and memory, against controls of the same size? | coupled oscillator network, uncoupled oscillator network, state-matched leaky-integrator bank; the spectrogram-only baseline on each lattice's rows | lattice 8 × 8, 16 × 16, 32 × 32, 64 × 64, 128 × 128; 1, 2, 4, 8 and 16 channels (64 to 262,144 oscillators); band mapping: one mel band per row, or paper 02's 16 bands mapped onto the rows; on recognition, at 64 and 128 bands, the analysis window (paper 02's 512 samples, or 1,024 and 2,048); task: recognition, and paper 02's order task on each of its five pairs; seeds 0–2 | Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1 met exactly; band-energy pathway; 0 dB; input gain 1 | 2,688 |
| trained | How do trained baselines of the same parameter count compare as both grow, and are oscillators more useful at the smallest sizes? | GRU, TCN, CNN, transformer and S4D, each sized to every recognition network of the size tier (2 · C · G² parameters, 128 to 524,288) and driven by that network's rows | as the size tier, on recognition | trained end to end as in paper 02; 0 dB | 825 |
| sequence | How much of a spoken sequence, and of its order, does a network of each size hold? | as the size tier | every lattice, channel count and band mapping; sequences of 2, 3 and 4 different digits, each position named by its own readout; seeds 0–2 | as the size tier; the whole-span read | 1,296 |
| design | Do coupling function and lattice geometry matter differently at different sizes? | coupled oscillator network only | coupling function (6) × lattice geometry (6), the Stuart–Landau functions on the torus only (25 configurations besides the reference); every lattice, channel count and band mapping; seeds 0–2 | as the size tier; recognition | 3,375 |
| quadrature | Does the quadrature pathway's standing change with size? | coupled and uncoupled oscillator networks; the pathway's own spectrogram-only baseline | as the size tier on recognition, windows included | quadrature pathway; 0 dB; input gain 1; the bank has no phase to take it | 363 |
| carrier | Does the carrier pathway's standing change with size? | coupled and uncoupled oscillator networks, state-matched bank at the sample rate; the pathway's own spectrogram-only baseline | every lattice, channel count and band mapping; seeds 0–2 | carrier pathway at 16 kHz; 0 dB; input gain 32 | 432 |
| design-quadrature | Design at size, on the quadrature pathway | coupled oscillator network only | the 4 phase coupling functions × 6 geometries (23 besides the reference); every lattice, channel count and band mapping | as the quadrature tier | 3,105 |
| design-carrier | Design at size, on the carrier pathway | coupled oscillator network only | the 25 configurations of the design tier; as the carrier tier | as the carrier tier | 3,375 |

All nine tiers together are 15,489 runs, 225 of them paper 02's. Every run is at 0 dB and input gain 1 (32 on the carrier pathway, paper 02's calibrated carrier gain), one of paper 02's two noise levels and one of its two gains.

**Arms and sizes.** A network of C channels on a G × G lattice has C · G² oscillators and 2 · C · G² parameters (its coupling kernels and natural frequencies). Channels are parallel copies, like the heads of one attention layer: each has its own kernel and natural frequencies, all receive the same input, and none acts on another. The state-matched leaky-integrator bank has the network's states and parameters at every size; paper 02's width-matched bank (twice the units) is the state-matched bank of twice the channels, which the size tier runs anyway up to 8 channels. Each trained baseline has one width knob, set so its parameter count is the widest at or under the network's (the CNN, the TCN's form, the narrowest at or over it): paper 02's widths at 2,048 parameters, and within 15% of the network's count from 2,048 parameters up. Below 2,048 some baselines cannot be sized closely; those cells are kept and flagged, since they are where the trained tier asks whether oscillators are more useful at small sizes. Input gain applies to the reservoirs only, as in paper 02.

**The lattices and the front end.** Every lattice other than 16 × 16 runs twice: driven by its own number of mel bands, one per row, and by paper 02's 16 bands mapped onto its rows (each band on G/16 adjacent rows above 16 × 16, each row the mean of two bands at 8 × 8). At 64 and 128 bands the front end's transform is zero-padded to 1,024 and 2,048 points behind paper 02's 512-sample window, so no mel filter is empty and the frames are paper 02's. Zero-padding adds no frequency resolution, so those own-band lattices also run, on recognition in the size, trained and quadrature tiers, under a real window as long as the transform, 1,024 or 2,048 samples, with every frame centred where paper 02's is: the frame count and the read are unchanged, and each frame blends 64 or 128 ms of audio instead of 32.

**The tasks.** Recognition (chance 10%) in every tier. Paper 02's order task (which of two digits came first, five fixed pairs, 2,048 training and 2,048 test clips per pair; chance 50%) in the size tier. The digit-sequence task (two, three or four different digits joined into one clip, each position named by a readout of its own; 2,048 training and 2,048 test clips per length) in the sequence tier: chance at each position is 10%, a reader that knows which digits a sequence holds but not their order reaches 50%, 33.3% and 25%, and every position right by chance is 1.11%, 0.139% and 0.0198% (DESIGN.md, section 3). Both memory tasks are read over the whole span in one window, which leaves a read of the spectrogram alone blind to order, and neither is run by the trained baselines, as in paper 02.

**The read.** Paper 02's read: the mean, standard deviation and mean absolute frame-to-frame change of each signal over four equal windows of frames 16 to 61 (recognition) or over the whole span from frame 16 (the memory tasks), standardized, projected by a Gaussian matrix to widths 192, 1,024 and 4,096 (never wider than the arm; the unprojected read also fitted where an arm has 1,024 states or fewer), and classified by the ridge readout, one per position on the digit-sequence task. Every coupled and uncoupled network also records the same read with its rotation rates added (paper 02's secondary read), and every spectrogram-only baseline its whole-clip read. Every projected read is recorded twice, under the fixed projection (paper 02's matrix, generator seed 4242, the same for every seed) and under a projection seeded by the run's seed (4242 + 1 + seed), and each cell is tagged with its projection. The primary cell is width 192 and 2,048 training clips, so the readout has 1,930 weights at every size: the network grows and the readout does not. Arms above 4,096 states are read channel by channel ([`stream.py`](src/harness/confirm/stream.py)), one read at a time, with the same result as the in-memory read on the same matrices.

**Instruments.** Every coupled and uncoupled network run records, over its test clips, its order parameter R, each oscillator's phase-locking value to its own band's drive, the share of oscillators locked above 0.5, and its mean amplitude, in memory or streamed and on every device. Nothing reads them, and no cell changes with them; the summary reports them beside each network's accuracy.

**Devices.** Every run takes `--device auto|cpu|mps|cuda` (auto: CUDA, else Apple Silicon's GPU, else the CPU) and records the device it used. Only the CPU is bit-identical to paper 02; a GPU's results are close, not identical. Paper 02's cells are reused as its CPUs recorded them, the reuse gate runs on the CPU, and a paper 02 cell that paper 03 must run itself runs on the CPU.

**Reporting.** As paper 02 reports: every accuracy as its mean over three seeds with the sample standard deviation and each seed's value; every comparison paired on task and set, lattice, band mapping, window, channel count, noise level, input gain, seed and projection, as the mean difference with its standard deviation over seeds and a 95% interval from resampling test clips; both projections side by side. No threshold is applied to either.

## Cost

Estimated with the cost model in `plan.py` (`uv run python -m harness.confirm.plan estimate`), per run, on one thread of the CPU and on the GPU of this project's Mac, an Apple M1 Max with 64 GB of unified memory. Both include what runs on the CPU either way: drawing the two projection matrices (48 million Gaussian draws a second per thread; a streamed arm draws a channel's two matrices in two threads, 1.65 times faster) and the ridge readout under each projection (about 10 s per read and projection).

What was measured, per channel and clip, for the reference network (Kuramoto, torus, band-energy pathway), with the coupling implementation each device uses:

| | 8 × 8 | 16 × 16 | 32 × 32 | 64 × 64 | 128 × 128 |
|---|---|---|---|---|---|
| CPU, one thread: simulate 61 frames (dense to 16 × 16, FFT above) | 0.29 ms | 0.97 ms | 2.1 ms | 7.3 ms | 38.5 ms |
| CPU: windowed statistics | 0.3 ms | 0.5 ms | 2.4 ms | 7.5 ms | 27 ms |
| MPS: simulate (dense to 64 × 64, FFT at 128) | 0.031 ms | 0.030 ms | 0.14 ms | 0.90 ms | 5.5 ms |
| MPS: FFT instead of dense, for comparison | 0.71 ms | 1.72 ms | 1.36 ms | 1.93 ms | 5.5 ms |
| MPS: windowed statistics | | 0.034 ms | | 0.57 ms | 2.05 ms |
| MPS: Stuart–Landau core (dense to 64 × 64) | | 0.063 ms | | 1.51 ms | 5.04 ms |
| MPS: leaky-integrator bank | | 0.014 ms | | 0.21 ms | 0.51 ms |

The CPU figures were taken on 2026-09-23 and the MPS ones on 2026-09-24, both while paper 02's runs held the CPU's other cores; the MPS ones in batches of 512 clips (8 and 16), 256 (32), 64 (64) and 32 (128), after warm-up. Paper 02's tier 2 measured 16 × 16 at 4 channels at about 5 ms a clip on the CPU, which the CPU figures reproduce. Also measured: the projection's matrix products (150 GFLOP/s on one CPU thread, 1.5 to 3 TFLOP/s on MPS); the CPU cost of each geometry and coupling function at 64 × 64 (0.74 to 2.6 times the reference) and on MPS with FFT (0.21 to 2.4 times); and the five trained baselines' training steps at 2,048, 65,536 and 524,288 parameters on both (0.2 to 154 minutes a run on the CPU, 0.2 to 7 on MPS). Extrapolated, not measured: the MPS statistics, bank and Stuart–Landau figures at 8 × 8 and 32 × 32 (interpolated), the dense path's independence of the geometry and the second harmonic's doubled cost on MPS, the quadrature pathway's 1.3 times the band-energy cost on MPS (measured on the CPU), the carrier pathway's 262 times (16,000 steps a clip instead of 61) on both, and a C-channel run as C single channels. CUDA was not measured, and the cost model does not estimate it: `uv run python -m harness.confirm.plan benchmark --device cuda` times one batch of every lattice and channel count on the band-energy and carrier pathways, every coupling function and geometry at one channel, the projection, the matrix draws, the ridge and the carrier's front end on the machine it runs on, extrapolates every run and tier the same way, and writes a JSON report (`results/benchmark/`); [`src/scripts/pod_run.sh`](src/scripts/pod_run.sh) sets up a pod, runs the tests and then the benchmark before any tier. Two whole runs on the real bank check the MPS model: paper 02's network (16 × 16, 4 channels, read in memory) took 17 s against an estimated 28, and a 32 × 32, 16-channel network (read channel by channel) took 147 s against an estimated 131. The memory tasks scale these by their clips and frames (4,096 clips of 147 to 284 frames against recognition's 8,048 of 61), a streamed run by its reads (it simulates its channels once per read), and the digit-sequence task's ridge by its positions.

| tier | runs | from paper 02 | CPU-hours, 8–32 | CPU-hours, 64 and 128 | MPS-hours, 8–32 | MPS-hours, 64 and 128 | longest run, CPU | longest run, MPS | peak memory |
|---|---|---|---|---|---|---|---|---|---|
| gate | 30 | 0 | 1.8 | 24 | 0.4 | 3.7 | 8.1 h | 1.3 h | 23 GB |
| size | 2,688 | 90 | 82 | 1,272 | 19 | 184 | 8.1 h | 1.3 h | 23 GB |
| trained | 825 | 15 | 6.9 | 76 | 4.9 | 8.6 | 2.6 h | 0.1 h | 1 GB |
| sequence | 1,296 | 0 | 67 | 761 | 24 | 108 | 10.2 h | 1.4 h | 7 GB |
| design | 3,375 | 75 | 265 | 4,375 | 45 | 495 | 22.5 h | 1.9 h | 23 GB |
| quadrature | 363 | 6 | 17 | 382 | 3.8 | 61 | 8.9 h | 1.4 h | 23 GB |
| carrier | 432 | 9 | 2,103 | 33,699 | 132 | 3,794 | 1,063 h | 130 h | 24 GB |
| design-quadrature | 3,105 | 12 | 286 | 4,818 | 43 | 501 | 27.6 h | 2.2 h | 23 GB |
| design-carrier | 3,375 | 18 | 42,048 | 762,634 | 1,713 | 51,148 | 4,849 h | 297 h | 24 GB |
| **all** | **15,489** | **225** | **44,877** | **808,042** | **1,985** | **56,303** | | | |

Off the carrier pathway the plan is about 1,500 hours on the M1 Max's GPU, against about 12,400 on one CPU thread; the order task is 137 of the size tier's 203 GPU-hours, and the longer window 44 across the size, trained and quadrature tiers. The carrier tiers' figures are extrapolated (16,000 steps a clip against 61) and dominate everything, 56,800 GPU-hours, 97% of it at 64 and 128; the benchmark replaces them before any is scheduled. A first probe with it on the M1 Max (2026-09-24, one batch of 32 clips each, with the instruments and both reads) measured the coupled network on the carrier at 63 ms a clip at 8 × 8 with one channel, 162 ms at 16 × 16 with four and 146 ms at 32 × 32 with one, 2 to 6 times the extrapolation, so the carrier's GPU-hours above are likely low by about that factor; those batches held about 35 GB of the GPU's memory at 1,024 states, more than a 24 GB GPU has, and the benchmark halves any batch that does not fit and says so. Memory is never the limit: an arm read in memory holds at most about 8 GB (4,096 states: its features for 8,048 clips and its two projection matrices), and a streamed arm one channel's training features and its rows of both matrices for one read, about 23 GB at 128 × 128 for the read with rotation rates. At 128 × 128 drawing the projection matrices on the CPU is about 30% of a GPU run's time, which drawing more channels at once would cut.

## Stages

The tiers are ordered so each stage answers a question on its own. `--grids` and `--channels` select a stage without changing any run's identity.

| stage | runs | runs here | CPU-hours | MPS-hours |
|---|---|---|---|---|
| A. gate, size, trained and sequence on lattices 8 to 32 | 2,553 | 2,448 | 158 | 49 |
| B. gate, size, trained and sequence on 64 and 128 | 2,286 | 2,286 | 2,133 | 304 |
| C. design and quadrature on lattices 8 to 32 | 2,040 | 1,959 | 282 | 49 |
| D. design and quadrature on 64 and 128 | 1,698 | 1,698 | 4,757 | 556 |
| E. carrier (extrapolated; 8 to 32 alone: 231 runs, 2,103 CPU-hours, 132 MPS-hours) | 432 | 423 | 35,801 | 3,926 |
| F. design-quadrature | 3,105 | 3,093 | 5,104 | 544 |
| G. design-carrier (extrapolated; 8 to 32 alone: 1,857 runs, 42,048 CPU-hours, 1,713 MPS-hours) | 3,375 | 3,357 | 804,682 | 52,861 |

Stages A to D and F are about 1,500 hours on the M1 Max's GPU. Options, for the author to choose from (runs to make here, CPU-hours, MPS-hours, same model):

- The design tier at one band mapping: 1,800 runs, 2,333 CPU-hours, 273 MPS-hours.
- The design-quadrature tier at one band mapping: 1,713 runs, 2,568 CPU-hours, 275 MPS-hours.
- The design-carrier tier at one band mapping: 1,857 runs, 404,443 CPU-hours, 26,489 MPS-hours (extrapolated).
- The sequence tier at one band mapping: 720 runs, 419 CPU-hours, 68 MPS-hours.
- Channel counts 1, 4 and 16 only, in every tier: 9,222 runs, 577,066 CPU-hours, 39,432 MPS-hours.
- Added: the order task in the design tier, 16,875 runs, 20,049 CPU-hours, 1,709 MPS-hours; at 4 channels only, 3,375 runs, 2,571 CPU-hours, 233 MPS-hours.
- Added: the longer window in the design tier, 750 runs, 2,187 CPU-hours, 247 MPS-hours; in design-quadrature, 690 runs, 2,409 CPU-hours, 250 MPS-hours.

## The record

Each tier writes `results/confirmatory/<tier>[-<task>]-<pathway>-<G>x<G>.json` (the task named unless it is recognition), the design tiers one file per coupling function as well. Each run's entry holds its spec, the arm's states, parameters and parameter budget (and a trained baseline's width and whether it is within 15% of its budget), how it was read (in memory or streamed by channel), one cell per read, width and projection (and, on the digit-sequence task, position) with its accuracy, chosen penalty and, at the primary cell, each test clip's correctness for the bootstrap, a network's instruments (mean, spread and mean per class over the test clips), and timing and the environment, including the device and GPU it ran on. The cells taken from paper 02 stay in paper 02's record under their own run identities; `uv run python -m harness.confirm.summary` reads both and writes every accuracy, instrument and comparison to `results/confirmatory/summary.json`, each marked with its source, with the order task's pairs pooled and the digit-sequence task's positions taken together as derived cells, and the primary cells as tables to `summary.md`.

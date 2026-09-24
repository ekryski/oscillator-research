# The tiers

**DRAFT.** What each tier of paper 03's run tests, what it costs, and what it takes from paper 02. The design and the reason for each choice are in [REGISTRATION.md](REGISTRATION.md), which is not frozen; the tiers as code are in [`src/harness/confirm/plan.py`](src/harness/confirm/plan.py), which is authoritative where the two could differ. Every run is one arm at one lattice, channel count, band mapping, input pathway and seed, at 0 dB and input gain 1 (32 on the carrier pathway), fitted on 2,048 training clips and read at widths 192, 1,024 and 4,096 under two projections. The terms are defined in the paper's glossary (Appendix A); the code and the record keep paper 02's labels, mapped in [the harness README](src/README.md#terms-in-the-code).

| tier | question | arms | varied | fixed | runs |
|---|---|---|---|---|---|
| gate | Does the pipeline leak at every size, in memory and streamed? | the coupled and uncoupled oscillator networks and the state-matched leaky-integrator bank, with no input | lattice 8 × 8 to 128 × 128; 1 and 16 channels | input gain 0; 0 dB; seed 0 | 30 |
| size | How do the number of oscillators and their layout move accuracy, against controls of the same size? | coupled oscillator network, uncoupled oscillator network, state-matched leaky-integrator bank; the spectrogram-only baseline on each lattice's rows | lattice 8 × 8, 16 × 16, 32 × 32, 64 × 64, 128 × 128; 1, 2, 4, 8 and 16 channels (64 to 262,144 oscillators); band mapping: one mel band per row, or paper 02's 16 bands mapped onto the rows; seeds 0–2 | Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1 met exactly; band-energy pathway; 0 dB; input gain 1 | 432 |
| trained | How do trained baselines of the same parameter count compare as both grow, and are oscillators more useful at the smallest sizes? | GRU, TCN, CNN, transformer and S4D, each sized to every network of the size tier (2 · C · G² parameters, 128 to 524,288) and driven by that network's rows | as the size tier | trained end to end as in paper 02; 0 dB | 675 |
| design | Do coupling function and lattice geometry matter differently at different sizes? | coupled oscillator network only | coupling function (6) × lattice geometry (6), the Stuart–Landau functions on the torus only (25 configurations besides the reference); every lattice, channel count and band mapping; seeds 0–2 | as the size tier | 3,375 |
| quadrature | Does the quadrature pathway's standing change with size? | coupled and uncoupled oscillator networks; the pathway's own spectrogram-only baseline | as the size tier | quadrature pathway; 0 dB; input gain 1; the bank has no phase to take it | 297 |
| carrier | Does the carrier pathway's standing change with size? | coupled and uncoupled oscillator networks, state-matched bank at the sample rate; the pathway's own spectrogram-only baseline | lattices 8 × 8 to 32 × 32; every channel count and band mapping; seeds 0–2 | carrier pathway at 16 kHz; 0 dB; input gain 32 | 240 |
| design-quadrature | Design at size, on the quadrature pathway | coupled oscillator network only | the 4 phase coupling functions × 6 geometries (23 besides the reference); as the quadrature tier | as the quadrature tier | 3,105 |
| design-carrier | Design at size, on the carrier pathway | coupled oscillator network only | the 25 configurations of the design tier; as the carrier tier | as the carrier tier | 1,875 |

All eight tiers together are 10,029 runs, 150 of them paper 02's. Every run is at 0 dB and input gain 1 (32 on the carrier pathway, paper 02's calibrated carrier gain): the author slimmed the design from paper 02's two noise levels and two gains on 2026-09-24.

**Arms and sizes.** A network of C channels on a G × G lattice has C · G² oscillators and 2 · C · G² parameters (its coupling kernels and natural frequencies). Channels are parallel copies, like the heads of one attention layer: each has its own kernel and natural frequencies, all receive the same input, and none acts on another. The state-matched leaky-integrator bank has the network's states and parameters at every size; paper 02's width-matched bank (twice the units) is the state-matched bank of twice the channels, which the size tier runs anyway up to 8 channels. Each trained baseline has one width knob, set so its parameter count is the widest at or under the network's (the CNN, the TCN's form, the narrowest at or over it): paper 02's widths at 2,048 parameters, and within 15% of the network's count from 2,048 parameters up. Below 2,048 some baselines cannot be sized closely; those cells are kept and flagged, since they are where the trained tier asks whether oscillators are more useful at small sizes. Input gain applies to the reservoirs only, as in paper 02.

**The lattices.** Every lattice other than 16 × 16 runs twice: driven by its own number of mel bands, one per row, and by paper 02's 16 bands mapped onto its rows (each band on G/16 adjacent rows above 16 × 16, each row the mean of two bands at 8 × 8). At 64 and 128 bands the front end's transform is zero-padded to 1,024 and 2,048 points behind paper 02's 512-sample window, so no mel filter is empty and the frames are paper 02's (an open decision, left as built). The carrier pathway above 32 × 32 is not planned (also open).

**The read.** Paper 02's read: the mean, standard deviation and mean absolute frame-to-frame change of each signal over four equal windows of frames 16 to 61, standardized, projected by a Gaussian matrix to widths 192, 1,024 and 4,096 (never wider than the arm; the unprojected read also fitted where an arm has 1,024 states or fewer), and classified by the ridge readout. Every projected read is recorded twice, under the fixed projection (paper 02's matrix, generator seed 4242, the same for every seed) and under a projection seeded by the run's seed (4242 + 1 + seed), and each cell is tagged with its projection. The primary cell is width 192 and 2,048 training clips, so the readout has 1,930 weights at every size: the network grows and the readout does not. Arms above 4,096 states are read channel by channel ([`stream.py`](src/harness/confirm/stream.py)), with the same result as the in-memory read on the same matrices.

**Devices.** Every run takes `--device auto|cpu|mps|cuda` (auto: CUDA, else Apple Silicon's GPU, else the CPU) and records the device it used. Only the CPU is bit-identical to paper 02; a GPU's results are close, not identical. Paper 02's cells are reused as its CPUs recorded them, the reuse gate runs on the CPU, and a paper 02 cell that paper 03 must run itself runs on the CPU.

**Reporting.** As paper 02 reports: every accuracy as its mean over three seeds with the sample standard deviation and each seed's value; every comparison paired on lattice, band mapping, channel count, noise level, input gain, seed and projection, as the mean difference with its standard deviation over seeds and a 95% interval from resampling test clips; both projections side by side. No threshold is applied to either.

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

The CPU figures were taken on 2026-09-23 and the MPS ones on 2026-09-24, both while paper 02's runs held the CPU's other cores; the MPS ones in batches of 512 clips (8 and 16), 256 (32), 64 (64) and 32 (128), after warm-up. Paper 02's tier 2 measured 16 × 16 at 4 channels at about 5 ms a clip on the CPU, which the CPU figures reproduce. Also measured: the projection's matrix products (150 GFLOP/s on one CPU thread, 1.5 to 3 TFLOP/s on MPS); the CPU cost of each geometry and coupling function at 64 × 64 (0.74 to 2.6 times the reference) and on MPS with FFT (0.21 to 2.4 times); and the five trained baselines' training steps at 2,048, 65,536 and 524,288 parameters on both (0.2 to 154 minutes a run on the CPU, 0.2 to 7 on MPS). Extrapolated, not measured: the MPS statistics, bank and Stuart–Landau figures at 8 × 8 and 32 × 32 (interpolated), the dense path's independence of the geometry and the second harmonic's doubled cost on MPS, the quadrature pathway's 1.3 times the band-energy cost on MPS (measured on the CPU), the carrier pathway's 262 times (16,000 steps a clip instead of 61) on both, and a C-channel run as C single channels. CUDA was not measured, and nothing here estimates it. Two whole runs on the real bank check the MPS model: paper 02's network (16 × 16, 4 channels, read in memory) took 17 s against an estimated 28, and a 32 × 32, 16-channel network (read channel by channel) took 147 s against an estimated 131.

| tier | runs | from paper 02 | CPU-hours, 8–32 | CPU-hours, 64 and 128 | MPS-hours, 8–32 | MPS-hours, 64 and 128 | longest run, CPU | longest run, MPS | peak memory |
|---|---|---|---|---|---|---|---|---|---|
| gate | 30 | 0 | 1.1 | 13 | 0.2 | 1.9 | 4.1 h | 0.6 h | 18 GB |
| size | 432 | 15 | 11 | 144 | 2.5 | 22 | 4.1 h | 0.6 h | 18 GB |
| trained | 675 | 15 | 6.9 | 51 | 4.9 | 5.8 | 2.6 h | 0.1 h | 1 GB |
| design | 3,375 | 75 | 152 | 2,207 | 22 | 236 | 11.3 h | 0.9 h | 18 GB |
| quadrature | 297 | 6 | 9.5 | 129 | 1.9 | 20 | 4.5 h | 0.7 h | 18 GB |
| carrier | 240 | 9 | 1,532 | not planned | 94 | not planned | 42 h | 2.7 h | 8 GB |
| design-quadrature | 3,105 | 12 | 167 | 2,428 | 21 | 240 | 13.9 h | 1.1 h | 18 GB |
| design-carrier | 1,875 | 18 | 28,022 | not planned | 1,147 | not planned | 146 h | 4.3 h | 8 GB |
| **all** | **10,029** | **150** | **29,901** | **4,971** | **1,294** | **526** | | | |

On the M1 Max's GPU the whole plan is about 1,800 hours, against about 34,900 on one CPU thread. Memory is never the limit: an arm read in memory holds at most about 8 GB (4,096 states: its features for 8,048 clips and its two projection matrices), and a streamed arm one channel's training features and its rows of both matrices, about 18 GB at 128 × 128, within the Mac's 64 GB. Time is: most of it is the design tiers on the carrier pathway (16,000 steps a clip), then the design tiers at 64 × 64 and 128 × 128. At 128 × 128 drawing the projection matrices on the CPU is about 30% of a GPU run's time, which drawing more channels at once would cut.

## Stages

The tiers are ordered so each stage answers a question on its own. `--grids` and `--channels` select a stage without changing any run's identity.

| stage | runs | runs here | CPU-hours | MPS-hours |
|---|---|---|---|---|
| A. gate, size and trained on lattices 8 to 32 | 633 | 603 | 19 | 7.6 |
| B. gate, size and trained on 64 and 128 | 504 | 504 | 208 | 29 |
| C. design and quadrature on lattices 8 to 32 | 2,040 | 1,959 | 162 | 24 |
| D. design and quadrature on 64 and 128 | 1,632 | 1,632 | 2,336 | 256 |
| E. carrier on lattices 8 to 32 | 240 | 231 | 1,532 | 94 |
| F. design on the quadrature and carrier pathways | 4,980 | 4,950 | 30,617 | 1,409 |

Stages A to E are about 410 hours on the M1 Max's GPU; stage F alone is 1,409. Ways to slim further, for the author to choose from (runs to make here, CPU-hours, MPS-hours, same model):

- The design tier at one band mapping: 1,800 runs, 1,189 CPU-hours, 131 MPS-hours.
- The design-quadrature tier at one band mapping: 1,713 runs, 1,309 CPU-hours, 132 MPS-hours.
- The design-carrier tier at one band mapping: 1,107 runs, 16,000 CPU-hours, 624 MPS-hours.
- Channel counts 1, 4 and 16 only, in every tier: 5,910 runs, 23,591 CPU-hours, 1,228 MPS-hours.

## The record

Each tier writes `results/confirmatory/<tier>-<pathway>-<G>x<G>.json`, the design tiers one file per coupling function as well. Each run's entry holds its spec, the arm's states, parameters and parameter budget (and a trained baseline's width and whether it is within 15% of its budget), how it was read (in memory or streamed by channel), one cell per width and projection with its accuracy, chosen penalty and, at the primary cell, each test clip's correctness for the bootstrap, and timing and the environment, including the device and GPU it ran on. The cells taken from paper 02 stay in paper 02's record under their own run identities; `uv run python -m harness.confirm.summary` reads both and writes every accuracy and comparison to `results/confirmatory/summary.json`, each marked with its source, and the primary cells as tables to `summary.md`.

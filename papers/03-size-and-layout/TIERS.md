# The tiers

**DRAFT.** What each tier of paper 03's run tests, what it costs, and what it takes from paper 02. The design and the reason for each choice are in [REGISTRATION.md](REGISTRATION.md), which is not frozen; the tiers as code are in [`src/harness/confirm/plan.py`](src/harness/confirm/plan.py), which is authoritative where the two could differ. Every run is one arm at one lattice, channel count, band mapping, input pathway, noise level, input gain and seed, fitted on 2,048 training clips and read at widths 192, 1,024 and 4,096. The terms are defined in the paper's glossary (Appendix A); the code and the record keep paper 02's labels, mapped in [the harness README](src/README.md#terms-in-the-code).

| tier | question | arms | varied | fixed | runs |
|---|---|---|---|---|---|
| gate | Does the pipeline leak at every size, in memory and streamed? | the coupled and uncoupled oscillator networks and the state-matched leaky-integrator bank, with no input | lattice 8 × 8 to 128 × 128; 1 and 16 channels | input gain 0; 0 dB; seed 0 | 30 |
| size | How do the number of oscillators and their layout move accuracy, against controls of the same size? | coupled oscillator network, uncoupled oscillator network, state-matched leaky-integrator bank; the spectrogram-only baseline on each lattice's rows | lattice 8 × 8, 16 × 16, 32 × 32, 64 × 64, 128 × 128; 1, 2, 4, 8 and 16 channels (64 to 262,144 oscillators); band mapping: one mel band per row, or paper 02's 16 bands mapped onto the rows; noise 0 and +5 dB; input gain 1 and 2; seeds 0–2 | Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1 met exactly; band-energy pathway | 1,674 |
| trained | How do trained baselines of the same parameter count compare as both grow? | GRU, TCN, CNN, transformer and S4D, each sized to every network of the size tier (2 · C · G² parameters, 128 to 524,288) and driven by that network's rows | as the size tier, without input gain | trained end to end as in paper 02 | 1,350 |
| design | Do coupling function and lattice geometry matter differently at different sizes? | coupled oscillator network only | coupling function (6) × lattice geometry (6), the Stuart–Landau functions on the torus only (25 configurations besides the reference); every lattice, channel count and band mapping; noise 0 and +5 dB; input gain 1 and 2; seeds 0–2 | as the size tier | 13,500 |
| quadrature | Does the quadrature pathway's standing change with size? | coupled and uncoupled oscillator networks; the pathway's own spectrogram-only baseline | as the size tier | quadrature pathway; the bank has no phase to take it | 1,134 |
| carrier | Does the carrier pathway's standing change with size? | coupled and uncoupled oscillator networks, state-matched bank at the sample rate; the pathway's own spectrogram-only baseline | lattices 8 × 8 to 32 × 32; every channel count and band mapping; seeds 0–2 | carrier pathway at 16 kHz; 0 dB; input gain 32 | 240 |
| design-quadrature | Design at size, on the quadrature pathway | coupled oscillator network only | the 4 phase coupling functions × 6 geometries (23 besides the reference); as the quadrature tier | as the quadrature tier | 12,420 |
| design-carrier | Design at size, on the carrier pathway | coupled oscillator network only | the 25 configurations of the design tier; as the carrier tier | as the carrier tier | 1,875 |

All eight tiers together are 32,223 runs, 477 of them paper 02's.

**Arms and sizes.** A network of C channels on a G × G lattice has C · G² oscillators and 2 · C · G² parameters (its coupling kernels and natural frequencies). Channels are parallel copies, like the heads of one attention layer: each has its own kernel and natural frequencies, all receive the same input, and none acts on another. The state-matched leaky-integrator bank has the network's states and parameters at every size; its width-matched bank of paper 02 (twice the units) is the state-matched bank of twice the channels, which the size tier runs anyway up to 8 channels. Each trained baseline has one width knob, set so its parameter count is the widest at or under the network's (the CNN, the TCN's form, the narrowest at or over it): paper 02's widths at 2,048 parameters, and within 15% of the network's count from 2,048 parameters up. Input gain applies to the reservoirs only, as in paper 02.

**The lattices.** Every lattice other than 16 × 16 runs twice: driven by its own number of mel bands, one per row, and by paper 02's 16 bands mapped onto its rows (each band on G/16 adjacent rows above 16 × 16, each row the mean of two bands at 8 × 8). At 64 and 128 bands the front end's transform is zero-padded to 1,024 and 2,048 points behind paper 02's 512-sample window, so no mel filter is empty and the frames are paper 02's. The carrier pathway above 32 × 32 is not planned: it integrates 16,000 steps a clip, and a 64 × 64 run would take days.

**The read.** Paper 02's read: the mean, standard deviation and mean absolute frame-to-frame change of each signal over four equal windows of frames 16 to 61, standardized, projected by a fixed Gaussian matrix to widths 192, 1,024 and 4,096 (never wider than the arm; the unprojected read also fitted where an arm has 1,024 states or fewer), and classified by the ridge readout. The primary cell is width 192 and 2,048 training clips, so the readout has 1,930 weights at every size: the network grows and the readout does not. Arms above 4,096 states are read channel by channel ([`stream.py`](src/harness/confirm/stream.py)), with the same result as the in-memory read on the same matrix.

**Reporting.** As paper 02 reports: every accuracy as its mean over three seeds with the sample standard deviation and each seed's value; every comparison paired on lattice, band mapping, channel count, noise level, input gain and seed, as the mean difference with its standard deviation over seeds and a 95% interval from resampling test clips. No threshold is applied to either.

## Cost

Estimated with the cost model in `plan.py` (`uv run python -m harness.confirm.plan estimate`), calibrated on one thread of an Apple M-series CPU while paper 02's runs held the other cores: per channel and clip, simulating 61 frames takes 0.29, 0.97, 2.1, 7.3 and 38.5 ms at 8, 16, 32, 64 and 128 rows (paper 02's tier 2 measured 16 × 16 at 4 channels at about 5 ms a clip, which this reproduces), the windowed statistics about as long again, and the projection 150 GFLOP/s. The design factors multiply the simulation by 0.7 (helix) to 2.6 (second harmonic); the trained baselines were timed per step at 2,048, 65,536 and 524,288 parameters. GPU times were not measured.

| tier | runs | from paper 02 | CPU-hours, lattices 8–32 | CPU-hours, 64 and 128 | longest run | peak memory |
|---|---|---|---|---|---|---|
| gate | 30 | 0 | 0.9 | 10 | 3.3 h | 11 GB |
| size | 1,674 | 54 | 33 | 458 | 3.3 h | 11 GB |
| trained | 1,350 | 30 | 12 | 100 | 2.6 h | 1 GB |
| design | 13,500 | 300 | 518 | 7,647 | 10.5 h | 11 GB |
| quadrature | 1,134 | 18 | 30 | 421 | 3.7 h | 11 GB |
| carrier | 240 | 9 | 1,529 | not planned | 42 h | 6 GB |
| design-quadrature | 12,420 | 48 | 584 | 8,625 | 13 h | 11 GB |
| design-carrier | 1,875 | 18 | 27,998 | not planned | 146 h | 6 GB |
| **all** | **32,223** | **477** | **30,705** | **17,261** | | |

Memory is never the limit: an arm read in memory holds at most about 6 GB (4,096 states: its features for 8,048 clips and its projection matrix), and a streamed arm holds one channel's training features and matrix rows, about 11 GB at 128 × 128. Time is. The longest runs are the 128 × 128, 16-channel networks (3.3 CPU-hours for the reference coupling function, up to 13 for the slowest design), the S4D sized to them (2.6 hours, its state grows with its width), and anything on the carrier pathway (16,000 steps a clip: from 10 CPU-minutes at 8 × 8 with one channel to 146 CPU-hours). The carrier tiers need a GPU, as in paper 02; the design tiers at 64 and 128 want one.

## Stages

The tiers are ordered so each stage answers a question on its own. `--grids` and `--channels` select a stage without changing any run's identity.

| stage | runs | runs here | CPU-hours | where |
|---|---|---|---|---|
| A. gate, size and trained on lattices 8 to 32 | 1,698 | 1,614 | 46 | this machine, about 8 hours at 6 workers |
| B. gate, size and trained on 64 and 128 | 1,356 | 1,356 | 568 | a 32-core machine for a day, or a GPU pod |
| C. design and quadrature on lattices 8 to 32 | 8,130 | 7,812 | 548 | a 32-core machine for a day |
| D. design and quadrature on 64 and 128 | 6,504 | 6,504 | 8,067 | GPU pods |
| E. carrier on lattices 8 to 32 | 240 | 231 | 1,529 | GPU pods |
| F. design on the quadrature and carrier pathways | 14,295 | 14,229 | 37,207 | GPU pods, after slimming |

Ways to slim, for the author to choose from (each measured with the same model):

- The design tier at one band mapping and input gain 1: 3,600 runs and 2,057 CPU-hours at every lattice, 2,100 runs and 146 CPU-hours on lattices 8 to 32.
- The design-quadrature tier the same way on lattices 8 to 32: 2,046 runs, 167 CPU-hours.
- The design-carrier tier at one band mapping: 1,107 runs, 15,987 CPU-hours; also at 1 to 4 channels only: 657 runs, 3,545 CPU-hours.
- The size tier at input gain 1 only: 834 runs, 246 CPU-hours instead of 491.
- Channel counts 1, 4 and 16 only: two fifths fewer runs in every tier.

## The record

Each tier writes `results/confirmatory/<tier>-<pathway>-<G>x<G>.json`, the design tiers one file per coupling function as well. Each run's entry holds its spec, the arm's states, parameters and parameter budget (and a trained baseline's width and whether it is within 15% of its budget), how it was read (in memory or streamed by channel), one cell per width with its accuracy, chosen penalty and, at the primary cell, each test clip's correctness for the bootstrap, and timing and the environment. The cells taken from paper 02 stay in paper 02's record under their own run identities; `uv run python -m harness.confirm.summary` reads both and writes every accuracy and comparison to `results/confirmatory/summary.json`, each marked with its source, and the primary cells as tables to `summary.md`.

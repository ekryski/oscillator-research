# Design: paper 03

The current design of paper 03, and its open questions (section 11). No run has been made.

Everything below is implemented in `src/harness/confirm/` and tested in `src/tests/`; where this text and the code could disagree, the tier definitions in `harness/confirm/plan.py` are the plan. [TIERS.md](TIERS.md) gives each tier's cost.

## 1. What this is

A size and layout study of the untrained coupled oscillator networks of paper 02 ([Kryski 2026b](https://github.com/ekryski/oscillator-research/blob/main/papers/02-untrained-reservoirs/)). Paper 02 read one network size, 1,024 oscillators in 4 channels of a 16 × 16 lattice (2,048 parameters), against a spectrogram-only baseline, an uncoupled copy, leaky-integrator banks and trained baselines of the same size. Paper 01's survey ([Kryski 2026a](https://doi.org/10.2139/ssrn.7445198)) found that no published oscillator network had ablated its parameter or oscillator count or its channel layout. This study does, for the networks of paper 02 and on paper 02's task, protocol and read, so that its 16 × 16, 4-channel cells are paper 02's own.

Paper 02's widened tier 4 (commit dd60fed: lattices of 8 × 8, 16 × 16 and 32 × 32 at 1 to 16 channels under two band mappings) is where this study began. The author moved it out of paper 02, which keeps only its 16 × 16, 4-channel network, and into this paper, extended to 64 × 64 and 128 × 128 and to every coupling function, lattice geometry and input pathway paper 02 ran.

## 2. Questions

The questions are the author's. The study states no hypotheses and applies no decision bar: as in paper 02, every accuracy and every paired difference is reported with its spread (section 10) and no verdict is drawn from a threshold.

- **Q1, size.** How does the coupled oscillator network's accuracy change with its number of oscillators, C · G², at a fixed readout?
- **Q2, size against controls.** Does the network's margin over its uncoupled copy, its state-matched leaky-integrator bank and the spectrogram-only baseline on the same rows change with size?
- **Q3, layout.** At a fixed number of oscillators, does it matter whether they form few large channels or many small ones? The design has nine such sets of two or three layouts, from 256 oscillators (8 × 8 at 4 channels, 16 × 16 at 1) to 65,536 (64 × 64 at 16, 128 × 128 at 4); 1,024 oscillators, for example, are 8 × 8 at 16 channels, 16 × 16 at 4 (paper 02's network) and 32 × 32 at 1.
- **Q4, input resolution.** At a fixed lattice, does driving each row with its own mel band read differently from mapping paper 02's 16 bands onto the rows?
- **Q5, trained baselines, and small sizes in particular.** How do trained baselines of the network's parameter count compare as both grow? Are untrained oscillator networks more useful than trained networks of the same size at the smallest sizes, from 128 to about 2,048 parameters, and does any such advantage persist, shrink or reverse as both grow? The cells below 2,048 parameters, where some baselines cannot be sized closely (section 4), are kept and flagged, because they are where this question is asked.
- **Q6, design and pathway at size.** Do paper 02's design effects (coupling function, lattice geometry) and pathway effects (quadrature, carrier) depend on size?

## 3. Data

Paper 02's data, unchanged: its 50-repetition AudioMNIST bank (30,000 clips, 60 speakers, 10 digits, 16 kHz), its Protocol A (speakers 1 to 48 train, speakers 49 to 60 test; the test set is all 6,000 test clips in one fixed order; a seed permutes the training pool and the training set is its first 2,048 clips), and its noise protocol (white noise relative to each clip's speech power, drawn from a generator seeded by the clip's identity, so a clip sounds the same in every arm, seed and tier). **Every run is at 0 dB**, noise as loud as the speech: paper 02 also ran +5 dB and clean audio, where the task saturates. Paper 02's Protocol B (Becker et al.'s folds) and its order task are not carried over; the calibration against Becker et al. is paper 02's.

**The front end at other band counts.** Paper 02's front end is a 512-point transform (257 bins, 32 ms Hann window) every 256 samples, mel filters, the log of each band's energy and a fixed rescaling. At 8, 16 and 32 bands paper 03 uses it exactly (tested against paper 02's formula). At 64 and 128 bands the narrowest mel filters fall between the 257 bins (at 128 bands one filter is empty and 25 touch a single bin), so the transform is zero-padded to 1,024 and 2,048 points, the lengths at which the narrowest filter spans three bins as the 32-band one does at 512. The window stays 512 samples and is shifted back onto paper 02's frames, so the frame count (61 for a 1 s clip) and every frame's timing are paper 02's; tested with a click that lands in the same two frames at 16, 64 and 128 bands. Zero-padding samples the same windowed spectrum more finely and adds no frequency resolution: the lowest bands at 64 and 128 are narrower than the window can resolve, so neighbouring low bands carry strongly correlated energies. The quadrature pathway takes its per-band phase from the same zero-padded transform, referenced to the start of paper 02's window, so it equals the unpadded phase wherever the two transforms share a bin. The alternative, a longer window, would give real frequency resolution at the cost of time resolution and a different frame count, which would change the read's windows; it is not used. Narrower bands hold less energy, so the drive's level falls slightly with band count; measured on 200 clips of the bank at 0 dB, the mean drive is 1.38, 1.32, 1.24, 1.23 and 1.22 at 8, 16, 32, 64 and 128 bands (the drive runs from 0 to about 1.6), the zero-padding's finer sampling offsetting the narrowing above 32 bands. It is not corrected.

**The band mapping.** A lattice of G rows is driven either by G mel bands, one per row, or by paper 02's 16 bands mapped onto its rows: above 16 rows each band drives G/16 adjacent rows, and an 8-row lattice gets the mean of each two adjacent bands. At 16 × 16 the two coincide and run once. Nothing in the mapping is fitted.

**The log-spaced bands.** The carrier pathway splits the waveform into log-spaced bands, and tonotopic natural frequencies take their centres. Paper 02 fixed four bands per octave from 96 Hz, which at more than 16 bands would run past the top of the range (32 bands would span eight octaves). Paper 03 spans the same four octaves at every band count, G/4 bands per octave; at 16 it is paper 02's exactly.

## 4. Arms, and how their size is matched

Every arm sits in paper 02's slot: front end, then the arm, then the shared readout. A network of C channels on a G × G lattice, with G in {8, 16, 32, 64, 128} and C in {1, 2, 4, 8, 16}, has C · G² oscillators (64 to 262,144) and 2 · C · G² parameters (128 to 524,288).

| arm | what it is | states | parameters |
|---|---|---|---|
| spectrogram-only baseline | the front end's rows, read directly; on each lattice's rows | 0 | 0 |
| coupled oscillator network | C channels of a G × G lattice | C · G² | 2 · C · G² |
| uncoupled oscillator network | the same network with its coupling kernels set to zero | C · G² | 2 · C · G² stored, C · G² in effect |
| leaky-integrator bank, state-matched | C · G² independent leaky integrators, band r driving row r | C · G² | 2 · C · G² |
| GRU, TCN, CNN, transformer, S4D | trained end to end, sized to the network | | at or near 2 · C · G², trained |

**Input gain** is 1 for every reservoir on the band-energy and quadrature pathways and 32 on the carrier pathway, paper 02's calibrated carrier gain. It does not apply to the spectrogram-only baseline or the trained baselines, as in paper 02.

**Channels** are parallel copies, like the heads of one attention layer, not stacked layers: every channel receives the same input, has its own coupling kernel and natural frequencies, and no channel acts on another. The readout reads all channels side by side.

**The network at size.** The reference configuration is paper 02's: Kuramoto coupling, torus, random natural frequencies, restoring strength λ = 0.3, coupling ceiling 1, one Euler step of dt = 0.1 per frame, each seed drawing the kernels and natural frequencies exactly as paper 02 drew them.

- *Natural frequencies* are drawn per oscillator from N(1, 0.1²) at every size, so their distribution does not change with the lattice.
- *Coupling kernels* are drawn per weight from N(0, 0.05²) at every size, and then each channel's kernel is scaled so its spectral peak, the largest factor by which it amplifies any pattern of phases across the channel, is exactly the coupling ceiling. Paper 02 scaled a kernel only down to the ceiling. That made no difference at 16 × 16, where a random kernel's peak is always above it (1.42 to 2.54 over every geometry, seed and channel either paper draws), but at 8 × 8 the peak is usually below it (median 0.80; 80% of torus channels below 1), so an 8 × 8 network would have had weaker coupling than a larger one for a reason that has nothing to do with size. Exact scaling fixes the strongest coupling gain at the ceiling at every size. What still changes with size is how that gain is spread: a random kernel's peak grows with G (medians 0.80, 1.84, 4.0, 9.1 and 20.0 on the torus from 8 × 8 to 128 × 128), so after scaling each weight is smaller (standard deviation 0.063 at 8 × 8, 0.0025 at 128 × 128), and the typical gain over all spatial patterns, the root mean square of the kernel's spectrum, drifts from 0.50 to 0.32. The per-weight initialization is not rescaled with G, since exact scaling already fixes the quantity the ceiling was meant to fix. Every 16 × 16 network is bit-identical under the two rules (tested for every coupling function, geometry, channel count, both ceilings and both coupling implementations), so paper 02's 16 × 16 runs stand.
- *The time step.* The scaled coupling's gain is at most 1 over any spatial pattern, so its contribution to a step is at most dt = 0.1 radian in root mean square over the lattice; the drive adds at most dt · g · 1.5 = 0.15 radian at input gain 1. Both are far inside the Euler integrator's validity bound (π per step) at every size, and the forward pass stays bounded and finite on every geometry and coupling function at 8 × 8 and 64 × 64 (tested).
- *Geometries.* The six geometries of paper 02, at every lattice. The cube cuts each row of G oscillators into an a × b slab; paper 02's 4 × 4 needs G to be a square, which 8, 32 and 128 are not, so a is the largest power of two whose square does not exceed G: 2 × 4, 4 × 4, 4 × 8, 8 × 8 and 8 × 16 from 8 to 128. The Stuart–Landau coupling functions run on the torus only, as in paper 02. FFT and the dense circulant implementation are the same operator (tested on every geometry at 8, 16 and 32); which one runs depends on the device (section 6).

**The uncoupled network** has no coupling term at all, so it does not depend on the coupling function or the geometry; one per size and pathway serves every design configuration (the Stuart–Landau cores differ in their amplitude dynamics and are not given one).

**The state-matched leaky-integrator bank** is paper 02's bank at C channels of a G × G lattice, matched to the network by construction at every size: C · G² units, each with a leak rate and an input weight (2 · C · G² parameters), band r driving every unit of row r, time constants log-spaced from 16 ms to 1 s across the C · G units of a row. Paper 02's width-matched bank, with twice the units to match the network's two signals per oscillator, is the state-matched bank of 2C channels on the same lattice, which the size tier runs for C up to 8.

**The trained baselines** are paper 02's architectures and recipe (AdamW, 3e-3 annealed to 3e-4, 30 epochs, batches of 64, a learned linear head on the shared statistics, the ridge readout on the same statistics). Each has one width knob and the rest of its design fixed: the GRU's hidden size, the TCN's and the CNN's hidden channels, the transformer's model width (two heads), the S4D's width with as many states per channel as its width. The width is the widest whose parameter count does not exceed the network's 2 · C · G², except for the CNN, which is the TCN's form and takes the narrowest width that reaches it, so the two stay one width apart. This gives paper 02's widths (18, 12, 13, 16, 16) and parameter counts (1,944, 1,948, 2,109, 1,968, 1,840) at 2,048 parameters on 16 rows, and every count within 15% of the network's from 2,048 parameters up, with widths up to 358 (GRU) at 524,288 parameters. Below 2,048 the steps between widths are coarse: in 10 of the 45 size cells, all at 8 × 8 with 1 to 8 channels or 16 × 16 with 1 or 2, one or more baselines fall 15% to 52% under the budget or up to 33% over it. Those cells are kept, since they are where Q5 asks whether oscillators are more useful than trained networks at small sizes, and the record flags each with its achieved count. A trained baseline is driven by exactly the rows its network is driven by, so it sees the same band mapping. Input gain does not apply to it, as in paper 02.

## 5. The read

Paper 02's read, unchanged: one function reads every arm, the mean, standard deviation and mean absolute frame-to-frame change of each signal (sin θ and cos θ of each oscillator; the units' states for the bank; the hidden trajectory for a trained baseline; the band energies for the spectrogram-only baseline), over each of four equal windows of the same frames for every clip, 16 to 61. No statistic depends on an endpoint. The spectrogram-only baseline is also read from frame 0, over the whole clip, and that whole-clip read is its primary one. Paper 03 records only this four-window read for the networks; paper 02's secondary reads (rotation rates, the whole-span read) are not recorded.

**At scale.** An arm with more than 4,096 states (32 × 32 at 8 or 16 channels, 64 × 64 at 2 or more, every 128 × 128 arm) is read channel by channel, because its features do not fit in memory: a 128 × 128, 16-channel network has 6,291,456 features per clip, 202 GB for a run's 8,048 clips, and paper 02's projection matrix for it would be 103 GB. The read is linear after standardization and the channels do not interact, so it splits exactly by channel: each channel is simulated alone (its own kernel, natural frequencies and starting phases, sliced from the full network), its training features are kept (3.2 GB at 128 × 128) and standardized with their own statistics, and their projection is added into the run's; its test clips are projected batch by batch. Given paper 02's projection matrix it reproduces the in-memory read's accuracy, chosen penalty and per-clip correctness exactly in every tested case; the projected features agree to float32 rounding, since the sums run in another order. Its own matrices are drawn channel by channel from seeds of its own (section 6), with paper 02's distribution and scale, N(0, 1/native width). Arms of 4,096 states or fewer, including every cell taken from paper 02, are read in memory by paper 02's code.

## 6. The readout, its two projections, and the devices

Paper 02's readout: standardize with the training set's own statistics, project by a fixed Gaussian matrix to a common width (each narrower width is the leading columns of the widest), and fit a closed-form ridge per class, its penalty chosen from {0.001, 0.01, 0.1, 1} on the last eighth of the training set.

**Two projections.** Every projected read is recorded twice, and each cell is tagged with its projection:

- *fixed*: paper 02's matrix, `torch.randn(native, 4096, generator=torch.Generator().manual_seed(4242)) / sqrt(native)`, the same for every seed;
- *seeded*: the same draw from generator seed 4242 + 1 + s for run seed s, so the matrix changes with the seed as the arms' own draws do.

Paper 02's matrix is fixed across seeds, so the spread over seeds omits the projection's own variability, and the draw matters at the narrow widths: one small network read under two different draws (paper 02's matrix and the streamed read's; 8 × 8, 2 channels, 0 dB, gain 2, seed 0) scored 72.6% and 73.6% at width 192, and 76.4% and 76.3% at 1,024. Under the seeded projection the spread over seeds includes it. A width at or above an arm's own number of features is read unprojected, once, and tagged "none". A streamed arm (section 5) draws each channel's rows of both matrices from seeds of their own, 100 times the in-memory read's generator seed plus the channel: 424,200 + c for the fixed matrix and 100 · (4,242 + 1 + s) + c for the seeded one; a channel's two matrices are drawn at once in two threads.

**Widths.** 192, 1,024 and 4,096, never wider than the arm itself; the unprojected read is also fitted for arms of 1,024 states or fewer, as in paper 02's widened tier 4. **The primary cell is width 192 and 2,048 training clips**, under each projection, so the readout has the same 1,930 weights at every size: the network grows, the readout does not, and any gain with size is the network's. Widths 1,024 and 4,096 are reported as secondary, to show whether a larger network needs a wider readout to show what it holds. A spectrogram-only baseline is never read wider than its own features: 96 at 8 rows (read unprojected there, beside networks read at 192), 192 at 16, and projected from 384, 768 and 1,536 above.

**Devices.** Every run takes `--device auto|cpu|mps|cuda`: `auto` is CUDA if present, else Apple Silicon's GPU (torch "mps"), else the CPU. The device each run used, and which GPU, is recorded in its environment. What runs where:

- The front-end rows are computed on the CPU and cached, so every device is driven by bit-identical input.
- The untrained arms are built on the CPU from their seed, so every device runs the same kernels and natural frequencies, then moved to the device and simulated there, with their statistics; a trained baseline is built on the CPU and trained on the device.
- The readout projects on the device and solves its ridge on the CPU in float64 (MPS has no float64).
- The coupling runs by FFT or as the same operator in a dense matrix, whichever measured faster on the device: dense up to 16 × 16 and FFT above on the CPU (paper 02's choice at 16 × 16); dense up to 64 × 64 and FFT at 128 × 128 on MPS, where FFT measured 2 to 80 times slower; FFT on CUDA, as in paper 02. The Stuart–Landau cores couple by FFT on the CPU and CUDA, as in paper 02, and densely on MPS up to 64 × 64.

**Only the CPU is bit-identical to paper 02.** A GPU's floating-point results are close to the CPU's but not identical: its FFTs, matrix products and reductions sum in other orders, and over 61 steps the phases drift apart at the level of rounding. The tests hold every model, front end and trained baseline on MPS to the CPU (trajectories within 2 · 10⁻³, whole runs within a few test clips on the synthetic bank). On the real bank, paper 02's network at 0 dB, gain 1 and seed 0, run on MPS, scored 77.98%, 79.37%, 80.97% and 81.70% at widths 192, 1,024, 4,096 and unprojected, the accuracies paper 02 recorded on its CPU to two decimals of a percent, in 17 s against about a minute on one CPU thread. Paper 02's cells are therefore reused only as recorded on its CPUs, the reuse gate runs on the CPU, and any paper 02 cell that paper 03 has to run itself (section 8) runs on the CPU whatever `--device` says. Comparisons within paper 03 pair runs of one device where the record allows, and a run's device is part of its record.

## 7. Tiers

| tier | what | runs |
|---|---|---|
| gate | no input at every lattice: the coupled and uncoupled networks and the state-matched bank at 1 and 16 channels; input gain 0, seed 0 | 30 |
| size | the reference network, its uncoupled copy and its state-matched bank at every lattice (5), channel count (5) and band mapping (2; one at 16 × 16), with the spectrogram-only baseline on each lattice's rows; seeds 0 to 2 | 432 |
| trained | the five trained baselines sized to every network of the size tier, on its rows; seeds 0 to 2 | 675 |
| design | the 25 other coupling-function and geometry configurations at every size, coupled network only; seeds 0 to 2 | 3,375 |
| quadrature | the reference and uncoupled networks at every size on the quadrature pathway, with that pathway's spectrogram-only baseline; seeds 0 to 2 | 297 |
| carrier | the reference and uncoupled networks and the state-matched bank on the carrier pathway at lattices up to 32 × 32, with its spectrogram-only baseline; gain 32; seeds 0 to 2 | 240 |
| design-quadrature | the 23 other phase coupling-function and geometry configurations on the quadrature pathway at every size | 3,105 |
| design-carrier | the 25 other configurations on the carrier pathway up to 32 × 32; gain 32 | 1,875 |

Every run is at 0 dB and, except on the carrier pathway, input gain 1. 10,029 runs in all, 150 of them paper 02's (section 8): about 34,900 hours on one CPU thread, or about 1,800 hours on the M1 Max's GPU, by the cost model measured on each (TIERS.md, which has the cost per tier, lattice and stage). Most of it is the design tiers on the carrier pathway (1,147 GPU-hours) and the design tiers at 64 × 64 and 128 × 128 (about 480); the size and trained tiers are 35 GPU-hours. **Running order:** gate, size and trained on lattices 8 to 32 (stage A), then on 64 and 128 (stage B), then design and quadrature (C, D), carrier (E), and the design tiers on the other pathways (F). Whether the later stages run, and at what scale, is decided by compute, not by results; any tier that does not run is reported as not run.

## 8. What is taken from paper 02

Paper 02 ran the 16 × 16 lattice, one band per row, at 4 channels. Wherever a paper 03 run is a run paper 02 recorded (same arm, pathway, noise level, input gain and seed; here 0 dB and gain 1, or 32 on the carrier), its cells are read from paper 02's record rather than run again. Paper 02's primary cell, 2,048 training clips at width 192, and its widths 1,024 and 4,096 are cells of those runs.

| paper 03 tier | cells from paper 02 | paper 02 record file | runs |
|---|---|---|---|
| size | spectrogram-only baseline; coupled and uncoupled networks; state-matched bank at 4 channels, and at 8 (paper 02's width-matched bank) | `tier1-recognition-envelope.json` (committed) | 15 |
| trained | the five trained baselines at 2,048 parameters | `tier1-recognition-envelope.json` | 15 |
| design | the 25 other configurations at 4 channels, random natural frequencies, λ = 0.3, ceiling 1 | `tier2-recognition-envelope-<coupling>.json` (in progress) | 75 |
| quadrature, design-quadrature | the pathway's baseline; the diagonal of paper 02's tier 3 (the four phase coupling functions on the torus, Kuramoto on the helix) | `tier3-recognition-quadrature.json` (not yet run) | 18 |
| carrier, design-carrier | the pathway's baseline; the 4-channel bank; the diagonal of paper 02's carrier tier (every coupling function on the torus, Kuramoto on the helix) | `tier3-recognition-carrier.json` (not yet run) | 27 |

The reuse rests on three checks. Exact kernel scaling leaves every 16 × 16 network bit-identical to paper 02's (section 4). The arms involved have at most 4,096 states and are read by paper 02's own code. And before any tier runs, **the reuse gate** (`gates reuse`) re-runs a sample of ten reused runs, one per record file and arm kind, with paper 03's harness and requires every cell's accuracy and per-clip correctness to equal paper 02's; before this design was drafted, and again after the device and projection changes of 2026-09-24, paper 02's and paper 03's harnesses gave identical trajectories and features on the CPU, bit for bit, on all 318 combinations of the 16 × 16 arms, three seeds, two gains and two pathways that were tried, and the five samples already recorded (paper 02's tier 1: the coupled and uncoupled networks, the 8-channel bank, the spectrogram-only baseline and the GRU) reproduced every recorded cell, accuracy and per-clip correctness alike. A reused cell is marked with its source in the record's summary and cited to paper 02 in the paper. A paper 02 run is used only if it carries the seeded projection as well as the fixed one; paper 02's runs recorded before it adopted the seeded projection carry only the fixed one. If paper 02 has not recorded a planned run completely when a tier runs, paper 03 runs it itself, on the CPU, and reports its own; its fixed cells equal paper 02's, which is what the reuse gate checks.

## 9. Integrity gates

1. The harness test suite passes.
2. **Reuse.** The sample of section 8, run on the CPU, reproduces paper 02's record exactly, cell for cell under the fixed projection (and the seeded one where paper 02 has recorded it). A sample whose paper 02 run is not recorded yet is reported as not checked. Paper 02's carrier tier runs on a GPU, so its carrier sample can only be checked if paper 02 also records it on a CPU.
3. **No input.** Every cell of the gate tier reads exactly 10%, at every lattice, in memory and streamed.
4. Every spectrogram-only baseline is recorded for every lattice, band mapping, pathway and noise level.

A failed gate stops the reporting of whatever it guards until it is fixed and the fix logged.

## 10. Reporting

As paper 02 reports:

- Every accuracy is its mean over the three seeds, with the sample standard deviation and each seed's value, in points, under the fixed projection and under the seeded one, side by side.
- Every comparison between two arms is paired: both scored on the same 6,000 test clips at the same lattice, band mapping, channel count, noise level, input gain, seed and projection, and reported as the mean difference, its standard deviation over seeds, and a 95% interval from resampling the test clips 2,000 times.
- No threshold is applied and no verdict drawn. Whether a difference matters is left to the reader and to further seeds.
- Every comparison named in section 2 is reported, whichever way it falls: at each lattice, band mapping and channel count, the network against its uncoupled copy, its state-matched bank, the whole-clip spectrogram-only baseline on its rows and each trained baseline of its size; each design configuration against the reference at the same size; each pathway's network against that pathway's baseline and uncoupled network.
- Accuracy is reported against the number of oscillators and against parameters, for every arm, as curves, with the layouts of equal size marked, not as single operating points.
- Arm parameters and readout weights are reported separately at every width.
- Cells taken from paper 02 are marked as such wherever they appear, and every run's device is in the record.

## 11. Open questions

1. Paper 02's secondary reads (rotation rates) are not recorded; whether to record them is undecided.
2. Scale: which of stages C to F to run, and whether to slim the design tiers further (TIERS.md gives the cost of each option).

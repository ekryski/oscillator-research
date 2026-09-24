# Design: paper 02

This document records the design of paper 02's study: its questions, data, arms, read, readout, tiers and integrity checks. It was written before the study's first tier started (commit f1d52fb, 2026-09-23 03:53 MDT), and the design has changed since where the experiment needed it; each change is in the decision log at the end, dated, with its reason.

Everything below is implemented in `src/harness/experiment/` and tested in `src/tests/test_experiment_*.py`; where this text and the code could disagree, the tier definitions in `harness/experiment/plan.py` are the plan.

## 1. What this is

An original study. Earlier pilot experiments (August 2026, in a separate development repository) suggested the questions and several of the comparisons, and this study runs similar comparisons again under a new protocol. Every number in the record and in the paper comes from this study's own runs; nothing from the pilot is in either, and the pilot's code is archived on the `ek/paper-02-pilot` branch.

The pilot also showed five weaknesses in how it read its arms, which the design below exists to correct:

- **Readout width.** The oscillator network was read with far more features than its spectrogram-only baseline, so the two readouts had very different numbers of coefficients.
- **Featurization.** The arms were not read alike. The trained baselines' features included their last hidden state, which carries temporal order directly; the network's and the baseline's did not.
- **The read window.** Each arm was read over its own clip's length, while the baseline was read over the whole padded clip. An undriven network read over each clip's own length still tells clips apart, because an oscillator keeps rotating and statistics over a span encode the span's length, which in speech carries the digit. This study measures that leak itself (diagnostic D1) and closes it with a fixed window.
- **Seeds.** The pilot's factorial was run at one seed.
- **Baselines.** The pilot's baselines were printed to a terminal and never written to its record.

## 2. Hypotheses

The direction and the hypotheses are the author's; the wording below was proposed on 18 September and accepted on 23 September. They say which comparisons the study makes. None is scored as passed or failed: every comparison is reported with its spread (section 9).

- **H1, dynamics.** An untrained oscillator network adds accuracy over its front end alone when both are read at the same width.
- **H2, oscillation.** That gain exceeds the gain from a non-oscillating bank of leaky integrators of the same size.
- **H3, coupling.** The coupled network reads higher than the same network uncoupled.
- **H4, design.** The coupling function, lattice geometry, natural frequencies, restoring strength and coupling ceiling each move accuracy.
- **H5, readout.** Readout width has a large effect on accuracy, for every arm.
- **H6, memory.** An untrained network reads temporal order through an order-free readout, and does so better than a leaky-integrator bank.

## 3. Data

AudioMNIST, 30,000 recordings, 60 speakers, 10 digits, 50 repetitions, resampled to 16 kHz, peak-normalized, energy-trimmed and capped at 1 s. The bank keeps all 50 repetitions.

**Protocol A, for every tier but B.** Speakers 1 to 48 train (24,000 clips), 49 to 60 test (6,000). **The test set is the whole test pool**, in one fixed order, identical for every arm, seed and condition. A seed permutes the training pool, and the training sets are the first 2,048, 8,192 and 24,000 clips of that permutation, so each contains the one before it.

**Protocol B, for comparison with published results only.** The five speaker folds of Becker et al., copied verbatim from `preprocess_data.py` in the corpus: three folds train (18,000 clips), one validates (6,000), one tests (6,000), clean audio, one run per fold.

**Noise.** Clean, 0 dB and +5 dB. The harness's convention is the reverse of the usual one: noise amplitude is speech RMS times 10^(dB/20), so 0 dB is noise at speech-equal power and +5 dB is louder noise, drawn over the whole padded window. Each clip's noise comes from a generator seeded by the clip's identity and the level, so a clip sounds the same in every arm, seed, tier and process. Seeds change which clips a readout is fitted on and an arm's own random draws, never the clips themselves. Front-end rows are computed once per input pathway and level and memory-mapped by every run.

**Gain.** g = 1 and g = 2 for every reservoir. The spectrogram-only baseline and the trained baselines read the rows as they are, so gain does not apply to them. g = 0 removes the input and runs as a sanity cell per reservoir.

## 4. Arms

Every arm sits in the same slot: front end, then the arm, then the shared readout.

| arm | what it is | states | stored parameters |
|---|---|---|---|
| spectrogram-only baseline | the arm slot left empty | 0 | 0 |
| coupled oscillator network | 4 channels on a 16 by 16 lattice | 1,024 | 2,048 |
| uncoupled oscillator network | the same network, same seed, with its coupling kernels set to zero | 1,024 | 2,048 (1,024 in effect) |
| leaky-integrator bank, state-matched | independent leaky integrators, matched to the network in states and parameters | 1,024 | 2,048 |
| leaky-integrator bank, width-matched | the same, matched to the network in exposed signals | 2,048 | 4,096 |
| GRU, TCN, CNN, transformer, S4D | trained baselines | | about 2,000, trained |

**The coupled network.** Where a tier does not vary it: Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1. Each seed draws the kernels and natural frequencies.

**The leaky-integrator bank.** Each unit follows x ← (1 − a)·x + a·tanh(g_in·g·u). Band r drives the units of row r, the routing the network uses. The time constants are one fixed log-spaced schedule from one hop frame (16 ms) to one clip (1 s), laid out so that every band is read at every time scale, and converted to a leak rate at the input's own frame rate. The input gains are drawn N(1, 0.1²), the distribution the network draws its natural frequencies from. Only the gains vary with the seed, so this arm's seed-to-seed spread is small by construction. Nothing about the schedule is tuned: it is a control, not a claim that the design is optimal. The state-matched bank matches the network's states and parameters and is half as wide at native width, because the network exposes two signals per state; the width-matched bank matches the native width at twice the parameters.

**The trained baselines.** Trained end to end with a learned linear head on the shared statistics of section 5 (four windows for recognition, the whole span for the order task), so each network is trained for the read it is judged by and cannot hand its head its last state. The recipe: AdamW at 3e-3 annealed to 3e-4 by cosine, 30 epochs, batches of 64, gradients clipped at 1, the last epoch kept. A run whose loss does not fall by 20% is flagged in the record as not having trained. Each baseline is trained at each training size and read by the shared ridge at that size; its own head's accuracy is recorded as a secondary number. They train on the CPU on every machine.

## 5. The read

One function reads every arm: over a window, or over each of four equal windows, it computes three statistics per signal, the mean, the standard deviation, and the mean absolute frame-to-frame difference. Arms differ only in the signals they expose: the front-end rows (the spectrogram-only baseline), sin θ and cos θ per oscillator (the networks; x and y for Stuart-Landau cores), the unit states (the banks), the hidden trajectory (the trained baselines).

**Every arm is read over one fixed window, the same frames for every clip:** from the end of the 16-frame integrator warm-up to the end of the padded clip (frames 16 to 61 for recognition, 16 to 147 for the order task). An arm with no input therefore reads exactly chance, and everything a read carries arrives through the arm's response to the input. **The spectrogram-only baseline is also read from the first frame, and that whole-clip baseline is the primary control**, because it is the harder one: it is given everything the network was driven with, including the onset the network can carry only as memory. The baseline over the arms' own window is reported beside it.

**No statistic may depend on an endpoint.** That excludes the last state, and the signed mean of a first difference, which telescopes to the last value minus the first.

**Secondary read S1, the networks' rotation rates.** The read above plus two signed rotation-rate features per oscillator and window, the mean cos Δθ and sin Δθ between consecutive frames: the oscillator's natural observable, which the generic read drops. It is recorded beside the primary read for every coupled and uncoupled network, labelled as the read that favours the networks, since no other arm has an analogue.

**Diagnostic D1, a per-clip window.** The coupled network at g = 0, 1 and 2, 0 dB, three seeds, read over each clip's own length, to measure how much a per-clip window hands a network. Not a hypothesis test.

**Coherence instruments.** Every coupled and uncoupled network run from Tier B on records, over its test clips, the order parameter R, each oscillator's phase-locking value to its own band's drive, the share of oscillators locked above 0.5 and the mean amplitude. They are diagnostics and reach no readout.

## 6. The readout

For each training size, an arm's native features are standardized with that training set's own statistics (a feature constant over the training set gets weight zero), brought to each common width by one fixed seeded Gaussian projection (each narrower width is the leading columns of the widest, so a wide read contains the narrow one), and classified by a closed-form ridge, one hyperplane per class, with its penalty chosen from {0.001, 0.01, 0.1, 1} (scaled by the number of fitted clips) on the last eighth of the training set, or on the validation fold under Protocol B. The ridge is solved in whichever of its two equivalent forms is smaller, and is tested to make the same penalty choice and the same prediction on every clip as the direct solution. The projection tier also reads Tier 1's reservoirs through a projection drawn per run seed.

**Width.** 192, 1,024 and 4,096, plus native. 192 is the spectrogram-only baseline's native width (16 bands, three statistics, four windows). An arm is never read wider than its native width; at or above it, it is read unprojected.

**Training-set size.** 2,048, 8,192 and 24,000 clips, crossed with width, because a wide ridge on few clips and a wide ridge on many are different estimators. The native read is fitted at 2,048 clips only: at 24,000 its system is too large to solve per run.

**The primary cell: width 192, 2,048 training clips, four windows (recognition) or the whole span (order).** Everything else is secondary and is reported whatever it shows.

## 7. Tasks and tiers

**Recognition.** Ten-way digit classification.

**Temporal order.** Two recordings joined behind a 272 ms silent leader and across a 100 ms gap, a then b against b then a, on five digit pairs (3-7, 1-8, 2-5, 4-9, 0-6). Per pair, 2,048 training clips drawn per seed from the training speakers and one fixed set of 2,048 test clips from the test speakers, with alternating labels so the classes are exactly balanced. Scored on the whole-span read only. For each pair and noise level, the spectrogram-only baseline's three-seed mean is reported with its 95% interval over test clips, to show whether the task leaks order to a read with no memory: it should sit at chance, 50%.

| tier | what | runs |
|---|---|---|
| gate | the g = 0 cells for the four reservoirs, and diagnostic D1 | 13 |
| 1, arms | the spectrogram-only baseline, both networks and both banks at every noise level, gain and seed, at all three training sizes; the five trained baselines at every size; the same arms on the order task, five pairs | 846 |
| 2, design | coupling function (4 phase functions; the 2 Stuart-Landau functions on the torus only), lattice geometry (6), natural frequencies (random, tonotopic, identical), restoring strength (0.3, 0.1), coupling ceiling (1, 0.5); 0 and +5 dB, both gains, three seeds; primary size; four-window read with and without rotation rates | 3,744 |
| B, Becker | every Tier 1 arm under Protocol B, both gains for the reservoirs | 70 |
| 3, quadrature | a diagonal (4 phase functions by random and tonotopic frequencies on the torus, plus a helix pair) and its baseline; 0 and +5 dB, both gains, three seeds | 126 |
| projection | Tier 1's reservoirs, recognition at 2,048 clips and the order task, each read under the fixed and a seeded projection | 432 |

Clean audio is left out of Tiers 2 and 3, because the task saturates there; it is run in Tier 1 and Tier B for the comparison with published results. **Running order:** gate, 1, 2, B, 3, projection. A carrier tier, driving the networks with the band-filtered waveform at 16 kHz, was planned and withdrawn before any of its runs (see the decision log).

## 8. Integrity checks

1. The harness test suite passes.
2. Every g = 0 cell reads exactly 0.100.
3. The spectrogram-only baseline is recorded as runs in the record, for every pathway, noise level, width and size.
4. The order task's baseline is reported against chance (section 7).
5. The projection tier's unseeded cells are compared with Tier 1's, cell for cell: the same runs, re-executed, on the GPU rather than the CPU.

A failed check is fixed and logged before any result it bears on is read.

## 9. Reporting

Every accuracy is reported as its mean over replicates (three seeds; the five folds under Protocol B) with the sample standard deviation and each replicate's value. Every comparison between two arms is paired on condition and replicate, and reported as the mean difference with its standard deviation over replicates and a 95% interval from resampling test clips (2,000 resamples). Seeds share one test set, so their per-clip differences are averaged; folds and order-task pairs each have their own test clips, so theirs are concatenated. No threshold is applied, and nothing is marked as passing or failing (`harness.experiment.summary`).

- Every planned comparison is reported, whichever way it falls.
- Arm parameters and readout coefficients are reported separately, at every width.
- Accuracy against width and against training size is reported as curves for every arm, not as one operating point.
- Diagnostic D1 is reported.
- Published AudioMNIST figures are given as context only. None found so far combines a speaker-disjoint split with added noise, so none is a head-to-head comparison. See `references/audiomnist-prior-art.md`.

## 10. Decisions made before the first run

1. Hypotheses, section 2: the proposed wording, accepted 23 September.
2. The leaky-integrator bank's time constants, section 4: as proposed.
3. The trained baselines: the trained-head protocol only, with the head on the shared statistics.
4. Widths and training sizes, section 6: as proposed, with the native read at 2,048 clips only.
5. The whole-clip spectrogram-only baseline is the primary control (the author, 23 September).
6. **Every arm is read over one fixed window.** Made while building the harness, after a per-clip window was measured to hand an undriven network a duration code (section 1).

Before the first tier, the only runs on this study's data were zero-gain network cells, the measurement behind decision 6, since repeated as the gate tier's diagnostic D1.

## Decision log

Each change to the design after the run began, with its date and reason.

**2026-09-23 20:04 MDT. Results are reported without the section 9 bars.** Every accuracy is reported as its mean over replicates (three seeds; the five folds under Protocol B) with the sample standard deviation and each replicate's value, and every comparison as the paired difference with its standard deviation over replicates and a 95% interval from resampling test clips (`harness.experiment.summary`). No threshold is applied and no verdict is drawn. Whether a difference is a real gain is left to further seeds and to a side-by-side comparison of the arms, to be published as supplementary material. Reason: the author judged a fixed bar such as +3 points arbitrary. **This decision was made after the Tier 1 results had been seen.** The bars are still scored, unchanged, by `harness.confirm.score`, and its Tier 1 verdicts remain in the record (`results/confirmatory/verdicts.json`, commit c68a1fb).

**2026-09-23 21:51 MDT. Terms.** The paper and the documents written for readers now use the terms of the paper's glossary (Appendix A) in place of several used here: the floor is the spectrogram-only baseline, the field the coupled oscillator network, the severed field the uncoupled oscillator network, bank A and bank B the state-matched and width-matched leaky-integrator banks, frozen is untrained, a twin is a matched pair, a coupling law or family is a coupling function, a shape a lattice geometry, designed and uniform natural frequencies are tonotopic and identical, pinning is the restoring strength, the spectral clamp is the coupling ceiling, the envelope pathway is the band-energy pathway. Nothing in the design changes. This document, the code and the run record keep the original labels, which are the runs' identities; `src/README.md` maps each term to its label.

**2026-09-23 22:07 MDT. Tier 4 widened.** The registered size tier ran the coupled oscillator network and its state-matched leaky-integrator bank at 1, 4 and 16 channels of a 16 × 16 lattice, at gain 2 only (36 runs). It now runs lattices of 8 × 8, 16 × 16 and 32 × 32 at 1, 2, 4, 8 and 16 channels (64 to 16,384 states), at gains 1 and 2, and a lattice other than 16 × 16 under two band mappings: its own number of mel bands, one per row, and the registered 16 bands mapped onto its rows (each band on two rows of a 32 × 32 lattice; each row of an 8 × 8 lattice the mean of two bands). Each lattice and mapping also gets the spectrogram-only baseline on exactly the rows its networks are driven with: 630 runs in all. Only the windowed read is recorded, because a 32 × 32, 16-channel network's full feature set would not fit in memory. Everything else about the tier is as registered, and the 16 × 16 runs keep their registered labels. Reason: the registered tier varied only the channel count, on one lattice at one gain, so it could not separate size from lattice shape, gain or input resolution; the author asked for all three. This amendment was made after the Tier 1 results and part of the Tier 2 results had been seen, and before any Tier 4 run.

**2026-09-23 22:44 MDT. Tier 4 withdrawn.** The size tier, registered as 1, 4 and 16 channels of a 16 × 16 lattice at gain 2 and widened in the entry above, is withdrawn from this study before any of its runs. Every network in this study is 4 channels of a 16 × 16 lattice. The question of size, now across lattice sizes, channel counts, band mappings, geometries, coupling functions and input pathways, is large enough to be its own study, which will build on and cite this one. The harness no longer defines the tier, so the code holds only what this study ran.

**2026-09-24 01:13 MDT. A seeded projection, and a tier to read Tier 1's reservoirs under it.** Every registered read reaches a common width through one fixed Gaussian projection, the same draw for every seed, so the spread over seeds leaves out the projection's own variability; on a small real run a different draw moved accuracy at width 192 by a point, the size of the differences the study compares. The spectrogram-only baseline and the trained baselines are 192 wide and never projected, so this touches only the reservoirs. A new tier, `projection`, reruns Tier 1's reservoir arms (both oscillator networks and both banks, both gains, every noise level and seed, recognition at 2,048 training clips and the order task on every pair) and reads each under the fixed projection and under a seeded projection, a draw of its own for each run seed: 432 runs, on the CPU like every tier before it. Its fixed cells must reproduce Tier 1's exactly. Results are reported under both projections. Nothing else changes, and every registered run's output is unchanged (tested). Reason: the author asked for the spread over seeds to include the projection. Decided after the Tier 1 results had been seen.

**2026-09-24 02:21 MDT. Coherence instruments.** Every coupled or uncoupled network run now also records, over its test clips, the global order parameter R, each oscillator's phase-locking value to its own band's drive, the share of oscillators locked above 0.5, and the mean amplitude, each as a mean, a spread and a mean per class. They are diagnostics: nothing reaches the readout, and no cell changes. The pilot measured them and this run did not; Tiers 1 and 2 have no instruments, while Tier B, Tier 3, the carrier tier and the projection tier (which reruns Tier 1's networks) do. Reason: the author wants coherence recorded even where the paper does not discuss it, and paper 03 will evaluate it.

**2026-09-24 03:20 MDT. An original study, not a confirmatory one.** This document and the paper no longer describe the study as confirmatory, registered or frozen: it is an original study that reruns comparisons similar to the pilot's under a new protocol. Every number in the record and the paper comes from this study's runs. With that: the pilot's harness is archived on the `ek/paper-02-pilot` branch, and this study's source holds only the code its runs use; the legacy-reproduction gate, which re-ran pilot runs against the pilot's record, is dropped with it (it ran before the first tier, on ten pilot runs, and reproduced every accuracy it checked; commit f1b2d25); the bank no longer claims bit-identity with the pilot's bank; the pass/fail scorer and its Tier 1 verdicts are removed (they remain in the history at commit c68a1fb); the order task's baseline is reported against chance rather than used to mark pairs invalid; the hypotheses name the comparisons and none is scored; and section 9's bars are replaced by the reporting rules they had already given way to. The carrier gain of 32 is the one design value carried over from the pilot. Reason: the author asked that the study stand on its own data.

**2026-09-24 03:20 MDT. The code and the record use the glossary terms.** Superseding the Terms entry of 21:51: the code and the record now use the glossary's terms in short form. Arm kinds are `baseline`, `network` (with `coupled` true or false), `bank` and `trained`; labels are `baseline`, `coupled-…`/`uncoupled-…` (coupling, geometry, frequencies, `restoring…`, `ceiling…`), `bank-state`, `bank-width` and `trained-<arch>`; the design factors are `coupling`, `geometry`, `frequencies` (random, tonotopic, identical), `restoring` and `ceiling`; the coupling functions are `kuramoto`, `kuramoto-sakaguchi`, `second-harmonic`, `winfree`, `stuart-landau` and `stuart-landau-fixed`; the input pathway field is `pathway`, and the band-energy pathway is the `spectrogram` pathway. The harness moves from `harness/confirm/` to `harness/experiment/` and the record from `results/confirmatory/` to `results/`. Every record entry was migrated in place with its numbers untouched: the regenerated summary matches the one before the migration in all 12,360 accuracies, 1,332 comparisons and the order baseline's intervals, and two recorded runs re-executed through the renamed code reproduce their cells exactly. The model layer (`src/harness/models/`) keeps its own parameter names, which the arms map onto.

**2026-09-24 03:20 MDT. Devices.** Tier B, Tier 3 and the projection tier simulate the reservoirs on the Apple M1 Max GPU, about four to six times faster than its CPU; the readout stays on the CPU. On the GPU an untrained arm's cells equal the CPU's exactly (tested for both networks and the bank), and the projection tier's fixed cells must equal Tier 1's, which ran on the CPU. The trained baselines stay on the CPU on every machine: at about 2,000 parameters in batches of 64 they train one to three times slower on the GPU. Superseding "on the CPU like every tier before it" in the projection entry. The carrier tier, about ten hours on the Mac's GPU, runs on a rented CUDA GPU, and after the paper is drafted the whole study is to be rerun on one to check that it reproduces across devices.

**2026-09-24 04:05 MDT. Two comparisons added to the summary.** Each coupling function at the Tier 1 configuration minus the whole-clip spectrogram-only baseline, pairing Tier 2's cells with Tier 1's baseline (they share test clips, seeds and training draws); and each Tier 3 network on the quadrature pathway minus the same network on the spectrogram pathway, pairing Tier 3 with Tier 2 on the same terms. Neither needs a new run. Reason: Tier 2's Stuart–Landau network read above its Kuramoto match by more than any other factor, which raised whether it reads above its input, and the quadrature networks' distance from the spectrogram pathway is the direct measure of what the pathway changes. Added after the Tier 2 results and the first Tier 3 results had been seen.

**2026-09-24 04:40 MDT. The Apple GPU does not reproduce the CPU exactly.** The Devices entry above expected the GPU's cells to equal the CPU's, from a test on a small synthetic bank. On the real data they do not: the projection tier re-executed Tier 1's reservoir runs on the GPU, and of their 1,296 unseeded cells 1,262 are identical to Tier 1's (per-clip correctness included), 33 differ by one or two test clips, and one differs by 13 clips (0.63 points), where a slightly different validation accuracy made the readout choose a penalty of 0.001 rather than 0.01. The two devices round the reservoir's integration differently, and the discrete penalty choice can amplify that. Tier B, Tier 3 and the projection tier are reported as run, on the GPU, and the agreement is reported beside them (`projection_reproduces_tier1` in the summary). The seeded-minus-fixed comparisons of the projection tier are unaffected, since both sides ran on the GPU.

**2026-09-24 04:35 MDT. The carrier tier stays on CUDA.** Started on the Apple GPU while it was idle, the carrier tier's first runs logged Metal command-buffer errors ("victim of GPU error/recovery") within minutes. PyTorch does not raise on these, so a run could record the output of an incomplete computation. The run was stopped before any carrier run was recorded. On the CPU the tier would take about 18 hours, so it runs on a CUDA GPU as first planned.

**2026-09-24 05:05 MDT. The carrier tier withdrawn.** The carrier tier (48 runs: the quadrature diagonal plus the two Stuart–Landau functions, driven by each band's filtered waveform at 16 kHz, with its baseline and the state-matched bank) is withdrawn from this study before any of its runs, and its code is removed from this study's source. Driving at the sample rate means 16,000 integration steps a clip rather than 61, about ten hours of GPU time on the Mac, whose GPU faulted on it, or a rented CUDA GPU; and the question it asks, whether a network does anything useful with the waveform itself, is the one a physical oscillator array raises, which the companion study of size and layout takes up with the pathway benchmarked on CUDA. The quadrature tier already tests a phase-referenced drive, and its networks read near chance. The study therefore compares two input pathways, the spectrogram and the quadrature. Reason: the author judged the cost, before the paper deadline and within its page limit, larger than what the result could add here.

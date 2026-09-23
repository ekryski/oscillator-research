# Registration: confirmatory run for Paper 02

**Status: FROZEN at 2026-09-23 03:53 MDT, in the commit that wrote this line. No registered run started before it.**

Freezing means one commit that changes the status line above to FROZEN, made and pushed before any registered run starts, so that the hosting service's timestamp corroborates the local one. After freezing, the body is not edited: deviations go in the change log at the end, dated, with the reason.

Everything below is implemented in `src/harness/confirm/` and tested in `src/tests/test_confirm_*.py`; where this text and the code could disagree, the code's tier definitions in `harness/confirm/plan.py` are the plan.

## 1. What this is

A confirmatory replication. An exploratory phase (August 2026, 1,940 runs, in a separate development repository whose commits timestamp its own pre-registrations) produced the design and the hypotheses. This run fixes the decision bars first and then measures. Because the bars are set after the exploratory results were seen, the paper describes them as confirmatory and does not present them as blind.

The exploratory phase also showed five things the design below exists to correct:

- **Readout width.** The field was read with 16,384 features and its no-dynamics floor with 192, so the ridge had 163,840 coefficients for one and 1,920 for the other. The one width-matched read in that record put the field below its floor in 936 of 936 runs.
- **Featurization.** The arms were not read alike. The conventional networks' features included their last hidden state, which carries temporal order directly; the field's and the floor's did not.
- **The read window.** Each arm was read over its own clip's length, while the floor was read over the whole padded clip. Measured before freezing, with no drive (so no hypothesis was tested): an undriven field read over each clip's own length classifies spoken digits at 16.6 to 18.0% against a chance of 10%, because an oscillator keeps rotating and statistics over a span encode the span's length, which in speech carries the digit. The same field read over a fixed window reads exactly 10.0%. That per-clip window handed the field a duration code the floor did not get in the same form.
- **Seeds.** The 1,800-run factorial was run at one seed.
- **Floors.** The floors were printed to a terminal and never written to the record.

## 2. Hypotheses

The direction and the hypotheses are the author's; the wording below was proposed on 18 September and accepted on 23 September.

- **H1, dynamics.** A frozen oscillator field adds accuracy over its front end alone when both are read at the same width.
- **H2, oscillation.** That gain exceeds the gain from a non-oscillating bank of leaky integrators of the same size.
- **H3, coupling.** The coupled field reads higher than the same field with coupling severed.
- **H4, design.** Coupling law, lattice geometry, natural-frequency structure, pinning and spectral clamp each move accuracy.
- **H5, readout.** Readout width has a large effect on accuracy, for every arm.
- **H6, memory.** A frozen field reads temporal order through an order-free readout, and does so better than a leaky bank.

## 3. Data

AudioMNIST, 30,000 recordings, 60 speakers, 10 digits, 50 repetitions, resampled to 16 kHz by the exploratory bank's own per-recording function. The confirmatory bank keeps all 50 repetitions; the exploratory bank kept 20, and every one of its 12,000 clips is bit-identical inside the new bank (checked before freezing, as was the exploratory bank against the one resonant built in August).

**Protocol A, for every verdict.** Speakers 1 to 48 train (24,000 clips), 49 to 60 test (6,000). **The test set is the whole test pool**, in one fixed order, identical for every arm, seed and condition; the exploratory phase scored 512 clips drawn with replacement. A seed permutes the training pool, and the training sets are the first 2,048, 8,192 and 24,000 clips of that permutation, so each contains the one before it.

**Protocol B, for comparison with published results only.** The five speaker folds of Becker et al., copied verbatim from `preprocess_data.py` in the corpus: three folds train (18,000 clips), one validates (6,000), one tests (6,000), clean audio, one run per fold. No design verdict is read from Protocol B.

**Noise.** Clean, 0 dB and +5 dB. The harness's convention is the reverse of the usual one: noise amplitude is speech RMS times 10^(dB/20), so 0 dB is noise at speech-equal power and +5 dB is louder noise, drawn over the whole padded window. Each clip's noise comes from a generator seeded by the clip's identity and the level, so a clip sounds the same in every arm, seed, tier and process. Seeds change which clips a readout is fitted on and an arm's own random draws, never the clips themselves. Front-end rows are computed once per drive and level and memory-mapped by every run.

**Gain.** g = 1 and g = 2 for every frozen dynamical arm. The floor and the trained networks read the rows as they are, so gain does not apply to them. g = 0 removes the drive and runs as a sanity cell per frozen arm.

## 4. Arms

Every arm sits in the same slot: front end, then the arm, then the shared readout.

| arm | what it is | states | stored parameters |
|---|---|---|---|
| floor | the arm slot left empty | 0 | 0 |
| field | coupled oscillator field, 4 channels on a 16 by 16 lattice | 1,024 | 2,048 |
| severed field | the same field, same seed, with its coupling kernel set to zero | 1,024 | 2,048 (1,024 in effect) |
| leaky bank A | independent leaky integrators, matched to the field in states and parameters | 1,024 | 2,048 |
| leaky bank B | the same, matched to the field in exposed signals | 2,048 | 4,096 |
| GRU, TCN, CNN, transformer, S4D | trained conventional networks | | about 2,000, trained |

**The field.** Where a tier does not vary it: Kuramoto coupling, torus, random natural frequencies, pinning 0.3, spectral clamp 1, the configuration the exploratory phase used for its gates and its order task. Each seed draws the kernel and natural frequencies exactly as the exploratory runner did (tested), so seed 0 is the exploratory field.

**The leaky bank.** Each unit follows x ← (1 − a)·x + a·tanh(g_in·g·u). Band r drives the units of row r, the routing the field uses. The time constants are one fixed log-spaced schedule from one hop frame (16 ms) to one clip (1 s), laid out so that every band is read at every time scale, and converted to a leak rate at the drive's own frame rate. The input gains are drawn N(1, 0.1²), the distribution the field draws its natural frequencies from. Only the gains vary with the seed, so this arm's seed-to-seed spread is small by construction. Nothing about the schedule is tuned: this is a falsification control, not a claim that the design is optimal. Bank A matches the field's states and parameters and is half as wide at native width, because the field exposes two signals per state; bank B matches the native width at twice the parameters.

**The conventional networks.** Trained end to end with a learned linear head on the shared statistics of section 5 (four windows for recognition, the whole span for the order task), so each network is trained for the read it is judged by and cannot hand its head its last state. The recipe is the exploratory trained-head one: AdamW at 3e-3 annealed to 3e-4 by cosine, 30 epochs, batches of 64, gradients clipped at 1, the last epoch kept. A network that does not cut its loss by 20% is recorded as an optimization failure. Each network is trained at each training size and read by the shared ridge at that size; its own head's accuracy is recorded as a secondary number.

## 5. The read

One function reads every arm: over a window, or over each of four equal windows, it computes three statistics per signal, the mean, the standard deviation, and the mean absolute frame-to-frame difference. Arms differ only in the signals they expose: the front-end rows (floor), sin θ and cos θ per oscillator (field; x and y for Stuart-Landau cores), the unit states (leaky banks), the hidden trajectory (networks).

**Every arm is read over one fixed window, the same frames for every clip:** from the end of the 16-frame integrator warm-up to the end of the padded clip (frames 16 to 61 for recognition, 16 to 147 for the order task). An arm with no input therefore reads exactly chance, and everything a read carries arrives through the arm's response to the input. **The floor is also read from the first frame, and that whole-clip floor is the primary control**, because it is the harder one: it is given everything the field was driven with, including the onset the field can carry only as memory. The floor over the arms' own window is reported as a secondary.

**No statistic may depend on an endpoint.** That excludes the last state, and the signed mean of a first difference, which telescopes to the last value minus the first.

**Secondary read S1, the field's rotation rate.** The read above plus two signed rotation-rate features per oscillator and window, the mean cos Δθ and sin Δθ between consecutive frames: the oscillator's natural observable, which the generic read drops. It is recorded beside the primary read for every field arm, labelled as the read that favours the field, since no other arm has an analogue.

**Diagnostic D1, the exploratory window.** The field at g = 0, 1 and 2, 0 dB, three seeds, read over each clip's own length as the exploratory harness read it, to measure how much that window inflated the exploratory field reads. Not a hypothesis test.

## 6. The readout, and the two ablations on it

For each training size, an arm's native features are standardized with that training set's own statistics (a feature constant over the training set gets weight zero), brought to each common width by one fixed seeded Gaussian projection (each narrower width is the leading columns of the widest, so a wide read contains the narrow one), and classified by a closed-form ridge, one hyperplane per class, with its penalty chosen from {0.001, 0.01, 0.1, 1} (scaled by the number of fitted clips) on the last eighth of the training set, or on the validation fold under Protocol B. The ridge is the exploratory one, solved in whichever of its two equivalent forms is smaller; it is tested to make the same penalty choice and the same prediction on every clip as the exploratory code.

**Width.** 192, 1,024 and 4,096, plus native. 192 is the floor's native width under the envelope drive. An arm is never read wider than its native width; at or above it, it is read unprojected.

**Training-set size.** 2,048, 8,192 and 24,000 clips, crossed with width, because a wide ridge on few clips and a wide ridge on many are different estimators. The native read is fitted at 2,048 clips only: at 24,000 its system is too large to solve per run.

**The primary read, for every verdict: width 192, 2,048 training clips, four windows (recognition) or the whole span (order).** Everything else is secondary and is reported whatever it shows.

## 7. Tasks and tiers

**Recognition.** Ten-way digit classification.

**Temporal order.** Two recordings joined behind the exploratory task's 272 ms silent leader and across a 100 ms gap, a then b against b then a, on five digit pairs (3-7, 1-8, 2-5, 4-9, 0-6). Per pair, 2,048 training clips drawn per seed from the training speakers and one fixed set of 2,048 test clips from the test speakers, with alternating labels so the classes are exactly balanced. Scored on the whole-span read only. **Gate:** for each pair and noise level, the three-seed mean of each floor read (whole clip, and arms' window) must have a 95% bootstrap interval over test clips that contains 0.5, or that pair is invalid at that level and is not scored. The first exploratory order design failed this check and was repaired; the paper names that failure.

| tier | what | runs |
|---|---|---|
| gate | the g = 0 cells for the four frozen arms, and diagnostic D1 | 13 |
| 1, arms | floor, field, severed field, bank A, bank B at every noise level, gain and seed, at all three training sizes; the five networks at every size; the same arms on the order task, five pairs | 846 |
| 2, design | coupling law (4 phase families; the 2 Stuart-Landau cores on the torus only), lattice geometry (6), natural-frequency structure (random, designed, uniform), pinning (0.3, 0.1), clamp (1, 0.5); 0 and +5 dB, both gains, three seeds; primary size; four-window read with and without rotation rates | 3,744 |
| B, Becker | every Tier 1 arm under Protocol B, both gains for the dynamical arms | 70 |
| 3, quadrature | a diagonal (4 phase families by random and designed frequencies on the torus, plus a helix pair) and its floor; 0 and +5 dB, both gains, three seeds | 126 |
| 4, size | the field and bank A at 1, 4 and 16 channels (256, 1,024, 4,096 states, same lattice and front end); 0 and +5 dB, g = 2, three seeds | 36 |
| 3c, carrier | the carrier diagonal (the quadrature diagonal plus the two Stuart-Landau cores), its floor, and bank A at the sample rate; 0 dB, g = 32 (the exploratory calibrated carrier gain), three seeds | 48 |

Clean audio carries no design verdict (Tiers 2 to 4), because the task saturates there; it is run in Tier 1 for the comparison with published results. **Running order:** gate, 1, 2, B, 3, 4, 3c. Tier 4 and the carrier tier run only if compute allows before the paper deadline; whether they run is decided by time, not by results, and any that do not run are reported as not run.

## 8. Integrity gates

1. The harness test suite passes.
2. **Legacy reproduction.** Six exploratory runs (a phase core, an amplitude core, a designed and a quadrature cell, an order run, a trained network), re-run with the exploratory harness from the configurations the record stored, reproduce every recorded accuracy exactly. Continuous diagnostics (ridge margins, locking instruments) are reported with their largest difference; they move in the last digits between library versions and no verdict reads them. Before freezing, three of the six were re-run and matched exactly in every accuracy.
3. Every g = 0 cell reads exactly 0.100.
4. The floors are recorded as floor runs in the confirmatory record, for every drive, noise level, width and size.
5. The order-task floor gate of section 7.

A failed gate stops the scoring of whatever it guards. It is fixed and logged before anything it guards is scored.

## 9. Decision bars

All in accuracy points, at the primary read. **A bar is met only if the three-seed mean clears it and all three seeds agree in sign.** Every comparison carries a 95% bootstrap interval over test clips.

Each hypothesis is scored separately at each discriminating condition: 0 dB and +5 dB, at g = 1 and g = 2 where gain applies. It is **supported** if its bar is met at every discriminating condition, **refuted** if it is missed at every one, and **mixed** otherwise, with the conditions named. Clean audio is reported and carries no verdict.

| hypothesis | comparison | bar |
|---|---|---|
| H1 | field minus the whole-clip floor (the floor over the arms' window reported beside it) | +3 |
| H2 | field minus bank A, and field minus bank B | +3 |
| H3 | field minus severed field | +3 |
| H4 | a shape minus its torus twin | +3 |
| H4 | designed minus randomized natural frequencies | +5 |
| H4 | a coupling family minus its Kuramoto twin | ±3 |
| H4 | a drive over its own floor | +5 |
| H5 | width 4,096 minus width 192, per arm | +5 |
| H6 | field order accuracy, on at least 3 of 5 valid pairs | 0.60 or above |
| H6 | field minus bank A, order task | +3 |

For H4, a factor level's effect is the mean over its twins, the runs identical to it in every other factor, of the paired difference. The +3 bars of H1 to H3 and H6 and the +5 bar of H5 were proposed for this registration; the H4 bars and the 0.60 bar are the exploratory phase's. The exploratory coherence bar, a Spearman correlation of −0.3 or below in two of three conditions, is carried over as a pre-specified analysis of these runs, not as an experiment.

## 10. Reporting commitments

- Every registered comparison is reported, whichever way it falls, including every bar not met.
- Arm parameters and readout coefficients are reported separately, at every width.
- Accuracy against width and against training size is reported as curves for every arm, not as one operating point.
- Diagnostic D1 is reported beside the exploratory numbers it bears on.
- Published AudioMNIST figures are given as context only. None found so far combines a speaker-disjoint split with added noise, so none is a head-to-head comparison. See `references/audiomnist-prior-art.md`.

## 11. Decisions

Resolved before freezing:

1. Hypotheses, section 2: the proposed wording, accepted 23 September.
2. Bars, section 9: as proposed, accepted 23 September.
3. The leaky bank's time constants, section 4: as proposed.
4. The conventional networks: the trained-head protocol only, with the head on the shared statistics.
5. Widths and training sizes, section 6: as proposed, with the native read at 2,048 clips only.
6. Tiers 3 and 4: run in the order given; Tier 4 and the carrier tier as compute allows.
7. The paper names the first order-task design that failed its blindness check.
8. The whole-clip floor is the primary control (the author, 23 September).
9. **Every arm is read over one fixed window.** Made while building the harness, after the per-clip window was measured to hand an undriven field a duration code (section 1). It replaces the per-clip masking the 18 September draft described, and is flagged to the author.

Before freezing, the only runs on the confirmatory data were zero-drive field cells (the section 1 measurement) and three legacy re-runs of exploratory runs whose results were already on record. No driven arm was run under this protocol before this registration was frozen.

## Change log

Empty. Entries are added only after freezing, each with a timestamp and a reason.

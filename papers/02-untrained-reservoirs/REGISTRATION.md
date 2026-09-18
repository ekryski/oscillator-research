# Registration: confirmatory run for Paper 02

**Status: DRAFT. Not frozen. No registered run may start until this is frozen.**

Freezing means one commit that changes the status line above to FROZEN, made and pushed before any registered run starts, so that the hosting service's timestamp corroborates the local one. Anything marked *proposed* below is the author's to confirm or change before that commit. After freezing, the body is not edited: deviations go in the change log at the end, dated, with the reason.

## 1. What this is

A confirmatory replication. An exploratory phase (August 2026, 1,940 runs, in a separate development repository) produced the design and the hypotheses. This run fixes the decision bars first and then measures. Because the bars are set after the exploratory results were seen, the paper describes them as confirmatory and does not present them as blind.

The exploratory phase also showed four things the design below exists to correct:

- The field was read with 16,384 features and its no-dynamics floor with 192, so the ridge had 163,840 coefficients for one and 1,920 for the other. The one width-matched read in that record put the field below its floor in 936 of 936 runs.
- The arms were not featurized alike. The conventional networks' features included their last hidden state, which carries temporal order directly; the field's and the floor's did not.
- The 1,800-run factorial was run at one seed.
- The floors were printed to a terminal and never written to the record.

## 2. Hypotheses

Set by the author. *Proposed wording, to be edited before freezing.*

- **H1, dynamics.** A frozen oscillator field adds accuracy over its front end alone when both are read at the same width.
- **H2, oscillation.** That gain exceeds the gain from a non-oscillating bank of leaky integrators of the same size.
- **H3, coupling.** The coupled field reads higher than the same field with coupling severed.
- **H4, design.** Coupling law, lattice geometry, natural-frequency structure, pinning and spectral clamp each move accuracy.
- **H5, readout.** Readout width has a large effect on accuracy, for every arm.
- **H6, memory.** A frozen field reads temporal order through an order-free readout, and does so better than a leaky bank.

## 3. Data

AudioMNIST, 30,000 recordings, 60 speakers, 10 digits, 50 repetitions, resampled to 16 kHz. The stimulus bank is rebuilt with all 50 repetitions. The exploratory bank kept 20.

**Protocol A, the factorial protocol.** Speakers 1 to 48 train, 49 to 60 test, which gives a 24,000-clip training pool and a 6,000-clip test pool. Training sets are drawn without replacement. **The test set is the whole 6,000-clip pool**, where the exploratory phase scored 512 clips drawn with replacement. The test set is identical for every arm, seed and condition.

**Protocol B, the Becker protocol, for comparison with prior art only.** The five speaker folds of Becker et al., taken verbatim from `preprocess_data.py` in the corpus: three folds train (18,000), one validates (6,000), one tests (6,000), five-fold cross-validation, clean audio. The data protocol is Becker's; the models are this paper's. No design verdict is read from Protocol B.

**Noise.** Clean, 0 dB and +5 dB. The harness's convention is the reverse of the usual one: noise amplitude is speech RMS times 10^(dB/20), so 0 dB is noise at speech-equal power and +5 dB is more noise. Noise is drawn fresh per clip and applied to training and test clips alike.

**Gain.** g = 1 and g = 2 in every sweep. g = 0 removes the drive, so it runs as one sanity cell per arm and must read chance.

## 4. Arms

Every arm sits in the same slot: front end, then the arm, then the shared readout.

| arm | what it is | states | frozen parameters |
|---|---|---|---|
| floor | the arm slot left empty | 0 | 0 |
| field | coupled oscillator field, 4 channels on a 16 by 16 lattice | 1,024 | 2,048 |
| severed field | the same field with coupling set to zero | 1,024 | 2,048 |
| leaky bank A | independent leaky integrators, matched to the field in states and parameters | 1,024 | 2,048 |
| leaky bank B | the same, matched to the field in exposed signals | 2,048 | 4,096 |
| GRU, TCN, CNN, transformer, S4D | trained conventional networks | | about 2,000 trained |

**The leaky bank.** Each unit follows x ← (1 − a)·x + a·tanh(g_in·u). It has one leak rate a and one input gain g_in, both frozen and seed-pinned, and no coupling, oscillation, bias or learned weight. Band r drives the units of row r, the routing the field uses. *Proposed:* leak rates log-spaced so that time constants run from one hop frame (16 ms) to the clip length (about 1 s); input gains drawn as the field's are. The schedule is fixed here and is not tuned: this is a falsification control, not a claim that the design is optimal.

Two banks are registered because the field exposes two signals per state, sin θ and cos θ, and a leaky unit exposes one. Bank A matches the field's states and parameters and is half as wide at native width. Bank B matches its native width and has twice the parameters. Reported side by side.

**The conventional networks.** *Proposed:* the trained-head protocol only, as in `scripts/trained_head_baselines.py`. The exploratory phase found the frozen-probe objective hostile to ReLU feed-forward nets, which makes it a poor reference.

## 5. The featurizer contract

One function reads every arm. Over the span, or over each of four windows, it computes three statistics per signal: mean, standard deviation, and mean absolute frame-to-frame difference. Arms differ only in the signals they expose.

| arm | signals exposed |
|---|---|
| floor | the front-end rows |
| field, severed field | sin θ and cos θ per oscillator |
| leaky banks | the unit states |
| conventional networks | the hidden trajectory |

**No statistic may depend on an endpoint.** That excludes the last state. It also excludes the signed mean of a first difference, which for a linear state telescopes to the last value minus the first and is the same leak by another route.

**Secondary read S1, the field's rotation rate.** The contract above, plus the field's two signed rotation-rate features per oscillator: mean cos Δθ and mean sin Δθ. These are the oscillator's natural observable, and the generic contract drops them. S1 is reported beside the primary read for the field and the severed field, so every result is visible with and without them. No other arm has an analogue, because for a linear state the signed mean difference is the endpoint leak just excluded. S1 is therefore labelled as the read that favours the field.

**Legacy reads** are kept as extra columns. Features are computed after the simulation, so they cost almost nothing and they make the integrity gate in section 8 possible.

## 6. The readout, and the two ablations on it

Closed-form ridge, one hyperplane per class. λ is chosen from {0.001, 0.01, 0.1, 1} on a held-out eighth of the training set, as in the exploratory phase.

**Width.** Every arm's feature vector is reduced by one fixed, seeded random projection to a common width. *Proposed* widths: **192, 1,024 and 4,096**, plus native. 192 is the floor's native width under the envelope drive. An arm is never read wider than its native width; above that it is read at native. The legacy 72-dimension read stays as a column.

**Training-set size.** *Proposed:* **2,048, 8,192 and 24,000** clips, nested. Width and training size are crossed, because a wide ridge on few clips and a wide ridge on many are different estimators.

Frozen arms are simulated once per run on the whole pool, and every width-by-size cell is fitted from those cached features, so the cross is nearly free for them. The conventional networks retrain at each training size.

**The primary read, for every verdict: width 192, 2,048 training clips, windowed.** Everything else is secondary and is reported whatever it shows.

## 7. Tasks and sweeps

**Recognition.** Ten-way digit classification.

**Temporal order.** Two-digit sequences with identical unordered content, a then b against b then a, on five digit pairs. Scored on the full-span read only, never the windowed one. **Gate:** the floor must read chance on a pair, within its bootstrap interval, or that pair is invalid and is not scored. The first exploratory design failed this check and was repaired; the gate is registered so a repeat is caught before any claim.

Tiers are in running order. Tier 1 decides the paper's central question and runs first.

| tier | what | runs | estimated cost |
|---|---|---|---|
| 1. arms | floor, field, severed field, bank A, bank B on Kuramoto and torus; 3 noise by 2 gain by 3 seeds; whole pool; all widths and sizes; recognition and order | about 90 frozen, plus 5 networks by 3 sizes by 18 cells | about 10 core-hours frozen; networks small |
| 2. design | coupling law by geometry by frequency structure by pinning by clamp, envelope drive; 3 noise by 2 gain by 3 seeds | about 5,600 | about 140 core-hours |
| 3. drives | quadrature, reduced to a registered diagonal since it read near chance throughout the exploratory phase; carrier, the 14-cell diagonal by 3 seeds | about 100 | carrier about 100 core-hours on CPU; GPU speedup unmeasured |
| 4. size, *proposed* | field and bank A at 256, 1,024 and 4,096 states | about 50 | grows with size |
| B. Becker | every Tier 1 arm under Protocol B | 5 folds by arms | small |

Costs scale the exploratory wall times linearly with clip count: a frozen envelope run took a median 30 s for 2,560 clips, a carrier run 2.4 hours, a network run 24 s. They are estimates. Envelope and quadrature sweeps are CPU-bound and want many cores. Carrier integrates at 16 kHz and is the one place a GPU may pay off; the harness accepts `--device cuda` and the speedup is measured before Tier 3 is committed to.

## 8. Integrity gates, before any scored run

1. The harness test suite passes.
2. **Legacy reproduction.** A fixed sample of exploratory cells, re-run under the exploratory configuration, matches the committed record bit for bit. This shows the harness that produces the new numbers is the one that produced the old.
3. Every g = 0 cell reads chance.
4. The floors are written to `results/baselines/floors.json` for every drive, noise level and width.
5. The order-task floor gate of section 7.

A failed gate stops the run. It is fixed and logged before anything is scored.

## 9. Decision bars

All in accuracy points, at the primary read. **A bar is met only if the three-seed mean clears it and all three seeds agree in sign.** Every comparison carries a 95% bootstrap interval over test clips.

| hypothesis | comparison | bar | source |
|---|---|---|---|
| H1 | field minus floor | +3 | *proposed* |
| H2 | field minus bank A, and minus bank B | +3 | *proposed* |
| H3 | field minus severed field | +3 | *proposed* |
| H4 | a shape against its torus twin | +3 | exploratory bar |
| H4 | designed against randomized frequencies | +5 | exploratory bar |
| H4 | a coupling family against Kuramoto | ±3 | exploratory bar |
| H4 | a drive over its own floor | +5 | exploratory bar |
| H5 | width 4,096 minus width 192, per arm | +5 | *proposed* |
| H6 | order accuracy, on at least 3 of 5 valid pairs | 0.60 or above | exploratory bar |
| H6 | field minus bank A, order task | +3 | *proposed* |

The exploratory coherence bar, a Spearman correlation of −0.3 or below in two of three conditions, is carried over as a pre-specified analysis of these runs, not as an experiment.

## 10. Reporting commitments

- Every registered comparison is reported, whichever way it falls, including every bar not met.
- Arm parameters and readout coefficients are reported separately, at every width.
- Accuracy against width and against training size is reported as curves for every arm, not as one operating point.
- Published AudioMNIST figures are given as context only. None found so far combines a speaker-disjoint split with added noise, so none is a head-to-head comparison. See `references/audiomnist-prior-art.md`.

## 11. Open decisions for the author, before freezing

1. The wording of the hypotheses in section 2.
2. The five *proposed* bars in section 9.
3. The leaky bank's time-constant range in section 4.
4. Trained-head only for the conventional networks, section 4.
5. The widths and the training sizes in section 6.
6. How much of Tier 3 to run, and whether to run Tier 4.
7. Whether the paper names the first order-task design that failed its blindness check. The gate in section 7 is written on the assumption that it does.

## Change log

Empty. Entries are added only after freezing, each with a timestamp and a reason.

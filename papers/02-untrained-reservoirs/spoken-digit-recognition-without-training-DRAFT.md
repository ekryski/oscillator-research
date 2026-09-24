# Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Frozen Oscillator Fields

## Abstract

Coupled-oscillator networks are returning as a machine-learning substrate, on the premise that oscillator physics turns signals into useful structure. Yet the ablations that would isolate what the oscillator dynamics contribute are rarely run. We run them for untrained oscillator fields, using spoken-digit classification on AudioMNIST as the measure. A small frozen oscillator field is read by a closed-form linear readout and compared with trained conventional networks matched in parameters and with two controls: a non-oscillating bank of leaky integrators with the same states and parameters, and the same readout fitted directly to the field's input with the field removed. Every experimental arm's state is summarized by the same time-pooled statistics and classified at the same number of features, so that readout capacity cannot pass for dynamics. A pre-registered confirmatory sweep then varies coupling law, lattice geometry, input pathway, readout width and training-set size, on digit recognition and on a temporal-order task that order-blind readouts cannot solve. The design separates three explanations for a frozen field's accuracy: its input representation, the width of its readout, and the oscillator dynamics themselves. It gives studies of trained oscillator dynamics an untrained baseline to compare trained results against and insights into which factors contribute to oscillator effectiveness on time series tasks.

## 1 Introduction

Every sense we have transduces a physical signal: light, sound, heat, touch and pressure arrive as waves, and taste and smell as chemistry. If neurons evolved to encode such stimuli, a first-principles reading suggests that models closer to biological computation might be built from the physics of waves and oscillation rather than from statistics alone, and the oscillation and synchronization recorded in neural populations as they track speech ([Assaneo et al. 2020](https://doi.org/10.1038/s41562-020-00962-0); [Doelling et al. 2023](https://doi.org/10.1371/journal.pcbi.1011669); [Pittman-Polletta et al. 2021](https://doi.org/10.1371/journal.pcbi.1008783); [Dogonasheva et al. 2026](https://doi.org/10.1016/j.neunet.2025.108194)) are consistent with neurons acting, in part, as coupled oscillators. We do not test that conjecture here. It motivates a narrower question: where are the useful boundaries of oscillatory neural networks, and what do their dynamics contribute?

A recent survey of oscillator networks in machine learning ([Kryski 2026](https://doi.org/10.2139/ssrn.7445198)) found the ablations that would answer that question largely missing: the coupling term had been removed as a single variable only three times, no study compared coupling functions, no study varied the lattice geometry or its boundaries as a single variable at a matched budget, and none contrasted two ways of injecting the input under controls. We address those gaps on one task, spoken-digit recognition, where automatic speech recognition began ([Davis et al. 1952](https://doi.org/10.1121/1.1906946)) and the one task on which an oscillator has run in hardware ([Torrejon 2017](https://doi.org/10.1038/nature23011)). An untrained coupled oscillator network, read by a linear readout, is tested under six coupling functions and six lattice geometries, and compared with the same network uncoupled, with leaky-integrator banks of the same size, with the readout applied to its input alone, and with five trained networks of the same parameter count. Results on AudioMNIST ([Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038)) are also placed beside the published figures on its own speaker folds.

## 2 Background

Coupled oscillators are among the most studied systems in physics: populations of them synchronize, form clusters, lock to a drive or split into coherent and incoherent domains, depending on how they are coupled and arranged ([Pikovsky et al. 2001](https://doi.org/10.1017/CBO9780511755743); [Strogatz 2000](https://doi.org/10.1016/S0167-2789%2800%2900094-4); [Kuramoto & Battogtokh 2002](https://arxiv.org/abs/cond-mat/0210694); [Abrams & Strogatz 2004](https://arxiv.org/abs/nlin/0407045)). Machine learning has used them two ways. Untrained, as reservoirs read by a trained linear readout ([Jaeger 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf); [Maass 2002](https://doi.org/10.1162/089976602760407955); [Tanaka 2019](https://doi.org/10.1016/j.neunet.2019.03.005)): a spintronic oscillator recognized spoken digits in hardware ([Torrejon 2017](https://doi.org/10.1038/nature23011)), though much of that accuracy came from its cochlear front end ([Abreu Araujo et al. 2020](https://doi.org/10.1038/s41598-019-56991-x)). Trained, as layers whose coupling or frequencies are learned: coupled oscillatory recurrent networks ([Rusch 2021](https://openreview.net/forum?id=F3s69XzWOia)), Neural Wave Machines on an oscillator lattice ([Keller & Welling 2023](https://proceedings.mlr.press/v202/keller23a.html)), AKOrN ([Miyato 2025](https://openreview.net/forum?id=nwDRD4AMoN)), WONN ([Dai & Song 2026](https://arxiv.org/abs/2605.20922)), FSN ([Nunley 2026](https://arxiv.org/abs/2606.18694)) and Un-0 ([unconv.ai 2026](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/)) among them. Each adopts one coupling function and one arrangement, and the survey found no benchmark any two share ([Kryski 2026](https://doi.org/10.2139/ssrn.7445198)). Here a single task is held fixed and the coupling function, the geometry and the input are varied across it, on untrained networks, so that each can be compared with controls before any training enters.

### 2.1 Coupling functions

Each oscillator's phase θᵢ evolves as θ̇ᵢ = ωᵢ + couplingᵢ + g·uᵢ − λ sin θᵢ: its natural frequency, the coupling from the other oscillators of its channel, the input and a restoring pull toward phase 0. The six coupling functions differ only in the coupling term. Four are phase models that machine-learning systems have built on or that physics offers as controlled departures from Kuramoto's; two are Stuart–Landau amplitude-phase oscillators, with and without their amplitude. The survey found no study that compares coupling functions, so the choice of one over another "remains a convention rather than a finding" ([Kryski 2026](https://doi.org/10.2139/ssrn.7445198)). Appendix D draws how each pushes an oscillator.

| coupling function | coupling term | what it adds | in machine learning |
|---|---|---|---|
| **Kuramoto** | Σⱼ Kᵢⱼ sin(θⱼ − θᵢ) | pulls each oscillator toward the others' phases: the minimal model of synchronization ([Kuramoto 1975](https://doi.org/10.1007/BFb0013365)), and the reference | trained all-to-all in Un-0 ([unconv.ai 2026](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/)); a D-dimensional generalization in AKOrN ([Miyato 2025](https://openreview.net/forum?id=nwDRD4AMoN)) |
| **Kuramoto–Sakaguchi** | Σⱼ Kᵢⱼ sin(θⱼ − θᵢ − α), α = π/4 | a phase lag that breaks the pull's symmetry and admits partial coherence ([Sakaguchi & Kuramoto 1986](https://doi.org/10.1143/PTP.76.576); [Abrams & Strogatz 2004](https://arxiv.org/abs/nlin/0407045)) | with Daido harmonics and a delay in FSN ([Nunley 2026](https://arxiv.org/abs/2606.18694)) |
| **Second harmonic** | Kuramoto plus β Σⱼ Kᵢⱼ sin 2(θⱼ − θᵢ), β = 0.5 | favours two-cluster states, pairs in phase or in opposition ([Daido 1992](https://doi.org/10.1143/ptp/88.6.1213); [Hansel et al. 1993](https://doi.org/10.1103/PhysRevE.48.3470)) | none recorded |
| **Winfree** | −sin θᵢ Σⱼ Kᵢⱼ (1 + cos θⱼ) | separates an oscillator's sensitivity, set by its own phase, from its influence, set by its neighbour's ([Winfree 1967](https://doi.org/10.1016/0022-5193%2867%2990051-3)) | generalized, with attention-defined coupling, in WONN ([Dai & Song 2026](https://arxiv.org/abs/2605.20922)) |
| **Stuart–Landau** | Σⱼ Kᵢⱼ (zⱼ − zᵢ), z = x + iy | amplitude as well as phase: each oscillator relaxes to a cycle of unit radius ([Stuart 1960](https://doi.org/10.1017/S002211206000116X); [Aranson & Kramer 2002](https://doi.org/10.1103/RevModPhys.74.99)) | one graph network |
| **Stuart–Landau, fixed amplitude** | the same, amplitude held at 1 | the phase-only limit, which reduces to Kuramoto: separates what the amplitude adds | none recorded |

Table: The six coupling functions. Kᵢⱼ is the kernel weight between oscillators i and j, positive or negative, and each term is summed over the other oscillators of the channel. Machine-learning uses are those recorded by the survey of [Kryski 2026](https://doi.org/10.2139/ssrn.7445198).

### 2.2 Lattice geometries

Every channel stores its 256 oscillators as a 16 × 16 grid whose row r is driven by mel band r. A lattice geometry changes only which oscillators are neighbours, that is, how the grid's edges are glued; each of a network's four channels is a separate copy of the same shape, and the kernel, the parameters and the oscillators are the same under every geometry, so a comparison between geometries varies one thing. The six vary the number of wrapped axes (sheet, cylinder, torus), the dimension (the cube, which shortens the paths between oscillators) and how frequency is laid out: the cylinder and the sphere leave the frequency axis open, as the cochlea does, and the helix puts bands an octave apart next to each other, as the pitch helix of music perception does ([Shepard 1982](https://doi.org/10.1037/0033-295X.89.4.305)). Oscillator lattices are a long-studied setting in physics ([Sakaguchi et al. 1987](https://doi.org/10.1143/PTP.77.1005)), and machine-learning systems couple oscillators all-to-all ([unconv.ai 2026](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/)), through attention, or locally on a lattice ([Keller & Welling 2023](https://proceedings.mlr.press/v202/keller23a.html); [Miyato 2025](https://openreview.net/forum?id=nwDRD4AMoN)). Geometry has been varied in four of the systems the survey covers, "and in none of them as a single variable", and it found no work that varies coupling range or boundary conditions at a matched budget ([Kryski 2026](https://doi.org/10.2139/ssrn.7445198)).

| geometry | how the grid is glued |
|---|---|
| **Torus** | Both axes wrap: the top row is coupled to the bottom, and the left column to the right. The reference geometry. |
| **Cylinder** | Columns wrap, rows do not: the frequency axis is open, as in the cochlea, so the highest band is not coupled around to the lowest. |
| **Sheet** | Neither axis wraps, and coupling stops at every edge. With the cylinder and the torus it varies the number of wrapped axes from zero to two. |
| **Helix** | The 256 sites read as one closed ring, one octave (64 sites, four rows) per turn, so an offset of one turn joins bands an octave apart. |
| **Cube** | Each row's 16 columns read as a 4 × 4 slab, giving a 16 × 4 × 4 lattice that wraps on all three axes: the same oscillators, with shorter paths between them. |
| **Sphere** | Rows as latitudes with open poles and columns as longitudes that wrap, each oscillator's influence weighted by the cosine of its latitude: an approximation to a sphere, not exact spherical coupling. |

Table: The six lattice geometries.

![Lattice geometries. Every channel stores its 256 oscillators the same way, as a 16 × 16 grid whose row r is driven by mel band r (colour, lowest band dark). A geometry changes only which oscillators are neighbours, that is, how the grid's edges are glued, and each of a network's four channels is a separate copy of the same shape. For each geometry, left: the grid, with glued edges marked by a shared colour and arrow, open edges dark, one oscillator (star) and its nearest neighbours (dots); right: the shape the gluing makes, with the same oscillator and neighbours. On the torus the highest band's row meets the lowest; on the cylinder and the sphere the frequency axis is open; the helix threads all 256 sites into one closed ring, one octave per turn; the cube folds each row into a 4 × 4 slab and wraps all three axes.](resources/figures/a2-lattice-geometries.png)

## 3 Methods

### 3.1 Experimental design

The study asks six questions. Does an untrained coupled oscillator network carry information for recognizing spoken digits beyond its input alone, beyond a non-oscillating memory of the same size, and beyond the same oscillators uncoupled? How do trained networks of the same parameter count compare? Do the coupling function, the lattice geometry, the natural frequencies, the restoring strength and the coupling ceiling change the answer? Does the way the audio enters the network matter? How do the answers change with the readout's width and the amount of training data? And does the network carry the order of events, on a task its input's summary statistics cannot solve? The study was registered before its first run; the registration, its dated change log and every run's record are in the supplementary material.

It ran in seven tiers of 5,279 runs (Appendix B). Every run is one arm, one of the systems compared, at one noise level, input gain and random seed; every oscillator network is 4 channels of a 16 × 16 lattice. Each arm's signals are summarized, brought to a common width and classified by the same linear readout (Section 3.4), each condition is run at three seeds, and every accuracy is measured on speakers held out from training (Section 3.2). The runs used the CPU of an Apple M1 Max (10 cores, 64 GB); the carrier pathway, integrated at the audio sample rate, used a CUDA GPU. Appendix A defines every term.

### 3.2 Task and data

AudioMNIST ([Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038)) holds 30,000 recordings of the ten spoken digits by 60 speakers, 50 of each digit per speaker. Each recording is resampled from 48 kHz to 16 kHz, peak-normalized, trimmed of leading and trailing silence below 1% of its peak, and capped at 1 s. We use it rather than TI-46, the corpus of the oscillator hardware result, because it is open, has four times as many speakers, and has published speaker-disjoint folds.

Every result keeps speakers apart. Under Protocol A, the arms are trained on speakers 1 to 48 (24,000 recordings) and tested on all 6,000 recordings of speakers 49 to 60. A seed fixes a permutation of the training recordings, and its training set is the first 2,048 of them (8,192 and 24,000 for the training-size comparison), so a larger set contains the smaller. Under Protocol B, used only to place results beside published ones, the arms follow the five folds of [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038): each fold splits the 60 speakers into 36 for training, 12 for validation and 12 for testing (18,000, 6,000 and 6,000 recordings), and every speaker is tested in exactly one fold, so a fold plays the part a seed plays under Protocol A.

The order task tests whether an arm carries the order of events. Two digits of a pair (0 and 6, 1 and 8, 2 and 5, 3 and 7, 4 and 9) are spoken one after the other, and the arm must say which came first; each pair has 2,048 training sequences from the training speakers and 2,048 test sequences from the test speakers, balanced between the two orders. The summary statistics of an input read over the whole span do not depend on the order of its frames, so the spectrogram-only baseline reads chance, 50%, and anything above it must come from the arm's own dynamics.

### 3.3 Front end: the mel spectrogram

Every arm hears the audio through the same fixed front end: a log-mel spectrogram of 16 mel bands, computed with a 512-point Fourier transform every 256 samples (16 ms, 62.5 frames per second) and rescaled by a fixed affine map, with nothing learned and no per-utterance normalization. We chose a fixed, basic front end over a learned one ([Zeghidour et al. 2021](https://openreview.net/forum?id=jM76BCb6F9m); [Lostanlen et al. 2023](https://arxiv.org/abs/2307.13821)) so that nothing trained sits before the arms being compared. A learned front end would be fitted differently for each arm, a difference in accuracy could come from it rather than from what follows, and it would favour the trained baselines, which can adapt to a richer input, over the untrained reservoirs, which cannot. Sixteen bands also fit every arm at once: mel band r drives row r of each of the lattice's 16 rows, and every trained baseline takes the same 16 signals as its input.

### 3.4 Readout

Every arm is read the same way. Each of its signals (a band energy, the sine or cosine of an oscillator's phase, a leaky integrator's state, or a trained baseline's hidden unit) is summarized by three statistics, its mean, its standard deviation and its mean absolute change from frame to frame, over each of four equal windows of the same frames for every clip: frames 16 to 61, after a 16-frame warm-up, and frames 16 to 147 as one window for the order task. The arms expose very different numbers of signals, from 16 band energies to 2,048 oscillator signals, so their summaries range from 192 to 24,576 numbers. So that readout capacity cannot pass for dynamics, every arm is brought to the same width: its summaries are standardized with the training set's own statistics and multiplied by a random Gaussian matrix to 192 features (and to 1,024 and 4,096 in the width comparison), a projection that approximately preserves distances between feature vectors ([Johnson & Lindenstrauss 1984](https://doi.org/10.1090/conm/026/737400)). The spectrogram-only baseline and four of the trained baselines are 192 wide and are not projected; the GRU's 216 features are.

A single linear layer then maps the 192 features to ten digit scores. It is fitted by ridge regression, least squares with an L2 penalty solved in closed form, with the penalty chosen from four values on the last eighth of the training clips. We use this readout, the reservoir-computing standard ([Jaeger 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf); [Lukoševičius 2012](https://doi.org/10.1007/978-3-642-35289-8_36)), rather than a learned decoder because it is the weakest reader that can be fitted identically for every arm: whatever it separates must already be in the features. The trained baselines are trained end to end through a learned linear layer on the same statistics and then read by the same readout.

The projection matrix is fixed, one draw used for every run, or seeded, a draw of its own for each run seed. Every result is reported under the fixed projection. Because a single draw leaves the projection's own variability out of the spread over seeds, Tier 1's reservoirs are also read under the seeded projection, and both are reported.

### 3.5 Input pathways

The survey found no study that contrasts two ways of injecting the input under controls ([Kryski 2026](https://doi.org/10.2139/ssrn.7445198)), so the audio drives a reservoir in one of three ways, each adding one kind of information over the last. **Band-energy**: each band energy adds to the rotation rate of the oscillators in its row, θ̇ᵢ += g·Aᵣ(t), the standard reservoir input; every tier uses it unless stated otherwise. **Quadrature**: each band's energy together with the phase of its signal relative to the band's centre frequency, entering as a phase-referenced push g·Aᵣ sin(φᵣ − θᵢ), the injection form of [Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930), so an oscillator can lock to the band's own cycle. **Carrier**: the band-filtered waveform itself, in 16 log-spaced bands from 96 to 1,536 Hz at the full 16 kHz, so the oscillators are driven by the cycles of the sound directly, as a physical oscillator array would be. Each pathway's spectrogram-only baseline reads that pathway's own front end.

### 3.6 Noise protocol

White noise is added to every clip, in training and in test alike, at a level set by the clip's own speech: at 0 dB the noise is as loud as the speech, and at +5 dB it is 5 dB louder, a signal-to-noise ratio of −5 dB. Each clip's noise is drawn from a generator seeded by the clip's identity, so every arm hears exactly the same noisy clip, and clean audio is also reported. We add noise because the clean task nearly saturates: in pilot runs before the study was registered, the untrained arms reached about 94% already with noise at a third of the speech's amplitude, leaving little room to tell designs apart. The two levels were chosen in those pilot runs as the ones that put the strongest untrained arm between 70% and 90%; two keep the design affordable while showing whether an effect holds as the noise grows.

### 3.7 Input gain and the integrator-validity bound

Input gain is the factor that scales the mel spectrogram before it drives a reservoir. It applies only to the reservoirs: the coupled and uncoupled oscillator networks and the leaky-integrator banks. Both of these types of reservoirs are nonlinear, so the strength of the input relative to their own dynamics changes how they respond. For an oscillator, the input competes with its natural frequency, its coupling and the restoring pull; for a leaky integrator, it sets how far into the saturating tanh the input reaches. How hard the input pushes, relative to a reservoir's own dynamics, therefore sets how much of the input its state carries: a weak push barely perturbs the oscillators' own rotation, and a strong one makes them follow the input.

The reservoirs are run at gains 1 and 2. At gain 1 the band energies, rescaled to lie roughly between 0 and 2, push with the same magnitude as the natural frequencies (about 1) and more than the restoring strength (0.3 or 0.1); gain 2 doubles that push. In pilot runs, accuracy moved by less than a point across gains of 0.5, 1 and 2, so these two bracket the working range rather than tune it. The gain is positive because the input is an energy: a negative gain would make louder sound slow the oscillators rather than speed them, a different model rather than a different strength. It is bounded above by the integrator: the network advances in time steps of 0.1, and the phase an oscillator gains from the input in one step must stay below π for the step to be valid ([Hairer et al. 1993](https://doi.org/10.1007/978-3-540-78862-1)), a bound the band-energy pathway reaches near gain 17. The carrier pathway's band-filtered waveform is about 60 times smaller than the band energies (root mean squares of 0.021 and 1.32 at 0 dB), so it runs at gain 32, which brings its push to a comparable size while staying well under its own bound, near gain 75.

The spectrogram-only baseline has no dynamics for gain to act on. Its features are summary statistics of the band energies, and each scales in proportion to the input: doubling the input doubles every mean, standard deviation and mean absolute change, and the readout's standardization divides that factor out exactly, so the baseline at gain 2 would be identical to the baseline at gain 1. The trained baselines learn their own input scale: a fixed gain could be undone by the first layer's weights, so it would relabel the same experiment rather than define a new one. Each trained baseline is therefore trained once per noise level, on the input as it is.

### 3.8 Reporting and uncertainty

No threshold decides a result. Every accuracy is reported as its mean over replicates, with the sample standard deviation and each replicate's value. A replicate is one seed under Protocol A, which sets the arm's random parameters and which training clips it draws, and one fold under Protocol B (Section 3.2). Every comparison between two arms, for example the coupled oscillator network against the spectrogram-only baseline, is paired: both are scored on the same test clips, at the same noise level, gain and seed. It is reported as the mean difference in accuracy, the standard deviation of that difference over seeds, and a 95% interval from a paired bootstrap: the test clips are resampled with replacement 2,000 times, the difference is recomputed on each resample, and the interval holds the middle 95% of those differences.

These measure different sources of variation. The standard deviation over seeds is how much a result moves when the random draws and the training clips change; the bootstrap interval is how precisely the fixed test set measures a difference. Scoring both arms on the same clips cancels the clips every arm finds easy or hard, so the paired interval is narrower than one for two unpaired accuracies, but it covers test-set sampling only. Neither covers the fixed projection's own draw, which is why Tier 1's reservoirs are also read under the seeded projection (Section 3.4). The primary cell for every comparison is width 192, 2,048 training clips and the four-window read (the whole-span read for the order task).

## 4 Results

### 4.1 The network against its controls

The table gives recognition accuracy at the primary cell for the spectrogram-only baseline, the three reservoirs at input gain 1, and the transformer, the trained baseline with the highest accuracy; the figure adds each reservoir at gain 2.

| arm | clean | 0 dB | +5 dB |
|---|---|---|---|
| **Spectrogram-only baseline**: the readout reads the 16 mel band energies directly; no reservoir | 93.6 ± 0.3 | 78.0 ± 0.6 | 71.6 ± 1.6 |
| **Coupled oscillator network** (gain = 1): 1,024 oscillators, untrained | 91.1 ± 0.4 | 77.8 ± 0.8 | 70.7 ± 0.3 |
| **Uncoupled oscillator network** (gain = 1): the same network with its coupling removed | 90.6 ± 0.3 | 76.9 ± 0.5 | 69.7 ± 0.4 |
| **Leaky-integrator bank, state-matched** (gain = 1): 1,024 leaky integrators with the network's states and parameters, untrained | 93.4 ± 0.2 | 73.8 ± 0.5 | 66.6 ± 0.8 |
| **Transformer**: a trained baseline, 1,968 parameters, trained end to end | 96.2 ± 0.4 | 76.0 ± 1.4 | 69.0 ± 1.8 |

Table: Single spoken-digit recognition accuracy (%) on AudioMNIST, mean ± standard deviation over three seeds.

![Recognition accuracy by noise level for each reservoir at input gain 1 (solid) and 2 (hatched), beside the spectrogram-only baseline and the transformer, which take no gain. Mean over three seeds; error bars, one standard deviation; dotted line, chance.](resources/figures/c3-recognition-both-gains.png)

### 4.2 Trained baselines

The five trained baselines, each trained end to end on the same clips and read by the same readout, reached the accuracies below at the primary cell. Training failed, by the registered criterion of not cutting the loss by 20%, in 12 of the TCN's 18 runs with added noise and in 5 of the CNN's; those runs are included, and the spread shows them. With 24,000 training clips at 0 dB, the transformer reached 89.0 ± 0.9%, the S4D 88.1 ± 0.4% and the GRU 84.6 ± 0.9%.

| trained baseline | parameters | clean | 0 dB | +5 dB | failed runs with noise |
|---|---|---|---|---|---|
| GRU | 1,944 | 92.7 ± 3.0 | 68.9 ± 1.9 | 64.0 ± 1.0 | 0 of 18 |
| TCN | 1,948 | 92.6 ± 0.3 | 44.4 ± 29.8 | 10.0 ± 0.0 | 12 of 18 |
| CNN | 2,109 | 92.5 ± 1.5 | 67.5 ± 6.5 | 40.9 ± 27.0 | 5 of 18 |
| transformer | 1,968 | 96.2 ± 0.4 | 76.0 ± 1.4 | 69.0 ± 1.8 | 0 of 18 |
| S4D | 1,840 | 90.5 ± 0.5 | 70.1 ± 1.0 | 63.4 ± 1.3 | 0 of 18 |

Table: The trained baselines: parameters, recognition accuracy (%) at the primary cell as mean ± standard deviation over three seeds, and runs with added noise, over all three training sizes, whose training failed.

### 4.3 Network design: coupling function, lattice geometry and natural frequencies

<!-- TODO: Add our results when they are ready. Don't make judgments. Just the facts. -->

### 4.4 Sensitivity of the canonical physics values

<!-- TODO: Add our results when they are ready. Don't make judgments. Just the facts. -->

### 4.5 Alternative input formats: quadrature and carrier

Rather than just test a mel spectrogram front-end drive we wanted to see what quadrature format and the direct raw input audio carrier signal would produce. The findings are...

<!-- TODO: Add our results when they are ready. Complete the sentence above and explain the results. Don't make judgments. Just the facts. -->

### 4.6 Readout sufficiency: width and training size

<!-- TODO: Add the width (192, 1,024, 4,096) and training-size (2,048, 8,192, 24,000) results from Tier 1 once the seeded-projection tier is in. Coherence is not reported: the pilot runs measured it (phase locking to the drive, the order parameter), but the confirmatory harness does not record those instruments, so the registration's coherence analysis cannot be run on these runs without adding them and rerunning. Decide whether to drop coherence or to add the instruments. -->

### 4.7 Temporal order

<!-- TODO: Add our results when they are ready. Don't make judgments. Just the facts. -->

## 5 Discussion

<!-- TODO: This is the interpretation of the results, where gaps might be and possible future directions. Future directions I think is running on different hardware, more seeds, different readout structure, different training protocols, not parameter matching across different model architectures (maybe we constrained some too much), measuring energy consumption, expanding parameters, more advanced physics functions, more advanced classification tasks or alternative speech detection, synthesis and generation tasks, training process for ANNs we are comparing to, how oscillator field size changes results, whether a trained coupled oscillator improves classification tasks. Can they learn? We think so and now we have some baselines to compare against! Sources of variation to cover here (moved from Section 3.8): the number of seeds, hardware (CPU and GPU floating point differ), the readout's structure and the projection's draw, parameter matching across architectures, and the trained baselines' training recipe, including the CNN and TCN runs whose training failed. -->

## 6 Conclusion

<!-- TODO: Brief conclusion of what was done, results, and future directions. -->

## Reproducibility statement {-}

The code, the registration with its dated change log, and the complete run record (every run's specification, results and per-clip correctness) are in the anonymized supplementary material and will be released with the paper; AudioMNIST is public ([Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038)). From the record alone, one command regenerates every table and figure in this paper, and from the audio, the harness reruns any tier, resuming without repeating a finished run. Every run records its code version, hardware and library versions. The runs reported here used the CPU of an Apple M1 Max, where runs with the same seed are bit-identical; the carrier pathway used a CUDA GPU. We welcome reproductions.

<!-- TODO: the harness runs on the CPU and on CUDA GPUs; Apple Silicon GPU (MPS) support is untested for paper 02. Test it before claiming it here. -->

## AI use statement {-}

This work was carried out by the author working with an AI coding agent, Claude (Anthropic), throughout. The disclosure covers both the uses ICLR requires to be disclosed and those it recommends.

**Implementing methods and running experiments.** Under the author's direction the agent wrote most of the experiment harness, the sweep driver, the scoring code and the test suite, launched and scored runs, and kept the dated experiment logs from which this paper's numbers are drawn.

**Designing experiments and interpreting results.** The author set the research questions and directed the work. The agent contributed to the design of controls, flagged flaws in experimental designs, and wrote first-draft interpretations of results, which the author reviewed and accepted, revised or rejected.

**Writing.** The agent drafted sections of this paper, including the abstract, from the experiment logs, and edited the text for readability; it also helped find and check related literature and built the tooling that produces the paper's formats and checks its bibliography.

**Verification.** The study was registered before its first run, and every deviation since is logged, dated, in the registration's change log. Every number in the paper is regenerated from the committed record by a script rather than transcribed. The harness carries tests for each mechanism the paper relies on, and same-seed runs on the CPU are bit-identical. The author takes responsibility for the final content of this work.

## References {-}

- [Abrams & Strogatz 2004](https://arxiv.org/abs/nlin/0407045): Chimera States for Coupled Oscillators.
- [Abreu Araujo et al. 2020](https://doi.org/10.1038/s41598-019-56991-x): Role of non-linear data processing on speech recognition task in the framework of reservoir computing.
- [Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930): A Study of Locking Phenomena in Oscillators.
- [Aranson & Kramer 2002](https://doi.org/10.1103/RevModPhys.74.99): The world of the complex Ginzburg-Landau equation.
- [Assaneo et al. 2020](https://doi.org/10.1038/s41562-020-00962-0): Speaking rhythmically can shape hearing.
- [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038): AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark.
- [Caranzano et al. 2025](https://arxiv.org/abs/2505.04300): Sparsity is All You Need: Rethinking Biological Pathway-Informed Approaches in Deep Learning.
- [Chandravadia & Imam 2026](https://doi.org/10.1016/j.patter.2026.101563): Neural rhythms as priors of speech computations.
- [Dai & Song 2026](https://arxiv.org/abs/2605.20922): Winfree Oscillatory Neural Network.
- [Daido 1992](https://doi.org/10.1143/ptp/88.6.1213): Order Function and Macroscopic Mutual Entrainment in Uniformly Coupled Limit-Cycle Oscillators.
- [Davis et al. 1952](https://doi.org/10.1121/1.1906946): Automatic Recognition of Spoken Digits.
- [Doelling et al. 2023](https://doi.org/10.1371/journal.pcbi.1011669): Adaptive oscillators support Bayesian prediction in temporal processing.
- [Dogonasheva et al. 2026](https://doi.org/10.1016/j.neunet.2025.108194): Neuro-oscillatory models of cortical speech processing.
- [Hairer et al. 1993](https://doi.org/10.1007/978-3-540-78862-1): Solving Ordinary Differential Equations I: Nonstiff Problems.
- [Hansel et al. 1993](https://doi.org/10.1103/PhysRevE.48.3470): Clustering and slow switching in globally coupled phase oscillators.
- [Jaeger 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf): The "echo state" approach to analysing and training recurrent neural networks.
- [Jaeger et al. 2007](https://doi.org/10.1016/j.neunet.2007.04.016): Optimization and applications of echo state networks with leaky-integrator neurons.
- [Johnson & Lindenstrauss 1984](https://doi.org/10.1090/conm/026/737400): Extensions of Lipschitz mappings into a Hilbert space.
- [Keller & Welling 2023](https://proceedings.mlr.press/v202/keller23a.html): Neural Wave Machines: Learning Spatiotemporally Structured Representations with Locally Coupled Oscillatory Recurrent Neural Networks.
- [Kryski 2026](https://doi.org/10.2139/ssrn.7445198): From Synchronization Physics to Trained Dynamics: A Survey of Oscillator Networks in Machine Learning.
- [Kuramoto & Battogtokh 2002](https://arxiv.org/abs/cond-mat/0210694): Coexistence of Coherence and Incoherence in Nonlocally Coupled Phase Oscillators.
- [Kuramoto 1975](https://doi.org/10.1007/BFb0013365): Self-entrainment of a population of coupled non-linear oscillators.
- [Lostanlen et al. 2023](https://arxiv.org/abs/2307.13821): Fitting Auditory Filterbanks with Multiresolution Neural Networks.
- [Lukoševičius 2012](https://doi.org/10.1007/978-3-642-35289-8_36): A Practical Guide to Applying Echo State Networks.
- [Maass 2002](https://doi.org/10.1162/089976602760407955): Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
- [Miyato 2025](https://openreview.net/forum?id=nwDRD4AMoN): Artificial Kuramoto Oscillatory Neurons.
- [Nunley 2026](https://arxiv.org/abs/2606.18694): Attention as Frustrated Synchronization.
- [Pikovsky et al. 2001](https://doi.org/10.1017/CBO9780511755743): Synchronization: A Universal Concept in Nonlinear Sciences.
- [Pittman-Polletta et al. 2021](https://doi.org/10.1371/journal.pcbi.1008783): Differential contributions of synaptic and intrinsic inhibitory currents to speech segmentation via flexible phase-locking in neural oscillators.
- [Rusch 2021](https://openreview.net/forum?id=F3s69XzWOia): Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies.
- [Saadatmand et al. 2026](https://arxiv.org/abs/2605.27406): A Simple State Space Model Excels at Multivariate Time Series Classification.
- [Sakaguchi & Kuramoto 1986](https://doi.org/10.1143/PTP.76.576): A Soluble Active Rotator Model Showing Phase Transitions via Mutual Entrainment.
- [Sakaguchi et al. 1987](https://doi.org/10.1143/PTP.77.1005): Local and Global Self-Entrainments in Oscillator Lattices.
- [Shepard 1982](https://doi.org/10.1037/0033-295X.89.4.305): Geometrical approximations to the structure of musical pitch.
- [Strogatz 2000](https://doi.org/10.1016/S0167-2789%2800%2900094-4): From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators.
- [Stuart 1960](https://doi.org/10.1017/S002211206000116X): On the non-linear mechanics of wave disturbances in stable and unstable parallel flows Part 1. The basic behaviour in plane Poiseuille flow.
- [Tanaka 2019](https://doi.org/10.1016/j.neunet.2019.03.005): Recent advances in physical reservoir computing: A review.
- [Torrejon 2017](https://doi.org/10.1038/nature23011): Neuromorphic computing with nanoscale spintronic oscillators.
- [unconv.ai 2026](https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/): Introducing Un-0: Generating Images with Coupled Oscillators.
- [Winfree 1967](https://doi.org/10.1016/0022-5193%2867%2990051-3): Biological rhythms and the behavior of populations of coupled oscillators.
- [Zeghidour et al. 2021](https://openreview.net/forum?id=jM76BCb6F9m): LEAF: A Learnable Frontend for Audio Classification.

<!-- appendix -->

## Appendix {-}

## A Glossary

The terms as this paper uses them. The code and the run record keep the labels the registration froze; src/README.md maps each term to its label.

### A.1 The pipeline

| Term | As used in this paper |
|---|---|
| **Front end** | The fixed first stage every arm shares. It turns a clip's 16 kHz waveform into a mel spectrogram: a 512-point Fourier transform every 256 samples (16 ms), 16 mel bands, the log of each band's energy, and a fixed rescaling. Nothing in it is trained or fitted to the data. |
| **Mel band** | One of the 16 frequency ranges the front end splits the audio into, spaced on the mel scale, which follows pitch perception: narrow at low frequencies, wide at high ones. |
| **Band energy** | The log energy in one mel band, one value per frame: the signal the front end produces for that band. |
| **Mel spectrogram** | A clip's 16 band energies together, frame by frame. The spectrogram-only baseline reads it directly; every reservoir is driven by it. |
| **Frame** | One 16 ms step of the front end, 62.5 per second. |
| **Reservoir** | An untrained dynamical system between the front end and the readout: here the coupled and uncoupled oscillator networks and the two leaky-integrator banks. The term is reservoir computing's ([Jaeger 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf); [Maass 2002](https://doi.org/10.1162/089976602760407955)). |
| **Readout** | The reader shared by every arm, and the only fitted part of an untrained arm: one linear layer from 192 features to 10 digit scores (1,930 weights), fitted by ridge regression, that is least squares with an L2 penalty, solved in closed form. The penalty is chosen from four values on the last eighth of the training clips, then the readout is refit on all of them. It is the same for every arm, so it is held constant across comparisons. |
| **Untrained** | Parameters drawn at random once, from the seed, and never updated. In an untrained arm only the readout is fitted. |
| **Input gain** | The factor that scales the input before it drives a reservoir: 1 or 2 (32 for the carrier pathway). It applies only to the reservoirs: the spectrogram-only baseline has no dynamics for it to act on, and a trained baseline learns its own input scale. |
| **Input pathway** | How the audio drives a reservoir. Band-energy: each band energy adds to the rotation rate of the oscillators in its row. Quadrature: each band's energy and phase, so the push depends on where an oscillator is relative to the band's own cycle, a phase-referenced drive of the kind [Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930) analysed. Carrier: the band-filtered waveform itself at 16 kHz, rather than its energy. |

### A.2 Arms and baselines

| Term | As used in this paper |
|---|---|
| **Arm** | One system under comparison, read the same way as every other. The term comes from experimental design. |
| **Spectrogram-only baseline** | The front end and the readout with nothing between them: the readout reads the band energies directly, so it measures what the input alone supports. Read from the first frame, over the whole clip, it is the primary control, because it sees everything a reservoir is driven with; read from frame 16, it sees the frames every other arm is read over. On the quadrature and carrier pathways it reads that pathway's front end instead. |
| **Coupled oscillator network** | The system under test: 1,024 untrained oscillators in 4 channels of a 16 × 16 lattice, coupled within each channel through a coupling kernel, with 2,048 parameters (the kernels and the natural frequencies). Also "the oscillator network". Its reference configuration: Kuramoto coupling, torus geometry, random natural frequencies, restoring strength 0.3, coupling ceiling 1. |
| **Uncoupled oscillator network** | The same network with every coupling kernel set to zero: each oscillator keeps its natural frequency, restoring strength and input, but none acts on another. It isolates what the coupling contributes. |
| **Leaky integrator** | A unit that holds one number x and, at each frame, moves a fraction a of the way toward its input: x ← (1 − a)·x + a·tanh(g_in·g·u), where u is the input, g the input gain, g_in the unit's own input weight, and tanh bounds the input. The old value decays exponentially, or "leaks", so the unit is a running average of its recent input with a time constant set by a; in signal-processing terms it is a first-order low-pass filter. The term is standard. In computational neuroscience it is the leaky-integrator neuron, the leaky integrate-and-fire model without the firing; in reservoir computing it is the unit of leaky-integrator echo state networks ([Jaeger et al. 2007](https://doi.org/10.1016/j.neunet.2007.04.016)), whose a is the "leaking rate" ([Lukoševičius 2012](https://doi.org/10.1007/978-3-642-35289-8_36)). |
| **Leaky-integrator bank** | A reservoir of independent leaky integrators, routed like the oscillator network: mel band r drives every unit in row r. Time constants are spaced logarithmically from 16 ms to 1 s, input weights g_in are drawn from N(1, 0.1²), and no unit is connected to another: nothing rotates and nothing is coupled. State-matched: 1,024 units, the oscillator network's number of states and parameters. Width-matched: 2,048 units, the number of signals the oscillator network exposes (sin θ and cos θ of each oscillator), at twice the parameters. |
| **Trained baselines** | Five conventional networks of 1,840 to 2,109 parameters, trained end to end (AdamW, 30 epochs) through a learned linear layer on the same statistics, then read by the same readout. GRU: a gated recurrent network. TCN: a small causal temporal convolutional network. CNN: two causal one-dimensional convolutions. Transformer: one causal self-attention layer, width 16 with two heads, and a two-layer feed-forward block. S4D: a diagonal state-space model. |
| **Matched pair** | Two runs identical in every factor but the one compared, at the same noise level, input gain and seed. A design factor's effect is the mean difference over its matched pairs. |
| **Seed** | Sets an arm's random parameters and which training clips it draws. Every condition is run at three seeds, and results are reported as their mean with the standard deviation. |

### A.3 The oscillator network

| Term | As used in this paper |
|---|---|
| **Oscillator** | A unit whose state advances around a cycle. A phase oscillator keeps only its phase θ, an angle; a Stuart–Landau oscillator also keeps an amplitude. The readout sees sin θ and cos θ of each. |
| **Natural frequency ω** | The rate at which an oscillator would advance if nothing acted on it. Random: drawn from N(1, 0.1²). Tonotopic: each row set to the centre frequency of the mel band that drives it, with small jitter. Identical: all 1. |
| **Lattice** | The 16 × 16 grid a channel's oscillators sit on. Rows follow frequency: mel band r drives row r, from lowest to highest. |
| **Row** | The 16 oscillators of one channel that a single mel band drives. The only grouping built into the network. |
| **Channel** | One independent 16 × 16 lattice of 256 oscillators, with its own coupling kernel and natural frequencies. Every channel receives the same input, and channels do not act on each other; the network's 4 channels are four differently drawn copies, read side by side (Appendix C). The name follows the channels of a convolutional network, where each channel likewise has its own kernel. |
| **Coupling kernel** | A channel's table of coupling weights: a 16 × 16 array whose entry at an offset of so many rows and columns sets how strongly an oscillator is acted on by the one at that offset. The same table applies at every site, so the coupling is a convolution, and "kernel" is meant as in a convolutional network, not as a GPU compute kernel. It covers every offset, so each oscillator is coupled to every other in its channel. The weights are drawn from N(0, 0.05²): a positive weight pulls a pair toward the same phase, a negative one pushes it apart. Each channel has its own kernel. |
| **Coupling function** | How the phases of two coupled oscillators turn into a push on one of them (A.4). |
| **Coupling ceiling** | A cap on each channel's overall coupling strength, 1 or 0.5. It limits the channel's coupling as a whole, the largest factor by which the kernel amplifies any spatial pattern of phases across the channel (the peak of the kernel's spectrum), and enforces it by scaling every weight of that channel's kernel by the same factor. It does not cap individual pairs, neighbours or regions. Random kernels always exceed it here (peaks of 1.4 to 2.9 on the torus), so in practice the ceiling sets every channel's coupling strength. |
| **Restoring strength λ** | The strength of a pull on every oscillator back toward phase 0, the term −λ sin θ, 0.3 or 0.1. It has the form of the restoring torque that returns a pendulum to rest, but the oscillators have no inertia, so nothing swings back past 0: an oscillator whose natural frequency is below λ is held in place, or locked ([Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930)), and a faster one keeps rotating, slowed where the pull opposes it. For Stuart–Landau oscillators it is a pull toward phase 0 at unit amplitude. |
| **Cluster** | A group of oscillators that move together, in phase or in fixed opposition, because of the dynamics rather than the wiring. Nothing in a kernel assigns an oscillator to a cluster; clusters form, or do not, as the network runs ([Pikovsky et al. 2001](https://doi.org/10.1017/CBO9780511755743)). |
| **Rotation rate** | How far an oscillator's phase advances between frames, read as the mean of cos Δθ and sin Δθ over a window. An additional read, beside the statistics. |

### A.4 Coupling functions

Defined in Section 2.1 and drawn in Appendix D.

### A.5 Lattice geometries

Defined and drawn in Section 2.2.

### A.6 The read and the evaluation

| Term | As used in this paper |
|---|---|
| **Signal** | A time series an arm exposes to the readout: a band energy for the spectrogram-only baseline, sin θ or cos θ of an oscillator, a leaky integrator's state, or a trained baseline's hidden unit. |
| **Summary statistics** | Per signal and window: the mean, the standard deviation and the mean absolute frame-to-frame change. None is an endpoint, and the absolute change cannot tell a rising signal from a falling one. |
| **Window** | For recognition, frames 16 to 61 split into four equal windows; for the order task, frames 16 to 147 as one. Every clip is read over the same frames. |
| **Warm-up** | The first 16 frames (256 ms), which every arm but the whole-clip spectrogram-only baseline skips. Reading every arm over the same frames after the warm-up means an arm with no input reads chance. |
| **Width** | The number of features the readout sees: 192, 1,024 or 4,096, reached by a random Gaussian projection shared by every arm, either the fixed draw or a draw of its own for each run seed (Section 3.4). An arm's native width is its feature count before projection: 192 for the spectrogram-only baseline, 24,576 for the oscillator network. |
| **Primary cell** | Width 192, 2,048 training clips, and the four-window read (the whole-span read for the order task). |
| **Protocols A and B** | A: train on speakers 1 to 48 and test on the 6,000 clips of speakers 49 to 60. B: the five speaker folds of [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038). |
| **Noise level** | White noise added to each clip at a level relative to its speech: 0 dB is as loud as the speech, and +5 dB is 5 dB louder (a signal-to-noise ratio of −5 dB). |
| **Order task** | Two digits of a pair spoken one after the other; the task is to say which came first. The mean and spread of a signal do not depend on the order of its frames, so the input alone read over the whole span cannot answer it, and the spectrogram-only baseline reads chance, 50%. |

## B Experimental design

The tiers of the study, with their registered run counts; the projection tier was added after the registration was frozen (Section 3.4).

| tier | question | arms | varied | fixed | runs |
|---|---|---|---|---|---|
| gate | Does the pipeline leak? | the coupled and uncoupled oscillator networks and both leaky-integrator banks, with no input; the coupled network read over each clip's own length | the per-clip read at input gain 0, 1 and 2, seeds 0–2 | 0 dB; the no-input cells at seed 0 | 13 |
| 1, arms | Do the dynamics add anything beyond the input, a leaky-integrator bank, or uncoupled oscillators, and how do trained baselines compare? | spectrogram-only baseline, coupled oscillator network, uncoupled oscillator network, state-matched and width-matched leaky-integrator banks, and the trained baselines GRU, TCN, CNN, transformer and S4D | noise clean, 0 and +5 dB; input gain 1 and 2 (reservoirs); seeds 0–2; recognition at 2,048, 8,192 and 24,000 training clips, each trained baseline trained at each size; the order task on 5 digit pairs | coupled network: Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1; order task at 2,048 training and 2,048 test sequences | 846 (216 recognition, 630 order) |
| 2, design | Does network design move accuracy? | coupled oscillator network only | coupling function (6), lattice geometry (6; the Stuart–Landau functions on the torus only), natural frequencies (random, tonotopic, identical), restoring strength (0.3, 0.1), coupling ceiling (1, 0.5); noise 0 and +5 dB; input gain 1 and 2; seeds 0–2 | 2,048 training clips; the four-window read, with and without rotation rates | 3,744 |
| B, Becker | Where do we sit against published numbers? | the Tier 1 arms | the 5 speaker folds of [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038); input gain 1 and 2 | clean audio; 18,000 training, 6,000 validation and 6,000 test clips; seed 0 | 70 |
| 3, quadrature | Does the quadrature pathway beat its own spectrogram-only baseline? | spectrogram-only baseline and 10 coupled-network configurations: the 4 phase coupling functions on the torus and Kuramoto on the helix, each with random and tonotopic natural frequencies | noise 0 and +5 dB; input gain 1 and 2; seeds 0–2 | quadrature pathway; 2,048 training clips | 126 |
| carrier | Does driving with the waveform itself help? | spectrogram-only baseline, state-matched leaky-integrator bank, the 10 Tier 3 configurations, and the 2 Stuart–Landau coupling functions with random and tonotopic natural frequencies | seeds 0–2 | 0 dB; input gain 32; carrier pathway at 16 kHz; 2,048 training clips | 48 |
| projection | Does the fixed projection's draw move Tier 1's reservoir results? | the coupled and uncoupled oscillator networks and both leaky-integrator banks | noise clean, 0 and +5 dB; input gain 1 and 2; seeds 0–2; recognition and the order task on 5 digit pairs; each read under the fixed and a seeded projection | 2,048 training clips; added after the registration was frozen | 432 |

**Arms.** The spectrogram-only baseline is the front end and the readout with nothing between them: the readout reads the 16 mel band energies directly. The coupled oscillator network is 1,024 untrained oscillators, 4 channels of a 16 × 16 lattice, with 2,048 parameters, exposing sin θ and cos θ of each oscillator. The uncoupled oscillator network is the same network with its coupling kernels set to zero. The state-matched leaky-integrator bank has the network's 1,024 states and 2,048 parameters; the width-matched bank has 2,048 units, matching the network's 2,048 exposed signals at twice the parameters. These four untrained dynamical systems are the reservoirs. The five trained baselines are trained end to end, at 1,840 to 2,109 parameters each. Input gain applies only to the reservoirs (Section 3.7).

## C Channels and layers

![Layers and channels. (a) A deep network or a transformer stacks layers: each transforms the output of the one before it, so layers add depth. (b) The coupled oscillator network's channels sit side by side: each is a 16 × 16 lattice driven by the same mel spectrogram, mel band r driving row r, with its own coupling kernel and natural frequencies and no coupling to the other channels, so channels add width. The readout reads the signals of all four channels together, as the outputs of the attention heads in one transformer layer are combined.](resources/figures/a1-channels-and-layers.png)

## D Coupling functions

![Coupling functions, each for one neighbour j with a positive kernel weight. Left: the push an oscillator would feel at each phase around the circle while j sits at the top; arrows point the way it is pushed. Right: that push against the phase difference θⱼ − θᵢ, with filled dots where a pair rests stably and open dots where it rests but is unstable or neutral. Kuramoto pulls each oscillator along the circle toward its neighbour. Kuramoto–Sakaguchi aims α = π/4 behind it, admitting partial coherence. The second harmonic (β = 0.5) strengthens the pull and removes the push away from opposite phase. Winfree pushes every oscillator toward phase 0, the neighbour's phase setting only how hard. Stuart–Landau pulls each oscillator straight toward its neighbour in the plane, so out-of-phase neighbours also shrink each other's amplitude; at fixed amplitude only the pull along the circle is left, which is Kuramoto. A negative kernel weight reverses every arrow, and each oscillator feels the sum over every other oscillator in its channel.](resources/figures/a3-coupling-functions.png)

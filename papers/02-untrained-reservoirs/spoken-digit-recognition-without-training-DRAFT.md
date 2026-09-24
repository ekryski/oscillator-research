# Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Frozen Oscillator Fields

## Abstract

Coupled-oscillator networks are returning as a machine-learning substrate, on the premise that oscillator physics turns signals into useful structure. Yet the ablations that would isolate what the oscillator dynamics contribute are rarely run. We run them for untrained oscillator fields, using spoken-digit classification on AudioMNIST as the measure. A small frozen oscillator field is read by a closed-form linear readout and compared with trained conventional networks matched in parameters and with two controls: a non-oscillating bank of leaky integrators with the same states and parameters, and the same readout fitted directly to the field's input with the field removed. Every experimental arm's state is summarized by the same time-pooled statistics and classified at the same number of features, so that readout capacity cannot pass for dynamics. A pre-registered confirmatory sweep then varies coupling law, lattice geometry, input pathway, readout width and training-set size, on digit recognition and on a temporal-order task that order-blind readouts cannot solve. The design separates three explanations for a frozen field's accuracy: its input representation, the width of its readout, and the oscillator dynamics themselves. It gives studies of trained oscillator dynamics an untrained baseline to compare trained results against and insights into which factors contribute to oscillator effectiveness on time series tasks.

## 1 Introduction

## 2 Background: the physics under test

### 2.1 Coupling laws

### 2.2 Lattice geometries

### 2.3 Why oscillator physics might matter for audio at all

## 3 The system under test

## 4 Methods

### 4.1 Experimental design

The confirmatory study was registered before its first run and is organized in seven tiers. Every run is one arm at one noise level, gain and seed, read at several readout widths and training sizes; the run counts below are the registered totals.

| tier | question | arms | varied | fixed | runs |
|---|---|---|---|---|---|
| gate | Does the pipeline leak? | field, severed field, bank A and bank B with no drive; the field read over each clip's own length | the per-clip read at gain 0, 1 and 2, seeds 0–2 | 0 dB; the zero-drive cells at seed 0 | 13 |
| 1, arms | Do the dynamics add anything beyond the input, a leaky bank, or severed coupling, and how do trained networks compare? (H1, H2, H3, H5, H6) | floor, field, severed field, bank A, bank B, GRU, TCN, CNN, transformer, S4D | noise clean, 0 and +5 dB; gain 1 and 2 (untrained dynamical arms); seeds 0–2; recognition at 2,048, 8,192 and 24,000 training clips, each network trained at each size; the order task on 5 digit pairs | field: Kuramoto, torus, random ω, pinning 0.3, clamp 1; order task at 2,048 training and 2,048 test sequences | 846 (216 recognition, 630 order) |
| 2, design | Does field design move accuracy? (H4) | field only | coupling law (6), geometry (6; the Stuart-Landau cores on the torus only), ω structure (random, designed, uniform), pinning (0.3, 0.1), clamp (1, 0.5); noise 0 and +5 dB; gain 1 and 2; seeds 0–2 | 2,048 training clips; the four-window read, with and without rotation rates | 3,744 |
| B, Becker | Where do we sit against published numbers? | the Tier 1 arms | the 5 speaker folds of [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038); gain 1 and 2 | clean audio; 18,000 training, 6,000 validation and 6,000 test clips; seed 0 | 70 |
| 3, quadrature | Does phase-referenced input beat its own floor? (H4) | floor and 10 field configurations: the 4 phase coupling laws and the helix, each with random and designed ω | noise 0 and +5 dB; gain 1 and 2; seeds 0–2 | quadrature front end; 2,048 training clips | 126 |
| 4, size | Does the field's size matter against a matched bank? | field and bank A | 1, 4 and 16 channels (256, 1,024 and 4,096 states); noise 0 and +5 dB; seeds 0–2 | 16 × 16 lattice; gain 2; 2,048 training clips | 36 |
| carrier | Does driving with the waveform itself help? (H4) | floor, bank A, the 10 Tier 3 field configurations, and the 2 Stuart-Landau cores with random and designed ω | seeds 0–2 | 0 dB; gain 32; 16 kHz drive; 2,048 training clips | 48 |

**Arms.** The floor is the front end alone: its 16 band envelopes, read directly. The field is 1,024 oscillators (4 channels on a 16 × 16 lattice) with 2,048 fixed parameters, exposing sin θ and cos θ. The severed field is the same field with its coupling kernel zeroed. Bank A is a leaky bank matching the field's 1,024 states and 2,048 parameters; bank B matches the field's 2,048 exposed signals, at twice the parameters. The five networks are trained end to end, at 1,840 to 2,109 parameters each. Gain, the scale of the input driving an arm, applies only to the untrained dynamical arms: the field, the severed field and the two banks, whose response it changes because they are nonlinear. The floor has no dynamics for it to act on, and the readout's standardization would remove any fixed scale from its features; a trained network learns its own input scale.

**The read.** Every arm is read by one function: the mean, standard deviation and mean absolute frame-to-frame change of each signal, over each of four equal windows of frames 16 to 61 for recognition, and over frames 16 to 147 as one window for the order task. The features are standardized, projected by one fixed Gaussian matrix to widths 192, 1,024 and 4,096 (never wider than the arm itself; the unprojected read is also fitted at 2,048 clips), and classified by a ridge whose penalty is chosen on the last eighth of the training set. The floor is also read from frame 0; that whole-clip floor is the primary control. The field arms also get a read with their rotation rates added.

**The primary cell** for every comparison is width 192, 2,048 training clips and the four-window read (the whole-span read for the order task). The test set is always the 6,000 clips of test speakers 49 to 60, except in Tier B, which uses the published test fold.

**Reporting.** Every accuracy is reported as its mean over replicates (three seeds; the five folds in Tier B) with the sample standard deviation and each replicate's value. Every comparison between two arms is paired on condition and replicate, and reported as the mean difference with its standard deviation over replicates and a 95% interval from resampling test clips. No threshold is applied to either.

### 4.2 Task and data

### 4.3 Primary frontend (mel envelope)

### 4.4 The transduction ladder (drive pathways)

### 4.5 Noise protocol

### 4.6 Drive scale, gain, and the integrator-validity bound

### 4.7 Instrumentation

### 4.8 Evidential standard

## 5 Results

| arm | clean | 0 dB | +5 dB |
|---|---|---|---|
| **Floor**: the front end alone, its 16 band envelopes read directly; no dynamics | 93.6 ± 0.3 | 78.0 ± 0.6 | 71.6 ± 1.6 |
| **Oscillator field** (gain = 1): 1,024 coupled oscillators, untrained | 91.1 ± 0.4 | 77.8 ± 0.8 | 70.7 ± 0.3 |
| **Severed field** (gain = 1): the same field with its coupling removed | 90.6 ± 0.3 | 76.9 ± 0.5 | 69.7 ± 0.4 |
| **Bank A** (gain = 1): 1,024 leaky integrators with the field's states and parameters, untrained | 93.4 ± 0.2 | 73.8 ± 0.5 | 66.6 ± 0.8 |
| **Transformer**: trained end to end, 1,968 parameters | 96.2 ± 0.4 | 76.0 ± 1.4 | 69.0 ± 1.8 |

Table: Digit-recognition accuracy (%) on AudioMNIST, mean ± standard deviation over three seeds. Training: 2,048 clips drawn from the 24,000 recordings of speakers 1 to 48, with the same white noise level in training and test audio (0 dB: noise as loud as the speech; +5 dB: noise 5 dB louder than the speech). A seed sets an arm's random parameters and which 2,048 clips it is trained on. For the floor, field, severed field and bank A only the readout is fitted; the transformer is also trained end to end on the same clips. Evaluation: each arm's signals are summarized by the same three statistics over four time windows, projected to 192 features and classified by a ridge readout, and scored on all 6,000 clips of the 12 held-out speakers (49 to 60). The floor is read from the first frame of each clip, the other arms from frame 16, after their warm-up. Gain scales the input into the untrained dynamical arms and does not apply to the floor or the transformer.

![Digit-recognition accuracy on the 6,000 test clips of AudioMNIST speakers 49 to 60, by the noise level of the training and test audio. Bars are the mean over three seeds and error bars one standard deviation. Every arm was trained on 2,048 clips from speakers 1 to 48 and read by the same ridge readout on 192 features; the transformer was also trained end to end. Gain applies only to the untrained dynamical arms (field, severed field, bank A). Dotted line: chance, 10%.](resources/figures/c1-recognition-gain1.png)

![Digit-recognition accuracy with the untrained dynamical arms (field, severed field, bank A) driven at gain = 2, hatched. The floor and the transformer take no gain and are the same as at gain = 1. Every arm was trained on 2,048 clips from speakers 1 to 48 and read by the same ridge readout on 192 features, and scored on the 6,000 test clips of speakers 49 to 60. Bars are the mean over three seeds and error bars one standard deviation. Dotted line: chance, 10%.](resources/figures/c2-recognition-gain2.png)

![Digit-recognition accuracy at both gains: each untrained dynamical arm at gain = 1 (solid) and gain = 2 (hatched), beside the floor and the transformer, which take no gain. Every arm was trained on 2,048 clips from speakers 1 to 48 and read by the same ridge readout on 192 features, and scored on the 6,000 test clips of speakers 49 to 60. Bars are the mean over three seeds and error bars one standard deviation. Dotted line: chance, 10%.](resources/figures/c3-recognition-both-gains.png)

### 5.1 Trained conventional networks

### 5.2 Field design: coupling law, geometry and natural frequencies

### 5.3 Input pathways: quadrature and carrier drive

### 5.4 Readout sufficiency

### 5.5 Sensitivity of the canonical physics values

### 5.6 Coherence and readability

### 5.7 Temporal order

## 6 Phase-referenced input: model and mechanism

## 7 Discussion

## 8 Related work

## 9 Reproducibility

## AI use statement {-}

This work was carried out by the author working with an AI coding agent, Claude (Anthropic), throughout. The disclosure covers both the uses ICLR requires to be disclosed and those it recommends.

**Implementing methods and running experiments.** Under the author's direction the agent wrote most of the experiment harness, the sweep driver, the scoring code and the test suite, launched and scored runs, and kept the dated experiment logs from which this paper's numbers are drawn.

**Designing experiments and interpreting results.** The author set the research questions and directed the work. The agent contributed to the design of controls, flagged flaws in experimental designs, and wrote first-draft interpretations of results, which the author reviewed and accepted, revised or rejected.

**Writing.** The agent drafted sections of this paper, including the abstract, from the experiment logs, and edited the text for readability; it also helped find and check related literature and built the tooling that produces the paper's formats and checks its bibliography.

**Verification.** Decision criteria were recorded before each run, and every verdict in the paper is regenerated from the committed record by a scoring script rather than transcribed. The harness carries contract tests for each mechanism the paper relies on, and same-seed runs are bit-identical. The author takes responsibility for the final content of this work.

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
- [Doelling et al. 2023](https://doi.org/10.1371/journal.pcbi.1011669): Adaptive oscillators support Bayesian prediction in temporal processing.
- [Dogonasheva et al. 2026](https://doi.org/10.1016/j.neunet.2025.108194): Neuro-oscillatory models of cortical speech processing.
- [Hairer et al. 1993](https://doi.org/10.1007/978-3-540-78862-1): Solving Ordinary Differential Equations I: Nonstiff Problems.
- [Hansel et al. 1993](https://doi.org/10.1103/PhysRevE.48.3470): Clustering and slow switching in globally coupled phase oscillators.
- [Jaeger 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf): The "echo state" approach to analysing and training recurrent neural networks.
- [Jaeger et al. 2007](https://doi.org/10.1016/j.neunet.2007.04.016): Optimization and applications of echo state networks with leaky-integrator neurons.
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

| Term | As used in this paper |
|---|---|
| **Leaky integrator** | A unit that holds one number x and, at each frame, moves a fraction a of the way toward its input: x ← (1 − a)·x + a·tanh(g_in·g·u), where u is the input, g the input gain, g_in the unit's own input weight, and tanh bounds the input. The old value decays exponentially, or "leaks", so the unit is a running average of its recent input with a time constant set by a; in signal-processing terms it is a first-order low-pass filter. The term is standard. In computational neuroscience it is the leaky-integrator neuron, the leaky integrate-and-fire model without the firing; in reservoir computing it is the unit of leaky-integrator echo state networks ([Jaeger et al. 2007](https://doi.org/10.1016/j.neunet.2007.04.016)), whose a is the "leaking rate" ([Lukoševičius 2012](https://doi.org/10.1007/978-3-642-35289-8_36)). The leaky-integrator banks used here have 1,024 or 2,048 units, time constants spaced logarithmically from 16 ms to 1 s, input weights g_in drawn from N(1, 0.1²), and no connections between units: nothing rotates and nothing is coupled. |

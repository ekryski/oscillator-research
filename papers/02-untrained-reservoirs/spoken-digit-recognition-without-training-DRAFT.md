# Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Frozen Oscillator Fields

## Abstract

Coupled-oscillator networks are returning as a machine-learning substrate, on the premise that oscillator physics turns signals into useful structure. Yet the ablations that would isolate what the oscillator dynamics contribute are rarely run. We run them for untrained oscillator fields, using spoken-digit classification on AudioMNIST as the measure. A small frozen oscillator field is read by a closed-form linear readout and compared with trained conventional networks matched in parameters and with two controls: a non-oscillating bank of leaky integrators with the same states and parameters, and the same readout fitted directly to the field's input with the field removed. Every experimental arm's state is summarized by the same time-pooled statistics and classified at the same number of features, so that readout capacity cannot pass for dynamics. A pre-registered confirmatory sweep then varies coupling law, lattice geometry, input pathway, readout width and training-set size, on digit recognition and on a temporal-order task that order-blind readouts cannot solve. The design separates three explanations for a frozen field's accuracy: its input representation, the width of its readout, and the oscillator dynamics themselves. It gives studies of trained oscillator dynamics an untrained baseline to compare trained results against and insights into which factors contribute to oscillator effectiveness on time series tasks.

## 1 Introduction

## 2 Background: the physics under test

### 2.1 Coupling functions

### 2.2 Lattice geometries

### 2.3 Why oscillator physics might matter for audio at all

## 3 The system under test

## 4 Methods

### 4.1 Experimental design

The confirmatory study was registered before its first run and is organized in six tiers. Every network is 4 channels of a 16 × 16 lattice; a registered tier on the size of the network was withdrawn before any of its runs and is the subject of a separate study. Every run is one arm at one noise level, input gain and seed, read at several readout widths and training sizes; the run counts below are the registered totals. Appendix A defines every term.

| tier | question | arms | varied | fixed | runs |
|---|---|---|---|---|---|
| gate | Does the pipeline leak? | the coupled and uncoupled oscillator networks and both leaky-integrator banks, with no input; the coupled network read over each clip's own length | the per-clip read at input gain 0, 1 and 2, seeds 0–2 | 0 dB; the no-input cells at seed 0 | 13 |
| 1, arms | Do the dynamics add anything beyond the input, a leaky-integrator bank, or uncoupled oscillators, and how do trained baselines compare? (H1, H2, H3, H5, H6) | spectrogram-only baseline, coupled oscillator network, uncoupled oscillator network, state-matched and width-matched leaky-integrator banks, and the trained baselines GRU, TCN, CNN, transformer and S4D | noise clean, 0 and +5 dB; input gain 1 and 2 (reservoirs); seeds 0–2; recognition at 2,048, 8,192 and 24,000 training clips, each trained baseline trained at each size; the order task on 5 digit pairs | coupled network: Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1; order task at 2,048 training and 2,048 test sequences | 846 (216 recognition, 630 order) |
| 2, design | Does network design move accuracy? (H4) | coupled oscillator network only | coupling function (6), lattice geometry (6; the Stuart–Landau functions on the torus only), natural frequencies (random, tonotopic, identical), restoring strength (0.3, 0.1), coupling ceiling (1, 0.5); noise 0 and +5 dB; input gain 1 and 2; seeds 0–2 | 2,048 training clips; the four-window read, with and without rotation rates | 3,744 |
| B, Becker | Where do we sit against published numbers? | the Tier 1 arms | the 5 speaker folds of [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038); input gain 1 and 2 | clean audio; 18,000 training, 6,000 validation and 6,000 test clips; seed 0 | 70 |
| 3, quadrature | Does the quadrature pathway beat its own spectrogram-only baseline? (H4) | spectrogram-only baseline and 10 coupled-network configurations: the 4 phase coupling functions on the torus and Kuramoto on the helix, each with random and tonotopic natural frequencies | noise 0 and +5 dB; input gain 1 and 2; seeds 0–2 | quadrature pathway; 2,048 training clips | 126 |
| carrier | Does driving with the waveform itself help? (H4) | spectrogram-only baseline, state-matched leaky-integrator bank, the 10 Tier 3 configurations, and the 2 Stuart–Landau coupling functions with random and tonotopic natural frequencies | seeds 0–2 | 0 dB; input gain 32; carrier pathway at 16 kHz; 2,048 training clips | 48 |

**Arms.** The spectrogram-only baseline is the front end and the readout with nothing between them: the readout reads the 16 mel band energies directly. The coupled oscillator network is 1,024 untrained oscillators, 4 channels of a 16 × 16 lattice, with 2,048 parameters, exposing sin θ and cos θ of each oscillator. The uncoupled oscillator network is the same network with its coupling kernels set to zero. The state-matched leaky-integrator bank has the network's 1,024 states and 2,048 parameters; the width-matched bank has 2,048 units, matching the network's 2,048 exposed signals at twice the parameters. These four untrained dynamical systems are the reservoirs. The five trained baselines are trained end to end, at 1,840 to 2,109 parameters each. Input gain applies only to the reservoirs (Section 4.6).

**The read.** Every arm is read by one function: the mean, standard deviation and mean absolute frame-to-frame change of each signal, over each of four equal windows of frames 16 to 61 for recognition, and over frames 16 to 147 as one window for the order task. The features are standardized, projected by one fixed Gaussian matrix to widths 192, 1,024 and 4,096 (never wider than the arm itself; the unprojected read is also fitted at 2,048 clips), and classified by a linear readout fitted by ridge regression, whose penalty is chosen on the last eighth of the training set. The spectrogram-only baseline is also read from frame 0; that whole-clip baseline is the primary control. The oscillator networks also get a read with their rotation rates added.

**The primary cell** for every comparison is width 192, 2,048 training clips and the four-window read (the whole-span read for the order task). The test set is always the 6,000 clips of test speakers 49 to 60, except in Tier B, which uses the published test fold. Section 4.8 describes how accuracies, differences and their uncertainty are reported.

### 4.2 Task and data

### 4.3 Front end: the mel spectrogram

### 4.4 Input pathways

### 4.5 Noise protocol

### 4.6 Input gain and the integrator-validity bound

Input gain is the factor that scales the mel spectrogram before it drives a reservoir. It applies only to the reservoirs: the coupled and uncoupled oscillator networks and the leaky-integrator banks. They are nonlinear, so the strength of the input relative to their own dynamics changes how they respond. For an oscillator, the input competes with its natural frequency, its coupling and the restoring pull; for a leaky integrator, it sets how far into the saturating tanh the input reaches. Gain therefore selects a regime and is a genuine experimental setting, and the reservoirs are run at gains 1 and 2 (32 on the carrier pathway, as calibrated in the exploratory phase).

The spectrogram-only baseline has no dynamics for gain to act on. Its features are summary statistics of the band energies, and each scales in proportion to the input: doubling the input doubles every mean, standard deviation and mean absolute change, and the readout's standardization divides that factor out exactly, so the baseline at gain 2 would be identical to the baseline at gain 1. The trained baselines learn their own input scale: a fixed gain could be undone by the first layer's weights, so it would relabel the same experiment rather than define a new one. Each trained baseline is therefore trained once per noise level, on the input as it is.

### 4.7 Instrumentation

### 4.8 Reporting and uncertainty

No threshold decides a result. Every accuracy is reported as its mean over replicates, three seeds under Protocol A and the five folds under Protocol B, with the sample standard deviation and each replicate's value. A seed sets an arm's random parameters and which training clips it draws. Every comparison between two arms is paired: both are scored on the same test clips, at the same condition and seed, and the comparison is reported as the mean difference, its standard deviation over replicates, and a 95% interval from resampling the test clips 2,000 times.

These describe two different sources of noise. The standard deviation over replicates is how much an arm's accuracy moves when its random draw and its training clips change; for the five arms of the first results table (Section 5) it is 0.2 to 1.8 points. The paired interval is how precisely the test set measures the mean difference. Because both arms are scored on the same clips, the clips that every arm finds easy or hard cancel in the difference, so the interval is narrower than a comparison of two unpaired accuracies would give. It covers test-set sampling only, not the spread over seeds, which is why each replicate's difference is reported beside it. Whether a difference is large enough to matter is left to the reader and to further seeds.

## 5 Results

| arm | clean | 0 dB | +5 dB |
|---|---|---|---|
| **Spectrogram-only baseline**: the readout reads the 16 mel band energies directly; no reservoir | 93.6 ± 0.3 | 78.0 ± 0.6 | 71.6 ± 1.6 |
| **Coupled oscillator network** (gain = 1): 1,024 oscillators, untrained | 91.1 ± 0.4 | 77.8 ± 0.8 | 70.7 ± 0.3 |
| **Uncoupled oscillator network** (gain = 1): the same network with its coupling removed | 90.6 ± 0.3 | 76.9 ± 0.5 | 69.7 ± 0.4 |
| **Leaky-integrator bank, state-matched** (gain = 1): 1,024 leaky integrators with the network's states and parameters, untrained | 93.4 ± 0.2 | 73.8 ± 0.5 | 66.6 ± 0.8 |
| **Transformer**: a trained baseline, 1,968 parameters, trained end to end | 96.2 ± 0.4 | 76.0 ± 1.4 | 69.0 ± 1.8 |

Table: Digit-recognition accuracy (%) on AudioMNIST, mean ± standard deviation over three seeds. Training: 2,048 clips drawn from the 24,000 recordings of speakers 1 to 48, with the same white noise level in training and test audio (0 dB: noise as loud as the speech; +5 dB: noise 5 dB louder than the speech). A seed sets an arm's random parameters and which 2,048 clips it is trained on. For the spectrogram-only baseline and the three reservoirs only the readout is fitted; the transformer is also trained end to end on the same clips. Evaluation: each arm's signals are summarized by the same three statistics over four time windows, projected to 192 features and classified by a linear readout fitted by ridge regression, and scored on all 6,000 clips of the 12 held-out speakers (49 to 60). The spectrogram-only baseline is read from the first frame of each clip, the other arms from frame 16, after their warm-up. Input gain applies only to the reservoirs, not to the spectrogram-only baseline or the transformer.

![Digit-recognition accuracy on the 6,000 test clips of AudioMNIST speakers 49 to 60, by the noise level of the training and test audio, with the reservoirs at input gain 1. Bars are the mean over three seeds and error bars one standard deviation. Every arm was trained on 2,048 clips from speakers 1 to 48 and read by the same linear readout on 192 features; the transformer was also trained end to end. Input gain applies only to the reservoirs: the coupled and uncoupled oscillator networks and the leaky-integrator bank. Dotted line: chance, 10%.](resources/figures/c1-recognition-gain1.png)

![Digit-recognition accuracy with the reservoirs (the coupled and uncoupled oscillator networks and the state-matched leaky-integrator bank) at input gain 2, hatched. The spectrogram-only baseline and the transformer take no gain and are the same as at gain 1. Every arm was trained on 2,048 clips from speakers 1 to 48, read by the same linear readout on 192 features, and scored on the 6,000 test clips of speakers 49 to 60. Bars are the mean over three seeds and error bars one standard deviation. Dotted line: chance, 10%.](resources/figures/c2-recognition-gain2.png)

![Digit-recognition accuracy at both input gains: each reservoir at gain 1 (solid) and gain 2 (hatched), beside the spectrogram-only baseline and the transformer, which take no gain. Every arm was trained on 2,048 clips from speakers 1 to 48, read by the same linear readout on 192 features, and scored on the 6,000 test clips of speakers 49 to 60. Bars are the mean over three seeds and error bars one standard deviation. Dotted line: chance, 10%.](resources/figures/c3-recognition-both-gains.png)

### 5.1 Trained baselines

### 5.2 Network design: coupling function, lattice geometry and natural frequencies

### 5.3 Input pathways: quadrature and carrier

### 5.4 Readout sufficiency

### 5.5 Sensitivity of the canonical physics values

### 5.6 Coherence and readability

### 5.7 Temporal order

## 6 The quadrature pathway: model and mechanism

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
| **Channel** | One independent 16 × 16 lattice of 256 oscillators, with its own coupling kernel and natural frequencies. Every channel receives the same input, and channels do not act on each other; the network's 4 channels are four differently drawn copies, read side by side (Appendix B). The name follows the channels of a convolutional network, where each channel likewise has its own kernel. |
| **Coupling kernel** | A channel's table of coupling weights: a 16 × 16 array whose entry at an offset of so many rows and columns sets how strongly an oscillator is acted on by the one at that offset. The same table applies at every site, so the coupling is a convolution, and "kernel" is meant as in a convolutional network, not as a GPU compute kernel. It covers every offset, so each oscillator is coupled to every other in its channel. The weights are drawn from N(0, 0.05²): a positive weight pulls a pair toward the same phase, a negative one pushes it apart. Each channel has its own kernel. |
| **Coupling function** | How the phases of two coupled oscillators turn into a push on one of them (A.4). |
| **Coupling ceiling** | A cap on each channel's overall coupling strength, 1 or 0.5. It limits the channel's coupling as a whole, the largest factor by which the kernel amplifies any spatial pattern of phases across the channel (the peak of the kernel's spectrum), and enforces it by scaling every weight of that channel's kernel by the same factor. It does not cap individual pairs, neighbours or regions. Random kernels always exceed it here (peaks of 1.4 to 2.9 on the torus), so in practice the ceiling sets every channel's coupling strength. |
| **Restoring strength λ** | The strength of a pull on every oscillator back toward phase 0, the term −λ sin θ, 0.3 or 0.1. It has the form of the restoring torque that returns a pendulum to rest, but the oscillators have no inertia, so nothing swings back past 0: an oscillator whose natural frequency is below λ is held in place, or locked ([Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930)), and a faster one keeps rotating, slowed where the pull opposes it. For Stuart–Landau oscillators it is a pull toward phase 0 at unit amplitude. |
| **Cluster** | A group of oscillators that move together, in phase or in fixed opposition, because of the dynamics rather than the wiring. Nothing in a kernel assigns an oscillator to a cluster; clusters form, or do not, as the network runs ([Pikovsky et al. 2001](https://doi.org/10.1017/CBO9780511755743)). |
| **Rotation rate** | How far an oscillator's phase advances between frames, read as the mean of cos Δθ and sin Δθ over a window. An additional read, beside the statistics. |

### A.4 Coupling functions

Each is the coupling term in θ̇ᵢ = ωᵢ + couplingᵢ + g·uᵢ − λ sin θᵢ, where the sum runs over the oscillators j of oscillator i's channel and Kᵢⱼ is the kernel weight at their offset.

| Term | As used in this paper |
|---|---|
| **Kuramoto** | Σⱼ Kᵢⱼ sin(θⱼ − θᵢ). Each oscillator is pulled toward the phases of the others, in proportion to the kernel weight: the minimal model of synchronization ([Kuramoto 1975](https://doi.org/10.1007/BFb0013365)), and the reference. |
| **Kuramoto–Sakaguchi** | Σⱼ Kᵢⱼ sin(θⱼ − θᵢ − α), with α = π/4. A phase lag that breaks the pull's symmetry and admits travelling waves ([Sakaguchi & Kuramoto 1986](https://doi.org/10.1143/PTP.76.576)). |
| **Second harmonic** | Kuramoto plus β Σⱼ Kᵢⱼ sin 2(θⱼ − θᵢ), with β = 0.5. The second harmonic favours two-cluster states, pairs in phase or in opposition ([Daido 1992](https://doi.org/10.1143/ptp/88.6.1213); [Hansel et al. 1993](https://doi.org/10.1103/PhysRevE.48.3470)). |
| **Winfree** | −sin θᵢ Σⱼ Kᵢⱼ (1 + cos θⱼ). How strongly an oscillator responds depends on its own phase, and how strongly it acts on the others on its phase ([Winfree 1967](https://doi.org/10.1016/0022-5193%2867%2990051-3)). |
| **Stuart–Landau** | Each oscillator is a complex amplitude z = x + iy that relaxes to a cycle of unit radius, and coupling is diffusive, Σⱼ Kᵢⱼ (zⱼ − zᵢ), so amplitude as well as phase carries the state ([Stuart 1960](https://doi.org/10.1017/S002211206000116X); [Aranson & Kramer 2002](https://doi.org/10.1103/RevModPhys.74.99)). The readout sees y and x in place of sin θ and cos θ. |
| **Stuart–Landau, fixed amplitude** | The same with the amplitude held at 1: the phase-only limit, which reduces to Kuramoto. It separates what the amplitude adds. |

### A.5 Lattice geometries

Every channel stores its oscillators as the same 16 × 16 grid; a geometry changes only which of them are neighbours, that is, how the grid's edges are glued. Appendix C draws each one.

| Term | As used in this paper |
|---|---|
| **Torus** | Both lattice axes wrap around: the top row is coupled to the bottom, and the left column to the right. The reference geometry. |
| **Cylinder** | Columns wrap, rows do not: the frequency axis is open, as in the cochlea, so the highest band is not coupled around to the lowest. |
| **Sheet** | Neither axis wraps, and coupling stops at every edge. With the cylinder and the torus it varies the number of wrapped axes from zero to two. |
| **Helix** | The 256 sites read as one closed ring, one octave (64 sites, four rows) per turn, so an offset of one turn joins bands an octave apart. |
| **Cube** | Each row's 16 columns read as a 4 × 4 slab, giving a 16 × 4 × 4 lattice that wraps on all three axes: the same oscillators, with shorter paths between them. |
| **Sphere** | Rows as latitudes with open poles and columns as longitudes that wrap, each oscillator's influence weighted by the cosine of its latitude: an approximation to a sphere, not exact spherical coupling. |

### A.6 The read and the evaluation

| Term | As used in this paper |
|---|---|
| **Signal** | A time series an arm exposes to the readout: a band energy for the spectrogram-only baseline, sin θ or cos θ of an oscillator, a leaky integrator's state, or a trained baseline's hidden unit. |
| **Summary statistics** | Per signal and window: the mean, the standard deviation and the mean absolute frame-to-frame change. None is an endpoint, and the absolute change cannot tell a rising signal from a falling one. |
| **Window** | For recognition, frames 16 to 61 split into four equal windows; for the order task, frames 16 to 147 as one. Every clip is read over the same frames. |
| **Warm-up** | The first 16 frames (256 ms), which every arm but the whole-clip spectrogram-only baseline skips. Reading every arm over the same frames after the warm-up means an arm with no input reads chance. |
| **Width** | The number of features the readout sees: 192, 1,024 or 4,096, reached by one fixed random projection shared by every arm. An arm's native width is its feature count before projection: 192 for the spectrogram-only baseline, 24,576 for the oscillator network. |
| **Primary cell** | Width 192, 2,048 training clips, and the four-window read (the whole-span read for the order task). |
| **Protocols A and B** | A: train on speakers 1 to 48 and test on the 6,000 clips of speakers 49 to 60. B: the five speaker folds of [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038). |
| **Noise level** | White noise added to each clip at a level relative to its speech: 0 dB is as loud as the speech, and +5 dB is 5 dB louder (a signal-to-noise ratio of −5 dB). |
| **Order task** | Two digits of a pair spoken one after the other; the task is to say which came first. The mean and spread of a signal do not depend on the order of its frames, so the input alone read over the whole span cannot answer it, and the spectrogram-only baseline reads chance, 50%. |

## B Channels and layers

![Layers and channels. (a) A deep network or a transformer stacks layers: each transforms the output of the one before it, so layers add depth. (b) The coupled oscillator network's channels sit side by side: each is a 16 × 16 lattice driven by the same mel spectrogram, mel band r driving row r, with its own coupling kernel and natural frequencies and no coupling to the other channels, so channels add width. The readout reads the signals of all four channels together, as the outputs of the attention heads in one transformer layer are combined.](resources/figures/a1-channels-and-layers.png)

## C Lattice geometries

![Lattice geometries. Every channel stores its 256 oscillators the same way, as a 16 × 16 grid whose row r is driven by mel band r (colour, lowest band dark). A geometry changes only which oscillators are neighbours, that is, how the grid's edges are glued, and each of a network's four channels is a separate copy of the same shape. For each geometry, left: the grid, with glued edges marked by a shared colour and arrow, open edges dark, one oscillator (star) and its nearest neighbours (dots); right: the shape the gluing makes, with the same oscillator and neighbours. On the torus the highest band's row meets the lowest; on the cylinder and the sphere the frequency axis is open; the helix threads all 256 sites into one closed ring, one octave per turn; the cube folds each row into a 4 × 4 slab and wraps all three axes.](resources/figures/a2-lattice-geometries.png)

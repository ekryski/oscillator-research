# Size and Channel Layout in Untrained Coupled Oscillator Networks for Spoken-Digit Recognition

## Abstract

<!-- DRAFT ABSTRACT, written by the agent from the registration for the author to rewrite. It states the design only: no run has been made. -->

*Draft abstract, to be rewritten by the author.* Networks of coupled oscillators are studied as a substrate for machine learning, yet the ablations that would show how much their size and arrangement matter have not been run: no published system varies its number of oscillators, its parameter count or its channel layout against controls of the same size. We run them for untrained coupled oscillator networks read by a linear readout on speaker-disjoint spoken-digit recognition (AudioMNIST), extending a previous study of one 1,024-oscillator network. The lattice ranges from 8 × 8 to 128 × 128 oscillators and the number of channels, independent parallel copies of the lattice, from 1 to 16, so the network ranges from 64 to 262,144 oscillators and from 128 to 524,288 parameters; each lattice is driven either by its own number of mel bands, one per row, or by 16 bands mapped onto its rows. At every size the network is compared with an uncoupled copy of itself, a bank of leaky integrators with its states and parameters, the same readout fitted to the input alone, and five trained networks of its parameter count, and the readout is held at the same 192 features, so that any gain with size belongs to the network rather than to its reader. The size study is crossed with six coupling functions, six lattice geometries and three input pathways. Every coupling kernel is scaled exactly to a fixed spectral ceiling, so that the strongest coupling gain is the same at every size, and networks too large to read in memory are read channel by channel, with the same result. The design separates the effect of how many oscillators a network has from how they are arranged, and from how finely its input is resolved.

## 1 Introduction

## 2 Background

### 2.1 Size in reservoirs and oscillator networks

### 2.2 Channels and layout

## 3 The system under test

## 4 Methods

### 4.1 Experimental design

| tier | question | arms | varied | fixed | runs |
|---|---|---|---|---|---|
| gate | Does the pipeline leak at every size? | the coupled and uncoupled oscillator networks and the state-matched leaky-integrator bank, with no input | lattice 8 × 8 to 128 × 128; 1 and 16 channels | input gain 0; 0 dB; seed 0 | 30 |
| size | How do the number of oscillators and their layout move accuracy, against controls of the same size? | coupled oscillator network, uncoupled oscillator network, state-matched leaky-integrator bank; the spectrogram-only baseline on each lattice's rows | lattice 8 × 8, 16 × 16, 32 × 32, 64 × 64 and 128 × 128; 1, 2, 4, 8 and 16 channels (64 to 262,144 oscillators); band mapping: one mel band per row, or 16 bands mapped onto the rows; seeds 0–2 | Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1 met exactly; band-energy pathway; 0 dB; input gain 1 | 432 |
| trained | How do trained baselines of the same parameter count compare as both grow, and are oscillators more useful at the smallest sizes? | GRU, TCN, CNN, transformer and S4D, each sized to every network of the size tier (128 to 524,288 parameters) and driven by its rows | as the size tier | trained end to end; 0 dB | 675 |
| design | Do coupling function and lattice geometry matter differently at different sizes? | coupled oscillator network | 6 coupling functions × 6 lattice geometries, the Stuart–Landau functions on the torus only; every lattice, channel count and band mapping; seeds 0–2 | as the size tier | 3,375 |
| quadrature | Does the quadrature pathway's standing change with size? | coupled and uncoupled oscillator networks; the pathway's spectrogram-only baseline | as the size tier | quadrature pathway; 0 dB; input gain 1 | 297 |
| carrier | Does the carrier pathway's standing change with size? | coupled and uncoupled oscillator networks, state-matched leaky-integrator bank; the pathway's spectrogram-only baseline | lattice 8 × 8 to 32 × 32; every channel count and band mapping; seeds 0–2 | carrier pathway at 16 kHz; 0 dB; input gain 32 | 240 |
| design-quadrature | Design at size, on the quadrature pathway | coupled oscillator network | the 4 phase coupling functions × 6 lattice geometries; as the quadrature tier | as the quadrature tier | 3,105 |
| design-carrier | Design at size, on the carrier pathway | coupled oscillator network | the design tier's configurations; as the carrier tier | as the carrier tier | 1,875 |

Table: The experimental design (draft registration, not frozen). Every run is one arm at one lattice, channel count, band mapping, input pathway and seed, at 0 dB and input gain 1 (32 on the carrier pathway), fitted on 2,048 training clips of speakers 1 to 48, read at widths 192, 1,024 and 4,096 under a fixed and a seeded projection, and scored on the 6,000 test clips of speakers 49 to 60. The size study is the ablation of oscillator count, parameter count and channel layout that [Kryski 2026a](https://doi.org/10.2139/ssrn.7445198) found no published oscillator network had run. The 16 × 16, 4-channel runs, 150 of the 10,029, are those of [Kryski 2026b](https://github.com/ekryski/oscillator-research/blob/main/papers/02-untrained-reservoirs/), read from its record. Appendix A defines every term.

### 4.2 Task and data

### 4.3 Front end and band mapping

### 4.4 Lattice size, channels and parameter count

### 4.5 Coupling kernels and the coupling ceiling

### 4.6 Controls matched in size

### 4.7 The read at scale

### 4.8 Reporting and uncertainty

## 5 Results

### 5.1 Number of oscillators

### 5.2 Layout at a fixed number of oscillators

### 5.3 Band mapping

### 5.4 Trained baselines of the same size

### 5.5 Coupling function and lattice geometry across sizes

### 5.6 Input pathways across sizes

### 5.7 Readout width

## 6 Discussion

## 7 Related work

## 8 Reproducibility

## AI use statement {-}

This work was carried out by the author working with an AI coding agent, Claude (Anthropic), throughout, as was the paper it extends. The disclosure covers the uses that machine-learning venues require to be disclosed and those they recommend.

**Implementing methods and running experiments.** Under the author's direction the agent adapted the previous paper's experiment harness to this one, wrote the planner, the cost model, the channel-by-channel read, the summary code and the test suite, and measured the timings the cost estimates rest on.

**Designing experiments and interpreting results.** The author set the research question and the scope of the study: the lattices, channel counts and band mappings, the crossing with every coupling function, lattice geometry and input pathway, controls matched in size, and exact scaling of the coupling kernels. The agent proposed the read for networks too large to hold in memory, the front end at 64 and 128 bands, the rule that sizes the trained baselines, the staging of the tiers, and the running of every part of a run on a GPU, each for the author to accept, revise or reject before the registration is frozen; the study's questions are the author's. It will write first-draft interpretations of results, which the author will review.

**Writing.** The agent drafted the registration, the tier plan, this paper's abstract, section structure, design table and glossary, and adapted the glossary from the previous paper's.

**Verification.** The design is recorded before any run. The harness carries contract tests for each mechanism the paper relies on, including that every 16 × 16 network is bit-identical to the previous paper's, that the channel-by-channel read gives the in-memory read's accuracies, and that the front end is the previous paper's up to 32 bands; the cells taken from the previous paper are re-run and checked against its record before they are used. The author takes responsibility for the final content of this work.

## References {-}

- [Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930): A Study of Locking Phenomena in Oscillators.
- [Aranson & Kramer 2002](https://doi.org/10.1103/RevModPhys.74.99): The world of the complex Ginzburg-Landau equation.
- [Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038): AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark.
- [Daido 1992](https://doi.org/10.1143/ptp/88.6.1213): Order Function and Macroscopic Mutual Entrainment in Uniformly Coupled Limit-Cycle Oscillators.
- [Hansel et al. 1993](https://doi.org/10.1103/PhysRevE.48.3470): Clustering and slow switching in globally coupled phase oscillators.
- [Jaeger 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf): The "echo state" approach to analysing and training recurrent neural networks.
- [Jaeger et al. 2007](https://doi.org/10.1016/j.neunet.2007.04.016): Optimization and applications of echo state networks with leaky-integrator neurons.
- [Kryski 2026a](https://doi.org/10.2139/ssrn.7445198): From Synchronization Physics to Trained Dynamics: A Survey of Oscillator Networks in Machine Learning.
- [Kryski 2026b](https://github.com/ekryski/oscillator-research/blob/main/papers/02-untrained-reservoirs/): Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Frozen Oscillator Fields.
- [Kuramoto 1975](https://doi.org/10.1007/BFb0013365): Self-entrainment of a population of coupled non-linear oscillators.
- [Lukoševičius 2012](https://doi.org/10.1007/978-3-642-35289-8_36): A Practical Guide to Applying Echo State Networks.
- [Maass 2002](https://doi.org/10.1162/089976602760407955): Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
- [Pikovsky et al. 2001](https://doi.org/10.1017/CBO9780511755743): Synchronization: A Universal Concept in Nonlinear Sciences.
- [Sakaguchi & Kuramoto 1986](https://doi.org/10.1143/PTP.76.576): A Soluble Active Rotator Model Showing Phase Transitions via Mutual Entrainment.
- [Stuart 1960](https://doi.org/10.1017/S002211206000116X): On the non-linear mechanics of wave disturbances in stable and unstable parallel flows Part 1. The basic behaviour in plane Poiseuille flow.
- [Winfree 1967](https://doi.org/10.1016/0022-5193%2867%2990051-3): Biological rhythms and the behavior of populations of coupled oscillators.

<!-- appendix -->

## Appendix {-}

## A Glossary

The terms as this paper uses them, carried over from [Kryski 2026b](https://github.com/ekryski/oscillator-research/blob/main/papers/02-untrained-reservoirs/) and extended for lattice size, channels, band mapping and parameter matching. The code and the run record keep the previous paper's labels; src/README.md maps each term to its label.

### A.1 The pipeline

| Term | As used in this paper |
|---|---|
| **Front end** | The fixed first stage every arm shares. It turns a clip's 16 kHz waveform into a mel spectrogram: a Fourier transform of a 512-sample (32 ms) Hann window every 256 samples (16 ms), G mel bands, the log of each band's energy, and a fixed rescaling. At 8, 16 and 32 bands it is the previous paper's exactly. At 64 and 128 bands the transform is zero-padded to 1,024 and 2,048 points behind the same window and frames, so that no band's filter falls between the transform's frequency bins; zero-padding samples the same spectrum more finely and adds no frequency resolution. Nothing in it is trained or fitted to the data. |
| **Mel band** | One of the G frequency ranges the front end splits the audio into, spaced on the mel scale, which follows pitch perception: narrow at low frequencies, wide at high ones. G is 8, 16, 32, 64 or 128. |
| **Band energy** | The log energy in one mel band, one value per frame: the signal the front end produces for that band. |
| **Mel spectrogram** | A clip's band energies together, frame by frame. The spectrogram-only baseline reads it directly; every reservoir is driven by it. |
| **Frame** | One 16 ms step of the front end, 62.5 per second, the same at every band count. |
| **Reservoir** | An untrained dynamical system between the front end and the readout: here the coupled and uncoupled oscillator networks and the leaky-integrator bank. The term is reservoir computing's ([Jaeger 2001](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf); [Maass 2002](https://doi.org/10.1162/089976602760407955)). |
| **Readout** | The reader shared by every arm, and the only fitted part of an untrained arm: one linear layer from 192 features to 10 digit scores (1,930 weights), fitted by ridge regression, that is least squares with an L2 penalty, solved in closed form. The penalty is chosen from four values on the last eighth of the training clips, then the readout is refit on all of them. It is the same for every arm at every size, so it is held constant across comparisons: as a network grows, its readout does not. |
| **Untrained** | Parameters drawn at random once, from the seed, and never updated. In an untrained arm only the readout is fitted. |
| **Input gain** | The factor that scales the input before it drives a reservoir: 1 here (32 for the carrier pathway). It applies only to the reservoirs: the spectrogram-only baseline has no dynamics for it to act on, and a trained baseline learns its own input scale. |
| **Input pathway** | How the audio drives a reservoir. Band-energy: each band energy adds to the rotation rate of the oscillators in its row. Quadrature: each band's energy and phase, so the push depends on where an oscillator is relative to the band's own cycle, a phase-referenced drive of the kind [Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930) analysed. Carrier: the waveform itself at 16 kHz, split into G log-spaced bands spanning the same four octaves at every G, rather than its energy. |

### A.2 Arms and baselines

| Term | As used in this paper |
|---|---|
| **Arm** | One system under comparison, read the same way as every other. The term comes from experimental design. |
| **Spectrogram-only baseline** | The front end and the readout with nothing between them: the readout reads the band energies directly, so it measures what the input alone supports. At every lattice it reads exactly the rows that lattice's networks are driven with. Read from the first frame, over the whole clip, it is the primary control, because it sees everything a reservoir is driven with; read from frame 16, it sees the frames every other arm is read over. On the quadrature and carrier pathways it reads that pathway's front end instead. |
| **Coupled oscillator network** | The system under test: C channels of a G × G lattice, C · G² untrained oscillators coupled within each channel through a coupling kernel, with 2 · C · G² parameters (the kernels and the natural frequencies). Also "the oscillator network". The previous paper's network is 4 channels of a 16 × 16 lattice: 1,024 oscillators and 2,048 parameters. Its reference configuration, used wherever a tier does not vary it: Kuramoto coupling, torus geometry, random natural frequencies, restoring strength 0.3, coupling ceiling 1. |
| **Uncoupled oscillator network** | The same network with every coupling kernel set to zero: each oscillator keeps its natural frequency, restoring strength and input, but none acts on another. It isolates what the coupling contributes. With no coupling term it does not depend on the coupling function or the lattice geometry. |
| **Leaky integrator** | A unit that holds one number x and, at each frame, moves a fraction a of the way toward its input: x ← (1 − a)·x + a·tanh(g_in·g·u), where u is the input, g the input gain, g_in the unit's own input weight, and tanh bounds the input. The old value decays exponentially, or "leaks", so the unit is a running average of its recent input with a time constant set by a; in signal-processing terms it is a first-order low-pass filter. In computational neuroscience it is the leaky-integrator neuron, the leaky integrate-and-fire model without the firing; in reservoir computing it is the unit of leaky-integrator echo state networks ([Jaeger et al. 2007](https://doi.org/10.1016/j.neunet.2007.04.016)), whose a is the "leaking rate" ([Lukoševičius 2012](https://doi.org/10.1007/978-3-642-35289-8_36)). |
| **Leaky-integrator bank** | A reservoir of independent leaky integrators, routed like the oscillator network: mel band r drives every unit in row r. Time constants are spaced logarithmically from 16 ms to 1 s across the units of each row, input weights g_in are drawn from N(1, 0.1²), and no unit is connected to another: nothing rotates and nothing is coupled. State-matched: C · G² units, the oscillator network's number of states and parameters, at every size. Width-matched: twice the units, matching the two signals the network exposes per oscillator, at twice the parameters; it is the state-matched bank of 2C channels. |
| **Trained baselines** | Five conventional networks, trained end to end (AdamW, 30 epochs) through a learned linear layer on the same statistics, then read by the same readout, each sized to a network and driven by that network's rows. GRU: a gated recurrent network. TCN: a small causal temporal convolutional network. CNN: two causal one-dimensional convolutions. Transformer: one causal self-attention layer with two heads and a two-layer feed-forward block. S4D: a diagonal state-space model. |
| **Matched pair** | Two runs identical in every factor but the one compared, at the same lattice, band mapping, channel count, noise level, input gain and seed. A factor's effect is the mean difference over its matched pairs. |
| **Seed** | Sets an arm's random parameters and which training clips it draws. Every condition is run at three seeds, and results are reported as their mean with the standard deviation. |

### A.3 Size and layout

| Term | As used in this paper |
|---|---|
| **Lattice size** | The side G of the G × G lattice a channel's oscillators sit on: 8, 16, 32, 64 or 128. |
| **Channel count** | The number C of channels in a network: 1, 2, 4, 8 or 16. |
| **Number of oscillators** | C · G², from 64 (8 × 8, one channel) to 262,144 (128 × 128, 16 channels). Also the number of states of the network and of its state-matched bank. |
| **Layout** | How a network's oscillators are divided: into few large channels or many small ones. Several layouts share a number of oscillators, because doubling G and quartering C leaves C · G² unchanged: 1,024 oscillators are 8 × 8 at 16 channels, 16 × 16 at 4 and 32 × 32 at 1. |
| **Parameter count** | The number of parameters of an arm. For the coupled oscillator network, 2 · C · G²: one kernel weight per offset and one natural frequency per oscillator, per channel. The uncoupled network stores as many and uses half; the state-matched bank has as many by construction; the spectrogram-only baseline has none. |
| **Parameter matching** | Sizing a trained baseline to a network's parameter count. Each architecture has one width knob (the GRU's hidden size, the TCN's and the CNN's hidden channels, the transformer's model width, the S4D's width and state count) and the rest of its design fixed; the width is the widest whose parameter count does not exceed the network's, except the CNN's, the narrowest that reaches it. At 2,048 parameters on 16 rows this gives the previous paper's baselines, 1,840 to 2,109 parameters; from 2,048 parameters up every baseline is within 15% of its network, and below that some are coarser, as the record shows. |

### A.4 The oscillator network

| Term | As used in this paper |
|---|---|
| **Oscillator** | A unit whose state advances around a cycle. A phase oscillator keeps only its phase θ, an angle; a Stuart–Landau oscillator also keeps an amplitude. The readout sees sin θ and cos θ of each. |
| **Natural frequency ω** | The rate at which an oscillator would advance if nothing acted on it. Random: drawn from N(1, 0.1²), the same distribution at every size. |
| **Lattice** | The G × G grid a channel's oscillators sit on. Rows follow frequency: the input to row r comes from the r-th band, from lowest to highest. |
| **Row** | The G oscillators of one channel that one input row drives. The only grouping built into the network. |
| **Band mapping** | How a front end's mel bands drive a lattice's rows. One band per row: a G × G lattice is driven by G mel bands. Mapped: the 16 bands of the previous paper drive the rows, each band G/16 adjacent rows of a larger lattice, each row of an 8 × 8 lattice the mean of two adjacent bands. At 16 × 16 the two are the same. Nothing in the mapping is fitted. |
| **Channel** | One independent G × G lattice of oscillators, with its own coupling kernel and natural frequencies. Every channel receives the same input, and channels do not act on each other: they are parallel copies, like the heads of one attention layer, read side by side, not layers stacked one on another. The name follows the channels of a convolutional network, where each channel likewise has its own kernel. |
| **Coupling kernel** | A channel's table of coupling weights: a G × G array whose entry at an offset of so many rows and columns sets how strongly an oscillator is acted on by the one at that offset. The same table applies at every site, so the coupling is a convolution, and "kernel" is meant as in a convolutional network, not as a GPU compute kernel. It covers every offset, so each oscillator is coupled to every other in its channel. The weights are drawn from N(0, 0.05²) and the kernel is then scaled to the coupling ceiling: a positive weight pulls a pair toward the same phase, a negative one pushes it apart. |
| **Coupling function** | How the phases of two coupled oscillators turn into a push on one of them (A.5). |
| **Coupling ceiling** | Each channel's overall coupling strength, 1: the largest factor by which its kernel amplifies any spatial pattern of phases across the channel (the peak of the kernel's spectrum). Every channel's kernel is scaled, all weights by one factor, so that its peak equals the ceiling exactly, raised or lowered as needed; the strongest coupling gain is therefore the same at every size. It does not cap individual pairs, neighbours or regions, and since a random kernel's peak grows with the lattice, the individual weights of a larger lattice end up smaller. The previous paper only lowered a kernel to the ceiling; random 16 × 16 kernels always exceed it, so on that lattice the two rules are the same. |
| **Restoring strength λ** | The strength of a pull on every oscillator back toward phase 0, the term −λ sin θ, 0.3. It has the form of the restoring torque that returns a pendulum to rest, but the oscillators have no inertia, so nothing swings back past 0: an oscillator whose natural frequency is below λ is held in place, or locked ([Adler 1946](https://doi.org/10.1109/JRPROC.1946.229930)), and a faster one keeps rotating, slowed where the pull opposes it. For Stuart–Landau oscillators it is a pull toward phase 0 at unit amplitude. |
| **Cluster** | A group of oscillators that move together, in phase or in fixed opposition, because of the dynamics rather than the wiring. Nothing in a kernel assigns an oscillator to a cluster; clusters form, or do not, as the network runs ([Pikovsky et al. 2001](https://doi.org/10.1017/CBO9780511755743)). |

### A.5 Coupling functions

Each is the coupling term in θ̇ᵢ = ωᵢ + couplingᵢ + g·uᵢ − λ sin θᵢ, where the sum runs over the oscillators j of oscillator i's channel and Kᵢⱼ is the kernel weight at their offset.

| Term | As used in this paper |
|---|---|
| **Kuramoto** | Σⱼ Kᵢⱼ sin(θⱼ − θᵢ). Each oscillator is pulled toward the phases of the others, in proportion to the kernel weight: the minimal model of synchronization ([Kuramoto 1975](https://doi.org/10.1007/BFb0013365)), and the reference. |
| **Kuramoto–Sakaguchi** | Σⱼ Kᵢⱼ sin(θⱼ − θᵢ − α), with α = π/4. A phase lag that breaks the pull's symmetry and admits travelling waves ([Sakaguchi & Kuramoto 1986](https://doi.org/10.1143/PTP.76.576)). |
| **Second harmonic** | Kuramoto plus β Σⱼ Kᵢⱼ sin 2(θⱼ − θᵢ), with β = 0.5. The second harmonic favours two-cluster states, pairs in phase or in opposition ([Daido 1992](https://doi.org/10.1143/ptp/88.6.1213); [Hansel et al. 1993](https://doi.org/10.1103/PhysRevE.48.3470)). |
| **Winfree** | −sin θᵢ Σⱼ Kᵢⱼ (1 + cos θⱼ). How strongly an oscillator responds depends on its own phase, and how strongly it acts on the others on its phase ([Winfree 1967](https://doi.org/10.1016/0022-5193%2867%2990051-3)). |
| **Stuart–Landau** | Each oscillator is a complex amplitude z = x + iy that relaxes to a cycle of unit radius, and coupling is diffusive, Σⱼ Kᵢⱼ (zⱼ − zᵢ), so amplitude as well as phase carries the state ([Stuart 1960](https://doi.org/10.1017/S002211206000116X); [Aranson & Kramer 2002](https://doi.org/10.1103/RevModPhys.74.99)). The readout sees y and x in place of sin θ and cos θ. Run on the torus only. |
| **Stuart–Landau, fixed amplitude** | The same with the amplitude held at 1: the phase-only limit, which reduces to Kuramoto. It separates what the amplitude adds. |

### A.6 Lattice geometries

| Term | As used in this paper |
|---|---|
| **Torus** | Both lattice axes wrap around: the top row is coupled to the bottom, and the left column to the right. The reference geometry. |
| **Cylinder** | Columns wrap, rows do not: the frequency axis is open, as in the cochlea, so the highest band is not coupled around to the lowest. |
| **Sheet** | Neither axis wraps, and coupling stops at every edge. With the cylinder and the torus it varies the number of wrapped axes from zero to two. |
| **Helix** | The G² sites read as one closed ring, G/4 rows per turn (four at 16 × 16, as in the previous paper), so an offset of one turn joins rows a quarter of the band range apart: an octave of the carrier pathway's log-spaced bands. |
| **Cube** | Each row's G oscillators read as an a × b slab, giving a G × a × b lattice that wraps on all three axes: the same oscillators, with shorter paths between them. The slab is as nearly square as the row allows: 4 × 4 and 8 × 8 at 16 and 64, and 2 × 4, 4 × 8 and 8 × 16 at 8, 32 and 128, where G is not a square. |
| **Sphere** | Rows as latitudes with open poles and columns as longitudes that wrap, each oscillator's influence weighted by the cosine of its latitude: an approximation to a sphere, not exact spherical coupling. |

### A.7 The read and the evaluation

| Term | As used in this paper |
|---|---|
| **Signal** | A time series an arm exposes to the readout: a band energy for the spectrogram-only baseline, sin θ or cos θ of an oscillator, a leaky integrator's state, or a trained baseline's hidden unit. |
| **Summary statistics** | Per signal and window: the mean, the standard deviation and the mean absolute frame-to-frame change. None is an endpoint, and the absolute change cannot tell a rising signal from a falling one. |
| **Window** | Frames 16 to 61 split into four equal windows. Every clip is read over the same frames. |
| **Warm-up** | The first 16 frames (256 ms), which every arm but the whole-clip spectrogram-only baseline skips. Reading every arm over the same frames after the warm-up means an arm with no input reads chance. |
| **Projection** | The Gaussian matrix that brings an arm's features to a common width. Fixed: the previous paper's, one draw for every seed. Seeded: drawn from the run's seed, so the spread over seeds includes the draw. Every projected read is reported under both. |
| **Width** | The number of features the readout sees: 192, 1,024 or 4,096, reached by a random projection. An arm's native width is its feature count before projection, 24 per oscillator for the network (two signals, three statistics, four windows): 24,576 for the previous paper's network and 6,291,456 for a 128 × 128 network of 16 channels. No arm is read wider than its native width. |
| **Channel-by-channel read** | How an arm of more than 4,096 states is read, since its features do not fit in memory: each channel is run alone, its features standardized with its own training statistics and projected, and the projections added. Because channels do not act on each other and standardization is per feature, this is the same read as holding every feature at once, up to the order of floating-point sums; its projection is drawn channel by channel, with the same distribution. |
| **Primary cell** | Width 192, 2,048 training clips, and the four-window read. |
| **Protocol A** | Train on speakers 1 to 48 and test on the 6,000 clips of speakers 49 to 60 of AudioMNIST ([Becker 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038)). |
| **Noise level** | White noise added to each clip at a level relative to its speech. Every run here is at 0 dB, noise as loud as the speech. |

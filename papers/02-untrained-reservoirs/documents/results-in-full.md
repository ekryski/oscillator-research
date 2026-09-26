# Results in full

Supplementary material for "Spoken-Digit Recognition Without Training: Geometry, Coupling, and Drive Effects in Oscillator Networks": every result the paper reports, in full. Sections and appendices named here are the paper's unless they say "below"; `methods-in-full.md` gives the methods, and `results/summary.md` every accuracy and paired difference in the record.

## 1 The network against its controls

Every coupled and uncoupled oscillator network in this section, and in Sections 2, 6 and 7 below, is the reference configuration: Kuramoto coupling on the torus, with random natural frequencies, restoring strength 0.3 and coupling ceiling 1; the other coupling functions, among them Stuart–Landau, are compared in Section 3 below. The table gives recognition accuracy at the primary cell for the spectrogram-only baseline, the three reservoirs at input gain 1, and the GRU, the trained baseline with the highest accuracy with noise; the figure in Section 4.1 of the paper adds each reservoir at gain 2.

| arm | clean | 0 dB | −5 dB |
|---|---|---|---|
| **Spectrogram-only baseline**: the readout reads the 16 mel band energies directly; no reservoir | 93.6 ± 0.3 | 78.0 ± 0.6 | 71.6 ± 1.6 |
| **Coupled oscillator network** (gain = 1): 1,024 Kuramoto oscillators, untrained | 91.1 ± 0.4 | 77.8 ± 0.8 | 70.7 ± 0.3 |
| **Uncoupled oscillator network** (gain = 1): the same network with its coupling removed | 90.6 ± 0.3 | 76.9 ± 0.5 | 69.7 ± 0.4 |
| **Leaky-integrator bank, state-matched** (gain = 1): 1,024 leaky integrators with the network's states and parameters, untrained | 93.4 ± 0.2 | 73.8 ± 0.5 | 66.6 ± 0.8 |
| **GRU**: a trained baseline, 1,944 parameters, trained end to end | 96.5 ± 0.3 | 82.8 ± 1.3 | 76.9 ± 1.6 |

Table: Single spoken-digit recognition accuracy (%) on AudioMNIST, mean ± standard deviation over three seeds.


## 2 Trained baselines

The five trained baselines, each trained end to end on the same clips and read by the same readout, reached the accuracies below at the primary cell. With noise the GRU and TCN read highest, 82.8% and 82.6% at 0 dB, and every trained baseline read above the Kuramoto network and the whole-clip spectrogram-only baseline there. Every recognition run trained, cutting its loss by more than 20%. With 24,000 training clips at 0 dB, the transformer reached 90.6%, the S4D 90.4%, the GRU 89.9%, the TCN 89.7% and the CNN 86.7%; the S4D is a diagonal state-space model, a family that compares well with other architectures on time-series classification, audio included ([Saadatmand et al. 2026](https://arxiv.org/abs/2605.27406)).

| trained baseline | parameters | clean | 0 dB | −5 dB | failed runs with noise |
|---|---|---|---|---|---|
| GRU | 1,944 | 96.5 ± 0.3 | 82.8 ± 1.3 | 76.9 ± 1.6 | 0 of 18 |
| TCN | 1,972 | 97.0 ± 0.7 | 82.6 ± 1.3 | 76.2 ± 0.4 | 0 of 18 |
| CNN | 2,109 | 94.7 ± 0.6 | 79.2 ± 2.0 | 73.9 ± 1.6 | 0 of 18 |
| transformer | 1,968 | 96.3 ± 0.7 | 79.8 ± 0.6 | 72.3 ± 0.4 | 0 of 18 |
| S4D | 1,840 | 95.5 ± 0.7 | 80.8 ± 1.2 | 74.6 ± 0.9 | 0 of 18 |

Table: The trained baselines: parameters, recognition accuracy (%) at the primary cell as mean ± standard deviation over three seeds, and runs with added noise, over all three training sizes, whose training failed (loss cut by less than 20%).

On the five speaker folds of [Becker et al. 2024](https://doi.org/10.1016/j.jfranklin.2023.11.038) (clean audio, 18,000 training clips, one run per fold), at width 192, the whole-clip spectrogram-only baseline read 96.0 ± 1.7%, the coupled oscillator network 91.9 ± 3.1% at gain 1 and 88.9 ± 3.1% at gain 2, the uncoupled network 92.3 ± 2.7%, and the state-matched leaky-integrator bank 93.9 ± 1.8% (gain 1). The trained baselines read 97.0% (CNN) to 98.4% (transformer). Read at width 4,096, the coupled network reached 96.9 ± 1.3% and the state-matched bank 96.8 ± 1.1% (gain 1). On the same folds Becker et al. report 95.82 ± 1.49% for AlexNet on spectrograms and 92.53 ± 2.04% for their AudioNet on the raw waveform.

## 3 Network design: coupling function, lattice geometry and natural frequencies

The design experiment varies the coupled oscillator network one factor at a time around the reference configuration (Kuramoto coupling, torus, random natural frequencies, restoring strength 0.3, coupling ceiling 1): six coupling functions, six lattice geometries and three kinds of natural frequencies, crossed with two restoring strengths and two coupling ceilings (Section 4 below). The two Stuart–Landau functions run on the torus only, so there are 312 configurations, each at 0 and −5 dB, gains 1 and 2 and three seeds, read at the primary cell. A level's difference from its reference is paired over every configuration that matches it in the other four factors, on the same test clips and seeds: 72 configurations for each phase coupling function, 12 for each Stuart–Landau function, 48 for each geometry and 104 for each kind of natural frequencies.

Across the 312 configurations, accuracy at gain 1 ran from 75.2% to 80.3% at 0 dB (median 77.4%) and from 68.3% to 73.0% at −5 dB (median 70.0%); the reference configuration read 77.8% and 70.7%. At gain 2 the ranges were 73.6% to 80.1% (median 74.8%) and 66.5% to 73.0% (median 68.3%).

The Stuart–Landau network, whose oscillators have a free amplitude, read 2.1 to 3.8 points above its Kuramoto match, the largest difference in the experiment. At gain 2 its twelve configurations were the twelve highest of the 312 at both noise levels; at gain 1 they ran from 78.1% to 80.3% at 0 dB, against 78.7% for the best phase-oscillator configuration. At the reference configuration it read above the whole-clip spectrogram-only baseline in all four conditions: by 2.31 points [1.63, 3.01] at 0 dB and 1.36 [0.62, 2.11] at −5 dB at gain 1, and by 1.14 [0.46, 1.88] and 1.38 [0.63, 2.16] at gain 2. With its amplitude fixed, the same network read within 0.3 points of Kuramoto. The other phase coupling functions differed from Kuramoto by at most 0.46 points: second harmonic above it by 0.15 to 0.46, Winfree below it by 0.15 to 0.34, and Kuramoto–Sakaguchi within 0.1.

Every lattice geometry read within 0.65 points of the torus. The sphere read lowest (0.30 to 0.64 points below), then the sheet (0.17 to 0.45 below) and the cylinder (0.01 to 0.26 below); the cube and the helix read within 0.21 points of the torus. Tonotopic natural frequencies read 0.38 to 1.16 points below random ones, and identical frequencies within 0.1 points of random ones.

The coil and the cochlea (Appendix C of the paper) ran for the four phase coupling functions and the three kinds of natural frequencies, and pair with the design experiment's torus and helix over those twelve configurations. The coil read within 0.22 points of the torus, and 0.10 to 0.28 points below the helix, the same line closed into a ring. The cochlea read 0.41 to 0.97 points below the torus and 0.27 to 0.83 points below the coil: 75.5% to 77.6% at 0 dB and gain 1 across its twelve configurations, against 77.8% for the torus reference, and 76.0% with Kuramoto coupling and tonotopic frequencies, its most cochlea-like configuration. Its curvature weighting also lowers its average coupling to 0.54 of the coil's, because the ceiling caps the coupling where it is strongest, at the apex, and its drop below the coil is the size of halving the coupling ceiling in the design experiment (0.39 to 0.68 points). A control with the cochlea's direction and shape of weighting at the coil's average coupling, run on the same twelve configurations, recovered 0.28 to 0.71 points of that drop, yet still read 0.40 to 0.47 points below the coil at gain 1 and within 0.12 points of it at gain 2.

| level minus reference | 0 dB, gain = 1 | 0 dB, gain = 2 | −5 dB, gain = 1 | −5 dB, gain = 2 |
|------------------|-------------|-------------|-------------|-------------|
| Kuramoto–Sakaguchi minus Kuramoto | −0.05 ± 0.23 | −0.01 ± 0.25 | +0.04 ± 0.07 | −0.09 ± 0.09 |
| second harmonic minus Kuramoto | +0.24 ± 0.09 | +0.15 ± 0.20 | +0.46 ± 0.21 | +0.30 ± 0.06 |
| Stuart–Landau minus Kuramoto | +2.06 ± 0.26 | +3.84 ± 0.32 | +2.30 ± 0.18 | +3.48 ± 0.12 |
| Stuart–Landau, fixed amplitude minus Kuramoto | +0.07 ± 0.12 | −0.24 ± 0.07 | +0.30 ± 0.23 | −0.26 ± 0.33 |
| Winfree minus Kuramoto | −0.34 ± 0.06 | −0.16 ± 0.14 | −0.17 ± 0.09 | −0.15 ± 0.09 |
| cube minus torus | +0.00 ± 0.30 | +0.04 ± 0.21 | +0.12 ± 0.10 | +0.05 ± 0.08 |
| cylinder minus torus | −0.26 ± 0.04 | −0.01 ± 0.13 | −0.26 ± 0.16 | −0.18 ± 0.10 |
| helix minus torus | +0.05 ± 0.13 | +0.04 ± 0.22 | +0.16 ± 0.17 | +0.21 ± 0.10 |
| sheet minus torus | −0.45 ± 0.32 | −0.17 ± 0.22 | −0.41 ± 0.14 | −0.26 ± 0.06 |
| sphere minus torus | −0.64 ± 0.15 | −0.30 ± 0.28 | −0.64 ± 0.26 | −0.57 ± 0.04 |
| identical minus random ω | −0.04 ± 0.05 | +0.09 ± 0.05 | −0.05 ± 0.17 | +0.01 ± 0.14 |
| tonotopic minus random ω | −1.16 ± 0.34 | −0.38 ± 0.09 | −0.68 ± 0.34 | −0.61 ± 0.11 |
| λ 0.1 minus 0.3 | −0.16 ± 0.09 | −0.13 ± 0.08 | −0.14 ± 0.09 | −0.23 ± 0.01 |
| ceiling 0.5 minus 1 | −0.59 ± 0.12 | −0.39 ± 0.08 | −0.68 ± 0.15 | −0.56 ± 0.08 |
| coil minus torus | −0.22 ± 0.31 | −0.15 ± 0.25 | +0.01 ± 0.48 | −0.06 ± 0.27 |
| cochlea minus torus | −0.97 ± 0.06 | −0.41 ± 0.11 | −0.76 ± 0.53 | −0.89 ± 0.20 |
| cochlea minus coil | −0.75 ± 0.26 | −0.27 ± 0.36 | −0.77 ± 0.21 | −0.83 ± 0.07 |
| coil minus helix | −0.28 ± 0.30 | −0.10 ± 0.13 | −0.17 ± 0.20 | −0.24 ± 0.11 |
| cochlea at the coil's coupling minus coil | −0.47 ± 0.38 | +0.07 ± 0.36 | −0.40 ± 0.35 | −0.12 ± 0.34 |
| cochlea minus cochlea at the coil's coupling | −0.28 ± 0.12 | −0.34 ± 0.16 | −0.37 ± 0.47 | −0.71 ± 0.32 |

Table: Each design level minus its reference level, over matched configurations: difference in recognition accuracy (points), mean ± standard deviation over three seeds; ω, natural frequencies; λ, restoring strength; ceiling, coupling ceiling. The coil and cochlea rows pair the four phase coupling functions and three kinds of natural frequencies at the reference restoring strength and ceiling. The 95% paired intervals for the coupling function, natural frequencies and lattice geometry rows are in the figures of Sections 4.3 and 4.4 of the paper.


## 4 Sensitivity of the canonical physics values

The restoring strength and the coupling ceiling are crossed with every other factor in the design experiment, so each difference is paired over 156 configurations. A restoring strength of 0.1 read 0.13 to 0.23 points below 0.3, and a coupling ceiling of 0.5 read 0.39 to 0.68 points below 1, in all four conditions (the last rows of the table in Section 3 below).

Adding each oscillator's rotation rates to the read raised accuracy by 0.13 to 0.17 points at gain 1 and by 0.47 to 0.49 points at gain 2, averaged over all 312 configurations. Per-clip correctness is kept only for the primary read in the design experiment, so this difference has no bootstrap interval.

A sweep extends the restoring strength, the coupling ceiling and the gain beyond the design experiment's levels, for every coupling function at the reference configuration, paired with the design experiment's reference cells on the same clips and seeds (the sweep's reservoirs ran on the Apple GPU and the design experiment's on its CPU, which agree within two test clips in all but one of 1,296 recorded cells; Section 3.5 of the paper). Restoring strengths of 0.5, 0.8 and 1.0 moved the phase-oscillator networks by −1.2 to +2.1 points against 0.3, with Kuramoto 0.2 to 0.8 points above; they lowered the Stuart–Landau network in every condition, by 0.4 to 2.8 points, more as the restoring strength grew. Coupling ceilings of 1.5 and 2 moved every coupling function by −2.4 to +1.7 points against 1, with no direction common to them.

Gain separates the Stuart–Landau network from the rest. From gain 1 to gain 8 it lost 4.3 points at 0 dB and 2.8 at −5 dB, while every phase-oscillator network lost 15 to 19 points and the Stuart–Landau network with its amplitude fixed 25 to 28. Beyond gain 8 it also gave way, losing 8.4 points by gain 10 and 20.8 by gain 12 at 0 dB (7.0 and 19.4 at −5 dB), against 24 to 38 points for the others.

Below gain 1 the order reverses. At gain 0.25 the four phase-oscillator networks read 0.27 to 1.12 points above gain 1 at 0 dB and 0.89 to 2.00 at −5 dB, and at gain 0.5 0.64 to 1.80 and 0.71 to 1.97, with every paired interval above zero except Winfree's at gain 0.25 and 0 dB (−0.19 to +0.72); the fixed-amplitude network gained 0.90 to 2.45. The Stuart–Landau network lost 1.73 and 0.99 points at gain 0.25 and 0.76 and 0.74 at gain 0.5. Against the whole-clip baseline, each phase-oscillator network at its better low gain read 0.73 to 1.57 points above at 0 dB, every interval above zero (Kuramoto 79.6% at gain 0.5), and the fixed-amplitude network 1.49; at −5 dB only the second-harmonic network read above, by 1.43 [+0.69, +2.22] at gain 0.25, and the rest within 0.2 points of the baseline. The best of them comes within 0.7 points of the Stuart–Landau network at gain 1.


## 5 Alternative input format: quadrature

Rather than just test a mel spectrogram front-end drive we wanted to see what a quadrature format would produce. The quadrature experiment ran ten coupled oscillator networks on the quadrature pathway: the four phase coupling functions on the torus and Kuramoto on the helix, each with random and tonotopic natural frequencies. They read 11.7% to 15.1% at gain 1 and 12.2% to 14.3% at gain 2, against a chance level of 10%. The spectrogram-only baseline read on the quadrature front end reached 36.1% at 0 dB and 28.2% at −5 dB over the whole clip, and 26.9% and 21.7% from frame 16. On the same clips and seeds, each network read 55.7 to 66.0 points below the same network driven through the spectrogram pathway. Within any one noise level and gain, the ten networks spanned at most 2.5 points, and Winfree with tonotopic frequencies read highest in three of the four conditions. The oscillators did not lock to the quadrature drive: their mean phase-locking value to their own band's drive ranged from 0.16 to 0.21 across runs, and in no run did more than 0.3% of them lock above 0.5.

## 6 Readout sufficiency: width and training size

Widening the readout from 192 to 4,096 features, at 2,048 training clips, raised the coupled oscillator network's recognition accuracy by 2.1 to 3.8 points at gain 1 and 2.8 to 6.3 points at gain 2, the uncoupled network's by 0.3 to 2.4 and 1.3 to 4.0 points, and the leaky-integrator banks' by 1.3 to 1.8 points on clean audio at gain 1, 2.9 to 3.9 points at gain 2, and 5.0 to 6.2 points with noise. The spectrogram-only baseline and four of the trained baselines are 192 wide, so a wider read leaves them unchanged; the GRU, 216 wide, gained 0.3 to 1.2 points.

At width 192, going from 2,048 to 24,000 training clips raised the whole-clip baseline by 4.1 points at 0 dB and 3.5 points at −5 dB, and the coupled network by 2.0 points at both; the trained baselines gained 7.1 to 10.8 points at 0 dB and 7.5 to 12.4 at −5 dB, the transformer most. With 24,000 clips and width 4,096, the coupled network read 85.9% at 0 dB and 78.3% at −5 dB, the state-matched bank 84.8% and 78.6%, and the whole-clip baseline 82.1% and 75.1%, against 86.7% to 90.6% at 0 dB and 81.4% to 85.4% at −5 dB for the trained baselines. At width 4,096 the coupled network read above the whole-clip baseline at every training size, and with 2,048 clips above three trained baselines at 0 dB, the S4D, CNN and transformer (81.4% against 79.2% to 80.8%); with 8,192 or 24,000 clips every trained baseline read above every reservoir at every width, the weakest above the best by 0.1 to 8.7 points.

![Recognition accuracy against the number of training clips, at 0 dB (left) and −5 dB (right), for the reservoirs at gain 1 read at width 192 (solid) and 4,096 (dashed), and for the arms that are 192 wide natively. Mean over three seeds; error bars, one standard deviation.](../resources/figures/c5-training-size.png)

Read through a projection drawn for each run seed instead of the one fixed draw, the controls experiment's reservoirs moved by −1.29 to +0.88 points on recognition and −0.63 to +1.22 points on order (seeded minus fixed, 48 paired comparisons, 35 of whose 95% intervals contain zero). Under the seeded projection the coupled network read 2.27, 0.40 and 0.69 points below the whole-clip baseline on recognition at gain 1 (clean, 0 dB, −5 dB), against 2.53, 0.23 and 0.88 under the fixed one; 0.58 to 1.46 points above the uncoupled network (0.43 to 1.08 fixed); and 4.0 to 4.8 points above the state-matched bank with noise (4.0 to 4.2 fixed).


## 7 Temporal order

The order task asks which of two recordings came first, digit a then b or b then a, on five digit pairs; the table averages each arm over the pairs, at 2,048 training clips and width 192, on the whole-span read. The spectrogram-only baseline read between 49.8% and 50.6% in every condition, whole clip or from frame 16. Its 95% interval over test clips contained chance (50%) in 14 of the 15 pair and noise cells; in the fifteenth, pair 3–7 at 0 dB, it read 48.4% [47.1, 49.8] over the whole clip.

At gain 1 the coupled oscillator network read 97.7%, 96.2% and 95.5% on clean audio, at 0 dB and at −5 dB, 45.4 to 47.9 points above the whole-clip baseline; at gain 2 it read 91.4% to 93.5%. The uncoupled network read below the coupled one by 2.2, 5.5 and 8.5 points at gain 1 (clean, 0 dB, −5 dB) and by 10.4, 15.7 and 18.7 points at gain 2. Both leaky-integrator banks read 98.7% to 99.9% at both gains, 2.2 to 7.5 points above the coupled network. Of the trained baselines, the transformer and the GRU read 99.0% to 99.9%, the S4D 96.9% to 99.1% and the TCN 91.1% to 99.9%. The CNN read 92.2% on clean audio and chance with noise (50.0%). Its two causal convolutions see 9 frames, 144 ms, less than a digit, and it has no memory beyond them, so statistics of its output over the whole sequence change with the digits' order only near the junction where one digit meets the other; on clean audio that cue carries it to 92.2%, and noise masks it. The limit is its window, not its training: 15 of its 30 runs with noise cut their training loss by more than 20% yet read chance on the test sequences, fitting the training sequences without learning order, while the TCN, a convolution that sees 31 frames, read 95.0% at 0 dB and 91.1% at −5 dB. The CNN therefore stands here as a learned model of local features with no memory, one step beyond the spectrogram-only baseline, which sees one frame at a time. The S4D first failed with noise as well: every state's time constant was initialized to about two frames, and all 30 of its runs with noise cut their loss by less than 20% and read chance. Initialized as S4D is, with its time constants spread across the sequence (here from 2 to 200 frames), and given a linear output, every S4D run in the study trained; the S4D reported throughout is this one, rerun in every experiment it appears in. Adding rotation rates to the coupled network's read changed its accuracy by −0.10 to +0.55 points at gain 1 and by +1.04 to +3.01 points at gain 2.

| arm | clean | 0 dB | −5 dB |
|------------------------------|----------|----------|----------|
| spectrogram-only baseline, whole clip | 49.9 ± 0.5 | 49.9 ± 0.3 | 50.1 ± 0.6 |
| coupled oscillator network (gain = 1) | 97.7 ± 0.6 | 96.2 ± 0.4 | 95.5 ± 0.7 |
| coupled oscillator network (gain = 2) | 92.9 ± 0.5 | 93.5 ± 0.1 | 91.4 ± 0.6 |
| uncoupled oscillator network (gain = 1) | 95.5 ± 0.6 | 90.7 ± 0.1 | 87.0 ± 0.4 |
| uncoupled oscillator network (gain = 2) | 82.5 ± 0.8 | 77.8 ± 0.4 | 72.7 ± 0.3 |
| leaky-integrator bank, state-matched (gain = 1) | 99.9 ± 0.0 | 99.6 ± 0.1 | 98.8 ± 0.1 |
| leaky-integrator bank, state-matched (gain = 2) | 99.7 ± 0.0 | 99.6 ± 0.1 | 98.9 ± 0.1 |
| leaky-integrator bank, width-matched (gain = 1) | 99.9 ± 0.1 | 99.5 ± 0.1 | 98.7 ± 0.3 |
| leaky-integrator bank, width-matched (gain = 2) | 99.8 ± 0.0 | 99.6 ± 0.1 | 99.0 ± 0.1 |
| transformer | 99.9 ± 0.0 | 99.6 ± 0.1 | 99.3 ± 0.1 |
| GRU | 99.8 ± 0.1 | 99.6 ± 0.1 | 99.0 ± 0.1 |
| S4D | 99.1 ± 0.1 | 98.3 ± 0.7 | 96.9 ± 0.2 |
| CNN | 92.2 ± 0.6 | 50.0 ± 0.4 | 50.0 ± 0.4 |
| TCN | 99.9 ± 0.1 | 95.0 ± 0.4 | 91.1 ± 0.3 |

Table: Temporal-order accuracy (%), averaged over the five digit pairs, mean ± standard deviation over three seeds; chance is 50%.

# Spoken-digit prior art

Published spoken-digit results relevant to papers 01 to 03, grouped by kind. The category tables hold the papers one of our papers cites; the last table holds papers found but not cited, with the reason. Only digit classification is listed, not AudioMNIST's speaker or sex tasks.

- **Speaker-disjoint**: *yes* when no speaker is in both training and test, *no* when speakers are shared, *not stated* when the paper does not say. A shared-speaker split lets any classifier, a trained network or a reservoir's readout, learn the test speakers' voices as well as their digits, so its accuracy likely overstates recognition of new speakers and is not comparable with a speaker-disjoint result.
- **Verified** (✓): the row's figures and protocol were checked against the paper's full text (the published version, PMC, arXiv or an author's copy). A blank means the row comes from an automated literature search: a lead, not a fact.
- **Cited in**: which of our papers cite it.
- Audio is clean unless the row says otherwise. SNR is signal-to-noise ratio.

## What the tables show

- Speaker-disjoint AudioMNIST results exist only for clean audio with 18,000 to 24,000 training clips: trained networks reach 92.5% to 98.6%, and Hydra, untrained random convolution kernels with a ridge classifier, 97.3%.
- No AudioMNIST result combines held-out speakers with added noise. Both noisy results found share speakers (Beoletto 2026, Young 2026).
- Most spoken-digit results for reservoirs and oscillator networks share speakers: every TI-46 study here, and on AudioMNIST, Buckley 2026 and Chandravadia & Imam 2026.
- Several find that the front end, not the reservoir, carries most of the accuracy: Abreu Araujo 2020, Buckley 2026, Hikasa 2026.
- No LinOSS, UnICORNN or Loihi-class result on AudioMNIST was found.

## A. AudioMNIST, clean, speaker-disjoint

| Source | Accuracy | Split, training size | Speaker-disjoint | Input, model | Verified | Cited in |
|---|---|---|---|---|---|---|
| Becker, Vielhaben, Ackermann, Müller, Lapuschkin, Samek 2024, J. Franklin Inst. 361(1):418–428, doi:10.1016/j.jfranklin.2023.11.038 (checked against arXiv:1807.03418; the journal version is paywalled) | AlexNet 95.82 ± 1.49%; AudioNet 92.53 ± 2.04% | 5 folds of 12 speakers; 18,000 train, 6,000 validation, 6,000 test | Yes | Spectrogram 228×230 or raw waveform, 8 kHz; model size not stated | ✓ | 02, 03 |
| Saadatmand, Webb, Rezatofighi, Salehi 2026, arXiv:2605.27406, Table 3 | Hydra 97.3% (untrained random kernels, ridge classifier). At 48 kHz also ConvTran 98.6, MS4N 98.2, MS4 96.1, Mamba2 90.4, Transformer 80.9; LSTM, GRU and FCN at chance. At 4 kHz: MS4N 95.2, ConvTran 94.9, GRU 94.5, Hydra 94.5 | MONSTER's 5 folds (Dempster et al. 2025, arXiv:2502.15122, §3.1.1); about 24,000 train, less 10% for early stopping | Yes | Raw waveform | ✓ (Hydra and the folds) | 02 |

## B. AudioMNIST with added noise

| Source | Accuracy | Split | Speaker-disjoint | Noise | Input, model | Verified | Cited in |
|---|---|---|---|---|---|---|---|
| Beoletto, Milano, Ricciardi, Bosia, Gliozzi 2026, Adv. Intell. Syst. 8(1):2500526, doi:10.1002/aisy.202500526 | Hardware, 3,000 clips of 6 speakers: tonogram 98.7 ± 0.6%, MFCC 98.6 ± 0.5%, cochleagram 98.4 ± 0.6%, linear spectrogram 75.5 ± 1.9%. Simulated, 30,000 clips: 94.4 ± 0.3% clean, 84.3 ± 0.6% at SNR −5 dB; about 88% at 0 dB and 90.5% at +5 dB are readings of Fig. 7e, not stated in the text | 10-fold cross-validation with each speaker's clips spread across folds; 27,000 train per fold | No | Additive Gaussian, SNR −5 to 20 dB, simulated with a linear filter-bank model of the resonator; whether training clips were noised is not stated | One passive spiral metamaterial resonator, not coupled oscillators; 16×16 features; linear pseudoinverse readout, 2,560 weights; 48 kHz | ✓ | 02 |
| Young, Fiorio, Yang, Karanov, van Houtum, Aarts 2026, EURASIP J. Audio Speech Music Process. 2026:27, doi:10.1186/s13636-026-00457-2 (arXiv:2504.03497) | 98% clean with 18k parameters. The preprint gives only test loss under noise; the published version adds a figure testing models trained clean, at 0 dB and at −5 dB, from −10 to +10 dB SNR | Random 80/10/10 of the clips | No | White Gaussian, SNR 0 and −5 dB in training | Hybrid real- and complex-valued CNN (the paper's HNN) against a real-valued CNN (its RVNN), both from architecture search; complex STFT at 48 kHz, RMS-normalized | ✓ | 02 |

## C. Untrained and physical reservoirs on spoken digits

| Source | Corpus | Accuracy | Split | Speaker-disjoint | Reservoir, input, readout | Verified | Cited in |
|---|---|---|---|---|---|---|---|
| Zhou et al. 2026, Nat. Nanotechnol. 21(4), doi:10.1038/s41565-026-02133-0 (arXiv:2512.22722) | AudioMNIST, the first 3,000 clips | 95.3%, above the same network uncoupled and above no processing | 5-fold 80/20 | Not stated | Simulated network of 128 coupled protonic nickelate devices, not oscillators; Lyon ear model, 64 channels, binary spikes; linear regression readout. Its supplement's "six speakers" is wrong for AudioMNIST | ✓ | 02 |
| Buckley, Schumm, Askenazi, Rietman 2026, arXiv:2604.00207 | AudioMNIST, 6,000 random clips | PZT cube 88.2%; logistic regression on the same input 88.1%; CNN 90.1% | Random 80/20 | No | A material reservoir, not an oscillator; 32×32 binarized MFCC map | ✓ | 02 |
| Torrejon et al. 2017, Nature 547:428, doi:10.1038/nature23011 | TI-46, 500 clips of 5 female speakers | Up to 99.6% with cochlear features; up to 80% with a linear spectrogram, where the readout alone is at chance | Same speakers in training and test | No | One time-multiplexed spin-torque oscillator in hardware | ✓ | 01, 02 |
| Abreu Araujo et al. 2020, Sci. Rep. 10:328, doi:10.1038/s41598-019-56991-x | TI-46, 500 clips of 5 speakers; AURORA-2 | Front end alone: cochleagram up to 95.8%, MFCC up to 77.2%, linear spectrogram at chance. With the experimental oscillator and MFCC, up to 99.8% (Table 2). On AURORA-2, noise at 20, 15 and 10 dB SNR, simulated, every result reported as a gain over the front end | TI-46: same speakers. AURORA-2: disjoint | TI-46 no; AURORA-2 yes | One spin-torque oscillator. The authors attribute the accuracy mainly to the front end and recommend reporting gain over it | ✓ | 01, 02 |
| Opala et al. 2019, Phys. Rev. Applied 11:064029, doi:10.1103/PhysRevApplied.11.064029 | TI-46 | Not recorded here | Shared speakers | No | Simulated Ginzburg–Landau lattice of untrained coupled oscillators; no added noise; no input-only or uncoupled control | ✓ | 02 |
| Coulombe, York, Sylvestre 2017, PLOS ONE 12:e0178663, doi:10.1371/journal.pone.0178663 | TI-46 | Not recorded here | Shared speakers | No | Simulated chain of 400 Duffing oscillators; no added noise; no input-only or uncoupled control | ✓ | 02 |
| Srinivasan, Panda, Roy 2018 (SpiLinC), Front. Neurosci. 12:524, doi:10.3389/fnins.2018.00524 | TI-46 digits, 16 speakers | One liquid: 77.7% with 800 neurons, 86.66% with 3,200; two liquids of 1,600: 85.14% | 1,594 train, 2,542 test, from TI-46's sets of the same speakers | No | A liquid of spiking neurons shaped by unsupervised STDP; accuracy grows with the liquid's size | ✓ (against the author's excerpt) | 02, 03 |
| Appeltant et al. 2011, Nat. Commun. 2:468 | TI-46 | 0.2% word error rate | Shared speakers | No | One electronic nonlinear node with delayed feedback; not self-oscillating | ✓ | 01 |
| Shougat et al. 2023, Sci. Rep. 13:8719, doi:10.1038/s41598-023-35760-x | FSDD, 1,000 clips | About 97% | 80/20; how clips were assigned is not stated | Not stated | One analog Hopf-oscillator circuit feeding a trained CNN | ✓ | 02 |

## D. Trained oscillator networks

| Source | Task | Accuracy | Split | Speaker-disjoint | Model | Verified | Cited in |
|---|---|---|---|---|---|---|---|
| Chandravadia & Imam 2026, Patterns 7(8):101563, doi:10.1016/j.patter.2026.101563 (PMC13494615) | AudioMNIST digits; Arabic digits | coRNN 89 ± 1% after 50 epochs (bioRxiv version 89.20 ± 0.88%); plain RNN 5 ± 2%; Arabic digits 78 ± 8%. The paper's 91% is the variance three principal components capture, not an accuracy | Random 90/10 of the clips; 27,000 train, 3,000 test; 10 initializations | No | 256 coupled oscillators on the raw waveform at 8 kHz, trained end to end | ✓ | 01, 02 |
| Romera et al. 2018, Nature 563:230, doi:10.1038/s41586-018-0632-y | Seven vowels from formant pairs | 88% test | Not recorded here | Not stated | Four coupled spin-torque oscillators in hardware, trained by tuning their frequencies | ✓ | 01 |
| Ahmadi 2026, arXiv:2604.10272 | Two vowels from formants | Not recorded here | Not recorded here | Not stated | Simulated nine-oscillator Kuramoto network trained by equilibrium propagation | ✓ | 01 |

## Found but not cited

| Source | Kind | What it reports | Speaker-disjoint | Verified | Why not cited |
|---|---|---|---|---|---|
| Peic Tukuljac, Ricaud, Aspert, Colbois 2022, Proc. NLDL 3, doi:10.7557/18.6279 | AudioMNIST, trained, clean | AudioNet reimplementation 94.9 ± 1.54%, LF-AudioNet 96.8 ± 1.22%, Gammatone 98.0 ± 0.41%, SincNet 97.2 ± 1.0%, Wavelet 89.9 ± 1.18% | Yes (Becker's folds, reused through Becker's code) | | Not verified; paper 02 places its results beside Becker et al.'s alone |
| Kalthoff, Rey, Wittkowski 2025, arXiv:2511.21313 | AudioMNIST, trained, clean | SincHSRNN at 8 kHz 96.51 ± 1.58% (95.11% constrained); 84.5% at 2 kHz, 73.9% at 1 kHz | Yes (80/20 by speaker; 24,000 train) | | Not verified; same reason |
| Choudhury, Singh, Roy, Bandyopadhyay 2026, LNNS 1668, doi:10.1007/978-3-032-07735-6_22 | AudioMNIST, noise | "Above 90% ... even in noisy environments", EfficientNet on mel images; babble and white noise | Not stated | | Abstract only; split and SNRs not stated |
| GLaSS-CNN | AudioMNIST, noise | Abstract claims 99.48% clean, 99.3% on FSDD and 90.5% at SNR 0 dB | Not stated | | Full text not accessible, so its split and noise protocol are unknown; looks low quality |
| Sebastian, O'Keefe, Trefzer 2026, arXiv:2603.24283 | AudioMNIST, echo state network | 93.08% on standard MFCCs, 84.81% on time-domain MFCCs; 400 nodes, ridge readout, 14 MFCCs at 8 kHz | Not stated (5 partitions, rule not stated) | | Split not stated |
| "Liquid-SNN 82.65% on Audio-MNIST", from a comparison table in Sebastian et al. 2026 | Secondary figure | Cites SpiLinC, which is a TI-46 study, and 82.65% is not among SpiLinC's figures | n/a | ✓ | Mislabels the corpus; cite SpiLinC directly for TI-46 |
| Sebastian et al. 2026, arXiv:2606.21335 | AudioMNIST, raw-audio reservoir | 43.59% and 64.10% test on raw frames of 250 samples | Not stated | | Split not stated; the figures suggest a 39-clip test set |
| Kumar, Jin, Bu, Kumar, Huang 2021, OSA Continuum 4(3):1086, doi:10.1364/OSAC.417996 | Photonic reservoir, first 5 speakers | 0.34 ± 0.51% word error rate; 77-channel cochleagram at 12 kHz, 400 virtual nodes | No (400 train, 100 test, same speakers) | | Five speakers, shared; Torrejon and Abreu Araujo make the point |
| Carpegna, Savino, Di Carlo 2024, IEEE TETC, doi:10.1109/TETC.2024.3511676 | AudioMNIST, spiking network | 95.23% in software, 89.82% on FPGA; 40-150-10 LIF network, 40-band binarized mel | Not stated (24,000 train, 6,000 test) | | Split rule not stated; spiking hardware is outside paper 02's comparison |
| Sekertzis & Dimitrakopoulos 2026, arXiv:2606.20675 | AudioMNIST, spiking network | 90.60% on FPGA, Carpegna's network with vector quantization | Not stated | | Same reason |
| Wang et al. 2025, Sci. Adv. 11(30):eadv3436, doi:10.1126/sciadv.adv3436 | AudioMNIST, memristor | 94.72%, a memristor analog DFT feeding a 4-layer CNN | Not stated | | Split not stated; outside the comparison |
| Diep, Phan, Truong 2024, PLoS ONE 19(4):e0302394, doi:10.1371/journal.pone.0302394 | AudioMNIST, CNNs | 1D-CNN 95.87%, spectrogram CNN 99.20%, mel CNN 99.76% | No (random 70/10/20) | | Shared speakers |
| Sridhar & Kanhe 2023, J. Phys.: Conf. Ser. 2466:012008, doi:10.1088/1742-6596/2466/1/012008 | CNN and LSTM | CNN 96.4% to 98.4%, LSTM 95.2% to 97.0%, Conv-LSTM up to 98.6%, validation accuracies | No (random 90/10) | | Describes "10000" samples and does not cite Becker, so the dataset is ambiguous |
| Pagadala & Chaitanya 2026, SIVP 20:418 | AudioMNIST, LSTM | "Up to 96.54%" | Not stated | | Abstract only |
| Bruesch et al. 2025, arXiv:2406.13584 | AudioMNIST | 96.9% | Not stated | | Split not stated |
| Tripathi & Paul 2022; Oliver & Hopkinson 2026; Fida & Mittal 2026; Sridhar & Kanhe 2026 | AudioMNIST | No accuracy could be verified | Not stated | | Nothing to cite |
| "98.3% on AudioMNIST (Becker)", from a secondary review | Secondary figure | Appears nowhere in Becker et al. | n/a | ✓ | Never cite it to Becker; its source and what it measures are unknown |
| Hikasa et al. 2026, arXiv:2603.29311 | TI-46, spin-wave reservoir | Cochleagram plus reservoir 95.2%; cochleagram alone 85.8%; reservoir on raw audio 29.0% | Not stated | ✓ | Supports the front-end finding paper 02 makes with Abreu Araujo and Buckley; a third source if needed |
| Zheng et al. 2021 | TI-46, two coupled MEMS resonators | Not recorded here | No | ✓ | Opala and Coulombe already show untrained coupled oscillators on TI-46 with shared speakers |
| Kimura et al. 2025 | TI-46, ring-oscillator lattice | Not recorded here; has an input-only control | No | ✓ | Same reason; its input-only control makes it a candidate if a reviewer asks |
| Zolfagharinejad et al., arXiv:2410.10434 | TI-46 | 96.2% | Not stated | | Not verified |
| Rajesh et al., arXiv:2411.19611 | FSDD | 98.95% | Not stated | | Not verified; another corpus |
| Kumari et al., arXiv:2605.29043 | FSDD | 95.67% | Not stated | | Not verified; another corpus |
| Ahmed 2025, doi:10.55730/1300-0632.4153 | FSDD, noise | 99.3% clean, 88.5% at SNR 0 dB | Not stated | | Not verified; another corpus |
| S4, arXiv:2111.00396 | Speech Commands, 10 classes, raw audio | 98.3% | Yes (the dataset's split is by speaker) | | Another corpus; paper 01 cites S4 for its architecture, not this result |
| Dutta et al. 2019; Abernot et al. 2021; Corti et al. 2020; Moy et al. 2022 | Oscillator hardware on vowels, patterns and optimization | Oscillator hardware on tasks other than spoken digits | n/a | | Paper 01 makes the point with Romera 2018 |

# AudioMNIST digit-classification prior art

Gathered by an automated literature search on 18 September 2026. **Treat every number here as unverified until checked against the paper**, except where a row says otherwise. Each row keeps the search's own confidence label: *Direct* means the raw PDF or HTML was inspected, *Extracted* means quotes were pulled through a fetch tool, *Abstract only* means the paper was paywalled. All audio is clean unless noted. Only digit classification is listed, not AudioMNIST's speaker or gender tasks.

## Headline

No published AudioMNIST result combines a speaker-disjoint split with additive noise, or with a small training set. The published numbers are loose upper context, never head-to-head comparisons.

- **Speaker-disjoint results exist only for clean audio with 18,000 to 24,000 training clips.** Trained networks reach 92.5 to 98.6%. Hydra, untrained random convolution kernels plus a ridge classifier, is reported at 97.3%.
- **The only digit accuracies under additive noise** come from Beoletto 2026: about 88% at 0 dB and about 90.5% at +5 dB. Those are readings of a figure, with speakers shared between train and test, 27,000 training clips, and simulated features.
- **Every small-training-set result leaks speakers.**

Several papers find that the front end, not the reservoir, carries most of the accuracy: Abreu Araujo 2020, Buckley 2026, Hikasa 2026.

## A citation problem in the draft

Draft line 281 reads "coRNN on AudioMNIST 78-91% under different frontends (Rusch 2021)". Rusch & Mishra 2021 reports no audio experiment: its tasks are the adding problem, sMNIST, psMNIST, noisy CIFAR-10, HAR-2, IMDB and Lorenz-96. The coRNN-on-AudioMNIST result belongs to Chandravadia & Imam 2026 (Patterns 7(8):101563), which the search reads as 89 +/- 1% with a **random** 90/10 split. It reads that paper's 77.95% as Arabic digits, and found no value near 91%. Paper 01's audited text independently describes Chandravadia & Imam as the coRNN raw-waveform speech study, which confirms the attribution. The numbers themselves are not yet checked.

## Numbers to avoid

- "98.3% on AudioMNIST (Becker)" appears in a secondary review. Becker does not report it.
- "Liquid-SNN 82.65% on Audio-MNIST" in Sebastian et al.'s table cites SpiLinC, which used TI-46.
- "GLaSS-CNN 99.48% clean, 90.5% at 0 dB" was found only in search snippets, with no primary source.

## A. Speaker-disjoint split, clean audio

| Source | Accuracy | Split, training size | Input | Model size |
|---|---|---|---|---|
| Becker et al. 2024, J. Franklin Inst. 361(1):418-428, doi:10.1016/j.jfranklin.2023.11.038. Peer-reviewed. Extracted. | AlexNet 95.82 +/- 1.49%; AudioNet 92.53 +/- 2.04% | 5-fold divided by speaker, 12 speakers per split; 18,000 train, 6,000 validation, 6,000 test | 8 kHz; spectrogram 228x230 or raw 8,000 samples | Not stated in Becker; two secondary sources conflict, about 17M versus about 500k |
| Peic Tukuljac, Ricaud, Aspert, Colbois 2022, Proc. NLDL 3, doi:10.7557/18.6279. Review status not stated. Direct. | AudioNet reimplementation 94.9 +/- 1.54%; LF-AudioNet 96.8 +/- 1.22%; Gammatone 98.0 +/- 0.41%; SincNet 97.2 +/- 1.0%; Wavelet 89.9 +/- 1.18% | Reused Becker's 5-fold code, so the speaker split is inherited rather than restated | Raw waveform, 8 kHz | 17M; 3.5M; 300k; 300k; 300k |
| Kalthoff, Rey, Wittkowski 2025, arXiv:2511.21313. Preprint. Extracted. | SincHSRNN at 8 kHz: 96.51 +/- 1.58% unconstrained, 95.11 +/- 1.67% constrained. At 2 kHz: 84.5% and 81.8%. At 1 kHz: 73.9% and 66.9% | 80/20 with no speaker in both sets; 24,000 train; 5 runs | Raw intensity, 1 to 8 kHz | Not stated |
| Saadatmand, Webb, Rezatofighi, Salehi 2026, arXiv:2605.27406v2, Table 3. Preprint. Direct. Uses the MONSTER speaker folds (Dempster et al. 2025, arXiv:2502.15122). | At 48 kHz raw: ConvTran 98.6; MS4N 98.2; Hydra 97.3; MS4 96.1; Mamba2 90.4; Quant 87.9; Mamba1 87.2; Informer 82.5; Transformer 80.9; HInception 78.9; ET 46.0; LSTM, GRU and FCN at chance. At 4 kHz: MS4N 95.2; ConvTran 94.9; GRU 94.5; Hydra 94.5; MS4 93.3 | 5-fold CV; about 24,000 train, minus 10% for early stopping | Raw waveform, length 47,998 or 4,000 | MS4N about 25K. Hydra is untrained random kernels plus a ridge classifier |

## B. Noise added (none is speaker-disjoint)

| Source | Accuracy | Split | Noise | Input, readout |
|---|---|---|---|---|
| Beoletto, Milano, Ricciardi, Bosia, Gliozzi 2026, Adv. Intell. Syst. 8(1):2500526, doi:10.1002/aisy.202500526. Peer-reviewed. Direct. | Hardware, 3,000 clips from 6 speakers: tonogram 98.7 +/- 0.6%; MFCC 98.6 +/- 0.5%; cochleagram 98.4 +/- 0.6%; linear spectrogram 75.5 +/- 1.9%. Simulated, 30,000 clips: 94.4 +/- 0.3% clean; 84.3 +/- 0.6% at -5 dB (stated). About 88% at 0 dB and about 90.5% at +5 dB are **readings of Fig. 7e, not stated in the text** | 10-fold CV with each speaker spread across folds, so speakers leak; 27,000 train per fold | Additive Gaussian, -5 to 20 dB. Whether training data was noised is not stated | 16x16 features from a cochlea-like resonator; linear pseudoinverse readout, 2,560 weights; 48 kHz |
| Young et al. 2026, EURASIP J. Audio Speech Music Process., doi:10.1186/s13636-026-00457-2. Peer-reviewed. Extracted from arXiv:2504.03497. | 98% clean. Under noise only the loss is given, no accuracy | Random 80/10/10 | White Gaussian at 0 dB and -5 dB | Complex STFT at 48 kHz; 18k params |
| Choudhury, Singh, Roy, Bandyopadhyay 2026, LNNS 1668, doi:10.1007/978-3-032-07735-6_22. Peer-reviewed. **Abstract only.** | "above 90% ... even in noisy environments" | Not stated | Babble and white; SNRs not stated | EfficientNet-B0 to B7 on mel images |

## C. Frozen or physical reservoirs

| Source | Accuracy | Split, size | Input, readout |
|---|---|---|---|
| Zhou et al. 2026, Nat. Nanotechnol., doi:10.1038/s41565-026-02133-0 (arXiv:2512.22722). Peer-reviewed. Direct. | 95.3%, simulated 128-node reservoir | 5-fold 80/20; whether by speaker is not stated; details in a supplement not seen | Lyon ear model, 64 channels, binary spikes; linear regression readout |
| Buckley, Schumm, Askenazi, Rietman 2026, arXiv:2604.00207. Preprint. Extracted. | PZT cube 88.2%; logistic regression on the same input 88.1%; CNN 90.1% | 6,000 random clips, 80/20 | 32x32 binarized MFCC map |
| Sebastian, O'Keefe, Trefzer 2026, arXiv:2603.24283. Preprint. Extracted. | ESN 93.08% with standard MFCCs; 84.81% with time-domain MFCCs | 5 partitions, rule not stated; train or test not stated | 14 MFCCs at 8 kHz; 400 nodes; ridge readout |
| Sebastian et al. 2026, arXiv:2606.21335. Preprint. Extracted. | Raw-audio reservoirs: 43.59% and 64.10% test | Not stated; values consistent with a 39-clip test set, which is the search's inference | Raw frames of 250 samples |
| Kumar, Jin, Bu, Kumar, Huang 2021, OSA Continuum 4(3):1086, doi:10.1364/OSAC.417996. Peer-reviewed. Extracted. | Word error rate 0.34 +/- 0.51% | First 5 speakers only; 400 train, 100 test; same speakers in both | 77-channel cochleagram at 12 kHz; 400 virtual nodes |

## D. Spiking, hardware, oscillatory RNN and random-split CNNs

| Source | Accuracy | Split | Notes |
|---|---|---|---|
| Carpegna, Savino, Di Carlo 2024, IEEE TETC, doi:10.1109/TETC.2024.3511676. Peer-reviewed. Direct. | SNN 95.23% software; 89.82% FPGA | 24,000 train, 6,000 test; rule not stated | 40-150-10 LIF network, 7,500 synapses; 40-band binarized mel |
| Sekertzis & Dimitrakopoulos 2026, arXiv:2606.20675. Preprint. Extracted. | 90.60% on FPGA | Not stated | Same network as Carpegna, with vector quantization |
| Wang et al. 2025, Sci. Adv. 11(30):eadv3436, doi:10.1126/sciadv.adv3436. Peer-reviewed. Extracted. | 94.72% | Not stated in the main text | Memristor analog DFT feeding a 4-layer CNN |
| Chandravadia & Imam 2026, Patterns 7(8):101563, doi:10.1016/j.patter.2026.101563. Peer-reviewed. Extracted. | coRNN 89 +/- 1%; bioRxiv version 89.20 +/- 0.88%; plain RNN 5 +/- 2% | **Random** 90/10; 27,000 train; 10 seeds | Raw waveform at 8 kHz; 256 units |
| Diep, Phan, Truong 2024, PLoS ONE 19(4):e0302394, doi:10.1371/journal.pone.0302394. Peer-reviewed. Extracted. | 1D-CNN 95.87%; spectrogram CNN 99.20%; mel CNN 99.76% | Random 70/10/20 | 85k; 336k; 151k params |
| Sridhar & Kanhe 2023, J. Phys.: Conf. Ser. 2466:012008, doi:10.1088/1742-6596/2466/1/012008. Peer-reviewed conference. Direct. | CNN 96.4 to 98.4%; LSTM 95.2 to 97.0%; Conv-LSTM up to 98.6% | 90/10 random; validation accuracies | Describes "10000" samples and does not cite Becker, so the dataset identity is ambiguous |

Minor or unverified: Pagadala & Chaitanya 2026 (SIVP 20:418), LSTM "up to 96.54%", abstract only. Bruesch et al. 2025 (arXiv:2406.13584), 96.9%, split not stated. No accuracy could be verified for Tripathi & Paul 2022, Oliver & Hopkinson 2026, Fida & Mittal 2026, or Sridhar & Kanhe 2026. No LinOSS, UnICORNN or Loihi-class result on AudioMNIST was found.

## Near-misses on other datasets (NOT AudioMNIST)

**TI-46.** Torrejon 2017: up to 99.6% with cochlear filtering, 80% with a linear spectrogram; speaker-dependent, the same 5 speakers in train and test. Abreu Araujo et al. 2020, Sci. Rep. 10:328, doi:10.1038/s41598-019-56991-x: the cochlear front end alone reaches 95.8%, MFCC alone 77.2%, a linear spectrogram alone about 10%, up to 99.8% with the oscillator; it concludes recognition "is mainly due to" the front end, "not to the reservoir itself". Appeltant 2011: word error rate 0.2%. Hikasa et al. 2026, arXiv:2603.29311: cochleagram plus spin-wave reservoir 95.2%, versus 29% with raw audio. Zolfagharinejad et al., arXiv:2410.10434: 96.2%.

**FSDD.** Shougat et al. 2023, Sci. Rep. 13:8719, a Hopf-oscillator reservoir, about 97% with a CNN readout and a random split. Rajesh et al., arXiv:2411.19611: 98.95%. Kumari et al., arXiv:2605.29043: 95.67%. Ahmed 2025, doi:10.55730/1300-0632.4153: 99.3% clean, 88.5% at 0 dB.

**Speech Commands.** S4 reaches 98.3% on the raw 10-class subset (arXiv:2111.00396).

## Not yet in either bibliography

Most useful additions: Peic Tukuljac 2022; Kalthoff 2025; Saadatmand 2026 with Dempster 2025 (MONSTER); Beoletto 2026; Zhou 2026; Buckley 2026; Carpegna 2024; Abreu Araujo 2020; Shougat 2023. Already cited: becker2024, rusch2021, torrejon2017 in Paper 02; chandravadia2026, appeltant2011, s4 in Paper 01 only.

# Calibration audio examples (the noise grid of §4.4)

> **Note:** These are just examples of what the different noise levels sounded like that we used to settle on the +0dB noise level for audio input in the experiments and examples of the ordered digit task audio input. Meant to support the reader in understanding the paper's experimental setup and results. These are NOT the only audio samples used in the paper, just a sample of what they sound like.

Same clip in every file — AudioMNIST digit 7 ("seven"), test-pool speaker 49, first rep (bank index 140), 11487 valid samples @ 16000 Hz — rendered clean and at each calibration noise level. Noise mirrors `harness.stimuli.make_digit_clips` exactly (white noise, sigma = speech-region RMS x 10^(dB/20), full padded window); one seeded draw (seed 424242) rescaled per level, so noise AMPLITUDE is the only variable across files. All files share one common scale factor (0.5313) so relative levels are exact and nothing clips at 16-bit.

| file | added noise (rel. speech RMS) |
|---|---|
| clean.wav | none (the unmodified sample) |
| noise_-10db.wav | 0.32x (-10 dB) |
| noise_-5db.wav | 0.56x (-5 dB) |
| noise_+0db.wav | 1.00x (+0 dB) |
| noise_+5db.wav | 1.78x (+5 dB) |
| noise_+10db.wav | 3.16x (+10 dB) |

Regenerate: `uv run python scripts/make_calibration_audio.py` (deterministic). The grid and the level-pick rule are stated in §4.4 of the paper.

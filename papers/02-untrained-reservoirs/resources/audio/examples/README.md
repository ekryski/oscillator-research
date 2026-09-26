# Audio examples

Example audio clips from the experiment: each is the waveform the front end received, sample for sample, with the same seeded noise, made by the harness's own `recognition_clips` and `order_clips` (`src/harness/experiment/protocol.py`). The recordings are from AudioMNIST (Becker et al. 2024), under its MIT licence (`LICENSE-AudioMNIST.txt`): resampled from 48 kHz to 16 kHz, peak-normalized, trimmed of leading and trailing silence and capped at 1 s. White noise is added over the whole window at a level set by the clip's own speech, given as the signal-to-noise ratio: at 0 dB the noise is as loud as the speech, at −5 dB it is 5 dB louder.

| file | what it is |
|---|---|
| `digit-7-speaker-49-clean.wav` | digit 7, test speaker 49, clean |
| `digit-7-speaker-49-snr0db.wav` | digit 7, test speaker 49, 0 dB |
| `digit-7-speaker-49-snr-5db.wav` | digit 7, test speaker 49, −5 dB |
| `order-3-then-7-clean.wav` | order task, pair 3–7: 3 then 7, clean |
| `order-7-then-3-clean.wav` | order task, pair 3–7: 7 then 3, clean |
| `order-3-then-7-snr0db.wav` | order task, pair 3–7: 3 then 7, 0 dB |
| `order-7-then-3-snr0db.wav` | order task, pair 3–7: 7 then 3, 0 dB |
| `order-3-then-7-snr-5db.wav` | order task, pair 3–7: 3 then 7, −5 dB |
| `order-7-then-3-snr-5db.wav` | order task, pair 3–7: 7 then 3, −5 dB |

The recognition clip is one second, the recording padded with silence before the noise is added. It is a test recording: the readout never saw speaker 49 in training. The order clips are the first two sequences of the pair's test set, one in each order: a silent leader, one recording, a 100 ms gap, the other recording, then silence to the fixed length, with the noise over all of it. Their four recordings come from test speakers 53, 55, 56, drawn independently as in every order sequence.

The files are 32-bit float WAV at 16,000 Hz. No sample reaches full scale, so they are unscaled.

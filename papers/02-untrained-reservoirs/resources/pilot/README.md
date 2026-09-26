# Pilot

The noise calibration from the pilot that preceded paper 02's study, kept for reference. It is not part of the study's record in `results/`, and the paper reports none of it.

| path | what it holds |
|---|---|
| `make_calibration_audio.py` | renders one AudioMNIST clip clean and at every noise level of the calibration grid (−10 to +10 dB, one seeded noise draw rescaled per level), so the noise's amplitude is the only thing that changes |
| `calibration/` | the clips it rendered, with their README |
| `noise-calibration.json` | the pilot's noise calibration record |

The script ran against the pilot's harness, whose `harness.stimuli.load_digit_bank` the study's code replaced with `harness.experiment.protocol.load_bank`; to rerun it, port that one call. Its README cites the pilot's section numbers. The noise levels written "+0 dB" and "+5 dB" in the pilot are the noise's level relative to the speech, which the paper reports as signal-to-noise ratios of 0 and −5 dB.

Taken from branch `ek/paper-02-pilot` at commit 4b7860d (tag `archive/paper-02-pilot`), which also keeps the pilot's harness and its full record, superseded by the study's.

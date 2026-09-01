# Order-task audio examples

> **Note:** These are just examples of what the different noise levels sounded like that we used to settle on the +0dB noise level for audio input in the experiments and examples of the ordered digit task audio input. Meant to support the reader in understanding the paper's experimental setup and results. These are NOT the only audio samples used in the paper, just a sample of what they sound like.

For a plain-language walkthrough of what the field is doing with these sounds,
see [How a machine hears a number](https://erickryski.com/articles/how-a-machine-hears-a-number).

A sample of the order-discrimination experiment audio inputs as the experiment
runs heard them: 272 ms silent leader, first digit, 100 ms gap, second
digit, calibrated white noise at 0 dB (speech-equal RMS) over the
full padded window. Both orders of two pairs, same recordings
swapped — the classifier's only distinguishing information is
temporal order. One shared output scale across all four files;
regenerate with `uv run python scripts/make_order_audio.py` (seed 20260816).

"""Spoken digits: one AudioMNIST recording, processed the way every bank processes it.

`load_clip` resamples, normalizes and trims a recording; the study's bank and
its order-task sequences are built from it in harness.experiment.protocol. The
order task's silent leader is load-bearing: it absorbs the featurization
warm-up, so the read window stays order-symmetric.
"""

from __future__ import annotations

from pathlib import Path

import torch

from harness.utils.constants import WARMUP_FRAMES  # noqa: F401  (the leader bound)

PAIR_GAP_SAMPLES = 1600  # 100 ms silence between the two digits


PAIR_LEADER_SAMPLES = 4352  # 272 ms silent leader: > WARMUP_FRAMES of hop


PAIR_MAX_SAMPLES = PAIR_LEADER_SAMPLES + 2 * 16000 + PAIR_GAP_SAMPLES


WAV_CHANNELS, WAV_SAMPLE_BYTES = 1, 2
PCM16_SCALE = 32768.0
DIGIT_SR = 16000


DIGIT_MAX_SAMPLES = 16000          # 1 s cap after trim (right-zero-padded at load)


DIGIT_TRIM_FRAC = 0.01             # trim below 1% of clip peak |x|


def load_clip(path: Path) -> torch.Tensor:
    """One AudioMNIST recording, processed exactly as every bank processes it:
    48 kHz -> 16 kHz, peak-normalized to 0.5, energy-trimmed below 1% of the
    peak, capped at 1 s, stored as int16."""
    import torchaudio
    x, sr = read_wav(path)
    x = torchaudio.functional.resample(x[None], sr, DIGIT_SR)[0]
    peak = x.abs().max().clamp_min(1e-8)
    x = 0.5 * x / peak
    keep = (x.abs() > DIGIT_TRIM_FRAC * 0.5).nonzero()
    if len(keep):
        x = x[keep[0, 0]:keep[-1, 0] + 1]
    x = x[:DIGIT_MAX_SAMPLES]
    return torch.round(x * 32767).to(torch.int16)


def read_wav(path: Path) -> tuple[torch.Tensor, int]:
    """A 16-bit PCM WAV as float32 in [-1, 1), and its sample rate.

    The standard-library reader, because torchaudio's own loader now needs a
    separate decoding package. The scaling is torchaudio's: int16 over 32768,
    so the samples match torchaudio's.
    """
    import wave
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (WAV_CHANNELS, WAV_SAMPLE_BYTES, "NONE"):
            raise ValueError(f"{path}: expected mono 16-bit PCM, got {w.getnchannels()} channel(s), "
                             f"{8 * w.getsampwidth()}-bit, {w.getcomptype()}")
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
    x = torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(torch.float32) / PCM16_SCALE
    return x, rate


def clip_path(root: Path, speaker: int, digit: int, rep: int) -> Path:
    return root / f"{speaker:02d}" / f"{digit}_{speaker:02d}_{rep}.wav"

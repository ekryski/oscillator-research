"""One AudioMNIST recording, loaded exactly as paper 02's bank loaded it.

Paper 03 reuses paper 02's 50-repetition bank (`digits_v2.pt`); the bank is
built by `harness.experiment.protocol --build-bank`, which calls `load_clip` for
every recording. Nothing here draws a random number: same corpus in, same
bank out.
"""

from __future__ import annotations

from pathlib import Path

import torch

DIGIT_SR = 16000
DIGIT_MAX_SAMPLES = 16000          # 1 s cap after trim (right-zero-padded in the bank)
DIGIT_TRIM_FRAC = 0.01             # trim below 1% of clip peak |x|
#: the only format AudioMNIST ships: mono, 16-bit, uncompressed PCM
WAV_CHANNELS, WAV_SAMPLE_BYTES = 1, 2
#: int16 full scale, the divisor torchaudio's loader used to map PCM to [-1, 1)
PCM16_SCALE = 32768.0


def load_clip(path: Path) -> torch.Tensor:
    """One recording: 48 kHz -> 16 kHz, peak-normalized to 0.5, energy-trimmed
    below 1% of the peak, capped at 1 s, stored as int16."""
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
    """A 16-bit PCM WAV as float32 in [-1, 1), and its sample rate (torchaudio's scaling)."""
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

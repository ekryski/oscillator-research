"""Calibration audio examples for the noise grid.

One clip — same digit, same speaker, same rep — rendered clean and at every
noise level of the calibration grid {-10, -5, 0, +5, +10 dB}, so the added
noise is the ONLY variable across files. Noise construction mirrors
`harness.stimuli.digits.make_digit_clips` exactly (white noise scaled to the clip's
speech-region RMS: sigma = rms * 10^(dB/20), added over the full padded
window), with ONE seeded draw reused across levels so even the noise
waveform shape is identical — only its amplitude differs.

All files share a single common scale factor (chosen so the loudest file
peaks at 0.99) before 16-bit encoding: relative levels across the set are
exactly preserved and nothing clips.

    uv run python scripts/make_calibration_audio.py

Output: resources/audio/calibration/
"""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from harness.stimuli import load_digit_bank
from harness.utils.paths import AUDIO_DIR

OUT_DIR = AUDIO_DIR / "calibration"
DIGIT = 7          # "seven": two syllables + the /s/ fricative (>1.5 kHz content)
NOISE_SEED = 424242
LEVELS_DB = [-10.0, -5.0, 0.0, 5.0, 10.0]  # the calibration grid of record


def main() -> None:
    bank = load_digit_bank()
    sr, max_samples = int(bank["sr"]), int(bank["max_samples"])
    pool = bank["test"]
    speaker = int(pool["speakers"].min())  # first (lowest-numbered) test speaker
    idx = int(((pool["speakers"] == speaker) & (pool["labels"] == DIGIT))
              .nonzero().flatten()[0])     # first rep of that speaker x digit

    # identical clip prep to make_digit_clips: int16 -> float, right-zero-pad
    w = pool["waves"][idx].to(torch.float32) / 32767.0
    n_valid = len(w)
    wave = torch.zeros(max_samples)
    wave[:n_valid] = w
    rms = float((wave.pow(2).sum() / n_valid).sqrt().clamp_min(1e-8))

    # one seeded noise draw, rescaled per level (amplitude is the only variable)
    gen = torch.Generator().manual_seed(NOISE_SEED)
    unit_noise = torch.randn(max_samples, generator=gen)

    files = {"clean.wav": wave}
    for db in LEVELS_DB:
        sigma = rms * (10.0 ** (db / 20.0))
        name = f"noise_{'+' if db >= 0 else ''}{int(db)}db.wav"
        files[name] = wave + unit_noise * sigma

    # common scale factor across the whole set: preserves relative levels
    peak = max(float(x.abs().max()) for x in files.values())
    scale = 0.99 / peak
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, x in files.items():
        sf.write(OUT_DIR / name, (x * scale).numpy(), sr, subtype="PCM_16")

    (OUT_DIR / "README.md").write_text(
        "# Calibration audio examples (noise grid)\n\n"
        "Same clip in every file — AudioMNIST digit "
        f"{DIGIT} (\"seven\"), test-pool speaker {speaker}, first rep "
        f"(bank index {idx}), {n_valid} valid samples @ {sr} Hz — rendered "
        "clean and at each calibration noise level. Noise mirrors "
        "`harness.stimuli.digits.make_digit_clips` exactly (white noise, sigma = "
        "speech-region RMS x 10^(dB/20), full padded window); one seeded "
        f"draw (seed {NOISE_SEED}) rescaled per level, so noise AMPLITUDE "
        "is the only variable across files. All files share one common "
        f"scale factor ({scale:.4f}) so relative levels are exact and "
        "nothing clips at 16-bit.\n\n"
        "| file | added noise (rel. speech RMS) |\n|---|---|\n"
        "| clean.wav | none (the unmodified sample) |\n"
        + "".join(
            f"| noise_{'+' if db >= 0 else ''}{int(db)}db.wav | "
            f"{10 ** (db / 20):.2f}x ({int(db):+d} dB) |\n"
            for db in LEVELS_DB)
        + "\nRegenerate: `uv run python scripts/make_calibration_audio.py` "
        "(deterministic).\n"
        "The grid and the level-pick rule are stated in section 4.4 of the paper.\n")
    for name in files:
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()

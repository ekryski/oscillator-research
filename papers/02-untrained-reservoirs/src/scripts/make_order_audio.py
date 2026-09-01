"""Order-task audio examples (deterministic, regenerable).

Two pairs x both orders, at the EXPERIMENTAL condition (272 ms leader,
100 ms gap, 0 dB speech-equal-RMS noise) — the stimuli as the order runs
heard them. One shared output scale across all files preserves relative
levels; 16-bit PCM, 16 kHz.

    uv run python scripts/make_order_audio.py
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from harness.stimuli import (
    PAIR_GAP_SAMPLES,
    PAIR_LEADER_SAMPLES,
    PAIR_MAX_SAMPLES,
    load_digit_bank,
)
from harness.utils.paths import AUDIO_DIR

OUT = AUDIO_DIR / "order"
SEED = 20260816
NOISE_DB = 0.0
PAIRS = [(3, 7), (1, 8)]


def compose(first: torch.Tensor, second: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    w = torch.zeros(PAIR_MAX_SAMPLES)
    w[PAIR_LEADER_SAMPLES:PAIR_LEADER_SAMPLES + len(first)] = first
    s2 = PAIR_LEADER_SAMPLES + len(first) + PAIR_GAP_SAMPLES
    w[s2:s2 + len(second)] = second
    speech_len = s2 + len(second)
    rms = w.pow(2).sum().div(speech_len).sqrt().clamp_min(1e-8)
    noise = torch.randn(PAIR_MAX_SAMPLES, generator=gen) * rms * (10.0 ** (NOISE_DB / 20.0))
    return w + noise


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bank = load_digit_bank()
    pool = bank["test"]
    gen = torch.Generator().manual_seed(SEED)
    clips = {}
    for a, b in PAIRS:
        ia = (pool["labels"] == a).nonzero().flatten()[0].item()
        ib = (pool["labels"] == b).nonzero().flatten()[0].item()
        wa = pool["waves"][ia].to(torch.float32) / 32767.0
        wb = pool["waves"][ib].to(torch.float32) / 32767.0
        clips[f"pair{a}{b}_order-{a}-then-{b}"] = compose(wa, wb, gen)
        clips[f"pair{a}{b}_order-{b}-then-{a}"] = compose(wb, wa, gen)
    peak = max(w.abs().max().item() for w in clips.values())
    scale = 0.98 / peak
    for name, w in clips.items():
        pcm = (w * scale * 32767.0).clamp(-32768, 32767).to(torch.int16).numpy()
        with wave.open(str(OUT / f"{name}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(pcm.tobytes())
        print(f"wrote {name}.wav")
    (OUT / "README.md").write_text(
        "# Order-task audio examples\n\n"
        "The order-discrimination stimuli exactly as the experiment's runs\n"
        "heard them: 272 ms silent leader, first digit, 100 ms gap, second\n"
        "digit, calibrated white noise at 0 dB (speech-equal RMS) over the\n"
        "full padded window. Both orders of two pairs, same recordings\n"
        "swapped — the classifier's only distinguishing information is\n"
        "temporal order. One shared output scale across all four files;\n"
        f"regenerate with `uv run python scripts/make_order_audio.py` (seed {SEED}).\n")


if __name__ == "__main__":
    main()

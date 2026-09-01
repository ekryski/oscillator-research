"""Spoken digits: the AudioMNIST bank, and the clips drawn from it.

The bank build is deterministic — same corpus in, same bank out, no RNG
anywhere — because every accuracy in the paper is conditioned on it. The
speaker split is fixed and disjoint (1-48 train, 49-60 test), so a reported
test accuracy is genuinely speaker-held-out.

Two tasks are drawn from the bank. `make_digit_clips` is digit identity.
`make_digitpair_clips` is the order task: digit a then b, or b then a, from
INDEPENDENT recordings joined by a gap, so the two classes differ in order and
nothing else. The silent leader in front is load-bearing — it absorbs the
featurization warmup, so the readout window stays order-symmetric; without it
the warmup asymmetry leaks order into an otherwise order-free read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from harness.utils.constants import WARMUP_FRAMES  # noqa: F401  (the leader bound)
from harness.utils.paths import AUDIOMNIST_DIR, CACHE_DIR

PAIR_GAP_SAMPLES = 1600  # 100 ms silence between the two digits
PAIR_LEADER_SAMPLES = 4352  # 272 ms silent leader: > WARMUP_FRAMES of hop
# frames, so the featurization warmup cut consumes leader (order-symmetric)
# rather than the first digit's onset — an audit found the original
# leaderless design leaked order through the warmup window asymmetry
# (warmup-matched floors 0.77-0.92); with the leader the matched floor is
# chance again and the construction is blind as intended.
PAIR_MAX_SAMPLES = PAIR_LEADER_SAMPLES + 2 * 16000 + PAIR_GAP_SAMPLES


def make_digitpair_clips(n: int, bank: dict, pool: str, gen: torch.Generator,
                         pair: tuple[int, int], noise_db: float | None = None):
    """Order-discrimination clips: digit a then digit b (label 0) or
    b then a (label 1), independent recordings joined by a 100 ms gap.

    Full-span pooled statistics of [A;B] and [B;A] are near-identical by
    construction (same frame multiset; only the junction frame differs), so
    an order-free floor sits at chance and any order signal read through
    full-span features must come from state memory. Calibrated noise is
    drawn over the whole padded window, as in make_digit_clips."""
    a, b = pair
    p = bank[pool]
    labels_all = p["labels"]
    idx_a = (labels_all == a).nonzero().flatten()
    idx_b = (labels_all == b).nonzero().flatten()
    waves = torch.zeros(n, PAIR_MAX_SAMPLES)
    lens = torch.empty(n, dtype=torch.long)
    ylab = torch.empty(n, dtype=torch.long)
    for i in range(n):
        ia = idx_a[torch.randint(len(idx_a), (1,), generator=gen)].item()
        ib = idx_b[torch.randint(len(idx_b), (1,), generator=gen)].item()
        wa = p["waves"][ia].to(torch.float32) / 32767.0
        wb = p["waves"][ib].to(torch.float32) / 32767.0
        flip = int(torch.randint(2, (1,), generator=gen).item())
        first, second = (wa, wb) if flip == 0 else (wb, wa)
        waves[i, PAIR_LEADER_SAMPLES:PAIR_LEADER_SAMPLES + len(first)] = first
        start2 = PAIR_LEADER_SAMPLES + len(first) + PAIR_GAP_SAMPLES
        waves[i, start2:start2 + len(second)] = second
        lens[i] = start2 + len(second)
        ylab[i] = flip
    if noise_db is not None:
        rms = (waves.pow(2).sum(1) / lens).sqrt().clamp_min(1e-8)
        scale = rms * (10.0 ** (noise_db / 20.0))
        waves = waves + torch.randn(n, PAIR_MAX_SAMPLES, generator=gen) * scale[:, None]
    return waves, lens, ylab


DIGIT_BANK_PATH = CACHE_DIR / "digits_v1.pt"
DIGIT_SR = 16000
DIGIT_MAX_SAMPLES = 16000          # 1 s cap after trim (right-zero-padded at load)
DIGIT_TRAIN_SPEAKERS = tuple(range(1, 49))   # speaker-disjoint split, fixed
DIGIT_TEST_SPEAKERS = tuple(range(49, 61))
DIGIT_REPS = 20                    # reps 0..19 per speaker per digit (deterministic)
DIGIT_TRIM_FRAC = 0.01             # trim below 1% of clip peak |x|


def build_digit_bank(root: Path = AUDIOMNIST_DIR,
                     out_path: Path = DIGIT_BANK_PATH) -> dict:
    """AudioMNIST -> bank: 48 kHz -> 16 kHz, peak-normalized to 0.5,
    energy-trimmed, TRUE LENGTHS stored (length-masked features depend on
    them), int16 storage. Deterministic file selection (reps 0..19)."""
    import torchaudio
    entries = {"train": [], "test": []}
    for spk in sorted(DIGIT_TRAIN_SPEAKERS + DIGIT_TEST_SPEAKERS):
        pool = "train" if spk in DIGIT_TRAIN_SPEAKERS else "test"
        d = root / f"{spk:02d}"
        for digit in range(10):
            for rep in range(DIGIT_REPS):
                f = d / f"{digit}_{spk:02d}_{rep}.wav"
                x, sr = torchaudio.load(str(f))
                x = torchaudio.functional.resample(x, sr, DIGIT_SR)[0]
                peak = x.abs().max().clamp_min(1e-8)
                x = 0.5 * x / peak
                keep = (x.abs() > DIGIT_TRIM_FRAC * 0.5).nonzero()
                if len(keep):
                    x = x[keep[0, 0]:keep[-1, 0] + 1]
                x = x[:DIGIT_MAX_SAMPLES]
                entries[pool].append((torch.round(x * 32767).to(torch.int16),
                                      digit, spk))
    bank = {"sr": DIGIT_SR, "max_samples": DIGIT_MAX_SAMPLES,
            "trim_frac": DIGIT_TRIM_FRAC, "reps": DIGIT_REPS,
            "train_speakers": DIGIT_TRAIN_SPEAKERS,
            "test_speakers": DIGIT_TEST_SPEAKERS}
    for pool in ("train", "test"):
        bank[pool] = {
            "waves": [w for w, _, _ in entries[pool]],
            "labels": torch.tensor([d for _, d, _ in entries[pool]]),
            "speakers": torch.tensor([s for _, _, s in entries[pool]]),
            "lens": torch.tensor([len(w) for w, _, _ in entries[pool]]),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out_path)
    return bank


def load_digit_bank(path: Path = DIGIT_BANK_PATH) -> dict:
    return torch.load(path, weights_only=True)


def make_digit_clips(n: int, bank: dict, pool: str, gen: torch.Generator,
                     noise_db: float | None = None):
    """-> (waves [n, MAX], lens [n], labels [n]); right-zero-padded floats;
    calibrated white noise (matched to the clip's speech-region RMS) added
    over the FULL padded window when noise_db is set."""
    p = bank[pool]
    idx = torch.randint(len(p["waves"]), (n,), generator=gen)
    L = bank["max_samples"]
    waves = torch.zeros(n, L)
    lens = torch.empty(n, dtype=torch.long)
    for i, j in enumerate(idx.tolist()):
        w = p["waves"][j].to(torch.float32) / 32767.0
        waves[i, :len(w)] = w
        lens[i] = len(w)
    if noise_db is not None:
        rms = (waves.pow(2).sum(1) / lens).sqrt().clamp_min(1e-8)
        scale = rms * (10.0 ** (noise_db / 20.0))
        waves = waves + torch.randn(n, L, generator=gen) * scale[:, None]
    return waves, lens, bank[pool]["labels"][idx]



def _main(argv: list[str] | None = None) -> None:
    """`python -m harness.stimuli.digits --build-bank` — the one-off bank build."""

    ap = argparse.ArgumentParser(description="build the derived stimulus bank")
    ap.add_argument("--build-bank", action="store_true",
                    help="build the spoken-digit bank from the AudioMNIST corpus")
    ap.add_argument("--corpus", type=Path, default=AUDIOMNIST_DIR,
                    help=f"AudioMNIST data root (default: {AUDIOMNIST_DIR})")
    a = ap.parse_args(argv)
    if not a.build_bank:
        ap.error("nothing to do — pass --build-bank")
    if not a.corpus.is_dir():
        ap.error(f"corpus not found at {a.corpus} — see src/data/README.md")
    bank = build_digit_bank(root=a.corpus)
    n_tr, n_te = len(bank["train"]["waves"]), len(bank["test"]["waves"])
    print(f"wrote {DIGIT_BANK_PATH}: {n_tr} train / {n_te} test clips "
          f"({len(bank['train_speakers'])} + {len(bank['test_speakers'])} speakers, "
          f"speaker-disjoint)")


if __name__ == "__main__":
    _main()

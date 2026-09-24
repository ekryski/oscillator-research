"""The data protocol, paper 02's Protocol A unchanged: which recordings, which noise, which splits.

Every clip is a fixed object. Its recording comes from the 50-repetition bank,
and its noise from a generator seeded by the clip's own identity and the noise
level, so a clip is the same clip in every arm, seed, tier and process. A seed
chooses which training clips a readout is fitted on and an arm's own random
draws; it never changes what a clip sounds like.

Protocol A: speakers 1-48 train (24,000 clips), speakers 49-60 test (6,000).
The test set is the whole test pool, in one fixed order. A seed permutes the
training pool, and a training set is a prefix of that permutation; paper 03
fits every readout on the first 2,048 clips, paper 02's primary size.

Paper 02's Protocol B (Becker et al.'s folds) and its order task are not
carried over: paper 03 reads recognition under Protocol A only.
"""

from __future__ import annotations

from pathlib import Path

import torch

from harness.stimuli.digits import DIGIT_MAX_SAMPLES, DIGIT_SR, DIGIT_TRIM_FRAC, clip_path, load_clip
from harness.stimuli.filterbank import bandpass_rows
from harness.stimuli.frontend import HOP_N_FFT, hop_num_frames, hop_rows, hop_rows_quad
from harness.utils.paths import AUDIOMNIST_DIR, CACHE_DIR

BANK_PATH = CACHE_DIR / "digits_v2.pt"
#: every repetition AudioMNIST has; the exploratory bank kept 20
REPS = 50
SPEAKERS = tuple(range(1, 61))
DIGITS = tuple(range(10))
#: int16 back to float, the scale the exploratory samplers used
INT16_SCALE = 32767.0

TRAIN_SPEAKERS = tuple(range(1, 49))
TEST_SPEAKERS = tuple(range(49, 61))
#: nested training-set sizes; the last is the whole training pool
SIZES = (2048, 8192, 24000)

#: seed families, kept apart so no two uses of a generator can collide
TRAIN_ORDER_SEED = 1000
NOISE_SEED = 20_260_923
#: clip identities: a bank clip is speaker*10,000 + digit*100 + rep
#: noise levels map to non-negative seed offsets
NOISE_LEVEL_OFFSET, NOISE_LEVEL_SCALE, NOISE_UID_STRIDE = 1000, 10, 10_007


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------

def build_bank(root: Path = AUDIOMNIST_DIR, out: Path = BANK_PATH, reps: int = REPS) -> dict:
    """AudioMNIST -> one tensor of clips, in canonical speaker, digit, rep order.

    Each recording is processed by `load_clip`, the exploratory bank's own
    function, so a recording in both banks is bit-identical in both.
    """
    n = len(SPEAKERS) * len(DIGITS) * reps
    waves = torch.zeros(n, DIGIT_MAX_SAMPLES, dtype=torch.int16)
    lens = torch.empty(n, dtype=torch.long)
    labels, speakers, repeats = (torch.empty(n, dtype=torch.long) for _ in range(3))
    i = 0
    for spk in SPEAKERS:
        for digit in DIGITS:
            for rep in range(reps):
                clip = load_clip(clip_path(root, spk, digit, rep))
                waves[i, :len(clip)] = clip
                lens[i], labels[i], speakers[i], repeats[i] = len(clip), digit, spk, rep
                i += 1
    bank = {"waves": waves, "lens": lens, "labels": labels, "speakers": speakers,
            "reps": repeats, "sr": DIGIT_SR, "trim_frac": DIGIT_TRIM_FRAC, "version": 2}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out)
    return bank


def load_bank(path: Path = BANK_PATH) -> dict:
    """Memory-mapped, so parallel workers share one copy of the clips."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: build it with "
                                "`uv run python -m harness.confirm.protocol --build-bank`")
    return torch.load(path, mmap=True, weights_only=True)


def uids(bank: dict, idx: torch.Tensor) -> torch.Tensor:
    """A bank clip's identity: speaker, digit and repetition, as one number."""
    return bank["speakers"][idx] * 10_000 + bank["labels"][idx] * 100 + bank["reps"][idx]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _of_speakers(bank: dict, speakers: tuple[int, ...]) -> torch.Tensor:
    return torch.isin(bank["speakers"], torch.tensor(speakers)).nonzero().flatten()


def protocol_a(bank: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(training pool, test pool) as bank indices, each in canonical order."""
    return _of_speakers(bank, TRAIN_SPEAKERS), _of_speakers(bank, TEST_SPEAKERS)


def training_order(pool: torch.Tensor, seed: int) -> torch.Tensor:
    """The seed's permutation of the training pool. Its prefixes are the nested sets."""
    gen = torch.Generator().manual_seed(TRAIN_ORDER_SEED + seed)
    return pool[torch.randperm(len(pool), generator=gen)]


# ---------------------------------------------------------------------------
# Noise and front ends
# ---------------------------------------------------------------------------

def add_noise(waves: torch.Tensor, lens: torch.Tensor, clip_ids: torch.Tensor,
              noise_db: float | None) -> torch.Tensor:
    """White noise at `noise_db` relative to each clip's speech RMS, over the padded window.

    The harness's convention: amplitude = speech RMS x 10^(dB/20), so 0 dB is
    noise at speech-equal power and +5 dB is louder noise. Each clip's noise
    comes from its own generator, seeded by its identity and the level, so it
    is the same noise whichever set, batch or process the clip turns up in.
    """
    if noise_db is None:
        return waves
    rms = (waves.pow(2).sum(1) / lens).sqrt().clamp_min(1e-8)
    level = int(round(noise_db * NOISE_LEVEL_SCALE)) + NOISE_LEVEL_OFFSET
    noise = torch.empty_like(waves)
    for i, uid in enumerate(clip_ids.tolist()):
        gen = torch.Generator().manual_seed(NOISE_SEED + uid * NOISE_UID_STRIDE + level)
        noise[i] = torch.randn(waves.shape[1], generator=gen)
    return waves + noise * (rms * 10.0 ** (noise_db / 20.0))[:, None]


def front_end(waves: torch.Tensor, drive: str, grid: int = 16, window: int = HOP_N_FFT) -> torch.Tensor:
    """The fixed, parameter-free front end of each input pathway, at `grid` bands and an analysis
    window of `window` samples (the band-energy and quadrature pathways; the carrier has no window)."""
    if drive == "envelope":
        return hop_rows(waves, grid, window=window)
    if drive == "quadrature":
        return hop_rows_quad(waves, grid, window=window)
    if drive == "carrier":
        if window != HOP_N_FFT:
            raise ValueError("the carrier pathway has no analysis window")
        return bandpass_rows(waves, grid)
    raise ValueError(f"unknown drive '{drive}'")


def to_rows(rows: torch.Tensor, grid: int) -> torch.Tensor:
    """[B, T, bands, ...] -> [B, T, grid, ...]: the fixed map from a front end's bands onto a lattice's rows.

    One band per row needs nothing. More rows than bands give each band
    grid // bands adjacent rows; fewer rows give each row the mean of
    bands // grid adjacent bands (on the quadrature pathway, the mean of their
    (A cos phi, A sin phi) pairs). Either way nothing is fitted, and the rows
    stay in frequency order.
    """
    bands = rows.shape[2]
    if bands == grid:
        return rows
    if grid % bands == 0:
        return rows.repeat_interleave(grid // bands, dim=2)
    if bands % grid == 0:
        return rows.unflatten(2, (grid, bands // grid)).mean(3)
    raise ValueError(f"cannot map {bands} bands onto {grid} rows")


def valid_frames(lens: torch.Tensor, drive: str) -> torch.Tensor:
    """How many of each clip's rows hold speech: hop frames, or samples for the carrier."""
    if drive == "carrier":
        return lens.clone()
    return torch.tensor([hop_num_frames(int(n)) for n in lens])


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def recognition_clips(bank: dict, idx: torch.Tensor, noise_db: float | None):
    """(waves [n, 16000] float, lens [n], labels [n]) for bank clips `idx`."""
    waves = bank["waves"][idx].to(torch.float32) / INT16_SCALE
    lens = bank["lens"][idx]
    return add_noise(waves, lens, uids(bank, idx), noise_db), lens, bank["labels"][idx]


# ---------------------------------------------------------------------------
# Row caches
# ---------------------------------------------------------------------------
#
# A clip's front-end rows depend only on the clip and the noise level, so they
# are computed once per (drive, level), saved, and memory-mapped by every run.
# That saves each run its noise draws and front end, and it means every run
# reads bit-identical rows: nothing depends on how a run happened to batch.
# The carrier's rows are 16,000 frames a clip and are always computed on the fly.

ROWS_DIR = CACHE_DIR / "rows"
#: clips per batch while building a cache
CACHE_BATCH = 1024
CACHED_DRIVES = ("envelope", "quadrature")


def level_name(noise_db: float | None) -> str:
    return "clean" if noise_db is None else f"{noise_db:g}db"


def rows_path(drive: str, noise_db: float | None, bands: int = 16, window: int = HOP_N_FFT) -> Path:
    """The cache for a drive, level, band count and window; 16 bands keep paper 02's file name."""
    size = ("" if bands == 16 else f"-{bands}bands") + ("" if window == HOP_N_FFT else f"-w{window}")
    return ROWS_DIR / f"{drive}{size}-{level_name(noise_db)}.pt"


def _save(out: Path, rows: torch.Tensor, tvalid: torch.Tensor, **meta) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"rows": rows, "tvalid": tvalid, **meta}, tmp)
    tmp.replace(out)                    # atomic: a reader never sees half a cache


def build_rows(bank: dict, drive: str, noise_db: float | None, bands: int = 16, window: int = HOP_N_FFT) -> Path:
    """Front-end rows for every clip in the bank, in canonical order."""
    n = len(bank["labels"])
    rows = tvalid = None
    for a in range(0, n, CACHE_BATCH):
        idx = torch.arange(a, min(a + CACHE_BATCH, n))
        waves, lens, _ = recognition_clips(bank, idx, noise_db)
        r = front_end(waves, drive, bands, window)
        if rows is None:
            rows = torch.empty((n, *r.shape[1:]))
            tvalid = torch.empty(n, dtype=torch.long)
        rows[idx], tvalid[idx] = r, valid_frames(lens, drive)
    out = rows_path(drive, noise_db, bands, window)
    _save(out, rows, tvalid, n_clips=n, drive=drive, noise_db=noise_db, bands=bands, window=window)
    return out


def load_rows(path: Path, n_clips: int) -> dict | None:
    """A cache if it exists and was built for this many clips, else None."""
    if not path.exists():
        return None
    cache = torch.load(path, mmap=True, weights_only=True)
    return cache if cache.get("n_clips") == n_clips else None


def _main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="build the 50-repetition bank (paper 02's, reused)")
    ap.add_argument("--build-bank", action="store_true")
    ap.add_argument("--corpus", type=Path, default=AUDIOMNIST_DIR)
    a = ap.parse_args(argv)
    if not a.build_bank:
        ap.error("nothing to do: pass --build-bank")
    bank = build_bank(a.corpus)
    print(f"wrote {BANK_PATH}: {len(bank['labels'])} clips from {len(SPEAKERS)} speakers")


if __name__ == "__main__":
    _main()

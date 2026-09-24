"""The confirmatory data protocol: which recordings, which noise, which splits.

Every clip is a fixed object. Its recording comes from the 50-repetition bank,
and its noise from a generator seeded by the clip's own identity and the noise
level, so a clip is the same clip in every arm, seed, tier and process. A seed
chooses which training clips a readout is fitted on and an arm's own random
draws; it never changes what a clip sounds like.

Protocol A, for every tier but B: speakers 1-48 train (24,000 clips), speakers
49-60 test (6,000). The test set is the whole test pool, in one fixed order. A
seed permutes the training pool, and training sets of 2,048, 8,192 and 24,000
clips are prefixes of that permutation, so each contains the one before it.

Protocol B, for comparison with published results only: the five speaker folds
of Becker et al., copied verbatim from the corpus's own preprocess_data.py.

The order task joins two recordings, digit a then digit b or b then a, behind a
silent leader and across a 100 ms gap, exactly as the exploratory task did. Its
labels alternate, so both classes are equally common in every set.
"""

from __future__ import annotations

from pathlib import Path

import torch

from harness.stimuli import bandpass_rows, hop_num_frames, hop_rows, hop_rows_quad
from harness.stimuli.digits import (
    DIGIT_MAX_SAMPLES,
    DIGIT_SR,
    DIGIT_TRIM_FRAC,
    PAIR_GAP_SAMPLES,
    PAIR_LEADER_SAMPLES,
    PAIR_MAX_SAMPLES,
    clip_path,
    load_clip,
)
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

#: Becker et al.'s digit folds, verbatim from preprocess_data.py in the corpus
BECKER_FOLDS = (
    {"train": (28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53, 36, 57, 9, 24, 37, 2,
               8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55),
     "validate": (12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50),
     "test": (26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51)},
    {"train": (36, 57, 9, 24, 37, 2, 8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38, 3,
               10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50),
     "validate": (26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51),
     "test": (28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53)},
    {"train": (43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41,
               4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51),
     "validate": (28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53),
     "test": (36, 57, 9, 24, 37, 2, 8, 17, 29, 39, 48, 54)},
    {"train": (12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,
               5, 13, 22, 33, 45, 51, 28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53),
     "validate": (36, 57, 9, 24, 37, 2, 8, 17, 29, 39, 48, 54),
     "test": (43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55)},
    {"train": (26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51, 28, 56, 7, 19, 35, 1,
               6, 16, 23, 34, 46, 53, 36, 57, 9, 24, 37, 2, 8, 17, 29, 39, 48, 54),
     "validate": (43, 58, 14, 25, 38, 3, 10, 20, 30, 40, 49, 55),
     "test": (12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50)},
)

#: the five order-task pairs, and the size of each set
PAIRS = ((3, 7), (1, 8), (2, 5), (4, 9), (0, 6))
ORDER_TRAIN, ORDER_TEST = 2048, 2048

#: seed families, kept apart so no two uses of a generator can collide
TRAIN_ORDER_SEED = 1000
ORDER_SET_SEED = 3000
ORDER_TEST_SET = 9999
NOISE_SEED = 20_260_923
#: clip identities: a bank clip is speaker*10,000 + digit*100 + rep; an order
#: clip lives in its own range above every bank identity
ORDER_UID_BASE = 10**8
ORDER_UID_PAIR, ORDER_UID_SET = 10**7, 10**5
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


def protocol_b(bank: dict, fold: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(train, validate, test) bank indices for one of Becker et al.'s folds."""
    f = BECKER_FOLDS[fold]
    return tuple(_of_speakers(bank, tuple(sorted(f[k]))) for k in ("train", "validate", "test"))


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


def front_end(waves: torch.Tensor, drive: str, grid: int = 16) -> torch.Tensor:
    """The fixed, parameter-free front end of each drive pathway."""
    if drive == "envelope":
        return hop_rows(waves, grid)
    if drive == "quadrature":
        return hop_rows_quad(waves, grid)
    if drive == "carrier":
        return bandpass_rows(waves, grid)
    raise ValueError(f"unknown drive '{drive}'")


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


def order_set(bank: dict, pool: torch.Tensor, pair: tuple[int, int], n: int,
              set_code: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Which recordings make up an order-task set: (first idx, second idx, labels).

    Label 0 is a then b, label 1 is b then a, alternating so the classes are
    exactly balanced. `set_code` 0 is the fixed test set; seed s draws training
    set s + 1. Recordings are independent draws from the pool.
    """
    a, b = pair
    pair_idx = PAIRS.index(pair)
    seed = ORDER_TEST_SET + pair_idx if set_code == 0 else ORDER_SET_SEED + 100 * pair_idx + set_code
    gen = torch.Generator().manual_seed(seed)
    of_a = pool[bank["labels"][pool] == a]
    of_b = pool[bank["labels"][pool] == b]
    pick_a = of_a[torch.randint(len(of_a), (n,), generator=gen)]
    pick_b = of_b[torch.randint(len(of_b), (n,), generator=gen)]
    labels = torch.arange(n) % 2
    first = torch.where(labels == 0, pick_a, pick_b)
    second = torch.where(labels == 0, pick_b, pick_a)
    return first, second, labels


def order_clips(bank: dict, first: torch.Tensor, second: torch.Tensor, pair: tuple[int, int],
                set_code: int, start: int, noise_db: float | None):
    """(waves [n, PAIR_MAX_SAMPLES], lens, None) for rows `start`.. of an order set."""
    n = len(first)
    waves = torch.zeros(n, PAIR_MAX_SAMPLES)
    lens = torch.empty(n, dtype=torch.long)
    for i in range(n):
        wa = bank["waves"][first[i], :bank["lens"][first[i]]].to(torch.float32) / INT16_SCALE
        wb = bank["waves"][second[i], :bank["lens"][second[i]]].to(torch.float32) / INT16_SCALE
        waves[i, PAIR_LEADER_SAMPLES:PAIR_LEADER_SAMPLES + len(wa)] = wa
        start2 = PAIR_LEADER_SAMPLES + len(wa) + PAIR_GAP_SAMPLES
        waves[i, start2:start2 + len(wb)] = wb
        lens[i] = start2 + len(wb)
    ids = (ORDER_UID_BASE + PAIRS.index(pair) * ORDER_UID_PAIR + set_code * ORDER_UID_SET
           + start + torch.arange(n))
    return add_noise(waves, lens, ids, noise_db), lens


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


def rows_path(drive: str, noise_db: float | None) -> Path:
    return ROWS_DIR / f"{drive}-{level_name(noise_db)}.pt"


def order_rows_path(pair: tuple[int, int], set_code: int, noise_db: float | None) -> Path:
    return ROWS_DIR / f"order-pair{pair[0]}{pair[1]}-set{set_code}-{level_name(noise_db)}.pt"


def _save(out: Path, rows: torch.Tensor, tvalid: torch.Tensor, **meta) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"rows": rows, "tvalid": tvalid, **meta}, tmp)
    tmp.replace(out)                    # atomic: a reader never sees half a cache


def build_rows(bank: dict, drive: str, noise_db: float | None) -> Path:
    """Front-end rows for every clip in the bank, in canonical order."""
    n = len(bank["labels"])
    rows = tvalid = None
    for a in range(0, n, CACHE_BATCH):
        idx = torch.arange(a, min(a + CACHE_BATCH, n))
        waves, lens, _ = recognition_clips(bank, idx, noise_db)
        r = front_end(waves, drive)
        if rows is None:
            rows = torch.empty((n, *r.shape[1:]))
            tvalid = torch.empty(n, dtype=torch.long)
        rows[idx], tvalid[idx] = r, valid_frames(lens, drive)
    out = rows_path(drive, noise_db)
    _save(out, rows, tvalid, n_clips=n, drive=drive, noise_db=noise_db)
    return out


def build_order_rows(bank: dict, pair: tuple[int, int], set_code: int, noise_db: float | None) -> Path:
    """Front-end rows for one order-task set: the test set (code 0) or seed s's training set (s + 1)."""
    train_pool, test_pool = protocol_a(bank)
    pool, n = (test_pool, ORDER_TEST) if set_code == 0 else (train_pool, ORDER_TRAIN)
    first, second, labels = order_set(bank, pool, pair, n, set_code)
    rows, tvalid = [], []
    for a in range(0, n, CACHE_BATCH):
        waves, lens = order_clips(bank, first[a:a + CACHE_BATCH], second[a:a + CACHE_BATCH],
                                  pair, set_code, a, noise_db)
        rows.append(front_end(waves, "envelope"))
        tvalid.append(valid_frames(lens, "envelope"))
    out = order_rows_path(pair, set_code, noise_db)
    _save(out, torch.cat(rows), torch.cat(tvalid), labels=labels, n_clips=n)
    return out


def load_rows(path: Path, n_clips: int) -> dict | None:
    """A cache if it exists and was built for this many clips, else None."""
    if not path.exists():
        return None
    cache = torch.load(path, mmap=True, weights_only=True)
    return cache if cache.get("n_clips") == n_clips else None


def _main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="build the confirmatory 50-repetition bank")
    ap.add_argument("--build-bank", action="store_true")
    ap.add_argument("--corpus", type=Path, default=AUDIOMNIST_DIR)
    a = ap.parse_args(argv)
    if not a.build_bank:
        ap.error("nothing to do: pass --build-bank")
    bank = build_bank(a.corpus)
    print(f"wrote {BANK_PATH}: {len(bank['labels'])} clips from {len(SPEAKERS)} speakers")


if __name__ == "__main__":
    _main()

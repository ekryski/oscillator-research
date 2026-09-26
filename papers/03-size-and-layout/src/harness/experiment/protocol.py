"""The data protocol, paper 02's Protocol A unchanged: which recordings, which noise, which splits.

Every clip is a fixed object. Its recording comes from the 50-repetition bank,
and its noise from a generator seeded by the clip's own identity and the noise
level, so a clip is the same clip in every arm, seed, experiment and process. A seed
chooses which training clips a readout is fitted on and an arm's own random
draws; it never changes what a clip sounds like.

Protocol A: speakers 1-48 train (24,000 clips), speakers 49-60 test (6,000).
The test set is the whole test pool, in one fixed order. A seed permutes the
training pool, and a training set is a prefix of that permutation; paper 03
fits every readout on the first 2,048 clips, paper 02's primary size.

Two memory tasks join recordings into one clip, behind a silent leader and
across 100 ms gaps, from the training speakers for training and the test
speakers for testing. The order task is paper 02's: digit a then digit b, or
b then a, on five digit pairs, its labels alternating so the classes are
exactly balanced. The digit-sequence task joins 2, 3 or 4 recordings of
different digits and asks for the digit at each position. Both are read over
the whole span as one window, a read that does not depend on the order of the
frames, so the spectrogram-only baseline can tell which digits are present but
not in what order, and whatever an arm adds must come from its own memory.

Paper 02's Protocol B (Becker et al.'s folds) is not carried over.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from harness.stimuli.digits import DIGIT_MAX_SAMPLES, DIGIT_SR, DIGIT_TRIM_FRAC, clip_path, load_clip
from harness.stimuli.frontend import HOP_N_FFT, hop_num_frames, hop_rows, hop_rows_quad
from harness.utils.paths import AUDIOMNIST_DIR, CACHE_DIR

BANK_PATH = CACHE_DIR / "digits_v2.pt"
#: every repetition AudioMNIST has
REPS = 50
SPEAKERS = tuple(range(1, 61))
DIGITS = tuple(range(10))
#: int16 back to float
INT16_SCALE = 32767.0

TRAIN_SPEAKERS = tuple(range(1, 49))
TEST_SPEAKERS = tuple(range(49, 61))

#: seed families, kept apart so no two uses of a generator can collide
TRAIN_ORDER_SEED = 1000
ORDER_SET_SEED = 3000
ORDER_TEST_SET = 9999
SEQUENCE_SET_SEED = 5000
SEQUENCE_TEST_SET = 9000
NOISE_SEED = 20_260_923
#: clip identities: a bank clip is speaker*10,000 + digit*100 + rep; an order
#: or a sequence clip lives in its own range above every bank identity
ORDER_UID_BASE = 10**8
ORDER_UID_PAIR, ORDER_UID_SET = 10**7, 10**5
SEQUENCE_UID_BASE = 2 * 10**8

#: the joined clips: a 272 ms silent leader (longer than the 16-frame warm-up, so the read starts
#: in silence whatever comes first), then the recordings, each padded to 1 s, 100 ms apart
LEADER_SAMPLES = 4352
GAP_SAMPLES = 1600
#: the order task's five pairs, and the size of each set (paper 02's)
PAIRS = ((3, 7), (1, 8), (2, 5), (4, 9), (0, 6))
ORDER_TRAIN, ORDER_TEST = 2048, 2048
#: the digit-sequence task's lengths, and the size of each set
SEQUENCE_LENGTHS = (2, 3, 4)
SEQUENCE_TRAIN, SEQUENCE_TEST = 2048, 2048
DIGIT_CHOICES = 10
#: noise levels map to non-negative seed offsets
NOISE_LEVEL_OFFSET, NOISE_LEVEL_SCALE, NOISE_UID_STRIDE = 1000, 10, 10_007


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------

def build_bank(root: Path = AUDIOMNIST_DIR, out: Path = BANK_PATH, reps: int = REPS) -> dict:
    """AudioMNIST -> one tensor of clips, in canonical speaker, digit, rep order.

    Each recording is trimmed, normalized and padded by `load_clip`, as paper 02 builds its bank, so the
    two banks are the same file.
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
                                "`uv run python -m harness.experiment.protocol --build-bank`")
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
    noise at speech-equal power and +5 dB is noise 5 dB louder than the speech,
    a signal-to-noise ratio of -5 dB, which is how the papers report it
    (summary.snr). The record keeps this level, since it seeds the noise. Paper
    03 runs 0 dB, where the two conventions agree. Each clip's noise
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


def front_end(waves: torch.Tensor, pathway: str, grid: int = 16, window: int = HOP_N_FFT) -> torch.Tensor:
    """The fixed, parameter-free front end of each input pathway, at `grid` bands and an analysis
    window of `window` samples."""
    if pathway == "spectrogram":
        return hop_rows(waves, grid, window=window)
    if pathway == "quadrature":
        return hop_rows_quad(waves, grid, window=window)
    raise ValueError(f"unknown pathway '{pathway}'")


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


def valid_frames(lens: torch.Tensor, pathway: str) -> torch.Tensor:
    """How many of each clip's rows hold speech, in hop frames."""
    return torch.tensor([hop_num_frames(int(n)) for n in lens])


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------

def joined_samples(length: int) -> int:
    """Samples in a clip of `length` joined recordings: the leader, the recordings and the gaps."""
    return LEADER_SAMPLES + length * DIGIT_MAX_SAMPLES + (length - 1) * GAP_SAMPLES


def joined_frames(length: int) -> int:
    """Hop frames in such a clip: 147, 216 and 284 for 2, 3 and 4 recordings."""
    return hop_num_frames(joined_samples(length))


def _join(bank: dict, parts: list[torch.Tensor], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(waves [n, joined_samples(length)], lens [n]): each row's recordings behind the leader, gap apart."""
    n = len(parts[0])
    waves = torch.zeros(n, joined_samples(length))
    lens = torch.empty(n, dtype=torch.long)
    for i in range(n):
        at = LEADER_SAMPLES
        for k, part in enumerate(parts):
            w = bank["waves"][part[i], :bank["lens"][part[i]]].to(torch.float32) / INT16_SCALE
            if k:
                at += GAP_SAMPLES
            waves[i, at:at + len(w)] = w
            at += len(w)
        lens[i] = at
    return waves, lens


def order_set(bank: dict, pool: torch.Tensor, pair: tuple[int, int], n: int,
              set_code: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Which recordings make up an order-task set: (first idx, second idx, labels). Paper 02's.

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
    """(waves [n, joined_samples(2)], lens) for rows `start`.. of an order set, as paper 02 built them."""
    waves, lens = _join(bank, [first, second], 2)
    ids = (ORDER_UID_BASE + PAIRS.index(pair) * ORDER_UID_PAIR + set_code * ORDER_UID_SET
           + start + torch.arange(len(first)))
    return add_noise(waves, lens, ids, noise_db), lens


def sequence_set(bank: dict, pool: torch.Tensor, length: int, n: int,
                 set_code: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Which recordings make up a digit-sequence set: (bank indices [n, length], digits [n, length]).

    Each sequence's digits are drawn uniformly, without repeats (so a
    sequence holds `length` different digits), and each position's recording
    is an independent draw from the pool's recordings of that digit. The digit
    at every position is therefore uniform over the ten, so chance at every
    position is 1/10. `set_code` 0 is the fixed test set; seed s draws training
    set s + 1.
    """
    seed = (SEQUENCE_TEST_SET + length if set_code == 0
            else SEQUENCE_SET_SEED + 100 * length + set_code)
    gen = torch.Generator().manual_seed(seed)
    digits = torch.stack([torch.randperm(DIGIT_CHOICES, generator=gen)[:length] for _ in range(n)])
    by_digit = [pool[bank["labels"][pool] == d] for d in range(DIGIT_CHOICES)]
    idx = torch.empty(n, length, dtype=torch.long)
    for p in range(length):
        for d in range(DIGIT_CHOICES):
            rows = (digits[:, p] == d).nonzero().flatten()
            idx[rows, p] = by_digit[d][torch.randint(len(by_digit[d]), (len(rows),), generator=gen)]
    return idx, digits


def sequence_clips(bank: dict, idx: torch.Tensor, length: int, set_code: int, start: int,
                   noise_db: float | None):
    """(waves [n, joined_samples(length)], lens) for rows `start`.. of a digit-sequence set."""
    waves, lens = _join(bank, [idx[:, p] for p in range(length)], length)
    ids = (SEQUENCE_UID_BASE + length * ORDER_UID_PAIR + set_code * ORDER_UID_SET
           + start + torch.arange(len(idx)))
    return add_noise(waves, lens, ids, noise_db), lens


def sequence_chance(length: int) -> dict:
    """Exact chance levels for a sequence of `length` different digits.

    per_position: an uninformed reader, 1/10. order_free: a reader that knows
    the digits present but not their order, 1/length. whole: every position
    right by chance, one sequence among 10 * 9 * ... (length factors).
    whole_order_free: every position right knowing the digits present, one
    ordering among length!.
    """
    return {"per_position": 1.0 / DIGIT_CHOICES, "order_free": 1.0 / length,
            "whole": 1.0 / math.perm(DIGIT_CHOICES, length), "whole_order_free": 1.0 / math.factorial(length)}


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
# are computed once per (pathway, level), saved, and memory-mapped by every run.
# That saves each run its noise draws and front end, and it means every run
# reads bit-identical rows: nothing depends on how a run happened to batch.

ROWS_DIR = CACHE_DIR / "rows"
#: clips per batch while building a cache
CACHE_BATCH = 1024
CACHED_PATHWAYS = ("spectrogram", "quadrature")


def level_name(noise_db: float | None) -> str:
    return "clean" if noise_db is None else f"{noise_db:g}db"


def rows_path(pathway: str, noise_db: float | None, bands: int = 16, window: int = HOP_N_FFT) -> Path:
    """The cache for a pathway, level, band count and window; 16 bands keep paper 02's file name."""
    size = ("" if bands == 16 else f"-{bands}bands") + ("" if window == HOP_N_FFT else f"-w{window}")
    return ROWS_DIR / f"{pathway}{size}-{level_name(noise_db)}.pt"


def _save(out: Path, rows: torch.Tensor, tvalid: torch.Tensor, **meta) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    torch.save({"rows": rows, "tvalid": tvalid, **meta}, tmp)
    tmp.replace(out)                    # atomic: a reader never sees half a cache


def build_rows(bank: dict, pathway: str, noise_db: float | None, bands: int = 16, window: int = HOP_N_FFT) -> Path:
    """Front-end rows for every clip in the bank, in canonical order."""
    n = len(bank["labels"])
    rows = tvalid = None
    for a in range(0, n, CACHE_BATCH):
        idx = torch.arange(a, min(a + CACHE_BATCH, n))
        waves, lens, _ = recognition_clips(bank, idx, noise_db)
        r = front_end(waves, pathway, bands, window)
        if rows is None:
            rows = torch.empty((n, *r.shape[1:]))
            tvalid = torch.empty(n, dtype=torch.long)
        rows[idx], tvalid[idx] = r, valid_frames(lens, pathway)
    out = rows_path(pathway, noise_db, bands, window)
    _save(out, rows, tvalid, n_clips=n, pathway=pathway, noise_db=noise_db, bands=bands, window=window)
    return out


def order_rows_path(pair: tuple[int, int], set_code: int, noise_db: float | None, bands: int = 16) -> Path:
    """An order-task set's cache; 16 bands keep paper 02's file name."""
    size = "" if bands == 16 else f"-{bands}bands"
    return ROWS_DIR / f"order-pair{pair[0]}{pair[1]}-set{set_code}{size}-{level_name(noise_db)}.pt"


def sequence_rows_path(length: int, set_code: int, noise_db: float | None, bands: int = 16) -> Path:
    size = "" if bands == 16 else f"-{bands}bands"
    return ROWS_DIR / f"sequence{length}-set{set_code}{size}-{level_name(noise_db)}.pt"


def order_pool(bank: dict, set_code: int, n_train: int, n_test: int) -> tuple[torch.Tensor, int]:
    """(the speakers' pool, the set's size): the test pool for the test set (code 0), else the training pool."""
    train_pool, test_pool = protocol_a(bank)
    return (test_pool, n_test) if set_code == 0 else (train_pool, n_train)


def build_order_rows(bank: dict, pair: tuple[int, int], set_code: int, noise_db: float | None,
                     bands: int = 16) -> Path:
    """Front-end rows for one order-task set: the test set (code 0) or seed s's training set (s + 1)."""
    pool, n = order_pool(bank, set_code, ORDER_TRAIN, ORDER_TEST)
    first, second, labels = order_set(bank, pool, pair, n, set_code)
    rows, tvalid = [], []
    for a in range(0, n, CACHE_BATCH):
        waves, lens = order_clips(bank, first[a:a + CACHE_BATCH], second[a:a + CACHE_BATCH],
                                  pair, set_code, a, noise_db)
        rows.append(front_end(waves, "spectrogram", bands))
        tvalid.append(valid_frames(lens, "spectrogram"))
    out = order_rows_path(pair, set_code, noise_db, bands)
    _save(out, torch.cat(rows), torch.cat(tvalid), labels=labels, n_clips=n)
    return out


def build_sequence_rows(bank: dict, length: int, set_code: int, noise_db: float | None,
                        bands: int = 16) -> Path:
    """Front-end rows for one digit-sequence set."""
    pool, n = order_pool(bank, set_code, SEQUENCE_TRAIN, SEQUENCE_TEST)
    idx, digits = sequence_set(bank, pool, length, n, set_code)
    rows, tvalid = [], []
    for a in range(0, n, CACHE_BATCH):
        waves, lens = sequence_clips(bank, idx[a:a + CACHE_BATCH], length, set_code, a, noise_db)
        rows.append(front_end(waves, "spectrogram", bands))
        tvalid.append(valid_frames(lens, "spectrogram"))
    out = sequence_rows_path(length, set_code, noise_db, bands)
    _save(out, torch.cat(rows), torch.cat(tvalid), labels=digits, n_clips=n)
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

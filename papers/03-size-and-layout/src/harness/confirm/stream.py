"""The read at scale: an arm too large to hold, standardized and projected one channel at a time.

Paper 02 reads an arm by keeping every clip's features, standardizing each
feature by its training statistics and projecting them with one fixed
Gaussian matrix. That needs the features of every clip and the whole matrix
in memory at once. At 128 x 128 with 16 channels the windowed read is
6,291,456 features a clip: 202 GB for the 8,048 clips of a run, and 103 GB for
the [native, 4,096] matrix. Neither fits.

The read is linear after standardization, and channels do not interact, so
it splits by channel exactly:

    z @ P = sum over channels c of z_c @ P_c

where z_c is channel c's standardized features and P_c the matrix rows that
meet them. Standardization is per feature, so channel c's statistics are
computed from channel c's training features alone, with paper 02's own code.
The streamed read therefore simulates one channel at a time (the channel's
own kernel, natural frequencies and starting phases, sliced from the full
arm), keeps that channel's training features only (3.2 GB at 128 x 128),
standardizes them, and adds their projection into the run's projected
features; the test clips of the channel are simulated in batches and
projected as they come. Peak memory is one channel's training features and
its rows of the two matrices (fixed and seeded, 6.4 GB each at 128 x 128),
whatever the channel count.

It computes the same function as the in-memory read. What differs:

1. The order of the floating-point sums: the projection is accumulated
   channel by channel rather than in one product. Tested against the
   in-memory read on the same matrix (tests/test_stream.py): the projected
   features agree to float32 rounding and every accuracy is identical.
2. The draw of the matrix. Paper 02's matrix is one sequential stream from
   one seed, and its rows for one channel are scattered through it (the
   features are ordered by window, then statistic, then signal, and channel c
   occupies 24 separate runs of rows), so reaching them means drawing the
   whole stream. The streamed read draws each channel's rows from a seed of
   their own, 100 times the in-memory read's generator seed plus the channel:
   424,200 + c for the fixed projection (the in-memory fixed seed is 4,242),
   and 100 * (4,242 + 1 + s) + c for the projection seeded by run seed s (the
   in-memory seeded rule is 4,242 + 1 + s). Same distribution, N(0, 1/native),
   same scale: another fixed Gaussian matrix, as paper 02's matrices already
   are from one native width to the next.

Arms at or below `STREAM_STATES` (4,096 states, the largest paper 02 read in
memory) are always read in memory, exactly as paper 02 reads them; every cell
paper 03 takes from paper 02's record is one of them.

On a GPU the channel is simulated there and its features and matrix rows are
projected there; the statistics and the ridge stay on the CPU in float64. The
two matrices of a channel are drawn on the CPU (the generator must be the
same on every device), in two threads at once.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import torch

from harness.confirm import arms as am
from harness.confirm import readout as ro
from harness.measurement.features import MAX_WIDTH, projection_seed

#: arms with more states than this are read channel by channel
STREAM_STATES = 4096
#: a channel's generator seed is this many times the in-memory generator seed, plus the channel
CHANNEL_STRIDE = 100


def channel_seed(c: int, seed: int | None = None) -> int:
    """Channel c's generator seed: for the fixed projection (None) or the one seeded by run seed `seed`."""
    return CHANNEL_STRIDE * projection_seed(seed) + c


def channel_projection(native: int, c: int, rows: int, width: int, seed: int | None = None) -> torch.Tensor:
    """[rows, width]: channel c's rows of the streamed read's fixed (None) or seeded matrix.

    Every channel draws all MAX_WIDTH columns, so each width is the leading
    columns of the widest, as in paper 02.
    """
    gen = torch.Generator().manual_seed(channel_seed(c, seed))
    return (torch.randn(rows, MAX_WIDTH, generator=gen) / math.sqrt(native))[:, :width]


def streamed_read(arm: am.Arm, model: torch.nn.Module, blocks: Callable[[int, int], Iterator],
                  n_train: int, n_test: int, widths: tuple[int, ...], task: str, seed: int, device: str = "cpu",
                  projection: Callable[..., torch.Tensor] | None = None, read: str | None = None,
                  drive: str | None = None,
                  ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], int, dict[str, torch.Tensor]]:
    """One read of a large arm: ({projection: (projected training features, projected test
    features)}, native width, {instrument: per test clip}).

    `read` is one of the arm's reads (`arms.reads`), its primary one by
    default; a composite read such as "windowed+rate" concatenates, channel by
    channel, the blocks it names. `blocks(c, part)` yields (rows, valid
    frames, slice) batches of the training clips (part 0, the first
    `n_train`) or the test clips (part 1), in readout order; the batch size is
    the caller's. `projection(native, c, rows, width, seed)` gives channel c's
    rows of the fixed (seed None) or the seeded matrix; the default is the
    streamed read's own draw, and a test passes paper 02's rows instead. With
    `drive` given and a network arm, the network's instruments
    (`arms.field_instruments`) are recorded over the test clips, channel by
    channel; they do not touch the read.
    """
    projection = channel_projection if projection is None else projection
    if arm.kind not in ("field", "bank"):
        raise ValueError(f"only an untrained network or bank is read channel by channel, not {arm.kind}")
    read = read or am.PRIMARY_READ[task]
    keys = am.reads(arm, task)[read]
    seeds = {"fixed": None, "seeded": seed}
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    instruments: dict[str, torch.Tensor] = {}
    per_channel = top = native = None

    def features(one, sub, rows, tvalid):
        sig = am.frozen_signals(one, sub, rows.to(device))
        f = am.frozen_features(one, sig, tvalid.to(device), task)
        return sig, torch.cat([f[k] for k in keys], dim=1)

    with ThreadPoolExecutor(len(seeds)) as pool:
        for c in range(arm.channels):
            one, sub = am.channel(arm, model, c)
            # the channel's training features, kept on the CPU: its statistics need all of them
            x_tr = None
            for rows, tvalid, where in blocks(c, 0):
                _, f = features(one, sub, rows, tvalid)
                f = f.cpu()
                if not out:                          # the first batch of the first channel sizes everything
                    per_channel = f.shape[1]
                    native = per_channel * arm.channels
                    top = max(e for _, e in ro.wanted_widths(widths, native, False) if e < native)
                    out = {k: (torch.zeros(n_train, top), torch.zeros(n_test, top)) for k in seeds}
                if x_tr is None:
                    x_tr = torch.zeros(n_train, per_channel)
                x_tr[where] = f
            draws = {k: pool.submit(projection, native, c, per_channel, top, s) for k, s in seeds.items()}
            mean, sd = ro._stats(x_tr)
            ps = {k: d.result().to(device) for k, d in draws.items()}
            for k, p in ps.items():
                out[k][0].add_(ro._projected([x_tr], slice(0, n_train), mean, sd, top, p, device))
            del x_tr
            for rows, tvalid, where in blocks(c, 1):
                sig, f = features(one, sub, rows, tvalid)
                for k, p in ps.items():
                    out[k][1][where] += ro._projected([f], slice(0, len(f)), mean, sd, top, p, device)
                if drive is not None and arm.kind == "field":
                    # channels are equal in size, so a channel mean over channels is the network's mean
                    for name, v in am.field_instruments(sig, rows.to(device), drive, 1, arm.grid).items():
                        instruments.setdefault(name, torch.zeros(n_test))[where] += v.cpu() / arm.channels
            del ps
    return out, native, instruments

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
its rows of the matrix (6.4 GB at 128 x 128), whatever the channel count.

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
   their own, `STREAM_SEED + c`, with the same distribution, N(0, 1/native),
   and the same scale. It is another fixed Gaussian matrix, as paper 02's
   matrices already are from one native width to the next.

Arms at or below `STREAM_STATES` (4,096 states, the largest paper 02 read in
memory) are always read in memory, exactly as paper 02 reads them; every cell
paper 03 takes from paper 02's record is one of them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import torch

from harness.confirm import arms as am
from harness.confirm import readout as ro
from harness.measurement.features import MAX_WIDTH

#: arms with more states than this are read channel by channel
STREAM_STATES = 4096
#: the streamed read's per-channel projection seeds start here (paper 02's single draw is seed 4242)
STREAM_SEED = 424_200
#: the read the streamed path records: the four-window statistics, paper 02's primary read
READ = "windowed"


def channel_projection(native: int, c: int, rows: int, width: int) -> torch.Tensor:
    """[rows, width]: channel c's rows of the streamed read's matrix, drawn from their own seed.

    Every channel draws all MAX_WIDTH columns, so each width is the leading
    columns of the widest, as in paper 02.
    """
    gen = torch.Generator().manual_seed(STREAM_SEED + c)
    return (torch.randn(rows, MAX_WIDTH, generator=gen) / math.sqrt(native))[:, :width]


def streamed_read(arm: am.Arm, model: torch.nn.Module, blocks: Callable[[int, int], Iterator],
                  n_train: int, n_test: int, widths: tuple[int, ...], task: str, device: str = "cpu",
                  projection: Callable[[int, int, int, int], torch.Tensor] | None = None,
                  ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(projected training features, projected test features, native width) for one read of a large arm.

    `blocks(c, part)` yields (rows, valid frames, slice) batches of the
    training clips (part 0, the first `n_train`) or the test clips (part 1),
    in readout order; the batch size is the caller's. `projection(native, c,
    rows, width)` gives channel c's rows of the matrix; the default is the
    streamed read's own draw, and a test passes paper 02's rows instead.
    """
    projection = channel_projection if projection is None else projection
    if arm.kind not in ("field", "bank"):
        raise ValueError(f"only an untrained network or bank is read channel by channel, not {arm.kind}")
    per_channel = None
    p_tr = p_te = None
    top = native = None
    for c in range(arm.channels):
        one, sub = am.channel(arm, model, c)
        # the channel's training features, kept: its statistics need all of them
        x_tr = None
        for rows, tvalid, where in blocks(c, 0):
            sig = am.frozen_signals(one, sub, rows.to(device))
            f = am.frozen_features(one, sig, tvalid.to(device), task)[READ].cpu()
            if p_tr is None:                     # the first batch of the first channel sizes everything
                per_channel = f.shape[1]
                native = per_channel * arm.channels
                top = max(e for _, e in ro.wanted_widths(widths, native, False) if e < native)
                p_tr, p_te = torch.zeros(n_train, top), torch.zeros(n_test, top)
            if x_tr is None:
                x_tr = torch.zeros(n_train, per_channel)
            x_tr[where] = f
        mean, sd = ro._stats(x_tr)
        p = projection(native, c, per_channel, top)
        p_tr += ro._projected([x_tr], slice(0, n_train), mean, sd, top, p=p)
        del x_tr
        for rows, tvalid, where in blocks(c, 1):
            sig = am.frozen_signals(one, sub, rows.to(device))
            f = am.frozen_features(one, sig, tvalid.to(device), task)[READ].cpu()
            p_te[where] += ro._projected([f], slice(0, len(f)), mean, sd, top, p=p)
    return p_tr, p_te, native

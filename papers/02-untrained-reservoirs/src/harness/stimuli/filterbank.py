"""The fixed log-spaced filterbank: frequency band -> lattice row.

Four bands per octave, so sixteen rows span exactly four octaves. This is the
tonotopy the whole harness is built around: band r drives lattice row r, in
every channel and every column, under every geometry. Nothing here is learned
or fitted.
"""

from __future__ import annotations

import math

import torch

from harness.utils.constants import BANDS_PER_OCTAVE, F_LO


def band_edges(grid: int) -> torch.Tensor:
    """Log-spaced filterbank edges [grid+1] in cycles/frame, 4 bands/octave."""
    r = torch.arange(grid + 1, dtype=torch.float64)
    return F_LO * (2.0 ** (r / BANDS_PER_OCTAVE))


def band_index(freq: float, grid: int) -> int:
    """Which filterbank row a frequency lands in."""
    return int(math.floor(BANDS_PER_OCTAVE * math.log2(freq / F_LO)))

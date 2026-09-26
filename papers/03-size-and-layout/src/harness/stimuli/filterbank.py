"""The fixed log-spaced filterbank: frequency band -> lattice row.

The bands span the same four octaves at every band count, G / 4 bands per
octave: four per octave at 16 bands, exactly as in paper 02, two at 8, eight
at 32, and so on. Paper 02 fixed four per octave, which at more than 16 bands
would run past the top of the range (32 bands would span eight octaves, the
upper half above the Nyquist frequency). Tonotopic natural frequencies take
their centres. Nothing here is learned or fitted.
"""

from __future__ import annotations

import math

import torch

from harness.utils.constants import F_LO, OCTAVES


def band_edges(grid: int) -> torch.Tensor:
    """Log-spaced band edges [grid+1] in cycles/frame, spanning F_LO .. F_LO * 2^4."""
    r = torch.arange(grid + 1, dtype=torch.float64)
    return F_LO * (2.0 ** (r * OCTAVES / grid))


def band_index(freq: float, grid: int) -> int:
    """Which of `grid` bands a frequency (cycles/frame) lands in."""
    return int(math.floor(grid / OCTAVES * math.log2(freq / F_LO)))

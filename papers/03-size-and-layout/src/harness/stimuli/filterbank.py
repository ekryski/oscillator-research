"""The fixed log-spaced filterbank: frequency band -> lattice row.

The bands span the same four octaves at every band count, G / 4 bands per
octave: four per octave at 16 bands, exactly as in paper 02, two at 8, eight
at 32, and so on. Paper 02 fixed four per octave, which at more than 16 bands
would run past the top of the range (32 bands would span eight octaves, the
upper half above the Nyquist frequency). The carrier pathway splits the
waveform into these bands, and tonotopic natural frequencies take their
centres. Nothing here is learned or fitted.
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


def bandpass_rows(wave: torch.Tensor, grid: int) -> torch.Tensor:
    """[B,T] -> [B,T,G]: row r carries the band-r-filtered signal, carrier kept.

    Full-clip FFT masking: bins outside [F_LO, F_LO*2^4) are dropped. The
    carrier, not an envelope, is what reaches the oscillators, so phase
    entrainment to the stimulus itself is possible."""
    b, t = wave.shape
    x = torch.fft.rfft(wave, dim=1)  # [B, T//2+1]
    bin_f = torch.arange(x.shape[1], dtype=torch.float64) / t
    e = band_edges(grid)
    mask = ((bin_f[None, :] >= e[:-1, None]) & (bin_f[None, :] < e[1:, None])).to(torch.float32)
    y = torch.fft.irfft(x[:, None, :] * mask[None, :, :], n=t, dim=2)  # [B, G, T]
    return y.transpose(1, 2).contiguous()  # [B, T, G]

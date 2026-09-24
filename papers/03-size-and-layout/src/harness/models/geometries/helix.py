"""Helix — the storage read as one long ring, an octave per turn.

The [G, G] grid becomes a 1-D ring of N = G^2 positions (p = row*G + col) with
a length-N circulant kernel. The point is tonotopic: at G=16 the ring spans
four octaves over 256 positions, so 64 positions is exactly one turn = one
octave, and a kernel tap at offset +/-64 couples octave-related bands
"vertically across turns" — the one venue where harmonic structure is a
neighbourhood relation rather than a distant one.

The ring is closed, so the seam couples the top band's end to the bottom band's
start at ring distance 1. That is a recorded property of the pre-registered
design, not an accident.
"""

from __future__ import annotations

import torch

from harness.models.geometries.base import Geometry


class Helix(Geometry):
    name = "helix"
    frequency_axis = "along the coil, one octave per turn"
    supports_kernel_support = False  # the ring indexes offsets 1-D, not (row, col)

    def kernel_spectrum(self, kernel: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(kernel.flatten(1))

    def spectrum_to_taps(self, kfft: torch.Tensor, embedded_shape: tuple) -> torch.Tensor:
        return torch.fft.irfft(kfft, n=self.grid * self.grid)

    def circulant_index(self) -> torch.Tensor:
        n = self.grid * self.grid
        pos = torch.arange(n)
        return (pos[:, None] - pos[None, :]) % n

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        n = self.grid * self.grid
        x = field.flatten(-2)
        return torch.fft.irfft(torch.fft.rfft(x) * coup, n=n).view_as(field)

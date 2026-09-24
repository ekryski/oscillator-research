"""Coil and cochlea: the storage read as one open coil, base to apex.

The [G, G] grid becomes one open line of N = G^2 positions (p = row*G + col),
like the helix but with its ends apart: the lowest band (row 0) sits at the
apex, p = 0, and the highest (row G-1) at the base, p = N-1. At G = 16 the coil
spans four octaves over 256 positions, so 64 positions make one turn and one
octave, and a kernel tap at offset +/-64 couples a site to the site beside it on
the next turn, as a spiral's adjacent turns sit side by side. Frequency falls
exponentially from base to apex, as it does along the cochlea, because the mel
bands are log-spaced; tonotopic natural frequencies follow the same map.

The coupling is a linear convolution along the line: the kernel's G^2 taps are
read as signed offsets -N/2 .. N/2 - 1, embedded in a zero-padded buffer of 2N,
so a site couples to sites up to two turns away in either direction and the
coil never closes on itself.

`Cochlea` adds the two features of the cochlea's physics a coil alone lacks:

  direction   the travelling wave runs from base to apex, so taps that carry
              influence toward the apex (offset < 0) are scaled by 1 + a and
              taps toward the base by 1 - a, with a = DIRECTION
  curvature   the coil tightens toward the apex; the coupling into each site is
              scaled by the curvature there relative to the apex's, which for a
              logarithmic spiral whose radius falls to APEX_RADIUS of the
              base's is APEX_RADIUS ** (p / (N - 1)): 1 at the apex, APEX_RADIUS
              at the base

Both scalings keep the coupling ceiling a true operator-norm bound: the
direction is applied to the kernel before its spectrum is capped, and the
curvature weights never exceed 1. Every coupling is still a random kernel tap:
the coil fixes which pairs share a tap and where the line ends, not which pairs
couple, as every other geometry here does.
"""

from __future__ import annotations

import torch

from harness.models.geometries.base import Geometry


class Coil(Geometry):
    name = "coil"
    frequency_axis = "along the coil, apex (low) to base (high), one octave per turn, ends open"
    supports_kernel_support = False  # offsets are 1-D along the coil, not (row, col)
    #: the travelling wave's direction; 0 on the plain coil
    direction = 0.0

    @property
    def n(self) -> int:
        return self.grid * self.grid

    def curvature(self, like: torch.Tensor) -> torch.Tensor | None:
        """[N] weights on the coupling into each site; None on the plain coil."""
        return None

    def embed_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        """[C, G, G] -> [C, 2N]: taps at signed offsets, zero-padded so the line stays open."""
        n, h = self.n, self.n // 2
        flat = kernel.flatten(1)
        out = kernel.new_zeros(kernel.shape[0], 2 * n)
        out[:, :h] = flat[:, :h]                         # offsets 0 .. N/2 - 1 (toward the base)
        out[:, n + h:] = flat[:, h:]                     # offsets -N/2 .. -1 (toward the apex)
        if self.direction:
            out[:, 1:h] = out[:, 1:h] * (1.0 - self.direction)
            out[:, n + h:] = out[:, n + h:] * (1.0 + self.direction)
        return out

    def kernel_spectrum(self, kernel: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(kernel)

    def spectrum_to_taps(self, kfft: torch.Tensor, embedded_shape: tuple) -> torch.Tensor:
        return torch.fft.irfft(kfft, n=2 * self.n)

    def circulant_index(self) -> torch.Tensor:
        pos = torch.arange(self.n)
        return (pos[:, None] - pos[None, :]) % (2 * self.n)

    def dense_operator(self, taps: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        op = taps[:, index]
        w = self.curvature(taps)
        return op if w is None else op * w[None, :, None]

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        n = self.n
        x = torch.nn.functional.pad(field.flatten(-2), (0, n))
        out = torch.fft.irfft(torch.fft.rfft(x) * coup, n=2 * n)[..., :n]
        w = self.curvature(out)
        return (out if w is None else out * w).view_as(field)


class Cochlea(Coil):
    name = "cochlea"
    frequency_axis = "along the coil, apex (low) to base (high), one octave per turn, ends open"
    #: taps toward the apex x (1 + a), toward the base x (1 - a)
    direction = 0.5
    #: the coil's radius at the apex, as a share of its radius at the base
    APEX_RADIUS = 0.25

    def curvature(self, like: torch.Tensor) -> torch.Tensor:
        def build() -> torch.Tensor:
            p = torch.arange(self.n, dtype=torch.float64)
            return self.APEX_RADIUS ** (p / (self.n - 1))
        return self._constant("curvature", build, like)

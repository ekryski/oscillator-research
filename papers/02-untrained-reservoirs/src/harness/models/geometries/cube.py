"""Cube — a fully periodic 3-torus prism.

Each storage row's G columns are read as a sqrt(G) x sqrt(G) slab, giving a
(G, sqrt(G), sqrt(G)) prism (16x4x4 at G=16) that is periodic on all three
axes. Its role in the matrix is dimensionality: at matched oscillator count it
shortens interaction path lengths, which is the mechanism by which higher
dimension could plausibly matter.

Fully periodic axes are symmetric, so which axis carries frequency is
arbitrary-but-fixed: z (= the storage row) is the tonotopic axis, band b drives
z-slice b. An open-z variant is a different venue, not this one.
"""

from __future__ import annotations

import math

import torch

from harness.models.geometries.base import Geometry


def cube_dims(grid: int) -> tuple[int, int, int]:
    """(G, s, s) prism dims with s = sqrt(G): z-slices are the G frequency
    bands, each an s x s slab of one storage row."""
    s = math.isqrt(grid)
    if s * s != grid:
        raise ValueError(f"cube boundary needs a perfect-square grid, got {grid}")
    return (grid, s, s)


class Cube(Geometry):
    name = "cube"
    frequency_axis = "the z axis (fully periodic)"
    supports_kernel_support = False  # 3-D prism offsets, not (row, col)

    def validate(self) -> None:
        self.zyx = cube_dims(self.grid)

    def kernel_spectrum(self, kernel: torch.Tensor) -> torch.Tensor:
        k = kernel.reshape(kernel.shape[0], *self.zyx)
        return torch.fft.rfftn(k, dim=(-3, -2, -1))

    def spectrum_to_taps(self, kfft: torch.Tensor, embedded_shape: tuple) -> torch.Tensor:
        return torch.fft.irfftn(kfft, s=self.zyx, dim=(-3, -2, -1)).flatten(1)

    def circulant_index(self) -> torch.Tensor:
        gz, s, _ = self.zyx
        pos = torch.arange(self.grid * self.grid)
        zi = pos.div(s * s, rounding_mode="floor")
        yi = pos.div(s, rounding_mode="floor") % s
        xi = pos % s
        dz = (zi[:, None] - zi[None, :]) % gz
        dy = (yi[:, None] - yi[None, :]) % s
        dx = (xi[:, None] - xi[None, :]) % s
        return dz * (s * s) + dy * s + dx

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        x = field.reshape(*field.shape[:-2], *self.zyx)
        y = torch.fft.irfftn(torch.fft.rfftn(x, dim=(-3, -2, -1)) * coup,
                             s=self.zyx, dim=(-3, -2, -1))
        return y.reshape(field.shape)

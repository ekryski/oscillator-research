"""Cube: a fully periodic three-dimensional prism.

Each storage row's G columns are read as an a x b slab, giving a (G, a, b)
prism that is periodic on all three axes. Its role is dimensionality: at the
same number of oscillators it shortens the paths between them.

Paper 02 ran it at 16 x 16, where the slab is square (4 x 4). A square slab
needs G to be a perfect square, which 8, 32 and 128 are not, so paper 03 cuts
each row into the most nearly square slab the lattice allows: a is the largest
power of two whose square does not exceed G, and b = G / a. At 16 and 64 that
is the square slab (4 x 4, 8 x 8), so the geometry is paper 02's there; at 8,
32 and 128 it is 2 x 4, 4 x 8 and 8 x 16.

Fully periodic axes are symmetric, so which axis carries frequency is
arbitrary but fixed: z (the storage row) is the tonotopic axis, and band b
drives z-slice b. An open-z variant is a different geometry, not this one.
"""

from __future__ import annotations

import math

import torch

from harness.models.geometries.base import Geometry


def cube_dims(grid: int) -> tuple[int, int, int]:
    """(G, a, b) prism dims: the G frequency bands, each an a x b slab of one storage row, a <= b."""
    if grid < 4 or grid & (grid - 1):
        # a non-power-of-two row still splits, but only the lattices paper 03 runs are pinned by tests
        s = math.isqrt(grid)
        while grid % s:
            s -= 1
        if s < 2:
            raise ValueError(f"cube boundary cannot split a row of {grid} into a slab")
        return (grid, s, grid // s)
    a = 1 << (grid.bit_length() - 1) // 2
    return (grid, a, grid // a)


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
        gz, a, b = self.zyx
        pos = torch.arange(self.grid * self.grid)
        zi = pos.div(a * b, rounding_mode="floor")
        yi = pos.div(b, rounding_mode="floor") % a
        xi = pos % b
        dz = (zi[:, None] - zi[None, :]) % gz
        dy = (yi[:, None] - yi[None, :]) % a
        dx = (xi[:, None] - xi[None, :]) % b
        return dz * (a * b) + dy * b + dx

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        x = field.reshape(*field.shape[:-2], *self.zyx)
        y = torch.fft.irfftn(torch.fft.rfftn(x, dim=(-3, -2, -1)) * coup, s=self.zyx, dim=(-3, -2, -1))
        return y.reshape(field.shape)

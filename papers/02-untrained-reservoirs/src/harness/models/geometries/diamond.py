"""Diamond — two interleaved FCC sublattices, tetrahedrally bonded.

Two sublattices A and B on a periodic (G/2, sqrt(G), sqrt(G)) grid of primitive
runs; at G=16 that is 8x4x4 runs x 2 sites = 128 A + 128 B = 256 sites per
channel, count-matched with every other venue. Coupling is strictly
INTER-sublattice, because diamond's tetrahedral bonds all connect A to B:

    out_A = K_AB * f_B     out_B = K_BA * f_A

each a 3-D circular convolution over the cell grid — a block-structured
transform over the 2-site basis.

The tetrahedral motif is the lattice's bond structure, not a mask laid over a
denser one: all C*G^2 taps are real parameters (2 direction blocks x G^2/2 run
offsets). Storage layer l = 2u + sublattice is the crystal layer along the a1
axis, matching diamond's alternating A/B stacking, so band b drives layer b and
the tetrahedral bonds couple ADJACENT bands only.
"""

from __future__ import annotations

import math

import torch

from harness.models.geometries.base import Geometry


def diamond_dims(grid: int) -> tuple[int, int, int]:
    """(G/2, s, s) primitive-cell grid with s = sqrt(G): G crystal layers along
    the a1 axis, each layer an s x s sheet of one sublattice."""
    s = math.isqrt(grid)
    if s * s != grid or grid % 2:
        raise ValueError(f"diamond boundary needs an even perfect-square grid, got {grid}")
    return (grid // 2, s, s)


class Diamond(Geometry):
    name = "diamond"
    frequency_axis = "the a1 cell axis, A/B sublattices interleaved"
    supports_kernel_support = False  # crystal offsets, not (row, col)

    def validate(self) -> None:
        self.uvw = diamond_dims(self.grid)

    def kernel_spectrum(self, kernel: torch.Tensor) -> torch.Tensor:
        # storage rows 0::2 hold K_AB (A hears B), rows 1::2 hold K_BA
        c, nu, s = kernel.shape[0], self.uvw[0], self.uvw[1]
        k_ab = kernel[:, 0::2].reshape(c, nu, s, s)
        k_ba = kernel[:, 1::2].reshape(c, nu, s, s)
        return torch.stack((torch.fft.rfftn(k_ab, dim=(-3, -2, -1)),
                            torch.fft.rfftn(k_ba, dim=(-3, -2, -1))), dim=1)

    def spectrum_to_taps(self, kfft: torch.Tensor, embedded_shape: tuple) -> torch.Tensor:
        spat = torch.fft.irfftn(kfft, s=self.uvw, dim=(-3, -2, -1))  # [C, 2, nu, s, s]
        c = spat.shape[0]
        taps = spat.transpose(1, 2).reshape(c, self.grid * self.grid)  # rows 2*du + d
        # a zero slot past the kernel, where same-sublattice pairs point:
        # diamond coupling is strictly bipartite, so those pairs are dead
        return torch.cat((taps, taps.new_zeros(c, 1)), dim=1)

    def circulant_index(self) -> torch.Tensor:
        g, (nu, s, _) = self.grid, self.uvw
        n = g * g
        pos = torch.arange(n)
        layer, colp = pos.div(g, rounding_mode="floor"), pos % g
        u, sub = layer.div(2, rounding_mode="floor"), layer % 2
        v, w = colp.div(s, rounding_mode="floor"), colp % s
        du = (u[:, None] - u[None, :]) % nu
        dv = (v[:, None] - v[None, :]) % s
        dw = (w[:, None] - w[None, :]) % s
        idx = (2 * du + sub[:, None]) * g + dv * s + dw
        return torch.where(sub[:, None] == sub[None, :], torch.full_like(idx, n), idx)

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        g = self.grid
        f = field.reshape(*field.shape[:-2], g, *self.uvw[1:])
        fa = torch.fft.rfftn(f[..., 0::2, :, :], dim=(-3, -2, -1))
        fb = torch.fft.rfftn(f[..., 1::2, :, :], dim=(-3, -2, -1))
        out_a = torch.fft.irfftn(fb * coup[:, 0], s=self.uvw, dim=(-3, -2, -1))
        out_b = torch.fft.irfftn(fa * coup[:, 1], s=self.uvw, dim=(-3, -2, -1))
        return torch.stack((out_a, out_b), dim=-3).reshape(field.shape)

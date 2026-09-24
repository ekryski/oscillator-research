"""Torus — both lattice axes periodic.

The reference venue: a plain 2-D circular convolution, and the one every other
shape is measured against. Rows are the tonotopic axis, so band b drives row b
and the wrap couples the top band back to the bottom.
"""

from __future__ import annotations

import torch

from harness.models.geometries.base import PlanarGeometry


class Torus(PlanarGeometry):
    name = "torus"
    frequency_axis = "grid rows (periodic: the top band wraps to the bottom)"
    embed_shape = (1, 1)

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        g = self.grid
        return torch.fft.irfft2(torch.fft.rfft2(field) * coup, s=(g, g))

"""Cylinder — the frequency axis OPEN, the other periodic.

The cochlea's own arrangement: band order is strictly linear, so energy at the
top band cannot couple around to the bottom. Implemented as an exact linear
convolution along rows by zero-padding to 2G; pad and crop are norm-1 maps, so
the spectral clamp on the padded kernel stays a true operator-norm bound.
"""

from __future__ import annotations

import torch

from harness.models.geometries.base import PlanarGeometry


class Cylinder(PlanarGeometry):
    name = "cylinder"
    frequency_axis = "grid rows (open: no wrap between the top and bottom band)"
    embed_shape = (2, 1)

    def embed_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        return self._signed_row_embedding(kernel)

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        g = self.grid
        f = torch.nn.functional.pad(field, (0, 0, 0, g))
        return torch.fft.irfft2(torch.fft.rfft2(f) * coup, s=(2 * g, g))[..., :g, :]

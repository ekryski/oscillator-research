"""Sheet — both axes open.

The zero-periodic-axis end of the sheet/cylinder/torus series, which is how the
paper varies the number of periodic axes as a single factor. Nothing wraps, so
travelling waves terminate at every edge instead of recirculating.
"""

from __future__ import annotations

import torch

from harness.models.geometries.base import PlanarGeometry


class Sheet(PlanarGeometry):
    name = "sheet"
    frequency_axis = "grid rows (open; the column axis is open too)"
    embed_shape = (2, 2)

    def embed_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        return self._signed_both_embedding(kernel)

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        g = self.grid
        f = torch.nn.functional.pad(field, (0, g, 0, g))
        return torch.fft.irfft2(torch.fft.rfft2(f) * coup, s=(2 * g, 2 * g))[..., :g, :g]

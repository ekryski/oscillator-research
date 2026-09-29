"""The non-orientable pair — moebius and klein.

Both glue one axis under a FLIP of the other, so a travelling wave crossing the
twisted seam comes back mirrored. They exist in the matrix as the orientability
control: same stencil, same budget, same tonotopy as their orientable parents
(moebius from the cylinder, klein from the torus), differing only in whether
the venue has a consistent global orientation.

Implemented via the orientation double cover: extend the field with a mirrored
copy, run one ordinary transform on the cover, crop back to the fundamental
domain. Exact — it matches the dense two-gather operator to f64 rounding.

Two honest caveats live here. The extension DOUBLES input energy (its operator
norm is sqrt(2)), so max |K-hat| alone is NOT a true bound; these venues clamp
their embedded spectrum to clamp/sqrt(2) instead, which is conservative by
design. And a non-orientable venue has no global "up the rows", so kernels
asymmetric in the twisted offset are chart-dependent by nature; only the
symmetric part commutes with the twisted translation.
"""

from __future__ import annotations

import math

import torch

from harness.models.geometries.base import PlanarGeometry

TWIST_NORM_FACTOR = math.sqrt(2.0)


class _Twisted(PlanarGeometry):
    @property
    def clamp_factor(self) -> float:
        return TWIST_NORM_FACTOR

    def dense_operator(self, taps: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # direct copy + deck-transformed mirror copy of the double cover; at
        # most one term is nonzero per pair, because the tap support fits
        # inside half the cover, so the sum IS the exact operator
        return taps[:, index[0]] + taps[:, index[1]]

    def _deck_index(self, deck_rows: torch.Tensor, deck_cols: torch.Tensor,
                    cols_pad: int) -> torch.Tensor:
        g = self.grid
        pos = torch.arange(g * g)
        yi, xi = pos.div(g, rounding_mode="floor"), pos % g
        dy_a = (yi[:, None] - yi[None, :]) % (2 * g)
        dx_a = (xi[:, None] - xi[None, :]) % cols_pad
        dy_b = (yi[:, None] - deck_rows[None, :]) % (2 * g)
        dx_b = (xi[:, None] - deck_cols[None, :]) % cols_pad
        return torch.stack((dy_a * cols_pad + dx_a, dy_b * cols_pad + dx_b))


class Moebius(_Twisted):
    """The cylinder with its periodic COLUMN axis glued under a flip of the
    open row axis — the only axis a moebius gluing can flip is the frequency
    axis, so the seam acts on frequency."""

    name = "moebius"
    frequency_axis = "grid rows, as cylinder (the seam flip acts on frequency)"
    embed_shape = (2, 2)

    def embed_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        return self._signed_both_embedding(kernel)

    def circulant_index(self) -> torch.Tensor:
        g = self.grid
        pos = torch.arange(g * g)
        yi, xi = pos.div(g, rounding_mode="floor"), pos % g
        return self._deck_index(g - 1 - yi, xi + g, 2 * g)  # (r, c) -> (G-1-r, c+G)

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        g = self.grid
        f = torch.cat((field, field.flip(-2)), dim=-1)      # row-mirrored copy
        f = torch.nn.functional.pad(f, (0, 0, 0, g))        # rows open, as cylinder
        return torch.fft.irfft2(torch.fft.rfft2(f) * coup, s=(2 * g, 2 * g))[..., :g, :g]


class Klein(_Twisted):
    """The torus with its ROW axis glued under a flip of the column axis:
    (r+G, c) ~ (r, G-1-c). Rows stay the frequency axis and column wraps stay
    untwisted, so the flip acts on the NON-frequency axis — the contrast with
    moebius that makes the pair a single-factor comparison."""

    name = "klein"
    frequency_axis = "grid rows, as torus (the seam flip acts on columns)"
    embed_shape = (2, 1)

    def embed_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        # the double cover is a genuine 2G x G torus, so the middle rows the
        # cylinder leaves zero are supplied by the mirrored copy instead
        return self._signed_row_embedding(kernel)

    def circulant_index(self) -> torch.Tensor:
        g = self.grid
        pos = torch.arange(g * g)
        yi, xi = pos.div(g, rounding_mode="floor"), pos % g
        return self._deck_index(yi + g, g - 1 - xi, g)  # (r, c) -> (r+G, G-1-c)

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        g = self.grid
        f = torch.cat((field, field.flip(-1)), dim=-2)  # column-mirrored copy
        return torch.fft.irfft2(torch.fft.rfft2(f) * coup, s=(2 * g, g))[..., :g, :]

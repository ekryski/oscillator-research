"""Sphere — a latitude/longitude lattice with open poles.

APPROXIMATION, STATED PLAINLY: this is a lat-long lattice, not true S^2
spectral coupling. Latitude rows use the cylinder's open-axis mechanism,
longitude is periodic, and each SOURCE oscillator's contribution is scaled by
cos(latitude) — the sphere's area element, so near-pole rings, which oversample
the surface, are weighted down.

`torch-harmonics` (genuine spherical harmonics) was evaluated and rejected: it
is not device-portable for this project's smoke ladder, and its real-SHT basis
on a 16x16 grid has eight dead directions, which would break the parameter
parity that makes a geometry comparison meaningful in the first place. The
status is exposed as `Sphere.implementation` so a run can record which it used.
"""

from __future__ import annotations

import math

import torch

from harness.models.geometries.base import PlanarGeometry


def sphere_latitudes(grid: int) -> torch.Tensor:
    """Cell-center latitudes [G], south -> north; the poles are OPEN (no ring
    sits at +-pi/2): lat_r = -pi/2 + pi*(r + 0.5)/G."""
    r = torch.arange(grid, dtype=torch.float32)
    return -math.pi / 2 + math.pi * (r + 0.5) / grid


def sphere_cos_weights(grid: int) -> torch.Tensor:
    """Per-latitude metric weights cos(lat) in (0, 1]: the area element at each
    ring, i.e. the Riemann-sum reading of the continuum coupling integral."""
    return torch.cos(sphere_latitudes(grid))


class Sphere(PlanarGeometry):
    name = "sphere"
    frequency_axis = "latitude (south = low frequency -> north = high)"
    #: "harmonics" is reserved for a future true-S^2 venue
    implementation = "lattice"
    # latitude offsets live in the padded buffer, so the 2-D wrapped-distance
    # mask would not mean what it means on the unpadded venues
    supports_kernel_support = False
    embed_shape = (2, 1)

    def embed_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        return self._signed_row_embedding(kernel)

    def weights(self, like: torch.Tensor) -> torch.Tensor:
        return self._constant("cos_lat", lambda: sphere_cos_weights(self.grid), like)

    def dense_operator(self, taps: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # fold the per-latitude source weights into the dense operator; the
        # spectral path applies them to the field instead, same operator
        w = self.weights(taps).repeat_interleave(self.grid)
        return taps[:, index] * w[None, None, :]

    def apply(self, field: torch.Tensor, coup: torch.Tensor) -> torch.Tensor:
        g = self.grid
        f = field * self.weights(field).view(1, 1, 1, g, 1)
        f = torch.nn.functional.pad(f, (0, 0, 0, g))
        return torch.fft.irfft2(torch.fft.rfft2(f) * coup, s=(2 * g, g))[..., :g, :]

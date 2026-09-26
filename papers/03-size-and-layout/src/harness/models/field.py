"""`OscillatorField`: the coupled oscillator network under test.

A core (phase or Stuart-Landau) behind the fixed band-to-row injection. Built
exactly as paper 02 built it, in the same order of random draws, so a seed
gives the same coupling kernels and natural frequencies in both papers: the
kernels are drawn first, then the natural frequencies, from the global
generator the caller seeds; the initial phases come from a generator of their
own. Model options paper 03 does not run (a random-graph core, an input-as-omega
encoder, stacked blocks, a trained probe and its featurizers) are not carried
over, and none of them drew from the global generator before the kernels.

Channels are parallel copies, like the heads of one attention layer: every
channel gets the same input, has its own kernel and natural frequencies, and
no channel acts on another. They are read side by side, never stacked.
"""

from __future__ import annotations

import torch
from torch import nn

from harness.models.phase import PhaseCore
from harness.models.stuart_landau import SLCore
from harness.stimuli.injection import quad_rows_to_drive, rows_to_drive


def physics_block(core) -> object:
    """The module owning kernel and natural_freqs (PhaseCore block 0, or SLCore itself)."""
    return core.blocks[0] if hasattr(core, "blocks") else core


class OscillatorField(nn.Module):
    """A phase or Stuart-Landau core behind the broadcast drive."""

    def __init__(self, channels: int = 4, grid: int = 16, coupling: str = "kuramoto",
                 damping: float = 0.3, spectral_clamp: float = 1.0, substeps: int = 1,
                 dt: float = 0.1, gain: float = 1.0, seed: int = 42, core: str = "phase",
                 boundary: str = "torus", sakaguchi_alpha: float = 0.0, harmonic2_beta: float = 0.0,
                 kernel_scaling: str = "exact"):
        super().__init__()
        self.channels, self.grid, self.gain = channels, grid, gain
        if core == "phase":
            self.core = PhaseCore(channels=channels, grid=grid, blocks=1, substeps=substeps, dt=dt,
                                  coupling=coupling, damping=damping, spectral_clamp=spectral_clamp,
                                  coupling_impl="auto", seed=seed, boundary=boundary,
                                  sakaguchi_alpha=sakaguchi_alpha, harmonic2_beta=harmonic2_beta,
                                  kernel_scaling=kernel_scaling)
        elif core in ("sl", "sl-fixedamp"):
            if boundary != "torus":
                raise ValueError("the Stuart-Landau cores run on the torus only")
            self.core = SLCore(channels=channels, grid=grid, substeps=substeps, dt=dt, damping=damping,
                               spectral_clamp=spectral_clamp, amplitude_frozen=(core == "sl-fixedamp"),
                               seed=seed, kernel_scaling=kernel_scaling)
        else:
            raise ValueError(f"unknown core '{core}'")

    def _drives(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """rows [B,T,G] -> (additive drives, None); quadrature rows [B,T,G,2] ->
        (zero additive drives, the phase-referenced drives). Broadcast views, never copies."""
        if rows.dim() == 4:
            if not hasattr(self.core, "blocks"):
                raise ValueError("the quadrature pathway drives the phase core only")
            b, t, g, _ = rows.shape
            zeros = rows.new_zeros(()).expand(b, t, self.channels, g, g)
            return zeros, quad_rows_to_drive(rows, self.channels, self.gain)
        return rows_to_drive(rows, self.channels, self.gain), None

    def _scan(self, rows: torch.Tensor) -> torch.Tensor:
        """[B, T, rows] -> [B, T, 2 * C * G * G]: sin θ of every oscillator, then cos θ."""
        drives, quad = self._drives(rows)
        if quad is not None:
            return self.core.forward_scan(drives, drives_quad=quad)[0]
        return self.core.forward_scan(drives)[0]


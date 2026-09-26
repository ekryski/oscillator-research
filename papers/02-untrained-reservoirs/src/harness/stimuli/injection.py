"""The fixed injection path: filterbank rows -> broadcast drive.

The drive is identical across channels and columns by construction, so any
differentiation within a lattice row has to come from the physics (omega, the
coupling kernel, the coupling law) and cannot come from the input map. That is
what makes the field, rather than the encoder, the thing under test.

`drive_kick_stats` is the integrator-validity instrument: explicit Euler is
only meaningful while the per-tick drive phase increment stays below pi, so
every run records its own increment statistics and a sweep point past the bound
is reported as invalid rather than quietly used.
"""

from __future__ import annotations

import torch


def drive_kick_stats(rows: torch.Tensor, gain: float, dt: float = 0.1,
                     max_n: int = 2_000_000) -> dict:
    """Per-tick drive phase-increment stats (rad): dt*gain*|row|.

    The integrator-validity instrument (rescope): a run is valid iff
    the MAX increment stays < pi; rms <= 0.5 rad is the accuracy-comfort
    annotation. Subsampled deterministically for large row tensors."""
    x = rows.abs().flatten()
    if x.numel() > max_n:
        x = x[:: x.numel() // max_n + 1]
    kick = dt * gain * x
    v = kick.sort().values
    return dict(kick_rms=float(kick.pow(2).mean().sqrt()),
                kick_p999=float(v[int(0.999 * (v.numel() - 1))]),
                kick_max=float(v[-1]))


def rows_to_drive(rows: torch.Tensor, channels: int, gain: float) -> torch.Tensor:
    """[B,T,G] -> expanded view [B,T,C,G,G]: identical drive across channels and
    columns; differentiation within a row must come from omega/K/coupling."""
    b, t, g = rows.shape
    return (rows * gain).view(b, t, 1, g, 1).expand(b, t, channels, g, g)


def quad_rows_to_drive(rows: torch.Tensor, channels: int, gain: float) -> torch.Tensor:
    """Quadrature frontend: [B,T,G,2] -> expanded view [B,T,C,G,G,2].

    Same broadcast contract as rows_to_drive (identical across channels and
    columns). gain scales the pair — the Adler torque is linear in (qcos, qsin),
    so gain*(A cos, A sin) == gain * A*sin(phi_bb - theta)."""
    b, t, g, _ = rows.shape
    return (rows * gain).view(b, t, 1, g, 1, 2).expand(b, t, channels, g, g, 2)

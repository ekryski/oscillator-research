"""The fixed injection path: filterbank rows -> broadcast drive.

The drive is identical across channels and columns by construction, so any
differentiation within a lattice row has to come from the physics (omega, the
coupling kernel, the coupling law) and cannot come from the input map. That is
what makes the network, rather than the encoder, the thing under test.
"""

from __future__ import annotations

import torch


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

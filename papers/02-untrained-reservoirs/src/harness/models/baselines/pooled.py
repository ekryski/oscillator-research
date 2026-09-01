"""Shared plumbing for the param-matched conventional references.

Recorded asymmetry, and it matters for how these numbers should be read: a
conventional net's input map is inherently a TRAINED injection, so its role
here is "is this task learnable by a tiny conventional net at matched budget",
never "physics-only learning". They are a budget reference, not a twin.

Every one exposes the identical arm contract — features, features_windowed,
features_settle, feat_dim, a frozen probe — so the evaluation applies to them
with no per-arm special cases.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from harness.measurement.probe import pooled_stats, windowed_stats
from harness.utils.constants import N_SETTLE, PROBE_SCALE, WARMUP_FRAMES


def register_probe(model: nn.Module, feat_dim: int, n_classes: int, probe_seed: int) -> None:
    """The harness's frozen random probe (a buffer — never trained), identical
    construction to GRUBaseline/TCNBaseline."""
    model.feat_dim = feat_dim
    pg = torch.Generator().manual_seed(probe_seed)
    probe = torch.randn(feat_dim, n_classes, generator=pg) / math.sqrt(feat_dim)
    model.register_buffer("probe_w", probe)


class PooledBaseline(nn.Module):
    """Shared feature plumbing: subclasses implement _hidden(rows) -> [B,T',H]
    (already WARMUP-cropped) and _hidden_tail(rows, n) -> the settle window."""

    def features(self, rows: torch.Tensor,
                 tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return pooled_stats(self._hidden(rows),
                             None if tvalid is None else tvalid - WARMUP_FRAMES)

    def features_windowed(self, rows: torch.Tensor, windows: int = 4,
                          tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return windowed_stats(self._hidden(rows), windows,
                               None if tvalid is None else tvalid - WARMUP_FRAMES)

    def features_settle(self, rows: torch.Tensor, tvalid: torch.Tensor | None = None,
                        n_settle: int = N_SETTLE) -> torch.Tensor:
        """Settle-read analog: pool the model's response to an
        n_settle-frame ZERO-INPUT tail. tvalid accepted-but-unused (same
        rationale as OscillatorField.features_settle). Per-family asymmetry recorded
        in each subclass's _hidden_tail docstring."""
        return pooled_stats(self._hidden_tail(rows, n_settle))

    def _hidden_tail(self, rows: torch.Tensor, n_settle: int) -> torch.Tensor:
        """Default (stateless families — CNN/transformer/S4D-as-concat): append
        n_settle zero-input frames and take the outputs at those positions.
        Like TCNBaseline, there is no recurrent state hand-off; the receptive field
        (or attention window) still sees the real past — recorded asymmetry vs
        the truly recurrent GRU/torus settle."""
        pad = rows.new_zeros(rows.shape[0], n_settle, rows.shape[2])
        return self._hidden(torch.cat((rows, pad), dim=1))[:, -n_settle:]

    def forward(self, rows: torch.Tensor,
                tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return self.features(rows, tvalid) @ self.probe_w * PROBE_SCALE

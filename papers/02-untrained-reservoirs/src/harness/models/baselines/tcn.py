"""Temporal-convolution reference — the feed-forward family the GRU misses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from harness.measurement.probe import pooled_stats, windowed_stats
from harness.utils.constants import N_SETTLE, PROBE_SCALE, WARMUP_FRAMES


class TCNBaseline(nn.Module):
    """: tiny causal temporal-conv reference (~1.9k params) — the
    feedforward family the GRU doesn't represent. Same recorded asymmetry as
    the GRU: its input conv is inherently a trained injection; its role is
    "matched-budget conventional net", not physics-only learning."""

    def __init__(self, grid: int = 16, hidden: int = 12, n_classes: int = 8,
                 probe_seed: int = 0, k: int = 5):
        super().__init__()
        self.k = k
        self.c1 = nn.Conv1d(grid, hidden, k)
        self.c2 = nn.Conv1d(hidden, grid, k)
        self.feat_dim = 4 * grid
        pg = torch.Generator().manual_seed(probe_seed)
        probe = torch.randn(self.feat_dim, n_classes, generator=pg) / math.sqrt(self.feat_dim)
        self.register_buffer("probe_w", probe)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        x = rows.transpose(1, 2)  # [B,G,T]
        x = F.relu(self.c1(F.pad(x, (self.k - 1, 0))))  # causal pad
        return F.relu(self.c2(F.pad(x, (self.k - 1, 0)))).transpose(1, 2)[:, WARMUP_FRAMES:]

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
        """Settle-read analog. Recorded ASYMMETRY vs the recurrent
        arms: a TCN has no recurrent state to continue from, so the settle
        window = the causal conv outputs over an n_settle-frame ZERO-PADDED
        input tail — the receptive field still sees the last k-1 real frames,
        then decays to the bias response. tvalid accepted-but-unused (same
        rationale as OscillatorField.features_settle)."""
        x = torch.cat((rows, rows.new_zeros(rows.shape[0], n_settle, rows.shape[2])), dim=1)
        return pooled_stats(self._hidden(x)[:, -n_settle:])

    def forward(self, rows: torch.Tensor,
                tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return self.features(rows, tvalid) @ self.probe_w * PROBE_SCALE

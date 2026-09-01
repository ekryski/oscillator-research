"""GRU reference — the recurrent family, at matched budget."""

from __future__ import annotations

import math

import torch
from torch import nn

from harness.measurement.probe import pooled_stats, windowed_stats
from harness.utils.constants import N_SETTLE, PROBE_SCALE, WARMUP_FRAMES


class GRUBaseline(nn.Module):
    """Param-matched conventional reference (~1.9k params vs physics 2.0k).

    NOT physics-isolated: a GRU's input matrix is inherently a trained
    injection. Its role is "is the task learnable by a tiny conventional
    net at matched budget", not "physics-only learning" — recorded asymmetry."""

    def __init__(self, grid: int = 16, hidden: int = 18, n_classes: int = 8,
                 probe_seed: int = 0):
        super().__init__()
        self.gru = nn.GRU(grid, hidden, batch_first=True)
        self.feat_dim = 4 * hidden
        pg = torch.Generator().manual_seed(probe_seed)
        probe = torch.randn(self.feat_dim, n_classes, generator=pg) / math.sqrt(self.feat_dim)
        self.register_buffer("probe_w", probe)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        return self.gru(rows)[0][:, WARMUP_FRAMES:]  # [B,T-warmup,H]

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
        """Settle-read analog: run the recurrence n_settle extra
        ZERO-INPUT steps from the final hidden state and pool those hidden
        states alone. tvalid accepted-but-unused (same rationale as
        OscillatorField.features_settle)."""
        _, h_n = self.gru(rows)
        settle, _ = self.gru(rows.new_zeros(rows.shape[0], n_settle, rows.shape[2]), h_n)
        return pooled_stats(settle)

    def forward(self, rows: torch.Tensor,
                tvalid: torch.Tensor | None = None) -> torch.Tensor:
        return self.features(rows, tvalid) @ self.probe_w * PROBE_SCALE

"""GRU: the recurrent trained baseline."""

from __future__ import annotations

import torch
from torch import nn

from harness.utils.constants import WARMUP_FRAMES


class GRUBaseline(nn.Module):
    """One GRU layer over the rows. Paper 02's hidden width is 18 (1,944 parameters on 16 rows);
    paper 03 sets the width from the parameter budget (`harness.experiment.arms.trained_width`)."""

    def __init__(self, grid: int = 16, hidden: int = 18):
        super().__init__()
        self.gru = nn.GRU(grid, hidden, batch_first=True)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        return self.gru(rows)[0][:, WARMUP_FRAMES:]  # [B, T - warm-up, H]

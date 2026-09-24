"""TCN: a small causal temporal convolutional network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from harness.utils.constants import WARMUP_FRAMES


class TCNBaseline(nn.Module):
    """Two causal Conv1d layers, rows -> hidden -> rows, kernel 5. Paper 02's hidden width is 12
    (1,948 parameters on 16 rows)."""

    def __init__(self, grid: int = 16, hidden: int = 12, k: int = 5):
        super().__init__()
        self.k = k
        self.c1 = nn.Conv1d(grid, hidden, k)
        self.c2 = nn.Conv1d(hidden, grid, k)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        x = rows.transpose(1, 2)  # [B, G, T]
        x = F.relu(self.c1(F.pad(x, (self.k - 1, 0))))  # causal pad
        return F.relu(self.c2(F.pad(x, (self.k - 1, 0)))).transpose(1, 2)[:, WARMUP_FRAMES:]

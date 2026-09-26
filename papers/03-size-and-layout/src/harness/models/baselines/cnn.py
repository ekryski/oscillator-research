"""CNN: two causal one-dimensional convolutions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from harness.utils.constants import WARMUP_FRAMES


class CNNBaseline(nn.Module):
    """Two causal Conv1d layers, rows -> hidden -> rows, kernel 5 (a receptive field of 9 frames), paper
    02's. Paper 02's hidden width is 13 (2,109 parameters on 16 rows), one notch over its 2,048 budget;
    paper 03 keeps that rule at every budget, taking the narrowest width that reaches it."""

    def __init__(self, grid: int = 16, hidden: int = 13, k: int = 5):
        super().__init__()
        self.k = k
        self.c1 = nn.Conv1d(grid, hidden, k)
        self.c2 = nn.Conv1d(hidden, grid, k)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        x = rows.transpose(1, 2)  # [B, G, T]
        x = F.relu(self.c1(F.pad(x, (self.k - 1, 0))))  # causal pad
        # a linear output layer: with a rectified one, a poor draw can leave every output unit
        # inactive on every clip, and the network then never learns
        return self.c2(F.pad(x, (self.k - 1, 0))).transpose(1, 2)[:, WARMUP_FRAMES:]

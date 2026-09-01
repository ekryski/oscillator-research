"""Small causal CNN reference — the shallow convolutional family."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from harness.models.baselines.pooled import PooledBaseline, register_probe
from harness.utils.constants import WARMUP_FRAMES


class CNNBaseline(PooledBaseline):
    """Small causal CNN: 2x causal Conv1d G -> hidden -> G, k=5, pooled stats.

    Budget note (recorded deviation): the program sketch said 16->24->16, which
    counts to 3,880 params — 89% over the 2,048 physics budget and outside the
    +-15% budget gate every baseline must pass. Width is the sanctioned sizing
    knob (as for the transformer), so hidden=13 -> 2,109 params (+3.0%). At
    G=16 this makes CNNBaseline the TCN family at one width notch up; its distinct
    distinct role is the 40-mel-band lens-cost control, where the class is
    re-sized via grid=40."""

    def __init__(self, grid: int = 16, hidden: int = 13, n_classes: int = 8,
                 probe_seed: int = 0, k: int = 5):
        super().__init__()
        self.k = k
        self.c1 = nn.Conv1d(grid, hidden, k)
        self.c2 = nn.Conv1d(hidden, grid, k)
        register_probe(self, 4 * grid, n_classes, probe_seed)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        x = rows.transpose(1, 2)  # [B,G,T]
        x = F.relu(self.c1(F.pad(x, (self.k - 1, 0))))  # causal pad
        return F.relu(self.c2(F.pad(x, (self.k - 1, 0)))).transpose(1, 2)[:, WARMUP_FRAMES:]

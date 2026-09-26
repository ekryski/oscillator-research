"""TCN: a small temporal convolutional network, dilated and residual."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from harness.utils.constants import WARMUP_FRAMES


class TCNBaseline(nn.Module):
    """A small temporal convolutional network in the form of Bai, Kolter & Koltun (2018), paper 02's.

    Two residual blocks, each a dilated causal convolution G -> hidden, a ReLU,
    and a dilated causal convolution hidden -> G, added back onto the block's
    input. Dilations 1, 2 and 4, 8 with kernel 3 give a receptive field of 31
    frames (about half a second). Paper 02's hidden width is 10 (1,972
    parameters on 16 rows); paper 03 sets the width from the parameter budget
    (`harness.experiment.arms.trained_width`). The residual path is linear, so
    no block can shut the signal off: a block whose rectified units are all
    inactive passes its input through unchanged, and the network still learns.
    """

    def __init__(self, grid: int = 16, hidden: int = 10, k: int = 3, dilations: tuple = ((1, 2), (4, 8))):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList(
            nn.ModuleList((nn.Conv1d(grid, hidden, k, dilation=d1), nn.Conv1d(hidden, grid, k, dilation=d2)))
            for d1, d2 in dilations)

    @staticmethod
    def _causal(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(x, ((conv.kernel_size[0] - 1) * conv.dilation[0], 0)))

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        x = rows.transpose(1, 2)  # [B, G, T]
        for c1, c2 in self.blocks:
            x = x + self._causal(c2, F.relu(self._causal(c1, x)))
        return x.transpose(1, 2)[:, WARMUP_FRAMES:]

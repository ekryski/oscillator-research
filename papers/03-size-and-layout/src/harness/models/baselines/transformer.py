"""Transformer: one causal self-attention layer (Vaswani et al. 2017)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from harness.utils.constants import WARMUP_FRAMES


class TransformerBaseline(nn.Module):
    """Input linear + fixed sinusoidal positions -> masked self-attention (2 heads) -> a two-layer
    feed-forward block, post-norm. Paper 02's width is 16 (1,968 parameters on 16 rows)."""

    def __init__(self, grid: int = 16, d: int = 16, nhead: int = 2):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(grid, d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff1, self.ff2 = nn.Linear(d, d), nn.Linear(d, d)

    def _pos(self, t: int, device: torch.device) -> torch.Tensor:
        """Standard sinusoidal encoding [T, d], computed on the fly."""
        pos = torch.arange(t, device=device, dtype=torch.float32)[:, None]
        i = torch.arange(self.d // 2, device=device, dtype=torch.float32)[None, :]
        ang = pos / (10000.0 ** (2 * i / self.d))
        return torch.cat((torch.sin(ang), torch.cos(ang)), dim=1)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        t = rows.shape[1]
        x = self.inp(rows) + self._pos(t, rows.device)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=rows.device), diagonal=1)
        a, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff2(F.relu(self.ff1(x))))
        return x[:, WARMUP_FRAMES:]

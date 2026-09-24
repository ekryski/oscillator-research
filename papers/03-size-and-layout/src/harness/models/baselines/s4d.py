"""S4D: a minimal diagonal state-space model (Gu, Gupta & Re 2022)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from harness.utils.constants import WARMUP_FRAMES


class S4DBaseline(nn.Module):
    """Per-channel learnable complex poles a = exp(-exp(log_decay) + i*freq), n_states diagonal
    states per channel, input and output mixing linears, a real readout per state and a skip, run as
    a recurrent scan. Paper 02's width is 16 with 16 states (1,840 parameters on 16 rows); paper 03
    scales the state count with the width."""

    def __init__(self, grid: int = 16, hidden: int = 16, n_states: int = 16):
        super().__init__()
        self.hidden, self.n_states = hidden, n_states
        self.inp = nn.Linear(grid, hidden)
        self.log_decay = nn.Parameter(torch.full((hidden, n_states), math.log(0.5)))
        self.freq = nn.Parameter(torch.linspace(0, math.pi, n_states).expand(hidden, n_states).clone())
        self.b = nn.Parameter(torch.ones(hidden, n_states))
        self.c_re = nn.Parameter(torch.randn(hidden, n_states) / math.sqrt(n_states))
        self.c_im = nn.Parameter(torch.randn(hidden, n_states) / math.sqrt(n_states))
        self.d_skip = nn.Parameter(torch.ones(hidden))
        self.out = nn.Linear(hidden, hidden)

    def _hidden(self, rows: torch.Tensor) -> torch.Tensor:
        u = self.inp(rows)                                   # [B,T,H]
        a = torch.exp(torch.complex(-torch.exp(self.log_decay), self.freq))  # [H,N], |a|<1
        c = torch.complex(self.c_re, self.c_im)
        x = torch.zeros(rows.shape[0], self.hidden, self.n_states, dtype=a.dtype, device=rows.device)
        ys = []
        for u_t in u.unbind(1):                              # recurrent scan
            x = a * x + self.b * u_t[:, :, None].to(a.dtype)
            ys.append((x * c).sum(dim=-1).real + self.d_skip * u_t)
        y = torch.stack(ys, dim=1)                           # [B,T,H]
        return F.relu(self.out(y))[:, WARMUP_FRAMES:]
